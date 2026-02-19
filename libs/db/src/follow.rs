//! Follower client that connects to a source elodin-db and replicates all data.
//!
//! Usage: `elodin-db run [::]:2241 --follows SOURCE_IP:2240 --follow-packet-size 1500`
//!
//! # Design
//!
//! All requests are sent in a single burst before reading any responses.
//! This is required because each connection handler on the source runs on
//! a stellarator `stellar()` thread whose I/O reactor only reliably delivers
//! data that is already buffered in the TCP socket at the time of the read.
//! By pipelining all requests, the source processes them sequentially from
//! its TCP receive buffer without needing I/O wake-ups between requests.
//!
//! The follower sends `DumpMetadata`, `DumpSchema`, and `FollowStream` in
//! one burst. The source handles DumpMetadata and DumpSchema inline (sending
//! responses), then transitions the connection into follow-stream mode.
//! The FollowStream handler streams all existing and new data.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use impeller2::types::{IntoLenPacket, Msg, OwnedPacket as Packet, PacketId, Timestamp};
use impeller2_stellar::{PacketSink, PacketStream};
use impeller2_wkt::*;
use stellarator::{io::SplitExt, net::TcpStream};
use tracing::{debug, info, warn};

use crate::{AtomicTimestampExt, ComponentSchema, DB, DBSink, Error};

/// Configuration for a follower connection.
pub struct FollowConfig {
    /// Address of the source elodin-db instance to follow.
    pub source_addr: SocketAddr,
    /// Target packet size in bytes for coalesced streaming (default: 1500).
    pub target_packet_size: usize,
    /// Delay before reconnecting after a dropped connection.
    /// Defaults to 2 seconds; tests may use a shorter value.
    pub reconnect_delay: Duration,
}

/// Default reconnect delay for production use.
const DEFAULT_RECONNECT_DELAY: Duration = Duration::from_secs(2);

impl Default for FollowConfig {
    fn default() -> Self {
        Self {
            source_addr: ([127, 0, 0, 1], 2240).into(),
            target_packet_size: 1500,
            reconnect_delay: DEFAULT_RECONNECT_DELAY,
        }
    }
}

/// Run the follower, connecting to the source and replicating all data.
///
/// This function reconnects on failure and should be spawned as a background task.
pub async fn run_follower(config: FollowConfig, db: Arc<DB>) -> Result<(), Error> {
    loop {
        info!(source = %config.source_addr, "connecting to source database");
        match run_follower_inner(&config, &db).await {
            Ok(()) => {
                info!("follow stream ended cleanly");
                return Ok(());
            }
            Err(err) if err.is_stream_closed() => {
                warn!(
                    "follow connection closed, reconnecting in {:?}",
                    config.reconnect_delay
                );
            }
            Err(err) => {
                warn!(
                    ?err,
                    "follow connection error, reconnecting in {:?}", config.reconnect_delay
                );
            }
        }
        stellarator::sleep(config.reconnect_delay).await;
    }
}

async fn run_follower_inner(config: &FollowConfig, db: &Arc<DB>) -> Result<(), Error> {
    let stream = TcpStream::connect(config.source_addr).await?;
    let (rx, tx) = stream.split();
    let mut rx = PacketStream::new(rx);
    let tx = PacketSink::new(tx);

    let mut buf = vec![0u8; 8 * 1024 * 1024];

    // ── Send ALL requests in one burst ──────────────────────────────────
    // This avoids the source's I/O reactor needing to wake up between
    // requests.  DumpMetadata + DumpSchema are processed inline by the
    // source and their responses sent back.  FollowStream transitions the
    // connection into streaming mode.

    let meta_req_id: u8 = 1;
    let schema_req_id: u8 = 2;
    let follow_req_id: u8 = 0; // streaming uses req_id 0

    tx.send(DumpMetadata.with_request_id(meta_req_id)).await.0?;
    tx.send(DumpSchema.with_request_id(schema_req_id)).await.0?;
    tx.send(
        FollowStream {
            target_packet_size: config.target_packet_size as u32,
        }
        .with_request_id(follow_req_id),
    )
    .await
    .0?;

    info!("sent DumpMetadata + DumpSchema + FollowStream in one burst");

    // ── Receive DumpMetadataResp and DumpSchemaResp ─────────────────────
    let mut metadata_resp: Option<DumpMetadataResp> = None;
    let mut schema_resp: Option<DumpSchemaResp> = None;

    while metadata_resp.is_none() || schema_resp.is_none() {
        let pkt = rx.next(buf).await?;
        match &pkt {
            Packet::Msg(m) if m.req_id == meta_req_id && m.id == DumpMetadataResp::ID => {
                metadata_resp = Some(m.parse::<DumpMetadataResp>()?);
            }
            Packet::Msg(m) if m.req_id == schema_req_id && m.id == DumpSchemaResp::ID => {
                schema_resp = Some(m.parse::<DumpSchemaResp>()?);
            }
            other => {
                debug!(?other, "unexpected packet during phase 1, ignoring");
            }
        }
        buf = pkt.into_buf().into_inner();
    }

    let metadata_resp = metadata_resp.unwrap();
    let schema_resp = schema_resp.unwrap();

    info!(
        components = metadata_resp.component_metadata.len(),
        messages = metadata_resp.msg_metadata.len(),
        schemas = schema_resp.schemas.len(),
        "phase 1: received metadata and schemas"
    );

    // ── Apply metadata ──────────────────────────────────────────────────

    // Apply db_config (includes schematic content/path, recording flag, etc.).
    // ── Apply metadata ──────────────────────────────────────────────────

    // Apply db_config (includes schematic content/path, recording flag, etc.).
    db.with_state_mut(|s| {
        s.db_config
            .metadata
            .extend(metadata_resp.db_config.metadata.clone());
        s.db_config.default_stream_time_step = metadata_resp.db_config.default_stream_time_step;
    });
    // Set earliest_timestamp to the source's DB creation time so the
    // editor's timeline starts at the correct point.
    if let Some(ts_micros) = metadata_resp.db_config.time_start_timestamp_micros() {
        let _ = db.set_earliest_timestamp(Timestamp(ts_micros));
    }
    db.save_db_state()?;

    // Apply component metadata.
    for metadata in &metadata_resp.component_metadata {
        db.with_state_mut(|s| s.set_component_metadata(metadata.clone(), &db.path))?;
    }

    // Apply message metadata.
    for msg_meta in &metadata_resp.msg_metadata {
        let msg_id = impeller2::types::msg_id(&msg_meta.name);
        db.with_state_mut(|s| s.set_msg_metadata(msg_id, msg_meta.clone(), &db.path))?;
    }

    // Create components from schemas using per-component start_timestamps.
    let source_start_ts = metadata_resp
        .db_config
        .time_start_timestamp_micros()
        .map(Timestamp)
        .unwrap_or_else(Timestamp::now);
    for (&component_id, schema) in &schema_resp.schemas {
        let cs = ComponentSchema::from(schema.clone());
        let start_ts = schema_resp
            .start_timestamps
            .get(&component_id)
            .copied()
            .unwrap_or(source_start_ts);
        db.with_state_mut(|s| {
            s.insert_component_with_start_timestamp(component_id, cs, start_ts, &db.path)
        })?;
    }

    // Track all known component IDs as followed.
    {
        let mut followed = db.followed_components.write().unwrap();
        for &cid in schema_resp.schemas.keys() {
            followed.insert(cid);
        }
    }

    // Track message timestamps for dedup.
    let msg_ids: Vec<PacketId> = metadata_resp
        .msg_metadata
        .iter()
        .map(|m| impeller2::types::msg_id(&m.name))
        .collect();
    let mut msg_last_ts: std::collections::HashMap<PacketId, Timestamp> =
        std::collections::HashMap::new();
    {
        let db_path = db.path.clone();
        db.with_state_mut(|state| {
            for &mid in &msg_ids {
                if let Ok(log) = state.get_or_insert_msg_log(mid, &db_path)
                    && let Some((ts, _)) = log.latest()
                {
                    msg_last_ts.insert(mid, ts);
                }
            }
        });
    }

    info!(
        target_packet_size = config.target_packet_size,
        "phase 2: entering follow stream (backfill + real-time)"
    );

    // ── Phase 2: Receive the unified FollowStream ───────────────────────
    // The source's FollowStream handler sends all existing data (backfill)
    // and then streams new data as it arrives.  No separate backfill phase
    // is needed.

    // Connection-local VTable-to-ComponentId mapping for follow-stream data.
    // Stored locally (not in the global vtable_registry) to prevent ID collisions
    // with VTables registered by local clients connecting to this follower DB.
    let mut follow_vtables: std::collections::HashMap<PacketId, impeller2::types::ComponentId> =
        std::collections::HashMap::new();

    // Track how many samples each component had at connection start.
    // On reconnection, the source resends ALL data; we skip samples we
    // already have using these counters (decremented as chunks arrive).
    let mut skip_remaining: std::collections::HashMap<impeller2::types::ComponentId, usize> = db
        .with_state(|state| {
            state
                .components
                .iter()
                .map(|(&cid, comp)| (cid, comp.time_series.sample_count()))
                .collect()
        });

    loop {
        let pkt = rx.next(buf).await?;
        match &pkt {
            // ComponentMetadata – apply and register as followed.
            Packet::Msg(m) if m.id == ComponentMetadata::ID => {
                let metadata: ComponentMetadata = m.parse()?;
                let cid = metadata.component_id;
                db.with_state_mut(|s| s.set_component_metadata(metadata, &db.path))?;
                db.followed_components.write().unwrap().insert(cid);
            }

            // DumpSchemaResp – create new components.
            Packet::Msg(m) if m.id == DumpSchemaResp::ID => {
                let resp: DumpSchemaResp = m.parse()?;
                for (component_id, schema) in resp.schemas {
                    let cs = ComponentSchema::from(schema);
                    let start_ts = resp
                        .start_timestamps
                        .get(&component_id)
                        .copied()
                        .unwrap_or(source_start_ts);
                    db.with_state_mut(|s| {
                        s.insert_component_with_start_timestamp(
                            component_id,
                            cs,
                            start_ts,
                            &db.path,
                        )
                    })?;
                    db.followed_components.write().unwrap().insert(component_id);
                }
            }

            // SetMsgMetadata – apply message metadata.
            Packet::Msg(m) if m.id == SetMsgMetadata::ID => {
                let meta: SetMsgMetadata = m.parse()?;
                db.with_state_mut(|s| s.set_msg_metadata(meta.id, meta.metadata, &db.path))?;
            }

            // VTableMsg – extract ComponentId and store in connection-local map.
            // Not inserted into the global vtable_registry to avoid ID collisions
            // with VTables from local clients.
            Packet::Msg(m) if m.id == VTableMsg::ID => {
                let vtable_msg = m.parse::<VTableMsg>()?;
                if let Some(Ok(field)) = vtable_msg.vtable.realize_fields(None).next() {
                    follow_vtables.insert(vtable_msg.id, field.component_id);
                } else {
                    warn!(vtable_id = ?vtable_msg.id, "failed to extract ComponentId from follow-stream VTable");
                }
            }

            // TimeSeries – bulk data from follow-stream.  Write
            // timestamps and data directly to the component's time
            // series without per-sample VTable decomposition.
            Packet::TimeSeries(ts) => 'ts: {
                let Ok(timestamps) = ts.timestamps() else {
                    warn!(vtable_id = ?ts.id, "failed to parse TimeSeries timestamps");
                    break 'ts;
                };
                let Ok(data_buf) = ts.data() else {
                    warn!(vtable_id = ?ts.id, "failed to parse TimeSeries data");
                    break 'ts;
                };
                // Resolve component_id from the connection-local VTable map.
                let Some(&component_id) = follow_vtables.get(&ts.id) else {
                    warn!(vtable_id = ?ts.id, "unknown follow-stream VTable ID in TimeSeries packet");
                    break 'ts;
                };
                // Dedup: skip samples we already had at connection start.
                let skip_count = skip_remaining.get(&component_id).copied().unwrap_or(0);
                let skip = skip_count.min(timestamps.len());
                if skip > 0 {
                    *skip_remaining.entry(component_id).or_insert(0) -= skip;
                }

                db.with_state(|state| -> Result<(), Error> {
                    let Some(component) = state.components.get(&component_id) else {
                        return Ok(());
                    };
                    let elem_size = component.schema.size();
                    for (i, &timestamp) in timestamps.iter().enumerate().skip(skip) {
                        let start = i * elem_size;
                        let end = start + elem_size;
                        if end > data_buf.len() {
                            break;
                        }
                        let _ = component
                            .time_series
                            .push_buf(timestamp, &data_buf[start..end]);
                    }
                    if timestamps.len() > skip {
                        db.last_updated.update_max(*timestamps.last().unwrap());
                    }
                    Ok(())
                })?;
            }

            // Table – decomponentize and write to local DB.
            Packet::Table(table) => {
                db.with_state(|state| -> Result<(), Error> {
                    let mut sink = DBSink {
                        components: &state.components,
                        snapshot_barrier: &db.snapshot_barrier,
                        last_updated: &db.last_updated,
                        earliest_timestamp: &db.earliest_timestamp,
                        sunk_new_time_series: false,
                        table_received: db.apply_implicit_timestamp(),
                        followed_components: &db.followed_components,
                        is_follower: true,
                    };
                    match table.sink(&state.vtable_registry, &mut sink) {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            warn!(?e, "error decomponentizing follow-stream table");
                        }
                        Err(e) => {
                            warn!(?e, "vtable error in follow-stream table");
                        }
                    }
                    if sink.sunk_new_time_series {
                        db.vtable_gen
                            .value
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        db.vtable_gen.wait_queue.wake_all();
                    }
                    Ok(())
                })?;
            }

            // MsgWithTimestamp – write message with original timestamp,
            // skipping duplicates already received during previous sessions.
            Packet::Msg(m) if m.timestamp.is_some() => {
                let timestamp = m.timestamp.unwrap();
                let last = msg_last_ts
                    .get(&m.id)
                    .copied()
                    .unwrap_or(Timestamp(i64::MIN));
                if timestamp > last {
                    db.push_msg(timestamp, m.id, &m.buf)?;
                    msg_last_ts.insert(m.id, timestamp);
                }
            }

            // Other messages – try to store as generic messages.
            Packet::Msg(m) => {
                let timestamp = m.timestamp.unwrap_or_else(|| db.apply_implicit_timestamp());
                db.push_msg(timestamp, m.id, &m.buf)?;
            }
        }
        buf = pkt.into_buf().into_inner();
    }
}
