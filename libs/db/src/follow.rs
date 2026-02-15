//! Follower client that connects to a source elodin-db and replicates all data.
//!
//! Usage: `elodin-db run [::]:2241 --follows SOURCE_IP:2240 --follow-packet-size 1500`
//!
//! # Phases
//!
//! 1. **Initial sync** – fetch metadata and schemas from the source via
//!    request/response.
//! 2. **Historical backfill** – fetch all existing time-series data and
//!    message logs.
//! 3. **Real-time streaming** – send a [`FollowStream`] request and
//!    continuously apply incoming packets to the local database.

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
    let mut next_req_id: u8 = 1;

    // ── Phase 1: Metadata & schema sync ─────────────────────────────────

    // Send DumpMetadata
    let meta_req_id = next_req_id;
    next_req_id += 1;
    tx.send(DumpMetadata.with_request_id(meta_req_id)).await.0?;

    // Send DumpSchema
    let schema_req_id = next_req_id;
    next_req_id += 1;
    tx.send(DumpSchema.with_request_id(schema_req_id)).await.0?;

    let mut metadata_resp: Option<DumpMetadataResp> = None;
    let mut schema_resp: Option<DumpSchemaResp> = None;

    // Receive both responses (order not guaranteed).
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

    // Apply db_config (includes schematic content/path, recording flag, etc.).
    db.with_state_mut(|s| {
        s.db_config
            .metadata
            .extend(metadata_resp.db_config.metadata.clone());
        s.db_config.default_stream_time_step = metadata_resp.db_config.default_stream_time_step;
    });
    // Match the source's earliest_timestamp so the follower's time range
    // is identical.
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
        // Derive the packet ID from the message name hash (same as the source).
        let msg_id = impeller2::types::msg_id(&msg_meta.name);
        db.with_state_mut(|s| s.set_msg_metadata(msg_id, msg_meta.clone(), &db.path))?;
    }

    // Create components from schemas using the source's per-component
    // start_timestamp so that the on-disk AppendLog headers match exactly
    // between source and follower (binary-level replication fidelity).
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

    // ── Phase 1b: Historical backfill ───────────────────────────────────

    // Backfill component time-series data.
    for (&component_id, schema) in &schema_resp.schemas {
        let ts_req_id = next_req_id;
        next_req_id = next_req_id.wrapping_add(1);
        if next_req_id == 0 {
            next_req_id = 1;
        }

        let schema_size = ComponentSchema::from(schema.clone()).size();
        // Use a unique-ish packet ID for request/response correlation.
        let id_bytes = (component_id.0 as u16).to_le_bytes();
        let get_ts = GetTimeSeries {
            id: id_bytes,
            range: Timestamp(i64::MIN)..Timestamp(i64::MAX),
            component_id,
            limit: None,
        };
        tx.send(get_ts.with_request_id(ts_req_id)).await.0?;

        // Receive the TimeSeries response.
        loop {
            let pkt = rx.next(buf).await?;
            match &pkt {
                Packet::TimeSeries(ts) if ts.req_id == ts_req_id => {
                    if let (Ok(timestamps), Ok(data)) = (ts.timestamps(), ts.data())
                        && !timestamps.is_empty()
                    {
                        let component = db.with_state(|s| s.components.get(&component_id).cloned());
                        if let Some(component) = component {
                            // Skip samples at or before the follower's current
                            // latest -- prevents duplicates on reconnect when
                            // the follower already has persisted data.
                            let local_latest = component
                                .time_series
                                .latest()
                                .map(|(ts, _)| *ts)
                                .unwrap_or(Timestamp(i64::MIN));
                            for (i, timestamp) in timestamps.iter().enumerate() {
                                if *timestamp <= local_latest {
                                    continue;
                                }
                                let start = i * schema_size;
                                let end = start + schema_size;
                                if end <= data.len() {
                                    let _ = component
                                        .time_series
                                        .push_buf(*timestamp, &data[start..end]);
                                }
                            }
                            db.last_updated.update_max(*timestamps.last().unwrap());
                        }
                        debug!(
                            component = ?component_id,
                            samples = timestamps.len(),
                            "backfilled component"
                        );
                    }
                    buf = pkt.into_buf().into_inner();
                    break;
                }
                _ => {
                    debug!("unexpected packet during backfill, ignoring");
                    buf = pkt.into_buf().into_inner();
                }
            }
        }
    }

    // Backfill message logs.
    let msg_ids: Vec<PacketId> = metadata_resp
        .msg_metadata
        .iter()
        .map(|m| impeller2::types::msg_id(&m.name))
        .collect();

    for &msg_id in &msg_ids {
        let msg_req_id = next_req_id;
        next_req_id = next_req_id.wrapping_add(1);
        if next_req_id == 0 {
            next_req_id = 1;
        }

        let get_msgs = GetMsgs {
            msg_id,
            range: Timestamp(i64::MIN)..Timestamp(i64::MAX),
            limit: None,
        };
        tx.send(get_msgs.with_request_id(msg_req_id)).await.0?;

        loop {
            let pkt = rx.next(buf).await?;
            match &pkt {
                Packet::Msg(m) if m.req_id == msg_req_id && m.id == MsgBatch::ID => {
                    let batch: MsgBatch = m.parse()?;
                    // Determine local latest to skip already-persisted messages.
                    let db_path = db.path.clone();
                    let local_msg_latest = db.with_state_mut(|s| {
                        s.get_or_insert_msg_log(msg_id, &db_path)
                            .ok()
                            .and_then(|log| log.latest().map(|(ts, _)| ts))
                            .unwrap_or(Timestamp(i64::MIN))
                    });
                    let mut written = 0usize;
                    for (timestamp, data) in &batch.data {
                        if *timestamp <= local_msg_latest {
                            continue;
                        }
                        db.push_msg(*timestamp, msg_id, data)?;
                        written += 1;
                    }
                    debug!(msg_id = ?msg_id, messages = written, total = batch.data.len(), "backfilled message log");
                    buf = pkt.into_buf().into_inner();
                    break;
                }
                _ => {
                    debug!("unexpected packet during msg backfill, ignoring");
                    buf = pkt.into_buf().into_inner();
                }
            }
        }
    }

    info!("phase 1b: historical backfill complete");

    // Track latest message timestamps so we can skip duplicates in Phase 2.
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

    // ── Phase 2: Real-time streaming via FollowStream ───────────────────

    let follow_req_id: u8 = 0; // streaming uses req_id 0
    tx.send(
        FollowStream {
            target_packet_size: config.target_packet_size as u32,
        }
        .with_request_id(follow_req_id),
    )
    .await
    .0?;

    info!(
        target_packet_size = config.target_packet_size,
        "phase 2: real-time follow stream started"
    );

    // Receive loop – process the unified stream from the source.
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

            // VTableMsg – register vtable locally.
            Packet::Msg(m) if m.id == VTableMsg::ID => {
                let vtable = m.parse::<VTableMsg>()?;
                db.insert_vtable(vtable)?;
            }

            // Table – decomponentize and write to local DB.
            Packet::Table(table) => {
                db.with_state(|state| -> Result<(), Error> {
                    let mut sink = DBSink {
                        components: &state.components,
                        snapshot_barrier: &db.snapshot_barrier,
                        last_updated: &db.last_updated,
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
            // skipping duplicates already received during backfill.
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

            _ => {}
        }
        buf = pkt.into_buf().into_inner();
    }
}
