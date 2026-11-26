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

use impeller::types::{
    ComponentId, IntoLenPacket, Msg, OwnedPacket as Packet, PacketId, Timestamp,
};
use impeller_stellar::{PacketSink, PacketStream};
use impeller_wkt::*;
#[cfg(feature = "axum")]
use stellarator::struc_con::Joinable;
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

/// Fetch metadata/schemas from the source, sync `db:` assets, and apply local state
/// before the follower accepts clients.
pub async fn prime_follower_state(config: &FollowConfig, db: &Arc<DB>) -> Result<(), Error> {
    mark_assets_read_only(db);
    let (_, metadata_resp, schema_resp) = request_source_snapshot(config, false).await?;
    apply_source_snapshot(
        config,
        db,
        &metadata_resp,
        &schema_resp,
        SchematicAssetSync::Await,
    )
    .await
}

/// Run the follower, connecting to the source and replicating all data.
///
/// This function reconnects on failure and should be spawned as a background task.
pub async fn run_follower(config: FollowConfig, db: Arc<DB>) -> Result<(), Error> {
    mark_assets_read_only(&db);
    loop {
        info!(source = %config.source_addr, "connecting to source database");
        match run_follower_inner(&config, &db).await {
            Ok(()) => {
                info!("follow stream ended cleanly");
                return Ok(());
            }
            Err(
                err @ (Error::FollowHistoryChanged { .. }
                | Error::FollowProtocolUnsupported
                | Error::Stellar(stellarator::Error::PacketTooLarge { .. })),
            ) => {
                // A changed history or unsupported protocol will not heal on
                // retry, and retrying cannot shrink an oversized packet.
                tracing::error!(error = %err, "follow replication stopped");
                return Err(err);
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

/// A follower's asset tree is a read-only mirror of its source. Setting this
/// makes the Impeller `StoreAsset` handler reject client writes, matching the
/// asset HTTP server's `PUT` 405 gate; otherwise a TCP client could diverge the
/// mirror (and bump `assets.revision`) until the next full mirror pass.
fn mark_assets_read_only(db: &Arc<DB>) {
    db.assets_read_only
        .store(true, std::sync::atomic::Ordering::Release);
}

struct FollowStreamSession {
    rx: PacketStream<stellarator::io::OwnedReader<stellarator::net::TcpStream>>,
    buf: Vec<u8>,
    message_logs: Vec<PacketId>,
}

async fn request_source_snapshot(
    config: &FollowConfig,
    start_stream: bool,
) -> Result<
    (
        Option<FollowStreamSession>,
        DumpMetadataResp,
        DumpSchemaResp,
    ),
    Error,
> {
    let stream = TcpStream::connect(config.source_addr).await?;
    let (rx, tx) = stream.split();
    let mut rx = PacketStream::new(rx).with_max_len(crate::MAX_PACKET_LEN);
    let tx = PacketSink::new(tx);
    let mut buf = vec![0u8; 8 * 1024 * 1024];

    let meta_req_id: u8 = 1;
    let schema_req_id: u8 = 2;

    tx.send(DumpMetadata.with_request_id(meta_req_id)).await.0?;
    tx.send(DumpSchema.with_request_id(schema_req_id)).await.0?;
    if start_stream {
        tx.send(
            FollowStreamV2 {
                target_packet_size: config.target_packet_size as u32,
                version: 2,
            }
            .with_request_id(0),
        )
        .await
        .0?;
        info!("sent DumpMetadata + DumpSchema + FollowStream in one burst");
    } else {
        info!("sent DumpMetadata + DumpSchema for follower priming");
    }

    let mut metadata_resp: Option<DumpMetadataResp> = None;
    let mut schema_resp: Option<DumpSchemaResp> = None;

    let mut stream_ready = !start_stream;
    let mut message_logs = Vec::new();
    while metadata_resp.is_none() || schema_resp.is_none() || !stream_ready {
        // Only time out the version acknowledgement. Large snapshots may need
        // much longer on slow links and must not get stuck in a reconnect loop.
        let pkt = if metadata_resp.is_some() && schema_resp.is_some() && !stream_ready {
            futures_lite::future::race(
                async { rx.next_grow(buf).await.map_err(Error::from) },
                async {
                    stellarator::sleep(Duration::from_secs(30)).await;
                    Err(Error::FollowProtocolUnsupported)
                },
            )
            .await?
        } else {
            rx.next_grow(buf).await?
        };
        match &pkt {
            Packet::Msg(m) if start_stream && m.id == FollowStreamReady::ID => {
                message_logs = m.parse::<FollowStreamReady>()?.message_logs;
                stream_ready = true;
            }
            Packet::Msg(m) if start_stream && m.req_id == 0 && m.id == ErrorResponse::ID => {
                return Err(Error::FollowProtocolUnsupported);
            }
            Packet::Msg(m) if m.req_id == meta_req_id && m.id == DumpMetadataResp::ID => {
                metadata_resp = Some(m.parse::<DumpMetadataResp>()?);
            }
            Packet::Msg(m) if m.req_id == schema_req_id && m.id == DumpSchemaResp::ID => {
                schema_resp = Some(m.parse::<DumpSchemaResp>()?);
            }
            _ if metadata_resp.is_some() && schema_resp.is_some() && !stream_ready => {
                return Err(Error::FollowProtocolUnsupported);
            }
            other => {
                debug!(
                    ?other,
                    "unexpected packet during snapshot handshake, ignoring"
                );
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
        "received metadata and schemas from source"
    );

    let session = if start_stream {
        Some(FollowStreamSession {
            rx,
            buf,
            message_logs,
        })
    } else {
        None
    };
    Ok((session, metadata_resp, schema_resp))
}

enum SchematicAssetSync {
    /// Block until HTTP asset sync completes (no FollowStream active).
    Await,
    /// Keep draining the follow stream while assets sync over HTTP.
    Background,
}

#[cfg(feature = "axum")]
async fn sync_schematic_assets_on_tokio(source_addr: SocketAddr, db: Arc<DB>) {
    if let Err(err) = stellarator::struc_con::tokio(move |_| async move {
        crate::assets_http::sync_schematic_assets_for_db_from_source(source_addr, &db).await;
    })
    .join()
    .await
    {
        warn!(?err, "failed to join schematic asset sync task");
    }
}

#[cfg(feature = "axum")]
fn spawn_schematic_asset_sync(source_addr: SocketAddr, db: Arc<DB>) {
    stellarator::struc_con::tokio(move |_| async move {
        crate::assets_http::sync_schematic_assets_for_db_from_source(source_addr, &db).await;
    });
}

async fn apply_source_snapshot(
    config: &FollowConfig,
    db: &Arc<DB>,
    metadata_resp: &DumpMetadataResp,
    schema_resp: &DumpSchemaResp,
    asset_sync: SchematicAssetSync,
) -> Result<(), Error> {
    db.with_state_mut(|s| {
        s.db_config.metadata = metadata_resp.db_config.metadata.clone();
        s.db_config.default_stream_time_step = metadata_resp.db_config.default_stream_time_step;
    });
    if let Some(ts_micros) = metadata_resp.db_config.time_start_timestamp_micros() {
        let _ = db.set_earliest_timestamp(Timestamp(ts_micros));
    }
    db.save_db_state()?;
    // The wholesale replace above bypasses `apply_set_db_config`, so bump the
    // config generation here too; otherwise `SubscribeLastUpdated` clients keep
    // a stale `schematic.active`/`assets.revision` until unrelated telemetry
    // moves `last_updated` (RFD #724).
    db.db_config_gen
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    for metadata in &metadata_resp.component_metadata {
        db.set_component_metadata(metadata.clone())?;
    }

    for msg_meta in &metadata_resp.msg_metadata {
        let msg_id = impeller::types::msg_id(&msg_meta.name);
        db.with_state_mut(|s| s.set_msg_metadata(msg_id, msg_meta.clone(), &db.path))?;
    }

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

    {
        let mut followed = db.followed_components.write().unwrap();
        for &cid in schema_resp.schemas.keys() {
            followed.insert(cid);
        }
        db.has_followed_components
            .store(!followed.is_empty(), std::sync::atomic::Ordering::Release);
    }

    match asset_sync {
        SchematicAssetSync::Await => {
            #[cfg(feature = "axum")]
            sync_schematic_assets_on_tokio(config.source_addr, db.clone()).await;
        }
        SchematicAssetSync::Background => {
            #[cfg(feature = "axum")]
            spawn_schematic_asset_sync(config.source_addr, db.clone());
        }
    }

    Ok(())
}

fn apply_db_config_update(
    config: &FollowConfig,
    db: &Arc<DB>,
    update: SetDbConfig,
) -> Result<(), Error> {
    let needs_asset_sync = db.apply_set_db_config(update)?;
    #[cfg(feature = "axum")]
    if needs_asset_sync {
        spawn_schematic_asset_sync(config.source_addr, db.clone());
    }
    Ok(())
}

// Apply the bulk follow ingress without per-sample VTable decomposition.
// Notify readers only for samples that were actually appended.
pub(super) fn apply_follow_time_series(
    db: &DB,
    component_id: ComponentId,
    timestamps: &[Timestamp],
    data_buf: &[u8],
    skip: usize,
) -> Result<(), Error> {
    db.with_state(|state| {
        let Some(component) = state.components.get(&component_id) else {
            return Ok(());
        };
        let was_empty = component.time_series.index().is_empty();
        let elem_size = component.schema.size();
        let mut latest = None;
        let mut earliest = None;
        for (i, &timestamp) in timestamps.iter().enumerate().skip(skip) {
            let start = i * elem_size;
            let end = start + elem_size;
            if end > data_buf.len() {
                break;
            }
            if component
                .time_series
                .push_buf(timestamp, &data_buf[start..end])
                .is_ok()
            {
                latest = Some(timestamp);
                if timestamp.0 > 0 && earliest.is_none() {
                    earliest = Some(timestamp);
                }
            }
        }
        if let Some(latest) = latest {
            if was_empty {
                db.vtable_gen
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            let is_timestamp_source = state
                .get_component_metadata(component_id)
                .is_some_and(|metadata| metadata.is_timestamp_source());
            if !is_timestamp_source {
                db.last_updated.update_max(latest);
                if let Some(earliest) = earliest {
                    db.earliest_timestamp.update_min(earliest);
                }
            }
        }
        Ok(())
    })
}

async fn run_follower_inner(config: &FollowConfig, db: &Arc<DB>) -> Result<(), Error> {
    let (session, metadata_resp, schema_resp) = request_source_snapshot(config, true).await?;
    let FollowStreamSession {
        mut rx,
        mut buf,
        message_logs,
    } = session.expect("follow stream session");
    validate_source_message_logs(db, &message_logs)?;
    apply_source_snapshot(
        config,
        db,
        &metadata_resp,
        &schema_resp,
        SchematicAssetSync::Background,
    )
    .await?;

    let source_start_ts = metadata_resp
        .db_config
        .time_start_timestamp_micros()
        .map(Timestamp)
        .unwrap_or_else(Timestamp::now);

    let mut message_positions = std::collections::HashMap::new();

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
    let mut follow_vtables: std::collections::HashMap<PacketId, impeller::types::ComponentId> =
        std::collections::HashMap::new();

    // Track how many samples each component had at connection start.
    // On reconnection, the source resends ALL data; we skip samples we
    // already have using these counters (decremented as chunks arrive).
    let mut skip_remaining: std::collections::HashMap<impeller::types::ComponentId, usize> = db
        .with_state(|state| {
            state
                .components
                .iter()
                .map(|(&cid, comp)| (cid, comp.time_series.sample_count()))
                .collect()
        });

    loop {
        let pkt = rx.next_grow(buf).await?;
        match &pkt {
            // ComponentMetadata – apply and register as followed.
            Packet::Msg(m) if m.id == ComponentMetadata::ID => {
                let metadata: ComponentMetadata = m.parse()?;
                let cid = metadata.component_id;
                db.set_component_metadata(metadata)?;
                db.followed_components.write().unwrap().insert(cid);
                db.has_followed_components
                    .store(true, std::sync::atomic::Ordering::Release);
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
                    db.has_followed_components
                        .store(true, std::sync::atomic::Ordering::Release);
                }
            }

            // SetMsgMetadata – apply message metadata.
            Packet::Msg(m) if m.id == SetMsgMetadata::ID => {
                let meta: SetMsgMetadata = m.parse()?;
                db.with_state_mut(|s| s.set_msg_metadata(meta.id, meta.metadata, &db.path))?;
            }

            // SetDbConfig – live db_config metadata (e.g. schematic.active).
            Packet::Msg(m) if m.id == SetDbConfig::ID => {
                let update: SetDbConfig = m.parse()?;
                apply_db_config_update(config, db, update)?;
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

                apply_follow_time_series(db, component_id, timestamps, data_buf, skip)?;
            }

            // Table – decomponentize and write to local DB.
            Packet::Table(table) => {
                db.with_state(|state| -> Result<(), Error> {
                    let mut sink = DBSink {
                        components: &state.components,
                        component_metadata: &state.component_metadata,
                        last_updated: &db.last_updated,
                        earliest_timestamp: &db.earliest_timestamp,
                        sunk_new_time_series: false,
                        table_received: db.apply_implicit_timestamp(),
                        followed_components: &db.followed_components,
                        has_followed_components: db
                            .has_followed_components
                            .load(std::sync::atomic::Ordering::Acquire),
                        is_follower: true,
                        batch_max_ts: Timestamp(i64::MIN),
                        batch_min_ts: Timestamp(i64::MAX),
                        batch_has_ts: false,
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
                    sink.flush_timestamps();
                    if sink.sunk_new_time_series {
                        db.vtable_gen
                            .value
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        db.vtable_gen.wait_queue.wake_all();
                    }
                    Ok(())
                })?;
            }

            Packet::Msg(m) if m.id == FollowMessage::ID => {
                apply_follow_message(db, &mut message_positions, m.parse()?)?;
            }
            Packet::Msg(m) if m.timestamp.is_some() => {
                return Err(Error::FollowProtocolUnsupported);
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

fn validate_source_message_logs(db: &DB, source_logs: &[PacketId]) -> Result<(), Error> {
    let source_logs: std::collections::HashSet<_> = source_logs.iter().copied().collect();
    db.with_state(|state| {
        for (id, log) in &state.msg_logs {
            if !source_logs.contains(id) {
                let binding = db
                    .path
                    .join("follow_messages")
                    .join(format!("{:02x}{:02x}", id[0], id[1]));
                if read_follow_binding(&binding, log)?.is_some() {
                    return Err(Error::FollowHistoryChanged { id: *id });
                }
            }
        }
        Ok(())
    })
}

// Bind the source identity to this incarnation of the local log. Truncation
// rotates the local generation, so an empty, explicitly reset replica can bind
// again without deleting files after the truncation has already committed.
fn read_follow_binding(
    path: &std::path::Path,
    log: &crate::msg_log::MsgLog,
) -> Result<Option<[u8; 16]>, Error> {
    let Some(binding) = crate::follow_identity::read::<32>(path)? else {
        return Ok(None);
    };
    if binding[16..] != *log.follow_generation.lock().unwrap() && log.timestamps().is_empty() {
        return Ok(None);
    }
    let mut source = [0; 16];
    source.copy_from_slice(&binding[..16]);
    Ok(Some(source))
}

/// Replay validates the existing prefix by ordinal, not by timestamp or payload
/// uniqueness. Identical entries are separate occurrences. The local log itself
/// is the durable progress record, so no separately committed counter can lag it.
fn apply_follow_message(
    db: &Arc<DB>,
    positions: &mut std::collections::HashMap<PacketId, ([u8; 16], u64)>,
    message: FollowMessage,
) -> Result<(), Error> {
    match message {
        FollowMessage::Log {
            id,
            generation,
            count,
        } => {
            let directory = db.path.join("follow_messages");
            match std::fs::create_dir(&directory) {
                Ok(()) => crate::sync_dir(&db.path)?,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
            let binding = directory.join(format!("{:02x}{:02x}", id[0], id[1]));
            if positions.contains_key(&id) {
                return Err(Error::FollowHistoryChanged { id });
            }
            db.with_state_mut(|state| {
                let log = state.get_or_insert_msg_log(id, &db.path)?;
                let existing = read_follow_binding(&binding, log)?;
                if existing.is_some_and(|old| old != generation) {
                    return Err(Error::FollowHistoryChanged { id });
                }
                if log.timestamps().len() as u64 > count {
                    return Err(Error::FollowHistoryChanged { id });
                }
                if existing.is_none() {
                    let mut identity = [0; 32];
                    identity[..16].copy_from_slice(&generation);
                    identity[16..].copy_from_slice(&*log.follow_generation.lock().unwrap());
                    crate::follow_identity::write(&binding, identity)?;
                }
                Ok(())
            })?;
            positions.insert(id, (generation, 0));
        }
        FollowMessage::Entry {
            id,
            generation,
            index,
            timestamp,
            payload,
        } => {
            if positions.get(&id) != Some(&(generation, index)) {
                return Err(Error::FollowHistoryChanged { id });
            }
            db.with_state_mut(|state| {
                let log = state.get_or_insert_msg_log(id, &db.path)?;
                let local_count = log.timestamps().len() as u64;
                if index < local_count {
                    let Some((old_ts, old_payload)) = log.get_index(index as usize) else {
                        return Err(Error::FollowHistoryChanged { id });
                    };
                    if old_ts != timestamp || old_payload.as_ref() != payload.as_slice() {
                        return Err(Error::FollowHistoryChanged { id });
                    }
                } else if index == local_count {
                    log.push(timestamp, &payload)?;
                } else {
                    return Err(Error::FollowHistoryChanged { id });
                }
                Ok(())
            })?;
            db.last_updated.update_max(timestamp);
            if timestamp.0 != 0 {
                db.earliest_timestamp.update_min(timestamp);
            }
            positions.insert(id, (generation, index + 1));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn indexed_entry(
        id: PacketId,
        generation: [u8; 16],
        index: u64,
        payload: &[u8],
    ) -> FollowMessage {
        FollowMessage::Entry {
            id,
            generation,
            index,
            timestamp: Timestamp(100),
            payload: payload.to_vec(),
        }
    }

    #[test]
    fn indexed_replay_preserves_occurrences_and_detects_divergence() {
        let directory = tempfile::tempdir().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let id = [77, 2];
        let generation = [3; 16];
        let mut positions = HashMap::new();
        apply_follow_message(
            &db,
            &mut positions,
            FollowMessage::Log {
                id,
                generation,
                count: 0,
            },
        )
        .unwrap();
        for (index, payload) in [b"A", b"B", b"B"].into_iter().enumerate() {
            apply_follow_message(
                &db,
                &mut positions,
                indexed_entry(id, generation, index as u64, payload),
            )
            .unwrap();
        }
        // New connection, including a crash after append but before recording an
        // in-memory cursor. The stored log is the authoritative progress record.
        positions.clear();
        apply_follow_message(
            &db,
            &mut positions,
            FollowMessage::Log {
                id,
                generation,
                count: 3,
            },
        )
        .unwrap();
        for (index, payload) in [b"A", b"B", b"B", b"C"].into_iter().enumerate() {
            apply_follow_message(
                &db,
                &mut positions,
                indexed_entry(id, generation, index as u64, payload),
            )
            .unwrap();
        }
        db.with_state(|s| {
            let log = &s.msg_logs[&id];
            assert_eq!(log.timestamps(), &[Timestamp(100); 4]);
            for (index, payload) in [b"A", b"B", b"B", b"C"].into_iter().enumerate() {
                assert_eq!(log.get_index(index).unwrap().1.as_ref(), payload);
            }
        });
        db.flush_all().unwrap();
        let path = db.path.clone();
        drop(db);
        let db = Arc::new(DB::open(path).unwrap());
        db.push_msg(Timestamp(100), [77, 8], b"local writer")
            .unwrap();
        validate_source_message_logs(&db, &[id]).unwrap();
        assert!(matches!(
            validate_source_message_logs(&db, &[]),
            Err(Error::FollowHistoryChanged { .. })
        ));
        positions.clear();
        assert!(matches!(
            apply_follow_message(
                &db,
                &mut positions,
                FollowMessage::Log {
                    id,
                    generation: [4; 16],
                    count: 4
                }
            ),
            Err(Error::FollowHistoryChanged { .. })
        ));
        assert!(matches!(
            apply_follow_message(
                &db,
                &mut positions,
                FollowMessage::Log {
                    id,
                    generation,
                    count: 2
                }
            ),
            Err(Error::FollowHistoryChanged { .. })
        ));
        apply_follow_message(
            &db,
            &mut positions,
            FollowMessage::Log {
                id,
                generation,
                count: 4,
            },
        )
        .unwrap();
        assert!(matches!(
            apply_follow_message(
                &db,
                &mut positions,
                indexed_entry(id, generation, 0, b"different")
            ),
            Err(Error::FollowHistoryChanged { .. })
        ));
        assert!(matches!(
            apply_follow_message(&db, &mut positions, indexed_entry(id, generation, 2, b"B")),
            Err(Error::FollowHistoryChanged { .. })
        ));
    }

    #[test]
    fn truncated_follower_can_rebind_after_reopen() {
        for truncate_all in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("db");
            let db = Arc::new(DB::create(path.clone()).unwrap());
            let id = [77, 9];
            let mut positions = HashMap::new();
            apply_follow_message(
                &db,
                &mut positions,
                FollowMessage::Log {
                    id,
                    generation: [1; 16],
                    count: 0,
                },
            )
            .unwrap();
            // An empty log alone is not permission to accept a different source.
            positions.clear();
            assert!(matches!(
                apply_follow_message(
                    &db,
                    &mut positions,
                    FollowMessage::Log {
                        id,
                        generation: [2; 16],
                        count: 0
                    },
                ),
                Err(Error::FollowHistoryChanged { .. })
            ));
            apply_follow_message(
                &db,
                &mut positions,
                FollowMessage::Log {
                    id,
                    generation: [1; 16],
                    count: 1,
                },
            )
            .unwrap();
            apply_follow_message(&db, &mut positions, indexed_entry(id, [1; 16], 0, b"old"))
                .unwrap();

            if truncate_all {
                db.truncate().unwrap();
            } else {
                db.truncate_msg_log(id).unwrap();
            }
            db.flush_all().unwrap();
            drop(db);
            let db = Arc::new(DB::open(path.clone()).unwrap());
            // A removed source log must not be blocked by an obsolete binding.
            validate_source_message_logs(&db, &[]).unwrap();
            positions.clear();
            apply_follow_message(
                &db,
                &mut positions,
                FollowMessage::Log {
                    id,
                    generation: [2; 16],
                    count: 1,
                },
            )
            .unwrap();
            apply_follow_message(&db, &mut positions, indexed_entry(id, [2; 16], 0, b"new"))
                .unwrap();
            db.with_state(|state| {
                let log = &state.msg_logs[&id];
                assert_eq!(log.timestamps().len(), 1);
                assert_eq!(log.get_index(0).unwrap().1.as_ref(), b"new");
            });
            db.flush_all().unwrap();
            drop(db);
            let db = Arc::new(DB::open(path).unwrap());
            positions.clear();
            assert!(matches!(
                apply_follow_message(
                    &db,
                    &mut positions,
                    FollowMessage::Log {
                        id,
                        generation: [1; 16],
                        count: 1
                    },
                ),
                Err(Error::FollowHistoryChanged { .. })
            ));
        }
    }

    #[test]
    fn prepared_but_uncommitted_truncate_keeps_nonempty_binding() {
        let directory = tempfile::tempdir().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let id = [77, 10];
        let mut positions = HashMap::new();
        apply_follow_message(
            &db,
            &mut positions,
            FollowMessage::Log {
                id,
                generation: [1; 16],
                count: 1,
            },
        )
        .unwrap();
        apply_follow_message(&db, &mut positions, indexed_entry(id, [1; 16], 0, b"keep")).unwrap();
        // A later log's preparation can fail during truncate_all. This log's
        // generation has changed, but its samples and source binding must stay.
        db.with_state(|state| drop(state.msg_logs[&id].prepare_truncate().unwrap()));
        positions.clear();
        assert!(matches!(
            apply_follow_message(
                &db,
                &mut positions,
                FollowMessage::Log {
                    id,
                    generation: [2; 16],
                    count: 1
                },
            ),
            Err(Error::FollowHistoryChanged { .. })
        ));
        assert!(matches!(
            validate_source_message_logs(&db, &[]),
            Err(Error::FollowHistoryChanged { .. })
        ));
        db.with_state(|state| {
            assert_eq!(
                state.msg_logs[&id].get_index(0).unwrap().1.as_ref(),
                b"keep"
            )
        });
    }

    #[test]
    fn old_incomplete_replica_is_rejected_instead_of_skipping_by_count() {
        let directory = tempfile::tempdir().unwrap();
        let db = Arc::new(DB::create(directory.path().join("db")).unwrap());
        let id = [77, 3];
        db.push_msg(Timestamp(100), id, b"A").unwrap();
        db.push_msg(Timestamp(101), id, b"C").unwrap();
        let mut positions = HashMap::new();
        let generation = [7; 16];
        apply_follow_message(
            &db,
            &mut positions,
            FollowMessage::Log {
                id,
                generation,
                count: 3,
            },
        )
        .unwrap();
        apply_follow_message(&db, &mut positions, indexed_entry(id, generation, 0, b"A")).unwrap();
        assert!(matches!(
            apply_follow_message(&db, &mut positions, indexed_entry(id, generation, 1, b"B")),
            Err(Error::FollowHistoryChanged { .. })
        ));
        assert_eq!(db.with_state(|s| s.msg_logs[&id].timestamps().len()), 2);
    }

    #[stellarator::test]
    async fn legacy_source_is_rejected_before_storing_data() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            // Read the complete burst, as an old server would. The extension
            // must remain a valid legacy FollowStream, not a generic DB write.
            for request in 0..3 {
                let mut prefix = [0; 4];
                socket.read_exact(&mut prefix).unwrap();
                let mut frame = vec![0; u32::from_le_bytes(prefix) as usize];
                socket.read_exact(&mut frame).unwrap();
                if request == 2 {
                    assert_eq!(&frame[1..3], &FollowStream::ID);
                    let legacy: FollowStream = postcard::from_bytes(&frame[4..]).unwrap();
                    assert_eq!(legacy.target_packet_size, 1500);
                }
            }
            let metadata = DumpMetadataResp {
                component_metadata: Vec::new(),
                msg_metadata: Vec::new(),
                db_config: DbConfig::default(),
            }
            .with_request_id(1)
            .into_len_packet();
            let schemas = DumpSchemaResp {
                schemas: HashMap::new(),
                start_timestamps: HashMap::new(),
            }
            .with_request_id(2)
            .into_len_packet();
            let mut legacy =
                impeller::types::LenPacket::msg_with_timestamp([77, 9], Timestamp(100), 1);
            legacy.extend_from_slice(b"A");
            for packet in [metadata, schemas, legacy] {
                socket.write_all(&packet.inner).unwrap();
            }
        });
        let directory = tempfile::tempdir().unwrap();
        let db = Arc::new(DB::create(directory.path().join("follower")).unwrap());
        let result = run_follower(
            FollowConfig {
                source_addr: address,
                ..Default::default()
            },
            db.clone(),
        )
        .await;
        assert!(matches!(result, Err(Error::FollowProtocolUnsupported)));
        assert!(db.with_state(|s| s.msg_logs.is_empty()));
        server.join().unwrap();
    }

    #[stellarator::test]
    async fn snapshot_larger_than_initial_buffer_is_received() {
        use stellarator::struc_con::{Joinable, ThreadBuilder};
        let directory = tempfile::tempdir().unwrap();
        let listener = stellarator::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            crate::Server::from_listener(listener, directory.path().join("source")).unwrap();
        let value = "x".repeat(9 * 1024 * 1024);
        server
            .db
            .apply_set_db_config(SetDbConfig {
                metadata: HashMap::from([("large".into(), value.clone())]),
                ..Default::default()
            })
            .unwrap();
        let cancel = stellarator::util::CancelToken::default();
        let task = ThreadBuilder::default()
            .cancel_token(cancel.clone())
            .stellar(move || async { server.run().await });
        let config = FollowConfig {
            source_addr: address,
            ..Default::default()
        };
        for start_stream in [false, true] {
            let (session, metadata, _) = request_source_snapshot(&config, start_stream)
                .await
                .unwrap();
            assert!(
                metadata.db_config.metadata["large"] == value,
                "large snapshot was changed"
            );
            assert_eq!(session.is_some(), start_stream);
        }
        cancel.cancel();
        assert!(task.join().await.unwrap().is_none());
    }

    #[stellarator::test]
    async fn oversized_follow_packet_is_terminal_without_allocating_payload() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            // Drain the three requests before closing, so an unread request
            // cannot reset the connection and hide the oversized response.
            for _ in 0..3 {
                let mut prefix = [0; 4];
                socket.read_exact(&mut prefix).unwrap();
                let mut request = vec![0; u32::from_le_bytes(prefix) as usize];
                socket.read_exact(&mut request).unwrap();
            }
            socket
                .write_all(&((crate::MAX_PACKET_LEN + 1) as u32).to_le_bytes())
                .unwrap();
        });
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(DB::create(dir.path().join("db")).unwrap());
        let config = FollowConfig {
            source_addr: address,
            ..Default::default()
        };
        let result =
            futures_lite::future::race(async { Some(run_follower(config, db).await) }, async {
                stellarator::sleep(Duration::from_secs(5)).await;
                None
            })
            .await;
        assert!(
            matches!(result, Some(Err(Error::Stellar(stellarator::Error::PacketTooLarge { len, max }))) if len == crate::MAX_PACKET_LEN + 1 && max == crate::MAX_PACKET_LEN)
        );
        server.join().unwrap();
    }

    #[stellarator::test]
    async fn source_snapshot_replaces_stale_db_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(DB::create(dir.path().join("db")).unwrap());
        db.apply_set_db_config(SetDbConfig {
            metadata: HashMap::from([
                (
                    "schematic.active".to_string(),
                    "schematics/old.kdl".to_string(),
                ),
                ("skybox.active".to_string(), "old_skybox".to_string()),
                ("local.only".to_string(), "stale".to_string()),
            ]),
            ..Default::default()
        })
        .unwrap();

        let mut source_config = DbConfig::default();
        source_config
            .metadata
            .insert("source.only".to_string(), "fresh".to_string());
        let metadata_resp = DumpMetadataResp {
            component_metadata: Vec::new(),
            msg_metadata: Vec::new(),
            db_config: source_config,
        };
        let schema_resp = DumpSchemaResp {
            schemas: HashMap::new(),
            start_timestamps: HashMap::new(),
        };
        let config = FollowConfig::default();

        let gen0 = db.db_config_gen.latest();
        apply_source_snapshot(
            &config,
            &db,
            &metadata_resp,
            &schema_resp,
            SchematicAssetSync::Await,
        )
        .await
        .unwrap();

        // The snapshot must wake `SubscribeLastUpdated` config pushes; without
        // the bump clients keep the stale config until telemetry moves.
        assert!(db.db_config_gen.latest() > gen0);

        db.with_state(|state| {
            assert_eq!(
                state
                    .db_config
                    .metadata
                    .get("source.only")
                    .map(String::as_str),
                Some("fresh")
            );
            assert!(!state.db_config.metadata.contains_key("schematic.active"));
            assert!(!state.db_config.metadata.contains_key("skybox.active"));
            assert!(!state.db_config.metadata.contains_key("local.only"));
        });
    }
}
