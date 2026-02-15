//! Source-side handler for the `FollowStream` protocol message.
//!
//! When a follower sends `FollowStream { target_packet_size }`, this module
//! streams ALL data (component metadata, schemas, table data, and messages)
//! through a [`CoalescingSink`] that batches small packets into writes of
//! approximately `target_packet_size` bytes.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use impeller2::types::{IntoLenPacket, LenPacket, PacketId, Timestamp};
use impeller2_wkt::*;
use stellarator::io::AsyncWrite;
use tracing::info;

use crate::{Component, DB, DBVisitor, Error, coalescing_sink::CoalescingSink};

/// Default flush interval for the coalescing sink.
const FLUSH_INTERVAL: Duration = Duration::from_millis(5);

/// Run the unified follow-stream handler on the source side.
///
/// This function never returns under normal operation – it streams data
/// continuously until the connection is closed.
///
/// `writer` is the raw writer extracted from the connection's `PacketSink`
/// (via `PacketSink::writer()`).  We wrap it in a [`CoalescingSink`] to
/// coalesce small packets into target-sized TCP writes.
pub async fn handle_follow_stream<W: AsyncWrite>(
    db: Arc<DB>,
    writer: &W,
    target_packet_size: u32,
    req_id: u8,
) -> Result<(), Error> {
    let target = target_packet_size as usize;
    let mut sink = CoalescingSink::new(writer, target, FLUSH_INTERVAL);

    let mut current_gen = u64::MAX;
    let mut table = LenPacket::table([0; 2], 2048 - 16);
    let mut components: HashMap<impeller2::types::ComponentId, Component> = HashMap::new();
    let mut known_msg_ids: HashSet<PacketId> = HashSet::new();

    // Track last-sent timestamp per component so we only send newer data
    // (avoids re-sending data the follower already backfilled).
    let mut component_last_sent: HashMap<impeller2::types::ComponentId, Timestamp> = HashMap::new();

    // Track last-sent timestamp per msg-log so we only send new messages.
    let mut msg_positions: HashMap<PacketId, Timestamp> = HashMap::new();

    info!(
        target_packet_size,
        "follow stream started – coalescing to {} byte target", target_packet_size
    );

    loop {
        // ── 1. Component discovery ──────────────────────────────────────
        let vtable_gen = db.vtable_gen.latest();
        if vtable_gen != current_gen {
            let new_components: Vec<(
                Component,
                Option<ComponentMetadata>,
                impeller2::schema::Schema<Vec<u64>>,
            )> = db.with_state(|state| {
                let mut new_comps = Vec::new();
                for component in state.components.values() {
                    if components.contains_key(&component.component_id) {
                        continue;
                    }
                    let metadata = state
                        .get_component_metadata(component.component_id)
                        .cloned();
                    let schema = component.schema.to_schema();
                    new_comps.push((component.clone(), metadata, schema));
                }
                new_comps
            });

            for (component, metadata, schema) in new_components {
                // Send ComponentMetadata
                if let Some(metadata) = metadata {
                    sink.send(metadata.into_len_packet().with_request_id(req_id))
                        .await?;
                }

                // Send DumpSchemaResp for this component
                let schema_msg = DumpSchemaResp {
                    schemas: [(component.component_id, schema)].into_iter().collect(),
                };
                sink.send(schema_msg.into_len_packet().with_request_id(req_id))
                    .await?;

                components.insert(component.component_id, component);
            }

            // Rebuild the VTable with the full component set and send it.
            let vtable_msg = DBVisitor.vtable(&components)?;
            let id: PacketId = fastrand::u16(..).to_le_bytes();
            table = LenPacket::table(id, 2048 - 16);
            sink.send(
                VTableMsg {
                    id,
                    vtable: vtable_msg,
                }
                .into_len_packet()
                .with_request_id(req_id),
            )
            .await?;

            current_gen = vtable_gen;
        }

        // ── 2. Discover new message logs ────────────────────────────────
        let new_msgs: Vec<(PacketId, Option<MsgMetadata>)> = db.with_state(|state| {
            state
                .msg_logs
                .iter()
                .filter(|(id, _)| !known_msg_ids.contains(*id))
                .map(|(id, log)| (*id, log.metadata().cloned()))
                .collect()
        });
        for (msg_id, metadata) in new_msgs {
            if let Some(metadata) = metadata {
                sink.send(
                    SetMsgMetadata {
                        id: msg_id,
                        metadata,
                    }
                    .into_len_packet()
                    .with_request_id(req_id),
                )
                .await?;
            }
            // Start tracking from the latest timestamp so we only send
            // newly arriving messages (backfill is handled in Phase 1).
            let latest_ts = db
                .with_state(|state| {
                    state
                        .msg_logs
                        .get(&msg_id)
                        .and_then(|log| log.latest().map(|(ts, _)| ts))
                })
                .unwrap_or(Timestamp(i64::MIN));
            msg_positions.insert(msg_id, latest_ts);
            known_msg_ids.insert(msg_id);
        }

        // ── 3. Component data (latest values) ──────────────────────────
        // Check if any component has new data before building the table.
        let any_new = components.iter().any(|(_, component)| {
            let Some((&ts, _)) = component.time_series.latest() else {
                return false;
            };
            let last = component_last_sent
                .get(&component.component_id)
                .copied()
                .unwrap_or(Timestamp(i64::MIN));
            ts > last
        });

        if any_new {
            table.clear();
            DBVisitor.populate_table_latest(&components, &mut table);

            // Track what we just sent.
            for (_, component) in components.iter() {
                if let Some((&ts, _)) = component.time_series.latest() {
                    component_last_sent.insert(component.component_id, ts);
                }
            }

            if table.inner.len() > 8 {
                table.set_request_id(req_id);
                table = sink.send_reusable(table).await?;
            }
        }

        // ── 4. New message data ─────────────────────────────────────────
        for &msg_id in &known_msg_ids {
            let last_ts = msg_positions
                .get(&msg_id)
                .copied()
                .unwrap_or(Timestamp(i64::MIN));
            let updates: Vec<(Timestamp, Vec<u8>)> = db.with_state(|state| {
                let Some(log) = state.msg_logs.get(&msg_id) else {
                    return vec![];
                };
                let range = Timestamp(last_ts.0.saturating_add(1))..Timestamp(i64::MAX);
                log.get_range(&range)
                    .map(|(ts, buf)| (ts, buf.to_vec()))
                    .collect()
            });

            if !updates.is_empty() {
                let new_last = updates.last().unwrap().0;
                msg_positions.insert(msg_id, new_last);

                for (timestamp, data) in updates {
                    let mut pkt = LenPacket::msg_with_timestamp(msg_id, timestamp, data.len());
                    pkt.extend_from_slice(&data);
                    pkt.set_request_id(req_id);
                    sink.send(pkt).await?;
                }
            }
        }

        // ── 5. Flush timer check ────────────────────────────────────────
        sink.maybe_flush().await?;

        // ── 6. Wait for next change ─────────────────────────────────────
        // Race between component data change and a short sleep so we
        // periodically check for new message logs / flush the sink.
        futures_lite::future::race(db.last_updated.wait(), stellarator::sleep(FLUSH_INTERVAL))
            .await;
    }
}
