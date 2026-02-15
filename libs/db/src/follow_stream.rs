//! Source-side handler for the `FollowStream` protocol message.
//!
//! When a follower sends `FollowStream { target_packet_size }`, this module
//! streams ALL data (component metadata, schemas, table data, and messages)
//! through a [`CoalescingSink`] that batches small per-component packets
//! into TCP writes of approximately `target_packet_size` bytes.
//!
//! Each component gets its own VTable (one field: timestamp + data), producing
//! small table packets that the [`CoalescingSink`] can actually batch.  This
//! ensures the `--follow-packet-size` flag controls the real TCP write size.

use std::{
    collections::{HashMap, HashSet},
    mem::size_of,
    sync::Arc,
    time::Duration,
};

use impeller2::{
    types::{ComponentId, IntoLenPacket, LenPacket, PacketId, PrimType, Timestamp},
    vtable::builder::{component, raw_field, raw_table, schema, timestamp, vtable},
};
use impeller2_wkt::*;
use stellarator::io::AsyncWrite;
use tracing::info;

use crate::{Component, DB, Error, coalescing_sink::CoalescingSink};

/// Default flush interval for the coalescing sink.
const FLUSH_INTERVAL: Duration = Duration::from_millis(5);

/// Per-component VTable info used for sending small table packets.
struct PerComponentVTable {
    vtable_id: PacketId,
    prim_type: PrimType,
    elem_size: usize,
}

/// Run the unified follow-stream handler on the source side.
pub async fn handle_follow_stream<W: AsyncWrite>(
    db: Arc<DB>,
    writer: &W,
    target_packet_size: u32,
    req_id: u8,
) -> Result<(), Error> {
    let target = target_packet_size as usize;
    let mut sink = CoalescingSink::new(writer, target, FLUSH_INTERVAL);

    let mut current_gen = u64::MAX;
    let mut components: HashMap<ComponentId, Component> = HashMap::new();
    let mut component_vtables: HashMap<ComponentId, PerComponentVTable> = HashMap::new();
    let mut known_msg_ids: HashSet<PacketId> = HashSet::new();

    // Track last-sent timestamp per component so we only send newer data.
    let mut component_last_sent: HashMap<ComponentId, Timestamp> = HashMap::new();

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
                for comp in state.components.values() {
                    if components.contains_key(&comp.component_id) {
                        continue;
                    }
                    let metadata = state.get_component_metadata(comp.component_id).cloned();
                    let sch = comp.schema.to_schema();
                    new_comps.push((comp.clone(), metadata, sch));
                }
                new_comps
            });

            for (comp, metadata, sch) in new_components {
                let cid = comp.component_id;

                // Send ComponentMetadata
                if let Some(metadata) = metadata {
                    sink.send(metadata.into_len_packet().with_request_id(req_id))
                        .await?;
                }

                // Send DumpSchemaResp for this component
                let start_ts = *comp.index_extra();
                let schema_msg = DumpSchemaResp {
                    schemas: [(cid, sch)].into_iter().collect(),
                    start_timestamps: [(cid, start_ts)].into_iter().collect(),
                };
                sink.send(schema_msg.into_len_packet().with_request_id(req_id))
                    .await?;

                // Create and send a per-component VTable
                let prim_type = comp.schema.prim_type;
                let elem_size = comp.schema.size();
                let timestamp_loc = raw_table(0, size_of::<Timestamp>() as u16);
                let per_vtable = vtable([raw_field(
                    (prim_type.padding(8) + size_of::<Timestamp>()) as u16,
                    elem_size as u16,
                    timestamp(
                        timestamp_loc,
                        schema(prim_type, &comp.schema.shape(), component(cid)),
                    ),
                )]);
                let vtable_id: PacketId = fastrand::u16(..).to_le_bytes();
                sink.send(
                    VTableMsg {
                        id: vtable_id,
                        vtable: per_vtable,
                    }
                    .into_len_packet()
                    .with_request_id(req_id),
                )
                .await?;

                component_vtables.insert(
                    cid,
                    PerComponentVTable {
                        vtable_id,
                        prim_type,
                        elem_size,
                    },
                );
                components.insert(cid, comp);
            }

            // Flush metadata/schema/VTable messages.
            sink.flush().await?;

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
            msg_positions.insert(msg_id, Timestamp(i64::MIN));
            known_msg_ids.insert(msg_id);
        }

        // ── 3. Component data (per-component packets) ───────────────────
        // For each component, send samples newer than the last-sent
        // timestamp.  A byte budget per iteration prevents overwhelming
        // the TCP send buffer (which would block write_all and stall the
        // loop).  During catch-up the budget is consumed quickly; once
        // caught up to real-time it's more than enough for a single new
        // sample per component.
        let byte_budget = target * 10; // ~15 KB at default 1500 MTU
        let mut bytes_sent = 0usize;
        'components: for (cid, comp) in components.iter() {
            let info = match component_vtables.get(cid) {
                Some(v) => v,
                None => continue,
            };
            let last = component_last_sent
                .get(cid)
                .copied()
                .unwrap_or(Timestamp(i64::MIN));
            let range = Timestamp(last.0.saturating_add(1))..Timestamp(i64::MAX);
            let Some((timestamps, data)) = comp.time_series.get_range(&range) else {
                continue;
            };
            if timestamps.is_empty() {
                continue;
            }
            for (i, &ts) in timestamps.iter().enumerate() {
                if bytes_sent >= byte_budget {
                    // Save progress for this component so we resume here
                    // on the next iteration.
                    if i > 0 {
                        component_last_sent.insert(*cid, timestamps[i - 1]);
                    }
                    break 'components;
                }
                let start = i * info.elem_size;
                let end = start + info.elem_size;
                if end > data.len() {
                    break;
                }
                let pkt_size = size_of::<Timestamp>() + info.elem_size + 8;
                let mut pkt = LenPacket::table(info.vtable_id, pkt_size);
                pkt.push_aligned(ts);
                pkt.pad_for_type(info.prim_type);
                pkt.extend_from_slice(&data[start..end]);
                pkt.set_request_id(req_id);
                sink.send(pkt).await?;
                bytes_sent += pkt_size + 8; // +8 for LenPacket header
            }
            component_last_sent.insert(*cid, *timestamps.last().unwrap());
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
        futures_lite::future::race(db.last_updated.wait(), stellarator::sleep(FLUSH_INTERVAL))
            .await;
    }
}
