//! Fixed recording blocks; disk replay belongs to the transport.
use super::*;
use impeller_wkt::GetTimeSeriesPredecessor;

pub const BLOCK_US: i64 = 1_000_000;
pub const PAGE_LIMIT: usize = 4096;

pub fn block_start(timestamp: i64) -> i64 {
    timestamp.div_euclid(BLOCK_US).saturating_mul(BLOCK_US)
}

#[derive(Resource, Clone, Debug)]
pub struct PlaybackCacheLimits {
    pub telemetry_bytes: usize,
    pub video_bytes: usize,
    pub decoded_bytes: usize,
}
impl Default for PlaybackCacheLimits {
    fn default() -> Self {
        Self {
            telemetry_bytes: 64 * 1024 * 1024,
            video_bytes: 192 * 1024 * 1024,
            decoded_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Resource, Default)]
pub struct PlaybackPager {
    pending: HashSet<ComponentId>,
    anchors: HashMap<(ComponentId, i64), Option<Timestamp>>,
}

#[allow(clippy::too_many_arguments)]
pub fn update(
    current: Timestamp,
    limits: &PlaybackCacheLimits,
    priority: &SeriesFetchPriority,
    schema: &ComponentSchemaRegistry,
    pager: &mut PlaybackPager,
    cache: &mut TelemetryCache,
    _backfill: &mut BackfillState,
    commands: &mut Commands,
) {
    cache.set_playback_budget(limits.telemetry_bytes, current);
    let center = block_start(current.0);
    pager.anchors.retain(|(id, block), time| {
        priority.high.contains(id)
            && *block == center
            && time.is_none_or(|time| {
                cache
                    .series(id)
                    .is_some_and(|series| series.contains_key(&time))
            })
    });
    for id in series_store_backfill_candidates(&priority.high, schema) {
        if pager.pending.len() >= 4 {
            break;
        }
        if pager.pending.contains(&id) {
            continue;
        }
        let anchor = !pager.anchors.contains_key(&(id, center));
        let start = if anchor {
            center
        } else {
            let Some(start) = [0, 1, -1, 2, 3]
                .into_iter()
                .map(|offset| center.saturating_add(offset * BLOCK_US))
                .find(|start| {
                    !cache.is_covered(
                        &id,
                        &(Timestamp(*start)..Timestamp(start.saturating_add(BLOCK_US))),
                    )
                })
            else {
                continue;
            };
            start
        };
        let bytes = schema
            .0
            .get(&id)
            .map_or(8, |schema| schema.size().saturating_add(8));
        if !cache.admit_block(
            start,
            bytes.saturating_mul(if anchor { 1 } else { PAGE_LIMIT }),
        ) {
            continue;
        }
        pager.pending.insert(id);
        send(id, start, start.saturating_add(BLOCK_US), anchor, commands);
    }
}

fn send(id: ComponentId, start: i64, end: i64, anchor: bool, commands: &mut Commands) {
    let callback = move |packet: InRef<OwnedPacket<PacketGrantR>>,
                         mut pager: ResMut<PlaybackPager>,
                         mut cache: ResMut<TelemetryCache>,
                         schema: Res<ComponentSchemaRegistry>,
                         priority: Res<SeriesFetchPriority>,
                         mut commands: Commands| {
        pager.pending.remove(&id);
        if !priority.high.contains(&id) {
            return true;
        }
        let OwnedPacket::TimeSeries(series) = &*packet else {
            return true;
        };
        let Some(schema) = schema.0.get(&id) else {
            return true;
        };
        let (Ok(times), Ok(data)) = (series.timestamps(), series.data()) else {
            return true;
        };
        let size = schema.size();
        if size == 0
            || data.len() != times.len().saturating_mul(size)
            || times.windows(2).any(|pair| pair[0] >= pair[1])
            || times.iter().any(|time| {
                if anchor {
                    time.0 > start
                } else {
                    time.0 < start || time.0 >= end
                }
            })
        {
            return true;
        }
        let block = block_start(start);
        if !cache.admit_block(block, times.len().saturating_mul(size.saturating_add(8))) {
            return true;
        }
        for (&time, bytes) in times.iter().zip(data.chunks_exact(size)) {
            let Ok(view) =
                ComponentView::try_from_bytes_shape(bytes, schema.shape(), schema.prim_type())
            else {
                return true;
            };
            if anchor {
                cache.insert_predecessor(id, time, block, ComponentValue::from_view(view));
            } else {
                cache.insert(id, time, ComponentValue::from_view(view));
            }
        }
        if anchor {
            pager.anchors.insert((id, block), times.first().copied());
        } else {
            let covered_end = if times.len() < PAGE_LIMIT {
                end
            } else {
                times.last().map_or(start, |last| last.0.saturating_add(1))
            };
            cache.mark_page_covered(id, Timestamp(start), Timestamp(covered_end), times);
            if covered_end < end {
                pager.pending.insert(id);
                send(id, covered_end, end, false, &mut commands);
            }
        }
        true
    };
    if anchor {
        commands.send_req_reply_raw(
            GetTimeSeriesPredecessor {
                id: PacketId::default(),
                timestamp: Timestamp(start),
                component_id: id,
            },
            callback,
        );
    } else {
        commands.send_req_reply_raw(
            GetTimeSeries {
                id: PacketId::default(),
                range: Timestamp(start)..Timestamp(end.saturating_sub(1)),
                component_id: id,
                limit: Some(PAGE_LIMIT),
            },
            callback,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocks_use_euclidean_seconds() {
        assert_eq!(block_start(1_250_001), 1_000_000);
        assert_eq!(block_start(-1), -1_000_000);
    }
    #[test]
    fn sparse_predecessors_do_not_trade_places_with_far_plot_blocks() {
        let mut cache = TelemetryCache::default();
        let id = ComponentId(7);
        let value = ComponentValue::F64(nox::array![1f64].to_dyn());
        let cost = TelemetryCache::value_bytes(&value);
        cache.set_playback_budget(cost, Timestamp(0));
        cache.insert(id, Timestamp(-60_000_000), value.clone());
        cache.mark_covered(id, Timestamp(-60_000_000), Timestamp(-59_000_000));
        cache.insert_predecessor(id, Timestamp(-60_000_000), 0, value);
        assert!(!cache.admit_block(5_000_000, cost));
        assert!(cache.get_at_or_before(&id, Timestamp(0)).is_some());
        cache.drop_block(id, 0);
        assert_eq!(cache.resident_bytes(), 0);
        assert!(!cache.is_covered(&id, &(Timestamp(-60_000_000)..Timestamp(-59_000_000))));
    }

    #[test]
    fn admission_only_replaces_strictly_farther_blocks() {
        let id = ComponentId(7);
        let value = ComponentValue::F64(nox::array![1f64].to_dyn());
        let cost = TelemetryCache::value_bytes(&value);
        let mut cache = TelemetryCache::default();
        cache.insert(id, Timestamp(60_000_000), value.clone());
        cache.mark_covered(id, Timestamp(60_000_000), Timestamp(61_000_000));
        cache.set_playback_budget(cost, Timestamp(0));
        assert!(cache.admit_block(5_000_000, cost));
        assert_eq!(cache.resident_bytes(), 0);
        assert!(!cache.is_covered(&id, &(Timestamp(60_000_000)..Timestamp(61_000_000))));
        cache.insert(id, Timestamp(2_000_000), value);
        assert!(!cache.admit_block(5_000_000, cost));
        assert!(!cache.admit_block(-2_000_000, cost));
        cache.drop_block(id, 2_000_000);
        assert_eq!(cache.resident_bytes(), 0);
    }
}
