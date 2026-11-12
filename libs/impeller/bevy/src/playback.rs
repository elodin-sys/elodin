//! Fixed recording blocks; disk replay belongs to the transport.
use std::ops::Range;

use super::prefetch::{self, VisiblePrefetchState};
use super::*;
use impeller_wkt::GetTimeSeries;

pub const BLOCK_US: i64 = 1_000_000;
pub const PAGE_LIMIT: usize = 4096;

pub fn block_start(timestamp: i64) -> i64 {
    timestamp.div_euclid(BLOCK_US).saturating_mul(BLOCK_US)
}

/// Inclusive DB range for one block. The range end is inclusive.
pub fn block_query_range(start: i64) -> Range<Timestamp> {
    let end = start.saturating_add(BLOCK_US).saturating_sub(1);
    Timestamp(start)..Timestamp(end)
}

pub fn block_query(component_id: ComponentId, start: i64, limit: usize) -> GetTimeSeries {
    GetTimeSeries {
        id: PacketId::default(),
        range: block_query_range(start),
        component_id,
        limit: Some(limit),
    }
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

/// The playhead block, one second behind, and three seconds ahead.
const PLAYHEAD_OFFSETS: [i64; 5] = [0, 1, -1, 2, 3];

pub fn update(
    current: Timestamp,
    limits: &PlaybackCacheLimits,
    priority: &SeriesFetchPriority,
    schema: &ComponentSchemaRegistry,
    prefetch: &mut VisiblePrefetchState,
    cache: &mut TelemetryCache,
    commands: &mut Commands,
) {
    cache.set_playback_budget(limits.telemetry_bytes, current);
    let center = block_start(current.0);
    for id in series_store_backfill_candidates(&priority.high, schema) {
        if !schema.0.contains_key(&id) {
            continue;
        }
        prefetch::prefetch_hold_anchor(id, Timestamp(center), cache, prefetch, commands);
        for offset in PLAYHEAD_OFFSETS {
            if !prefetch.has_capacity() {
                return;
            }
            let start = center.saturating_add(offset.saturating_mul(BLOCK_US));
            prefetch::request_block(id, start, cache, prefetch, commands);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller::types::IntoLenPacket;
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

    #[test]
    fn large_components_request_a_page_that_fits() {
        let mut cache = TelemetryCache::default();
        assert_eq!(
            cache.fitting_page_limit(10_000, Some(PAGE_LIMIT), 0),
            Some(PAGE_LIMIT)
        );
        assert_eq!(cache.fitting_page_limit(10_000, None, 0), None);
        assert_eq!(cache.fitting_page_limit(10_000, Some(0), 0), Some(0));
        cache.set_playback_budget(64 * 1024, Timestamp(0));
        let sample = 64usize;
        let limit = cache
            .fitting_page_limit(sample, Some(PAGE_LIMIT), 0)
            .expect("one sample fits");
        assert!(limit < PAGE_LIMIT);
        assert!(cache.admit_block(0, sample.saturating_mul(limit)));
        assert!(!cache.admit_block(1_000_000, sample.saturating_mul(PAGE_LIMIT)));
        assert_eq!(cache.fitting_page_limit(sample, Some(4), 0), Some(4));
        assert_eq!(cache.fitting_page_limit(sample, Some(0), 0), Some(0));
        assert_eq!(
            cache.fitting_page_limit(sample, None, 0),
            Some(64 * 1024 / sample)
        );
        assert!(
            cache
                .fitting_page_limit(64 * 1024 + 1, Some(PAGE_LIMIT), 0)
                .is_none()
        );
    }

    #[test]
    fn plot_and_playback_predecessors_stay_independent() {
        let mut cache = TelemetryCache::default();
        let id = ComponentId(7);
        let playhead = ComponentValue::F64(nox::array![1f64].to_dyn());
        let plot = ComponentValue::F64(nox::array![2f64].to_dyn());
        let sample = TelemetryCache::value_bytes(&playhead);
        cache.set_playback_budget(sample * 2, Timestamp(0));
        cache.insert_predecessor(id, Timestamp(-120_000_000), 0, playhead.clone());
        cache.insert_predecessor(id, Timestamp(-120_000_000), 5_000_000, playhead);
        cache.drop_block(id, 5_000_000);
        assert!(
            cache
                .series(&id)
                .is_some_and(|series| series.contains_key(&Timestamp(-120_000_000)))
        );
        cache.insert_predecessor(id, Timestamp(-30_000_000), 5_000_000, plot);
        assert!(cache.admit_block(1_000_000, sample));
        let series = cache.series(&id).expect("both holds were stored");
        assert!(series.contains_key(&Timestamp(-120_000_000)));
        assert!(!series.contains_key(&Timestamp(-30_000_000)));
    }

    #[test]
    fn page_after_predecessor_leaves_room_for_the_block() {
        let mut cache = TelemetryCache::default();
        let id = ComponentId(7);
        let value = ComponentValue::F64(nox::array![1f64].to_dyn());
        let sample = TelemetryCache::value_bytes(&value);
        cache.set_playback_budget(sample * 4, Timestamp(0));
        cache.insert_predecessor(id, Timestamp(-120_000_000), 0, value);
        let limit = cache
            .fitting_page_limit(sample, Some(PAGE_LIMIT), 0)
            .expect("remaining budget fits a page");
        assert_eq!(limit, 2);
        assert!(limit.is_power_of_two());
        assert!(cache.admit_block(0, sample * limit));
        assert!(cache.admit_block(5_000_000, sample * limit));
        assert!(!cache.admit_block(0, sample * 4));
        assert!(
            cache
                .series(&id)
                .is_some_and(|series| series.contains_key(&Timestamp(-120_000_000)))
        );
    }

    #[test]
    fn cleared_prefetch_does_not_stick_a_component() {
        let mut prefetch = VisiblePrefetchState::default();
        let key = prefetch::PrefetchKey::Window {
            component_id: ComponentId(7),
            start: 0,
            end: BLOCK_US,
        };
        assert!(prefetch.begin(key).is_some());
        assert!(prefetch.begin(key).is_none());
        prefetch.clear_in_flight();
        assert!(prefetch.begin(key).is_some());
    }

    #[test]
    fn playback_and_plot_blocks_share_one_request() {
        let id = ComponentId(7);
        let limit = 256;
        let playback = block_query(id, BLOCK_US, limit);
        let plot = block_query(id, BLOCK_US, limit);
        assert_eq!(
            (&playback).into_len_packet().inner,
            (&plot).into_len_packet().inner
        );
        let again = block_query(id, BLOCK_US, limit);
        assert_eq!(
            again.range,
            Timestamp(BLOCK_US)..Timestamp(BLOCK_US * 2 - 1)
        );
        assert_eq!(again.limit, Some(limit));
    }

    #[test]
    fn retained_bytes_match_a_recount() {
        let mut cache = TelemetryCache::default();
        let id = ComponentId(7);
        let near = ComponentValue::F64(nox::array![1f64].to_dyn());
        let far = ComponentValue::F64(nox::array![2f64].to_dyn());
        let hold = ComponentValue::F64(nox::array![3f64].to_dyn());
        let sample = TelemetryCache::value_bytes(&near);
        cache.set_playback_budget(sample * 8, Timestamp(0));
        cache.insert(id, Timestamp(0), near);
        cache.insert(id, Timestamp(5_000_000), far);
        assert_eq!(cache.retained_bytes(0), sample);
        assert_eq!(cache.retained_bytes(5_000_000), sample * 2);
        cache.insert_predecessor(id, Timestamp(-120_000_000), 0, hold.clone());
        cache.insert_predecessor(id, Timestamp(-120_000_000), 5_000_000, hold);
        assert_eq!(cache.retained_bytes(0), sample * 2);
        assert_eq!(recount(&cache, 0), cache.retained_bytes(0));
        cache.drop_block(id, 5_000_000);
        assert!(
            cache
                .series(&id)
                .is_some_and(|series| series.contains_key(&Timestamp(-120_000_000)))
        );
        assert!(
            cache
                .series(&id)
                .is_some_and(|series| !series.contains_key(&Timestamp(5_000_000)))
        );
        assert_eq!(cache.retained_bytes(0), sample * 2);
        assert_eq!(recount(&cache, 0), cache.retained_bytes(0));
        cache.drop_block(id, 0);
        assert!(
            cache
                .series(&id)
                .is_none_or(|series| !series.contains_key(&Timestamp(0)))
        );
        assert_eq!(recount(&cache, 0), cache.retained_bytes(0));
        assert_eq!(cache.retained_bytes(0), 0);
    }

    fn recount(cache: &TelemetryCache, start: i64) -> usize {
        let wanted = start.abs_diff(cache.playback_pivot().0);
        cache
            .components
            .iter()
            .flat_map(|(id, series)| {
                let claims = cache.predecessors.get(id);
                series.iter().filter_map(move |(time, value)| {
                    let kept = match claims {
                        Some(claims) if claims.values().any(|claimed| claimed == time) => {
                            claims.iter().any(|(block, claimed)| {
                                claimed == time
                                    && block.abs_diff(cache.playback_pivot().0) <= wanted
                            })
                        }
                        _ => block_start(time.0).abs_diff(cache.playback_pivot().0) <= wanted,
                    };
                    kept.then_some(TelemetryCache::value_bytes(value))
                })
            })
            .fold(0, usize::saturating_add)
    }
}
