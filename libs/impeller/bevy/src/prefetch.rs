//! In-flight plot and playback page requests.
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::time::Duration;

use bevy::prelude::{Commands, InRef, Res, ResMut, Resource};
use impeller::types::{ComponentId, ComponentView, OwnedPacket, PacketId, Timestamp};
use impeller_wkt::{ComponentValue, GetTimeSeries, GetTimeSeriesPredecessor};
use web_time::Instant;

use super::playback::{self, block_query_range};
use super::{
    CommandsExt, ComponentSchemaRegistry, PacketGrantR, SeriesFetchPriority, TelemetryCache,
};

#[cfg(target_family = "wasm")]
pub const VISIBLE_PREFETCH_LIMIT: usize = 4096;
#[cfg(not(target_family = "wasm"))]
pub const VISIBLE_PREFETCH_LIMIT: usize = 8192;
const MAX_VISIBLE_PREFETCH_IN_FLIGHT: usize = 32;
const PREFETCH_RETRY_DELAY: Duration = Duration::from_millis(100);

/// Identifies logical sparse-prefetch requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefetchKey {
    Window {
        component_id: ComponentId,
        start: i64,
        end: i64,
    },
    Anchor {
        component_id: ComponentId,
        start: i64,
    },
}

impl PrefetchKey {
    pub fn component_id(self) -> ComponentId {
        match self {
            Self::Window { component_id, .. } | Self::Anchor { component_id, .. } => component_id,
        }
    }

    fn same_slot(self, other: Self) -> bool {
        self.component_id() == other.component_id()
            && matches!(
                (self, other),
                (Self::Window { .. }, Self::Window { .. })
                    | (Self::Anchor { .. }, Self::Anchor { .. })
            )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefetchRequestState {
    InFlight(u64),
    RetryAt(Instant),
}

#[derive(Resource, Default)]
pub struct VisiblePrefetchState {
    pub requests: HashMap<PrefetchKey, PrefetchRequestState>,
    next_attempt: u64,
}

impl VisiblePrefetchState {
    pub fn clear_in_flight(&mut self) {
        self.requests.clear();
    }

    pub fn request_count(&self) -> usize {
        self.requests
            .values()
            .filter(|state| matches!(state, PrefetchRequestState::InFlight(_)))
            .count()
    }

    pub fn has_capacity(&self) -> bool {
        self.request_count() < MAX_VISIBLE_PREFETCH_IN_FLIGHT
    }

    fn drop_superseded_retries(&mut self, key: PrefetchKey) {
        self.requests.retain(|existing, state| {
            *existing == key
                || !existing.same_slot(key)
                || matches!(state, PrefetchRequestState::InFlight(_))
        });
    }

    pub fn begin(&mut self, key: PrefetchKey) -> Option<u64> {
        self.drop_superseded_retries(key);
        let now = Instant::now();
        if self.requests.iter().any(|(existing, state)| {
            existing.same_slot(key) && matches!(state, PrefetchRequestState::InFlight(_))
        }) {
            return None;
        }
        if self.requests.get(&key).is_some_and(
            |state| matches!(state, PrefetchRequestState::RetryAt(retry_at) if now < *retry_at),
        ) {
            return None;
        }
        self.next_attempt = self.next_attempt.wrapping_add(1);
        let attempt = self.next_attempt;
        self.requests
            .insert(key, PrefetchRequestState::InFlight(attempt));
        Some(attempt)
    }

    pub fn is_current(&self, key: PrefetchKey, attempt: u64) -> bool {
        self.requests.get(&key) == Some(&PrefetchRequestState::InFlight(attempt))
    }

    pub fn finish(&mut self, key: PrefetchKey, attempt: u64, success: bool) {
        if !self.is_current(key, attempt) {
            return;
        }
        if success {
            self.requests.remove(&key);
        } else {
            self.requests.insert(
                key,
                PrefetchRequestState::RetryAt(Instant::now() + PREFETCH_RETRY_DELAY),
            );
        }
    }

    pub fn cancel_component(&mut self, component_id: ComponentId) {
        self.requests
            .retain(|key, _| key.component_id() != component_id);
    }

    fn remove(&mut self, key: PrefetchKey) {
        self.requests.remove(&key);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum HoldAnchorDecision {
    Satisfied,
    Pending,
    Request,
}

pub fn hold_anchor_decision(
    component_id: ComponentId,
    start: Timestamp,
    series_store: &TelemetryCache,
    prefetch: &VisiblePrefetchState,
) -> HoldAnchorDecision {
    if series_store.has_covered_hold(&component_id, start)
        || series_store.is_covered(
            &component_id,
            &(Timestamp(i64::MIN)..Timestamp(start.0.saturating_add(1))),
        )
    {
        return HoldAnchorDecision::Satisfied;
    }
    let key = PrefetchKey::Anchor {
        component_id,
        start: start.0,
    };
    if prefetch
        .requests
        .get(&key)
        .is_some_and(|state| match state {
            PrefetchRequestState::InFlight(_) => true,
            PrefetchRequestState::RetryAt(retry_at) => Instant::now() < *retry_at,
        })
    {
        return HoldAnchorDecision::Pending;
    }
    HoldAnchorDecision::Request
}

pub fn apply_hold_anchor_payload(
    timestamps: &[Timestamp],
    buf: &[u8],
    component_id: ComponentId,
    start: Timestamp,
    series_store: &mut TelemetryCache,
    schema_reg: &ComponentSchemaRegistry,
) -> bool {
    if timestamps.is_empty() {
        // An empty reply cannot confirm a cache that already contains an older
        // sample. Keep the retry delay for this inconsistent response.
        if !buf.is_empty()
            || series_store
                .get_at_or_before(&component_id, start)
                .is_some()
        {
            return false;
        }
        let Some(end) = start.0.checked_add(1) else {
            return false;
        };
        let range = Timestamp(i64::MIN)..Timestamp(end);
        series_store.mark_covered(component_id, range.start, range.end);
        return series_store.is_covered(&component_id, &range);
    }
    let (Some(&timestamp), Some(schema)) = (timestamps.first(), schema_reg.0.get(&component_id))
    else {
        return false;
    };
    let size = schema.size();
    if timestamps.len() != 1 || timestamp > start || buf.len() != size {
        return false;
    }
    let Ok(view) = ComponentView::try_from_bytes_shape(buf, schema.shape(), schema.prim_type())
    else {
        return false;
    };
    // Admit against the wanted block. The sample's own block is often too
    // far to keep, and a dropped insert must not count as confirmed.
    series_store.insert_predecessor(
        component_id,
        timestamp,
        start.0,
        ComponentValue::from_view(view),
    );
    series_store.has_covered_hold(&component_id, start)
}

fn apply_hold_anchor_reply(
    packet: &OwnedPacket<PacketGrantR>,
    component_id: ComponentId,
    start: Timestamp,
    series_store: &mut TelemetryCache,
    schema_reg: &ComponentSchemaRegistry,
) -> bool {
    let OwnedPacket::TimeSeries(time_series) = packet else {
        return false;
    };
    let (Ok(timestamps), Ok(buf)) = (time_series.timestamps(), time_series.data()) else {
        return false;
    };
    apply_hold_anchor_payload(
        timestamps,
        buf,
        component_id,
        start,
        series_store,
        schema_reg,
    )
}

pub fn prefetch_hold_anchor(
    component_id: ComponentId,
    start: Timestamp,
    series_store: &TelemetryCache,
    prefetch: &mut VisiblePrefetchState,
    commands: &mut Commands,
) -> bool {
    let key = PrefetchKey::Anchor {
        component_id,
        start: start.0,
    };
    match hold_anchor_decision(component_id, start, series_store, prefetch) {
        HoldAnchorDecision::Satisfied => {
            prefetch.remove(key);
            return false;
        }
        HoldAnchorDecision::Pending => return false,
        HoldAnchorDecision::Request => {}
    }

    let Some(attempt) = prefetch.begin(key) else {
        return false;
    };
    commands.send_req_reply_raw(
        GetTimeSeriesPredecessor {
            id: PacketId::default(),
            timestamp: start,
            component_id,
        },
        move |pkt: InRef<OwnedPacket<PacketGrantR>>,
              mut series_store: ResMut<TelemetryCache>,
              schema_reg: Res<ComponentSchemaRegistry>,
              priority: Res<SeriesFetchPriority>,
              mut prefetch: ResMut<VisiblePrefetchState>|
              -> bool {
            if !priority.high.contains(&component_id) {
                prefetch.cancel_component(component_id);
                return true;
            }
            if !prefetch.is_current(key, attempt) {
                return true;
            }
            let confirmed =
                apply_hold_anchor_reply(&pkt, component_id, start, &mut series_store, &schema_reg);
            prefetch.finish(key, attempt, confirmed);
            true
        },
    );
    true
}

/// One block for a component. Playback and plots share this request.
pub fn request_block(
    component_id: ComponentId,
    start: i64,
    series_store: &TelemetryCache,
    prefetch: &mut VisiblePrefetchState,
    commands: &mut Commands,
) {
    let end = start.saturating_add(playback::BLOCK_US);
    if series_store.is_covered(&component_id, &(Timestamp(start)..Timestamp(end))) {
        return;
    }
    if !prefetch.has_capacity() {
        return;
    }
    let key = PrefetchKey::Window {
        component_id,
        start,
        end,
    };
    let Some(attempt) = prefetch.begin(key) else {
        return;
    };
    send_series_page(
        key,
        attempt,
        component_id,
        block_query_range(start),
        Timestamp(end),
        commands,
    );
}

/// Prefetch only plot/3D consumers (`fetch_ids`); monitors/viewport/arrows are
/// allowlisted in `update_series_fetch_priority` and filled by live + begin→end backfill.
pub fn prefetch_visible_window(
    sync_range: &Range<Timestamp>,
    range_key: (i64, i64),
    fetch_ids: &HashSet<ComponentId>,
    series_store: &TelemetryCache,
    schema_reg: &ComponentSchemaRegistry,
    prefetch: &mut VisiblePrefetchState,
    commands: &mut Commands,
) {
    // Playback shares these requests. Subscription removal is handled by
    // update_series_fetch_priority using the complete set of consumers.
    if fetch_ids.is_empty() {
        return;
    }
    if cfg!(target_family = "wasm") {
        let first = playback::block_start(sync_range.start.0);
        let last = playback::block_start(sync_range.end.0.saturating_sub(1));
        for component_id in fetch_ids.iter().copied() {
            prefetch_hold_anchor(
                component_id,
                Timestamp(first),
                series_store,
                prefetch,
                commands,
            );
        }
        // Walk outward from the playhead so nearby blocks win admission first.
        let center =
            playback::block_start(series_store.playback_pivot().0).clamp(first, last.max(first));
        let count = last
            .saturating_sub(first)
            .div_euclid(playback::BLOCK_US)
            .saturating_add(1);
        for offset in 0..count {
            for start in [
                center.saturating_add(offset.saturating_mul(playback::BLOCK_US)),
                center.saturating_sub(offset.saturating_mul(playback::BLOCK_US)),
            ] {
                if start < first || start > last {
                    continue;
                }
                for component_id in fetch_ids.iter().copied() {
                    if !prefetch.has_capacity() {
                        return;
                    }
                    if !schema_reg.0.contains_key(&component_id) {
                        continue;
                    }
                    request_block(component_id, start, series_store, prefetch, commands);
                }
            }
        }
        return;
    }
    for component_id in fetch_ids.iter().copied() {
        prefetch.drop_superseded_retries(PrefetchKey::Anchor {
            component_id,
            start: sync_range.start.0,
        });
        prefetch.drop_superseded_retries(PrefetchKey::Window {
            component_id,
            start: range_key.0,
            end: range_key.1,
        });
    }
    for component_id in fetch_ids.iter().copied() {
        if !prefetch.has_capacity() {
            break;
        }
        if !schema_reg.0.contains_key(&component_id) {
            continue;
        }
        prefetch_hold_anchor(
            component_id,
            sync_range.start,
            series_store,
            prefetch,
            commands,
        );
        if !prefetch.has_capacity() {
            continue;
        }
        let key = PrefetchKey::Window {
            component_id,
            start: range_key.0,
            end: range_key.1,
        };
        // Only the gap: a trailing window's new quantum would otherwise refetch it all.
        let Some(gap_start) = series_store.first_uncovered(&component_id, sync_range) else {
            prefetch.remove(key);
            continue;
        };
        let Some(attempt) = prefetch.begin(key) else {
            continue;
        };
        send_visible_prefetch_page(
            key,
            attempt,
            component_id,
            gap_start,
            sync_range.end,
            commands,
        );
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum VisiblePageOutcome {
    Complete,
    Next(Timestamp),
    Invalid,
}

fn apply_visible_prefetch_page(
    pkt: &OwnedPacket<PacketGrantR>,
    component_id: ComponentId,
    req_start: Timestamp,
    req_end: Timestamp,
    limit: usize,
    series_store: &mut TelemetryCache,
    schema_reg: &ComponentSchemaRegistry,
) -> VisiblePageOutcome {
    let OwnedPacket::TimeSeries(time_series) = pkt else {
        return VisiblePageOutcome::Invalid;
    };
    let (Ok(timestamps), Ok(buf)) = (time_series.timestamps(), time_series.data()) else {
        return VisiblePageOutcome::Invalid;
    };
    apply_visible_prefetch_payload(
        timestamps,
        buf,
        component_id,
        req_start,
        req_end,
        limit,
        series_store,
        schema_reg,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn apply_visible_prefetch_payload(
    timestamps: &[Timestamp],
    buf: &[u8],
    component_id: ComponentId,
    req_start: Timestamp,
    req_end: Timestamp,
    limit: usize,
    series_store: &mut TelemetryCache,
    schema_reg: &ComponentSchemaRegistry,
) -> VisiblePageOutcome {
    let Some(schema) = schema_reg.0.get(&component_id) else {
        return VisiblePageOutcome::Invalid;
    };
    let elem_size = schema.size();
    if buf.len() != timestamps.len().saturating_mul(elem_size) {
        return VisiblePageOutcome::Invalid;
    }
    if cfg!(target_family = "wasm")
        && !series_store.admit_block(
            playback::block_start(req_start.0),
            timestamps.len().saturating_mul(elem_size.saturating_add(8)),
        )
    {
        return VisiblePageOutcome::Invalid;
    }
    for (i, &timestamp) in timestamps.iter().enumerate() {
        let offset = i * elem_size;
        let Ok(view) = ComponentView::try_from_bytes_shape(
            &buf[offset..offset + elem_size],
            schema.shape(),
            schema.prim_type(),
        ) else {
            return VisiblePageOutcome::Invalid;
        };
        series_store.insert(component_id, timestamp, ComponentValue::from_view(view));
    }
    if timestamps.len() < limit {
        series_store.mark_page_covered(component_id, req_start, req_end, timestamps);
        return VisiblePageOutcome::Complete;
    }
    let Some(last_ts) = timestamps.last().copied() else {
        series_store.mark_page_covered(component_id, req_start, req_end, timestamps);
        return VisiblePageOutcome::Complete;
    };
    let next_start = Timestamp(last_ts.0.saturating_add(1));
    series_store.mark_page_covered(component_id, req_start, next_start, timestamps);
    if next_start < req_end {
        VisiblePageOutcome::Next(next_start)
    } else {
        VisiblePageOutcome::Complete
    }
}

fn send_visible_prefetch_page(
    key: PrefetchKey,
    attempt: u64,
    component_id: ComponentId,
    req_start: Timestamp,
    req_end: Timestamp,
    commands: &mut Commands,
) {
    send_series_page(
        key,
        attempt,
        component_id,
        req_start..req_end,
        req_end,
        commands,
    );
}

fn send_series_page(
    key: PrefetchKey,
    attempt: u64,
    component_id: ComponentId,
    query: Range<Timestamp>,
    cover_end: Timestamp,
    commands: &mut Commands,
) {
    commands.queue(move |world: &mut bevy::prelude::World| {
        let mut limit = VISIBLE_PREFETCH_LIMIT;
        if cfg!(target_family = "wasm") {
            let size = world
                .resource::<ComponentSchemaRegistry>()
                .0
                .get(&component_id)
                .map_or(8, |schema| schema.size().saturating_add(8));
            let block = playback::block_start(query.start.0);
            let Some(fit) = world.resource::<TelemetryCache>().fitting_page_limit(
                size,
                Some(VISIBLE_PREFETCH_LIMIT),
                block,
            ) else {
                world
                    .resource_mut::<VisiblePrefetchState>()
                    .finish(key, attempt, false);
                return;
            };
            limit = fit;
            if !world
                .resource_mut::<TelemetryCache>()
                .admit_block(block, size.saturating_mul(limit))
            {
                world
                    .resource_mut::<VisiblePrefetchState>()
                    .finish(key, attempt, false);
                return;
            }
        }
        send_admitted_page(
            key,
            attempt,
            component_id,
            query,
            cover_end,
            limit,
            &mut world.commands(),
        );
    });
}

fn send_admitted_page(
    key: PrefetchKey,
    attempt: u64,
    component_id: ComponentId,
    query: Range<Timestamp>,
    cover_end: Timestamp,
    limit: usize,
    commands: &mut Commands,
) {
    commands.send_req_reply_raw(
        GetTimeSeries {
            id: PacketId::default(),
            range: query.clone(),
            component_id,
            limit: Some(limit),
        },
        move |pkt: InRef<OwnedPacket<PacketGrantR>>,
              mut series_store: ResMut<TelemetryCache>,
              schema_reg: Res<ComponentSchemaRegistry>,
              priority: Res<SeriesFetchPriority>,
              mut prefetch: ResMut<VisiblePrefetchState>,
              mut commands: Commands|
              -> bool {
            if !priority.high.contains(&component_id) {
                prefetch.cancel_component(component_id);
                return true;
            }
            if !prefetch.is_current(key, attempt) {
                return true;
            }
            match apply_visible_prefetch_page(
                &pkt,
                component_id,
                query.start,
                cover_end,
                limit,
                &mut series_store,
                &schema_reg,
            ) {
                VisiblePageOutcome::Complete => prefetch.finish(key, attempt, true),
                VisiblePageOutcome::Next(next_start) => send_series_page(
                    key,
                    attempt,
                    component_id,
                    next_start..query.end,
                    cover_end,
                    &mut commands,
                ),
                VisiblePageOutcome::Invalid => prefetch.finish(key, attempt, false),
            }
            true
        },
    );
}

#[cfg(test)]
mod coverage_tests {
    use super::*;

    #[test]
    fn plot_refresh_preserves_pending_playback_requests_after_seek() {
        let playback_id = ComponentId(1);
        let start = 450 * playback::BLOCK_US;
        let end = start + playback::BLOCK_US;
        let mut prefetch = VisiblePrefetchState::default();
        let window = PrefetchKey::Window {
            component_id: playback_id,
            start,
            end,
        };
        let anchor = PrefetchKey::Anchor {
            component_id: playback_id,
            start,
        };
        let window_attempt = prefetch.begin(window).unwrap();
        let anchor_attempt = prefetch.begin(anchor).unwrap();

        let mut world = bevy::prelude::World::new();
        prefetch_visible_window(
            &(Timestamp(start)..Timestamp(end)),
            (start, end),
            &HashSet::from([ComponentId(2)]),
            &TelemetryCache::default(),
            &ComponentSchemaRegistry::default(),
            &mut prefetch,
            &mut world.commands(),
        );

        assert!(prefetch.is_current(window, window_attempt));
        assert!(prefetch.is_current(anchor, anchor_attempt));
        assert!(prefetch.begin(window).is_none());
    }

    #[test]
    fn rejected_or_inconsistent_hold_replies_keep_retry_delay() {
        let id = ComponentId(981);
        let query = Timestamp(40_500_000);
        let mut schemas = ComponentSchemaRegistry::default();
        schemas.0.insert(
            id,
            impeller::schema::Schema::new(impeller::types::PrimType::F64, [0usize; 0]).unwrap(),
        );
        for empty in [false, true] {
            let mut cache = TelemetryCache::default();
            cache.insert(
                id,
                Timestamp(40_100_000),
                ComponentValue::F64(nox::array![10f64].to_dyn()),
            );
            cache.set_playback_budget(cache.resident_bytes(), query);
            let mut prefetch = VisiblePrefetchState::default();
            let key = PrefetchKey::Anchor {
                component_id: id,
                start: query.0,
            };
            let attempt = prefetch.begin(key).unwrap();
            let confirmed = if empty {
                apply_hold_anchor_payload(&[], &[], id, query, &mut cache, &schemas)
            } else {
                apply_hold_anchor_payload(
                    &[Timestamp(40_300_000)],
                    &30f64.to_le_bytes(),
                    id,
                    query,
                    &mut cache,
                    &schemas,
                )
            };
            assert!(!confirmed);
            assert_eq!(cache.total_sample_count(), 1);
            prefetch.finish(key, attempt, confirmed);
            assert!(matches!(
                prefetch.requests.get(&key),
                Some(PrefetchRequestState::RetryAt(_))
            ));
            assert_eq!(
                hold_anchor_decision(id, query, &cache, &prefetch),
                HoldAnchorDecision::Pending
            );
        }
    }

    #[test]
    fn empty_hold_reply_confirms_absence_without_repeated_fetches() {
        let id = ComponentId(981);
        let query = Timestamp(40_500_000);
        let mut cache = TelemetryCache::default();
        assert!(apply_hold_anchor_payload(
            &[],
            &[],
            id,
            query,
            &mut cache,
            &ComponentSchemaRegistry::default()
        ));
        assert_eq!(
            hold_anchor_decision(id, query, &cache, &VisiblePrefetchState::default()),
            HoldAnchorDecision::Satisfied
        );
        assert_eq!(cache.total_sample_count(), 0);
    }

    #[test]
    fn unaligned_predecessor_reply_confirms_exact_query_until_eviction() {
        let id = ComponentId(981);
        let mut cache = TelemetryCache::default();
        let prefetch = VisiblePrefetchState::default();
        let mut schemas = ComponentSchemaRegistry::default();
        schemas.0.insert(
            id,
            impeller::schema::Schema::new(impeller::types::PrimType::F64, [0usize; 0]).unwrap(),
        );
        let query = Timestamp(40_500_000);
        assert_eq!(
            hold_anchor_decision(id, query, &cache, &prefetch),
            HoldAnchorDecision::Request
        );
        assert!(apply_hold_anchor_payload(
            &[Timestamp(30_000_000)],
            &30f64.to_le_bytes(),
            id,
            query,
            &mut cache,
            &schemas
        ));
        assert_eq!(
            hold_anchor_decision(id, query, &cache, &prefetch),
            HoldAnchorDecision::Satisfied
        );
        assert!(cache.has_covered_hold(&id, Timestamp(query.0 - 1)));
        assert!(!cache.has_covered_hold(&id, Timestamp(query.0 + 1)));
        cache.drop_block(id, playback::block_start(query.0));
        cache.insert(
            id,
            Timestamp(30_000_000),
            ComponentValue::F64(nox::array![30f64].to_dyn()),
        );
        assert_eq!(
            hold_anchor_decision(id, query, &cache, &prefetch),
            HoldAnchorDecision::Request
        );
        assert!(apply_hold_anchor_payload(
            &[Timestamp(30_000_000)],
            &30f64.to_le_bytes(),
            id,
            query,
            &mut cache,
            &schemas
        ));
        cache.remove_series(&id);
        cache.insert(
            id,
            Timestamp(30_000_000),
            ComponentValue::F64(nox::array![30f64].to_dyn()),
        );
        assert!(!cache.has_covered_hold(&id, query));
    }

    #[test]
    fn seek_across_gap_requests_a_new_predecessor() {
        let id = ComponentId(981);
        let mut cache = TelemetryCache::default();
        let prefetch = VisiblePrefetchState::default();
        let value = ComponentValue::F64(nox::array![10f64].to_dyn());
        cache.insert(id, Timestamp(10), value.clone());
        cache.mark_covered(id, Timestamp(0), Timestamp(11));
        assert_eq!(
            hold_anchor_decision(id, Timestamp(40), &cache, &prefetch),
            HoldAnchorDecision::Request
        );
        cache.mark_covered(id, Timestamp(35), Timestamp(41));
        assert_eq!(
            hold_anchor_decision(id, Timestamp(40), &cache, &prefetch),
            HoldAnchorDecision::Request
        );
        cache.insert_predecessor(
            id,
            Timestamp(30),
            40,
            ComponentValue::F64(nox::array![30f64].to_dyn()),
        );
        assert_eq!(
            hold_anchor_decision(id, Timestamp(40), &cache, &prefetch),
            HoldAnchorDecision::Satisfied
        );
        assert!(!cache.has_covered_hold(&id, Timestamp(42)));
        cache.mark_covered(id, Timestamp(41), Timestamp(43));
        assert!(cache.has_covered_hold(&id, Timestamp(42)));
    }

    #[test]
    fn complete_coverage_and_exact_samples_keep_valid_holds() {
        let id = ComponentId(981);
        let mut cache = TelemetryCache::default();
        cache.insert(
            id,
            Timestamp(10),
            ComponentValue::F64(nox::array![10f64].to_dyn()),
        );
        assert!(cache.has_covered_hold(&id, Timestamp(10)));
        assert!(!cache.has_covered_hold(&id, Timestamp(9)));
        cache.mark_covered(id, Timestamp(11), Timestamp(40));
        assert!(!cache.has_covered_hold(&id, Timestamp(40)));
        cache.mark_covered(id, Timestamp(40), Timestamp(41));
        assert!(cache.has_covered_hold(&id, Timestamp(40)));
        assert!(!cache.has_covered_hold(&id, Timestamp(i64::MAX)));
    }
}
