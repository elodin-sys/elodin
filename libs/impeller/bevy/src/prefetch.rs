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

    fn retain_subscribed(&mut self, ids: &HashSet<ComponentId>) {
        self.requests
            .retain(|key, _| ids.contains(&key.component_id()));
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
    if series_store
        .get_at_or_before(&component_id, start)
        .is_some()
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
        return buf.is_empty();
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
        playback::block_start(start.0),
        ComponentValue::from_view(view),
    );
    series_store
        .get_at_or_before(&component_id, start)
        .is_some()
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
            if confirmed
                && series_store
                    .get_at_or_before(&component_id, start)
                    .is_none()
            {
                series_store.mark_covered(
                    component_id,
                    Timestamp(i64::MIN),
                    Timestamp(start.0.saturating_add(1)),
                );
            }
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
    prefetch.retain_subscribed(fetch_ids);
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
