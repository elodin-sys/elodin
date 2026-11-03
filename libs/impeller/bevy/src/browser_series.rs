use super::*;
use crate::browser_cache::{self, BrowserCacheScope, CacheRead, PlaybackCacheLimits};
use bevy::prelude::*;
use impeller_wkt::GetTimeSeriesPredecessor;

const WINDOW_US: i64 = 1_000_000;
const PAGE_LIMIT: usize = 4096;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SeriesPage {
    pub timestamps: Vec<Timestamp>,
    pub data: Vec<u8>,
}

#[derive(Clone)]
struct Page {
    scope: String,
    component: ComponentId,
    block: i64,
    start: i64,
    end: i64,
    generation: u64,
    key: String,
    anchor: bool,
}

struct Pending {
    page: Page,
    read: CacheRead,
    started: Instant,
}

#[derive(Resource, Default)]
pub struct BrowserSeriesState {
    scope: String,
    center: i64,
    updated: Option<Instant>,
    generation: u64,
    pending: Vec<Pending>,
    network: HashMap<ComponentId, (u64, String, Instant)>,
    cursors: HashMap<(ComponentId, i64), i64>,
    anchors: HashMap<(ComponentId, i64), Option<Timestamp>>,
}

impl SeriesPage {
    pub fn from_packet<B: stellarator_buf::IoBuf>(packet: &OwnedPacket<B>) -> Option<Self> {
        let OwnedPacket::TimeSeries(series) = packet else {
            return None;
        };
        Some(Self {
            timestamps: series.timestamps().ok()?.to_vec(),
            data: series.data().ok()?.to_vec(),
        })
    }

    pub fn insert(
        &self,
        component: ComponentId,
        cache: &mut TelemetryCache,
        schema: &Schema<Vec<u64>>,
    ) -> bool {
        let size = schema.size();
        if self.timestamps.windows(2).any(|times| times[0] >= times[1]) {
            return false;
        }
        if size == 0 || self.data.len() != self.timestamps.len().saturating_mul(size) {
            return false;
        }
        for (&time, bytes) in self.timestamps.iter().zip(self.data.chunks_exact(size)) {
            let Ok(view) =
                ComponentView::try_from_bytes_shape(bytes, schema.shape(), schema.prim_type())
            else {
                return false;
            };
            cache.insert(component, time, ComponentValue::from_view(view));
        }
        true
    }
}

fn accept(
    page: &Page,
    value: SeriesPage,
    state: &mut BrowserSeriesState,
    cache: &mut TelemetryCache,
    schema: &ComponentSchemaRegistry,
    backfill: &mut BackfillState,
) -> bool {
    if state.scope != page.scope || state.generation != page.generation {
        return false;
    }
    let Some(schema) = schema.0.get(&page.component) else {
        return false;
    };
    if page.anchor {
        if value.timestamps.len() > 1
            || value.timestamps.iter().any(|time| time.0 > page.start)
            || !value.insert(page.component, cache, schema)
        {
            return false;
        }
        let first = value.timestamps.first().copied();
        state.anchors.insert((page.component, page.block), first);
        cache.mark_page_covered(
            page.component,
            first.unwrap_or(Timestamp(i64::MIN)),
            Timestamp(page.start.saturating_add(1)),
            &value.timestamps,
        );
        backfill.initial.insert(page.component);
        return true;
    }
    if value
        .timestamps
        .iter()
        .any(|time| time.0 < page.start || time.0 >= page.end)
        || !value.insert(page.component, cache, schema)
    {
        return false;
    }
    backfill.initial.insert(page.component);
    let end = if value.timestamps.len() < PAGE_LIMIT {
        page.end
    } else {
        value
            .timestamps
            .last()
            .map_or(page.start, |last| last.0.saturating_add(1))
    };
    cache.mark_page_covered(
        page.component,
        Timestamp(page.start),
        Timestamp(end),
        &value.timestamps,
    );
    if end < page.end {
        state.cursors.insert((page.component, page.block), end);
    } else {
        state.cursors.remove(&(page.component, page.block));
    }
    true
}

fn has_playback_value(cache: &TelemetryCache, id: ComponentId, time: Timestamp) -> bool {
    cache
        .series(&id)
        .and_then(|series| series.range(..=time).next_back())
        .is_some_and(|(&previous, _)| {
            cache.is_covered(&id, &(previous..Timestamp(time.0.saturating_add(1))))
        })
}

#[allow(clippy::too_many_arguments)]
pub fn update(
    scope: &BrowserCacheScope,
    limits: &PlaybackCacheLimits,
    current: Timestamp,
    priority: &SeriesFetchPriority,
    schema: &ComponentSchemaRegistry,
    state: &mut BrowserSeriesState,
    cache: &mut TelemetryCache,
    backfill: &mut BackfillState,
    commands: &mut Commands,
) {
    let Some(scope) = scope.key.as_ref() else {
        if !state.scope.is_empty() {
            *state = BrowserSeriesState::default();
            cache.reset();
            *backfill = BackfillState::default();
        }
        return;
    };
    let center = browser_cache::window_start(current.0, WINDOW_US);
    browser_cache::report_resident("telemetryBytes", cache.resident_bytes());
    cache.pivot = current;
    cache.byte_limit = limits.telemetry_bytes.max(1);
    while cache.resident_bytes > cache.byte_limit && cache.evict_farthest() {}
    if &state.scope != scope {
        *state = BrowserSeriesState {
            scope: scope.clone(),
            center,
            ..Default::default()
        };
    }
    let now = Instant::now();
    let elapsed = state
        .updated
        .map_or(Duration::ZERO, |last| now.duration_since(last));
    if browser_cache::is_seek(state.center, center, elapsed, 2_000_000) {
        state.generation = state.generation.wrapping_add(1);
    }
    state.updated = Some(now);
    state.center = center;
    state
        .cursors
        .retain(|(id, time), _| priority.high.contains(id) && time.abs_diff(center) <= 3_000_000);
    state
        .network
        .retain(|_, (_, _, started)| started.elapsed() < browser_cache::REQUEST_TIMEOUT);
    let mut index = 0;
    while index < state.pending.len() {
        let ready = state.pending[index].read.poll();
        if ready.is_none() && state.pending[index].started.elapsed() < Duration::from_secs(3) {
            index += 1;
            continue;
        }
        let pending = state.pending.swap_remove(index);
        if pending.page.generation != state.generation
            || pending.page.scope != state.scope
            || !priority.high.contains(&pending.page.component)
        {
            continue;
        }
        let accepted = ready
            .flatten()
            .as_deref()
            .and_then(browser_cache::decode::<SeriesPage>)
            .is_some_and(|value| accept(&pending.page, value, state, cache, schema, backfill));
        if !accepted {
            send(pending.page, state, commands);
        }
    }
    let mut slots = 4usize.saturating_sub(state.pending.len() + state.network.len());
    for id in series_store_backfill_candidates(&priority.high, schema) {
        if slots == 0 {
            break;
        }
        if state.network.contains_key(&id)
            || state
                .pending
                .iter()
                .any(|pending| pending.page.component == id)
        {
            continue;
        }
        state.anchors.retain(|(component, block), time| {
            block.abs_diff(center) <= 3_000_000
                && time.is_none_or(|time| {
                    cache
                        .series(component)
                        .is_some_and(|series| series.contains_key(&time))
                })
        });
        let anchor =
            !has_playback_value(cache, id, current) && !state.anchors.contains_key(&(id, center));
        let block = if anchor {
            center
        } else {
            let Some(block) = [
                center,
                center.saturating_sub(WINDOW_US),
                center.saturating_add(WINDOW_US),
            ]
            .into_iter()
            .find(|start| {
                !cache.is_covered(
                    &id,
                    &(Timestamp(*start)..Timestamp(start.saturating_add(WINDOW_US))),
                )
            }) else {
                continue;
            };
            block
        };
        let start = if anchor {
            current.0
        } else {
            state.cursors.get(&(id, block)).copied().unwrap_or(block)
        };
        let end = block.saturating_add(WINDOW_US);
        let key = if anchor {
            format!("anchor/{}/{start}", id.0)
        } else {
            format!("series/{}/{start}/{end}/{PAGE_LIMIT}", id.0)
        };
        let read = browser_cache::lookup(scope, &key);
        state.pending.push(Pending {
            page: Page {
                scope: scope.clone(),
                component: id,
                block,
                start,
                end,
                generation: state.generation,
                key,
                anchor,
            },
            read,
            started: Instant::now(),
        });
        slots -= 1;
    }
}

fn send(page: Page, state: &mut BrowserSeriesState, commands: &mut Commands) {
    state.network.insert(
        page.component,
        (page.generation, page.key.clone(), Instant::now()),
    );
    let request_page = page.clone();
    let callback = move |packet: InRef<OwnedPacket<PacketGrantR>>,
                         mut state: ResMut<BrowserSeriesState>,
                         mut cache: ResMut<TelemetryCache>,
                         schema: Res<ComponentSchemaRegistry>,
                         mut backfill: ResMut<BackfillState>,
                         priority: Res<SeriesFetchPriority>,
                         active_scope: Res<BrowserCacheScope>| {
        if state
            .network
            .get(&page.component)
            .is_some_and(|(generation, key, _)| *generation == page.generation && key == &page.key)
        {
            state.network.remove(&page.component);
        }
        if active_scope.key.as_ref() != Some(&page.scope)
            || !priority.high.contains(&page.component)
            || state.generation != page.generation
            || state.scope != page.scope
        {
            return true;
        }
        if let Some(value) = SeriesPage::from_packet(&packet) {
            browser_cache::store(&page.scope, &page.key, &value);
            accept(&page, value, &mut state, &mut cache, &schema, &mut backfill);
        }
        true
    };
    if request_page.anchor {
        commands.send_req_reply_raw(
            GetTimeSeriesPredecessor {
                id: PacketId::default(),
                timestamp: Timestamp(request_page.start),
                component_id: request_page.component,
            },
            callback,
        );
    } else {
        commands.send_req_reply_raw(
            GetTimeSeries {
                id: PacketId::default(),
                range: Timestamp(request_page.start)..Timestamp(request_page.end.saturating_sub(1)),
                component_id: request_page.component,
                limit: Some(PAGE_LIMIT),
            },
            callback,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn setup() -> (Page, BrowserSeriesState, ComponentSchemaRegistry) {
        let page = Page {
            scope: "recording".into(),
            component: ComponentId(7),
            block: 0,
            start: 0,
            end: WINDOW_US,
            generation: 0,
            key: "series".into(),
            anchor: false,
        };
        let state = BrowserSeriesState {
            scope: page.scope.clone(),
            ..Default::default()
        };
        let mut registry = ComponentSchemaRegistry::default();
        registry.0.insert(
            page.component,
            Schema::new(impeller::types::PrimType::F64, [1usize]).unwrap(),
        );
        (page, state, registry)
    }
    fn payload(times: &[i64]) -> SeriesPage {
        SeriesPage {
            timestamps: times.iter().copied().map(Timestamp).collect(),
            data: times
                .iter()
                .flat_map(|time| (*time as f64).to_le_bytes())
                .collect(),
        }
    }
    #[test]
    fn cached_pages_keep_the_wire_values_and_timestamps() {
        let (page, mut state, schema) = setup();
        let encoded = postcard::to_allocvec(&payload(&[0, 10, 20])).unwrap();
        let decoded = browser_cache::decode::<SeriesPage>(&encoded).unwrap();
        let mut cache = TelemetryCache::default();
        accept(
            &page,
            decoded,
            &mut state,
            &mut cache,
            &schema,
            &mut BackfillState::default(),
        );
        assert_eq!(cache.total_sample_count(), 3);
        assert!(cache.is_covered(&page.component, &(Timestamp(0)..Timestamp(WINDOW_US))));
        assert_eq!(
            cache
                .get_at_or_before(&page.component, Timestamp(15))
                .unwrap()
                .as_view()
                .as_bytes(),
            10f64.to_le_bytes()
        );
    }
    #[test]
    fn late_pages_and_malformed_payloads_cannot_populate_the_cache() {
        let (page, mut state, schema) = setup();
        let mut cache = TelemetryCache::default();
        let mut backfill = BackfillState::default();
        state.generation += 1;
        accept(
            &page,
            payload(&[10]),
            &mut state,
            &mut cache,
            &schema,
            &mut backfill,
        );
        assert_eq!(cache.total_sample_count(), 0);
        state.generation = 0;
        accept(
            &page,
            payload(&[-1]),
            &mut state,
            &mut cache,
            &schema,
            &mut backfill,
        );
        assert_eq!(cache.total_sample_count(), 0);
        let mut malformed = payload(&[10]);
        malformed.data.pop();
        accept(
            &page,
            malformed,
            &mut state,
            &mut cache,
            &schema,
            &mut backfill,
        );
        assert_eq!(cache.total_sample_count(), 0);
        assert!(!cache.is_covered(&page.component, &(Timestamp(0)..Timestamp(WINDOW_US))));
    }
    #[test]
    fn sparse_history_uses_the_preceding_value() {
        let (mut page, mut state, schema) = setup();
        page.anchor = true;
        page.start = 500_000;
        let mut cache = TelemetryCache::default();
        accept(
            &page,
            payload(&[-10_000_000]),
            &mut state,
            &mut cache,
            &schema,
            &mut BackfillState::default(),
        );
        assert!(
            cache
                .get_at_or_before(&page.component, Timestamp(500_000))
                .is_some()
        );
        assert!(cache.is_covered(
            &page.component,
            &(Timestamp(-10_000_000)..Timestamp(500_001))
        ));
    }
    #[test]
    fn telemetry_eviction_drops_coverage_and_retains_byte_accounting() {
        let (page, _, _) = setup();
        let mut cache = TelemetryCache::default();
        let value = ComponentValue::F64(nox::array![1f64].to_dyn());
        let cost = TelemetryCache::value_bytes(&value);
        cache.byte_limit = cost * 2;
        cache.pivot = Timestamp(20);
        for time in [0, 10] {
            cache.insert(page.component, Timestamp(time), value.clone());
        }
        cache.mark_covered(page.component, Timestamp(0), Timestamp(21));
        cache.insert(page.component, Timestamp(20), value);
        assert_eq!(cache.total_sample_count(), 2);
        assert_eq!(cache.resident_bytes(), cost * 2);
        assert!(!cache.is_covered(&page.component, &(Timestamp(0)..Timestamp(21))));
        assert!(
            cache
                .get_at_or_before(&page.component, Timestamp(0))
                .is_none()
        );
        cache.remove_series(&page.component);
        assert_eq!(cache.resident_bytes(), 0);
    }
    #[test]
    fn an_evicted_page_prefix_is_not_declared_present() {
        let (page, mut state, schema) = setup();
        let mut cache = TelemetryCache::default();
        let value = ComponentValue::F64(nox::array![1f64].to_dyn());
        cache.byte_limit = TelemetryCache::value_bytes(&value) * 2;
        cache.pivot = Timestamp(20);
        assert!(accept(
            &page,
            payload(&[0, 10, 20]),
            &mut state,
            &mut cache,
            &schema,
            &mut BackfillState::default()
        ));
        assert!(!cache.is_covered(&page.component, &(Timestamp(0)..Timestamp(21))));
        assert!(cache.is_covered(&page.component, &(Timestamp(10)..Timestamp(21))));
    }
    #[test]
    fn an_old_value_does_not_replace_a_predecessor_query_after_a_seek() {
        let (page, _, schema) = setup();
        let mut cache = TelemetryCache::default();
        assert!(payload(&[0]).insert(page.component, &mut cache, &schema.0[&page.component]));
        cache.mark_covered(page.component, Timestamp(0), Timestamp(1_000_000));
        assert!(!has_playback_value(
            &cache,
            page.component,
            Timestamp(5_000_000)
        ));
        cache.mark_covered(page.component, Timestamp(0), Timestamp(5_000_001));
        assert!(has_playback_value(
            &cache,
            page.component,
            Timestamp(5_000_000)
        ));
    }
}
