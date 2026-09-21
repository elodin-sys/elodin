//! Playback speed, loop, and discontinuity skipping.
//!
//! Speed, loop, and the selected region are session state. They are not
//! written into the KDL schematic: a schematic is shared layout (colors, the
//! graph window, whether to arm live follow), and reopening it must not resume
//! the previous session's 10x loop. Rerun stores the equivalent in its
//! blueprint because that blueprint *is* the session.

use std::collections::HashMap;

use bevy::prelude::*;
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::{CurrentStreamId, SeriesFetchPriority, TelemetryCache};
use impeller2_wkt::{CurrentTimestamp, DbConfig};

use super::{AutoFollowLatestState, LatestFollow, PlaybackSpeed};

/// Presets shared by the timeline control and the command palette.
pub const PLAYBACK_SPEED_PRESETS: [f64; 15] = [
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0,
];

/// A hole at least this long, with no subscribed sample inside it, is a
/// discontinuity the playhead jumps rather than sits in.
const DISCONTINUITY_MICROS: i64 = 2_000_000;

/// How many sample steps the gap scan may take per frame. The playhead's own
/// gap is detected separately and does not wait on this.
const SCAN_BUDGET: usize = 8_192;

/// Set the playback multiplier and leave live follow.
///
/// A speed change is a request to replay. Leaving [`LatestFollow`] on would
/// pin the playhead back to the live edge in the same frame, so the label
/// would change and nothing else would.
pub(crate) fn set_playback_speed(
    speed: f64,
    playback_speed: &mut PlaybackSpeed,
    latest_follow: &mut LatestFollow,
    auto_follow: &mut AutoFollowLatestState,
) {
    playback_speed.0 = speed;
    latest_follow.0 = false;
    auto_follow.cancel();
}

/// Next preset above `current`, or the previous one when `direction` is negative.
pub fn adjacent_playback_speed(current: f64, direction: i32) -> f64 {
    if direction >= 0 {
        PLAYBACK_SPEED_PRESETS
            .into_iter()
            .find(|speed| *speed > current + 1e-9)
            .unwrap_or(PLAYBACK_SPEED_PRESETS[PLAYBACK_SPEED_PRESETS.len() - 1])
    } else {
        PLAYBACK_SPEED_PRESETS
            .into_iter()
            .rev()
            .find(|speed| *speed < current - 1e-9)
            .unwrap_or(PLAYBACK_SPEED_PRESETS[0])
    }
}

pub fn same_playback_speed(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// When true, [`advance_playback`](crate::advance_playback) wraps inside
/// [`PlaybackRegion`], or inside the whole recording when no region is set.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct PlaybackLoop(pub bool);

/// Range chosen on the timeline with shift-drag.
///
/// This is a playback region, not the graph window. [`crate::SelectedTimeRange`]
/// stays the view; confusing the two would resize every plot when the user
/// only wanted to loop a passage.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct PlaybackRegion(pub Option<(Timestamp, Timestamp)>);

/// Merged spans where no subscribed series has a sample, drawn on the timeline
/// and jumped by [`skip_discontinuities`].
#[derive(Resource, Default)]
pub struct PlaybackDiscontinuities {
    pub gaps: Vec<(i64, i64)>,
    scans: HashMap<ComponentId, SeriesScan>,
}

struct SeriesScan {
    origin: Option<i64>,
    cursor: Option<i64>,
    holes: Vec<(i64, i64)>,
    done: bool,
}

/// Advance `current` by `delta`, wrapping into `[start, end)` when that lands
/// past the end. A playhead already outside the region is pulled to `start`.
pub fn step_loop(current: i64, delta: i64, start: i64, end: i64) -> i64 {
    if start >= end {
        return current.saturating_add(delta);
    }
    let span = end - start;
    if current < start || current >= end {
        return start;
    }
    let next = current.saturating_add(delta.max(0));
    if next < end {
        return next;
    }
    let over = next - end;
    start + over % span
}

pub fn loop_bounds(
    looping: bool,
    region: Option<(Timestamp, Timestamp)>,
    earliest: Timestamp,
    latest: Timestamp,
) -> Option<(i64, i64)> {
    if !looping {
        return None;
    }
    let (start, end) = match region {
        Some((a, b)) => {
            let (start, end) = if a.0 <= b.0 { (a.0, b.0) } else { (b.0, a.0) };
            (start, end)
        }
        None => (earliest.0, latest.0),
    };
    (start < end).then_some((start, end))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SpeedApply {
    stream: u64,
    applied: Option<f64>,
    saw_config: bool,
}

impl Default for SpeedApply {
    fn default() -> Self {
        Self {
            stream: u64::MAX,
            applied: None,
            saw_config: false,
        }
    }
}

/// Returns the speed to publish, if this config change should override the
/// session. A later config bump that carries the same recorded speed (asset
/// revision, for example) returns `None`, so a speed the user picked stays.
fn consider_recorded_speed(
    state: &mut SpeedApply,
    stream: u64,
    stream_changed: bool,
    config_changed: bool,
    recorded: Option<f64>,
) -> Option<f64> {
    if stream_changed || state.stream != stream {
        *state = SpeedApply {
            stream,
            applied: None,
            saw_config: false,
        };
    }
    if !config_changed {
        return None;
    }
    if state.saw_config && state.applied == recorded {
        return None;
    }
    state.saw_config = true;
    state.applied = recorded;
    recorded
}

pub(crate) fn apply_recorded_playback_speed(
    config: Res<DbConfig>,
    stream: Res<CurrentStreamId>,
    mut playback_speed: ResMut<PlaybackSpeed>,
    mut state: Local<SpeedApply>,
) {
    let Some(speed) = consider_recorded_speed(
        &mut state,
        **stream,
        stream.is_changed(),
        config.is_changed(),
        config.default_playback_speed(),
    ) else {
        return;
    };
    playback_speed.0 = speed;
}

pub fn skip_discontinuities(
    cache: Res<TelemetryCache>,
    priority: Res<SeriesFetchPriority>,
    paused: Res<crate::ui::Paused>,
    latest_follow: Res<LatestFollow>,
    mut current: ResMut<CurrentTimestamp>,
    mut index: ResMut<PlaybackDiscontinuities>,
) {
    let imminent = imminent_gap(&cache, &priority.high, current.0.0);
    index.scan(&cache, &priority.high);
    if !paused.0
        && !latest_follow.0
        && let Some((_, end)) = imminent
    {
        current.0 = Timestamp(end);
    }
    if let Some(gap) = imminent
        && !index
            .gaps
            .iter()
            .any(|known| known.0 <= gap.0 && known.1 >= gap.1)
    {
        index.gaps.push(gap);
        index.gaps.sort_by_key(|gap| gap.0);
    }
}

fn imminent_gap(
    cache: &TelemetryCache,
    ids: &std::collections::HashSet<ComponentId>,
    now: i64,
) -> Option<(i64, i64)> {
    let mut any = false;
    let mut prev_any: Option<i64> = None;
    let mut next_any: Option<i64> = None;
    for id in ids {
        let Some(series) = cache.series(id) else {
            continue;
        };
        if series.is_empty() {
            continue;
        }
        any = true;
        if let Some((ts, _)) = series.range(..=Timestamp(now)).next_back() {
            prev_any = Some(prev_any.map_or(ts.0, |prev| prev.max(ts.0)));
        }
        if let Some((ts, _)) = series.range(Timestamp(now.saturating_add(1))..).next() {
            next_any = Some(next_any.map_or(ts.0, |next| next.min(ts.0)));
        }
    }
    if !any {
        return None;
    }
    match (prev_any, next_any) {
        (Some(prev), Some(next)) if next.saturating_sub(prev) >= DISCONTINUITY_MICROS => {
            (now > prev && now < next).then_some((prev, next))
        }
        _ => None,
    }
}

impl PlaybackDiscontinuities {
    fn scan(&mut self, cache: &TelemetryCache, ids: &std::collections::HashSet<ComponentId>) {
        self.scans.retain(|id, _| ids.contains(id));
        let mut budget = SCAN_BUDGET;
        for id in ids {
            if budget == 0 {
                break;
            }
            let Some(series) = cache.series(id) else {
                continue;
            };
            let origin = series.keys().next().map(|ts| ts.0);
            let scan = self.scans.entry(*id).or_insert_with(|| SeriesScan {
                origin,
                cursor: None,
                holes: Vec::new(),
                done: false,
            });
            if scan.origin != origin {
                *scan = SeriesScan {
                    origin,
                    cursor: None,
                    holes: Vec::new(),
                    done: false,
                };
            }
            if scan.done || origin.is_none() {
                continue;
            }
            let mut prev = scan.cursor;
            let start = scan
                .cursor
                .map(|cursor| Timestamp(cursor.saturating_add(1)))
                .unwrap_or(Timestamp(i64::MIN));
            for (ts, _) in series.range(start..) {
                if let Some(prev) = prev
                    && ts.0.saturating_sub(prev) >= DISCONTINUITY_MICROS
                {
                    scan.holes.push((prev, ts.0));
                }
                prev = Some(ts.0);
                scan.cursor = Some(ts.0);
                budget = budget.saturating_sub(1);
                if budget == 0 {
                    break;
                }
            }
            let last = series.keys().next_back().map(|ts| ts.0);
            scan.done = scan.cursor == last;
        }

        let complete: Vec<&Vec<(i64, i64)>> = ids
            .iter()
            .filter_map(|id| {
                let scan = self.scans.get(id)?;
                scan.done.then_some(&scan.holes)
            })
            .collect();
        let subscribed = ids
            .iter()
            .filter(|id| cache.series(id).is_some_and(|series| !series.is_empty()))
            .count();
        self.gaps = if complete.len() == subscribed && subscribed > 0 {
            intersect_holes(&complete)
        } else {
            Vec::new()
        };
    }
}

fn intersect_holes(sets: &[&Vec<(i64, i64)>]) -> Vec<(i64, i64)> {
    let Some((first, rest)) = sets.split_first() else {
        return Vec::new();
    };
    let mut acc = (*first).clone();
    for set in rest {
        acc = intersect_two(&acc, set);
        if acc.is_empty() {
            break;
        }
    }
    acc
}

fn intersect_two(a: &[(i64, i64)], b: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        let start = a[i].0.max(b[j].0);
        let end = a[i].1.min(b[j].1);
        if start < end {
            out.push((start, end));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_preset_steps_and_clamps() {
        assert_eq!(adjacent_playback_speed(1.0, 1), 2.0);
        assert_eq!(adjacent_playback_speed(1.0, -1), 0.5);
        assert_eq!(adjacent_playback_speed(100.0, 1), 100.0);
        assert_eq!(adjacent_playback_speed(0.05, -1), 0.05);
        assert_eq!(adjacent_playback_speed(1.5, 1), 2.0);
        assert_eq!(adjacent_playback_speed(1.5, -1), 1.0);
    }

    #[test]
    fn changing_speed_is_a_replay_request() {
        let mut speed = PlaybackSpeed(1.0);
        let mut follow = LatestFollow(true);
        let mut auto = AutoFollowLatestState::default();

        set_playback_speed(10.0, &mut speed, &mut follow, &mut auto);

        assert_eq!(speed.0, 10.0);
        assert!(!follow.0);
        assert!(!auto.armed);
    }

    #[test]
    fn loop_wraps_and_pulls_an_outside_playhead_in() {
        assert_eq!(step_loop(0, 30, 0, 100), 30);
        assert_eq!(step_loop(90, 30, 0, 100), 20);
        assert_eq!(step_loop(250, 10, 0, 100), 0);
    }

    #[test]
    fn a_region_outranks_the_whole_recording() {
        let region = Some((Timestamp(5), Timestamp(15)));
        assert_eq!(
            loop_bounds(true, region, Timestamp(0), Timestamp(1_000)),
            Some((5, 15))
        );
        assert_eq!(
            loop_bounds(true, None, Timestamp(0), Timestamp(1_000)),
            Some((0, 1_000))
        );
        assert_eq!(
            loop_bounds(false, region, Timestamp(0), Timestamp(1_000)),
            None
        );
    }

    #[test]
    fn recorded_speed_applies_once_per_value() {
        let mut state = SpeedApply::default();
        assert_eq!(
            consider_recorded_speed(&mut state, 1, true, false, Some(30.0)),
            None,
            "a config still belonging to the previous connection must wait"
        );
        assert_eq!(
            consider_recorded_speed(&mut state, 1, false, true, Some(30.0)),
            Some(30.0)
        );
        assert_eq!(
            consider_recorded_speed(&mut state, 1, false, true, Some(30.0)),
            None,
            "an asset bump carrying the same speed must not clobber the session"
        );
        assert_eq!(
            consider_recorded_speed(&mut state, 2, true, true, Some(2.0)),
            Some(2.0)
        );
    }

    #[test]
    fn holes_intersect_only_where_every_series_is_quiet() {
        let imu = vec![(0, 100), (500, 800)];
        let gps = vec![(50, 90), (400, 900)];
        assert_eq!(intersect_holes(&[&imu, &gps]), vec![(50, 90), (500, 800)]);
        let continuous = Vec::new();
        assert!(intersect_holes(&[&imu, &continuous]).is_empty());
    }

    #[test]
    fn the_playhead_gap_is_the_union_of_subscribed_series() {
        use impeller2_bevy::ComponentValue;

        let sample = || ComponentValue::F64(nox::array![0.0f64].to_dyn());
        let mut cache = TelemetryCache::default();
        let imu = ComponentId::new("imu");
        let gps = ComponentId::new("gps");
        cache.insert(imu, Timestamp(0), sample());
        cache.insert(imu, Timestamp(5_000_000), sample());
        cache.insert(gps, Timestamp(1_000_000), sample());

        let mut ids = std::collections::HashSet::new();
        ids.insert(imu);
        assert_eq!(
            imminent_gap(&cache, &ids, 2_000_000),
            Some((0, 5_000_000)),
            "a single series' hole is a discontinuity"
        );
        ids.insert(gps);
        assert_eq!(
            imminent_gap(&cache, &ids, 500_000),
            None,
            "gps still has a sample ahead of the playhead"
        );
        assert_eq!(
            imminent_gap(&cache, &ids, 2_000_000),
            Some((1_000_000, 5_000_000)),
            "once that sample is behind the playhead, the remaining hole is the gap"
        );
    }
}
