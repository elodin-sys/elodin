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
use impeller2_wkt::{CurrentTimestamp, DbConfig, EarliestTimestamp, LastUpdated};

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

/// Bounds for a typed speed. Wider than the presets so `0.02` or `250` work,
/// narrow enough that a stray keystroke cannot stall or fling the playhead.
pub const MIN_PLAYBACK_SPEED: f64 = 0.01;
pub const MAX_PLAYBACK_SPEED: f64 = 1000.0;

pub const SPEED_INPUT_MAX_CHARS: usize = 6;

/// Keep what the speed field accepts: digits and a single `.` or `,`
/// separator, so `0.5`, `2,4`, and `.25` type naturally and nothing else does.
pub fn sanitize_speed_input(text: &str) -> String {
    let mut separator_seen = false;
    text.chars()
        .filter(|c| match c {
            '0'..='9' => true,
            '.' | ',' if !separator_seen => {
                separator_seen = true;
                true
            }
            _ => false,
        })
        .take(SPEED_INPUT_MAX_CHARS)
        .collect()
}

/// Filter one keystroke or paste before it reaches the speed field, so a
/// rejected character never shows. `separator_allowed` is false while the
/// field already holds a separator that the edit does not replace, and flips
/// off once this input supplies one.
pub fn filter_speed_keystrokes(typed: &str, separator_allowed: &mut bool) -> String {
    typed
        .chars()
        .filter(|c| match c {
            '0'..='9' => true,
            '.' | ',' if *separator_allowed => {
                *separator_allowed = false;
                true
            }
            _ => false,
        })
        .collect()
}

/// Parse a typed speed, accepting either decimal separator and clamping to
/// [`MIN_PLAYBACK_SPEED`]..=[`MAX_PLAYBACK_SPEED`]. `None` for empty or zero
/// input, which leaves the current speed alone.
pub fn parse_playback_speed(text: &str) -> Option<f64> {
    let normalized = sanitize_speed_input(text).replace(',', ".");
    let speed: f64 = normalized.parse().ok()?;
    (speed.is_finite() && speed > 0.0).then(|| speed.clamp(MIN_PLAYBACK_SPEED, MAX_PLAYBACK_SPEED))
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
    /// Samples and tail observed on the last pass, so a sample landing inside
    /// the already-scanned span (backfill, prefetch) is told apart from a live
    /// append and forces the holes to be rebuilt.
    seen: usize,
    tail: Option<i64>,
}

/// Advance `current` by `delta`, wrapping to `start` when that lands past the
/// end. The band is `[start, end]`, matching the overlay and seek, so the end
/// sample holds. A playhead already outside the region is pulled to `start`.
pub fn step_loop(current: i64, delta: i64, start: i64, end: i64) -> i64 {
    if start >= end {
        return current.saturating_add(delta);
    }
    let span = end - start;
    if current < start || current > end {
        return start;
    }
    let next = current.saturating_add(delta.max(0));
    if next <= end {
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

/// Where the playhead lands when skipping the gap `(gap_start, gap_end)`, or
/// `None` to leave it where it is. Without a loop it lands on the gap end. When
/// a loop is active and the gap reaches or passes the loop end, it wraps to the
/// loop start rather than jumping outside — leaving the region would only make
/// the next frame's [`step_loop`] snap it back, bouncing at the gap.
///
/// The one case that must not wrap is a gap that starts at or before the loop start: the
/// whole region is quiet, so wrapping would land back inside the gap and freeze
/// the playhead. Then skip nothing and let `step_loop` advance through it.
fn skip_target(gap: (i64, i64), loop_bounds: Option<(i64, i64)>) -> Option<i64> {
    let (gap_start, gap_end) = gap;
    match loop_bounds {
        Some((start, loop_end)) if gap_end >= loop_end => (start < gap_start).then_some(start),
        _ => Some(gap_end),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn skip_discontinuities(
    cache: Res<TelemetryCache>,
    priority: Res<SeriesFetchPriority>,
    paused: Res<crate::ui::Paused>,
    latest_follow: Res<LatestFollow>,
    playback_loop: Res<PlaybackLoop>,
    region: Res<PlaybackRegion>,
    earliest: Res<EarliestTimestamp>,
    last_updated: Res<LastUpdated>,
    mut current: ResMut<CurrentTimestamp>,
    mut index: ResMut<PlaybackDiscontinuities>,
) {
    let imminent = imminent_gap(&cache, &priority.high, current.0.0);
    index.scan(&cache, &priority.high);
    if !paused.0
        && !latest_follow.0
        && let Some((gap_start, end)) = imminent
    {
        let bounds = loop_bounds(playback_loop.0, region.0, earliest.0, last_updated.0);
        if let Some(target) = skip_target((gap_start, end), bounds) {
            current.0 = Timestamp(target);
        }
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
            let last = series.keys().next_back().map(|ts| ts.0);
            let len = series.len();
            let scan = self.scans.entry(*id).or_insert_with(|| SeriesScan {
                origin,
                cursor: None,
                holes: Vec::new(),
                done: false,
                seen: 0,
                tail: None,
            });
            // A live append only ever adds samples past the tail, so the scan
            // resumes from its cursor. Anything else — a sample landing inside
            // the span already scanned (backfill or prefetch filling a hole), a
            // removal, a rewind, or a new stream reusing this id — invalidates
            // the holes, which are rebuilt from the start.
            let rewound =
                matches!((scan.cursor, last), (Some(cursor), Some(last)) if last < cursor);
            let appended_at_tail = match scan.tail {
                Some(tail) if len > scan.seen => {
                    series.range(Timestamp(tail.saturating_add(1))..).count() == len - scan.seen
                }
                _ => false,
            };
            if scan.origin != origin || rewound || (len != scan.seen && !appended_at_tail) {
                *scan = SeriesScan {
                    origin,
                    cursor: None,
                    holes: Vec::new(),
                    done: false,
                    seen: 0,
                    tail: None,
                };
            }
            if origin.is_none() {
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
            scan.seen = len;
            scan.tail = last;
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
    fn speed_input_keeps_digits_and_one_separator() {
        assert_eq!(sanitize_speed_input("2.4"), "2.4");
        assert_eq!(sanitize_speed_input("2,4"), "2,4");
        assert_eq!(sanitize_speed_input("1.2.3"), "1.23");
        assert_eq!(sanitize_speed_input("0,5.1"), "0,51");
        assert_eq!(sanitize_speed_input("-3x "), "3");
        assert_eq!(sanitize_speed_input("12345678"), "123456");
    }

    #[test]
    fn keystrokes_other_than_digits_and_a_free_separator_are_dropped() {
        let mut allowed = true;
        assert_eq!(filter_speed_keystrokes("a", &mut allowed), "");
        assert_eq!(filter_speed_keystrokes("2", &mut allowed), "2");
        assert_eq!(filter_speed_keystrokes(",", &mut allowed), ",");
        assert!(!allowed);
        assert_eq!(filter_speed_keystrokes(".", &mut allowed), "");
        assert_eq!(filter_speed_keystrokes("4x", &mut allowed), "4");

        let mut allowed = true;
        assert_eq!(filter_speed_keystrokes(" 2.4.5x ", &mut allowed), "2.45");
    }

    #[test]
    fn typed_speed_parses_either_separator_and_clamps() {
        assert_eq!(parse_playback_speed("0.5"), Some(0.5));
        assert_eq!(parse_playback_speed("2,4"), Some(2.4));
        assert_eq!(parse_playback_speed(".25"), Some(0.25));
        assert_eq!(parse_playback_speed("3."), Some(3.0));
        assert_eq!(parse_playback_speed("0.001"), Some(MIN_PLAYBACK_SPEED));
        assert_eq!(parse_playback_speed("5000"), Some(MAX_PLAYBACK_SPEED));
        assert_eq!(parse_playback_speed(""), None);
        assert_eq!(parse_playback_speed("0"), None);
        assert_eq!(parse_playback_speed("."), None);
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
    fn loop_end_is_inside_the_band() {
        assert_eq!(step_loop(90, 10, 0, 100), 100);
        assert_eq!(step_loop(100, 0, 0, 100), 100);
        assert_eq!(step_loop(100, 30, 0, 100), 30);
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

    #[test]
    fn a_skip_stays_inside_an_active_loop() {
        assert_eq!(
            skip_target((0, 500), None),
            Some(500),
            "no loop: land on the gap end"
        );
        assert_eq!(
            skip_target((100, 500), Some((0, 1_000))),
            Some(500),
            "gap ends inside the loop: land on it"
        );
        assert_eq!(
            skip_target((100, 1_000), Some((0, 1_000))),
            Some(0),
            "gap ends at the loop end: wrap rather than bounce"
        );
        assert_eq!(
            skip_target((100, 1_500), Some((0, 1_000))),
            Some(0),
            "gap runs past the loop end: wrap to the start"
        );
        assert_eq!(
            skip_target((0, 2_000), Some((500, 1_000))),
            None,
            "gap covers the whole loop: stay put instead of wrapping back into it"
        );
        assert_eq!(
            skip_target((0, 1_000), Some((0, 1_000))),
            None,
            "gap starts at the loop start: wrapping would re-enter it"
        );
    }

    #[test]
    fn the_scan_resumes_when_later_samples_arrive() {
        use impeller2_bevy::ComponentValue;
        let sample = || ComponentValue::F64(nox::array![0.0f64].to_dyn());
        let mut cache = TelemetryCache::default();
        let id = ComponentId::new("imu");
        cache.insert(id, Timestamp(0), sample());
        cache.insert(id, Timestamp(1_000_000), sample());

        let mut ids = std::collections::HashSet::new();
        ids.insert(id);
        let mut index = PlaybackDiscontinuities::default();
        index.scan(&cache, &ids);
        assert!(index.gaps.is_empty(), "contiguous samples have no gap");

        // A late append opens a >2 s hole after the scan already caught up.
        cache.insert(id, Timestamp(5_000_000), sample());
        index.scan(&cache, &ids);
        assert_eq!(
            index.gaps,
            vec![(1_000_000, 5_000_000)],
            "the resumed scan must find the new hole"
        );
    }

    #[test]
    fn the_scan_drops_holes_from_a_rewound_series() {
        use impeller2_bevy::ComponentValue;
        let sample = || ComponentValue::F64(nox::array![0.0f64].to_dyn());
        let mut cache = TelemetryCache::default();
        let id = ComponentId::new("imu");
        cache.insert(id, Timestamp(0), sample());
        cache.insert(id, Timestamp(5_000_000), sample());

        let mut ids = std::collections::HashSet::new();
        ids.insert(id);
        let mut index = PlaybackDiscontinuities::default();
        index.scan(&cache, &ids);
        assert_eq!(index.gaps, vec![(0, 5_000_000)]);

        // A new stream reuses the id with earlier, contiguous data.
        cache.remove_series(&id);
        cache.insert(id, Timestamp(0), sample());
        cache.insert(id, Timestamp(1_000), sample());
        index.scan(&cache, &ids);
        assert!(
            index.gaps.is_empty(),
            "the rewound series must not keep the old stream's hole"
        );
    }

    #[test]
    fn backfill_into_a_scanned_span_drops_the_stale_hole() {
        use impeller2_bevy::ComponentValue;
        let sample = || ComponentValue::F64(nox::array![0.0f64].to_dyn());
        let mut cache = TelemetryCache::default();
        let id = ComponentId::new("imu");
        cache.insert(id, Timestamp(0), sample());
        cache.insert(id, Timestamp(3_000_000), sample());

        let mut ids = std::collections::HashSet::new();
        ids.insert(id);
        let mut index = PlaybackDiscontinuities::default();
        index.scan(&cache, &ids);
        assert_eq!(index.gaps, vec![(0, 3_000_000)]);

        // Backfill lands a sample inside the hole, splitting it into two
        // stretches that are each too short to be a discontinuity.
        cache.insert(id, Timestamp(1_500_000), sample());
        index.scan(&cache, &ids);
        assert!(
            index.gaps.is_empty(),
            "a hole that backfill has filled in must not stay drawn"
        );
    }
}
