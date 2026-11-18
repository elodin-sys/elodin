//! Always-on, customer-facing per-phase simulation-cycle timing summary.
//!
//! The `run` / `serve` simulation loop in
//! [`crate::impeller_server::tick`] walks through eight distinct
//! phases on every iteration (pre_step → db read → kernel →
//! pre_commit → commit → write-barrier → post_step → real-time
//! pacing). One iteration is one **simulation cycle** — one pass from
//! pre_step through pacing, running `ticks_per_telemetry` simulation
//! steps inside `world.run()` and committing once to the DB. At
//! simulation shutdown we show the customer a one-block stdout
//! summary with mean / p95 / max wall time per phase, so they can
//! see where their cycles are spending time without needing a
//! debug flag, and build a [`Receipt`] for `on_receipt`,
//! `Exec.last_receipt()` and `ELODIN_SIM_SUMMARY_JSON`.
//!
//! Design goals:
//!
//! - **Zero runtime allocation.** Fixed-size histograms, and a
//!   per-cycle sample buffer sized before the loop starts.
//! - **Zero I/O during the loop.** Only the final summary prints,
//!   via a `Drop` impl on [`TickMetrics`].
//! - **~300 ns per-cycle overhead.** Seven `Instant::now` + seven
//!   observe calls (atomic-free, since the metrics struct is
//!   owned by the async task's stack, not shared).
//! - **stdout.** Customers can redirect stderr and still see the
//!   summary. Existing profile / tracing output already lives on
//!   stderr.

use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;

/// Number of log2 buckets in [`PhaseStats::buckets`]. Bucket `i`
/// covers `[2^i, 2^(i+1))` ns, so 32 buckets reach ~2.1 s — well
/// beyond any realistic single-phase wall time.
const BUCKETS: usize = 32;

/// Rolling statistics for one simulation-cycle phase.
///
/// Kept deliberately small: four `u64`s plus a 32×`u32` histogram
/// (128 bytes of buckets). Fits easily in one cache line group.
#[derive(Debug, Default, Clone)]
pub struct PhaseStats {
    sum_ns: u64,
    count: u64,
    min_ns: u64,
    max_ns: u64,
    /// Log2(ns) histogram. `buckets[i]` counts observations where
    /// `floor(log2(max(ns, 1))) == i`.
    buckets: [u32; BUCKETS],
}

#[derive(Debug, Clone, Serialize)]
pub struct PhaseStatsSnapshot {
    sum_ns: u64,
    count: u64,
    min_ns: u64,
    max_ns: u64,
    buckets: [u32; BUCKETS],
    p50_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
}

impl PhaseStatsSnapshot {
    /// Percentiles are nearest-rank over `samples` when given, else estimated from the histogram.
    pub(crate) fn new(stats: &PhaseStats, samples: Option<Vec<u32>>) -> Self {
        let [p50_ns, p95_ns, p99_ns] = match samples {
            Some(mut samples) => {
                samples.sort_unstable();
                [0.50, 0.95, 0.99].map(|q| nearest_rank(&samples, q))
            }
            None => [0.50, 0.95, 0.99].map(|q| stats.percentile_ns(q)),
        };
        Self {
            sum_ns: stats.sum_ns,
            count: stats.count,
            min_ns: stats.min_ns,
            max_ns: stats.max_ns,
            buckets: stats.buckets,
            p50_ns,
            p95_ns,
            p99_ns,
        }
    }
}

fn nearest_rank(sorted: &[u32], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    u64::from(sorted[rank.min(sorted.len()) - 1])
}

/// The compiled tick function and what it did over the measured cycles.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BackendReport {
    pub name: &'static str,
    pub workers: usize,
    pub invocations: u64,
    pub input_bytes_per_invocation: u64,
    pub output_bytes_per_invocation: u64,
    pub compile_ms: Option<CompileReport>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CompileReport {
    pub lower: f64,
    pub parse: f64,
    pub fold: f64,
    pub ir_build: f64,
    pub codegen: f64,
    pub link: f64,
}

/// How much a run measures, from least to most perturbing.
#[pyo3::pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Timing {
    #[default]
    Summary,
    Systems,
    Operations,
}

#[pyo3::pymethods]
impl Timing {
    /// Phase times of every cycle; always on.
    #[staticmethod]
    fn summary() -> Self {
        Self::Summary
    }

    /// Also the time in each compiled function, so each named system, on the Cranelift
    /// backend; a system small enough to inline counts toward its caller. Its probes cost
    /// `systems.probe_pair_ns` per call and keep calls serial.
    #[staticmethod]
    fn systems() -> Self {
        Self::Systems
    }

    /// Also sampled time per kind of operation.
    #[staticmethod]
    fn operations() -> Self {
        Self::Operations
    }
}

/// Versioned record of one run, written to `ELODIN_SIM_SUMMARY_JSON` and passed to
/// `on_receipt`. Phase times cover the cycles after warmup.
#[derive(Debug, Clone, Serialize)]
pub struct Receipt {
    schema: &'static str,
    timing: Timing,
    /// `exact` when every measured cycle was sampled, else `histogram`.
    percentile_source: &'static str,
    pre_step: PhaseStatsSnapshot,
    copy_db_to_world: PhaseStatsSnapshot,
    world_run: PhaseStatsSnapshot,
    pre_commit: PhaseStatsSnapshot,
    commit: PhaseStatsSnapshot,
    /// Appending the `pre_commit` hooks' deferred writes, after the commit.
    deferred_writes: PhaseStatsSnapshot,
    wait_for_write: PhaseStatsSnapshot,
    post_step: PhaseStatsSnapshot,
    /// pre_step + pre_commit + deferred_writes + post_step of each cycle.
    callback: PhaseStatsSnapshot,
    real_time_pacing: PhaseStatsSnapshot,
    real_time_oversleep: PhaseStatsSnapshot,
    real_time_behind: PhaseStatsSnapshot,
    pacing_behind_events: u64,
    pacing_resets: u64,
    pacing_no_sleep: u64,
    total_cycle: PhaseStatsSnapshot,
    cycles: u64,
    steps: u64,
    requested_ticks: Option<u64>,
    warmup_excluded_ticks: u64,
    simulation_rate_hz: f64,
    telemetry_rate_hz: f64,
    wall_ns: u64,
    /// Estimated cost of taking this receipt's timestamps, per cycle.
    timing_overhead_ns: u64,
    backend: Option<BackendReport>,
    sync: Option<crate::impeller_server::SyncCounters>,
    callbacks: Option<crate::prepared::CallbackReport>,
    systems: Option<cranelift_mlir::profile::Timings>,
    failure: Option<String>,
    failed_hook: Option<FailedHook>,
    /// Sections this run did not collect.
    unavailable: Vec<&'static str>,
    entry_unix_ns: Option<u64>,
    compile_done_unix_ns: Option<u64>,
    loop_start_unix_ns: u64,
    loop_end_unix_ns: u64,
    summary_written_unix_ns: u64,
}

/// The `pre_commit` hook that stopped a run.
#[derive(Debug, Clone, Serialize)]
pub struct FailedHook {
    /// Its position in the `pre_commit` list.
    pub index: usize,
    pub name: String,
    /// Whether the cycle's world commit landed before the failure, as it has for a deferred write.
    pub committed: bool,
}

impl Receipt {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("receipt serializes")
    }

    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    pub fn to_py<'py>(
        &self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        use pyo3::types::PyAnyMethods;
        py.import("json")?.call_method1("loads", (self.to_json(),))
    }
}

impl PhaseStats {
    /// Record one phase sample.
    #[inline]
    pub fn observe(&mut self, d: Duration) {
        // Clamp absurd durations (>= 2^64 ns = ~585 years) rather
        // than wrapping. Saturating also protects against pathological
        // clock-source behaviour on suspended-then-resumed machines.
        let ns = u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        self.sum_ns = self.sum_ns.saturating_add(ns);
        self.count += 1;
        if self.count == 1 {
            self.min_ns = ns;
            self.max_ns = ns;
        } else {
            if ns < self.min_ns {
                self.min_ns = ns;
            }
            if ns > self.max_ns {
                self.max_ns = ns;
            }
        }
        let bucket = if ns < 2 {
            0
        } else {
            // floor(log2(ns)) for ns >= 2. Using `leading_zeros` so
            // we can stay u64 without a float detour. 63 - lz gives
            // the highest set-bit index.
            let idx = 63usize - ns.leading_zeros() as usize;
            idx.min(BUCKETS - 1)
        };
        self.buckets[bucket] = self.buckets[bucket].saturating_add(1);
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn mean_ns(&self) -> u64 {
        self.sum_ns.checked_div(self.count).unwrap_or(0)
    }

    pub fn max_ns(&self) -> u64 {
        self.max_ns
    }

    /// Approximate the `q`-th quantile (`q` in `[0, 1]`) from the
    /// log2 histogram. Within a bucket we linearly interpolate
    /// between the bucket's lower edge `2^i` and upper edge
    /// `2^(i+1)`, so the worst-case error is one bucket width —
    /// plenty for a "is my sim fast?" customer view.
    pub fn percentile_ns(&self, q: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        if q <= 0.0 {
            return self.min_ns;
        }
        if q >= 1.0 {
            return self.max_ns;
        }
        let target = (q * self.count as f64).ceil().max(1.0) as u64;
        let mut cum: u64 = 0;
        for (i, &c) in self.buckets.iter().enumerate() {
            let c = c as u64;
            if c == 0 {
                continue;
            }
            let prev = cum;
            cum += c;
            if cum >= target {
                // Linear interpolation within bucket [2^i, 2^(i+1)).
                let lo = 1u64 << i;
                let hi = if i + 1 < 64 {
                    1u64 << (i + 1)
                } else {
                    u64::MAX
                };
                let frac = (target - prev) as f64 / c as f64;
                let within = (lo as f64 + frac * (hi - lo) as f64) as u64;
                // Clamp inside observed min / max so percentiles
                // never lie below the true minimum.
                return within.clamp(self.min_ns, self.max_ns);
            }
        }
        self.max_ns
    }
}

/// Most cycles [`TickMetrics::record_cycles`] keeps (56 bytes each).
const MAX_SAMPLES: usize = 2_000_000;

/// Timestamps the simulation loop takes per cycle, for the overhead estimate.
const TIMESTAMPS_PER_CYCLE: u64 = 18;

/// Phase times of one simulation cycle in nanoseconds, saturating at about 4.3 s.
#[derive(Clone, Copy, Default)]
struct CycleSample {
    pre_step: u32,
    copy_db_to_world: u32,
    world_run: u32,
    pre_commit: u32,
    commit: u32,
    deferred_writes: u32,
    wait_for_write: u32,
    post_step: u32,
    real_time_pacing: u32,
    total_cycle: u32,
    /// Compute time, from the top of the loop to the pacing block.
    work: u32,
    /// Sleep asked for; 0 when the lead target or the deadline had passed.
    requested_sleep: u32,
    actual_sleep: u32,
    /// `now - deadline` at the pacing check; negative is ahead of schedule.
    deadline_err: i32,
    /// The drift cap moved the deadline forward.
    reset: bool,
}

impl CycleSample {
    fn callback(&self) -> u32 {
        self.pre_step
            .saturating_add(self.pre_commit)
            .saturating_add(self.deferred_writes)
            .saturating_add(self.post_step)
    }
}

fn ns32(d: Duration) -> u32 {
    u32::try_from(d.as_nanos()).unwrap_or(u32::MAX)
}

fn instant_now_cost_ns() -> u64 {
    const CALLS: u32 = 1024;
    let start = Instant::now();
    for _ in 0..CALLS {
        std::hint::black_box(Instant::now());
    }
    u64::try_from(start.elapsed().as_nanos() / u128::from(CALLS)).unwrap_or(u64::MAX)
}

fn write_trace(path: &PathBuf, samples: &[CycleSample]) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut out = String::with_capacity(samples.len() * 96 + 256);
    out.push_str(
        "cycle,work_ns,requested_sleep_ns,actual_sleep_ns,oversleep_ns,deadline_err_ns,reset,\
         pre_step_ns,copy_db_to_world_ns,world_run_ns,pre_commit_ns,commit_ns,\
         deferred_writes_ns,wait_for_write_ns,post_step_ns,real_time_pacing_ns,total_cycle_ns\n",
    );
    for (i, s) in samples.iter().enumerate() {
        out.push_str(&format!(
            "{i},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            s.work,
            s.requested_sleep,
            s.actual_sleep,
            s.actual_sleep.saturating_sub(s.requested_sleep),
            s.deadline_err,
            s.reset as u8,
            s.pre_step,
            s.copy_db_to_world,
            s.world_run,
            s.pre_commit,
            s.commit,
            s.deferred_writes,
            s.wait_for_write,
            s.post_step,
            s.real_time_pacing,
            s.total_cycle,
        ));
    }
    let _ = fs::write(path, out);
}

/// Owner of one simulation loop's per-phase statistics. Dropping
/// this struct prints the final summary to stdout; the `tick` async
/// fn binds one of these at the top of the function so the summary
/// prints naturally when the task returns (normal exit, cancel, or
/// panic unwind).
pub struct TickMetrics {
    pub pre_step: PhaseStats,
    pub copy_db_to_world: PhaseStats,
    pub world_run: PhaseStats,
    pub pre_commit: PhaseStats,
    pub commit: PhaseStats,
    pub deferred_writes: PhaseStats,
    pub wait_for_write: PhaseStats,
    pub post_step: PhaseStats,
    /// pre_step + pre_commit + deferred_writes + post_step of each cycle.
    pub callback: PhaseStats,
    pub real_time_pacing: PhaseStats,
    /// Real-time pacing diagnostics (only populated when
    /// `generate_real_time` is on). `real_time_oversleep` is the
    /// amount each pacing sleep ran *past* the duration we asked for
    /// (`actual_sleep - requested_sleep`); it isolates timer / OS /
    /// executor wakeup jitter from the intended sleep. `real_time_behind`
    /// records, for the cycles where we entered the pacing block already
    /// past the deadline, how far past we were (the positive lateness).
    pub real_time_oversleep: PhaseStats,
    pub real_time_behind: PhaseStats,
    /// Count of cycles where, at the pacing check, wall time had already
    /// passed the per-cycle deadline (i.e. the "cannot achieve real-time"
    /// warning condition was true for that cycle).
    pub pacing_behind_events: u64,
    /// Count of cycles where the deadline was reset forward to `now`
    /// because we were more than two tick-periods behind (drift cap).
    pub pacing_resets: u64,
    /// Count of cycles where no sleep happened because the *real* deadline had
    /// already elapsed (we were behind and trying to catch up). Does not count
    /// cycles that only consumed the lead buffer (`deadline - lead` target in
    /// the past while the real deadline is still in the future).
    pub pacing_no_sleep: u64,
    /// Every cycle's phase times, warmup included, sized by [`Self::record_cycles`].
    /// `None` when the run has no known length and no trace was requested.
    samples: Option<Vec<CycleSample>>,
    /// Index in `samples` of the first cycle after warmup.
    measured_from: usize,
    /// `ELODIN_PACING_TRACE`: the CSV that `samples` is written to on drop.
    trace_path: Option<PathBuf>,
    /// Phase times of the cycle in progress.
    cycle: CycleSample,
    instant_now_ns: u64,
    pub requested_ticks: Option<u64>,
    pub backend: Option<BackendReport>,
    pub sync: Option<crate::impeller_server::SyncCounters>,
    pub callbacks: Option<crate::prepared::CallbackReport>,
    pub timing: Timing,
    pub systems: Option<cranelift_mlir::profile::Timings>,
    /// Why the run stopped early, when a callback failed.
    pub failure: Option<String>,
    pub failed_hook: Option<FailedHook>,
    /// Wall-clock duration of one simulation cycle (from immediately
    /// after the cancel / paused checks down to just before the next
    /// iteration starts). Feeds the "mean cycle" header and
    /// "effective cycles/s" estimate. Equals the sum of the other
    /// phase durations plus minor un-attributed overhead (timestamp
    /// calculations, should_cancel, etc. — typically < 1 µs/cycle).
    pub total_cycle: PhaseStats,
    /// Number of completed simulation cycles (outer-loop iterations,
    /// one per telemetry commit).
    pub cycles: u64,
    /// Number of simulation steps executed inside `world.run()`
    /// across all cycles. Equals `cycles × ticks_per_telemetry` when
    /// `simulation_rate == telemetry_rate`, otherwise larger.
    pub steps: u64,
    /// Configured simulation rate (Hz). 0.0 until `set_rates` runs.
    simulation_rate_hz: f64,
    /// Configured telemetry rate (Hz). Equals `simulation_rate_hz`
    /// when the user didn't opt in to a slower telemetry rate.
    telemetry_rate_hz: f64,
    started_at: Instant,
    started_unix_ns: u64,
    loop_end_unix_ns: Option<u64>,
    /// Number of initial ticks excluded from the stats as "warmup" (the
    /// simulator/SITL spin-up). Set by [`Self::reset_after_warmup`]; shown in
    /// the summary header so the reader knows the boot was not measured. 0 means
    /// no warmup reset happened (e.g. a run shorter than the warmup window).
    warmup_excluded_ticks: u64,
    /// Suppresses the `Drop` summary (e.g. when no cycles ever ran).
    /// Currently unused — the summary itself gracefully handles
    /// zero-cycle cases — but kept as an escape hatch for callers
    /// that explicitly want silence.
    pub silent: bool,
}

impl TickMetrics {
    pub fn new() -> Self {
        Self {
            pre_step: PhaseStats::default(),
            copy_db_to_world: PhaseStats::default(),
            world_run: PhaseStats::default(),
            pre_commit: PhaseStats::default(),
            commit: PhaseStats::default(),
            deferred_writes: PhaseStats::default(),
            wait_for_write: PhaseStats::default(),
            post_step: PhaseStats::default(),
            callback: PhaseStats::default(),
            real_time_pacing: PhaseStats::default(),
            real_time_oversleep: PhaseStats::default(),
            real_time_behind: PhaseStats::default(),
            pacing_behind_events: 0,
            pacing_resets: 0,
            pacing_no_sleep: 0,
            samples: None,
            measured_from: 0,
            trace_path: std::env::var_os("ELODIN_PACING_TRACE").map(PathBuf::from),
            cycle: CycleSample::default(),
            instant_now_ns: instant_now_cost_ns(),
            requested_ticks: None,
            backend: None,
            sync: None,
            callbacks: None,
            timing: Timing::Summary,
            systems: None,
            failure: None,
            failed_hook: None,
            total_cycle: PhaseStats::default(),
            cycles: 0,
            steps: 0,
            simulation_rate_hz: 0.0,
            telemetry_rate_hz: 0.0,
            started_at: Instant::now(),
            started_unix_ns: unix_now_ns(),
            loop_end_unix_ns: None,
            warmup_excluded_ticks: 0,
            silent: false,
        }
    }

    pub fn mark_loop_end(&mut self) {
        self.loop_end_unix_ns.get_or_insert_with(unix_now_ns);
    }

    /// Records that `pre_commit` hook `index`, named `name`, stopped the run `when`, unless an
    /// earlier failure did.
    pub fn fail(&mut self, index: usize, name: &str, when: &str, cause: &str, committed: bool) {
        if self.failure.is_some() {
            return;
        }
        self.failure = Some(format!(
            "pre_commit hook {index} ({name}) failed {when}: {cause}"
        ));
        self.failed_hook = Some(FailedHook {
            index,
            name: name.to_string(),
            committed,
        });
    }

    /// Discard everything accumulated so far and restart the clock, so the
    /// final summary reflects steady state rather than the simulator/SITL
    /// spin-up (the first cycle can stall for seconds while a flight controller
    /// boots, which otherwise poisons mean cycle / post_step / lateness). Called
    /// once by the loop after the warmup tick threshold is crossed. The
    /// per-cycle samples are kept so the trace still shows the boot; only
    /// the ones after this point feed the receipt.
    pub fn reset_after_warmup(&mut self, excluded_ticks: u64) {
        self.pre_step = PhaseStats::default();
        self.copy_db_to_world = PhaseStats::default();
        self.world_run = PhaseStats::default();
        self.pre_commit = PhaseStats::default();
        self.commit = PhaseStats::default();
        self.deferred_writes = PhaseStats::default();
        self.wait_for_write = PhaseStats::default();
        self.post_step = PhaseStats::default();
        self.callback = PhaseStats::default();
        self.real_time_pacing = PhaseStats::default();
        self.real_time_oversleep = PhaseStats::default();
        self.real_time_behind = PhaseStats::default();
        self.total_cycle = PhaseStats::default();
        self.pacing_behind_events = 0;
        self.pacing_resets = 0;
        self.pacing_no_sleep = 0;
        self.cycles = 0;
        self.steps = 0;
        self.started_at = Instant::now();
        self.started_unix_ns = unix_now_ns();
        self.warmup_excluded_ticks = excluded_ticks;
        self.measured_from = self.samples.as_ref().map_or(0, Vec::len);
        self.cycle = CycleSample::default();
    }

    /// Feed the configured rates so `print_summary` can show the
    /// `X Hz sim / Y Hz telemetry` annotation. Call once, right
    /// after `new`, from the site that knows the world's
    /// `sim_time_step` and `ticks_per_telemetry`.
    pub fn set_rates(&mut self, simulation_rate_hz: f64, telemetry_rate_hz: f64) {
        self.simulation_rate_hz = simulation_rate_hz;
        self.telemetry_rate_hz = telemetry_rate_hz;
    }

    /// Sample every cycle, so the receipt has exact percentiles, when the run's length is known
    /// or a trace was requested. Call before the loop: the buffer never grows, and cycles past
    /// its capacity leave the receipt with histogram estimates.
    pub fn record_cycles(&mut self, expected_cycles: Option<u64>) {
        let trace_max = self.trace_path.as_ref().map(|_| {
            std::env::var("ELODIN_PACING_TRACE_MAX")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(MAX_SAMPLES as u64)
        });
        let Some(capacity) = trace_max.max(expected_cycles) else {
            return;
        };
        let capacity = usize::try_from(capacity).unwrap_or(MAX_SAMPLES);
        self.samples = Some(Vec::with_capacity(capacity.min(MAX_SAMPLES)));
    }

    #[inline]
    pub fn observe_pre_step(&mut self, d: Duration) {
        self.pre_step.observe(d);
        self.cycle.pre_step = ns32(d);
    }
    #[inline]
    pub fn observe_copy_db_to_world(&mut self, d: Duration) {
        self.copy_db_to_world.observe(d);
        self.cycle.copy_db_to_world = ns32(d);
    }
    #[inline]
    pub fn observe_world_run(&mut self, d: Duration) {
        self.world_run.observe(d);
        self.cycle.world_run = ns32(d);
    }
    #[inline]
    pub fn observe_pre_commit(&mut self, d: Duration) {
        self.pre_commit.observe(d);
        self.cycle.pre_commit = ns32(d);
    }
    #[inline]
    pub fn observe_commit(&mut self, d: Duration) {
        self.commit.observe(d);
        self.cycle.commit = ns32(d);
    }
    #[inline]
    pub fn observe_deferred_writes(&mut self, d: Duration) {
        self.deferred_writes.observe(d);
        self.cycle.deferred_writes = ns32(d);
    }
    #[inline]
    pub fn observe_wait_for_write(&mut self, d: Duration) {
        self.wait_for_write.observe(d);
        self.cycle.wait_for_write = ns32(d);
    }
    #[inline]
    pub fn observe_post_step(&mut self, d: Duration) {
        self.post_step.observe(d);
        self.cycle.post_step = ns32(d);
    }
    #[inline]
    pub fn observe_real_time_pacing(&mut self, d: Duration) {
        self.real_time_pacing.observe(d);
        self.cycle.real_time_pacing = ns32(d);
    }
    /// Ends the cycle in progress.
    #[inline]
    pub fn observe_total_cycle(&mut self, d: Duration) {
        self.total_cycle.observe(d);
        self.cycle.total_cycle = ns32(d);
        self.callback
            .observe(Duration::from_nanos(self.cycle.callback().into()));
        if let Some(samples) = self.samples.as_mut()
            && samples.len() < samples.capacity()
        {
            samples.push(self.cycle);
        }
        self.cycle = CycleSample::default();
    }

    /// Record one real-time pacing cycle. `work` is the cycle's compute time,
    /// `requested` the sleep we asked for (`Duration::ZERO` when either the
    /// lead-buffer target or the real deadline was already past), `actual` the
    /// sleep that actually elapsed, `deadline_err_ns` the signed `now -
    /// deadline` at the pacing check (negative = ahead), and `reset` whether
    /// the drift cap fired.
    #[inline]
    pub fn observe_pacing(
        &mut self,
        work: Duration,
        requested: Duration,
        actual: Duration,
        deadline_err_ns: i64,
        reset: bool,
    ) {
        if deadline_err_ns > 0 {
            self.pacing_behind_events += 1;
            self.real_time_behind
                .observe(Duration::from_nanos(deadline_err_ns as u64));
        }
        if reset {
            self.pacing_resets += 1;
        }
        if requested.is_zero() && deadline_err_ns > 0 {
            self.pacing_no_sleep += 1;
        } else if !requested.is_zero() {
            self.real_time_oversleep
                .observe(actual.saturating_sub(requested));
        }
        self.cycle.work = ns32(work);
        self.cycle.requested_sleep = ns32(requested);
        self.cycle.actual_sleep = ns32(actual);
        self.cycle.deadline_err = deadline_err_ns.clamp(i32::MIN.into(), i32::MAX.into()) as i32;
        self.cycle.reset = reset;
    }

    /// Format a ns duration into a compact human-readable cell
    /// ("12.3 µs", "1.8 ms", "2.05 s"). Stays in a unit as long as
    /// the integer part is non-trivial, so readings don't collapse
    /// to "0.0 ms" for sub-millisecond phases.
    fn fmt_ns(ns: u64) -> String {
        if ns < 1_000 {
            format!("{:>6} ns", ns)
        } else if ns < 1_000_000 {
            format!("{:>6.1} µs", ns as f64 / 1e3)
        } else if ns < 1_000_000_000 {
            format!("{:>6.1} ms", ns as f64 / 1e6)
        } else {
            format!("{:>6.2} s ", ns as f64 / 1e9)
        }
    }

    fn fmt_cell(p: &PhaseStats, q: f64) -> String {
        if p.count == 0 {
            "        —".to_string()
        } else {
            let val = if q == 1.0 {
                p.max_ns
            } else if q < 0.0 {
                p.mean_ns()
            } else {
                p.percentile_ns(q)
            };
            format!("{:>9}", Self::fmt_ns(val))
        }
    }

    fn fmt_row(name: &str, p: &PhaseStats) -> String {
        format!(
            "    {:<18} {}  {}  {}",
            name,
            Self::fmt_cell(p, -1.0), // mean
            Self::fmt_cell(p, 0.95),
            Self::fmt_cell(p, 1.0),
        )
    }

    pub fn print_summary(&self) {
        if self.silent {
            return;
        }
        let wall = self.started_at.elapsed();
        println!();
        println!("──── elodin simulation summary ────");
        if self.warmup_excluded_ticks > 0 {
            println!(
                "  (excluding first {} ticks as warmup / spin-up)",
                format_commas(self.warmup_excluded_ticks),
            );
        }
        println!("  wall time:          {:.3} s", wall.as_secs_f64());
        if self.cycles == 0 {
            println!("  simulation cycles:  0  (no cycles executed)");
            println!("───────────────────────────────────");
            return;
        }
        // Mean cycle wall time (sum of phases + minor un-attributed
        // overhead). For sims pacing at `generate_real_time=True`
        // this is dominated by the real-time sleep; for unpaced sims
        // it's the compute phase.
        let mean_cycle_ns = self.total_cycle.mean_ns();
        let cycles_per_sec = if wall.as_secs_f64() > 0.0 {
            self.cycles as f64 / wall.as_secs_f64()
        } else {
            0.0
        };
        println!(
            "  simulation cycles:  {}    mean cycle: {}  (effective {:.0} cycles/s)",
            format_commas(self.cycles),
            Self::fmt_ns(mean_cycle_ns).trim(),
            cycles_per_sec,
        );
        // When the user configured a telemetry rate slower than the
        // simulation rate, surface the raw sim-step count and the
        // rate ratio so the reader can see why `steps > cycles`.
        if self.steps > 0
            && self.simulation_rate_hz > 0.0
            && self.telemetry_rate_hz > 0.0
            && (self.simulation_rate_hz - self.telemetry_rate_hz).abs() > 1e-9
        {
            println!(
                "  simulation steps:   {}    at {} Hz sim / {} Hz telemetry",
                format_commas(self.steps),
                fmt_hz(self.simulation_rate_hz),
                fmt_hz(self.telemetry_rate_hz),
            );
        }
        println!();
        println!("  per-cycle phase timing (mean / p95 / max):");
        println!("{}", Self::fmt_row("pre_step", &self.pre_step));
        println!(
            "{}",
            Self::fmt_row("read (db → world)", &self.copy_db_to_world)
        );
        println!("{}", Self::fmt_row("tick function", &self.world_run));
        if self.pre_commit.count > 0 {
            println!("{}", Self::fmt_row("pre_commit", &self.pre_commit));
        }
        println!("{}", Self::fmt_row("commit world", &self.commit));
        if self
            .callbacks
            .as_ref()
            .is_some_and(|c| c.deferred_batches > 0)
        {
            println!(
                "{}",
                Self::fmt_row("deferred writes", &self.deferred_writes)
            );
        }
        println!("{}", Self::fmt_row("wait_for_write", &self.wait_for_write));
        println!("{}", Self::fmt_row("post_step", &self.post_step));
        println!(
            "{}",
            Self::fmt_row("real-time pacing", &self.real_time_pacing)
        );
        self.print_pacing_diagnostics();
        println!("───────────────────────────────────");
    }

    /// Extra real-time pacing diagnostics, printed only when pacing was
    /// active. Surfaces *why* a sim with compute headroom can still slip
    /// behind wall-clock: sleep oversleep (timer/OS jitter), how often we
    /// crossed the deadline, and how often the drift cap reset.
    fn print_pacing_diagnostics(&self) {
        if self.real_time_pacing.count == 0 {
            return;
        }
        let paced = self.cycles.max(1);
        let behind_pct = 100.0 * self.pacing_behind_events as f64 / paced as f64;
        println!();
        println!("  real-time pacing diagnostics:");
        println!(
            "{}",
            Self::fmt_row("  oversleep (jitter)", &self.real_time_oversleep)
        );
        println!(
            "{}",
            Self::fmt_row("  lateness when behind", &self.real_time_behind)
        );
        println!(
            "    {:<18} {} cycles ({:.1}% of {})",
            "behind-deadline:",
            format_commas(self.pacing_behind_events),
            behind_pct,
            format_commas(paced),
        );
        println!(
            "    {:<18} {}    catch-up no sleep: {}",
            "drift resets:",
            format_commas(self.pacing_resets),
            format_commas(self.pacing_no_sleep),
        );
    }

    pub fn receipt(&self) -> Receipt {
        let summary_written_unix_ns = unix_now_ns();
        let measured = self
            .samples
            .as_deref()
            .map(|samples| &samples[self.measured_from..])
            .filter(|measured| measured.len() as u64 == self.cycles);
        let phase = |stats: &PhaseStats, field: fn(&CycleSample) -> u32| {
            PhaseStatsSnapshot::new(stats, measured.map(|m| m.iter().map(field).collect()))
        };
        let mut unavailable = Vec::new();
        if self.systems.is_none() {
            unavailable.push("systems");
        }
        if self
            .systems
            .as_ref()
            .is_none_or(|systems| systems.ops.is_empty())
        {
            unavailable.push("operations");
        }
        if self.backend.is_none() {
            unavailable.push("backend");
        }
        if self.sync.is_none() {
            unavailable.push("sync");
        }
        if self.callbacks.is_none() {
            unavailable.push("callbacks");
        }
        Receipt {
            schema: "elodin.run_receipt.v1",
            timing: self.timing,
            percentile_source: if measured.is_some() {
                "exact"
            } else {
                "histogram"
            },
            pre_step: phase(&self.pre_step, |s| s.pre_step),
            copy_db_to_world: phase(&self.copy_db_to_world, |s| s.copy_db_to_world),
            world_run: phase(&self.world_run, |s| s.world_run),
            pre_commit: phase(&self.pre_commit, |s| s.pre_commit),
            commit: phase(&self.commit, |s| s.commit),
            deferred_writes: phase(&self.deferred_writes, |s| s.deferred_writes),
            wait_for_write: phase(&self.wait_for_write, |s| s.wait_for_write),
            post_step: phase(&self.post_step, |s| s.post_step),
            callback: phase(&self.callback, CycleSample::callback),
            real_time_pacing: phase(&self.real_time_pacing, |s| s.real_time_pacing),
            real_time_oversleep: PhaseStatsSnapshot::new(&self.real_time_oversleep, None),
            real_time_behind: PhaseStatsSnapshot::new(&self.real_time_behind, None),
            pacing_behind_events: self.pacing_behind_events,
            pacing_resets: self.pacing_resets,
            pacing_no_sleep: self.pacing_no_sleep,
            total_cycle: phase(&self.total_cycle, |s| s.total_cycle),
            cycles: self.cycles,
            steps: self.steps,
            requested_ticks: self.requested_ticks,
            warmup_excluded_ticks: self.warmup_excluded_ticks,
            simulation_rate_hz: self.simulation_rate_hz,
            telemetry_rate_hz: self.telemetry_rate_hz,
            wall_ns: u64::try_from(self.started_at.elapsed().as_nanos()).unwrap_or(u64::MAX),
            timing_overhead_ns: TIMESTAMPS_PER_CYCLE * self.instant_now_ns,
            backend: self.backend.clone().map(|backend| BackendReport {
                invocations: self.steps,
                ..backend
            }),
            sync: self.sync.clone(),
            callbacks: self.callbacks.clone(),
            systems: self.systems.clone(),
            failure: self.failure.clone(),
            failed_hook: self.failed_hook.clone(),
            unavailable,
            entry_unix_ns: read_env_u64("ELODIN_SIM_ENTRY_UNIX_NS"),
            compile_done_unix_ns: read_env_u64("ELODIN_SIM_COMPILE_DONE_UNIX_NS"),
            loop_start_unix_ns: self.started_unix_ns,
            loop_end_unix_ns: self.loop_end_unix_ns.unwrap_or(summary_written_unix_ns),
            summary_written_unix_ns,
        }
    }

    fn write_summary_json_from_env(&self) {
        if self.silent {
            return;
        }
        let Ok(path) = std::env::var("ELODIN_SIM_SUMMARY_JSON") else {
            return;
        };
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(path, self.receipt().to_json());
    }
}

fn unix_now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
        .unwrap_or_default()
}

fn read_env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok()?.parse().ok()
}

/// Format a rate in Hz, stripping the decimal when the value is
/// close enough to an integer. The tolerance of 0.01 Hz absorbs the
/// round-trip error from storing `1 / rate` as a `Duration` and
/// back. `300.0 → "300"`, `33.333333 → "33.3"`.
fn fmt_hz(hz: f64) -> String {
    let rounded = hz.round();
    if (hz - rounded).abs() < 0.01 {
        format!("{:.0}", rounded)
    } else {
        format!("{:.1}", hz)
    }
}

impl Default for TickMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TickMetrics {
    fn drop(&mut self) {
        self.mark_loop_end();
        self.print_summary();
        self.write_summary_json_from_env();
        if let (Some(path), Some(samples)) = (&self.trace_path, &self.samples) {
            write_trace(path, samples);
        }
    }
}

/// Format an integer with thousands separators (US locale). No
/// `num-format` dep; we only need u64 and ASCII commas.
fn format_commas(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let first = bytes.len() % 3;
    if first > 0 {
        out.push_str(std::str::from_utf8(&bytes[..first]).unwrap());
    }
    let rest = &bytes[first..];
    for (i, chunk) in rest.chunks(3).enumerate() {
        if i > 0 || first > 0 {
            out.push(',');
        }
        out.push_str(std::str::from_utf8(chunk).unwrap());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_tracks_sum_count_min_max() {
        let mut s = PhaseStats::default();
        s.observe(Duration::from_micros(10));
        s.observe(Duration::from_micros(50));
        s.observe(Duration::from_micros(5));
        assert_eq!(s.count, 3);
        assert_eq!(s.sum_ns, 65_000);
        assert_eq!(s.min_ns, 5_000);
        assert_eq!(s.max_ns, 50_000);
        assert_eq!(s.mean_ns(), 65_000 / 3);
    }

    #[test]
    fn histogram_bucket_indices_are_log2() {
        let mut s = PhaseStats::default();
        s.observe(Duration::from_nanos(1)); // ns < 2 → bucket 0
        s.observe(Duration::from_nanos(2)); // log2(2)=1 → bucket 1
        s.observe(Duration::from_nanos(3)); // log2(3)=1 → bucket 1
        s.observe(Duration::from_nanos(4)); // log2(4)=2 → bucket 2
        s.observe(Duration::from_nanos(1023)); // log2(1023)=9 → bucket 9
        s.observe(Duration::from_nanos(1024)); // log2(1024)=10 → bucket 10
        assert_eq!(s.buckets[0], 1);
        assert_eq!(s.buckets[1], 2);
        assert_eq!(s.buckets[2], 1);
        assert_eq!(s.buckets[9], 1);
        assert_eq!(s.buckets[10], 1);
    }

    #[test]
    fn percentile_interpolates_within_bucket() {
        // 100 samples all in the 1024..2048 ns range (bucket 10).
        // The median (q=0.5) should land roughly mid-bucket
        // (~1536 ns) under linear interpolation.
        let mut s = PhaseStats::default();
        for ns in 1024..1124 {
            s.observe(Duration::from_nanos(ns));
        }
        let p50 = s.percentile_ns(0.5);
        assert!(
            (1024..=2048).contains(&p50),
            "p50 should fall inside bucket [1024, 2048), got {}",
            p50
        );
        // Clamped at min/max:
        assert_eq!(s.percentile_ns(0.0), s.min_ns);
        assert_eq!(s.percentile_ns(1.0), s.max_ns);
    }

    #[test]
    fn empty_phase_formats_as_dash() {
        let s = PhaseStats::default();
        let cell = TickMetrics::fmt_cell(&s, 0.95);
        assert!(cell.trim() == "—");
    }

    #[test]
    fn empty_phase_percentile_is_zero() {
        let s = PhaseStats::default();
        assert_eq!(s.mean_ns(), 0);
        assert_eq!(s.percentile_ns(0.95), 0);
    }

    #[test]
    fn drop_prints_without_panic_on_empty() {
        // Drop on a brand-new TickMetrics must not panic (no
        // division by zero, no out-of-bounds indexing).
        let mut m = TickMetrics::new();
        m.silent = true; // keep the test output clean
        drop(m);
    }

    #[test]
    fn format_commas_basic() {
        assert_eq!(format_commas(0), "0");
        assert_eq!(format_commas(999), "999");
        assert_eq!(format_commas(1_000), "1,000");
        assert_eq!(format_commas(60_000), "60,000");
        assert_eq!(format_commas(1_234_567), "1,234,567");
    }

    #[test]
    fn fmt_hz_strips_trailing_zero_when_integer() {
        assert_eq!(fmt_hz(300.0), "300");
        assert_eq!(fmt_hz(100.0), "100");
        // Duration round-trip noise: `1.0 / 300.0` stored as Duration
        // then reconverted isn't exactly 300.0. Treat it as 300.
        assert_eq!(fmt_hz(300.00003), "300");
        assert_eq!(fmt_hz(33.333333), "33.3");
        assert_eq!(fmt_hz(120.5), "120.5");
    }

    #[test]
    fn set_rates_stores_configuration() {
        let mut m = TickMetrics::new();
        m.silent = true;
        m.set_rates(300.0, 100.0);
        assert_eq!(m.simulation_rate_hz, 300.0);
        assert_eq!(m.telemetry_rate_hz, 100.0);
    }

    #[test]
    fn no_sleep_counts_only_when_behind_real_deadline() {
        let mut m = TickMetrics::new();
        m.silent = true;

        // Lead target already elapsed, but the real deadline is still ahead.
        m.observe_pacing(
            Duration::from_micros(500),
            Duration::ZERO,
            Duration::ZERO,
            -1_000,
            false,
        );
        assert_eq!(m.pacing_no_sleep, 0);

        // Real deadline elapsed: this is a catch-up no-sleep cycle.
        m.observe_pacing(
            Duration::from_micros(500),
            Duration::ZERO,
            Duration::ZERO,
            1_000,
            false,
        );
        assert_eq!(m.pacing_no_sleep, 1);
    }

    fn cycle(m: &mut TickMetrics, post_step_us: u64) {
        m.observe_pre_commit(Duration::from_micros(1));
        m.observe_post_step(Duration::from_micros(post_step_us));
        m.observe_total_cycle(Duration::from_micros(post_step_us + 1));
        m.cycles += 1;
    }

    #[test]
    fn receipt_percentiles_are_exact_over_measured_cycles() {
        let mut m = TickMetrics::new();
        m.silent = true;
        m.record_cycles(Some(101));
        cycle(&mut m, 50_000);
        m.reset_after_warmup(1);
        for us in 1..=100 {
            cycle(&mut m, us);
        }
        let receipt = m.receipt();
        assert_eq!(receipt.percentile_source, "exact");
        assert_eq!(receipt.post_step.p95_ns, 95_000);
        assert_eq!(receipt.post_step.p99_ns, 99_000);
        assert_eq!(receipt.callback.p50_ns, 51_000);
        assert_eq!(receipt.world_run.p95_ns, 0);
    }

    #[test]
    fn receipt_estimates_percentiles_once_samples_overflow() {
        let mut m = TickMetrics::new();
        m.silent = true;
        m.record_cycles(Some(10));
        for us in 1..=20 {
            cycle(&mut m, us);
        }
        assert_eq!(m.receipt().percentile_source, "histogram");
    }
}
