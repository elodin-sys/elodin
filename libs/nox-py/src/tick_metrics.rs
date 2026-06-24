//! Always-on, customer-facing per-phase simulation-cycle timing summary.
//!
//! The `run` / `serve` simulation loop in
//! [`crate::impeller2_server::tick`] walks through seven distinct
//! phases on every iteration (pre_step → db read → kernel →
//! commit → write-barrier → post_step → real-time pacing). One
//! iteration is one **simulation cycle** — one pass from pre_step
//! through pacing, running `ticks_per_telemetry` simulation steps
//! inside `world.run()` and committing once to the DB. At
//! simulation shutdown we show the customer a one-block stdout
//! summary with mean / p95 / max wall time per phase, so they can
//! see where their cycles are spending time without needing a
//! debug flag.
//!
//! Design goals:
//!
//! - **Zero runtime allocation.** Fixed-size histograms, no
//!   `Vec` growth during the loop.
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
}

impl From<&PhaseStats> for PhaseStatsSnapshot {
    fn from(stats: &PhaseStats) -> Self {
        Self {
            sum_ns: stats.sum_ns,
            count: stats.count,
            min_ns: stats.min_ns,
            max_ns: stats.max_ns,
            buckets: stats.buckets,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TickSummaryJson {
    pre_step: PhaseStatsSnapshot,
    copy_db_to_world: PhaseStatsSnapshot,
    world_run: PhaseStatsSnapshot,
    commit: PhaseStatsSnapshot,
    wait_for_write: PhaseStatsSnapshot,
    post_step: PhaseStatsSnapshot,
    real_time_pacing: PhaseStatsSnapshot,
    real_time_oversleep: PhaseStatsSnapshot,
    real_time_behind: PhaseStatsSnapshot,
    pacing_behind_events: u64,
    pacing_resets: u64,
    pacing_no_sleep: u64,
    total_cycle: PhaseStatsSnapshot,
    cycles: u64,
    steps: u64,
    warmup_excluded_ticks: u64,
    simulation_rate_hz: f64,
    telemetry_rate_hz: f64,
    wall_ns: u64,
    entry_unix_ns: Option<u64>,
    compile_done_unix_ns: Option<u64>,
    loop_start_unix_ns: u64,
    loop_end_unix_ns: u64,
    summary_written_unix_ns: u64,
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

/// One row of the optional per-cycle pacing trace. Compact (40 bytes)
/// so a long run's worth fits comfortably in the pre-sized buffer.
#[derive(Clone, Copy)]
struct PacingSample {
    /// Compute time for the cycle (loop-top to pacing-block entry):
    /// pre_step + read + tick + commit + wait_for_write + post_step.
    work_ns: u64,
    /// Sleep duration we *asked* for (`deadline - now`), 0 if behind.
    requested_sleep_ns: u64,
    /// Sleep duration that actually elapsed on the wall clock.
    actual_sleep_ns: u64,
    /// Signed deadline error at the pacing check: `now - deadline`.
    /// Negative = ahead of schedule (good), positive = behind.
    deadline_err_ns: i64,
    /// Whether the drift cap reset the deadline forward this cycle.
    reset: bool,
}

/// Pre-allocated ring of pacing samples plus the CSV destination.
struct PacingTrace {
    path: PathBuf,
    samples: Vec<PacingSample>,
    /// Cap so unbounded (interactive) runs can't grow without limit.
    max: usize,
}

impl PacingTrace {
    fn from_env() -> Option<Self> {
        let path = std::env::var("ELODIN_PACING_TRACE").ok()?;
        let max = std::env::var("ELODIN_PACING_TRACE_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2_000_000usize);
        Some(Self {
            path: PathBuf::from(path),
            samples: Vec::with_capacity(max.min(1_000_000)),
            max,
        })
    }

    #[inline]
    fn push(&mut self, sample: PacingSample) {
        if self.samples.len() < self.max {
            self.samples.push(sample);
        }
    }

    fn flush(&self) {
        if let Some(parent) = self.path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let mut out = String::with_capacity(self.samples.len() * 48 + 64);
        out.push_str(
            "cycle,work_ns,requested_sleep_ns,actual_sleep_ns,oversleep_ns,deadline_err_ns,reset\n",
        );
        for (i, s) in self.samples.iter().enumerate() {
            let oversleep = s.actual_sleep_ns.saturating_sub(s.requested_sleep_ns);
            out.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                i,
                s.work_ns,
                s.requested_sleep_ns,
                s.actual_sleep_ns,
                oversleep,
                s.deadline_err_ns,
                s.reset as u8,
            ));
        }
        let _ = fs::write(&self.path, out);
    }
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
    pub commit: PhaseStats,
    pub wait_for_write: PhaseStats,
    pub post_step: PhaseStats,
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
    /// Count of cycles where no sleep happened because the deadline had
    /// already elapsed (we were behind and trying to catch up).
    pub pacing_no_sleep: u64,
    /// Optional per-cycle pacing trace. Allocated up front (capacity from
    /// `ELODIN_PACING_TRACE_MAX`) and flushed to the CSV path in
    /// `ELODIN_PACING_TRACE` on drop. `None` unless the env var is set, so
    /// default runs allocate nothing and stay on the zero-I/O hot path.
    pacing_trace: Option<PacingTrace>,
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
            commit: PhaseStats::default(),
            wait_for_write: PhaseStats::default(),
            post_step: PhaseStats::default(),
            real_time_pacing: PhaseStats::default(),
            real_time_oversleep: PhaseStats::default(),
            real_time_behind: PhaseStats::default(),
            pacing_behind_events: 0,
            pacing_resets: 0,
            pacing_no_sleep: 0,
            pacing_trace: PacingTrace::from_env(),
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

    /// Discard everything accumulated so far and restart the clock, so the
    /// final summary reflects steady state rather than the simulator/SITL
    /// spin-up (the first cycle can stall for seconds while a flight controller
    /// boots, which otherwise poisons mean cycle / post_step / lateness). Called
    /// once by the loop after the warmup tick threshold is crossed. The
    /// per-cycle pacing trace (if enabled) is intentionally kept intact so the
    /// boot is still visible for offline debugging.
    pub fn reset_after_warmup(&mut self, excluded_ticks: u64) {
        self.pre_step = PhaseStats::default();
        self.copy_db_to_world = PhaseStats::default();
        self.world_run = PhaseStats::default();
        self.commit = PhaseStats::default();
        self.wait_for_write = PhaseStats::default();
        self.post_step = PhaseStats::default();
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
    }

    /// Feed the configured rates so `print_summary` can show the
    /// `X Hz sim / Y Hz telemetry` annotation. Call once, right
    /// after `new`, from the site that knows the world's
    /// `sim_time_step` and `ticks_per_telemetry`.
    pub fn set_rates(&mut self, simulation_rate_hz: f64, telemetry_rate_hz: f64) {
        self.simulation_rate_hz = simulation_rate_hz;
        self.telemetry_rate_hz = telemetry_rate_hz;
    }

    #[inline]
    pub fn observe_pre_step(&mut self, d: Duration) {
        self.pre_step.observe(d);
    }
    #[inline]
    pub fn observe_copy_db_to_world(&mut self, d: Duration) {
        self.copy_db_to_world.observe(d);
    }
    #[inline]
    pub fn observe_world_run(&mut self, d: Duration) {
        self.world_run.observe(d);
    }
    #[inline]
    pub fn observe_commit(&mut self, d: Duration) {
        self.commit.observe(d);
    }
    #[inline]
    pub fn observe_wait_for_write(&mut self, d: Duration) {
        self.wait_for_write.observe(d);
    }
    #[inline]
    pub fn observe_post_step(&mut self, d: Duration) {
        self.post_step.observe(d);
    }
    #[inline]
    pub fn observe_real_time_pacing(&mut self, d: Duration) {
        self.real_time_pacing.observe(d);
    }
    #[inline]
    pub fn observe_total_cycle(&mut self, d: Duration) {
        self.total_cycle.observe(d);
    }

    /// Record one real-time pacing cycle. `work` is the cycle's compute
    /// time, `requested` the sleep we asked for (`Duration::ZERO` when we
    /// were behind and skipped sleeping), `actual` the sleep that actually
    /// elapsed, `deadline_err_ns` the signed `now - deadline` at the pacing
    /// check (negative = ahead), and `reset` whether the drift cap fired.
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
        if requested.is_zero() {
            self.pacing_no_sleep += 1;
        } else {
            self.real_time_oversleep
                .observe(actual.saturating_sub(requested));
        }
        if let Some(trace) = self.pacing_trace.as_mut() {
            trace.push(PacingSample {
                work_ns: u64::try_from(work.as_nanos()).unwrap_or(u64::MAX),
                requested_sleep_ns: u64::try_from(requested.as_nanos()).unwrap_or(u64::MAX),
                actual_sleep_ns: u64::try_from(actual.as_nanos()).unwrap_or(u64::MAX),
                deadline_err_ns,
                reset,
            });
        }
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
        println!("{}", Self::fmt_row("commit world", &self.commit));
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
            "    {:<18} {}    skipped sleep: {}",
            "drift resets:",
            format_commas(self.pacing_resets),
            format_commas(self.pacing_no_sleep),
        );
    }

    pub fn summary_json(&self) -> TickSummaryJson {
        let summary_written_unix_ns = unix_now_ns();
        TickSummaryJson {
            pre_step: (&self.pre_step).into(),
            copy_db_to_world: (&self.copy_db_to_world).into(),
            world_run: (&self.world_run).into(),
            commit: (&self.commit).into(),
            wait_for_write: (&self.wait_for_write).into(),
            post_step: (&self.post_step).into(),
            real_time_pacing: (&self.real_time_pacing).into(),
            real_time_oversleep: (&self.real_time_oversleep).into(),
            real_time_behind: (&self.real_time_behind).into(),
            pacing_behind_events: self.pacing_behind_events,
            pacing_resets: self.pacing_resets,
            pacing_no_sleep: self.pacing_no_sleep,
            total_cycle: (&self.total_cycle).into(),
            cycles: self.cycles,
            steps: self.steps,
            warmup_excluded_ticks: self.warmup_excluded_ticks,
            simulation_rate_hz: self.simulation_rate_hz,
            telemetry_rate_hz: self.telemetry_rate_hz,
            wall_ns: u64::try_from(self.started_at.elapsed().as_nanos()).unwrap_or(u64::MAX),
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
        if let Ok(json) = serde_json::to_string_pretty(&self.summary_json()) {
            let _ = fs::write(path, json);
        }
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
        if let Some(trace) = self.pacing_trace.as_ref() {
            trace.flush();
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
}
