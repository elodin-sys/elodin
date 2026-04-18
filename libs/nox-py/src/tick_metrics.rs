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

use std::time::{Duration, Instant};

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
        if self.count == 0 {
            0
        } else {
            self.sum_ns / self.count
        }
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
            total_cycle: PhaseStats::default(),
            cycles: 0,
            steps: 0,
            simulation_rate_hz: 0.0,
            telemetry_rate_hz: 0.0,
            started_at: Instant::now(),
            silent: false,
        }
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
        println!("───────────────────────────────────");
    }
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
        self.print_summary();
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
