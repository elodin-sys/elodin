use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Lightweight profiling counters for elodin-db hot paths.
/// All counters are global atomics — no allocation on the hot path.
/// Enabled only with `--features profile`.

// ── Phase timers (cumulative nanoseconds) ───────────────────

pub static SINK_TABLE_NS: AtomicU64 = AtomicU64::new(0);
pub static SINK_TABLE_COUNT: AtomicU64 = AtomicU64::new(0);

pub static VTABLE_RESOLVE_NS: AtomicU64 = AtomicU64::new(0);
pub static VTABLE_RESOLVE_FIELDS: AtomicU64 = AtomicU64::new(0);

pub static APPLY_VALUE_NS: AtomicU64 = AtomicU64::new(0);
pub static APPLY_VALUE_COUNT: AtomicU64 = AtomicU64::new(0);

pub static PUSH_BUF_NS: AtomicU64 = AtomicU64::new(0);
pub static PUSH_BUF_COUNT: AtomicU64 = AtomicU64::new(0);

pub static PUSH_BUF_DATA_WRITE_NS: AtomicU64 = AtomicU64::new(0);
pub static PUSH_BUF_INDEX_WRITE_NS: AtomicU64 = AtomicU64::new(0);
pub static PUSH_BUF_TS_CHECK_NS: AtomicU64 = AtomicU64::new(0);

pub static WAKE_ALL_COUNT: AtomicU64 = AtomicU64::new(0);
pub static WAKE_ALL_DATA_WAKER_NS: AtomicU64 = AtomicU64::new(0);
pub static WAKE_ALL_LAST_UPDATED_NS: AtomicU64 = AtomicU64::new(0);
pub static WAKE_ALL_EARLIEST_TS_NS: AtomicU64 = AtomicU64::new(0);

pub static RWLOCK_ACQUIRE_NS: AtomicU64 = AtomicU64::new(0);
pub static RWLOCK_ACQUIRE_COUNT: AtomicU64 = AtomicU64::new(0);

pub static HASHMAP_LOOKUP_NS: AtomicU64 = AtomicU64::new(0);
pub static HASHMAP_LOOKUP_COUNT: AtomicU64 = AtomicU64::new(0);

// ── Per-tick histograms (ring buffer of recent ticks) ───────

pub static TICK_NS_RING: [AtomicU64; RING_SIZE] = {
    const INIT: AtomicU64 = AtomicU64::new(0);
    [INIT; RING_SIZE]
};
pub static TICK_INDEX: AtomicU64 = AtomicU64::new(0);
pub const RING_SIZE: usize = 8192;

// ── Outlier detection ───────────────────────────────────────

pub static PUSH_BUF_MAX_NS: AtomicU64 = AtomicU64::new(0);
pub static APPLY_VALUE_MAX_NS: AtomicU64 = AtomicU64::new(0);

/// RAII guard that accumulates elapsed time into an atomic counter.
pub struct TimedSection {
    start: Instant,
    target_ns: &'static AtomicU64,
}

impl TimedSection {
    #[inline(always)]
    pub fn begin(target_ns: &'static AtomicU64) -> Self {
        Self {
            start: Instant::now(),
            target_ns,
        }
    }

    #[inline(always)]
    pub fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }
}

impl Drop for TimedSection {
    #[inline(always)]
    fn drop(&mut self) {
        let ns = self.start.elapsed().as_nanos() as u64;
        self.target_ns.fetch_add(ns, Ordering::Relaxed);
    }
}

/// Record a tick duration into the ring buffer.
#[inline]
pub fn record_tick(ns: u64) {
    let idx = TICK_INDEX.fetch_add(1, Ordering::Relaxed) as usize % RING_SIZE;
    TICK_NS_RING[idx].store(ns, Ordering::Relaxed);
}

/// Reset all counters to zero.
pub fn reset_all() {
    for counter in [
        &SINK_TABLE_NS,
        &SINK_TABLE_COUNT,
        &VTABLE_RESOLVE_NS,
        &VTABLE_RESOLVE_FIELDS,
        &APPLY_VALUE_NS,
        &APPLY_VALUE_COUNT,
        &PUSH_BUF_NS,
        &PUSH_BUF_COUNT,
        &PUSH_BUF_DATA_WRITE_NS,
        &PUSH_BUF_INDEX_WRITE_NS,
        &PUSH_BUF_TS_CHECK_NS,
        &WAKE_ALL_COUNT,
        &WAKE_ALL_DATA_WAKER_NS,
        &WAKE_ALL_LAST_UPDATED_NS,
        &WAKE_ALL_EARLIEST_TS_NS,
        &RWLOCK_ACQUIRE_NS,
        &RWLOCK_ACQUIRE_COUNT,
        &HASHMAP_LOOKUP_NS,
        &HASHMAP_LOOKUP_COUNT,
        &PUSH_BUF_MAX_NS,
        &APPLY_VALUE_MAX_NS,
        &TICK_INDEX,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
    for slot in &TICK_NS_RING {
        slot.store(0, Ordering::Relaxed);
    }
}

/// Snapshot all counters into a struct for reporting.
pub fn snapshot() -> ProfileSnapshot {
    let tick_count = TICK_INDEX.load(Ordering::Relaxed).min(RING_SIZE as u64);
    let mut tick_durations: Vec<u64> = (0..tick_count as usize)
        .map(|i| TICK_NS_RING[i].load(Ordering::Relaxed))
        .filter(|&v| v > 0)
        .collect();
    tick_durations.sort_unstable();

    ProfileSnapshot {
        sink_table_ns: SINK_TABLE_NS.load(Ordering::Relaxed),
        sink_table_count: SINK_TABLE_COUNT.load(Ordering::Relaxed),
        vtable_resolve_ns: VTABLE_RESOLVE_NS.load(Ordering::Relaxed),
        vtable_resolve_fields: VTABLE_RESOLVE_FIELDS.load(Ordering::Relaxed),
        apply_value_ns: APPLY_VALUE_NS.load(Ordering::Relaxed),
        apply_value_count: APPLY_VALUE_COUNT.load(Ordering::Relaxed),
        push_buf_ns: PUSH_BUF_NS.load(Ordering::Relaxed),
        push_buf_count: PUSH_BUF_COUNT.load(Ordering::Relaxed),
        push_buf_data_write_ns: PUSH_BUF_DATA_WRITE_NS.load(Ordering::Relaxed),
        push_buf_index_write_ns: PUSH_BUF_INDEX_WRITE_NS.load(Ordering::Relaxed),
        push_buf_ts_check_ns: PUSH_BUF_TS_CHECK_NS.load(Ordering::Relaxed),
        wake_all_count: WAKE_ALL_COUNT.load(Ordering::Relaxed),
        wake_all_data_waker_ns: WAKE_ALL_DATA_WAKER_NS.load(Ordering::Relaxed),
        wake_all_last_updated_ns: WAKE_ALL_LAST_UPDATED_NS.load(Ordering::Relaxed),
        wake_all_earliest_ts_ns: WAKE_ALL_EARLIEST_TS_NS.load(Ordering::Relaxed),
        rwlock_acquire_ns: RWLOCK_ACQUIRE_NS.load(Ordering::Relaxed),
        rwlock_acquire_count: RWLOCK_ACQUIRE_COUNT.load(Ordering::Relaxed),
        hashmap_lookup_ns: HASHMAP_LOOKUP_NS.load(Ordering::Relaxed),
        hashmap_lookup_count: HASHMAP_LOOKUP_COUNT.load(Ordering::Relaxed),
        push_buf_max_ns: PUSH_BUF_MAX_NS.load(Ordering::Relaxed),
        apply_value_max_ns: APPLY_VALUE_MAX_NS.load(Ordering::Relaxed),
        tick_durations,
    }
}

pub struct ProfileSnapshot {
    pub sink_table_ns: u64,
    pub sink_table_count: u64,
    pub vtable_resolve_ns: u64,
    pub vtable_resolve_fields: u64,
    pub apply_value_ns: u64,
    pub apply_value_count: u64,
    pub push_buf_ns: u64,
    pub push_buf_count: u64,
    pub push_buf_data_write_ns: u64,
    pub push_buf_index_write_ns: u64,
    pub push_buf_ts_check_ns: u64,
    pub wake_all_count: u64,
    pub wake_all_data_waker_ns: u64,
    pub wake_all_last_updated_ns: u64,
    pub wake_all_earliest_ts_ns: u64,
    pub rwlock_acquire_ns: u64,
    pub rwlock_acquire_count: u64,
    pub hashmap_lookup_ns: u64,
    pub hashmap_lookup_count: u64,
    pub push_buf_max_ns: u64,
    pub apply_value_max_ns: u64,
    pub tick_durations: Vec<u64>,
}

impl ProfileSnapshot {
    pub fn percentile(&self, p: f64) -> u64 {
        if self.tick_durations.is_empty() {
            return 0;
        }
        let idx = ((p / 100.0) * (self.tick_durations.len() - 1) as f64) as usize;
        self.tick_durations[idx.min(self.tick_durations.len() - 1)]
    }

    pub fn mean_tick_ns(&self) -> u64 {
        if self.tick_durations.is_empty() {
            return 0;
        }
        let sum: u64 = self.tick_durations.iter().sum();
        sum / self.tick_durations.len() as u64
    }

    fn ns_to_us(ns: u64) -> f64 {
        ns as f64 / 1000.0
    }

    fn pct_of(part: u64, total: u64) -> f64 {
        if total == 0 {
            0.0
        } else {
            (part as f64 / total as f64) * 100.0
        }
    }

    fn mean_ns(total_ns: u64, count: u64) -> f64 {
        if count == 0 {
            0.0
        } else {
            total_ns as f64 / count as f64
        }
    }

    pub fn generate_report(&self, scenario: &str, components: usize, frequency: u32, duration_secs: u64, mode: &str) -> String {
        let total = self.sink_table_ns;
        let mut report = String::with_capacity(8192);

        report.push_str(&format!("# elodin-db Profiling Report\n\n"));
        report.push_str(&format!("**Generated:** {}\n\n", chrono_now()));
        report.push_str("## Configuration\n\n");
        report.push_str(&format!("| Parameter | Value |\n|---|---|\n"));
        report.push_str(&format!("| Scenario | {} |\n", scenario));
        report.push_str(&format!("| Components | {} |\n", components));
        report.push_str(&format!("| Frequency | {} Hz |\n", frequency));
        report.push_str(&format!("| Duration | {} s |\n", duration_secs));
        report.push_str(&format!("| Mode | {} |\n", mode));
        report.push_str(&format!("| Total ticks recorded | {} |\n", self.tick_durations.len()));
        report.push_str(&format!("| Total `sink_table` calls | {} |\n\n", self.sink_table_count));

        // Tick latency distribution
        report.push_str("## Tick Latency Distribution\n\n");
        report.push_str("| Percentile | Latency |\n|---|---|\n");
        for (label, p) in [("p50", 50.0), ("p75", 75.0), ("p90", 90.0), ("p95", 95.0), ("p99", 99.0), ("max", 100.0)] {
            report.push_str(&format!("| {} | {:.1} µs |\n", label, Self::ns_to_us(self.percentile(p))));
        }
        report.push_str(&format!("| mean | {:.1} µs |\n\n", Self::ns_to_us(self.mean_tick_ns())));

        // Phase breakdown
        report.push_str("## Phase Breakdown (cumulative)\n\n");
        report.push_str("| Phase | Total Time | % of sink_table | Calls | Mean/call | Source |\n");
        report.push_str("|---|---|---|---|---|---|\n");

        let phases: &[(&str, u64, u64, &str)] = &[
            ("RwLock acquire", self.rwlock_acquire_ns, self.rwlock_acquire_count, "`lib.rs` `db.with_state()`"),
            ("VTable resolve", self.vtable_resolve_ns, self.vtable_resolve_fields, "`vtable.rs` `realize_fields()`"),
            ("apply_value total", self.apply_value_ns, self.apply_value_count, "`lib.rs` `DBSink::apply_value()`"),
            ("  push_buf", self.push_buf_ns, self.push_buf_count, "`time_series.rs` `push_buf()`"),
            ("    data write", self.push_buf_data_write_ns, self.push_buf_count, "`append_log.rs` `data.write()`"),
            ("    index write", self.push_buf_index_write_ns, self.push_buf_count, "`append_log.rs` `index.write()`"),
            ("    timestamp check", self.push_buf_ts_check_ns, self.push_buf_count, "`time_series.rs` last_ts read"),
            ("  wake_all (data)", self.wake_all_data_waker_ns, self.apply_value_count, "`time_series.rs` `data_waker`"),
            ("  wake_all (last_upd)", self.wake_all_last_updated_ns, self.apply_value_count, "`lib.rs` `update_max()`"),
            ("  wake_all (earliest)", self.wake_all_earliest_ts_ns, self.apply_value_count, "`lib.rs` `update_min()`"),
            ("  HashMap lookup", self.hashmap_lookup_ns, self.hashmap_lookup_count, "`lib.rs` `components.get()`"),
        ];

        for (name, ns, count, source) in phases {
            report.push_str(&format!(
                "| {} | {:.1} ms | {:.1}% | {} | {:.0} ns | {} |\n",
                name,
                *ns as f64 / 1_000_000.0,
                Self::pct_of(*ns, total),
                count,
                Self::mean_ns(*ns, *count),
                source,
            ));
        }

        // Bottleneck analysis
        report.push_str("\n## Identified Bottlenecks\n\n");

        let mut bottlenecks: Vec<(&str, u64, String)> = vec![
            ("wake_all() cascade", self.wake_all_data_waker_ns + self.wake_all_last_updated_ns + self.wake_all_earliest_ts_ns,
             format!(
                 "{} total calls ({}/tick). `update_min` rarely changes the value but calls `wake_all()` unconditionally. \
                  `update_max` is called {}× per tick with the same timestamp — only the first matters.",
                 self.wake_all_count,
                 self.wake_all_count / self.sink_table_count.max(1),
                 self.apply_value_count / self.sink_table_count.max(1),
             )),
            ("VTable realize chain", self.vtable_resolve_ns,
             format!(
                 "{} field resolutions ({}/tick). Each field triggers ~7 recursive `realize()` calls. \
                  The VTable structure is static — results could be cached after first resolve.",
                 self.vtable_resolve_fields,
                 self.vtable_resolve_fields / self.sink_table_count.max(1),
             )),
            ("push_buf Mutex churn", self.push_buf_data_write_ns + self.push_buf_index_write_ns,
             format!(
                 "2 Mutex acquisitions per component per tick = {} lock/unlock pairs. \
                  Uncontested in single-client batch mode but still ~20 ns each.",
                 self.push_buf_count * 2,
             )),
            ("HashMap lookups", self.hashmap_lookup_ns,
             format!(
                 "{} lookups. With {} components the table fits in L2 cache, \
                  but pointer chasing to heap-allocated Components causes cache misses.",
                 self.hashmap_lookup_count, components,
             )),
        ];

        bottlenecks.sort_by(|a, b| b.1.cmp(&a.1));

        for (i, (name, ns, explanation)) in bottlenecks.iter().enumerate() {
            report.push_str(&format!(
                "### #{} — {} ({:.1} ms, {:.1}% of total)\n\n{}\n\n",
                i + 1,
                name,
                *ns as f64 / 1_000_000.0,
                Self::pct_of(*ns, total),
                explanation,
            ));
        }

        // Outliers
        report.push_str("## Outlier Detection\n\n");
        report.push_str(&format!(
            "| Metric | Value |\n|---|---|\n\
             | Worst single `push_buf` | {:.1} µs |\n\
             | Worst single `apply_value` | {:.1} µs |\n\
             | Tick p99/p50 ratio | {:.1}× |\n\n",
            Self::ns_to_us(self.push_buf_max_ns),
            Self::ns_to_us(self.apply_value_max_ns),
            if self.percentile(50.0) > 0 {
                self.percentile(99.0) as f64 / self.percentile(50.0) as f64
            } else {
                0.0
            },
        ));

        if self.push_buf_max_ns > self.mean_tick_ns() / 2 {
            report.push_str(
                "> **Warning:** A single `push_buf` call took more than half the mean tick time. \
                 This is likely a **mmap page fault** on first write to a new 4 KB page.\n\n"
            );
        }

        // Recommendations
        report.push_str("## Optimization Recommendations\n\n");

        let wake_total = self.wake_all_data_waker_ns + self.wake_all_last_updated_ns + self.wake_all_earliest_ts_ns;
        if Self::pct_of(wake_total, total) > 15.0 {
            report.push_str(&format!(
                "1. **Debounce `wake_all()` in `apply_value`** — Currently {:.0}% of tick time. \
                 Call `update_max`/`update_min` once after the entire batch instead of per-component. \
                 Skip `wake_all()` when the atomic value didn't actually change.\n\n",
                Self::pct_of(wake_total, total),
            ));
        }

        if Self::pct_of(self.vtable_resolve_ns, total) > 20.0 {
            report.push_str(&format!(
                "2. **Cache VTable resolution** — Currently {:.0}% of tick time. \
                 Pre-compile the field→component mapping on first `apply()` and reuse it on subsequent ticks, \
                 bypassing the recursive `realize()` chain.\n\n",
                Self::pct_of(self.vtable_resolve_ns, total),
            ));
        }

        if Self::pct_of(self.push_buf_data_write_ns + self.push_buf_index_write_ns, total) > 10.0 {
            report.push_str(
                "3. **Batch mmap commits** — Instead of 2 separate `AppendLog::write()` calls \
                 (data + index) per component, accumulate all writes and commit once per tick.\n\n"
            );
        }

        report
    }
}

fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let hours = (secs / 3600) % 24;
    let minutes = (secs / 60) % 60;
    let seconds = secs % 60;
    format!("{:02}:{:02}:{:02} UTC", hours, minutes, seconds)
}
