//! Shared plumbing for terrain-data fetch binaries: worker-count parsing and
//! compact progress reporting for tile phases.
//!
//! The heavy fetch/stitch/sampling primitives live in [`crate::fetch`]. This
//! module intentionally stays UI/orchestration-only so binaries don't each grow
//! their own copy of the same progress-bar and worker parsing helpers.

use crate::fetch::{FetchStats, TileRange};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::Duration;

/// Read a positive worker count from `env_var`, otherwise return `default`.
pub fn workers_from_env_or(env_var: &str, default: usize) -> usize {
    std::env::var(env_var)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Read a positive worker count from `env_var`, otherwise use OS-reported CPU
/// parallelism, falling back to `fallback` if that cannot be detected.
pub fn workers_from_env_or_available(env_var: &str, fallback: usize) -> usize {
    std::env::var(env_var)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| available_parallelism_or(fallback))
}

/// Return OS-reported CPU parallelism, or `fallback` if unavailable.
pub fn available_parallelism_or(fallback: usize) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(fallback)
}

/// Metadata for one progress-reported tile phase.
#[derive(Debug, Clone, Copy)]
pub struct TilePhase {
    pub label: &'static str,
    pub z: u8,
    pub cols: u32,
    pub rows: u32,
    pub total: u64,
}

impl TilePhase {
    pub fn from_range(label: &'static str, range: TileRange) -> Self {
        let cols = range.x_max - range.x_min + 1;
        let rows = range.y_max - range.y_min + 1;
        Self::from_grid(label, range.z, cols, rows)
    }

    pub fn square(label: &'static str, z: u8, side: u32) -> Self {
        Self::from_grid(label, z, side, side)
    }

    pub fn from_grid(label: &'static str, z: u8, cols: u32, rows: u32) -> Self {
        Self {
            label,
            z,
            cols,
            rows,
            total: u64::from(cols) * u64::from(rows),
        }
    }

    /// Print the standard `label z=Z CxR = N tile(s)` phase banner. Plain
    /// `eprintln!` keeps the line visible for non-TTY/scripted runs.
    pub fn print_banner(self) {
        eprintln!(
            "    {:12} z={} {}x{} = {} tile(s)",
            self.label, self.z, self.cols, self.rows, self.total
        );
    }

    pub fn progress_bar(self, mp: &MultiProgress) -> ProgressBar {
        mp.add(make_progress_bar(self.total, self.label))
    }

    /// Finalise a phase: stop the bar and emit the `done in HH:MM:SS — ...`
    /// summary text. Plain `eprintln!` keeps the line visible in non-TTY runs.
    pub fn print_summary(self, pb: &ProgressBar, stats: FetchStats) {
        let elapsed = pb.elapsed();
        pb.finish_and_clear();
        let rate = self.total as f64 / elapsed.as_secs_f64().max(1e-6);
        let retries_suffix = if stats.retries == 0 {
            String::new()
        } else {
            format!(", {} retries", stats.retries)
        };
        eprintln!(
            "    {:12} z={} done in {} — {} net + {} cached{} ({rate:.1} tile/s avg)",
            self.label,
            self.z,
            format_duration(elapsed),
            stats.network_hits,
            stats.cache_hits,
            retries_suffix,
        );
    }
}

/// Build an [`indicatif::ProgressBar`] sized to `total` with our standard
/// fetch-phase template.
pub fn make_progress_bar(total: u64, label: &'static str) -> ProgressBar {
    let style = ProgressStyle::with_template(
        "    {prefix:12} [{elapsed_precise}] [{bar:40.cyan/blue}] \
         {pos}/{len} {msg} {per_sec} ETA {eta}",
    )
    .expect("static progress template parses")
    .progress_chars("#>-");
    let pb = ProgressBar::new(total);
    pb.set_style(style);
    pb.set_prefix(label);
    pb
}

fn format_duration(d: Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

/// Render provider fetch counters for the progress bar message slot.
pub fn format_stats_message(s: FetchStats) -> String {
    if s.retries == 0 {
        format!("net={} cached={}", s.network_hits, s.cache_hits)
    } else {
        format!(
            "net={} cached={} retries={}",
            s.network_hits, s.cache_hits, s.retries
        )
    }
}
