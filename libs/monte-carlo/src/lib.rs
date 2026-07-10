use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::io::Write;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use miette::{Context, IntoDiagnostic, Result, miette};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use stellarator::util::CancelToken;
use tokio::task::JoinSet;

mod fs_util;
pub mod ports;

pub const CONTEXT_ENV: &str = "ELODIN_MONTE_CARLO_CONTEXT";
pub const CACHE_ENV: &str = "ELODIN_CACHE_DIR";

/// Hook scalar keys that have dedicated, typed columns elsewhere (`pass` is the
/// raw scoring verdict surfaced as `scored_pass`/`passed`; `valid` as
/// `scored_valid`/`valid`). They are excluded from the dynamic `hook_scalars`
/// columns so results.csv never emits a duplicate `valid`/`pass` header.
const RESERVED_HOOK_KEYS: &[&str] = &["pass", "valid"];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CampaignConfig {
    /// Concurrent worker slots. The runner sizes the s10 admission budget to
    /// `workers * recipe_weight` so exactly this many runs execute at once.
    /// Precedence: `--workers` flag > `S10_MAX_INFLIGHT` env (escape hatch) >
    /// this key > logical-core default.
    pub workers: Option<usize>,
    pub timeout: Option<String>,
    pub retries: usize,
    pub cache_dir: Option<PathBuf>,
    /// Run per-run IO on a fast scratch filesystem, moving each run's
    /// surviving artifacts to `--out` as it finishes. `"auto"` picks
    /// `/dev/shm` when present; any other value is used as the scratch root;
    /// unset/`"off"` runs directly in `--out`.
    pub scratch_dir: Option<String>,
    /// Stagger first-wave worker starts (`"auto"`, default) so N simultaneous
    /// cold interpreter starts and JIT compiles don't distort the real-time
    /// pacing of the runs that start first. Steady state staggers naturally
    /// through slot turnover; only the first wave needs help. `"off"` starts
    /// every worker at once.
    pub rampup: String,
    /// Build steps executed once before workers start (`[[build]]` tables).
    pub build: Vec<BuildConfig>,
    /// Extra environment for every run's processes. Runner-managed variables
    /// (ports, context paths, seeds) always win over these.
    pub env: BTreeMap<String, String>,
    pub resources: ResourceConfig,
    pub hooks: HookConfig,
    pub params_delivery: Option<ParamsDeliveryConfig>,
    pub retention: RetentionConfig,
    pub quality: QualityConfig,
    pub continue_on_error: bool,
    /// When true, the campaign process exits non-zero if any run failed scoring,
    /// crashed, or was marked invalid/no-data. Defaults to false so exploratory
    /// campaigns can finish with partial failures and still produce reports.
    pub fail_on_run_errors: bool,
}

impl Default for CampaignConfig {
    fn default() -> Self {
        Self {
            workers: None,
            timeout: None,
            retries: 0,
            cache_dir: None,
            scratch_dir: None,
            rampup: "auto".to_string(),
            build: Vec::new(),
            env: BTreeMap::new(),
            resources: ResourceConfig::default(),
            hooks: HookConfig::default(),
            params_delivery: None,
            retention: RetentionConfig::default(),
            quality: QualityConfig::default(),
            continue_on_error: true,
            fail_on_run_errors: false,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BuildConfig {
    pub command: Option<String>,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env: HashMap<String, String>,
}

/// Pacing-integrity gate: real-time-paced sims degrade physics (not exit
/// codes) under oversubscription, so violating runs are marked `degraded` —
/// distinct from `invalid` — and excluded from passes.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct QualityConfig {
    /// Maximum fraction of paced cycles allowed behind deadline (e.g. 0.05).
    pub max_behind_deadline_frac: Option<f64>,
    /// Maximum wall/sim-time ratio for the sim loop (e.g. 1.05).
    pub max_real_time_factor: Option<f64>,
    /// Count degraded runs toward `fail_on_run_errors`. Off by default.
    pub fail_on_degraded: bool,
}

/// A port base: a static u16 (shifted by `worker_id * port_stride`) or
/// `"auto"` (allocated fresh for every run, collision-free by construction).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum PortSpec {
    Static(u16),
    Auto(AutoTag),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AutoTag {
    #[serde(rename = "auto")]
    Auto,
}

impl PortSpec {
    pub fn is_auto(&self) -> bool {
        matches!(self, PortSpec::Auto(_))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ResourceConfig {
    pub bind_ip: IpAddr,
    pub port_stride: u16,
    pub db_port: PortSpec,
    pub ports: BTreeMap<String, PortSpec>,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            bind_ip: IpAddr::V6(Ipv6Addr::UNSPECIFIED),
            port_stride: 20,
            db_port: PortSpec::Static(2240),
            // Named sidecar ports default to dynamic allocation: an empty
            // `[resources]` block never collides or overflows at any worker
            // count. Sims read them via `el.monte_carlo.port(name)`.
            ports: BTreeMap::from([
                ("state".to_string(), PortSpec::Auto(AutoTag::Auto)),
                ("command".to_string(), PortSpec::Auto(AutoTag::Auto)),
            ]),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ParamsDeliveryConfig {
    pub file: PathBuf,
    pub format: ParamsDeliveryFormat,
    pub env_var: Option<String>,
    pub env: BTreeMap<String, String>,
}

impl Default for ParamsDeliveryConfig {
    fn default() -> Self {
        Self {
            file: PathBuf::from("params.json"),
            format: ParamsDeliveryFormat::Json,
            env_var: None,
            env: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ParamsDeliveryFormat {
    #[default]
    Json,
    Toml,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RetentionConfig {
    pub keep_run_db: RunDbRetention,
    /// Glob patterns (relative to the run dir) removed after the post-run
    /// hook when the run passed, so hooks don't carry bespoke cleanup code.
    pub prune_on_pass: Vec<String>,
    /// Same, for runs that failed or were invalid.
    pub prune_on_fail: Vec<String>,
    /// Truncate kept run DBs' preallocated (sparse) files to their real size
    /// once the run's processes have exited. See `elodin-db compact`.
    pub compact_run_db: bool,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            keep_run_db: RunDbRetention::default(),
            prune_on_pass: Vec::new(),
            prune_on_fail: Vec::new(),
            compact_run_db: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum RunDbRetention {
    #[default]
    Always,
    Never,
    OnFail,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HookConfig {
    pub post_run: Option<PathBuf>,
    pub post_campaign: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub sim_path: PathBuf,
    pub plan_path: PathBuf,
    pub campaign_path: Option<PathBuf>,
    pub out_dir: PathBuf,
    pub cache_dir: Option<PathBuf>,
    /// `--workers` flag; wins over `S10_MAX_INFLIGHT` and `campaign.toml`.
    pub workers: Option<usize>,
    pub scratch_dir: Option<String>,
    pub retries: Option<usize>,
    pub timeout: Option<String>,
    pub post_run: Option<PathBuf>,
    pub post_campaign: Option<PathBuf>,
    pub fail_fast: bool,
    pub fail_on_run_errors: bool,
    pub dry_run: bool,
    pub memory_probe: bool,
    pub keep_existing: bool,
    pub clean: bool,
    /// Escalate ephemeral-range port warnings to errors.
    pub strict_ports: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedRunShape {
    pub runtime_threads: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanRow {
    pub run_id: String,
    pub seed: Option<u64>,
    pub params: BTreeMap<String, Value>,
    pub meta: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceSlot {
    pub worker_id: usize,
    pub db_addr: SocketAddr,
    pub db_port: u16,
    pub ports: BTreeMap<String, u16>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetric {
    pub run_id: String,
    /// Worker slot that executed the run, or `None` for runs that were never
    /// scheduled onto a worker (e.g. skipped after `fail_fast`).
    pub worker_id: Option<usize>,
    pub attempt: usize,
    pub status: String,
    pub exit_ok: bool,
    /// Pass/fail outcome reported by the `post_run` hook via the `pass` field of
    /// `post_run_result.json`. `None` when no hook ran or it reported no `pass`
    /// outcome, in which case the run is judged purely on `exit_ok`.
    #[serde(default)]
    pub scored_pass: Option<bool>,
    #[serde(default)]
    pub scored_valid: Option<bool>,
    /// One-line machine-readable reason for a failed/invalid run: the recipe
    /// error chain, a timeout, a preflight port conflict, or a setup error.
    #[serde(default)]
    pub failure_reason: Option<String>,
    /// The run completed but violated the `[quality]` pacing gate: its
    /// physics ran under degraded real-time pacing and the sample should be
    /// re-run or excluded rather than silently ingested.
    #[serde(default)]
    pub degraded: bool,
    /// Fraction of paced cycles that missed their real-time deadline.
    #[serde(default)]
    pub behind_deadline_frac: Option<f64>,
    /// Sim-loop wall time divided by simulated time (1.0 = exact real time).
    #[serde(default)]
    pub real_time_factor: Option<f64>,
    /// Times the real-time pacer hit its drift cap and re-anchored.
    #[serde(default)]
    pub drift_resets: Option<u64>,
    #[serde(default)]
    pub hook_scalars: BTreeMap<String, String>,
    pub wall_ms: u128,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub spawn_unix_ns: u64,
    pub exit_unix_ns: u64,
    pub entry_unix_ns: Option<u64>,
    pub compile_done_unix_ns: Option<u64>,
    pub loop_start_unix_ns: Option<u64>,
    pub loop_end_unix_ns: Option<u64>,
    pub summary_written_unix_ns: Option<u64>,
    pub python_import_ms: Option<f64>,
    pub compile_ms: Option<f64>,
    pub loop_ms: Option<f64>,
    pub teardown_ms: Option<f64>,
    pub process_shutdown_ms: Option<f64>,
    pub db_path: PathBuf,
    pub run_dir: PathBuf,
}

impl RunMetric {
    /// Whether the run counts as a pass: it must exit cleanly, not be
    /// quality-degraded, *and*, if a `post_run` hook scored it, the hook's
    /// `pass` outcome must be true. A run can exit successfully yet fail its
    /// scoring criteria.
    pub fn passed(&self) -> bool {
        self.valid() && !self.degraded && self.exit_ok && self.scored_pass.unwrap_or(true)
    }

    pub fn valid(&self) -> bool {
        self.status != "skipped" && self.scored_valid.unwrap_or(true)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignSummary {
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub total_runs: usize,
    pub passed: usize,
    pub failed: usize,
    #[serde(default)]
    pub invalid: usize,
    #[serde(default)]
    pub degraded: usize,
    pub skipped: usize,
    pub workers: usize,
    pub wall_ms: u128,
    pub total_run_wall_ms: u128,
    pub average_run_wall_ms: f64,
    pub max_run_wall_ms: u128,
    pub parallel_efficiency: f64,
    pub disk_bytes: u64,
    pub resource_summary: ResourceSummary,
    pub sim_phase_summary: Option<SimSummaryAggregate>,
    pub phase_attribution: PhaseAttributionSummary,
    pub concurrency_summary: ConcurrencySummary,
    #[serde(default)]
    pub hook_metrics: BTreeMap<String, ScalarMetricSummary>,
    /// Real-time pacing rollup across paced runs (worst offenders called out
    /// so load-skewed samples are visible without opening results.csv).
    #[serde(default)]
    pub pacing: Option<PacingSummary>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PacingSummary {
    pub paced_runs: usize,
    pub max_behind_deadline_frac: f64,
    pub max_behind_run: String,
    pub max_real_time_factor: f64,
    pub max_real_time_factor_run: String,
}

impl CampaignSummary {
    /// Runs that should trip a `fail_on_run_errors` CI gate: scored failures
    /// plus invalid (no-data / infrastructure) runs that never produced a
    /// scoreable result. Matches the documented "failed or missed scoring".
    pub fn error_runs(&self) -> usize {
        self.failed + self.invalid
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScalarMetricSummary {
    pub count: usize,
    pub min: f64,
    pub mean: f64,
    pub p95: f64,
    pub max: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhaseAttributionSummary {
    pub samples: usize,
    pub average_python_import_ms: f64,
    pub average_compile_ms: f64,
    pub average_loop_ms: f64,
    pub average_teardown_ms: f64,
    pub average_process_shutdown_ms: f64,
    pub p95_python_import_ms: f64,
    pub p95_compile_ms: f64,
    pub p95_loop_ms: f64,
    pub p95_teardown_ms: f64,
    pub p95_process_shutdown_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimelineSample {
    pub start_ms: f64,
    pub end_ms: f64,
    pub active_runs: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConcurrencyBucket {
    pub concurrency: usize,
    pub runs: usize,
    pub average_run_wall_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConcurrencySummary {
    pub mean_active_runs: f64,
    pub peak_active_runs: usize,
    pub buckets: Vec<ConcurrencyBucket>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheMappingSample {
    pub pid: u32,
    pub virtual_kib: u64,
    pub rss_kib: u64,
    pub pss_kib: u64,
    pub cmd: String,
    pub paths: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceSample {
    pub elapsed_ms: u128,
    pub cpu_percent: f64,
    pub cpu_min_percent: f64,
    pub cpu_max_percent: f64,
    pub load_average_1m: f64,
    pub context_switches_per_sec: f64,
    pub mem_total_kib: u64,
    pub mem_available_kib: u64,
    pub campaign_disk_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceSummary {
    pub samples: usize,
    pub average_cpu_percent: f64,
    pub peak_cpu_percent: f64,
    pub peak_cpu_core_percent: f64,
    pub peak_load_average_1m: f64,
    pub peak_context_switches_per_sec: f64,
    pub peak_memory_used_kib: u64,
    pub peak_campaign_disk_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessSample {
    pub elapsed_ms: u128,
    pub run_id: String,
    pub pid: u32,
    pub rss_kib: u64,
    pub utime_ticks: u64,
    pub stime_ticks: u64,
    pub cmd: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimPhaseSummary {
    pub sum_ns: u64,
    pub count: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub buckets: [u32; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimRunSummary {
    pub pre_step: SimPhaseSummary,
    pub copy_db_to_world: SimPhaseSummary,
    pub world_run: SimPhaseSummary,
    pub commit: SimPhaseSummary,
    pub wait_for_write: SimPhaseSummary,
    pub post_step: SimPhaseSummary,
    pub real_time_pacing: SimPhaseSummary,
    pub total_cycle: SimPhaseSummary,
    // Pacing-integrity fields written by the sim's tick metrics
    // (libs/nox-py/src/tick_metrics.rs `TickSummaryJson`).
    #[serde(default)]
    pub real_time_oversleep: Option<SimPhaseSummary>,
    #[serde(default)]
    pub real_time_behind: Option<SimPhaseSummary>,
    #[serde(default)]
    pub pacing_behind_events: u64,
    #[serde(default)]
    pub pacing_resets: u64,
    #[serde(default)]
    pub pacing_no_sleep: u64,
    pub cycles: u64,
    pub steps: u64,
    pub simulation_rate_hz: f64,
    pub telemetry_rate_hz: f64,
    pub wall_ns: u64,
    pub entry_unix_ns: Option<u64>,
    pub compile_done_unix_ns: Option<u64>,
    pub loop_start_unix_ns: Option<u64>,
    pub loop_end_unix_ns: Option<u64>,
    pub summary_written_unix_ns: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimSummaryAggregate {
    pub runs: usize,
    pub pre_step: SimPhaseSummary,
    pub copy_db_to_world: SimPhaseSummary,
    pub world_run: SimPhaseSummary,
    pub commit: SimPhaseSummary,
    pub wait_for_write: SimPhaseSummary,
    pub post_step: SimPhaseSummary,
    pub real_time_pacing: SimPhaseSummary,
    pub total_cycle: SimPhaseSummary,
    pub cycles: u64,
    pub steps: u64,
    pub simulation_rate_hz: f64,
    pub telemetry_rate_hz: f64,
    pub wall_ns: u64,
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("campaign generated no runs")]
    EmptyPlan,
}

const MAX_WORKER_PROGRESS_BARS: usize = 16;
const PROGRESS_TICK_MS: u64 = 120;

#[derive(Clone)]
struct ActiveRun {
    run_id: String,
    started: Instant,
}

#[derive(Clone)]
struct CampaignReporter {
    multi: MultiProgress,
    progress: ProgressBar,
    worker_bars: Arc<Vec<ProgressBar>>,
    slots: Arc<Vec<Mutex<Option<ActiveRun>>>>,
    durations: Arc<Mutex<Vec<u128>>>,
    log_file: Arc<Mutex<fs::File>>,
    ok: Arc<AtomicUsize>,
    failed: Arc<AtomicUsize>,
    invalid: Arc<AtomicUsize>,
    degraded: Arc<AtomicUsize>,
    skipped: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    workers: usize,
}

impl CampaignReporter {
    fn new(
        out_dir: &Path,
        total_runs: usize,
        workers: usize,
        worker_resolution: impl AsRef<str>,
    ) -> Result<Self> {
        let log_path = out_dir.join("campaign.log");
        let log_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .into_diagnostic()
            .with_context(|| format!("open {}", log_path.display()))?;
        let multi = MultiProgress::new();
        let progress = multi.add(ProgressBar::new(total_runs as u64));
        progress.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("#>-"),
        );

        let worker_bars = if workers <= MAX_WORKER_PROGRESS_BARS {
            (0..workers)
                .map(|worker| {
                    let bar = multi.add(ProgressBar::new(100));
                    bar.set_style(
                        ProgressStyle::with_template("  {prefix:>3} {bar:30.cyan/blue} {msg}")
                            .unwrap_or_else(|_| ProgressStyle::default_bar())
                            .progress_chars("#>-"),
                    );
                    bar.set_prefix(format!("w{worker}"));
                    bar.set_message("idle");
                    bar
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let reporter = Self {
            multi,
            progress,
            worker_bars: Arc::new(worker_bars),
            slots: Arc::new((0..workers).map(|_| Mutex::new(None)).collect()),
            durations: Arc::new(Mutex::new(Vec::new())),
            log_file: Arc::new(Mutex::new(log_file)),
            ok: Arc::new(AtomicUsize::new(0)),
            failed: Arc::new(AtomicUsize::new(0)),
            invalid: Arc::new(AtomicUsize::new(0)),
            degraded: Arc::new(AtomicUsize::new(0)),
            skipped: Arc::new(AtomicUsize::new(0)),
            active: Arc::new(AtomicUsize::new(0)),
            workers,
        };
        reporter.progress.set_message(reporter.main_message());
        reporter.line(format!(
            "starting monte-carlo campaign: runs={total_runs} workers={workers} {} out_dir={}",
            worker_resolution.as_ref(),
            out_dir.display()
        ));
        Ok(reporter)
    }

    fn line(&self, line: impl AsRef<str>) {
        let line = line.as_ref();
        self.multi.suspend(|| eprintln!("{line}"));
        self.log_only(line);
    }

    fn log_only(&self, line: &str) {
        if let Ok(mut file) = self.log_file.lock() {
            let _ = writeln!(file, "{line}");
        }
    }

    fn main_message(&self) -> String {
        let degraded = self.degraded.load(Ordering::SeqCst);
        let degraded = if degraded > 0 {
            format!(" degraded={degraded}")
        } else {
            String::new()
        };
        format!(
            "ok={} fail={} invalid={}{degraded} skip={} active={}/{}",
            self.ok.load(Ordering::SeqCst),
            self.failed.load(Ordering::SeqCst),
            self.invalid.load(Ordering::SeqCst),
            self.skipped.load(Ordering::SeqCst),
            self.active.load(Ordering::SeqCst),
            self.workers,
        )
    }

    fn run_started(&self, worker_id: usize, run_id: &str) {
        if let Some(slot) = self.slots.get(worker_id)
            && let Ok(mut slot) = slot.lock()
        {
            *slot = Some(ActiveRun {
                run_id: run_id.to_string(),
                started: Instant::now(),
            });
            self.active.fetch_add(1, Ordering::SeqCst);
            self.progress.set_message(self.main_message());
        }
    }

    fn run_finished(&self, worker_id: usize, wall_ms: Option<u128>) {
        if let Some(slot) = self.slots.get(worker_id)
            && let Ok(mut slot) = slot.lock()
            && slot.take().is_some()
        {
            self.active.fetch_sub(1, Ordering::SeqCst);
        }
        if let Some(wall_ms) = wall_ms {
            self.durations
                .lock()
                .expect("duration mutex poisoned")
                .push(wall_ms);
        }
        self.progress.set_message(self.main_message());
    }

    fn record(&self, metric: &RunMetric) {
        if metric.status == "skipped" {
            self.skipped.fetch_add(1, Ordering::SeqCst);
        } else if !metric.valid() {
            self.invalid.fetch_add(1, Ordering::SeqCst);
        } else if metric.degraded {
            self.degraded.fetch_add(1, Ordering::SeqCst);
        } else if metric.passed() {
            self.ok.fetch_add(1, Ordering::SeqCst);
        } else {
            self.failed.fetch_add(1, Ordering::SeqCst);
        }
        let worker = worker_label(metric.worker_id);
        self.progress.inc(1);
        self.progress.set_message(self.main_message());
        let reason = metric
            .failure_reason
            .as_deref()
            .map(|reason| format!(" reason={reason:?}"))
            .unwrap_or_default();
        let line = format!(
            "[{}] {} worker={} wall_ms={}{reason} log={}",
            metric.status,
            metric.run_id,
            worker,
            metric.wall_ms,
            metric.run_dir.join("logs").display()
        );
        self.log_only(&line);
        if !metric.passed() {
            self.multi.suspend(|| eprintln!("{line}"));
        }
    }

    fn spawn_progress_ticker(&self, stop: Arc<AtomicBool>) -> tokio::task::JoinHandle<()> {
        let progress = self.progress.clone();
        let worker_bars = self.worker_bars.clone();
        let slots = self.slots.clone();
        let durations = self.durations.clone();
        let reporter = self.clone();
        tokio::spawn(async move {
            while !stop.load(Ordering::SeqCst) {
                reporter.refresh_progress(&progress, &worker_bars, &slots, &durations);
                tokio::time::sleep(Duration::from_millis(PROGRESS_TICK_MS)).await;
            }
            reporter.refresh_progress(&progress, &worker_bars, &slots, &durations);
        })
    }

    fn refresh_progress(
        &self,
        progress: &ProgressBar,
        worker_bars: &[ProgressBar],
        slots: &[Mutex<Option<ActiveRun>>],
        durations: &Mutex<Vec<u128>>,
    ) {
        progress.set_message(self.main_message());
        let estimate_ms = median_duration_ms(durations);
        for (worker, bar) in worker_bars.iter().enumerate() {
            let active = slots
                .get(worker)
                .and_then(|slot| slot.lock().ok().and_then(|slot| slot.clone()));
            if let Some(active) = active {
                let elapsed = active.started.elapsed();
                let elapsed_ms = elapsed.as_millis();
                if let Some(estimate_ms) = estimate_ms {
                    let percent = elapsed_ms
                        .saturating_mul(100)
                        .checked_div(estimate_ms)
                        .unwrap_or(0)
                        .min(99) as u64;
                    bar.set_position(percent);
                    bar.set_message(format!(
                        "~{percent}% {} ({}s)",
                        active.run_id,
                        elapsed.as_secs()
                    ));
                } else {
                    bar.set_position(0);
                    bar.set_message(format!("{} ({}s)", active.run_id, elapsed.as_secs()));
                }
            } else {
                bar.set_position(0);
                bar.set_message("idle");
            }
        }
    }

    fn finish(&self) {
        self.clear_progress();
        self.line(format!(
            "finished monte-carlo campaign: ok={} failed={} invalid={} degraded={} skipped={}",
            self.ok.load(Ordering::SeqCst),
            self.failed.load(Ordering::SeqCst),
            self.invalid.load(Ordering::SeqCst),
            self.degraded.load(Ordering::SeqCst),
            self.skipped.load(Ordering::SeqCst)
        ));
    }

    fn clear_progress(&self) {
        for bar in self.worker_bars.iter() {
            bar.finish_and_clear();
        }
        self.progress.finish_and_clear();
    }
}

fn median_duration_ms(durations: &Mutex<Vec<u128>>) -> Option<u128> {
    let mut values = durations.lock().expect("duration mutex poisoned").clone();
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[values.len() / 2])
}

fn log_campaign_line(out_dir: &Path, line: &str) -> Result<()> {
    eprintln!("{line}");
    let log_path = out_dir.join("campaign.log");
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .into_diagnostic()
        .with_context(|| format!("open {}", log_path.display()))?;
    writeln!(file, "{line}").into_diagnostic()
}

fn run_build_steps(builds: &[BuildConfig], out_dir: &Path) -> Result<()> {
    for build in builds {
        let Some(program) = build.command.as_ref().filter(|command| !command.is_empty()) else {
            continue;
        };
        let display_args = build.args.join(" ");
        log_campaign_line(
            out_dir,
            &format!("building campaign artifacts: {program} {display_args}"),
        )?;
        let mut command = Command::new(program);
        command.args(&build.args);
        if let Some(cwd) = &build.cwd {
            command.current_dir(cwd);
        }
        command.envs(&build.env);
        let status = command
            .status()
            .into_diagnostic()
            .with_context(|| format!("run campaign build step `{program} {display_args}`"))?;
        if !status.success() {
            return Err(miette!(
                "campaign build step failed with status {status}: {program} {display_args}"
            ));
        }
    }
    Ok(())
}

pub async fn run_campaign(mut options: RunOptions) -> Result<()> {
    let started_at = Utc::now();
    let mut config = load_config(options.campaign_path.as_deref())?;
    apply_overrides(&mut config, &mut options);

    let plan = read_plan(&options.plan_path)?;
    if plan.is_empty() {
        return Err(Error::EmptyPlan).into_diagnostic();
    }
    warn_stale_plan(&options.plan_path);

    if options.dry_run {
        let workers = match requested_workers(options.workers, &config) {
            Some(workers) => workers.to_string(),
            None => match s10::admission::max_inflight() {
                // The recipe is not planned during a dry run, so report the
                // admission budget itself.
                Some(max) => format!("(S10_MAX_INFLIGHT={max}/recipe_weight)"),
                None => "off".to_string(),
            },
        };
        println!(
            "monte-carlo dry run: runs={} workers={} out_dir={}",
            plan.len(),
            workers,
            options.out_dir.display()
        );
        for row in &plan {
            println!(
                "run_id={} seed={:?} params={} meta={}",
                row.run_id,
                row.seed,
                row.params.len(),
                row.meta.len()
            );
        }
        return Ok(());
    }

    fs::create_dir_all(&options.out_dir)
        .into_diagnostic()
        .with_context(|| format!("create campaign output {}", options.out_dir.display()))?;
    let cache_dir = resolve_cache_dir(&config, &options.out_dir);
    fs::create_dir_all(&cache_dir)
        .into_diagnostic()
        .with_context(|| format!("create cache dir {}", cache_dir.display()))?;
    let plan_dest = options.out_dir.join("plan.csv");
    if !same_file_or_path(&options.plan_path, &plan_dest) {
        fs::copy(&options.plan_path, &plan_dest).into_diagnostic()?;
    }
    if options.clean {
        prune_stale_run_dirs(&options.out_dir, &plan)?;
    }
    write_json(&options.out_dir.join("campaign.resolved.json"), &config)?;
    write_json(
        &options.out_dir.join("campaign.manifest.json"),
        &ResumeManifest {
            sim_path: options.sim_path.clone(),
            memory_probe: options.memory_probe,
            keep_existing: options.keep_existing,
        },
    )?;

    execute_campaign(ExecuteParams {
        workers_flag: options.workers,
        config,
        sim_path: options.sim_path,
        out_dir: options.out_dir,
        cache_dir,
        plan,
        preserved: Vec::new(),
        started_at,
        memory_probe: options.memory_probe,
        keep_existing: options.keep_existing,
        strict_ports: options.strict_ports,
    })
    .await
}

/// The default compile cache lives in the user cache home — outside the
/// campaign out dir — so `--clean` and fresh out dirs never trigger an
/// N-worker JIT compile storm. Entries are content-addressed, making
/// cross-campaign sharing safe by construction.
fn resolve_cache_dir(config: &CampaignConfig, out_dir: &Path) -> PathBuf {
    if let Some(cache_dir) = &config.cache_dir {
        return cache_dir.clone();
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("elodin/monte-carlo/const-cache");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        return PathBuf::from(home).join(".cache/elodin/monte-carlo/const-cache");
    }
    out_dir.join("const-cache")
}

/// Warn when the plan CSV predates a sibling `spec.toml`: the user probably
/// edited the spec and forgot to re-run `elodin monte-carlo sample`.
fn warn_stale_plan(plan_path: &Path) {
    let Some(spec_path) = plan_path.parent().map(|dir| dir.join("spec.toml")) else {
        return;
    };
    let (Ok(spec_meta), Ok(plan_meta)) = (fs::metadata(&spec_path), fs::metadata(plan_path)) else {
        return;
    };
    // Tolerance absorbs same-checkout mtimes (git writes both within seconds).
    if let (Ok(spec_mtime), Ok(plan_mtime)) = (spec_meta.modified(), plan_meta.modified())
        && spec_mtime
            .duration_since(plan_mtime)
            .is_ok_and(|delta| delta > Duration::from_secs(5))
    {
        eprintln!(
            "warning: {} is newer than {}; regenerate the plan with `elodin monte-carlo sample {} --output {}` (or pass --spec instead of --plan)",
            spec_path.display(),
            plan_path.display(),
            spec_path.display(),
            plan_path.display(),
        );
    }
}

/// The worker count explicitly requested by the user, honoring precedence:
/// `--workers` flag > `S10_MAX_INFLIGHT` env > `campaign.toml workers`.
/// `None` falls through to the admission budget (env or logical cores).
fn requested_workers(flag: Option<usize>, config: &CampaignConfig) -> Option<usize> {
    if let Some(workers) = flag {
        return Some(workers.max(1));
    }
    if std::env::var_os("S10_MAX_INFLIGHT").is_some() {
        return None;
    }
    config.workers.map(|workers| workers.max(1))
}

fn prune_stale_run_dirs(out_dir: &Path, plan: &[PlanRow]) -> Result<()> {
    let runs_dir = out_dir.join("runs");
    if !runs_dir.exists() {
        return Ok(());
    }
    let planned = plan
        .iter()
        .map(|row| row.run_id.as_str())
        .collect::<HashSet<_>>();
    for entry in fs::read_dir(&runs_dir).into_diagnostic()? {
        let entry = entry.into_diagnostic()?;
        if !entry.file_type().into_diagnostic()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !planned.contains(name.as_ref()) {
            fs::remove_dir_all(entry.path())
                .into_diagnostic()
                .with_context(|| format!("remove stale run dir {}", entry.path().display()))?;
        }
    }
    Ok(())
}

/// Persisted next to the campaign so `resume` can reconstruct the run inputs
/// that aren't already captured in `campaign.resolved.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ResumeManifest {
    sim_path: PathBuf,
    #[serde(default)]
    memory_probe: bool,
    /// Whether the original run kept pre-existing elodin / elodin-db processes
    /// alive (`--keep-existing`). Persisted so resume honors the same intent;
    /// defaults to `false` for campaigns created before this field existed.
    #[serde(default)]
    keep_existing: bool,
}

struct ExecuteParams {
    /// `--workers` from the CLI (resume passes `None`).
    workers_flag: Option<usize>,
    config: CampaignConfig,
    sim_path: PathBuf,
    out_dir: PathBuf,
    cache_dir: PathBuf,
    /// Rows still to run.
    plan: Vec<PlanRow>,
    /// Metrics for runs that already passed (resume); merged into the report.
    preserved: Vec<RunMetric>,
    started_at: DateTime<Utc>,
    memory_probe: bool,
    keep_existing: bool,
    strict_ports: bool,
}

/// Schedules `params.plan` across workers, merges in any `params.preserved`
/// metrics, and writes the full campaign report. Shared by `run_campaign` and
/// `resume_campaign`.
async fn execute_campaign(params: ExecuteParams) -> Result<()> {
    let ExecuteParams {
        workers_flag,
        config,
        sim_path,
        out_dir,
        cache_dir,
        plan,
        preserved,
        started_at,
        memory_probe,
        keep_existing,
        strict_ports,
    } = params;
    fs::create_dir_all(&out_dir)
        .into_diagnostic()
        .with_context(|| format!("create campaign output {}", out_dir.display()))?;
    let _campaign_lock = acquire_campaign_lock(&out_dir)?;
    let wall_start = Instant::now();
    let total_runs = preserved.len() + plan.len();

    run_build_steps(&config.build, &out_dir)?;
    // Only one `elodin monte-carlo` campaign is expected per host: campaigns share
    // the default DB port (2240) and the `elodin-mc-` cgroup namespace, so this
    // prefix reap is intentionally host-wide. It clears scopes (and their
    // reparented sidecars) left by a crashed or previous campaign before we start;
    // `--keep-existing` opts out when the operator manages those processes.
    if !keep_existing {
        s10::CgroupScope::reap_prefix("elodin-mc-").into_diagnostic()?;
    }
    let campaign_cgroup =
        s10::CgroupScope::create(format!("elodin-mc-{}", std::process::id())).into_diagnostic()?;
    // Prioritize the campaign's sims over background work under contention
    // (Linux cgroup cpu.weight; best-effort, no-op elsewhere).
    if s10::priority_enabled()
        && let Some(scope) = &campaign_cgroup
    {
        scope.set_cpu_weight(s10::sim_cpu_weight());
    }
    let base_recipe = plan_recipe(&sim_path, &out_dir).await?;
    let recipe_weight = s10::admission::recipe_weight(&base_recipe);
    let requested = requested_workers(workers_flag, &config);
    if let Some(workers) = requested {
        // Size the s10 admission budget so exactly `workers` runs execute at
        // once regardless of recipe shape.
        s10::admission::configure(Some(workers.saturating_mul(recipe_weight)));
    }
    let admission_max = s10::admission::max_inflight();
    let workers = resolve_auto_workers(total_runs, recipe_weight, admission_max);
    let worker_resolution = match (workers_flag, requested) {
        (Some(_), _) => "(--workers)".to_string(),
        (None, Some(_)) => "(campaign.toml workers)".to_string(),
        (None, None) => worker_resolution_label(recipe_weight, admission_max),
    };

    // Validate the entire static port plan before anything starts, so a bad
    // stride/base combination fails fast instead of as a mid-campaign worker
    // death several hundred runs in.
    let planned_ports = ports::validate_port_plan(&config.resources, workers)?;
    if let Some(warning) = ports::ephemeral_conflicts(&planned_ports) {
        if strict_ports {
            return Err(miette!("{warning} (--strict-ports)"));
        }
        log_campaign_line(&out_dir, &format!("warning: {warning}"))?;
    }
    if !keep_existing {
        let ports: HashSet<u16> = planned_ports.iter().map(|planned| planned.port).collect();
        for line in ports::reap_port_squatters(&ports)? {
            log_campaign_line(&out_dir, &line)?;
        }
    }
    raise_fd_limit(&out_dir, workers, recipe_weight)?;
    let scratch = resolve_scratch_dir(config.scratch_dir.as_deref(), &out_dir)?;
    if let Some(scratch) = &scratch {
        log_campaign_line(
            &out_dir,
            &format!(
                "running per-run IO on scratch {} (artifacts move to {} as runs finish)",
                scratch.display(),
                out_dir.display()
            ),
        )?;
    }

    let reporter = CampaignReporter::new(&out_dir, total_runs, workers, worker_resolution)?;
    // Count already-passed runs toward the report without re-running them.
    for metric in &preserved {
        reporter.record(metric);
    }

    let rows = Arc::new(Mutex::new(VecDeque::from(plan)));
    let metrics = Arc::new(Mutex::new(preserved));
    let failure = Arc::new(Mutex::new(None::<String>));
    let stop_sampling = Arc::new(AtomicBool::new(false));
    let peak_memory = Arc::new(Mutex::new(Vec::<CacheMappingSample>::new()));
    let resource_samples = Arc::new(Mutex::new(Vec::<ResourceSample>::new()));
    let process_samples = Arc::new(Mutex::new(Vec::<ProcessSample>::new()));
    let memory_sampler = memory_probe.then(|| {
        spawn_memory_sampler(
            cache_dir.clone(),
            stop_sampling.clone(),
            peak_memory.clone(),
        )
    });
    let resource_sampler = spawn_resource_sampler(
        out_dir.clone(),
        wall_start,
        stop_sampling.clone(),
        resource_samples.clone(),
        memory_probe.then_some(process_samples.clone()),
    );
    let progress_ticker = reporter.spawn_progress_ticker(stop_sampling.clone());
    let mut worker_tasks = JoinSet::new();

    let worker_count = workers;
    for worker_id in 0..workers {
        let rows = rows.clone();
        let metrics = metrics.clone();
        let failure = failure.clone();
        let base_recipe = base_recipe.clone();
        let config = config.clone();
        let out_dir = out_dir.clone();
        let cache_dir = cache_dir.clone();
        let scratch = scratch.clone();
        let reporter = reporter.clone();
        let campaign_cgroup = campaign_cgroup.clone();
        worker_tasks.spawn(async move {
            let template = ports::slot_template(worker_id, &config.resources)?;
            if let Some(delay) = rampup_delay(&config.rampup, worker_id, worker_count) {
                tokio::time::sleep(delay).await;
            }
            loop {
                if failure.lock().expect("failure mutex poisoned").is_some()
                    && !config.continue_on_error
                {
                    break;
                }
                let Some(row) = rows.lock().expect("rows mutex poisoned").pop_front() else {
                    break;
                };
                let run_id = row.run_id.clone();
                reporter.run_started(worker_id, &run_id);
                let metric = match run_with_retries(
                    row,
                    worker_id,
                    &template,
                    base_recipe.clone(),
                    &config,
                    &out_dir,
                    &cache_dir,
                    scratch.as_deref(),
                    worker_count,
                    campaign_cgroup.as_ref(),
                )
                .await
                {
                    Ok(metric) => metric,
                    Err(err) => {
                        let mut metric =
                            placeholder_metric(&run_id, Some(worker_id), &out_dir, "failed");
                        metric.failure_reason = Some(error_chain(&err));
                        reporter.run_finished(worker_id, None);
                        reporter.line(format!(
                            "[failed] {run_id} worker={worker_id} setup_error={err}"
                        ));
                        *failure.lock().expect("failure mutex poisoned") = Some(run_id);
                        reporter.record(&metric);
                        metrics.lock().expect("metrics mutex poisoned").push(metric);
                        continue;
                    }
                };
                // A clean process exit that fails post_run scoring is still a
                // campaign failure, so --fail-fast halts on scored misses too.
                let failed = !metric.passed();
                if failed {
                    *failure.lock().expect("failure mutex poisoned") = Some(metric.run_id.clone());
                }
                reporter.run_finished(worker_id, Some(metric.wall_ms));
                reporter.record(&metric);
                metrics.lock().expect("metrics mutex poisoned").push(metric);
            }
            Ok::<(), miette::Report>(())
        });
    }

    while let Some(result) = worker_tasks.join_next().await {
        result.into_diagnostic()??;
    }
    let skipped_rows = rows
        .lock()
        .expect("rows mutex poisoned")
        .drain(..)
        .collect::<Vec<_>>();
    for row in skipped_rows {
        let metric = placeholder_metric(&row.run_id, None, &out_dir, "skipped");
        reporter.record(&metric);
        metrics.lock().expect("metrics mutex poisoned").push(metric);
    }
    stop_sampling.store(true, Ordering::SeqCst);
    if let Some(memory_sampler) = memory_sampler {
        memory_sampler.await.into_diagnostic()?;
    }
    resource_sampler.await.into_diagnostic()?;
    progress_ticker.await.into_diagnostic()?;
    if let Some(scratch) = &scratch {
        // Run artifacts move out incrementally, leaving only empty skeleton
        // dirs behind — unless a finalize move failed, in which case the
        // scratch copy is that run's only data: preserve the tree.
        if dir_has_files(scratch) {
            log_campaign_line(
                &out_dir,
                &format!(
                    "warning: preserving scratch {} — it still holds artifacts of run(s) whose move to {} failed",
                    scratch.display(),
                    out_dir.display()
                ),
            )?;
        } else {
            let _ = fs::remove_dir_all(scratch);
        }
    }

    let metrics = metrics.lock().expect("metrics mutex poisoned").clone();
    write_perf_csv(&out_dir.join("perf.csv"), &metrics)?;
    let (timeline, concurrency_summary) = build_timeline(&metrics);
    write_timeline_csv(&out_dir.join("timeline.csv"), &timeline)?;
    write_results_csv(&out_dir.join("results.csv"), &out_dir, &metrics)?;
    let memory = if memory_probe {
        peak_memory.lock().expect("memory mutex poisoned").clone()
    } else {
        Vec::new()
    };
    let resource_samples = resource_samples
        .lock()
        .expect("resource samples mutex poisoned")
        .clone();
    write_resource_csv(&out_dir.join("resources.csv"), &resource_samples)?;
    if memory_probe {
        write_json(&out_dir.join("memory.json"), &memory)?;
        write_process_csv(
            &out_dir.join("processes.csv"),
            &process_samples
                .lock()
                .expect("process samples mutex poisoned")
                .clone(),
        )?;
    }

    let summary = summarize_campaign(
        &out_dir,
        &metrics,
        &resource_samples,
        started_at,
        Utc::now(),
        wall_start.elapsed().as_millis(),
        workers,
        &concurrency_summary,
    );
    write_json(&out_dir.join("summary.json"), &summary)?;
    reporter.clear_progress();
    write_campaign_summary(
        &out_dir,
        &summary,
        summary.sim_phase_summary.as_ref(),
        &memory,
    )?;
    reporter.finish();

    if let Some(hook) = &config.hooks.post_campaign {
        let context = out_dir.join("campaign_hook_context.json");
        write_json(
            &context,
            &json!({
                "out_dir": out_dir,
                "results": out_dir.join("results.csv"),
                "perf": out_dir.join("perf.csv"),
                "memory": memory_probe.then(|| out_dir.join("memory.json")),
                "resources": out_dir.join("resources.csv"),
                "summary": out_dir.join("summary.json"),
            }),
        )?;
        run_hook("post_campaign", hook, &context).await?;
    }

    if config.fail_on_run_errors {
        let mut error_runs = summary.error_runs();
        if config.quality.fail_on_degraded {
            error_runs += summary.degraded;
        }
        if error_runs > 0 {
            return Err(miette!(
                "monte-carlo campaign finished with {} failed, {} invalid, and {} degraded run(s) (fail_on_run_errors=true)",
                summary.failed,
                summary.invalid,
                summary.degraded
            ));
        }
    }

    Ok(())
}

/// Exclusive lock on a campaign out dir, held for the campaign's lifetime.
/// A second campaign pointed at the same `--out` fails fast with the holder's
/// pid instead of silently interleaving with it (dueling port plans, run-dir
/// races). The kernel releases the lock when the holder dies, however it
/// died, so reruns after a kill never block.
#[cfg(unix)]
struct CampaignLock {
    _lock: nix::fcntl::Flock<fs::File>,
}

#[cfg(not(unix))]
struct CampaignLock {}

#[cfg(unix)]
fn acquire_campaign_lock(out_dir: &Path) -> Result<CampaignLock> {
    let path = out_dir.join("campaign.lock");
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .into_diagnostic()
        .with_context(|| format!("open {}", path.display()))?;
    match nix::fcntl::Flock::lock(file, nix::fcntl::FlockArg::LockExclusiveNonblock) {
        Ok(lock) => {
            use std::io::Write;
            let mut lock = lock;
            let _ = lock.set_len(0);
            let _ = write!(&mut *lock, "{}", std::process::id());
            let _ = lock.flush();
            Ok(CampaignLock { _lock: lock })
        }
        Err((_, _)) => {
            let holder = fs::read_to_string(&path).unwrap_or_default();
            let holder = holder.trim();
            let holder = if holder.is_empty() {
                "unknown pid".to_string()
            } else {
                format!("pid {holder}")
            };
            Err(miette!(
                "campaign out dir {} is in use by another `elodin monte-carlo` run ({holder}); \
                 kill it or choose a different --out",
                out_dir.display()
            ))
        }
    }
}

#[cfg(not(unix))]
fn acquire_campaign_lock(_out_dir: &Path) -> Result<CampaignLock> {
    Ok(CampaignLock {})
}

/// Raise the soft fd limit to the hard limit: the runner pipes stdout/stderr
/// for every child (`workers x recipe_weight` processes), and the stock soft
/// limit of 1024 starves it below ~48 workers.
#[cfg(unix)]
fn raise_fd_limit(out_dir: &Path, workers: usize, recipe_weight: usize) -> Result<()> {
    use nix::sys::resource::{Resource, getrlimit, setrlimit};
    let Ok((soft, hard)) = getrlimit(Resource::RLIMIT_NOFILE) else {
        return Ok(());
    };
    if soft < hard {
        let _ = setrlimit(Resource::RLIMIT_NOFILE, hard, hard);
    }
    let needed = (workers as u64)
        .saturating_mul(recipe_weight as u64)
        .saturating_mul(4);
    if hard < needed {
        log_campaign_line(
            out_dir,
            &format!(
                "warning: fd hard limit {hard} may be too low for {workers} workers \
                 (~{needed} fds needed); raise it with `ulimit -Hn` or systemd LimitNOFILE"
            ),
        )?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn raise_fd_limit(_out_dir: &Path, _workers: usize, _recipe_weight: usize) -> Result<()> {
    Ok(())
}

/// First-wave stagger: without it, every worker starts a cold interpreter +
/// JIT compile at t=0 and the earliest runs' real-time-paced loops execute
/// under peak transient load, biasing wave-1 pacing (and occasionally
/// physics) relative to steady state. 500 ms per worker, capped at one
/// minute, halves the concurrent-startup peak for ~1 run of added wall time.
/// Small pools skip it entirely; steady state staggers itself via slot
/// turnover.
fn rampup_delay(rampup: &str, worker_id: usize, workers: usize) -> Option<Duration> {
    if rampup == "off" || workers < 8 || worker_id == 0 {
        return None;
    }
    let delay = Duration::from_millis(500).saturating_mul(worker_id as u32);
    Some(delay.min(Duration::from_secs(60)))
}

/// Resolve the scratch base for per-run IO. `"auto"` picks `/dev/shm` when it
/// exists and is writable; any other value is used as the scratch root; unset
/// or `"off"` disables scratch. The campaign gets a unique subdirectory.
fn resolve_scratch_dir(scratch: Option<&str>, out_dir: &Path) -> Result<Option<PathBuf>> {
    let scratch = match scratch {
        None => return Ok(None),
        Some(value) if value.is_empty() || value == "off" => return Ok(None),
        Some("auto") => {
            let shm = Path::new("/dev/shm");
            if !shm.is_dir() {
                return Ok(None);
            }
            shm.to_path_buf()
        }
        Some(path) => PathBuf::from(path),
    };
    let campaign_name = out_dir
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "campaign".to_string());
    let dir = scratch.join(format!("elodin-mc-{campaign_name}-{}", std::process::id()));
    fs::create_dir_all(&dir)
        .into_diagnostic()
        .with_context(|| format!("create scratch dir {}", dir.display()))?;
    Ok(Some(dir))
}

/// Whether `dir` recursively contains anything other than empty directories.
/// Decides if a scratch tree still holds run artifacts worth preserving.
fn dir_has_files(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        match entry.file_type() {
            Ok(file_type) if file_type.is_dir() => {
                if dir_has_files(&entry.path()) {
                    return true;
                }
            }
            // Files, symlinks, and unreadable entries all count as content.
            _ => return true,
        }
    }
    false
}

/// Join an error's chain into one line for `failure_reason`.
fn error_chain(err: &miette::Report) -> String {
    let mut parts = Vec::new();
    for cause in err.chain() {
        let text = cause.to_string();
        if !text.is_empty() && parts.last() != Some(&text) {
            parts.push(text);
        }
    }
    parts.join(": ")
}

/// Resume a previously started campaign: keep the runs that already passed,
/// re-run everything else, then rewrite the merged report.
pub async fn resume_campaign(campaign_dir: PathBuf) -> Result<()> {
    let out_dir = campaign_dir;
    if !out_dir.exists() {
        return Err(miette!(
            "campaign directory {} does not exist",
            out_dir.display()
        ));
    }
    let manifest: ResumeManifest = read_json(&out_dir.join("campaign.manifest.json"))
        .with_context(|| {
            format!(
                "{} has no campaign.manifest.json; only campaigns created by `monte-carlo run` can be resumed",
                out_dir.display()
            )
        })?;
    let config: CampaignConfig = read_json(&out_dir.join("campaign.resolved.json"))
        .with_context(|| format!("read resolved config in {}", out_dir.display()))?;
    let plan = read_plan(&out_dir.join("plan.csv"))?;
    if plan.is_empty() {
        return Err(Error::EmptyPlan).into_diagnostic();
    }

    let mut preserved = Vec::new();
    let mut pending = Vec::new();
    for row in plan {
        match load_existing_metric(&out_dir, &row.run_id) {
            Some(metric) if metric.passed() => preserved.push(metric),
            _ => pending.push(row),
        }
    }

    if pending.is_empty() {
        eprintln!(
            "monte-carlo resume: all {} run(s) already passed; rebuilding report",
            preserved.len()
        );
        return rebuild_report(&out_dir);
    }
    eprintln!(
        "monte-carlo resume: {} done / {} pending in {}",
        preserved.len(),
        pending.len(),
        out_dir.display()
    );

    let cache_dir = resolve_cache_dir(&config, &out_dir);
    fs::create_dir_all(&cache_dir)
        .into_diagnostic()
        .with_context(|| format!("create cache dir {}", cache_dir.display()))?;

    execute_campaign(ExecuteParams {
        workers_flag: None,
        config,
        sim_path: manifest.sim_path,
        out_dir,
        cache_dir,
        plan: pending,
        preserved,
        started_at: Utc::now(),
        memory_probe: manifest.memory_probe,
        keep_existing: manifest.keep_existing,
        strict_ports: false,
    })
    .await
}

/// Rebuild the human-readable report and `summary.json` from the per-run
/// metrics already written to a campaign directory.
pub fn rebuild_report(campaign_dir: &Path) -> Result<()> {
    let out_dir = campaign_dir;
    if !out_dir.exists() {
        return Err(miette!(
            "campaign directory {} does not exist",
            out_dir.display()
        ));
    }
    let mut metrics = read_perf_metrics(&out_dir.join("perf.csv"))?;
    if metrics.is_empty() {
        return Err(miette!(
            "no runs recorded in {}",
            out_dir.join("perf.csv").display()
        ));
    }
    for metric in &mut metrics {
        // perf.csv is a flat schema and cannot represent the dynamic
        // hook_scalars map, so promoted columns / hook_metrics would be lost on
        // rebuild. Prefer the per-run metrics.json (the full RunMetric written
        // after scoring) when present; otherwise keep the perf.csv row and fix
        // up its paths.
        if let Some(stored) = load_existing_metric(out_dir, &metric.run_id) {
            *metric = stored;
        } else {
            let run_dir = out_dir.join("runs").join(&metric.run_id);
            metric.db_path = run_dir.join("db");
            metric.run_dir = run_dir;
        }
    }

    let started_at = metrics
        .iter()
        .map(|metric| metric.started_at)
        .min()
        .unwrap_or_else(Utc::now);
    let finished_at = metrics
        .iter()
        .map(|metric| metric.finished_at)
        .max()
        .unwrap_or_else(Utc::now);
    let wall_ms = (finished_at - started_at).num_milliseconds().max(0) as u128;
    let workers = read_json::<CampaignSummary>(&out_dir.join("summary.json"))
        .ok()
        .map(|summary| summary.workers)
        .filter(|workers| *workers > 0)
        .unwrap_or_else(|| {
            metrics
                .iter()
                .filter_map(|metric| metric.worker_id)
                .max()
                .map(|id| id + 1)
                .unwrap_or(1)
        });

    let resource_samples = read_resource_samples(&out_dir.join("resources.csv"));
    let memory =
        read_json::<Vec<CacheMappingSample>>(&out_dir.join("memory.json")).unwrap_or_default();
    let (timeline, concurrency_summary) = build_timeline(&metrics);
    write_timeline_csv(&out_dir.join("timeline.csv"), &timeline)?;
    write_results_csv(&out_dir.join("results.csv"), out_dir, &metrics)?;

    let summary = summarize_campaign(
        out_dir,
        &metrics,
        &resource_samples,
        started_at,
        finished_at,
        wall_ms,
        workers,
        &concurrency_summary,
    );
    write_json(&out_dir.join("summary.json"), &summary)?;
    write_campaign_summary(
        out_dir,
        &summary,
        summary.sim_phase_summary.as_ref(),
        &memory,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn summarize_campaign(
    out_dir: &Path,
    metrics: &[RunMetric],
    resource_samples: &[ResourceSample],
    started_at: DateTime<Utc>,
    finished_at: DateTime<Utc>,
    wall_ms: u128,
    workers: usize,
    concurrency_summary: &ConcurrencySummary,
) -> CampaignSummary {
    let passed = metrics.iter().filter(|metric| metric.passed()).count();
    let skipped = metrics
        .iter()
        .filter(|metric| metric.status == "skipped")
        .count();
    let invalid = metrics
        .iter()
        .filter(|metric| metric.status != "skipped" && !metric.valid())
        .count();
    let degraded = metrics
        .iter()
        .filter(|metric| metric.status != "skipped" && metric.valid() && metric.degraded)
        .count();
    // Invalid runs are infrastructure/no-data samples, so they are reported
    // separately and excluded from the scored pass/fail distribution.
    let failed = metrics
        .iter()
        .filter(|metric| {
            metric.status != "skipped" && metric.valid() && !metric.degraded && !metric.passed()
        })
        .count();
    let sim_phase_summary = aggregate_sim_summaries(metrics);
    let phase_attribution = summarize_phase_attribution(metrics);
    let total_run_wall_ms = metrics.iter().map(|metric| metric.wall_ms).sum::<u128>();
    let max_run_wall_ms = metrics
        .iter()
        .map(|metric| metric.wall_ms)
        .max()
        .unwrap_or_default();
    let disk_bytes = dir_size(out_dir);
    let mut resource_summary = summarize_resources(resource_samples);
    resource_summary.peak_campaign_disk_bytes = disk_bytes;
    CampaignSummary {
        started_at,
        finished_at,
        total_runs: metrics.len(),
        passed,
        failed,
        invalid,
        degraded,
        skipped,
        workers,
        wall_ms,
        total_run_wall_ms,
        average_run_wall_ms: if metrics.is_empty() {
            0.0
        } else {
            total_run_wall_ms as f64 / metrics.len() as f64
        },
        max_run_wall_ms,
        parallel_efficiency: if wall_ms == 0 || workers == 0 {
            0.0
        } else {
            total_run_wall_ms as f64 / (wall_ms as f64 * workers as f64)
        },
        disk_bytes,
        resource_summary,
        sim_phase_summary,
        phase_attribution,
        concurrency_summary: concurrency_summary.clone(),
        hook_metrics: summarize_hook_metrics(metrics),
        pacing: summarize_pacing(metrics),
    }
}

fn summarize_pacing(metrics: &[RunMetric]) -> Option<PacingSummary> {
    let mut pacing = PacingSummary::default();
    for metric in metrics {
        let mut paced = false;
        if let Some(frac) = metric.behind_deadline_frac {
            paced = true;
            if frac >= pacing.max_behind_deadline_frac {
                pacing.max_behind_deadline_frac = frac;
                pacing.max_behind_run = metric.run_id.clone();
            }
        }
        if let Some(rtf) = metric.real_time_factor {
            paced = true;
            if rtf >= pacing.max_real_time_factor {
                pacing.max_real_time_factor = rtf;
                pacing.max_real_time_factor_run = metric.run_id.clone();
            }
        }
        if paced {
            pacing.paced_runs += 1;
        }
    }
    (pacing.paced_runs > 0).then_some(pacing)
}

fn summarize_hook_metrics(metrics: &[RunMetric]) -> BTreeMap<String, ScalarMetricSummary> {
    let mut grouped: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for metric in metrics.iter().filter(|metric| metric.valid()) {
        for (key, value) in &metric.hook_scalars {
            if RESERVED_HOOK_KEYS.contains(&key.as_str()) {
                continue;
            }
            if let Ok(value) = value.parse::<f64>()
                && value.is_finite()
            {
                grouped.entry(key.clone()).or_default().push(value);
            }
        }
    }
    grouped
        .into_iter()
        .filter_map(|(key, mut values)| {
            values.sort_by(f64::total_cmp);
            let count = values.len();
            if count == 0 {
                return None;
            }
            let sum = values.iter().sum::<f64>();
            let p95_idx = ((count as f64 * 0.95).ceil() as usize)
                .saturating_sub(1)
                .min(count - 1);
            Some((
                key,
                ScalarMetricSummary {
                    count,
                    min: values[0],
                    mean: sum / count as f64,
                    p95: values[p95_idx],
                    max: values[count - 1],
                },
            ))
        })
        .collect()
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path)
        .into_diagnostic()
        .with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text)
        .into_diagnostic()
        .with_context(|| format!("parse {}", path.display()))
}

fn read_perf_metrics(path: &Path) -> Result<Vec<RunMetric>> {
    let mut reader = csv::Reader::from_path(path)
        .into_diagnostic()
        .with_context(|| format!("read {}", path.display()))?;
    let mut metrics = Vec::new();
    for record in reader.deserialize::<RunMetric>() {
        metrics.push(
            record
                .into_diagnostic()
                .with_context(|| format!("parse row in {}", path.display()))?,
        );
    }
    Ok(metrics)
}

fn read_resource_samples(path: &Path) -> Vec<ResourceSample> {
    let Ok(mut reader) = csv::Reader::from_path(path) else {
        return Vec::new();
    };
    reader
        .deserialize::<ResourceSample>()
        .filter_map(Result::ok)
        .collect()
}

fn load_existing_metric(out_dir: &Path, run_id: &str) -> Option<RunMetric> {
    let run_dir = out_dir.join("runs").join(run_id);
    let text = fs::read_to_string(run_dir.join("metrics.json")).ok()?;
    let mut metric: RunMetric = serde_json::from_str(&text).ok()?;
    metric.db_path = run_dir.join("db");
    metric.run_dir = run_dir;
    Some(metric)
}

#[allow(clippy::too_many_arguments)]
async fn run_with_retries(
    row: PlanRow,
    worker_id: usize,
    template: &ports::SlotTemplate,
    base_recipe: s10::Recipe,
    config: &CampaignConfig,
    out_dir: &Path,
    cache_dir: &Path,
    scratch: Option<&Path>,
    worker_count: usize,
    campaign_cgroup: Option<&Arc<s10::CgroupScope>>,
) -> Result<RunMetric> {
    let mut last_metric = None;
    let ctx = RunContext {
        base_recipe,
        config,
        out_dir,
        cache_dir,
        scratch,
        worker_count,
        campaign_cgroup,
    };
    for attempt in 0..=config.retries {
        let metric = run_one(&row, worker_id, template, &ctx, attempt).await?;
        let ok = metric.exit_ok;
        last_metric = Some(metric);
        if ok {
            break;
        }
    }
    Ok(last_metric.expect("attempt loop always runs at least once"))
}

struct RunContext<'a> {
    base_recipe: s10::Recipe,
    config: &'a CampaignConfig,
    out_dir: &'a Path,
    cache_dir: &'a Path,
    /// Campaign scratch base; run dirs live here during execution and move to
    /// `out_dir` when the run finalizes.
    scratch: Option<&'a Path>,
    worker_count: usize,
    campaign_cgroup: Option<&'a Arc<s10::CgroupScope>>,
}

/// Resolve a worker's slot for one run: static ports come from the template,
/// `"auto"` ports are freshly allocated and guarded until the recipe spawns.
/// An auto DB port is allocated as a consecutive pair with its `db_port + 1`
/// assets port (a static DB port already carries `db_assets` in the template).
fn resolve_run_slot(
    template: &ports::SlotTemplate,
    run_id: &str,
) -> Result<(ResourceSlot, Vec<ports::PortGuard>)> {
    let mut guards = Vec::new();
    let mut ports = BTreeMap::new();
    let db_port = match template.db_port {
        Some(port) => port,
        None => {
            let (db_guard, assets_guard) = ports::allocate_port_pair(template.bind_ip)
                .with_context(|| format!("allocate dynamic db/assets ports for {run_id}"))?;
            let port = db_guard.port;
            ports.insert("db_assets".to_string(), assets_guard.port);
            guards.push(db_guard);
            guards.push(assets_guard);
            port
        }
    };
    let mut allocate = || -> Result<u16> {
        let guard = ports::allocate_port(template.bind_ip)
            .with_context(|| format!("allocate dynamic port for {run_id}"))?;
        let port = guard.port;
        guards.push(guard);
        Ok(port)
    };
    for (name, port) in &template.ports {
        let port = match port {
            Some(port) => *port,
            None => allocate()?,
        };
        ports.insert(name.clone(), port);
    }
    Ok((
        ResourceSlot {
            worker_id: template.worker_id,
            db_addr: SocketAddr::new(template.bind_ip, db_port),
            db_port,
            ports,
        },
        guards,
    ))
}

/// Fail fast (with pid attribution) when another process is squatting on one
/// of this run's static ports — otherwise the failure surfaces minutes later
/// as an opaque readiness timeout or, worse, corrupted handshakes. Brief
/// re-probes absorb the teardown latency of the previous run on this slot.
async fn preflight_static_ports(template: &ports::SlotTemplate, slot: &ResourceSlot) -> Result<()> {
    let static_ports: HashSet<u16> = template
        .db_port
        .iter()
        .copied()
        .chain(
            template
                .ports
                .values()
                .filter_map(|port| port.as_ref().copied()),
        )
        .collect();
    if static_ports.is_empty() {
        return Ok(());
    }
    let mut owners = Vec::new();
    for attempt in 0..4 {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        owners = ports::find_port_owners(&static_ports);
        if owners.is_empty() {
            return Ok(());
        }
    }
    let owner = &owners[0];
    let name = slot
        .ports
        .iter()
        .find(|(_, port)| **port == owner.port)
        .map(|(name, _)| name.as_str())
        .unwrap_or(if slot.db_port == owner.port {
            "db_port"
        } else {
            "port"
        });
    Err(miette!("preflight: `{name}` {owner}"))
}

async fn run_one(
    row: &PlanRow,
    worker_id: usize,
    template: &ports::SlotTemplate,
    ctx: &RunContext<'_>,
    attempt: usize,
) -> Result<RunMetric> {
    // The run executes in `active_run_dir` (scratch when configured) and its
    // surviving artifacts move to `final_run_dir` when it completes.
    let final_run_dir = ctx.out_dir.join("runs").join(&row.run_id);
    let active_run_dir = match ctx.scratch {
        Some(scratch) => scratch.join("runs").join(&row.run_id),
        None => final_run_dir.clone(),
    };
    let run_dir = active_run_dir.clone();
    let db_path = run_dir.join("db");
    for dir in [&final_run_dir, &run_dir] {
        if dir.exists() {
            fs::remove_dir_all(dir)
                .into_diagnostic()
                .with_context(|| format!("remove stale run dir {}", dir.display()))?;
        }
    }
    fs::create_dir_all(&run_dir).into_diagnostic()?;

    let (slot, port_guards) = resolve_run_slot(template, &row.run_id)?;
    let mut preflight_failure = preflight_static_ports(template, &slot).await.err();

    let context_path = run_dir.join("context.json");
    write_run_context(
        &context_path,
        row,
        &slot,
        &db_path,
        &run_dir,
        ctx.cache_dir,
        ctx.config,
    )?;

    let recipe = patch_recipe(
        ctx.base_recipe.clone(),
        row,
        &PatchContext {
            slot: &slot,
            context_path: &context_path,
            cache_dir: ctx.cache_dir,
            db_path: &db_path,
            run_dir: &run_dir,
            config: ctx.config,
            worker_count: ctx.worker_count,
            recipe_weight: s10::admission::recipe_weight(&ctx.base_recipe),
        },
    )?;
    let admission_permit = if preflight_failure.is_none() {
        s10::admission::acquire_run_slot(s10::admission::recipe_weight(&recipe)).await
    } else {
        None
    };

    let started_at = Utc::now();
    let spawn_unix_ns = unix_now_ns();
    let start = Instant::now();
    let token = CancelToken::new();
    let run_cgroup = ctx.campaign_cgroup.and_then(|scope| {
        s10::CgroupScope::create_child(scope, &row.run_id)
            .ok()
            .flatten()
    });
    // Release dynamic-port guards only now, just before the recipe spawns.
    drop(port_guards);
    let preflight_failed = preflight_failure.is_some();
    let mut failure_reason: Option<String> = None;
    let result = if let Some(reason) = preflight_failure.take() {
        Err(reason)
    } else {
        let fut = s10::cli::run_recipe_with_token_admitted_in_cgroup(
            row.run_id.clone(),
            recipe,
            false,
            false,
            token.clone(),
            admission_permit,
            run_cgroup.clone(),
        );
        if let Some(timeout) = parse_duration(ctx.config.timeout.as_deref())? {
            let mut fut = Box::pin(fut);
            match tokio::time::timeout(timeout, &mut fut).await {
                Ok(result) => result,
                Err(_) => {
                    token.cancel();
                    // Let the recipes' cancel branches run their graceful
                    // SIGTERM -> SIGKILL process-group teardown (the only
                    // kill path when no cgroup is available) before dropping
                    // the future.
                    let _ = tokio::time::timeout(Duration::from_secs(10), &mut fut).await;
                    if let Some(scope) = &run_cgroup {
                        let _ = scope.kill();
                    }
                    failure_reason = Some(format!("timed out after {timeout:?}"));
                    Err(miette!("run {} timed out after {:?}", row.run_id, timeout))
                }
            }
        } else {
            fut.await
        }
    };
    if let Some(scope) = &run_cgroup {
        let _ = scope.kill();
        let _ = scope.remove();
    }
    let exit_unix_ns = unix_now_ns();
    let exit_ok = result.is_ok();
    if failure_reason.is_none()
        && let Err(err) = &result
    {
        failure_reason = Some(error_chain(err));
    }
    let sim_summary = read_sim_run_summary(&run_dir.join("sim_summary.json"));
    let entry_unix_ns = sim_summary
        .as_ref()
        .and_then(|summary| summary.entry_unix_ns);
    let compile_done_unix_ns = sim_summary
        .as_ref()
        .and_then(|summary| summary.compile_done_unix_ns);
    let loop_start_unix_ns = sim_summary
        .as_ref()
        .and_then(|summary| summary.loop_start_unix_ns);
    let loop_end_unix_ns = sim_summary
        .as_ref()
        .and_then(|summary| summary.loop_end_unix_ns);
    let summary_written_unix_ns = sim_summary
        .as_ref()
        .and_then(|summary| summary.summary_written_unix_ns);
    let pacing = sim_summary.as_ref().map(extract_pacing).unwrap_or_default();
    let mut metric = RunMetric {
        run_id: row.run_id.clone(),
        worker_id: Some(worker_id),
        attempt,
        status: if exit_ok { "ok" } else { "failed" }.to_string(),
        exit_ok,
        scored_pass: None,
        scored_valid: None,
        failure_reason,
        degraded: false,
        behind_deadline_frac: pacing.behind_deadline_frac,
        real_time_factor: pacing.real_time_factor,
        drift_resets: pacing.drift_resets,
        hook_scalars: BTreeMap::new(),
        wall_ms: start.elapsed().as_millis(),
        started_at,
        finished_at: Utc::now(),
        spawn_unix_ns,
        exit_unix_ns,
        entry_unix_ns,
        compile_done_unix_ns,
        loop_start_unix_ns,
        loop_end_unix_ns,
        summary_written_unix_ns,
        python_import_ms: diff_ms(Some(spawn_unix_ns), entry_unix_ns),
        compile_ms: diff_ms(entry_unix_ns, compile_done_unix_ns),
        loop_ms: diff_ms(loop_start_unix_ns, loop_end_unix_ns),
        teardown_ms: diff_ms(loop_end_unix_ns, Some(exit_unix_ns)),
        process_shutdown_ms: diff_ms(summary_written_unix_ns, Some(exit_unix_ns)),
        db_path,
        run_dir: run_dir.clone(),
    };
    // Setup-only failures (port preflight) never spawned the recipe, so there
    // is no run database or telemetry to score: skip the hook instead of
    // having it error (or emit misleading scores) against a missing DB.
    if let Some(hook) = &ctx.config.hooks.post_run
        && !preflight_failed
    {
        let hook_ctx = run_dir.join("post_run_context.json");
        write_json(
            &hook_ctx,
            &json!({
                "run_id": row.run_id,
                "params": row.params,
                "meta": row.meta,
                "metrics": metric,
                "db_path": metric.db_path,
                "run_dir": metric.run_dir,
            }),
        )?;
        run_hook("post_run", hook, &hook_ctx).await?;
        // The hook scores the run via `post_run_result.json`; fold its verdict
        // into the metric so summaries don't count a criteria-failing run as
        // passed just because the process exited cleanly.
        let outcome = read_post_run_outcome(&run_dir.join("post_run_result.json"));
        metric.scored_pass = outcome.pass;
        metric.scored_valid = outcome.valid;
        metric.hook_scalars = outcome.scalars;
        if !metric.valid() {
            metric.status = "invalid".to_string();
        }
    }
    // Quality gate: a clean, valid run whose pacing degraded answers a
    // different statistical question — flag it instead of counting a pass.
    if metric.exit_ok
        && metric.valid()
        && let Some(reason) = quality_violation(&ctx.config.quality, &metric)
    {
        metric.degraded = true;
        metric.status = "degraded".to_string();
        metric.failure_reason.get_or_insert(reason);
    }
    // Truncate the (now finalized) DB's preallocated files before retention
    // decides to keep it, so kept DBs have apparent size ~= real size.
    let keep_db = match ctx.config.retention.keep_run_db {
        RunDbRetention::Always => true,
        RunDbRetention::Never => false,
        RunDbRetention::OnFail => !metric.passed(),
    };
    if keep_db && ctx.config.retention.compact_run_db {
        let _ = fs_util::compact_run_db(&metric.db_path);
    }
    apply_retention(ctx.config.retention.keep_run_db, &metric);
    apply_prune_globs(&ctx.config.retention, &metric);
    write_json(&run_dir.join("metrics.json"), &metric)?;
    // Scratch finalize: move the surviving artifacts to the real out dir,
    // sparse-aware, and point the metric at their final home.
    if run_dir != final_run_dir {
        match fs_util::move_dir_sparse(&run_dir, &final_run_dir) {
            Ok(()) => {
                metric.db_path = final_run_dir.join("db");
                metric.run_dir = final_run_dir.clone();
                write_json(&final_run_dir.join("metrics.json"), &metric)?;
            }
            Err(err) => {
                // The scratch copy is the only complete copy of the run.
                // Don't error the worker (that would record a placeholder and
                // let campaign teardown delete scratch): mark the run failed,
                // keep the metric pointed at the surviving scratch artifacts
                // (teardown preserves non-empty scratch trees), and drop any
                // partial destination so a later resume cannot mistake it for
                // a completed run.
                if run_dir.join("metrics.json").exists() {
                    let _ = fs::remove_dir_all(&final_run_dir);
                }
                let reason = format!(
                    "finalize: move run artifacts to {} failed: {err}; artifacts preserved on scratch {}",
                    final_run_dir.display(),
                    run_dir.display()
                );
                metric.status = "failed".to_string();
                metric.exit_ok = false;
                metric.failure_reason = Some(match metric.failure_reason.take() {
                    Some(existing) => format!("{existing}; {reason}"),
                    None => reason,
                });
                write_json(&run_dir.join("metrics.json"), &metric)?;
            }
        }
    }
    Ok(metric)
}

#[derive(Default)]
struct PacingMetrics {
    behind_deadline_frac: Option<f64>,
    real_time_factor: Option<f64>,
    drift_resets: Option<u64>,
}

/// Derive pacing-integrity metrics from a sim summary. Only meaningful for
/// real-time-paced sims (`real_time_pacing` observed at least one cycle).
fn extract_pacing(summary: &SimRunSummary) -> PacingMetrics {
    let mut pacing = PacingMetrics::default();
    if summary.real_time_pacing.count == 0 {
        return pacing;
    }
    if summary.cycles > 0 {
        pacing.behind_deadline_frac =
            Some(summary.pacing_behind_events as f64 / summary.cycles as f64);
    }
    pacing.drift_resets = Some(summary.pacing_resets);
    // Wall time of the sim loop vs the simulated time it covered.
    if summary.steps > 0 && summary.simulation_rate_hz > 0.0 && summary.wall_ns > 0 {
        let sim_seconds = summary.steps as f64 / summary.simulation_rate_hz;
        if sim_seconds > 0.0 {
            pacing.real_time_factor = Some((summary.wall_ns as f64 / 1e9) / sim_seconds);
        }
    }
    pacing
}

fn quality_violation(quality: &QualityConfig, metric: &RunMetric) -> Option<String> {
    if let (Some(max), Some(frac)) = (
        quality.max_behind_deadline_frac,
        metric.behind_deadline_frac,
    ) && frac > max
    {
        return Some(format!(
            "pacing degraded: {:.1}% of cycles behind deadline (gate {:.1}%)",
            frac * 100.0,
            max * 100.0
        ));
    }
    if let (Some(max), Some(rtf)) = (quality.max_real_time_factor, metric.real_time_factor)
        && rtf > max
    {
        return Some(format!(
            "pacing degraded: real-time factor {rtf:.3} (gate {max:.3})"
        ));
    }
    None
}

/// Remove retention-glob matches (relative to the run dir) after scoring, so
/// hooks don't each carry bespoke cleanup code.
fn apply_prune_globs(retention: &RetentionConfig, metric: &RunMetric) {
    let patterns = if metric.passed() {
        &retention.prune_on_pass
    } else {
        &retention.prune_on_fail
    };
    let Ok(run_dir) = fs::canonicalize(&metric.run_dir) else {
        return;
    };
    for pattern in patterns {
        if Path::new(pattern)
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            continue;
        }
        let full = metric.run_dir.join(pattern);
        let Some(full) = full.to_str() else { continue };
        let Ok(paths) = glob::glob(full) else {
            continue;
        };
        for path in paths.flatten() {
            // Never delete outside the run dir, including symlink escapes.
            let Ok(canonical) = fs::canonicalize(&path) else {
                continue;
            };
            if !canonical.starts_with(&run_dir) {
                continue;
            }
            let _ = if canonical.is_dir() {
                fs::remove_dir_all(&canonical)
            } else {
                fs::remove_file(&canonical)
            };
        }
    }
}

#[derive(Default)]
struct HookOutcome {
    pass: Option<bool>,
    valid: Option<bool>,
    scalars: BTreeMap<String, String>,
}

/// Reads scalar output from a `post_run_result.json` produced by a scoring hook.
fn read_post_run_outcome(path: &Path) -> HookOutcome {
    let Ok(contents) = fs::read_to_string(path) else {
        return HookOutcome::default();
    };
    let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&contents) else {
        return HookOutcome::default();
    };
    let pass = map.get("pass").and_then(Value::as_bool);
    let valid = map.get("valid").and_then(Value::as_bool).or_else(|| {
        map.get("status")
            .and_then(Value::as_str)
            .map(|status| status != "invalid")
    });
    let scalars = map
        .into_iter()
        .filter_map(|(key, value)| scalar_value(&value).map(|value| (key, value)))
        .collect();
    HookOutcome {
        pass,
        valid,
        scalars,
    }
}

fn scalar_value(value: &Value) -> Option<String> {
    match value {
        Value::Bool(value) => Some(value.to_string()),
        Value::Number(value) => Some(value.to_string()),
        Value::String(value) => Some(value.clone()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

fn apply_retention(policy: RunDbRetention, metric: &RunMetric) {
    let keep = match policy {
        RunDbRetention::Always => true,
        RunDbRetention::Never => false,
        RunDbRetention::OnFail => !metric.passed(),
    };
    if !keep {
        let _ = fs::remove_dir_all(&metric.db_path);
    }
}

fn load_config(path: Option<&Path>) -> Result<CampaignConfig> {
    let Some(path) = path else {
        return Ok(CampaignConfig::default());
    };
    let text = fs::read_to_string(path)
        .into_diagnostic()
        .with_context(|| format!("read campaign config {}", path.display()))?;
    toml::from_str(&text)
        .into_diagnostic()
        .with_context(|| format!("parse campaign config {}", path.display()))
}

pub fn resolve_run_shape(
    campaign_path: Option<&Path>,
    plan_path: &Path,
    runtime_threads_override: Option<usize>,
    workers_flag: Option<usize>,
) -> Result<ResolvedRunShape> {
    // Validate the campaign config up front (surfaces errors before the run).
    // Concurrency is governed by `workers` / the s10 admission budget, so the
    // only thing left to size here is the orchestrator's I/O thread pool.
    let config = load_config(campaign_path)?;
    let plan = read_plan(plan_path)?;
    if plan.is_empty() {
        return Err(Error::EmptyPlan).into_diagnostic();
    }
    // The recipe is not planned yet, so a requested worker count stands in
    // for the admission budget when sizing IO threads.
    let budget = requested_workers(workers_flag, &config).or(s10::admission::max_inflight());
    Ok(ResolvedRunShape {
        runtime_threads: resolve_runtime_threads(budget, plan.len(), runtime_threads_override),
    })
}

/// Number of runs allowed in flight at once: the s10 admission budget
/// (`S10_MAX_INFLIGHT`) divided by the per-run recipe weight, clamped to the
/// plan size. `S10_MAX_INFLIGHT=off` falls back to logical cores.
fn resolve_auto_workers(total_runs: usize, weight: usize, admission_max: Option<usize>) -> usize {
    let cap = total_runs.max(1);
    let weight = weight.max(1);
    let slots = match admission_max {
        Some(max) => (max / weight).max(1),
        None => available_cpus().max(1),
    };
    slots.min(cap)
}

fn worker_resolution_label(weight: usize, admission_max: Option<usize>) -> String {
    let weight = weight.max(1);
    match admission_max {
        Some(max) => format!("(S10_MAX_INFLIGHT={max}/recipe_weight={weight})"),
        None => format!(
            "(S10_MAX_INFLIGHT=off, cores={}, recipe_weight={weight})",
            available_cpus()
        ),
    }
}

/// Sizes the orchestrator's Tokio worker threads to track the admission budget
/// (each in-flight run streams its logs/IO), capped at logical cores so a large
/// `S10_MAX_INFLIGHT` does not oversubscribe the thread pool. Falls back to
/// cores when admission is disabled; never exceeds the plan size.
fn resolve_runtime_threads(
    admission_max: Option<usize>,
    plan_len: usize,
    override_threads: Option<usize>,
) -> usize {
    if let Some(threads) = override_threads.filter(|threads| *threads > 0) {
        return threads.max(1);
    }
    let available = available_cpus();
    if available <= 1 {
        return 1;
    }
    let budget = admission_max.unwrap_or(available);
    budget.min(plan_len.max(1)).clamp(2, available)
}

fn available_cpus() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

fn apply_overrides(config: &mut CampaignConfig, options: &mut RunOptions) {
    if let Some(cache_dir) = options.cache_dir.take() {
        config.cache_dir = Some(cache_dir);
    }
    if let Some(scratch_dir) = options.scratch_dir.take() {
        config.scratch_dir = Some(scratch_dir);
    }
    if let Some(workers) = options.workers {
        config.workers = Some(workers.max(1));
    }
    if let Some(retries) = options.retries {
        config.retries = retries;
    }
    if let Some(timeout) = options.timeout.take() {
        config.timeout = Some(timeout);
    }
    if let Some(hook) = options.post_run.take() {
        config.hooks.post_run = Some(hook);
    }
    if let Some(hook) = options.post_campaign.take() {
        config.hooks.post_campaign = Some(hook);
    }
    if options.fail_fast {
        config.continue_on_error = false;
    }
    if options.fail_on_run_errors {
        config.fail_on_run_errors = true;
    }
}

fn same_file_or_path(left: &Path, right: &Path) -> bool {
    left == right
        || match (fs::canonicalize(left), fs::canonicalize(right)) {
            (Ok(left), Ok(right)) => left == right,
            _ => false,
        }
}

fn read_plan(path: &Path) -> Result<Vec<PlanRow>> {
    let mut reader = csv::Reader::from_path(path)
        .into_diagnostic()
        .with_context(|| format!("read plan {}", path.display()))?;
    let headers = reader.headers().into_diagnostic()?.clone();
    let mut rows = Vec::new();
    for (idx, record) in reader.records().enumerate() {
        let record = record.into_diagnostic()?;
        let mut params = BTreeMap::new();
        let mut meta = BTreeMap::new();
        let mut run_id = None;
        let mut seed = None;
        for (header, value) in headers.iter().zip(record.iter()) {
            if value.is_empty() {
                continue;
            }
            match header {
                "run_id" => run_id = Some(value.to_string()),
                "seed" => seed = Some(value.parse::<u64>().into_diagnostic()?),
                _ if header.starts_with("param.") => {
                    params.insert(
                        header.trim_start_matches("param.").to_string(),
                        parse_cell(value),
                    );
                }
                _ if header.starts_with("meta.") => {
                    meta.insert(
                        header.trim_start_matches("meta.").to_string(),
                        parse_cell(value),
                    );
                }
                _ => {}
            }
        }
        rows.push(PlanRow {
            run_id: run_id.unwrap_or_else(|| format!("run_{idx:07}")),
            seed,
            params,
            meta,
        });
    }
    Ok(rows)
}

fn parse_cell(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
}

async fn plan_recipe(sim_path: &Path, out_dir: &Path) -> Result<s10::Recipe> {
    let plan_dir = out_dir.join("base-plan");
    fs::create_dir_all(&plan_dir).into_diagnostic()?;
    let output = s10::python_command()
        .into_diagnostic()?
        .env("ELODIN_MONTE_CARLO_PLANNING", "1")
        .arg(sim_path)
        .arg("plan")
        .arg(&plan_dir)
        .output()
        .into_diagnostic()
        .with_context(|| format!("generate s10 plan for {}", sim_path.display()))?;
    if !output.status.success() {
        return Err(miette!(
            "failed to generate s10 plan: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let plan_path = plan_dir.join("s10.toml");
    let text = fs::read_to_string(&plan_path)
        .into_diagnostic()
        .with_context(|| format!("read {}", plan_path.display()))?;
    toml::from_str(&text)
        .into_diagnostic()
        .with_context(|| format!("parse {}", plan_path.display()))
}

fn patch_recipe(recipe: s10::Recipe, row: &PlanRow, ctx: &PatchContext<'_>) -> Result<s10::Recipe> {
    let port_offset = ctx
        .slot
        .worker_id
        .checked_mul(ctx.config.resources.port_stride as usize)
        .ok_or_else(|| miette!("worker port offset overflow"))?;
    let mut env = HashMap::new();
    // Campaign `[env]` table first: runner-managed variables always win.
    for (key, value) in &ctx.config.env {
        env.insert(key.clone(), render_template(value, row, ctx));
    }
    // Machine-sympathy defaults, only when the user set them nowhere (shell,
    // campaign [env]): cap per-process thread pools so N concurrent sims do
    // not each size their pools to every core on the box. The box runs
    // `workers x recipe_weight` processes, so that is the divisor — at full
    // packing every process gets one thread.
    let processes = ctx.worker_count.max(1) * ctx.recipe_weight.max(1);
    let threads = (available_cpus() / processes).max(1);
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"] {
        if std::env::var_os(key).is_none() && !env.contains_key(key) {
            env.insert(key.to_string(), threads.to_string());
        }
    }
    if std::env::var_os("XLA_FLAGS").is_none() && !env.contains_key("XLA_FLAGS") {
        env.insert(
            "XLA_FLAGS".to_string(),
            format!("--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={threads}"),
        );
    }
    env.extend([
        (
            CONTEXT_ENV.to_string(),
            ctx.context_path.to_string_lossy().to_string(),
        ),
        (
            CACHE_ENV.to_string(),
            ctx.cache_dir.to_string_lossy().to_string(),
        ),
        (
            "ELODIN_MONTE_CARLO_WORKER_ID".to_string(),
            ctx.slot.worker_id.to_string(),
        ),
        (
            "ELODIN_MONTE_CARLO_DB_PORT".to_string(),
            ctx.slot.db_port.to_string(),
        ),
        (
            "ELODIN_MONTE_CARLO_PORT_OFFSET".to_string(),
            port_offset.to_string(),
        ),
        (
            "ELODIN_DB_PATH".to_string(),
            ctx.db_path.to_string_lossy().to_string(),
        ),
        (
            "ELODIN_MONTE_CARLO_RUN_DIR".to_string(),
            ctx.run_dir.to_string_lossy().to_string(),
        ),
        (
            "ELODIN_SIM_SUMMARY_JSON".to_string(),
            ctx.run_dir
                .join("sim_summary.json")
                .to_string_lossy()
                .to_string(),
        ),
    ]);
    if let Some(seed) = row.seed {
        env.insert("ELODIN_MONTE_CARLO_SEED".to_string(), seed.to_string());
    }
    for (name, port) in &ctx.slot.ports {
        env.insert(named_port_env(name), port.to_string());
    }
    if let Some(delivery) = &ctx.config.params_delivery {
        write_params_delivery(delivery, row, ctx, &mut env)?;
    }

    Ok(patch_recipe_env(
        recipe,
        &env,
        ctx.slot.db_addr,
        ctx.run_dir,
        "recipe",
    ))
}

fn named_port_env(name: &str) -> String {
    let suffix = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("ELODIN_MC_PORT_{suffix}")
}

fn write_params_delivery(
    delivery: &ParamsDeliveryConfig,
    row: &PlanRow,
    ctx: &PatchContext<'_>,
    env: &mut HashMap<String, String>,
) -> Result<()> {
    let relative = render_template(&delivery.file.to_string_lossy(), row, ctx);
    let path = if Path::new(&relative).is_absolute() {
        PathBuf::from(relative)
    } else {
        ctx.run_dir.join(relative)
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).into_diagnostic()?;
    }
    match delivery.format {
        ParamsDeliveryFormat::Json => write_json(&path, &row.params)?,
        ParamsDeliveryFormat::Toml => {
            let text = toml::to_string_pretty(&row.params).into_diagnostic()?;
            fs::write(&path, text).into_diagnostic()?;
        }
    }
    if let Some(env_var) = &delivery.env_var {
        env.insert(env_var.clone(), path.to_string_lossy().to_string());
    }
    for (key, value) in &delivery.env {
        env.insert(key.clone(), render_template(value, row, ctx));
    }
    Ok(())
}

fn render_template(template: &str, row: &PlanRow, ctx: &PatchContext<'_>) -> String {
    template
        .replace(
            "{seed}",
            &row.seed.map(|seed| seed.to_string()).unwrap_or_default(),
        )
        .replace("{run_id}", &row.run_id)
        .replace("{db_path}", &ctx.db_path.to_string_lossy())
        .replace("{run_dir}", &ctx.run_dir.to_string_lossy())
}

struct PatchContext<'a> {
    slot: &'a ResourceSlot,
    context_path: &'a Path,
    cache_dir: &'a Path,
    db_path: &'a Path,
    run_dir: &'a Path,
    config: &'a CampaignConfig,
    worker_count: usize,
    recipe_weight: usize,
}

fn patch_recipe_env(
    recipe: s10::Recipe,
    env: &HashMap<String, String>,
    db_addr: SocketAddr,
    run_dir: &Path,
    name: &str,
) -> s10::Recipe {
    match recipe {
        s10::Recipe::Group(mut group) => {
            group.recipes = group
                .recipes
                .into_iter()
                .map(|(name, recipe)| {
                    let patched = patch_recipe_env(recipe, env, db_addr, run_dir, &name);
                    (name, patched)
                })
                .collect();
            s10::Recipe::Group(group)
        }
        s10::Recipe::Process(mut process) => {
            process.process_args.env.extend(env.clone());
            process.process_args.fail_on_error = true;
            process.process_args.log_path = Some(log_path_for(run_dir, name));
            // Group-kill fallback: without a delegated cgroup, this is what
            // keeps a timed-out run's daemons from squatting on worker ports.
            process.process_args.own_process_group = true;
            s10::Recipe::Process(process)
        }
        s10::Recipe::Cargo(mut cargo) => {
            cargo.process_args.env.extend(env.clone());
            cargo.process_args.fail_on_error = true;
            cargo.process_args.log_path = Some(log_path_for(run_dir, name));
            cargo.process_args.own_process_group = true;
            s10::Recipe::Cargo(cargo)
        }
        #[cfg(not(target_os = "windows"))]
        s10::Recipe::Sim(mut sim) => {
            sim.addr = db_addr;
            sim.env.extend(env.clone());
            sim.log_path = Some(log_path_for(run_dir, name));
            sim.own_process_group = true;
            s10::Recipe::Sim(sim)
        }
    }
}

fn log_path_for(run_dir: &Path, name: &str) -> PathBuf {
    let sanitized = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    run_dir.join("logs").join(format!("{sanitized}.log"))
}

fn write_run_context(
    path: &Path,
    row: &PlanRow,
    slot: &ResourceSlot,
    db_path: &Path,
    run_dir: &Path,
    cache_dir: &Path,
    _config: &CampaignConfig,
) -> Result<()> {
    let value = json!({
        "run_id": row.run_id,
        "seed": row.seed,
        "db_path": db_path,
        "db_addr": slot.db_addr.to_string(),
        "cache_dir": cache_dir,
        "run_dir": run_dir,
        "params": row.params,
        "meta": row.meta,
        "slots": {
            "worker_id": slot.worker_id,
            "db_port": slot.db_port,
            "ports": slot.ports,
        },
    });
    write_json(path, &value)
}

async fn run_hook(kind: &str, hook: &Path, context: &Path) -> Result<()> {
    let output = s10::python_command()
        .into_diagnostic()?
        .arg("-m")
        .arg("elodin.monte_carlo.run_hook")
        .arg(hook)
        .arg(kind)
        .arg(context)
        .output()
        .into_diagnostic()
        .with_context(|| format!("run {kind} hook {}", hook.display()))?;
    let hook_log = context.with_file_name(format!("{kind}.log"));
    if let Some(parent) = hook_log.parent() {
        fs::create_dir_all(parent).into_diagnostic()?;
    }
    fs::write(
        &hook_log,
        [
            output.stdout.as_slice(),
            b"\n--- stderr ---\n",
            output.stderr.as_slice(),
        ]
        .concat(),
    )
    .into_diagnostic()
    .with_context(|| format!("write {}", hook_log.display()))?;
    if kind == "post_campaign" && !output.stdout.is_empty() {
        print!("{}", String::from_utf8_lossy(&output.stdout));
    }
    if !output.status.success() {
        return Err(miette!("{kind} hook failed: {}", hook.display()));
    }
    Ok(())
}

fn write_perf_csv(path: &Path, metrics: &[RunMetric]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path).into_diagnostic()?;
    writer
        .write_record([
            "run_id",
            "worker_id",
            "attempt",
            "status",
            "exit_ok",
            "scored_pass",
            "scored_valid",
            "failure_reason",
            "degraded",
            "behind_deadline_frac",
            "real_time_factor",
            "drift_resets",
            "wall_ms",
            "started_at",
            "finished_at",
            "spawn_unix_ns",
            "exit_unix_ns",
            "entry_unix_ns",
            "compile_done_unix_ns",
            "loop_start_unix_ns",
            "loop_end_unix_ns",
            "summary_written_unix_ns",
            "python_import_ms",
            "compile_ms",
            "loop_ms",
            "teardown_ms",
            "process_shutdown_ms",
            "db_path",
            "run_dir",
        ])
        .into_diagnostic()?;
    for metric in metrics {
        writer
            .write_record([
                metric.run_id.clone(),
                metric
                    .worker_id
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric.attempt.to_string(),
                metric.status.clone(),
                metric.exit_ok.to_string(),
                metric
                    .scored_pass
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .scored_valid
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric.failure_reason.clone().unwrap_or_default(),
                metric.degraded.to_string(),
                metric
                    .behind_deadline_frac
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .real_time_factor
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .drift_resets
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric.wall_ms.to_string(),
                metric.started_at.to_rfc3339(),
                metric.finished_at.to_rfc3339(),
                metric.spawn_unix_ns.to_string(),
                metric.exit_unix_ns.to_string(),
                metric
                    .entry_unix_ns
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .compile_done_unix_ns
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .loop_start_unix_ns
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .loop_end_unix_ns
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .summary_written_unix_ns
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .python_import_ms
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .compile_ms
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .loop_ms
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .teardown_ms
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric
                    .process_shutdown_ms
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metric.db_path.to_string_lossy().to_string(),
                metric.run_dir.to_string_lossy().to_string(),
            ])
            .into_diagnostic()?;
    }
    writer.flush().into_diagnostic()
}

fn write_timeline_csv(path: &Path, samples: &[TimelineSample]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path).into_diagnostic()?;
    for sample in samples {
        writer.serialize(sample).into_diagnostic()?;
    }
    writer.flush().into_diagnostic()
}

fn write_resource_csv(path: &Path, samples: &[ResourceSample]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path).into_diagnostic()?;
    for sample in samples {
        writer.serialize(sample).into_diagnostic()?;
    }
    writer.flush().into_diagnostic()
}

fn write_process_csv(path: &Path, samples: &[ProcessSample]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path).into_diagnostic()?;
    for sample in samples {
        writer.serialize(sample).into_diagnostic()?;
    }
    writer.flush().into_diagnostic()
}

fn write_results_csv(path: &Path, out_dir: &Path, metrics: &[RunMetric]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path).into_diagnostic()?;
    let hook_keys = metrics
        .iter()
        .flat_map(|metric| metric.hook_scalars.keys().cloned())
        .filter(|key| !RESERVED_HOOK_KEYS.contains(&key.as_str()))
        .collect::<BTreeSet<_>>();
    let mut header = vec![
        "run_id".to_string(),
        "status".to_string(),
        "passed".to_string(),
        "valid".to_string(),
        "degraded".to_string(),
        "scored_pass".to_string(),
        "scored_valid".to_string(),
        "failure_reason".to_string(),
        "behind_deadline_frac".to_string(),
        "real_time_factor".to_string(),
        "drift_resets".to_string(),
        "worker_id".to_string(),
        "wall_ms".to_string(),
        "db_path".to_string(),
        "result_json".to_string(),
    ];
    header.extend(hook_keys.iter().cloned());
    writer.write_record(&header).into_diagnostic()?;
    for metric in metrics {
        let result_json = metric.run_dir.join("result.json");
        let scored_pass = metric
            .scored_pass
            .map(|pass| pass.to_string())
            .unwrap_or_default();
        let scored_valid = metric
            .scored_valid
            .map(|valid| valid.to_string())
            .unwrap_or_default();
        let mut row = vec![
            metric.run_id.clone(),
            metric.status.clone(),
            metric.passed().to_string(),
            metric.valid().to_string(),
            metric.degraded.to_string(),
            scored_pass,
            scored_valid,
            metric.failure_reason.clone().unwrap_or_default(),
            metric
                .behind_deadline_frac
                .map(|value| value.to_string())
                .unwrap_or_default(),
            metric
                .real_time_factor
                .map(|value| value.to_string())
                .unwrap_or_default(),
            metric
                .drift_resets
                .map(|value| value.to_string())
                .unwrap_or_default(),
            metric
                .worker_id
                .map(|id| id.to_string())
                .unwrap_or_default(),
            metric.wall_ms.to_string(),
            metric.db_path.to_string_lossy().to_string(),
            result_json
                .strip_prefix(out_dir)
                .unwrap_or(&result_json)
                .to_string_lossy()
                .to_string(),
        ];
        row.extend(
            hook_keys
                .iter()
                .map(|key| metric.hook_scalars.get(key).cloned().unwrap_or_default()),
        );
        writer.write_record(&row).into_diagnostic()?;
    }
    writer.flush().into_diagnostic()
}

fn aggregate_sim_summaries(metrics: &[RunMetric]) -> Option<SimSummaryAggregate> {
    let mut aggregate: Option<SimSummaryAggregate> = None;
    for metric in metrics {
        let Some(summary) = read_sim_run_summary(&metric.run_dir.join("sim_summary.json")) else {
            continue;
        };
        match &mut aggregate {
            Some(aggregate) => aggregate.merge(summary),
            None => aggregate = Some(SimSummaryAggregate::from_run(summary)),
        }
    }
    aggregate
}

fn read_sim_run_summary(path: &Path) -> Option<SimRunSummary> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str::<SimRunSummary>(&text).ok()
}

fn diff_ms(start: Option<u64>, end: Option<u64>) -> Option<f64> {
    Some(end?.saturating_sub(start?) as f64 / 1_000_000.0)
}

/// Human-readable label for a metric's worker slot. Skipped runs are never
/// assigned to a worker, so they render as `-` rather than a fake slot.
fn worker_label(worker_id: Option<usize>) -> String {
    match worker_id {
        Some(id) => id.to_string(),
        None => "-".to_string(),
    }
}

fn placeholder_metric(
    run_id: &str,
    worker_id: Option<usize>,
    out_dir: &Path,
    status: &str,
) -> RunMetric {
    let now = Utc::now();
    let unix_ns = unix_now_ns();
    let run_dir = out_dir.join("runs").join(run_id);
    RunMetric {
        run_id: run_id.to_string(),
        worker_id,
        attempt: 0,
        status: status.to_string(),
        exit_ok: false,
        scored_pass: None,
        scored_valid: None,
        failure_reason: None,
        degraded: false,
        behind_deadline_frac: None,
        real_time_factor: None,
        drift_resets: None,
        hook_scalars: BTreeMap::new(),
        wall_ms: 0,
        started_at: now,
        finished_at: now,
        spawn_unix_ns: unix_ns,
        exit_unix_ns: unix_ns,
        entry_unix_ns: None,
        compile_done_unix_ns: None,
        loop_start_unix_ns: None,
        loop_end_unix_ns: None,
        summary_written_unix_ns: None,
        python_import_ms: None,
        compile_ms: None,
        loop_ms: None,
        teardown_ms: None,
        process_shutdown_ms: None,
        db_path: run_dir.join("db"),
        run_dir,
    }
}

impl SimSummaryAggregate {
    fn from_run(summary: SimRunSummary) -> Self {
        Self {
            runs: 1,
            pre_step: summary.pre_step,
            copy_db_to_world: summary.copy_db_to_world,
            world_run: summary.world_run,
            commit: summary.commit,
            wait_for_write: summary.wait_for_write,
            post_step: summary.post_step,
            real_time_pacing: summary.real_time_pacing,
            total_cycle: summary.total_cycle,
            cycles: summary.cycles,
            steps: summary.steps,
            simulation_rate_hz: summary.simulation_rate_hz,
            telemetry_rate_hz: summary.telemetry_rate_hz,
            wall_ns: summary.wall_ns,
        }
    }

    fn merge(&mut self, summary: SimRunSummary) {
        self.runs += 1;
        self.pre_step.merge(summary.pre_step);
        self.copy_db_to_world.merge(summary.copy_db_to_world);
        self.world_run.merge(summary.world_run);
        self.commit.merge(summary.commit);
        self.wait_for_write.merge(summary.wait_for_write);
        self.post_step.merge(summary.post_step);
        self.real_time_pacing.merge(summary.real_time_pacing);
        self.total_cycle.merge(summary.total_cycle);
        self.cycles = self.cycles.saturating_add(summary.cycles);
        self.steps = self.steps.saturating_add(summary.steps);
        self.wall_ns = self.wall_ns.saturating_add(summary.wall_ns);
    }
}

impl SimPhaseSummary {
    fn merge(&mut self, other: SimPhaseSummary) {
        if self.count == 0 {
            *self = other;
            return;
        }
        if other.count == 0 {
            return;
        }
        self.sum_ns = self.sum_ns.saturating_add(other.sum_ns);
        self.count = self.count.saturating_add(other.count);
        self.min_ns = self.min_ns.min(other.min_ns);
        self.max_ns = self.max_ns.max(other.max_ns);
        for (slot, value) in self.buckets.iter_mut().zip(other.buckets) {
            *slot = slot.saturating_add(value);
        }
    }

    fn mean_ns(&self) -> u64 {
        self.sum_ns.checked_div(self.count).unwrap_or(0)
    }

    // Mirrors `PhaseStats::percentile_ns` in `libs/nox-py/src/tick_metrics.rs`.
    fn percentile_ns(&self, q: f64) -> u64 {
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
        for (i, &count) in self.buckets.iter().enumerate() {
            let count = count as u64;
            if count == 0 {
                continue;
            }
            let prev = cum;
            cum += count;
            if cum >= target {
                let lo = 1u64 << i;
                let hi = if i + 1 < 64 {
                    1u64 << (i + 1)
                } else {
                    u64::MAX
                };
                let frac = (target - prev) as f64 / count as f64;
                let within = (lo as f64 + frac * (hi - lo) as f64) as u64;
                return within.clamp(self.min_ns, self.max_ns);
            }
        }
        self.max_ns
    }
}

fn write_campaign_summary(
    out_dir: &Path,
    summary: &CampaignSummary,
    sim_summary: Option<&SimSummaryAggregate>,
    memory: &[CacheMappingSample],
) -> Result<()> {
    let mut rendered = String::new();
    rendered.push_str("──── elodin monte-carlo campaign summary ────\n");
    rendered.push_str(&format!(
        "  runs:              {} ok / {} failed / {} invalid / {} degraded / {} skipped / {} total\n",
        summary.passed,
        summary.failed,
        summary.invalid,
        summary.degraded,
        summary.skipped,
        summary.total_runs
    ));
    rendered.push_str(&format!("  workers:           {}\n", summary.workers));
    if let Some(pacing) = &summary.pacing {
        rendered.push_str(&format!(
            "  pacing:            worst behind-deadline {:.1}% ({}), worst real-time factor {:.3} ({}) over {} paced run(s)\n",
            pacing.max_behind_deadline_frac * 100.0,
            pacing.max_behind_run,
            pacing.max_real_time_factor,
            pacing.max_real_time_factor_run,
            pacing.paced_runs
        ));
    }
    rendered.push_str(&format!(
        "  wall time:         {:.3} s\n",
        summary.wall_ms as f64 / 1000.0
    ));
    rendered.push_str(&format!(
        "  avg run wall:      {:.3} s\n",
        summary.average_run_wall_ms / 1000.0
    ));
    rendered.push_str(&format!(
        "  parallel eff.:     {:.1}%\n",
        summary.parallel_efficiency * 100.0
    ));
    rendered.push_str(&format!(
        "  disk allocated:    {} bytes\n",
        summary.disk_bytes
    ));
    rendered.push_str(&format!(
        "  cpu avg/peak:      {:.1}% / {:.1}%\n",
        summary.resource_summary.average_cpu_percent, summary.resource_summary.peak_cpu_percent
    ));
    rendered.push_str(&format!(
        "  load/ctx-switch:   peak load1={:.2} peak ctxt/s={:.0}\n",
        summary.resource_summary.peak_load_average_1m,
        summary.resource_summary.peak_context_switches_per_sec
    ));
    rendered.push_str(&format!(
        "  concurrency:       mean={:.1} peak={}\n",
        summary.concurrency_summary.mean_active_runs, summary.concurrency_summary.peak_active_runs
    ));
    if !summary.concurrency_summary.buckets.is_empty() {
        rendered.push('\n');
        rendered.push_str("  run cost by concurrency (active / runs / avg wall):\n");
        for bucket in &summary.concurrency_summary.buckets {
            rendered.push_str(&format!(
                "    {:>6} {:>8} {}\n",
                bucket.concurrency,
                bucket.runs,
                fmt_ms(bucket.average_run_wall_ms)
            ));
        }
    }
    if !summary.hook_metrics.is_empty() {
        rendered.push('\n');
        rendered.push_str("  hook metrics (count / min / mean / p95 / max):\n");
        for (name, metric) in &summary.hook_metrics {
            rendered.push_str(&format!(
                "    {:<20} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>10.4}\n",
                name, metric.count, metric.min, metric.mean, metric.p95, metric.max
            ));
        }
    }
    if summary.phase_attribution.samples > 0 {
        rendered.push('\n');
        rendered.push_str("  per-run phase attribution (avg / p95):\n");
        rendered.push_str(&fmt_ms_row(
            "python import",
            summary.phase_attribution.average_python_import_ms,
            summary.phase_attribution.p95_python_import_ms,
        ));
        rendered.push('\n');
        rendered.push_str(&fmt_ms_row(
            "compile",
            summary.phase_attribution.average_compile_ms,
            summary.phase_attribution.p95_compile_ms,
        ));
        rendered.push('\n');
        rendered.push_str(&fmt_ms_row(
            "loop",
            summary.phase_attribution.average_loop_ms,
            summary.phase_attribution.p95_loop_ms,
        ));
        rendered.push('\n');
        rendered.push_str(&fmt_ms_row(
            "teardown",
            summary.phase_attribution.average_teardown_ms,
            summary.phase_attribution.p95_teardown_ms,
        ));
        rendered.push('\n');
        rendered.push_str(&fmt_ms_row(
            "process exit",
            summary.phase_attribution.average_process_shutdown_ms,
            summary.phase_attribution.p95_process_shutdown_ms,
        ));
        rendered.push('\n');
    }
    if !memory.is_empty() {
        let virtual_kib = memory.iter().map(|sample| sample.virtual_kib).sum::<u64>();
        let rss_kib = memory.iter().map(|sample| sample.rss_kib).sum::<u64>();
        let pss_kib = memory.iter().map(|sample| sample.pss_kib).sum::<u64>();
        rendered.push_str(&format!(
            "  shared constants:  mappings={} virtual={:.1} MiB rss={:.1} MiB pss={:.1} MiB\n",
            memory.len(),
            virtual_kib as f64 / 1024.0,
            rss_kib as f64 / 1024.0,
            pss_kib as f64 / 1024.0
        ));
    }
    if let Some(sim_summary) = sim_summary {
        rendered.push('\n');
        rendered.push_str(&render_sim_summary(sim_summary));
    }
    rendered.push_str("────────────────────────────────────────────\n");
    print!("{rendered}");
    fs::write(out_dir.join("campaign_summary.txt"), rendered)
        .into_diagnostic()
        .with_context(|| format!("write {}", out_dir.join("campaign_summary.txt").display()))
}

fn unix_now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
        .unwrap_or_default()
}

fn render_sim_summary(summary: &SimSummaryAggregate) -> String {
    let mut rendered = String::new();
    rendered.push_str(&format!(
        "──── elodin campaign simulation summary ({} runs) ────\n",
        summary.runs
    ));
    rendered.push_str(&format!(
        "  aggregate sim wall: {:.3} s\n",
        summary.wall_ns as f64 / 1e9
    ));
    if summary.cycles == 0 {
        rendered.push_str("  simulation cycles:  0  (no cycles executed)\n");
        return rendered;
    }
    rendered.push_str(&format!(
        "  simulation cycles:  {}    mean cycle: {}  (effective {:.0} cycles/s)\n",
        format_commas(summary.cycles),
        fmt_ns(summary.total_cycle.mean_ns()).trim(),
        if summary.wall_ns > 0 {
            summary.cycles as f64 / (summary.wall_ns as f64 / 1e9)
        } else {
            0.0
        }
    ));
    rendered.push('\n');
    rendered.push_str("  per-cycle phase timing (mean / p95 / max):\n");
    rendered.push_str(&fmt_row("pre_step", &summary.pre_step));
    rendered.push('\n');
    rendered.push_str(&fmt_row("read (db → world)", &summary.copy_db_to_world));
    rendered.push('\n');
    rendered.push_str(&fmt_row("tick function", &summary.world_run));
    rendered.push('\n');
    rendered.push_str(&fmt_row("commit world", &summary.commit));
    rendered.push('\n');
    rendered.push_str(&fmt_row("wait_for_write", &summary.wait_for_write));
    rendered.push('\n');
    rendered.push_str(&fmt_row("post_step", &summary.post_step));
    rendered.push('\n');
    rendered.push_str(&fmt_row("real-time pacing", &summary.real_time_pacing));
    rendered.push('\n');
    rendered
}

fn fmt_row(name: &str, phase: &SimPhaseSummary) -> String {
    format!(
        "    {:<18} {}  {}  {}",
        name,
        fmt_cell(phase, -1.0),
        fmt_cell(phase, 0.95),
        fmt_cell(phase, 1.0)
    )
}

fn fmt_ms_row(name: &str, avg_ms: f64, p95_ms: f64) -> String {
    format!("    {:<14} {}  {}", name, fmt_ms(avg_ms), fmt_ms(p95_ms))
}

fn fmt_cell(phase: &SimPhaseSummary, q: f64) -> String {
    if phase.count == 0 {
        "        —".to_string()
    } else {
        let value = if q == 1.0 {
            phase.max_ns
        } else if q < 0.0 {
            phase.mean_ns()
        } else {
            phase.percentile_ns(q)
        };
        format!("{:>9}", fmt_ns(value))
    }
}

fn fmt_ms(ms: f64) -> String {
    format!("{ms:>8.1} ms")
}

fn fmt_ns(ns: u64) -> String {
    if ns < 1_000 {
        format!("{ns:>6} ns")
    } else if ns < 1_000_000 {
        format!("{:>6.1} µs", ns as f64 / 1e3)
    } else if ns < 1_000_000_000 {
        format!("{:>6.1} ms", ns as f64 / 1e6)
    } else {
        format!("{:>6.2} s ", ns as f64 / 1e9)
    }
}

fn format_commas(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let first = bytes.len() % 3;
    if first > 0 {
        out.push_str(std::str::from_utf8(&bytes[..first]).unwrap_or_default());
    }
    for (i, chunk) in bytes[first..].chunks(3).enumerate() {
        if i > 0 || first > 0 {
            out.push(',');
        }
        out.push_str(std::str::from_utf8(chunk).unwrap_or_default());
    }
    out
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).into_diagnostic()?;
    }
    let json = serde_json::to_string_pretty(value).into_diagnostic()?;
    fs::write(path, json)
        .into_diagnostic()
        .with_context(|| format!("write {}", path.display()))
}

fn build_timeline(metrics: &[RunMetric]) -> (Vec<TimelineSample>, ConcurrencySummary) {
    if metrics.is_empty() {
        return (Vec::new(), ConcurrencySummary::default());
    }
    let base = metrics
        .iter()
        .map(|metric| metric.spawn_unix_ns)
        .min()
        .unwrap_or_default();
    let mut events = Vec::with_capacity(metrics.len() * 2);
    for metric in metrics {
        events.push((metric.spawn_unix_ns, 1_i32));
        events.push((metric.exit_unix_ns, -1_i32));
    }
    events.sort_by_key(|(timestamp, delta)| (*timestamp, *delta));

    let mut active = 0_i32;
    let mut prev = events[0].0;
    let mut timeline = Vec::new();
    let mut weighted_active_ns = 0_f64;
    let mut peak_active_runs = 0_usize;
    for (timestamp, delta) in events {
        if timestamp > prev && active > 0 {
            let duration = timestamp - prev;
            weighted_active_ns += duration as f64 * active as f64;
            let active_runs = active as usize;
            peak_active_runs = peak_active_runs.max(active_runs);
            timeline.push(TimelineSample {
                start_ms: (prev.saturating_sub(base)) as f64 / 1_000_000.0,
                end_ms: (timestamp.saturating_sub(base)) as f64 / 1_000_000.0,
                active_runs,
            });
        }
        active = (active + delta).max(0);
        prev = timestamp;
    }
    let wall_ns = metrics
        .iter()
        .map(|metric| metric.exit_unix_ns)
        .max()
        .unwrap_or(base)
        .saturating_sub(base);
    let mut buckets: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for metric in metrics {
        let avg_concurrency = average_concurrency_for_run(metric, &timeline, base);
        let bucket = avg_concurrency.round().max(1.0) as usize;
        buckets
            .entry(bucket)
            .or_default()
            .push(metric.wall_ms as f64);
    }
    let buckets = buckets
        .into_iter()
        .map(|(concurrency, values)| ConcurrencyBucket {
            concurrency,
            runs: values.len(),
            average_run_wall_ms: average(&values),
        })
        .collect();
    (
        timeline,
        ConcurrencySummary {
            mean_active_runs: if wall_ns == 0 {
                0.0
            } else {
                weighted_active_ns / wall_ns as f64
            },
            peak_active_runs,
            buckets,
        },
    )
}

fn average_concurrency_for_run(metric: &RunMetric, timeline: &[TimelineSample], base: u64) -> f64 {
    let run_start = (metric.spawn_unix_ns.saturating_sub(base)) as f64 / 1_000_000.0;
    let run_end = (metric.exit_unix_ns.saturating_sub(base)) as f64 / 1_000_000.0;
    let mut weighted = 0.0;
    let mut duration = 0.0;
    for sample in timeline {
        let start = run_start.max(sample.start_ms);
        let end = run_end.min(sample.end_ms);
        if end > start {
            let overlap = end - start;
            weighted += overlap * sample.active_runs as f64;
            duration += overlap;
        }
    }
    if duration > 0.0 {
        weighted / duration
    } else {
        0.0
    }
}

#[cfg(target_os = "linux")]
fn sample_cache_mappings(cache_dir: &Path) -> Vec<CacheMappingSample> {
    let cache_prefix = cache_dir.to_string_lossy().to_string();
    let Ok(proc_entries) = fs::read_dir("/proc") else {
        return Vec::new();
    };
    let mut samples = Vec::new();
    for entry in proc_entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Ok(pid) = name.parse::<u32>() else {
            continue;
        };
        let proc_dir = entry.path();
        let Ok(smaps) = fs::read_to_string(proc_dir.join("smaps")) else {
            continue;
        };
        let mut paths = Vec::<String>::new();
        let mut virtual_kib = 0_u64;
        let mut rss_kib = 0_u64;
        let mut pss_kib = 0_u64;
        let mut in_cache_mapping = false;
        for line in smaps.lines() {
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.first().is_some_and(|part| {
                part.contains('-') && part.chars().filter(|c| *c == '-').count() == 1
            }) {
                in_cache_mapping = parts
                    .last()
                    .is_some_and(|path| path.starts_with(cache_prefix.as_str()));
                if in_cache_mapping {
                    if let Some(range) = parts.first()
                        && let Some((start, end)) = range.split_once('-')
                        && let (Ok(start), Ok(end)) =
                            (u64::from_str_radix(start, 16), u64::from_str_radix(end, 16))
                    {
                        virtual_kib += (end.saturating_sub(start)) / 1024;
                    }
                    if let Some(path) = parts.last() {
                        paths.push((*path).to_string());
                    }
                }
                continue;
            }
            if in_cache_mapping && line.starts_with("Rss:") {
                rss_kib += parse_kib(line);
            } else if in_cache_mapping && line.starts_with("Pss:") {
                pss_kib += parse_kib(line);
            }
        }
        paths.sort();
        paths.dedup();
        if paths.is_empty() {
            continue;
        }
        let cmd = fs::read(proc_dir.join("cmdline"))
            .map(|bytes| {
                String::from_utf8_lossy(&bytes)
                    .replace('\0', " ")
                    .trim()
                    .chars()
                    .take(240)
                    .collect::<String>()
            })
            .unwrap_or_default();
        samples.push(CacheMappingSample {
            pid,
            virtual_kib,
            rss_kib,
            pss_kib,
            cmd,
            paths,
        });
    }
    samples
}

#[cfg(not(target_os = "linux"))]
fn sample_cache_mappings(_cache_dir: &Path) -> Vec<CacheMappingSample> {
    Vec::new()
}

fn spawn_memory_sampler(
    cache_dir: PathBuf,
    stop: Arc<AtomicBool>,
    peak: Arc<Mutex<Vec<CacheMappingSample>>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut best_score = (0_usize, 0_u64, 0_u64);
        while !stop.load(Ordering::SeqCst) {
            let samples = sample_cache_mappings(&cache_dir);
            let score = memory_score(&samples);
            if score > best_score {
                best_score = score;
                *peak.lock().expect("memory mutex poisoned") = samples;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let samples = sample_cache_mappings(&cache_dir);
        let score = memory_score(&samples);
        if score > best_score {
            *peak.lock().expect("memory mutex poisoned") = samples;
        }
    })
}

fn spawn_resource_sampler(
    out_dir: PathBuf,
    started_at: Instant,
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<ResourceSample>>>,
    process_samples: Option<Arc<Mutex<Vec<ProcessSample>>>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut previous_cpu = read_cpu_sample();
        let mut previous_per_core = read_per_core_cpu_samples();
        let mut previous_context_switches = read_context_switches();
        let mut previous_sample_at = Instant::now();
        while !stop.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(250)).await;
            let now = Instant::now();
            let current_cpu = read_cpu_sample();
            let cpu_percent = match (previous_cpu, current_cpu) {
                (Some(previous), Some(current)) => cpu_percent(previous, current),
                _ => 0.0,
            };
            previous_cpu = current_cpu;
            let current_per_core = read_per_core_cpu_samples();
            let per_core = per_core_cpu_percent(&previous_per_core, &current_per_core);
            let cpu_min_percent = if per_core.is_empty() {
                0.0
            } else {
                per_core.iter().copied().fold(f64::INFINITY, f64::min)
            };
            let cpu_max_percent = per_core.iter().copied().fold(0.0, f64::max);
            previous_per_core = current_per_core;
            let current_context_switches = read_context_switches();
            let context_switches_per_sec =
                match (previous_context_switches, current_context_switches) {
                    (Some(previous), Some(current)) => {
                        let elapsed = now.duration_since(previous_sample_at).as_secs_f64();
                        if elapsed > 0.0 {
                            current.saturating_sub(previous) as f64 / elapsed
                        } else {
                            0.0
                        }
                    }
                    _ => 0.0,
                };
            previous_context_switches = current_context_switches;
            previous_sample_at = now;
            let (mem_total_kib, mem_available_kib) = read_meminfo();
            let elapsed_ms = started_at.elapsed().as_millis();
            samples
                .lock()
                .expect("resource samples mutex poisoned")
                .push(ResourceSample {
                    elapsed_ms,
                    cpu_percent,
                    cpu_min_percent,
                    cpu_max_percent,
                    load_average_1m: read_loadavg_1m(),
                    context_switches_per_sec,
                    mem_total_kib,
                    mem_available_kib,
                    campaign_disk_bytes: 0,
                });
            if let Some(process_samples) = &process_samples {
                process_samples
                    .lock()
                    .expect("process samples mutex poisoned")
                    .extend(sample_run_processes(&out_dir, elapsed_ms));
            }
        }
    })
}

fn summarize_resources(samples: &[ResourceSample]) -> ResourceSummary {
    if samples.is_empty() {
        return ResourceSummary::default();
    }
    let average_cpu_percent =
        samples.iter().map(|sample| sample.cpu_percent).sum::<f64>() / samples.len() as f64;
    let peak_cpu_percent = samples
        .iter()
        .map(|sample| sample.cpu_percent)
        .fold(0.0, f64::max);
    let peak_cpu_core_percent = samples
        .iter()
        .map(|sample| sample.cpu_max_percent)
        .fold(0.0, f64::max);
    let peak_load_average_1m = samples
        .iter()
        .map(|sample| sample.load_average_1m)
        .fold(0.0, f64::max);
    let peak_context_switches_per_sec = samples
        .iter()
        .map(|sample| sample.context_switches_per_sec)
        .fold(0.0, f64::max);
    let peak_memory_used_kib = samples
        .iter()
        .map(|sample| {
            sample
                .mem_total_kib
                .saturating_sub(sample.mem_available_kib)
        })
        .max()
        .unwrap_or_default();
    let peak_campaign_disk_bytes = samples
        .iter()
        .map(|sample| sample.campaign_disk_bytes)
        .max()
        .unwrap_or_default();
    ResourceSummary {
        samples: samples.len(),
        average_cpu_percent,
        peak_cpu_percent,
        peak_cpu_core_percent,
        peak_load_average_1m,
        peak_context_switches_per_sec,
        peak_memory_used_kib,
        peak_campaign_disk_bytes,
    }
}

fn summarize_phase_attribution(metrics: &[RunMetric]) -> PhaseAttributionSummary {
    let python_import = metrics
        .iter()
        .filter_map(|metric| metric.python_import_ms)
        .collect::<Vec<_>>();
    let compile = metrics
        .iter()
        .filter_map(|metric| metric.compile_ms)
        .collect::<Vec<_>>();
    let loop_ms = metrics
        .iter()
        .filter_map(|metric| metric.loop_ms)
        .collect::<Vec<_>>();
    let teardown = metrics
        .iter()
        .filter_map(|metric| metric.teardown_ms)
        .collect::<Vec<_>>();
    let process_shutdown = metrics
        .iter()
        .filter_map(|metric| metric.process_shutdown_ms)
        .collect::<Vec<_>>();
    PhaseAttributionSummary {
        samples: [
            python_import.len(),
            compile.len(),
            loop_ms.len(),
            teardown.len(),
            process_shutdown.len(),
        ]
        .into_iter()
        .min()
        .unwrap_or_default(),
        average_python_import_ms: average(&python_import),
        average_compile_ms: average(&compile),
        average_loop_ms: average(&loop_ms),
        average_teardown_ms: average(&teardown),
        average_process_shutdown_ms: average(&process_shutdown),
        p95_python_import_ms: percentile(&python_import, 0.95),
        p95_compile_ms: percentile(&compile, 0.95),
        p95_loop_ms: percentile(&loop_ms, 0.95),
        p95_teardown_ms: percentile(&teardown, 0.95),
        p95_process_shutdown_ms: percentile(&process_shutdown, 0.95),
    }
}

fn average(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_by(|a, b| a.total_cmp(b));
    let idx = ((values.len() as f64 * q).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[idx]
}

#[derive(Clone, Copy)]
struct CpuSample {
    idle: u64,
    total: u64,
}

#[cfg(target_os = "linux")]
fn read_cpu_sample() -> Option<CpuSample> {
    let stat = fs::read_to_string("/proc/stat").ok()?;
    let line = stat.lines().find(|line| line.starts_with("cpu "))?;
    let values = line
        .split_whitespace()
        .skip(1)
        .filter_map(|value| value.parse::<u64>().ok())
        .collect::<Vec<_>>();
    if values.len() < 4 {
        return None;
    }
    let idle =
        values.get(3).copied().unwrap_or_default() + values.get(4).copied().unwrap_or_default();
    let total = values.iter().sum();
    Some(CpuSample { idle, total })
}

#[cfg(not(target_os = "linux"))]
fn read_cpu_sample() -> Option<CpuSample> {
    None
}

#[cfg(target_os = "linux")]
fn read_per_core_cpu_samples() -> Vec<CpuSample> {
    let Ok(stat) = fs::read_to_string("/proc/stat") else {
        return Vec::new();
    };
    stat.lines()
        .filter(|line| {
            line.starts_with("cpu")
                && line
                    .as_bytes()
                    .get(3)
                    .is_some_and(|byte| byte.is_ascii_digit())
        })
        .filter_map(parse_cpu_line)
        .collect()
}

#[cfg(not(target_os = "linux"))]
fn read_per_core_cpu_samples() -> Vec<CpuSample> {
    Vec::new()
}

#[cfg(target_os = "linux")]
fn parse_cpu_line(line: &str) -> Option<CpuSample> {
    let values = line
        .split_whitespace()
        .skip(1)
        .filter_map(|value| value.parse::<u64>().ok())
        .collect::<Vec<_>>();
    if values.len() < 4 {
        return None;
    }
    let idle =
        values.get(3).copied().unwrap_or_default() + values.get(4).copied().unwrap_or_default();
    let total = values.iter().sum();
    Some(CpuSample { idle, total })
}

fn cpu_percent(previous: CpuSample, current: CpuSample) -> f64 {
    let total = current.total.saturating_sub(previous.total);
    if total == 0 {
        return 0.0;
    }
    let idle = current.idle.saturating_sub(previous.idle);
    100.0 * (total.saturating_sub(idle)) as f64 / total as f64
}

fn per_core_cpu_percent(previous: &[CpuSample], current: &[CpuSample]) -> Vec<f64> {
    previous
        .iter()
        .zip(current)
        .map(|(previous, current)| cpu_percent(*previous, *current))
        .collect()
}

#[cfg(target_os = "linux")]
fn read_context_switches() -> Option<u64> {
    let stat = fs::read_to_string("/proc/stat").ok()?;
    stat.lines()
        .find_map(|line| line.strip_prefix("ctxt "))
        .and_then(|value| value.trim().parse().ok())
}

#[cfg(not(target_os = "linux"))]
fn read_context_switches() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn read_loadavg_1m() -> f64 {
    fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|text| text.split_whitespace().next()?.parse().ok())
        .unwrap_or_default()
}

#[cfg(not(target_os = "linux"))]
fn read_loadavg_1m() -> f64 {
    0.0
}

#[cfg(target_os = "linux")]
fn sample_run_processes(out_dir: &Path, elapsed_ms: u128) -> Vec<ProcessSample> {
    let prefix = out_dir.join("runs").to_string_lossy().to_string();
    let Ok(entries) = fs::read_dir("/proc") else {
        return Vec::new();
    };
    let mut samples = Vec::new();
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse().ok())
        else {
            continue;
        };
        let proc_dir = entry.path();
        let Ok(environ) = fs::read(proc_dir.join("environ")) else {
            continue;
        };
        let environ = String::from_utf8_lossy(&environ);
        let Some(context_path) = environ
            .split('\0')
            .find_map(|entry| entry.strip_prefix("ELODIN_MONTE_CARLO_CONTEXT="))
        else {
            continue;
        };
        if !context_path.starts_with(&prefix) {
            continue;
        }
        let run_id = context_path
            .split("/runs/")
            .nth(1)
            .and_then(|rest| rest.split('/').next())
            .unwrap_or("unknown")
            .to_string();
        let (utime_ticks, stime_ticks) = read_proc_stat(&proc_dir);
        let rss_kib = read_proc_rss_kib(&proc_dir);
        let cmd = fs::read(proc_dir.join("cmdline"))
            .map(|bytes| {
                String::from_utf8_lossy(&bytes)
                    .replace('\0', " ")
                    .trim()
                    .chars()
                    .take(240)
                    .collect::<String>()
            })
            .unwrap_or_default();
        samples.push(ProcessSample {
            elapsed_ms,
            run_id,
            pid,
            rss_kib,
            utime_ticks,
            stime_ticks,
            cmd,
        });
    }
    samples
}

#[cfg(not(target_os = "linux"))]
fn sample_run_processes(_out_dir: &Path, _elapsed_ms: u128) -> Vec<ProcessSample> {
    Vec::new()
}

#[cfg(target_os = "linux")]
fn read_proc_stat(proc_dir: &Path) -> (u64, u64) {
    let Ok(stat) = fs::read_to_string(proc_dir.join("stat")) else {
        return (0, 0);
    };
    let Some((_, rest)) = stat.rsplit_once(") ") else {
        return (0, 0);
    };
    let fields = rest.split_whitespace().collect::<Vec<_>>();
    let utime = fields
        .get(11)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let stime = fields
        .get(12)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    (utime, stime)
}

#[cfg(target_os = "linux")]
fn read_proc_rss_kib(proc_dir: &Path) -> u64 {
    let Ok(status) = fs::read_to_string(proc_dir.join("status")) else {
        return 0;
    };
    status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .map(parse_kib)
        .unwrap_or_default()
}

#[cfg(target_os = "linux")]
fn read_meminfo() -> (u64, u64) {
    let Ok(text) = fs::read_to_string("/proc/meminfo") else {
        return (0, 0);
    };
    let mut total = 0;
    let mut available = 0;
    for line in text.lines() {
        if line.starts_with("MemTotal:") {
            total = parse_kib(line);
        } else if line.starts_with("MemAvailable:") {
            available = parse_kib(line);
        }
    }
    (total, available)
}

#[cfg(not(target_os = "linux"))]
fn read_meminfo() -> (u64, u64) {
    (0, 0)
}

fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            match entry.metadata() {
                Ok(metadata) if metadata.is_dir() => dir_size(&path),
                Ok(metadata) => allocated_bytes(&metadata),
                Err(_) => 0,
            }
        })
        .sum()
}

#[cfg(unix)]
fn allocated_bytes(metadata: &fs::Metadata) -> u64 {
    metadata.blocks().saturating_mul(512)
}

#[cfg(not(unix))]
fn allocated_bytes(metadata: &fs::Metadata) -> u64 {
    metadata.len()
}

fn memory_score(samples: &[CacheMappingSample]) -> (usize, u64, u64) {
    (
        samples.len(),
        samples.iter().map(|sample| sample.virtual_kib).sum(),
        samples.iter().map(|sample| sample.rss_kib).sum(),
    )
}

#[cfg(target_os = "linux")]
fn parse_kib(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn parse_duration(value: Option<&str>) -> Result<Option<Duration>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_empty() {
        return Ok(None);
    }
    let (digits, unit) = value.split_at(
        value
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(value.len()),
    );
    let amount = digits.parse::<u64>().into_diagnostic()?;
    let duration = match unit {
        "" | "s" => Duration::from_secs(amount),
        "ms" => Duration::from_millis(amount),
        "m" => Duration::from_secs(amount * 60),
        "h" => Duration::from_secs(amount * 60 * 60),
        other => return Err(miette!("unsupported duration unit `{other}` in `{value}`")),
    };
    Ok(Some(duration))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phase(ns: &[u64]) -> SimPhaseSummary {
        let mut summary = SimPhaseSummary {
            sum_ns: 0,
            count: 0,
            min_ns: 0,
            max_ns: 0,
            buckets: [0; 32],
        };
        for &ns in ns {
            summary.sum_ns += ns;
            summary.count += 1;
            if summary.count == 1 {
                summary.min_ns = ns;
                summary.max_ns = ns;
            } else {
                summary.min_ns = summary.min_ns.min(ns);
                summary.max_ns = summary.max_ns.max(ns);
            }
            let bucket = if ns < 2 {
                0
            } else {
                (63usize - ns.leading_zeros() as usize).min(31)
            };
            summary.buckets[bucket] += 1;
        }
        summary
    }

    #[test]
    fn phase_merge_preserves_counts_min_max_and_mean() {
        let mut a = phase(&[1_000, 2_000]);
        a.merge(phase(&[4_000, 8_000]));
        assert_eq!(a.count, 4);
        assert_eq!(a.min_ns, 1_000);
        assert_eq!(a.max_ns, 8_000);
        assert_eq!(a.mean_ns(), 3_750);
    }

    #[test]
    fn phase_percentile_uses_log2_bucket_bounds() {
        let p = phase(&(1024..1124).collect::<Vec<_>>());
        let p50 = p.percentile_ns(0.5);
        assert!((1024..=2048).contains(&p50));
        assert_eq!(p.percentile_ns(0.0), p.min_ns);
        assert_eq!(p.percentile_ns(1.0), p.max_ns);
    }

    #[test]
    fn skipped_runs_are_unassigned_not_a_phantom_worker() {
        let out_dir = Path::new("/tmp/mc-test");
        let metric = placeholder_metric("run_0000000", None, out_dir, "skipped");
        assert_eq!(metric.status, "skipped");
        assert_eq!(metric.worker_id, None);
        assert!(!metric.exit_ok);
    }

    #[test]
    fn setup_failures_retain_their_worker_slot() {
        let out_dir = Path::new("/tmp/mc-test");
        let metric = placeholder_metric("run_0000000", Some(3), out_dir, "failed");
        assert_eq!(metric.status, "failed");
        assert_eq!(metric.worker_id, Some(3));
    }

    #[test]
    fn passed_respects_post_run_hook_verdict() {
        let out_dir = Path::new("/tmp/mc-test");
        let mut metric = placeholder_metric("run_0000000", Some(0), out_dir, "ok");
        metric.exit_ok = true;

        // Clean exit, no hook outcome -> judged on exit_ok alone.
        metric.scored_pass = None;
        assert!(metric.passed());

        // Clean exit but the scoring hook failed the criteria -> not a pass.
        metric.scored_pass = Some(false);
        assert!(!metric.passed());

        // Clean exit and the hook passed -> pass.
        metric.scored_pass = Some(true);
        assert!(metric.passed());

        // A hard failure never passes, even if a stale hook verdict says true.
        metric.exit_ok = false;
        metric.scored_pass = Some(true);
        assert!(!metric.passed());
    }

    #[test]
    fn invalid_runs_count_toward_fail_on_errors() {
        let out_dir = std::env::temp_dir().join(format!("mc-invalid-{}", unix_now_ns()));
        fs::create_dir_all(&out_dir).unwrap();

        let mut passed = placeholder_metric("run_0000000", Some(0), &out_dir, "ok");
        passed.exit_ok = true;
        passed.scored_pass = Some(true);
        passed.scored_valid = Some(true);

        let mut invalid = placeholder_metric("run_0000001", Some(0), &out_dir, "ok");
        invalid.exit_ok = true;
        invalid.scored_pass = Some(true);
        invalid.scored_valid = Some(false);

        let metrics = vec![passed, invalid];
        let (_, concurrency_summary) = build_timeline(&metrics);
        let now = Utc::now();
        let summary = summarize_campaign(
            &out_dir,
            &metrics,
            &[],
            now,
            now,
            0,
            1,
            &concurrency_summary,
        );

        assert_eq!(summary.failed, 0);
        assert_eq!(summary.invalid, 1);
        assert_eq!(summary.error_runs(), 1);

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn read_post_run_outcome_extracts_hook_fields() {
        let dir = std::env::temp_dir().join(format!("mc-post-run-{}", unix_now_ns()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("post_run_result.json");

        fs::write(&path, r#"{"pass": false, "traj_rmse_m": 12.5}"#).unwrap();
        let outcome = read_post_run_outcome(&path);
        assert_eq!(outcome.pass, Some(false));
        assert_eq!(outcome.valid, None);
        assert_eq!(
            outcome.scalars.get("traj_rmse_m"),
            Some(&"12.5".to_string())
        );

        fs::write(&path, r#"{"pass": true, "valid": false}"#).unwrap();
        let outcome = read_post_run_outcome(&path);
        assert_eq!(outcome.pass, Some(true));
        assert_eq!(outcome.valid, Some(false));

        // No `pass` field -> fall back to exit-based judgement.
        fs::write(&path, r#"{"traj_rmse_m": 1.0}"#).unwrap();
        let outcome = read_post_run_outcome(&path);
        assert_eq!(outcome.pass, None);
        assert_eq!(outcome.scalars.get("traj_rmse_m"), Some(&"1.0".to_string()));

        // Missing file -> empty outcome.
        let outcome = read_post_run_outcome(&dir.join("missing.json"));
        assert_eq!(outcome.pass, None);
        assert!(outcome.scalars.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn perf_metrics_round_trip_through_csv() {
        let dir = std::env::temp_dir().join(format!("mc-perf-{}", unix_now_ns()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("perf.csv");

        let mut metric = placeholder_metric("run_0000001", Some(2), &dir, "ok");
        metric.exit_ok = true;
        metric.scored_pass = Some(true);
        metric.wall_ms = 1234;
        write_perf_csv(&path, std::slice::from_ref(&metric)).unwrap();

        let loaded = read_perf_metrics(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].run_id, "run_0000001");
        assert_eq!(loaded[0].worker_id, Some(2));
        assert_eq!(loaded[0].wall_ms, 1234);
        assert!(loaded[0].passed());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebuild_report_recovers_hook_scalars() {
        let out_dir = std::env::temp_dir().join(format!("mc-rebuild-{}", unix_now_ns()));
        let run_dir = out_dir.join("runs").join("run_0000000");
        fs::create_dir_all(&run_dir).unwrap();

        let mut metric = placeholder_metric("run_0000000", Some(0), &out_dir, "ok");
        metric.exit_ok = true;
        metric.scored_pass = Some(true);
        metric.scored_valid = Some(true);
        metric
            .hook_scalars
            .insert("traj_rmse_m".to_string(), "12.5".to_string());

        write_perf_csv(&out_dir.join("perf.csv"), std::slice::from_ref(&metric)).unwrap();
        write_json(&run_dir.join("metrics.json"), &metric).unwrap();

        rebuild_report(&out_dir).unwrap();

        let results = fs::read_to_string(out_dir.join("results.csv")).unwrap();
        assert!(results.contains("traj_rmse_m"));
        assert!(results.contains("12.5"));

        let summary: CampaignSummary = read_json(&out_dir.join("summary.json")).unwrap();
        let rmse = summary
            .hook_metrics
            .get("traj_rmse_m")
            .expect("traj_rmse_m hook metric");
        assert_eq!(rmse.count, 1);
        assert_eq!(rmse.mean, 12.5);

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn results_csv_includes_passed_and_scored_pass() {
        let dir = std::env::temp_dir().join(format!("mc-results-{}", unix_now_ns()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("results.csv");

        let mut metric = placeholder_metric("run_0000000", Some(0), &dir, "ok");
        metric.exit_ok = true;
        metric.scored_pass = Some(false);
        write_results_csv(&path, &dir, std::slice::from_ref(&metric)).unwrap();

        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("passed"));
        assert!(text.contains("valid"));
        assert!(text.contains("scored_pass"));
        assert!(text.contains("run_0000000,ok,false,true,false,"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn results_csv_does_not_duplicate_reserved_hook_columns() {
        let dir = std::env::temp_dir().join(format!("mc-results-reserved-{}", unix_now_ns()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("results.csv");

        let mut metric = placeholder_metric("run_0000000", Some(0), &dir, "ok");
        metric.exit_ok = true;
        metric.scored_pass = Some(true);
        metric.scored_valid = Some(true);
        // Scoring hooks echo `pass`/`valid` into their scalar map; those must not
        // become duplicate columns alongside the dedicated `valid`/`scored_*` ones.
        metric
            .hook_scalars
            .insert("valid".to_string(), "true".to_string());
        metric
            .hook_scalars
            .insert("pass".to_string(), "true".to_string());
        metric
            .hook_scalars
            .insert("traj_rmse_m".to_string(), "12.5".to_string());
        write_results_csv(&path, &dir, std::slice::from_ref(&metric)).unwrap();

        let text = fs::read_to_string(&path).unwrap();
        let header = text.lines().next().expect("header row");
        let columns: Vec<&str> = header.split(',').collect();
        assert_eq!(columns.iter().filter(|col| **col == "valid").count(), 1);
        assert_eq!(columns.iter().filter(|col| **col == "pass").count(), 0);
        assert_eq!(
            columns.iter().filter(|col| **col == "traj_rmse_m").count(),
            1
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_existing_metric_only_returns_recorded_runs() {
        let out_dir = std::env::temp_dir().join(format!("mc-load-{}", unix_now_ns()));
        let run_dir = out_dir.join("runs").join("run_0000000");
        fs::create_dir_all(&run_dir).unwrap();

        // No metrics.json yet -> nothing to preserve.
        assert!(load_existing_metric(&out_dir, "run_0000000").is_none());

        let mut metric = placeholder_metric("run_0000000", Some(0), &out_dir, "ok");
        metric.exit_ok = true;
        write_json(&run_dir.join("metrics.json"), &metric).unwrap();

        let loaded = load_existing_metric(&out_dir, "run_0000000").expect("metric");
        assert!(loaded.passed());
        // Paths are reconstructed relative to the campaign dir.
        assert_eq!(loaded.run_dir, run_dir);
        assert_eq!(loaded.db_path, run_dir.join("db"));

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn worker_label_marks_unassigned_runs() {
        assert_eq!(worker_label(Some(0)), "0");
        assert_eq!(worker_label(Some(7)), "7");
        assert_eq!(worker_label(None), "-");
    }

    #[test]
    fn build_config_parses_from_toml() {
        let config: CampaignConfig = toml::from_str(
            r#"
            [[build]]
            command = "cargo"
            args = ["build", "--release"]
            cwd = "."
            [build.env]
            FOO = "bar"

            [[build]]
            command = "make"
            "#,
        )
        .expect("parse campaign config");
        assert_eq!(config.build.len(), 2);
        let build = &config.build[0];
        assert_eq!(build.command.as_deref(), Some("cargo"));
        assert_eq!(build.args, ["build", "--release"]);
        assert_eq!(build.cwd.as_deref(), Some(Path::new(".")));
        assert_eq!(build.env.get("FOO").map(String::as_str), Some("bar"));
        assert_eq!(config.build[1].command.as_deref(), Some("make"));
    }

    #[test]
    fn campaign_config_parses_new_keys() {
        let config: CampaignConfig = toml::from_str(
            r#"
            workers = 96
            scratch_dir = "auto"

            [env]
            SITL = "1"

            [resources]
            db_port = 20000
            port_stride = 32
            [resources.ports]
            sitl_socket = 20002
            weaver = "auto"

            [retention]
            keep_run_db = "on-fail"
            prune_on_pass = ["postprocess/merged"]

            [quality]
            max_behind_deadline_frac = 0.05
            "#,
        )
        .expect("parse campaign config");
        assert_eq!(config.workers, Some(96));
        assert_eq!(config.scratch_dir.as_deref(), Some("auto"));
        assert_eq!(config.env.get("SITL").map(String::as_str), Some("1"));
        assert_eq!(config.resources.db_port, PortSpec::Static(20000));
        assert_eq!(
            config.resources.ports.get("sitl_socket"),
            Some(&PortSpec::Static(20002))
        );
        assert_eq!(
            config.resources.ports.get("weaver"),
            Some(&PortSpec::Auto(AutoTag::Auto))
        );
        assert_eq!(config.retention.prune_on_pass, ["postprocess/merged"]);
        assert!(config.retention.compact_run_db);
        assert_eq!(config.quality.max_behind_deadline_frac, Some(0.05));
        assert!(!config.quality.fail_on_degraded);
    }

    #[test]
    fn degraded_runs_are_counted_separately() {
        let out_dir = std::env::temp_dir().join(format!("mc-degraded-{}", unix_now_ns()));
        fs::create_dir_all(&out_dir).unwrap();

        let mut degraded = placeholder_metric("run_0000000", Some(0), &out_dir, "degraded");
        degraded.exit_ok = true;
        degraded.degraded = true;
        degraded.behind_deadline_frac = Some(0.4);
        assert!(!degraded.passed());
        assert!(degraded.valid());

        let metrics = vec![degraded];
        let (_, concurrency) = build_timeline(&metrics);
        let now = Utc::now();
        let summary = summarize_campaign(&out_dir, &metrics, &[], now, now, 0, 1, &concurrency);
        assert_eq!(summary.degraded, 1);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.passed, 0);
        // Degraded runs don't trip fail_on_run_errors by default.
        assert_eq!(summary.error_runs(), 0);
        let pacing = summary.pacing.expect("pacing summary");
        assert_eq!(pacing.max_behind_run, "run_0000000");

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn quality_gate_flags_violations() {
        let quality = QualityConfig {
            max_behind_deadline_frac: Some(0.05),
            max_real_time_factor: None,
            fail_on_degraded: false,
        };
        let out = std::env::temp_dir();
        let mut metric = placeholder_metric("run_0000000", Some(0), &out, "ok");
        metric.behind_deadline_frac = Some(0.2);
        let reason = quality_violation(&quality, &metric).expect("violation");
        assert!(reason.contains("behind deadline"), "got: {reason}");
        metric.behind_deadline_frac = Some(0.01);
        assert!(quality_violation(&quality, &metric).is_none());
    }

    #[test]
    fn extract_pacing_computes_rtf_and_frac() {
        let mut summary = SimRunSummary {
            pre_step: phase(&[]),
            copy_db_to_world: phase(&[]),
            world_run: phase(&[]),
            commit: phase(&[]),
            wait_for_write: phase(&[]),
            post_step: phase(&[]),
            real_time_pacing: phase(&[1_000; 10]),
            total_cycle: phase(&[1_000; 10]),
            real_time_oversleep: None,
            real_time_behind: None,
            pacing_behind_events: 5,
            pacing_resets: 2,
            pacing_no_sleep: 0,
            cycles: 100,
            steps: 1_000,
            simulation_rate_hz: 100.0,
            telemetry_rate_hz: 100.0,
            // 20 s wall for 10 s of sim time -> rtf 2.0.
            wall_ns: 20_000_000_000,
            entry_unix_ns: None,
            compile_done_unix_ns: None,
            loop_start_unix_ns: None,
            loop_end_unix_ns: None,
            summary_written_unix_ns: None,
        };
        let pacing = extract_pacing(&summary);
        assert_eq!(pacing.behind_deadline_frac, Some(0.05));
        assert_eq!(pacing.drift_resets, Some(2));
        assert!((pacing.real_time_factor.unwrap() - 2.0).abs() < 1e-9);

        // Unpaced sims produce no pacing metrics.
        summary.real_time_pacing = phase(&[]);
        let pacing = extract_pacing(&summary);
        assert!(pacing.behind_deadline_frac.is_none());
        assert!(pacing.real_time_factor.is_none());
    }

    #[test]
    fn prune_globs_remove_only_within_run_dir() {
        let run_dir = std::env::temp_dir().join(format!("mc-prune-{}", unix_now_ns()));
        fs::create_dir_all(run_dir.join("postprocess/merged")).unwrap();
        fs::write(run_dir.join("postprocess/merged/big.csv"), b"x").unwrap();
        fs::write(run_dir.join("keep.txt"), b"x").unwrap();

        let mut metric = placeholder_metric("run_0000000", Some(0), &run_dir, "ok");
        metric.exit_ok = true;
        metric.run_dir = run_dir.clone();
        let retention = RetentionConfig {
            prune_on_pass: vec!["postprocess/merged".to_string()],
            ..RetentionConfig::default()
        };
        apply_prune_globs(&retention, &metric);
        assert!(!run_dir.join("postprocess/merged").exists());
        assert!(run_dir.join("keep.txt").exists());

        let _ = fs::remove_dir_all(&run_dir);
    }

    #[test]
    fn prune_globs_reject_parent_dir_patterns() {
        let root = std::env::temp_dir().join(format!("mc-prune-traversal-{}", unix_now_ns()));
        let run_dir = root.join("run");
        let sibling = root.join("sibling");
        fs::create_dir_all(&run_dir).unwrap();
        fs::create_dir_all(&sibling).unwrap();
        fs::write(sibling.join("keep.txt"), b"x").unwrap();

        let mut metric = placeholder_metric("run_0000000", Some(0), &run_dir, "ok");
        metric.exit_ok = true;
        metric.run_dir = run_dir.clone();
        let retention = RetentionConfig {
            prune_on_pass: vec!["../sibling".to_string()],
            ..RetentionConfig::default()
        };
        apply_prune_globs(&retention, &metric);
        assert!(sibling.join("keep.txt").exists());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn dir_has_files_ignores_empty_skeletons() {
        let root = std::env::temp_dir().join(format!("mc-scratch-{}", unix_now_ns()));
        fs::create_dir_all(root.join("runs/run_0000000")).unwrap();
        // Only empty directories left behind -> safe to delete.
        assert!(!dir_has_files(&root));
        // A stranded artifact anywhere in the tree preserves it.
        fs::write(root.join("runs/run_0000000/metrics.json"), b"{}").unwrap();
        assert!(dir_has_files(&root));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn apply_overrides_persists_explicit_workers_for_resume() {
        let mut config = CampaignConfig {
            workers: Some(8),
            ..CampaignConfig::default()
        };
        let mut options = RunOptions {
            sim_path: PathBuf::from("sim.py"),
            plan_path: PathBuf::from("plan.csv"),
            campaign_path: None,
            out_dir: PathBuf::from("out"),
            cache_dir: None,
            workers: Some(96),
            scratch_dir: None,
            retries: None,
            timeout: None,
            post_run: None,
            post_campaign: None,
            fail_fast: false,
            fail_on_run_errors: false,
            dry_run: false,
            memory_probe: false,
            keep_existing: false,
            clean: false,
            strict_ports: false,
        };

        apply_overrides(&mut config, &mut options);
        assert_eq!(config.workers, Some(96));
    }

    #[test]
    fn requested_workers_precedence() {
        // Flag always wins.
        let config = CampaignConfig {
            workers: Some(8),
            ..CampaignConfig::default()
        };
        assert_eq!(requested_workers(Some(96), &config), Some(96));
        // Without flag or env, campaign.toml applies. (The env branch is
        // process-global, so it is not exercised here.)
        if std::env::var_os("S10_MAX_INFLIGHT").is_none() {
            assert_eq!(requested_workers(None, &config), Some(8));
        }
    }

    #[test]
    fn rampup_staggers_first_wave_only() {
        // Small pools and worker 0 start immediately.
        assert_eq!(rampup_delay("auto", 3, 4), None);
        assert_eq!(rampup_delay("auto", 0, 96), None);
        // Later workers stagger, capped at one minute.
        assert_eq!(
            rampup_delay("auto", 10, 96),
            Some(Duration::from_millis(5000))
        );
        assert_eq!(
            rampup_delay("auto", 95, 200),
            Some(Duration::from_millis(47_500))
        );
        assert_eq!(
            rampup_delay("auto", 150, 200),
            Some(Duration::from_secs(60))
        );
        assert_eq!(rampup_delay("off", 50, 96), None);
    }

    #[cfg(unix)]
    #[test]
    fn campaign_lock_is_exclusive_per_out_dir() {
        let out_dir = std::env::temp_dir().join(format!("mc-lock-{}", unix_now_ns()));
        fs::create_dir_all(&out_dir).unwrap();
        let lock = acquire_campaign_lock(&out_dir).expect("first lock");
        let Err(err) = acquire_campaign_lock(&out_dir) else {
            panic!("second lock must fail");
        };
        assert!(err.to_string().contains("in use"), "got: {err}");
        drop(lock);
        acquire_campaign_lock(&out_dir).expect("lock re-acquirable after release");
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn error_chain_joins_contexts() {
        let err = miette::Report::msg("io error")
            .wrap_err("spawn fsw")
            .wrap_err("run_0000012");
        let chain = error_chain(&err);
        assert_eq!(chain, "run_0000012: spawn fsw: io error");
    }

    #[test]
    fn campaign_config_defaults_and_ci_gate_flag() {
        let config: CampaignConfig = toml::from_str("").expect("parse empty campaign config");
        assert!(config.continue_on_error);
        assert!(!config.fail_on_run_errors);

        let ci: CampaignConfig = toml::from_str(
            r#"
            continue_on_error = true
            fail_on_run_errors = true
            "#,
        )
        .expect("parse ci campaign config");
        assert!(ci.fail_on_run_errors);
    }

    #[test]
    fn resolve_auto_workers_uses_admission_budget() {
        // S10_MAX_INFLIGHT / recipe_weight, clamped to the plan size.
        assert_eq!(resolve_auto_workers(100, 2, Some(8)), 4);
        assert_eq!(resolve_auto_workers(100, 9, Some(8)), 1);
        assert_eq!(resolve_auto_workers(3, 2, Some(8)), 3);
        // Admission disabled falls back to logical cores (clamped to the plan).
        assert_eq!(
            resolve_auto_workers(100, 2, None),
            available_cpus().max(1).min(100)
        );
    }

    #[test]
    fn resolve_runtime_threads_track_admission_budget() {
        // An explicit override always wins.
        assert_eq!(resolve_runtime_threads(Some(4), 100, Some(6)), 6);
        let cores = available_cpus();
        if cores > 1 {
            // A large budget scales threads but never exceeds logical cores.
            assert_eq!(
                resolve_runtime_threads(Some(1000), 100, None),
                cores.min(100).max(2)
            );
            // Never spins up more threads than there are runs (floor of 2).
            assert_eq!(resolve_runtime_threads(Some(1000), 1, None), 2);
        }
    }

    #[cfg(unix)]
    #[test]
    fn build_steps_run_in_order() {
        let out_dir = std::env::temp_dir().join(format!(
            "mc-build-step-{}-{}",
            std::process::id(),
            unix_now_ns()
        ));
        fs::create_dir_all(&out_dir).expect("create temp dir");
        let first = out_dir.join("first");
        let second = out_dir.join("second");
        let steps = vec![
            BuildConfig {
                command: Some("touch".to_string()),
                args: vec![first.to_string_lossy().to_string()],
                cwd: None,
                env: HashMap::new(),
            },
            BuildConfig {
                command: Some("touch".to_string()),
                args: vec![second.to_string_lossy().to_string()],
                cwd: None,
                env: HashMap::new(),
            },
        ];
        run_build_steps(&steps, &out_dir).expect("build steps");
        assert!(first.exists());
        assert!(second.exists());
        let _ = fs::remove_dir_all(out_dir);
    }

    #[cfg(unix)]
    #[test]
    fn build_step_failure_is_error() {
        let out_dir = std::env::temp_dir().join(format!(
            "mc-build-step-fail-{}-{}",
            std::process::id(),
            unix_now_ns()
        ));
        fs::create_dir_all(&out_dir).expect("create temp dir");
        let steps = vec![BuildConfig {
            command: Some("false".to_string()),
            args: Vec::new(),
            cwd: None,
            env: HashMap::new(),
        }];
        assert!(run_build_steps(&steps, &out_dir).is_err());
        let _ = fs::remove_dir_all(out_dir);
    }
}
