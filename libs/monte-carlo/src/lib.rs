use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{IsTerminal, Write};
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use miette::{Context, IntoDiagnostic, Result, miette};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use stellarator::util::CancelToken;
use sysinfo::{Pid, ProcessesToUpdate, Signal, System};
use tokio::task::JoinSet;

pub const CONTEXT_ENV: &str = "ELODIN_MONTE_CARLO_CONTEXT";
pub const CACHE_ENV: &str = "ELODIN_CACHE_DIR";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CampaignConfig {
    pub workers: Option<usize>,
    pub timeout: Option<String>,
    pub retries: usize,
    pub cache_dir: Option<PathBuf>,
    pub resources: ResourceConfig,
    pub hooks: HookConfig,
    pub params_compat: Option<String>,
    pub continue_on_error: bool,
}

impl Default for CampaignConfig {
    fn default() -> Self {
        Self {
            workers: None,
            timeout: None,
            retries: 0,
            cache_dir: None,
            resources: ResourceConfig::default(),
            hooks: HookConfig::default(),
            params_compat: None,
            continue_on_error: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ResourceConfig {
    pub bind_ip: IpAddr,
    pub port_stride: u16,
    pub db_port: u16,
    pub state_port: u16,
    pub command_port: u16,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            bind_ip: IpAddr::V6(Ipv6Addr::UNSPECIFIED),
            port_stride: 20,
            db_port: 2240,
            state_port: 9003,
            command_port: 9002,
        }
    }
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
    pub workers: Option<usize>,
    pub cache_dir: Option<PathBuf>,
    pub retries: Option<usize>,
    pub timeout: Option<String>,
    pub post_run: Option<PathBuf>,
    pub post_campaign: Option<PathBuf>,
    pub params_compat: Option<String>,
    pub fail_fast: bool,
    pub dry_run: bool,
    pub progress: ProgressMode,
    pub memory_probe: bool,
    pub keep_existing: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProgressMode {
    #[default]
    Auto,
    Always,
    Never,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedRunShape {
    pub workers: usize,
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
    pub state_port: u16,
    pub command_port: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetric {
    pub run_id: String,
    pub worker_id: usize,
    pub attempt: usize,
    pub status: String,
    pub exit_ok: bool,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignSummary {
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub total_runs: usize,
    pub passed: usize,
    pub failed: usize,
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
    #[error("run {0} failed")]
    RunFailed(String),
    #[error("unsupported params compatibility mode: {0}")]
    UnsupportedCompat(String),
}

#[derive(Clone)]
struct CampaignReporter {
    progress: Option<ProgressBar>,
    log_file: Arc<Mutex<fs::File>>,
    ok: Arc<AtomicUsize>,
    failed: Arc<AtomicUsize>,
    skipped: Arc<AtomicUsize>,
}

impl CampaignReporter {
    fn new(out_dir: &Path, total_runs: usize, workers: usize, mode: ProgressMode) -> Result<Self> {
        let log_path = out_dir.join("campaign.log");
        let log_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .into_diagnostic()
            .with_context(|| format!("open {}", log_path.display()))?;
        let progress_enabled = match mode {
            ProgressMode::Always => true,
            ProgressMode::Never => false,
            ProgressMode::Auto => std::io::stderr().is_terminal(),
        };
        let progress = progress_enabled.then(|| {
            let bar = ProgressBar::new(total_runs as u64);
            bar.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ok={msg}",
                )
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("#>-"),
            );
            bar.set_message(format!("0 fail=0 workers={workers}"));
            bar
        });
        let reporter = Self {
            progress,
            log_file: Arc::new(Mutex::new(log_file)),
            ok: Arc::new(AtomicUsize::new(0)),
            failed: Arc::new(AtomicUsize::new(0)),
            skipped: Arc::new(AtomicUsize::new(0)),
        };
        reporter.line(format!(
            "starting monte-carlo campaign: runs={total_runs} workers={workers} out_dir={}",
            out_dir.display()
        ));
        Ok(reporter)
    }

    fn line(&self, line: impl AsRef<str>) {
        let line = line.as_ref();
        if let Some(progress) = &self.progress {
            progress.suspend(|| eprintln!("{line}"));
        } else {
            eprintln!("{line}");
        }
        self.log_only(line);
    }

    fn log_only(&self, line: &str) {
        if let Ok(mut file) = self.log_file.lock() {
            let _ = writeln!(file, "{line}");
        }
    }

    fn record(&self, metric: &RunMetric) {
        if metric.status == "skipped" {
            self.skipped.fetch_add(1, Ordering::SeqCst);
        } else if metric.exit_ok {
            self.ok.fetch_add(1, Ordering::SeqCst);
        } else {
            self.failed.fetch_add(1, Ordering::SeqCst);
        }
        let ok = self.ok.load(Ordering::SeqCst);
        let failed = self.failed.load(Ordering::SeqCst);
        let skipped = self.skipped.load(Ordering::SeqCst);
        if let Some(progress) = &self.progress {
            progress.inc(1);
            progress.set_message(format!(
                "{ok} fail={failed} skip={skipped} worker={}",
                metric.worker_id
            ));
            let line = format!(
                "[{}] {} worker={} wall_ms={} log={}",
                metric.status,
                metric.run_id,
                metric.worker_id,
                metric.wall_ms,
                metric.run_dir.join("logs").display()
            );
            self.log_only(&line);
            if !metric.exit_ok {
                progress.suspend(|| eprintln!("{line}"));
            }
        } else {
            self.line(format!(
                "[{}] {} worker={} wall_ms={} log={}",
                metric.status,
                metric.run_id,
                metric.worker_id,
                metric.wall_ms,
                metric.run_dir.join("logs").display()
            ));
        }
    }

    fn finish(&self) {
        if let Some(progress) = &self.progress {
            progress.finish_and_clear();
        }
        self.line(format!(
            "finished monte-carlo campaign: ok={} failed={} skipped={}",
            self.ok.load(Ordering::SeqCst),
            self.failed.load(Ordering::SeqCst),
            self.skipped.load(Ordering::SeqCst)
        ));
    }
}

fn reap_existing_elodin(reporter: &CampaignReporter) {
    let current_pid = Pid::from_u32(std::process::id());
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::All, true);
    let protected_pids = current_process_ancestry(&system, current_pid);
    let current_start_time = system
        .process(current_pid)
        .map(|process| process.start_time())
        .unwrap_or_default();
    let targets = system
        .processes()
        .iter()
        .filter_map(|(pid, process)| {
            if protected_pids.contains(pid)
                || process_cmd_contains(process, "monte-carlo")
                || (process.start_time() >= current_start_time
                    && has_protected_ancestor(&system, *pid, &protected_pids))
            {
                return None;
            }
            let name = elodin_process_name(process)?;
            matches!(name.as_str(), "elodin" | "elodin-db").then_some((*pid, name))
        })
        .collect::<Vec<_>>();

    if targets.is_empty() {
        return;
    }

    reporter.line(format!(
        "reaping {} existing elodin process(es) before monte-carlo campaign",
        targets.len()
    ));
    reporter.log_only(&format!(
        "protected current process ancestry: {:?}",
        protected_pids
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    ));

    for (pid, name) in &targets {
        reporter.log_only(&format!("reaping existing {name} process {pid}"));
        match system
            .process(*pid)
            .and_then(|process| process.kill_with(Signal::Term))
        {
            Some(true) => reporter.log_only(&format!("reaped {pid} {name} with SIGTERM")),
            Some(false) | None => reporter.log_only(&format!(
                "warning: failed to send SIGTERM to existing {name} process {pid}"
            )),
        }
    }

    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        system.refresh_processes(ProcessesToUpdate::All, true);
        if targets
            .iter()
            .all(|(pid, _)| system.process(*pid).is_none())
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    system.refresh_processes(ProcessesToUpdate::All, true);
    for (pid, name) in &targets {
        if let Some(process) = system.process(*pid) {
            if process.kill() {
                reporter.log_only(&format!("force-killed lingering {name} process {pid}"));
            } else {
                reporter.log_only(&format!(
                    "warning: failed to force-kill lingering {name} process {pid}"
                ));
            }
        }
    }
}

fn current_process_ancestry(system: &System, current_pid: Pid) -> HashSet<Pid> {
    let mut protected = HashSet::from([current_pid]);
    let mut pid = current_pid;
    while let Some(parent) = system.process(pid).and_then(|process| process.parent()) {
        if !protected.insert(parent) {
            break;
        }
        pid = parent;
    }
    protected
}

fn has_protected_ancestor(system: &System, mut pid: Pid, protected: &HashSet<Pid>) -> bool {
    loop {
        if protected.contains(&pid) {
            return true;
        }
        let Some(parent) = system.process(pid).and_then(|process| process.parent()) else {
            return false;
        };
        if parent == pid {
            return false;
        }
        pid = parent;
    }
}

fn process_cmd_contains(process: &sysinfo::Process, needle: &str) -> bool {
    process
        .cmd()
        .iter()
        .any(|arg| arg.to_string_lossy() == needle)
}

fn elodin_process_name(process: &sysinfo::Process) -> Option<String> {
    process
        .exe()
        .and_then(|path| path.file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .or_else(|| Some(process.name().to_string_lossy().into_owned()))
}

pub async fn run_campaign(mut options: RunOptions) -> Result<()> {
    let started_at = Utc::now();
    let wall_start = Instant::now();
    let mut config = load_config(options.campaign_path.as_deref())?;
    apply_overrides(&mut config, &mut options);

    let plan = read_plan(&options.plan_path)?;
    if plan.is_empty() {
        return Err(Error::EmptyPlan).into_diagnostic();
    }
    let workers = resolve_workers(config.workers, plan.len());
    config.workers = Some(workers);

    if options.dry_run {
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
    let cache_dir = config
        .cache_dir
        .clone()
        .unwrap_or_else(|| options.out_dir.join("const-cache"));
    fs::create_dir_all(&cache_dir)
        .into_diagnostic()
        .with_context(|| format!("create cache dir {}", cache_dir.display()))?;
    let plan_dest = options.out_dir.join("plan.csv");
    if !same_file_or_path(&options.plan_path, &plan_dest) {
        fs::copy(&options.plan_path, &plan_dest).into_diagnostic()?;
    }
    write_json(&options.out_dir.join("campaign.resolved.json"), &config)?;
    let reporter = CampaignReporter::new(&options.out_dir, plan.len(), workers, options.progress)?;
    if !options.keep_existing {
        reap_existing_elodin(&reporter);
    }

    let base_recipe = plan_recipe(&options.sim_path, &options.out_dir).await?;
    let rows = Arc::new(Mutex::new(VecDeque::from(plan)));
    let metrics = Arc::new(Mutex::new(Vec::<RunMetric>::new()));
    let failure = Arc::new(Mutex::new(None::<String>));
    let stop_sampling = Arc::new(AtomicBool::new(false));
    let peak_memory = Arc::new(Mutex::new(Vec::<CacheMappingSample>::new()));
    let resource_samples = Arc::new(Mutex::new(Vec::<ResourceSample>::new()));
    let process_samples = Arc::new(Mutex::new(Vec::<ProcessSample>::new()));
    let memory_sampler = options.memory_probe.then(|| {
        spawn_memory_sampler(
            cache_dir.clone(),
            stop_sampling.clone(),
            peak_memory.clone(),
        )
    });
    let resource_sampler = spawn_resource_sampler(
        options.out_dir.clone(),
        wall_start,
        stop_sampling.clone(),
        resource_samples.clone(),
        options.memory_probe.then_some(process_samples.clone()),
    );
    let mut worker_tasks = JoinSet::new();

    for worker_id in 0..workers {
        let rows = rows.clone();
        let metrics = metrics.clone();
        let failure = failure.clone();
        let base_recipe = base_recipe.clone();
        let config = config.clone();
        let out_dir = options.out_dir.clone();
        let cache_dir = cache_dir.clone();
        let reporter = reporter.clone();
        worker_tasks.spawn(async move {
            let slot = resource_slot(worker_id, &config.resources)?;
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
                let metric = match run_with_retries(
                    row,
                    worker_id,
                    slot.clone(),
                    base_recipe.clone(),
                    &config,
                    &out_dir,
                    &cache_dir,
                )
                .await
                {
                    Ok(metric) => metric,
                    Err(err) => {
                        let metric = placeholder_metric(&run_id, worker_id, &out_dir, "failed");
                        reporter.line(format!(
                            "[failed] {run_id} worker={worker_id} setup_error={err}"
                        ));
                        *failure.lock().expect("failure mutex poisoned") = Some(run_id);
                        reporter.record(&metric);
                        metrics.lock().expect("metrics mutex poisoned").push(metric);
                        continue;
                    }
                };
                let failed = !metric.exit_ok;
                if failed {
                    *failure.lock().expect("failure mutex poisoned") = Some(metric.run_id.clone());
                }
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
        let metric = placeholder_metric(&row.run_id, workers, &options.out_dir, "skipped");
        reporter.record(&metric);
        metrics.lock().expect("metrics mutex poisoned").push(metric);
    }
    stop_sampling.store(true, Ordering::SeqCst);
    if let Some(memory_sampler) = memory_sampler {
        memory_sampler.await.into_diagnostic()?;
    }
    resource_sampler.await.into_diagnostic()?;

    let metrics = metrics.lock().expect("metrics mutex poisoned").clone();
    write_perf_csv(&options.out_dir.join("perf.csv"), &metrics)?;
    let (timeline, concurrency_summary) = build_timeline(&metrics);
    write_timeline_csv(&options.out_dir.join("timeline.csv"), &timeline)?;
    write_results_csv(
        &options.out_dir.join("results.csv"),
        &options.out_dir,
        &metrics,
    )?;
    let memory = if options.memory_probe {
        peak_memory.lock().expect("memory mutex poisoned").clone()
    } else {
        Vec::new()
    };
    let resource_samples = resource_samples
        .lock()
        .expect("resource samples mutex poisoned")
        .clone();
    write_resource_csv(&options.out_dir.join("resources.csv"), &resource_samples)?;
    if options.memory_probe {
        write_json(&options.out_dir.join("memory.json"), &memory)?;
        write_process_csv(
            &options.out_dir.join("processes.csv"),
            &process_samples
                .lock()
                .expect("process samples mutex poisoned")
                .clone(),
        )?;
    }

    let passed = metrics.iter().filter(|metric| metric.exit_ok).count();
    let failed = metrics
        .iter()
        .filter(|metric| metric.status == "failed")
        .count();
    let skipped = metrics
        .iter()
        .filter(|metric| metric.status == "skipped")
        .count();
    let sim_phase_summary = aggregate_sim_summaries(&metrics);
    let phase_attribution = summarize_phase_attribution(&metrics);
    let total_run_wall_ms = metrics.iter().map(|metric| metric.wall_ms).sum::<u128>();
    let max_run_wall_ms = metrics
        .iter()
        .map(|metric| metric.wall_ms)
        .max()
        .unwrap_or_default();
    let wall_ms = wall_start.elapsed().as_millis();
    let disk_bytes = dir_size(&options.out_dir);
    let mut resource_summary = summarize_resources(&resource_samples);
    resource_summary.peak_campaign_disk_bytes = disk_bytes;
    let summary = CampaignSummary {
        started_at,
        finished_at: Utc::now(),
        total_runs: metrics.len(),
        passed,
        failed,
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
        sim_phase_summary: sim_phase_summary.clone(),
        phase_attribution,
        concurrency_summary,
    };
    write_json(&options.out_dir.join("summary.json"), &summary)?;
    write_campaign_summary(
        &options.out_dir,
        &summary,
        sim_phase_summary.as_ref(),
        &memory,
    )?;
    reporter.finish();

    if let Some(hook) = &config.hooks.post_campaign {
        let context = options.out_dir.join("campaign_hook_context.json");
        write_json(
            &context,
            &json!({
                "out_dir": options.out_dir,
                "results": options.out_dir.join("results.csv"),
                "perf": options.out_dir.join("perf.csv"),
                "memory": options.memory_probe.then(|| options.out_dir.join("memory.json")),
                "resources": options.out_dir.join("resources.csv"),
                "summary": options.out_dir.join("summary.json"),
            }),
        )?;
        run_hook("post_campaign", hook, &context).await?;
    }

    if let Some(run_id) = failure.lock().expect("failure mutex poisoned").clone()
        && !config.continue_on_error
    {
        return Err(Error::RunFailed(run_id)).into_diagnostic();
    }

    Ok(())
}

async fn run_with_retries(
    row: PlanRow,
    worker_id: usize,
    slot: ResourceSlot,
    base_recipe: s10::Recipe,
    config: &CampaignConfig,
    out_dir: &Path,
    cache_dir: &Path,
) -> Result<RunMetric> {
    let mut last_metric = None;
    let ctx = RunContext {
        base_recipe,
        config,
        out_dir,
        cache_dir,
    };
    for attempt in 0..=config.retries {
        let metric = run_one(&row, worker_id, &slot, &ctx, attempt).await?;
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
}

async fn run_one(
    row: &PlanRow,
    worker_id: usize,
    slot: &ResourceSlot,
    ctx: &RunContext<'_>,
    attempt: usize,
) -> Result<RunMetric> {
    let run_dir = ctx.out_dir.join("runs").join(&row.run_id);
    let db_path = run_dir.join("db");
    if run_dir.exists() {
        fs::remove_dir_all(&run_dir)
            .into_diagnostic()
            .with_context(|| format!("remove stale run dir {}", run_dir.display()))?;
    }
    fs::create_dir_all(&run_dir).into_diagnostic()?;
    let context_path = run_dir.join("context.json");
    write_run_context(
        &context_path,
        row,
        slot,
        &db_path,
        &run_dir,
        ctx.cache_dir,
        ctx.config,
    )?;

    let started_at = Utc::now();
    let spawn_unix_ns = unix_now_ns();
    let start = Instant::now();
    let recipe = patch_recipe(
        ctx.base_recipe.clone(),
        row,
        &PatchContext {
            slot,
            context_path: &context_path,
            cache_dir: ctx.cache_dir,
            db_path: &db_path,
            run_dir: &run_dir,
            config: ctx.config,
        },
    )?;
    let token = CancelToken::new();
    let fut =
        s10::cli::run_recipe_with_token(row.run_id.clone(), recipe, false, false, token.clone());
    let result = if let Some(timeout) = parse_duration(ctx.config.timeout.as_deref())? {
        match tokio::time::timeout(timeout, fut).await {
            Ok(result) => result,
            Err(_) => {
                token.cancel();
                Err(miette!("run {} timed out after {:?}", row.run_id, timeout))
            }
        }
    } else {
        fut.await
    };
    let exit_unix_ns = unix_now_ns();
    let exit_ok = result.is_ok();
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
    let metric = RunMetric {
        run_id: row.run_id.clone(),
        worker_id,
        attempt,
        status: if exit_ok { "ok" } else { "failed" }.to_string(),
        exit_ok,
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
    write_json(&run_dir.join("metrics.json"), &metric)?;
    if let Some(hook) = &ctx.config.hooks.post_run {
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
    }
    Ok(metric)
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
    workers_override: Option<usize>,
    runtime_threads_override: Option<usize>,
) -> Result<ResolvedRunShape> {
    let mut config = load_config(campaign_path)?;
    if let Some(workers) = workers_override {
        config.workers = Some(workers.max(1));
    }
    let plan = read_plan(plan_path)?;
    if plan.is_empty() {
        return Err(Error::EmptyPlan).into_diagnostic();
    }
    let workers = resolve_workers(config.workers, plan.len());
    Ok(ResolvedRunShape {
        workers,
        runtime_threads: resolve_runtime_threads(workers, runtime_threads_override),
    })
}

fn resolve_workers(config_workers: Option<usize>, plan_len: usize) -> usize {
    if let Some(workers) = config_workers {
        return workers.max(1).min(plan_len.max(1));
    }
    let reserve = if available_cpus() > 4 { 2 } else { 1 };
    available_cpus()
        .saturating_sub(reserve)
        .max(1)
        .min(plan_len.max(1))
}

fn resolve_runtime_threads(workers: usize, override_threads: Option<usize>) -> usize {
    if let Some(threads) = override_threads.filter(|threads| *threads > 0) {
        return threads.max(1);
    }
    let available = available_cpus();
    if available <= 1 {
        1
    } else {
        workers.clamp(2, available.min(8))
    }
}

fn available_cpus() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

fn apply_overrides(config: &mut CampaignConfig, options: &mut RunOptions) {
    if let Some(workers) = options.workers {
        config.workers = Some(workers.max(1));
    }
    if let Some(cache_dir) = options.cache_dir.take() {
        config.cache_dir = Some(cache_dir);
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
    if let Some(compat) = options.params_compat.take() {
        config.params_compat = Some(compat);
    }
    if options.fail_fast {
        config.continue_on_error = false;
    }
    config.workers = config.workers.map(|workers| workers.max(1));
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
    let mut env = HashMap::from([
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
            "ELODIN_MONTE_CARLO_STATE_PORT".to_string(),
            ctx.slot.state_port.to_string(),
        ),
        (
            "ELODIN_MONTE_CARLO_COMMAND_PORT".to_string(),
            ctx.slot.command_port.to_string(),
        ),
        (
            "ELODIN_DB_PATH".to_string(),
            ctx.db_path.to_string_lossy().to_string(),
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
    if ctx.config.params_compat.as_deref() == Some("revere-overrides-file") {
        let overrides_path = ctx.context_path.with_file_name("revere_overrides.json");
        write_json(&overrides_path, &row.params)?;
        env.insert(
            "REVERE_SIM_OVERRIDES_FILE".to_string(),
            overrides_path.to_string_lossy().to_string(),
        );
        if let Some(seed) = row.seed {
            env.insert("SIM_SEED".to_string(), seed.to_string());
        }
        env.insert(
            "REVERE_DB_PATH".to_string(),
            ctx.db_path.to_string_lossy().to_string(),
        );
    } else if let Some(mode) = ctx.config.params_compat.as_deref() {
        return Err(Error::UnsupportedCompat(mode.to_string())).into_diagnostic();
    }

    Ok(patch_recipe_env(
        recipe,
        &env,
        ctx.slot.db_addr,
        ctx.run_dir,
        "recipe",
    ))
}

struct PatchContext<'a> {
    slot: &'a ResourceSlot,
    context_path: &'a Path,
    cache_dir: &'a Path,
    db_path: &'a Path,
    run_dir: &'a Path,
    config: &'a CampaignConfig,
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
            s10::Recipe::Process(process)
        }
        s10::Recipe::Cargo(mut cargo) => {
            cargo.process_args.env.extend(env.clone());
            cargo.process_args.fail_on_error = true;
            cargo.process_args.log_path = Some(log_path_for(run_dir, name));
            s10::Recipe::Cargo(cargo)
        }
        #[cfg(not(target_os = "windows"))]
        s10::Recipe::Sim(mut sim) => {
            sim.addr = db_addr;
            sim.env.extend(env.clone());
            sim.log_path = Some(log_path_for(run_dir, name));
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

fn resource_slot(worker_id: usize, resources: &ResourceConfig) -> Result<ResourceSlot> {
    let offset = worker_id
        .checked_mul(resources.port_stride as usize)
        .ok_or_else(|| miette!("worker port offset overflow"))?;
    let shift = |base: u16| -> Result<u16> {
        let port = base as usize + offset;
        u16::try_from(port).into_diagnostic()
    };
    let db_port = shift(resources.db_port)?;
    Ok(ResourceSlot {
        worker_id,
        db_port,
        db_addr: SocketAddr::new(resources.bind_ip, db_port),
        state_port: shift(resources.state_port)?,
        command_port: shift(resources.command_port)?,
    })
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
            "state_port": slot.state_port,
            "command_port": slot.command_port,
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
    for metric in metrics {
        writer.serialize(metric).into_diagnostic()?;
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
    writer
        .write_record([
            "run_id",
            "status",
            "worker_id",
            "wall_ms",
            "db_path",
            "result_json",
        ])
        .into_diagnostic()?;
    for metric in metrics {
        let result_json = metric.run_dir.join("result.json");
        writer
            .write_record([
                metric.run_id.as_str(),
                metric.status.as_str(),
                &metric.worker_id.to_string(),
                &metric.wall_ms.to_string(),
                &metric.db_path.to_string_lossy(),
                &result_json
                    .strip_prefix(out_dir)
                    .unwrap_or(&result_json)
                    .to_string_lossy(),
            ])
            .into_diagnostic()?;
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

fn placeholder_metric(run_id: &str, worker_id: usize, out_dir: &Path, status: &str) -> RunMetric {
    let now = Utc::now();
    let unix_ns = unix_now_ns();
    let run_dir = out_dir.join("runs").join(run_id);
    RunMetric {
        run_id: run_id.to_string(),
        worker_id,
        attempt: 0,
        status: status.to_string(),
        exit_ok: false,
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
        "  runs:              {} ok / {} failed / {} skipped / {} total\n",
        summary.passed, summary.failed, summary.skipped, summary.total_runs
    ));
    rendered.push_str(&format!("  workers:           {}\n", summary.workers));
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
}
