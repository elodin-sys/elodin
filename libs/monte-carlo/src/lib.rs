use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use miette::{Context, IntoDiagnostic, Result, miette};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use stellarator::util::CancelToken;
use tokio::task::JoinSet;

pub const CONTEXT_ENV: &str = "ELODIN_MONTE_CARLO_CONTEXT";
pub const CACHE_ENV: &str = "ELODIN_CACHE_DIR";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CampaignConfig {
    pub workers: usize,
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
            workers: 1,
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanRow {
    pub run_id: String,
    pub seed: Option<u64>,
    pub params: BTreeMap<String, Value>,
    pub meta: BTreeMap<String, Value>,
    pub explicit_db_port: Option<u16>,
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
    pub workers: usize,
    pub wall_ms: u128,
    pub total_run_wall_ms: u128,
    pub average_run_wall_ms: f64,
    pub max_run_wall_ms: u128,
    pub parallel_efficiency: f64,
    pub disk_bytes: u64,
    pub resource_summary: ResourceSummary,
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
    pub mem_total_kib: u64,
    pub mem_available_kib: u64,
    pub campaign_disk_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceSummary {
    pub samples: usize,
    pub average_cpu_percent: f64,
    pub peak_cpu_percent: f64,
    pub peak_memory_used_kib: u64,
    pub peak_campaign_disk_bytes: u64,
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

pub async fn run_campaign(mut options: RunOptions) -> Result<()> {
    let started_at = Utc::now();
    let wall_start = Instant::now();
    let mut config = load_config(options.campaign_path.as_deref())?;
    apply_overrides(&mut config, &mut options);

    let plan = read_plan(&options.plan_path)?;
    if plan.is_empty() {
        return Err(Error::EmptyPlan).into_diagnostic();
    }

    if options.dry_run {
        println!(
            "monte-carlo dry run: runs={} workers={} out_dir={}",
            plan.len(),
            config.workers,
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
    fs::copy(&options.plan_path, options.out_dir.join("plan.csv")).into_diagnostic()?;
    write_json(&options.out_dir.join("campaign.resolved.json"), &config)?;

    let base_recipe = plan_recipe(&options.sim_path, &options.out_dir).await?;
    let rows = Arc::new(Mutex::new(VecDeque::from(plan)));
    let metrics = Arc::new(Mutex::new(Vec::<RunMetric>::new()));
    let failure = Arc::new(Mutex::new(None::<String>));
    let stop_sampling = Arc::new(AtomicBool::new(false));
    let peak_memory = Arc::new(Mutex::new(Vec::<CacheMappingSample>::new()));
    let resource_samples = Arc::new(Mutex::new(Vec::<ResourceSample>::new()));
    let sampler = spawn_memory_sampler(
        cache_dir.clone(),
        stop_sampling.clone(),
        peak_memory.clone(),
    );
    let resource_sampler = spawn_resource_sampler(
        options.out_dir.clone(),
        wall_start,
        stop_sampling.clone(),
        resource_samples.clone(),
    );
    let mut workers = JoinSet::new();

    for worker_id in 0..config.workers {
        let rows = rows.clone();
        let metrics = metrics.clone();
        let failure = failure.clone();
        let base_recipe = base_recipe.clone();
        let config = config.clone();
        let out_dir = options.out_dir.clone();
        let cache_dir = cache_dir.clone();
        workers.spawn(async move {
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
                let metric = run_with_retries(
                    row,
                    worker_id,
                    slot.clone(),
                    base_recipe.clone(),
                    &config,
                    &out_dir,
                    &cache_dir,
                )
                .await?;
                let failed = !metric.exit_ok;
                if failed {
                    *failure.lock().expect("failure mutex poisoned") = Some(metric.run_id.clone());
                }
                metrics.lock().expect("metrics mutex poisoned").push(metric);
            }
            Ok::<(), miette::Report>(())
        });
    }

    while let Some(result) = workers.join_next().await {
        result.into_diagnostic()??;
    }
    stop_sampling.store(true, Ordering::SeqCst);
    sampler.await.into_diagnostic()?;
    resource_sampler.await.into_diagnostic()?;

    let metrics = metrics.lock().expect("metrics mutex poisoned").clone();
    write_perf_csv(&options.out_dir.join("perf.csv"), &metrics)?;
    write_results_csv(
        &options.out_dir.join("results.csv"),
        &options.out_dir,
        &metrics,
    )?;
    let memory = peak_memory.lock().expect("memory mutex poisoned").clone();
    let resource_samples = resource_samples
        .lock()
        .expect("resource samples mutex poisoned")
        .clone();
    write_json(&options.out_dir.join("memory.json"), &memory)?;
    write_resource_csv(&options.out_dir.join("resources.csv"), &resource_samples)?;

    if let Some(hook) = &config.hooks.post_campaign {
        let context = options.out_dir.join("campaign_hook_context.json");
        write_json(
            &context,
            &json!({
                "out_dir": options.out_dir,
                "results": options.out_dir.join("results.csv"),
                "perf": options.out_dir.join("perf.csv"),
                "memory": options.out_dir.join("memory.json"),
                "resources": options.out_dir.join("resources.csv"),
            }),
        )?;
        run_hook("post_campaign", hook, &context).await?;
    }

    let failed = metrics.iter().filter(|metric| !metric.exit_ok).count();
    let total_run_wall_ms = metrics.iter().map(|metric| metric.wall_ms).sum::<u128>();
    let max_run_wall_ms = metrics
        .iter()
        .map(|metric| metric.wall_ms)
        .max()
        .unwrap_or_default();
    let wall_ms = wall_start.elapsed().as_millis();
    let resource_summary = summarize_resources(&resource_samples);
    let summary = CampaignSummary {
        started_at,
        finished_at: Utc::now(),
        total_runs: metrics.len(),
        passed: metrics.len().saturating_sub(failed),
        failed,
        workers: config.workers,
        wall_ms,
        total_run_wall_ms,
        average_run_wall_ms: if metrics.is_empty() {
            0.0
        } else {
            total_run_wall_ms as f64 / metrics.len() as f64
        },
        max_run_wall_ms,
        parallel_efficiency: if wall_ms == 0 || config.workers == 0 {
            0.0
        } else {
            total_run_wall_ms as f64 / (wall_ms as f64 * config.workers as f64)
        },
        disk_bytes: dir_size(&options.out_dir),
        resource_summary,
    };
    write_json(&options.out_dir.join("summary.json"), &summary)?;

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
    let start = Instant::now();
    let recipe = patch_recipe(
        ctx.base_recipe.clone(),
        row,
        slot,
        &context_path,
        ctx.cache_dir,
        &db_path,
        ctx.config,
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
    let exit_ok = result.is_ok();
    let metric = RunMetric {
        run_id: row.run_id.clone(),
        worker_id,
        attempt,
        status: if exit_ok { "ok" } else { "failed" }.to_string(),
        exit_ok,
        wall_ms: start.elapsed().as_millis(),
        started_at,
        finished_at: Utc::now(),
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

fn apply_overrides(config: &mut CampaignConfig, options: &mut RunOptions) {
    if let Some(workers) = options.workers {
        config.workers = workers.max(1);
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
    config.workers = config.workers.max(1);
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
        let mut explicit_db_port = None;
        for (header, value) in headers.iter().zip(record.iter()) {
            if value.is_empty() {
                continue;
            }
            match header {
                "run_id" => run_id = Some(value.to_string()),
                "seed" => seed = Some(value.parse::<u64>().into_diagnostic()?),
                "db_port" => explicit_db_port = Some(value.parse::<u16>().into_diagnostic()?),
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
            explicit_db_port,
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

fn patch_recipe(
    recipe: s10::Recipe,
    row: &PlanRow,
    slot: &ResourceSlot,
    context_path: &Path,
    cache_dir: &Path,
    db_path: &Path,
    config: &CampaignConfig,
) -> Result<s10::Recipe> {
    let mut env = HashMap::from([
        (
            CONTEXT_ENV.to_string(),
            context_path.to_string_lossy().to_string(),
        ),
        (
            CACHE_ENV.to_string(),
            cache_dir.to_string_lossy().to_string(),
        ),
        (
            "ELODIN_MONTE_CARLO_WORKER_ID".to_string(),
            slot.worker_id.to_string(),
        ),
        (
            "ELODIN_MONTE_CARLO_DB_PORT".to_string(),
            slot.db_port.to_string(),
        ),
        (
            "ELODIN_MONTE_CARLO_STATE_PORT".to_string(),
            slot.state_port.to_string(),
        ),
        (
            "ELODIN_MONTE_CARLO_COMMAND_PORT".to_string(),
            slot.command_port.to_string(),
        ),
        (
            "ELODIN_DB_PATH".to_string(),
            db_path.to_string_lossy().to_string(),
        ),
    ]);
    if let Some(seed) = row.seed {
        env.insert("ELODIN_MONTE_CARLO_SEED".to_string(), seed.to_string());
    }
    if config.params_compat.as_deref() == Some("revere-overrides-file") {
        let overrides_path = context_path.with_file_name("revere_overrides.json");
        write_json(&overrides_path, &row.params)?;
        env.insert(
            "REVERE_SIM_OVERRIDES_FILE".to_string(),
            overrides_path.to_string_lossy().to_string(),
        );
        if let Some(seed) = row.seed {
            env.insert("SIM_SEED".to_string(), seed.to_string());
        }
    } else if let Some(mode) = config.params_compat.as_deref() {
        return Err(Error::UnsupportedCompat(mode.to_string())).into_diagnostic();
    }

    Ok(patch_recipe_env(recipe, &env, slot.db_addr))
}

fn patch_recipe_env(
    recipe: s10::Recipe,
    env: &HashMap<String, String>,
    db_addr: SocketAddr,
) -> s10::Recipe {
    match recipe {
        s10::Recipe::Group(mut group) => {
            group.recipes = group
                .recipes
                .into_iter()
                .map(|(name, recipe)| (name, patch_recipe_env(recipe, env, db_addr)))
                .collect();
            s10::Recipe::Group(group)
        }
        s10::Recipe::Process(mut process) => {
            process.process_args.env.extend(env.clone());
            process.process_args.fail_on_error = true;
            s10::Recipe::Process(process)
        }
        s10::Recipe::Cargo(mut cargo) => {
            cargo.process_args.env.extend(env.clone());
            cargo.process_args.fail_on_error = true;
            s10::Recipe::Cargo(cargo)
        }
        #[cfg(not(target_os = "windows"))]
        s10::Recipe::Sim(mut sim) => {
            sim.addr = db_addr;
            sim.env.extend(env.clone());
            s10::Recipe::Sim(sim)
        }
    }
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
    let status = s10::python_command()
        .into_diagnostic()?
        .arg("-m")
        .arg("elodin.monte_carlo.run_hook")
        .arg(hook)
        .arg(kind)
        .arg(context)
        .status()
        .into_diagnostic()
        .with_context(|| format!("run {kind} hook {}", hook.display()))?;
    if !status.success() {
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

fn write_resource_csv(path: &Path, samples: &[ResourceSample]) -> Result<()> {
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

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).into_diagnostic()?;
    }
    let json = serde_json::to_string_pretty(value).into_diagnostic()?;
    fs::write(path, json)
        .into_diagnostic()
        .with_context(|| format!("write {}", path.display()))
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
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut previous_cpu = read_cpu_sample();
        while !stop.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(250)).await;
            let current_cpu = read_cpu_sample();
            let cpu_percent = match (previous_cpu, current_cpu) {
                (Some(previous), Some(current)) => cpu_percent(previous, current),
                _ => 0.0,
            };
            previous_cpu = current_cpu;
            let (mem_total_kib, mem_available_kib) = read_meminfo();
            samples
                .lock()
                .expect("resource samples mutex poisoned")
                .push(ResourceSample {
                    elapsed_ms: started_at.elapsed().as_millis(),
                    cpu_percent,
                    mem_total_kib,
                    mem_available_kib,
                    campaign_disk_bytes: dir_size(&out_dir),
                });
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
        peak_memory_used_kib,
        peak_campaign_disk_bytes,
    }
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

fn cpu_percent(previous: CpuSample, current: CpuSample) -> f64 {
    let total = current.total.saturating_sub(previous.total);
    if total == 0 {
        return 0.0;
    }
    let idle = current.idle.saturating_sub(previous.idle);
    100.0 * (total.saturating_sub(idle)) as f64 / total as f64
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
