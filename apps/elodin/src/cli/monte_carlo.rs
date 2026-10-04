use std::path::PathBuf;

use clap::{Args as ClapArgs, Subcommand};
use miette::{IntoDiagnostic, Result, miette};

use super::Cli;

#[derive(ClapArgs, Clone)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Clone)]
enum Command {
    /// Generate a starter Monte Carlo spec or plan from a simulation
    Template {
        sim: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Scaffold a runnable campaign (spec + campaign + hooks) from a simulation
    Quickstart { sim: PathBuf, output: PathBuf },
    /// Materialize a sampling spec into a plan CSV
    Sample {
        spec: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Run a Monte Carlo campaign
    Run(Box<RunArgs>),
    /// Resume a previous campaign
    Resume {
        campaign_dir: PathBuf,
        /// Do not re-exec under `systemd-run --user --scope` when no
        /// delegated cgroup is available.
        #[arg(long)]
        no_self_scope: bool,
    },
    /// Rebuild a campaign report
    Report { campaign_dir: PathBuf },
}

#[derive(ClapArgs, Clone)]
pub struct RunArgs {
    pub sim: PathBuf,
    #[arg(long)]
    pub plan: Option<PathBuf>,
    #[arg(long)]
    pub spec: Option<PathBuf>,
    #[arg(long)]
    pub campaign: Option<PathBuf>,
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Concurrent worker slots (exactly this many runs execute at once).
    /// Overrides `S10_MAX_INFLIGHT` and `campaign.toml`'s `workers`.
    #[arg(long)]
    pub workers: Option<usize>,
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Run per-run IO on a scratch filesystem ("auto" picks /dev/shm),
    /// moving artifacts to --out as each run finishes.
    #[arg(long)]
    pub scratch_dir: Option<String>,
    #[arg(long)]
    pub retries: Option<usize>,
    #[arg(long)]
    pub timeout: Option<String>,
    #[arg(long)]
    pub post_run: Option<PathBuf>,
    #[arg(long)]
    pub post_campaign: Option<PathBuf>,
    #[arg(long)]
    pub fail_fast: bool,
    #[arg(long)]
    pub fail_on_errors: bool,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub memory_probe: bool,
    #[arg(long)]
    pub keep_existing: bool,
    #[arg(long)]
    pub clean: bool,
    /// Error (instead of warn) when planned ports fall inside the kernel
    /// ephemeral range.
    #[arg(long)]
    pub strict_ports: bool,
    /// Do not re-exec under `systemd-run --user --scope` when no delegated
    /// cgroup is available.
    #[arg(long)]
    pub no_self_scope: bool,
    #[arg(long)]
    pub runtime_threads: Option<usize>,
}

impl Cli {
    pub fn monte_carlo(self, args: Args, rt: tokio::runtime::Runtime) -> Result<()> {
        match args.command {
            Command::Template { sim, output } => {
                python_module(["-m", "elodin.monte_carlo.template"], [sim, output])
            }
            Command::Quickstart { sim, output } => {
                python_module(["-m", "elodin.monte_carlo.quickstart"], [sim, output])
            }
            Command::Sample { spec, output } => {
                python_module(["-m", "elodin.monte_carlo.sample"], [spec, output])
            }
            Command::Run(args) => run_with_runtime(*args, rt),
            Command::Resume {
                campaign_dir,
                no_self_scope,
            } => {
                // Resumed runs need the same reliable teardown as fresh ones.
                if !no_self_scope {
                    self_scope_reexec();
                }
                rt.block_on(monte_carlo::resume_campaign(campaign_dir))
            }
            Command::Report { campaign_dir } => monte_carlo::rebuild_report(&campaign_dir),
        }
    }
}

fn run_with_runtime(args: RunArgs, rt: tokio::runtime::Runtime) -> Result<()> {
    if !args.dry_run && !args.no_self_scope {
        self_scope_reexec();
    }
    let runtime_threads = args.runtime_threads;
    let workers = args.workers;
    let options = run_options(args)?;
    let shape = monte_carlo::resolve_run_shape(
        options.campaign_path.as_deref(),
        &options.plan_path,
        runtime_threads,
        workers,
    )?;
    if shape.runtime_threads > 1 {
        let threaded_rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(shape.runtime_threads)
            .enable_all()
            .build()
            .map_err(|err| miette!("failed to build monte-carlo runtime: {err}"))?;
        threaded_rt.block_on(monte_carlo::run_campaign(options))
    } else {
        rt.block_on(monte_carlo::run_campaign(options))
    }
}

/// Without a delegated cgroup (e.g. a plain ssh session scope), the runner
/// cannot kill a timed-out run's whole process tree. When `systemd-run
/// --user --scope` can provide one, transparently re-exec the campaign inside
/// it; `--scope` runs the command as our child with inherited stdio/env/cwd,
/// so the progress UI is unaffected. `--no-self-scope` opts out.
#[cfg(target_os = "linux")]
fn self_scope_reexec() {
    const GUARD: &str = "ELODIN_MC_SELF_SCOPED";
    if std::env::var_os(GUARD).is_some() || s10::cgroups_available() {
        return;
    }
    let probe = std::process::Command::new("systemd-run")
        .args(["--user", "--scope", "--collect", "-q", "true"])
        .status();
    if !probe.map(|status| status.success()).unwrap_or(false) {
        eprintln!(
            "warning: no delegated cgroup and systemd-run --user unavailable; \
             a timed-out run may leak child processes (fallback process-group kill still applies)"
        );
        return;
    }
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(_) => return,
    };
    eprintln!("re-executing under `systemd-run --user --scope` for reliable run teardown");
    let status = std::process::Command::new("systemd-run")
        .args(["--user", "--scope", "--collect", "-q"])
        .arg(exe)
        .args(std::env::args_os().skip(1))
        .env(GUARD, "1")
        .status();
    match status {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(err) => {
            eprintln!("warning: systemd-run re-exec failed ({err}); continuing unscoped");
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn self_scope_reexec() {}

fn run_options(args: RunArgs) -> Result<monte_carlo::RunOptions> {
    let out_dir = args.out.unwrap_or_else(|| {
        PathBuf::from(format!(
            "mc-{}",
            chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
        ))
    });
    let plan_path = match (args.plan, args.spec) {
        (Some(plan), None) => plan,
        (None, Some(spec)) => {
            let plan = out_dir.join("plan.csv");
            python_module(["-m", "elodin.monte_carlo.sample"], [spec, plan.clone()])?;
            plan
        }
        (Some(_), Some(_)) => return Err(miette!("use either --plan or --spec, not both")),
        (None, None) => return Err(miette!("monte-carlo run requires --plan or --spec")),
    };
    Ok(monte_carlo::RunOptions {
        sim_path: args.sim,
        plan_path,
        campaign_path: args.campaign,
        out_dir,
        cache_dir: args.cache_dir,
        workers: args.workers,
        scratch_dir: args.scratch_dir,
        retries: args.retries,
        timeout: args.timeout,
        post_run: args.post_run,
        post_campaign: args.post_campaign,
        fail_fast: args.fail_fast,
        fail_on_run_errors: args.fail_on_errors,
        dry_run: args.dry_run,
        memory_probe: args.memory_probe,
        keep_existing: args.keep_existing,
        clean: args.clean,
        strict_ports: args.strict_ports,
    })
}

fn python_module<const N: usize, const M: usize>(
    fixed_args: [&str; N],
    path_args: [PathBuf; M],
) -> Result<()> {
    let mut command = s10::python_command().into_diagnostic()?;
    command.args(fixed_args);
    command.args(path_args);
    let status = command.status().into_diagnostic()?;
    if status.success() {
        Ok(())
    } else {
        Err(miette!("python helper failed with status {status}"))
    }
}
