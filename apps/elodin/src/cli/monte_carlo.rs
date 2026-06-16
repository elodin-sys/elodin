use std::path::PathBuf;

use clap::{Args as ClapArgs, Subcommand, ValueEnum};
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
    Resume { campaign_dir: PathBuf },
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
    pub workers: Option<usize>,
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    #[arg(long)]
    pub retries: Option<usize>,
    #[arg(long)]
    pub timeout: Option<String>,
    #[arg(long)]
    pub post_run: Option<PathBuf>,
    #[arg(long)]
    pub post_campaign: Option<PathBuf>,
    #[arg(long)]
    pub params_compat: Option<String>,
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
    #[arg(long, value_enum, default_value = "auto")]
    pub progress: ProgressMode,
    #[arg(long)]
    pub runtime_threads: Option<usize>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ProgressMode {
    Auto,
    Always,
    Never,
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
            Command::Resume { campaign_dir } => rt.block_on(monte_carlo::resume_campaign(
                campaign_dir,
                monte_carlo::ProgressMode::Auto,
            )),
            Command::Report { campaign_dir } => monte_carlo::rebuild_report(&campaign_dir),
        }
    }
}

fn run_with_runtime(args: RunArgs, rt: tokio::runtime::Runtime) -> Result<()> {
    let runtime_threads = args.runtime_threads;
    let options = run_options(args)?;
    let shape = monte_carlo::resolve_run_shape(
        options.campaign_path.as_deref(),
        &options.plan_path,
        options.workers,
        runtime_threads,
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
        workers: args.workers,
        cache_dir: args.cache_dir,
        retries: args.retries,
        timeout: args.timeout,
        post_run: args.post_run,
        post_campaign: args.post_campaign,
        params_compat: args.params_compat,
        fail_fast: args.fail_fast,
        fail_on_run_errors: args.fail_on_errors,
        dry_run: args.dry_run,
        memory_probe: args.memory_probe,
        keep_existing: args.keep_existing,
        progress: match args.progress {
            ProgressMode::Auto => monte_carlo::ProgressMode::Auto,
            ProgressMode::Always => monte_carlo::ProgressMode::Always,
            ProgressMode::Never => monte_carlo::ProgressMode::Never,
        },
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
