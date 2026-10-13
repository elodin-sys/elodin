use std::path::PathBuf;

use clap::Subcommand;
use elodin_mcp::index::Index;
use elodin_mcp::source::local_dbs;
use miette::{IntoDiagnostic, miette};

use super::Cli;

#[derive(clap::Args, Clone, Debug)]
pub struct Args {
    /// Local database directory, optionally renamed (repeatable)
    #[arg(long = "db", value_name = "[NAME=]PATH", global = true)]
    pub db: Vec<String>,
    /// Elodin Cloud project name
    #[arg(long, value_name = "NAME", global = true, conflicts_with = "db")]
    pub project: Option<String>,
    /// Server only: parent directory for execution directories [default: ./elodin-artifacts]
    #[arg(long = "artifacts", value_name = "DIR")]
    pub artifacts: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Clone, Debug)]
pub enum Command {
    /// Search runs, tables, signals, and the SDK reference; prints an overview without EXPR
    Search(SearchArgs),
    /// Run a Python analysis script against the recordings
    Execute(ExecuteArgs),
    /// Write the synthetic test recording to DIR
    #[command(hide = true)]
    Fixture { dir: PathBuf },
}

#[derive(clap::Args, Clone, Debug)]
pub struct SearchArgs {
    /// One expression: runs(), tables(q), signals(q), describe(name), sdk(q), or plain text
    pub expr: Option<String>,
    /// Print compact JSON
    #[arg(long)]
    pub json: bool,
    /// Maximum results for tables() and signals()
    #[arg(long)]
    pub limit: Option<usize>,
}

#[derive(clap::Args, Clone, Debug)]
pub struct ExecuteArgs {
    /// Python script, or - for stdin
    pub script: PathBuf,
    /// Script parameter as KEY=VALUE; VALUE is parsed as JSON when valid (repeatable)
    #[arg(long = "param", value_name = "KEY=VALUE")]
    pub param: Vec<String>,
    /// Parameters from a JSON object file
    #[arg(long, value_name = "FILE.json")]
    pub params: Option<PathBuf>,
    /// Execution directory [default: ./elodin-artifacts/<execution_id>/]
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Wall timeout in seconds [default: 600 local, 300 hosted]
    #[arg(long)]
    pub timeout: Option<u64>,
    /// Maximum rows one read may return [default: 5000000]
    #[arg(long)]
    pub max_rows: Option<u64>,
    /// Print compact JSON
    #[arg(long)]
    pub json: bool,
}

impl Cli {
    pub fn mcp(self, args: Args, rt: tokio::runtime::Runtime) -> miette::Result<()> {
        match self.mcp_inner(args, rt) {
            Ok(true) => Ok(()),
            // The script ran and failed (error or timeout).
            Ok(false) => std::process::exit(1),
            // Usage, source, or authentication errors.
            Err(e) => {
                eprintln!("{e:?}");
                std::process::exit(2)
            }
        }
    }

    fn mcp_inner(&self, args: Args, rt: tokio::runtime::Runtime) -> miette::Result<bool> {
        match &args.command {
            Some(Command::Fixture { dir }) => {
                elodin_db::fixture::write(dir).into_diagnostic()?;
                println!("{}", dir.display());
                Ok(true)
            }
            Some(Command::Search(search)) => {
                let index = self.index(&args, &rt)?;
                let out = elodin_mcp::search::search(&index, search.expr.as_deref(), search.limit);
                print_json(&out, search.json);
                Ok(true)
            }
            Some(Command::Execute(exec)) => {
                let code = if exec.script.as_os_str() == "-" {
                    std::io::read_to_string(std::io::stdin()).into_diagnostic()?
                } else {
                    std::fs::read_to_string(&exec.script)
                        .map_err(|e| miette!("read {}: {e}", exec.script.display()))?
                };
                let req = elodin_mcp::exec::Request {
                    code,
                    parameters: elodin_mcp::exec::parse_params(exec.params.as_deref(), &exec.param)
                        .map_err(|e| miette!("{e:#}"))?,
                    limits: elodin_mcp::exec::Limits {
                        timeout_s: exec.timeout,
                        max_rows: exec.max_rows,
                    },
                    out: exec.out.clone(),
                    root: None,
                };
                let envelope = if let Some(name) = &args.project {
                    let (cloud, project) = self.cloud_project(name, &rt)?;
                    rt.block_on(cloud.execute(&project, &req))
                } else {
                    let index = self.index(&args, &rt)?;
                    let dbs = local_dbs(&args.db).map_err(|e| miette!("{e:#}"))?;
                    elodin_mcp::exec::execute_local(&dbs, &index.runs, &req)
                }
                .map_err(|e| miette!("{e:#}"))?;
                print_json(&envelope, exec.json);
                Ok(envelope["status"] == "completed")
            }
            None => {
                // Cloud tools block on API calls from blocking threads, which needs a
                // multi-threaded runtime.
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .into_diagnostic()?;
                let backend: std::sync::Arc<dyn elodin_mcp::server::Backend> =
                    if let Some(name) = &args.project {
                        let (cloud, project) = self.cloud_project(name, &rt)?;
                        std::sync::Arc::new(elodin_mcp::cloud::CloudBackend::new(
                            cloud,
                            project,
                            args.artifacts.clone(),
                        ))
                    } else {
                        let dbs = local_dbs(&args.db).map_err(|e| miette!("{e:#}"))?;
                        if dbs.is_empty() {
                            return Err(miette!("pass --db PATH (repeatable) or --project NAME"));
                        }
                        std::sync::Arc::new(elodin_mcp::source::LocalBackend::new(
                            dbs,
                            args.artifacts.clone(),
                        ))
                    };
                rt.block_on(elodin_mcp::server::serve(backend))
                    .map_err(|e| miette!("{e:#}"))?;
                Ok(true)
            }
        }
    }

    fn cloud_project(
        &self,
        name: &str,
        rt: &tokio::runtime::Runtime,
    ) -> miette::Result<(elodin_mcp::cloud::Cloud, elodin_mcp::cloud::Project)> {
        let cloud = self.cloud()?;
        let project = rt
            .block_on(cloud.project(name))
            .map_err(|e| miette!("{e:#}"))?;
        Ok((cloud, project))
    }

    fn index(&self, args: &Args, rt: &tokio::runtime::Runtime) -> miette::Result<Index> {
        if let Some(name) = &args.project {
            let (cloud, project) = self.cloud_project(name, rt)?;
            let (runs, _) = rt
                .block_on(cloud.catalog(&project, None))
                .map_err(|e| miette!("{e:#}"))?
                .expect("no etag sent");
            return Ok(Index::new(runs));
        }
        if args.db.is_empty() {
            return Err(miette!("pass --db PATH (repeatable) or --project NAME"));
        }
        let dbs = local_dbs(&args.db).map_err(|e| miette!("{e:#}"))?;
        let runs = dbs
            .iter()
            .map(|d| d.run())
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|e| miette!("{e:#}"))?;
        Ok(Index::new(runs))
    }
}

fn print_json(v: &serde_json::Value, compact: bool) {
    let s = if compact {
        serde_json::to_string(v)
    } else {
        serde_json::to_string_pretty(v)
    };
    println!("{}", s.expect("json"));
}
