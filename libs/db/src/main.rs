use std::{io::Write, net::SocketAddr, path::PathBuf};

use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use elodin_db::Server;
use impeller2::vtable;
use miette::IntoDiagnostic;
use postcard_c_codegen::SchemaExt;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Clone)]
enum Commands {
    #[command(about = "Run the Elodin database server")]
    Run(RunArgs),
    #[command(about = "Run a Lua script or launch a REPL")]
    Lua(impeller2_cli::Args),
    #[command(about = "Generate C++ header files")]
    GenCpp,
    #[command(
        name = "fix-timestamps",
        about = "Fix monotonic timestamps in a database"
    )]
    FixTimestamps(FixTimestampsArgs),
    #[command(about = "Merge two databases into one with optional prefixes")]
    Merge(MergeArgs),
}

#[derive(clap::Args, Clone, Debug)]
struct RunArgs {
    #[clap(default_value = "[::]:2240", help = "Address to bind the server to")]
    addr: SocketAddr,
    #[clap(help = "Path to the data directory")]
    path: Option<PathBuf>,
    #[clap(
        long,
        value_enum,
        default_value = "info",
        help = "Log level (error, warn, info, debug, trace)"
    )]
    log_level: LogLevel,
    #[clap(long, help = "Start timestamp in microseconds")]
    start_timestamp: Option<i64>,
    #[clap(long, help = "Path to the configuration file")]
    pub config: Option<PathBuf>,
    #[cfg(feature = "axum")]
    #[clap(long, help = "Address to bind the HTTP server to")]
    http_addr: Option<SocketAddr>,
    #[clap(long, hide = true)]
    reset: bool,
}

#[derive(clap::Args, Clone, Debug)]
struct FixTimestampsArgs {
    #[clap(help = "Path to the database directory")]
    path: PathBuf,
    #[clap(long, help = "Show what would be changed without modifying")]
    dry_run: bool,
    #[clap(long, short, help = "Skip confirmation prompt")]
    yes: bool,
    #[clap(
        long = "no-prune",
        action = ArgAction::SetFalse,
        default_value_t = true,
        help = "Do not prune empty components"
    )]
    prune: bool,
    #[clap(
        long,
        value_enum,
        default_value = "wall-clock",
        help = "Clock to use as reference when computing offsets"
    )]
    reference: ReferenceClockArg,
}

#[derive(ValueEnum, Clone, Debug)]
enum ReferenceClockArg {
    WallClock,
    Monotonic,
}

#[derive(clap::Args, Clone, Debug)]
pub struct MergeArgs {
    #[clap(help = "Path to the first source database")]
    pub db1: PathBuf,
    #[clap(help = "Path to the second source database")]
    pub db2: PathBuf,
    #[clap(long, short, help = "Path for the merged output database")]
    pub output: PathBuf,
    #[clap(long, help = "Prefix to apply to first database components")]
    pub prefix1: Option<String>,
    #[clap(long, help = "Prefix to apply to second database components")]
    pub prefix2: Option<String>,
    #[clap(long, help = "Show what would be merged without creating output")]
    pub dry_run: bool,
    #[clap(long, short, help = "Skip confirmation prompt")]
    pub yes: bool,
    #[clap(
        long,
        help = "Alignment timestamp (seconds) in DB1 - defines the output timeline"
    )]
    pub align1: Option<f64>,
    #[clap(
        long,
        help = "Alignment timestamp (seconds) in DB2 - will be shifted to align with align1"
    )]
    pub align2: Option<f64>,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

#[stellarator::main]
async fn main() -> miette::Result<()> {
    let args = Cli::parse();
    let log_level = match &args.command {
        Commands::Run(args) => args.log_level,
        _ => LogLevel::Info,
    };
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::builder().from_env_lossy()
    } else {
        EnvFilter::builder().parse_lossy(format!("elodin_db={}", log_level.as_str()))
    };

    let _ = tracing_subscriber::fmt::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new(
            "%Y-%m-%d %H:%M:%S%.3f".to_string(),
        ))
        .try_init();
    match args.command {
        Commands::Run(RunArgs {
            addr,
            http_addr,
            path,
            config,
            reset,
            start_timestamp,
            ..
        }) => {
            let path = path.unwrap_or_else(|| {
                let dirs =
                    directories::ProjectDirs::from("systems", "elodin", "db").expect("no dirs");
                dirs.data_dir().join("data")
            });
            if reset && path.exists() {
                info!(?path, "resetting db");
                std::fs::remove_dir_all(&path).unwrap_or_else(|_| {
                    tracing::warn!("failed to remove existing data directory");
                });
            }
            info!(?path, "starting db");
            let server = Server::new(&path, addr).into_diagnostic()?;
            if let Some(start_timestamp) = start_timestamp {
                server
                    .db
                    .set_earliest_timestamp(impeller2::types::Timestamp(start_timestamp))
                    .into_diagnostic()?;
            }
            let axum_db = server.db.clone();
            let db = stellarator::spawn(server.run());
            if let Some(http_addr) = http_addr {
                stellarator::struc_con::tokio(move |_| async move {
                    elodin_db::axum::serve(http_addr, axum_db).await.unwrap()
                });
            }
            if let Some(lua_config) = config {
                let args = impeller2_cli::Args {
                    config: Some(lua_config),
                    db: Some(path.clone()),
                    lua_args: vec![],
                };
                impeller2_cli::run(args)
                    .await
                    .map_err(|e| miette::miette!(e))?;
            }
            db.await.unwrap().into_diagnostic()
        }
        Commands::Lua(args) => impeller2_cli::run(args)
            .await
            .map_err(|e| miette::miette!(e)),
        Commands::GenCpp => {
            let header = postcard_c_codegen::hpp_header(
                "ELODIN_DB",
                [
                    include_str!("../../postcard-c/postcard.h").to_string(),
                    impeller2_wkt::InitialTimestamp::to_cpp()?,
                    impeller2_wkt::FixedRateBehavior::to_cpp()?,
                    impeller2_wkt::StreamBehavior::to_cpp()?,
                    impeller2_wkt::Stream::to_cpp()?,
                    impeller2_wkt::MsgStream::to_cpp()?,
                    vtable::Field::to_cpp()?,
                    vtable::Op::to_cpp()?,
                    vtable::OpRef::to_cpp()?,
                    impeller2::types::PrimType::to_cpp()?,
                    vtable::VTable::<Vec<vtable::Op>, Vec<u8>, Vec<vtable::Field>>::to_cpp()?,
                    impeller2_wkt::VTableMsg::to_cpp()?,
                    impeller2_wkt::VTableStream::to_cpp()?,
                    impeller2_wkt::ComponentMetadata::to_cpp()?,
                    impeller2_wkt::SetComponentMetadata::to_cpp()?,
                    include_str!("../cpp/helpers.hpp").to_string(),
                    include_str!("../cpp/vtable.hpp").to_string(),
                ],
            )?;
            std::io::stdout()
                .write_all(header.as_bytes())
                .into_diagnostic()?;
            Ok(())
        }
        Commands::FixTimestamps(FixTimestampsArgs {
            path,
            dry_run,
            yes,
            prune,
            reference,
        }) => {
            let reference = match reference {
                ReferenceClockArg::WallClock => {
                    elodin_db::fix_timestamps::ReferenceClock::WallClock
                }
                ReferenceClockArg::Monotonic => {
                    elodin_db::fix_timestamps::ReferenceClock::Monotonic
                }
            };
            elodin_db::fix_timestamps::run(path, dry_run, yes, reference, prune).into_diagnostic()
        }
        Commands::Merge(MergeArgs {
            db1,
            db2,
            output,
            prefix1,
            prefix2,
            dry_run,
            yes,
            align1,
            align2,
        }) => elodin_db::merge::run(
            db1, db2, output, prefix1, prefix2, dry_run, yes, align1, align2,
        )
        .into_diagnostic(),
    }
}
