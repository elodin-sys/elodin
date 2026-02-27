use std::{io::Write, net::SocketAddr, path::PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
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
    #[command(about = "Remove empty components from a database")]
    Prune(PruneArgs),
    #[command(about = "Trim a database to a time range, removing data outside the window")]
    Trim(TrimArgs),
    #[command(about = "Clear all data from a database, preserving schemas")]
    Truncate(TruncateArgs),
    #[command(
        name = "time-align",
        about = "Align component timestamps to a target timestamp"
    )]
    TimeAlign(TimeAlignArgs),
    #[command(about = "Drop (delete) components from a database")]
    Drop(DropArgs),
    #[command(about = "Display information about a database")]
    Info(InfoArgs),
    #[command(about = "Export database contents to parquet, arrow-ipc, or csv files")]
    Export(ExportArgs),
    #[cfg(feature = "video-export")]
    #[command(
        name = "export-videos",
        about = "Export video message logs to MP4 files"
    )]
    ExportVideos(ExportVideosArgs),
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
    #[clap(
        long,
        help = "Replay recorded data as live telemetry (advances last_updated with playback)"
    )]
    replay: bool,
    #[clap(long, help = "Follow another elodin-db instance, replicating all data")]
    follows: Option<SocketAddr>,
    #[clap(
        long,
        default_value = "1500",
        help = "Target packet size in bytes for follow streaming (data is buffered to this size before sending)"
    )]
    follow_packet_size: usize,
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
        long,
        value_enum,
        default_value = "wall-clock",
        help = "Clock to use as reference when computing offsets"
    )]
    reference: ReferenceClockArg,
}

#[derive(clap::Args, Clone, Debug)]
struct PruneArgs {
    #[clap(help = "Path to the database directory")]
    path: PathBuf,
    #[clap(long, help = "Show what would be pruned without modifying")]
    dry_run: bool,
    #[clap(long, short, help = "Skip confirmation prompt")]
    yes: bool,
}

#[derive(clap::Args, Clone, Debug)]
struct TrimArgs {
    #[clap(help = "Path to the database directory")]
    path: PathBuf,
    #[clap(
        long,
        help = "Remove the first N microseconds of data (from the start of the recording)"
    )]
    from_start: Option<i64>,
    #[clap(
        long,
        help = "Remove the last N microseconds of data (from the end of the recording)"
    )]
    from_end: Option<i64>,
    #[clap(long, short, help = "Output path (modifies in place if not specified)")]
    output: Option<PathBuf>,
    #[clap(long, help = "Show what would be trimmed without modifying")]
    dry_run: bool,
    #[clap(long, short, help = "Skip confirmation prompt")]
    yes: bool,
}

#[derive(clap::Args, Clone, Debug)]
struct TruncateArgs {
    #[clap(help = "Path to the database directory")]
    path: PathBuf,
    #[clap(long, help = "Show what would be truncated without modifying")]
    dry_run: bool,
    #[clap(long, short, help = "Skip confirmation prompt")]
    yes: bool,
}

#[derive(clap::Args, Clone, Debug)]
struct TimeAlignArgs {
    #[clap(help = "Path to the database directory")]
    path: PathBuf,
    #[clap(long, help = "Target timestamp (seconds) to align first sample to")]
    timestamp: f64,
    #[clap(long, help = "Align all components")]
    all: bool,
    #[clap(long, help = "Specific component name to align")]
    component: Option<String>,
    #[clap(long, help = "Show what would be changed without modifying")]
    dry_run: bool,
    #[clap(long, short, help = "Skip confirmation prompt")]
    yes: bool,
}

#[derive(clap::Args, Clone, Debug)]
struct DropArgs {
    #[clap(help = "Path to the database directory")]
    path: PathBuf,
    #[clap(long, help = "Component name to match (fuzzy)")]
    component: Option<String>,
    #[clap(
        long,
        help = "Glob pattern to match component names (e.g., 'rocket.*')"
    )]
    pattern: Option<String>,
    #[clap(long, help = "Drop all components")]
    all: bool,
    #[clap(long, help = "Show what would be dropped without modifying")]
    dry_run: bool,
    #[clap(long, short, help = "Skip confirmation prompt")]
    yes: bool,
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
    #[clap(long, help = "Alignment timestamp (microseconds) for an event in DB1")]
    pub align1: Option<i64>,
    #[clap(
        long,
        help = "Alignment timestamp (microseconds) for the same event in DB2. DB2 is shifted to align with DB1."
    )]
    pub align2: Option<i64>,
    #[clap(
        long,
        help = "Interpret --align1/--align2 as offsets from each database's playback start rather than absolute timestamps"
    )]
    pub from_playback_start: bool,
}

#[derive(clap::Args, Clone, Debug)]
struct InfoArgs {
    #[clap(help = "Path to the database directory (defaults to standard location)")]
    path: Option<PathBuf>,
}

#[derive(clap::Args, Clone, Debug)]
struct ExportArgs {
    #[clap(help = "Path to the database directory")]
    path: PathBuf,
    #[clap(long, short, help = "Output directory for exported files")]
    output: PathBuf,
    #[clap(
        long,
        value_enum,
        default_value = "parquet",
        help = "Export format (parquet, arrow-ipc, csv)"
    )]
    format: elodin_db::export::ExportFormat,
    #[clap(
        long,
        help = "Flatten vector columns to separate columns (e.g., vel_ned -> vel_ned_x, vel_ned_y, vel_ned_z)"
    )]
    flatten: bool,
    #[clap(long, help = "Filter components by glob pattern (e.g., 'NavNED.*')")]
    pattern: Option<String>,
}

#[cfg(feature = "video-export")]
#[derive(clap::Args, Clone, Debug)]
struct ExportVideosArgs {
    #[clap(help = "Path to the database directory")]
    path: PathBuf,
    #[clap(long, short, help = "Output directory for MP4 files")]
    output: PathBuf,
    #[clap(long, help = "Filter message logs by name glob (e.g., 'test-*')")]
    pattern: Option<String>,
    #[clap(
        long,
        default_value = "30",
        help = "Frame rate when SPS has no timing_info"
    )]
    fps: u32,
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
            replay,
            follows,
            follow_packet_size,
            ..
        }) => {
            let path = path.unwrap_or_else(|| {
                if follows.is_some() {
                    // Follower mode without an explicit path: use a temp directory
                    // so we don't pollute (or inherit stale data from) the default
                    // system data directory.
                    let tmp = std::env::temp_dir()
                        .join(format!("elodin-db-follower-{}", std::process::id()));
                    info!(?tmp, "follower mode: using temp data directory");
                    tmp
                } else {
                    let dirs =
                        directories::ProjectDirs::from("systems", "elodin", "db").expect("no dirs");
                    dirs.data_dir().join("data")
                }
            });
            if reset && path.exists() {
                info!(?path, "resetting db");
                std::fs::remove_dir_all(&path).unwrap_or_else(|_| {
                    tracing::warn!("failed to remove existing data directory");
                });
            }
            if replay && !path.exists() {
                return Err(miette::miette!(
                    "--replay cannot be used when the database path does not exist; create and record data first"
                ));
            }
            info!(?path, "starting db");
            let server = Server::new(&path, addr).into_diagnostic()?;
            // Apply start_timestamp before enable_replay_mode so replay uses the
            // correct earliest_timestamp; otherwise last_updated can end up before
            // earliest_timestamp and the editor shows NoData.
            if let Some(start_timestamp) = start_timestamp {
                server
                    .db
                    .set_earliest_timestamp(impeller2::types::Timestamp(start_timestamp))
                    .into_diagnostic()?;
            }
            if replay {
                if server.db.last_updated.latest().0 == i64::MIN {
                    return Err(miette::miette!(
                        "--replay cannot be used on an empty database; record data first"
                    ));
                }
                info!("replay mode enabled: last_updated will advance with playback");
                server.db.enable_replay_mode();
            }
            let axum_db = server.db.clone();
            // Spawn follower before server.run() consumes server.
            if let Some(source_addr) = follows {
                let follow_db = server.db.clone();
                stellarator::struc_con::stellar(move || {
                    elodin_db::follow::run_follower(
                        elodin_db::follow::FollowConfig {
                            source_addr,
                            target_packet_size: follow_packet_size,
                            reconnect_delay: std::time::Duration::from_secs(2),
                        },
                        follow_db,
                    )
                });
            }
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
            elodin_db::fix_timestamps::run(path, dry_run, yes, reference).into_diagnostic()
        }
        Commands::Prune(PruneArgs { path, dry_run, yes }) => {
            elodin_db::prune::run(path, dry_run, yes).into_diagnostic()
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
            from_playback_start,
        }) => {
            use elodin_db::MetadataExt;
            let (align1, align2) = if from_playback_start {
                let resolve = |db_path: &std::path::Path,
                               val: Option<i64>|
                 -> miette::Result<Option<i64>> {
                    match val {
                        Some(v) => {
                            let config = impeller2_wkt::DbConfig::read(db_path.join("db_state"))
                                .into_diagnostic()?;
                            let start = config.time_start_timestamp_micros().unwrap_or(0);
                            Ok(Some(start.saturating_add(v)))
                        }
                        None => Ok(None),
                    }
                };
                (resolve(&db1, align1)?, resolve(&db2, align2)?)
            } else {
                (align1, align2)
            };
            elodin_db::merge::run(
                db1, db2, output, prefix1, prefix2, dry_run, yes, align1, align2,
            )
            .into_diagnostic()
        }
        Commands::Trim(TrimArgs {
            path,
            from_start,
            from_end,
            output,
            dry_run,
            yes,
        }) => {
            elodin_db::trim::run(path, from_start, from_end, output, dry_run, yes).into_diagnostic()
        }
        Commands::Truncate(TruncateArgs { path, dry_run, yes }) => {
            elodin_db::truncate::run(path, dry_run, yes).into_diagnostic()
        }
        Commands::TimeAlign(TimeAlignArgs {
            path,
            timestamp,
            all,
            component,
            dry_run,
            yes,
        }) => elodin_db::time_align::run(path, timestamp, all, component, dry_run, yes)
            .into_diagnostic(),
        Commands::Drop(DropArgs {
            path,
            component,
            pattern,
            all,
            dry_run,
            yes,
        }) => {
            // Validate that exactly one matching mode is specified
            let mode_count = [component.is_some(), pattern.is_some(), all]
                .iter()
                .filter(|&&x| x)
                .count();

            if mode_count == 0 {
                return Err(miette::miette!(
                    "Must specify one of --component, --pattern, or --all"
                ));
            }
            if mode_count > 1 {
                return Err(miette::miette!(
                    "Cannot combine --component, --pattern, and --all. Specify only one."
                ));
            }

            let match_mode = if let Some(name) = component {
                elodin_db::drop::MatchMode::Fuzzy(name)
            } else if let Some(pat) = pattern {
                elodin_db::drop::MatchMode::Pattern(pat)
            } else {
                elodin_db::drop::MatchMode::All
            };

            elodin_db::drop::run(path, match_mode, dry_run, yes).into_diagnostic()
        }
        Commands::Info(args) => run_info(args),
        Commands::Export(ExportArgs {
            path,
            output,
            format,
            flatten,
            pattern,
        }) => {
            // Install signal handlers only for Export command which uses check_cancelled()
            elodin_db::cancellation::install_signal_handlers();

            elodin_db::export::run(path, output, format, flatten, pattern).into_diagnostic()
        }
        #[cfg(feature = "video-export")]
        Commands::ExportVideos(args) => {
            elodin_db::export_videos::run(args.path, args.output, args.pattern, args.fps)
                .into_diagnostic()
        }
    }
}

fn run_info(args: InfoArgs) -> miette::Result<()> {
    use impeller2_wkt::DbConfig;

    let db_state_path = match args.path {
        Some(path) => {
            let state_path = path.join("db_state");
            if state_path.exists() {
                state_path
            } else if path.is_dir() {
                return Err(miette::miette!(
                    "db_state not found in directory: {}",
                    path.display()
                ));
            } else {
                path
            }
        }
        None => {
            let dirs = directories::ProjectDirs::from("systems", "elodin", "db").expect("no dirs");
            dirs.data_dir().join("data").join("db_state")
        }
    };

    if !db_state_path.exists() {
        return Err(miette::miette!(
            "db_state not found: {}",
            db_state_path.display()
        ));
    }

    let bytes = std::fs::read(&db_state_path)
        .map_err(|e| miette::miette!("failed to read {}: {e}", db_state_path.display()))?;
    let config: DbConfig =
        postcard::from_bytes(&bytes).map_err(|e| miette::miette!("decode error: {e}"))?;

    println!("db_state: {}", db_state_path.display());

    // Display version information prominently
    if let Some(version) = config.version_created() {
        println!("version_created: {}", version);
    }
    if let Some(version) = config.version_last_opened() {
        println!("version_last_opened: {}", version);
    }

    println!("recording: {}", config.recording);
    println!(
        "default_stream_time_step: {}",
        format_duration(config.default_stream_time_step)
    );

    print_metadata(&config);

    Ok(())
}

fn format_duration(duration: std::time::Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos >= 1_000_000_000 {
        format!("{} s", nanos / 1_000_000_000)
    } else if nanos >= 1_000_000 {
        format!("{} ms", nanos / 1_000_000)
    } else if nanos >= 1_000 {
        format!("{} us", nanos / 1_000)
    } else {
        format!("{} ns", nanos)
    }
}

fn print_metadata(config: &impeller2_wkt::DbConfig) {
    let meta = &config.metadata;

    // Filter out version keys (displayed separately) and collect remaining metadata
    let mut keys: Vec<&String> = meta.keys().filter(|k| !k.starts_with("version.")).collect();
    keys.sort();

    if keys.is_empty() {
        println!("metadata: <empty>");
        return;
    }

    println!("metadata:");
    for key in keys {
        if let Some(value) = meta.get(key) {
            println!("  {key}: {value}");
        }
    }
}
