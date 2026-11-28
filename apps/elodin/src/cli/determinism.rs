use std::path::PathBuf;

use clap::{Args as ClapArgs, Subcommand};
use miette::{IntoDiagnostic, Result};

use super::Cli;

#[derive(ClapArgs, Clone)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Clone)]
enum Command {
    /// Validate a witness and print its schema, labels, trailer and tick inventory.
    Inspect {
        trace: PathBuf,
        /// Print the complete inspection as JSON.
        #[arg(long)]
        json: bool,
        /// Write the validated witness schema to this path.
        #[arg(long)]
        schema: Option<PathBuf>,
    },
    /// Report the first tick, stage, hook, record, entity and byte where two witnesses differ.
    /// Exits 1 at a mismatch.
    Compare {
        left: PathBuf,
        right: PathBuf,
        /// Stop reading at the first mismatch instead of counting every differing tick.
        #[arg(long)]
        stop_at_first: bool,
    },
    /// Replay a native pre_commit hook on the rows a witness kept exactly, without the
    /// simulation, and compare what it writes with the recording. Exits 1 at a mismatch.
    ReplayHook {
        trace: PathBuf,
        /// The shared library the hook was loaded from.
        #[arg(long)]
        library: PathBuf,
        /// The hook's symbol.
        #[arg(long)]
        symbol: String,
        /// Let the hook do its own I/O instead of reading the recorded external channels.
        #[arg(long)]
        live: bool,
    },
}

impl Cli {
    pub fn determinism(self, args: Args) -> Result<()> {
        let mut command = s10::python_command().into_diagnostic()?;
        command.args([
            "-c",
            "from elodin.determinism import main; raise SystemExit(main())",
        ]);
        match args.command {
            Command::Inspect {
                trace,
                json,
                schema,
            } => {
                command.arg("inspect").arg(trace);
                if json {
                    command.arg("--json");
                }
                if let Some(schema) = schema {
                    command.arg("--schema").arg(schema);
                }
            }
            Command::Compare {
                left,
                right,
                stop_at_first,
            } => {
                command.arg("compare").arg(left).arg(right);
                if stop_at_first {
                    command.arg("--stop-at-first");
                }
            }
            Command::ReplayHook {
                trace,
                library,
                symbol,
                live,
            } => {
                command
                    .arg("replay-hook")
                    .arg(trace)
                    .arg("--library")
                    .arg(library);
                command.args(["--symbol", &symbol]);
                if live {
                    command.arg("--live");
                }
            }
        }
        let status = command.status().into_diagnostic()?;
        if !status.success() {
            std::process::exit(status.code().unwrap_or(2));
        }
        Ok(())
    }
}
