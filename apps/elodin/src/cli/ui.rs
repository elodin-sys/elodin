use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use clap::{Args as ClapArgs, Subcommand};
use miette::{IntoDiagnostic, Result, miette};

use super::Cli;

#[derive(ClapArgs, Clone)]
pub struct Args {
    #[command(subcommand)]
    command: UiCommand,
}

#[derive(Subcommand, Clone)]
enum UiCommand {
    /// Watch a Python schematic script and push on change
    Watch {
        /// Python script defining `build() -> Schematic`
        script: PathBuf,
        /// Impeller DB address
        #[arg(long, default_value = "127.0.0.1:2240")]
        db: String,
        /// Build and push once, then exit
        #[arg(long)]
        once: bool,
        /// Debounce interval for file changes (ms)
        #[arg(long, default_value_t = 200)]
        debounce_ms: u64,
    },
}

impl Cli {
    pub fn ui(self, args: Args) -> Result<()> {
        match args.command {
            UiCommand::Watch {
                script,
                db,
                once,
                debounce_ms,
            } => {
                let python = std::env::var("ELODIN_PYTHON")
                    .ok()
                    .or_else(|| which("python3"))
                    .or_else(|| which("python"))
                    .ok_or_else(|| miette!("python3 not found on PATH"))?;
                let mut cmd = ProcessCommand::new(python);
                cmd.arg("-m")
                    .arg("elodin.ui.watch")
                    .arg(&script)
                    .arg("--db")
                    .arg(&db)
                    .arg("--debounce-ms")
                    .arg(debounce_ms.to_string());
                if once {
                    cmd.arg("--once");
                }
                let status = cmd.status().into_diagnostic()?;
                if status.success() {
                    Ok(())
                } else {
                    Err(miette!("elodin ui watch exited with {status}"))
                }
            }
        }
    }
}

fn which(bin: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            return Some(candidate.display().to_string());
        }
    }
    None
}
