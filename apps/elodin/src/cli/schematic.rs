use std::path::PathBuf;

use clap::{Args as ClapArgs, Subcommand};
use miette::{IntoDiagnostic, Result, WrapErr};

use super::Cli;

#[derive(ClapArgs, Clone)]
pub struct Args {
    #[command(subcommand)]
    command: SchematicCommand,
}

#[derive(Subcommand, Clone)]
enum SchematicCommand {
    /// Convert a KDL schematic to an executable Python migration scaffold
    ToPython {
        /// KDL schematic to convert
        input: PathBuf,
        /// Write Python to this path instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

impl Cli {
    pub fn schematic(self, args: Args) -> Result<()> {
        match args.command {
            SchematicCommand::ToPython { input, output } => {
                let source = std::fs::read_to_string(&input)
                    .into_diagnostic()
                    .wrap_err_with(|| format!("failed to read {}", input.display()))?;
                let source_name = input.file_name().and_then(|name| name.to_str());
                let python = impeller2_kdl::schematic_to_python(&source, source_name)
                    .into_diagnostic()
                    .wrap_err_with(|| format!("failed to parse {}", input.display()))?;
                if let Some(output) = output {
                    std::fs::write(&output, python)
                        .into_diagnostic()
                        .wrap_err_with(|| format!("failed to write {}", output.display()))?;
                    eprintln!("wrote {}", output.display());
                } else {
                    print!("{python}");
                }
                Ok(())
            }
        }
    }
}
