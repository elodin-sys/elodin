use clap::Subcommand;

use super::Cli;

#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a series of Monte Carlo simulations
    Run,
}

impl Cli {
    pub async fn monte_carlo(&self, _args: &Args) -> anyhow::Result<()> {
        let _access_token = self.access_token()?;
        Ok(())
    }
}
