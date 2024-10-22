mod cli;
mod config;

fn main() -> anyhow::Result<()> {
    cli::Cli::from_os_args().run()
}
