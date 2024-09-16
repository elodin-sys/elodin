mod cli;

fn main() -> miette::Result<()> {
    cli::Cli::from_os_args().run()
}
