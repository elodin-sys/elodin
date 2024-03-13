mod cli;

fn main() {
    cli::Cli::from_os_args().run();
}
