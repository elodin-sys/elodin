use std::io::Seek;
use std::path::PathBuf;

use clap::Subcommand;
use elodin_types::api::{api_client::ApiClient, *};
use tonic::{service::interceptor::InterceptedService, transport};

use super::auth::AuthInterceptor;
use super::Cli;

#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create and submit a Monte Carlo run
    Run(RunArgs),
}

#[derive(clap::Args, Clone)]
struct RunArgs {
    /// Name of the Monte Carlo run
    #[arg(short, long)]
    name: String,
    /// Number of samples to run
    #[arg(short, long, default_value_t = 100)]
    samples: u32,
    /// Max simulation duration in seconds
    #[arg(short, long, default_value_t = 30)]
    max_duration: u64,
    /// Path to the simulation configuration
    file: PathBuf,
    /// Open the dashboard in the browser
    #[arg(long)]
    open: bool,
}

type Client = ApiClient<InterceptedService<transport::Channel, AuthInterceptor>>;

impl Cli {
    pub async fn client(&self) -> anyhow::Result<Client> {
        let auth_interceptor = self.auth_interceptor()?;
        let channel = transport::Endpoint::from_shared(self.url.clone())?
            .timeout(std::time::Duration::from_secs(5))
            .connect_timeout(std::time::Duration::from_secs(5))
            .connect()
            .await?;
        let client = ApiClient::with_interceptor(channel, auth_interceptor);
        Ok(client)
    }

    pub async fn monte_carlo(&self, args: &Args) -> anyhow::Result<()> {
        match &args.command {
            Commands::Run(run_args) => self.monte_carlo_run(run_args).await,
        }
    }

    async fn monte_carlo_run(&self, args: &RunArgs) -> anyhow::Result<()> {
        let RunArgs {
            name,
            samples,
            max_duration,
            file,
            open,
        } = args.clone();
        let mut client = self.client().await?;
        let create_req = CreateMonteCarloRunReq {
            name: name.clone(),
            samples,
            max_duration,
        };
        let create_res = client
            .create_monte_carlo_run(create_req)
            .await?
            .into_inner();
        let id = uuid::Uuid::from_slice(&create_res.id)?;
        let upload_url = create_res.upload_url;

        let artifacts_file = tokio::task::spawn_blocking(|| prepare_artifacts(file))
            .await
            .unwrap()?;
        let artifacts_file = tokio::fs::File::from_std(artifacts_file);
        reqwest::Client::new()
            .put(&upload_url)
            .body(artifacts_file)
            .send()
            .await?;
        println!("Uploaded simulation artifacts");

        let start_req = StartMonteCarloRunReq {
            id: id.as_bytes().to_vec(),
        };
        client.start_monte_carlo_run(start_req).await?.into_inner();
        println!("Created Monte Carlo run with id: {id}");

        let dashboard_url = format!("{}/monte_carlo/{}/{}", self.url, name, id);
        println!("Monitor the Monte Carlo run at: {dashboard_url}");
        if open {
            opener::open_browser(dashboard_url)?;
        }

        Ok(())
    }
}

fn prepare_artifacts(file: PathBuf) -> anyhow::Result<std::fs::File> {
    let tmp_dir = tempfile::tempdir()?;
    if !file.is_file() {
        anyhow::bail!("Not a file: {}", file.display());
    }
    let file_name = file.file_name().unwrap();

    let status = std::process::Command::new("python3")
        .arg(&file)
        .arg("--")
        .arg("build")
        .arg("--dir")
        .arg(tmp_dir.path())
        .spawn()?
        .wait()?;

    if !status.success() {
        anyhow::bail!("Failed to prepare simulation artifacts: {}", status);
    }

    // Copy the original file into the temporary directory for debugging purposes
    // TODO(Akhil): Remove this once this actually works e2e
    std::fs::copy(&file, tmp_dir.path().join(file_name))?;

    let archive_file = tempfile::tempfile()?;
    let buf = std::io::BufWriter::new(archive_file);
    let zstd = zstd::Encoder::new(buf, 0)?;
    let mut ar = tar::Builder::new(zstd);
    ar.append_dir_all("artifacts", tmp_dir.path())?;
    let mut archive_file = ar.into_inner()?.finish()?.into_inner()?;
    archive_file.rewind()?;
    let len = archive_file.metadata()?.len();
    println!("Bundled simulation artifacts (size: {} bytes)", len);

    Ok(archive_file)
}
