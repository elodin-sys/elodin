use clap::Subcommand;
use elodin_types::api::{api_client::ApiClient, *};
use tonic::{service::interceptor::InterceptedService, transport};

use crate::auth::AuthInterceptor;
use crate::Cli;

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

#[derive(clap::Args)]
struct RunArgs {
    #[arg(short, long)]
    name: String,
    #[arg(short, long)]
    samples: u32,
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
        let mut client = self.client().await?;
        let create_req = CreateMonteCarloRunReq {
            name: args.name.clone(),
            samples: args.samples,
        };
        let create_res = client
            .create_monte_carlo_run(create_req)
            .await?
            .into_inner();
        let id = uuid::Uuid::from_slice(&create_res.id)?;
        let upload_url = create_res.upload_url;

        let start_req = StartMonteCarloRunReq {
            id: id.as_bytes().to_vec(),
        };
        client.start_monte_carlo_run(start_req).await?.into_inner();
        println!("Created Monte Carlo run with id: {id}, upload url: {upload_url}");

        Ok(())
    }
}
