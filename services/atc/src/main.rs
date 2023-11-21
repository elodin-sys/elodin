use api::Api;
use config::Config;
use futures::future;
use tracing::info;

mod api;
mod config;
mod error;
mod orca;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = Config::parse()?;
    info!(?config, "config");
    let mut services = vec![];
    if let Some(api_config) = config.api {
        let api = Api::new(api_config, config.database_url.clone()).await?;
        services.push(tokio::spawn(api.run()));
    }
    let (res, _, _) = future::select_all(services.into_iter()).await;
    res?
}
