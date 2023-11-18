use api::Api;
use config::Config;
use futures::future;
use tracing::info;

mod api;
mod config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = Config::parse()?;
    info!(?config, "config");
    let mut services = vec![];
    if let Some(api_config) = config.api {
        services.push(tokio::spawn(Api.run(api_config)));
    }
    let (res, _, _) = future::select_all(services.into_iter()).await;
    res?
}
