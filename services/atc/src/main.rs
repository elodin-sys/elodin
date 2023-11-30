use api::Api;
use config::Config;
use futures::future;
use sea_orm::Database;
use tracing::info;

use crate::orca::Orca;

mod api;
mod config;
mod error;
mod events;
mod orca;
mod sandbox;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = Config::parse()?;
    info!(?config, "config");
    let mut services = vec![];
    let db = Database::connect(config.database_url).await?;
    let redis = redis::Client::open(config.redis_url)?;

    if let Some(orca_config) = config.orca {
        let redis = redis.get_tokio_connection().await?;
        let redis = redis.into_pubsub();
        let orca = Orca::new(orca_config, db.clone()).await?;
        let handle = orca.run(redis);
        services.push(handle);
    }
    if let Some(api_config) = config.api {
        let redis = redis.get_multiplexed_tokio_connection().await?;
        let api = Api::new(api_config, db.clone(), redis).await?;
        services.push(tokio::spawn(api.run()));
    }

    let (res, _, _) = future::select_all(services.into_iter()).await;
    res?
}
