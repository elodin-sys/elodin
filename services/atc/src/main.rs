use api::Api;
use config::Config;
use futures::future;
use migration::{Migrator, MigratorTrait};
use sea_orm::Database;
use tracing::info;

use crate::{events::EventMonitor, orca::Orca};

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
    if config.migrate {
        Migrator::up(&db, None).await?;
    }

    let redis = redis::Client::open(config.redis_url)?;

    if let Some(orca_config) = config.orca {
        let pubsub = redis.get_tokio_connection().await?;
        let pubsub = pubsub.into_pubsub();
        let redis = redis.get_multiplexed_tokio_connection().await?;
        let orca = Orca::new(orca_config, db.clone(), redis).await?;
        let handle = orca.run(pubsub);
        services.push(handle);
    }
    if let Some(api_config) = config.api {
        let (sandbox_monitor, sandbox_events) = EventMonitor::<atc_entity::sandbox::Model>::pair();
        {
            let redis = redis.get_tokio_connection().await?;
            let redis = redis.into_pubsub();
            services.push(sandbox_monitor.run(redis))
        };
        let redis = redis.get_multiplexed_tokio_connection().await?;
        let api = Api::new(api_config, db.clone(), redis, sandbox_events).await?;
        services.push(tokio::spawn(api.run()));
    }

    let (res, _, _) = future::select_all(services.into_iter()).await;
    res?
}
