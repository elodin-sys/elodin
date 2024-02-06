use api::Api;
use config::Config;
use migration::{Migrator, MigratorTrait};
use sea_orm::Database;
use tokio::signal::unix::{signal, SignalKind};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn, Instrument};

use crate::{events::EventMonitor, orca::Orca, sandbox::garbage_collect};

mod api;
mod config;
mod error;
mod events;
mod monte_carlo;
mod orca;
mod sandbox;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = Config::new()?;
    info!(?config, "config");

    // this cancellation token is hooked into all of the spawned tasks
    // any of them can trigger a full shutdown via a dead-man's switch
    let cancel_token = CancellationToken::new();
    let signal_cancel_token = cancel_token.clone();
    tokio::spawn(async move {
        // initiate graceful shutdown on sigterm (but not sigint or ctrl+c)
        let sigterm = SignalKind::terminate();
        signal(sigterm).unwrap().recv().await;
        warn!("received SIGTERM, initiating graceful shutdown");
        signal_cancel_token.cancel();
    });

    let mut services = tokio::task::JoinSet::new();
    let mut opt = sea_orm::ConnectOptions::new(config.database_url);
    opt.sqlx_logging(false);
    let db = Database::connect(opt).await?;
    if config.migrate {
        Migrator::up(&db, None).await?;
    }

    let redis = redis::Client::open(config.redis_url)?;

    if let Some(orca_config) = config.orca {
        let pubsub = redis.get_tokio_connection().await?;
        let pubsub = pubsub.into_pubsub();
        let redis = redis.get_multiplexed_tokio_connection().await?;
        let orca = Orca::new(orca_config, db.clone(), redis).await?;
        services.spawn(
            orca.run(pubsub, cancel_token.clone())
                .instrument(tracing::info_span!("orca")),
        );
    }
    if let Some(api_config) = config.api {
        let (sandbox_monitor, sandbox_events) = EventMonitor::<atc_entity::sandbox::Model>::pair();
        {
            let redis = redis.get_tokio_connection().await?;
            let redis = redis.into_pubsub();
            services.spawn(
                sandbox_monitor
                    .run(redis, cancel_token.clone())
                    .instrument(tracing::info_span!("sandbox_monitor")),
            )
        };
        let msg_queue = redmq::MsgQueue::new(&redis, "atc", &config.pod_name).await?;
        let redis = redis.get_multiplexed_tokio_connection().await?;
        let api = Api::new(api_config, db.clone(), redis, msg_queue, sandbox_events).await?;
        let cancel_on_drop = cancel_token.clone().drop_guard();
        services.spawn(async move {
            // don't add graceful shutdown for API server because "always be serving" is the best strategy for no downtime
            // load balancers and ingress should be responsible for draining connections instead of the application
            api.run().instrument(tracing::info_span!("api")).await?;
            // if api server returns an error, other tasks should still gracefully shutdown
            drop(cancel_on_drop);
            Ok(())
        });
    }

    if let Some(gc) = config.garbage_collect {
        if gc.enabled {
            let redis = redis.get_multiplexed_tokio_connection().await?;
            services.spawn(
                garbage_collect(db.clone(), redis, gc.timeout, cancel_token.clone())
                    .instrument(tracing::info_span!("gc")),
            );
        }
    }

    if config.monte_carlo.spawn_batches {
        let batch_spawner = monte_carlo::BatchSpawner::new(&redis, &config.pod_name).await?;
        services.spawn(
            batch_spawner
                .run(cancel_token.clone())
                .instrument(tracing::info_span!("batch_spawner")),
        );
    }

    if config.monte_carlo.collect_results {
        let aggregator = monte_carlo::Aggregator::new(&redis, db, &config.pod_name).await?;
        services.spawn(
            aggregator
                .run(cancel_token.clone())
                .instrument(tracing::info_span!("aggregator")),
        );
    }

    // wait for cancellation
    cancel_token.cancelled().await;
    info!("waiting for services to terminate gracefully");
    while let Some(res) = services.join_next().await {
        res.unwrap()?
    }
    Ok(())
}
