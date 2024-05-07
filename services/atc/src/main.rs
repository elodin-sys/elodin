use api::Api;
use config::Config;
use migration::{Migrator, MigratorTrait};
use sea_orm::Database;
use stripe::EventFilter;
use tokio::signal::unix::{signal, SignalKind};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn, Instrument};

use crate::{orca::Orca, sandbox::garbage_collect};

mod api;
mod config;
mod error;
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
    let sim_storage_client = monte_carlo::SimStorageClient::new(&config.monte_carlo).await?;

    if let Some(orca_config) = config.orca {
        let pubsub = redis.get_async_pubsub().await?;
        let redis = redis.get_multiplexed_tokio_connection().await?;
        let orca = Orca::new(orca_config, db.clone(), redis).await?;
        services.spawn(
            orca.run(pubsub, cancel_token.clone())
                .instrument(tracing::info_span!("orca")),
        );
    }
    if let Some(api_config) = config.api {
        let sandbox_events = {
            let redis = redis.get_async_pubsub().await?;
            atc_entity::events::subscribe(redis).await?
        };
        let monte_carlo_run_events = {
            let redis = redis.get_async_pubsub().await?;
            atc_entity::events::subscribe(redis).await?
        };
        let monte_carlo_batch_events = {
            let redis = redis.get_async_pubsub().await?;
            atc_entity::events::subscribe(redis).await?
        };
        let msg_queue = redmq::MsgQueue::new(&redis, "atc", &config.pod_name).await?;
        let redis = redis.get_multiplexed_tokio_connection().await?;
        let stripe = stripe::Client::new(api_config.stripe_secret_key.clone());
        let stripe_webhook_secret = if let Some(secret) = api_config.stripe_webhook_secret.clone() {
            secret
        } else {
            let endpoint = stripe::WebhookEndpoint::create(
                &stripe,
                stripe::CreateWebhookEndpoint::new(
                    vec![
                        EventFilter::CustomerSubscriptionPaused,
                        EventFilter::CustomerSubscriptionDeleted,
                        EventFilter::CustomerSubscriptionCreated,
                        EventFilter::CustomerSubscriptionUpdated,
                        EventFilter::CustomerSubscriptionResumed,
                        EventFilter::CustomerSubscriptionTrialWillEnd,
                        EventFilter::CustomerSubscriptionPendingUpdateExpired,
                        EventFilter::CustomerSubscriptionPendingUpdateApplied,
                    ],
                    &format!("{}/stripe/webhook", api_config.base_url),
                ),
            )
            .await?;
            endpoint
                .secret
                .ok_or_else(|| anyhow::anyhow!("stripe webhook secret not found"))?
        };
        let api = Api::new(
            api_config,
            db.clone(),
            redis,
            msg_queue,
            sim_storage_client,
            sandbox_events,
            monte_carlo_run_events,
            monte_carlo_batch_events,
            stripe,
            stripe_webhook_secret,
        )
        .await?;
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

    // wait for cancellation
    cancel_token.cancelled().await;
    info!("waiting for services to terminate gracefully");
    while let Some(res) = services.join_next().await {
        res.unwrap()?
    }
    Ok(())
}
