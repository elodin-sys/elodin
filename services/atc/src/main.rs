use std::time::Duration;

use api::{stripe::sync_monte_carlo_billing, Api};
use atc_entity::{events, vm};
use config::{Config, ElodinEnvironment};
use fred::prelude::*;
use migration::{Migrator, MigratorTrait};
use sea_orm::Database;
use stripe::EventFilter;
use tokio::signal::unix::{signal, SignalKind};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::{orca::Orca, sandbox::garbage_collect};

mod api;
mod config;
mod error;
mod monte_carlo;
mod orca;
mod sandbox;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::new()?;

    match config.env {
        ElodinEnvironment::Local | ElodinEnvironment::DevBranch => {
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::INFO)
                .with_env_filter("atc=debug")
                .init();
        }
        ElodinEnvironment::Dev | ElodinEnvironment::Prod => {
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::INFO)
                .with_env_filter("atc=debug")
                .json()
                .init();
        }
    };
    tracing::info!(?config, "config");

    // this cancellation token is hooked into all of the spawned tasks
    // any of them can trigger a full shutdown via a dead-man's switch
    let cancel_token = CancellationToken::new();
    let signal_cancel_token = cancel_token.clone();
    tokio::spawn(async move {
        // initiate graceful shutdown on sigterm (but not sigint or ctrl+c)
        let sigterm = SignalKind::terminate();
        signal(sigterm).unwrap().recv().await;
        tracing::warn!("received SIGTERM, initiating graceful shutdown");
        signal_cancel_token.cancel();
    });

    let mut services = tokio::task::JoinSet::new();
    let mut opt = sea_orm::ConnectOptions::new(config.database_url);
    opt.sqlx_logging(false);
    let db = Database::connect(opt).await?;
    if config.migrate {
        Migrator::up(&db, None).await?;
    }

    let redis_config = RedisConfig::from_url(&config.redis_url)?;
    let redis_builder = Builder::from_config(redis_config);
    let redis = redis_builder.build()?;
    let conn_task = redis.init().await?;
    let sim_storage_client = monte_carlo::SimStorageClient::new(&config.monte_carlo).await?;

    if let Some(orca_config) = config.orca {
        let vm_events = events::subscribe::<vm::Model>(&redis_builder).await?;
        let orca = Orca::new(orca_config, db.clone(), redis.clone(), vm_events).await?;
        services.spawn(
            orca.run(cancel_token.clone())
                .instrument(tracing::info_span!("orca")),
        );
    }
    let mut stripe_webhook_id = None;
    if let Some(ref api_config) = config.api {
        let sandbox_events = events::subscribe(&redis_builder).await?;
        let monte_carlo_run_events = events::subscribe(&redis_builder).await?;
        let monte_carlo_batch_events = events::subscribe(&redis_builder).await?;
        let msg_queue = redmq::MsgQueue::new(&redis, "atc", &config.pod_name).await?;
        let stripe = stripe::Client::new(api_config.stripe_secret_key.expose_secret().to_string());
        let stripe_webhook_secret =
            if let Some(secret) = api_config.stripe_webhook_secret.expose_secret().clone() {
                secret.to_string()
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
                stripe_webhook_id = Some(endpoint.id.clone());
                endpoint
                    .secret
                    .ok_or_else(|| anyhow::anyhow!("stripe webhook secret not found"))?
            };
        let api = Api::new(
            api_config.clone(),
            db.clone(),
            redis.clone(),
            msg_queue,
            sim_storage_client,
            sandbox_events,
            monte_carlo_run_events,
            monte_carlo_batch_events,
            stripe.clone(),
            stripe_webhook_secret.to_string(),
        )
        .await?;
        let cancel_on_drop = cancel_token.clone().drop_guard();

        let db_connection = db.clone();
        let stripe_secret_key = api_config.stripe_secret_key.expose_secret().to_string();

        services.spawn(async move {
            let client = reqwest::Client::new();

            loop {
                tokio::time::sleep(Duration::from_secs(5 * 60)).await;

                sync_monte_carlo_billing(&client, &db_connection, &stripe_secret_key).await?;
            }
        });
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
            services.spawn(
                garbage_collect(db.clone(), redis.clone(), gc.timeout, cancel_token.clone())
                    .instrument(tracing::info_span!("gc")),
            );
        }
    }

    // wait for cancellation
    cancel_token.cancelled().await;
    tracing::info!("waiting for services to terminate gracefully");
    // cleanup webhook
    if let Some(stripe_webhook_id) = stripe_webhook_id {
        stripe::WebhookEndpoint::delete(
            &stripe::Client::new(config.api.unwrap().stripe_secret_key.expose_secret()),
            &stripe_webhook_id,
        )
        .await?;
    }

    redis.quit().await?;
    while let Some(res) = services.join_next().await {
        res.unwrap()?
    }
    conn_task.await.unwrap()?;

    Ok(())
}
