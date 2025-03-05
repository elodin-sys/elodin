use api::Api;
use config::{Config, ElodinEnvironment};
use fred::prelude::*;
use migration::{Migrator, MigratorTrait};
use sea_orm::Database;
use stripe::EventFilter;
use tokio::signal::unix::{SignalKind, signal};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

mod api;
mod config;
mod error;

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

    let mut services = tokio::task::JoinSet::<anyhow::Result<()>>::new();
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

    let mut stripe_webhook_id = None;
    if let Some(ref api_config) = config.api {
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
            stripe.clone(),
            stripe_webhook_secret.to_string(),
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
