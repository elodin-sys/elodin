use crate::{
    config::{Auth0Config, StripePlansConfig},
    current_user_route, current_user_route_txn,
    error::Error,
};
use axum::{body::Body, http::Request, routing::post};
use axum::{extract::MatchedPath, routing::get};
use elodin_types::api::*;
use fred::prelude::*;
use jsonwebtoken::jwk::JwkSet;
use sea_orm::{ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter, TransactionTrait};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, time::Duration};
use tonic::async_trait;
use tonic::service::Routes;
use tonic::{Response, Status};
use tower::{make::Shared, steer::Steer};
use tower_http::{classify::ServerErrorsFailureClass, trace::TraceLayer};
use tracing::{Instrument, Span};

use crate::config::ApiConfig;

mod billing_account;
mod license;
pub(crate) mod stripe;
mod user;
mod utils;
use utils::*;

pub struct Api {
    base_url: String,
    address: SocketAddr,
    db: DatabaseConnection,
    auth_context: AuthContext,
    redis: RedisClient,
    stripe: ::stripe::Client,
    stripe_webhook_secret: String,
    stripe_plans_config: StripePlansConfig,
}

#[derive(Clone)]
pub struct AuthContext {
    auth0_keys: JwkSet,
    auth_config: Auth0Config,
}

#[derive(Clone)]
pub struct AxumContext {
    db: DatabaseConnection,
    redis: RedisClient,
    webhook_secret: String,
    stripe_plans_config: StripePlansConfig,
}

impl Api {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        config: ApiConfig,
        db: DatabaseConnection,
        redis: RedisClient,
        stripe: ::stripe::Client,
        stripe_webhook_secret: String,
    ) -> anyhow::Result<Self> {
        let auth0_keys = get_keyset(&config.auth0.domain).await?;
        let auth_context = AuthContext {
            auth0_keys,
            auth_config: config.auth0.clone(),
        };
        Ok(Self {
            base_url: config.base_url,
            address: config.address,
            db,
            redis,
            auth_context,
            stripe,
            stripe_webhook_secret,
            stripe_plans_config: config.stripe_plans.clone(),
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let address = self.address;
        tracing::info!(api.addr = ?address, "api listening");

        let rest = axum::Router::new()
            .layer(
                TraceLayer::new_for_http()
                    .make_span_with(|req: &Request<_>| {
                        let matched_path = req
                            .extensions()
                            .get::<MatchedPath>()
                            .map(MatchedPath::as_str);
                        tracing::info_span!(
                            "req",
                            method = ?req.method(),
                            matched_path,
                            some_other_field = tracing::field::Empty,
                        )
                    })
                    .on_failure(
                        |err: ServerErrorsFailureClass, _latency: Duration, _span: &Span| {
                            tracing::error!(error = ?err, "request failed");
                        },
                    ),
            )
            .route("/healthz", get(healthz))
            .route("/stripe/webhook", post(stripe::stripe_webhook))
            .with_state(AxumContext {
                db: self.db.clone(),
                redis: self.redis.clone(),
                webhook_secret: self.stripe_webhook_secret.clone(),
                stripe_plans_config: self.stripe_plans_config.clone(),
            });
        let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter
            .set_serving::<api_server::ApiServer<Api>>()
            .await;
        let svc = api_server::ApiServer::new(self);
        let reflection = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(elodin_types::FILE_DESCRIPTOR_SET)
            .build_v1()?;

        let grpc = Routes::default()
            .add_service(health_service)
            .add_service(svc)
            .add_service(reflection)
            .prepare()
            .into_axum_router();
        let service = Steer::new(
            vec![rest, grpc],
            |req: &Request<Body>, _services: &[_]| {
                if is_grpc_request(req) {
                    1
                } else {
                    0
                }
            },
        );

        let listener = tokio::net::TcpListener::bind(address).await?;
        axum::serve(listener, Shared::new(service)).await?;

        tracing::debug!("done");
        Ok(())
    }
}

fn is_grpc_request<B>(req: &Request<B>) -> bool {
    req.headers()
        .get(hyper::header::CONTENT_TYPE)
        .map(|content_type| content_type.as_bytes())
        .filter(|content_type| content_type.starts_with(b"application/grpc"))
        .is_some()
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct CurrentUser {
    user: atc_entity::user::Model,
    claims: Claims,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserInfo {
    sub: String,
    name: String,
    email: String,
    picture: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    sub: String,
}

#[async_trait]
impl api_server::Api for Api {
    async fn create_user(
        &self,
        req: tonic::Request<CreateUserReq>,
    ) -> Result<Response<CreateUserResp>, Status> {
        self.authed_route_userinfo(req, |req, userinfo| self.create_user(req, userinfo))
            .await
    }

    async fn current_user(
        &self,
        req: tonic::Request<CurrentUserReq>,
    ) -> Result<Response<CurrentUserResp>, Status> {
        self.authed_route(req, |_, claims| async {
            let claims = claims.ok_or(Error::Unauthorized)?;
            self.current_user(claims).await
        })
        .await
    }

    async fn update_user(
        &self,
        req: tonic::Request<UpdateUserReq>,
    ) -> Result<Response<UpdateUserResp>, Status> {
        current_user_route_txn!(self, req, Self::update_user)
    }

    async fn create_billing_account(
        &self,
        req: tonic::Request<CreateBillingAccountReq>,
    ) -> Result<Response<BillingAccount>, Status> {
        current_user_route_txn!(self, req, Self::create_billing_account)
    }

    async fn get_stripe_subscription_status(
        &self,
        req: tonic::Request<GetStripeSubscriptionStatusReq>,
    ) -> Result<Response<StripeSubscriptionStatus>, Status> {
        current_user_route_txn!(self, req, Self::get_stripe_subscription_status)
    }

    async fn generate_license(
        &self,
        req: tonic::Request<GenerateLicenseReq>,
    ) -> Result<Response<GenerateLicenseResp>, Status> {
        current_user_route!(self, req, Self::generate_license)
    }
}

async fn healthz() -> impl axum::response::IntoResponse {
    "OK"
}
