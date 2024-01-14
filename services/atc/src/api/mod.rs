use crate::{
    api::{multiplex::MultiplexService, sandbox::sim_socket},
    config::Auth0Config,
    current_user_route, current_user_route_txn,
    error::Error,
    events::DbEvent,
};
use axum::routing::get;
use elodin_types::api::{
    api_server::{self, ApiServer},
    BootSandboxReq, BootSandboxResp, CreateSandboxReq, CreateSandboxResp, CreateUserReq,
    CreateUserResp, CurrentUserReq, CurrentUserResp, GetSandboxReq, ListSandboxesReq,
    ListSandboxesResp, Sandbox, UpdateSandboxReq, UpdateSandboxResp,
};
use futures::Stream;
use jsonwebtoken::jwk::JwkSet;
use redis::aio::MultiplexedConnection;
use sea_orm::{ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter, TransactionTrait};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, pin::Pin};
use tokio::sync::broadcast;
use tonic::async_trait;
use tonic::{transport::Server, Response, Status};
use tracing::info;

use crate::config::ApiConfig;

mod multiplex;
mod sandbox;
mod user;
mod utils;
use utils::*;

pub struct Api {
    address: SocketAddr,
    db: DatabaseConnection,
    auth_context: AuthContext,
    redis: MultiplexedConnection,
    sandbox_events: broadcast::Receiver<DbEvent<atc_entity::sandbox::Model>>,
}

#[derive(Clone)]
pub struct AuthContext {
    auth0_keys: JwkSet,
    auth_config: Auth0Config,
}

#[derive(Clone)]
pub struct WSContext {
    auth_context: AuthContext,
    db: DatabaseConnection,
}

impl Api {
    pub async fn new(
        config: ApiConfig,
        db: DatabaseConnection,
        redis: MultiplexedConnection,
        sandbox_events: broadcast::Receiver<DbEvent<atc_entity::sandbox::Model>>,
    ) -> anyhow::Result<Self> {
        let auth0_keys = get_keyset(&config.auth0.domain).await?;
        let auth_context = AuthContext {
            auth0_keys,
            auth_config: config.auth0.clone(),
        };
        Ok(Self {
            address: config.address,
            db,
            redis,
            auth_context,
            sandbox_events,
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let address = self.address;
        info!(api.addr = ?address, "api listening");

        let rest = axum::Router::new()
            .route("/sim/ws/:id", get(sim_socket))
            .route("/healthz", get(healthz))
            .with_state(WSContext {
                auth_context: self.auth_context.clone(),
                db: self.db.clone(),
            });
        let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter.set_serving::<ApiServer<Api>>().await;
        let svc = ApiServer::new(self);
        let reflection = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(elodin_types::FILE_DESCRIPTOR_SET)
            .build()?;

        let grpc = Server::builder()
            .add_service(health_service)
            .add_service(svc)
            .add_service(reflection)
            .into_service();
        let service = MultiplexService::new(rest, grpc);
        axum::Server::bind(&address)
            .serve(tower::make::Shared::new(service))
            .await?;

        Ok(())
    }
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
    async fn current_user(
        &self,
        req: tonic::Request<CurrentUserReq>,
    ) -> Result<Response<CurrentUserResp>, Status> {
        self.authed_route(req, |_, claims| self.current_user(claims))
            .await
    }

    async fn create_user(
        &self,
        req: tonic::Request<CreateUserReq>,
    ) -> Result<Response<CreateUserResp>, Status> {
        self.authed_route_userinfo(req, |req, userinfo| self.create_user(req, userinfo))
            .await
    }

    async fn create_sandbox(
        &self,
        req: tonic::Request<CreateSandboxReq>,
    ) -> Result<Response<CreateSandboxResp>, Status> {
        current_user_route_txn!(self, req, Self::create_sandbox)
    }

    async fn update_sandbox(
        &self,
        req: tonic::Request<UpdateSandboxReq>,
    ) -> Result<Response<UpdateSandboxResp>, Status> {
        current_user_route_txn!(self, req, Self::update_sandbox)
    }

    async fn boot_sandbox(
        &self,
        req: tonic::Request<BootSandboxReq>,
    ) -> Result<Response<BootSandboxResp>, Status> {
        current_user_route!(self, req, Self::boot_sandbox)
    }

    async fn list_sandboxes(
        &self,
        req: tonic::Request<ListSandboxesReq>,
    ) -> Result<Response<ListSandboxesResp>, Status> {
        current_user_route_txn!(self, req, Self::list_sandbox)
    }

    async fn get_sandbox(
        &self,
        req: tonic::Request<GetSandboxReq>,
    ) -> Result<Response<Sandbox>, Status> {
        current_user_route_txn!(self, req, Self::get_sandbox)
    }

    type SandboxEventsStream = Pin<Box<dyn Stream<Item = Result<Sandbox, Status>> + Send + Sync>>;
    async fn sandbox_events(
        &self,
        req: tonic::Request<GetSandboxReq>,
    ) -> Result<Response<Self::SandboxEventsStream>, Status> {
        current_user_route!(self, req, Self::sandbox_events)
    }
}

async fn healthz() -> impl axum::response::IntoResponse {
    "OK"
}
