use std::{net::SocketAddr, str::FromStr};

use crate::{
    error::Error,
    orca::{OrcaMsg, VmManager},
};
use atc_entity::{
    sandbox,
    user::{self, EntityType, Permission, Permissions, Verb},
    vm, User,
};
use enumflags2::BitFlag;
use flume::Sender;
use futures::Future;
use jsonwebtoken::{
    decode, decode_header,
    jwk::{AlgorithmParameters, JwkSet},
    Algorithm, DecodingKey, Validation,
};
use paracosm_types::api::{
    api_server::{self, ApiServer},
    BootSandboxReq, BootSandboxResp, CreateSandboxReq, CreateSandboxResp, CreateUserReq,
    CreateUserResp, UpdateSandboxReq, UpdateSandboxResp,
};
use sea_orm::{
    prelude::Uuid, ActiveModelTrait, ActiveValue, ColumnTrait, Database, DatabaseConnection,
    DatabaseTransaction, EntityTrait, QueryFilter, Set, TransactionTrait, Unchanged,
};
use serde::{Deserialize, Serialize};
use tonic::{async_trait, transport::Server, Request, Response, Status};
use tracing::{info, warn};

use crate::config::ApiConfig;

pub struct Api {
    address: SocketAddr,
    db: DatabaseConnection,
    auth0_keys: JwkSet,
    config: ApiConfig,
    vm_manager: VmManager,
}

impl Api {
    pub async fn new(
        config: ApiConfig,
        db_url: String,
        vm_manager: VmManager,
    ) -> anyhow::Result<Self> {
        let db = Database::connect(&db_url).await?;
        let auth0_keys = get_keyset(&config.auth0.domain).await?;
        Ok(Self {
            address: config.address,
            db,
            auth0_keys,
            config,
            vm_manager,
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let address = self.address.clone();
        let svc = ApiServer::new(self);
        info!(api.addr = ?address, "api listening");
        let reflection = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(paracosm_types::FILE_DESCRIPTOR_SET)
            .build()
            .unwrap();

        Server::builder()
            .add_service(svc)
            .add_service(reflection)
            .serve(address)
            .await?;
        Ok(())
    }
}

impl Api {
    async fn create_user(
        &self,
        req: CreateUserReq,
        claims: Claims,
    ) -> Result<CreateUserResp, Error> {
        let id = Uuid::now_v7();
        user::ActiveModel {
            id: Set(id),
            email: Set(req.email),
            name: Set(req.name),
            auth0_id: Set(claims.sub),
            permissions: Set(Permissions::default()),
        }
        .insert(&self.db)
        .await?;
        Ok(CreateUserResp {
            id: id.as_bytes().to_vec(),
        })
    }

    async fn create_sandbox(
        &self,
        req: CreateSandboxReq,
        CurrentUser { mut user, .. }: CurrentUser,
        txn: &DatabaseTransaction,
    ) -> Result<CreateSandboxResp, Error> {
        let id = Uuid::now_v7();
        sandbox::ActiveModel {
            id: Set(id),
            user_id: Set(user.id),
            name: Set(req.name),
            code: Set(req.code),
            status: Set(sandbox::Status::Off),
            vm_id: Set(None),
        }
        .insert(txn)
        .await?;
        user.permissions
            .insert(id, Permission::new(EntityType::Sandbox, Verb::all()));
        user::ActiveModel {
            id: Unchanged(user.id),
            permissions: ActiveValue::Set(user.permissions),
            ..Default::default()
        }
        .update(txn)
        .await?;
        Ok(CreateSandboxResp {
            id: id.as_bytes().to_vec(),
        })
    }

    async fn update_sandbox(
        &self,
        req: UpdateSandboxReq,
        CurrentUser { user, .. }: CurrentUser,
        txn: &DatabaseTransaction,
    ) -> Result<UpdateSandboxResp, Error> {
        let id = req.id()?;
        if !user
            .permissions
            .has_perm(&id, EntityType::Sandbox, Verb::Write.into())
        {
            return Err(Error::Unauthorized);
        }
        sandbox::ActiveModel {
            id: Unchanged(id),
            name: Set(req.name),
            code: Set(req.code),
            ..Default::default()
        }
        .update(txn)
        .await?;
        Ok(UpdateSandboxResp {})
    }

    async fn boot_sandbox(
        &self,
        req: BootSandboxReq,
        CurrentUser { user, .. }: CurrentUser,
    ) -> Result<BootSandboxResp, Error> {
        let id = req.id()?;
        if !user
            .permissions
            .has_perm(&id, EntityType::Sandbox, Verb::Write.into())
        {
            return Err(Error::Unauthorized);
        }
        let Some(sandbox) = sandbox::Entity::find_by_id(id).one(&self.db).await? else {
            return Err(Error::NotFound);
        };
        let vm_id = if let Some(vm_id) = sandbox.vm_id {
            vm_id
        } else {
            let vm_id = Uuid::now_v7();
            vm::ActiveModel {
                id: Set(vm_id),
                pod_name: Set(vm_id.to_string()),
                status: Set(vm::Status::Pending),
            }
            .insert(&self.db)
            .await?;
            sandbox::ActiveModel {
                id: Unchanged(id),
                vm_id: Set(Some(vm_id)),
                ..Default::default()
            }
            .update(&self.db)
            .await?;
            vm_id
        };
        if let Some(vm) = vm::Entity::find_by_id(vm_id).one(&self.db).await? {
            if vm.status == vm::Status::Pending {
                self.vm_manager.spawn_vm(vm_id, "nginx".to_string()).await?;
            }
        }

        Ok(BootSandboxResp {})
    }

    async fn authed_route<Req, Resp, RespFuture>(
        &self,
        req: Request<Req>,
        handler: impl FnOnce(Req, Claims) -> RespFuture,
    ) -> Result<tonic::Response<Resp>, Status>
    where
        RespFuture: Future<Output = Result<Resp, Error>>,
    {
        let auth_header = req
            .metadata()
            .get("Authorization")
            .ok_or(Error::Unauthorized)?;
        let auth_header = auth_header.to_str().map_err(|_| Error::Unauthorized)?;
        let token = auth_header
            .split("Bearer ")
            .nth(1)
            .ok_or(Error::Unauthorized)?;
        let header = decode_header(token).map_err(|_| Error::Unauthorized)?;
        let kid = header.kid.ok_or(Error::Unauthorized)?;
        let Some(j) = self.auth0_keys.find(&kid) else {
            return Err(Error::Unauthorized.status());
        };

        let AlgorithmParameters::RSA(rsa) = &j.algorithm else {
            return Err(Error::Unauthorized.into());
        };

        let decoding_key = DecodingKey::from_rsa_components(&rsa.n, &rsa.e).unwrap();

        let mut validation = Validation::new(
            Algorithm::from_str(
                j.common
                    .key_algorithm
                    .ok_or_else(|| {
                        warn!("missing key algo field in jwks");
                        Error::Unauthorized
                    })?
                    .to_string()
                    .as_str(),
            )
            .map_err(|_| {
                warn!("invalid jwks algo");
                Error::Unauthorized
            })?,
        );
        validation.validate_exp = true;
        validation.set_audience(&[&self.config.auth0.client_id]);
        let claims = dbg!(decode::<Claims>(token, &decoding_key, &validation))
            .map_err(|_| Error::Unauthorized)?;

        handler(req.into_inner(), claims.claims)
            .await
            .map_err(Error::status)
            .map(Response::new)
    }
}

macro_rules! current_user_route_txn {
    ($self:ident, $req:ident, $handler:expr) => {
        $self
            .authed_route($req, move |req, claims| async move {
                let txn = $self.db.begin().await?;
                let user = User::find()
                    .filter(user::Column::Auth0Id.eq(&claims.sub))
                    .one(&txn)
                    .await?
                    .ok_or_else(|| Error::Unauthorized)?;
                let res = $handler($self, req, CurrentUser { user, claims }, &txn).await;
                if res.is_ok() {
                    txn.commit().await?;
                } else {
                    txn.rollback().await?;
                }
                res
            })
            .await
    };
}

macro_rules! current_user_route {
    ($self:ident, $req:ident, $handler:expr) => {
        $self
            .authed_route($req, move |req, claims| async move {
                let user = User::find()
                    .filter(user::Column::Auth0Id.eq(&claims.sub))
                    .one(&$self.db)
                    .await?
                    .ok_or_else(|| Error::Unauthorized)?;
                $handler($self, req, CurrentUser { user, claims }).await
            })
            .await
    };
}

#[allow(dead_code)]
#[derive(Debug)]
struct CurrentUser {
    user: user::Model,
    claims: Claims,
}

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    name: String,
}

#[async_trait]
impl api_server::Api for Api {
    async fn create_user(
        &self,
        req: tonic::Request<CreateUserReq>,
    ) -> Result<Response<CreateUserResp>, Status> {
        self.authed_route(req, |req, id| self.create_user(req, id))
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
}

#[allow(dead_code)]
async fn route<Req, Resp, RespFuture>(
    req: Request<Req>,
    handler: impl Fn(Req) -> RespFuture,
) -> Result<tonic::Response<Resp>, Status>
where
    RespFuture: Future<Output = Result<Resp, Error>>,
{
    handler(req.into_inner())
        .await
        .map_err(Error::status)
        .map(Response::new)
}

async fn get_keyset(domain: &str) -> Result<JwkSet, Error> {
    reqwest::get(&format!("https://{}/.well-known/jwks.json", domain))
        .await?
        .json()
        .await
        .map_err(Error::from)
}
