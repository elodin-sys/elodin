use super::{utils::validate_auth_header, Api, CurrentUser, WSContext};
use crate::{error::Error, sandbox::update_sandbox_code};
use atc_entity::{
    events::DbExt,
    sandbox::{self},
    user::{EntityType, Permission, Verb},
    vm,
};
use axum::{
    extract::{ws, Path, Query, State, WebSocketUpgrade},
    response::IntoResponse,
};
use chrono::Utc;
use elodin_types::api::{
    api_server, BootSandboxReq, BootSandboxResp, CreateSandboxReq, CreateSandboxResp,
    GetSandboxReq, ListSandboxesReq, ListSandboxesResp, Page, Sandbox, UpdateSandboxReq,
    UpdateSandboxResp,
};
use enumflags2::BitFlag;
use futures::{StreamExt, TryFutureExt, TryStreamExt};
use sea_orm::{
    prelude::Uuid, ActiveValue, ColumnTrait, DatabaseTransaction, EntityTrait, NotSet, QueryFilter,
    Set, Unchanged,
};
use serde::Deserialize;
use std::io;
use tokio::net::TcpSocket;
use tokio_util::{
    bytes::Bytes,
    codec::{FramedRead, FramedWrite, LengthDelimitedCodec},
};
use tracing::trace;

impl Api {
    pub async fn get_sandbox(
        &self,
        req: GetSandboxReq,
        user: Option<CurrentUser>,
        txn: &DatabaseTransaction,
    ) -> Result<Sandbox, Error> {
        let id = req.id()?;
        let Some(sandbox) = atc_entity::sandbox::Entity::find_by_id(id).one(txn).await? else {
            return Err(Error::NotFound);
        };
        if sandbox.user_id.is_some() {
            let CurrentUser { user, .. } = user.ok_or(Error::Unauthorized)?;
            if !user
                .permissions
                .has_perm(&id, EntityType::Sandbox, Verb::Read.into())
                && !sandbox.public
            {
                return Err(Error::NotFound);
            }
        }
        Ok(sandbox.into())
    }

    pub async fn list_sandbox(
        &self,
        req: ListSandboxesReq,
        CurrentUser { user, .. }: CurrentUser,
        txn: &DatabaseTransaction,
    ) -> Result<ListSandboxesResp, Error> {
        let ids: Vec<Uuid> = if let Some(ref page) = req.page {
            if page.last_id.is_empty() {
                user.permissions
                    .0
                    .iter()
                    .filter(|(_, p)| p.entity_type == EntityType::Sandbox)
                    .map(|(id, _)| id)
                    .take(page.count as usize)
                    .copied()
                    .collect()
            } else {
                let last_id = page.last_id()?;
                user.permissions
                    .0
                    .range(last_id..)
                    .filter(|(_, p)| p.entity_type == EntityType::Sandbox)
                    .map(|(id, _)| id)
                    .take(page.count as usize)
                    .copied()
                    .collect()
            }
        } else {
            user.permissions
                .0
                .iter()
                .filter(|(_, p)| p.entity_type == EntityType::Sandbox)
                .map(|(id, _)| id)
                .copied()
                .collect()
        };
        let res = sandbox::Entity::find()
            .filter(sandbox::Column::Id.is_in(ids))
            .all(txn)
            .await?;
        let sandboxes = res.into_iter().map(Sandbox::from).collect::<Vec<_>>();
        let next_page = if let Some(last_sandbox) = sandboxes.last() {
            if let Some(page) = req.page {
                let last_id = last_sandbox.id.clone();
                Some(Page {
                    last_id,
                    count: page.count,
                })
            } else {
                None
            }
        } else {
            None
        };
        Ok(ListSandboxesResp {
            sandboxes,
            next_page,
        })
    }

    pub async fn create_sandbox(
        &self,
        req: CreateSandboxReq,
        user: Option<CurrentUser>,
        txn: &DatabaseTransaction,
    ) -> Result<CreateSandboxResp, Error> {
        let code = match req.template.as_deref() {
            Some("three-body") => {
                include_str!("../../../../libs/nox-py/examples/three-body.py").to_string()
            }
            Some("cube-sat") => {
                include_str!("../../../../libs/nox-py/examples/cube-sat.py").to_string()
            }
            Some(_) | None => req.code,
        };
        let mut redis = self.redis.clone();
        let id = Uuid::now_v7();
        let user_id = user.as_ref().map(|u| u.user.id);
        sandbox::ActiveModel {
            id: Set(id),
            user_id: Set(user_id),
            name: Set(req.name),
            code: Set(code.clone()),
            draft_code: Set(code),
            status: Set(sandbox::Status::Off),
            vm_id: Set(None),
            public: Set(false),
            last_used: ActiveValue::Set(Utc::now()),
        }
        .insert_with_event(txn, &mut redis)
        .await?;
        if let Some(CurrentUser { mut user, .. }) = user {
            user.permissions
                .insert(id, Permission::new(EntityType::Sandbox, Verb::all()));
            atc_entity::user::ActiveModel {
                id: Unchanged(user.id),
                permissions: ActiveValue::Set(user.permissions),
                ..Default::default()
            }
            .update_with_event(txn, &mut redis)
            .await?;
        }
        Ok(CreateSandboxResp {
            id: id.as_bytes().to_vec(),
        })
    }

    pub async fn update_sandbox(
        &self,
        req: UpdateSandboxReq,
        user: Option<CurrentUser>,
        txn: &DatabaseTransaction,
    ) -> Result<UpdateSandboxResp, Error> {
        let mut redis = self.redis.clone();
        let id = req.id()?;
        let Some(sandbox) = sandbox::Entity::find_by_id(id).one(&self.db).await? else {
            return Err(Error::NotFound);
        };
        if sandbox.user_id.is_some() {
            let CurrentUser { user, .. } = user.ok_or(Error::Unauthorized)?;

            if !user
                .permissions
                .has_perm(&id, EntityType::Sandbox, Verb::Write.into())
            {
                return Err(Error::Unauthorized);
            }
        }
        let sandbox = sandbox::ActiveModel {
            id: Unchanged(id),
            name: Set(req.name),
            code: req.code.clone().map(Set).unwrap_or(NotSet),
            draft_code: req.draft_code.map(Set).unwrap_or(NotSet),
            last_used: Set(Utc::now()),
            public: Set(req.public),
            ..Default::default()
        }
        .update_with_event(txn, &mut redis)
        .await?;
        let Some(code) = req.code else {
            return Ok(UpdateSandboxResp::default());
        };
        let Some(vm_id) = sandbox.vm_id else {
            return Ok(UpdateSandboxResp::default());
        };
        let Some(vm_ip) = vm::Entity::find_by_id(vm_id)
            .one(txn)
            .await?
            .and_then(|vm| vm.pod_ip)
        else {
            return Ok(UpdateSandboxResp::default());
        };
        let resp = update_sandbox_code(&vm_ip, code).await?;
        if resp.status() == elodin_types::sandbox::Status::Error {
            return Ok(UpdateSandboxResp {
                errors: resp.errors,
            });
        }
        Ok(UpdateSandboxResp::default())
    }

    pub async fn boot_sandbox(
        &self,
        req: BootSandboxReq,
        user: Option<CurrentUser>,
    ) -> Result<BootSandboxResp, Error> {
        let mut redis = self.redis.clone();
        let id = req.id()?;
        let Some(sandbox) = sandbox::Entity::find_by_id(id).one(&self.db).await? else {
            return Err(Error::NotFound);
        };

        if sandbox.user_id.is_some() {
            let CurrentUser { user, .. } = user.ok_or(Error::Unauthorized)?;

            if !user
                .permissions
                .has_perm(&id, EntityType::Sandbox, Verb::Write.into())
                && !sandbox.public
            {
                return Err(Error::Unauthorized);
            }
        }

        if sandbox.vm_id.is_some() {
            sandbox::ActiveModel {
                id: Unchanged(id),
                last_used: ActiveValue::Set(Utc::now()),
                ..Default::default()
            }
            .update_with_event(&self.db, &mut redis)
            .await?;
            return Ok(BootSandboxResp {});
        }
        let vm_id = Uuid::now_v7();
        vm::ActiveModel {
            id: Set(vm_id),
            pod_name: Set(vm_id.to_string()),
            status: Set(vm::Status::Pending),
            sandbox_id: Set(Some(id)),
            ..Default::default()
        }
        .insert_with_event(&self.db, &mut redis)
        .await?;
        sandbox::ActiveModel {
            id: Unchanged(id),
            vm_id: Set(Some(vm_id)),
            last_used: ActiveValue::Set(Utc::now()),
            ..Default::default()
        }
        .update_with_event(&self.db, &mut redis)
        .await?;

        Ok(BootSandboxResp {})
    }

    pub async fn sandbox_events(
        &self,
        req: GetSandboxReq,
        user: Option<CurrentUser>,
    ) -> Result<<Api as api_server::Api>::SandboxEventsStream, Error> {
        let id = req.id()?;

        let Some(sandbox) = atc_entity::sandbox::Entity::find_by_id(id)
            .one(&self.db)
            .await?
        else {
            return Err(Error::NotFound);
        };

        if sandbox.user_id.is_some() {
            let CurrentUser { user, .. } = user.ok_or(Error::Unauthorized)?;
            if !user
                .permissions
                .has_perm(&id, EntityType::Sandbox, Verb::Read.into())
                && !sandbox.public
            {
                return Err(Error::NotFound);
            }
        }

        let sandbox_events = self.sandbox_events.resubscribe();
        let stream = tokio_stream::wrappers::BroadcastStream::new(sandbox_events);
        let stream = stream.filter_map(move |res| async move {
            if let Ok(event) = res {
                let model = event.into_model();
                if model.id == id {
                    return Some(Ok(model.into()));
                }
            }
            None
        });
        Ok(Box::pin(stream))
    }
}

#[derive(Deserialize)]
pub struct WsAuth {
    token: String,
}

pub async fn sim_socket(
    ws: WebSocketUpgrade,
    Path(sandbox_id): Path<Uuid>,
    State(context): State<WSContext>,
    auth: Query<WsAuth>,
) -> Result<impl IntoResponse, Error> {
    trace!(?sandbox_id, "sandbox id");
    let Some(sandbox) = atc_entity::sandbox::Entity::find_by_id(sandbox_id)
        .one(&context.db)
        .await?
    else {
        return Err(Error::NotFound);
    };

    if sandbox.user_id.is_some() {
        if let Ok(claims) = validate_auth_header(
            &auth.token,
            &context.auth_context.auth_config.domain,
            &context.auth_context.auth0_keys,
        ) {
            let user = atc_entity::User::find()
                .filter(atc_entity::user::Column::Auth0Id.eq(&claims.sub))
                .one(&context.db)
                .await?
                .ok_or_else(|| Error::Unauthorized)?;
            if !user
                .permissions
                .has_perm(&sandbox_id, EntityType::Sandbox, Verb::Read.into())
                && !sandbox.public
            {
                return Err(Error::NotFound);
            }
        } else {
            return Err(Error::Unauthorized);
        }
    }
    trace!(?sandbox, "found sandbox");
    let Some(vm_id) = sandbox.vm_id else {
        return Err(Error::SandboxNotBooted);
    };
    let Some(vm) = vm::Entity::find_by_id(vm_id).one(&context.db).await? else {
        return Err(Error::SandboxNotBooted);
    };
    let Some(pod_ip) = vm.pod_ip else {
        return Err(Error::SandboxNotBooted);
    };
    let sim_socket = TcpSocket::new_v4()?;
    let Ok(ip) = format!("{}:3563", pod_ip).parse() else {
        return Err(Error::VMBootFailed("vm has invalid ip".to_string()));
    };
    tracing::debug!(%ip, "connecting to sim-agent");
    let sim_stream = sim_socket.connect(ip).await?;
    let (rx, tx) = sim_stream.into_split();
    let sim_tx = FramedWrite::new(tx, LengthDelimitedCodec::new());
    let sim_rx = FramedRead::new(rx, LengthDelimitedCodec::new());

    Ok(ws.on_upgrade(move |socket| async move {
        tracing::debug!(%ip, "upgraded to websocket");
        let (ws_tx, ws_rx) = socket.split();
        let ws_rx = ws_rx
            .try_filter_map(|msg| async move {
                let ws::Message::Binary(bytes) = msg else {
                    return Ok(None);
                };
                Ok(Some(Bytes::from(bytes)))
            })
            .map_err(|err| std::io::Error::new(io::ErrorKind::Other, err));
        let ws_to_sim = ws_rx
            .inspect_ok(|b| {
                tracing::trace!(bytes = b.len(), "ws -> sim");
            })
            .forward(sim_tx)
            .map_err(Error::from);
        let sim_to_ws = sim_rx
            .inspect_ok(|b| {
                tracing::trace!(bytes = b.len(), "sim -> ws");
            })
            .map(|m| m.map(|b| ws::Message::Binary(b.to_vec())))
            .map_err(axum::Error::new)
            .forward(ws_tx)
            .map_err(Error::from);
        let res = tokio::select! {
            res = ws_to_sim => { res }
            res = sim_to_ws=> { res}
        };
        if let Err(err) = res {
            tracing::error!(?err, "error in sim proxy");
        }
    }))
}
