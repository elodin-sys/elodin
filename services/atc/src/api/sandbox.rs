use super::{utils::validate_auth_header, Api, CurrentUser, WSContext};
use crate::{error::Error, events::DbExt, sandbox::update_sandbox_code};
use atc_entity::{
    sandbox,
    user::{EntityType, Permission, Verb},
    vm,
};
use axum::{
    extract::{ws, Path, Query, State, WebSocketUpgrade},
    response::IntoResponse,
};
use enumflags2::BitFlag;
use futures::{StreamExt, TryFutureExt, TryStreamExt};
use paracosm_types::api::{
    BootSandboxReq, BootSandboxResp, CreateSandboxReq, CreateSandboxResp, GetSandboxReq,
    ListSandboxesReq, ListSandboxesResp, Page, Sandbox, UpdateSandboxReq, UpdateSandboxResp,
};
use sea_orm::{
    prelude::Uuid, ActiveValue, ColumnTrait, DatabaseTransaction, EntityTrait, QueryFilter, Set,
    Unchanged,
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
        CurrentUser { user, .. }: CurrentUser,
        txn: &DatabaseTransaction,
    ) -> Result<Sandbox, Error> {
        let id = req.id()?;
        if !user
            .permissions
            .has_perm(&id, EntityType::Sandbox, Verb::Read.into())
        {
            return Err(Error::NotFound);
        }
        let Some(sandbox) = atc_entity::sandbox::Entity::find_by_id(id).one(txn).await? else {
            return Err(Error::NotFound);
        };
        Ok(Sandbox {
            id: sandbox.id.as_bytes().to_vec(),
            name: sandbox.name,
            code: sandbox.code,
        })
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
        let sandboxes = res
            .into_iter()
            .map(|s| Sandbox {
                id: s.id.as_bytes().to_vec(),
                name: s.name,
                code: s.code,
            })
            .collect::<Vec<_>>();
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
        CurrentUser { mut user, .. }: CurrentUser,
        txn: &DatabaseTransaction,
    ) -> Result<CreateSandboxResp, Error> {
        let mut redis = self.redis.clone();
        let id = Uuid::now_v7();
        sandbox::ActiveModel {
            id: Set(id),
            user_id: Set(user.id),
            name: Set(req.name),
            code: Set(req.code),
            status: Set(sandbox::Status::Off),
            vm_id: Set(None),
        }
        .insert_with_event(txn, &mut redis)
        .await?;
        user.permissions
            .insert(id, Permission::new(EntityType::Sandbox, Verb::all()));
        atc_entity::user::ActiveModel {
            id: Unchanged(user.id),
            permissions: ActiveValue::Set(user.permissions),
            ..Default::default()
        }
        .update_with_event(txn, &mut redis)
        .await?;
        Ok(CreateSandboxResp {
            id: id.as_bytes().to_vec(),
        })
    }

    pub async fn update_sandbox(
        &self,
        req: UpdateSandboxReq,
        CurrentUser { user, .. }: CurrentUser,
        txn: &DatabaseTransaction,
    ) -> Result<UpdateSandboxResp, Error> {
        let mut redis = self.redis.clone();
        let id = req.id()?;
        if !user
            .permissions
            .has_perm(&id, EntityType::Sandbox, Verb::Write.into())
        {
            return Err(Error::Unauthorized);
        }
        let sandbox = sandbox::ActiveModel {
            id: Unchanged(id),
            name: Set(req.name),
            code: Set(req.code),
            ..Default::default()
        }
        .update_with_event(txn, &mut redis)
        .await?;
        if let Some(vm_id) = sandbox.vm_id {
            if let Some(vm_ip) = vm::Entity::find_by_id(vm_id)
                .one(txn)
                .await?
                .and_then(|vm| vm.pod_ip)
            {
                update_sandbox_code(&vm_ip, sandbox.code).await?;
            }
        }
        Ok(UpdateSandboxResp {})
    }

    pub async fn boot_sandbox(
        &self,
        req: BootSandboxReq,
        CurrentUser { user, .. }: CurrentUser,
    ) -> Result<BootSandboxResp, Error> {
        let mut redis = self.redis.clone();
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
        if sandbox.vm_id.is_some() {
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
            ..Default::default()
        }
        .update_with_event(&self.db, &mut redis)
        .await?;

        Ok(BootSandboxResp {})
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
    let claims = validate_auth_header(
        &auth.token,
        &context.auth_context.auth_config.client_id,
        &context.auth_context.auth0_keys,
    )?;
    let user = atc_entity::User::find()
        .filter(atc_entity::user::Column::Auth0Id.eq(&claims.sub))
        .one(&context.db)
        .await?
        .ok_or_else(|| Error::Unauthorized)?;
    if !user
        .permissions
        .has_perm(&sandbox_id, EntityType::Sandbox, Verb::Read.into())
    {
        return Err(Error::NotFound);
    }
    let Some(sandbox) = atc_entity::sandbox::Entity::find_by_id(sandbox_id)
        .one(&context.db)
        .await?
    else {
        return Err(Error::NotFound);
    };
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
    let sim_stream = sim_socket.connect(ip).await?;
    let (rx, tx) = sim_stream.into_split();
    let sim_tx = FramedWrite::new(tx, LengthDelimitedCodec::new());
    let sim_rx = FramedRead::new(rx, LengthDelimitedCodec::new());

    Ok(ws.on_upgrade(move |socket| async move {
        let (ws_tx, ws_rx) = socket.split();
        let ws_rx = ws_rx
            .try_filter_map(|msg| async move {
                let ws::Message::Binary(bytes) = msg else {
                    return Ok(None);
                };
                Ok(Some(Bytes::from(bytes)))
            })
            .map_err(|err| std::io::Error::new(io::ErrorKind::Other, err));
        let ws_to_sim = ws_rx.forward(sim_tx).map_err(Error::from);
        let sim_to_ws = sim_rx
            .map(|m| m.map(|b| ws::Message::Binary(b.to_vec())))
            .map_err(|err| axum::Error::new(err))
            .forward(ws_tx)
            .map_err(Error::from);
        let res = tokio::select! {
            res = ws_to_sim => { res }
            res = sim_to_ws=> { res }
        };
        if let Err(err) = res {
            trace!(?err, "error in sim proxy");
        }
    }))
}
