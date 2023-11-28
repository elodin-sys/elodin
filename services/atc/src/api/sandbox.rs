use super::{Api, CurrentUser};
use crate::{error::Error, events::DbExt};
use atc_entity::{
    sandbox,
    user::{EntityType, Permission, Verb},
    vm,
};
use enumflags2::BitFlag;
use paracosm_types::api::{
    BootSandboxReq, BootSandboxResp, CreateSandboxReq, CreateSandboxResp, GetSandboxReq,
    ListSandboxesReq, ListSandboxesResp, Page, Sandbox, UpdateSandboxReq, UpdateSandboxResp,
};
use sea_orm::{
    prelude::Uuid, ActiveValue, ColumnTrait, DatabaseTransaction, EntityTrait, QueryFilter, Set,
    Unchanged,
};

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
        sandbox::ActiveModel {
            id: Unchanged(id),
            name: Set(req.name),
            code: Set(req.code),
            ..Default::default()
        }
        .update_with_event(txn, &mut redis)
        .await?;
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
