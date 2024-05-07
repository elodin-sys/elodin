use super::{Api, Claims, CurrentUser, UserInfo};
use crate::error::Error;
use atc_entity::events::DbExt;
use atc_entity::user::{self, Permissions};
use elodin_types::api::{CreateUserReq, CreateUserResp, CurrentUserResp};
use elodin_types::api::{LicenseType, UpdateUserReq, UpdateUserResp};
use sea_orm::{prelude::Uuid, ColumnTrait, EntityTrait, QueryFilter, Set};
use sea_orm::{DatabaseTransaction, Unchanged};
use serde_json::json;

impl Api {
    pub async fn current_user(&self, claims: Claims) -> Result<CurrentUserResp, Error> {
        let user = atc_entity::User::find()
            .filter(atc_entity::user::Column::Auth0Id.eq(&claims.sub))
            .one(&self.db)
            .await?
            .ok_or_else(|| Error::NotFound)?;
        let onboarding_data = serde_json::from_value(user.onboarding_data).ok();

        Ok(CurrentUserResp {
            id: user.id.as_bytes().to_vec(),
            email: user.email,
            name: user.name,
            avatar: user.avatar,
            license_type: LicenseType::from(user.license_type).into(),
            billing_account_id: user.billing_account_id.map(|id| id.as_bytes().to_vec()),
            onboarding_data,
        })
    }

    pub async fn create_user(
        &self,
        req: CreateUserReq,
        userinfo: UserInfo,
    ) -> Result<CreateUserResp, Error> {
        let mut redis = self.redis.clone();
        let id = Uuid::now_v7();
        let name = req.name.unwrap_or(userinfo.name);
        let email = req.email.unwrap_or(userinfo.email);
        user::ActiveModel {
            id: Set(id),
            email: Set(email),
            name: Set(name),
            auth0_id: Set(userinfo.sub),
            permissions: Set(Permissions::default()),
            avatar: Set(userinfo.picture),
            license_type: Set(user::LicenseType::None),
            monte_carlo_active: Set(false),
            onboarding_data: Set(json!({})),
            billing_account_id: Set(None),
        }
        .insert_with_event(&self.db, &mut redis)
        .await?;
        Ok(CreateUserResp {
            id: id.as_bytes().to_vec(),
        })
    }

    pub async fn update_user(
        &self,
        req: UpdateUserReq,
        CurrentUser { user, .. }: CurrentUser,
        txn: &DatabaseTransaction,
    ) -> Result<UpdateUserResp, Error> {
        let onboarding_data = if let Some(onboarding_data) = req.onboarding_data {
            let onboarding_data = serde_json::to_value(onboarding_data)?;
            sea_orm::Set(onboarding_data)
        } else {
            sea_orm::NotSet
        };
        let mut redis = self.redis.clone();
        user::ActiveModel {
            id: Unchanged(user.id),
            onboarding_data,
            ..Default::default()
        }
        .update_with_event(txn, &mut redis)
        .await?;
        Ok(UpdateUserResp {})
    }
}
