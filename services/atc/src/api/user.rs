use super::{Api, Claims, CurrentUser, UserInfo};
use crate::error::Error;
use atc_entity::events::DbExt;
use atc_entity::user::{self, Permissions};
use elodin_types::api::{CreateUserReq, CreateUserResp, CurrentUserResp};
use elodin_types::api::{LicenseType, UpdateUserReq, UpdateUserResp};
use sea_orm::{prelude::Uuid, ColumnTrait, EntityTrait, QueryFilter, Set};
use sea_orm::{DatabaseTransaction, Unchanged};

impl Api {
    pub async fn current_user(&self, claims: Claims) -> Result<CurrentUserResp, Error> {
        let user = atc_entity::User::find()
            .filter(atc_entity::user::Column::Auth0Id.eq(&claims.sub))
            .one(&self.db)
            .await?
            .ok_or_else(|| Error::NotFound)?;

        let onboarding_data = user
            .onboarding_data
            .and_then(|data| serde_json::from_value(data).ok());

        let (billing_account_id, subscription_status) =
            if let Some(billing_account_id) = user.billing_account_id {
                (
                    Some(billing_account_id.as_bytes().to_vec()),
                    Some(
                        self.get_subscription_status(billing_account_id, user.id, &self.db)
                            .await?,
                    ),
                )
            } else {
                (None, None)
            };

        Ok(CurrentUserResp {
            id: user.id.as_bytes().to_vec(),
            email: user.email,
            name: user.name,
            avatar: user.avatar,
            license_type: LicenseType::from(user.license_type).into(),
            billing_account_id,
            subscription_status,
            onboarding_data,
        })
    }

    pub async fn create_user(
        &self,
        req: CreateUserReq,
        userinfo: UserInfo,
    ) -> Result<CreateUserResp, Error> {
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
            onboarding_data: Set(None),
            billing_account_id: Set(None),
        }
        .insert_with_event(&self.db, &self.redis)
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
            sea_orm::Set(Some(onboarding_data))
        } else {
            sea_orm::NotSet
        };
        user::ActiveModel {
            id: Unchanged(user.id),
            onboarding_data,
            ..Default::default()
        }
        .update_with_event(txn, &self.redis)
        .await?;
        Ok(UpdateUserResp {})
    }
}
