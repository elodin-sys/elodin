use super::{Api, Claims, CurrentUser, UserInfo};
use crate::error::Error;
use atc_entity::events::DbExt;
use atc_entity::user::{self, Permissions};
use elodin_types::api::{CreateUserReq, CreateUserResp, CurrentUserResp};
use elodin_types::api::{LicenseType, UpdateUserReq, UpdateUserResp};
use sea_orm::{ColumnTrait, EntityTrait, QueryFilter, Set, prelude::Uuid};
use sea_orm::{DatabaseTransaction, Unchanged};

impl Api {
    pub async fn current_user(&self, claims: Claims) -> Result<CurrentUserResp, Error> {
        let tracing_debug_span = tracing::debug_span!(
            "current_user",
            claims_sub = %claims.sub,
            user_id = tracing::field::Empty,
            user_email = tracing::field::Empty
        );
        tracing_debug_span.in_scope(|| {
            tracing::debug!("get current_user");
        });

        let user = atc_entity::User::find()
            .filter(atc_entity::user::Column::Auth0Id.eq(&claims.sub))
            .one(&self.db)
            .await?
            .ok_or_else(|| Error::NotFound)?;

        tracing_debug_span
            .record("user_id", user.id.to_string())
            .record("user_email", user.email.to_string());

        tracing_debug_span.in_scope(|| {
            tracing::debug!("get current_user - start");
        });

        let onboarding_data = user
            .onboarding_data
            .and_then(|data| serde_json::from_value(data).ok());

        let (billing_account_id, subscription_status) =
            if let Some(billing_account_id) = user.billing_account_id {
                let subscription_status = self
                    .get_subscription_status(billing_account_id, user.id, &self.db)
                    .await?;

                tracing_debug_span.in_scope(|| {
                    tracing::debug!(
                        billing_account_id = %billing_account_id,
                        subscription_end = ?subscription_status.subscription_end,
                        trial_start = ?subscription_status.trial_start,
                        trial_end = ?subscription_status.trial_start,
                        "get current_user - billing_account_id is ok"
                    );
                });

                (
                    Some(billing_account_id.as_bytes().to_vec()),
                    Some(subscription_status),
                )
            } else {
                tracing_debug_span.in_scope(|| {
                    tracing::warn!("get current_user - billing_account_id is missing");
                });

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
            monte_carlo_minutes_used: 0,
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

        let tracing_info_span =
            tracing::info_span!("create_user", user_id = %name, user_email = %email);

        tracing_info_span.in_scope(|| {
            tracing::info!("create user - start");
        });

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

        tracing_info_span.in_scope(|| {
            tracing::info!("create user - done");
        });

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
        tracing::info!(onboarding_data = ?req.onboarding_data, "update user");

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
