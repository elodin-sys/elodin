use super::{Api, Claims};
use crate::{error::Error, events::DbExt};
use atc_entity::user::{self, Permissions};
use paracosm_types::api::{CreateUserReq, CreateUserResp, CurrentUserResp};
use sea_orm::{prelude::Uuid, ColumnTrait, EntityTrait, QueryFilter, Set};

impl Api {
    pub async fn current_user(&self, claims: Claims) -> Result<CurrentUserResp, Error> {
        let user = atc_entity::User::find()
            .filter(atc_entity::user::Column::Auth0Id.eq(&claims.sub))
            .one(&self.db)
            .await?
            .ok_or_else(|| Error::NotFound)?;

        Ok(CurrentUserResp {
            id: user.id.as_bytes().to_vec(),
            email: user.email,
            name: user.name,
        })
    }

    pub async fn create_user(
        &self,
        req: CreateUserReq,
        claims: Claims,
    ) -> Result<CreateUserResp, Error> {
        let mut redis = self.redis.clone();
        let id = Uuid::now_v7();
        let name = req.name.unwrap_or(claims.name);
        let email = req.email.unwrap_or(claims.email);
        user::ActiveModel {
            id: Set(id),
            email: Set(email),
            name: Set(name),
            auth0_id: Set(claims.sub),
            permissions: Set(Permissions::default()),
        }
        .insert_with_event(&self.db, &mut redis)
        .await?;
        Ok(CreateUserResp {
            id: id.as_bytes().to_vec(),
        })
    }
}
