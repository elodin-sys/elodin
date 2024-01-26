use crate::{api, error};

use elodin_types::api::*;
use sea_orm::prelude::*;

impl api::Api {
    pub async fn create_monte_carlo_run(
        &self,
        req: CreateMonteCarloRunReq,
        api::CurrentUser { user, .. }: api::CurrentUser,
        txn: &sea_orm::DatabaseTransaction,
    ) -> Result<CreateMonteCarloRunResp, error::Error> {
        // # of samples must be a multiple of 100 for easy math
        if (req.samples % 100 != 0) || (req.samples > 100_000) {
            return Err(error::Error::InvalidRequest);
        }
        let samples = i32::try_from(req.samples).map_err(|_| error::Error::InvalidRequest)?;
        let mc_run = atc_entity::mc::ActiveModel {
            id: sea_orm::Set(Uuid::now_v7()),
            user_id: sea_orm::Set(user.id),
            samples: sea_orm::Set(samples),
            manifest: sea_orm::Set(atc_entity::mc::Manifest {
                name: req.name,
                artifact_dir_uri: req.artifact_dir_uri,
                ..Default::default()
            }),
            ..Default::default()
        }
        .insert(txn)
        .await?;
        Ok(CreateMonteCarloRunResp {
            id: mc_run.id.as_bytes().to_vec(),
        })
    }
}
