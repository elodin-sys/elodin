use crate::{api, error, monte_carlo};

use atc_entity::mc;
use elodin_types::api::*;
use sea_orm::prelude::*;

impl api::Api {
    pub async fn create_monte_carlo_run(
        &self,
        req: CreateMonteCarloRunReq,
        api::CurrentUser { user, .. }: api::CurrentUser,
        txn: &sea_orm::DatabaseTransaction,
    ) -> Result<CreateMonteCarloRunResp, error::Error> {
        tracing::debug!(%user.id, "create monte carlo run");

        if req.samples > monte_carlo::MAX_SAMPLE_COUNT as u32 {
            return Err(error::Error::InvalidRequest);
        }
        let samples = i32::try_from(req.samples).map_err(|_| error::Error::InvalidRequest)?;
        let mc_run = atc_entity::mc::ActiveModel {
            id: sea_orm::Set(Uuid::now_v7()),
            user_id: sea_orm::Set(user.id),
            samples: sea_orm::Set(samples),
            name: sea_orm::Set(req.name),
            status: sea_orm::Set(mc::Status::Pending),
            metadata: sea_orm::Set(Json::Null),
        }
        .insert(txn)
        .await?;
        tracing::debug!(%user.id, id = %mc_run.id, "created monte carlo run");

        Ok(CreateMonteCarloRunResp {
            id: mc_run.id.as_bytes().to_vec(),
        })
    }

    pub async fn start_monte_carlo_run(
        &self,
        req: StartMonteCarloRunReq,
        api::CurrentUser { user, .. }: api::CurrentUser,
        txn: &sea_orm::DatabaseTransaction,
    ) -> Result<StartMonteCarloRunResp, error::Error> {
        let id = req.id()?;
        tracing::debug!(%user.id, %id, "start monte carlo run");

        let mc_run = atc_entity::MonteCarloRun::find_by_id(id)
            .filter(mc::Column::UserId.eq(user.id))
            .one(txn)
            .await?
            .ok_or(error::Error::NotFound)?;

        // change status from pending -> running
        if mc_run.status != mc::Status::Pending {
            return Err(error::Error::InvalidRequest);
        }
        let mut mc_run: mc::ActiveModel = mc_run.into();
        mc_run.status = sea_orm::Set(mc::Status::Running);
        let mc_run = mc_run.update(txn).await?;

        let mc_run_msg = monte_carlo::Run {
            id: mc_run.id,
            name: mc_run.name,
            samples: mc_run.samples as usize,
            batch_size: monte_carlo::BATCH_SIZE,
            start_time: chrono::Utc::now(),
        };
        self.msg_queue
            .send(monte_carlo::RUN_TOPIC, vec![mc_run_msg])
            .await?;
        tracing::debug!(%user.id, %id, "started monte carlo run");

        Ok(StartMonteCarloRunResp {})
    }
}
