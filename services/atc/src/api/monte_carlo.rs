use crate::{api, error, monte_carlo};

use atc_entity::mc_run;
use elodin_types::{api::*, Run, RUN_TOPIC};
use sea_orm::prelude::*;

impl api::Api {
    pub async fn create_monte_carlo_run(
        &self,
        req: CreateMonteCarloRunReq,
        api::CurrentUser { user, .. }: api::CurrentUser,
        txn: &sea_orm::DatabaseTransaction,
    ) -> Result<CreateMonteCarloRunResp, error::Error> {
        tracing::debug!(%user.id, "create monte carlo run");

        let id = Uuid::now_v7();
        let upload_url = self.sim_storage_client.signed_upload_url(id).await?;

        if req.samples > monte_carlo::MAX_SAMPLE_COUNT as u32 {
            return Err(error::Error::InvalidRequest);
        }
        let samples = i32::try_from(req.samples).map_err(|_| error::Error::InvalidRequest)?;
        let max_duration =
            i64::try_from(req.max_duration).map_err(|_| error::Error::InvalidRequest)?;
        let mc_run = atc_entity::mc_run::ActiveModel {
            id: sea_orm::Set(id),
            user_id: sea_orm::Set(user.id),
            samples: sea_orm::Set(samples),
            name: sea_orm::Set(req.name),
            status: sea_orm::Set(mc_run::Status::Pending),
            metadata: sea_orm::Set(Json::Null),
            max_duration: sea_orm::Set(max_duration),
            started: sea_orm::Set(None),
        }
        .insert(txn)
        .await?;
        tracing::debug!(%user.id, id = %mc_run.id, "created monte carlo run");

        Ok(CreateMonteCarloRunResp {
            id: mc_run.id.as_bytes().to_vec(),
            upload_url,
        })
    }

    pub async fn start_monte_carlo_run(
        &self,
        req: StartMonteCarloRunReq,
        api::CurrentUser { user, .. }: api::CurrentUser,
        txn: &sea_orm::DatabaseTransaction,
    ) -> Result<StartMonteCarloRunResp, error::Error> {
        let id = req.id()?;
        let start_time = chrono::Utc::now();
        tracing::debug!(%user.id, %id, "start monte carlo run");

        let mc_run = atc_entity::MonteCarloRun::find_by_id(id)
            .filter(mc_run::Column::UserId.eq(user.id))
            .one(txn)
            .await?
            .ok_or(error::Error::NotFound)?;

        // change status from pending -> running
        if mc_run.status != mc_run::Status::Pending {
            return Err(error::Error::InvalidRequest);
        }
        let mut mc_run: mc_run::ActiveModel = mc_run.into();
        mc_run.status = sea_orm::Set(mc_run::Status::Running);
        mc_run.started = sea_orm::Set(Some(start_time));
        let mc_run = mc_run.update(txn).await?;

        let mc_run_msg = Run {
            id: mc_run.id,
            name: mc_run.name,
            samples: mc_run.samples as usize,
            batch_size: monte_carlo::BATCH_SIZE,
            start_time,
            max_duration: mc_run.max_duration as u64,
        };
        self.msg_queue.send(RUN_TOPIC, vec![mc_run_msg]).await?;
        tracing::debug!(%user.id, %id, "started monte carlo run");

        Ok(StartMonteCarloRunResp {})
    }
}
