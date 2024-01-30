use crate::{api, error, montecarlo};

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
        if req.samples > montecarlo::MAX_SAMPLE_COUNT as u32 {
            return Err(error::Error::InvalidRequest);
        }
        let samples = i32::try_from(req.samples).map_err(|_| error::Error::InvalidRequest)?;
        let mc_run = atc_entity::mc::ActiveModel {
            id: sea_orm::Set(Uuid::now_v7()),
            user_id: sea_orm::Set(user.id),
            samples: sea_orm::Set(samples),
            name: sea_orm::Set(req.name),
            ..Default::default()
        }
        .insert(txn)
        .await?;

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

        let mc_run_msg = montecarlo::Run {
            id: redmq::StringAdapter(mc_run.id),
            name: mc_run.name,
            samples: mc_run.samples as usize,
            batch_size: montecarlo::BATCH_SIZE,
            start_time: redmq::StringAdapter(chrono::Utc::now()),
        };
        self.msg_queue
            .send(montecarlo::RUN_TOPIC, vec![mc_run_msg])
            .await?;

        Ok(StartMonteCarloRunResp {})
    }
}
