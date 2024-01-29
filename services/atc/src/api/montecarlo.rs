use crate::{api, error, montecarlo};

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
            manifest: sea_orm::Set(atc_entity::mc::Manifest {
                name: req.name,
                artifact_dir_uri: req.artifact_dir_uri,
                ..Default::default()
            }),
            ..Default::default()
        }
        .insert(txn)
        .await?;

        let run = montecarlo::Run {
            id: redmq::StringAdapter(mc_run.id),
            name: mc_run.manifest.name,
            samples: mc_run.samples as usize,
            batch_size: montecarlo::BATCH_SIZE,
            artifact_dir_uri: mc_run.manifest.artifact_dir_uri,
            start_time: redmq::StringAdapter(chrono::Utc::now()),
        };

        self.msg_queue
            .send(montecarlo::RUN_TOPIC, vec![run])
            .await?;

        Ok(CreateMonteCarloRunResp {
            id: mc_run.id.as_bytes().to_vec(),
        })
    }
}
