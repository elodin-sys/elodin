use crate::error::Error;
use crate::monte_carlo::{BATCH_SIZE, MAX_SAMPLE_COUNT};
use crate::{api, error};

use atc_entity::{batches, mc_run};
use chrono::{DateTime, Months};
use elodin_types::{api::*, Batch, BATCH_TOPIC};
use futures::StreamExt;
use sea_orm::{prelude::*, FromQueryResult, JoinType, QueryOrder, QuerySelect};
use serde::{Deserialize, Serialize};

impl api::Api {
    pub async fn list_monte_carlo_runs(
        &self,
        _req: ListMonteCarloRunsReq,
        api::CurrentUser { user, .. }: api::CurrentUser,
    ) -> Result<ListMonteCarloRunsResp, error::Error> {
        tracing::debug!("list monte-carlo runs");

        let mc_runs = atc_entity::MonteCarloRun::find()
            .filter(mc_run::Column::UserId.eq(user.id))
            .all(&self.db)
            .await?
            .into_iter()
            .map(MonteCarloRun::from)
            .collect::<Vec<_>>();

        tracing::debug!(mc_run_count = ?mc_runs.len(), "list monte-carlo runs - done");

        Ok(ListMonteCarloRunsResp {
            monte_carlo_runs: mc_runs,
        })
    }

    pub async fn create_monte_carlo_run(
        &self,
        req: CreateMonteCarloRunReq,
        api::CurrentUser { user, .. }: api::CurrentUser,
        txn: &sea_orm::DatabaseTransaction,
    ) -> Result<CreateMonteCarloRunResp, error::Error> {
        tracing::debug!("create monte carlo run");

        let id = Uuid::now_v7();
        let upload_url = self.sim_storage_client.upload_artifacts_url(id).await?;

        if req.samples > MAX_SAMPLE_COUNT as u32 {
            return Err(error::Error::InvalidRequest);
        }
        let samples = i32::try_from(req.samples).map_err(|_| error::Error::InvalidRequest)?;
        let max_duration =
            i64::try_from(req.max_duration).map_err(|_| error::Error::InvalidRequest)?;
        let metadata = serde_json::from_str::<Json>(&req.metadata)
            .map_err(|_| error::Error::InvalidRequest)?;
        let mc_run = atc_entity::mc_run::ActiveModel {
            id: sea_orm::Set(id),
            user_id: sea_orm::Set(user.id),
            samples: sea_orm::Set(samples),
            name: sea_orm::Set(req.name),
            status: sea_orm::Set(mc_run::Status::Pending),
            metadata: sea_orm::Set(metadata),
            max_duration: sea_orm::Set(max_duration),
            started: sea_orm::Set(None),
            billed: sea_orm::Set(false),
        }
        .insert(txn)
        .await?;
        let batch_size = BATCH_SIZE as i32;
        let batch_count = (samples + batch_size - 1) / batch_size;
        for i in 0..batch_count {
            let samples = batch_size.min(samples);
            let byte_count = (samples as usize).div_ceil(8);
            let failures = vec![0; byte_count];
            atc_entity::batches::ActiveModel {
                run_id: sea_orm::Set(id),
                batch_number: sea_orm::Set(i),
                samples: sea_orm::Set(samples),
                failures: sea_orm::Set(failures),
                finished: sea_orm::Set(None),
                status: sea_orm::Set(batches::Status::Pending),
                runtime: sea_orm::Set(0),
            }
            .insert(txn)
            .await?;
        }
        tracing::debug!(mc_run_id = ?id, "create monte carlo run - done");

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
        tracing::debug!(mc_run_id = ?id, "start monte carlo run");

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

        self.spawn_batches(&mc_run).await?;
        tracing::debug!(mc_run_id = ?id, "start monte carlo run - done");
        Ok(StartMonteCarloRunResp {})
    }

    async fn spawn_batches(&self, run: &mc_run::Model) -> Result<(), error::Error> {
        tracing::debug!(mc_run_id = ?run.id, mc_run_name = ?run.name, "spawn monte carlo run batches");
        let samples = run.samples as usize;
        let batch_count = samples.div_ceil(BATCH_SIZE);
        let batches = (0..batch_count)
            .map(|batch_no| Batch {
                run_id: run.id,
                batch_no,
                buffer: false,
            })
            .collect();
        self.msg_queue.send(BATCH_TOPIC, batches).await?;
        Ok(())
    }

    pub async fn get_monte_carlo_run(
        &self,
        req: GetMonteCarloRunReq,
        api::CurrentUser { user, .. }: api::CurrentUser,
    ) -> Result<MonteCarloRun, error::Error> {
        let id = req.id()?;

        tracing::debug!(mc_run_id = ?id, "get monte-carlo run");

        let mc_run = atc_entity::MonteCarloRun::find_by_id(id)
            .filter(mc_run::Column::UserId.eq(user.id))
            .one(&self.db)
            .await?
            .ok_or(error::Error::NotFound)?;
        let batches = atc_entity::Batches::find()
            .filter(batches::Column::RunId.eq(id))
            .order_by_asc(batches::Column::BatchNumber)
            .all(&self.db)
            .await?;
        let batches = batches
            .into_iter()
            .map(|b| b.into())
            .collect::<Vec<MonteCarloBatch>>();

        tracing::debug!(mc_run_id = ?id, batch_count = ?batches.len(), "get monte-carlo run - done");

        Ok(MonteCarloRun {
            batches,
            ..mc_run.into()
        })
    }

    pub async fn monte_carlo_run_events(
        &self,
        req: GetMonteCarloRunReq,
        api::CurrentUser { user, .. }: api::CurrentUser,
    ) -> Result<<super::Api as api_server::Api>::MonteCarloRunEventsStream, error::Error> {
        let id = req.id()?;

        tracing::debug!(mc_run_id = ?id, "get monte-carlo run events");

        let _ = atc_entity::MonteCarloRun::find_by_id(id)
            .filter(mc_run::Column::UserId.eq(user.id))
            .one(&self.db)
            .await?
            .ok_or(error::Error::NotFound)?;

        let monte_carlo_run_events = self.monte_carlo_run_events.resubscribe();
        let stream = tokio_stream::wrappers::BroadcastStream::new(monte_carlo_run_events);
        let stream = stream.filter_map(move |res| async move {
            if let Ok(event) = res {
                let model = event.into_model();
                if model.id == id {
                    return Some(Ok(model.into()));
                }
            }
            None
        });
        Ok(Box::pin(stream))
    }

    pub async fn monte_carlo_batch_events(
        &self,
        req: GetMonteCarloRunReq,
        api::CurrentUser { user, .. }: api::CurrentUser,
    ) -> Result<<super::Api as api_server::Api>::MonteCarloBatchEventsStream, error::Error> {
        let id = req.id()?;

        tracing::debug!(mc_run_id = ?id, "get monte-carlo batch events");

        let _ = atc_entity::MonteCarloRun::find_by_id(id)
            .filter(mc_run::Column::UserId.eq(user.id))
            .one(&self.db)
            .await?
            .ok_or(error::Error::NotFound)?;

        let monte_carlo_batch_events = self.monte_carlo_batch_events.resubscribe();
        let stream = tokio_stream::wrappers::BroadcastStream::new(monte_carlo_batch_events);
        let stream = stream.filter_map(move |res| async move {
            if let Ok(event) = res {
                let model = event.into_model();
                if model.run_id == id {
                    return Some(Ok(model.into()));
                }
            }
            None
        });
        Ok(Box::pin(stream))
    }

    pub async fn get_monte_carlo_results(
        &self,
        req: GetMonteCarloResultsReq,
        api::CurrentUser { .. }: api::CurrentUser,
    ) -> Result<GetMonteCarloResultsResp, error::Error> {
        tracing::debug!("get monte-carlo sample results");
        let GetMonteCarloResultsReq { id, batch_number } = req;
        let id = Uuid::from_slice(&id).map_err(|_| error::Error::InvalidRequest)?;
        let download_url = self
            .sim_storage_client
            .download_results_url(id, batch_number)
            .await?;

        tracing::debug!(%download_url, "get monte-carlo sample results - done");

        Ok(GetMonteCarloResultsResp { download_url })
    }
}

#[derive(FromQueryResult, Debug, Serialize, Deserialize)]
pub struct MonteCarloRuntime {
    pub id: Uuid,
    pub runtime_sum: i64,
}

pub(crate) async fn get_monte_carlo_runtime_for_current_month(
    db_connection: &DatabaseConnection,
    user_id: Uuid,
    subscription_end: i64,
) -> Result<u32, Error> {
    let billing_period_end = DateTime::from_timestamp_millis(subscription_end * 1000);
    let billing_period_start =
        billing_period_end.and_then(|dt| dt.checked_sub_months(Months::new(1)));

    let tracing_debug_span = tracing::debug_span!(
        "get_monte_carlo_runtime_for_current_month",
        %user_id,
        ?billing_period_start,
        ?billing_period_end,
    );

    tracing_debug_span.in_scope(|| {
        tracing::debug!("get monte-carlo runtime for current period - start");
    });

    let mc_run_billings = atc_entity::MonteCarloRun::find()
        .select_only()
        .columns([atc_entity::mc_run::Column::Id])
        .column_as(atc_entity::batches::Column::Runtime.sum(), "runtime_sum")
        .filter(atc_entity::mc_run::Column::Billed.eq(true))
        .filter(atc_entity::mc_run::Column::UserId.eq(user_id))
        .filter(atc_entity::mc_run::Column::Started.gt(billing_period_start))
        .filter(atc_entity::mc_run::Column::Started.lte(billing_period_end))
        .join_rev(
            JoinType::InnerJoin,
            atc_entity::Batches::belongs_to(atc_entity::MonteCarloRun)
                .from(atc_entity::batches::Column::RunId)
                .to(atc_entity::mc_run::Column::Id)
                .into(),
        )
        .group_by(atc_entity::mc_run::Column::Id)
        .into_model::<MonteCarloRuntime>()
        .all(db_connection)
        .await?;

    let minutes_used: u32 = mc_run_billings
        .iter()
        .map(|mc_run| (mc_run.runtime_sum as f64 / 60.0).ceil() as u32)
        .sum();

    tracing_debug_span.in_scope(|| {
        tracing::debug!(%minutes_used, "get monte-carlo runtime for current period - done");
    });

    Ok(minutes_used)
}
