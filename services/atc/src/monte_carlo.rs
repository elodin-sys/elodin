use atc_entity::mc;
use elodin_types::{Batch, BatchResults, Run, BATCH_TOPIC, RUN_TOPIC};
use google_cloud_storage::client::{Client as GcsClient, ClientConfig};
use google_cloud_storage::sign::{SignedURLMethod, SignedURLOptions};
use sea_orm::prelude::*;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::config::MonteCarloConfig;
use crate::error::Error;

pub const BATCH_SIZE: usize = 100;
pub const MAX_SAMPLE_COUNT: usize = 100_000;

pub const SPAWN_GROUP: &str = "atc:spawn";
pub const RESULTS_GROUP: &str = "atc:results";

// Buffer batches help with work alignment
// E.g. if an agent can do 10 batches at once, adding some buffer between runs allows for the agent
// to only work on 1 run at a time. These batches are effectively a no-op, and are ignored during
// results aggregation.
// pub const BUFFER_BATCH_COUNT: usize = 10;

pub struct SimStorageClient {
    gcs_client: GcsClient,
    sim_artifacts_bucket_name: String,
    _sim_results_bucket_name: String,
}

pub struct BatchSpawner {
    msg_queue: redmq::MsgQueue,
}

impl SimStorageClient {
    pub async fn new(config: &MonteCarloConfig) -> anyhow::Result<Self> {
        let gcp_config = ClientConfig::default().with_auth().await?;
        let gcs_client = GcsClient::new(gcp_config);
        Ok(SimStorageClient {
            gcs_client,
            sim_artifacts_bucket_name: config.sim_artifacts_bucket_name.clone(),
            _sim_results_bucket_name: config.sim_results_bucket_name.clone(),
        })
    }

    pub async fn signed_upload_url(&self, id: uuid::Uuid) -> Result<String, Error> {
        let object_name = format!("runs/{}.tar.zst", id);
        let options = SignedURLOptions {
            method: SignedURLMethod::PUT,
            ..Default::default()
        };
        let url = self
            .gcs_client
            .signed_url(
                &self.sim_artifacts_bucket_name,
                &object_name,
                None,
                None,
                options,
            )
            .await?;
        Ok(url)
    }
}

impl BatchSpawner {
    pub async fn new(client: &redis::Client, pod_name: &str) -> anyhow::Result<Self> {
        let msg_queue = redmq::MsgQueue::new(client, SPAWN_GROUP, pod_name).await?;
        let batch_spawner = BatchSpawner { msg_queue };
        Ok(batch_spawner)
    }

    pub async fn run(mut self, cancel_token: CancellationToken) -> anyhow::Result<()> {
        let cancel_on_drop = cancel_token.clone().drop_guard();
        loop {
            let work = tokio::select! {
                r = self.msg_queue.recv::<Run>(RUN_TOPIC, 1, None) => r?,
                _ = cancel_token.cancelled() => break,
            };
            let mut batches = Vec::default();
            for run in &work {
                let since_start = chrono::Utc::now().signed_duration_since(run.start_time);
                let span = tracing::debug_span!("create_batches", %run.id, %run.name, %since_start);
                let _enter = span.enter();
                if since_start.num_minutes() > 30 {
                    tracing::warn!("stale run, skipping batch creation");
                    continue;
                }
                tracing::debug!("creating batches");
                let batch_count = run.samples.div_ceil(run.batch_size);
                let new_batches = (0..batch_count).map(|batch_no| Batch {
                    id: run.id().to_string(),
                    batch_no,
                    buffer: false,
                });
                batches.extend(new_batches);
            }
            // TODO(Akhil): make the send + ack operations atomic
            self.msg_queue.send(BATCH_TOPIC, batches).await?;
            self.msg_queue.ack(RUN_TOPIC, &work).await?;
        }
        drop(cancel_on_drop);
        tracing::debug!("done");
        Ok(())
    }
}

pub struct Aggregator {
    msg_queue: redmq::MsgQueue,
    db: DatabaseConnection,
}

impl Aggregator {
    pub async fn new(
        client: &redis::Client,
        db: DatabaseConnection,
        pod_name: &str,
    ) -> anyhow::Result<Self> {
        let msg_queue = redmq::MsgQueue::new(client, RESULTS_GROUP, pod_name).await?;
        let results_collector = Aggregator { msg_queue, db };
        Ok(results_collector)
    }

    pub async fn run(mut self, cancel_token: CancellationToken) -> anyhow::Result<()> {
        let cancel_on_drop = cancel_token.clone().drop_guard();
        loop {
            let mut work = tokio::select! {
                r = self.msg_queue.recv::<Run>(RUN_TOPIC, 1, None) => r?,
                _ = cancel_token.cancelled() => break,
            };
            let Some(run) = work.pop() else { continue };

            let span = tracing::debug_span!("collect_results", %run.name);
            self.collect_results(run, &cancel_token)
                .instrument(span)
                .await?;
        }
        drop(cancel_on_drop);
        tracing::debug!("done");
        Ok(())
    }

    async fn collect_results(
        &mut self,
        run: redmq::Received<Run>,
        cancel_token: &CancellationToken,
    ) -> anyhow::Result<()> {
        let results_topic = format!("mc:results:{}", run.id);
        let batch_count = run.samples.div_ceil(run.batch_size);
        let expires_at = run.start_time + chrono::Duration::hours(2);
        tracing::info!(results_topic, batch_count, %expires_at, "collecting results");
        let mut last_id: Option<String> = None;
        let mut all_results = Vec::default();
        loop {
            let remaining = expires_at.signed_duration_since(chrono::Utc::now());
            let results = tokio::select! {
                r = self.msg_queue.recv::<BatchResults>(results_topic.as_str(), 100, last_id.as_deref()) => r?,
                _ = cancel_token.cancelled() => return Ok(()),
                _ = tokio::time::sleep(remaining.to_std().unwrap_or_default()) => {
                    tracing::warn!("stale run, abandon results collection");
                    break;
                }
            };
            for r in results {
                last_id = Some(r.id().to_string());
                if r.batch_no >= batch_count {
                    tracing::debug!(%r.batch_no, "ignoring buffer batch results");
                    continue;
                }
                let runtime = Duration::from_secs(r.run_time_seconds);
                tracing::debug!(batch_no = %r.batch_no, ?runtime, failed = ?r.failed, "received results");
                all_results.push(r.clone());
            }

            if all_results.len() >= batch_count {
                break;
            }
            // avoid hot loop, encourage batching
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        let failures = all_results.iter().fold(0, |f, r| f + r.failed);
        let total_runtime = all_results.iter().fold(0, |rt, r| rt + r.run_time_seconds);
        let total_runtime = Duration::from_secs(total_runtime);
        let average_runtime = total_runtime / batch_count as u32;
        tracing::info!(failures, ?average_runtime, "collected results");

        let mc_run = atc_entity::mc::ActiveModel {
            id: sea_orm::Set(run.id),
            status: sea_orm::Set(mc::Status::Done),
            ..Default::default()
        };
        mc_run.update(&self.db).await?;

        let work = &[run];
        self.msg_queue.del_topic(results_topic.as_str()).await?;
        self.msg_queue.ack(RUN_TOPIC, work).await?;
        self.msg_queue.del(RUN_TOPIC, work).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Batch, Run};

    #[test]
    fn ser_de_run() {
        let run = Run {
            id: uuid::Uuid::now_v7(),
            name: "test_run".to_string(),
            samples: 774,
            batch_size: 100,
            start_time: chrono::Utc::now(),
            max_duration: 30,
        };

        let run_de = redmq::from_redis::<Run>(redmq::to_redis(&run)).unwrap();
        assert_eq!(run_de, run);
    }

    #[test]
    fn ser_de_batch() {
        let batch = Batch {
            id: "0-0".to_string(),
            batch_no: 6,
            buffer: false,
        };

        let batch_de = redmq::from_redis::<Batch>(redmq::to_redis(&batch)).unwrap();
        assert_eq!(batch_de, batch);
    }
}
