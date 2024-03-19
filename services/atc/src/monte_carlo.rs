use elodin_types::{Batch, Run, BATCH_TOPIC, RUN_TOPIC};
use google_cloud_storage::client::{Client as GcsClient, ClientConfig};
use google_cloud_storage::sign::{SignedURLMethod, SignedURLOptions};
use tokio_util::sync::CancellationToken;

use crate::config::MonteCarloConfig;
use crate::error::Error;

pub const BATCH_SIZE: usize = 10;
pub const MAX_SAMPLE_COUNT: usize = 100_000;

pub const SPAWN_GROUP: &str = "atc:spawn";

// Buffer batches help with work alignment
// E.g. if an agent can do 10 batches at once, adding some buffer between runs allows for the agent
// to only work on 1 run at a time. These batches are effectively a no-op, and are ignored during
// results aggregation.
// pub const BUFFER_BATCH_COUNT: usize = 10;

pub struct SimStorageClient {
    gcs_client: GcsClient,
    sim_artifacts_bucket_name: String,
    sim_results_bucket_name: String,
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
            sim_results_bucket_name: config.sim_results_bucket_name.clone(),
        })
    }

    pub async fn upload_artifacts_url(&self, id: uuid::Uuid) -> Result<String, Error> {
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

    pub async fn download_results_url(
        &self,
        id: uuid::Uuid,
        sample_number: u32,
    ) -> Result<String, Error> {
        let object_name = format!("runs/{}/samples/{}.tar.zst", id, sample_number);
        let options = SignedURLOptions {
            method: SignedURLMethod::GET,
            ..Default::default()
        };
        let url = self
            .gcs_client
            .signed_url(
                &self.sim_results_bucket_name,
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

#[cfg(test)]
mod tests {
    use super::{Batch, Run, BATCH_SIZE};

    #[test]
    fn ser_de_run() {
        let run = Run {
            id: uuid::Uuid::now_v7(),
            name: "test_run".to_string(),
            samples: 774,
            batch_size: BATCH_SIZE,
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
