use google_cloud_storage::client::{Client as GcsClient, ClientConfig};
use google_cloud_storage::sign::{SignedURLMethod, SignedURLOptions};

use crate::config::MonteCarloConfig;
use crate::error::Error;

pub const BATCH_SIZE: usize = 10;
pub const MAX_SAMPLE_COUNT: usize = 100_000;

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
        batch_number: u32,
    ) -> Result<String, Error> {
        let object_name = format!("runs/{}/batches/{}.tar.zst", id, batch_number);
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

#[cfg(test)]
mod tests {
    use elodin_types::Batch;

    #[test]
    fn ser_de_batch() {
        let batch = Batch {
            run_id: uuid::Uuid::now_v7(),
            batch_no: 6,
            buffer: false,
        };

        let batch_de = redmq::from_redis::<Batch>(redmq::to_redis(&batch)).unwrap();
        assert_eq!(batch_de, batch);
    }
}
