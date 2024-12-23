use std::time::Duration;

use azure_storage::shared_access_signature::service_sas::BlobSasPermissions;
use azure_storage_blobs::prelude::ClientBuilder;

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
    azure_blob: ClientBuilder,
    sim_artifacts_bucket_name: String,
    sim_results_bucket_name: String,
}

impl SimStorageClient {
    pub async fn new(config: &MonteCarloConfig) -> anyhow::Result<Self> {
        // let gcp_config = ClientConfig::default().with_auth().await?;
        // let gcs_client = GcsClient::new(gcp_config);
        let credentials = azure_identity::create_credential()?;
        let credentials = azure_storage::StorageCredentials::token_credential(credentials);
        let azure_blob = ClientBuilder::new(config.azure_account_name.clone(), credentials);
        Ok(SimStorageClient {
            azure_blob,
            sim_artifacts_bucket_name: config.sim_artifacts_bucket_name.clone(),
            sim_results_bucket_name: config.sim_results_bucket_name.clone(),
        })
    }

    pub async fn upload_artifacts_url(&self, id: uuid::Uuid) -> Result<String, Error> {
        let object_name = format!("runs/{}.tar.zst", id);
        let blob_client = self
            .azure_blob
            .clone()
            .blob_client(self.sim_artifacts_bucket_name.clone(), object_name);
        let sas_token = blob_client
            .shared_access_signature(
                BlobSasPermissions {
                    write: true,
                    create: true,
                    ..Default::default()
                },
                time::OffsetDateTime::now_utc() + Duration::from_secs(3600),
            )
            .await?;
        let url = blob_client.generate_signed_blob_url(&sas_token)?;
        Ok(url.to_string())
    }

    pub async fn download_results_url(
        &self,
        id: uuid::Uuid,
        batch_number: u32,
    ) -> Result<String, Error> {
        let object_name = format!("runs/{}/batches/{}.tar.zst", id, batch_number);

        let blob_client = self
            .azure_blob
            .clone()
            .blob_client(self.sim_results_bucket_name.clone(), object_name);
        let sas_token = blob_client
            .shared_access_signature(
                BlobSasPermissions {
                    read: true,
                    ..Default::default()
                },
                time::OffsetDateTime::now_utc() + Duration::from_secs(3600),
            )
            .await?;
        let url = blob_client.generate_signed_blob_url(&sas_token)?;
        Ok(url.to_string())
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
