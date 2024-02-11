use elodin_types::{Batch, BatchResults, Run, BATCH_TOPIC, RUN_TOPIC};
use google_cloud_storage::client::{Client as GcsClient, ClientConfig};
use google_cloud_storage::http::objects::download::Range;
use google_cloud_storage::http::objects::get::GetObjectRequest;
use nox::Client as NoxClient;

use crate::config::MonteCarloConfig;

pub struct Runner {
    msg_queue: redmq::MsgQueue,
    nox_client: NoxClient,
    gcs_client: GcsClient,
    sim_artifacts_bucket_name: String,
    _sim_results_bucket_name: String,
}

impl Runner {
    pub async fn new(config: MonteCarloConfig) -> anyhow::Result<Self> {
        let redis = redis::Client::open(config.redis_url)?;
        let msg_queue = redmq::MsgQueue::new(&redis, "sim-agent", config.pod_name).await?;
        let gcs_config = ClientConfig::default().with_auth().await?;
        Ok(Self {
            msg_queue,
            nox_client: NoxClient::cpu()?,
            gcs_client: GcsClient::new(gcs_config),
            sim_artifacts_bucket_name: config.sim_artifacts_bucket_name,
            _sim_results_bucket_name: config.sim_results_bucket_name,
        })
    }

    pub async fn run(mut self) -> anyhow::Result<()> {
        tracing::info!("running monte carlo agent");
        loop {
            let batches = self.msg_queue.recv::<Batch>(BATCH_TOPIC, 1, None).await?;

            match self.process_batches(&batches).await {
                Ok(_) => {}
                Err(err) => tracing::error!(?err, "error processing batches"),
            }

            self.msg_queue.ack(BATCH_TOPIC, &batches).await?;
            self.msg_queue.del(BATCH_TOPIC, &batches).await?;
        }
    }

    async fn process_batches(&mut self, batches: &[redmq::Received<Batch>]) -> anyhow::Result<()> {
        let Some(run_id) = batches.iter().find(|b| !b.buffer).map(|b| b.id.clone()) else {
            return Ok(());
        };
        let Some(run) = self.msg_queue.get::<Run>(RUN_TOPIC, &run_id).await? else {
            anyhow::bail!("monte carlo run {} not found", run_id);
        };

        // TODO: run the batch simulations, collect and upload the results
        let start_time = chrono::Utc::now();

        // TODO: use streaming API to avoid buffering the entire archive in memory
        // TODO: cache the artifacts based on the run id to avoid downloading them multiple times
        tracing::info!("downloading sim artifacts for run {}", run.id);
        let data = self
            .gcs_client
            .download_object(
                &GetObjectRequest {
                    bucket: self.sim_artifacts_bucket_name.clone(),
                    object: format!("runs/{}.tar.zst", run.id),
                    ..Default::default()
                },
                &Range::default(),
            )
            .await?;

        let results = tokio::task::block_in_place(|| {
            let zstd = zstd::Decoder::new(data.as_slice())?;
            let mut tar = tar::Archive::new(zstd);
            let temp_dir = tempfile::tempdir()?;
            tar.unpack(temp_dir.path())?;

            let hlo = temp_dir.path().join("artifacts").join("xlacomp.hlo");
            let hlo = std::fs::read(hlo)?;
            tracing::debug!("downloaded XLA hlo: {} bytes", hlo.len());

            let world = temp_dir.path().join("artifacts").join("world.bin");
            let world = std::fs::read(world)?;
            let world: Vec<Vec<u8>> = postcard::from_bytes(&world)?;
            tracing::debug!("downloaded {} component buffers", world.len());

            let comp = nox::xla::HloModuleProto::parse_binary(&hlo)?.computation();
            let _exec = self.nox_client.0.compile(&comp)?;

            let mut results = Vec::default();
            for b in batches {
                tracing::info!(%run.name, %run.id, batch_no = %b.batch_no, "simulating batch");
                let elapsed = chrono::Utc::now().signed_duration_since(start_time);
                results.push(BatchResults {
                    batch_no: b.batch_no,
                    failed: 0,
                    start_time,
                    run_time_seconds: elapsed.num_seconds() as u64,
                });
            }
            Ok::<_, anyhow::Error>(results)
        })?;

        let results_topic = format!("mc:results:{}", run.id);
        self.msg_queue.send(&results_topic, results).await?;
        Ok(())
    }
}
