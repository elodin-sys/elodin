use std::io::Seek;

use anyhow::Context;
use elodin_types::{Batch, BatchResults, Run, BATCH_TOPIC, RUN_TOPIC};
use google_cloud_storage::client::{Client as GcsClient, ClientConfig};
use google_cloud_storage::http::objects::download::Range;
use google_cloud_storage::http::objects::get::GetObjectRequest;
use google_cloud_storage::http::objects::upload::{Media, UploadObjectRequest, UploadType};
use nox::Client as NoxClient;
use nox_ecs::WorldExec;
use tracing::Instrument;

use crate::config::MonteCarloConfig;

pub struct Runner {
    msg_queue: redmq::MsgQueue,
    nox_client: NoxClient,
    gcs_client: GcsClient,
    sim_artifacts_bucket_name: String,
    sim_results_bucket_name: String,
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
            sim_results_bucket_name: config.sim_results_bucket_name,
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

        let mut uploads = tokio::task::JoinSet::new();
        let results = tokio::task::block_in_place(|| {
            let span = tracing::info_span!("run", name = %run.name);
            let _guard = span.enter();

            let zstd = zstd::Decoder::new(data.as_slice())?;
            let mut tar = tar::Archive::new(zstd);
            let temp_dir = tempfile::tempdir()?;
            tar.unpack(temp_dir.path())?;
            let artifacts = temp_dir.path().join("artifacts");
            let run_exec = WorldExec::read_from_dir(artifacts)?;

            let mut results = Vec::default();
            for b in batches {
                let span = tracing::info_span!("batch", no = %b.batch_no);
                let _guard = span.enter();

                let batch_exec = run_exec.fork();
                let mut failed = 0;
                // TODO: parallelize batch sim execution
                for i in 0..run.batch_size {
                    let mut sample_exec = batch_exec.fork();
                    let sample_no = b.batch_no * run.batch_size + i;
                    let span = tracing::info_span!("sample", no = %sample_no);
                    let _guard = span.enter();

                    if let Err(err) = self.run_sim(&run, &mut sample_exec) {
                        tracing::error!(?err, "simulation failed");
                        failed += 1;
                    } else {
                        tracing::debug!("simulation completed");
                    }

                    let gcs_client = self.gcs_client.clone();
                    let bucket = self.sim_results_bucket_name.clone();
                    let file_name = format!("runs/{}/samples/{}.tar.zst", run.id, sample_no);
                    uploads.spawn(
                        async move {
                            // TODO: if/when polars supports streaming from compressed parquet files,
                            // upload files directly instead of archiving them into a tarball
                            tracing::trace!("generating replay archive");
                            let results_dir = tempfile::tempdir()?;
                            let results_archive = tempfile::tempfile()?;
                            sample_exec.history.write_to_dir(&results_dir)?;
                            let buf = std::io::BufWriter::new(results_archive);
                            let zstd = zstd::Encoder::new(buf, 0)?;
                            let mut ar = tar::Builder::new(zstd);
                            ar.append_dir_all("results", &results_dir)?;
                            // TODO: write directly to compressed archive instead of through through tmp fs
                            let mut results_archive = ar.into_inner()?.finish()?.into_inner()?;
                            results_archive.rewind()?;
                            let len = results_archive.metadata()?.len();

                            tracing::trace!(file_name, len, "uploading replay archive");
                            gcs_client
                                .upload_object(
                                    &UploadObjectRequest {
                                        bucket,
                                        ..Default::default()
                                    },
                                    tokio::fs::File::from_std(results_archive),
                                    &UploadType::Simple(Media::new(file_name.clone())),
                                )
                                .await
                                .with_context(|| format!("gcs upload failed: {}", file_name))?;
                            tracing::trace!(file_name, "uploaded replay archive");
                            Ok::<_, anyhow::Error>(())
                        }
                        .in_current_span(),
                    );
                }

                let runtime = (chrono::Utc::now() - start_time).to_std()?;
                let batch_results = BatchResults {
                    batch_no: b.batch_no,
                    failed,
                    start_time,
                    run_time_seconds: runtime.as_secs(),
                };
                tracing::info!(failed = %batch_results.failed, ?runtime, "simulated batch");
                results.push(batch_results);
            }
            Ok::<_, anyhow::Error>(results)
        })?;

        while let Some(result) = uploads.join_next().await {
            if let Err(err) = result.unwrap() {
                tracing::error!(?err, "upload failed");
            }
        }

        let results_topic = format!("mc:results:{}", run.id);
        self.msg_queue.send(&results_topic, results).await?;
        Ok(())
    }

    fn run_sim(&self, run: &Run, exec: &mut WorldExec) -> Result<(), nox_ecs::Error> {
        let ticks = run.max_duration * 60;
        for _ in 0..ticks {
            exec.run(&self.nox_client)?;
        }
        Ok(())
    }
}
