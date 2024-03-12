use std::io::Seek;

use anyhow::Context;
use atc_entity::events::DbExt;
use elodin_types::{Batch, BitVec, Run, BATCH_TOPIC, RUN_TOPIC};
use google_cloud_storage::client::{Client as GcsClient, ClientConfig};
use google_cloud_storage::http::objects::download::Range;
use google_cloud_storage::http::objects::get::GetObjectRequest;
use google_cloud_storage::http::objects::upload::{Media, UploadObjectRequest, UploadType};
use nox::Client as NoxClient;
use nox_ecs::WorldExec;
use sea_orm::{prelude::*, TransactionTrait};
use tracing::Instrument;

use crate::config::MonteCarloConfig;

pub struct Runner {
    db: DatabaseConnection,
    redis_conn: redis::aio::MultiplexedConnection,
    msg_queue: redmq::MsgQueue,
    nox_client: NoxClient,
    gcs_client: GcsClient,
    sim_artifacts_bucket_name: String,
    sim_results_bucket_name: String,
    uploads: tokio::task::JoinSet<anyhow::Result<()>>,
}

struct BatchResults {
    pub batch_no: usize,
    pub failed: BitVec,
    pub finish_time: chrono::DateTime<chrono::Utc>,
}

impl Runner {
    pub async fn new(config: MonteCarloConfig) -> anyhow::Result<Self> {
        let mut opt = sea_orm::ConnectOptions::new(config.database_url);
        opt.sqlx_logging(false);
        let db = sea_orm::Database::connect(opt).await?;

        let redis = redis::Client::open(config.redis_url)?;
        let redis_conn = redis.get_multiplexed_tokio_connection().await?;
        let msg_queue = redmq::MsgQueue::new(&redis, "sim-agent", config.pod_name).await?;
        let gcs_config = ClientConfig::default().with_auth().await?;
        Ok(Self {
            db,
            redis_conn,
            msg_queue,
            nox_client: NoxClient::cpu()?,
            gcs_client: GcsClient::new(gcs_config),
            sim_artifacts_bucket_name: config.sim_artifacts_bucket_name,
            sim_results_bucket_name: config.sim_results_bucket_name,
            uploads: tokio::task::JoinSet::new(),
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

        for batch in batches {
            let txn = self.db.begin().await?;
            let batch_model = atc_entity::batches::ActiveModel {
                run_id: sea_orm::Unchanged(run.id),
                batch_number: sea_orm::Unchanged(batch.batch_no as i32),
                status: sea_orm::Set(atc_entity::batches::Status::Running),
                ..Default::default()
            };
            batch_model
                .update_with_event(&txn, &mut self.redis_conn.clone())
                .await?;
            txn.commit().await?;
        }

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
                let batch_results = self.process_batch(&run, &run_exec, b.batch_no);
                results.push(batch_results);
            }
            Ok::<_, anyhow::Error>(results)
        })?;

        while let Some(result) = self.uploads.join_next().await {
            if let Err(err) = result.unwrap() {
                tracing::error!(?err, "upload failed");
            }
        }

        for r in results.iter() {
            let results_model = atc_entity::batches::ActiveModel {
                run_id: sea_orm::Unchanged(run.id),
                batch_number: sea_orm::Unchanged(r.batch_no as i32),
                samples: sea_orm::Set(run.batch_size as i32),
                failures: sea_orm::Set(r.failed.to_bytes()),
                finished: sea_orm::Set(Some(r.finish_time)),
                status: sea_orm::Set(atc_entity::batches::Status::Done),
            };
            results_model
                .update_with_event(&self.db, &mut self.redis_conn.clone())
                .await?;
        }
        Ok(())
    }

    fn process_batch(&mut self, run: &Run, run_exec: &WorldExec, batch_no: usize) -> BatchResults {
        let span = tracing::info_span!("batch", no = %batch_no);
        let _guard = span.enter();
        let start_time = chrono::Utc::now();

        let batch_exec = run_exec.fork();
        let mut failed = BitVec::with_capacity(run.batch_size);
        // TODO: parallelize batch sim execution
        for i in 0..run.batch_size {
            let mut sample_exec = batch_exec.fork();
            let sample_no = batch_no * run.batch_size + i;
            let span = tracing::info_span!("sample", no = %sample_no);
            let _guard = span.enter();

            if let Err(err) = self.run_sim(run, &mut sample_exec) {
                tracing::error!(?err, "simulation failed");
                failed.push(true);
            } else {
                tracing::debug!("simulation completed");
                failed.push(false);
            }

            let gcs_client = self.gcs_client.clone();
            let bucket = self.sim_results_bucket_name.clone();
            let file_name = format!("runs/{}/samples/{}.tar.zst", run.id, sample_no);
            self.uploads.spawn(
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

        let finish_time = chrono::Utc::now();
        let runtime = (finish_time - start_time).to_std().unwrap();
        let failed_count = failed.iter().filter(|b| *b).count();
        tracing::info!(failed_count, ?runtime, "simulated batch");
        BatchResults {
            batch_no,
            failed,
            finish_time,
        }
    }

    fn run_sim(&self, run: &Run, exec: &mut WorldExec) -> Result<(), nox_ecs::Error> {
        let ticks = run.max_duration * 60;
        for _ in 0..ticks {
            exec.run(&self.nox_client)?;
        }
        Ok(())
    }
}
