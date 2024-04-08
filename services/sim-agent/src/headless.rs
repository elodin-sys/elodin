use std::fs::File;
use std::io::{BufRead, Seek};
use std::time::Instant;

use atc_entity::events::DbExt;
use elodin_types::{sandbox::*, Batch, BitVec, Run, BATCH_TOPIC, RUN_TOPIC};
use google_cloud_storage::client::{Client as GcsClient, ClientConfig};
use google_cloud_storage::http::objects::download::Range;
use google_cloud_storage::http::objects::get::GetObjectRequest;
use google_cloud_storage::http::objects::upload::{Media, UploadObjectRequest, UploadType};
use nox::Client as NoxClient;
use nox_ecs::{polars::PolarsWorld, ComponentExt, Seed, WorldExec};
use sea_orm::{prelude::*, TransactionTrait};
use tokio::sync::mpsc;
use tokio::task::block_in_place;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;
use tracing::Instrument;

use crate::config::MonteCarloConfig;
use crate::pytest;

pub struct Runner {
    db: DatabaseConnection,
    redis_conn: redis::aio::MultiplexedConnection,
    msg_queue: redmq::MsgQueue,
    nox_client: NoxClient,
    gcs_client: GcsClient,
    vm_client: sandbox_client::SandboxClient<Channel>,
    sim_artifacts_bucket_name: String,
    sim_results_bucket_name: String,
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

        let channel = super::builder_channel(config.tester_addr);
        let vm_client = sandbox_client::SandboxClient::new(channel.clone());
        Ok(Self {
            db,
            redis_conn,
            msg_queue,
            nox_client: NoxClient::cpu()?,
            gcs_client: GcsClient::new(gcs_config),
            vm_client,
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

        let (run_exec, sim_code) = self.download_artifacts(run.id).await?;

        for b in batches {
            let start_time = chrono::Utc::now();
            let failed = self
                .process_batch(&run, &run_exec, sim_code.clone(), b.batch_no)
                .instrument(tracing::info_span!("batch", no = %b.batch_no))
                .await
                .inspect_err(|err| tracing::error!(?err, "batch failed"))
                .unwrap_or(BitVec::from_elem(run.batch_size, true));
            let finish_time = chrono::Utc::now();
            let runtime = (finish_time - start_time).to_std().unwrap();
            let failed_count = failed.iter().filter(|b| *b).count();
            tracing::info!(failed_count, ?runtime, "processed batch");

            let results_model = atc_entity::batches::ActiveModel {
                run_id: sea_orm::Unchanged(run.id),
                batch_number: sea_orm::Unchanged(b.batch_no as i32),
                samples: sea_orm::Set(run.batch_size as i32),
                failures: sea_orm::Set(failed.to_bytes()),
                finished: sea_orm::Set(Some(finish_time)),
                status: sea_orm::Set(atc_entity::batches::Status::Done),
            };
            results_model
                .update_with_event(&self.db, &mut self.redis_conn.clone())
                .await?;
        }
        Ok(())
    }

    async fn process_batch(
        &mut self,
        run: &Run,
        run_exec: &WorldExec,
        code: String,
        batch_no: usize,
    ) -> Result<BitVec<u32>, Error> {
        let sim_start = Instant::now();
        let mut batch_world = PolarsWorld::default();
        // TODO: parallelize batch sim execution
        for i in 0..run.batch_size {
            let mut sample_exec = run_exec.fork();
            let sample_no = batch_no * run.batch_size + i;
            let span = tracing::info_span!("sample", no = %sample_no);
            let _guard = span.enter();

            if let Ok(mut col) = sample_exec.column_mut(Seed::component_id()) {
                tracing::trace!("setting seed");
                col.typed_buf_mut::<u64>()
                    .unwrap_or_default()
                    .iter_mut()
                    .for_each(|s| *s = sample_no as u64);
            }
            block_in_place(|| self.run_sim(run, &mut sample_exec))?;

            let mut history = sample_exec.history.compact_to_world()?;
            history.add_sample_number(sample_no)?;
            batch_world.vstack(&history)?;
        }
        tracing::debug!(elapsed = ?sim_start.elapsed(), "simulated batch");
        let archive_start = Instant::now();
        let mut batch_tar = write_history_tar(&mut batch_world)?;
        let batch_tar_zst = compress(&mut batch_tar)?;
        tracing::debug!(elapsed = ?archive_start.elapsed(), "wrote batch archive");
        let file_name = format!("runs/{}/batches/{}.tar.zst", run.id, batch_no);
        self.upload_archive(batch_tar_zst, file_name.clone());

        let results_file = file_name.replace('/', "_");
        let file_stream = stream_file(batch_tar, results_file.clone());
        self.vm_client.clone().send_file(file_stream).await?;
        let test_start = Instant::now();
        let report = self
            .vm_client
            .clone()
            .test(TestReq { code, results_file })
            .await?
            .into_inner()
            .report;
        let report = pytest::Report::from_json(&report)?;
        tracing::debug!(elasped = ?test_start.elapsed(), ?report.summary, "received test results");
        let failed = report.failed(run.batch_size, batch_no * run.batch_size);
        Ok(failed)
    }

    async fn download_artifacts(&self, run_id: Uuid) -> anyhow::Result<(WorldExec, String)> {
        tracing::info!("downloading sim artifacts for run {}", run_id);
        let data = self
            .gcs_client
            .download_object(
                &GetObjectRequest {
                    bucket: self.sim_artifacts_bucket_name.clone(),
                    object: format!("runs/{}.tar.zst", run_id),
                    ..Default::default()
                },
                &Range::default(),
            )
            .await?;
        let zstd = zstd::Decoder::new(data.as_slice())?;
        let mut tar = tar::Archive::new(zstd);
        let temp_dir = tempfile::tempdir()?;
        tar.unpack(temp_dir.path())?;
        let artifacts = temp_dir.path().join("artifacts");
        let sim_code = std::fs::read_to_string(artifacts.join("sim_code.py"))?;
        let run_exec = WorldExec::read_from_dir(artifacts)?;
        Ok((run_exec, sim_code))
    }

    fn upload_archive(&self, archive: File, file_name: String) -> tokio::task::JoinHandle<()> {
        let start = Instant::now();
        let gcs_client = self.gcs_client.clone();
        let bucket = self.sim_results_bucket_name.clone();
        let len = archive.metadata().unwrap().len();
        tracing::trace!(file_name, len, "uploading replay archive");
        tokio::spawn(async move {
            let result = gcs_client
                .upload_object(
                    &UploadObjectRequest {
                        bucket,
                        ..Default::default()
                    },
                    tokio::fs::File::from_std(archive),
                    &UploadType::Simple(Media::new(file_name.clone())),
                )
                .await;
            let elapsed = start.elapsed();
            match result {
                Ok(_) => tracing::debug!(?elapsed, file_name, "uploaded replay archive"),
                Err(err) => tracing::error!(?err, "gcs upload failed"),
            }
        })
    }

    fn run_sim(&self, run: &Run, exec: &mut WorldExec) -> Result<(), nox_ecs::Error> {
        let ticks = run.max_duration * 60;
        for _ in 0..ticks {
            exec.run(&self.nox_client)?;
        }
        Ok(())
    }
}

fn stream_file(file: File, file_name: String) -> ReceiverStream<FileChunk> {
    let start = Instant::now();
    let mut reader = std::io::BufReader::with_capacity(128 * 1024, file);
    let (tx, rx) = mpsc::channel::<FileChunk>(1);
    let span = tracing::info_span!("stream_file", file = %file_name);
    tokio::task::spawn_blocking(move || {
        let _guard = span.enter();
        loop {
            let buf = reader.fill_buf()?;
            let len = buf.len();
            if buf.is_empty() {
                break;
            }
            let chunk = FileChunk {
                data: buf.to_vec(),
                name: file_name.clone(),
            };
            tracing::trace!(len, "sending chunk");
            let _ = tx.blocking_send(chunk);
            reader.consume(len);
        }
        tracing::debug!(elapsed = ?start.elapsed(), "sent");
        Ok::<_, Error>(())
    });
    ReceiverStream::new(rx)
}

fn write_history_tar(history: &mut PolarsWorld) -> Result<File, Error> {
    let results_dir = tempfile::tempdir()?;
    history.write_to_dir(&results_dir)?;

    let mut ar = tar::Builder::new(tempfile::tempfile()?);
    ar.append_dir_all("results", &results_dir)?;
    let mut results_tar = ar.into_inner()?;
    results_tar.sync_all()?;
    results_tar.rewind()?;
    Ok(results_tar)
}

fn compress(file: &mut File) -> Result<File, Error> {
    let mut zstd = zstd::Encoder::new(tempfile::tempfile()?, 0)?;
    std::io::copy(file, &mut zstd)?;
    let mut file_zst = zstd.finish()?;
    file_zst.sync_all()?;
    file.rewind()?;
    file_zst.rewind()?;
    Ok(file_zst)
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("nox ecs error: {0}")]
    NoxEcs(#[from] nox_ecs::Error),
    #[error("tonic error: {0}")]
    Tonic(#[from] tonic::Status),
    #[error("pytest error: {0}")]
    Pytest(#[from] pytest::Error),
}
