use std::fs::File;
use std::io::{BufRead, Seek};
use std::path::{Path, PathBuf};
use std::time::Instant;

use atc_entity::batches;
use atc_entity::events::DbExt;
use elodin_types::{sandbox::*, Batch, BitVec, SampleMetadata, BATCH_TOPIC};
use fred::prelude::*;
use google_cloud_storage::client::{Client as GcsClient, ClientConfig};
use google_cloud_storage::http::objects::download::Range;
use google_cloud_storage::http::objects::get::GetObjectRequest;
use google_cloud_storage::http::objects::upload::{Media, UploadObjectRequest, UploadType};
use nox_ecs::nox::Client as NoxClient;
use nox_ecs::Compiled;
use nox_ecs::{Seed, WorldExec};
use sea_orm::{prelude::*, IntoActiveModel, TransactionTrait};
use tokio::sync::mpsc;
use tokio::task::block_in_place;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;
use tracing::Instrument;

use crate::config::MonteCarloConfig;
use crate::pytest;

pub struct Runner {
    db: DatabaseConnection,
    redis: RedisClient,
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

        let redis_config = RedisConfig::from_url(&config.redis_url)?;
        let redis = Builder::from_config(redis_config).build()?;
        let msg_queue = redmq::MsgQueue::new(&redis, "sim-agent", config.pod_name).await?;
        let gcs_config = ClientConfig::default().with_auth().await?;

        let channel = super::builder_channel(config.tester_addr);
        let vm_client = sandbox_client::SandboxClient::new(channel.clone());
        Ok(Self {
            db,
            redis,
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
        self.redis.init().await?;
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
        let Some(run_id) = batches.iter().find(|b| !b.buffer).map(|b| b.run_id) else {
            return Ok(());
        };

        let run = atc_entity::MonteCarloRun::find_by_id(run_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| anyhow::anyhow!("monte carlo run {} not found", run_id))?;

        let run_max_duration = run.max_duration as usize;
        let mut run_active = run.into_active_model();

        let temp_dir = tempfile::tempdir()?;
        let artifacts = self.download_artifacts(&temp_dir, run_id).await?;
        let context_dir = artifacts.join("context");
        let run_exec = WorldExec::read_from_dir(artifacts)?;
        let run_exec = run_exec.compile(self.nox_client.clone())?;

        for b in batches {
            let txn = self.db.begin().await?;
            let mut batch = atc_entity::Batches::find_by_id((run_id, b.batch_no as i32))
                .one(&txn)
                .await?
                .ok_or_else(|| anyhow::anyhow!("batch {}/{} not found", run_id, b.batch_no))?
                .into_active_model();
            batch.status = sea_orm::Set(atc_entity::batches::Status::Running);
            let batch = batch.update_with_event(&txn, &self.redis).await?;
            txn.commit().await?;

            let start_time = chrono::Utc::now();
            let failed = self
                .process_batch(run_id, run_max_duration, &batch, &run_exec, &context_dir)
                .instrument(tracing::info_span!("batch", no = %b.batch_no))
                .await
                .inspect_err(|err| tracing::error!(?err, "batch failed"))
                .unwrap_or(BitVec::from_elem(batch.samples as usize, true));
            let finish_time = chrono::Utc::now();
            let runtime = (finish_time - start_time).to_std().unwrap();
            let failed_count = failed.iter().filter(|b| *b).count();
            tracing::info!(failed_count, ?runtime, "processed batch");

            let mut batch = batch.into_active_model();
            batch.failures = sea_orm::Set(failed.to_bytes());
            batch.finished = sea_orm::Set(Some(finish_time));
            batch.status = sea_orm::Set(atc_entity::batches::Status::Done);
            batch.runtime = sea_orm::Set(runtime.as_secs() as i32);
            batch.update_with_event(&self.db, &self.redis).await?;
        }

        let not_done_batches_count = atc_entity::Batches::find()
            .filter(batches::Column::RunId.eq(run_id))
            .filter(batches::Column::Status.ne(atc_entity::batches::Status::Done))
            .count(&self.db)
            .await?;

        if not_done_batches_count == 0 {
            run_active.status = sea_orm::Set(atc_entity::mc_run::Status::Done);
            run_active.update(&self.db).await?;
        }

        Ok(())
    }

    async fn process_batch(
        &mut self,
        run_id: Uuid,
        run_max_duration: usize,
        batch: &atc_entity::batches::Model,
        run_exec: &WorldExec<Compiled>,
        context_dir: &Path,
    ) -> Result<BitVec<u32>, Error> {
        let sim_start = Instant::now();
        let mut batch_worlds = Vec::new();
        // TODO: parallelize batch sim execution
        let batch_no = batch.batch_number as usize;
        let samples = batch.samples as usize;

        for i in 0..samples {
            let mut sample_exec = run_exec.fork();
            let sample_no = batch_no * samples + i;
            let span = tracing::info_span!("sample", no = %sample_no);
            let _guard = span.enter();

            if let Some(mut col) = sample_exec.world.column_mut::<Seed>() {
                tracing::trace!("setting seed");
                col.typed_buf_mut::<u64>()
                    .unwrap_or_default()
                    .iter_mut()
                    .for_each(|s| *s = sample_no as u64);
            }
            block_in_place(|| self.run_sim(run_max_duration, &mut sample_exec))?;

            let profile = sample_exec
                .profile()
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            let sample_metadata = SampleMetadata {
                run_id,
                batch_no,
                sample_no,
                profile,
            };
            batch_worlds.push((sample_metadata, sample_exec));
        }
        tracing::debug!(elapsed = ?sim_start.elapsed(), "simulated batch");

        let archive_start = Instant::now();
        let mut batch_tar = write_results_tar(context_dir, batch_worlds)?;
        let batch_tar_zst = compress(&mut batch_tar)?;
        tracing::debug!(elapsed = ?archive_start.elapsed(), "wrote batch archive");
        let file_name = format!("runs/{}/batches/{}.tar.zst", run_id, batch_no);
        self.upload_archive(batch_tar_zst, file_name.clone());

        // run pytest:
        let results_file = file_name.replace('/', "_");
        let file_stream = stream_file(batch_tar, results_file.clone());
        self.vm_client.clone().send_file(file_stream).await?;
        let test_start = Instant::now();
        let report = self
            .vm_client
            .clone()
            .test(TestReq { results_file })
            .await?
            .into_inner()
            .report;
        let report = pytest::Report::from_json(&report)?;
        tracing::debug!(elapsed = ?test_start.elapsed(), ?report.summary, "received test results");
        let failed = report.failed(samples, batch_no * samples);
        Ok(failed)
    }

    async fn download_artifacts(
        &self,
        temp_dir: &tempfile::TempDir,
        run_id: Uuid,
    ) -> anyhow::Result<PathBuf> {
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
        tar.unpack(temp_dir.path())?;
        let artifacts = temp_dir.path().join("artifacts");
        Ok(artifacts)
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

    fn run_sim(
        &self,
        max_duration: usize,
        exec: &mut WorldExec<Compiled>,
    ) -> Result<(), nox_ecs::Error> {
        let ticks = max_duration * 60;
        for _ in 0..ticks {
            exec.run()?;
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

fn write_results_tar(
    context_dir: &Path,
    batch_worlds: Vec<(SampleMetadata, WorldExec<Compiled>)>,
) -> Result<File, Error> {
    let results_dir = tempfile::tempdir()?;
    for (metadata, mut exec) in batch_worlds.into_iter() {
        let sample_dir = results_dir.path().join(metadata.sample_no.to_string());
        let metadata = serde_json::to_string(&metadata)?;
        exec.world.write_to_dir(&sample_dir)?;
        std::fs::write(sample_dir.join("sample.json"), metadata)?;
    }
    let mut ar = tar::Builder::new(tempfile::tempfile()?);
    ar.append_dir_all("results", &results_dir)?;
    ar.append_dir_all("context", context_dir)?;
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
    #[error("impeller error: {0}")]
    Impeller(#[from] impeller::Error),
    #[error("nox ecs error: {0}")]
    NoxEcs(#[from] nox_ecs::Error),
    #[error("tonic error: {0}")]
    Tonic(#[from] tonic::Status),
    #[error("pytest error: {0}")]
    Pytest(#[from] pytest::Error),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
}
