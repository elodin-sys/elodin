use std::io::{BufRead, Write};
use std::process::Stdio;
use std::time::Instant;

use elodin_types::sandbox::{sandbox_server::Sandbox, *};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::async_trait;
use tonic::{Request, Response, Status, Streaming};

pub use elodin_types::sandbox::sandbox_server::SandboxServer;

#[derive(Default)]
pub struct Service();

impl Service {
    pub async fn set_serving(reporter: &mut tonic_health::server::HealthReporter) {
        reporter.set_serving::<SandboxServer<Self>>().await;
    }
}

#[async_trait]
impl Sandbox for Service {
    type RecvFileStream = ReceiverStream<Result<FileChunk, Status>>;

    async fn build(&self, req: Request<BuildReq>) -> Result<Response<BuildResp>, Status> {
        let req = req.into_inner();

        let artifacts_file = build(req.code).await.map_err(|err| {
            tracing::error!(?err, "failed to build code");
            Status::internal(err.to_string())
        })?;

        Ok(Response::new(BuildResp { artifacts_file }))
    }

    async fn test(&self, req: Request<TestReq>) -> Result<Response<TestResp>, Status> {
        let req = req.into_inner();

        let results = test(req.code, req.results_file).await.map_err(|err| {
            tracing::error!(?err, "failed to test results");
            Status::internal(err.to_string())
        })?;

        Ok(Response::new(results))
    }

    async fn send_file(
        &self,
        req: Request<Streaming<FileChunk>>,
    ) -> Result<Response<FileTransferResp>, Status> {
        let mut stream = req.into_inner().peekable();
        let req = stream
            .peek()
            .await
            .ok_or(Status::invalid_argument("empty stream"))?
            .clone()?;
        let path = std::env::temp_dir().join(req.name);
        let mut file = std::fs::File::create(&path)?;
        while let Some(chunk) = stream.try_next().await? {
            file.write_all(&chunk.data)?;
            tracing::trace!(len = chunk.data.len(), path = %path.display(), "wrote chunk to file");
        }
        file.sync_all()?;
        let metadata = file.metadata()?;
        tracing::debug!(size = metadata.len(), path = %path.display(), "received file");
        Ok(Response::new(FileTransferResp::default()))
    }

    async fn recv_file(
        &self,
        req: Request<FileTransferReq>,
    ) -> Result<Response<Self::RecvFileStream>, Status> {
        let req = req.into_inner();
        let path = std::env::temp_dir().join(&req.name);
        let file = std::fs::File::open(&path)?;
        let mut reader = std::io::BufReader::with_capacity(128 * 1024, file);
        let (tx, rx) = mpsc::channel::<Result<FileChunk, Status>>(1);
        tokio::task::spawn_blocking(move || {
            loop {
                match reader.fill_buf() {
                    Ok(buf) => {
                        let len = buf.len();
                        if buf.is_empty() {
                            break;
                        }
                        let chunk = FileChunk {
                            data: buf.to_vec(),
                            name: req.name.clone(),
                        };
                        let _ = tx.blocking_send(Ok(chunk));
                        tracing::trace!(len, path = %path.display(), "sent chunk");
                        reader.consume(len);
                    }
                    Err(err) => {
                        tracing::error!(?err, "failed to read file");
                        let err = Err(Status::internal(err.to_string()));
                        let _ = tx.blocking_send(err);
                        return;
                    }
                };
            }
            tracing::info!(path = %path.display(), "sent file");
            if let Err(err) = std::fs::remove_file(&path) {
                tracing::error!(?err, "failed to delete file");
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

async fn build(code: String) -> anyhow::Result<String> {
    tracing::debug!(len = code.len(), "building code");
    let mut code_file = tempfile::NamedTempFile::new()?;
    tracing::debug!(file = %code_file.path().display(), "writing code to temp file");
    code_file.write_all(code.as_bytes())?;

    let start = Instant::now();
    let artifact_dir = tempfile::tempdir()?;
    tracing::debug!(dir = %artifact_dir.path().display(), "building artifacts");
    let output = tokio::process::Command::new("python3")
        .arg(code_file.path())
        .arg("--")
        .arg("build")
        .arg("--dir")
        .arg(artifact_dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?
        .wait_with_output()
        .await?;
    if !output.status.success() {
        let status = output.status;
        let stderr = String::from_utf8(output.stderr).unwrap_or_default();
        let stdout = String::from_utf8(output.stdout).unwrap_or_default();
        anyhow::bail!("python command failed: {} {} {}", status, stderr, stdout);
    }
    tracing::debug!(elapsed = ?start.elapsed(), "built artifacts");

    let file_name = format!("{}.tar", uuid::Uuid::now_v7());
    let path = std::env::temp_dir().join(&file_name);
    let file = std::fs::File::create(path)?;
    let buf = std::io::BufWriter::new(file);
    let mut ar = tar::Builder::new(buf);
    ar.append_dir_all("artifacts", artifact_dir.path())?;
    let artifacts = ar.into_inner()?.into_inner()?;
    artifacts.sync_all()?;

    Ok(file_name)
}

async fn test(code: String, results_file: String) -> anyhow::Result<TestResp> {
    let tempdir = tempfile::tempdir()?;
    let code_file = tempdir.path().join("test_code.py");
    tracing::debug!(len = code.len(), file = %code_file.display(), "writing code to temp file");
    std::fs::write(&code_file, code)?;

    let report = tempdir.path().join("report.json");

    let start = Instant::now();
    let mut cmd = tokio::process::Command::new("pytest");

    if !results_file.is_empty() {
        let results_file_path = std::env::temp_dir().join(&results_file);
        let results_file = std::fs::File::open(&results_file_path)?;
        let metadata = results_file.metadata()?;
        tracing::debug!(path = %results_file_path.display(), size = metadata.len(), "unpacking");
        let mut tar = tar::Archive::new(results_file);
        tar.unpack(tempdir.path())?;
        let results = tempdir.path().join("results");
        cmd.arg("--batch-results").arg(&results);
    }

    let status = cmd
        .arg(&code_file)
        .arg("--json-report")
        .arg("--json-report-file")
        .arg(&report)
        .spawn()?
        .wait()
        .await?;
    let report = std::fs::read_to_string(report)?;
    tracing::debug!(elapsed = ?start.elapsed(), status = ?status.code(), "tested results");

    if !results_file.is_empty() {
        let results_file_path = std::env::temp_dir().join(&results_file);
        if let Err(err) = std::fs::remove_file(results_file_path) {
            tracing::error!(?err, "failed to delete results file");
        }
    }

    Ok(TestResp { report })
}
