use std::io::Write;
use std::time::Instant;

use elodin_types::sandbox::{sandbox_server::Sandbox, *};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
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

        let span = tracing::Span::current();
        let build_task = tokio::task::spawn_blocking(move || {
            let _guard = span.enter();
            build(req.code)
        });
        let artifacts_file = build_task.await.unwrap().map_err(|err| {
            tracing::error!(?err, "failed to build code");
            Status::internal(err.to_string())
        })?;

        Ok(Response::new(BuildResp { artifacts_file }))
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
        let mut file = tokio::fs::File::create(&path).await?;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk.data).await?;
            tracing::debug!(len = chunk.data.len(), path = %path.display(), "wrote chunk to file");
        }
        tracing::info!(path = %path.display(), "received file");
        Ok(Response::new(FileTransferResp::default()))
    }

    async fn recv_file(
        &self,
        req: Request<FileTransferReq>,
    ) -> Result<Response<Self::RecvFileStream>, Status> {
        let req = req.into_inner();
        let path = std::env::temp_dir().join(&req.name);
        let file = tokio::fs::File::open(&path).await?;
        let (tx, rx) = mpsc::channel::<Result<FileChunk, Status>>(1);
        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(file);
            let mut buf = Vec::with_capacity(128 * 1024);
            loop {
                buf.clear();
                match reader.read_buf(&mut buf).await {
                    Ok(0) => break,
                    Ok(len) => {
                        let chunk = FileChunk {
                            data: buf.clone(),
                            name: req.name.clone(),
                        };
                        let _ = tx.send(Ok(chunk)).await;
                        tracing::debug!(len, path = %path.display(), "sent chunk");
                    }
                    Err(err) => {
                        tracing::error!(?err, "failed to read file");
                        let err = Err(Status::internal(err.to_string()));
                        let _ = tx.send(err).await;
                        return;
                    }
                }
                if tx.is_closed() {
                    break;
                }
            }
            tracing::info!(path = %path.display(), "sent file");
            if let Err(err) = tokio::fs::remove_file(&path).await {
                tracing::error!(?err, "failed to delete file");
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn build(code: String) -> anyhow::Result<String> {
    tracing::debug!(len = code.len(), "building code");
    let mut code_file = tempfile::NamedTempFile::new()?;
    tracing::debug!(file = %code_file.path().display(), "writing code to temp file");
    code_file.write_all(code.as_bytes())?;

    let start = Instant::now();
    let artifact_dir = tempfile::tempdir()?;
    tracing::debug!(dir = %artifact_dir.path().display(), "building artifacts");
    let status = std::process::Command::new("python3")
        .arg(code_file.path())
        .arg("--")
        .arg("build")
        .arg("--dir")
        .arg(artifact_dir.path())
        .spawn()?
        .wait()?;
    if !status.success() {
        anyhow::bail!("python command failed: {}", status);
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
