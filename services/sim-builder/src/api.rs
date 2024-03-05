use std::io::Write;
use std::time::Instant;

use elodin_types::sandbox::{build_sim_server::BuildSim, BuildReq, BuildResp};
use tonic::async_trait;
use tonic::{Request, Response, Status};

pub use elodin_types::sandbox::build_sim_server::BuildSimServer;

#[derive(Default)]
pub struct Service();

impl Service {
    pub async fn set_serving(reporter: &mut tonic_health::server::HealthReporter) {
        reporter.set_serving::<BuildSimServer<Self>>().await;
    }
}

#[async_trait]
impl BuildSim for Service {
    async fn build(&self, req: Request<BuildReq>) -> Result<Response<BuildResp>, Status> {
        let req = req.into_inner();

        let span = tracing::Span::current();
        let build_task = tokio::task::spawn_blocking(move || {
            let _guard = span.enter();
            build(req.code)
        });
        let artifacts = build_task.await.unwrap().map_err(|err| {
            tracing::error!(?err, "failed to build code");
            Status::internal(err.to_string())
        })?;

        Ok(Response::new(BuildResp { artifacts }))
    }
}

fn build(code: String) -> anyhow::Result<Vec<u8>> {
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

    let buf = Vec::default();
    let mut ar = tar::Builder::new(buf);
    ar.append_dir_all("artifacts", artifact_dir.path())?;
    let artifacts = ar.into_inner()?;
    Ok(artifacts)
}
