use nox_ecs::{
    impeller::{self, client::MsgPair, Connection},
    nox, Compiled, ImpellerExec, WorldExec,
};
use std::{
    iter,
    net::{Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
use tokio::{process::Command, sync::Mutex};
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use which::which;

use crate::{error::Error, recipe::DEFAULT_WATCH_TIMEOUT, watch::watch};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
#[derive(Debug, Clone)]
pub struct SimRecipe {
    pub path: PathBuf,
    #[serde(default = "default_addr")]
    pub addr: SocketAddr,
}

fn default_addr() -> SocketAddr {
    SocketAddr::new(Ipv4Addr::new(0, 0, 0, 0).into(), 2240)
}

impl SimRecipe {
    async fn build_with_client(&self, client: nox::Client) -> Result<WorldExec<Compiled>, Error> {
        let tmpdir = tempfile::tempdir()?;
        let mut start = Instant::now();
        info!("building sim");

        let status = python_tokio_command()?
            .arg(&self.path)
            .arg("build")
            .arg("--dir")
            .arg(tmpdir.path())
            .status()
            .await?;

        if !status.success() {
            return Err(Error::SimBuildFailed(status.code()));
        }

        let exec = nox_ecs::WorldExec::read_from_dir(tmpdir.path())?;
        info!(elapsed = ?start.elapsed(), "built sim");
        start = Instant::now();
        let exec = exec.compile(client)?;
        info!(elapsed = ?start.elapsed(), "compiled sim");
        Ok(exec)
    }
    async fn build(&self) -> Result<WorldExec<Compiled>, Error> {
        let client = nox_ecs::nox::Client::cpu().map_err(nox_ecs::Error::from)?;
        self.build_with_client(client).await
    }

    pub async fn run(self, cancel_token: CancellationToken) -> Result<(), Error> {
        let exec = self.build().await?;
        let (tx, rx) = flume::unbounded();
        let server = impeller::server::TcpServer::bind(tx, self.addr)
            .await
            .map_err(nox_ecs::Error::from)?;
        let exec = tokio::task::spawn_blocking(move || {
            run_exec(exec, rx, cancel_token, std::iter::empty()).map(|_| ())
        });
        tokio::select! {
            res = server.run() => res.map_err(nox_ecs::Error::from).map_err(Error::from),
            res = exec => res.map(|_| Ok(())).map_err(|_| Error::JoinError)?,
        }
    }

    pub async fn watch(self, cancel_token: CancellationToken) -> Result<(), Error> {
        let dir = if self.path.is_dir() {
            self.path.clone()
        } else {
            let path = std::fs::canonicalize(&self.path)?;
            path.parent()
                .expect("path does not have a parent directory")
                .to_path_buf()
        };
        let (tx, rx) = flume::unbounded();
        let server = impeller::server::TcpServer::bind(tx, self.addr)
            .await
            .map_err(nox_ecs::Error::from)?;
        let connections = Arc::new(Mutex::new(vec![]));
        let client = nox::Client::cpu().map_err(nox_ecs::Error::from)?;
        let watch = watch(
            DEFAULT_WATCH_TIMEOUT,
            |token| {
                let client = client.clone();
                let rx = rx.clone();
                let this = self.clone();
                let existing_conns = connections.clone();
                async move {
                    let exec = this.build_with_client(client).await?;
                    let mut conns = {
                        let mut guard = existing_conns.lock().await;
                        std::mem::take(&mut *guard)
                    };
                    let conns = tokio::task::spawn_blocking(move || {
                        run_exec(exec, rx.clone(), token, conns.drain(..))
                    })
                    .await
                    .map_err(|_| Error::JoinError)??;
                    let mut existing_conns = existing_conns.lock().await;
                    *existing_conns = conns;
                    Ok(())
                }
            },
            cancel_token,
            iter::once(dir),
        );
        tokio::select! {
            res = server.run() => res.map_err(nox_ecs::Error::from).map_err(Error::from),
            res = watch => res,
        }
    }
}

fn run_exec(
    exec: WorldExec<Compiled>,
    server_rx: flume::Receiver<MsgPair>,
    cancel_token: CancellationToken,
    existing_connections: impl Iterator<Item = Connection>,
) -> Result<Vec<Connection>, Error> {
    let mut impeller_exec = ImpellerExec::new(exec, server_rx.clone());
    for conn in existing_connections {
        impeller_exec.add_connection(conn)?;
    }
    let mut start = Instant::now();
    let time_step = impeller_exec.run_time_step();
    loop {
        if let Err(err) = impeller_exec.run() {
            error!(?err, "failed to run impeller exec");
            return Err(err.into());
        }
        let sleep_time = time_step.saturating_sub(start.elapsed());
        std::thread::sleep(sleep_time);
        start += time_step;
        if cancel_token.is_cancelled() {
            return Ok(impeller_exec.into_connections());
        }
    }
}

pub fn python_command() -> Result<std::process::Command, Error> {
    if let Ok(uv) = which("uv") {
        let mut cmd = std::process::Command::new(uv);
        cmd.arg("run");
        Ok(cmd)
    } else if let Ok(py) = which("python3") {
        Ok(std::process::Command::new(py))
    } else {
        Err(Error::PythonNotFound)
    }
}

pub fn python_tokio_command() -> Result<Command, Error> {
    Ok(python_command()?.into())
}
