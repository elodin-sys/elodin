use std::iter;
use std::{
    net::{Ipv4Addr, SocketAddr},
    path::PathBuf,
};
use stellarator::util::CancelToken;
use tokio::process::Command;
use tracing::{debug, error};
use which::which;

use crate::DEFAULT_WATCH_TIMEOUT;
use crate::{error::Error, watch::watch};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
#[derive(Debug, Clone)]
pub struct SimRecipe {
    pub path: PathBuf,
    #[serde(default = "default_addr")]
    pub addr: SocketAddr,
    #[serde(default)]
    pub optimize: bool,
}

fn default_addr() -> SocketAddr {
    SocketAddr::new(Ipv4Addr::new(0, 0, 0, 0).into(), 2240)
}

impl SimRecipe {
    pub async fn run(self, cancel_token: CancelToken) -> Result<(), Error> {
        debug!("running sim");

        let mut cmd = python_tokio_command()?;
        if !cfg!(target_os = "linux") {
            cmd.process_group(0); // NOTE(sphw): this causes all sorts of issues on linux, not sure why
        }
        let port = crate::liveness::serve_tokio().await?;
        let mut child = cmd
            .arg(&self.path)
            .arg("run")
            .arg("--no-s10")
            .arg("--liveness-port")
            .arg(port.to_string())
            .spawn()?;

        tokio::select! {
            _ = cancel_token.wait() => {
                if let Some(pid) = child.id() {
                    let _ = nix::sys::signal::killpg(
                        nix::unistd::Pid::from_raw(pid as i32),
                        nix::sys::signal::Signal::SIGTERM,
                    );
                }
                tracing::info!("Waiting for sim process to exit");
                let _ = child.wait().await;
                Ok(())
            }
            res = child.wait() => {
                let status = res?;
                if !status.success() {
                    Err(Error::SimBuildFailed(status.code()))
                } else {
                    Ok(())
                }
            }
        }
    }

    pub async fn watch(self, cancel_token: CancelToken) -> Result<(), Error> {
        let dir = if self.path.is_dir() {
            self.path.clone()
        } else {
            let path = std::fs::canonicalize(&self.path)?;
            path.parent()
                .expect("path does not have a parent directory")
                .to_path_buf()
        };
        watch(
            DEFAULT_WATCH_TIMEOUT,
            |token| {
                let this = self.clone();
                async move {
                    if let Err(err) = this.run(token).await {
                        error!(?err, "error running sim");
                    }
                    Ok(())
                }
            },
            cancel_token,
            iter::once(dir),
        )
        .await
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
