use std::iter;
use std::process::Stdio;
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
use std::time::Duration;

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

#[cfg(target_os = "linux")]
fn collect_process_tree(root_pid: nix::unistd::Pid) -> Vec<nix::unistd::Pid> {
    let mut stack = vec![root_pid];
    let mut seen = HashSet::new();
    let mut pids = Vec::new();

    while let Some(pid) = stack.pop() {
        if !seen.insert(pid.as_raw()) {
            continue;
        }

        pids.push(pid);
        let tasks_dir = format!("/proc/{}/task", pid.as_raw());
        let Ok(tasks) = fs::read_dir(tasks_dir) else {
            continue;
        };

        for task in tasks.flatten() {
            let children_path = task.path().join("children");
            let Ok(children) = fs::read_to_string(children_path) else {
                continue;
            };

            for child in children
                .split_whitespace()
                .filter_map(|raw| raw.parse::<i32>().ok())
            {
                stack.push(nix::unistd::Pid::from_raw(child));
            }
        }
    }

    pids
}

#[cfg(not(target_os = "linux"))]
fn collect_process_tree(root_pid: nix::unistd::Pid) -> Vec<nix::unistd::Pid> {
    vec![root_pid]
}

#[cfg(unix)]
fn signal_process_tree(root_pid: nix::unistd::Pid, signal: nix::sys::signal::Signal) {
    let pids = collect_process_tree(root_pid);
    tracing::debug!(
        root_pid = root_pid.as_raw(),
        pid_count = pids.len(),
        ?signal,
        "signalling sim process tree"
    );

    for pid in pids.into_iter().rev() {
        let _ = nix::sys::signal::kill(pid, signal);
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
fn configure_sim_command(cmd: &mut Command) {
    cmd.process_group(0);
}

#[cfg(target_os = "linux")]
fn configure_sim_command(_cmd: &mut Command) {}

#[cfg(all(unix, not(target_os = "linux")))]
fn signal_process_group(root_pid: nix::unistd::Pid, signal: nix::sys::signal::Signal) {
    let _ = nix::sys::signal::killpg(root_pid, signal);
}

#[cfg(target_os = "linux")]
fn signal_process_group(_root_pid: nix::unistd::Pid, _signal: nix::sys::signal::Signal) {}

impl SimRecipe {
    pub async fn run(self, cancel_token: CancelToken) -> Result<(), Error> {
        debug!("running sim");

        let mut cmd = python_tokio_command()?;
        configure_sim_command(&mut cmd);
        // Close stdin to prevent SIGTTIN when child is in background process group
        cmd.stdin(Stdio::null());
        cmd.env("TRACY_PORT", "8089");
        let port = crate::liveness::serve_tokio().await?;
        let mut child = cmd
            .arg(&self.path)
            .arg("run")
            .arg("--no-s10")
            .arg("--liveness-port")
            .arg(port.to_string())
            .spawn()?;
        let child_pid = child.id().map(|pid| nix::unistd::Pid::from_raw(pid as i32));

        tokio::select! {
            _ = cancel_token.wait() => {
                if let Some(pid) = child_pid {
                    signal_process_group(pid, nix::sys::signal::Signal::SIGTERM);
                    signal_process_tree(pid, nix::sys::signal::Signal::SIGTERM);
                }
                tracing::info!("Waiting for sim process tree to exit");
                match tokio::time::timeout(Duration::from_secs(2), child.wait()).await {
                    Ok(res) => {
                        let _ = res;
                    }
                    Err(_) => {
                        tracing::warn!("Sim process did not exit after SIGTERM, forcing kill");
                        if let Some(pid) = child_pid {
                            signal_process_group(pid, nix::sys::signal::Signal::SIGKILL);
                            signal_process_tree(pid, nix::sys::signal::Signal::SIGKILL);
                        }
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                    }
                }
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
    let venv_python = std::path::Path::new(".venv/bin/python");
    if venv_python.exists() {
        let mut cmd = std::process::Command::new(venv_python);
        // When built with tracy, the nox-py .so is large enough (IREE+TracyClient)
        // to exceed the default static TLS reservation. Increase the optional
        // static TLS allocation so dlopen() succeeds. Unlike LD_PRELOAD, this
        // env var is safe to inherit into child processes (e.g. iree-compile).
        if std::env::var("TRACY_PORT").is_ok() {
            cmd.env("GLIBC_TUNABLES", "glibc.rtld.optional_static_tls=16384");
        }
        return Ok(cmd);
    }
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
