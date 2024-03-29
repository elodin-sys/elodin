use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use conduit::client::MsgPair;
use nox_ecs::{ConduitExec, WorldExec};
use pyo3::Python;
use tracing::{debug, error, info, trace};

pub struct SimSupervisor;

impl SimSupervisor {
    pub fn spawn(path: PathBuf) -> JoinHandle<anyhow::Result<()>> {
        std::thread::spawn(move || Self::run(path))
    }

    pub fn run(path: PathBuf) -> anyhow::Result<()> {
        let addr = "0.0.0.0:2240".parse::<SocketAddr>().unwrap();
        let (notify_tx, notify_rx) = flume::bounded(1);
        let mut debouncer =
            notify_debouncer_mini::new_debouncer(Duration::from_millis(500), move |res| {
                if let Ok(event) = res {
                    debug!(?event, "received notify");
                    let _ = notify_tx.try_send(());
                }
            })?;

        debouncer
            .watcher()
            .watch(&path, notify::RecursiveMode::NonRecursive)?;

        let (tx, rx) = flume::unbounded();
        let sim_runner = SimRunner::new(rx);
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let server = conduit::server::TcpServer::bind(tx, addr).await.unwrap();
                server.run().await
            })
            .unwrap();
        });

        loop {
            let _ = sim_runner.try_update_sim(&path).inspect_err(eprint_err);
            notify_rx.recv().unwrap();
        }
    }
}

fn eprint_err<E: std::fmt::Debug>(err: &E) {
    eprintln!("{err:?}");
}

#[derive(Clone)]
struct SimRunner {
    exec_tx: flume::Sender<WorldExec>,
}

impl SimRunner {
    fn new(server_rx: flume::Receiver<MsgPair>) -> Self {
        let (exec_tx, exec_rx) = flume::bounded(1);
        std::thread::spawn(move || -> anyhow::Result<()> {
            let client = nox_ecs::nox::Client::cpu()?;
            let exec: WorldExec = exec_rx.recv()?;
            let mut conduit_exec = ConduitExec::new(exec, server_rx.clone());
            loop {
                let start = Instant::now();
                if let Err(err) = conduit_exec.run(&client) {
                    error!(?err, "failed to run conduit exec");
                    return Err(err.into());
                }
                let sleep_time = conduit_exec.time_step().saturating_sub(start.elapsed());
                std::thread::sleep(sleep_time);

                if let Ok(exec) = exec_rx.try_recv() {
                    trace!("received new code, updating sim");
                    let conns = conduit_exec.connections().to_vec();
                    conduit_exec = ConduitExec::new(exec, server_rx.clone());
                    for conn in conns {
                        conduit_exec.add_connection(conn)?;
                    }
                }
            }
        });
        Self { exec_tx }
    }

    fn try_update_sim(&self, path: &Path) -> anyhow::Result<()> {
        let tmpdir = tempfile::tempdir()?;
        let start = Instant::now();
        info!("building sim");
        let script = std::fs::read_to_string(path)?;
        std::env::set_var("ELODIN_FORCE_BUILD_DIR", tmpdir.path());
        Python::with_gil(|py| py.run(&script, None, None))?;

        let exec = nox_ecs::WorldExec::read_from_dir(tmpdir.path())?;
        info!(elapsed = ?start.elapsed(), "built sim");
        self.exec_tx.send(exec)?;
        Ok(())
    }
}
