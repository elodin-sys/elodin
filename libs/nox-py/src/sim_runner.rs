use clap::Parser;
use conduit::client::MsgPair;
use nox_ecs::{Compiled, ConduitExec, WorldExec};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    thread::JoinHandle,
    time::{Duration, Instant},
};
use tracing::{debug, error, info, trace};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub enum Args {
    Build {
        #[arg(long)]
        dir: PathBuf,
    },
    Repl {
        #[arg(default_value = "0.0.0.0:2240")]
        addr: SocketAddr,
    },
    Run {
        #[arg(default_value = "0.0.0.0:2240")]
        addr: SocketAddr,
        #[arg(long)]
        no_repl: bool,
        #[arg(long)]
        watch: bool,
    },
    #[clap(hide = true)]
    Bench {
        #[arg(long, default_value = "1000")]
        ticks: usize,
    },
}

pub struct SimSupervisor;

impl SimSupervisor {
    pub fn spawn(path: PathBuf) -> JoinHandle<anyhow::Result<()>> {
        std::thread::spawn(move || Self::run(path))
    }

    pub fn run(path: PathBuf) -> anyhow::Result<()> {
        let parent_dir = path.parent().unwrap();

        let addr = "0.0.0.0:2240".parse::<SocketAddr>().unwrap();
        let (notify_tx, notify_rx) = flume::bounded(0);
        let mut debouncer = notify_debouncer_mini::new_debouncer(
            Duration::from_millis(500),
            move |res: Result<Vec<notify_debouncer_mini::DebouncedEvent>, _>| {
                if let Ok(events) = res {
                    let pycache_only = events
                        .iter()
                        .all(|event| event.path.ancestors().any(|p| p.ends_with("__pycache__")));
                    if pycache_only {
                        debug!("ignoring __pycache__ changes");
                        return;
                    }
                    for event in events {
                        info!(path = %event.path.display(), "detected change");
                    }
                    let _ = notify_tx.try_send(());
                }
            },
        )?;

        debouncer
            .watcher()
            .watch(parent_dir, notify::RecursiveMode::Recursive)?;

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
    exec_tx: flume::Sender<WorldExec<Compiled>>,
}

impl SimRunner {
    fn new(server_rx: flume::Receiver<MsgPair>) -> Self {
        let (exec_tx, exec_rx) = flume::bounded(1);
        std::thread::spawn(move || -> anyhow::Result<()> {
            let exec: WorldExec<Compiled> = exec_rx.recv()?;
            let mut conduit_exec = ConduitExec::new(exec, server_rx.clone());
            let mut start = Instant::now();
            let time_step = conduit_exec.time_step();
            loop {
                if let Err(err) = conduit_exec.run() {
                    error!(?err, "failed to run conduit exec");
                    return Err(err.into());
                }
                let sleep_time = time_step.saturating_sub(start.elapsed());
                std::thread::sleep(sleep_time);
                start += time_step;

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
        let mut start = Instant::now();
        info!("building sim");

        let status = std::process::Command::new("python")
            .arg(path)
            .arg("build")
            .arg("--dir")
            .arg(tmpdir.path())
            .status()?;

        if !status.success() {
            anyhow::bail!("failed to build sim: {}", status);
        }

        let exec = nox_ecs::WorldExec::read_from_dir(tmpdir.path())?;
        info!(elapsed = ?start.elapsed(), "built sim");
        start = Instant::now();
        let client = nox_ecs::nox::Client::cpu()?;
        let exec = exec.compile(client)?;
        info!(elapsed = ?start.elapsed(), "compiled sim");
        self.exec_tx.send(exec)?;
        Ok(())
    }
}
