use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::net::TcpStream;

use super::Cli;
use bevy::{prelude::*, utils::tracing};
use conduit::{client::MsgPair, server::handle_socket};
use notify::Watcher;
use nox_ecs::{ConduitExec, WorldExec};

#[derive(clap::Args, Clone)]
pub struct Args {
    #[clap(name = "addr/path")]
    sim: Simulator,
}

#[derive(Clone)]
enum Simulator {
    Addr(SocketAddr),
    File(PathBuf),
}

impl std::str::FromStr for Simulator {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        if let Ok(addr) = s.parse() {
            Ok(Self::Addr(addr))
        } else {
            Ok(Self::File(s.into()))
        }
    }
}

impl Cli {
    pub fn editor(&self, args: Args) -> anyhow::Result<()> {
        use conduit::bevy::{ConduitSubscribePlugin, Subscriptions};
        use conduit::bevy_sync::SyncPlugin;
        use elodin_editor::EditorPlugin;
        let (sub, bevy_tx) = ConduitSubscribePlugin::pair();

        let (addr, path) = match args.sim {
            Simulator::Addr(addr) => (addr, None),
            Simulator::File(path) => ("127.0.0.1:2240".parse()?, Some(path)),
        };

        if let Some(path) = &path {
            std::process::Command::new("python3")
                .arg(path)
                .arg("--")
                .arg("repl")
                .arg(addr.to_string())
                .spawn()?;
        }

        App::new()
            .add_plugins(EditorPlugin)
            .add_plugins(SimClient { addr, bevy_tx })
            .add_plugins(SimSupervisor { path })
            .add_plugins(SyncPlugin {
                plugin: sub,
                subscriptions: Subscriptions::default(),
            })
            .run();

        Ok(())
    }
}

#[derive(Clone)]
struct SimClient {
    addr: SocketAddr,
    bevy_tx: flume::Sender<MsgPair>,
}

#[derive(Clone)]
struct SimSupervisor {
    path: Option<PathBuf>,
}

impl Plugin for SimClient {
    fn build(&self, _: &mut App) {
        let c = self.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime failed to start");
            rt.block_on(async move {
                loop {
                    let Ok(socket) = TcpStream::connect(c.addr).await else {
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        continue;
                    };
                    let (rx_socket, tx_socket) = socket.into_split();

                    if let Err(err) = handle_socket(c.bevy_tx.clone(), tx_socket, rx_socket).await {
                        tracing::warn!(?err, "socket error");
                    }
                }
            });
        });
    }
}

impl Plugin for SimSupervisor {
    fn build(&self, _: &mut App) {
        let Some(path) = self.path.clone() else {
            return;
        };
        std::thread::spawn(move || {
            if let Err(err) = Self::run(path) {
                tracing::error!(?err);
            }
        });
    }
}

impl SimSupervisor {
    fn run(path: PathBuf) -> anyhow::Result<()> {
        let addr = "0.0.0.0:2240".parse::<SocketAddr>().unwrap();
        let (notify_tx, notify_rx) = flume::bounded(1);
        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                if let Ok(event) = res {
                    tracing::debug!(?event, "received notify");
                    let _ = notify_tx.try_send(());
                }
            })?;
        watcher.watch(&path, notify::RecursiveMode::NonRecursive)?;

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
                    tracing::error!(?err, "failed to run conduit exec");
                    return Err(err.into());
                }
                let sleep_time = conduit_exec.time_step().saturating_sub(start.elapsed());
                std::thread::sleep(sleep_time);

                if let Ok(exec) = exec_rx.try_recv() {
                    tracing::info!("received new code, updating sim");
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
        let status = std::process::Command::new("python3")
            .arg(path)
            .arg("--")
            .arg("build")
            .arg("--dir")
            .arg(tmpdir.path())
            .spawn()?
            .wait()?;
        if !status.success() {
            anyhow::bail!("failed to build sim: {}", status);
        }
        let exec = nox_ecs::WorldExec::read_from_dir(tmpdir.path())?;
        tracing::info!(elapsed = ?start.elapsed(), "built sim");
        self.exec_tx.send(exec)?;
        Ok(())
    }
}
