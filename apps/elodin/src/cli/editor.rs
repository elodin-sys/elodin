use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::net::TcpStream;

use crate::Cli;
use bevy::{prelude::*, utils::tracing};
use elodin_conduit::{client::MsgPair, server::handle_socket};
use notify::Watcher;

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
        use elodin_conduit::bevy::{ConduitSubscribePlugin, Subscriptions};
        use elodin_conduit::bevy_sync::SyncPlugin;
        use elodin_editor::EditorPlugin;
        let (sub, bevy_tx) = ConduitSubscribePlugin::pair();

        let (addr, path) = match args.sim {
            Simulator::Addr(addr) => (addr, None),
            Simulator::File(path) => ("127.0.0.1:2240".parse()?, Some(path)),
        };

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
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                if let Ok(event) = res {
                    tracing::debug!(?event, "received notify");
                    let _ = tx.try_send(());
                }
            })?;
        watcher.watch(&path, notify::RecursiveMode::NonRecursive)?;
        loop {
            let mut sim = std::process::Command::new("python3")
                .arg(&path)
                .arg("--")
                .arg("run")
                .spawn()?;
            // 500ms debounce
            std::thread::sleep(std::time::Duration::from_millis(500));
            if rx.recv().is_err() {
                break;
            }
            println!("{} was updated, restarting sim ...", path.display());
            sim.kill()?;
        }
        Ok(())
    }
}
