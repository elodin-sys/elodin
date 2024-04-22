use anyhow::Context;
use bevy::window::WindowResized;
use core::fmt;
use std::io::{Read, Seek, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::path::PathBuf;
use tokio::net::TcpStream;

use super::Cli;
use bevy::{prelude::*, utils::tracing};
use conduit::{client::MsgPair, server::handle_socket};

const DEFAULT_SIM: Simulator =
    Simulator::Addr(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 2240)));

#[derive(clap::Args, Clone, Default)]
pub struct Args {
    #[clap(name = "addr/path", default_value_t = DEFAULT_SIM)]
    sim: Simulator,
}

#[derive(Clone)]
enum Simulator {
    Addr(SocketAddr),
    File(PathBuf),
}

#[derive(Resource)]
struct WindowStateFile(std::fs::File);

impl Default for Simulator {
    fn default() -> Self {
        DEFAULT_SIM
    }
}

impl fmt::Display for Simulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Addr(addr) => write!(f, "{}", addr),
            Self::File(path) => write!(f, "{}", path.display()),
        }
    }
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
                .arg("run")
                .arg("--watch")
                .arg(addr.to_string())
                .spawn()?;
        }

        let mut window_state_file = self.window_state_file()?;
        let mut window_state = String::new();
        window_state_file.read_to_string(&mut window_state)?;
        let editor_plugin = if let [width, height] = window_state
            .split_whitespace()
            .collect::<Vec<_>>()
            .as_slice()
        {
            let width = width.parse::<f32>()?;
            let height = height.parse::<f32>()?;
            EditorPlugin::new(width, height)
        } else {
            EditorPlugin::default()
        };

        App::new()
            .insert_resource(WindowStateFile(window_state_file))
            .add_plugins(editor_plugin)
            .add_plugins(SimClient { addr, bevy_tx })
            .add_plugins(SyncPlugin {
                plugin: sub,
                subscriptions: Subscriptions::default(),
                enable_pbr: true,
            })
            .add_systems(Update, on_window_resize)
            .run();

        Ok(())
    }

    fn window_state_file(&self) -> anyhow::Result<std::fs::File> {
        let dirs = self.dirs().context("failed to get data directory")?;
        let data_dir = dirs.data_dir();
        std::fs::create_dir_all(data_dir).context("failed to create data directory")?;
        let window_state_path = data_dir.join(".window-state");
        std::fs::File::options()
            .write(true)
            .read(true)
            .create(true)
            .truncate(false)
            .open(window_state_path)
            .context("failed to open window state file")
    }
}

fn on_window_resize(
    mut window_state_file: ResMut<WindowStateFile>,
    mut resize_reader: EventReader<WindowResized>,
) {
    if let Some(e) = resize_reader.read().last() {
        let window_state = format!("{:.1} {:.1}\n", e.width, e.height);
        if let Err(err) = window_state_file.0.rewind() {
            tracing::warn!(?err, "failed to rewind window state file");
            return;
        }
        if let Err(err) = window_state_file.0.write_all(window_state.as_bytes()) {
            tracing::warn!(?err, "failed to write window state");
        }
    }
}

#[derive(Clone)]
struct SimClient {
    addr: SocketAddr,
    bevy_tx: flume::Sender<MsgPair>,
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
