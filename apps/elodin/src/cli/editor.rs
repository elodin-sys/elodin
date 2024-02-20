use std::net::SocketAddr;
use tokio::net::TcpStream;

use crate::Cli;
use bevy::{prelude::*, utils::tracing};
use elodin_conduit::{client::MsgPair, server::handle_socket};

#[derive(clap::Args, Clone)]
pub struct Args {
    addr: SocketAddr,
}

impl Cli {
    pub fn editor(&self, args: Args) -> anyhow::Result<()> {
        use elodin_conduit::bevy::{ConduitSubscribePlugin, Subscriptions};
        use elodin_conduit::bevy_sync::SyncPlugin;
        use elodin_editor::EditorPlugin;
        let (sub, bevy_tx) = ConduitSubscribePlugin::pair();

        App::new()
            .add_plugins(SimClient {
                addr: args.addr,
                bevy_tx,
            })
            .add_plugins(EditorPlugin)
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
