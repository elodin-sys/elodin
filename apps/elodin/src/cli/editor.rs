use std::net::SocketAddr;
use tokio::net::TcpStream;

use crate::Cli;
use bevy::{prelude::App, utils::tracing};
use elodin_conduit::server::handle_socket;

#[derive(clap::Args, Clone)]
pub struct Args {
    addr: SocketAddr,
}

impl Cli {
    pub async fn editor(&self, args: Args) -> anyhow::Result<()> {
        use elodin_conduit::bevy::{ConduitSubscribePlugin, Subscriptions};
        use elodin_conduit::bevy_sync::SyncPlugin;
        use elodin_editor::EditorPlugin;
        let (sub, bevy_tx) = ConduitSubscribePlugin::pair();
        tokio::spawn(async move {
            loop {
                let Ok(socket) = TcpStream::connect(args.addr).await else {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    continue;
                };
                let (rx_socket, tx_socket) = socket.into_split();
                if let Err(err) = handle_socket(bevy_tx.clone(), tx_socket, rx_socket).await {
                    tracing::warn!(?err, "socket error");
                }
            }
        });

        let mut app = App::new();
        app.add_plugins(EditorPlugin).add_plugins(SyncPlugin {
            plugin: sub,
            subscriptions: Subscriptions::default(),
        });
        tokio::task::block_in_place(|| app.run());

        Ok(())
    }
}
