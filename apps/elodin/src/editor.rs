use std::net::SocketAddr;
use tokio::net::TcpStream;

use crate::Cli;
use bevy::prelude::App;
use elodin_conduit::bevy_sync::DEFAULT_SUB_FILTERS;

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
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let stream = TcpStream::connect(args.addr).await?;
                let (rx_socket, tx_socket) = stream.into_split();
                elodin_conduit::bevy::handle_socket(
                    bevy_tx.clone(),
                    tx_socket,
                    rx_socket,
                    DEFAULT_SUB_FILTERS,
                )
                .await;
                anyhow::Ok(())
            })
            .unwrap();
        });
        let mut app = App::new();
        app.add_plugins(EditorPlugin)
            .add_plugins(SyncPlugin {
                plugin: sub,
                subscriptions: Subscriptions::default(),
            })
            .run();

        Ok(())
    }
}
