use crate::*;
use bevy::app::{Plugin, Update};
use impeller2::types::FilledRecycle;
use impeller2::types::LenPacket;
use impeller2_stella::thingbuf::tcp_connect;
use std::{net::SocketAddr, time::Duration};
use thingbuf::mpsc;

pub struct TcpImpellerPlugin {
    addr: SocketAddr,
}

impl TcpImpellerPlugin {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }
}

impl Plugin for TcpImpellerPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        let addr = self.addr;
        let (mut incoming_packet_tx, incoming_packet_rx) = mpsc::with_recycle(512, FilledRecycle);
        let (outgoing_packet_tx, mut outgoing_packet_rx) = mpsc::channel::<Option<LenPacket>>(512);
        let stream_id = fastrand::u64(..);
        std::thread::spawn(move || {
            let res: Result<(), miette::Error> = stellarator::run(|| async move {
                loop {
                    if let Err(err) = tcp_connect(
                        addr,
                        &mut outgoing_packet_rx,
                        &mut incoming_packet_tx,
                        stream_id,
                        &new_connection_packets,
                    )
                    .await
                    {
                        bevy::log::trace!(?err, "connection ended with error");
                        stellarator::sleep(Duration::from_millis(50)).await;
                    }
                }
            });
            if let Err(err) = res {
                bevy::log::error!(?err, "tcp plugin error");
            }
        });
        app.insert_resource(PacketTx(outgoing_packet_tx))
            .insert_resource(PacketRx(incoming_packet_rx))
            .insert_resource(CurrentStreamId(stream_id))
            .add_systems(Update, sink);
    }
}
