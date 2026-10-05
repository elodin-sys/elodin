use crate::*;
use bevy::app::{Plugin, PreUpdate};
use impeller::types::LenPacket;
use impeller_bbq::AsyncArcQueueTx;
use impeller_stellar::queue::tcp_connect;
use impeller_wkt::StreamId;
use std::{net::SocketAddr, time::Duration};
use thingbuf::mpsc;

pub struct TcpImpellerPlugin {
    addr: Option<SocketAddr>,
}

impl TcpImpellerPlugin {
    pub fn new(addr: Option<SocketAddr>) -> Self {
        Self { addr }
    }
}

impl Plugin for TcpImpellerPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        let (packet_tx, packet_rx, outgoing_packet_rx, incoming_packet_tx) =
            crate::channels::channels();
        let (msg_tx, msg_rx, msg_outgoing_rx, msg_incoming_tx) = crate::channels::msg_channels();
        let stream_id = fastrand::u64(..);
        let status = if let Some(addr) = self.addr {
            app.insert_resource(ConnectionAddr(addr));
            spawn_msg_tcp_connect(addr, msg_outgoing_rx, msg_incoming_tx);
            spawn_tcp_connect(
                addr,
                outgoing_packet_rx,
                incoming_packet_tx,
                stream_id,
                true,
            )
        } else {
            ThreadConnectionStatus::new(ConnectionStatus::NoConnection)
        };
        app.insert_resource(packet_tx)
            .insert_resource(packet_rx)
            .insert_resource(msg_tx)
            .insert_resource(msg_rx)
            .insert_resource(CurrentStreamId(stream_id))
            .insert_resource(status)
            .add_systems(PreUpdate, (sink, msg_sink));
    }
}

pub fn spawn_tcp_connect(
    addr: SocketAddr,
    mut outgoing_packet_rx: mpsc::Receiver<Option<LenPacket>>,
    mut incoming_packet_tx: AsyncArcQueueTx,
    stream_id: StreamId,
    mut reconnect: bool,
) -> ThreadConnectionStatus {
    let connection_status = ThreadConnectionStatus::new(ConnectionStatus::NoConnection);
    let ret_connection_status = connection_status.clone();
    std::thread::spawn(move || {
        let res: Result<(), miette::Error> = stellarator::run(|| async move {
            loop {
                connection_status.set_status(ConnectionStatus::Connecting);
                match tcp_connect(
                    addr,
                    &mut outgoing_packet_rx,
                    &mut incoming_packet_tx,
                    stream_id,
                    &new_connection_packets,
                    || {
                        reconnect = true;
                        connection_status.set_status(ConnectionStatus::Success);
                    },
                )
                .await
                {
                    Err(err) => {
                        bevy::log::trace!(?err, "connection ended with error");
                        connection_status.set_status(ConnectionStatus::Error);
                        if !reconnect {
                            return Ok(());
                        }
                        stellarator::sleep(Duration::from_millis(250)).await;
                    }
                    Ok(_) => return Ok(()),
                }
            }
        });
        if let Err(err) = res {
            bevy::log::error!(?err, "tcp plugin error");
        }
    });
    ret_connection_status
}

pub fn spawn_msg_tcp_connect(
    addr: SocketAddr,
    mut outgoing_packet_rx: mpsc::Receiver<Option<LenPacket>>,
    mut incoming_packet_tx: AsyncArcQueueTx,
) {
    std::thread::spawn(move || {
        let _: Result<(), miette::Error> = stellarator::run(|| async move {
            let stream_id = fastrand::u64(..);
            loop {
                match tcp_connect(
                    addr,
                    &mut outgoing_packet_rx,
                    &mut incoming_packet_tx,
                    stream_id,
                    &crate::msg_connection_packets,
                    || {},
                )
                .await
                {
                    Err(err) => {
                        bevy::log::trace!(?err, "msg connection ended");
                        stellarator::sleep(Duration::from_millis(250)).await;
                    }
                    Ok(_) => return Ok(()),
                }
            }
        });
    });
}
