use crate::*;
use bevy::app::{Plugin, Update};
use impeller2::types::FilledRecycle;
use impeller2::types::LenPacket;
use impeller2_stella::thingbuf::tcp_connect;
use std::sync::atomic::{self, AtomicU64};
use std::sync::Arc;
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
        let (packet_tx, packet_rx, outgoing_packet_rx, incoming_packet_tx) = channels();
        let stream_id = fastrand::u64(..);
        let status = if let Some(addr) = self.addr {
            spawn_tcp_connect(addr, outgoing_packet_rx, incoming_packet_tx, stream_id)
        } else {
            ThreadConnectionStatus::new(ConnectionStatus::NoConnection)
        };
        app.insert_resource(packet_tx)
            .insert_resource(packet_rx)
            .insert_resource(CurrentStreamId(stream_id))
            .insert_resource(status)
            .add_systems(Update, sink);
    }
}

pub fn channels() -> (
    PacketTx,
    PacketRx,
    mpsc::Receiver<Option<LenPacket>>,
    mpsc::Sender<MaybeFilledPacket, FilledRecycle>,
) {
    let (incoming_packet_tx, incoming_packet_rx) = mpsc::with_recycle(4096, FilledRecycle);
    let (outgoing_packet_tx, outgoing_packet_rx) = mpsc::channel::<Option<LenPacket>>(4096);
    (
        PacketTx(outgoing_packet_tx),
        PacketRx(incoming_packet_rx),
        outgoing_packet_rx,
        incoming_packet_tx,
    )
}

pub fn spawn_tcp_connect(
    addr: SocketAddr,
    mut outgoing_packet_rx: mpsc::Receiver<Option<LenPacket>>,
    mut incoming_packet_tx: mpsc::Sender<MaybeFilledPacket, FilledRecycle>,
    stream_id: StreamId,
) -> ThreadConnectionStatus {
    let connection_status = ThreadConnectionStatus(Arc::new(AtomicU64::new(0)));
    let ret_connection_status = connection_status.clone();
    std::thread::spawn(move || {
        let res: Result<(), miette::Error> = stellarator::run(|| async move {
            loop {
                connection_status.set_status(ConnectionStatus::Connecting);
                if let Err(err) = tcp_connect(
                    addr,
                    &mut outgoing_packet_rx,
                    &mut incoming_packet_tx,
                    stream_id,
                    &new_connection_packets,
                    || {
                        connection_status.set_status(ConnectionStatus::Success);
                    },
                )
                .await
                {
                    bevy::log::trace!(?err, "connection ended with error");
                    connection_status.set_status(ConnectionStatus::Error);
                    stellarator::sleep(Duration::from_millis(250)).await;
                }
            }
        });
        if let Err(err) = res {
            bevy::log::error!(?err, "tcp plugin error");
        }
    });
    ret_connection_status
}

#[repr(u64)]
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionStatus {
    #[default]
    NoConnection = 0,
    Success,
    Connecting,
    Error,
}

#[derive(Clone, Resource)]
pub struct ThreadConnectionStatus(Arc<AtomicU64>);

impl ThreadConnectionStatus {
    pub fn new(status: ConnectionStatus) -> Self {
        ThreadConnectionStatus(Arc::new(AtomicU64::new(status as u64)))
    }

    pub fn status(&self) -> ConnectionStatus {
        match self.0.load(atomic::Ordering::SeqCst) {
            0 => ConnectionStatus::NoConnection,
            1 => ConnectionStatus::Success,
            2 => ConnectionStatus::Connecting,
            3 => ConnectionStatus::Error,
            _ => ConnectionStatus::NoConnection,
        }
    }
    pub fn set_status(&self, status: ConnectionStatus) {
        self.0.store(status as u64, atomic::Ordering::SeqCst);
    }
}
