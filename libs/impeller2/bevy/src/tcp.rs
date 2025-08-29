use crate::*;
use bbq2::queue::ArcBBQueue;
use bbq2::traits::storage::BoxedSlice;
use bevy::app::{Plugin, PreUpdate};
use impeller2::types::LenPacket;
use impeller2_bbq::*;
use impeller2_stellar::queue::tcp_connect;
use std::sync::Arc;
use std::sync::atomic::{self, AtomicU64};
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
            app.insert_resource(ConnectionAddr(addr));
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
            .insert_resource(CurrentStreamId(stream_id))
            .insert_resource(status)
            .add_systems(PreUpdate, sink);
    }
}

pub fn channels() -> (
    PacketTx,
    PacketRx,
    mpsc::Receiver<Option<LenPacket>>,
    AsyncArcQueueTx,
) {
    let queue = ArcBBQueue::new_with_storage(BoxedSlice::new(QUEUE_LEN));
    let (incoming_packet_rx, incoming_packet_tx) = queue.framed_split();
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
    mut incoming_packet_tx: AsyncArcQueueTx,
    stream_id: StreamId,
    mut reconnect: bool,
) -> ThreadConnectionStatus {
    let connection_status = ThreadConnectionStatus(Arc::new(AtomicU64::new(0)));
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

#[derive(Clone, Resource, Deref)]
pub struct ConnectionAddr(pub SocketAddr);
