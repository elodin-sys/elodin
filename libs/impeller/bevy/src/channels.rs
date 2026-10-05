use bevy::prelude::{Deref, Resource};
use impeller::types::{IntoLenPacket, LenPacket, Msg, OwnedPacket};
use impeller_bbq::{AsyncArcQueueRx, AsyncArcQueueTx, PacketGrantR, RxExt};
use std::{
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{self, AtomicU64},
    },
};

/// Size of the BBQ queue for incoming packets.
#[cfg(target_family = "wasm")]
pub const QUEUE_LEN: usize = 2 * 1024 * 1024;
/// Size of the BBQ queue for incoming packets.
/// Increased from 64MB to 256MB to handle large Arrow IPC responses from SQL queries.
#[cfg(not(target_family = "wasm"))]
pub const QUEUE_LEN: usize = 256 * 1024 * 1024;

#[derive(Resource)]
pub struct PacketRx(AsyncArcQueueRx);

impl From<AsyncArcQueueRx> for PacketRx {
    fn from(rx: AsyncArcQueueRx) -> Self {
        Self(rx)
    }
}

impl PacketRx {
    #[inline]
    pub fn try_recv_pkt(&mut self) -> Option<OwnedPacket<PacketGrantR>> {
        self.0.try_recv_pkt()
    }
}

#[derive(Resource)]
pub struct PacketTx(pub thingbuf::mpsc::Sender<Option<LenPacket>>);

impl PacketTx {
    pub fn send_msg(&self, msg: impl Msg) {
        let pkt = msg.into_len_packet();
        let _ = self.0.try_send(Some(pkt));
    }
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

#[derive(Resource)]
pub struct MsgPacketRx(AsyncArcQueueRx);

impl From<AsyncArcQueueRx> for MsgPacketRx {
    fn from(rx: AsyncArcQueueRx) -> Self {
        Self(rx)
    }
}

impl MsgPacketRx {
    #[inline]
    pub fn try_recv_pkt(&mut self) -> Option<OwnedPacket<PacketGrantR>> {
        self.0.try_recv_pkt()
    }
}

#[derive(Resource, Clone)]
pub struct MsgPacketTx(pub thingbuf::mpsc::Sender<Option<LenPacket>>);

impl MsgPacketTx {
    pub fn send_msg(&self, msg: impl Msg) {
        let pkt = msg.into_len_packet();
        let _ = self.0.try_send(Some(pkt));
    }
}

pub fn channels() -> (
    PacketTx,
    PacketRx,
    thingbuf::mpsc::Receiver<Option<LenPacket>>,
    AsyncArcQueueTx,
) {
    use bbqueue::ArcBBQueue;
    use bbqueue::traits::storage::BoxedSlice;
    let queue = ArcBBQueue::new_with_storage(BoxedSlice::new(QUEUE_LEN));
    let incoming_packet_rx = queue.framed_consumer_with_header::<usize>();
    let incoming_packet_tx = queue.framed_producer_with_header::<usize>();
    let (outgoing_packet_tx, outgoing_packet_rx) =
        thingbuf::mpsc::channel::<Option<LenPacket>>(4096);
    (
        PacketTx(outgoing_packet_tx),
        PacketRx(incoming_packet_rx),
        outgoing_packet_rx,
        incoming_packet_tx,
    )
}

pub fn msg_channels() -> (
    MsgPacketTx,
    MsgPacketRx,
    thingbuf::mpsc::Receiver<Option<LenPacket>>,
    AsyncArcQueueTx,
) {
    use bbqueue::ArcBBQueue;
    use bbqueue::traits::storage::BoxedSlice;
    let queue = ArcBBQueue::new_with_storage(BoxedSlice::new(QUEUE_LEN));
    let incoming_packet_rx = queue.framed_consumer_with_header::<usize>();
    let incoming_packet_tx = queue.framed_producer_with_header::<usize>();
    let (outgoing_packet_tx, outgoing_packet_rx) =
        thingbuf::mpsc::channel::<Option<LenPacket>>(4096);
    (
        MsgPacketTx(outgoing_packet_tx),
        MsgPacketRx(incoming_packet_rx),
        outgoing_packet_rx,
        incoming_packet_tx,
    )
}
