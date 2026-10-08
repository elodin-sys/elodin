use crate::*;
use bbqueue::ArcBBQueue;
use bbqueue::traits::storage::BoxedSlice;
use bevy::app::{Plugin, PreUpdate};
use impeller::types::{IntoLenPacket, LenPacket};
use impeller_bbq::*;
use impeller_stellar::queue::{StreamHandshake, tcp_connect};
use impeller_wkt::{DbConfig, SetStreamFilter, Stream, StreamBehavior, StreamId};
use std::sync::Arc;
use std::sync::atomic::{self, AtomicU64};
use std::{net::SocketAddr, time::Duration};
use thingbuf::mpsc;

pub struct TcpImpellerPlugin {
    addr: Option<SocketAddr>,
    component_filtered: bool,
}

impl TcpImpellerPlugin {
    pub fn new(addr: Option<SocketAddr>) -> Self {
        Self {
            addr,
            component_filtered: false,
        }
    }

    /// Open each connection's live stream with an empty component allowlist
    /// when the DB supports it. A client system must publish the component IDs
    /// it consumes once [`ComponentFilteredStream::supported`] is true.
    pub fn with_component_filtering(mut self) -> Self {
        self.component_filtered = true;
        self
    }
}

impl Plugin for TcpImpellerPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        let (packet_tx, packet_rx, outgoing_packet_rx, incoming_packet_tx) = channels();
        let (msg_tx, msg_rx, msg_outgoing_rx, msg_incoming_tx) = msg_channels();
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
                self.component_filtered,
            )
        } else {
            ThreadConnectionStatus::new(ConnectionStatus::NoConnection)
        };
        if self.component_filtered {
            app.init_resource::<ComponentFilteredStream>();
        }
        app.insert_resource(packet_tx)
            .insert_resource(packet_rx)
            .insert_resource(msg_tx)
            .insert_resource(msg_rx)
            .insert_resource(CurrentStreamId(stream_id))
            .insert_resource(status)
            .add_systems(PreUpdate, (sink, msg_sink));
    }
}

pub fn channels() -> (
    PacketTx,
    PacketRx,
    mpsc::Receiver<Option<LenPacket>>,
    AsyncArcQueueTx,
) {
    let queue = ArcBBQueue::new_with_storage(BoxedSlice::new(QUEUE_LEN));
    let incoming_packet_rx = queue.framed_consumer_with_header::<usize>();
    let incoming_packet_tx = queue.framed_producer_with_header::<usize>();
    let (outgoing_packet_tx, outgoing_packet_rx) = mpsc::channel::<Option<LenPacket>>(4096);
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
    mpsc::Receiver<Option<LenPacket>>,
    AsyncArcQueueTx,
) {
    let queue = ArcBBQueue::new_with_storage(BoxedSlice::new(QUEUE_LEN));
    let incoming_packet_rx = queue.framed_consumer_with_header::<usize>();
    let incoming_packet_tx = queue.framed_producer_with_header::<usize>();
    let (outgoing_packet_tx, outgoing_packet_rx) = mpsc::channel::<Option<LenPacket>>(4096);
    (
        MsgPacketTx(outgoing_packet_tx),
        MsgPacketRx(incoming_packet_rx),
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
    component_filtered: bool,
) -> ThreadConnectionStatus {
    let connection_status = ThreadConnectionStatus(Arc::new(AtomicU64::new(0)));
    let ret_connection_status = connection_status.clone();
    std::thread::spawn(move || {
        let res: Result<(), miette::Error> = stellarator::run(|| async move {
            let connection_packets = |stream_id| connection_packets(stream_id, component_filtered);
            let stream_handshake = component_filtered
                .then_some(component_filtered_stream_handshake as StreamHandshake);
            loop {
                connection_status.set_status(ConnectionStatus::Connecting);
                match tcp_connect(
                    addr,
                    &mut outgoing_packet_rx,
                    &mut incoming_packet_tx,
                    stream_id,
                    &connection_packets,
                    stream_handshake,
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

/// Component-filtered clients defer `Stream` to the per-connection handshake.
fn connection_packets(
    stream_id: StreamId,
    component_filtered: bool,
) -> impl Iterator<Item = LenPacket> {
    new_connection_packets(stream_id)
        .filter(move |packet| !component_filtered || packet.as_packet().header.id != Stream::ID)
}

/// Packets that open a component-filtered client's live stream. A supporting
/// DB gets an empty allowlist first so the stream starts with no components;
/// older DBs get a plain unfiltered stream, since they either ignore an empty
/// filter or persist the filter message as telemetry.
fn component_filtered_stream_handshake(stream_id: StreamId, config: &DbConfig) -> Vec<LenPacket> {
    let mut packets = Vec::with_capacity(2);
    if config.supports_exact_stream_filter() {
        packets.push(
            SetStreamFilter {
                id: stream_id,
                component_ids: Vec::new(),
                frequency: None,
            }
            .into_len_packet(),
        );
    }
    packets.push(
        Stream {
            behavior: StreamBehavior::RealTimeBatched,
            id: stream_id,
        }
        .into_len_packet(),
    );
    packets
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
                    None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use impeller::types::{Msg, PacketId};
    use impeller_wkt::GetDbSettings;

    fn packet_ids(packets: impl IntoIterator<Item = LenPacket>) -> Vec<PacketId> {
        packets
            .into_iter()
            .map(|packet| packet.as_packet().header.id)
            .collect()
    }

    #[test]
    fn component_filtered_connection_defers_stream_to_handshake() {
        let ids = packet_ids(connection_packets(42, true));
        assert!(ids.contains(&GetDbSettings::ID));
        assert!(!ids.contains(&Stream::ID));
        assert!(packet_ids(connection_packets(42, false)).contains(&Stream::ID));
    }

    #[test]
    fn supporting_db_gets_empty_filter_before_stream() {
        let mut config = DbConfig::default();
        config.advertise_exact_stream_filter();
        let packets = component_filtered_stream_handshake(42, &config);
        let filter: SetStreamFilter =
            postcard::from_bytes(&packets[0].as_packet().body).expect("valid filter packet");
        assert_eq!(filter.id, 42);
        assert!(filter.component_ids.is_empty());
        assert_eq!(packet_ids(packets), [SetStreamFilter::ID, Stream::ID]);
    }

    #[test]
    fn older_db_gets_only_an_unfiltered_stream() {
        let packets = component_filtered_stream_handshake(42, &DbConfig::default());
        assert_eq!(packet_ids(packets), [Stream::ID]);
    }
}
