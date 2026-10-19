use address::{Authority, HostRef, IPAddress, ParseError};
use bevy::prelude::Resource;
use impeller::types::{IntoLenPacket, LenPacket, Msg, OwnedPacket};
use impeller_bbq::{AsyncArcQueueRx, AsyncArcQueueTx, PacketGrantR, RxExt};
use std::{
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    str::FromStr,
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

    /// Queue a packet whose loss would leave the client silently
    /// out of sync with the DB; panics naming `operation` if it cannot be queued.
    pub fn send_required(&self, pkt: LenPacket, operation: &str) {
        if let Err(err) = self.0.try_send(Some(pkt)) {
            let reason = if err.is_full() { "full" } else { "closed" };
            panic!("{operation} failed: outgoing packet queue is {reason}");
        }
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

/// Database address typed by a client: an IP literal or DNS name plus port.
/// DNS names are kept so wasm can hand them to the browser.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DbEndpoint(Authority);

impl DbEndpoint {
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        input.trim().parse::<Authority>().map(Self)
    }

    pub fn from_socket(addr: SocketAddr) -> Self {
        Self(addr.into())
    }

    pub fn port(&self) -> u16 {
        self.0.port()
    }

    pub fn host(&self) -> HostRef<'_> {
        self.0.host()
    }

    /// DNS name when `host` is not an IP literal.
    pub fn dns_host(&self) -> Option<&str> {
        match self.0.host() {
            HostRef::Domain(domain) => Some(domain.name()),
            HostRef::IPAddress(_) => None,
        }
    }

    pub fn socket_addr(&self) -> Option<SocketAddr> {
        match self.0.host() {
            HostRef::IPAddress(ip) => Some(SocketAddr::new(ip.to_std(), self.port())),
            HostRef::Domain(_) => None,
        }
    }

    /// Socket stored for session identity. DNS names use loopback at the same
    /// port; callers that build URLs must use [`Self::authority`].
    pub fn identity_socket(&self) -> SocketAddr {
        self.socket_addr()
            .unwrap_or_else(|| SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), self.port()))
    }

    /// Authority for http/ws URLs at `port`. Unspecified bind addresses become loopback.
    fn client_authority(&self, port: u16) -> Authority {
        match self.0.host() {
            HostRef::IPAddress(ip) => {
                let ip = match ip.to_std() {
                    IpAddr::V4(v4) if v4.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
                    IpAddr::V6(v6) if v6.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
                    other => other,
                };
                Authority::new(IPAddress::from(ip).to_host(), port)
            }
            host @ HostRef::Domain(_) => Authority::new(host.into(), port),
        }
    }

    pub fn authority(&self) -> Authority {
        self.client_authority(self.port())
    }

    fn assets_authority(&self) -> Authority {
        self.client_authority(
            self.port()
                .saturating_add(impeller::ASSETS_HTTP_PORT_OFFSET),
        )
    }

    pub fn http_base(&self) -> String {
        format!("http://{}", self.assets_authority())
    }

    pub fn ws_urls(&self) -> (String, String) {
        let host = self.assets_authority();
        (
            format!("ws://{host}/impeller"),
            format!("ws://{host}/impeller/msg"),
        )
    }
}

impl FromStr for DbEndpoint {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl From<SocketAddr> for DbEndpoint {
    fn from(addr: SocketAddr) -> Self {
        Self::from_socket(addr)
    }
}

impl From<Authority> for DbEndpoint {
    fn from(authority: Authority) -> Self {
        Self(authority)
    }
}

impl fmt::Display for DbEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

#[derive(Clone, Debug, Resource)]
pub struct ConnectionAddr {
    pub socket: SocketAddr,
    endpoint: DbEndpoint,
}

impl ConnectionAddr {
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            endpoint: DbEndpoint::from_socket(addr),
            socket: addr,
        }
    }

    pub fn from_endpoint(endpoint: DbEndpoint) -> Self {
        let socket = endpoint.identity_socket();
        Self { socket, endpoint }
    }

    pub fn endpoint(&self) -> &DbEndpoint {
        &self.endpoint
    }
}

impl std::ops::Deref for ConnectionAddr {
    type Target = SocketAddr;

    fn deref(&self) -> &Self::Target {
        &self.socket
    }
}

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

#[cfg(test)]
mod tests {
    use super::DbEndpoint;

    #[test]
    fn parse_accepts_ip_and_dns_names() {
        let ip = DbEndpoint::parse("127.0.0.1:2240").unwrap();
        assert_eq!(ip.authority().to_string(), "127.0.0.1:2240");
        assert!(ip.dns_host().is_none());
        assert_eq!(ip.socket_addr(), Some("127.0.0.1:2240".parse().unwrap()));

        let local = DbEndpoint::parse(" localhost:2240 ").unwrap();
        assert_eq!(local.to_string(), "localhost:2240");
        assert_eq!(local.http_base(), "http://localhost:2241");
        assert_eq!(local.dns_host(), Some("localhost"));
        assert!(local.socket_addr().is_none());

        let named = DbEndpoint::parse("DB.Local:2240").unwrap();
        assert_eq!(named.ws_urls().0, "ws://db.local:2241/impeller");

        assert!(DbEndpoint::parse("not a host:2240").is_err());
        assert!(DbEndpoint::parse("localhost").is_err());
        assert!(DbEndpoint::parse("").is_err());
    }

    #[test]
    fn unspecified_ip_becomes_loopback_in_urls() {
        let v4 = DbEndpoint::parse("0.0.0.0:2240").unwrap();
        assert_eq!(v4.http_base(), "http://127.0.0.1:2241");
        assert_eq!(v4.to_string(), "0.0.0.0:2240");
        let v6 = DbEndpoint::parse("[::]:2240").unwrap();
        assert_eq!(v6.authority().to_string(), "[::1]:2240");
        assert_eq!(v6.ws_urls().1, "ws://[::1]:2241/impeller/msg");
    }
}
