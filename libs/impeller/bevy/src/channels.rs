use bevy::prelude::Resource;
use impeller::types::{IntoLenPacket, LenPacket, Msg, OwnedPacket};
use impeller_bbq::{AsyncArcQueueRx, AsyncArcQueueTx, PacketGrantR, RxExt};
use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
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

/// Database address typed by a client. IP literals stay `SocketAddr`s; DNS
/// names are kept so wasm can hand them to the browser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DbEndpoint {
    host: String,
    port: u16,
}

impl DbEndpoint {
    pub fn parse(input: &str) -> Result<Self, String> {
        let input = input.trim();
        if input.is_empty() {
            return Err("address is empty".into());
        }
        if let Ok(addr) = input.parse::<SocketAddr>() {
            return Ok(Self::from_socket(addr));
        }
        let Some((host, port_str)) = input.rsplit_once(':') else {
            return Err("expected host:port".into());
        };
        if !is_dns_host(host) {
            return Err(format!("invalid host {host}"));
        }
        let port: u16 = port_str
            .parse()
            .map_err(|_| format!("invalid port {port_str}"))?;
        Ok(Self {
            host: host.to_string(),
            port,
        })
    }

    pub fn from_socket(addr: SocketAddr) -> Self {
        let host = match addr.ip() {
            IpAddr::V4(ip) => ip.to_string(),
            IpAddr::V6(ip) => ip.to_string(),
        };
        Self {
            host,
            port: addr.port(),
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    /// DNS name when `host` is not an IP literal.
    pub fn dns_host(&self) -> Option<&str> {
        if self.socket_addr().is_some() {
            None
        } else {
            Some(self.host.as_str())
        }
    }

    pub fn socket_addr(&self) -> Option<SocketAddr> {
        let ip: IpAddr = self.host.parse().ok()?;
        Some(SocketAddr::new(ip, self.port))
    }

    /// Socket stored for session identity. DNS names use loopback at the same
    /// port; callers that build URLs must use [`Self::client_url_host`].
    pub fn identity_socket(&self) -> SocketAddr {
        self.socket_addr()
            .unwrap_or_else(|| SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), self.port))
    }

    /// Host portion of an http/ws URL. Unspecified bind addresses become loopback.
    pub fn client_url_host(&self) -> String {
        if let Some(addr) = self.socket_addr() {
            let ip = match addr.ip() {
                IpAddr::V4(v4) if v4.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
                IpAddr::V6(v6) if v6.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
                other => other,
            };
            return match ip {
                IpAddr::V4(v4) => v4.to_string(),
                IpAddr::V6(v6) => format!("[{v6}]"),
            };
        }
        self.host.clone()
    }

    pub fn authority(&self) -> String {
        format!("{}:{}", self.client_url_host(), self.port)
    }

    pub fn http_base(&self) -> String {
        let port = self.port.saturating_add(impeller::ASSETS_HTTP_PORT_OFFSET);
        format!("http://{}:{port}", self.client_url_host())
    }

    pub fn ws_urls(&self) -> (String, String) {
        let port = self.port.saturating_add(impeller::ASSETS_HTTP_PORT_OFFSET);
        let host = format!("{}:{port}", self.client_url_host());
        (
            format!("ws://{host}/impeller"),
            format!("ws://{host}/impeller/msg"),
        )
    }
}

fn is_dns_host(host: &str) -> bool {
    if host.is_empty() || host.len() > 253 || host.starts_with('.') || host.ends_with('.') {
        return false;
    }
    host.split('.').all(|label| {
        !label.is_empty()
            && label.len() <= 63
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    })
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
        assert_eq!(ip.authority(), "127.0.0.1:2240");
        assert!(ip.dns_host().is_none());

        let local = DbEndpoint::parse("localhost:2240").unwrap();
        assert_eq!(local.client_url_host(), "localhost");
        assert_eq!(local.http_base(), "http://localhost:2241");
        assert_eq!(local.dns_host(), Some("localhost"));

        let named = DbEndpoint::parse("db.local:2240").unwrap();
        assert_eq!(named.ws_urls().0, "ws://db.local:2241/impeller");

        assert!(DbEndpoint::parse("not a host:2240").is_err());
        assert!(DbEndpoint::parse("localhost").is_err());
    }

    #[test]
    fn unspecified_ip_becomes_loopback_in_urls() {
        let v4 = DbEndpoint::parse("0.0.0.0:2240").unwrap();
        assert_eq!(v4.http_base(), "http://127.0.0.1:2241");
        let v6 = DbEndpoint::parse("[::]:2240").unwrap();
        assert_eq!(v6.client_url_host(), "[::1]");
    }
}
