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

/// Incoming BBQ queue capacity.
///
/// Raised from 64 MiB to 256 MiB so a large Arrow IPC SQL reply fits alongside
/// other in-flight packets. WASM uses the same size: a 2 MiB queue makes
/// `push_incoming` drop those frames while the socket still looks connected.
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

/// Why [`HostOrUrl`] rejected an `&assets=` value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostOrUrlError {
    /// Blank, or only slashes.
    Empty,
    /// A URL scheme other than `http` or `https`, or a query or fragment.
    UnsupportedScheme,
    /// An `http`/`https` URL with no host.
    MissingHost,
    /// A full URL did not parse.
    Url(url::ParseError),
    /// `host:port` did not parse.
    Host(ParseError),
}

impl fmt::Display for HostOrUrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("empty assets URL"),
            Self::UnsupportedScheme => {
                f.write_str("assets URL scheme must be http or https, without a query or fragment")
            }
            Self::MissingHost => f.write_str("assets URL is missing a host"),
            Self::Url(err) => write!(f, "invalid assets URL: {err}"),
            Self::Host(err) => write!(f, "invalid assets address: {err}"),
        }
    }
}

impl std::error::Error for HostOrUrlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Url(err) => Some(err),
            Self::Host(err) => Some(err),
            _ => None,
        }
    }
}

/// `&assets=` value: a `host:port` or an `http`/`https` URL.
///
/// The host form keeps the authority the user typed. Loopback mapping happens
/// in [`Self::base_url`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HostOrUrl {
    Host(Authority),
    Url(url::Url),
}

impl HostOrUrl {
    /// HTTP base for asset fetches. The path ends with `/` so [`url::Url::join`] appends.
    pub fn base_url(&self) -> url::Url {
        match self {
            Self::Host(authority) => {
                authority_base_url(&client_host_authority(authority.host(), authority.port()))
            }
            Self::Url(url) => url.clone(),
        }
    }
}

impl FromStr for HostOrUrl {
    type Err = HostOrUrlError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let input = input.trim();
        if input.is_empty() {
            return Err(HostOrUrlError::Empty);
        }
        if input.contains("://") {
            return parse_assets_url(input);
        }
        let authority_text = input.trim_end_matches('/');
        if authority_text.is_empty() {
            return Err(HostOrUrlError::Empty);
        }
        authority_text
            .parse()
            .map(Self::Host)
            .map_err(HostOrUrlError::Host)
    }
}

impl fmt::Display for HostOrUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host(authority) => fmt::Display::fmt(authority, f),
            Self::Url(url) => fmt::Display::fmt(url, f),
        }
    }
}

fn parse_assets_url(input: &str) -> Result<HostOrUrl, HostOrUrlError> {
    let mut url = url::Url::parse(input).map_err(HostOrUrlError::Url)?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(HostOrUrlError::UnsupportedScheme);
    }
    if url.host().is_none() {
        return Err(HostOrUrlError::MissingHost);
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(HostOrUrlError::UnsupportedScheme);
    }
    if !url.path().ends_with('/') {
        url.path_segments_mut()
            .map_err(|_| HostOrUrlError::MissingHost)?
            .push("");
    }
    Ok(HostOrUrl::Url(url))
}

fn authority_base_url(authority: &Authority) -> url::Url {
    url::Url::parse(&format!("http://{authority}/")).expect("authority is a valid http URL")
}

/// Database address typed by a client: an IP literal or DNS name plus port.
/// DNS names are kept so wasm can hand them to the browser.
///
/// `assets`, when set, is the HTTP base for `db:` assets. WebSocket URLs still
/// come from this address.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DbEndpoint {
    authority: Authority,
    assets: Option<HostOrUrl>,
}

impl DbEndpoint {
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        input.trim().parse::<Authority>().map(Self::from)
    }

    pub fn from_socket(addr: SocketAddr) -> Self {
        let authority: Authority = addr.into();
        Self::from(authority)
    }

    pub fn with_assets(mut self, assets: HostOrUrl) -> Self {
        self.assets = Some(assets);
        self
    }

    pub fn assets(&self) -> Option<&HostOrUrl> {
        self.assets.as_ref()
    }

    pub fn port(&self) -> u16 {
        self.authority.port()
    }

    pub fn host(&self) -> HostRef<'_> {
        self.authority.host()
    }

    /// DNS name when `host` is not an IP literal.
    pub fn dns_host(&self) -> Option<&str> {
        match self.authority.host() {
            HostRef::Domain(domain) => Some(domain.name()),
            HostRef::IPAddress(_) => None,
        }
    }

    pub fn socket_addr(&self) -> Option<SocketAddr> {
        match self.authority.host() {
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
        client_host_authority(self.authority.host(), port)
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

    pub fn http_base(&self) -> url::Url {
        if let Some(assets) = &self.assets {
            return assets.base_url();
        }
        authority_base_url(&self.assets_authority())
    }

    /// URL of one `db:` asset key under [`Self::http_base`].
    pub fn assets_url(&self, key: &str) -> url::Url {
        let key = key.trim_start_matches('/');
        let base = self.http_base();
        match base.join(key) {
            Ok(url) => url,
            Err(_) => {
                let mut fallback = base.to_string();
                if !fallback.ends_with('/') {
                    fallback.push('/');
                }
                fallback.push_str(key);
                url::Url::parse(&fallback).unwrap_or(base)
            }
        }
    }

    pub fn ws_urls(&self) -> (String, String) {
        let host = self.assets_authority();
        (
            format!("ws://{host}/impeller"),
            format!("ws://{host}/impeller/msg"),
        )
    }
}

/// Authority for client URLs. Unspecified bind addresses become loopback.
fn client_host_authority(host: HostRef<'_>, port: u16) -> Authority {
    match host {
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
        Self {
            authority,
            assets: None,
        }
    }
}

impl fmt::Display for DbEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.authority, f)
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
    use super::{DbEndpoint, QUEUE_LEN};

    #[test]
    fn parse_accepts_ip_and_dns_names() {
        let ip = DbEndpoint::parse("127.0.0.1:2240").unwrap();
        assert_eq!(ip.authority().to_string(), "127.0.0.1:2240");
        assert!(ip.dns_host().is_none());
        assert_eq!(ip.socket_addr(), Some("127.0.0.1:2240".parse().unwrap()));

        let local = DbEndpoint::parse(" localhost:2240 ").unwrap();
        assert_eq!(local.to_string(), "localhost:2240");
        assert_eq!(local.http_base().as_str(), "http://localhost:2241/");
        assert_eq!(local.dns_host(), Some("localhost"));
        assert!(local.socket_addr().is_none());

        let named = DbEndpoint::parse("DB.Local:2240").unwrap();
        assert_eq!(named.ws_urls().0, "ws://db.local:2241/impeller");

        assert!(DbEndpoint::parse("not a host:2240").is_err());
        assert!(DbEndpoint::parse("localhost").is_err());
        assert!(DbEndpoint::parse("").is_err());
    }

    #[test]
    fn incoming_queue_holds_large_arrow_ipc_frames() {
        // Websocket proxy body cap (`MAX_IMPELLER_PACKET_BYTES`) plus the u32
        // length prefix, the 16-byte alignment reserve in `push_incoming`, and
        // the framed `usize` header. The queue is larger than one max frame so
        // a SQL reply still fits when other packets are in flight.
        const MAX_WS_BODY: usize = 64 * 1024 * 1024;
        let one_frame = 4 + MAX_WS_BODY + 16 + size_of::<usize>();
        assert!(
            QUEUE_LEN >= 256 * 1024 * 1024,
            "incoming queue {QUEUE_LEN} is below the 256 MiB Arrow IPC sizing"
        );
        assert!(
            QUEUE_LEN > one_frame,
            "incoming queue {QUEUE_LEN} cannot hold a {one_frame}-byte Arrow IPC frame"
        );
    }

    #[test]
    fn unspecified_ip_becomes_loopback_in_urls() {
        let v4 = DbEndpoint::parse("0.0.0.0:2240").unwrap();
        assert_eq!(v4.http_base().as_str(), "http://127.0.0.1:2241/");
        assert_eq!(v4.to_string(), "0.0.0.0:2240");
        let v6 = DbEndpoint::parse("[::]:2240").unwrap();
        assert_eq!(v6.authority().to_string(), "[::1]:2240");
        assert_eq!(v6.ws_urls().1, "ws://[::1]:2241/impeller/msg");
    }

    #[test]
    fn host_or_url_accepts_authority_and_urls() {
        use super::HostOrUrl;

        let host: HostOrUrl = "127.0.0.1:2241".parse().unwrap();
        assert_eq!(host.base_url().as_str(), "http://127.0.0.1:2241/");
        assert_eq!(host.to_string(), "127.0.0.1:2241");
        assert_eq!(host.to_string().parse::<HostOrUrl>().unwrap(), host);

        let unspecified: HostOrUrl = " 0.0.0.0:2241 ".parse().unwrap();
        assert_eq!(unspecified.to_string(), "0.0.0.0:2241");
        assert_eq!(unspecified.base_url().as_str(), "http://127.0.0.1:2241/");
        assert_eq!(
            unspecified.to_string().parse::<HostOrUrl>().unwrap(),
            unspecified
        );

        let v6: HostOrUrl = "[::]:2241".parse().unwrap();
        assert_eq!(v6.to_string(), "[::]:2241");
        assert_eq!(v6.base_url().as_str(), "http://[::1]:2241/");

        let named: HostOrUrl = "localhost:2241/".parse().unwrap();
        assert_eq!(named.to_string(), "localhost:2241");
        assert_eq!(named.base_url().as_str(), "http://localhost:2241/");

        let cdn: HostOrUrl = "https://cdn.example/assets".parse().unwrap();
        assert_eq!(cdn.to_string(), "https://cdn.example/assets/");
        assert_eq!(cdn.to_string().parse::<HostOrUrl>().unwrap(), cdn);
        assert_eq!(
            cdn.base_url().join("rocket.glb").unwrap().as_str(),
            "https://cdn.example/assets/rocket.glb"
        );

        let upper: HostOrUrl = "HTTP://127.0.0.1:2241/".parse().unwrap();
        assert_eq!(upper.to_string(), "http://127.0.0.1:2241/");

        assert!("".parse::<HostOrUrl>().is_err());
        assert!("http://".parse::<HostOrUrl>().is_err());
        assert!("https:///".parse::<HostOrUrl>().is_err());
        assert!("ftp://files/a".parse::<HostOrUrl>().is_err());
        assert!("not a host".parse::<HostOrUrl>().is_err());
        assert!(
            "https://cdn.example/assets?x=1"
                .parse::<HostOrUrl>()
                .is_err()
        );
    }

    #[test]
    fn assets_override_replaces_http_base_only() {
        use super::HostOrUrl;

        let assets: HostOrUrl = "10.0.0.5:9000".parse().unwrap();
        let endpoint = DbEndpoint::parse("127.0.0.1:2240")
            .unwrap()
            .with_assets(assets.clone());
        assert_eq!(endpoint.assets(), Some(&assets));
        assert_eq!(endpoint.http_base().as_str(), "http://10.0.0.5:9000/");
        assert_eq!(
            endpoint.assets_url("rocket.glb").as_str(),
            "http://10.0.0.5:9000/rocket.glb"
        );
        assert_eq!(
            endpoint.assets_url("icons/marker.png").as_str(),
            "http://10.0.0.5:9000/icons/marker.png"
        );
        assert_eq!(endpoint.ws_urls().0, "ws://127.0.0.1:2241/impeller");
        assert_eq!(endpoint.ws_urls().1, "ws://127.0.0.1:2241/impeller/msg");
        assert_eq!(endpoint.to_string(), "127.0.0.1:2240");
    }
}
