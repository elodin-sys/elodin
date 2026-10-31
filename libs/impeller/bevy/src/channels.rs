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

/// Complete browser routes. The browser supplies cookies for these URLs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BrowserEndpoints {
    impeller: url::Url,
    messages: url::Url,
    assets: url::Url,
}

impl BrowserEndpoints {
    pub fn new(impeller: &str, messages: &str, assets: &str) -> Result<Self, String> {
        fn parse(value: &str, schemes: &[&str]) -> Result<url::Url, String> {
            let url = url::Url::parse(value).map_err(|_| "Invalid browser endpoint URL")?;
            if !schemes.contains(&url.scheme())
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.fragment().is_some()
            {
                return Err("Browser endpoints require absolute HTTP(S) or WS(S) URLs without credentials or fragments".into());
            }
            Ok(url)
        }
        let mut assets = parse(assets, &["http", "https"])?;
        if !assets.path().ends_with('/') {
            assets.set_path(&format!("{}/", assets.path()));
        }
        Ok(Self {
            impeller: parse(impeller, &["ws", "wss"])?,
            messages: parse(messages, &["ws", "wss"])?,
            assets,
        })
    }

    pub fn impeller(&self) -> &str {
        self.impeller.as_str()
    }
    pub fn messages(&self) -> &str {
        self.messages.as_str()
    }
    pub fn assets(&self) -> &str {
        self.assets.as_str()
    }
}

/// A desktop database address or a complete set of browser routes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DbEndpoint {
    Tcp(Authority),
    Browser(Box<BrowserEndpoints>),
}

impl DbEndpoint {
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        input.trim().parse::<Authority>().map(Self::Tcp)
    }

    pub fn from_socket(addr: SocketAddr) -> Self {
        Self::Tcp(addr.into())
    }

    pub fn dns_host(&self) -> Option<&str> {
        match self {
            Self::Tcp(authority) => match authority.host() {
                HostRef::Domain(domain) => Some(domain.name()),
                _ => None,
            },
            Self::Browser(_) => None,
        }
    }

    pub fn socket_addr(&self) -> Option<SocketAddr> {
        match self {
            Self::Tcp(authority) => match authority.host() {
                HostRef::IPAddress(ip) => Some(SocketAddr::new(ip.to_std(), authority.port())),
                _ => None,
            },
            Self::Browser(_) => None,
        }
    }

    fn client_authority(authority: &Authority, port: u16) -> Authority {
        match authority.host() {
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

    pub fn authority(&self) -> Option<Authority> {
        match self {
            Self::Tcp(a) => Some(Self::client_authority(a, a.port())),
            Self::Browser(_) => None,
        }
    }

    pub fn http_base(&self) -> String {
        match self {
            Self::Tcp(a) => format!(
                "http://{}",
                Self::client_authority(
                    a,
                    a.port().saturating_add(impeller::ASSETS_HTTP_PORT_OFFSET)
                )
            ),
            Self::Browser(b) => b.assets().to_owned(),
        }
    }

    /// Resolves a DB key without treating its query or fragment characters as URL syntax.
    /// Existing percent escapes remain encoded. Routing parameters stay on the base URL.
    pub fn asset_url(&self, key: &str) -> Result<String, String> {
        let key = key.trim_start_matches('/');
        if key.is_empty()
            || key.contains('\\')
            || key.split('/').any(|part| {
                let dots = part.to_ascii_lowercase().replace("%2e", ".");
                dots == "."
                    || dots == ".."
                    || part.to_ascii_lowercase().contains("%2f")
                    || part.to_ascii_lowercase().contains("%5c")
            })
        {
            return Err("Invalid database asset key".into());
        }
        let mut url = url::Url::parse(&self.http_base()).map_err(|e| e.to_string())?;
        let path = format!("{}/{}", url.path().trim_end_matches('/'), key);
        url.set_path(&path);
        Ok(url.into())
    }

    pub fn ws_urls(&self) -> (String, String) {
        match self {
            Self::Tcp(a) => {
                let host = Self::client_authority(
                    a,
                    a.port().saturating_add(impeller::ASSETS_HTTP_PORT_OFFSET),
                );
                (
                    format!("ws://{host}/impeller"),
                    format!("ws://{host}/impeller/msg"),
                )
            }
            Self::Browser(b) => (b.impeller().into(), b.messages().into()),
        }
    }

    /// Restores routing parameters on relative dependencies loaded by Bevy,
    /// such as an external glTF buffer or texture. Other origins stay untouched.
    pub fn asset_request_url(&self, request: &str) -> Result<String, String> {
        let Self::Browser(browser) = self else {
            return Ok(request.into());
        };
        let mut url = url::Url::parse(request).map_err(|e| e.to_string())?;
        if url.origin() == browser.assets.origin() && url.path().starts_with(browser.assets.path())
        {
            let routing: Vec<_> = browser.assets.query_pairs().collect();
            let mut query: Vec<_> = url
                .query()
                .unwrap_or_default()
                .split('&')
                .filter(|pair| !pair.is_empty())
                .filter(|pair| {
                    url::form_urlencoded::parse(pair.as_bytes())
                        .next()
                        .is_none_or(|(key, _)| !routing.iter().any(|(k, _)| *k == key))
                })
                .map(str::to_owned)
                .collect();
            if let Some(routing) = browser.assets.query() {
                query.push(routing.to_owned());
            }
            let query = query.join("&");
            url.set_query((!query.is_empty()).then_some(query.as_str()));
        }
        Ok(url.into())
    }

    /// Resolves a glTF dependency and keeps the session query in its asset identity.
    pub fn asset_dependency_url(&self, parent: &str, relative: &str) -> Result<String, String> {
        let url = url::Url::parse(parent)
            .and_then(|base| base.join(relative))
            .map_err(|e| e.to_string())?;
        self.asset_request_url(url.as_str())
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
        Self::Tcp(authority)
    }
}
impl From<BrowserEndpoints> for DbEndpoint {
    fn from(endpoints: BrowserEndpoints) -> Self {
        Self::Browser(Box::new(endpoints))
    }
}
impl fmt::Display for DbEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tcp(a) => fmt::Display::fmt(a, f),
            Self::Browser(b) => f.write_str(b.impeller()),
        }
    }
}

#[derive(Clone, Debug, Resource)]
pub struct ConnectionAddr {
    endpoint: DbEndpoint,
}
impl ConnectionAddr {
    pub fn new(addr: SocketAddr) -> Self {
        Self::from_endpoint(addr.into())
    }
    pub fn from_endpoint(endpoint: DbEndpoint) -> Self {
        Self { endpoint }
    }
    pub fn endpoint(&self) -> &DbEndpoint {
        &self.endpoint
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
    use super::{BrowserEndpoints, DbEndpoint, QUEUE_LEN};

    fn browser(session: &str) -> DbEndpoint {
        BrowserEndpoints::new(
            &format!("wss://example.com/sessions/{session}/impeller?route=main"),
            &format!("wss://example.com/sessions/{session}/messages?route=messages"),
            &format!("https://cdn.example.com/sessions/{session}/assets?route=a%2Fb"),
        )
        .unwrap()
        .into()
    }

    #[test]
    fn browser_routes_preserve_paths_queries_and_encoded_asset_names() {
        let endpoint = browser("one");
        assert_eq!(
            endpoint.ws_urls().1,
            "wss://example.com/sessions/one/messages?route=messages"
        );
        assert_eq!(
            endpoint.asset_url("models/rocket%20one.glb").unwrap(),
            "https://cdn.example.com/sessions/one/assets/models/rocket%20one.glb?route=a%2Fb"
        );
        assert_eq!(
            endpoint.asset_url("icons/a #?.png").unwrap(),
            "https://cdn.example.com/sessions/one/assets/icons/a%20%23%3F.png?route=a%2Fb"
        );
        assert!(endpoint.socket_addr().is_none());
        assert_ne!(endpoint, browser("two"));
    }

    #[test]
    fn browser_asset_dependencies_inherit_only_their_own_session_routing() {
        let endpoint = browser("one");
        assert_eq!(
            endpoint
                .asset_request_url("https://cdn.example.com/sessions/one/assets/models/buffer.bin")
                .unwrap(),
            "https://cdn.example.com/sessions/one/assets/models/buffer.bin?route=a%2Fb"
        );
        for url in [
            "https://other.example.com/sessions/one/assets/a.png",
            "https://cdn.example.com/sessions/two/assets/a.png",
        ] {
            assert_eq!(endpoint.asset_request_url(url).unwrap(), url);
        }
    }

    #[test]
    fn routing_parameters_keep_their_original_encoding() {
        let endpoint: DbEndpoint = BrowserEndpoints::new(
            "wss://example.com/main",
            "wss://example.com/msg",
            "https://example.com/assets?route=a%2fb&name=a%20b",
        )
        .unwrap()
        .into();
        assert_eq!(
            endpoint
                .asset_request_url("https://example.com/assets/a.bin?other=%2f")
                .unwrap(),
            "https://example.com/assets/a.bin?other=%2f&route=a%2fb&name=a%20b"
        );
    }

    #[test]
    fn browser_endpoints_reject_invalid_schemes_credentials_and_fragments() {
        for bad in [
            "",
            "example.com/main",
            "https://example.com/main",
            "wss://user:secret@example.com/main",
            "wss://example.com/main#fragment",
        ] {
            assert!(
                BrowserEndpoints::new(
                    bad,
                    "wss://example.com/messages",
                    "https://example.com/assets"
                )
                .is_err()
            );
        }
        assert!(
            BrowserEndpoints::new(
                "wss://example.com/main",
                "wss://example.com/messages",
                "file:///assets"
            )
            .is_err()
        );
        for key in [
            "../escape.glb",
            "%2e%2e/escape.glb",
            "a/%2Fescape",
            "a/./b",
            "a\\b",
            "",
        ] {
            assert!(browser("one").asset_url(key).is_err(), "accepted {key}");
        }
    }

    #[test]
    fn parse_accepts_ip_and_dns_names() {
        let ip = DbEndpoint::parse("127.0.0.1:2240").unwrap();
        assert_eq!(ip.authority().unwrap().to_string(), "127.0.0.1:2240");
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
    fn incoming_queue_holds_large_arrow_ipc_frames() {
        // Websocket proxy body cap (`MAX_IMPELLER_PACKET_BYTES`) plus the u32
        // length prefix, the 16-byte alignment reserve in `push_incoming`, and
        // the framed `usize` header. The queue is larger than one max frame so
        // a SQL reply still fits when other packets are in flight.
        const MAX_WS_BODY: usize = 64 * 1024 * 1024;
        let one_frame = 4 + MAX_WS_BODY + 16 + size_of::<usize>();
        const {
            assert!(
                QUEUE_LEN >= 256 * 1024 * 1024,
                "incoming queue is below the 256 MiB Arrow IPC sizing"
            );
        }
        assert!(
            QUEUE_LEN > one_frame,
            "incoming queue {QUEUE_LEN} cannot hold a {one_frame}-byte Arrow IPC frame"
        );
    }

    #[test]
    fn unspecified_ip_becomes_loopback_in_urls() {
        let v4 = DbEndpoint::parse("0.0.0.0:2240").unwrap();
        assert_eq!(v4.http_base(), "http://127.0.0.1:2241");
        assert_eq!(v4.to_string(), "0.0.0.0:2240");
        let v6 = DbEndpoint::parse("[::]:2240").unwrap();
        assert_eq!(v6.authority().unwrap().to_string(), "[::1]:2240");
        assert_eq!(v6.ws_urls().1, "ws://[::1]:2241/impeller/msg");
    }
}
