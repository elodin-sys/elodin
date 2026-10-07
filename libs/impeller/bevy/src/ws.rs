use crate::{
    CurrentStreamId,
    channels::{ConnectionAddr, ConnectionStatus, ThreadConnectionStatus, channels, msg_channels},
    msg_sink, new_connection_packets, sink,
};
use bevy::prelude::*;
use impeller::types::{LenPacket, Msg, PacketTy};
use impeller_bbq::{AsyncArcQueueTx, PacketGrantW};
use impeller_wkt::{NewConnection, StreamId};
use js_sys::Uint8Array;
use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use thingbuf::mpsc;
use wasm_bindgen::{JsCast, JsValue, prelude::Closure};
use web_sys::{BinaryType, MessageEvent, WebSocket};
use web_time::{Duration, Instant};

const RECONNECT_DELAY: Duration = Duration::from_millis(250);

/// Rebuild the two Impeller WebSockets against `addr`.
#[derive(Message, Clone, Copy, Debug)]
pub struct WsConnect(pub SocketAddr);

pub struct WsImpellerPlugin {
    addr: Option<SocketAddr>,
}

impl WsImpellerPlugin {
    pub fn new(addr: Option<SocketAddr>) -> Self {
        Self { addr }
    }
}

impl Plugin for WsImpellerPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<WsConnect>()
            .init_resource::<PendingWsConnect>()
            .add_systems(PreUpdate, queue_ws_connect)
            .add_systems(PreUpdate, apply_ws_connect.after(queue_ws_connect))
            .add_systems(
                PreUpdate,
                (ws_reconnect, ws_flush, sink, msg_sink)
                    .chain()
                    .after(apply_ws_connect),
            );
        install_connection(app.world_mut(), self.addr, true);
    }
}

#[derive(Resource, Default)]
struct PendingWsConnect(Option<SocketAddr>);

fn install_connection(world: &mut World, addr: Option<SocketAddr>, initial: bool) {
    let (packet_tx, packet_rx, outgoing_packet_rx, incoming_packet_tx) = channels();
    let (msg_tx, msg_rx, msg_outgoing_rx, msg_incoming_tx) = msg_channels();
    let stream_id = fastrand::u64(..);
    let status = ThreadConnectionStatus::new(if addr.is_some() {
        ConnectionStatus::Connecting
    } else {
        ConnectionStatus::NoConnection
    });
    if let Some(addr) = addr {
        world.insert_resource(ConnectionAddr(addr));
    } else if initial {
        world.remove_resource::<ConnectionAddr>();
    }
    world.insert_resource(packet_tx);
    world.insert_resource(packet_rx);
    world.insert_resource(msg_tx);
    world.insert_resource(msg_rx);
    world.insert_resource(CurrentStreamId(stream_id));
    world.insert_resource(status.clone());
    world.insert_resource(PendingWsConnect(None));
    world.insert_resource(WsIo {
        addr,
        stream_id,
        generation: Arc::new(AtomicU64::new(0)),
        main_outgoing: outgoing_packet_rx,
        msg_outgoing: msg_outgoing_rx,
        incoming_main: Arc::new(incoming_packet_tx),
        incoming_msg: Arc::new(msg_incoming_tx),
        retry_at: None,
        status,
    });
    world.remove_non_send::<WsSockets>();
    if addr.is_some() {
        open_sockets(world);
    }
}

#[derive(Resource)]
struct WsIo {
    addr: Option<SocketAddr>,
    stream_id: StreamId,
    generation: Arc<AtomicU64>,
    main_outgoing: mpsc::Receiver<Option<LenPacket>>,
    msg_outgoing: mpsc::Receiver<Option<LenPacket>>,
    incoming_main: Arc<AsyncArcQueueTx>,
    incoming_msg: Arc<AsyncArcQueueTx>,
    retry_at: Option<Instant>,
    status: ThreadConnectionStatus,
}

struct WsSockets {
    main: Option<WebSocket>,
    msg: Option<WebSocket>,
}

fn queue_ws_connect(mut events: MessageReader<WsConnect>, mut pending: ResMut<PendingWsConnect>) {
    if let Some(event) = events.read().last() {
        pending.0 = Some(event.0);
    }
}

fn apply_ws_connect(world: &mut World) {
    let Some(addr) = world
        .get_resource_mut::<PendingWsConnect>()
        .and_then(|mut pending| pending.0.take())
    else {
        return;
    };
    install_connection(world, Some(addr), false);
}

fn ws_reconnect(world: &mut World) {
    let Some(addr) = world.get_resource::<WsIo>().and_then(|io| io.addr) else {
        return;
    };
    let needs_open = match world.get_non_send::<WsSockets>() {
        None => true,
        Some(sockets) => {
            sockets
                .main
                .as_ref()
                .is_none_or(|ws| !is_active(ws.ready_state()))
                || sockets
                    .msg
                    .as_ref()
                    .is_none_or(|ws| !is_active(ws.ready_state()))
        }
    };
    if !needs_open {
        if let Some(mut io) = world.get_resource_mut::<WsIo>() {
            io.retry_at = None;
        }
        return;
    }
    let now = Instant::now();
    let due = {
        let Some(mut io) = world.get_resource_mut::<WsIo>() else {
            return;
        };
        if io.addr != Some(addr) {
            return;
        }
        match io.retry_at {
            None => {
                io.status.set_status(ConnectionStatus::Error);
                io.retry_at = Some(now + RECONNECT_DELAY);
                false
            }
            Some(at) => now >= at,
        }
    };
    if due {
        open_sockets(world);
    }
}

fn ws_flush(world: &mut World) {
    let (main, msg) = {
        let Some(sockets) = world.get_non_send::<WsSockets>() else {
            return;
        };
        (sockets.main.clone(), sockets.msg.clone())
    };
    let Some(mut io) = world.get_resource_mut::<WsIo>() else {
        return;
    };
    flush_outgoing(&mut io.main_outgoing, main.as_ref());
    flush_outgoing(&mut io.msg_outgoing, msg.as_ref());
}

fn flush_outgoing(rx: &mut mpsc::Receiver<Option<LenPacket>>, socket: Option<&WebSocket>) {
    let Some(socket) = socket else {
        return;
    };
    if socket.ready_state() != WebSocket::OPEN {
        return;
    }
    loop {
        match rx.try_recv() {
            Ok(Some(pkt)) => {
                if socket.send_with_u8_array(&pkt.inner).is_err() {
                    break;
                }
            }
            Ok(None) => {}
            Err(_) => break,
        }
    }
}

fn is_active(ready_state: u16) -> bool {
    ready_state == WebSocket::CONNECTING || ready_state == WebSocket::OPEN
}

fn open_sockets(world: &mut World) {
    let (addr, stream_id, incoming_main, incoming_msg, generation, my_gen, status) = {
        let Some(io) = world.get_resource::<WsIo>() else {
            return;
        };
        let Some(addr) = io.addr else {
            return;
        };
        let my_gen = io.generation.fetch_add(1, Ordering::SeqCst) + 1;
        (
            addr,
            io.stream_id,
            io.incoming_main.clone(),
            io.incoming_msg.clone(),
            io.generation.clone(),
            my_gen,
            io.status.clone(),
        )
    };
    world.remove_non_send::<WsSockets>();
    status.set_status(ConnectionStatus::Connecting);
    if let Some(mut io) = world.get_resource_mut::<WsIo>() {
        io.retry_at = None;
    }

    let (main_url, msg_url) = ws_urls(addr);
    let main = open_socket(
        &main_url,
        incoming_main,
        generation.clone(),
        my_gen,
        Some(status.clone()),
        new_connection_packets(stream_id).collect(),
        true,
    );
    let msg = open_socket(
        &msg_url,
        incoming_msg,
        generation,
        my_gen,
        None,
        Vec::new(),
        false,
    );
    if main.is_none() || msg.is_none() {
        status.set_status(ConnectionStatus::Error);
        if let Some(mut io) = world.get_resource_mut::<WsIo>() {
            io.retry_at = Some(Instant::now() + RECONNECT_DELAY);
        }
    }
    world.insert_non_send(WsSockets { main, msg });
}

fn open_socket(
    url: &str,
    incoming: Arc<AsyncArcQueueTx>,
    generation: Arc<AtomicU64>,
    my_gen: u64,
    status: Option<ThreadConnectionStatus>,
    hello: Vec<LenPacket>,
    inject_new_connection: bool,
) -> Option<WebSocket> {
    let ws = match WebSocket::new(url) {
        Ok(ws) => ws,
        Err(err) => {
            bevy::log::warn!("impeller websocket open failed: {err:?}");
            return None;
        }
    };
    ws.set_binary_type(BinaryType::Arraybuffer);

    let ws_open = ws.clone();
    let incoming_open = incoming.clone();
    let generation_open = generation.clone();
    let status_open = status.clone();
    let on_open = Closure::wrap(Box::new(move || {
        if generation_open.load(Ordering::SeqCst) != my_gen {
            return;
        }
        if inject_new_connection {
            push_incoming(
                &incoming_open,
                LenPacket::new(PacketTy::Msg, NewConnection::ID, 0),
            );
        }
        for pkt in &hello {
            let _ = ws_open.send_with_u8_array(&pkt.inner);
        }
        if let Some(status) = &status_open {
            status.set_status(ConnectionStatus::Success);
        }
    }) as Box<dyn FnMut()>);
    ws.set_onopen(Some(on_open.as_ref().unchecked_ref()));
    on_open.forget();

    let incoming_msg = incoming;
    let generation_msg = generation.clone();
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        if generation_msg.load(Ordering::SeqCst) != my_gen {
            return;
        }
        let Ok(buf) = event.data().dyn_into::<js_sys::ArrayBuffer>() else {
            return;
        };
        let arr = Uint8Array::new(&buf);
        let mut bytes = vec![0u8; arr.length() as usize];
        arr.copy_to(&mut bytes);
        if bytes.len() < 4 {
            return;
        }
        push_incoming(&incoming_msg, LenPacket { inner: bytes });
    }) as Box<dyn FnMut(_)>);
    ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();

    let status_err = status.clone();
    let generation_err = generation.clone();
    let on_error = Closure::wrap(Box::new(move |_e: JsValue| {
        if generation_err.load(Ordering::SeqCst) != my_gen {
            return;
        }
        if let Some(status) = &status_err {
            status.set_status(ConnectionStatus::Error);
        }
    }) as Box<dyn FnMut(_)>);
    ws.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    on_error.forget();

    let generation_close = generation;
    let on_close = Closure::wrap(Box::new(move |_e: JsValue| {
        if generation_close.load(Ordering::SeqCst) != my_gen {
            return;
        }
        if let Some(status) = &status {
            status.set_status(ConnectionStatus::Error);
        }
    }) as Box<dyn FnMut(_)>);
    ws.set_onclose(Some(on_close.as_ref().unchecked_ref()));
    on_close.forget();

    Some(ws)
}

fn push_incoming(incoming: &AsyncArcQueueTx, pkt: LenPacket) {
    let needed = pkt.inner.len().saturating_add(16);
    let Ok(grant) = incoming.grant(needed) else {
        bevy::log::warn!("dropping impeller websocket packet; rx queue full");
        return;
    };
    PacketGrantW::new(grant).commit_len_pkt(pkt);
}

fn ws_urls(tcp: SocketAddr) -> (String, String) {
    let ip = match tcp.ip() {
        IpAddr::V4(v4) if v4.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(v6) if v6.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        other => other,
    };
    let port = tcp.port().saturating_add(impeller::ASSETS_HTTP_PORT_OFFSET);
    let host = match ip {
        IpAddr::V4(v4) => format!("{v4}:{port}"),
        IpAddr::V6(v6) => format!("[{v6}]:{port}"),
    };
    (
        format!("ws://{host}/impeller"),
        format!("ws://{host}/impeller/msg"),
    )
}
