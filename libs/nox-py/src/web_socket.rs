use std::io;

use axum::{
    extract::{ws, State, WebSocketUpgrade},
    http::HeaderMap,
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use bytes::{Bytes, BytesMut};
use conduit::{client::MsgPair, server::handle_stream_sink};
use futures::{SinkExt, StreamExt, TryStreamExt};
use nox_ecs::{nox, ConduitExec, Error, WorldExec};
use std::net::SocketAddr;

pub fn spawn_ws_server(
    socket_addr: std::net::SocketAddr,
    exec: WorldExec,
    client: &nox::Client,
    cancel_token: Option<tokio_util::sync::CancellationToken>,
    check_canceled: impl Fn() -> bool,
    addr_tx: flume::Sender<SocketAddr>,
) -> Result<(), Error> {
    use std::time::Instant;

    let time_step = exec.time_step();
    let (tx, rx) = flume::unbounded();
    let mut conduit_exec = ConduitExec::new(exec, rx);
    let axum_token = cancel_token.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let app = Router::new()
                .route("/ws", get(sim_socket))
                .route("/", get(viewer))
                .route("/editor-web.js", get(editor_web_js))
                .route("/editor-web_bg.wasm", get(editor_web_wasm))
                .with_state(WSContext {
                    socket: tx.clone(),
                    cancel_token: axum_token.clone(),
                });

            let listener = tokio::net::TcpListener::bind(&socket_addr).await.unwrap();
            let _ = addr_tx.send(listener.local_addr().unwrap());
            axum::serve(listener, app.into_make_service())
                .with_graceful_shutdown(async move {
                    if let Some(axum_token) = axum_token {
                        axum_token.cancelled().await
                    } else {
                        std::future::pending().await
                    }
                })
                .await
        })
        .unwrap();
    });
    loop {
        let start = Instant::now();
        conduit_exec.run(client)?;
        if check_canceled() {
            break Ok(());
        }
        let sleep_time = time_step.saturating_sub(start.elapsed());
        if !sleep_time.is_zero() {
            std::thread::sleep(sleep_time)
        }
    }
}

#[derive(Clone)]
pub struct WSContext {
    socket: flume::Sender<MsgPair>,
    cancel_token: Option<tokio_util::sync::CancellationToken>,
}

async fn sim_socket(ws: WebSocketUpgrade, State(context): State<WSContext>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| async move {
        let (ws_tx, ws_rx) = socket.split();
        let ws_rx = ws_rx
            .try_filter_map(|msg| async move {
                let ws::Message::Binary(bytes) = msg else {
                    return Ok(None);
                };
                Ok(Some(BytesMut::from(&bytes[..])))
            })
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err));
        let ws_rx = Box::pin(ws_rx);
        let ws_tx = ws_tx
            .sink_map_err(|err| io::Error::new(io::ErrorKind::Other, err))
            .with(
                |m: Bytes| async move { Ok::<_, std::io::Error>(ws::Message::Binary(m.to_vec())) },
            );
        let ws_tx = Box::pin(ws_tx);

        if let Err(_) = handle_stream_sink(context.socket.clone(), ws_tx, ws_rx).await {
            if let Some(cancel_token) = context.cancel_token.as_ref() {
                cancel_token.cancel();
            }
        }
    })
}

async fn viewer() -> impl IntoResponse {
    Html(
        r##"
<style>
* {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
}
</style>
<script type="module">
import init from '/editor-web.js'
init("/editor-web_bg.wasm")
</script>
<div id="editor-container" data-ws-url="/ws" style="width: 100%; height: 100%;">
<canvas id="editor" style="width: 100%; height: 100%;" oncontextmenu="return false;" />
</div>
"##,
    )
}

async fn editor_web_js() -> impl IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "application/javascript".parse().unwrap());
    (headers, include_str!("../assets/editor-web.js"))
}

async fn editor_web_wasm() -> impl IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "application/wasm".parse().unwrap());

    (headers, include_bytes!("../assets/editor-web_bg.wasm"))
}
