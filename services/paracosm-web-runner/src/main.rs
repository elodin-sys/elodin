use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        ConnectInfo, TypedHeader,
    },
    headers,
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use paracosm::{
    runner::IntoSimRunner,
    sync::{ClientChannel, ClientTransport, ServerChannel},
};
use paracosm_py::SimBuilder;
use pyo3::{types::PyModule, Python};
use std::{net::SocketAddr, thread, time::Duration};
use tracing::{error, info, trace};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    {
        use paracosm_py::paracosm_py;
        pyo3::append_to_inittab!(paracosm_py);
    }

    let app = Router::new()
        .layer(tower_http::cors::CorsLayer::very_permissive())
        .route("/ws", get(ws_handler));
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service_with_connect_info::<SocketAddr>())
        .await?;
    Ok(())
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    user_agent: Option<TypedHeader<headers::UserAgent>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    let user_agent = if let Some(TypedHeader(user_agent)) = user_agent {
        user_agent.to_string()
    } else {
        String::from("unknown")
    };
    info!(?user_agent, ?addr, "client connected");
    ws.on_upgrade(move |socket| handle_socket(socket, addr))
}

async fn handle_socket(socket: WebSocket, _who: SocketAddr) {
    let (server, client) = paracosm::sync::channel_pair();
    let code_client = client.clone();
    let ClientChannel { rx, .. } = client;
    let (mut sender, mut receiver) = socket.split();
    let mut loaded = false;
    let mut recv = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(code) => {
                    if loaded {
                        code_client.send_msg(paracosm::ServerMsg::Exit);
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    loaded = true;
                    if let Err(err) = load_python(code, server.clone()) {
                        trace!(?err, "error loading python");
                    }
                }
                Message::Binary(buf) => {
                    let msg: paracosm::ServerMsg = postcard::from_bytes(&buf).unwrap();
                    code_client.send_msg(msg);
                }
                _ => {}
            }
        }
    });

    let mut send = tokio::spawn(async move {
        while let Ok(msg) = rx.recv_async().await {
            let bytes = postcard::to_allocvec(&msg).unwrap();
            sender.send(Message::Binary(bytes)).await.unwrap();
        }
    });
    tokio::select! {
        res = (&mut send) => {
            if let Err(err) = res {
                error!(?err, "socket send error");
            }
            recv.abort();
        }
        res = (&mut recv) => {
            if let Err(err) = res {
                error!(?err, "socket rec error");
            }
            send.abort();
        }
    }
}

fn load_python(code: String, server: ServerChannel) -> anyhow::Result<()> {
    thread::spawn(move || -> anyhow::Result<()> {
        pyo3::prepare_freethreaded_python();
        let builder = Python::with_gil(|py| {
            let sim = PyModule::from_code(py, &code, "./test.py", "./")?;
            let callable = sim.getattr("sim")?;
            let builder = callable.call0()?;
            let builder: SimBuilder = builder.extract().unwrap();

            pyo3::PyResult::Ok(builder.0)
        })
        .map_err(|e| {
            Python::with_gil(|py| {
                e.print_and_set_sys_last_vars(py);
                e
            })
        })?;
        let runner = builder.into_runner();
        let mut app = runner
            .run_mode(paracosm::runner::RunMode::RealTime)
            .build(server);
        app.run();
        Ok(())
    });
    Ok(())
}
