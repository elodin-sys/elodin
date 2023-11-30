use bytes::Bytes;
use futures::future::select_all;
use futures::{SinkExt, StreamExt};
use paracosm::runner::IntoSimRunner;
use paracosm::sync::ServerTransport;
use paracosm::{ClientMsg, ServerMsg};
use paracosm_py::SimBuilder;
use paracosm_types::sandbox;
use paracosm_types::sandbox::sandbox_control_server::SandboxControlServer;
use paracosm_types::sandbox::{
    sandbox_control_server::SandboxControl, UpdateCodeReq, UpdateCodeResp,
};
use pyo3::types::PyModule;
use pyo3::Python;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};
use tonic::transport::Server;
use tonic::{async_trait, Response, Status};
use tracing::info;

mod config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    {
        use paracosm_py::paracosm_py;
        pyo3::append_to_inittab!(paracosm_py);
    }

    let (sim_channels, client_channels) = channel_pair();
    let config = crate::config::Config::new()?;
    let control = ControlService::new(
        sim_channels,
        client_channels.tx.clone(),
        config.control_addr,
    );
    let control = tokio::spawn(control.run());
    let sim = SimServer::new(client_channels, config.sim_addr);
    let sim = tokio::spawn(sim.run());
    let (res, _, _) = select_all([control, sim]).await;
    res?
}

struct SimServer {
    address: SocketAddr,
    client_channels: ClientChannels,
}

impl SimServer {
    fn new(client_channels: ClientChannels, address: SocketAddr) -> Self {
        Self {
            address,
            client_channels,
        }
    }

    async fn run(self) -> anyhow::Result<()> {
        let listener = TcpListener::bind(self.address).await?;
        loop {
            let (socket, _addr) = listener.accept().await?;
            let mut client_channels = self.client_channels.clone();
            tokio::spawn(async move {
                let (rx, tx) = socket.into_split();
                let mut tx = FramedWrite::new(tx, LengthDelimitedCodec::new());
                let mut rx = FramedRead::new(rx, LengthDelimitedCodec::new());
                let tx = async move {
                    while let Ok(msg) = client_channels.rx.recv().await {
                        tx.send(msg).await?;
                    }
                    Ok::<(), anyhow::Error>(())
                };
                let rx = async move {
                    while let Some(buf) = rx.next().await {
                        let buf = buf?;
                        let Ok(msg) = postcard::from_bytes::<ServerMsg>(&buf[..]) else {
                            continue;
                        };
                        client_channels.tx.send_async(msg).await?;
                    }
                    Ok::<(), anyhow::Error>(())
                };
                tokio::select! {
                    res = tx => { res }
                    res = rx => { res }
                }
            });
        }
    }
}

#[derive(Clone)]
struct ControlService {
    sim_channels: SimChannels,
    server_tx: flume::Sender<ServerMsg>,
    loaded: Arc<AtomicBool>,
    address: SocketAddr,
}

impl ControlService {
    fn new(
        sim_channels: SimChannels,
        server_tx: flume::Sender<ServerMsg>,
        address: SocketAddr,
    ) -> Self {
        Self {
            sim_channels,
            address,
            server_tx,
            loaded: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let address = self.address;
        let svc = SandboxControlServer::new(self);
        info!(api.addr = ?address, "control api listening");
        let reflection = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(paracosm_types::FILE_DESCRIPTOR_SET)
            .build()
            .unwrap();

        Server::builder()
            .add_service(svc)
            .add_service(reflection)
            .serve(address)
            .await?;
        Ok(())
    }
}

#[async_trait]
impl SandboxControl for ControlService {
    async fn update_code(
        &self,
        req: tonic::Request<UpdateCodeReq>,
    ) -> Result<Response<UpdateCodeResp>, Status> {
        let req = req.into_inner();
        pyo3::prepare_freethreaded_python();
        let loaded = self.loaded.clone();
        let server_tx = self.server_tx.clone();
        let builder = match Python::with_gil(|py| {
            let sim = PyModule::from_code(py, &req.code, "./test.py", "./")?;
            let callable = sim.getattr("sim")?;
            let builder = callable.call0()?;
            let builder: SimBuilder = builder.extract().unwrap();

            pyo3::PyResult::Ok(builder.0)
        }) {
            Ok(builder) => builder,
            Err(e) => {
                let err = Python::with_gil(|py| {
                    e.print_and_set_sys_last_vars(py);
                    e
                });
                let err = err.to_string();
                return Ok(Response::new(UpdateCodeResp {
                    status: sandbox::Status::Error.into(),
                    errors: vec![err],
                }));
            }
        };
        let channels = self.sim_channels.clone();
        thread::spawn(move || -> anyhow::Result<()> {
            if loaded.swap(true, Ordering::SeqCst) {
                server_tx.send(paracosm::ServerMsg::Exit)?;
                thread::sleep(Duration::from_millis(100));
            }

            let runner = builder.into_runner();
            let mut app = runner
                .run_mode(paracosm::runner::RunMode::RealTime)
                .build(channels);
            app.run();
            Ok(())
        });

        Ok(Response::new(UpdateCodeResp {
            status: sandbox::Status::Success.into(),
            errors: vec![],
        }))
    }
}

#[derive(Clone, bevy::prelude::Resource)]
struct SimChannels {
    tx: tokio::sync::broadcast::Sender<Bytes>,
    rx: flume::Receiver<ServerMsg>,
}

impl ServerTransport for SimChannels {
    fn send_msg(&self, msg: ClientMsg) {
        let bytes = postcard::to_allocvec(&msg).unwrap();
        let bytes = Bytes::from(bytes);
        self.tx.send(bytes).unwrap();
    }

    fn try_recv_msg(&self) -> Option<ServerMsg> {
        self.rx.try_recv().ok()
    }
}

struct ClientChannels {
    rx: tokio::sync::broadcast::Receiver<Bytes>,
    tx: flume::Sender<ServerMsg>,
}

fn channel_pair() -> (SimChannels, ClientChannels) {
    let (client_tx, client_rx) = broadcast::channel(128);
    let (server_tx, server_rx) = flume::unbounded();
    (
        SimChannels {
            tx: client_tx,
            rx: server_rx,
        },
        ClientChannels {
            tx: server_tx,
            rx: client_rx,
        },
    )
}

impl Clone for ClientChannels {
    fn clone(&self) -> Self {
        Self {
            rx: self.rx.resubscribe(),
            tx: self.tx.clone(),
        }
    }
}
