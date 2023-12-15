use elo_conduit::bevy::{ConduitSubscribePlugin, Msg, Subscriptions};
use futures::future::select_all;
use paracosm::{
    runner::IntoSimRunner,
    sync::{SendPlbPlugin, SyncPlugin},
};
use paracosm_py::SimBuilder;
use paracosm_types::sandbox::{
    self,
    sandbox_control_server::{SandboxControl, SandboxControlServer},
    UpdateCodeReq, UpdateCodeResp,
};
use pyo3::{types::PyModule, Python};
use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};
use tokio::net::TcpListener;
use tonic::{async_trait, transport::Server, Response, Status};
use tracing::info;

mod config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    {
        use paracosm_py::paracosm_py;
        pyo3::append_to_inittab!(paracosm_py);
    }

    let (server_tx, server_rx) = flume::unbounded();
    let config = crate::config::Config::new()?;
    let control = ControlService::new(
        server_tx.clone(),
        server_rx,
        config.control_addr,
        Subscriptions::default(),
    );
    let control = tokio::spawn(control.run());
    let sim = SimServer::new(server_tx, config.sim_addr);
    let sim = tokio::spawn(sim.run());
    let (res, _, _) = select_all([control, sim]).await;
    res?
}

struct SimServer {
    address: SocketAddr,
    bevy_tx: flume::Sender<Msg<'static>>,
}

impl SimServer {
    fn new(bevy_tx: flume::Sender<elo_conduit::bevy::Msg<'static>>, address: SocketAddr) -> Self {
        Self { address, bevy_tx }
    }

    async fn run(self) -> anyhow::Result<()> {
        let listener = TcpListener::bind(self.address).await?;
        loop {
            let (socket, _addr) = listener.accept().await?;
            let (rx_socket, tx_socket) = socket.into_split();
            tokio::spawn(elo_conduit::bevy::handle_socket(
                self.bevy_tx.clone(),
                tx_socket,
                rx_socket,
            ));
        }
    }
}

#[derive(Clone)]
struct ControlService {
    server_tx: flume::Sender<Msg<'static>>,
    server_rx: flume::Receiver<Msg<'static>>,
    subscriptions: Subscriptions,
    loaded: Arc<AtomicBool>,
    address: SocketAddr,
}

impl ControlService {
    fn new(
        server_tx: flume::Sender<Msg<'static>>,
        server_rx: flume::Receiver<Msg<'static>>,
        address: SocketAddr,
        subscriptions: Subscriptions,
    ) -> Self {
        Self {
            address,
            server_tx,
            server_rx,
            loaded: Arc::new(AtomicBool::new(false)),
            subscriptions,
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
        let rx = self.server_rx.clone();
        let subscriptions = self.subscriptions.clone();
        thread::spawn(move || -> anyhow::Result<()> {
            if loaded.swap(true, Ordering::SeqCst) {
                server_tx.send(Msg::Exit)?;
                thread::sleep(Duration::from_millis(100));
            }
            let runner = builder.into_runner();
            let mut app = runner
                .run_mode(paracosm::runner::RunMode::RealTime)
                .build_with_plugins((
                    SyncPlugin {
                        plugin: ConduitSubscribePlugin::new(rx),
                        subscriptions,
                    },
                    SendPlbPlugin,
                ));
            app.run();
            Ok(())
        });

        Ok(Response::new(UpdateCodeResp {
            status: sandbox::Status::Success.into(),
            errors: vec![],
        }))
    }
}
