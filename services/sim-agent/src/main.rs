use elodin_conduit::bevy::{ConduitSubscribePlugin, Subscriptions};
use elodin_conduit::bevy_sync::{SendPlbPlugin, SyncPlugin};
use elodin_conduit::client::MsgPair;
use elodin_conduit::server::TcpServer;
use elodin_core::runner::IntoSimRunner;
use elodin_py::SimBuilder;
use elodin_types::sandbox::{
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

use tonic::{async_trait, transport::Server, Response, Status};
use tracing::{info, info_span, Instrument};

mod config;
mod headless;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    {
        use elodin_py::elodin_py;
        pyo3::append_to_inittab!(elodin_py);
    }

    let config = config::Config::new()?;
    let mut tasks = tokio::task::JoinSet::new();

    if let Some(sandbox_config) = config.sandbox {
        let (server_tx, server_rx) = flume::unbounded();
        let control = ControlService::new(
            server_tx.clone(),
            server_rx,
            sandbox_config.control_addr,
            Subscriptions::default(),
        );
        tasks.spawn(control.run().instrument(info_span!("control").or_current()));
        let sim = async move {
            let sim = TcpServer::bind(server_tx, sandbox_config.sim_addr).await?;
            sim.run().instrument(info_span!("sim").or_current()).await?;
            Ok(())
        };
        tasks.spawn(sim);
    }

    if let Some(mc_agent_config) = config.monte_carlo {
        let runner = headless::Runner::new(mc_agent_config).await?;
        tasks.spawn(runner.run().instrument(info_span!("mc").or_current()));
    }

    while let Some(res) = tasks.join_next().await {
        res.unwrap()?;
    }
    Ok(())
}

#[derive(Clone)]
struct ControlService {
    server_tx: flume::Sender<MsgPair>,
    server_rx: flume::Receiver<MsgPair>,
    subscriptions: Subscriptions,
    loaded: Arc<AtomicBool>,
    address: SocketAddr,
}

impl ControlService {
    fn new(
        server_tx: flume::Sender<MsgPair>,
        server_rx: flume::Receiver<MsgPair>,
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
            .register_encoded_file_descriptor_set(elodin_types::FILE_DESCRIPTOR_SET)
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
        let res = Python::with_gil(|py| {
            let sim = PyModule::from_code(py, &req.code, "./test.py", "./")?;
            let callable = sim.getattr("sim")?;
            let builder = callable.call0()?;
            let builder: SimBuilder = builder.extract().unwrap();

            pyo3::PyResult::Ok(builder.0)
        });
        let builder = match res {
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
                let (tx, _) = flume::unbounded();
                server_tx.send(MsgPair {
                    msg: elodin_conduit::client::Msg::Control(elodin_conduit::ControlMsg::Exit),
                    tx: tx.downgrade(),
                })?;

                thread::sleep(Duration::from_millis(100));
            }
            let runner = builder.into_runner();
            let mut app = runner
                .run_mode(elodin_core::runner::RunMode::RealTime)
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
