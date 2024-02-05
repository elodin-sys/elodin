use elodin_conduit::bevy::{ConduitSubscribePlugin, Msg, Subscriptions};
use elodin_conduit::bevy_sync::{SendPlbPlugin, SyncPlugin, DEFAULT_SUB_FILTERS};
use elodin_core::runner::IntoSimRunner;
use elodin_py::SimBuilder;
use elodin_types::sandbox::{
    self,
    sandbox_control_server::{SandboxControl, SandboxControlServer},
    UpdateCodeReq, UpdateCodeResp,
};
use elodin_types::{Batch, BatchResults, Run, BATCH_TOPIC, RUN_TOPIC};
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
use tracing::{error, info, info_span, Instrument};

mod config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    {
        use elodin_py::elodin_py;
        pyo3::append_to_inittab!(elodin_py);
    }

    // TODO(Akhil): Handle graceful shutdown
    match config::Config::new()? {
        config::Config::WebRunner(config) => {
            run_web_runner(config)
                .instrument(info_span!("web_runner"))
                .await?
        }
        config::Config::MonteCarlo(config) => {
            run_mc_agent(config)
                .instrument(info_span!("mc_agent"))
                .await?
        }
    }
    Ok(())
}

async fn run_mc_agent(config: config::MonteCarloConfig) -> anyhow::Result<()> {
    info!("running monte carlo agent");
    let redis = redis::Client::open(config.redis_url)?;
    let mut msg_queue = redmq::MsgQueue::new(&redis, "sim-agent", config.pod_name).await?;
    loop {
        let batches = msg_queue.recv::<Batch>(BATCH_TOPIC, 1).await?;

        match process_batches(&mut msg_queue, &batches).await {
            Ok(_) => {}
            Err(err) => error!(?err, "error processing batches"),
        }

        msg_queue.ack(BATCH_TOPIC, &batches).await?;
        msg_queue.del(BATCH_TOPIC, &batches).await?;
    }
}

async fn process_batches(
    msg_queue: &mut redmq::MsgQueue,
    batches: &[redmq::Received<Batch>],
) -> anyhow::Result<()> {
    let Some(run_id) = batches.iter().find(|b| !b.buffer).map(|b| b.id.clone()) else {
        return Ok(());
    };
    let Some(run) = msg_queue.get::<Run>(RUN_TOPIC, &run_id).await? else {
        anyhow::bail!("monte carlo run {} not found", run_id);
    };

    // TODO: run the batch simulations, collect and upload the results
    let start_time = chrono::Utc::now();
    let failed = 0;
    let elapsed = chrono::Utc::now().signed_duration_since(start_time);
    let results_topic = format!("mc:results:{}", run.id);

    let mut results = Vec::default();
    for b in batches {
        info!(%run.name, %run.id, batch_no = %b.batch_no, "simulating batch");
        results.push(BatchResults {
            batch_no: b.batch_no,
            failed,
            start_time,
            run_time_seconds: elapsed.num_seconds() as u64,
        });
    }

    msg_queue.send(&results_topic, results).await?;
    Ok(())
}

async fn run_web_runner(config: config::WebRunnerConfig) -> anyhow::Result<()> {
    let (server_tx, server_rx) = flume::unbounded();
    let control = ControlService::new(
        server_tx.clone(),
        server_rx,
        config.control_addr,
        Subscriptions::default(),
    );
    let mut tasks = tokio::task::JoinSet::new();
    tasks.spawn(control.run().instrument(info_span!("control").or_current()));
    let sim = SimServer::new(server_tx, config.sim_addr);
    tasks.spawn(sim.run().instrument(info_span!("sim").or_current()));
    while let Some(res) = tasks.join_next().await {
        res.unwrap()?;
    }
    Ok(())
}

struct SimServer {
    address: SocketAddr,
    bevy_tx: flume::Sender<Msg<'static>>,
}

impl SimServer {
    fn new(
        bevy_tx: flume::Sender<elodin_conduit::bevy::Msg<'static>>,
        address: SocketAddr,
    ) -> Self {
        Self { address, bevy_tx }
    }

    async fn run(self) -> anyhow::Result<()> {
        let listener = TcpListener::bind(self.address).await?;
        loop {
            let (socket, _addr) = listener.accept().await?;
            let (rx_socket, tx_socket) = socket.into_split();
            tokio::spawn(elodin_conduit::bevy::handle_socket(
                self.bevy_tx.clone(),
                tx_socket,
                rx_socket,
                DEFAULT_SUB_FILTERS,
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
