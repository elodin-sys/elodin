use anyhow::Context;
use conduit::client::MsgPair;
use conduit::server::TcpServer;
use config::SandboxConfig;
use elodin_types::sandbox::{
    self,
    build_sim_client::BuildSimClient,
    sandbox_control_server::{SandboxControl, SandboxControlServer},
    BuildReq, UpdateCodeReq, UpdateCodeResp,
};
use nox::Client;
use nox_ecs::{ConduitExec, WorldExec};
use std::time::Duration;
use std::{net::SocketAddr, thread, time::Instant};
use tonic::{
    async_trait,
    codec::CompressionEncoding,
    transport::{Channel, Endpoint, Server, Uri},
    Response, Status,
};
use tonic_health::pb::{health_client::HealthClient, HealthCheckRequest};
use tracing::{debug, error, info, info_span, Instrument};

mod config;
mod headless;

const WAIT_DURATION: Duration = Duration::from_millis(200);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let config = config::Config::new()?;
    let mut tasks = tokio::task::JoinSet::new();

    if let Some(SandboxConfig {
        control_addr,
        sim_addr,
        builder_cid,
    }) = config.sandbox
    {
        let (server_tx, server_rx) = flume::unbounded();
        let sim_runner = SimRunner::new(server_rx);
        let control = ControlService::new(sim_runner, control_addr, builder_cid)?;
        tasks.spawn(control.run().instrument(info_span!("control").or_current()));
        let sim = async move {
            let sim = TcpServer::bind(server_tx, sim_addr).await?;
            sim.run().await?;
            Ok(())
        };
        tasks.spawn(sim.instrument(info_span!("sim").or_current()));
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
    sim_builder_health: HealthClient<Channel>,
    sim_builder: BuildSimClient<Channel>,
    sim_runner: SimRunner,
    address: SocketAddr,
}

impl ControlService {
    fn new(sim_runner: SimRunner, address: SocketAddr, builder_cid: u32) -> anyhow::Result<Self> {
        let addr = format!("vsock://{}:50051", builder_cid).parse::<Uri>()?;
        let channel =
            Endpoint::from(addr).connect_with_connector_lazy(tower::service_fn(vsock_connect));
        let sim_builder = BuildSimClient::new(channel.clone())
            .send_compressed(CompressionEncoding::Zstd)
            .accept_compressed(CompressionEncoding::Zstd);
        let sim_builder_health = HealthClient::new(channel);
        Ok(Self {
            sim_builder_health,
            sim_builder,
            sim_runner,
            address,
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let health_checker = self.clone();

        let address = self.address;
        let svc = SandboxControlServer::new(self);
        info!(api.addr = ?address, "control api listening");
        let reflection = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(elodin_types::FILE_DESCRIPTOR_SET)
            .build()
            .unwrap();

        let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
        health_checker.wait_for_builder().await;
        health_reporter
            .set_serving::<SandboxControlServer<ControlService>>()
            .await;

        let span = tracing::Span::current();
        Server::builder()
            .trace_fn(move |_| span.clone())
            .add_service(svc)
            .add_service(reflection)
            .add_service(health_service)
            .serve(address)
            .await?;
        Ok(())
    }

    async fn handle_update_code(&self, req: UpdateCodeReq) -> anyhow::Result<UpdateCodeResp> {
        let request = tonic::Request::new(BuildReq { code: req.code });
        tracing::debug!("sending code to sim builder");
        let artifacts = match self.sim_builder.clone().build(request).await {
            Ok(res) => res.into_inner().artifacts,
            Err(err) => {
                return Ok(UpdateCodeResp {
                    status: sandbox::Status::Error.into(),
                    errors: vec![err.to_string()],
                })
            }
        };

        tracing::debug!(len = artifacts.len(), "received artifacts");
        let mut tar = tar::Archive::new(artifacts.as_slice());
        let tmp_dir = tempfile::tempdir()?;
        tar.unpack(tmp_dir.path())?;
        let artifacts = tmp_dir.path().join("artifacts");

        let exec = match nox_ecs::WorldExec::read_from_dir(artifacts) {
            Ok(exec) => exec,
            Err(err) => {
                return Ok(UpdateCodeResp {
                    status: sandbox::Status::Error.into(),
                    errors: vec![err.to_string()],
                });
            }
        };

        self.sim_runner.update(exec)?;

        Ok(UpdateCodeResp {
            status: sandbox::Status::Success.into(),
            errors: vec![],
        })
    }

    async fn wait_for_builder(&self) {
        let req = HealthCheckRequest::default();
        let mut client = self.sim_builder_health.clone();
        loop {
            debug!("checking sim builder status");
            match client.check(req.clone()).await {
                Ok(_) => break,
                Err(err) => {
                    let code = err.code();
                    let message = err.message();
                    debug!(%code, message, "sim builder is not ready, waiting {WAIT_DURATION:?}")
                }
            }
            tokio::time::sleep(WAIT_DURATION).await;
        }
        info!("sim builder is ready");
    }
}

#[async_trait]
impl SandboxControl for ControlService {
    async fn update_code(
        &self,
        req: tonic::Request<UpdateCodeReq>,
    ) -> Result<Response<UpdateCodeResp>, Status> {
        let req = req.into_inner();
        match self.handle_update_code(req).await {
            Ok(resp) => {
                if !resp.errors.is_empty() {
                    error!(err = ?resp.errors, "failed to update code");
                } else {
                    info!("updated code")
                }
                Ok(Response::new(resp))
            }
            Err(err) => {
                error!(?err, "failed to update code");
                let status = Status::internal(err.to_string());
                Err(status)
            }
        }
    }
}

#[derive(Clone)]
struct SimRunner {
    exec_tx: flume::Sender<WorldExec>,
}

impl SimRunner {
    fn new(server_rx: flume::Receiver<MsgPair>) -> Self {
        let (exec_tx, exec_rx) = flume::bounded(1);
        std::thread::spawn(move || -> anyhow::Result<()> {
            let client = Client::cpu()?;
            let exec: WorldExec = exec_rx.recv()?;
            let mut conduit_exec = ConduitExec::new(exec, server_rx.clone());
            loop {
                let start = Instant::now();
                if let Err(err) = conduit_exec.run(&client) {
                    error!(?err, "failed to run conduit exec");
                    return Err(err.into());
                }
                let sleep_time = conduit_exec.time_step().saturating_sub(start.elapsed());
                thread::sleep(sleep_time);

                if let Ok(exec) = exec_rx.try_recv() {
                    tracing::info!("received new code, updating sim");
                    let conns = conduit_exec.connections().to_vec();
                    conduit_exec = ConduitExec::new(exec, server_rx.clone());
                    for conn in conns {
                        conduit_exec.add_connection(conn)?;
                    }
                }
            }
        });
        Self { exec_tx }
    }

    fn update(&self, exec: WorldExec) -> anyhow::Result<()> {
        self.exec_tx
            .try_send(exec)
            .context("failed to send new exec to sim runner")
    }
}

async fn vsock_connect(uri: Uri) -> Result<tokio_vsock::VsockStream, std::io::Error> {
    let cid = uri.host().unwrap().parse::<u32>().unwrap();
    let port = uri.port_u16().unwrap();
    let addr = tokio_vsock::VsockAddr::new(cid, port as u32);
    tracing::debug!(cid, port, "connecting to vsock");
    tokio_vsock::VsockStream::connect(addr).await
}
