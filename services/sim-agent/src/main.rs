use anyhow::Context;
use conduit::client::MsgPair;
use conduit::server::TcpServer;
use config::SandboxConfig;
use elodin_types::sandbox::{
    self,
    sandbox_client::SandboxClient,
    sandbox_control_server::{SandboxControl, SandboxControlServer},
    BuildReq, FileTransferReq, UpdateCodeReq, UpdateCodeResp,
};
use nox::Client;
use nox_ecs::{Compiled, ConduitExec, WorldExec};
use std::{io::Seek, time::Duration};
use std::{net::SocketAddr, thread, time::Instant};
use tokio::io::AsyncWriteExt;
use tonic::{
    async_trait,
    transport::{Channel, Endpoint, Server, Uri},
    Response, Status,
};
use tonic_health::pb::{health_client::HealthClient, HealthCheckRequest};
use tracing::{debug, error, info, info_span, Instrument};

mod config;
mod headless;
mod pytest;

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
        builder_addr,
    }) = config.sandbox
    {
        let (server_tx, server_rx) = flume::unbounded();
        let sim_runner = SimRunner::new(server_rx)?;
        let control = ControlService::new(sim_runner, control_addr, builder_addr)?;
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
    sim_builder: SandboxClient<Channel>,
    sim_runner: SimRunner,
    address: SocketAddr,
}

impl ControlService {
    fn new(sim_runner: SimRunner, address: SocketAddr, addr: Uri) -> anyhow::Result<Self> {
        let channel = builder_channel(addr);
        let sim_builder = SandboxClient::new(channel.clone());
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
        let start = std::time::Instant::now();
        tracing::debug!("sending code to sim builder");
        let artifacts_file_name = match self.sim_builder.clone().build(request).await {
            Ok(res) => res.into_inner().artifacts_file,
            Err(err) => {
                return Ok(UpdateCodeResp {
                    status: sandbox::Status::Error.into(),
                    errors: vec![err.to_string()],
                })
            }
        };

        let mut artifacts = tokio::fs::File::from_std(tempfile::tempfile()?);
        let file_req = FileTransferReq {
            name: artifacts_file_name,
        };
        let mut stream = self
            .sim_builder
            .clone()
            .recv_file(file_req)
            .await?
            .into_inner();
        while let Some(chunk) = stream.message().await? {
            artifacts.write_all(&chunk.data).await?;
        }
        let mut artifacts = artifacts.into_std().await;
        artifacts.rewind()?;

        let buf = std::io::BufReader::new(artifacts);
        let mut tar = tar::Archive::new(buf);
        let tmp_dir = tempfile::tempdir()?;
        tar.unpack(tmp_dir.path())?;
        let artifacts = tmp_dir.path().join("artifacts");

        tracing::debug!(duration = ?start.elapsed(), path = %artifacts.display(), "received artifacts");
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
    client: Client,
    exec_tx: flume::Sender<WorldExec<Compiled>>,
}

impl SimRunner {
    fn new(server_rx: flume::Receiver<MsgPair>) -> anyhow::Result<Self> {
        let mut client = Client::cpu()?;
        client.disable_optimizations();
        let (exec_tx, exec_rx) = flume::bounded(1);
        std::thread::spawn(move || -> anyhow::Result<()> {
            let exec: WorldExec<Compiled> = exec_rx.recv()?;
            let mut conduit_exec = ConduitExec::new(exec, server_rx.clone());
            loop {
                let start = Instant::now();
                if let Err(err) = conduit_exec.run() {
                    error!(?err, "failed to run conduit exec");
                    return Err(err.into());
                }
                let sleep_time = conduit_exec.run_time_step().saturating_sub(start.elapsed());
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
        Ok(Self { client, exec_tx })
    }

    fn update(&self, exec: WorldExec) -> anyhow::Result<()> {
        let exec = exec.compile(self.client.clone())?;
        self.exec_tx
            .try_send(exec)
            .context("failed to send new exec to sim runner")
    }
}

pub fn builder_channel(addr: Uri) -> Channel {
    let scheme = addr.scheme().map(|s| s.as_str());
    let use_vsock = scheme == Some("vsock");
    let endpoint = Endpoint::from(addr);

    if use_vsock {
        #[cfg(not(target_os = "linux"))]
        panic!("vsock not supported on os");
        #[cfg(target_os = "linux")]
        return endpoint.connect_with_connector_lazy(tower::service_fn(vsock_connect));
    }
    endpoint.connect_lazy()
}

#[cfg(target_os = "linux")]
async fn vsock_connect(uri: Uri) -> Result<tokio_vsock::VsockStream, std::io::Error> {
    let cid = uri.host().unwrap().parse::<u32>().unwrap();
    let port = uri.port_u16().unwrap();
    let addr = tokio_vsock::VsockAddr::new(cid, port as u32);
    tracing::debug!(cid, port, "connecting to vsock");
    tokio_vsock::VsockStream::connect(addr).await
}
