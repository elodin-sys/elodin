use config::SandboxConfig;
use elodin_types::sandbox::{
    self,
    sandbox_client::SandboxClient,
    sandbox_control_server::{SandboxControl, SandboxControlServer},
    BuildReq, FileTransferReq, UpdateCodeReq, UpdateCodeResp,
};
use nox_ecs::WorldExec;
use std::net::SocketAddr;
use std::{
    io::Seek,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use stellarator::struc_con::Joinable;
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
        let (world_tx, world_rx) = flume::unbounded();
        let control = ControlService::new(control_addr, builder_addr, world_tx)?;
        tasks.spawn(control.run().instrument(info_span!("control").or_current()));
        tasks.spawn(async move {
            stellarator::struc_con::stellar(move || run_sim(world_rx, sim_addr))
                .join()
                .await?;
            Ok(())
        });
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

async fn run_sim(world_rx: flume::Receiver<WorldExec>, sim_addr: SocketAddr) -> anyhow::Result<()> {
    let mut cancel = Arc::new(AtomicBool::new(false));
    let client = nox_ecs::nox::Client::cpu()?;
    while let Ok(world_exec) = world_rx.recv_async().await {
        cancel.store(true, Ordering::SeqCst);
        cancel = Arc::new(AtomicBool::new(false));
        let Ok(world_exec) = world_exec.compile(client.clone()) else {
            continue;
        };
        let cancel = cancel.clone();
        let tmpfile = tempfile::tempdir().unwrap().into_path();
        stellarator::spawn(
            nox_ecs::impeller2_server::Server::new(
                elodin_db::Server::new(tmpfile.join("db"), sim_addr).unwrap(),
                world_exec,
            )
            .run_with_cancellation(move || cancel.load(Ordering::SeqCst)),
        );
    }
    Ok(())
}

#[derive(Clone)]
struct ControlService {
    sim_builder_health: HealthClient<Channel>,
    sim_builder: SandboxClient<Channel>,
    address: SocketAddr,
    world_channel: flume::Sender<WorldExec>,
}

impl ControlService {
    fn new(
        address: SocketAddr,
        addr: Uri,
        world_channel: flume::Sender<WorldExec>,
    ) -> anyhow::Result<Self> {
        let channel = builder_channel(addr);
        let sim_builder = SandboxClient::new(channel.clone());
        let sim_builder_health = HealthClient::new(channel);
        Ok(Self {
            sim_builder_health,
            sim_builder,
            address,
            world_channel,
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let health_checker = self.clone();

        let address = self.address;
        let svc = SandboxControlServer::new(self);
        info!(api.addr = ?address, "control api listening");
        let reflection = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(elodin_types::FILE_DESCRIPTOR_SET)
            .build_v1()
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

        self.world_channel.send_async(exec).await?;

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
async fn vsock_connect(
    uri: Uri,
) -> Result<hyper_util::rt::TokioIo<tokio_vsock::VsockStream>, std::io::Error> {
    let cid = uri.host().unwrap().parse::<u32>().unwrap();
    let port = uri.port_u16().unwrap();
    let addr = tokio_vsock::VsockAddr::new(cid, port as u32);
    tracing::debug!(cid, port, "connecting to vsock");
    let vsock_stream = tokio_vsock::VsockStream::connect(addr).await?;
    Ok(hyper_util::rt::TokioIo::new(vsock_stream))
}
