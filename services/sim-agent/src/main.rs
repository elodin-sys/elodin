use elodin_conduit::client::MsgPair;
use elodin_conduit::server::TcpServer;
use elodin_types::sandbox::{
    self,
    sandbox_control_server::{SandboxControl, SandboxControlServer},
    UpdateCodeReq, UpdateCodeResp,
};
use nox::Client;
use nox_ecs::ConduitExec;
use std::{
    io::{Seek, Write},
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};
use tonic::{async_trait, transport::Server, Response, Status};
use tracing::{error, info, info_span, Instrument};

mod config;
mod headless;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let config = config::Config::new()?;
    let mut tasks = tokio::task::JoinSet::new();

    if let Some(sandbox_config) = config.sandbox {
        let (server_tx, server_rx) = flume::unbounded();
        let control = ControlService::new(server_rx, sandbox_config.control_addr);
        tasks.spawn(control.run().instrument(info_span!("control").or_current()));
        let sim = async move {
            let sim = TcpServer::bind(server_tx, sandbox_config.sim_addr).await?;
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
    server_rx: flume::Receiver<MsgPair>,
    loaded: Arc<AtomicBool>,
    address: SocketAddr,
    client: Client,
}

impl ControlService {
    fn new(server_rx: flume::Receiver<MsgPair>, address: SocketAddr) -> Self {
        Self {
            address,
            server_rx,
            loaded: Arc::new(AtomicBool::new(false)),
            client: Client::cpu().unwrap(),
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

        let span = info_span!("grpc");
        Server::builder()
            .trace_fn(move |_| span.clone())
            .add_service(svc)
            .add_service(reflection)
            .serve(address)
            .await?;
        Ok(())
    }

    fn handle_update_code(&self, req: UpdateCodeReq) -> anyhow::Result<UpdateCodeResp> {
        let loaded = self.loaded.clone();

        let mut code = tempfile::NamedTempFile::new()?;
        tracing::debug!(tmp_file = %code.path().display(), "writing code to temp file");
        code.write_all(req.code.as_bytes())?;
        code.rewind()?;

        let tmp_dir = tempfile::tempdir()?;
        tracing::debug!(tmp_dir = %tmp_dir.path().display(), "building artifacts");
        let status = std::process::Command::new("python3")
            .arg(code.path())
            .arg("--")
            .arg("build")
            .arg("--dir")
            .arg(tmp_dir.path())
            .spawn()?
            .wait()?;

        if !status.success() {
            return Ok(UpdateCodeResp {
                status: sandbox::Status::Error.into(),
                errors: vec![status.to_string()],
            });
        }

        let exec = match nox_ecs::WorldExec::read_from_dir(tmp_dir.path()) {
            Ok(exec) => exec,
            Err(err) => {
                return Ok(UpdateCodeResp {
                    status: sandbox::Status::Error.into(),
                    errors: vec![err.to_string()],
                });
            }
        };

        let rx = self.server_rx.clone();
        let client = self.client.clone();
        let mut conduit_exec = ConduitExec::new(exec, rx);

        let span = info_span!("sim");
        thread::spawn(move || -> anyhow::Result<()> {
            let _guard = span.enter();
            if loaded.swap(true, Ordering::SeqCst) {
                // TODO: implement update handling
                return Ok(());
            }
            let tick_period = Duration::from_secs_f64(1.0 / 60.0);
            loop {
                let start = Instant::now();
                if let Err(err) = conduit_exec.run(&client) {
                    error!(?err, "failed to run conduit exec");
                    return Err(err.into());
                }
                let sleep_time = tick_period.saturating_sub(start.elapsed());
                if !sleep_time.is_zero() {
                    thread::sleep(sleep_time)
                }
            }
        });

        Ok(UpdateCodeResp {
            status: sandbox::Status::Success.into(),
            errors: vec![],
        })
    }
}

#[async_trait]
impl SandboxControl for ControlService {
    async fn update_code(
        &self,
        req: tonic::Request<UpdateCodeReq>,
    ) -> Result<Response<UpdateCodeResp>, Status> {
        let req = req.into_inner();
        match self.handle_update_code(req) {
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
