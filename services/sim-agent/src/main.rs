use anyhow::Context;
use conduit::client::MsgPair;
use conduit::server::TcpServer;
use elodin_types::sandbox::{
    self,
    sandbox_control_server::{SandboxControl, SandboxControlServer},
    UpdateCodeReq, UpdateCodeResp,
};
use nox::Client;
use nox_ecs::{ConduitExec, WorldExec};
use std::{
    io::{Seek, Write},
    net::SocketAddr,
    thread,
    time::Instant,
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
        let sim_runner = SimRunner::new(server_rx);
        let control = ControlService::new(sim_runner, sandbox_config.control_addr);
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
    sim_runner: SimRunner,
    address: SocketAddr,
}

impl ControlService {
    fn new(sim_runner: SimRunner, address: SocketAddr) -> Self {
        Self {
            sim_runner,
            address,
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

        self.sim_runner.update(exec)?;

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
