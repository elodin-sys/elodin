use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowResized};
use core::fmt;
use elodin_editor::EditorPlugin;
use miette::{IntoDiagnostic, miette};
use std::io::{Read, Seek, Write};
use std::net::{Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::thread::JoinHandle;
#[cfg(not(target_os = "windows"))]
use std::time::Duration;
use stellarator::util::CancelToken;
use tokio::runtime::Runtime;

use super::Cli;

const DEFAULT_SIM: Simulator = Simulator::None;

fn default_sim_addr() -> SocketAddr {
    SocketAddr::new(Ipv6Addr::UNSPECIFIED.into(), 2240)
}

#[derive(clap::Args, Clone)]
#[command(
    after_help = "Environment:\n  BLOCKADE_API_KEY    Optional. Enables Skybox AI generation from the command palette. Get one from https://skybox.blockadelabs.com/api and keep it out of source control."
)]
pub struct Args {
    #[clap(name = "addr/path", default_value_t = DEFAULT_SIM)]
    sim: Simulator,

    /// Address to use when launching a Python simulation or serving a database directory.
    /// Assets use its port + 1. Existing s10.toml plans control their own addresses.
    #[clap(long, default_value = "[::]:2240")]
    addr: SocketAddr,

    /// Open this KDL schematic file after connecting to the database.
    #[clap(long)]
    pub kdl: Option<PathBuf>,

    /// Replay recorded data as if it were streaming in real time. The timeline
    /// reveals data progressively as the playback marker advances, simulating
    /// a live session.
    #[clap(long)]
    pub replay: bool,
}

#[derive(clap::Args, Clone)]
pub struct RenderServerArgs {
    /// Address of the Elodin DB to connect to.
    #[clap(long, default_value = "[::]:2240")]
    pub addr: SocketAddr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Simulator {
    None,
    Addr(SocketAddr),
    File(PathBuf),
    Db(PathBuf),
}

#[derive(Resource)]
struct WindowStateFile(std::fs::File);

impl Default for Args {
    fn default() -> Self {
        Self {
            sim: DEFAULT_SIM,
            addr: default_sim_addr(),
            kdl: None,
            replay: false,
        }
    }
}

impl Default for Simulator {
    fn default() -> Self {
        DEFAULT_SIM
    }
}

#[cfg(not(target_os = "windows"))]
fn recipe_sim_addr(recipe: &s10::Recipe) -> Option<SocketAddr> {
    match recipe {
        s10::Recipe::Sim(sim) => Some(sim.addr),
        s10::Recipe::Group(group) => group.recipes.values().find_map(recipe_sim_addr),
        _ => None,
    }
}

#[cfg(not(target_os = "windows"))]
fn use_plan_addr(args: &mut Args) -> miette::Result<()> {
    let Simulator::File(path) = &args.sim else {
        return Ok(());
    };
    if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
        return Ok(());
    }
    if args.addr != default_sim_addr() {
        return Err(miette!(
            "--addr cannot override an existing s10.toml plan; edit the plan instead"
        ));
    }

    let contents = std::fs::read_to_string(path).into_diagnostic()?;
    let recipe: s10::Recipe = toml::from_str(&contents).into_diagnostic()?;
    if let Some(addr) = recipe_sim_addr(&recipe) {
        args.addr = addr;
    }
    Ok(())
}

impl fmt::Display for Simulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, ""),
            Self::Addr(addr) => write!(f, "{}", addr),
            Self::File(path) => write!(f, "{}", path.display()),
            Self::Db(path) => write!(f, "{}", path.display()),
        }
    }
}

impl std::str::FromStr for Simulator {
    type Err = miette::Error;
    fn from_str(s: &str) -> miette::Result<Self> {
        if s.is_empty() {
            return Ok(Self::None);
        }
        if let Ok(addr) = s.parse() {
            Ok(Self::Addr(addr))
        } else {
            let path = PathBuf::from(s);
            if path.is_dir() {
                if path.join("db_state").exists() {
                    Ok(Self::Db(path))
                } else if path.join("s10.toml").is_file() {
                    Ok(Self::File(path.join("s10.toml")))
                } else if path.join("main.py").is_file() {
                    Ok(Self::File(path.join("main.py")))
                } else {
                    Err(miette!(
                        "directory {} is not an Elodin database or simulation; \
                         expected db_state, s10.toml, or main.py",
                        path.display()
                    ))
                }
            } else {
                Ok(Self::File(path))
            }
        }
    }
}

impl Cli {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub fn run_sim(
        &self,
        args: &Args,
        rt: Runtime,
        cancel_token: CancelToken,
    ) -> miette::Result<JoinHandle<miette::Result<()>>> {
        let sim = args.sim.clone();
        let sim_addr = args.addr;
        let dirs = self.dirs().into_diagnostic()?;
        let cache_dir = dirs.cache_dir().to_owned();
        let thread = std::thread::spawn(move || {
            rt.block_on(async move {
                let cancel_on_ctrl_c = {
                    let cancel_token = cancel_token.clone();
                    async move {
                        match tokio::signal::ctrl_c().await {
                            Ok(()) => {
                                info!("Received Ctrl-C, shutting down");
                                cancel_token.cancel();
                                tokio::time::sleep(Duration::from_millis(2000)).await;
                                std::process::exit(130);
                            }
                            Err(err) => {
                                warn!(?err, "failed to listen for Ctrl-C");
                            }
                        }
                    }
                };

                match &sim {
                    Simulator::File(path) => {
                        let mut res = None;
                        let mut recipe_fut = Box::pin(elodin_editor::run::run_recipe_at(
                            cache_dir,
                            path.clone(),
                            cancel_token.clone(),
                            sim_addr,
                        ));
                        tokio::select! {
                            r = &mut recipe_fut => res = Some(r),
                            _ = cancel_on_ctrl_c => {
                                cancel_token.cancel();
                            }
                        }
                        if res.is_none() {
                            res = Some(recipe_fut.await);
                        }
                        cancel_token.cancel();
                        res.expect("run_recipe result missing")
                    }
                    _ => {
                        tokio::select! {
                            _ = cancel_on_ctrl_c => Ok(()),
                            _ = cancel_token.wait() => Ok(()),
                        }
                    }
                }
            })
        });
        Ok(thread)
    }

    #[cfg(target_os = "windows")]
    pub fn run_sim(
        &self,
        _args: &Args,
        rt: Runtime,
        cancel_token: CancelToken,
    ) -> miette::Result<JoinHandle<miette::Result<()>>> {
        Ok(std::thread::spawn(move || {
            rt.block_on(async move {
                wait_for_shutdown(cancel_token, tokio::signal::ctrl_c()).await;
                Ok(())
            })
        }))
    }

    #[cfg_attr(target_os = "windows", allow(unused_mut))]
    pub fn editor(self, mut args: Args, rt: Runtime) -> miette::Result<()> {
        #[cfg(not(target_os = "windows"))]
        use_plan_addr(&mut args)?;

        let cancel_token = CancelToken::new();
        let db_server = match &args.sim {
            Simulator::Db(path) => Some(super::db::serve(
                path.clone(),
                args.addr,
                cancel_token.clone(),
            )?),
            _ => None,
        };
        let thread = self.run_sim(&args, rt, cancel_token.clone())?;
        let mut app = self.editor_app()?;
        match args.sim {
            Simulator::None => {
                app.add_plugins(impeller2_bevy::TcpImpellerPlugin::new(None));
            }
            Simulator::Addr(addr) => {
                app.add_plugins(impeller2_bevy::TcpImpellerPlugin::new(Some(addr)));
            }
            Simulator::File(_) | Simulator::Db(_) => {
                app.add_plugins(impeller2_bevy::TcpImpellerPlugin::new(Some(args.addr)));
            }
        };
        app.insert_resource(BevyCancelToken(cancel_token.clone()))
            .add_systems(Update, check_cancel_token);
        if args.replay {
            app.init_resource::<elodin_editor::ReplayMode>();
        }
        if let Some(path) = &args.kdl {
            app.insert_resource(elodin_editor::ui::schematic::InitialKdlPath(Some(
                path.clone(),
            )));
        }
        app.run();
        cancel_token.cancel();
        let sim_result = thread
            .join()
            .map_err(|_| miette!("simulation thread panicked"))
            .and_then(|result| result);
        let db_result = match db_server {
            Some(server) => server.join(),
            None => Ok(()),
        };
        sim_result?;
        db_result
    }

    /// Run a simulation in headless mode. The render-server (if sensor cameras
    /// are configured) is started as a separate s10-managed process — see the
    /// auto-registered recipe in `world_builder.rs`.
    #[cfg(not(target_os = "windows"))]
    pub fn run_headless(self, mut args: Args, rt: Runtime) -> miette::Result<()> {
        use_plan_addr(&mut args)?;

        let cancel_token = CancelToken::new();
        let db_server = match &args.sim {
            Simulator::Db(path) => Some(super::db::serve(
                path.clone(),
                args.addr,
                cancel_token.clone(),
            )?),
            _ => None,
        };
        let thread = self.run_sim(&args, rt, cancel_token.clone())?;
        let result = thread
            .join()
            .map_err(|_| miette!("simulation thread panicked"))
            .and_then(|result| result);
        cancel_token.cancel();
        let db_result = match db_server {
            Some(server) => server.join(),
            None => Ok(()),
        };
        result?;
        db_result
    }

    /// Start the headless sensor camera render server. This is spawned as an
    /// s10-managed child process — not called directly by users.
    #[cfg(not(target_os = "windows"))]
    pub fn render_server(self, args: RenderServerArgs) -> miette::Result<()> {
        let mut app = App::new();
        app.add_plugins(elodin_editor::headless::HeadlessEditorPlugin);
        app.add_plugins(impeller2_bevy::TcpImpellerPlugin::new(Some(args.addr)));
        app.run();
        Ok(())
    }

    pub fn editor_app(&self) -> miette::Result<App> {
        let mut window_state_file = self.window_state_file()?;
        let mut window_state = String::new();
        window_state_file
            .read_to_string(&mut window_state)
            .into_diagnostic()?;
        let editor_plugin = if let [width, height] = window_state
            .split_whitespace()
            .collect::<Vec<_>>()
            .as_slice()
        {
            let width = width.parse::<f32>().into_diagnostic()?;
            let height = height.parse::<f32>().into_diagnostic()?;
            EditorPlugin::new(width, height)
        } else {
            EditorPlugin::default()
        };

        let mut app = App::new();
        app.insert_resource(WindowStateFile(window_state_file))
            .add_plugins(editor_plugin)
            .add_systems(Update, on_window_resize);
        Ok(app)
    }

    fn window_state_file(&self) -> miette::Result<std::fs::File> {
        use miette::Context;
        let dirs = self.dirs().into_diagnostic()?;
        let data_dir = dirs.data_dir();
        std::fs::create_dir_all(data_dir)
            .into_diagnostic()
            .context("failed to create data directory")?;
        let window_state_path = data_dir.join(".window-state");
        std::fs::File::options()
            .write(true)
            .read(true)
            .create(true)
            .truncate(false)
            .open(window_state_path)
            .into_diagnostic()
            .context("failed to open window state file")
    }
}

#[cfg(any(target_os = "windows", test))]
async fn wait_for_shutdown(
    cancel_token: CancelToken,
    ctrl_c: impl std::future::Future<Output = std::io::Result<()>>,
) {
    tokio::select! {
        result = ctrl_c => {
            match result {
                Ok(()) => {
                    info!("Received Ctrl-C, shutting down");
                    cancel_token.cancel();
                }
                Err(err) => {
                    warn!(?err, "failed to listen for Ctrl-C");
                }
            }
        }
        _ = cancel_token.wait() => {}
    }
}

#[derive(Resource)]
struct BevyCancelToken(CancelToken);

fn check_cancel_token(token: Res<BevyCancelToken>, mut exit: MessageWriter<AppExit>) {
    if token.0.is_cancelled() {
        exit.write(AppExit::Success);
    }
}

fn on_window_resize(
    mut window_state_file: ResMut<WindowStateFile>,
    mut resize_reader: MessageReader<WindowResized>,
    query: Query<Entity, With<PrimaryWindow>>,
) {
    if let Some(e) = resize_reader.read().last() {
        if query.get(e.window).is_err() {
            return;
        }
        let window_state = format!("{:.1} {:.1}\n", e.width, e.height);
        if let Err(err) = window_state_file.0.rewind() {
            warn!(?err, "failed to rewind window state file");
            return;
        }
        if let Err(err) = window_state_file.0.write_all(window_state.as_bytes()) {
            warn!(?err, "failed to write window state");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn parses_socket_address() {
        let addr = "127.0.0.1:2240".parse().unwrap();
        assert_eq!(
            Simulator::from_str("127.0.0.1:2240").unwrap(),
            Simulator::Addr(addr)
        );
    }

    #[test]
    fn parses_file_path() {
        assert_eq!(
            Simulator::from_str("examples/drone/main.py").unwrap(),
            Simulator::File(PathBuf::from("examples/drone/main.py"))
        );
    }

    #[test]
    fn recognizes_database_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("db_state"), []).unwrap();
        assert_eq!(
            Simulator::from_str(dir.path().to_str().unwrap()).unwrap(),
            Simulator::Db(dir.path().to_path_buf())
        );
    }

    #[test]
    fn resolves_plan_from_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("s10.toml"), []).unwrap();
        assert_eq!(
            Simulator::from_str(dir.path().to_str().unwrap()).unwrap(),
            Simulator::File(dir.path().join("s10.toml"))
        );
    }

    #[test]
    fn resolves_python_entrypoint_from_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("main.py"), []).unwrap();
        assert_eq!(
            Simulator::from_str(dir.path().to_str().unwrap()).unwrap(),
            Simulator::File(dir.path().join("main.py"))
        );
    }

    #[test]
    fn rejects_unrecognized_directory() {
        let dir = tempfile::tempdir().unwrap();
        let error = Simulator::from_str(dir.path().to_str().unwrap()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("expected db_state, s10.toml, or main.py")
        );
    }

    #[tokio::test]
    async fn shutdown_waits_for_cancellation() {
        let cancel = CancelToken::new();
        let handle = tokio::spawn(wait_for_shutdown(
            cancel.clone(),
            std::future::pending::<std::io::Result<()>>(),
        ));
        tokio::task::yield_now().await;
        assert!(!handle.is_finished());
        assert!(!cancel.is_cancelled());

        cancel.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), handle)
            .await
            .expect("shutdown waiter did not observe cancellation")
            .expect("shutdown waiter panicked");
    }

    #[tokio::test]
    async fn shutdown_signal_cancels_token() {
        let cancel = CancelToken::new();
        wait_for_shutdown(cancel.clone(), std::future::ready(Ok(()))).await;
        assert!(cancel.is_cancelled());
    }
}
