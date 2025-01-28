use bevy::window::WindowResized;
use bevy::{prelude::*, utils::tracing};
use core::fmt;
use elodin_editor::EditorPlugin;
use miette::{miette, Context, IntoDiagnostic};
use std::io::{Read, Seek, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::path::PathBuf;
use std::thread::JoinHandle;
use stellarator::util::CancelToken;
use tokio::runtime::Runtime;

use super::Cli;

const DEFAULT_SIM: Simulator =
    Simulator::Addr(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 2240)));

#[derive(clap::Args, Clone, Default)]
pub struct Args {
    #[clap(name = "addr/path", default_value_t = DEFAULT_SIM)]
    sim: Simulator,
}

#[derive(Clone)]
enum Simulator {
    Addr(SocketAddr),
    File(PathBuf),
    ReplayDir(PathBuf),
}

#[derive(Resource)]
struct WindowStateFile(std::fs::File);

impl Default for Simulator {
    fn default() -> Self {
        DEFAULT_SIM
    }
}

impl fmt::Display for Simulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Addr(addr) => write!(f, "{}", addr),
            Self::File(path) => write!(f, "{}", path.display()),
            Self::ReplayDir(path) => write!(f, "{}", path.display()),
        }
    }
}

impl std::str::FromStr for Simulator {
    type Err = miette::Error;
    fn from_str(s: &str) -> miette::Result<Self> {
        if let Ok(addr) = s.parse() {
            Ok(Self::Addr(addr))
        } else {
            let path = PathBuf::from(s);
            if path.is_dir() {
                Ok(Self::ReplayDir(path))
            } else {
                Ok(Self::File(path))
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
async fn run_recipe(
    cache_dir: PathBuf,
    path: PathBuf,
    cancel_token: CancelToken,
) -> miette::Result<()> {
    let mut path = if path.is_dir() {
        let toml = path.join("s10.toml");
        let py = path.join("main.py");
        if path.join("s10.toml").exists() {
            toml
        } else if py.exists() {
            py
        } else {
            return Err(miette!("couldn't find a elodin config, please add either a main.py or s10.toml file to the directory"));
        }
    } else {
        path.clone()
    };

    // If python file, generate s10.toml
    if path.extension().and_then(|ext| ext.to_str()) == Some("py") {
        let date_time = chrono::Local::now().to_rfc3339();
        let out_dir = cache_dir.join(date_time);
        std::fs::create_dir_all(&out_dir).into_diagnostic()?;
        let output = s10::python_command()?
            .arg(path.clone())
            .arg("plan")
            .arg(&out_dir)
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .output()
            .into_diagnostic()?;
        if output.status.code() != Some(0) {
            return Err(miette!("error generating s10 plan from python file"));
        }
        path = out_dir.join("s10.toml");
        tracing::info!("Generated s10 plan: {}", path.display());
    }

    // If not a s10 plan file, bail out
    if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
        return Err(miette!(
            "invalid file type, must be a s10 plan or python file"
        ));
    }

    let contents = std::fs::read_to_string(&path).into_diagnostic()?;
    let recipe: s10::Recipe = toml::from_str(&contents)
        .into_diagnostic()
        .with_context(|| format!("failed to parse s10 recipe from file: {}", path.display()))?;

    recipe
        .watch("sim".to_string(), false, cancel_token.clone())
        .await?;
    cancel_token.cancel();
    Ok(())
}

impl Cli {
    #[cfg(not(target_os = "windows"))]
    pub fn run_sim(
        &self,
        args: &Args,
        rt: Runtime,
        cancel_token: CancelToken,
    ) -> miette::Result<JoinHandle<miette::Result<()>>> {
        let sim = args.sim.clone();
        let dirs = self.dirs().into_diagnostic()?;
        let cache_dir = dirs.cache_dir().to_owned();
        let thread = std::thread::spawn(move || {
            rt.block_on(async move {
                let ctrl_c_cancel_token = cancel_token.clone();
                tokio::spawn(async move {
                    let _drop = ctrl_c_cancel_token.drop_guard(); // binding needs to be named to ensure drop is called at end of scope
                    tokio::signal::ctrl_c().await.unwrap();
                    tracing::info!("Received Ctrl-C, shutting down");
                });
                if let Simulator::File(path) = &sim {
                    let res = run_recipe(cache_dir, path.clone(), cancel_token.clone()).await;
                    cancel_token.cancel();
                    res
                } else {
                    Ok(())
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
                tokio::spawn(async move {
                    let _drop = cancel_token.drop_guard(); // binding needs to be named to ensure drop is called at end of scope
                    tokio::signal::ctrl_c().await
                });
                Ok(())
            })
        }))
    }

    pub fn editor(self, args: Args, rt: Runtime) -> miette::Result<()> {
        let cancel_token = CancelToken::new();
        let thread = self.run_sim(&args, rt, cancel_token.clone())?;
        let mut app = self.editor_app()?;
        match args.sim {
            Simulator::Addr(addr) => {
                app.add_plugins(impeller2_bevy::TcpImpellerPlugin::new(addr));
            }
            Simulator::File(_) => {
                app.add_plugins(impeller2_bevy::TcpImpellerPlugin::new(SocketAddr::new(
                    [127, 0, 0, 1].into(),
                    2240,
                )));
            }
            Simulator::ReplayDir(_) => {
                // TODO
            }
        };
        app.insert_resource(BevyCancelToken(cancel_token.clone()))
            .add_systems(Update, check_cancel_token);
        app.run();
        cancel_token.cancel();
        thread.join().map_err(|_| miette!("join error"))?
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

#[derive(Resource)]
struct BevyCancelToken(CancelToken);

fn check_cancel_token(token: Res<BevyCancelToken>, mut exit: EventWriter<AppExit>) {
    if token.0.is_cancelled() {
        exit.send(AppExit::Success);
    }
}

fn on_window_resize(
    mut window_state_file: ResMut<WindowStateFile>,
    mut resize_reader: EventReader<WindowResized>,
) {
    if let Some(e) = resize_reader.read().last() {
        let window_state = format!("{:.1} {:.1}\n", e.width, e.height);
        if let Err(err) = window_state_file.0.rewind() {
            tracing::warn!(?err, "failed to rewind window state file");
            return;
        }
        if let Err(err) = window_state_file.0.write_all(window_state.as_bytes()) {
            tracing::warn!(?err, "failed to write window state");
        }
    }
}
