use cargo_metadata::camino::Utf8PathBuf;
use core::iter;
use futures::{
    future::{maybe_done, BoxFuture},
    pin_mut, FutureExt,
};
use miette::Diagnostic;
use nu_ansi_term::{Color, Style};
use std::{
    collections::HashMap,
    env,
    io::{self, stdout, Write},
    path::PathBuf,
    process::Stdio,
    time::Duration,
};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, BufReader},
    process::Command,
    task::JoinSet,
};
use tokio_util::sync::CancellationToken;

use crate::{error::Error, watch::watch};

pub const DEFAULT_WATCH_TIMEOUT: Duration = Duration::from_millis(200);

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
#[derive(Debug, Clone)]
pub enum Recipe {
    Cargo(CargoRecipe),
    Process(ProcessRecipe),
    Group(GroupRecipe),
    #[cfg(not(target_os = "windows"))]
    Sim(crate::sim::SimRecipe),
}

impl Recipe {
    pub fn run(
        self,
        name: String,
        release: bool,
        cancel_token: CancellationToken,
    ) -> BoxFuture<'static, Result<(), Error>> {
        match self {
            Recipe::Cargo(c) => c.run(name, release, cancel_token).boxed(),
            Recipe::Process(p) => p.run(name, cancel_token).boxed(),
            Recipe::Group(g) => g.run(release, cancel_token).boxed(),
            #[cfg(not(target_os = "windows"))]
            Recipe::Sim(s) => s.run(cancel_token).boxed(),
        }
    }

    pub fn watch(
        self,
        name: String,
        release: bool,
        cancel_token: CancellationToken,
    ) -> BoxFuture<'static, Result<(), Error>> {
        match self {
            Recipe::Cargo(c) => c.watch(name, release, cancel_token).boxed(),
            Recipe::Process(p) => p.watch(name, cancel_token).boxed(),
            Recipe::Group(g) => g.watch(release, cancel_token).boxed(),
            #[cfg(not(target_os = "windows"))]
            Recipe::Sim(s) => s.watch(cancel_token).boxed(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct GroupRecipe {
    #[serde(default)]
    pub refs: Vec<String>,
    #[serde(default)]
    pub recipes: HashMap<String, Recipe>,
}

impl GroupRecipe {
    async fn run(self, release: bool, cancel_token: CancellationToken) -> Result<(), Error> {
        let mut recipes: JoinSet<_> = self
            .recipes
            .into_iter()
            .map(|(name, r)| {
                let token = cancel_token.clone();
                Ok(r.run(name, release, token))
            })
            .collect::<Result<_, Error>>()?;

        if let Some(res) = recipes.join_next().await {
            res.unwrap()?;
        }
        Ok(())
    }

    async fn watch(self, release: bool, cancel_token: CancellationToken) -> Result<(), Error> {
        let mut recipes: JoinSet<_> = self
            .recipes
            .into_iter()
            .map(|(name, r)| {
                let token = cancel_token.clone();
                Ok(r.watch(name, release, token))
            })
            .collect::<Result<_, Error>>()?;

        if let Some(res) = recipes.join_next().await {
            res.unwrap()?;
        }
        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ProcessRecipe {
    pub cmd: String,
    #[serde(flatten)]
    pub process_args: ProcessArgs,
    pub no_watch: bool,
}

impl ProcessRecipe {
    pub async fn run(self, name: String, cancel_token: CancellationToken) -> Result<(), Error> {
        self.process_args.run(name, self.cmd, cancel_token).await?;
        Ok(())
    }

    pub async fn watch(self, name: String, cancel_token: CancellationToken) -> Result<(), Error> {
        if self.no_watch {
            return self.run(name, cancel_token).await;
        }
        let dirs = self.process_args.watch_dirs();
        watch(
            DEFAULT_WATCH_TIMEOUT,
            |token| self.clone().run(name.clone(), token),
            cancel_token,
            dirs,
        )
        .await
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CargoRecipe {
    pub path: PathBuf,
    pub package: Option<String>,
    pub bin: Option<String>,
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(flatten)]
    pub process_args: ProcessArgs,
    #[serde(default)]
    pub destination: Destination,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ProcessArgs {
    #[serde(default)]
    pub args: Vec<String>,
    pub cwd: Option<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub restart_policy: RestartPolicy,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "kebab-case")]
pub enum RestartPolicy {
    Never,
    #[default]
    Instant,
}

impl ProcessArgs {
    pub async fn run(
        self,
        name: String,
        cmd: String,
        cancel_token: CancellationToken,
    ) -> Result<(), ProcessError> {
        loop {
            let mut child = Command::new(&cmd);
            child
                .args(self.args.iter())
                .envs(self.env.iter())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            if let Some(cwd) = &self.cwd {
                child.current_dir(cwd);
            }
            let mut child = child.spawn().map_err(|source| ProcessError::Spawn {
                source,
                cmd: cmd.clone(),
                recipe: name.clone(),
            })?;
            let stdout = child
                .stdout
                .take()
                .ok_or(ProcessError::ProcessMissingStdout)?;
            let stderr = child
                .stderr
                .take()
                .ok_or(ProcessError::ProcessMissingStderr)?;

            let stdout_name = name.clone();
            let stderr_name = name.clone();
            tokio::spawn(async move { print_logs(stdout, &stdout_name, Color::Blue).await });
            tokio::spawn(async move { print_logs(stderr, &stderr_name, Color::Red).await });
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    child.kill().await?;
                    return Ok(())
                }
                res = child.wait() => {
                    let status = res?;
                    if let Some(code) = status.code() {
                        let color = if code == 0 {
                            Color::Green
                        }else{
                            Color::Red
                        };
                        let style = Style::new().bold().fg(color);
                        println!("{} killed with code {}", style.paint(&name), style.paint(code.to_string()))
                    }else{
                        println!("{} killed by signal", Style::new().bold().paint(&name))
                    }
                    match self.restart_policy {
                        RestartPolicy::Never => {
                            return Ok(())
                        }
                        RestartPolicy::Instant => {
                            continue;
                        }
                    }
                }
            }
        }
    }

    pub fn watch_dirs(&self) -> impl Iterator<Item = PathBuf> {
        self.cwd.clone().into_iter().map(PathBuf::from)
    }
}

#[derive(Error, Diagnostic, Debug)]
pub enum ProcessError {
    #[error("error while running `{cmd}` for recipe \"{recipe}\"")]
    #[diagnostic(
        help = "try checking the cmd and args for {recipe}",
        code = "elodin::recipe_spawn_error"
    )]
    Spawn {
        source: io::Error,
        cmd: String,
        recipe: String,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("process missing stdout")]
    ProcessMissingStdout,
    #[error("process missing stderr")]
    ProcessMissingStderr,
}

async fn print_logs(
    input: impl AsyncRead + Unpin,
    proc_name: &str,
    color: nu_ansi_term::Color,
) -> io::Result<()> {
    let color = Style::default().fg(color).bold();
    let mut buf_reader = BufReader::new(input);
    let mut line = String::new();
    loop {
        let read_fut = maybe_done(buf_reader.read_line(&mut line));
        pin_mut!(read_fut);
        if read_fut.as_mut().now_or_never().is_none() {
            stdout().flush()?;
            read_fut.as_mut().await
        }
        let Some(res) = read_fut.take_output() else {
            return Err(io::Error::other("read_fut did not a return an output"));
        };
        if res? == 0 {
            break;
        }
        writeln!(stdout(), "{} {}", color.paint(proc_name), line.trim_end())?;

        line.clear();
    }
    Ok(())
}

fn cargo_path() -> PathBuf {
    env::var("CARGO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("cargo"))
}

impl CargoRecipe {
    pub async fn build(
        &mut self,
        release: bool,
        cancel_token: CancellationToken,
    ) -> Result<Utf8PathBuf, Error> {
        let mut cmd = Command::new(cargo_path());
        let manifest_path = if self.path.is_dir() {
            self.path.join("Cargo.toml")
        } else {
            self.path.clone()
        };
        let metadata = cargo_metadata::MetadataCommand::new()
            .manifest_path(&manifest_path)
            .exec()?;
        let Some(package_name) = self
            .package
            .as_ref()
            .or_else(|| metadata.root_package().map(|p| &p.name))
        else {
            return Err(Error::MustSelectPackage);
        };
        cmd.arg("build")
            .arg("--manifest-path")
            .arg(&manifest_path)
            .stderr(Stdio::inherit())
            .stdout(Stdio::inherit());
        if let Some(package_name) = &self.package {
            cmd.args(["--package", package_name]);
        }
        if let Some(bin_name) = &self.bin {
            cmd.args(["--bin", bin_name]);
        }
        if release {
            cmd.arg("--release");
        }
        let mut child = cmd.spawn()?;
        let status = tokio::select! {
             _ = cancel_token.cancelled() => {
                child.kill().await?;
                return Err(Error::PackageBuild(package_name.to_string()));
            }
            res = child.wait() => {
                res?
            }
        };
        if status.code() != Some(0) {
            return Err(Error::PackageBuild(package_name.to_string()));
        }
        let target_dir = &metadata.target_directory;
        let bin_dir = if release {
            target_dir.join("release")
        } else {
            target_dir.join("debug")
        };
        Ok(bin_dir.join(package_name))
    }

    pub async fn run(
        mut self,
        name: String,
        release: bool,
        cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        let path = self.build(release, cancel_token.clone()).await?;
        self.process_args
            .run(name, path.to_string(), cancel_token)
            .await?;
        Ok(())
    }

    pub async fn watch(
        self,
        name: String,
        release: bool,
        cancel_token: CancellationToken,
    ) -> Result<(), Error> {
        let dirs = self.watch_dirs();

        watch(
            DEFAULT_WATCH_TIMEOUT,
            |token| self.clone().run(name.clone(), release, token),
            cancel_token,
            dirs,
        )
        .await
    }

    pub fn watch_dirs(&self) -> impl Iterator<Item = PathBuf> {
        iter::once(if self.path.is_dir() {
            self.path.clone()
        } else {
            let path = std::fs::canonicalize(&self.path).expect("failed to canoncilize path");
            path.parent()
                .expect("manifest path didn't have a parent")
                .to_path_buf()
        })
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
#[derive(Debug, Clone, Default)]
pub enum Destination {
    #[default]
    Local,
}
