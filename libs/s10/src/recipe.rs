use cargo_metadata::camino::Utf8PathBuf;
use core::iter;
use futures::{
    FutureExt,
    future::{BoxFuture, maybe_done},
    pin_mut,
};
use miette::Diagnostic;
use nu_ansi_term::{Color, Style};
use std::{
    collections::HashMap,
    env,
    io::{self, Write, stdout},
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};
use stellarator::util::CancelToken;
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader},
    process::Command,
    sync::watch as sync_watch,
    task::JoinSet,
};

use crate::{
    cgroup::CgroupScope,
    error::Error,
    probe::{ReadyProbe, expand_env, parse_duration},
    watch::watch,
};

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

/// Whether a finished group member should tear down its siblings.
///
/// Simulations are leaders: when the sim ends, forever-running FSW sidecars
/// must be cancelled (Monte Carlo, `elodin run`). Successful Process/Cargo
/// exits must not cancel the group — optional sidecars (e.g. video-stream's
/// rtsp-receiver when `RTSP_URL` is unset) may exit 0 without killing the sim.
/// Any `Err` still cancels (fail-fast). Parent `CancelToken` cancellation
/// (editor close, Ctrl-C, MC timeout) is independent of this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecipeRole {
    Sim,
    Sidecar,
}

impl Recipe {
    fn role(&self) -> RecipeRole {
        match self {
            #[cfg(not(target_os = "windows"))]
            Recipe::Sim(_) => RecipeRole::Sim,
            _ => RecipeRole::Sidecar,
        }
    }

    fn depends_on(&self) -> &[String] {
        match self {
            Recipe::Cargo(c) => &c.process_args.depends_on,
            Recipe::Process(p) => &p.process_args.depends_on,
            Recipe::Group(_) => &[],
            #[cfg(not(target_os = "windows"))]
            Recipe::Sim(s) => &s.depends_on,
        }
    }

    fn ready_probe(&self) -> Option<&ReadyProbe> {
        match self {
            Recipe::Cargo(c) => c.process_args.ready.as_ref(),
            Recipe::Process(p) => p.process_args.ready.as_ref(),
            Recipe::Group(_) => None,
            #[cfg(not(target_os = "windows"))]
            Recipe::Sim(s) => s.ready.as_ref(),
        }
    }

    fn ready_timeout(&self) -> Option<&str> {
        match self {
            Recipe::Cargo(c) => c.process_args.ready_timeout.as_deref(),
            Recipe::Process(p) => p.process_args.ready_timeout.as_deref(),
            Recipe::Group(_) => None,
            #[cfg(not(target_os = "windows"))]
            Recipe::Sim(s) => s.ready_timeout.as_deref(),
        }
    }

    fn log_path(&self) -> Option<&Path> {
        match self {
            Recipe::Cargo(c) => c.process_args.log_path.as_deref(),
            Recipe::Process(p) => p.process_args.log_path.as_deref(),
            Recipe::Group(_) => None,
            #[cfg(not(target_os = "windows"))]
            Recipe::Sim(s) => s.log_path.as_deref(),
        }
    }

    fn env(&self) -> Option<&HashMap<String, String>> {
        match self {
            Recipe::Cargo(c) => Some(&c.process_args.env),
            Recipe::Process(p) => Some(&p.process_args.env),
            Recipe::Group(_) => None,
            #[cfg(not(target_os = "windows"))]
            Recipe::Sim(s) => Some(&s.env),
        }
    }

    pub fn run(
        self,
        name: String,
        release: bool,
        cancel_token: CancelToken,
        cgroup: Option<Arc<CgroupScope>>,
    ) -> BoxFuture<'static, Result<(), Error>> {
        match self {
            Recipe::Cargo(c) => c.run(name, release, cancel_token, cgroup).boxed(),
            Recipe::Process(p) => p.run(name, cancel_token, cgroup).boxed(),
            Recipe::Group(g) => g.run(release, cancel_token, cgroup).boxed(),
            #[cfg(not(target_os = "windows"))]
            Recipe::Sim(s) => s.run(cancel_token, cgroup).boxed(),
        }
    }

    pub fn watch(
        self,
        name: String,
        release: bool,
        cancel_token: CancelToken,
        cgroup: Option<Arc<CgroupScope>>,
    ) -> BoxFuture<'static, Result<(), Error>> {
        match self {
            Recipe::Cargo(c) => c.watch(name, release, cancel_token, cgroup).boxed(),
            Recipe::Process(p) => p.watch(name, cancel_token, cgroup).boxed(),
            Recipe::Group(g) => g.watch(release, cancel_token, cgroup).boxed(),
            #[cfg(not(target_os = "windows"))]
            Recipe::Sim(s) => s.watch(cancel_token, cgroup).boxed(),
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
    async fn run(
        self,
        release: bool,
        cancel_token: CancelToken,
        cgroup: Option<Arc<CgroupScope>>,
    ) -> Result<(), Error> {
        // One readiness channel per recipe. Receivers don't keep a channel open,
        // so clone them up front for dependency wiring, then move each Sender into
        // its own task as the sole owner: when a task ends without signaling ready,
        // the channel closes and dependents observe it instead of blocking forever.
        let mut channels = self
            .recipes
            .keys()
            .map(|name| (name.clone(), sync_watch::channel(false)))
            .collect::<HashMap<_, _>>();
        let receivers = channels
            .iter()
            .map(|(name, (_, rx))| (name.clone(), rx.clone()))
            .collect::<HashMap<_, _>>();
        let recipes: JoinSet<_> = self
            .recipes
            .into_iter()
            .map(|(name, r)| {
                let token = cancel_token.clone();
                let (ready_tx, _) = channels
                    .remove(&name)
                    .expect("recipe has a readiness signal");
                let dependencies = r
                    .depends_on()
                    .iter()
                    .map(|dep| {
                        receivers
                            .get(dep)
                            .map(|rx| (dep.clone(), rx.clone()))
                            .ok_or_else(|| Error::UnresolvedRecipe(dep.clone()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(run_with_readiness(
                    name,
                    r,
                    release,
                    token,
                    cgroup.clone(),
                    dependencies,
                    ready_tx,
                ))
            })
            .collect::<Result<_, Error>>()?;

        await_group_members(recipes, cancel_token).await
    }

    async fn watch(
        self,
        release: bool,
        cancel_token: CancelToken,
        cgroup: Option<Arc<CgroupScope>>,
    ) -> Result<(), Error> {
        // One readiness channel per recipe. Receivers don't keep a channel open,
        // so clone them up front for dependency wiring, then move each Sender into
        // its own task as the sole owner: when a task ends without signaling ready,
        // the channel closes and dependents observe it instead of blocking forever.
        let mut channels = self
            .recipes
            .keys()
            .map(|name| (name.clone(), sync_watch::channel(false)))
            .collect::<HashMap<_, _>>();
        let receivers = channels
            .iter()
            .map(|(name, (_, rx))| (name.clone(), rx.clone()))
            .collect::<HashMap<_, _>>();
        let recipes: JoinSet<_> = self
            .recipes
            .into_iter()
            .map(|(name, r)| {
                let token = cancel_token.clone();
                let (ready_tx, _) = channels
                    .remove(&name)
                    .expect("recipe has a readiness signal");
                let dependencies = r
                    .depends_on()
                    .iter()
                    .map(|dep| {
                        receivers
                            .get(dep)
                            .map(|rx| (dep.clone(), rx.clone()))
                            .ok_or_else(|| Error::UnresolvedRecipe(dep.clone()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(watch_with_readiness(
                    name,
                    r,
                    release,
                    token,
                    cgroup.clone(),
                    dependencies,
                    ready_tx,
                ))
            })
            .collect::<Result<_, Error>>()?;

        await_group_members(recipes, cancel_token).await
    }
}

/// Join group members until the group should end.
///
/// - Successful **sidecar** exits leave siblings running.
/// - **Sim** exit or any **Err** cancels the shared token and drains the rest.
async fn await_group_members(
    mut recipes: JoinSet<(RecipeRole, Result<(), Error>)>,
    cancel_token: CancelToken,
) -> Result<(), Error> {
    let mut result = Ok(());
    while let Some(joined) = recipes.join_next().await {
        let (role, res) = joined.unwrap();
        let failed = res.is_err();
        if result.is_ok() {
            result = res;
        }
        if matches!(role, RecipeRole::Sim) || failed {
            cancel_token.cancel();
            while let Some(rest) = recipes.join_next().await {
                let (_, r) = rest.unwrap();
                if result.is_ok() {
                    result = r;
                }
            }
            break;
        }
    }
    result
}

async fn run_with_readiness(
    name: String,
    recipe: Recipe,
    release: bool,
    cancel_token: CancelToken,
    cgroup: Option<Arc<CgroupScope>>,
    dependencies: Vec<(String, sync_watch::Receiver<bool>)>,
    ready_tx: sync_watch::Sender<bool>,
) -> (RecipeRole, Result<(), Error>) {
    // Capture before `prebuild`: Cargo becomes Process, but stays a sidecar.
    let role = recipe.role();
    let result = async {
        let probe = expand_ready_probe(&recipe);
        let ready_timeout = recipe.ready_timeout().map(str::to_string);
        let log_path = recipe.log_path().map(PathBuf::from);
        wait_for_dependencies(&name, dependencies).await?;

        // Compile Cargo recipes before the readiness window opens so time-based
        // probes (e.g. `Ready.delay`) measure from process spawn, not from build
        // start. Otherwise a slow compile can elapse the delay and release
        // `depends_on` dependents while the sidecar is still building.
        let recipe = prebuild(recipe, release, &cancel_token).await?;
        let mut run_fut = recipe.run(name, release, cancel_token, cgroup);
        mark_ready_or_wait(probe, ready_timeout, log_path, ready_tx, &mut run_fut).await
    }
    .await;
    (role, result)
}

/// Performs a recipe's build phase (if any) up front so it does not overlap the
/// readiness probe. A built Cargo recipe becomes a prebuilt `Process` recipe so
/// the subsequent `run` only spawns the binary. Non-Cargo recipes pass through
/// unchanged. Only used on the `run` path; `watch` keeps its own rebuild loop.
async fn prebuild(
    recipe: Recipe,
    release: bool,
    cancel_token: &CancelToken,
) -> Result<Recipe, Error> {
    match recipe {
        Recipe::Cargo(mut cargo) => {
            let bin = cargo.build(release, cancel_token.clone()).await?;
            Ok(Recipe::Process(ProcessRecipe {
                cmd: bin.to_string(),
                process_args: cargo.process_args,
                no_watch: true,
            }))
        }
        other => Ok(other),
    }
}

async fn watch_with_readiness(
    name: String,
    recipe: Recipe,
    release: bool,
    cancel_token: CancelToken,
    cgroup: Option<Arc<CgroupScope>>,
    dependencies: Vec<(String, sync_watch::Receiver<bool>)>,
    ready_tx: sync_watch::Sender<bool>,
) -> (RecipeRole, Result<(), Error>) {
    let role = recipe.role();
    let result = async {
        let probe = expand_ready_probe(&recipe);
        let ready_timeout = recipe.ready_timeout().map(str::to_string);
        let log_path = recipe.log_path().map(PathBuf::from);
        wait_for_dependencies(&name, dependencies).await?;

        let mut run_fut = recipe.watch(name, release, cancel_token, cgroup);
        mark_ready_or_wait(probe, ready_timeout, log_path, ready_tx, &mut run_fut).await
    }
    .await;
    (role, result)
}

/// Resolves a recipe's readiness probe placeholders against its env (overlaid
/// on the inherited environment), so probes can target per-run ports/sockets.
fn expand_ready_probe(recipe: &Recipe) -> Option<ReadyProbe> {
    let probe = recipe.ready_probe()?;
    let env = recipe.env().cloned().unwrap_or_default();
    Some(probe.expand(move |name| env.get(name).cloned().or_else(|| std::env::var(name).ok())))
}

async fn wait_for_dependencies(
    name: &str,
    dependencies: Vec<(String, sync_watch::Receiver<bool>)>,
) -> Result<(), Error> {
    for (dep, mut rx) in dependencies {
        while !*rx.borrow() {
            // A closed channel means the dependency's task ended without ever
            // signaling ready (early exit or readiness failure), so name it
            // rather than emitting a generic "task exited" message.
            rx.changed().await.map_err(|_| {
                Error::Readiness(format!(
                    "dependency \"{dep}\" exited before \"{name}\" became ready"
                ))
            })?;
        }
    }
    Ok(())
}

async fn mark_ready_or_wait(
    probe: Option<ReadyProbe>,
    ready_timeout: Option<String>,
    log_path: Option<PathBuf>,
    ready_tx: sync_watch::Sender<bool>,
    run_fut: &mut BoxFuture<'static, Result<(), Error>>,
) -> Result<(), Error> {
    let Some(probe) = probe else {
        let _ = ready_tx.send(true);
        return run_fut.await;
    };

    let timeout = parse_duration(ready_timeout.as_deref(), Duration::from_secs(30))
        .map_err(|err| Error::Readiness(err.to_string()))?;
    tokio::select! {
        // The child exited before the probe signaled ready: return its result
        // and let `ready_tx` drop, closing the channel so dependents fail fast
        // instead of blocking. Group join policy then decides whether to cancel
        // siblings (Sim exit / Err) or leave them running (sidecar Ok).
        result = &mut *run_fut => result,
        result = probe.wait(log_path.as_deref(), timeout) => {
            result.map_err(|err| Error::Readiness(err.to_string()))?;
            let _ = ready_tx.send(true);
            run_fut.await
        }
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
    pub async fn run(
        self,
        name: String,
        cancel_token: CancelToken,
        cgroup: Option<Arc<CgroupScope>>,
    ) -> Result<(), Error> {
        self.process_args
            .run(name, self.cmd, cancel_token, cgroup)
            .await?;
        Ok(())
    }

    pub async fn watch(
        self,
        name: String,
        cancel_token: CancelToken,
        cgroup: Option<Arc<CgroupScope>>,
    ) -> Result<(), Error> {
        if self.no_watch {
            return self.run(name, cancel_token, cgroup).await;
        }
        let dirs = self.process_args.watch_dirs();
        watch(
            DEFAULT_WATCH_TIMEOUT,
            |token| self.clone().run(name.clone(), token, cgroup.clone()),
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
    #[serde(default)]
    pub fail_on_error: bool,
    #[serde(default)]
    pub log_path: Option<PathBuf>,
    #[serde(default)]
    pub silence: bool,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub ready: Option<ReadyProbe>,
    #[serde(default)]
    pub ready_timeout: Option<String>,
    /// Spawn the child in its own process group and tear the whole group down
    /// (SIGTERM, then SIGKILL) on cancel and after the leader exits. This is
    /// the fallback kill path for orchestrated runs (monte-carlo) on hosts
    /// without a delegated cgroup, where daemonizing grandchildren would
    /// otherwise outlive the recipe and keep its ports bound.
    #[serde(default)]
    pub own_process_group: bool,
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
        cancel_token: CancelToken,
        cgroup: Option<Arc<CgroupScope>>,
    ) -> Result<(), ProcessError> {
        // Resolve `${VAR:-default}` placeholders in args/cwd from the recipe env
        // (overlaid on the inherited environment). This lets a single planned
        // recipe carry per-run values (e.g. monte-carlo worker ports) that are
        // only known when the process is spawned.
        let (args, cwd) = {
            let lookup = |name: &str| {
                self.env
                    .get(name)
                    .cloned()
                    .or_else(|| std::env::var(name).ok())
            };
            let args: Vec<String> = self
                .args
                .iter()
                .map(|arg| expand_env(arg, lookup))
                .collect();
            let cwd = self.cwd.as_ref().map(|cwd| expand_env(cwd, lookup));
            (args, cwd)
        };
        loop {
            let mut child = Command::new(&cmd);
            child.args(args.iter()).envs(self.env.iter());
            if self.silence {
                child.stdout(Stdio::null()).stderr(Stdio::null());
            } else {
                child.stdout(Stdio::piped()).stderr(Stdio::piped());
            }
            if let Some(cwd) = &cwd {
                child.current_dir(cwd);
            }
            #[cfg(unix)]
            if self.own_process_group {
                // Own group + no terminal stdin: stdin(null) prevents SIGTTIN
                // stops when the child is in a background process group.
                child.process_group(0);
                child.stdin(Stdio::null());
            }
            let mut child = child.spawn().map_err(|source| ProcessError::Spawn {
                source,
                cmd: cmd.clone(),
                recipe: name.clone(),
            })?;
            #[cfg(unix)]
            let child_pgid = self
                .own_process_group
                .then(|| child.id().map(|pid| nix::unistd::Pid::from_raw(pid as i32)))
                .flatten();
            if let (Some(scope), Some(pid)) = (&cgroup, child.id()) {
                scope.add_pid(pid);
            }
            if !self.silence {
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
                let stdout_log_path = self.log_path.clone();
                let stderr_log_path = self.log_path.clone();
                tokio::spawn(async move {
                    print_logs(stdout, &stdout_name, Color::Blue, stdout_log_path).await
                });
                tokio::spawn(async move {
                    print_logs(stderr, &stderr_name, Color::Red, stderr_log_path).await
                });
            }
            tokio::select! {
                _ = cancel_token.wait() => {
                    // Graceful shutdown: send SIGTERM, wait up to 2 seconds, then force kill
                    #[cfg(unix)]
                    {
                        if let Some(pgid) = child_pgid {
                            let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGTERM);
                        }
                        if let Some(pid) = child.id() {
                            let _ = nix::sys::signal::kill(
                                nix::unistd::Pid::from_raw(pid as i32),
                                nix::sys::signal::Signal::SIGTERM,
                            );
                        }
                        if let Some(log_path) = &self.log_path {
                            let _ = append_log_line(
                                log_path,
                                &name,
                                "waiting for process to exit gracefully",
                            )
                            .await;
                        } else {
                            tracing::info!("Waiting for {} to exit gracefully", name);
                        }
                        match tokio::time::timeout(Duration::from_secs(2), child.wait()).await {
                            Ok(_) => {}
                            Err(_) => {
                                if let Some(log_path) = &self.log_path {
                                    let _ = append_log_line(
                                        log_path,
                                        &name,
                                        "did not exit after SIGTERM, forcing kill",
                                    )
                                    .await;
                                } else {
                                    tracing::warn!(
                                        "{} did not exit after SIGTERM, forcing kill",
                                        name
                                    );
                                }
                                if let Some(pgid) = child_pgid {
                                    let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
                                }
                                let _ = child.start_kill();
                                let _ = child.wait().await;
                            }
                        }
                        // Reap any group members that survived the leader.
                        if let Some(pgid) = child_pgid {
                            let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        child.kill().await?;
                    }
                    return Ok(())
                }
                res = child.wait() => {
                    let status = res?;
                    // The leader is gone; sweep daemonized group members so they
                    // cannot outlive the recipe and squat on its ports.
                    #[cfg(unix)]
                    if let Some(pgid) = child_pgid {
                        let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
                    }
                    emit_process_status(&self.log_path, &name, &status).await?;
                    match self.restart_policy {
                        RestartPolicy::Never => {
                            if self.fail_on_error && !status.success() {
                                return Err(ProcessError::Exited {
                                    recipe: name,
                                    status,
                                });
                            }
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
    #[error("process `{recipe}` exited unsuccessfully with status {status}")]
    Exited {
        recipe: String,
        status: std::process::ExitStatus,
    },
}

async fn print_logs(
    input: impl AsyncRead + Unpin,
    proc_name: &str,
    color: nu_ansi_term::Color,
    log_path: Option<PathBuf>,
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
        if let Some(log_path) = &log_path {
            append_log_line(log_path, proc_name, line.trim_end()).await?;
        } else {
            writeln!(stdout(), "{} {}", color.paint(proc_name), line.trim_end())?;
        }

        line.clear();
    }
    Ok(())
}

async fn emit_process_status(
    log_path: &Option<PathBuf>,
    name: &str,
    status: &std::process::ExitStatus,
) -> io::Result<()> {
    if let Some(log_path) = log_path {
        let line = if let Some(code) = status.code() {
            format!("killed with code {code}")
        } else {
            "killed by signal".to_string()
        };
        append_log_line(log_path, name, &line).await
    } else {
        if let Some(code) = status.code() {
            let color = if code == 0 { Color::Green } else { Color::Red };
            let style = Style::new().bold().fg(color);
            println!(
                "{} killed with code {}",
                style.paint(name),
                style.paint(code.to_string())
            )
        } else {
            println!("{} killed by signal", Style::new().bold().paint(name))
        }
        Ok(())
    }
}

pub(crate) async fn append_log_line(
    log_path: &Path,
    proc_name: &str,
    line: &str,
) -> io::Result<()> {
    if let Some(parent) = log_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .await?;
    file.write_all(format!("{proc_name} {line}\n").as_bytes())
        .await
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
        cancel_token: CancelToken,
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
             _ = cancel_token.wait() => {
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
        cancel_token: CancelToken,
        cgroup: Option<Arc<CgroupScope>>,
    ) -> Result<(), Error> {
        let path = self.build(release, cancel_token.clone()).await?;
        self.process_args
            .run(name, path.to_string(), cancel_token, cgroup)
            .await?;
        Ok(())
    }

    pub async fn watch(
        self,
        name: String,
        release: bool,
        cancel_token: CancelToken,
        cgroup: Option<Arc<CgroupScope>>,
    ) -> Result<(), Error> {
        let dirs = self.watch_dirs();

        watch(
            DEFAULT_WATCH_TIMEOUT,
            |token| {
                self.clone()
                    .run(name.clone(), release, token, cgroup.clone())
            },
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::time::Duration;

    fn process_recipe(
        cmd: &str,
        args: &[&str],
        fail_on_error: bool,
        ready: Option<ReadyProbe>,
        depends_on: Vec<String>,
    ) -> Recipe {
        Recipe::Process(ProcessRecipe {
            cmd: cmd.to_string(),
            process_args: ProcessArgs {
                args: args.iter().map(|a| a.to_string()).collect(),
                cwd: None,
                env: HashMap::new(),
                restart_policy: RestartPolicy::Never,
                fail_on_error,
                log_path: None,
                silence: false,
                depends_on,
                ready,
                ready_timeout: None,
                own_process_group: false,
            },
            no_watch: false,
        })
    }

    /// A leaf with `own_process_group` must not leak grandchildren after the
    /// leader exits: the group sweep reaps the backgrounded sleeper.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn own_process_group_reaps_grandchildren_on_exit() {
        // Unique sleep duration so pgrep can identify this test's grandchild.
        let secs = 200_000 + (std::process::id() % 100_000);
        let recipe = Recipe::Process(ProcessRecipe {
            cmd: "sh".to_string(),
            process_args: ProcessArgs {
                args: vec!["-c".to_string(), format!("sleep {secs} & true")],
                cwd: None,
                env: HashMap::new(),
                restart_policy: RestartPolicy::Never,
                fail_on_error: false,
                log_path: None,
                silence: true,
                depends_on: vec![],
                ready: None,
                ready_timeout: None,
                own_process_group: true,
            },
            no_watch: false,
        });
        recipe
            .run("pg-test".to_string(), false, CancelToken::new(), None)
            .await
            .expect("recipe run");
        // The backgrounded sleeper was in the leader's process group and must
        // have been swept when the leader exited.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let pattern = format!("^sleep {secs}$");
        let leaked = std::process::Command::new("pgrep")
            .args(["-f", &pattern])
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false);
        if leaked {
            let _ = std::process::Command::new("pkill")
                .args(["-f", &pattern])
                .status();
        }
        assert!(!leaked, "grandchild survived own_process_group sweep");
    }

    /// A dependency that exits before its readiness probe fires must close its
    /// channel so dependents stop waiting; otherwise the whole group hangs until
    /// an external timeout. The wrapping `timeout` turns that hang into a failure.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dependency_exit_before_ready_does_not_hang() {
        let group = GroupRecipe {
            refs: vec![],
            recipes: HashMap::from([
                (
                    "a".to_string(),
                    process_recipe(
                        "false",
                        &[],
                        true,
                        Some(ReadyProbe::Delay { ms: 60_000 }),
                        vec![],
                    ),
                ),
                (
                    "b".to_string(),
                    process_recipe("sleep", &["30"], false, None, vec!["a".to_string()]),
                ),
            ]),
        };
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            group.run(false, CancelToken::new(), None),
        )
        .await
        .expect("group hung waiting on a dependency that exited before becoming ready");
        assert!(
            result.is_err(),
            "the failed dependency should fail the group"
        );
    }

    /// When a dependency exits cleanly before signaling ready, the dependent's
    /// error should name the dead dependency rather than a generic message.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dependent_names_dependency_that_exited_before_ready() {
        let group = GroupRecipe {
            refs: vec![],
            recipes: HashMap::from([
                (
                    "a".to_string(),
                    process_recipe(
                        "true",
                        &[],
                        false,
                        Some(ReadyProbe::Delay { ms: 60_000 }),
                        vec![],
                    ),
                ),
                (
                    "b".to_string(),
                    process_recipe("sleep", &["30"], false, None, vec!["a".to_string()]),
                ),
            ]),
        };
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            group.run(false, CancelToken::new(), None),
        )
        .await
        .expect("group hung");
        let err = result.expect_err("expected a readiness error");
        assert!(
            err.to_string()
                .contains("dependency \"a\" exited before \"b\" became ready"),
            "unexpected error: {err}"
        );
    }

    /// Optional sidecars that exit 0 must not tear down the rest of the group
    /// (regression from cancel-on-any-finish). The sleeper stays alive until
    /// the outer timeout fires — proving the group did not cancel early.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sidecar_ok_exit_does_not_cancel_siblings() {
        let group = GroupRecipe {
            refs: vec![],
            recipes: HashMap::from([
                (
                    "done".to_string(),
                    process_recipe("true", &[], false, None, vec![]),
                ),
                (
                    "linger".to_string(),
                    process_recipe("sleep", &["30"], false, None, vec![]),
                ),
            ]),
        };
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            group.run(false, CancelToken::new(), None),
        )
        .await;
        assert!(
            result.is_err(),
            "group returned early after a successful sidecar exit; siblings were cancelled"
        );
    }

    /// A failing sidecar (`fail_on_error`) must cancel siblings (fail-fast).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sidecar_err_cancels_siblings() {
        let group = GroupRecipe {
            refs: vec![],
            recipes: HashMap::from([
                (
                    "boom".to_string(),
                    process_recipe("false", &[], true, None, vec![]),
                ),
                (
                    "linger".to_string(),
                    process_recipe("sleep", &["30"], false, None, vec![]),
                ),
            ]),
        };
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            group.run(false, CancelToken::new(), None),
        )
        .await
        .expect("group hung after a failing sidecar");
        assert!(
            result.is_err(),
            "fail_on_error sidecar should fail the group"
        );
    }

    /// Sim exit cancels remaining sidecars even on Ok — the Monte Carlo /
    /// `elodin run` contract. Uses synthetic JoinSet members so we do not need
    /// a real Python sim binary in unit tests.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sim_ok_exit_cancels_siblings() {
        let cancel_token = CancelToken::new();
        let child = cancel_token.child();
        let mut set = JoinSet::new();
        set.spawn(async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            (RecipeRole::Sim, Ok(()))
        });
        set.spawn(async move {
            child.wait().await;
            (RecipeRole::Sidecar, Ok(()))
        });
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            await_group_members(set, cancel_token),
        )
        .await
        .expect("sim exit did not cancel the waiting sidecar");
        assert!(result.is_ok());
    }

    /// Successful sidecar-only finishes leave the group waiting (no cancel).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn await_group_members_ignores_sidecar_ok() {
        let cancel_token = CancelToken::new();
        let child = cancel_token.child();
        let mut set = JoinSet::new();
        set.spawn(async { (RecipeRole::Sidecar, Ok(())) });
        set.spawn(async move {
            tokio::select! {
                _ = child.wait() => (RecipeRole::Sidecar, Ok(())),
                _ = tokio::time::sleep(Duration::from_secs(30)) => (RecipeRole::Sidecar, Ok(())),
            }
        });
        let result = tokio::time::timeout(
            Duration::from_millis(400),
            await_group_members(set, cancel_token.clone()),
        )
        .await;
        assert!(
            result.is_err(),
            "sidecar Ok should not finish the group while another member is alive"
        );
        cancel_token.cancel();
    }
}
