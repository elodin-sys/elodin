#[cfg(target_os = "linux")]
use std::fs;
use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug)]
pub struct CgroupScope {
    path: PathBuf,
}

impl CgroupScope {
    #[cfg(target_os = "linux")]
    pub fn create(name: impl AsRef<str>) -> io::Result<Option<Arc<Self>>> {
        let Some(base) = writable_base()? else {
            return Ok(None);
        };
        Self::create_in(base, name)
    }

    #[cfg(not(target_os = "linux"))]
    pub fn create(_name: impl AsRef<str>) -> io::Result<Option<Arc<Self>>> {
        Ok(None)
    }

    #[cfg(target_os = "linux")]
    pub fn create_child(
        parent: &Arc<Self>,
        name: impl AsRef<str>,
    ) -> io::Result<Option<Arc<Self>>> {
        Self::create_in(parent.path.clone(), name)
    }

    #[cfg(not(target_os = "linux"))]
    pub fn create_child(
        _parent: &Arc<Self>,
        _name: impl AsRef<str>,
    ) -> io::Result<Option<Arc<Self>>> {
        Ok(None)
    }

    #[cfg(target_os = "linux")]
    fn create_in(base: PathBuf, name: impl AsRef<str>) -> io::Result<Option<Arc<Self>>> {
        let path = base.join(sanitize_name(name.as_ref()));
        match fs::create_dir(&path) {
            Ok(()) => Ok(Some(Arc::new(Self { path }))),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
                Ok(Some(Arc::new(Self { path })))
            }
            Err(err) if is_unavailable(&err) => Ok(None),
            Err(err) => Err(err),
        }
    }

    #[cfg(target_os = "linux")]
    pub fn add_pid(&self, pid: u32) -> io::Result<()> {
        match fs::write(self.path.join("cgroup.procs"), pid.to_string()) {
            Ok(()) => Ok(()),
            Err(err) if is_unavailable(&err) => Ok(()),
            Err(err) => Err(err),
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn add_pid(&self, _pid: u32) -> io::Result<()> {
        Ok(())
    }

    /// Give every process in this cgroup a larger share of the CPU under
    /// contention (cgroup-v2 `cpu.weight`, 1..=10000, default 100), so the
    /// real-time simulation stack is scheduled promptly when other work (an
    /// editor render loop, co-located services) competes for cores.
    ///
    /// Entirely best-effort: the `cpu.weight` knob only exists when the `cpu`
    /// controller is delegated to this cgroup (we try to enable it in the
    /// parent's `cgroup.subtree_control` first), and we never fail a simulation
    /// because priority could not be raised. A no-op on non-Linux and on hosts
    /// where the controller isn't available.
    #[cfg(target_os = "linux")]
    pub fn set_cpu_weight(&self, weight: u32) {
        let weight = weight.clamp(1, 10_000);
        if let Some(parent) = self.path.parent() {
            let _ = fs::write(parent.join("cgroup.subtree_control"), "+cpu");
        }
        let _ = fs::write(self.path.join("cpu.weight"), weight.to_string());
    }

    #[cfg(not(target_os = "linux"))]
    pub fn set_cpu_weight(&self, _weight: u32) {}

    #[cfg(target_os = "linux")]
    pub fn kill(&self) -> io::Result<()> {
        match fs::write(self.path.join("cgroup.kill"), "1") {
            Ok(()) => Ok(()),
            Err(err) if is_unavailable(&err) => Ok(()),
            Err(err) => Err(err),
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn kill(&self) -> io::Result<()> {
        Ok(())
    }

    #[cfg(target_os = "linux")]
    pub fn remove(&self) -> io::Result<()> {
        match fs::remove_dir(&self.path) {
            Ok(()) => Ok(()),
            Err(err)
                if matches!(
                    err.kind(),
                    io::ErrorKind::NotFound
                        | io::ErrorKind::PermissionDenied
                        | io::ErrorKind::DirectoryNotEmpty
                ) =>
            {
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn remove(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(target_os = "linux")]
    pub fn reap_prefix(prefix: &str) -> io::Result<()> {
        let Some(base) = writable_base()? else {
            return Ok(());
        };
        for entry in fs::read_dir(base)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with(prefix) {
                continue;
            }
            let scope = Self { path: entry.path() };
            let _ = scope.kill();
            let _ = remove_cgroup_tree(scope.path());
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn reap_prefix(_prefix: &str) -> io::Result<()> {
        Ok(())
    }
}

/// Whether this process can create (and therefore kill) cgroup scopes: a
/// writable cgroup-v2 base was found. Orchestrators use this to decide whether
/// to self-scope under `systemd-run --user --scope` for reliable teardown.
#[cfg(target_os = "linux")]
pub fn cgroups_available() -> bool {
    matches!(writable_base(), Ok(Some(_)))
}

#[cfg(not(target_os = "linux"))]
pub fn cgroups_available() -> bool {
    false
}

/// Whether s10 should prioritize the simulation stack (via cgroup `cpu.weight`).
/// On by default; set `ELODIN_S10_PRIORITY=off` to disable (for A/B comparison
/// or debugging).
pub fn priority_enabled() -> bool {
    !matches!(
        std::env::var("ELODIN_S10_PRIORITY").as_deref(),
        Ok("off") | Ok("0") | Ok("false")
    )
}

/// cgroup-v2 `cpu.weight` to apply to the simulation stack. Defaults to the
/// maximum share (the sim is latency-critical but low-duty, so a high weight
/// gets it scheduled promptly without meaningfully starving others). Override
/// with `ELODIN_S10_CPU_WEIGHT`.
pub fn sim_cpu_weight() -> u32 {
    std::env::var("ELODIN_S10_CPU_WEIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000)
}

#[cfg(target_os = "linux")]
fn remove_cgroup_tree(path: &Path) -> io::Result<()> {
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                let _ = remove_cgroup_tree(&entry.path());
            }
        }
    }
    match fs::remove_dir(path) {
        Ok(()) => Ok(()),
        Err(err)
            if matches!(
                err.kind(),
                io::ErrorKind::NotFound
                    | io::ErrorKind::PermissionDenied
                    | io::ErrorKind::DirectoryNotEmpty
            ) =>
        {
            Ok(())
        }
        Err(err) => Err(err),
    }
}

#[cfg(target_os = "linux")]
fn writable_base() -> io::Result<Option<PathBuf>> {
    if fs::metadata("/sys/fs/cgroup/cgroup.controllers").is_err() {
        return Ok(None);
    }

    let cgroup = fs::read_to_string("/proc/self/cgroup")?;
    let rel = cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .unwrap_or("/");
    let current = Path::new("/sys/fs/cgroup").join(rel.trim_start_matches('/'));
    let mut candidates = Vec::new();
    let mut cursor = Some(current.as_path());
    while let Some(path) = cursor {
        candidates.push(path.to_path_buf());
        cursor = path.parent();
    }

    for base in candidates {
        let test = base.join(format!(".elodin-cgroup-test-{}", std::process::id()));
        match fs::create_dir(&test) {
            Ok(()) => {
                let _ = fs::remove_dir(&test);
                return Ok(Some(base));
            }
            Err(err) if is_unavailable(&err) => continue,
            Err(err) => return Err(err),
        }
    }

    Ok(None)
}

#[cfg(target_os = "linux")]
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn is_unavailable(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::NotFound
            | io::ErrorKind::PermissionDenied
            | io::ErrorKind::ReadOnlyFilesystem
    )
}
