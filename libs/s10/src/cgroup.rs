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
