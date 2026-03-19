use bevy::asset::io::{AssetSourceBuilder, AssetSourceEvent, AssetWatcher};
use bevy::prelude::*;
use notify_debouncer_mini::{
    DebounceEventResult, DebouncedEvent, Debouncer, new_debouncer,
    notify::{self, RecommendedWatcher, RecursiveMode},
};
use std::{
    env,
    path::{Path, PathBuf},
    time::Duration,
};

pub(crate) const KDL_ASSET_SOURCE: &str = "kdl";
const KDL_WATCH_DEBOUNCE: Duration = Duration::from_millis(300);

struct KdlAssetWatcher {
    _watcher: Debouncer<RecommendedWatcher>,
}

impl KdlAssetWatcher {
    fn new(
        root: PathBuf,
        sender: crossbeam_channel::Sender<AssetSourceEvent>,
    ) -> Result<Self, notify::Error> {
        let event_root = canonicalize_or_original(&root);
        let mut debouncer = new_debouncer(
            KDL_WATCH_DEBOUNCE,
            move |result: DebounceEventResult| match result {
                Ok(events) => forward_kdl_events(&event_root, events, &sender),
                Err(err) => {
                    error!(?err, root = ?event_root, "KDL asset watcher error");
                }
            },
        )?;
        debouncer.watcher().watch(&root, RecursiveMode::Recursive)?;
        Ok(Self {
            _watcher: debouncer,
        })
    }
}

impl AssetWatcher for KdlAssetWatcher {}

fn canonicalize_or_original(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn forward_kdl_events(
    root: &Path,
    events: Vec<DebouncedEvent>,
    sender: &crossbeam_channel::Sender<AssetSourceEvent>,
) {
    for event in events {
        let _ = send_modified_asset(root, &event.path, sender);
    }
}

fn send_modified_asset(
    root: &Path,
    event_path: &Path,
    sender: &crossbeam_channel::Sender<AssetSourceEvent>,
) -> bool {
    let Some(path) = to_asset_path(root, event_path) else {
        debug!(
            root = ?root,
            event_path = ?event_path,
            "Ignoring KDL watcher event outside of asset root"
        );
        return false;
    };
    let _ = sender.send(AssetSourceEvent::ModifiedAsset(path));
    true
}

fn to_asset_path(root: &Path, path: &Path) -> Option<PathBuf> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("kdl") {
        return None;
    }
    let relative = path
        .strip_prefix(root)
        .ok()
        .map(Path::to_path_buf)
        .or_else(|| path_under_canonical_root(root, path))?;
    Some(relative.to_path_buf())
}

fn path_under_canonical_root(root: &Path, path: &Path) -> Option<PathBuf> {
    let canonical_root = canonicalize_or_original(root);
    let canonical_path = canonicalize_or_original(path);
    canonical_path
        .strip_prefix(&canonical_root)
        .ok()
        .map(Path::to_path_buf)
}

fn resolve_kdl_source_root() -> Option<PathBuf> {
    let mut kdl_dir = match impeller2_kdl::env::schematic_dir_or_cwd() {
        Ok(path) => path,
        Err(err) => {
            error!("{err}, cannot register KDL asset source");
            return None;
        }
    };

    if kdl_dir.is_relative() {
        let Ok(mut cur_dir) = env::current_dir().inspect_err(|err| {
            error!("Cannot set KDL asset source. Could not get current directory: {err}.")
        }) else {
            return None;
        };
        cur_dir.push(&kdl_dir);
        kdl_dir = cur_dir;
    }

    Some(kdl_dir)
}

fn log_kdl_source_registration(root: &Path) {
    if !root.exists() {
        warn!("KDL asset directory {:?} does not exist.", root.display());
    } else if !root.is_dir() {
        warn!(
            "KDL asset directory {:?} is not a directory.",
            root.display()
        );
    } else {
        info!(
            "Registered KDL asset source {:?} at {:?}",
            KDL_ASSET_SOURCE,
            root.display()
        );
    }
}

pub(crate) fn plugin(app: &mut App) {
    let Some(kdl_dir) = resolve_kdl_source_root() else {
        return;
    };

    let Some(str_path) = kdl_dir.to_str() else {
        error!(
            "KDL asset directory contains invalid UTF-8 characters: {:?}",
            kdl_dir.display()
        );
        return;
    };

    let watcher_root = kdl_dir.clone();
    app.register_asset_source(
        KDL_ASSET_SOURCE,
        AssetSourceBuilder::platform_default(str_path, None).with_watcher(move |sender| {
            match KdlAssetWatcher::new(watcher_root.clone(), sender) {
                Ok(watcher) => Some(Box::new(watcher)),
                Err(err) => {
                    error!(
                        ?err,
                        root = ?watcher_root,
                        "Failed to create KDL asset watcher"
                    );
                    None
                }
            }
        }),
    );

    log_kdl_source_registration(&kdl_dir);
}

#[cfg(test)]
mod tests {
    use super::{KdlAssetWatcher, to_asset_path};
    use bevy::asset::io::AssetSourceEvent;
    use std::{
        fs,
        path::{Path, PathBuf},
        time::Duration,
        time::{SystemTime, UNIX_EPOCH},
    };

    struct TempTestDir(PathBuf);

    impl TempTestDir {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time went backwards")
                .as_nanos();
            let path = std::env::temp_dir().join(format!("elodin-kdl-watcher-{name}-{unique}"));
            fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempTestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn to_asset_path_returns_relative_path_for_kdl_file() {
        let root = TempTestDir::new("relative");
        let file = root.path().join("drone.kdl");
        fs::write(&file, "viewport").expect("write kdl");

        let asset_path = to_asset_path(root.path(), &file);

        assert_eq!(asset_path, Some(PathBuf::from("drone.kdl")));
    }

    #[cfg(unix)]
    #[test]
    fn to_asset_path_handles_symlinked_root() {
        use std::os::unix::fs::symlink;

        let temp = TempTestDir::new("symlink");
        let real_root = temp.path().join("real");
        fs::create_dir_all(&real_root).expect("create real root");
        let linked_root = temp.path().join("linked");
        symlink(&real_root, &linked_root).expect("create symlink root");

        let file = real_root.join("drone.kdl");
        fs::write(&file, "viewport").expect("write kdl");

        let asset_path = to_asset_path(&linked_root, &file);

        assert_eq!(asset_path, Some(PathBuf::from("drone.kdl")));
    }

    #[cfg(unix)]
    #[test]
    #[ignore = "requires OS file watcher events; run with --ignored"]
    fn watcher_emits_modified_asset_for_symlinked_root() {
        use crossbeam_channel::unbounded;
        use std::os::unix::fs::symlink;

        let temp = TempTestDir::new("watcher-symlink");
        let real_root = temp.path().join("real");
        fs::create_dir_all(&real_root).expect("create real root");
        let linked_root = temp.path().join("linked");
        symlink(&real_root, &linked_root).expect("create symlink root");

        let file = real_root.join("drone.kdl");
        fs::write(&file, "viewport").expect("write kdl");

        let (sender, receiver) = unbounded();
        let _watcher = KdlAssetWatcher::new(linked_root.clone(), sender).expect("create watcher");

        std::thread::sleep(Duration::from_millis(400));
        fs::write(&file, "viewport\n// watcher test\n").expect("update kdl");

        let event = receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("watcher event");

        assert_eq!(
            event,
            AssetSourceEvent::ModifiedAsset(PathBuf::from("drone.kdl"))
        );
    }
}
