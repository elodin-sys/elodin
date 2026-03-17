use bevy::asset::io::{AssetSourceBuilder, AssetSourceEvent, AssetWatcher};
use bevy::prelude::*;
use notify_debouncer_mini::{
    DebounceEventResult, Debouncer, new_debouncer,
    notify::{self, RecommendedWatcher, RecursiveMode},
};
use std::{env, path::PathBuf};
use std::{path::Path, time::Duration};

pub(crate) const KDL_ASSET_SOURCE: &str = "kdl";

struct KdlAssetWatcher {
    _watcher: Debouncer<RecommendedWatcher>,
}

impl KdlAssetWatcher {
    fn new(
        root: PathBuf,
        sender: crossbeam_channel::Sender<AssetSourceEvent>,
    ) -> Result<Self, notify::Error> {
        let event_root = root.clone();
        let mut debouncer = new_debouncer(
            Duration::from_millis(300),
            move |result: DebounceEventResult| match result {
                Ok(events) => {
                    for event in events {
                        let Some(path) = to_asset_path(&event_root, &event.path) else {
                            continue;
                        };
                        let _ = sender.send(AssetSourceEvent::ModifiedAsset(path));
                    }
                }
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

fn to_asset_path(root: &Path, path: &Path) -> Option<PathBuf> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("kdl") {
        return None;
    }
    let relative = path.strip_prefix(root).ok()?;
    Some(relative.to_path_buf())
}

pub(crate) fn plugin(app: &mut App) {
    let mut kdl_dir = match impeller2_kdl::env::schematic_dir_or_cwd() {
        Ok(path) => path,
        Err(err) => {
            error!("{err}, cannot register KDL asset source");
            return;
        }
    };

    if kdl_dir.is_relative() {
        let Ok(mut cur_dir) = env::current_dir().inspect_err(|err| {
            error!("Cannot set KDL asset source. Could not get current directory: {err}.")
        }) else {
            return;
        };
        cur_dir.push(&kdl_dir);
        kdl_dir = cur_dir;
    }

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

    if !kdl_dir.exists() {
        warn!(
            "KDL asset directory {:?} does not exist.",
            kdl_dir.display()
        );
    } else if !kdl_dir.is_dir() {
        warn!(
            "KDL asset directory {:?} is not a directory.",
            kdl_dir.display()
        );
    } else {
        info!(
            "Registered KDL asset source {:?} at {:?}",
            KDL_ASSET_SOURCE,
            PathBuf::from(str_path).display()
        );
    }
}
