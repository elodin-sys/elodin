use bevy::asset::io::{AssetSource, AssetSourceBuilder};
use bevy::prelude::*;
use std::{
    env,
    path::{Path, PathBuf},
};

pub(crate) const KDL_ASSET_SOURCE: &str = "kdl";

pub(crate) fn canonicalize_or_original(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn resolve_kdl_source_root() -> Option<PathBuf> {
    let mut kdl_dir = env::current_dir().ok()?;

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
            "Registered KDL asset source {:?} at {:?} (hot-reload: off)",
            KDL_ASSET_SOURCE,
            root.display(),
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

    // The cwd can be a large dev workspace; skip the recursive file watcher so
    // startup does not exhaust inotify limits. Offline `--kdl` loads are explicit.
    let source_builder =
        AssetSourceBuilder::new(AssetSource::get_default_reader(str_path.to_string()))
            .with_writer(AssetSource::get_default_writer(str_path.to_string()));

    app.register_asset_source(KDL_ASSET_SOURCE, source_builder);
    log_kdl_source_registration(&kdl_dir);
}
