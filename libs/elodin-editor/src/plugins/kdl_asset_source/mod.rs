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

/// Source of the KDL root directory: either the `ELODIN_KDL_DIR` env var, or
/// the cwd fallback. The fallback variant controls whether we attach a
/// recursive Bevy file watcher (see `plugin` below).
enum KdlRootSource {
    /// User explicitly pointed `ELODIN_KDL_DIR` at this path.
    EnvVar,
    /// Env var unset; we defaulted to the current working directory.
    CwdFallback,
}

fn resolve_kdl_source_root() -> Option<(PathBuf, KdlRootSource)> {
    let (mut kdl_dir, origin) = match impeller2_kdl::env::schematic_dir() {
        Ok(Some(path)) => (path, KdlRootSource::EnvVar),
        Ok(None) => match env::current_dir() {
            Ok(cwd) => (cwd, KdlRootSource::CwdFallback),
            Err(err) => {
                error!("Cannot set KDL asset source. Could not get current directory: {err}.");
                return None;
            }
        },
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

    Some((kdl_dir, origin))
}

fn log_kdl_source_registration(root: &Path, hot_reload: bool) {
    if !root.exists() {
        warn!("KDL asset directory {:?} does not exist.", root.display());
    } else if !root.is_dir() {
        warn!(
            "KDL asset directory {:?} is not a directory.",
            root.display()
        );
    } else {
        info!(
            "Registered KDL asset source {:?} at {:?} (hot-reload: {})",
            KDL_ASSET_SOURCE,
            root.display(),
            if hot_reload { "on" } else { "off" },
        );
    }
}

pub(crate) fn plugin(app: &mut App) {
    let Some((kdl_dir, origin)) = resolve_kdl_source_root() else {
        return;
    };

    let Some(str_path) = kdl_dir.to_str() else {
        error!(
            "KDL asset directory contains invalid UTF-8 characters: {:?}",
            kdl_dir.display()
        );
        return;
    };

    // When `ELODIN_KDL_DIR` is unset we fall back to cwd, which in a dev
    // workspace can be tens of thousands of directories deep (e.g., this repo
    // with a populated `target/`). Bevy's default file watcher recursively
    // registers an inotify watch per directory, blowing past
    // `fs.inotify.max_user_watches` and panicking during app start-up. Only
    // attach the watcher when the user has explicitly opted in via the env
    // var, which in practice means they have pointed it at a small,
    // KDL-specific directory.
    let source_builder = match origin {
        KdlRootSource::EnvVar => AssetSourceBuilder::platform_default(str_path, None),
        KdlRootSource::CwdFallback => AssetSourceBuilder::default()
            .with_reader(AssetSource::get_default_reader(str_path.to_string()))
            .with_writer(AssetSource::get_default_writer(str_path.to_string())),
    };

    app.register_asset_source(KDL_ASSET_SOURCE, source_builder);

    let hot_reload = matches!(origin, KdlRootSource::EnvVar);
    log_kdl_source_registration(&kdl_dir, hot_reload);
    if !hot_reload {
        info!(
            "KDL hot-reload disabled because ELODIN_KDL_DIR is unset. \
             Set ELODIN_KDL_DIR=<schematic-dir> to enable it."
        );
    }
}
