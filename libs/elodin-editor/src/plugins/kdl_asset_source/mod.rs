use bevy::asset::io::AssetSourceBuilder;
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

    app.register_asset_source(
        KDL_ASSET_SOURCE,
        AssetSourceBuilder::platform_default(str_path, None),
    );

    log_kdl_source_registration(&kdl_dir);
}
