use bevy::{
    asset::io::{AssetSourceBuilder, AssetSourceId},
    prelude::*,
};
use std::{env, path::PathBuf};

const DEFAULT_ASSETS_DIR: Option<&str> = Some("assets");

/// Use the environment variable `ELODIN_ASSETS_DIR` value as the asset source.
///
/// If it is not present use "assets".
///
/// If it is a relative path, prepend the current working directory onto it.
///
/// Register that path as the asset source.
///
/// Warn if the path does not exist or is not a directory.
///
/// If no `ELODIN_ASSETS_DIR` is given and `DEFAULT_ASSETS_DIR` is none, then
/// report the likely path Bevy will use to look for assets.
pub(crate) fn plugin(app: &mut App) {
    let var_set = env::var_os("ELODIN_ASSETS_DIR").is_some();
    if let Some(dir_name) = env::var_os("ELODIN_ASSETS_DIR")
        .or_else(|| DEFAULT_ASSETS_DIR.map(|s| std::ffi::OsString::from(String::from(s))))
    {
        let mut assets_dir: PathBuf = dir_name.into();
        if var_set {
            info!("ELODIN_ASSETS_DIR set to {:?}", assets_dir.display());
        } else {
            info!("ELODIN_ASSETS_DIR defaulted to {:?}", assets_dir.display());
        }
        if assets_dir.is_relative() {
            let Ok(mut cur_dir) = env::current_dir()
                .inspect_err(|e| error!("Cannot set asset source. Could not get current directory: {e}. Consider using an absolute path for ELODIN_ASSETS_DIR.")) else {
                    return;
                };
            cur_dir.push(&assets_dir);
            assets_dir = cur_dir;
        }
        // Bevy requires a UTF-8 string path.
        let str_path = assets_dir.to_str().or_else(|| {
            if let Some(default_dir) = &DEFAULT_ASSETS_DIR {
                error!(
                    "ELODIN_ASSETS_DIR contains invalid UTF-8 characters, falling back to {:?}",
                    default_dir
                );
            } else {
                error!("ELODIN_ASSETS_DIR contains invalid UTF-8 characters; no default available");
            }
            DEFAULT_ASSETS_DIR
        });

        if let Some(str_path) = str_path {
            app.register_asset_source(
                &AssetSourceId::Default,
                AssetSourceBuilder::platform_default(str_path, None),
            );
            if !assets_dir.exists() {
                warn!(
                    "ELODIN_ASSETS_DIR {:?} does not exist.",
                    assets_dir.display()
                );
            } else if !assets_dir.is_dir() {
                warn!(
                    "ELODIN_ASSETS_DIR {:?} is not a directory.",
                    assets_dir.display()
                );
            }
        }
    } else {
        // No assets directory. Report which one will likely be used by Bevy.
        match env::current_exe() {
            Ok(bin_path) => {
                if let Some(dir) = bin_path.parent() {
                    info!(
                        "ELODIN_ASSETS_DIR env not set; use {:?} by default.",
                        dir.display()
                    );
                } else {
                    warn!(
                        "ELODIN_ASSETS_DIR env not set, for binary {:?}.",
                        bin_path.display()
                    );
                }
            }
            Err(e) => {
                warn!("ELODIN_ASSETS_DIR env not set. Cannot determine binary path {e}");
            }
        }
    }
}
