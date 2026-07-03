use bevy::{
    asset::io::{AssetSourceBuilder, AssetSourceId},
    prelude::*,
};
use std::path::PathBuf;

/// Read the asset-root override, honoring the deprecated `ELODIN_ASSETS_DIR`
/// name (pre-rename deployments, e.g. AlephOS session vars) with a one-time
/// deprecation warning.
pub(crate) fn assets_env_override() -> Option<std::ffi::OsString> {
    if let Some(val) = std::env::var_os("ELODIN_ASSETS") {
        return Some(val);
    }
    let legacy = std::env::var_os("ELODIN_ASSETS_DIR")?;
    static WARN_ONCE: std::sync::Once = std::sync::Once::new();
    WARN_ONCE.call_once(|| {
        warn!("ELODIN_ASSETS_DIR is deprecated; rename it to ELODIN_ASSETS");
    });
    Some(legacy)
}

pub(crate) fn resolve_assets_dir() -> Option<PathBuf> {
    if let Some(explicit) = assets_env_override() {
        let path = PathBuf::from(explicit);
        return Some(if path.is_absolute() {
            path
        } else {
            std::env::current_dir().unwrap_or_default().join(path)
        });
    }
    Some(std::env::current_dir().ok()?.join("assets"))
}

/// Register the resolved assets directory as Bevy's default asset source.
pub(crate) fn plugin(app: &mut App) {
    let Some(assets_dir) = resolve_assets_dir() else {
        warn!("Cannot resolve asset source: could not get current directory");
        return;
    };
    if assets_env_override().is_some() {
        info!("ELODIN_ASSETS set to {:?}", assets_dir.display());
    } else {
        info!("Assets directory defaulted to {:?}", assets_dir.display());
    }
    let Some(str_path) = assets_dir.to_str() else {
        error!(
            "Assets directory contains invalid UTF-8 characters: {:?}",
            assets_dir.display()
        );
        return;
    };
    app.register_asset_source(
        &AssetSourceId::Default,
        AssetSourceBuilder::platform_default(str_path, None),
    );
    if !assets_dir.exists() {
        warn!(
            "Assets directory {:?} does not exist.",
            assets_dir.display()
        );
    } else if !assets_dir.is_dir() {
        warn!(
            "Assets directory {:?} is not a directory.",
            assets_dir.display()
        );
    }
}
