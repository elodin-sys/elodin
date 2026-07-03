use bevy::{
    asset::io::{AssetSourceBuilder, AssetSourceId},
    prelude::*,
};
use std::path::PathBuf;

pub(crate) fn resolve_assets_dir() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("ELODIN_ASSETS") {
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
    if std::env::var_os("ELODIN_ASSETS").is_some() {
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
