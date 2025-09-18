use bevy::{
    asset::{
        io::{AssetSourceBuilder, AssetSourceId},
    },
    prelude::*,
};
use std::{
    env,
    path::PathBuf,
};


pub(crate) fn plugin(app: &mut App) {
    if let Some(dir_name) = env::var_os("ELODIN_ASSETS_DIR") {
        let mut asset_dir: PathBuf = dir_name.into();
        if asset_dir.is_relative() {
            // let mut cur_dir = env::current_dir()?;
            let Ok(mut cur_dir) = env::current_dir()
                .inspect_err(|e| error!("Could not get current directory: {e}")) else {
                    return;
                };
            cur_dir.push(&asset_dir);
            asset_dir = cur_dir;
        }
        app.register_asset_source(
            &AssetSourceId::Default,
            AssetSourceBuilder::platform_default(asset_dir.to_str().expect("asset dir"), None),
        );
        info!("Using ELODIN_ASSETS_DIR {:?}", asset_dir.display());
    }
}
