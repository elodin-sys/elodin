use bevy::asset::{AssetLoader, AssetPath, io::Reader};
use impeller2_kdl::FromKdl;
use impeller2_wkt::Schematic;
use std::path::Path;

use super::types::{
    SchematicDocumentAsset, SchematicDocumentLoader, SchematicDocumentLoaderError, SchematicWindow,
};

impl AssetLoader for SchematicDocumentLoader {
    type Asset = SchematicDocumentAsset;
    type Settings = ();
    type Error = SchematicDocumentLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        load_context: &mut bevy::asset::LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let root = Schematic::from_kdl(&String::from_utf8(bytes)?)?;
        let mut windows = Vec::new();
        let base_dir = load_context
            .asset_path()
            .path()
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let source = load_context.asset_path().source().clone_owned();

        for window in root.elems.iter().filter_map(|elem| match elem {
            impeller2_wkt::SchematicElem::Window(window) => Some(window),
            _ => None,
        }) {
            let Some(path) = window.path.as_deref() else {
                continue;
            };
            let asset_path =
                AssetPath::from_path_buf(base_dir.join(path)).with_source(source.clone());
            let handle = load_context.load(asset_path.clone());
            windows.push(SchematicWindow {
                handle,
                asset_path: asset_path.clone_owned(),
            });
        }

        Ok(SchematicDocumentAsset { root, windows })
    }

    fn extensions(&self) -> &[&str] {
        &["kdl"]
    }
}
