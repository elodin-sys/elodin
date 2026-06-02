use serde::{Deserialize, Serialize};

pub const ASSETS_MANIFEST_KEY: &str = "assets.manifest";
pub const ASSET_MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct AssetManifest {
    pub version: u32,
    pub assets: Vec<AssetEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssetEntry {
    pub logical_path: String,
    pub hash: String,
    pub media_type: String,
    pub byte_len: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_path: Option<String>,
}

impl AssetManifest {
    pub fn new() -> Self {
        Self {
            version: ASSET_MANIFEST_VERSION,
            assets: Vec::new(),
        }
    }

    pub fn find(&self, logical_path: &str) -> Option<&AssetEntry> {
        self.assets
            .iter()
            .find(|entry| entry.logical_path == logical_path)
    }
}
