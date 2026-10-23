//! Types stubbed under `wasm-cut` so the editor still compiles without
//! native-only crates (`bevy_ai_skybox`, `directories`, …). Search
//! `feature = "wasm-cut"` for remaining wasm work.

#![allow(dead_code)]

use bevy::prelude::*;
use std::path::PathBuf;

#[derive(Component)]
pub struct PrimarySkybox;

#[derive(Resource, Default)]
pub struct SkyboxCache {
    pub active: Option<String>,
    pub manifest: SkyboxManifest,
}

impl SkyboxCache {
    pub fn empty(_manifest_path: PathBuf) -> Self {
        Self::default()
    }
}

#[derive(Default)]
pub struct SkyboxManifest {
    pub entries: Vec<ManifestEntry>,
}

pub struct ManifestEntry {
    pub name: String,
}

#[derive(Clone, Message)]
pub enum SetActiveSkybox {
    Clear,
    ByName(String),
}

#[derive(Resource, Default)]
pub struct SkyboxGenerationUi {
    pub phase: SkyboxGenerationPhase,
    pub message: Option<String>,
    pub revert_name: Option<String>,
    pub target_name: Option<String>,
}

impl SkyboxGenerationUi {
    pub fn is_busy(&self) -> bool {
        false
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum SkyboxGenerationPhase {
    #[default]
    Idle,
    Generating,
    PendingApply,
}

#[derive(Resource, Default)]
pub struct SkyboxCacheHealth {
    pub load_error: Option<String>,
}

#[derive(Resource, Default)]
pub struct LocallyPushedSkyboxActive;

impl LocallyPushedSkyboxActive {
    pub fn consume_matching(&mut self, _active: Option<&str>) -> bool {
        false
    }
}

#[derive(Resource, Default)]
pub struct DbSkyboxAssetMirror;

impl DbSkyboxAssetMirror {
    pub fn note_local_clear(&mut self, _desired: Option<String>) {}
}

#[derive(Resource, Default)]
pub struct DbSkyboxSyncInFlight;

#[derive(Resource, Default)]
pub struct DbSkyboxUploaded;
