use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_ai_skybox::prelude::{
    SetActiveSkybox, SkyboxCache, SkyboxGenerated, SkyboxGenerationPhase, SkyboxGenerationUi,
};
use impeller2_bevy::PacketTx;
use impeller2_kdl::ToKdl;
use impeller2_wkt::{DbConfig, SetDbConfig, SkyboxConfig, StoreAsset};
use std::collections::VecDeque;

use crate::plugins::kdl_document::{
    ACTIVE_SCHEMATIC_KEY, CurrentDocument, DocumentLoaded, DocumentReloaded, LastSyncedActiveKey,
    SchematicDocumentAsset, sync_document_skybox,
};
use crate::ui::schematic::CurrentSchematic;

pub(crate) struct SkyboxDocumentSyncMut<'a> {
    pub schematic: &'a mut CurrentSchematic,
    pub current_document: &'a mut CurrentDocument,
    pub document_assets: &'a mut Assets<SchematicDocumentAsset>,
    pub last_synced_key: &'a mut LastSyncedActiveKey,
    pub locally_pushed: &'a mut LocallyPushedSkyboxActive,
    pub cache: &'a mut SkyboxCache,
    pub tx: &'a PacketTx,
    /// DB key the schematic is currently stored under, so a skybox edit writes
    /// back to the active (possibly "Save As"-named) schematic rather than
    /// always clobbering `schematics/main.kdl`.
    pub active_key: &'a str,
}

impl SkyboxDocumentSyncMut<'_> {
    pub(crate) fn sync_skybox_to_document_and_db(&mut self, skybox: Option<SkyboxConfig>) {
        let kdl = sync_document_skybox(
            skybox.clone(),
            self.current_document,
            self.document_assets,
            self.schematic,
        );
        sync_cache_active_from_skybox(self.cache, skybox.as_ref());
        self.last_synced_key.0 = Some(self.active_key.to_string());
        let active = self
            .schematic
            .skybox
            .as_ref()
            .map(|entry| entry.name.as_str());
        push_skybox_db_sync(
            self.tx,
            Some(kdl),
            active,
            self.active_key,
            self.locally_pushed,
        );
    }
}

#[derive(Resource, Default)]
pub(crate) struct LocallyPushedSkyboxActive {
    pending: VecDeque<Option<String>>,
}

impl LocallyPushedSkyboxActive {
    const MAX_PENDING: usize = 8;

    pub(crate) fn mark(&mut self, skybox: Option<&str>) {
        if self
            .pending
            .iter()
            .any(|pending| pending.as_deref() == skybox)
        {
            return;
        }
        self.pending.push_back(skybox.map(str::to_string));
        while self.pending.len() > Self::MAX_PENDING {
            self.pending.pop_front();
        }
    }

    pub(crate) fn consume_matching(&mut self, skybox: Option<&str>) -> bool {
        let Some(index) = self
            .pending
            .iter()
            .position(|pending| pending.as_deref() == skybox)
        else {
            return false;
        };
        self.pending.remove(index);
        true
    }

    /// Whether `skybox` is still queued as a locally pushed state whose DB echo
    /// hasn't arrived yet. Lets the DB mirror tell a user-initiated change
    /// (authoritative until the config catches up) apart from an external drift
    /// it should re-assert.
    pub(crate) fn is_pending(&self, skybox: Option<&str>) -> bool {
        self.pending
            .iter()
            .any(|pending| pending.as_deref() == skybox)
    }
}

pub(crate) fn sync_cache_active_from_skybox(
    cache: &mut SkyboxCache,
    skybox: Option<&SkyboxConfig>,
) {
    cache.active = skybox.map(|entry| entry.name.clone());
}

#[derive(SystemParam)]
pub(crate) struct SyncGeneratedSkyboxParams<'w> {
    schematic: ResMut<'w, CurrentSchematic>,
    current_document: ResMut<'w, CurrentDocument>,
    document_assets: ResMut<'w, Assets<SchematicDocumentAsset>>,
    last_synced_key: ResMut<'w, LastSyncedActiveKey>,
    locally_pushed: ResMut<'w, LocallyPushedSkyboxActive>,
    cache: ResMut<'w, SkyboxCache>,
    config: Res<'w, DbConfig>,
    tx: Res<'w, PacketTx>,
}

pub(crate) fn sync_generated_skybox_to_schematic(
    mut reader: MessageReader<SkyboxGenerated>,
    mut params: SyncGeneratedSkyboxParams,
) {
    for event in reader.read() {
        let mut sync = SkyboxDocumentSyncMut {
            schematic: &mut params.schematic,
            current_document: &mut params.current_document,
            document_assets: &mut params.document_assets,
            last_synced_key: &mut params.last_synced_key,
            locally_pushed: &mut params.locally_pushed,
            cache: &mut params.cache,
            tx: &params.tx,
            active_key: params
                .config
                .schematic_active()
                .unwrap_or(ACTIVE_SCHEMATIC_KEY),
        };
        sync.sync_skybox_to_document_and_db(Some(SkyboxConfig {
            name: event.name.clone(),
        }));
    }
}

pub(crate) fn on_document_loaded(
    mut loaded: MessageReader<DocumentLoaded>,
    config: Res<DbConfig>,
    mut locally_pushed: ResMut<LocallyPushedSkyboxActive>,
    mut last_synced_key: ResMut<LastSyncedActiveKey>,
    mut caches: Query<&mut crate::ui::video_stream::VideoFrameCache>,
    tx: Option<Res<PacketTx>>,
) {
    let Some(event) = loaded.read().last() else {
        return;
    };
    let document = &event.document;
    let loaded_skybox = document
        .root
        .skybox
        .as_ref()
        .map(|entry| entry.name.as_str());

    crate::ui::video_stream::invalidate_sensor_frames_if_loaded_skybox_differs(
        loaded_skybox,
        &config,
        &mut caches,
    );

    if should_push_loaded_skybox_to_db(loaded_skybox, config.skybox_active())
        && let Some(tx) = tx.as_ref()
    {
        let kdl = document.root.to_kdl();
        let active_key = config.schematic_active().unwrap_or(ACTIVE_SCHEMATIC_KEY);
        push_skybox_db_sync(
            tx,
            Some(kdl),
            loaded_skybox,
            active_key,
            &mut locally_pushed,
        );
    }

    last_synced_key.0 = config.schematic_active().map(str::to_string);
}

pub(crate) fn record_reloaded_schematic_key(
    mut reloaded: MessageReader<DocumentReloaded>,
    config: Res<DbConfig>,
    mut last_synced_key: ResMut<LastSyncedActiveKey>,
) {
    if reloaded.read().last().is_some() {
        last_synced_key.0 = config.schematic_active().map(str::to_string);
    }
}

/// Push skybox metadata to the DB. When `kdl` is `None`, only `skybox.active` is updated
/// (used while a generated cubemap is still loading on the render server).
pub(crate) fn push_skybox_db_sync(
    tx: &PacketTx,
    kdl: Option<String>,
    skybox: Option<&str>,
    active_key: &str,
    locally_pushed: &mut LocallyPushedSkyboxActive,
) {
    locally_pushed.mark(skybox);
    push_schematic_metadata(tx, kdl, skybox, active_key);
}

pub(crate) fn push_schematic_metadata(
    tx: &PacketTx,
    kdl: Option<String>,
    skybox: Option<&str>,
    active_key: &str,
) {
    let mut metadata = std::collections::HashMap::new();
    // DB-centric write-back (RFD #724): store schematic bytes as an asset, then
    // repoint `schematic.active` at the current key so a named schematic's
    // skybox edit doesn't silently revert to `schematics/main.kdl`.
    if let Some(kdl) = kdl {
        tx.send_msg(StoreAsset {
            key: active_key.to_string(),
            bytes: kdl.into_bytes(),
        });
        metadata.insert("schematic.active".to_string(), active_key.to_string());
    }
    metadata.insert(
        "skybox.active".to_string(),
        skybox.unwrap_or("").to_string(),
    );
    tx.send_msg(SetDbConfig {
        metadata,
        ..Default::default()
    });
}

/// Notify render-server while the cubemap is still loading on the editor.
pub(crate) fn push_skybox_active_on_pending(
    ui: Res<SkyboxGenerationUi>,
    mut last_pushed: Local<Option<String>>,
    mut locally_pushed: ResMut<LocallyPushedSkyboxActive>,
    tx: Res<PacketTx>,
) {
    if ui.phase != SkyboxGenerationPhase::PendingApply {
        *last_pushed = None;
        return;
    }
    let Some(name) = ui.target_name.clone() else {
        return;
    };
    if last_pushed.as_deref() == Some(name.as_str()) {
        return;
    }
    // No schematic KDL here (only `skybox.active` is pushed while the cubemap
    // loads), so the active key is unused; pass the default.
    push_skybox_db_sync(
        &tx,
        None,
        Some(&name),
        ACTIVE_SCHEMATIC_KEY,
        &mut locally_pushed,
    );
    *last_pushed = Some(name);
}

pub fn decay_skybox_status_message(
    time: Res<Time>,
    mut ui: ResMut<SkyboxGenerationUi>,
    mut shown_at: Local<Option<f32>>,
) {
    match ui.phase {
        SkyboxGenerationPhase::Ready | SkyboxGenerationPhase::Failed => {
            let start = *shown_at.get_or_insert(time.elapsed_secs());
            let timeout = match ui.phase {
                SkyboxGenerationPhase::Ready => 2.5,
                SkyboxGenerationPhase::Failed => 5.0,
                _ => 2.5,
            };
            if time.elapsed_secs() - start > timeout {
                ui.phase = SkyboxGenerationPhase::Idle;
                ui.message = None;
                ui.prompt = None;
                ui.target_name = None;
                *shown_at = None;
            }
        }
        _ => {
            *shown_at = None;
        }
    }
}

#[derive(SystemParam)]
pub(crate) struct RevertSkyboxParams<'w> {
    ui: ResMut<'w, SkyboxGenerationUi>,
    schematic: ResMut<'w, CurrentSchematic>,
    current_document: ResMut<'w, CurrentDocument>,
    document_assets: ResMut<'w, Assets<SchematicDocumentAsset>>,
    last_synced_key: ResMut<'w, LastSyncedActiveKey>,
    locally_pushed: ResMut<'w, LocallyPushedSkyboxActive>,
    cache: ResMut<'w, SkyboxCache>,
    config: Res<'w, DbConfig>,
    tx: Res<'w, PacketTx>,
    skyboxes: MessageWriter<'w, SetActiveSkybox>,
}

pub(crate) fn revert_previous_skybox(mut params: RevertSkyboxParams) {
    let Some(name) = params.ui.revert_name.take() else {
        return;
    };
    let skybox = SkyboxConfig { name: name.clone() };
    let mut sync = SkyboxDocumentSyncMut {
        schematic: &mut params.schematic,
        current_document: &mut params.current_document,
        document_assets: &mut params.document_assets,
        last_synced_key: &mut params.last_synced_key,
        locally_pushed: &mut params.locally_pushed,
        cache: &mut params.cache,
        tx: &params.tx,
        active_key: params
            .config
            .schematic_active()
            .unwrap_or(ACTIVE_SCHEMATIC_KEY),
    };
    sync.sync_skybox_to_document_and_db(Some(skybox));
    params.skyboxes.write(SetActiveSkybox::ByName(name.clone()));
    params.ui.phase = SkyboxGenerationPhase::Idle;
    params.ui.message = Some(format!("Reverted to skybox `{name}`"));
}

fn should_push_loaded_skybox_to_db(loaded: Option<&str>, db: Option<&str>) -> bool {
    loaded != db
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::kdl_document::{LastSyncedActiveKey, SchematicDocumentAsset};
    use impeller2_wkt::Schematic;
    use std::path::PathBuf;

    #[test]
    fn on_document_loaded_updates_last_synced_without_db_push() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<DbConfig>()
            .init_resource::<LocallyPushedSkyboxActive>()
            .add_message::<DocumentLoaded>()
            .add_systems(Update, on_document_loaded);

        app.world_mut().write_message(DocumentLoaded {
            save_path: None,
            document: SchematicDocumentAsset {
                root: Schematic {
                    skybox: Some(SkyboxConfig {
                        name: "seaport".to_string(),
                    }),
                    ..default()
                },
                windows: Vec::new(),
            },
        });
        app.update();

        let last = app.world().resource::<LastSyncedActiveKey>();
        assert!(
            last.0.as_ref().is_some_and(|kdl| kdl.contains("seaport")),
            "expected loaded schematic KDL to be recorded locally"
        );
    }

    #[test]
    fn loaded_skybox_db_push_needed_when_names_differ() {
        assert!(should_push_loaded_skybox_to_db(
            Some("seaport"),
            Some("machu_picchu")
        ));
        assert!(!should_push_loaded_skybox_to_db(
            Some("seaport"),
            Some("seaport")
        ));
        assert!(should_push_loaded_skybox_to_db(None, Some("seaport")));
        assert!(!should_push_loaded_skybox_to_db(None, None));
    }

    #[test]
    fn sync_cache_active_from_skybox_aligns_save_source_with_schematic() {
        let mut cache = SkyboxCache::empty(PathBuf::from("manifest.ron"));
        cache.active = Some("desert_night".to_string());

        sync_cache_active_from_skybox(
            &mut cache,
            Some(&SkyboxConfig {
                name: "seaport".to_string(),
            }),
        );
        assert_eq!(cache.active.as_deref(), Some("seaport"));

        sync_cache_active_from_skybox(&mut cache, None);
        assert!(cache.active.is_none());
    }
}
