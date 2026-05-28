use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_ai_skybox::prelude::{
    SetActiveSkybox, SkyboxCache, SkyboxGenerated, SkyboxGenerationPhase, SkyboxGenerationUi,
};
use impeller2_bevy::PacketTx;
use impeller2_kdl::ToKdl;
use impeller2_wkt::{DbConfig, SetDbConfig, SkyboxConfig};
use std::collections::VecDeque;

use crate::plugins::kdl_document::{
    CurrentDocument, DocumentLoaded, DocumentReloaded, LastSyncedSchematicContent,
    SchematicDocumentAsset, sync_document_skybox,
};
use crate::ui::schematic::CurrentSchematic;

pub(crate) struct SkyboxDocumentSyncMut<'a> {
    pub schematic: &'a mut CurrentSchematic,
    pub current_document: &'a mut CurrentDocument,
    pub document_assets: &'a mut Assets<SchematicDocumentAsset>,
    pub last_synced_content: &'a mut LastSyncedSchematicContent,
    pub locally_pushed: &'a mut LocallyPushedSkyboxActive,
    pub cache: &'a mut SkyboxCache,
    pub tx: &'a PacketTx,
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
        record_synced_schematic_content(self.last_synced_content, &kdl);
        let active = self
            .schematic
            .skybox
            .as_ref()
            .map(|entry| entry.name.as_str());
        push_skybox_db_sync(self.tx, Some(kdl), active, self.locally_pushed);
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
    last_synced_content: ResMut<'w, LastSyncedSchematicContent>,
    locally_pushed: ResMut<'w, LocallyPushedSkyboxActive>,
    cache: ResMut<'w, SkyboxCache>,
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
            last_synced_content: &mut params.last_synced_content,
            locally_pushed: &mut params.locally_pushed,
            cache: &mut params.cache,
            tx: &params.tx,
        };
        sync.sync_skybox_to_document_and_db(Some(SkyboxConfig {
            name: event.name.clone(),
        }));
    }
}

pub(crate) fn record_synced_schematic_content(
    last_synced_content: &mut LastSyncedSchematicContent,
    kdl: &str,
) {
    last_synced_content.0 = Some(kdl.to_string());
}

/// Push loaded document skybox metadata to the DB when it differs from the current config.
/// Editor viewports update locally on load; the render-server (sensor_view) reads `skybox.active`.
pub(crate) fn sync_loaded_document_skybox_to_db(
    mut loaded: MessageReader<DocumentLoaded>,
    config: Res<DbConfig>,
    mut locally_pushed: ResMut<LocallyPushedSkyboxActive>,
    tx: Res<PacketTx>,
) {
    let Some(document) = loaded.read().last().map(|event| &event.document) else {
        return;
    };
    let loaded_skybox = document
        .root
        .skybox
        .as_ref()
        .map(|entry| entry.name.as_str());
    if !should_push_loaded_skybox_to_db(loaded_skybox, config.skybox_active()) {
        return;
    }
    let kdl = document.root.to_kdl();
    push_skybox_db_sync(&tx, Some(kdl), loaded_skybox, &mut locally_pushed);
}

pub(crate) fn record_loaded_schematic_content(
    mut loaded: MessageReader<DocumentLoaded>,
    mut reloaded: MessageReader<DocumentReloaded>,
    mut last_synced_content: ResMut<LastSyncedSchematicContent>,
) {
    let loaded = loaded.read().map(|event| &event.document);
    let reloaded = reloaded.read().map(|event| &event.document);
    let Some(document) = loaded.chain(reloaded).last() else {
        return;
    };

    record_synced_schematic_content(&mut last_synced_content, &document.root.to_kdl());
}

/// Push skybox metadata to the DB. When `kdl` is `None`, only `skybox.active` is updated
/// (used while a generated cubemap is still loading on the render server).
pub(crate) fn push_skybox_db_sync(
    tx: &PacketTx,
    kdl: Option<String>,
    skybox: Option<&str>,
    locally_pushed: &mut LocallyPushedSkyboxActive,
) {
    locally_pushed.mark(skybox);
    push_schematic_metadata(tx, kdl, skybox);
}

pub(crate) fn push_schematic_metadata(tx: &PacketTx, kdl: Option<String>, skybox: Option<&str>) {
    let mut metadata = std::collections::HashMap::new();
    if let Some(kdl) = kdl {
        metadata.insert("schematic.content".to_string(), kdl);
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
    push_skybox_db_sync(&tx, None, Some(&name), &mut locally_pushed);
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
    last_synced_content: ResMut<'w, LastSyncedSchematicContent>,
    locally_pushed: ResMut<'w, LocallyPushedSkyboxActive>,
    cache: ResMut<'w, SkyboxCache>,
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
        last_synced_content: &mut params.last_synced_content,
        locally_pushed: &mut params.locally_pushed,
        cache: &mut params.cache,
        tx: &params.tx,
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
    use crate::plugins::kdl_document::{LastSyncedSchematicContent, SchematicDocumentAsset};
    use impeller2_wkt::Schematic;
    use std::path::PathBuf;

    #[test]
    fn record_loaded_schematic_content_updates_last_synced_without_db_push() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<LastSyncedSchematicContent>()
            .add_message::<DocumentLoaded>()
            .add_message::<DocumentReloaded>()
            .add_systems(Update, record_loaded_schematic_content);

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

        let last = app.world().resource::<LastSyncedSchematicContent>();
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
        cache.active = Some("alien_swamp".to_string());

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
