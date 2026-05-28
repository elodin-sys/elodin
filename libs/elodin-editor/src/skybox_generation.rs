use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_ai_skybox::prelude::{
    SetActiveSkybox, SkyboxGenerated, SkyboxGenerationPhase, SkyboxGenerationUi,
};
use impeller2_bevy::PacketTx;
use impeller2_kdl::ToKdl;
use impeller2_wkt::{SetDbConfig, SkyboxConfig};
use std::collections::VecDeque;

use crate::plugins::kdl_document::{
    CurrentDocument, DocumentLoaded, DocumentReloaded, LastSyncedSchematicContent,
    SchematicDocumentAsset, sync_document_skybox,
};
use crate::ui::schematic::CurrentSchematic;

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

pub(crate) fn sync_skybox_to_document_and_db(
    skybox: Option<SkyboxConfig>,
    schematic: &mut CurrentSchematic,
    current_document: &mut CurrentDocument,
    document_assets: &mut Assets<SchematicDocumentAsset>,
    last_synced_content: &mut LastSyncedSchematicContent,
    locally_pushed: &mut LocallyPushedSkyboxActive,
    tx: &PacketTx,
) {
    let kdl = sync_document_skybox(skybox, current_document, document_assets, schematic);
    record_synced_schematic_content(last_synced_content, &kdl);
    let active = schematic.skybox.as_ref().map(|entry| entry.name.as_str());
    push_skybox_db_sync(tx, Some(kdl), active, locally_pushed);
}

pub(crate) fn sync_generated_skybox_to_schematic(
    mut reader: MessageReader<SkyboxGenerated>,
    mut schematic: ResMut<CurrentSchematic>,
    mut current_document: ResMut<CurrentDocument>,
    mut document_assets: ResMut<Assets<SchematicDocumentAsset>>,
    mut last_synced_content: ResMut<LastSyncedSchematicContent>,
    mut locally_pushed: ResMut<LocallyPushedSkyboxActive>,
    tx: Res<PacketTx>,
) {
    for event in reader.read() {
        let kdl = sync_document_skybox(
            Some(SkyboxConfig {
                name: event.name.clone(),
            }),
            &mut current_document,
            &mut document_assets,
            &mut schematic,
        );
        last_synced_content.0 = Some(kdl.clone());
        push_skybox_db_sync(&tx, Some(kdl), Some(&event.name), &mut locally_pushed);
    }
}

pub(crate) fn record_synced_schematic_content(
    last_synced_content: &mut LastSyncedSchematicContent,
    kdl: &str,
) {
    last_synced_content.0 = Some(kdl.to_string());
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
    tx: Res<'w, PacketTx>,
    skyboxes: MessageWriter<'w, SetActiveSkybox>,
}

pub(crate) fn revert_previous_skybox(mut params: RevertSkyboxParams) {
    let Some(name) = params.ui.revert_name.take() else {
        return;
    };
    let kdl = sync_document_skybox(
        Some(SkyboxConfig { name: name.clone() }),
        &mut params.current_document,
        &mut params.document_assets,
        &mut params.schematic,
    );
    params.last_synced_content.0 = Some(kdl.clone());
    push_skybox_db_sync(
        &params.tx,
        Some(kdl),
        Some(&name),
        &mut params.locally_pushed,
    );
    params.skyboxes.write(SetActiveSkybox::ByName(name.clone()));
    params.ui.phase = SkyboxGenerationPhase::Idle;
    params.ui.message = Some(format!("Reverted to skybox `{name}`"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::kdl_document::{LastSyncedSchematicContent, SchematicDocumentAsset};
    use impeller2_wkt::Schematic;

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
}
