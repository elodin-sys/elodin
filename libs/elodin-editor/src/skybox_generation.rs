use bevy::prelude::*;
use bevy_ai_skybox::prelude::{
    SetActiveSkybox, SkyboxGenerated, SkyboxGenerationPhase, SkyboxGenerationUi,
};
use impeller2_bevy::PacketTx;
use impeller2_wkt::{SetDbConfig, SkyboxConfig};

use crate::plugins::kdl_document::{
    CurrentDocument, LastSyncedSchematicContent, SchematicDocumentAsset, sync_document_skybox,
};
use crate::ui::schematic::CurrentSchematic;

pub fn sync_generated_skybox_to_schematic(
    mut reader: MessageReader<SkyboxGenerated>,
    mut schematic: ResMut<CurrentSchematic>,
    current_document: Res<CurrentDocument>,
    mut document_assets: ResMut<Assets<SchematicDocumentAsset>>,
    mut last_synced_content: ResMut<LastSyncedSchematicContent>,
    tx: Res<PacketTx>,
) {
    for event in reader.read() {
        let kdl = sync_document_skybox(
            Some(SkyboxConfig {
                name: event.name.clone(),
            }),
            &current_document,
            &mut document_assets,
            &mut schematic,
        );
        last_synced_content.0 = Some(kdl.clone());
        push_schematic_metadata(&tx, kdl, Some(Some(&event.name)));
    }
}

pub(crate) fn record_synced_schematic_content(
    last_synced_content: &mut LastSyncedSchematicContent,
    kdl: &str,
) {
    last_synced_content.0 = Some(kdl.to_string());
}

pub(crate) fn push_skybox_active_metadata(tx: &PacketTx, skybox: Option<&str>) {
    let metadata = match skybox {
        Some(name) => {
            std::collections::HashMap::from([("skybox.active".to_string(), name.to_string())])
        }
        None => std::collections::HashMap::from([("skybox.active".to_string(), String::new())]),
    };
    tx.send_msg(SetDbConfig {
        metadata,
        ..Default::default()
    });
}

/// Notify render-server while the cubemap is still loading on the editor.
pub fn push_skybox_active_on_pending(
    ui: Res<SkyboxGenerationUi>,
    mut last_pushed: Local<Option<String>>,
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
    push_skybox_active_metadata(&tx, Some(&name));
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

pub(crate) fn push_schematic_metadata(tx: &PacketTx, kdl: String, skybox: Option<Option<&str>>) {
    let mut metadata = std::collections::HashMap::from([("schematic.content".to_string(), kdl)]);
    match skybox {
        Some(Some(name)) => {
            metadata.insert("skybox.active".to_string(), name.to_string());
        }
        Some(None) => {
            metadata.insert("skybox.active".to_string(), String::new());
        }
        None => {}
    }
    tx.send_msg(SetDbConfig {
        metadata,
        ..Default::default()
    });
}

pub fn revert_previous_skybox(
    mut ui: ResMut<SkyboxGenerationUi>,
    mut schematic: ResMut<CurrentSchematic>,
    current_document: Res<CurrentDocument>,
    mut document_assets: ResMut<Assets<SchematicDocumentAsset>>,
    mut last_synced_content: ResMut<LastSyncedSchematicContent>,
    tx: Res<PacketTx>,
    mut skyboxes: MessageWriter<SetActiveSkybox>,
) {
    let Some(name) = ui.revert_name.take() else {
        return;
    };
    let kdl = sync_document_skybox(
        Some(SkyboxConfig { name: name.clone() }),
        &current_document,
        &mut document_assets,
        &mut schematic,
    );
    last_synced_content.0 = Some(kdl.clone());
    push_schematic_metadata(&tx, kdl, Some(Some(&name)));
    skyboxes.write(SetActiveSkybox::ByName(name.clone()));
    ui.phase = SkyboxGenerationPhase::Idle;
    ui.message = Some(format!("Reverted to skybox `{name}`"));
}
