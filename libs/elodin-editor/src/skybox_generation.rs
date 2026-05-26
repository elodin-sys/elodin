use bevy::prelude::*;
use bevy_ai_skybox::prelude::{
    SetActiveSkybox, SkyboxGenerationComplete, SkyboxGenerationPhase, SkyboxGenerationUi,
};
use impeller2_bevy::PacketTx;
use impeller2_wkt::{SetDbConfig, SkyboxConfig};

use crate::plugins::kdl_document::{CurrentDocument, SchematicDocumentAsset, sync_document_skybox};
use crate::ui::schematic::CurrentSchematic;

pub fn sync_generated_skybox_to_schematic(
    mut reader: MessageReader<SkyboxGenerationComplete>,
    mut schematic: ResMut<CurrentSchematic>,
    current_document: Res<CurrentDocument>,
    mut document_assets: ResMut<Assets<SchematicDocumentAsset>>,
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
        push_schematic_metadata(&tx, kdl, Some(Some(&event.name)));
    }
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
    tx.send_msg(SetDbConfig {
        metadata: std::collections::HashMap::from([(
            "skybox.active".to_string(),
            name.clone(),
        )]),
        ..Default::default()
    });
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
            if time.elapsed_secs() - start > 12.0 {
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

pub(crate) fn push_schematic_metadata(
    tx: &PacketTx,
    kdl: String,
    skybox: Option<Option<&str>>,
) {
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
    push_schematic_metadata(&tx, kdl, Some(Some(&name)));
    skyboxes.write(SetActiveSkybox::ByName(name.clone()));
    ui.phase = SkyboxGenerationPhase::Idle;
    ui.message = Some(format!("Reverted to skybox `{name}`"));
}
