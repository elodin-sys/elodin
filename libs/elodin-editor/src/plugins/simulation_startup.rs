//! Release the simulation startup gate once the editor can display its schematic.
use bevy::prelude::*;

pub(crate) fn release_simulation(
    config: Res<impeller_wkt::DbConfig>,
    packet_tx: Res<impeller_bevy::PacketTx>,
    mut loaded: MessageReader<crate::ui::schematic::DocumentLoaded>,
    mut document_loaded: Local<bool>,
) {
    *document_loaded |= loaded.read().next().is_some();
    if !config
        .metadata
        .contains_key(impeller_wkt::EDITOR_STARTUP_GATE_METADATA_KEY)
    {
        return;
    }
    if config.schematic_active().is_some() && !*document_loaded {
        return;
    }
    packet_tx.send_msg(impeller_wkt::SetDbConfig {
        recording: Some(true),
        metadata: [(
            impeller_wkt::EDITOR_STARTUP_GATE_METADATA_KEY.to_string(),
            String::new(),
        )]
        .into(),
    });
}
