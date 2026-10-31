//! Embedding controls and recording lifecycle messages.

use crate::plugins::kdl_document::{
    DocumentCommandFailed, DocumentLoadFailed, DocumentLoaded, DocumentReloaded,
};
use bevy::prelude::*;
use impeller_bevy::{
    BackfillState, ConnectionAddr, ConnectionStatus, DbEndpoint, SeriesFetchPriority,
    ThreadConnectionStatus,
};
use impeller_wkt::DbConfig;

/// Controls remote actions. Local camera, playback, and layout edits remain available.
/// These controls are a UI contract. The server must enforce its own permissions.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewerCapabilities {
    pub recording_write: bool,
    pub sql_queries: bool,
}

impl Default for ViewerCapabilities {
    fn default() -> Self {
        Self {
            recording_write: true,
            sql_queries: true,
        }
    }
}

impl ViewerCapabilities {
    pub const READ_ONLY: Self = Self {
        recording_write: false,
        sql_queries: false,
    };
}

pub(crate) fn recording_write_enabled(capabilities: Option<Res<ViewerCapabilities>>) -> bool {
    capabilities.is_none_or(|c| c.recording_write)
}

/// Emitted for each connection or recording transition, after the editor updates.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub enum ViewerEvent {
    Connecting,
    Ready,
    Failed(String),
}

#[derive(Default)]
struct RecordingLifecycle {
    identity: Option<(DbEndpoint, Option<i64>, Option<String>)>,
    schematic: bool,
    ready: bool,
    failed: bool,
    last_status: ConnectionStatus,
}

impl RecordingLifecycle {
    fn update(
        &mut self,
        identity: Option<(DbEndpoint, Option<i64>, Option<String>)>,
        status: ConnectionStatus,
        schematic_loaded: bool,
        data_loaded: bool,
    ) -> Vec<ViewerEvent> {
        let mut events = Vec::new();
        if self.identity != identity {
            self.identity = identity;
            self.schematic = false;
            self.ready = false;
            self.failed = false;
            if self.identity.is_some() {
                events.push(ViewerEvent::Connecting);
            }
        }
        if status == ConnectionStatus::Connecting
            && self.last_status != status
            && self.identity.is_some()
            && !events.contains(&ViewerEvent::Connecting)
        {
            self.failed = false;
            events.push(ViewerEvent::Connecting);
        }
        if schematic_loaded {
            self.schematic = true;
            self.failed = false;
        }
        if status != ConnectionStatus::Success {
            if self.last_status != status && status == ConnectionStatus::Error {
                events.push(ViewerEvent::Failed("The database connection closed".into()));
            }
            self.last_status = status;
            self.ready = false;
            return events;
        }
        self.last_status = status;
        let no_schematic = self
            .identity
            .as_ref()
            .is_some_and(|(_, _, key)| key.is_none());
        if !self.ready
            && !self.failed
            && !schematic_loaded
            && (self.schematic || no_schematic)
            && data_loaded
        {
            self.ready = true;
            events.push(ViewerEvent::Ready);
        }
        events
    }
}

pub(crate) fn plugin(app: &mut App) {
    app.add_message::<ViewerEvent>()
        .add_systems(PostUpdate, recording_lifecycle);
}

fn initial_data_loaded(
    backfill: &BackfillState,
    priority: &SeriesFetchPriority,
    schema: &impeller_bevy::ComponentSchemaRegistry,
) -> bool {
    priority
        .high
        .iter()
        .all(|id| !schema.0.contains_key(id) || backfill.is_complete(*id))
}

#[allow(clippy::too_many_arguments)]
fn recording_lifecycle(
    connection: Option<Res<ConnectionAddr>>,
    status: Res<ThreadConnectionStatus>,
    config: Option<Res<DbConfig>>,
    backfill: Res<BackfillState>,
    priority: Res<SeriesFetchPriority>,
    schema: Res<impeller_bevy::ComponentSchemaRegistry>,
    handshake: Res<impeller_bevy::InitialConnectionState>,
    pending_windows: Option<Res<crate::ui::schematic::PendingWindowSchematics>>,
    mut loaded: MessageReader<DocumentLoaded>,
    mut reloaded: MessageReader<DocumentReloaded>,
    mut failed: MessageReader<DocumentLoadFailed>,
    mut commands_failed: MessageReader<DocumentCommandFailed>,
    mut asset_failures: MessageReader<bevy::asset::UntypedAssetLoadFailedEvent>,
    mut events: MessageWriter<ViewerEvent>,
    mut lifecycle: Local<RecordingLifecycle>,
    mut windows_were_pending: Local<bool>,
) {
    let identity = connection.as_ref().map(|c| {
        (
            c.endpoint().clone(),
            config
                .as_ref()
                .and_then(|c| c.time_start_timestamp_micros()),
            config
                .as_ref()
                .and_then(|c| c.schematic_active().map(str::to_owned)),
        )
    });
    let schematic_loaded = loaded.read().count() + reloaded.read().count() > 0;
    // An empty subscription needs no telemetry fetch. A populated subscription
    // waits for its first complete backfill, including an empty server response.
    let windows_pending = pending_windows.as_ref().is_some_and(|w| !w.is_empty());
    let data_loaded = !windows_pending
        && !*windows_were_pending
        && handshake.config_received
        && handshake.schema_received
        && initial_data_loaded(&backfill, &priority, &schema);
    *windows_were_pending = windows_pending;
    for event in lifecycle.update(identity, status.status(), schematic_loaded, data_loaded) {
        events.write(event);
    }
    for failure in failed.read() {
        lifecycle.ready = false;
        lifecycle.failed = true;
        lifecycle.schematic = false;
        events.write(ViewerEvent::Failed(failure.message.clone()));
    }
    for failure in commands_failed.read().filter(|e| {
        matches!(
            e.title.as_str(),
            "Failed to Load Active Schematic" | "Invalid Active Schematic"
        )
    }) {
        lifecycle.ready = false;
        lifecycle.failed = true;
        lifecycle.schematic = false;
        events.write(ViewerEvent::Failed(failure.message.clone()));
    }
    for failure in asset_failures.read() {
        lifecycle.ready = false;
        lifecycle.failed = true;
        lifecycle.schematic = false;
        events.write(ViewerEvent::Failed(failure.error.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_viewport_components_do_not_block_an_empty_recording() {
        let mut priority = SeriesFetchPriority::default();
        let id = impeller::types::ComponentId(1);
        priority.high.insert(id);
        let mut schema = impeller_bevy::ComponentSchemaRegistry::default();
        let backfill = BackfillState::default();
        assert!(initial_data_loaded(&backfill, &priority, &schema));
        schema.0.insert(
            id,
            impeller::schema::Schema::new(impeller::types::PrimType::F64, Vec::<u64>::new())
                .unwrap(),
        );
        assert!(!initial_data_loaded(&backfill, &priority, &schema));
    }

    #[test]
    fn a_schematic_loaded_before_both_channels_open_still_counts() {
        let mut state = RecordingLifecycle::default();
        state.update(identity("a"), ConnectionStatus::Connecting, true, false);
        assert_eq!(
            state.update(identity("a"), ConnectionStatus::Success, false, true),
            vec![ViewerEvent::Ready]
        );
    }

    #[test]
    fn initial_connection_failure_is_observable_once() {
        let mut state = RecordingLifecycle::default();
        state.update(identity("a"), ConnectionStatus::Connecting, false, false);
        assert!(matches!(
            state
                .update(identity("a"), ConnectionStatus::Error, false, false)
                .as_slice(),
            [ViewerEvent::Failed(_)]
        ));
        assert!(
            state
                .update(identity("a"), ConnectionStatus::Error, false, false)
                .is_empty()
        );
    }
    fn identity(session: &str) -> Option<(DbEndpoint, Option<i64>, Option<String>)> {
        Some((
            impeller_bevy::BrowserEndpoints::new(
                &format!("wss://example.com/{session}/main"),
                &format!("wss://example.com/{session}/messages"),
                &format!("https://example.com/{session}/assets"),
            )
            .unwrap()
            .into(),
            Some(10),
            Some("schematics/main.kdl".into()),
        ))
    }
    #[test]
    fn readiness_waits_for_schematic_and_initial_data() {
        let mut state = RecordingLifecycle::default();
        assert_eq!(
            state.update(identity("a"), ConnectionStatus::Success, false, false),
            vec![ViewerEvent::Connecting]
        );
        assert!(
            state
                .update(identity("a"), ConnectionStatus::Success, true, false)
                .is_empty()
        );
        assert_eq!(
            state.update(identity("a"), ConnectionStatus::Success, false, true),
            vec![ViewerEvent::Ready]
        );
        assert!(
            state
                .update(identity("a"), ConnectionStatus::Success, false, true)
                .is_empty()
        );
    }
    #[test]
    fn reconnect_and_recording_change_rearm_readiness() {
        let mut state = RecordingLifecycle::default();
        state.update(identity("a"), ConnectionStatus::Success, true, true);
        assert!(matches!(
            state
                .update(identity("a"), ConnectionStatus::Error, false, false)
                .as_slice(),
            [ViewerEvent::Failed(_)]
        ));
        assert_eq!(
            state.update(identity("a"), ConnectionStatus::Success, false, true),
            vec![ViewerEvent::Ready]
        );
        assert_eq!(
            state.update(identity("b"), ConnectionStatus::Success, false, true),
            vec![ViewerEvent::Connecting]
        );
        assert!(
            state
                .update(identity("b"), ConnectionStatus::Success, true, true)
                .is_empty()
        );
        assert_eq!(
            state.update(identity("b"), ConnectionStatus::Success, false, true),
            vec![ViewerEvent::Ready]
        );
    }
}
