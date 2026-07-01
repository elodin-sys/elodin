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
    LastSyncedAssetsRevision, PendingActiveSchematic, SchematicDocumentAsset, sync_document_skybox,
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
        // Drop the echoed state and every older push it superseded. A coalesced
        // `DbConfig` echo can skip intermediate states, so removing only the
        // matched entry could leave a stale older push as the deque tail and
        // mislead `latest_supersedes`.
        self.pending.drain(..=index);
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

    /// Whether the most recent locally pushed desired state (still awaiting its
    /// DB echo) differs from `skybox`. When true, a mirror action targeting
    /// `skybox` is stale: the user has since moved to another skybox (or a
    /// clear) that the live `DbConfig` hasn't caught up to yet, so applying or
    /// starting a download for `skybox` would resurrect a superseded state.
    pub(crate) fn latest_supersedes(&self, skybox: Option<&str>) -> bool {
        self.pending
            .back()
            .is_some_and(|latest| latest.as_deref() != skybox)
    }
}

pub(crate) fn sync_cache_active_from_skybox(
    cache: &mut SkyboxCache,
    skybox: Option<&SkyboxConfig>,
) {
    cache.active = skybox.map(|entry| entry.name.clone());
}

/// The DB key a skybox edit should write to. Prefers an in-flight repoint pin
/// (`PendingActiveSchematic`) then the last synced key over the echoed
/// `DbConfig`, so a skybox edit made right after Save As / Open (before the
/// repoint echo lands) targets the schematic the user is actually on rather
/// than the stale/default `schematics/main.kdl`.
pub(crate) fn active_write_key(
    pending_active: &PendingActiveSchematic,
    last_synced: &LastSyncedActiveKey,
    config: &DbConfig,
) -> String {
    pending_active
        .target
        .clone()
        .or_else(|| last_synced.0.clone())
        .or_else(|| config.schematic_active().map(str::to_string))
        .unwrap_or_else(|| ACTIVE_SCHEMATIC_KEY.to_string())
}

#[derive(SystemParam)]
pub(crate) struct SyncGeneratedSkyboxParams<'w> {
    schematic: ResMut<'w, CurrentSchematic>,
    current_document: ResMut<'w, CurrentDocument>,
    document_assets: ResMut<'w, Assets<SchematicDocumentAsset>>,
    last_synced_key: ResMut<'w, LastSyncedActiveKey>,
    locally_pushed: ResMut<'w, LocallyPushedSkyboxActive>,
    cache: ResMut<'w, SkyboxCache>,
    pending_active: Res<'w, PendingActiveSchematic>,
    config: Res<'w, DbConfig>,
    tx: Res<'w, PacketTx>,
}

pub(crate) fn sync_generated_skybox_to_schematic(
    mut reader: MessageReader<SkyboxGenerated>,
    mut params: SyncGeneratedSkyboxParams,
) {
    for event in reader.read() {
        let write_key = active_write_key(
            &params.pending_active,
            &params.last_synced_key,
            &params.config,
        );
        let mut sync = SkyboxDocumentSyncMut {
            schematic: &mut params.schematic,
            current_document: &mut params.current_document,
            document_assets: &mut params.document_assets,
            last_synced_key: &mut params.last_synced_key,
            locally_pushed: &mut params.locally_pushed,
            cache: &mut params.cache,
            tx: &params.tx,
            active_key: &write_key,
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
    mut last_synced_revision: Option<ResMut<LastSyncedAssetsRevision>>,
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

    // An explicit clear (`skybox.active=""` → `Some(None)`) is sticky: don't let
    // a reloaded asset that still carries a `skybox` node silently re-add it and
    // undo the clear for other clients / the render server. For an unset or named
    // skybox, the loaded schematic seeds/updates the DB metadata as before.
    let clear_is_sticky = matches!(config.skybox_active_desired(), Some(None));
    if !clear_is_sticky
        && should_push_loaded_skybox_to_db(loaded_skybox, config.skybox_active())
        && let Some(tx) = tx.as_ref()
    {
        let kdl = document.root.to_kdl();
        // Write the skybox back to the *loaded* document's own asset key, not the
        // DB's current `schematic.active`: an HTTP load can finish before the
        // repoint echoes, so `config.schematic_active()` may still point at the
        // previous schematic and we'd stamp these bytes onto the wrong asset
        // (RFD #724).
        let active_key = event
            .save_path
            .as_deref()
            .and_then(std::path::Path::to_str)
            .or_else(|| config.schematic_active())
            .unwrap_or(ACTIVE_SCHEMATIC_KEY);
        push_skybox_db_sync(
            tx,
            Some(kdl),
            loaded_skybox,
            active_key,
            &mut locally_pushed,
        );
    }

    last_synced_key.0 = config.schematic_active().map(str::to_string);
    // Record the revision we just loaded so config sync only reloads on a later
    // byte change at this key (RFD #724, Bug 1).
    if let Some(revision) = last_synced_revision.as_deref_mut() {
        revision.revision = Some(config.assets_revision());
        revision.suppress_next = false;
    }
}

pub(crate) fn record_reloaded_schematic_key(
    mut reloaded: MessageReader<DocumentReloaded>,
    config: Res<DbConfig>,
    mut last_synced_key: ResMut<LastSyncedActiveKey>,
    mut last_synced_revision: Option<ResMut<LastSyncedAssetsRevision>>,
) {
    if reloaded.read().last().is_some() {
        last_synced_key.0 = config.schematic_active().map(str::to_string);
        if let Some(revision) = last_synced_revision.as_deref_mut() {
            revision.revision = Some(config.assets_revision());
            revision.suppress_next = false;
        }
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
    pending_active: Res<'w, PendingActiveSchematic>,
    config: Res<'w, DbConfig>,
    tx: Res<'w, PacketTx>,
    skyboxes: MessageWriter<'w, SetActiveSkybox>,
}

pub(crate) fn revert_previous_skybox(mut params: RevertSkyboxParams) {
    let Some(name) = params.ui.revert_name.take() else {
        return;
    };
    let skybox = SkyboxConfig { name: name.clone() };
    let write_key = active_write_key(
        &params.pending_active,
        &params.last_synced_key,
        &params.config,
    );
    let mut sync = SkyboxDocumentSyncMut {
        schematic: &mut params.schematic,
        current_document: &mut params.current_document,
        document_assets: &mut params.document_assets,
        last_synced_key: &mut params.last_synced_key,
        locally_pushed: &mut params.locally_pushed,
        cache: &mut params.cache,
        tx: &params.tx,
        active_key: &write_key,
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
    fn on_document_loaded_records_active_key() {
        let mut app = App::new();
        let mut config = DbConfig::default();
        config.set_schematic_active("schematics/main.kdl");
        app.add_plugins(MinimalPlugins)
            .init_resource::<LastSyncedActiveKey>()
            .insert_resource(config)
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

        // Phase 4: `last_synced` tracks the active *key*, not inline KDL, so the
        // DB's echo of our own load isn't mistaken for an external change.
        let last = app.world().resource::<LastSyncedActiveKey>();
        assert_eq!(last.0.as_deref(), Some("schematics/main.kdl"));
    }

    #[test]
    fn latest_supersedes_flags_newer_local_push() {
        let mut pushed = LocallyPushedSkyboxActive::default();
        // Nothing pending: never supersedes.
        assert!(!pushed.latest_supersedes(Some("seaport")));
        assert!(!pushed.latest_supersedes(None));

        // Latest push is a clear: a download for any named skybox is stale.
        pushed.mark(Some("seaport"));
        pushed.mark(None);
        assert!(pushed.latest_supersedes(Some("seaport")));
        // ...but a mirror action matching the latest state is not superseded.
        assert!(!pushed.latest_supersedes(None));

        // Latest push is a name: a different name (or clear) is stale.
        pushed.mark(Some("machu_picchu"));
        assert!(pushed.latest_supersedes(Some("seaport")));
        assert!(pushed.latest_supersedes(None));
        assert!(!pushed.latest_supersedes(Some("machu_picchu")));
    }

    #[test]
    fn consume_matching_drops_superseded_older_pushes() {
        let mut pushed = LocallyPushedSkyboxActive::default();
        pushed.mark(Some("a"));
        pushed.mark(None);
        pushed.mark(Some("b"));

        // A coalesced echo jumps straight to the latest state; older pushes must
        // be drained so the tail reflects reality rather than a stale entry.
        assert!(pushed.consume_matching(Some("b")));
        assert!(!pushed.latest_supersedes(Some("b")));
        assert!(!pushed.is_pending(Some("a")));
        assert!(!pushed.is_pending(None));
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
