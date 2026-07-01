use bevy::asset::{AssetEvent, AssetLoadFailedEvent};
use bevy::prelude::*;
use bevy::tasks::{IoTaskPool, Task, futures_lite::future};
use bevy_ai_skybox::prelude::{SetActiveSkybox, SkyboxCache};
use impeller2_wkt::{DbConfig, SkyboxConfig};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::commands::*;
use super::messages::*;
use super::operations::{
    fetch_active_schematic_kdl, open_document_from_content, open_document_path,
};
use super::types::*;

/// In-flight async fetch of the active schematic's KDL from the DB Asset Server.
/// Keeps the blocking HTTP request off the main thread (RFD #724): an
/// `OpenDocumentFromActiveRequest` spawns the fetch here and the same system
/// applies the result once it lands, so a slow or unreachable DB never freezes
/// the UI mid-frame.
/// Maximum number of fetch attempts for one active-schematic key before the
/// load is surfaced as a failure. Covers the transition window where the asset
/// is still being synced/uploaded (follower lag, slow `PUT`).
const MAX_ACTIVE_FETCH_ATTEMPTS: u32 = 10;

/// Delay between active-schematic fetch attempts.
const ACTIVE_FETCH_RETRY_DELAY: Duration = Duration::from_millis(400);

#[derive(Resource, Default)]
pub(crate) struct ActiveSchematicFetch {
    key: Option<String>,
    /// Attempt number (0-based) of the in-flight fetch, so a transient failure
    /// can schedule the next bounded retry.
    attempts: u32,
    task: Option<Task<ActiveSchematicFetched>>,
    /// Scheduled re-attempt after a transient failure.
    retry: Option<PendingRetry>,
    /// A new request for the *same* key arrived while a fetch was in flight
    /// (e.g. the asset bytes were replaced under an unchanged `schematic.active`).
    /// The in-flight result may be stale, so re-fetch fresh bytes on completion
    /// instead of applying it.
    refetch_pending: bool,
}

/// A bounded re-attempt of an active-schematic fetch whose asset was not yet
/// available (e.g. a follower still mirroring it, or an in-flight upload).
struct PendingRetry {
    request: OpenDocumentFromActiveRequest,
    attempts: u32,
    next_at: Instant,
}

struct ActiveSchematicFetched {
    request: OpenDocumentFromActiveRequest,
    result: Result<String, String>,
}

/// Whether the in-flight fetch already covers `request`, so a new request must
/// not supersede (and cancel) it. A repeated request for the same key is
/// ignored, so unrelated `DbConfig` churn mid-load can't cancel a nearly-
/// complete fetch.
fn fetch_covers_request(fetch_key: Option<&str>, request: &OpenDocumentFromActiveRequest) -> bool {
    fetch_key == Some(request.key.as_str())
}

/// After a fetch failure, the next attempt count to retry with, or `None` once
/// attempts are exhausted (surface the failure).
fn next_fetch_attempt(attempts: u32) -> Option<u32> {
    (attempts + 1 < MAX_ACTIVE_FETCH_ATTEMPTS).then_some(attempts + 1)
}

/// Spawn an active-schematic fetch on the IO pool; clears any scheduled retry
/// it supersedes.
fn spawn_active_fetch(
    fetch: &mut ActiveSchematicFetch,
    request: OpenDocumentFromActiveRequest,
    attempts: u32,
    addr: Option<SocketAddr>,
) {
    let key = request.key.clone();
    fetch.key = Some(request.key.clone());
    fetch.attempts = attempts;
    fetch.retry = None;
    fetch.refetch_pending = false;
    fetch.task = Some(IoTaskPool::get().spawn(async move {
        let result = fetch_active_schematic_kdl(&key, addr);
        ActiveSchematicFetched { request, result }
    }));
}

fn cloned_current_document_asset(
    current_document: &CurrentDocument,
    document_assets: &Assets<SchematicDocumentAsset>,
) -> Option<(Option<PathBuf>, SchematicDocumentAsset)> {
    let handle = current_document.handle.clone()?;
    let save_path = current_document.save_path.clone();
    let document = document_assets.get(&handle).cloned()?;
    Some((save_path, document))
}

pub(super) fn handle_open_document_requests(
    mut requests: MessageReader<OpenDocumentRequest>,
    asset_server: Res<AssetServer>,
    mut current_document: ResMut<CurrentDocument>,
    mut loaded: MessageWriter<DocumentLoaded>,
    mut failed: MessageWriter<DocumentCommandFailed>,
) {
    for request in requests.read() {
        match open_document_path(&request.0, &asset_server, &mut current_document) {
            Ok(document) => {
                loaded.write(DocumentLoaded {
                    save_path: current_document.save_path.clone(),
                    document,
                });
            }
            Err(error) => {
                failed.write(DocumentCommandFailed {
                    title: format!("Invalid Schematic in {}", request.0.display()),
                    message: error.to_string(),
                });
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_open_document_from_active_requests(
    mut requests: MessageReader<OpenDocumentFromActiveRequest>,
    connection_addr: Option<Res<impeller2_bevy::ConnectionAddr>>,
    pending_active: Res<PendingActiveSchematic>,
    mut fetch: ResMut<ActiveSchematicFetch>,
    mut current_document: ResMut<CurrentDocument>,
    mut loaded: MessageWriter<DocumentLoaded>,
    mut failed: MessageWriter<DocumentCommandFailed>,
) {
    let addr = connection_addr.as_deref().map(|c| c.0);

    // Latest request wins: a newer active schematic supersedes any in-flight
    // fetch so we never apply a stale load after the user switched schematics.
    // A repeated request for the *same* key doesn't cancel a nearly-complete
    // fetch, but it does mark a re-fetch: the asset bytes at that key may have
    // been replaced (e.g. another client's save under an unchanged
    // `schematic.active`), so the in-flight result could be stale.
    if let Some(request) = requests.read().last().cloned() {
        let already_fetching = fetch_covers_request(fetch.key.as_deref(), &request);
        if already_fetching {
            fetch.refetch_pending = true;
        } else {
            spawn_active_fetch(&mut fetch, request, 0, addr);
        }
    }

    // A scheduled retry (previous attempt failed) is re-spawned once due — unless
    // the user has since pinned a different key.
    if fetch.task.is_none()
        && let Some(retry) = fetch.retry.as_ref()
        && Instant::now() >= retry.next_at
    {
        if pending_active
            .0
            .as_deref()
            .is_some_and(|pending| pending != retry.request.key.as_str())
        {
            fetch.retry = None;
        } else {
            let request = retry.request.clone();
            let attempts = retry.attempts;
            spawn_active_fetch(&mut fetch, request, attempts, addr);
        }
    }

    let Some(task) = fetch.task.as_mut() else {
        return;
    };
    let Some(ActiveSchematicFetched { request, result }) =
        future::block_on(future::poll_once(task))
    else {
        return;
    };
    let attempts = fetch.attempts;
    let refetch_pending = std::mem::take(&mut fetch.refetch_pending);
    fetch.task = None;
    fetch.key = None;

    // The user may have switched schematics (via "Save As…"/"Open Schematic…")
    // while this fetch was in flight: `PendingActiveSchematic` pins the key they
    // now want. Drop a completed load for any other key so a late fetch of the
    // previous active schematic can't briefly show — or strand us on — the wrong
    // one before the DB echoes the repoint.
    if let Some(pending) = pending_active.0.as_deref()
        && pending != request.key.as_str()
    {
        return;
    }

    // A newer request for the same key arrived mid-flight: the bytes we just
    // fetched may already be stale, so re-fetch fresh ones (attempt budget
    // reset) rather than applying this result.
    if refetch_pending {
        spawn_active_fetch(&mut fetch, request, 0, addr);
        return;
    }

    let content = match result {
        Ok(content) => content,
        Err(error) => match next_fetch_attempt(attempts) {
            Some(next_attempts) => {
                bevy::log::debug!(
                    "Active schematic fetch failed ({error}); retrying \
                     ({next_attempts}/{MAX_ACTIVE_FETCH_ATTEMPTS})"
                );
                fetch.retry = Some(PendingRetry {
                    request,
                    attempts: next_attempts,
                    next_at: Instant::now() + ACTIVE_FETCH_RETRY_DELAY,
                });
                return;
            }
            None => {
                failed.write(DocumentCommandFailed {
                    title: "Failed to Load Active Schematic".to_string(),
                    message: error,
                });
                return;
            }
        },
    };
    match open_document_from_content(&content, request.save_path.clone(), &mut current_document) {
        Ok(document) => {
            loaded.write(DocumentLoaded {
                save_path: current_document.save_path.clone(),
                document,
            });
        }
        Err(error) => {
            failed.write(DocumentCommandFailed {
                title: "Invalid Active Schematic".to_string(),
                message: error.to_string(),
            });
        }
    }
}

pub(super) fn emit_document_reloads(
    mut events: MessageReader<AssetEvent<SchematicDocumentAsset>>,
    mut current_document: ResMut<CurrentDocument>,
    document_assets: Res<Assets<SchematicDocumentAsset>>,
    mut reloaded: MessageWriter<DocumentReloaded>,
) {
    let window_handles = current_document.window_handles(&document_assets);

    // Lazily expand the suppress set to include window sub-asset IDs once they
    // become available (they aren't known at save time).
    if !current_document.suppress_ids.is_empty() {
        current_document
            .suppress_ids
            .extend(window_handles.iter().copied());
    }

    let mut root_changed = false;
    let mut changed_window_indices = Vec::new();

    for event in events.read() {
        let id = match event {
            AssetEvent::LoadedWithDependencies { id } | AssetEvent::Modified { id } => *id,
            _ => continue,
        };

        let is_root = current_document.matches(id);
        let window_idx = window_handles.iter().position(|wid| *wid == id);

        if !is_root && window_idx.is_none() {
            continue;
        }

        if current_document.suppress_ids.remove(&id) {
            continue;
        }

        if is_root {
            root_changed = true;
        }
        if let Some(i) = window_idx {
            changed_window_indices.push(i);
        }
    }

    if !root_changed && changed_window_indices.is_empty() {
        return;
    }

    // Genuine external changes detected; discard any remaining suppress IDs.
    current_document.suppress_ids.clear();

    // When only window sub-assets changed, report the specific indices.
    // When the root changed, report an empty vec to signal a full reload.
    let changed = if root_changed {
        Vec::new()
    } else {
        changed_window_indices
    };

    if let Some((save_path, document)) =
        cloned_current_document_asset(&current_document, &document_assets)
    {
        reloaded.write(DocumentReloaded {
            save_path,
            document,
            changed_window_indices: changed,
        });
    }
}

pub(super) fn activate_document_skybox(
    mut loaded: MessageReader<DocumentLoaded>,
    mut reloaded: MessageReader<DocumentReloaded>,
    mut skyboxes: MessageWriter<SetActiveSkybox>,
    mut cache: Option<ResMut<SkyboxCache>>,
    config: Option<Res<DbConfig>>,
) {
    // An explicit clear (`skybox.active=""` → `Some(None)`) is sticky: honor it
    // even if the loaded/reloaded KDL still carries a `skybox` node, so a stale
    // asset can't resurrect a skybox the DB says was cleared.
    let clear_is_sticky = config
        .as_deref()
        .is_some_and(|config| matches!(config.skybox_active_desired(), Some(None)));

    for event in loaded.read() {
        let skybox = event
            .document
            .root
            .skybox
            .as_ref()
            .filter(|_| !clear_is_sticky);
        activate_skybox_config(skybox, &mut skyboxes, &mut cache);
    }

    for event in reloaded.read() {
        if !event.changed_window_indices.is_empty() {
            continue;
        }
        let skybox = event
            .document
            .root
            .skybox
            .as_ref()
            .filter(|_| !clear_is_sticky);
        activate_skybox_config(skybox, &mut skyboxes, &mut cache);
    }
}

fn activate_skybox_config(
    skybox: Option<&SkyboxConfig>,
    skyboxes: &mut MessageWriter<SetActiveSkybox>,
    cache: &mut Option<ResMut<SkyboxCache>>,
) {
    if let Some(cache) = cache.as_mut() {
        // Drop stale cache state; bevy_ai_skybox sets `active` once the cubemap is ready.
        cache.active = None;
    }
    match skybox {
        Some(skybox) => {
            skyboxes.write(SetActiveSkybox::ByName(skybox.name.clone()));
        }
        None => {
            skyboxes.write(SetActiveSkybox::Clear);
        }
    }
}

pub(super) fn emit_document_load_failures(
    mut events: MessageReader<AssetLoadFailedEvent<SchematicDocumentAsset>>,
    current_document: Res<CurrentDocument>,
    document_assets: Res<Assets<SchematicDocumentAsset>>,
    mut failed: MessageWriter<DocumentLoadFailed>,
) {
    let window_handles = current_document.window_handles(&document_assets);
    for event in events.read() {
        if !current_document.matches(event.id) && !window_handles.contains(&event.id) {
            continue;
        }
        failed.write(DocumentLoadFailed {
            path: format!("{:?}", event.path),
            message: event.error.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(key: &str) -> OpenDocumentFromActiveRequest {
        OpenDocumentFromActiveRequest {
            key: key.to_string(),
            save_path: None,
        }
    }

    #[test]
    fn in_flight_fetch_ignores_same_key() {
        let req = request("schematics/main.kdl");
        assert!(fetch_covers_request(Some("schematics/main.kdl"), &req,));
    }

    #[test]
    fn in_flight_fetch_superseded_for_different_key() {
        let req = request("schematics/other.kdl");
        assert!(!fetch_covers_request(Some("schematics/main.kdl"), &req,));
    }

    #[test]
    fn no_in_flight_fetch_never_covers() {
        let req = request("schematics/main.kdl");
        assert!(!fetch_covers_request(None, &req));
    }

    #[test]
    fn fetch_retries_until_attempts_exhausted() {
        let mut attempts = 0;
        let mut spawns = 1;
        while let Some(next) = next_fetch_attempt(attempts) {
            attempts = next;
            spawns += 1;
        }
        assert_eq!(attempts, MAX_ACTIVE_FETCH_ATTEMPTS - 1);
        assert_eq!(spawns, MAX_ACTIVE_FETCH_ATTEMPTS);
    }

    #[test]
    fn fetch_gives_up_on_last_attempt() {
        assert_eq!(next_fetch_attempt(MAX_ACTIVE_FETCH_ATTEMPTS - 1), None);
        assert_eq!(
            next_fetch_attempt(MAX_ACTIVE_FETCH_ATTEMPTS - 2),
            Some(MAX_ACTIVE_FETCH_ATTEMPTS - 1)
        );
    }
}
