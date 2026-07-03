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
/// Number of fast fetch attempts for one active-schematic key before the
/// failure is surfaced. Covers the transition window where the asset is still
/// being synced/uploaded (follower lag, slow `PUT`).
const MAX_ACTIVE_FETCH_ATTEMPTS: u32 = 10;

/// Delay between fast active-schematic fetch attempts.
const ACTIVE_FETCH_RETRY_DELAY: Duration = Duration::from_millis(400);

/// Retry cadence once the fast attempts are exhausted. The failure is surfaced
/// once and retries then continue at this slower pace instead of stranding the
/// editor: `schematic.active` still names this key, so it must eventually load
/// (e.g. a follower whose mirror outlasts the fast budget) (Bug 3).
const ACTIVE_FETCH_SLOW_RETRY_DELAY: Duration = Duration::from_secs(5);

/// Total attempt budget for a *gated* refetch whose bytes keep coming back
/// unchanged. On a follower the fetched bytes can predate the mirror of the
/// revision that triggered the refetch, so "unchanged" is retried (fast, then
/// slow) before the bump is finally deemed unrelated and the revision baseline
/// adopted (Bug 1). Bounded — unlike failure retries — so an unrelated bump
/// (skybox cubemap, mesh `PUT`) doesn't refetch the schematic forever.
const MAX_GATED_UNCHANGED_ATTEMPTS: u32 = 2 * MAX_ACTIVE_FETCH_ATTEMPTS;

#[derive(Resource, Default)]
pub(crate) struct ActiveSchematicFetch {
    key: Option<String>,
    /// Attempt number (0-based) of the in-flight fetch, so a transient failure
    /// can schedule the next retry.
    attempts: u32,
    task: Option<Task<ActiveSchematicFetched>>,
    /// Scheduled re-attempt after a transient failure.
    retry: Option<PendingRetry>,
    /// A new request for the *same* key arrived while a fetch was in flight
    /// (e.g. the asset bytes were replaced under an unchanged `schematic.active`).
    /// The in-flight result may be stale, so re-fetch fresh bytes (with this
    /// newer request) on completion instead of applying it.
    refetch_pending: Option<OpenDocumentFromActiveRequest>,
}

/// A scheduled re-attempt of an active-schematic fetch whose asset was not yet
/// available (e.g. a follower still mirroring it, or an in-flight upload).
struct PendingRetry {
    request: OpenDocumentFromActiveRequest,
    attempts: u32,
    next_at: Instant,
}

/// Result of one active-schematic fetch pass.
struct FetchedActiveSchematic {
    content: String,
    /// Whether any `db:` window sub-schematic referenced by the fetched root
    /// differs from the recorded stored contents. Computed only for gated
    /// refetches whose root bytes are unchanged: a remote save that touched
    /// only a window leaves the root identical, so gating on the root alone
    /// would make that change invisible (Bug 2).
    windows_changed: bool,
}

struct ActiveSchematicFetched {
    request: OpenDocumentFromActiveRequest,
    result: Result<FetchedActiveSchematic, String>,
}

/// Whether the in-flight fetch already covers `request`, so a new request must
/// not supersede (and cancel) it. A repeated request for the same key is
/// ignored, so unrelated `DbConfig` churn mid-load can't cancel a nearly-
/// complete fetch.
fn fetch_covers_request(fetch_key: Option<&str>, request: &OpenDocumentFromActiveRequest) -> bool {
    fetch_key == Some(request.key.as_str())
}

/// After a gated refetch returned unchanged bytes, the next attempt count to
/// retry with, or `None` once the bounded budget is spent (the bump is then
/// deemed unrelated and the revision baseline adopted).
fn next_gated_unchanged_attempt(attempts: u32) -> Option<u32> {
    (attempts + 1 < MAX_GATED_UNCHANGED_ATTEMPTS).then_some(attempts + 1)
}

/// Delay before the retry carrying `attempts`: fast within the initial budget,
/// slow afterwards.
fn fetch_retry_delay(attempts: u32) -> Duration {
    if attempts >= MAX_ACTIVE_FETCH_ATTEMPTS {
        ACTIVE_FETCH_SLOW_RETRY_DELAY
    } else {
        ACTIVE_FETCH_RETRY_DELAY
    }
}

/// Failures are surfaced to the user exactly once per request cycle — when the
/// fast attempts run out — while retries keep going at the slow cadence.
fn should_surface_fetch_failure(attempts: u32) -> bool {
    attempts == MAX_ACTIVE_FETCH_ATTEMPTS
}

/// For a gated refetch, decide on the IO pool whether any `db:` window
/// sub-schematic referenced by the fetched root changed relative to the
/// recorded stored contents.
///
/// Skipped (`false`) when the root itself differs from the snapshot: the gate
/// fails on the root alone and a full reload follows regardless, so the window
/// fetches would be wasted. A window that cannot be fetched or was never
/// recorded counts as changed — the reload path (with its own retries) then
/// re-establishes ground truth rather than this gate silently keeping a stale
/// document.
fn gated_windows_changed(
    key: &str,
    root_kdl: &str,
    snapshot: &StoredSchematicSnapshot,
    addr: Option<SocketAddr>,
) -> bool {
    use impeller2_kdl::FromKdl;
    if !snapshot.root_matches(key, root_kdl) {
        return false;
    }
    let Ok(root) = impeller2_wkt::Schematic::from_kdl(root_kdl) else {
        return false;
    };
    for elem in &root.elems {
        let impeller2_wkt::SchematicElem::Window(window) = elem else {
            continue;
        };
        let Some(window_key) = window.path.as_deref().and_then(|p| p.strip_prefix("db:")) else {
            continue;
        };
        let Ok(content) = fetch_active_schematic_kdl(window_key, addr) else {
            return true;
        };
        if !snapshot.window_matches(window_key, &content) {
            return true;
        }
    }
    false
}

/// Spawn an active-schematic fetch on the IO pool; clears any scheduled retry
/// it supersedes. Gated refetches carry a snapshot of the recorded stored
/// contents so the task can also compare window sub-schematics (Bug 2).
fn spawn_active_fetch(
    fetch: &mut ActiveSchematicFetch,
    request: OpenDocumentFromActiveRequest,
    attempts: u32,
    addr: Option<SocketAddr>,
    last_content: &LastActiveSchematicContent,
) {
    let key = request.key.clone();
    let snapshot = request.only_if_changed.then(|| last_content.snapshot());
    fetch.key = Some(request.key.clone());
    fetch.attempts = attempts;
    fetch.retry = None;
    fetch.refetch_pending = None;
    fetch.task = Some(IoTaskPool::get().spawn(async move {
        let result = fetch_active_schematic_kdl(&key, addr).map(|content| {
            let windows_changed = snapshot
                .as_ref()
                .is_some_and(|snapshot| gated_windows_changed(&key, &content, snapshot, addr));
            FetchedActiveSchematic {
                content,
                windows_changed,
            }
        });
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
                    // A file open (`--kdl`, path request) is always deliberate.
                    explicit: true,
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
    mut last_content: ResMut<LastActiveSchematicContent>,
    mut last_synced_revision: Option<ResMut<LastSyncedAssetsRevision>>,
    mut loaded: MessageWriter<DocumentLoaded>,
    mut failed: MessageWriter<DocumentCommandFailed>,
) {
    let addr = connection_addr.as_deref().map(|c| c.0);

    // Latest request wins: a newer active schematic supersedes any in-flight
    // fetch so we never apply a stale load after the user switched schematics.
    // A repeated request for the *same* key doesn't cancel a nearly-complete
    // fetch, but it does mark a re-fetch: the asset bytes at that key may have
    // been replaced (e.g. another client's save under an unchanged
    // `schematic.active`), so the in-flight result could be stale. The newer
    // request is kept for that re-fetch so its intent (e.g. an explicit open
    // that must not be skipped as "unchanged") wins over the in-flight one.
    if let Some(request) = requests.read().last().cloned() {
        let already_fetching = fetch_covers_request(fetch.key.as_deref(), &request);
        if already_fetching {
            fetch.refetch_pending = Some(request);
        } else {
            spawn_active_fetch(&mut fetch, request, 0, addr, &last_content);
        }
    }

    // A scheduled retry (previous attempt failed) is re-spawned once due — unless
    // the user has since pinned a different key.
    if fetch.task.is_none()
        && let Some(retry) = fetch.retry.as_ref()
        && Instant::now() >= retry.next_at
    {
        if pending_active
            .target()
            .is_some_and(|pending| pending != retry.request.key.as_str())
        {
            fetch.retry = None;
        } else {
            let request = retry.request.clone();
            let attempts = retry.attempts;
            spawn_active_fetch(&mut fetch, request, attempts, addr, &last_content);
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
    if let Some(pending) = pending_active.target()
        && pending != request.key.as_str()
    {
        return;
    }

    // A newer request for the same key arrived mid-flight: the bytes we just
    // fetched may already be stale, so re-fetch fresh ones (attempt budget
    // reset) rather than applying this result.
    if let Some(newer) = refetch_pending {
        spawn_active_fetch(&mut fetch, newer, 0, addr, &last_content);
        return;
    }

    let fetched = match result {
        Ok(fetched) => fetched,
        // Fetch failures retry without ever giving up: `schematic.active` still
        // names this key, so it must eventually load. The failure is surfaced
        // once when the fast budget runs out, then retries continue at the slow
        // cadence rather than stranding the editor on a stale document until an
        // unrelated `DbConfig` change (Bug 3).
        Err(error) => {
            let next_attempts = attempts.saturating_add(1);
            if should_surface_fetch_failure(next_attempts) {
                failed.write(DocumentCommandFailed {
                    title: "Failed to Load Active Schematic".to_string(),
                    message: format!("{error} — still retrying in the background"),
                });
            } else {
                bevy::log::debug!(
                    "Active schematic fetch failed ({error}); retrying ({next_attempts})"
                );
            }
            fetch.retry = Some(PendingRetry {
                request,
                attempts: next_attempts,
                next_at: Instant::now() + fetch_retry_delay(next_attempts),
            });
            return;
        }
    };
    // A revision-triggered refetch whose schematic bytes (root and window
    // sub-schematics) are unchanged. On a follower the fetched bytes can
    // predate the mirror of the revision that triggered this refetch, so
    // "unchanged" is not yet proof the bump was unrelated: retry within the
    // bounded budget, and only once it is spent conclude the bump came from an
    // unrelated asset write (skybox cubemap, mesh…) and adopt the revision
    // baseline — never before (Bug 1). Keeping the current document instead of
    // tearing it down matters because a full reload resets the viewport and
    // window layout (Bug 1/2).
    if request.only_if_changed
        && last_content.matches(&request.key, &fetched.content)
        && !fetched.windows_changed
    {
        match next_gated_unchanged_attempt(attempts) {
            Some(next_attempts) => {
                bevy::log::debug!(
                    key = %request.key,
                    "active schematic unchanged after revision bump; re-checking \
                     ({next_attempts}/{MAX_GATED_UNCHANGED_ATTEMPTS})"
                );
                fetch.retry = Some(PendingRetry {
                    request,
                    attempts: next_attempts,
                    next_at: Instant::now() + fetch_retry_delay(next_attempts),
                });
            }
            None => {
                bevy::log::debug!(
                    key = %request.key,
                    "active schematic still unchanged; adopting revision baseline"
                );
                if let (Some(target), Some(revision)) =
                    (request.revision, last_synced_revision.as_deref_mut())
                {
                    revision.revision = Some(target);
                }
            }
        }
        return;
    }
    match open_document_from_content(
        &fetched.content,
        request.save_path.clone(),
        &mut current_document,
    ) {
        Ok(document) => {
            last_content.record(&request.key, &fetched.content);
            loaded.write(DocumentLoaded {
                save_path: current_document.save_path.clone(),
                document,
                explicit: request.explicit,
            });
        }
        // A parse failure is often transient: a multi-`PUT` DB-native save bumps
        // `assets.revision` while its bytes are still landing, so config sync can
        // fetch a torn, momentarily-invalid schematic. Retry like a fetch error
        // (surface once, then keep retrying slowly) instead of only reporting and
        // returning — otherwise `sync_document_from_config` won't reload until
        // `DbConfig` changes again, stranding the editor on the failure.
        Err(error) => {
            let next_attempts = attempts.saturating_add(1);
            if should_surface_fetch_failure(next_attempts) {
                failed.write(DocumentCommandFailed {
                    title: "Invalid Active Schematic".to_string(),
                    message: format!("{error} — still retrying in the background"),
                });
            } else {
                bevy::log::debug!(
                    "Active schematic parse failed ({error}); retrying ({next_attempts})"
                );
            }
            fetch.retry = Some(PendingRetry {
                request,
                attempts: next_attempts,
                next_at: Instant::now() + fetch_retry_delay(next_attempts),
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
    // asset can't resurrect a skybox the DB says was cleared. A user-initiated
    // open is the exception: the user asked for this schematic, skybox
    // included, so the document wins over the earlier clear.
    let clear_is_sticky = config
        .as_deref()
        .is_some_and(|config| matches!(config.skybox_active_desired(), Some(None)));

    for event in loaded.read() {
        let skybox = event
            .document
            .root
            .skybox
            .as_ref()
            .filter(|_| !clear_is_sticky || event.explicit);
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
            only_if_changed: false,
            explicit: false,
            revision: None,
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
    fn fetch_failure_retries_never_give_up() {
        // Fast cadence within the initial budget, slow afterwards — but the
        // retry chain itself is unbounded (Bug 3: the editor must not strand
        // on a stale document because a follower mirror outlasted the budget).
        assert_eq!(fetch_retry_delay(1), ACTIVE_FETCH_RETRY_DELAY);
        assert_eq!(
            fetch_retry_delay(MAX_ACTIVE_FETCH_ATTEMPTS - 1),
            ACTIVE_FETCH_RETRY_DELAY
        );
        assert_eq!(
            fetch_retry_delay(MAX_ACTIVE_FETCH_ATTEMPTS),
            ACTIVE_FETCH_SLOW_RETRY_DELAY
        );
        assert_eq!(fetch_retry_delay(u32::MAX), ACTIVE_FETCH_SLOW_RETRY_DELAY);
    }

    #[test]
    fn fetch_failure_is_surfaced_exactly_once() {
        let surfaced: Vec<u32> = (1..=3 * MAX_ACTIVE_FETCH_ATTEMPTS)
            .filter(|attempts| should_surface_fetch_failure(*attempts))
            .collect();
        assert_eq!(surfaced, vec![MAX_ACTIVE_FETCH_ATTEMPTS]);
    }

    #[test]
    fn gated_unchanged_retries_are_bounded() {
        let mut attempts = 0;
        let mut checks = 1;
        while let Some(next) = next_gated_unchanged_attempt(attempts) {
            attempts = next;
            checks += 1;
        }
        assert_eq!(attempts, MAX_GATED_UNCHANGED_ATTEMPTS - 1);
        assert_eq!(checks, MAX_GATED_UNCHANGED_ATTEMPTS);
    }
}
