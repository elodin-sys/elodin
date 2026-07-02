//! DB-centric asset ingest (RFD #724, Phase 0).
//!
//! A single boundary copies a source `assets/` tree into `{db}/assets/` exactly
//! once, at DB creation. Downstream the simulation and editor are pure network
//! consumers of the DB Asset Server. This replaces the selective, type-aware copy
//! that previously lived in `nox-py`.

use std::io;
use std::path::{Component, Path, PathBuf};

use crate::assets_http::{assets_dir, write_asset_file};

/// Marker written into `{db}/assets/` after a successful ingest. Its presence is
/// the copy-once guard and carries provenance for `elodin-db info`.
pub const INGEST_MARKER: &str = ".elodin-ingested";

/// Result of an [`ingest_asset_dir`] call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IngestReport {
    /// Source tree that was copied in.
    pub source_root: PathBuf,
    /// Number of files copied (0 when skipped).
    pub file_count: usize,
    /// Total bytes copied (0 when skipped).
    pub byte_count: u64,
    /// True when the copy-once guard tripped and nothing was copied.
    #[serde(default)]
    pub skipped: bool,
}

/// A single entry in the asset index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssetEntry {
    pub key: String,
    pub size: u64,
}

/// Resolve the source assets root to ingest, in priority order:
///   1. `$ELODIN_ASSETS` (explicit; absolute or cwd-relative)
///   2. `<entry_dir>/assets` (next to the sim entry / `main.py`, when known)
///   3. `<cwd>/assets`
///   4. nearest ancestor of `entry_dir` containing an `assets/` directory
///
/// `entry` is the simulation entrypoint (a file or directory). An explicit
/// `$ELODIN_ASSETS` is trusted and returned even if it does not exist yet; the
/// other candidates are only returned when they exist.
pub fn resolve_assets_root(entry: Option<&Path>) -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("ELODIN_ASSETS") {
        let path = PathBuf::from(explicit);
        return Some(if path.is_absolute() {
            path
        } else {
            std::env::current_dir().unwrap_or_default().join(path)
        });
    }

    let entry_dir = entry.and_then(|entry| {
        if entry.is_dir() {
            Some(entry.to_path_buf())
        } else {
            entry.parent().map(Path::to_path_buf)
        }
    });

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = &entry_dir {
        candidates.push(dir.join("assets"));
    }
    if let Ok(cwd) = std::env::current_dir() {
        let cwd_assets = cwd.join("assets");
        if !candidates.contains(&cwd_assets) {
            candidates.push(cwd_assets);
        }
    }
    if let Some(dir) = &entry_dir {
        for ancestor in dir.ancestors() {
            let candidate = ancestor.join("assets");
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
    }

    candidates.into_iter().find(|candidate| candidate.is_dir())
}

/// Copy every file under `source_root` into `{db}/assets/`, preserving relative
/// paths. Path traversal is rejected by the shared `write_asset_file` sanitizer.
///
/// Ingest is a *one-time bootstrap into an empty DB*, not a per-run sync: it
/// seeds a brand-new DB from a local `assets/` tree and is then a no-op forever.
/// It is invoked on every `elodin-db run` / `world.run(...)`, including reopens,
/// so it must be safe against a DB that already owns assets. Two guards enforce
/// copy-once and, above all, **never destroy existing data**:
///
///   1. The ingest marker is present → a prior ingest committed; skip.
///   2. The destination already holds real assets (any non-dotfile) → the DB is
///      already populated by some legitimate path — recorded simulation output,
///      editor `PUT`/`StoreAsset`, or a legacy DB created before ingest existed
///      (which have no marker) — so leave it untouched and skip. This is what
///      keeps `elodin-db run <existing-db>` from wiping recorded assets just
///      because a stray `./assets` resolves in the cwd.
///
/// Only when the destination has no real assets do we (re)create it and copy.
/// The trade-off: a rare ingest that crashed part-way (files written, marker
/// not) is treated as "already populated" and left as-is rather than wiped and
/// retried; recovering a partial seed is a `--reset`, which is far preferable to
/// silently deleting a user's recorded assets.
pub fn ingest_asset_dir(db_path: &Path, source_root: &Path) -> io::Result<IngestReport> {
    let dest = assets_dir(db_path);
    let skipped = || {
        Ok(IngestReport {
            source_root: source_root.to_path_buf(),
            file_count: 0,
            byte_count: 0,
            skipped: true,
        })
    };

    if dest.join(INGEST_MARKER).exists() {
        return skipped();
    }
    // `index_assets_in` excludes dotfiles, so a present-but-uncommitted marker
    // (or other dotfiles) alone does not count as "populated"; any real asset
    // does, and we must not clobber it.
    if !index_assets_in(&dest, None)?.is_empty() {
        return skipped();
    }

    if !source_root.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "assets source root does not exist: {}",
                source_root.display()
            ),
        ));
    }

    // Destination carries no real assets: safe to start from a clean slate.
    // Removing here only clears dotfile-only leftovers (e.g. a stray marker) —
    // the guard above already bailed out on any actual asset — so a partial run
    // from a prior crash cannot merge into this (possibly different) tree.
    if dest.exists() {
        std::fs::remove_dir_all(&dest)?;
    }
    std::fs::create_dir_all(&dest)?;

    let mut keys = Vec::new();
    collect_relative_keys(source_root, source_root, &mut keys)?;
    // Skip keys reserved by the asset layer (the `.elodin-ingested` marker and
    // the `__index__` listing namespace). Copying them would either trip the
    // copy-once guard on a partial tree or shadow a stored asset behind the
    // index route; `write_asset_file` rejects them anyway, so they must be
    // filtered here to avoid aborting the whole ingest.
    keys.retain(|key| !crate::assets_http::is_reserved_asset_key(key));

    let mut byte_count = 0u64;
    for key in &keys {
        let data = std::fs::read(source_root.join(key))?;
        byte_count += data.len() as u64;
        write_asset_file(&dest, key, &data)?;
    }

    // Every schematic now lives in the DB, so rewrite the local asset paths
    // inside each stored `.kdl` (window sub-schematics included) to `db:` refs.
    // Done before the marker so a genuine I/O failure aborts without committing
    // copy-once. The partial tree it leaves behind is then preserved (not wiped)
    // by the "already populated" guard on the next run; recover with `--reset`.
    rewrite_stored_schematics(&dest)?;

    let report = IngestReport {
        source_root: source_root.to_path_buf(),
        file_count: keys.len(),
        byte_count,
        skipped: false,
    };
    write_ingest_marker(&dest.join(INGEST_MARKER), &report)?;
    Ok(report)
}

/// Rewrite local asset paths in a single schematic KDL document to `db:`
/// references, keeping only those whose bytes are present under `assets_dir`.
///
/// Returns the rewritten KDL, or `None` when the content is not a valid
/// schematic (callers store it verbatim) or already needs no change. This is the
/// single rewrite entry point shared by tree ingest and `StoreAsset` uploads, so
/// every `.kdl` that enters the DB routes its assets through the Asset Server
/// regardless of how it got there. Idempotent: a path already in `db:` form maps
/// to `None` via `local_asset_name` and is left as-is.
pub fn rewrite_schematic_kdl_to_db(assets_dir: &Path, content: &str) -> Option<String> {
    let mut schematic = impeller2_kdl::parse_schematic(content).ok()?;
    impeller2_kdl::rewrite_asset_paths(&mut schematic, |path| {
        let name = impeller2_kdl::local_asset_name(path)?;
        resolve_stored_asset_key(assets_dir, &name).map(|key| format!("db:{key}"))
    });
    let serialized = impeller2_kdl::serialize_schematic(&schematic);
    (serialized != content).then_some(serialized)
}

/// Resolve the stored asset key for a local path referenced by a schematic to
/// the key actually present under `assets_dir`, or `None` to leave it local.
///
/// A direct hit wins. A `.kdl` window reference that is not already under
/// `schematics/` also resolves there: window sub-schematics live in
/// `schematics/` but are referenced relative to their sibling schematic (often a
/// bare `telemetry.kdl`), so without this they would stay local and fail to load
/// over HTTP after ingest.
fn resolve_stored_asset_key(assets_dir: &Path, name: &str) -> Option<String> {
    if assets_dir.join(name).is_file() {
        return Some(name.to_string());
    }
    if name.ends_with(".kdl") && !name.starts_with("schematics/") {
        let prefixed = format!("schematics/{name}");
        if assets_dir.join(&prefixed).is_file() {
            return Some(prefixed);
        }
    }
    None
}

/// Rewrite local asset paths inside every stored `.kdl` schematic under `dir` to
/// `db:` references, so nested `glb`/`icon` paths in window sub-schematics also
/// resolve through the DB Asset Server. Only paths whose bytes are actually
/// present in the tree are rewritten (mirrors the active-schematic rewrite).
///
/// A `.kdl` that is not a valid schematic is intentionally left untouched (a
/// non-schematic file is not an error). Genuine I/O failures (read/write) are
/// propagated so the caller does not commit the copy-once marker over a tree
/// whose schematics were only partially rewritten.
fn rewrite_stored_schematics(dir: &Path) -> io::Result<()> {
    let mut keys = Vec::new();
    collect_relative_keys(dir, dir, &mut keys)?;
    for key in keys {
        if !key.ends_with(".kdl") {
            continue;
        }
        let content = std::fs::read_to_string(dir.join(&key))?;
        if let Some(serialized) = rewrite_schematic_kdl_to_db(dir, &content) {
            write_asset_file(dir, &key, serialized.as_bytes())?;
        }
    }
    Ok(())
}

/// Read the ingest provenance marker, if present.
pub fn ingest_report(db_path: &Path) -> Option<IngestReport> {
    let marker = assets_dir(db_path).join(INGEST_MARKER);
    let bytes = std::fs::read(marker).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// True once an asset tree has been ingested into this DB.
pub fn assets_ingested(db_path: &Path) -> bool {
    assets_dir(db_path).join(INGEST_MARKER).exists()
}

/// List the assets stored under `{db}/assets/`, optionally filtered to a key
/// prefix. Hidden dotfiles (e.g. the ingest marker) are excluded. Keys use
/// forward slashes regardless of platform.
pub fn index_assets(db_path: &Path, prefix: Option<&str>) -> io::Result<Vec<AssetEntry>> {
    index_assets_in(&assets_dir(db_path), prefix)
}

/// Like [`index_assets`] but operating directly on an `assets/` directory.
pub fn index_assets_in(dir: &Path, prefix: Option<&str>) -> io::Result<Vec<AssetEntry>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut keys = Vec::new();
    collect_relative_keys(dir, dir, &mut keys)?;
    let mut entries = Vec::new();
    for key in keys {
        if key.split('/').any(|segment| segment.starts_with('.')) {
            continue;
        }
        if let Some(prefix) = prefix
            && !key.starts_with(prefix)
        {
            continue;
        }
        let size = std::fs::metadata(dir.join(&key))
            .map(|m| m.len())
            .unwrap_or(0);
        entries.push(AssetEntry { key, size });
    }
    entries.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(entries)
}

fn write_ingest_marker(marker: &Path, report: &IngestReport) -> io::Result<()> {
    let json = serde_json::to_vec_pretty(report).map_err(io::Error::other)?;
    if let Some(parent) = marker.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = marker.with_extension("elodin-upload");
    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, marker)?;
    Ok(())
}

/// Recursively collect file paths under `dir` as `/`-joined keys relative to
/// `root`. Symlinks are skipped to avoid escaping the tree.
fn collect_relative_keys(root: &Path, dir: &Path, out: &mut Vec<String>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        // Skip symlinks so the walk cannot follow a link out of the assets
        // root. `DirEntry::file_type` does not traverse links, so a symlinked
        // dir/file reports neither `is_dir` nor `is_file`; reject it explicitly.
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            collect_relative_keys(root, &path, out)?;
        } else if file_type.is_file() {
            let Ok(rel) = path.strip_prefix(root) else {
                continue;
            };
            let key = rel
                .components()
                .filter_map(|component| match component {
                    Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("/");
            if !key.is_empty() {
                out.push(key);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write(path: &Path, data: &[u8]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, data).unwrap();
    }

    #[test]
    fn ingest_copies_whole_tree_preserving_paths() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        write(&src.join("meshes/rocket.glb"), b"glb");
        write(&src.join("schematics/main.kdl"), b"kdl");
        write(&src.join("terrains/planar/region.toml"), b"toml");

        let db = dir.path().join("db");
        let report = ingest_asset_dir(&db, &src).unwrap();
        assert!(!report.skipped);
        assert_eq!(report.file_count, 3);
        assert_eq!(
            report.byte_count,
            (b"glb".len() + b"kdl".len() + b"toml".len()) as u64
        );

        let assets = assets_dir(&db);
        assert_eq!(
            std::fs::read(assets.join("meshes/rocket.glb")).unwrap(),
            b"glb"
        );
        assert_eq!(
            std::fs::read(assets.join("schematics/main.kdl")).unwrap(),
            b"kdl"
        );
        assert_eq!(
            std::fs::read(assets.join("terrains/planar/region.toml")).unwrap(),
            b"toml"
        );
    }

    #[test]
    fn ingest_is_copy_once() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        write(&src.join("meshes/rocket.glb"), b"glb");
        let db = dir.path().join("db");

        let first = ingest_asset_dir(&db, &src).unwrap();
        assert!(!first.skipped);
        assert!(assets_ingested(&db));

        // Mutating the source after ingest must not change the DB.
        write(&src.join("meshes/rocket.glb"), b"changed");
        let second = ingest_asset_dir(&db, &src).unwrap();
        assert!(second.skipped);
        assert_eq!(
            std::fs::read(assets_dir(&db).join("meshes/rocket.glb")).unwrap(),
            b"glb"
        );
    }

    #[test]
    fn ingest_preserves_existing_assets_without_marker() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("db");

        // A DB that already owns assets but has no ingest marker: recorded
        // simulation output, an editor PUT/StoreAsset, or a legacy DB created
        // before ingest existed. This is the data-loss scenario — a stray
        // `./assets` resolving must never wipe these.
        let dest = assets_dir(&db);
        write(&dest.join("meshes/recorded.glb"), b"recorded");
        assert!(!assets_ingested(&db));

        let src = dir.path().join("src_assets");
        write(&src.join("meshes/new.glb"), b"new");
        let report = ingest_asset_dir(&db, &src).unwrap();

        // Ingest must be a no-op: existing assets kept, nothing copied over them.
        assert!(report.skipped, "populated DB must not be re-ingested");
        assert_eq!(report.file_count, 0);
        assert_eq!(
            std::fs::read(dest.join("meshes/recorded.glb")).unwrap(),
            b"recorded",
            "recorded asset must survive untouched"
        );
        assert!(
            !dest.join("meshes/new.glb").exists(),
            "a populated DB must not be seeded from a local tree"
        );
    }

    #[test]
    fn ingest_seeds_when_only_dotfile_leftovers_present() {
        let dir = tempdir().unwrap();
        let db = dir.path().join("db");

        // Dotfile-only leftovers (e.g. a stray marker from a crashed pre-commit
        // run) are not real assets, so a fresh seed may still proceed and clear
        // them.
        let dest = assets_dir(&db);
        write(&dest.join(".leftover"), b"x");
        assert!(!assets_ingested(&db));

        let src = dir.path().join("src_assets");
        write(&src.join("meshes/new.glb"), b"new");
        let report = ingest_asset_dir(&db, &src).unwrap();

        assert!(!report.skipped);
        assert_eq!(report.file_count, 1);
        assert!(dest.join("meshes/new.glb").is_file());
        assert!(
            !dest.join(".leftover").exists(),
            "dotfile-only leftovers are cleared by a clean seed"
        );
        assert!(assets_ingested(&db));
    }

    #[test]
    fn ingest_ignores_source_file_named_like_marker() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        // A source tree that happens to contain a file with the marker's name.
        write(&src.join(INGEST_MARKER), b"not-our-marker");
        write(&src.join("meshes/a.glb"), b"a");

        let db = dir.path().join("db");
        let report = ingest_asset_dir(&db, &src).unwrap();
        assert!(!report.skipped);
        // The marker-named source file is excluded from the copy.
        assert_eq!(report.file_count, 1);
        assert!(assets_dir(&db).join("meshes/a.glb").is_file());

        // The committed marker is our report, not the source file's content.
        let stored = ingest_report(&db).expect("marker should parse as our report");
        assert_eq!(stored.file_count, 1);

        // And copy-once still holds afterwards.
        let second = ingest_asset_dir(&db, &src).unwrap();
        assert!(second.skipped);
    }

    #[test]
    fn ingest_skips_index_namespace_source_files() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        // A source file under the `__index__` listing namespace would be shadowed
        // by the index route, so it must be dropped rather than stored.
        write(&src.join("__index__/foo.glb"), b"shadow");
        write(&src.join("meshes/a.glb"), b"a");

        let db = dir.path().join("db");
        let report = ingest_asset_dir(&db, &src).unwrap();
        assert!(!report.skipped);
        // Only the real asset is copied; the reserved key is dropped.
        assert_eq!(report.file_count, 1);
        assert!(assets_dir(&db).join("meshes/a.glb").is_file());
        assert!(!assets_dir(&db).join("__index__/foo.glb").exists());
    }

    #[test]
    fn ingest_rewrites_nested_paths_in_window_schematics() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        write(&src.join("meshes/drone.glb"), b"glb");
        // A window sub-schematic with a nested, filesystem-relative glb path.
        write(
            &src.join("schematics/telemetry.kdl"),
            b"object_3d \"drone.world_pos\" {\n    glb path=\"meshes/drone.glb\"\n}\n",
        );

        let db = dir.path().join("db");
        ingest_asset_dir(&db, &src).unwrap();

        // The stored window schematic's nested path now points at the DB asset.
        let stored =
            std::fs::read_to_string(assets_dir(&db).join("schematics/telemetry.kdl")).unwrap();
        assert!(
            stored.contains("path=\"db:meshes/drone.glb\""),
            "nested window asset path should be rewritten to db:, got:\n{stored}"
        );
    }

    #[test]
    fn ingest_rewrites_bare_window_path_under_schematics() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        // Root schematic references a sibling window by bare filename; the window
        // file itself lives alongside it under schematics/.
        write(
            &src.join("schematics/main.kdl"),
            b"window title=\"detail\" path=\"telemetry.kdl\"\n",
        );
        write(&src.join("schematics/telemetry.kdl"), b"viewport\n");

        let db = dir.path().join("db");
        ingest_asset_dir(&db, &src).unwrap();

        let stored = std::fs::read_to_string(assets_dir(&db).join("schematics/main.kdl")).unwrap();
        assert!(
            stored.contains("path=\"db:schematics/telemetry.kdl\""),
            "bare window path should resolve under schematics/, got:\n{stored}"
        );
    }

    #[test]
    fn ingest_leaves_nested_path_local_when_asset_absent() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        // Window references a glb that does not exist in the tree.
        write(
            &src.join("schematics/telemetry.kdl"),
            b"object_3d \"drone.world_pos\" {\n    glb path=\"meshes/missing.glb\"\n}\n",
        );

        let db = dir.path().join("db");
        ingest_asset_dir(&db, &src).unwrap();

        let stored =
            std::fs::read_to_string(assets_dir(&db).join("schematics/telemetry.kdl")).unwrap();
        assert!(
            stored.contains("path=\"meshes/missing.glb\""),
            "absent nested asset must stay local, got:\n{stored}"
        );
        assert!(!stored.contains("db:meshes/missing.glb"));
    }

    #[test]
    fn ingest_commits_marker_despite_unparsable_kdl() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        // Not a schematic: parse fails, but that must not abort ingest.
        write(&src.join("notes/readme.kdl"), b"blah \"not a schematic\"\n");
        write(&src.join("meshes/a.glb"), b"a");

        let db = dir.path().join("db");
        let report = ingest_asset_dir(&db, &src).unwrap();
        assert!(!report.skipped);
        assert!(
            assets_ingested(&db),
            "an unparsable .kdl is non-fatal; the marker must still commit"
        );
        // The non-schematic file is left exactly as-is.
        assert_eq!(
            std::fs::read(assets_dir(&db).join("notes/readme.kdl")).unwrap(),
            b"blah \"not a schematic\"\n"
        );
    }

    #[test]
    fn ingest_missing_source_errors() {
        let dir = tempdir().unwrap();
        let err = ingest_asset_dir(&dir.path().join("db"), &dir.path().join("nope")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn index_excludes_marker_and_filters_prefix() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        write(&src.join("meshes/rocket.glb"), b"glb");
        write(&src.join("schematics/main.kdl"), b"kdl");
        write(&src.join("schematics/overview.kdl"), b"kdl2");
        let db = dir.path().join("db");
        ingest_asset_dir(&db, &src).unwrap();

        let all = index_assets(&db, None).unwrap();
        let keys: Vec<_> = all.iter().map(|e| e.key.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "meshes/rocket.glb",
                "schematics/main.kdl",
                "schematics/overview.kdl"
            ]
        );
        assert!(!keys.iter().any(|k| k.starts_with('.')));

        let schematics = index_assets(&db, Some("schematics/")).unwrap();
        assert_eq!(schematics.len(), 2);
        assert!(schematics.iter().all(|e| e.key.starts_with("schematics/")));
    }

    #[test]
    fn resolve_prefers_explicit_env() {
        temp_env_assets(Some("/tmp/explicit-assets"), || {
            assert_eq!(
                resolve_assets_root(None),
                Some(PathBuf::from("/tmp/explicit-assets"))
            );
        });
    }

    #[test]
    fn resolve_finds_assets_next_to_entry() {
        let dir = tempdir().unwrap();
        let sim = dir.path().join("simulation");
        std::fs::create_dir_all(sim.join("assets")).unwrap();
        let main = sim.join("main.py");
        std::fs::write(&main, b"# sim").unwrap();

        temp_env_assets(None, || {
            assert_eq!(resolve_assets_root(Some(&main)), Some(sim.join("assets")));
        });
    }

    /// Serialize env access for the two tests that read `ELODIN_ASSETS`.
    fn temp_env_assets(value: Option<&str>, f: impl FnOnce()) {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("ELODIN_ASSETS");
        match value {
            Some(value) => unsafe { std::env::set_var("ELODIN_ASSETS", value) },
            None => unsafe { std::env::remove_var("ELODIN_ASSETS") },
        }
        f();
        match prev {
            Some(prev) => unsafe { std::env::set_var("ELODIN_ASSETS", prev) },
            None => unsafe { std::env::remove_var("ELODIN_ASSETS") },
        }
    }
}
