//! DB-centric asset ingest (RFD #724, Phase 0).
//!
//! A single boundary copies a source `assets/` tree into `{db}/assets/` exactly
//! once, at DB creation. Downstream the simulation and editor are pure network
//! consumers of the DB Asset Server. This replaces the selective, type-aware copy
//! that previously lived in `nox-py`.

use std::collections::HashMap;
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
    /// True when nothing was copied: the copy-once guard tripped, or the
    /// source tree carried no real assets to seed from.
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
/// An empty source tree commits nothing — in particular not the marker — so a
/// DB first opened against a not-yet-populated assets root (e.g. Aleph before
/// the seed lands) still ingests once the host tree gains content.
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

    let mut keys = Vec::new();
    collect_relative_keys(source_root, source_root, &mut keys)?;
    // Skip keys reserved by the asset layer (the `.elodin-ingested` marker and
    // the `__index__` listing namespace). Copying them would either trip the
    // copy-once guard on a partial tree or shadow a stored asset behind the
    // index route; `write_asset_file` rejects them anyway, so they must be
    // filtered here to avoid aborting the whole ingest.
    keys.retain(|key| !crate::assets_http::is_reserved_asset_key(key));

    // Nothing to seed from: do NOT commit the marker. An initially empty source
    // (e.g. Aleph's shared asset root before it is populated) must leave the DB
    // eligible for ingest on a later run, once the host tree has content.
    if keys.is_empty() {
        return skipped();
    }

    // Destination carries no real assets: safe to start from a clean slate.
    // Removing here only clears dotfile-only leftovers (e.g. a stray marker) —
    // the guard above already bailed out on any actual asset — so a partial run
    // from a prior crash cannot merge into this (possibly different) tree.
    if dest.exists() {
        std::fs::remove_dir_all(&dest)?;
    }
    std::fs::create_dir_all(&dest)?;

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

/// Write an uploaded asset under `assets_dir`, routing `.kdl` content through
/// [`rewrite_schematic_kdl_to_db`] first. This is the shared write path for
/// network uploads (Impeller `StoreAsset` and asset HTTP `PUT`), so a schematic
/// gets its local asset paths rewritten to `db:` no matter which transport
/// delivered it. Unparsable or non-schematic `.kdl` is stored verbatim.
pub fn write_uploaded_asset(assets_dir: &Path, key: &str, bytes: &[u8]) -> io::Result<()> {
    if key.ends_with(".kdl")
        && let Ok(content) = std::str::from_utf8(bytes)
        && let Some(rewritten) = rewrite_schematic_kdl_to_db(assets_dir, content)
    {
        write_asset_file(assets_dir, key, rewritten.as_bytes())
    } else {
        write_asset_file(assets_dir, key, bytes)
    }
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

/// Search directories for resolving `window path=` references at record time,
/// mirroring [`resolve_assets_root`]: the sim entry directory, the cwd, then
/// the entry's ancestors — so a reference like `examples/drone/panel.kdl`
/// (repo-root-relative) resolves whether the sim runs from the repo root or
/// from its own directory.
pub fn schematic_window_search_dirs(entry: Option<&Path>) -> Vec<PathBuf> {
    let entry_dir = entry.and_then(|entry| {
        if entry.is_dir() {
            Some(entry.to_path_buf())
        } else {
            entry.parent().map(Path::to_path_buf)
        }
    });

    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(dir) = &entry_dir {
        dirs.push(dir.clone());
    }
    if let Ok(cwd) = std::env::current_dir()
        && !dirs.contains(&cwd)
    {
        dirs.push(cwd);
    }
    if let Some(dir) = &entry_dir {
        for ancestor in dir.ancestors() {
            if !dirs.iter().any(|d| d == ancestor) {
                dirs.push(ancestor.to_path_buf());
            }
        }
    }
    dirs
}

/// Copy the window sub-schematics referenced by local paths in `content` into
/// `assets_dir` and return the content with those references rewritten to
/// `db:` keys (`None` when nothing changed).
///
/// This is the record-time companion of the editor's DB-native save: the
/// stored active schematic must be self-contained, so windows living *outside*
/// the ingested `assets/` tree (e.g. `window path="examples/drone/panel.kdl"`)
/// are pulled in under `schematics/<file>.kdl` — flattened exactly like the
/// editor keys freshly created windows — instead of staying machine-local
/// paths that 404 on a follower or another machine. A reference already
/// resolvable inside the stored tree is skipped here and handled by the normal
/// `store_asset` rewrite.
///
/// Windows are followed transitively (a window may reference further windows,
/// resolved relative to the referencing file first, then `search_dirs`). Each
/// ingested `.kdl` is written through [`write_uploaded_asset`] so its own
/// mesh/icon paths get the same local → `db:` rewrite. Producer-only: network
/// uploads never reach this path (a server has no filesystem to read from).
pub fn ingest_window_schematics(
    assets_dir: &Path,
    content: &str,
    search_dirs: &[PathBuf],
) -> io::Result<Option<String>> {
    let Ok(mut root) = impeller2_kdl::parse_schematic(content) else {
        return Ok(None);
    };

    let mut ingest = WindowIngest {
        assets_dir,
        search_dirs,
        file_keys: HashMap::new(),
    };
    let map = ingest.ingest_referenced_windows(&root, None)?;
    if map.is_empty() {
        return Ok(None);
    }

    impeller2_kdl::rewrite_asset_paths(&mut root, |path| {
        map.get(path).map(|key| format!("db:{key}"))
    });
    let serialized = impeller2_kdl::serialize_schematic(&root);
    Ok((serialized != content).then_some(serialized))
}

struct WindowIngest<'a> {
    assets_dir: &'a Path,
    search_dirs: &'a [PathBuf],
    /// Canonical source file → assigned asset key. Doubles as the cycle guard:
    /// a key is recorded *before* the file's own references are followed, so
    /// mutually referencing windows terminate with each other's key.
    file_keys: HashMap<PathBuf, String>,
}

impl WindowIngest<'_> {
    /// Resolve, ingest and key every local window reference of `schematic`
    /// (itself loaded from `base_dir`, `None` for the root content). Returns
    /// the reference-string → key map to rewrite that schematic with.
    ///
    /// The map is per-schematic on purpose: the same reference string can
    /// resolve to different files from different referencing directories.
    fn ingest_referenced_windows(
        &mut self,
        schematic: &impeller2_wkt::Schematic,
        base_dir: Option<&Path>,
    ) -> io::Result<HashMap<String, String>> {
        let mut map = HashMap::new();
        for elem in &schematic.elems {
            let impeller2_wkt::SchematicElem::Window(window) = elem else {
                continue;
            };
            let Some(reference) = window.path.as_deref() else {
                continue;
            };
            if !impeller2_kdl::is_local_asset_path(reference) || map.contains_key(reference) {
                continue;
            }
            // Already stored in the tree (ingested alongside the assets):
            // the normal `store_asset`/`rewrite_stored_schematics` rewrite
            // covers it; only files outside the tree need pulling in.
            if let Some(name) = impeller2_kdl::local_asset_name(reference)
                && resolve_stored_asset_key(self.assets_dir, &name).is_some()
            {
                continue;
            }
            let Some(source) = resolve_window_source(reference, base_dir, self.search_dirs) else {
                tracing::warn!(
                    window = %reference,
                    "window sub-schematic not found on disk; leaving the local path as-is"
                );
                continue;
            };
            let key = self.ingest_window_file(&source)?;
            map.insert(reference.to_string(), key);
        }
        Ok(map)
    }

    /// Ingest one window file (and, transitively, the windows it references)
    /// and return its assigned asset key. A file already ingested in this pass
    /// reuses its key, so shared sub-windows are stored once.
    fn ingest_window_file(&mut self, source: &Path) -> io::Result<String> {
        let canonical = source
            .canonicalize()
            .unwrap_or_else(|_| source.to_path_buf());
        if let Some(key) = self.file_keys.get(&canonical) {
            return Ok(key.clone());
        }
        let key = self.assign_key(&canonical);
        self.file_keys.insert(canonical.clone(), key.clone());

        let mut bytes = std::fs::read(&canonical)?;
        // An unparsable window file is stored verbatim (matching the editor's
        // save fallback): it still becomes fetchable over HTTP, it just cannot
        // be scanned for further references.
        if let Ok(content) = std::str::from_utf8(&bytes)
            && let Ok(mut schematic) = impeller2_kdl::parse_schematic(content)
        {
            let map = self.ingest_referenced_windows(&schematic, canonical.parent())?;
            if !map.is_empty() {
                impeller2_kdl::rewrite_asset_paths(&mut schematic, |path| {
                    map.get(path).map(|key| format!("db:{key}"))
                });
                bytes = impeller2_kdl::serialize_schematic(&schematic).into_bytes();
            }
        }

        write_uploaded_asset(self.assets_dir, &key, &bytes)?;
        tracing::info!(
            source = %canonical.display(),
            key = %key,
            "ingested window sub-schematic into db assets"
        );
        Ok(key)
    }

    /// First free `schematics/<stem>.kdl` key, suffixing `-2`, `-3`, … when the
    /// stem is already claimed by this pass or by an existing stored asset.
    fn assign_key(&self, source: &Path) -> String {
        let file_name = source
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "window.kdl".to_string());
        let stem = file_name.strip_suffix(".kdl").unwrap_or(&file_name);
        let mut n = 1usize;
        loop {
            let key = if n == 1 {
                format!("schematics/{stem}.kdl")
            } else {
                format!("schematics/{stem}-{n}.kdl")
            };
            let claimed = self.file_keys.values().any(|existing| *existing == key)
                || self.assets_dir.join(&key).exists();
            if !claimed {
                return key;
            }
            n += 1;
        }
    }
}

/// Resolve a window reference to a readable file: absolute paths as-is, then
/// relative to the referencing file's directory, then each search directory.
fn resolve_window_source(
    reference: &str,
    base_dir: Option<&Path>,
    search_dirs: &[PathBuf],
) -> Option<PathBuf> {
    let path = Path::new(reference);
    if path.is_absolute() {
        return path.is_file().then(|| path.to_path_buf());
    }
    base_dir
        .map(|dir| dir.join(path))
        .into_iter()
        .chain(search_dirs.iter().map(|dir| dir.join(path)))
        .find(|candidate| candidate.is_file())
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
    fn ingest_empty_source_does_not_commit_marker() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        std::fs::create_dir_all(&src).unwrap();
        let db = dir.path().join("db");

        // First run against a not-yet-populated source tree: nothing to copy,
        // and crucially no marker — the DB must stay eligible for ingest.
        let first = ingest_asset_dir(&db, &src).unwrap();
        assert!(first.skipped);
        assert_eq!(first.file_count, 0);
        assert!(
            !assets_ingested(&db),
            "an empty source must not commit the copy-once marker"
        );

        // The host tree is populated later (e.g. Aleph seed): the next run
        // must ingest for real.
        write(&src.join("meshes/rocket.glb"), b"glb");
        let second = ingest_asset_dir(&db, &src).unwrap();
        assert!(!second.skipped, "populated source must now be ingested");
        assert_eq!(second.file_count, 1);
        assert!(assets_ingested(&db));
        assert_eq!(
            std::fs::read(assets_dir(&db).join("meshes/rocket.glb")).unwrap(),
            b"glb"
        );
    }

    #[test]
    fn ingest_source_with_only_reserved_keys_does_not_commit_marker() {
        let dir = tempdir().unwrap();
        let src = dir.path().join("src_assets");
        // Only reserved keys: effectively empty after filtering.
        write(&src.join(INGEST_MARKER), b"not-our-marker");
        let db = dir.path().join("db");

        let report = ingest_asset_dir(&db, &src).unwrap();
        assert!(report.skipped);
        assert!(!assets_ingested(&db));
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

    #[test]
    fn window_ingest_pulls_external_kdl_into_schematics() {
        let dir = tempdir().unwrap();
        // The window lives outside the assets tree, referenced repo-relative
        // (the drone layout).
        write(
            &dir.path().join("examples/drone/motor-panel.kdl"),
            b"tabs {\n    graph \"drone.motor_input\"\n}\n",
        );
        let assets = dir.path().join("db/assets");
        std::fs::create_dir_all(&assets).unwrap();

        let content = "window title=\"motors\" path=\"examples/drone/motor-panel.kdl\"\n";
        let rewritten = ingest_window_schematics(&assets, content, &[dir.path().to_path_buf()])
            .unwrap()
            .expect("reference should be rewritten");

        assert!(
            rewritten.contains("path=\"db:schematics/motor-panel.kdl\""),
            "window reference should point at the stored key, got:\n{rewritten}"
        );
        let stored = std::fs::read_to_string(assets.join("schematics/motor-panel.kdl")).unwrap();
        assert!(stored.contains("drone.motor_input"));
    }

    #[test]
    fn window_ingest_rewrites_nested_assets_of_pulled_window() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("db/assets");
        // The mesh is already stored in the tree; the external window references
        // it by its local name.
        write(&assets.join("meshes/drone.glb"), b"glb");
        write(
            &dir.path().join("panels/view.kdl"),
            b"object_3d \"drone.world_pos\" {\n    glb path=\"meshes/drone.glb\"\n}\n",
        );

        let content = "window path=\"panels/view.kdl\"\n";
        ingest_window_schematics(&assets, content, &[dir.path().to_path_buf()])
            .unwrap()
            .expect("reference should be rewritten");

        let stored = std::fs::read_to_string(assets.join("schematics/view.kdl")).unwrap();
        assert!(
            stored.contains("path=\"db:meshes/drone.glb\""),
            "pulled window's mesh path should be rewritten to db:, got:\n{stored}"
        );
    }

    #[test]
    fn window_ingest_follows_transitive_windows_and_cycles() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("db/assets");
        std::fs::create_dir_all(&assets).unwrap();
        // a references b; b references a back (worst case, must terminate).
        write(&dir.path().join("panels/a.kdl"), b"window path=\"b.kdl\"\n");
        write(&dir.path().join("panels/b.kdl"), b"window path=\"a.kdl\"\n");

        let content = "window path=\"panels/a.kdl\"\n";
        let rewritten = ingest_window_schematics(&assets, content, &[dir.path().to_path_buf()])
            .unwrap()
            .expect("reference should be rewritten");
        assert!(rewritten.contains("db:schematics/a.kdl"));

        let a = std::fs::read_to_string(assets.join("schematics/a.kdl")).unwrap();
        let b = std::fs::read_to_string(assets.join("schematics/b.kdl")).unwrap();
        assert!(
            a.contains("db:schematics/b.kdl"),
            "a should point at b: {a}"
        );
        assert!(
            b.contains("db:schematics/a.kdl"),
            "b should point back at a: {b}"
        );
    }

    #[test]
    fn window_ingest_suffixes_key_on_stem_collision() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("db/assets");
        // A different panel.kdl already stored in the tree claims the stem.
        write(&assets.join("schematics/panel.kdl"), b"tabs {\n}\n");
        write(
            &dir.path().join("other/panel.kdl"),
            b"graph \"drone.thrust\"\n",
        );

        let content = "window path=\"other/panel.kdl\"\n";
        let rewritten = ingest_window_schematics(&assets, content, &[dir.path().to_path_buf()])
            .unwrap()
            .expect("reference should be rewritten");

        assert!(
            rewritten.contains("db:schematics/panel-2.kdl"),
            "colliding stem should be suffixed, got:\n{rewritten}"
        );
        // The pre-existing asset is untouched.
        assert_eq!(
            std::fs::read(assets.join("schematics/panel.kdl")).unwrap(),
            b"tabs {\n}\n"
        );
        assert!(assets.join("schematics/panel-2.kdl").is_file());
    }

    #[test]
    fn window_ingest_skips_references_already_stored_in_tree() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("db/assets");
        write(&assets.join("schematics/detail.kdl"), b"tabs {\n}\n");

        // Resolvable in the tree (bare name under schematics/): the normal
        // store_asset rewrite handles it; this pass must not duplicate it.
        let content = "window path=\"detail.kdl\"\n";
        let result =
            ingest_window_schematics(&assets, content, &[dir.path().to_path_buf()]).unwrap();
        assert!(result.is_none(), "in-tree reference needs no window ingest");
    }

    #[test]
    fn window_ingest_leaves_unresolvable_reference_local() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("db/assets");
        std::fs::create_dir_all(&assets).unwrap();

        let content = "window path=\"nowhere/missing.kdl\"\n";
        let result =
            ingest_window_schematics(&assets, content, &[dir.path().to_path_buf()]).unwrap();
        assert!(result.is_none(), "missing file must stay a local reference");
    }

    #[test]
    fn window_ingest_stores_shared_window_once() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("db/assets");
        std::fs::create_dir_all(&assets).unwrap();
        write(&dir.path().join("panels/shared.kdl"), b"tabs {\n}\n");

        let content = "window title=\"one\" path=\"panels/shared.kdl\"\nwindow title=\"two\" path=\"panels/shared.kdl\"\n";
        let rewritten = ingest_window_schematics(&assets, content, &[dir.path().to_path_buf()])
            .unwrap()
            .expect("references should be rewritten");

        assert_eq!(rewritten.matches("db:schematics/shared.kdl").count(), 2);
        assert!(!assets.join("schematics/shared-2.kdl").exists());
    }

    #[test]
    fn window_search_dirs_cover_entry_cwd_and_ancestors() {
        let dir = tempdir().unwrap();
        let sim = dir.path().join("examples/drone");
        std::fs::create_dir_all(&sim).unwrap();
        let entry = sim.join("main.py");
        std::fs::write(&entry, b"# sim").unwrap();

        let dirs = schematic_window_search_dirs(Some(&entry));
        assert_eq!(dirs.first(), Some(&sim), "entry dir must come first");
        assert!(
            dirs.iter().any(|d| d == dir.path()),
            "ancestors of the entry dir must be searched (repo-root-relative refs)"
        );
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
