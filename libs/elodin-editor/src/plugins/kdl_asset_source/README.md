# kdl_asset_source

Registers a custom Bevy `AssetSource` that serves `.kdl` schematic files from the local filesystem, with live-reload via debounced file watching.

## Why a separate plugin from `kdl_document`?

Bevy requires that every `AssetSource` is registered **before** `AssetPlugin` finalizes its source list (during `App::build`). The `kdl_document` plugin operates at a higher level — it defines assets, loaders, resources, and systems that depend on a working asset pipeline. Mixing the low-level source registration (which talks directly to `notify` / OS file watchers) into the document lifecycle plugin would violate separation of concerns:

| Layer | Plugin | Responsibility |
|---|---|---|
| **Transport** | `kdl_asset_source` | *Where* KDL bytes come from (filesystem root, symlink resolution, watcher) |
| **Document** | `kdl_document` | *What* those bytes mean (parsing, loading, saving, hot-reload events) |

This separation also means a future alternative source (e.g. fetching KDL from a remote database) could replace `kdl_asset_source` without touching the document lifecycle.

## What it does

1. Resolves the KDL root directory from `ELODIN_KDL_DIR` (or cwd).
2. Calls `app.register_asset_source("kdl", ...)` with a platform-default reader backed by that directory.
3. Attaches a `KdlAssetWatcher` (powered by `notify-debouncer-mini`) so that any `.kdl` file change under the root emits `AssetSourceEvent::ModifiedAsset`, which Bevy's asset server picks up for hot-reload.
4. Handles symlinked root directories transparently by canonicalizing paths before computing relative asset paths.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `ELODIN_KDL_DIR` | current working directory | Root directory for KDL schematics |

## Internals

- **Debounce**: 300 ms (`KDL_WATCH_DEBOUNCE`) to coalesce rapid editor saves.
- **Filtering**: Only `.kdl` files trigger asset events; other file types are silently ignored.
- **Symlink support**: Both the root and individual file paths are canonicalized to match watcher events that report real paths against a symlinked root.

## Status
Active.
