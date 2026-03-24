# kdl_asset_source

Registers a custom Bevy `AssetSource` that serves `.kdl` schematic files from the local filesystem, with live-reload via Bevy's built-in `FileWatcher` (enabled by the `file_watcher` feature).

## Why a separate plugin from `kdl_document`?

Bevy requires that every `AssetSource` is registered **before** `AssetPlugin` finalizes its source list (during `App::build`). The `kdl_document` plugin operates at a higher level — it defines assets, loaders, resources, and systems that depend on a working asset pipeline. Mixing the low-level source registration into the document lifecycle plugin would violate separation of concerns:

| Layer | Plugin | Responsibility |
|---|---|---|
| **Transport** | `kdl_asset_source` | *Where* KDL bytes come from (filesystem root, reader, watcher) |
| **Document** | `kdl_document` | *What* those bytes mean (parsing, loading, saving, hot-reload events) |

This separation also means a future alternative source (e.g. fetching KDL from a remote database) could replace `kdl_asset_source` without touching the document lifecycle.

## What it does

1. Resolves the KDL root directory from `ELODIN_KDL_DIR` (or cwd).
2. Calls `app.register_asset_source("kdl", ...)` with `AssetSourceBuilder::platform_default`, which provides a filesystem reader/writer and Bevy's `FileWatcher` for change detection.
3. Symlinked root directories are handled transparently by Bevy's `FileWatcher`, which canonicalizes both the root and event paths before computing relative asset paths.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `ELODIN_KDL_DIR` | current working directory | Root directory for KDL schematics |

## Internals

- **File watching** is provided by Bevy's `FileWatcher` (backed by `notify-debouncer-full`) with a 300 ms debounce window to coalesce rapid editor saves.
- **Symlink support**: `FileWatcher` canonicalizes both the root path and individual event paths, so symlinked roots work correctly.

## Status
Active.
