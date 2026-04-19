# kdl_asset_source

Registers a custom Bevy `AssetSource` that serves `.kdl` schematic files from the local filesystem. Hot-reload via Bevy's built-in `FileWatcher` is enabled only when the user explicitly scopes the KDL root via `ELODIN_KDL_DIR`.

## Why a separate plugin from `kdl_document`?

Bevy requires that every `AssetSource` is registered **before** `AssetPlugin` finalizes its source list (during `App::build`). The `kdl_document` plugin operates at a higher level — it defines assets, loaders, resources, and systems that depend on a working asset pipeline. Mixing the low-level source registration into the document lifecycle plugin would violate separation of concerns:

| Layer | Plugin | Responsibility |
|---|---|---|
| **Transport** | `kdl_asset_source` | *Where* KDL bytes come from (filesystem root, reader, watcher) |
| **Document** | `kdl_document` | *What* those bytes mean (parsing, loading, saving, hot-reload events) |

This separation also means a future alternative source (e.g. fetching KDL from a remote database) could replace `kdl_asset_source` without touching the document lifecycle.

## What it does

1. Resolves the KDL root directory from `ELODIN_KDL_DIR` (or falls back to cwd).
2. Calls `app.register_asset_source("kdl", ...)` with:
   - `AssetSourceBuilder::platform_default` — includes Bevy's recursive `FileWatcher` — **when `ELODIN_KDL_DIR` is set**; or
   - A reader/writer-only builder (no watcher) when the env var is unset and we've defaulted to cwd.
3. Symlinked root directories are handled transparently by Bevy's `FileWatcher`, which canonicalizes both the root and event paths before computing relative asset paths.

### Why no watcher in the cwd-fallback case?

Bevy's `FileWatcher` is backed by `notify` in `RecursiveMode::Recursive`. It registers one inotify watch **per directory** under the root. If the editor is launched from a developer's workspace root (which can easily contain 30 k+ directories between `target/`, `node_modules/`, etc.), this exceeds the default `fs.inotify.max_user_watches = 65 536` on Linux and panics during app start-up. Users who want KDL hot-reload should point `ELODIN_KDL_DIR` at a scoped directory that actually holds schematics.

## Configuration

| Env var | Default | Hot-reload | Description |
|---|---|---|---|
| `ELODIN_KDL_DIR` set | the given path | enabled | Root directory for KDL schematics |
| `ELODIN_KDL_DIR` unset | current working directory | disabled | Loading still works, but edits are not picked up live |

## Internals

- **File watching** (when enabled) is provided by Bevy's `FileWatcher` (backed by `notify-debouncer-full`) with a 300 ms debounce window to coalesce rapid editor saves.
- **Symlink support**: `FileWatcher` canonicalizes both the root path and individual event paths, so symlinked roots work correctly.

## Status
Active.
