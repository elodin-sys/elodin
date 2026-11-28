# Terrain profiling diagnostics

The experimental branch now uses main's terrain behavior. It keeps only optional browser profiling hooks.
The former terrain ordering modes, fixed nearby region, asset head start, backdrop integration, and custom renderer patches are removed.
The merge commit preserves the earlier experiments in Git history.

The normal `elodin-web` package is unchanged. Build `packages.aarch64-linux.elodin-web-diagnostics` for the optional hooks.
This package applies `tiles-core.patch` to the pinned upstream dependency. The patch observes requests and selection results.
It does not change ordering, admission limits, rendering, transport, provider setup, or capture readiness.
No editor or renderer patch is applied.

## Optional browser hooks

Install `globalThis.__elodinTerrainAudit = (kind, epochMilliseconds, payload) => { ... }` to observe `request` and `selection` events.
The callback is optional. Its exceptions are ignored. Payloads are not constructed when the callback is absent.
Request URLs exclude the query and fragment.

Request payload positions remain compatible with the earlier captures:

`[tileId, null, depth, geometricError, visibleScreenError, null, null, hierarchy01, provider01, radius, maximumScreenError, null, url, null, null]`

The parent ID and retired policy fields are `null`. They do not describe upstream scheduling.
Selection events contain `[selectedIds, cameras, selectionComplete]`. Each camera contains position XYZ, direction XYZ, and viewport height.

Set `globalThis.__elodinTerrainFrames = []` to collect at most 6,000 frame rows:

`[epochMilliseconds, visibleCount, pendingCount, residentCount, renderReadyCount, null, selectionComplete01, decodedCount, requestCount]`

Completion remains at position six. It describes full selection completion, not first visible ground or GPU presentation time.
Main can report capture readiness before full selection completion.

## Historical tools

`profile.py`, `chromium.py`, and `serve.py` are archived helpers for the earlier baseline/progressive comparison.
Their build labels, ports, and captured results describe that historical experiment. They do not enable current terrain ordering modes.
`serve.py` also injects the old Safari shader literal workaround. It is not a passive diagnostic server for an unmodified build.
Old captures remain unchanged. Use main's normal hosting and the diagnostics package for new captures.

After editing an isolated copy of the dependency, regenerate the patch from the repository root:

```sh
uv run --no-project experiments/terrain-progressive/update-patches.py <upstream-checkout> <edited-checkout>
```

Run the command inside `nix develop`. The patch must contain only observations in `crates/tiles-core/src/lib.rs`.
