# Browser editor

From the repository root, inside `nix develop`:

```sh
just install web
just web
```

Open the URL printed by Trunk. An empty `<meta name="elodin-web-db">` opens the
connect screen. A value names the database to open on load. That value is the
asset server (`tcp_port + 1`): `http://127.0.0.1:2241`, or `https://…` when the
page is served over TLS. `just web http://127.0.0.1:2241` (or
`ELODIN_WEB_DB=http://127.0.0.1:2241 just web`) writes the meta tag for a Trunk
dev page. A successful browser build does not by itself validate a Cloud data
source or its authentication.

## Platform boundary

Big Space is required on both native and WASM targets. It provides the coordinate
conversion used by the editor and is independent of terrain streaming.

The current `bevy-cesium` integration uses Cesium Native through a C++ bridge.
It is included only on native targets, along with its terrain-readiness playback
condition. Selecting an example such as Falcon9 does not select Cargo dependencies:
without this target boundary, even a browser editor that never requests terrain
tries to build the native dependency.

WASM therefore runs without this terrain integration. Supporting Cesium terrain
in the browser requires a browser-compatible implementation; removing the target
boundary is not sufficient. Data backends and transport adapters are a separate
concern and must not require Cesium merely to build or connect the editor.

See [TODO.md](TODO.md) for other browser limitations. CI checks compilation of
`elodin-web` for `wasm32-unknown-unknown`; rendering and backend connectivity also
need a browser smoke test.

## Lifecycle events

The editor emits `ViewerEvent` messages in Rust and custom events on its browser window.
The host can listen on the frame window when both pages share an origin.

`elodin-connecting` marks a connection or recording change.
`elodin-ready` follows the database handshake, schematic load, and initial telemetry backfill.
If the recording contains no requested telemetry, the editor does not wait for a backfill.
The event can repeat after a reconnect or recording change.
`elodin-error` reports connection, schematic, and asset failures. Its `detail` field contains an error string.

The host owns its loading screen and event handlers.
The read-only controls govern editor actions. The server must enforce its own access rules.

## Experimental playback cache

Browser connections fetch camera frames and telemetry near the playhead.
Seeks keep the full recording available through new range requests.
Late responses cannot replace data for a newer seek or session.
The native connection retains its existing playback path.

The Cache API stores disposable chunks for the active viewer session.
The cache does not store cookies, authentication headers, or credential-bearing keys.
Storage failures fall back to bounded memory and authorized network requests.
The same adapter caches static GLB and glTF files, external buffers, and textures.
This experiment does not enable the native `world_mesh` terrain renderer.

Default limits are 192 MiB for camera payloads, 32 MiB for decoded camera images, and 64 MiB for telemetry.
Each browser transport queue has a 64 MiB capacity.
The disk payload limit is 512 MiB, with 8 MiB chunks and at most 4096 index entries.
An asset has a 64 MiB download limit. This limit does not bound total decoded scene or graphics memory.
The regenerated static terrain still needs separate graphics and memory qualification.
H.264 decoding uses WebCodecs in the WASM build.

Rust callers can supply `impeller_bevy::browser_cache::PlaybackCacheLimits` through `EditorPlugin::with_cache_limits`.
Set `disk_bytes` to zero to disable persistence.
For browser tests, set `globalThis.__elodinDisableDiskCache` before startup.
`globalThis.__elodinCacheStats` exposes cache counters and resident payload estimates for this experiment.
WASM linear memory needs a separate measurement because these counters do not include allocator capacity.

Run the cache adapter tests inside the Nix shell:

```sh
node --test tests/browser-cache/cache.test.mjs
```


The host can opt into read-only capabilities with `?read_only=true`. Invalid or duplicate capability parameters stop startup with `elodin-error`.

## Tests

Run native tests from the repository root inside `nix develop`:

```sh
cargo test -p impeller-bevy --lib
cargo test -p elodin-editor --lib
cargo test -p elodin-web
cargo check -p elodin-web --target wasm32-unknown-unknown
```

The browser test needs Node.js, `wasm-bindgen-test-runner`, and Chrome with `chromedriver` on `PATH`.
Use a runner version that matches `wasm-bindgen` in `Cargo.lock`.
Start the fixture in a separate Nix shell from the repository root:

```sh
node libs/impeller/bevy/tests/browser_server.mjs
```

Run the browser test from the repository root:

```sh
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test -p impeller-bevy --no-default-features --target wasm32-unknown-unknown --lib
```

The test covers derived session routes, cookies, encoded asset names, reconnects after either channel closes, and session replacement on the same host.
