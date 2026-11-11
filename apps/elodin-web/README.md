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
dev page. An HTTPS deployment needs a browser-accessible secure WebSocket
endpoint. A successful browser build does not by itself validate a Cloud data
source or its authentication.

## Run an example

The browser editor does not launch a simulation. Start the example so its
database is listening, then serve the editor with that database's asset URL.
`just install` provides the `elodin` binary and the Python package; `just web`
only serves the editor.

In one shell, inside `nix develop`:

```sh
elodin run examples/three-body/main.py
```

In another shell, inside `nix develop`:

```sh
just web http://127.0.0.1:2241
```

Open the URL Trunk prints. The default listen address is `[::]:2240`. The
asset server, and the editor's WebSocket, are the next port (2240 → 2241).
Use another `--addr` when 2240 is taken, and pass the next port to `just web`:

```sh
elodin run --addr 127.0.0.1:3240 examples/ball/main.py
```

```sh
just web http://127.0.0.1:3241
```

Stop `elodin run` before starting a different example, and reload the page.
The editor keeps the connection it opened with the document until then.

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

## Playback cache

WASM keeps telemetry in aligned one-second blocks near the playhead and in visible plot ranges.
The RAM limits are configured through `impeller_bevy::PlaybackCacheLimits` and
`EditorPlugin::with_cache_limits`. Native backfill and frame limits stay unchanged.
A plot wider than the RAM budget fills only the region nearest the playhead.
Assets use the browser HTTP cache; downloads larger than 64 MiB are rejected.

A host page can install `globalThis.elodinReplayCache` before loading the editor:

```ts
interface ElodinReplayCache {
  lookup(scope: string, request: Uint8Array): Promise<Uint8Array | undefined>;
  store(scope: string, request: Uint8Array, reply: Uint8Array): void;
}
```

The scope is the socket URL. Requests and replies are full Impeller frames with
request-id byte 7 set to zero. Only time-series, predecessor and message queries
are replayed, and only after a stopped database reports `DbConfig.recording == false`.
The host owns storage, quotas and session cleanup. Missing storage, errors and
lookups slower than two seconds fall back to the server. Replies from an old
socket generation are discarded. Seeks retain outstanding requests; late replies
reach disk before RAM admission decides whether to retain them.

The browser replay regression test uses the local fixture:

```sh
node libs/impeller/bevy/tests/browser_server.mjs
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test -p impeller-bevy --no-default-features --target wasm32-unknown-unknown --lib
```

Use `wasm-bindgen-test-runner` matching `Cargo.lock`, with Chrome and ChromeDriver available.
