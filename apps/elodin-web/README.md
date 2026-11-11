# Browser editor

From the repository root, inside `nix develop`:

```sh
just install web
just web
```

Open the URL printed by Trunk. The editor connects to the page origin, unless
`<meta name="elodin-web-db" content="…">` names a database. That value is the
asset server (`tcp_port + 1`): `http://127.0.0.1:2241`, or `https://…` when the
page is served over TLS. `just web http://127.0.0.1:2241` (or
`ELODIN_WEB_DB=http://127.0.0.1:2241 just web`) writes the meta tag for a Trunk
dev page. An empty meta tag keeps same-origin. A successful browser build does
not by itself validate a Cloud data source or its authentication.

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
