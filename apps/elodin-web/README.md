# Browser editor

From the repository root, inside `nix develop`:

```sh
just install web
just web
```

Open the URL printed by Trunk. To connect to an Elodin DB, append
`?db=127.0.0.1:2240` (use the DB's TCP port; its WebSocket endpoint uses the
following port). An HTTPS deployment needs a browser-accessible secure WebSocket
endpoint. A successful browser build does not by itself validate a Cloud data
source or its authentication.

## Platform boundary

WebGPU enables Hanabi particles. The editor adapts Bevy's shadow comparison
sampling for WebGPU while retaining the WebGL shaders for browsers that use
the fallback renderer.

With WebGPU, `atmosphere` or `earth` inside the schematic's `environment`
also enables the shared atmosphere renderer on the active viewport. This does
not include Cinematic Earth's globe or night-side material. WebGL retains
the schematic lighting and clear color without atmosphere rendering.

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
