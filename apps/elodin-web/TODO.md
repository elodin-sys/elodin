# WASM editor checklist

Status of WASM in Elodin. Checked items compile into the Trunk build and are
wired up. Unchecked items are compiled out, stubbed, or still native-only.

## Finished

- [x] Trunk app on `#editor`, `wasm-release` profile (`opt-level = "z"`, fat LTO)
- [x] Boot log, `console_error_panic_hook`, and Bevy `LogPlugin` so panics reach the browser console
- [x] `strip = "symbols"` drops the Rust `name` section (bundle ~45 MB; ~3.5 KB of wasm-bindgen closure names remain)
- [x] Directional shadow maps use 2048 on WASM, same as native (Bevy’s 8192 default is not used)
- [x] WebSocket Impeller client (`?db=host:port`), connection screen, and connection-failure UI
- [x] Schematics, tiles, timeline, graphs, and EQL-backed gauges
- [x] Timeseries graph lines (x uploaded in seconds; triangle lists instead of line strips)
- [x] 3D viewports: camera, touch input, ViewCube, navigation gizmo
- [x] ViewCube and nav-gizmo hit regions register before camera input, so a cube click does not also pan the viewport
- [x] ViewCube font meshes
- [x] `object_3d`, `line_3d`, `point_trails`, gizmos, frustums, world meshes, infinite grid, wireframe
- [x] Embedded image-based lighting (`diffuse.ktx2` / `specular.ktx2`)
- [x] Geo frames and big-space origins
- [x] Display kernels via the StableHLO interpreter (Cranelift JIT stays native)
- [x] `web_time` instead of `std::time::Instant`
- [x] Malformed `WorldPos` samples are ignored instead of panicking

## Not finished


- [ ] Skyboxes: cache, generation, schematic activation, and DB asset sync (`wasm_cut.rs` stubs)
- [ ] Skybox status overlay
- [ ] Lua action tiles (the button renders; `LuaActor::spawn` does not connect)
- [ ] H.264 video decode (no decoder thread on WASM)
- [ ] Sensor-camera GPU readback
- [ ] Unix `sensor_h264`
- [ ] Thruster particles
- [ ] Cinematic Earth and the night-side material
- [ ] Screenshots and the FPS log
- [ ] Hardware stats, system-information diagnostics, and the status bar
- [ ] Component stream filter (native sends `SetStreamFilter`; WASM does not)
- [ ] Saved and on-disk color schemes (built-in presets still apply; nothing is written)
- [ ] Disk asset cache and asset-directory watching (WASM uses `NoCache`)
- [ ] Local display-kernel files (WASM fetches kernels over HTTP only)
- [ ] Interpreter coverage for every StableHLO op (`erfinv`, batched `dot_general`, and other ops still return errors)
- [ ] `wasm-opt` (`data-wasm-opt="0"` in `index.html`)
- [ ] Rust symbol names in browser stack traces (`strip = "symbols"` replaces them with `wasm-function[N]`)
- [ ] Click-through check of the ViewCube fix in a running browser session
