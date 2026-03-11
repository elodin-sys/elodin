---
name: elodin-tracy
description: Profile Elodin with Tracy. Use when profiling the editor, simulation, or IREE runtime, building with tracy features, capturing traces, analyzing performance, or adding custom instrumentation.
---

# Profiling Elodin with Tracy

> Tracy is a real-time, nanosecond-resolution hybrid frame and sampling profiler.
> https://github.com/wolfpld/tracy | BSD 3-clause

---

## 1. Quick Start

```bash
nix develop
just install tracy
elodin editor examples/sensor-camera/main.py
```

In a second terminal:

```bash
tracy                    # launches the Tracy profiler GUI
```

Click **Connect** in the Tracy UI. You will see both the **editor process** (Bevy systems, UI rendering) and the **simulation subprocess** (IREE kernel execution) as separate clients.

For full sampling and context-switch data on Linux, run the profiled binary with elevated privileges:

```bash
sudo elodin editor examples/sensor-camera/main.py
```

### Tracy Ports

Each Elodin process uses a dedicated Tracy port so they can be profiled independently or simultaneously:

| Process | Tracy Port |
|---------|------------|
| Editor / `elodin run` | 8087 |
| Render server | 8088 |
| Simulation (IREE) | 8089 |

When using the Tracy GUI, connect to each port separately. When using `tracy-capture`, specify the port with `-p`.

The simulation process (IREE) uses a different Tracy protocol version than the editor/render-server. Use `iree-tracy-capture` (built from IREE's vendored Tracy source) for port 8089, and the standard `tracy-capture` for ports 8087/8088.

### CLI Capture and Export

To capture traces headlessly (useful for CI, remote machines, or agentic workflows):

```bash
# Editor + render-server (Tracy v0.13.x protocol):
tracy-capture -a 127.0.0.1 -p 8087 -o /tmp/trace-editor.tracy -s 30 &
tracy-capture -a 127.0.0.1 -p 8088 -o /tmp/trace-render.tracy -s 30 &

# Simulation/IREE (IREE's pinned Tracy protocol -- must use iree-tracy-capture):
# https://iree.dev/developers/performance/profiling-with-tracy/#building-the-tracy-capture-cli-tool
iree-tracy-capture -a 127.0.0.1 -p 8089 -o /tmp/trace-sim.tracy -s 30 &

sleep 1
source .venv/bin/activate
elodin editor examples/sensor-camera/main.py

# After capture completes, export to CSV for analysis:
tracy-csvexport /tmp/trace-editor.tracy > /tmp/trace-editor.csv
tracy-csvexport /tmp/trace-render.tracy > /tmp/trace-render.csv
tracy-csvexport /tmp/trace-sim.tracy > /tmp/trace-sim.csv
```

Start the capture **before** launching Elodin so the servers are listening when the Tracy clients initialize. The `-s 30` flag records for 30 seconds; adjust as needed.

You can also open saved `.tracy` files in the GUI later:

```bash
tracy /tmp/trace-editor.tracy
```

---

## 2. What Gets Profiled

### Editor Process

When the `tracy` feature is enabled, the editor binary (`apps/elodin`) sets up a `tracing_tracy::TracyLayer` in its tracing subscriber (`apps/elodin/src/cli/mod.rs`). This means every Rust `tracing` span becomes a Tracy zone.

**Automatic (zero code changes):**
- All Bevy systems, schedules, and stages (via `bevy/trace_tracy`)
- Any function annotated with `#[tracing::instrument]`

**Tracy-specific runtime behavior:**
- Present mode switches to `AutoNoVsync` (eliminates vsync idle from profiles)
- Winit uses `continuous()` mode instead of reactive/game mode

### Simulation Subprocess (IREE)

The simulation runs as a Python subprocess (spawned by `s10`). When `just install tracy` builds nox-py with the `tracy` feature, the IREE runtime is compiled with `IREE_ENABLE_RUNTIME_TRACING=ON`. This activates Tracy instrumentation across all of IREE's internal subsystems:

- **VM execution** -- zone spans around bytecode dispatch
- **HAL dispatch** -- hardware abstraction layer operations
- **Task scheduling** -- work distribution across CPU threads
- **Memory allocations** -- tracked with pool names
- **Log messages** -- IREE log output forwarded to Tracy messages

Both processes connect to Tracy independently over TCP. The Tracy UI displays them on a unified timeline.

---

## 3. Build Details

### What `just install tracy` does

1. Builds `nox-py` (Python extension) with `maturin develop -F tracy`, which activates `iree-runtime/tracy` and links against the Tracy-instrumented IREE runtime (`$IREE_RUNTIME_TRACY_DIR`)
2. Builds the `elodin` editor binary with `cargo build --release -p elodin --features tracy`, which activates `bevy/trace_tracy` and adds `tracing-tracy`

### Feature chain

```
apps/elodin          tracy = ["elodin-editor/tracy", "bevy/trace_tracy", "dep:tracing-tracy"]
libs/elodin-editor   tracy = ["bevy/trace_tracy"]
libs/nox-py          tracy = ["iree-runtime/tracy"]
libs/iree-runtime    tracy = []  (selects IREE_RUNTIME_TRACY_DIR in build.rs, links TracyClient)
```

### Nix environment

The dev shell provides two IREE runtime builds:
- `IREE_RUNTIME_DIR` -- standard build (no tracing overhead)
- `IREE_RUNTIME_TRACY_DIR` -- built with `IREE_ENABLE_RUNTIME_TRACING=ON` and `IREE_TRACING_MODE=2`

The `tracy` feature in `iree-runtime` selects which one to link against.

---

## 4. IREE Tracing In Depth

### Tracing modes

IREE's tracing verbosity is controlled by `IREE_TRACING_MODE` (set in `nix/pkgs/iree-runtime.nix`):

| Mode | Features |
|------|----------|
| 1 | Instrumentation + log messages |
| 2 | + device instrumentation + allocation tracking (default) |
| 3 | + allocation callstacks + fiber support |
| 4 | + instrumentation callstacks (highest overhead) |

### Compiler flags for richer traces

Elodin's IREE compilation pipeline (`libs/nox-py/src/iree_compile.rs`) already uses `--iree-llvmcpu-link-embedded=false` (system library loader), which lets Tracy see deeper into generated native code than the embedded ELF loader.

For even richer trace data, these flags can be added to the `iree-compile` invocation:

- `--iree-hal-executable-debug-level=3` -- embeds source-level info (MLIR locations) into the `.vmfb`
- `--iree-llvmcpu-debug-symbols=true` -- includes debug symbols (already the default)

### Viewing IREE-generated code in Tracy

Set this environment variable before running the simulation to preserve temporary `.so` files that IREE generates, allowing Tracy to resolve symbols and show disassembly:

```bash
IREE_PRESERVE_DYLIB_TEMP_FILES=1 elodin editor examples/sensor-camera/main.py
```

### Understanding the Tracy timeline

In the Tracy UI, look for:
- **Main thread** of the editor process -- Bevy frame loop, system execution
- **Render thread** -- GPU submission, render passes
- **Worker threads** in the simulation process -- IREE task dispatch, VM execution
- **Messages** panel -- IREE log messages forwarded to Tracy

Click the **Statistics** button to see instrumentation vs. sampling data. The **ghost** icon on threads toggles between instrumentation zones (explicit) and sampling zones (statistical).

---

## 5. Adding Custom Instrumentation

### Rust (editor/runtime)

Any `tracing` span automatically appears in Tracy when the `tracy` feature is enabled:

```rust
#[tracing::instrument]
fn my_hot_function() {
    // entire function is a Tracy zone
}

fn partial_instrumentation() {
    let _span = tracing::info_span!("critical_section").entered();
    // only this block is a Tracy zone
}
```

The current `EnvFilter` (`s10=info,elodin=info,impeller=info,...`) controls which spans reach Tracy. To capture more detail:

```bash
RUST_LOG=debug elodin editor examples/sensor-camera/main.py
```

### IREE (simulation internals)

IREE's instrumentation uses C macros defined in `iree/base/tracing.h`:

```c
IREE_TRACE_ZONE_BEGIN(z0);
// ... work ...
IREE_TRACE_ZONE_END(z0);
```

These are compiled into the IREE static libraries and activate automatically when the tracy-enabled build is linked. No changes to simulation Python code are needed.

---

## 6. Tips and Troubleshooting

### Tracy won't connect

- Ensure the profiler UI is running and listening **before** starting the Elodin binary
- Or set `TRACY_NO_EXIT=1` to keep the app alive until Tracy connects:
  ```bash
  TRACY_NO_EXIT=1 elodin editor examples/sensor-camera/main.py
  ```

### Missing sampling / CPU data

Sampling and context-switch capture require elevated privileges on Linux:

```bash
sudo elodin editor examples/sensor-camera/main.py
```

If you see the Tracy timeline but no ghost zones or CPU core list, this is the cause.

### "RESOURCE_EXHAUSTED; failed to open file"

Tracy keeps many file descriptors open. Increase the limit:

```bash
sudo sh -c "ulimit -n 65536 && elodin editor examples/sensor-camera/main.py"
```

### GPU timeline drift

On some Linux systems, GPU and CPU timelines drift due to network time sync:

```bash
sudo systemctl stop systemd-timesyncd
# Re-enable when done:
sudo systemctl start systemd-timesyncd
```

### Headless render server crash (resolved)

With Tracy enabled, Bevy's `RenderDiagnosticsPlugin` requires a `DiagnosticsStore` resource. The headless render server disables `DiagnosticsPlugin` but now explicitly initializes `DiagnosticsStore` to prevent panics (`libs/elodin-editor/src/headless.rs`).

---

## Appendix A: Tracy Tools

### tracy-profiler (GUI)

```
tracy-profiler [file.tracy]               # Open saved trace
tracy-profiler -a 127.0.0.1 [-p 8086]    # Auto-connect to address
```

### tracy-capture (CLI)

```
tracy-capture -o out.tracy [-a addr] [-p port] [-f] [-s seconds] [-m mem%]
```

### tracy-csvexport

```
tracy-csvexport trace.tracy [-f name] [-c] [-s sep] [-e] [-u] > out.csv
```

Columns: `name`, `src_file`, `src_line`, `total_ns`, `total_perc`, `counts`, `mean_ns`, `min_ns`, `max_ns`, `std_ns`.

### tracy-update

```
tracy-update old.tracy new.tracy [-4|-h|-e|-z level] [-j streams] [-d] [-c] [-r] [-s flags]
```

Strip flags: `l`ocks `m`essages `p`lots `M`emory `i`mages `c`tx-switches `s`ampling `C`ode `S`ource-cache.

---

## Appendix B: Compile-Time Macros Reference

All defined project-wide. In Elodin, these are managed by the `tracy-client-sys` crate (editor) and IREE's CMake build (simulation).

### Core

- `TRACY_ENABLE` -- required; without it all macros are no-ops
- `TRACY_ON_DEMAND` -- profile only when server is connected (saves memory)
- `TRACY_NO_EXIT` -- wait for server before exiting (also env var)

### Network

- `TRACY_NO_BROADCAST` -- no UDP presence announcement
- `TRACY_ONLY_LOCALHOST` -- localhost only (also env var)
- `TRACY_PORT` -- data+broadcast port (default 8086; also env var)

### Feature Toggles

- `TRACY_NO_SYSTEM_TRACING` -- no kernel data (also env var)
- `TRACY_NO_CONTEXT_SWITCH` -- no context switch capture
- `TRACY_NO_SAMPLING` -- no call stack sampling
- `TRACY_NO_CALLSTACK` -- no call stack support at all
- `TRACY_NO_CODE_TRANSFER` -- no executable code retrieval
- `TRACY_FIBERS` -- fiber/coroutine support (small perf hit)
- `TRACY_CALLSTACK=<depth>` -- force callstack capture on all macros
- `TRACY_SAMPLING_HZ=<freq>` -- sampling frequency (default 10 kHz Linux)

---

## Appendix C: Viewer Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A`/`D` | Scroll left/right |
| `W`/`S` | Zoom in/out |
| `Ctrl+F` | Find Zone |
| `Ctrl+drag` | Define time range |
| `Ctrl+click` zone | Zone statistics |
| Middle-click | Zoom to extent |
| Left-click zone | Zone info |
| Right-click zone | Set time range |
| `Ctrl+Alt+R` | Reconnect (live) |

---

## Appendix D: Limits

**Hard limits:** 64 threads/lock, 65534 source locations, 255 recursive zone appearances, 1.6-day max session, 4B memory frees, 16M unique callstacks. Little-endian only, 48-bit VA.

**Pitfalls:**
- `exit()` inside a zone = hang. Use exception workaround.
- Every `Free` needs matching `Alloc`. Mismatch kills session.
- Without `TRACY_ON_DEMAND`, events buffer unbounded in RAM.

**Linux notes:**
- Sampling and context switches require root or `perf_event_paranoid` set to -1.
- Docker: `--privileged --pid=host --user 0:0`, mount `/sys/kernel/debug`.
