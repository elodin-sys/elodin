---
name: elodin-tracy
description: Profile Elodin with Tracy. Use when profiling the editor, simulation, building with tracy features, capturing traces, analyzing performance, or adding custom instrumentation.
---

# Profiling Elodin with Tracy

> Tracy is a real-time, nanosecond-resolution hybrid frame and sampling profiler.
> https://github.com/wolfpld/tracy | BSD 3-clause

---

## 1. Quick Start

**Linux only.** Tracy profiling requires Linux. On macOS, `just install tracy` will abort with an explanation. Use a Linux machine or an OrbStack NixOS VM (see the `elodin-nix` skill) for profiling workflows.

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
| Elodin-DB | 8090 |

When using the Tracy GUI, connect to each port separately. When using `tracy-capture`, specify the port with `-p`.

### CLI Capture and Export

To capture traces headlessly (useful for CI, remote machines, or agentic workflows):

```bash
# Editor + render-server (Tracy v0.13.x protocol):
tracy-capture -a 127.0.0.1 -p 8087 -o /tmp/trace-editor.tracy -s 30 &
tracy-capture -a 127.0.0.1 -p 8088 -o /tmp/trace-render.tracy -s 30 &

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

### Cranelift-MLIR JIT (sim subprocess)

When `cranelift-mlir` is built with `--features tracy` (propagated via `nox-py/tracy` from `just install tracy`), each JIT-compiled function emits a Tracy zone named after its `FuncId → name` mapping (e.g. `main`, `inner_929`, `svd`). The zones appear in the same sim-subprocess Tracy port (8089) alongside the existing sim instrumentation.

Activation requires **both**:
- Build with `--features tracy` (or `just install tracy`)
- Runtime: `ELODIN_CRANELIFT_DEBUG_DIR=<path>`

Without `ELODIN_CRANELIFT_DEBUG_DIR`, the Cranelift JIT IR emits no probe calls, so Tracy produces zero zones for JIT'd functions — the feature is runtime-toggled orthogonally to the Cargo feature. See [`libs/cranelift-mlir/PERFORMANCE.md`](../../libs/cranelift-mlir/PERFORMANCE.md) for the full workflow.

### Elodin-DB Process

When built with `--features tracy`, the `elodin-db` binary (`libs/db/src/main.rs`) adds a `TracyLayer` to its tracing subscriber on port 8090. Instrumented hot paths:

- **`handle_conn`** -- per-connection lifetime
- **`handle_packet`** -- per-packet dispatch
- **`sink_table`** -- table decomposition + write-lock acquisition (the primary write path)
- **`apply_value`** (trace-level) -- per-component write within a table
- **`push_buf`** (trace-level) -- mmap append to index + data files
- **`follow_stream`** -- follow/replication egress path
- **`coalescing_flush`** (trace-level) -- TCP write coalescing

The `apply_value` and `push_buf` spans use `trace_span!` to minimize overhead at default log levels. Set `RUST_LOG=trace` or connect Tracy to capture them.

A throughput benchmark is available:

```bash
# Customer scenario: 400 components at 250Hz, per-component connections, with a reader
elodin-db-bench --scenario customer --json

# Same workload but batched into single-table packets (faster)
elodin-db-bench --scenario customer --mode batch --json

# Custom configuration
elodin-db-bench --components 1000 --frequency 100 --duration 20 --mode per-component
```

---

## 3. Build Details

### What `just install tracy` does

1. Builds `nox-py` (Python extension) with `maturin develop -F tracy`
2. Builds the `elodin` editor and `elodin-db` binaries with `cargo build --release -p elodin -p elodin-db --features tracy`, which activates `bevy/trace_tracy` and adds `tracing-tracy` to both processes

### Feature chain

```
apps/elodin          tracy = ["elodin-editor/tracy", "bevy/trace_tracy", "dep:tracing-tracy"]
libs/elodin-editor   tracy = ["bevy/trace_tracy"]
libs/db              tracy = ["dep:tracing-tracy"]  (adds TracyLayer to subscriber, port 8090)
libs/nox-py          tracy = ["cranelift-mlir/tracy"]  (forwards to JIT profiling layer)
libs/cranelift-mlir  tracy = ["dep:tracy-client"]  (emits per-JIT-function zones, port 8089)
```

---

## 4. Adding Custom Instrumentation

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

#### Instrument Options

```rust
#[tracing::instrument(skip(graph))]
fn my_hot_function() {
    // Entire function is a Tracy zone.
}

The `skip(graph)` option tells [`#[tracing::instrument]`](https://docs.rs/tracing/0.1/tracing/attr.instrument.html) not to attach the `graph` argument as a span field. By default the macro would try to record every parameter (usually via `Debug`), which is noisy for large values, can fail if a type has no useful `Debug`, and is rarely needed when you only want a named zone in Tracy.

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

All defined project-wide. In Elodin, these are managed by the `tracy-client-sys` crate (editor).

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
