# cranelift-mlir Performance Profiling

One environment variable, one output directory, every diagnostic:

```bash
ELODIN_BACKEND=cranelift \
ELODIN_CRANELIFT_DEBUG_DIR=/tmp/dbg \
    python path/to/customer_sim.py
```

When `ELODIN_CRANELIFT_DEBUG_DIR` is set, the backend turns on probe
emission, op-category wall-time sampling, per-tick waveform capture,
instr and fold reports, inliner and slot-pool traces, StableHLO MLIR
dump, and first-tick XLA-reference checkpoint capture — all at once.
File artifacts land flat under `<dir>`; human-readable summaries go
to stderr. Unset, the JIT emits bit-identical machine code to a
plain build and there is zero stderr chatter beyond the compile
banner.

## Table of contents

1. [Outputs at a glance](#1-outputs-at-a-glance)
2. [Quick start](#2-quick-start)
3. [Reading the stderr report](#3-reading-the-stderr-report)
4. [Per-tick waveform visualization](#4-per-tick-waveform-visualization)
5. [Comparing two captures](#5-comparing-two-captures)
6. [Tracy deep-dive workflow](#6-tracy-deep-dive-workflow)
7. [Known limits](#7-known-limits)
8. [Manual validation checklist](#8-manual-validation-checklist)

---

## 1. Outputs at a glance

Stderr (human-readable, at compile time + sim exit):

| Section | Trigger |
|---|---|
| `[elodin-cranelift] fold: ...` one-line summary | always |
| `[elodin-cranelift] fold histogram: ...` per-rule | debug mode |
| `[elodin-cranelift] abi classification: scalar=N pointer=M` | debug mode |
| `[elodin-cranelift] inliner: inlined N single-caller callee(s)` | debug mode |
| `[inliner] inlined <callee> into <caller> at pos N …` per splice | debug mode |
| Scalar/SIMD IR counts + per-function opcode histogram | debug mode |
| `[elodin-cranelift] slot_pool: N allocs, H hits (…% hit rate)` | debug mode |
| `[elodin-cranelift] profile report` block at sim exit | debug mode |

Files written to `$ELODIN_CRANELIFT_DEBUG_DIR/` (flat, overwritten each run):

| File | Content |
|---|---|
| `stablehlo.mlir` | Lowered MLIR (pre-cranelift) |
| `compile_context.json` | Input array shape/dtype summary |
| `input_<i>.bin` | Raw first-tick input buffers |
| `cranelift_output_<i>.bin` | Cranelift JIT output, first tick |
| `xla_output_<i>.bin` | XLA reference output for comparison |
| `checkpoint.json` | Component IDs, byte sizes, slot counts |
| `profile.json` | Runtime profile (see `diff_profile.py`) |

The JSON profile includes a `main_tick_waveform[]` array of per-tick
wall ns, an `op_category_timing[]` array of sampled per-op ns, a
probe-overhead calibration, and a `call_graph[]` of hot parent→callee
edges. See [`scripts/diff_profile.py`](scripts/diff_profile.py) and
[`scripts/plot_tick_waveform.py`](scripts/plot_tick_waveform.py) for
downstream consumers.

### Tracy (build-time feature, orthogonal)

`--features tracy` on `cranelift-mlir` (forwarded via `nox-py/tracy`)
adds `tracy_client::Span` emission inside the profile probes. Active
only when the tracy build *and* `ELODIN_CRANELIFT_DEBUG_DIR` are both
set. Zones carry the StableHLO source line so Tracy's "go to source"
lands on the correct `func.func` in the MLIR.

---

## 2. Quick start

From a `nix develop` shell with `just install` already run:

```bash
ELODIN_BACKEND=cranelift \
ELODIN_CRANELIFT_DEBUG_DIR=/tmp/dbg \
    python path/to/customer_sim.py
```

When the sim exits, the stderr block includes:

```
[elodin-cranelift] profile report
  wall: 41.23 s   ticks: 60000   tick_time: min=545us mean=687us max=2.34ms
  top functions by cumulative time:
    main               41.23 s (100.0%)   60000 calls  mean= 687.2us  max=2340.1us  vec%=11.1
    inner_929          18.45 s ( 44.7%)   60000 calls  mean= 307.5us  max= 612.4us  vec%=43.7
    inner_1040         11.82 s ( 28.7%)   60000 calls  mean= 197.0us  max= 390.8us  vec%=44.0
    inner_536           4.58 s ( 11.1%)   60000 calls  mean=  76.3us  max= 160.1us  vec%= 0.0 (ptr-ABI)
    svd                 1.20 s (  2.9%)   60000 calls  mean=  20.0us  max=  45.7us  vec%= 0.7
    ...
  per-op-category estimate (static × runtime call counts):
    fmul                 376 M executed  (18.62%)
    fadd                 324 M executed  (16.04%)
    ...
  simd utilization:
    static (unweighted):    18.9% vector
    runtime-weighted:       44.2% vector
  cross-ABI marshal:
    scalar→pointer:     720000 calls,    334M bytes
    pointer→scalar:     720000 calls,    220M bytes
  transcendental calls:
    libm scalar fallback:    180000 calls
    wide-SIMD batch:         540000 calls  ( 75.0% of all xcend)
[elodin-cranelift] profile: JSON report written to /tmp/dbg/profile.json
```

Every line is explained in the next section.

---

## 3. Reading the stderr report

### Header

```
wall: 41.23 s   ticks: 60000   tick_time: min=545us mean=687us max=2.34ms
```

- **wall**: total time spent inside `main` across the entire sim.
- **ticks**: number of `main` invocations (= sim ticks).
- **tick_time**: per-tick distribution. A `max` >5× `mean` typically
  means a GC pause, page fault, or a cold-path branch that only runs
  on some ticks.

### Top-N hot functions

```
inner_929   18.45 s ( 44.7%)   60000 calls  mean= 307.5us  max= 612.4us  vec%=43.7
```

Columns:
- **cumulative s + %**: total wall time in this function (including
  callees) and its share of `main`'s total.
- **calls**: invocations across the sim.
- **mean / max**: per-call timing distribution.
- **vec%**: fraction of this function's Cranelift IR that is
  SIMD-vector ops (a *static* measure; dynamic execution depends on
  loop trip counts).
- **(ptr-ABI)**: marker for pointer-ABI functions. These typically
  have `vec%=0` because the SIMD packing happens in the scalar-ABI
  caller.

### Per-op-category estimate

```
fmul    376 M executed  (18.62%)
fadd    324 M executed  (16.04%)
```

This is `Σ (calls[f] × static_opcount[f][op])` per opcode — a cheap
alternative to instruction-level perf counters. Off by at most the
per-function loop-trip variance (generally within 2×). For measured
(not estimated) per-op wall time, see the `op_category_timing[]`
array in `profile.json`.

### SIMD utilization

```
static (unweighted):    18.9% vector
runtime-weighted:       44.2% vector
```

- **static**: count all IR instructions equally across compiled
  functions.
- **runtime-weighted**: `Σ calls[f] × vector[f] / Σ calls[f] × total[f]`.
  Runtime-weighted higher than static means the SIMD rollout
  targeted the hot functions — that's the signal you want.

### Cross-ABI marshal

```
scalar→pointer:     720000 calls,    334M bytes
pointer→scalar:     720000 calls,    220M bytes
```

Every call from a scalar-ABI function into a pointer-ABI callee (or
vice versa on return) copies tensor operand bytes through a stack
slot. A high byte count with a slow tick suggests the per-function
ABI split is costing you. On a 60 000-tick sim, 720 k marshal calls
≈ 12 per tick; anything >100 per tick is suspicious.

### Transcendental calls

```
libm scalar fallback:    180000 calls
wide-SIMD batch:         540000 calls  ( 75.0% of all xcend)
```

The SIMD transcendentals (`sin`, `cos`, `exp`, `log`, …) require ≥ 2
elements per call to beat the scalar `libm` path. A sim with mostly
scalar state will show a low SIMD share here; that's expected. If
the ratio is unexpectedly low for a vector-heavy workload,
investigate whether `get_vals()` is unpacking large tensors before
the transcendental (forcing the scalar path).

---

## 4. Per-tick waveform visualization

When a sim shows a high p99 / p99.9 tick time but the mean looks
fine, the next question is usually: *are slow ticks random,
periodic, or clustered at sim start?* Every debug run already
captures the waveform into `$ELODIN_CRANELIFT_DEBUG_DIR/profile.json`
(the `main_tick_waveform` array). Render it:

```bash
libs/cranelift-mlir/scripts/plot_tick_waveform.py \
    /tmp/dbg/profile.json --out /tmp/dbg/sim.png
```

Typical patterns:

- **First-tick spike only** — cold-cache + JIT warmup; not a
  problem for long-running sims.
- **Regular periodic spikes** — allocator / GC / log-flush at a
  fixed interval. Investigate via Tracy or `samply`.
- **Isolated random outliers** — OS preemption or page faults.
  If p99 is within budget, ignore.
- **Rising baseline** — memory pressure or fragmentation;
  investigate via `heaptrack`.

The waveform itself is a JSON array of ns per tick; downstream tools
can consume it directly without invoking the plotter.

---

## 5. Comparing two captures

After a performance-affecting change, the quickest way to see the
delta is to run both captures into their own debug dir and diff:

```bash
# Baseline:
ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/base \
    python customer_sim.py

# After change:
ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/new \
    python customer_sim.py

libs/cranelift-mlir/scripts/diff_profile.py \
    /tmp/base/profile.json /tmp/new/profile.json
```

Output:

```
main tick timing:
  base:  687.21 us / tick × 60000 ticks =   41.23 s
  new:   612.08 us / tick × 60000 ticks =   36.72 s
  mean delta: -10.9%

top 15 functions by absolute time delta (new − base):
  name                         base         new       delta       pct
  inner_929                  18.45 s     14.20 s    -4.25 s    -23.0%
  ...

simd utilization:
  static:            18.9% →  28.4%
  runtime-weighted:  44.2% →  58.7%

cross-ABI marshal:
  ...
```

Absolute-time-delta sort surfaces the functions that moved most in
either direction; percentage deltas tell you relative impact. The
first line (`mean delta`) is the bottom line — did the change make
the sim faster overall?

---

## 6. Tracy deep-dive workflow

When the stderr report points at a hot function whose timing is
inconsistent with its mean (long tail), switch to Tracy.

One-time setup (Linux only):

```bash
nix develop .#tracy        # Tracy GUI + tracy-capture
just install tracy         # builds nox-py with --features tracy
```

Run with Tracy listening:

```bash
# Terminal 1: GUI, listening for sim subprocess
tracy -a 127.0.0.1 -p 8089

# Terminal 2: the sim
ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/dbg \
    python customer_sim.py
```

Each JIT-compiled function's lifetime becomes a Tracy zone labelled
with its StableHLO function name (`main`, `inner_929`, `svd`, …).
The Tracy timeline shows exact ordering, overlap, and per-call
duration. Combined with Tracy's callstack sampling (enable in the
GUI settings), you also get per-call CPU-hot lines.

Tracy limitations specific to cranelift-mlir:
- No distinction between the JIT code and the libm / tensor_rt
  extern calls it makes — those appear as opaque zones unless you
  instrument `tensor_rt` separately.
- Source locations all point at `profile.rs:0`; use the function
  name + the stderr profile to cross-reference.

See [`.cursor/skills/elodin-tracy/SKILL.md`](../../.cursor/skills/elodin-tracy/SKILL.md)
for GUI shortcuts, port conventions, and `tracy-capture` +
`tracy-csvexport` headless flows.

---

## 7. Known limits

The profiling layer is **coarse-grained by design**: it instruments
whole StableHLO functions, not individual ops or loop iterations.
For finer analysis:

| Need | Use |
|---|---|
| Per-source-line CPU time | `samply record` |
| L1/L2 cache-miss rates | `perf stat -e cache-misses` |
| Branch-miss rates | `perf stat -e branch-misses` |
| Heap / allocator pressure | `heaptrack` |
| Per-op timing within a function | Tracy zones by hand, or the `op_category_timing` array in `profile.json` |

A common workflow: (1) use this layer to find the hot function,
(2) Tracy to confirm the call pattern, (3) `samply` or `perf` for
line-level attribution inside that function.

**Runtime overhead**: the enter/exit probes do a thread-local
`Instant::now()` + stack push/pop (~40 ns per function on modern
x86). Sampled op-category probes add ~0.5% on top. For a sim
running at 1 kHz × 200 fn-calls/tick that's ~8 ms/sec of
overhead — tolerable for profiling but disable for production.

**Compile-time overhead**: enabling debug mode adds one enter / one
exit probe per function. Compile time grows 2-5 %; not measurable on
a warm cache.

**Counter reset**: `dump_report` resets the atomic marshal/xcend
counters and the TL stats map. Subsequent `CompiledModule`
lifecycles start from zero, so multiple sims in one process can be
profiled sequentially.

**Loop iteration counting**: only `while` bodies are instrumented.
Conditional branches (`case`, `stablehlo.if`) contribute their
static op counts regardless of which branch executed. If a sim has
hot `case` branches with very different op profiles,
`op_kind_executed` is a modest over/under-estimate.

**Multi-threaded support**: Elodin's JIT today is single-threaded.
The infrastructure is in place for future multi-threaded sims:
`profile::flush_current_thread()` can be called from a worker
before exit, depositing stats into a process-global accumulator
the main `dump_report` merges. Tracy / waveform assume a single
main-tick-producing thread.

---

## 8. Manual validation checklist

Pre-customer validation. Every step should pass without manual
intervention on a Linux dev machine.

### 8.1 Build gates

```bash
# Default build — must be warning-free.
cargo build -p cranelift-mlir --release
cargo clippy -p cranelift-mlir --release -- -Dwarnings
cargo test -p cranelift-mlir --release

# Tracy-enabled build — must also be warning-free.
cargo build -p cranelift-mlir --release --features tracy
cargo clippy -p cranelift-mlir --release --features tracy -- -Dwarnings
cargo test -p cranelift-mlir --release --features tracy

# nox-py tracy chain — must compile cleanly.
cargo build -p nox-py --release --features tracy
```

### 8.2 Correctness gate (debug mode off = zero behavior change)

```bash
ELODIN_CRANELIFT_DEBUG_DIR=$(pwd)/libs/cranelift-mlir/testdata/checkpoints/cube-sat \
    cargo test -p cranelift-mlir --test checkpoint_test --release \
    -- verify_checkpoint --ignored --nocapture 2>&1 | tail -3
```

Wait — the checkpoint test reads from `$ELODIN_CRANELIFT_DEBUG_DIR`.
If debug mode is unset in a production run, the backend is silent
and emits bit-identical code. The easiest way to confirm that
separately is:

```bash
ELODIN_BACKEND=cranelift python examples/cube-sat/main.py bench
```

No `[elodin-cranelift]` stderr chatter should appear beyond the
standard compile banner (`fold: N -> M instr`). No files are
written.

### 8.3 Correctness gate (debug mode on = still bit-for-bit)

```bash
ELODIN_CRANELIFT_DEBUG_DIR=$(pwd)/libs/cranelift-mlir/testdata/checkpoints/cube-sat \
    cargo test -p cranelift-mlir --test checkpoint_test --release \
    -- verify_checkpoint --ignored --nocapture 2>&1 | tail -40
```

Expected (abbreviated):

```
ALL 205 outputs match XLA reference
[elodin-cranelift] profile report
  probe overhead: XX ns/probe-pair  (calibrated via 100000 iterations)
  wall (raw): 0.00 s   wall (corrected): 0.00 s   ticks: 1
  tick_time: min=... mean=... (corrected=...) max=...
  tick_latency: p50=... p95=... p99=... p99.9=...
  top functions by cumulative time:
    inner_xxx   ... excl=... in_calls=... inline=... vec%=...
  per-op-category estimate (static × runtime call counts):
    ...
  simd utilization: ...
  cross-ABI marshal: ...
  transcendental calls: ...
  hot edges (parent → callee, by cumulative time): ...
  tick waveform: 1 samples captured (JSON contains full array)
[elodin-cranelift] profile: JSON report written to .../profile.json
test verify_checkpoint ... ok
```

Still 205/205 bit-for-bit. All report sections present.

### 8.4 Verify JSON shape

```bash
python3 -c "
import json, os
path = os.path.join(
    os.environ['ELODIN_CRANELIFT_DEBUG_DIR'], 'profile.json')
d = json.load(open(path))
assert 'profile_overhead' in d, 'missing profile_overhead'
assert 'main_wall_ns_corrected' in d, 'missing main_wall_ns_corrected'
assert 'call_graph' in d, 'missing call_graph'
assert 'loop_stats' in d, 'missing loop_stats'
f = d['functions'][0]
for k in ('exclusive_ns', 'time_in_calls_ns', 'p50_ns', 'p95_ns',
          'p99_ns', 'p99_9_ns', 'histogram', 'corrected_total_ns'):
    assert k in f, f'missing functions[].{k}'
print('shape OK')
"
```

Expected: `shape OK`.

### 8.5 Diff round-trip

```bash
ELODIN_CRANELIFT_DEBUG_DIR=/tmp/cr_p_a \
    cargo test -p cranelift-mlir --test checkpoint_test --release \
    -- verify_checkpoint --ignored --nocapture >/dev/null

ELODIN_CRANELIFT_DEBUG_DIR=/tmp/cr_p_b \
    cargo test -p cranelift-mlir --test checkpoint_test --release \
    -- verify_checkpoint --ignored --nocapture >/dev/null

python3 libs/cranelift-mlir/scripts/diff_profile.py \
    /tmp/cr_p_a/profile.json /tmp/cr_p_b/profile.json
```

Expected: both JSON files exist and are non-empty; `diff_profile.py`
prints a `main tick timing` header with sub-10 % run-to-run noise.
Marshal and xcend totals should be identical (same code path, same
inputs).

### 8.6 Regression suite

```bash
ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh --all
```

Expected: all outputs bit-identical to CSV baseline, no `FAIL`
lines. Full-system correctness gate.

### 8.7 Spot-check on a real sim

```bash
source .venv/bin/activate
ELODIN_BACKEND=cranelift \
ELODIN_CRANELIFT_DEBUG_DIR=/tmp/real_sim \
    python examples/rocket/main.py --ticks 500 2>&1 | tail -30
```

Expected: non-trivial `wall:` line, `ticks: 500` matches, top-N
function names reflect the sim structure, and
`/tmp/real_sim/profile.json` is valid JSON.

### 8.8 Tracy visual spot-check (Linux, optional)

```bash
nix develop .#tracy
just install tracy
# Terminal 1:
tracy -a 127.0.0.1 -p 8089 &
# Terminal 2:
ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/dbg \
    python examples/rocket/main.py --ticks 500
```

Expected: Tracy GUI shows per-function zones (`main`, named inners)
in the sim subprocess timeline. Zone counts should roughly match
`calls` in the stderr report.

---

If all gates pass, the profiling layer is ready for customer use.
If any one fails, stop and triage before shipping.
