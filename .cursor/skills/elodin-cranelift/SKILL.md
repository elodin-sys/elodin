---
name: elodin-cranelift
description: Work with the Cranelift JIT MLIR backend. Use when modifying libs/cranelift-mlir/, adding new StableHLO ops, debugging simulation correctness issues, running the checkpoint diagnostic tool, or working on the pointer-ABI tensor runtime.
---

# Elodin Cranelift Backend

`libs/cranelift-mlir/` compiles StableHLO MLIR to native code via Cranelift JIT — the default CPU backend for Elodin simulations. Deep internals live in [`ARCHITECTURE.md`](../../libs/cranelift-mlir/ARCHITECTURE.md); profiling is in [`PERFORMANCE.md`](../../libs/cranelift-mlir/PERFORMANCE.md).

## Default backend

Cranelift is the default (`backend="cranelift"` in `WorldBuilder.run` / `.build`). Override per-run:

```bash
ELODIN_BACKEND=jax-cpu python examples/<name>/main.py run   # XLA native, reference for correctness
ELODIN_BACKEND=jax-gpu python examples/<name>/main.py run   # XLA CUDA
```

Or in Python:

```python
w.run(system, backend="jax-cpu")
```

## Where things live

| Path | Purpose |
|------|---------|
| `src/ir.rs` | Internal IR: `Module`, `FuncDef`, `Instruction` variants |
| `src/parser.rs` | StableHLO text → IR (Winnow parser, child contexts for while/case) |
| `src/lower.rs` | IR → Cranelift JIT. Dual ABI, cross-ABI marshaling, SIMD, slot pooling |
| `src/tensor_rt.rs` | Runtime: broadcast_nd, slice, transpose, reduce, gather_nd, scatter, matmul |
| `tests/ops.rs` | Per-op golden tests, both ABI paths |
| `tests/checkpoint_test.rs` | XLA-vs-Cranelift comparator |
| `libs/nox-py/src/cranelift_compile.rs` | JAX → StableHLO, XLA reference checkpoint |
| `libs/nox-py/src/cranelift_exec.rs` | `CraneliftExec` tick loop |
| `libs/nox-py/src/exec.rs` | `WorldExec` enum (dispatches Cranelift vs JAX) |

## Everyday commands

Always run inside `nix develop`.

```bash
cargo test -p cranelift-mlir                    # all unit + golden tests
cargo test -p cranelift-mlir --test ops         # per-op only
cargo clippy -p cranelift-mlir -- -Dwarnings
cargo fmt -p cranelift-mlir -- --check

# Full simulation regression (requires `just install` first):
just install
source .venv/bin/activate
ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh --all                                 # every example
ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh ball examples/ball/main.py            # one example
ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh --update ball examples/ball/main.py   # re-baseline after verifying correctness
```

Baselines live in `scripts/ci/baseline/`; tolerances in `scripts/ci/baseline/tolerances.json`.

## Shared Large Constants

Large `stablehlo.constant dense<"0x...">` blobs over 1 MB are transparently interned by `libs/cranelift-mlir`:

- Parser: `src/parser.rs` decodes large hex constants as raw bytes and interns them through `src/const_cache.rs`.
- Cache: content-addressed by element type, byte length, and raw little-endian bytes under `${ELODIN_CACHE_DIR:-~/.cache/elodin/const-cache}`.
- IR: `ConstantValue::DenseExternal` is cheap to clone and carries the cached mapping handle.
- Lowering: `src/lower.rs` registers each mapping with `JITBuilder::symbol` and lowers it through `declare_data(Linkage::Import)`, so constants are not copied into the JIT arena.
- Lifetime: `CompiledModule` must retain the `Arc<CachedConst>` handles for every imported constant. The JIT symbol stores raw pointers, so dropping the parsed IR module before live execution would otherwise leave dangling mmap pointers.
- Concurrent publishing: cache writes must never replace an already-published file. Use a temp file plus no-clobber publish so simultaneous first writers converge on one inode; otherwise one process can map a now-deleted inode and lose page-cache sharing.

Relevant tests:

```bash
cargo test -p cranelift-mlir --test large_constant_cache
cargo test -p cranelift-mlir --test imported_data_arena -- --ignored --nocapture
cargo test -p cranelift-mlir large_external_constant_8d_dynamic_slice
```

Customer-style checkpoint repro:

```bash
ELODIN_CRANELIFT_DEBUG_DIR=/tmp/dbg \
  cargo test -p cranelift-mlir --test checkpoint_test --release verify_checkpoint -- --ignored --nocapture

# To hold two verifier processes for mmap/PSS sampling:
ELODIN_CACHE_DIR=/tmp/elodin-const-cache-proof \
ELODIN_CRANELIFT_DEBUG_DIR=/tmp/dbg \
ELODIN_CRANELIFT_CHECKPOINT_HOLD_AFTER_TICK_MS=120000 \
  cargo test -p cranelift-mlir --test checkpoint_test --release verify_checkpoint -- --ignored --nocapture
```

## Adding a new op

1. **IR**: add `Instruction` variant in `src/ir.rs`
2. **Parser**: add arm in `parse_op()` in `src/parser.rs`
3. **Lowering (scalar)**: add match arm in `lower_instruction()` in `src/lower.rs`
4. **Lowering (pointer-ABI)**: add match arm in `lower_instruction_mem()` in `src/lower.rs`
5. **Runtime** (if N-D): add `tensor_<op>_f64` in `src/tensor_rt.rs`, register in `TensorRtIds`
6. **Tests**: golden-value tests in `tests/ops.rs` exercising both `run_mlir` and `run_mlir_mem`
7. `cargo test -p cranelift-mlir`

The dual-ABI / SIMD / tensor-runtime design constraints each step has to satisfy are documented in [`ARCHITECTURE.md`](../../libs/cranelift-mlir/ARCHITECTURE.md).

## Adding a new simulation example

1. Dump MLIR: `ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/dbg python examples/<name>/main.py run`
2. Catalog ops: `python3 libs/cranelift-mlir/scripts/catalog_ops.py /tmp/dbg/stablehlo.mlir` (safe on multi-GB files — do NOT grep them)
3. Copy the MLIR into `testdata/`, create an e2e test
4. Implement any missing ops (see "Adding a new op")
5. Regression: `ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh <name> examples/<name>/main.py`

## Debugging

### Simulation produces wrong values

The tick checkpoint diagnostic tool is the workhorse. Full usage and MLIR-bisection workflow in [`ARCHITECTURE.md`](../../libs/cranelift-mlir/ARCHITECTURE.md) — _Checkpoint Diagnostic Tool_ section. Quick commands:

```bash
# Capture XLA reference + Cranelift outputs for every tick input:
ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/ckpt \
  bash scripts/ci/regress.sh <example> examples/<example>/main.py

# Compare, per output, element-by-element:
ELODIN_CRANELIFT_DEBUG_DIR=/tmp/ckpt \
  cargo test -p cranelift-mlir --test checkpoint_test --release -- --ignored --nocapture
```

Once the diverging output is identified, reduce it to a minimal `tests/ops.rs` reproducer before fixing.

### Simulation crashes (segfault / abort)

- Try `--release` first: some JIT paths trip `ptr::copy_nonoverlapping` debug-mode UB checks.
- Stack overflow on complex sims: the checkpoint test pins a 64 MB thread stack; mirror this if reproducing outside the test.
- NULL pointer in JIT code is almost always a cross-ABI marshaling bug.

### Compiler says "unsupported instruction" or "unsupported custom_call target"

Follow "Adding a new op".

## Environment variables

Only two matter day-to-day:

- `ELODIN_BACKEND` — `cranelift` (default), `jax-cpu`, `jax-gpu`.
- `ELODIN_CRANELIFT_DEBUG_DIR=<dir>` — single flag for every diagnostic (profile probes, op-category sampling, tick waveform, instr/fold reports, inliner and slot-pool traces, MLIR dump, first-tick XLA reference checkpoint). Files land flat under `<dir>`. Zero overhead when unset.

Full outputs and reading guide: [`PERFORMANCE.md`](../../libs/cranelift-mlir/PERFORMANCE.md).

## Further reading

- [`libs/cranelift-mlir/ARCHITECTURE.md`](../../libs/cranelift-mlir/ARCHITECTURE.md) — compilation pipeline, dual ABI, SIMD (LaneRepr), JIT memory layout, tensor runtime, LAPACK via faer, gather patterns, while-loop scoping, checkpoint tool, testing strategy, opportunities.
- [`libs/cranelift-mlir/PERFORMANCE.md`](../../libs/cranelift-mlir/PERFORMANCE.md) — `ELODIN_CRANELIFT_DEBUG_DIR` outputs, profile report fields, diff + waveform scripts, Tracy workflow.
- [`libs/cranelift-mlir/README.md`](../../libs/cranelift-mlir/README.md) — crate-level quick start.
