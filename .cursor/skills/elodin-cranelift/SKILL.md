---
name: elodin-cranelift
description: Work with the Cranelift JIT MLIR backend. Use when modifying libs/cranelift-mlir/, adding new StableHLO ops, debugging simulation correctness issues, running the checkpoint diagnostic tool, or working on the pointer-ABI tensor runtime.
---

# Elodin Cranelift Backend

The `libs/cranelift-mlir/` crate compiles StableHLO MLIR to native code via Cranelift JIT. It is the **high-performance alternative** to IREE for CPU-bound Elodin simulations, delivering 14x-170x speedups over IREE with millisecond compile times.

## When This Backend Is Used

The Cranelift backend is selected via:

```bash
ELODIN_BACKEND=cranelift python examples/<name>/main.py run
```

Or in Python:
```python
w.run(system, backend="cranelift")
```

When not set, the default is `backend="jax-cpu"` (XLA native), which is the reference for correctness.

## Architecture

```
JAX function
  → jax.jit().lower().compiler_ir("stablehlo")
  → StableHLO MLIR text
  → parser.rs (Winnow parser → IR)
  → lower.rs (IR → Cranelift IR → native function pointer)
  → tensor_rt.rs (runtime library for N-D tensor operations)
  → CraneliftExec (per-tick execution from Rust)
```

### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/ir.rs` | ~437 | Internal IR: `Module`, `FuncDef`, 35+ `Instruction` variants |
| `src/parser.rs` | ~2,276 | StableHLO text → IR. Child contexts for while/case scoping |
| `src/lower.rs` | ~5,756 | IR → Cranelift JIT. Dual ABI (scalar + pointer), cross-ABI marshaling |
| `src/tensor_rt.rs` | ~1,125 | Runtime: broadcast_nd, slice, transpose, reduce, gather_nd, scatter, matmul |
| `tests/ops.rs` | ~2,769 | 114 golden-value tests covering all ops in both ABI paths |
| `tests/checkpoint_test.rs` | ~158 | Checkpoint verifier for XLA-vs-Cranelift comparison |

### Integration Points

| File | Purpose |
|------|---------|
| `libs/nox-py/src/cranelift_compile.rs` | JAX → StableHLO lowering, XLA reference checkpoint |
| `libs/nox-py/src/cranelift_exec.rs` | `CraneliftExec` tick loop, checkpoint dumper |
| `libs/nox-py/src/exec.rs` | `WorldExec` enum dispatching to Cranelift/IREE/JAX |
| `scripts/ci/regress.sh` | Regression test runner (CSV + profile comparison) |

## Dual ABI System

Functions are classified at parse time by `classify_function()`:

- **Scalar ABI** (all tensors ≤ 64 elements): tensor elements are individual SSA values. Fast for small simulations (ball, three-body).
- **Pointer ABI** (any tensor > 64 elements): tensors are stack-allocated buffers. Operations call `tensor_rt` functions. Used for EGM08 gravity (65x65 matrices).

The `LARGE_TENSOR_THRESHOLD = 64` element cutoff determines classification. Cross-ABI calls (scalar calling pointer or vice versa) marshal arguments and return values at the call site.

Key lowering functions:
- `lower_instruction()` — scalar path dispatch
- `lower_instruction_mem()` — pointer-ABI path dispatch
- `lower_call()` — scalar path, handles cross-ABI to pointer callees
- `lower_pointer_body()` — pointer-ABI function entry/exit
- `lower_while()` — scalar path while loops
- `Instruction::While` in `lower_instruction_mem()` — pointer-ABI while loops

## Building and Testing

Always use the nix develop shell:

```bash
nix develop

cargo test -p cranelift-mlir                    # all tests (~114 pass)
cargo test -p cranelift-mlir --test ops         # per-op golden tests
cargo clippy -p cranelift-mlir -- -Dwarnings    # lint
cargo fmt -p cranelift-mlir -- --check          # format

# Full simulation regression (requires just install first):
just install
source .venv/bin/activate
ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh --all
```

## Tick Checkpoint Diagnostic Tool

The most effective tool for diagnosing compilation correctness bugs. It captures tick-function inputs/outputs from both XLA (reference) and Cranelift (candidate), enabling fast Rust-only comparison without Python.

### Generate a checkpoint

```bash
ELODIN_BACKEND=cranelift \
ELODIN_CRANELIFT_DEBUG_DIR=/tmp/ckpt \
  bash scripts/ci/regress.sh <example> examples/<example>/main.py
```

This produces in `/tmp/ckpt/`:
- `stablehlo.mlir` — the compiled MLIR
- `input_0.bin` ... `input_N.bin` — tick function input byte buffers
- `xla_output_0.bin` ... `xla_output_N.bin` — XLA reference outputs
- `cranelift_output_0.bin` ... — Cranelift outputs
- `checkpoint.json` — metadata (sizes, component IDs)
- `compile_context.json` — input shape/dtype summary
- `profile.json` — runtime profile (see [`PERFORMANCE.md`](../../libs/cranelift-mlir/PERFORMANCE.md))

### Verify with Rust test

```bash
ELODIN_CRANELIFT_DEBUG_DIR=/tmp/ckpt \
  cargo test -p cranelift-mlir --test checkpoint_test --release -- --ignored --nocapture
```

Output shows per-output pass/fail with element-level diffs:
```
output 0: FAIL at elem 3/12: got=-0.000236, want=-8.680, max_abs=8.680e0
output 1: OK (1 elems, max_abs=0.000e0)
...
9 of 28 outputs FAILED: [0, 8, 12, 14, 15, 16, 19, 23, 24]
```

### Bisecting a bug

1. Generate checkpoint → identifies which outputs diverge
2. Modify `stablehlo.mlir` in the checkpoint to expose intermediate values as additional returns
3. Regenerate XLA reference (re-run Python with checkpoint env)
4. Re-run verifier → identifies which intermediate first diverges

Checkpoints are committed to `testdata/checkpoints/<example>/` for permanent regression testing.

## Adding a New Op

1. **IR**: Add `Instruction` variant in `src/ir.rs`
2. **Parser**: Add arm in `parse_op()` in `src/parser.rs`
3. **Lowering (scalar)**: Add match arm in `lower_instruction()` in `src/lower.rs`
4. **Lowering (pointer-ABI)**: Add match arm in `lower_instruction_mem()` in `src/lower.rs`
5. **Runtime** (if N-D): Add `tensor_<op>_f64` in `src/tensor_rt.rs`, register in `TensorRtIds`
6. **Tests**: Add golden-value tests in `tests/ops.rs` for both `run_mlir` and `run_mlir_mem` paths
7. Run `cargo test -p cranelift-mlir`

## Adding a New Simulation Example

1. Dump MLIR: `ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/dbg python examples/<name>/main.py run`
2. Catalog ops: `python3 libs/cranelift-mlir/scripts/catalog_ops.py /tmp/dbg/stablehlo.mlir` (handles large files safely; do NOT use grep on multi-GB MLIR)
3. Copy MLIR to testdata, create e2e test
4. Implement missing ops
5. Run regression: `ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh <name> examples/<name>/main.py`

## Debugging Playbook

### Simulation produces wrong values

1. Run regression to confirm failure: `ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh <example> examples/<example>/main.py`
2. Generate checkpoint: add `ELODIN_CRANELIFT_DEBUG_DIR=/tmp/ckpt` to the command
3. Run verifier: `ELODIN_CRANELIFT_DEBUG_DIR=/tmp/ckpt cargo test ... --release -- --ignored --nocapture`
4. Identify diverging outputs → trace back to the responsible MLIR function
5. Write a minimal reproducer test in `tests/ops.rs` that exercises the failing pattern
6. Fix the bug, verify with both the reproducer test and the checkpoint

### Simulation crashes (segfault/abort)

- In debug builds, `ptr::copy_nonoverlapping` precondition checks may fire. Try `--release`.
- Stack overflow: complex functions need large thread stacks. Checkpoint test uses 64MB.
- NULL pointer in JIT code: check cross-ABI marshaling, verify all input buffers are non-null.

### Simulation compiles but an op is unsupported

The compiler prints `"unsupported instruction: ..."` or `"unsupported custom_call target: ..."`. Follow "Adding a New Op" above.

## While Loop Implementation Details

While loops are the most complex part of the compiler. Key design:

- **Parser**: `parse_while_op()` creates `ctx.child()` contexts for cond/body blocks, preserving outer-scope name mappings. `iter_arg_ids` are stored in the IR.
- **Scalar path** (`lower_while`): Uses Cranelift block parameters for loop-carried state. Header block evaluates condition, body block executes iteration.
- **Pointer-ABI path** (`lower_instruction_mem` While handler): Uses stack slots for loop-carried state with memcpy for updates. `stack_addr` of each slot is mapped to the parser-assigned `iter_arg_ids`.

Common pitfalls:
- Outer-scope variable references in while bodies require `ctx.child()` (not `ValueCtx::new()`)
- The `iter_arg_ids` must match between parser and compiler — hardcoded `ValueId(i)` will break
- Large loop-carried tensors (65x65) need 33,800-byte stack slots

## Regression Testing

```bash
# All examples:
ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh --all

# Single example:
ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh ball examples/ball/main.py

# Update baseline (after confirming correctness):
ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh --update ball examples/ball/main.py
```

The regression runner exports telemetry to CSV and compares against baselines in `scripts/ci/baseline/`. Tolerances are in `scripts/ci/baseline/tolerances.json`.

## Known Issues

- **Cube-sat gravity magnitude**: EGM08 gravity ~37,000x too small. 19/28 outputs match XLA perfectly, 9 diverge (all from inner_375). Under investigation with checkpoint tool.
- **Debug-mode UB**: Some JIT paths trigger `ptr::copy_nonoverlapping` precondition in debug builds. Use `--release`.
- **No SIMD**: All ops are scalar Cranelift instructions. Future opportunity: faer-rs for matmul/transpose.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ELODIN_BACKEND=cranelift` | Select Cranelift backend |
| `ELODIN_CRANELIFT_DEBUG_DIR=<dir>` | One flag for every diagnostic: profile probes, op-category sampling, tick waveform, instr/fold reports, inliner and slot-pool traces, MLIR dump, first-tick XLA-reference checkpoint. Files land flat under `<dir>`. Zero overhead when unset. See [`PERFORMANCE.md`](../../libs/cranelift-mlir/PERFORMANCE.md). |
