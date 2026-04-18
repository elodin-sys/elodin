# cranelift-mlir

StableHLO MLIR to Cranelift JIT compiler for Elodin simulations. Parses StableHLO MLIR
text (as emitted by JAX) and compiles it to native code via Cranelift JIT, replacing IREE
as the CPU execution backend. Designed for single-threaded CPU-bound physics simulations
where IREE's runtime overhead dominates.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design documentation, data flow diagrams,
and architectural decisions.

## Performance

All six regression examples pass with matched correctness:

| Example | IREE RTF | Cranelift RTF | Speedup |
|------------|----------|---------------|---------|
| ball | 79x | ~10,000x | **~130x** |
| three-body | 29x | ~4,700x | **~160x** |
| drone | 2.9x | ~300x | **~100x** |
| rocket | 2.3x | ~33x | **~14x** |
| linalg | 0.1x | ~800x | **~8,000x** |
| cube-sat | 0.56x | ~3.6x | **~6.4x** |

## Quick Start

```bash
cargo test -p cranelift-mlir --release          # all 220+ tests
ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh --all  # full regression suite
```

### Dumping StableHLO MLIR and capturing a debug snapshot

```bash
ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/dbg \
  python examples/ball/main.py run
```

Lowered MLIR, first-tick checkpoint, structured profile JSON, and
every stderr report all land flat under `/tmp/dbg/`. See
[`PERFORMANCE.md`](PERFORMANCE.md) for the full list of outputs.

## Adding a New Op

1. Add the `Instruction` variant in `src/ir.rs`
2. Add the parser arm in `src/parser.rs`
3. Add lowering in `src/lower.rs` (both scalar and pointer-ABI paths)
4. If needed, add a runtime function in `src/tensor_rt.rs` and register in `TensorRtIds`
5. Add golden-value tests in `tests/ops.rs` for both paths
6. Run `cargo test -p cranelift-mlir`

## Debugging a Failing Example

1. Generate a checkpoint:
   ```bash
   ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/ckpt \
     bash scripts/ci/regress.sh <name> examples/<name>/main.py
   ```
2. Run the Rust verifier (no Python needed, ~0.06s):
   ```bash
   ELODIN_CRANELIFT_DEBUG_DIR=/tmp/ckpt \
     cargo test -p cranelift-mlir --test checkpoint_test --release -- --ignored --nocapture
   ```
3. The verifier shows which outputs diverge and the first mismatching element
4. To bisect: modify the checkpoint MLIR to expose intermediates, regenerate, re-verify

See [ARCHITECTURE.md](ARCHITECTURE.md) for details on the checkpoint tool and bisection
technique.

Dependency documentation sources:
[Cranelift JIT](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/docs/index.md)
[OpenXLA - StableHLO](https://openxla.org/stablehlo)
[faer - linear algebra library](https://faer.veganb.tw/docs/getting-started/introduction/)
[Wide](https://docs.rs/wide/latest/wide/)
