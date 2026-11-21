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
ELODIN_BACKEND=cranelift bash scripts/ci/regress.sh --all  # every example on Cranelift, then jax-cpu
```

### Dumping StableHLO MLIR and capturing a debug snapshot

```bash
ELODIN_BACKEND=cranelift ELODIN_CRANELIFT_DEBUG_DIR=/tmp/dbg \
  python examples/ball/main.py run
```

Lowered MLIR, first-tick checkpoint, structured profile JSON, and
every stderr report all land flat under `/tmp/dbg/`. See
[`PERFORMANCE.md`](PERFORMANCE.md) for the full list of outputs.

## Ahead-of-Time Objects

`lower::compile_module_to_object` runs the same lowering into a position-independent object
file instead of JIT memory. The object defines `{prefix}_step` with the JIT `main`'s signature;
its calls into the runtime (libm shims, `tensor_rt`, LAPACK) go through a table it exports
alongside their names, which the loader fills from `lower::runtime_symbol`. With `workers` above
1, independent calls form groups as in the JIT and run through `pool::run` on whatever pool
threads the loader starts; the compiler starts none. `libs/elodin-model` is that loader;
[examples/ball-cpp](../../examples/ball-cpp) and [examples/many-agents-cpp](../../examples/many-agents-cpp)
run it end to end.

Object export accepts `cpu="native"` or `"generic"`. Native enables detected host features;
generic keeps the target architecture's baseline for delivery to another CPU of that
architecture. The target OS and architecture remain those of the export process, so for another
platform the object is compiled there: [`examples/compile-object.rs`](examples/compile-object.rs)
turns a StableHLO file into an object with Cargo alone, and `libs/elodin-model/package.py --cpu`
runs it.

## Adding a New Op

1. Add the `Instruction` variant in `src/ir.rs`
2. Add the parser arm in `src/parser.rs`
3. Add lowering in `src/lower.rs` (both scalar and pointer-ABI paths)
4. If needed, add a runtime function in `src/tensor_rt.rs` and register in `TensorRtIds`
5. Add golden-value tests in `tests/ops.rs` for both paths
6. Add a case to `libs/nox-py/python/tests/test_backend_parity.py`, which checks the op
   against jax-cpu on both ABIs
7. Run `cargo test -p cranelift-mlir` and that test

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
