---
name: elodin-iree
description: Work with the IREE runtime FFI crate. Use when modifying libs/iree-runtime/, writing code that executes VMFB modules from Rust, compiling MLIR test fixtures, updating the IREE nix package, or working on the IREE execution backend.
---

# Elodin IREE Runtime

The `libs/iree-runtime/` crate provides Rust FFI bindings and safe wrappers for IREE's C runtime API. It enables executing compiled VMFB modules (IREE's portable bytecode format) from pure Rust with no Python dependency at runtime.

## Background

IREE is the **default execution backend** for Elodin simulations. When users call `w.run(system)` or `w.build(system)`, the system is compiled through IREE by default (`backend="iree"`). A JAX backend (`backend="jax"`) is available for simulations that use JAX features IREE does not yet support.

The two-phase architecture:

- **Compilation** (Python, happens once at startup): JAX function -> `jax.jit().lower()` -> StableHLO MLIR -> `iree-compile` (subprocess) -> VMFB bytes
- **Execution** (Pure Rust via IREE C API, happens every tick): VMFB -> `Instance` -> `Session` -> `Call` -> `BufferView` I/O. No GIL, no PyO3, no numpy.

## Backend Selection

Simulations select their backend via the `backend` parameter:

```python
w.run(system, backend="iree")   # Default: fast, Python-free tick loop
w.run(system, backend="jax")    # Full JAX compatibility, Python per-tick
```

IREE does not yet support all JAX operations. Known unsupported patterns:
- `.at[].set()` scatter patterns (IREE scatter validation too strict)
- `jnp.linalg.pinv` (not yet wired to `elodin_lapack`)
- Complex control flow with `np.block` in loops (`tensor.expand_shape` limitation)

Most linalg operations (`svd`, `solve`, `cholesky`, `qr`, `det`, `slogdet`, `eigh`, `inv`) are supported via the `elodin_lapack` VM module.

When IREE compilation fails, a helpful error message tells the user to set `backend="jax"`.

## Crate Structure

```
libs/iree-runtime/
  Cargo.toml          # links = "iree", bindgen build-dep
  build.rs            # Reads IREE_RUNTIME_DIR, runs bindgen, links ~50 static libs
  src/
    lib.rs            # Public re-exports
    ffi.rs            # Private bindgen-generated bindings + manual iree_allocator_system()
    instance.rs       # Instance (RAII wrapper for iree_runtime_instance_t)
    device.rs         # Device (RAII wrapper for iree_hal_device_t)
    session.rs        # Session (VMFB loading, call creation)
    call.rs           # Call (push inputs, invoke, pop outputs)
    buffer_view.rs    # BufferView (from_bytes, to_bytes, shape/dtype introspection)
    element_type.rs   # ElementType enum (Float32, Float64, Int32, etc.)
    error.rs          # Error type with full IREE status message extraction
  tests/
    fixtures/
      *.mlir                # MLIR sources
      x86_64/*.vmfb         # Precompiled for x86_64 (CI)
      aarch64/*.vmfb        # Precompiled for aarch64 (e.g. Apple Silicon)
      compile_fixtures.sh   # Regenerate .vmfb for current host
    integration.rs          # Integration tests (select arch via cfg!(target_arch))
```

## Public API

The core usage pattern for executing a compiled VMFB:

```rust
use iree_runtime::{Instance, Session, BufferView, ElementType};
use std::path::Path;

// 1. Create runtime instance (registers all available HAL drivers)
let instance = Instance::new()?;

// 2. Create a device ("local-sync" for single-threaded, "local-task" for multi-threaded)
let device = instance.create_device("local-sync")?;

// 3. Create a session and load the compiled module
let session = Session::new(&instance, &device)?;
session.load_vmfb_file(Path::new("model.vmfb"))?;
// Or from bytes (IREE copies into aligned memory internally):
// session.load_vmfb(&vmfb_bytes)?;

// 4. Create a call to a function in the module
let mut call = session.call("module.my_function")?;

// 5. Create input buffer views from raw bytes
let input_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
let input_bytes = unsafe {
    std::slice::from_raw_parts(input_data.as_ptr() as *const u8, std::mem::size_of_val(&input_data))
};
let buf = BufferView::from_bytes(&session, input_bytes, &[4], ElementType::Float32)?;

// 6. Push inputs, invoke, pop outputs
call.push_input(&buf)?;
call.invoke()?;
let output = call.pop_output()?;

// 7. Read results back as raw bytes
let result_bytes = output.to_bytes()?;

// Introspect output metadata
assert_eq!(output.shape(), vec![4]);
assert_eq!(output.element_type(), Some(ElementType::Float32));
```

All types implement `Drop` for RAII cleanup. `Instance`, `Device`, and `Session` implement `Send`.

## Building and Testing

Requires the nix develop shell (which sets `IREE_RUNTIME_DIR`):

```bash
nix develop

cargo build -p iree-runtime
cargo test -p iree-runtime
cargo clippy -p iree-runtime -- -Dwarnings
cargo fmt -p iree-runtime -- --check
```

The crate will not build outside the nix shell because `IREE_RUNTIME_DIR` must point to the IREE runtime headers and static libraries.

## Compiling VMFB Test Fixtures

Fixtures live in `tests/fixtures/` with architecture-specific subdirectories: `x86_64/` and `aarch64/`. The integration tests choose the correct set via `cfg!(target_arch)`. Both sets are checked in so CI (x86_64) and Mac/ARM developers get passing tests without manual steps.

To regenerate fixtures (e.g. after an IREE version upgrade or when adding a new MLIR module):

```bash
# Install the IREE compiler (version must match nix/pkgs/iree-runtime.nix, currently 3.11.0)
uv venv /tmp/iree-venv --python 3.13
source /tmp/iree-venv/bin/activate
uv pip install iree-base-compiler==3.11.0

# From libs/iree-runtime/tests/fixtures/, compile for current host arch
cd libs/iree-runtime/tests/fixtures
./compile_fixtures.sh
```

The script compiles all `.mlir` files into `x86_64/` or `aarch64/` based on `uname -m`. Omitting the target triple makes `iree-compile` use the host architecture. When upgrading IREE or adding fixtures, regenerate **both** `x86_64/` and `aarch64/` (run the script on an x86_64 and an aarch64 host, or use CI for x86_64 and a Mac for aarch64).

## Nix Integration

Both the IREE runtime and compiler are built from source via nix:

- **Runtime package**: `nix/pkgs/iree-runtime.nix` builds IREE v3.11.0 with CMake (runtime-only: `IREE_BUILD_COMPILER=OFF`)
- **Compiler package**: `nix/pkgs/iree-compiler-source.nix` builds `iree-compile` and `iree-lld` from source. This is where `iree-fix-lapack-custom-call.patch` is applied. Modifying the patch triggers a full LLVM/MLIR/IREE C++ rebuild (~20-40 min on a fast machine).
- **Shell**: `nix/shell.nix` imports both and sets `IREE_RUNTIME_DIR` and `IREE_COMPILER_DIR`
- **Build flags**: Runtime: static libs, CPU drivers (`local-sync` + `local-task`), embedded ELF + VMVX executable loaders. Compiler: llvm-cpu backend, StableHLO input.
- **Submodules**: Runtime vendors `flatcc`, `benchmark`, and `printf`. Compiler additionally vendors `llvm-project` and `stablehlo`.

The nix package produces:
- `$IREE_RUNTIME_DIR/include/iree/runtime/api.h` (and transitive headers)
- `$IREE_RUNTIME_DIR/lib/*.a` (~115 static libraries)

The Rust `build.rs` reads `IREE_RUNTIME_DIR`, runs bindgen on `iree/runtime/api.h`, and links all required static libraries.

## Known Gotchas

### 1. VMFB data must be aligned

IREE's FlatBuffer verifier requires aligned buffer headers. Data from `include_bytes!` lives in `.rodata` without alignment guarantees. Always use `Session::load_vmfb_file()` for file-based loading, or `Session::load_vmfb()` which passes the system allocator so IREE copies into aligned memory.

### 2. Never hardcode element type hex values

IREE's element type encoding changed between versions. The values are NOT the same as XLA's. Always use the bindgen-generated constants from `ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_*`. The `ElementType` enum in `element_type.rs` handles this mapping.

### 3. IREE aborts on completely invalid VMFB data

Passing garbage bytes to `load_vmfb` causes IREE's FlatBuffer verifier to call C `abort()`, which kills the process. This is not recoverable from Rust. Only pass valid VMFB data or data that at least has a valid FlatBuffer header.

### 4. `iree_allocator_system()` is manually constructed

This is an inline C function that bindgen cannot generate. It is manually implemented in `ffi.rs` by constructing `iree_allocator_t { self_: null, ctl: Some(iree_allocator_libc_ctl) }`. The `iree_allocator_libc_ctl` symbol is declared as an `unsafe extern "C"` block in `ffi.rs`.

### 5. VMFB function names use `module.` prefix

Functions compiled from MLIR are namespaced under `module`. A function `func.func @simple_mul(...)` in MLIR becomes `module.simple_mul` in the VMFB. Always use `session.call("module.<name>")`.

### 6. f64 compilation requires two IREE flags

Without both `--iree-vm-target-extension-f64` and `--iree-input-demote-f64-to-f32=false`, IREE silently demotes f64 to f32. This breaks physics simulations that require double precision.

## Upgrading IREE Version

1. Update `version` in `nix/pkgs/iree-runtime.nix`
2. Update the source `hash` (use `nix-prefetch-url --unpack` on the new tag's tarball)
3. Check if submodule commits changed for `flatcc` and `benchmark` (query GitHub API for the new tag's `third_party/` contents) and update their `rev` and `hash`
4. Run `nix develop` to rebuild (first build of a new version takes several minutes)
5. Install the matching `iree-base-compiler` version in a venv
6. Recompile fixtures for both architectures: run `compile_fixtures.sh` in `tests/fixtures/` on x86_64 and aarch64 (or use CI + local Mac)
7. Run `cargo test -p iree-runtime` to verify

## Key Integration Points

- `libs/nox-py/src/iree_compile.rs` — JAX -> StableHLO -> VMFB compilation pipeline
- `libs/nox-py/src/iree_exec.rs` — `IREEExec` and `IREEWorldExec` for per-tick execution
- `libs/nox-py/src/jax_exec.rs` — `JaxExec` and `JaxWorldExec` for the JAX backend
- `libs/nox-py/src/exec.rs` — `WorldExec` enum with `Iree` and `Jax` variants
- `libs/nox-py/src/world_builder.rs` — `build_with_backend()` dispatches to IREE or JAX

Key lessons:
- `jax.jit(func, keep_unused=True)` is required to prevent JAX from optimizing away simulation inputs
- JAX 0.7+ requires tuples (not lists) for `GatherDimensionNumbers` parameters
- Math functions (`log`, `exp`) may require VMVX backend fallback if embedded ELF linking fails on macOS

## IREE Debugging Playbook (nox-py)

When debugging simulation compile failures in `libs/nox-py/src/iree_compile.rs` and `world_builder.rs`, follow this order:

1. **Run in nix shell first**
   - Always use `nix develop` for both Rust and Python runs.
   - Toolchain-dependent linker behavior differs significantly outside nix.

2. **Rebuild editable Python extension after Rust changes**
   - Rust edits in `libs/nox-py/src/*.rs` are not picked up until reinstall:
   - `nix develop -c just install py`

3. **Use artifact bundles**
   - Set `ELODIN_IREE_DUMP_DIR=/tmp/<name>` to persist:
   - `stablehlo.mlir`, `iree_compile_stderr.txt`, `iree_compile_cmd.sh`, `versions.json`, and `system_names.txt`.
   - Each compile attempt writes to a **unique subdirectory** to prevent overwriting between main and subsystem diagnostic retries.
   - If not setting `ELODIN_IREE_DUMP_DIR`, dumps default to a temp directory (`$TMPDIR/elodin_iree_debug/...`) so builds do not require a writable CWD.

4. **Interpret stage/class output first**
   - `Failure stage` tells where it broke (`jax_lower`, `stablehlo_emit`, `iree_compile`, etc.).
   - `Failure class` drives next action:
     - `UnsupportedIreeFeature`: rewrite model/system patterns or temporarily use `backend="jax"`.
     - `ToolchainMisconfigured`: investigate compiler/linker environment and dump scripts.

### Linux linker learnings (critical)

Recent drone validation exposed Linux-specific IREE LLVM CPU linker pitfalls:

- `undefined symbol: cos/sin` can occur during `iree-lld` linking.
- Embedded loader can fail const-eval with imports like `_ITM_deregisterTMCloneTable`.
- Effective mitigations in `iree_compile.rs` include:
  - forcing non-embedded linking intent (`--iree-llvmcpu-link-embedded=false`)
  - explicit Linux target triple (`--iree-llvmcpu-target-triple=<arch>-unknown-linux-gnu`)
  - disabling static linking (`--iree-llvmcpu-link-static=false`)
  - disabling const eval (`--iree-opt-const-eval=false`)
  - passing explicit linker wrappers for both:
    - `--iree-llvmcpu-system-linker-path=...`
    - `--iree-llvmcpu-embedded-linker-path=...`
  - wrapper should call `cc`/`clang` from PATH (not hardcoded `/usr/bin/cc`), link with `-lm`, and preserve non-object linker flags instead of silently dropping them.


## IREE Flags Usage (`ELODIN_IREE_FLAGS` and `iree_flags`)

Elodin supports passing extra `iree-compile` flags through:

- Environment variable: `ELODIN_IREE_FLAGS`
- Python API parameter: `iree_flags=[...]` on `w.run(...)` / `w.build(...)`

Both are useful and intentionally supported.

### When to use `ELODIN_IREE_FLAGS`

- Quick local debugging without changing Python code
- Session-wide defaults while iterating on compiler behavior
- Capturing extra diagnostics in CI/local triage runs

Example:

```bash
ELODIN_IREE_DUMP_DIR=/tmp/elodin_iree_debug \
ELODIN_IREE_FLAGS="--mlir-timing --mlir-timing-display=list --iree-hal-dump-executable-intermediates-to=/tmp/elodin_iree_debug/llvm" \
elodin run examples/drone/main.py
```

### When to use Python `iree_flags`

- Per-call control for a single test/simulation invocation
- Keeping flag usage explicit in code under test

Example:

```python
w.run(system, backend="iree", iree_flags=["--iree-opt-const-eval=false"])
```

### Precedence / composition

- Elodin appends env flags first, then API `iree_flags`.
- If both set the same option, later arguments generally win (tool-dependent).
- Prefer API `iree_flags` for targeted overrides; use env flags for broad sessions.

### Common useful flags

- `--mlir-timing --mlir-timing-display=list`: pass timing breakdowns
- `--iree-hal-dump-executable-intermediates-to=<dir>`: dump lower-level executable artifacts
- `--iree-opt-const-eval=false`: can avoid const-eval/import failures in some linker scenarios

### Cautions

- Invalid flags fail compilation immediately (helpful for catching typos).
- Some flags are backend/platform specific; verify against `iree-compile --help`.

---

## elodin_lapack VM Module (LAPACK via OpenBLAS)

The `elodin_lapack` VM module provides native LAPACK functions to the IREE runtime, replacing the deleted `_iree_linalg.py` pure-JAX workaround. It enables `jnp.linalg.svd`, `solve`, `cholesky`, `qr`, `det`, `slogdet`, `eigh`, and `inv` to compile and run through IREE.

### Architecture

```
JAX linalg op (e.g. jnp.linalg.svd)
    → stablehlo.custom_call @lapack_dgesdd_ffi
    → [iree-fix-lapack-custom-call.patch] func.call @elodin_lapack.dgesdd_6_6
    → [IREE VM] vm.call @elodin_lapack.dgesdd_6_6
    → [lapack_module.c] LAPACKE_dgesdd() via dlopen'd OpenBLAS

jnp.linalg.solve (special case — fused by SolvePatternRewriter):
    → stablehlo.custom_call @lapack_dgetrf_ffi + stablehlo.while + dtrsm
    → [rewriteSolvePattern] stablehlo.custom_call @lapack_dgetrs_ffi  (single call)
    → [LapackCustomCallRewriter] func.call @elodin_lapack.dgetrs_3_3_3
    → [lapack_module.c] LAPACKE_dgetrf() + LAPACKE_dgetrs() via OpenBLAS
```

### Key files

- `libs/iree-runtime/src/lapack_module.c` -- C implementation (LAPACKE API, dlopen, shape-suffixed exports)
- `libs/iree-runtime/src/lapack_module.h` -- public header
- `libs/iree-runtime/src/lapack.rs` -- Rust FFI (`unsafe fn create_module()` -- unsafe because it takes raw IREE pointers)
- `libs/iree-runtime/build.rs` -- compiles lapack_module.c, bakes in OpenBLAS path
- `nix/pkgs/iree-fix-lapack-custom-call.patch` -- compiler patch (func.call conversion + SolvePatternRewriter + dead while-loop removal)
- `nix/pkgs/iree-compiler-source.nix` -- nix derivation for source-built IREE compiler (applies the patch)
- `examples/linalg-iree/` -- regression test exercising all LAPACK operations

### Design decisions

- **LAPACKE C API** (not Fortran): avoids hidden string-length parameters and calling convention issues that caused segfaults with the Fortran API across different compilers/platforms.
- **dlopen at runtime**: avoids link-time ABI mismatches between the OpenBLAS version used at build time and the one available at runtime. The Nix store path is baked in via `ELODIN_OPENBLAS_PATH` define.
- **func.call** (no `nosideeffects`): the compiler patch converts custom_calls to standard `func.call` ops without `nosideeffects`, so IREE treats them as sequential host operations. The patch also includes `rewriteSolvePattern()` which fuses JAX's multi-step solve IR (`dgetrf` + `stablehlo.while` pivot loop + `dtrsm` x2) into a single `dgetrs` custom_call before conversion. This eliminates a fatal MLIR dominance error caused by external call results crossing while-loop region boundaries. A companion `removeDeadWhileLoops()` erases unused while loops (e.g. in `slogdet`).
- **Shape-suffixed exports**: each unique operand-shape combination gets its own export name (e.g., `dgesdd_3_3`, `dgesdd_6_6`). The export table must be alphabetically sorted (IREE VM binary search). All variants map to the same underlying LAPACKE function.

### Adding a new matrix size

If a simulation uses matrix sizes not in the pre-registered set (3, 4, 6, 9, 12), add entries to the alphabetically-sorted export and function tables in `lapack_module.c`. The implementation is shape-generic -- only the export name needs to match.

### Solve pattern fusion

`jnp.linalg.solve` generates `lapack_dgetrf_ffi` + `stablehlo.while` (pivot permutation) + `_lu_solve` (two `lapack_dtrsm_ffi` calls). IREE's stream scheduler cannot thread external `func.call` results across while-loop region boundaries, causing a fatal MLIR dominance error. The compiler patch fixes this with `rewriteSolvePattern()`: it detects functions containing this pattern (2-arg function with dgetrf + while + _lu_solve) and replaces the entire body with a single `lapack_dgetrs_ffi` custom_call. For matrix RHS, transposes are inserted before/after to match JAX's column-major convention.

`jnp.linalg.slogdet` also emits `dgetrf` + while, but the while-loop outputs are unused. `removeDeadWhileLoops()` erases these, preventing the same dominance error.

### dgetrs NaN semantics

`elodin_lapack_dgetrs` in `lapack_module.c` calls `LAPACKE_dgetrs` even for singular matrices (`info >= 0` from `dgetrf`), letting NaN/Inf propagate naturally from the zero diagonal of U. For bad arguments (`info < 0` from `dgetrf`, or `info2 < 0` from `dgetrs`), the output is filled with NaN. This matches JAX's native `linalg.solve` behavior.
