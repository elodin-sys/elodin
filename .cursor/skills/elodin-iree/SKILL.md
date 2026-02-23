---
name: elodin-iree
description: Work with the IREE runtime FFI crate. Use when modifying libs/iree-runtime/, writing code that executes VMFB modules from Rust, compiling MLIR test fixtures, updating the IREE nix package, or continuing the XLA-to-IREE migration.
---

# Elodin IREE Runtime

The `libs/iree-runtime/` crate provides Rust FFI bindings and safe wrappers for IREE's C runtime API. It enables executing compiled VMFB modules (IREE's portable bytecode format) from pure Rust with no Python dependency at runtime.

## Background

The old `libs/noxla/` crate wraps XLA's C++ API via `cpp!`/`cxx` macros: it downloads a prebuilt XLA binary from an Elodin fork, compiles vendored JAXlib C++ kernels, statically links ~hundreds of MB, and historically required tightly pinned JAX/JAXlib versions. This is the single largest source of build pain in the repo.

A first migration attempt (PR #445, Jan 2026) replaced XLA with IREE's Python runtime (`iree.runtime`), but called Python every tick via PyO3 -- unacceptable for a real-time simulation loop.

This crate takes a different two-phase approach:

- **Compilation** (Python, happens once at startup): JAX function -> `jax.export` -> StableHLO MLIR -> `iree.compiler.compile_str()` -> VMFB bytes
- **Execution** (Pure Rust via IREE C API, happens every tick): VMFB -> `Instance` -> `Session` -> `Call` -> `BufferView` I/O. No GIL, no PyO3, no numpy.

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
      simple_mul.mlir       # f32 element-wise multiply source
      simple_mul.vmfb       # Precompiled for llvm-cpu
      simple_add_f64.mlir   # f64 element-wise add source
      simple_add_f64.vmfb   # Precompiled with f64 support
    integration.rs          # 4 integration tests
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

Test fixtures are precompiled `.vmfb` files checked into `tests/fixtures/`. To regenerate or create new ones:

```bash
# Install the IREE compiler (in a venv, not the nix Python)
uv venv /tmp/iree-venv --python 3.12
source /tmp/iree-venv/bin/activate
uv pip install iree-base-compiler==3.10.0

# Compile f32 module (no target triple -- produces cross-platform VMFB)
iree-compile --iree-hal-target-backends=llvm-cpu \
  simple_mul.mlir -o simple_mul.vmfb

# Compile f64 module (critical flags for physics simulations)
iree-compile --iree-hal-target-backends=llvm-cpu \
  --iree-vm-target-extension-f64 \
  --iree-input-demote-f64-to-f32=false \
  simple_add_f64.mlir -o simple_add_f64.vmfb
```

The `iree-base-compiler` version must match the IREE runtime version in `nix/pkgs/iree-runtime.nix` (currently 3.10.0).

Do NOT use `--iree-llvmcpu-target-triple` for checked-in fixtures -- omitting it produces host-independent VMFBs that work on both aarch64 and x86_64 CI runners.

## Nix Integration

The IREE C runtime library is built from source via nix:

- **Package**: `nix/pkgs/iree-runtime.nix` builds IREE v3.10.0 with CMake
- **Shell**: `nix/shell.nix` imports it and sets `IREE_RUNTIME_DIR`
- **Build flags**: Runtime-only (`IREE_BUILD_COMPILER=OFF`), static libs, CPU drivers (`local-sync` + `local-task`), embedded ELF + VMVX executable loaders
- **Submodules**: Only `flatcc` and `benchmark` are needed (not LLVM -- that's compiler-only)

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
6. Recompile all `.vmfb` fixtures in `tests/fixtures/` with the new compiler
7. Run `cargo test -p iree-runtime` to verify

## Next Steps

The broader migration (replacing `libs/noxla/` entirely) involves:

1. Create Rust-native `Literal`/`ElementType`/`ArrayElement` types in `libs/nox/` to replace `xla::Literal` etc.
2. Add a Python compilation module in `libs/nox-py/` that uses `jax.export` -> StableHLO -> `iree.compiler` -> VMFB
3. Wire `IREEWorldExec` in `libs/nox-py/` to use this crate's C API for the per-tick execution loop
4. Update `impeller2_server` to use `IREEWorldExec` instead of `WorldExec<Compiled>`
5. Delete `libs/noxla/` and all XLA feature flags

Key lessons from the first attempt (PR #445):
- `jax.jit(func, keep_unused=True)` is required to prevent JAX from optimizing away simulation inputs
- JAX 0.7+ requires tuples (not lists) for `GatherDimensionNumbers` parameters
- IREE may reorder I/O relative to JAX's function signature; match by shape+dtype via ABI declaration
- Math functions (`log`, `exp`) may require VMVX backend fallback if embedded ELF linking fails
