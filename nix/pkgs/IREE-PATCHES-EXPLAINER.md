# IREE Compiler Patches: Architecture and Explainer

**IREE version:** 3.11.0
**LLVM commit:** `66395ad94c3283d938a2957e7c7b439711764680` (iree-org/llvm-project)
**Branch:** `fix/iree-compat-2`
**Date:** 2026-03-31

---

## 1. Introduction

Elodin builds the IREE compiler from source rather than using the pip-installed `iree-base-compiler` binary. This enables patching IREE's bundled LLVM/MLIR and StableHLO conversion passes to fix bugs that would otherwise block customer simulations.

Two separate nix derivations handle IREE:

| Derivation | File | Patches | Purpose |
|---|---|---|---|
| `iree-compiler-source` | `nix/pkgs/iree-compiler-source.nix` | 7 custom patches | Builds `iree-compile` + `iree-lld` from IREE v3.11.0 source |
| `iree-runtime` | `nix/pkgs/iree-runtime.nix` | None | Builds IREE C runtime static libraries (headers + ~115 `.a` files) |

The `nix/shell.nix` wires both into the dev environment:
- `IREE_COMPILER_DIR` -- points to the patched compiler (consumed by `libs/nox-py/src/iree_compile.rs`)
- `IREE_RUNTIME_DIR` -- points to runtime headers and static libs (consumed by `libs/iree-runtime/build.rs`)

The compiler vendors `llvm-project`, `stablehlo`, `flatcc`, `benchmark`, and `printf` as submodules. All are pinned to specific commits in the nix derivation.

---

## 2. Patch Application Order and Pipeline Context

### 2.1 Nix Patch Application Order

Nix applies patches sequentially in the array order declared in `iree-compiler-source.nix` (lines 66-74). Each patch modifies the IREE source tree before CMake runs:

| # | Patch file | Target file(s) |
|---|---|---|
| 1 | `iree-fix-i1-hex-parsing.patch` | `third_party/llvm-project/mlir/lib/AsmParser/AttributeParser.cpp` |
| 2 | `iree-fix-scalar-concat.patch` | `compiler/.../Preprocessing/Canonicalization.cpp` |
| 3 | `iree-fix-lapack-custom-call.patch` | `compiler/.../Conversion/StableHLOCustomCalls.cpp` |
| 4 | `iree-fix-power-zero.patch` | `compiler/.../Conversion/StableHLOCustomCalls.cpp` |
| 5 | `iree-fix-large-constant-promotion.patch` | `StableHLOCustomCalls.cpp`, `Passes.cpp`, `Passes.h` |
| 6 | `iree-fix-case-to-if.patch` | `StableHLOCustomCalls.cpp`, `Passes.cpp` |
| 7 | `iree-fix-scf-to-cf.patch` | `Passes.cpp` |

Patches 3-6 all modify `StableHLOCustomCalls.cpp`. They apply cleanly because each inserts at different line offsets. Patches 5-7 all modify `Passes.cpp`, each appending at different pipeline positions.

### 2.2 MLIR Pass Pipeline Execution Order

The nix patch application order (1-7) is **not** the same as the runtime pass execution order. The pass pipeline is defined in `Passes.cpp` (`buildStableHLOInputConversionPassPipeline`). The patches inject logic at the following pipeline positions:

```
StableHLO input from JAX
    |
    v
[MLIR ASM Parser]  <-- Patch 1 (i1 hex unpacking, fires during parsing)
    |
    v
[Canonicalize + StableHLOCanonicalize]  <-- Patch 2 (ScalarConcatTo2D pattern)
    |
    v
[CSE]
    |
    v
[elodin-promote-large-constants]  <-- Patch 5 (module-level pass)
    |
    v
[elodin-outline-case-ops]  <-- Patch 6 (module-level pass)
    |
    v
[LegalizeStableHLOCustomCalls]  <-- Patches 3 + 4 (per-function pass)
    |   fixupPowerZeroExponent  (Patch 4)
    |   rewriteSolvePattern     (Patch 3)
    |   removeDeadWhileLoops    (Patch 3)
    |   LapackCustomCallRewriter   (Patch 3)
    |   ScatterIntoConstantRewriter (Patch 3)
    |
    v
[LegalizeControlFlow]  (stablehlo.case -> scf.if)
    |
    v
... InlinerPass, Canonicalize, CSE, LegalizeShapeComputations,
    ConvertStableHloToLinalgExt ...
    |
    v
[ChloLegalizeToStablehlo]
    |
    v
[elodin-lower-scf-if-to-cf]  <-- Patch 7 (module-level pass)
    |
    v
[ConvertStableHloToIreeInputDialects]
    |
    v
[LLVM-CPU Codegen + iree-lld]
```

Two distinct pipeline phases:

- **Early pipeline** (StableHLO preprocessing): patches 2, 5, 6, 3, 4 all run before `LegalizeControlFlow` converts `stablehlo.case` to `scf.if`.
- **Late pipeline** (post-inliner): patch 7 runs after the `InlinerPass` has re-inlined everything into `@main`, right before the partial conversion that triggers the dominance bug.

---

## 3. Itemized Patch Reference

### 3.1 `iree-fix-i1-hex-parsing.patch`

**Layer:** LLVM/MLIR parser (pre-pass, fires during IR deserialization)

**File modified:** `third_party/llvm-project/mlir/lib/AsmParser/AttributeParser.cpp`

**Problem:** Newer JAX/jaxlib versions serialize `i1` (boolean) dense attributes using packed bits (1 bit per element, `ceil(N/8)` bytes total). IREE 3.11.0's bundled MLIR expects the older byte-per-element format (1 byte per `i1`), causing `elements hex data size is invalid` parse errors.

**Fix:** After hex data is parsed, detects when the data size matches the packed-bits representation (`ceil(numElements/8)` bytes) but not the byte-per-element representation. Unpacks each bit into a separate byte.

**Lines of code:** ~17 lines inserted.

**When to remove:** When IREE upgrades its bundled LLVM/MLIR to a version that natively handles packed `i1` serialization.

---

### 3.2 `iree-fix-scalar-concat.patch`

**Layer:** StableHLO canonicalization (runs during the Canonicalize pass)

**File modified:** `compiler/plugins/input/StableHLO/Conversion/Preprocessing/Canonicalization.cpp`

**Problem:** IREE's tiling pass generates invalid `tensor.expand_shape` ops when tiling odd-sized 1D tensors (e.g., `tensor<3xf64>` tiled to `tensor<2x1xf64>` -- dimension mismatch). This occurs when JAX emits 1D concatenation of N scalar `tensor<1xTy>` inputs producing an odd-length result.

**Fix:** Adds `ScalarConcatTo2D` rewrite pattern. Matches dim-0 concatenations of N `tensor<1xTy>` inputs. Rewrites: reshape each input to `tensor<1x1xTy>`, concatenate along dim-0 to produce `tensor<Nx1xTy>`, reshape back to `tensor<NxTy>`. The 2D intermediate avoids the tiling bug entirely. Registered before `ConcatenateOpCanon` in the canonicalization pattern set.

**Lines of code:** ~50 lines inserted.

**When to remove:** When IREE fixes the tiling pass that generates invalid `tensor.expand_shape` for odd-sized 1D tensors.

---

### 3.3 `iree-fix-lapack-custom-call.patch`

**Layer:** StableHLO custom call legalization (runs in `LegalizeStableHLOCustomCalls` per-function pass)

**File modified:** `compiler/plugins/input/StableHLO/Conversion/StableHLOCustomCalls.cpp`

**Problem:** JAX's linear algebra operations (`jnp.linalg.svd`, `solve`, `cholesky`, etc.) emit `stablehlo.custom_call @lapack_*` ops. IREE has no built-in LAPACK support, so these would fail compilation. Additionally, JAX's `.at[idx].set()` on constant arrays produces scatter ops that IREE's llvm-cpu backend cannot distribute correctly.

**Fix:** Four additions to the `LegalizeStableHLOCustomCalls` pass:

1. **`LapackCustomCallRewriter`** -- Rewrites `stablehlo.custom_call @lapack_*_ffi` to `func.call @elodin_lapack.*` with dimension-suffixed names (e.g., `@elodin_lapack.dgesdd_6_6`). Declares the function as a private `func.func` in the module if not already present. At runtime, these resolve to the `elodin_lapack` VM module in `libs/iree-runtime/src/lapack_module.c`.

2. **`ScatterIntoConstantRewriter`** -- Converts scatter-into-constant patterns (`stablehlo.scatter` on a `stablehlo.constant` operand) into `stablehlo.dynamic_update_slice`. Only matches 1D operands with single-element updates and `unique_indices=true`.

3. **`rewriteSolvePattern()`** -- Replaces JAX's `jnp.linalg.solve` IR pattern (`lapack_dgetrf_ffi` + `stablehlo.while` pivot loop + `_lu_solve` with two `dtrsm` calls) with a single `lapack_dgetrs_ffi` custom_call. For matrix RHS, inserts transposes before/after. This prevents a fatal MLIR dominance error caused by external `func.call` results crossing while-loop region boundaries.

4. **`removeDeadWhileLoops()`** -- Erases `stablehlo.while` ops whose results are all unused. JAX's `slogdet` emits `dgetrf` + `while` where the while outputs are never consumed; without removal, the dead while triggers the same dominance error as solve.

**Lines of code:** ~218 lines inserted.

**When to remove:** When IREE natively supports LAPACK custom_calls or when a different LAPACK integration strategy is adopted.

---

### 3.4 `iree-fix-power-zero.patch`

**Layer:** StableHLO custom call legalization (runs in `LegalizeStableHLOCustomCalls`, before LAPACK rewriters)

**File modified:** `compiler/plugins/input/StableHLO/Conversion/StableHLOCustomCalls.cpp`

**Problem:** LLVM's `math.powf` expansion computes `power(x, y) = exp(y * log(x))`. When `x=0, y=0`: `log(0) = -inf`, `0 * -inf = NaN` (IEEE 754 indeterminate), `exp(NaN) = NaN`. IEEE 754 mandates `pow(x, +/-0) = 1` for ALL `x`, including 0, NaN, and inf.

Discovered via customer FT16 simulation: a power-law noise model uses exponent `p=0.0` (constant noise) with speed `v=0.0` (rocket on pad at t=0). The NaN propagated through `sqrt(covariance)` into sensor noise, then through the EKF into all physics state.

**Fix:** Adds `fixupPowerZeroExponent()` which walks every `stablehlo.PowOp` in a function and wraps the result with:
```
select(y == 0, 1.0, power(x, y))
```
When `y==0`, the select short-circuits the broken `exp(y*log(x))` expansion.

**Lines of code:** ~74 lines inserted.

**When to remove:** When IREE upgrades its bundled LLVM to include upstream fixes:
- https://github.com/llvm/llvm-project/pull/124402
- https://github.com/llvm/llvm-project/pull/126338

---

### 3.5 `iree-fix-large-constant-promotion.patch`

**Layer:** Module-level pass (`elodin-promote-large-constants`), registered before `LegalizeStableHLOCustomCalls`

**Files modified:** `StableHLOCustomCalls.cpp`, `Passes.cpp`, `Passes.h`

**Problem:** Large constant tensors (e.g., lookup tables, precomputed matrices) embedded inside inner functions cause IREE's const-eval and flow passes to spend excessive time materializing and duplicating them. In customer simulations with multi-MB constants, this caused compilation timeouts.

**Fix:** Three components:

1. **`promoteLargeConstants()`** in `StableHLOCustomCalls.cpp` -- Walks the module for `stablehlo.constant` ops whose raw data exceeds a threshold (default 1 MB, configurable via `ELODIN_IREE_CONSTANT_PROMOTE_THRESHOLD` env var). For each, finds the call chain from `@main` to the containing function via BFS, then threads the constant as a new function parameter up the entire chain. The constant is erased from the inner function and becomes a runtime buffer passed from `@main`.

2. **Pipeline registration** in `Passes.cpp` -- Registered as a module-level pass via `createModulePass` immediately before `LegalizeStableHLOCustomCalls`.

3. **`createModulePass` helper** in `Passes.h` -- A generic lambda-to-pass adapter that wraps a `std::function<void(ModuleOp)>` into a proper `OperationPass<ModuleOp>`. Accepts a customizable pass name. This helper is reused by patches 6 and 7.

**Lines of code:** ~152 lines in `StableHLOCustomCalls.cpp`, ~6 lines in `Passes.cpp`, ~24 lines in `Passes.h`.

**When to remove:** When IREE improves its handling of large constants in inner functions, or when an upstream constant-hoisting pass addresses this pattern.

---

### 3.6 `iree-fix-case-to-if.patch`

**Layer:** Module-level pass (`elodin-outline-case-ops`), registered before `LegalizeStableHLOCustomCalls`

**Files modified:** `StableHLOCustomCalls.cpp`, `Passes.cpp`

**Problem:** Customer flight software uses `jax.lax.cond` and `jax.lax.switch`, which JAX lowers to `stablehlo.case`. In large fused system functions (~1200 ops with 19 case ops), IREE's `ConvertStableHloToIreeInputDialects` partial conversion triggers an MLIR `DialectConversion` value-remapping dominance failure: `operand #0 does not dominate this use`.

This was investigated over 5 attempts (documented in `IREE-CASE-DOMINANCE-STATUS.md`). Outlining alone proved insufficient because the InlinerPass re-inlines everything, and the dominance bug triggers in `scf.if` branches regardless of function size. Patch 7 is the definitive fix for the dominance error itself; this patch provides early-pipeline isolation.

**Fix:** Adds `outlineCaseOps()` and `outlineAllCaseOps()` in `StableHLOCustomCalls.cpp`. For each pre-existing `func::FuncOp`, walks for `stablehlo.case` ops and outlines each into a new private function. The outlined function captures all values used in case branches plus the case index, clones the entire case op via `IRMapping`, and returns the cloned results. The original case op is replaced with `func.call @<name>_case_N(captures...)`. Only pre-existing functions are processed to prevent infinite recursion.

Registered as a module-level pass in `Passes.cpp` via `createModulePass`, running after `elodin-promote-large-constants` and before `LegalizeStableHLOCustomCalls`.

**Role in layered defense:** Patch 6 isolates case ops during the early pipeline so patch 3's LAPACK rewriters see clean function bodies (no case ops mixing with solve patterns). Patch 7 then handles the late-pipeline dominance bug.

**Lines of code:** ~77 lines in `StableHLOCustomCalls.cpp`, ~4 lines in `Passes.cpp`.

**When to remove:** When upstream IREE resolves the dominance issue in control-flow legalization/conversion. May also be removable if patch 7 alone proves sufficient for all customer artifacts -- see improvement suggestions.

---

### 3.7 `iree-fix-scf-to-cf.patch`

**Layer:** Module-level pass (`elodin-lower-scf-if-to-cf`), registered late in the pipeline right before `ConvertStableHloToIreeInputDialects`

**File modified:** `Passes.cpp`

**Problem:** The MLIR `DialectConversion` framework has a value-remapping dominance failure when partially converting ops inside `scf.if` branches. This is the root cause of the dominance error that patches 6 addressed at the outlining level. After the InlinerPass re-inlines outlined functions back into `@main`, the `scf.if` ops are present again, and the partial conversion in `ConvertStableHloToIreeInputDialects` breaks.

The fix was chosen from three analyzed options (documented in `big-fix.md`). Option 2 (SCF-to-CF lowering) was selected for its favorable effort-to-value ratio: it addresses the root cause with minimal custom code using existing MLIR infrastructure.

**Fix:** Adds a module-level pass that walks every non-declaration `func::FuncOp`. For functions containing `scf::IfOp`, applies MLIR's existing `populateSCFToControlFlowConversionPatterns` as a partial conversion targeting only `scf::IfOp` (marking it illegal) while legalizing `cf::BranchOp` and `cf::CondBranchOp`. All other ops are marked dynamically legal and left untouched. This converts `scf.if` regions into flat `cf.cond_br`/`cf.br` basic-block control flow, eliminating region-holding ops before the partial StableHLO-to-Linalg conversion runs.

**Lines of code:** ~25 lines in `Passes.cpp`.

**When to remove:** When upstream MLIR fixes the `DialectConversion` value-remapping for region-holding ops during partial conversion, or when IREE moves `ConvertStableHloToIreeInputDialects` to full conversion.

---

## 4. Patch Interaction Map

### 4.1 File Overlap

Five of seven patches modify files in `compiler/plugins/input/StableHLO/Conversion/`:

| File | Patches |
|---|---|
| `StableHLOCustomCalls.cpp` | 3, 4, 5, 6 |
| `Passes.cpp` | 5, 6, 7 |
| `Passes.h` | 5 |
| `Preprocessing/Canonicalization.cpp` | 2 |
| `third_party/.../AttributeParser.cpp` | 1 |

Patches 3 and 4 both insert code at the same `StableHLOCustomCalls.cpp` insertion point (before the pass definition block), but at different line offsets. Patch 4 is applied after patch 3, so it inserts relative to the already-patched file. Similarly, patches 5, 6, and 7 each append to `Passes.cpp` at progressively later pipeline positions.

Because multiple patches modify overlapping files with offset-dependent hunks, **patch application order matters**. Reordering patches in the nix array can cause fuzz/offset failures. This was observed during development: patch 6's Attempt 3 failed because the code landed at the wrong file position due to ambiguous context matching.

### 4.2 Shared Infrastructure

Patch 5 introduces the `createModulePass` helper in `Passes.h`:

```cpp
inline std::unique_ptr<OperationPass<ModuleOp>>
createModulePass(std::function<void(ModuleOp)> body,
                 StringRef name = "elodin-promote-large-constants");
```

This adapter is the foundation for all three module-level Elodin passes:
- Patch 5: `elodin-promote-large-constants` (default name)
- Patch 6: `elodin-outline-case-ops`
- Patch 7: `elodin-lower-scf-if-to-cf`

### 4.3 Pipeline Ordering Dependencies

The pass execution order creates implicit dependencies between patches:

1. **Patch 5 before Patch 6:** Constants must be promoted before case outlining, because outlining clones the case op body. If a large constant lives inside a case branch, promoting it first avoids duplicating the constant into the outlined function.

2. **Patch 6 before Patches 3/4:** Outlining isolates case ops into separate functions before `LegalizeStableHLOCustomCalls` runs. This ensures the LAPACK rewriters and power-zero fixup see clean function bodies without case ops that could confuse pattern matching.

3. **Patches 3/4 before LegalizeControlFlow:** The solve pattern rewriter (patch 3) must replace `stablehlo.while` ops before `LegalizeControlFlow` converts `stablehlo.case` to `scf.if`. If the while ops persisted into `scf.if` lowering, they would create additional dominance issues.

4. **Patch 7 after InlinerPass:** Patch 7 runs in the late pipeline, after the InlinerPass has re-inlined all functions into `@main`. At this point, patch 6's outlining has been undone. Patch 7 converts `scf.if` to flat control flow right before the conversion pass where the dominance bug lives.

### 4.4 The Layered Defense (Patches 6 + 7)

Patches 6 and 7 address the same root problem (the MLIR `DialectConversion` dominance bug) from two angles:

- **Patch 6 (early pipeline):** Outlines `stablehlo.case` ops so that patch 3's LAPACK rewriters process clean functions. Without this, solve patterns inside case branches would not be rewritten before control-flow legalization.

- **Patch 7 (late pipeline):** After the InlinerPass re-inlines everything (undoing patch 6's outlining), patch 7 converts all `scf.if` ops to flat `cf.cond_br`/`cf.br`. With no region-holding ops, the partial conversion cannot trigger the dominance bug.

Together they form a layered defense: patch 6 ensures correct LAPACK lowering in the early pipeline, and patch 7 prevents the dominance crash in the late pipeline.

---

## 5. Nix Build Architecture

### 5.1 Build Flow

```
fetchFromGitHub (IREE v3.11.0 source)
    |
    v
postUnpack: vendor submodules
    |  llvm-project (iree-org fork, commit 66395ad)
    |  stablehlo (iree-org fork)
    |  flatcc, benchmark, printf
    |
    v
patches: apply 7 patches sequentially
    |  1. i1 hex parsing (LLVM/MLIR)
    |  2. scalar concat (StableHLO canonicalization)
    |  3. LAPACK custom call (StableHLO conversion)
    |  4. power zero (StableHLO conversion)
    |  5. large constant promotion (StableHLO conversion + pipeline)
    |  6. case outlining (StableHLO conversion + pipeline)
    |  7. SCF-to-CF lowering (pipeline)
    |
    v
CMake build (ninja)
    |  IREE_BUILD_COMPILER=ON
    |  IREE_TARGET_BACKEND_LLVM_CPU=ON (only backend)
    |  IREE_INPUT_STABLEHLO=ON
    |  IREE_ENABLE_LLD=ON
    |  BUILD_SHARED_LIBS=OFF
    |
    v
installPhase:
    |  $out/bin/iree-compile       (compiler binary)
    |  $out/lib/libIREECompiler.so (shared library)
    |  $out/libexec/iree-lld       (real linker binary)
    |  $out/bin/iree-lld           (wrapper script, appends musl libc.a)
    |
    v
patchelf: fix RPATH for libstdc++ resolution
```

### 5.2 The iree-lld Wrapper

The install phase creates a wrapper script at `$out/bin/iree-lld` that calls the real `iree-lld` binary and appends `musl/lib/libc.a`. This provides math symbols (`sin`, `cos`, `log`, `exp`) needed by the embedded ELF linker when generating f64 code. Without musl's libc.a, `iree-compile` fails with `undefined symbol: cos` during embedded ELF linking.

### 5.3 Dev Shell Integration

`nix/shell.nix` exposes the patched compiler to the development environment:

```nix
IREE_COMPILER_DIR = "${iree_compiler_source}";  # patched compiler
IREE_RUNTIME_DIR  = "${iree_runtime}";           # unpatched runtime
OPENBLAS_DIR      = "${openblas}";               # for LAPACK VM module
```

At runtime, `libs/nox-py/src/iree_compile.rs` locates `iree-compile` via `IREE_COMPILER_DIR` (or the baked-in path in the installed Python package) and invokes it as a subprocess with the appropriate flags.

---

## 6. Improvement Suggestions

### 6.1 Performance Optimization

**Batch constant promotion (Patch 5):** `promoteLargeConstants` currently calls `findCallChain` + `promoteConstant` per large constant, each building a BFS from scratch. Batching all promotions to share a single caller map traversal would reduce redundant work in modules with many large constants.

**Register proper named passes:** The `createModulePass` lambda pattern (used by patches 5, 6, 7) prevents MLIR's pass statistics and timing infrastructure (`--mlir-timing`) from reporting meaningful names. Consider registering proper passes via MLIR's TableGen `.td` mechanism, which would enable accurate per-pass timing breakdowns.

**Reduce IR churn in solve rewriter (Patch 3):** `rewriteSolvePattern` erases the entire function body and rebuilds it. An in-place pattern match + replacement would produce less IR churn and enable MLIR's greedy pattern driver to optimize the replacement more efficiently.

**Skip functions without scf.if (Patch 7):** The pass already checks for `scf::IfOp` presence before applying conversion, which is good. A minor optimization: instead of walking the function body to detect `scf::IfOp`, check the op statistics if MLIR exposes them at the pass manager level.

### 6.2 Code Maintainability

**Consolidate overlapping patches:** Patches 3, 4, 5, and 6 all modify `StableHLOCustomCalls.cpp`. Consolidating them into one or two larger patches would eliminate fuzz/offset drift risk. Patch 6's Attempt 3 failed precisely because a hunk landed at the wrong file position due to ambiguous context matching. A single unified patch for all `StableHLOCustomCalls.cpp` changes would be immune to this class of error.

**Evaluate patch 6 necessity:** Now that patch 7 addresses the dominance bug at the root (eliminating `scf.if` before partial conversion), investigate whether patch 6 (case outlining) is still required. Patch 6 serves two purposes: (a) isolating case ops during early pipeline for LAPACK rewriter correctness, and (b) preventing the dominance error. Purpose (b) is now handled by patch 7. If purpose (a) can be validated independently, and if removing patch 6 does not break customer artifacts, it would be a significant simplification.

**Document env vars in SKILL.md:** The `ELODIN_IREE_CONSTANT_PROMOTE_THRESHOLD` environment variable (patch 5) is not documented in `.cursor/skills/elodin-iree/SKILL.md`. It should be added alongside the existing `ELODIN_IREE_FLAGS` and `ELODIN_IREE_DUMP_DIR` documentation.

**Add provenance comments to patches:** Each `.patch` file should include a header comment block explaining the upstream issue, when it can be removed, and any tracking links. This would make it easier for future developers to evaluate whether a patch is still needed after an IREE version upgrade.

**Regression test coverage:** The `linalg-iree` test now covers solve-inside-cond patterns (added in commit `997862b0`). Customer `.stablehlo.mlir` artifact compilation was added to `scripts/ci/regress.sh` as end-to-end validation. Both should be maintained as the gold-standard regression gate for compiler patches.

### 6.3 Upstream Tracking

| Patch | Upstream dependency | Removable when |
|---|---|---|
| 1 (i1 hex) | LLVM/MLIR packed i1 support | IREE upgrades bundled LLVM/MLIR |
| 2 (scalar concat) | IREE tiling pass fix | IREE fixes `tensor.expand_shape` for odd-sized 1D |
| 3 (LAPACK) | IREE LAPACK support | IREE natively supports LAPACK custom_calls |
| 4 (power zero) | LLVM `math.powf` fix | LLVM PRs [#124402](https://github.com/llvm/llvm-project/pull/124402), [#126338](https://github.com/llvm/llvm-project/pull/126338) land in IREE's bundled LLVM |
| 5 (constants) | IREE constant handling | IREE improves large-constant hoisting in inner functions |
| 6 (case outline) | MLIR DialectConversion fix | Dominance bug is resolved upstream; may also be removable if patch 7 alone suffices |
| 7 (SCF-to-CF) | MLIR DialectConversion fix | Upstream MLIR fixes value-remapping for region-holding ops, or IREE moves to full conversion |

---

## 7. Runtime-Side Fixes (commit `b53e0fee`)

The following changes were made to the runtime C code (`lapack_module.c`) and the compilation pipeline (`iree_compile.rs`) to fix correctness issues identified during review. These are not compiler patches -- they affect the Elodin runtime and the Python-to-VMFB compilation bridge.

### 7.1 LAPACK VM Module: Memory Leak Fixes

**File:** `libs/iree-runtime/src/lapack_module.c`

**Problem:** All 8 LAPACK functions (`dgesdd`, `dpotrf`, `dgetrf`, `dgeqrf`, `dorgqr`, `dtrsm`, `dsyevd`, `dgetrs`) allocated heap memory via `malloc`/`calloc`, then used `IREE_RETURN_IF_ERROR` which returns immediately on failure -- leaking all previously allocated buffers. For example, if `map_f64` or any `alloc_f64_view` call failed in `elodin_lapack_dgesdd`, the `a`, `s`, `u`, and `vt` arrays were never freed.

**Fix:** All 8 functions were rewritten to use goto-based cleanup:
- All heap pointers initialized to `NULL` at declaration.
- `IREE_RETURN_IF_ERROR` replaced with explicit `iree_status_is_ok()` checks that jump to a `cleanup:` label.
- Output `iree_hal_buffer_view_t*` pointers set to `NULL` after `iree_vm_ref_wrap_assign` transfers ownership.
- `cleanup:` block frees all heap allocations and releases any non-null, non-transferred output views.

### 7.2 LAPACK VM Module: Static Module Storage

**File:** `libs/iree-runtime/src/lapack_module.c`

**Problem:** `elodin_lapack_module_create` used `static elodin_lapack_module_t module_storage`, so calling it more than once (e.g., during `IREEExec::fork` which creates a fresh IREE session) silently overwrote the `device` pointer. If the previous module's `alloc_state` hadn't already run, it would capture the wrong device.

**Fix:** `module_storage` is now heap-allocated via `calloc` per call, giving each module instance its own `device` pointer. On failure in `iree_vm_module_initialize` or `iree_vm_native_module_create`, the allocation is freed before returning. The `elodin_lapack_original_lookup` static is unchanged -- it's always the same function pointer for a given descriptor, so sharing across instances is safe.

### 7.3 Promoted-Constant Extraction: `dense_resource` Support

**File:** `libs/nox-py/src/iree_compile.rs` (embedded Python in `_extract_large_constants`)

**Problem:** `_extract_large_constants` only recognized `stablehlo.constant dense<"0x...">` text, but the compiler pass (`promoteLargeConstants` in patch 5) promotes both `DenseElementsAttr` and `DenseResourceElementsAttr`. When StableHLO emits large constants as `dense_resource<...>`, they would be promoted by the compiler (adding extra function parameters) but never uploaded by the runtime, causing input arity mismatch.

**Fix:** Added a two-pass scan:
1. **First pass:** Collects resource blob hex data keyed by blob name from the MLIR metadata section (after `{-#`). The scan is gated to only the metadata section to prevent false-matching hex strings in the module body (e.g., `backend_config` attributes), which was discovered to break the drone example at runtime with a corrupted buffer length (`OUT_OF_RANGE; length=274877906960`).
2. **Second pass:** Scans constant definitions matching both `dense<"0x...">` and `dense_resource<NAME>` patterns in module order, looking up resource blob data from the first pass.

### 7.4 Promoted-Constant Extraction: Dtype Error Handling

**File:** `libs/nox-py/src/iree_compile.rs` (Rust dtype match)

**Problem:** Unknown promoted-constant dtypes fell back to `ElementType::F64`, silently reinterpreting raw bytes with the wrong element type. For example, `i1`/boolean constants or unsigned integer types would produce invalid buffer sizes or corrupted values.

**Fix:** The `_ => nox::ElementType::F64` fallback replaced with an explicit error: `return Err(Error::IreeCompilationFailed(...))`. Added explicit mappings for `i1`, `ui8`, `ui16`, `ui32`, `ui64` (mapped to signed equivalents, since IREE uses signless integers).

### 7.5 Promoted-Constant Extraction: Threshold Alignment

**File:** `libs/nox-py/src/iree_compile.rs` (embedded Python in `_extract_large_constants`)

**Problem:** The Python extraction check used `<` (`if len(raw_bytes) < threshold_bytes`), meaning constants at exactly the threshold size were extracted. But the compiler pass uses strictly-greater-than (`> threshold` at lines 141/145 of `iree-fix-large-constant-promotion.patch`). For exact-threshold constants, the runtime would upload an extra promoted buffer that the compiled module did not expect, causing input arity mismatch.

**Fix:** Changed `<` to `<=` (`if len(raw_bytes) <= threshold_bytes: continue`), so both the compiler and runtime use the same strictly-greater-than semantics.
