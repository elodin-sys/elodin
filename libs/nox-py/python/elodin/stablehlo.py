"""Shared JAX → StableHLO lowering used by simulations and UI display kernels."""

from __future__ import annotations

import json
import os
import re
from typing import Any


def lower_to_stablehlo(func: Any, input_arrays: list[Any]) -> str:
    """Lower a JAX function against concrete or shapely inputs to StableHLO text."""
    import jax

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    jax.config.update("jax_enable_x64", True)

    jit_fn = jax.jit(func, keep_unused=True)
    lowered = jit_fn.lower(*input_arrays)
    stablehlo_module = lowered.compiler_ir(dialect="stablehlo")
    stablehlo_mlir = str(stablehlo_module)
    stablehlo_mlir = re.sub(r"module @\S+", "module @module", stablehlo_mlir, count=1)

    debug_dir = os.environ.get("ELODIN_CRANELIFT_DEBUG_DIR")
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "stablehlo.mlir"), "w") as handle:
            handle.write(stablehlo_mlir)
        input_summaries = [
            {"shape": [int(dim) for dim in arr.shape], "dtype": str(arr.dtype)}
            for arr in input_arrays
        ]
        with open(os.path.join(debug_dir, "compile_context.json"), "w") as handle:
            json.dump({"inputs": input_summaries}, handle, indent=2)
        print(f"[elodin-cranelift] dumped StableHLO to {debug_dir}", file=__import__("sys").stderr)

    return stablehlo_mlir


def run_xla_reference(func: Any, real_input_arrays: list[Any], debug_dir: str) -> None:
    """Run the function with XLA and save reference outputs."""
    import sys

    import jax
    import numpy as np

    try:
        os.makedirs(debug_dir, exist_ok=True)
        jit_fn = jax.jit(func, keep_unused=True)
        results = jit_fn(*real_input_arrays)
        if not isinstance(results, (list, tuple)):
            results = (results,)
        for index, result in enumerate(results):
            arr = np.asarray(result)
            path = os.path.join(debug_dir, f"xla_output_{index}.bin")
            arr.tofile(path)
        print(
            f"[elodin-cranelift] checkpoint: saved {len(results)} XLA reference outputs to {debug_dir}",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[elodin-cranelift] checkpoint: XLA reference failed: {exc}", file=sys.stderr)


def eval_output_specs(func: Any, input_specs: list[Any]) -> list[dict[str, Any]]:
    """Return output tensor specs from `jax.eval_shape`."""
    import jax

    shaped = jax.eval_shape(func, *input_specs)
    if isinstance(shaped, dict):
        raise TypeError("display kernels must return tensors, not mappings")
    outputs = shaped if isinstance(shaped, tuple) else (shaped,)
    specs: list[dict[str, Any]] = []
    for output in outputs:
        if not hasattr(output, "shape") or not hasattr(output, "dtype"):
            raise TypeError("display kernels must return tensors")
        shape = [int(dim) for dim in output.shape]
        dtype = str(output.dtype)
        specs.append({"shape": shape, "dtype": _canon_dtype(dtype)})
    return specs


def _canon_dtype(dtype: str) -> str:
    text = str(dtype).rsplit(".", maxsplit=1)[-1].lower()
    aliases = {
        "float64": "f64",
        "float32": "f32",
        "float16": "f16",
        "int64": "i64",
        "int32": "i32",
        "int16": "i16",
        "int8": "i8",
        "uint64": "u64",
        "uint32": "u32",
        "uint16": "u16",
        "uint8": "u8",
        "bool": "bool",
        "bool_": "bool",
    }
    return aliases.get(text, text)
