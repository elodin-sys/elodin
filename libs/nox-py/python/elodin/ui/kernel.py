"""Python-authored JAX display kernels for graphs and 3D objects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import update_wrapper
from typing import Any, Callable

from .expr import ComponentHandle, Expr, ExprError

BATCH_SIZE = 256
ASSET_PREFIX = "schematics/kernels/"


class KernelError(ExprError):
    """Raised when a display kernel cannot be traced or lowered."""


@dataclass(frozen=True)
class KernelInput:
    name: str
    component: str
    shape: tuple[int, ...]
    dtype: str


@dataclass
class KernelArtifact:
    hash: str
    batch_size: int
    inputs: list[KernelInput]
    outputs: list[dict[str, Any]]
    scalar_mlir: str
    batched_mlir: str

    @property
    def asset_key(self) -> str:
        return f"{ASSET_PREFIX}{self.hash}"

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "hash": self.hash,
            "batch_size": self.batch_size,
            "inputs": [
                {
                    "name": item.name,
                    "component": item.component,
                    "shape": list(item.shape),
                    "dtype": item.dtype,
                }
                for item in self.inputs
            ],
            "outputs": self.outputs,
            "scalar_mlir": self.scalar_mlir,
            "batched_mlir": self.batched_mlir,
        }


class KernelExpr:
    """Compiled display-kernel binding produced by calling a `@ui.kernel` function."""

    __slots__ = ("_func", "_inputs", "_name", "_artifact")

    def __init__(self, func: Callable[..., Any], inputs: list[KernelInput], name: str):
        self._func = func
        self._inputs = inputs
        self._name = name
        self._artifact: KernelArtifact | None = None

    def __repr__(self) -> str:
        args = ", ".join(item.component for item in self._inputs)
        return f"KernelExpr({self._name}({args}))"

    def artifact(self) -> KernelArtifact:
        if self._artifact is None:
            self._artifact = compile_kernel(self._func, self._inputs, name=self._name)
        return self._artifact

    @property
    def _display_kernel(self) -> dict[str, Any]:
        artifact = self.artifact()
        return {
            "hash": artifact.hash,
            "asset": artifact.asset_key,
            "inputs": [
                {
                    "component": item.component,
                    "shape": list(item.shape),
                    "dtype": item.dtype,
                }
                for item in self._inputs
            ],
            "sidecar": artifact.to_json_bytes(),
        }


class _KernelDecorator:
    def __init__(self, func: Callable[..., Any]):
        self._func = func
        update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> KernelExpr:
        if kwargs:
            raise KernelError("@ui.kernel does not accept keyword bindings")
        expected = _positional_count(self._func)
        if len(args) != expected:
            raise KernelError(f"{self._func.__name__} expected {expected} inputs, got {len(args)}")
        inputs = [
            _bind_input(arg, name) for arg, name in zip(args, _arg_names(self._func), strict=True)
        ]
        return KernelExpr(self._func, inputs, self._func.__name__)


def kernel(func: Callable[..., Any] | None = None) -> Any:
    """Mark a JAX function as a display kernel.

    Calling the wrapped function with component handles compiles scalar and
    batched StableHLO modules and returns a :class:`KernelExpr`.
    """

    if func is None:
        return kernel
    return _KernelDecorator(func)


def compile_kernel(
    func: Callable[..., Any],
    inputs: list[KernelInput],
    *,
    name: str = "kernel",
    batch_size: int = BATCH_SIZE,
    validate: bool = True,
) -> KernelArtifact:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise KernelError("JAX is required to compile display kernels") from exc

    from elodin.stablehlo import eval_output_specs, lower_to_stablehlo

    jax.config.update("jax_enable_x64", True)
    scalar_specs = [_shape_dtype(item, jnp) for item in inputs]
    for spec in scalar_specs:
        if any(dim is None for dim in spec.shape):
            raise KernelError(f"{name} has a dynamic input shape {spec.shape}")

    try:
        outputs = eval_output_specs(func, scalar_specs)
    except Exception as exc:  # noqa: BLE001
        raise KernelError(f"{name} failed shape inference: {exc}") from exc
    if not outputs:
        raise KernelError(f"{name} produced no outputs")
    for output in outputs:
        if any(dim < 0 for dim in output["shape"]):
            raise KernelError(f"{name} produced a dynamic output shape {output['shape']}")

    batched = jax.vmap(func)
    batched_specs = [
        jax.ShapeDtypeStruct((batch_size, *item.shape), spec.dtype)
        for item, spec in zip(inputs, scalar_specs, strict=True)
    ]
    try:
        scalar_mlir = lower_to_stablehlo(func, scalar_specs)
        batched_mlir = lower_to_stablehlo(batched, batched_specs)
    except Exception as exc:  # noqa: BLE001
        raise KernelError(f"{name} failed StableHLO lowering: {exc}") from exc

    digest = _artifact_hash(batch_size, inputs, outputs, scalar_mlir, batched_mlir)
    artifact = KernelArtifact(
        hash=digest,
        batch_size=batch_size,
        inputs=list(inputs),
        outputs=outputs,
        scalar_mlir=scalar_mlir,
        batched_mlir=batched_mlir,
    )
    if validate:
        _validate_artifact(artifact)
    return artifact


def _validate_artifact(artifact: KernelArtifact) -> None:
    try:
        from elodin.elodin import ui as native
    except ImportError:
        return
    validate = getattr(native, "validate_stablehlo", None)
    if validate is None:
        return
    try:
        validate(artifact.scalar_mlir)
        validate(artifact.batched_mlir)
    except Exception as exc:  # noqa: BLE001
        raise KernelError(f"Cranelift rejected display kernel: {exc}") from exc


def _artifact_hash(
    batch_size: int,
    inputs: list[KernelInput],
    outputs: list[dict[str, Any]],
    scalar_mlir: str,
    batched_mlir: str,
) -> str:
    payload = {
        "batch_size": batch_size,
        "inputs": [
            {
                "name": item.name,
                "component": item.component,
                "shape": list(item.shape),
                "dtype": item.dtype,
            }
            for item in inputs
        ],
        "outputs": outputs,
        "scalar_mlir": scalar_mlir,
        "batched_mlir": batched_mlir,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _bind_input(value: Any, name: str) -> KernelInput:
    if isinstance(value, ComponentHandle):
        shape = tuple(value.shape or ())
        dtype = _canon_prim(getattr(value, "prim_type", None))
        return KernelInput(name=name, component=value._name, shape=shape, dtype=dtype)
    if isinstance(value, Expr):
        return KernelInput(name=name, component=str(value), shape=(), dtype="f64")
    if isinstance(value, str):
        return KernelInput(name=name, component=value, shape=(), dtype="f64")
    raise TypeError(f"kernel input {name} must be a component handle or EQL name")


def _canon_prim(prim_type: Any) -> str:
    if prim_type is None:
        return "f64"
    text = str(prim_type).rsplit(".", maxsplit=1)[-1].lower()
    aliases = {
        "f64": "f64",
        "float64": "f64",
        "f32": "f32",
        "float32": "f32",
        "i64": "i64",
        "int64": "i64",
        "i32": "i32",
        "int32": "i32",
        "u64": "u64",
        "u32": "u32",
        "bool": "bool",
        "bool_": "bool",
    }
    return aliases.get(text, "f64")


def _shape_dtype(item: KernelInput, jnp: Any) -> Any:
    import jax

    dtype = {
        "f64": jnp.float64,
        "f32": jnp.float32,
        "i64": jnp.int64,
        "i32": jnp.int32,
        "u64": jnp.uint64,
        "u32": jnp.uint32,
        "bool": jnp.bool_,
    }.get(item.dtype, jnp.float64)
    return jax.ShapeDtypeStruct(item.shape, dtype)


def _arg_names(func: Callable[..., Any]) -> list[str]:
    code = getattr(func, "__code__", None)
    if code is None:
        return []
    return list(code.co_varnames[: code.co_argcount])


def _positional_count(func: Callable[..., Any]) -> int:
    code = getattr(func, "__code__", None)
    if code is None:
        return 0
    return code.co_argcount
