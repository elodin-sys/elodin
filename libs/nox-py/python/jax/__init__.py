from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as _np

_STUB_DIR = Path(__file__).resolve().parent
_PARENT = _STUB_DIR.parent.resolve()


def _real_jax_module() -> types.ModuleType | None:
    if os.environ.get("ELODIN_FORCE_JAX_STUB") == "1":
        return None

    search_paths = [
        entry
        for entry in sys.path
        if not entry or Path(entry).resolve() != _PARENT
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONSAFEPATH", "0")
    env["PYTHONPATH"] = os.pathsep.join(search_paths)
    try:
        subprocess.run(
            [sys.executable, "-c", "import jax"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except Exception:
        return None

    spec = importlib.machinery.PathFinder.find_spec(__name__, search_paths)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


_real = _real_jax_module()
if _real is not None:
    sys.modules[__name__] = _real
    globals().update(_real.__dict__)
else:

    class _Config:
        def update(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
            return None

    def _tree_flatten(tree: Any) -> tuple[list[Any], Any]:
        if isinstance(tree, tuple):
            children = [_tree_flatten(item) for item in tree]
            return (
                [leaf for flat, _ in children for leaf in flat],
                ("tuple", tuple(spec for _, spec in children)),
            )
        if isinstance(tree, list):
            children = [_tree_flatten(item) for item in tree]
            return (
                [leaf for flat, _ in children for leaf in flat],
                ("list", tuple(spec for _, spec in children)),
            )
        if isinstance(tree, Mapping):
            items = [(key, _tree_flatten(value)) for key, value in tree.items()]
            return (
                [leaf for _, (flat, _) in items for leaf in flat],
                ("dict", tuple((key, spec) for key, (_, spec) in items)),
            )
        return [tree], ("leaf",)

    def _tree_unflatten(meta: Any, leaves: Iterable[Any]) -> Any:
        iterator = iter(leaves)

        def _build(spec: Any) -> Any:
            kind = spec[0]
            if kind == "leaf":
                return next(iterator)
            if kind == "tuple":
                return tuple(_build(child) for child in spec[1])
            if kind == "list":
                return [_build(child) for child in spec[1]]
            if kind == "dict":
                return {key: _build(child) for key, child in spec[1]}
            raise TypeError(f"Unsupported pytree spec: {spec!r}")

        result = _build(meta)
        try:
            next(iterator)
        except StopIteration:
            return result
        raise ValueError("Too many leaves provided to tree_unflatten")

    def _slice_arg(arg: Any, index: int) -> Any:
        if isinstance(arg, tuple):
            return tuple(_slice_arg(item, index) for item in arg)
        if isinstance(arg, list):
            return [_slice_arg(item, index) for item in arg]
        if isinstance(arg, Mapping):
            return {key: _slice_arg(value, index) for key, value in arg.items()}
        if hasattr(arg, "__getitem__"):
            return arg[index]
        return arg

    def _length_of(arg: Any) -> int:
        if isinstance(arg, Mapping):
            values = iter(arg.values())
            first = next(values, None)
            return 0 if first is None else _length_of(first)
        if isinstance(arg, (list, tuple)):
            if not arg:
                return 0
            try:
                return _length_of(arg[0])
            except TypeError:
                return len(arg)
        if hasattr(arg, "shape") and getattr(arg, "shape", ()):
            return int(arg.shape[0])  # type: ignore[index]
        if hasattr(arg, "__len__"):
            return len(arg)  # type: ignore[arg-type]
        raise TypeError("Cannot determine length for vmap input")

    def _stack_outputs(outputs: list[Any]) -> Any:
        if not outputs:
            return []
        first = outputs[0]
        if isinstance(first, tuple):
            return tuple(_stack_outputs([o[i] for o in outputs]) for i in range(len(first)))
        if isinstance(first, list):
            return [_stack_outputs([o[i] for o in outputs]) for i in range(len(first))]
        if isinstance(first, Mapping):
            return {key: _stack_outputs([o[key] for o in outputs]) for key in first}
        if isinstance(first, _np.ndarray):
            return _np.stack(outputs, axis=0)
        return _np.asarray(outputs)

    def vmap(func, in_axes: int | tuple[int, ...] = 0, out_axes: int = 0):  # pragma: no cover - thin wrapper
        del in_axes, out_axes

        def wrapper(*args: Any) -> Any:
            if not args:
                raise TypeError("vmap wrapper requires at least one argument")
            target = args if len(args) > 1 else args[0]
            results = [
                func(*(_slice_arg(arg, idx) for arg in args))
                for idx in range(_length_of(target))
            ]
            return _stack_outputs(results)

        return wrapper

    def _scan(func, init, xs):  # pragma: no cover - thin wrapper
        carry = init
        ys: list[Any] = []
        for idx in range(_length_of(xs)):
            carry, y = func(carry, _slice_arg(xs, idx))
            ys.append(y)
        return carry, _stack_outputs(ys)

    def _as_int(value: Any) -> int:
        array = _np.asarray(value).reshape(-1)
        return int(array[0])

    def _random_key(seed: Any) -> int:
        return _as_int(seed) & 0xFFFFFFFF

    def _random_fold_in(key: int, data: Any) -> int:
        return (_as_int(key) * 1103515245 + 12345 + _as_int(data)) & 0xFFFFFFFF

    def _random_uniform(
        key: int,
        shape: tuple[int, ...] | int | None = (),
        minval: float = 0.0,
        maxval: float = 1.0,
    ) -> _np.ndarray:
        rng = _np.random.default_rng(_as_int(key))
        size = None if shape in (None, ()) else shape
        return rng.uniform(minval, maxval, size=size)

    numpy = _np
    config = _Config()
    Array = _np.ndarray

    tree_util = types.ModuleType("jax.tree_util")
    tree_util.tree_flatten = _tree_flatten  # type: ignore[attr-defined]
    tree_util.tree_unflatten = _tree_unflatten  # type: ignore[attr-defined]
    tree_util.register_pytree_node = lambda *_, **__: None  # type: ignore[attr-defined]

    lax = types.ModuleType("jax.lax")
    lax.scan = _scan  # type: ignore[attr-defined]

    random = types.ModuleType("jax.random")
    random.key = _random_key  # type: ignore[attr-defined]
    random.fold_in = _random_fold_in  # type: ignore[attr-defined]
    random.uniform = _random_uniform  # type: ignore[attr-defined]

    typing = types.ModuleType("jax.typing")
    typing.ArrayLike = _np.ndarray  # type: ignore[attr-defined]

    sys.modules.setdefault("jax", sys.modules[__name__])
    sys.modules.setdefault("jax.numpy", numpy)
    sys.modules.setdefault("jax.random", random)
    sys.modules.setdefault("jax.tree_util", tree_util)
    sys.modules.setdefault("jax.lax", lax)
    sys.modules.setdefault("jax.typing", typing)

    __all__ = [
        "Array",
        "config",
        "numpy",
        "random",
        "lax",
        "tree_util",
        "typing",
        "vmap",
    ]

