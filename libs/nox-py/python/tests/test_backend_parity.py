"""Cranelift must match jax-cpu on `lax` control flow.

`stablehlo.case` / `stablehlo.while` regions read values from the enclosing
scope. The state is rotated and rescaled every tick so each predicate takes
both branches.
"""

import typing as ty
from dataclasses import dataclass

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
import pytest

VEC4 = el.ComponentType(el.PrimitiveType.F64, (4,))
X = ty.Annotated[jax.Array, el.Component("parity_x", VEC4)]
Y = ty.Annotated[jax.Array, el.Component("parity_y", VEC4)]


@dataclass
class Body(el.Archetype):
    x: X
    y: Y


def cond(pred, true_fn, false_fn):
    return jax.lax.cond(pred, true_fn, false_fn, operand=None)


def cond_reads_max(x):
    return x * cond(jnp.max(x) > 0.0, lambda _: jnp.max(x), lambda _: 1.0)


def cond_nested(x):
    def positive(_):
        return cond(jnp.max(x) > 1.0, lambda _: 3.0 * jnp.max(x), lambda _: 2.0 * jnp.max(x))

    return x * cond(jnp.max(x) > 0.0, positive, lambda _: 1.0)


def switch(x):
    # The index runs 3, 0, 1, 2, -1, 2; switch clamps the -1 to branch 0.
    branches = [lambda: 1.0, lambda: jnp.max(x), lambda: jnp.min(x), lambda: jnp.sum(x)]
    return x * jax.lax.switch(jnp.sum(x > 0.0) - 1, branches)


def fori_loop(x):
    return jax.lax.fori_loop(0, 3, lambda i, acc: 0.5 * acc + jnp.max(x) * i, x)


def drone_mixer(out):
    # Saturation guard from examples/drone/motors.py.
    low, high = jnp.min(out), jnp.max(out)
    scale = cond(high - low > 1.0, lambda _: 1.0 / (high - low), lambda _: 1.0)
    scale = cond(
        low + 0.3 < 0.0, lambda _: jnp.min(jnp.array([scale, -0.3 / low])), lambda _: scale
    )
    return out * scale


def history(fn, backend: str) -> np.ndarray:
    @el.map
    def step(x: X) -> tuple[X, Y]:
        return jnp.array([-1.3, -0.9, -0.6, 1.2]) * jnp.roll(x, 1), fn(x)

    w = el.World()
    w.spawn(Body(x=jnp.array([0.3, 0.7, 0.2, 0.9]), y=jnp.zeros(4)), "e")
    exec = w.build(step, backend=backend)
    exec.run(6, show_progress=False)
    return np.stack(exec.history("e.parity_y")["e.parity_y"].to_numpy())


CASES = [cond_reads_max, cond_nested, switch, fori_loop, drone_mixer]


@pytest.mark.parametrize("fn", CASES, ids=lambda fn: fn.__name__)
def test_cranelift_matches_jax_cpu(fn):
    expected = history(fn, "jax-cpu")
    np.testing.assert_allclose(history(fn, "cranelift"), expected, rtol=1e-12, atol=1e-12)
