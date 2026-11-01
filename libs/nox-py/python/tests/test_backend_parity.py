"""Cranelift must match jax-cpu on `lax` control flow and on the ops in #922.

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


def outer(x):
    return jnp.outer(x, x[::-1] + 1.0)


def weighted_rows(m):
    return (m * jnp.arange(1.0, 5.0)).sum(axis=1)


def big(x):
    # 144 elements put the whole function on the pointer ABI.
    return jnp.outer(jnp.tile(x, 3), jnp.tile(x, 3))


def scatter_column(x):
    return weighted_rows(outer(x).at[:, 1].set(x))


def scatter_dynamic_column(x):
    return weighted_rows(outer(x).at[:, jnp.int32(x[0] > 0.0) + 1].set(x))


def scatter_element(x):
    return weighted_rows(outer(x).at[2, 1].set(jnp.sum(x)))


def scatter_slice(x):
    return weighted_rows(outer(x).at[1:3, 2:4].set(jnp.outer(x[:2], x[2:])))


def scatter_add(x):
    return weighted_rows(outer(x).at[1].add(x))


def scatter_vector(x):
    return x.at[2].set(jnp.max(x))


def egm08_roll(x):
    return weighted_rows(jnp.roll(outer(x), 1, axis=1).at[:, -1].set(0.0))


def prod(x):
    return x * jnp.prod(x) + jnp.prod(outer(x), axis=1)


def argmax(x):
    return x * jnp.argmax(x) + jnp.argmin(x)


def argmax_axis(x):
    m = outer(x)
    return x * jnp.argmax(m, axis=0) + jnp.argmin(m, axis=1)


def argmax_nan(x):
    y = jnp.where(x > 0.5, jnp.nan, x)
    return x * jnp.argmax(y) + jnp.argmin(y)


def select(x):
    return jnp.select([x < -0.2, x < 0.2], [x * 2.0, x * 3.0], x)


def erfc(x):
    return jax.scipy.special.erfc(x) + jax.scipy.special.erfc(4.0 * x)


def where_scalar(x):
    return jnp.where(jnp.sum(x) > 0.0, big(x), -2.0 * big(x)).reshape(4, 36).sum(axis=1)


def reduce_axes(x):
    m = outer(x)
    return m.sum(axis=0) + m.max(axis=0) + jnp.sum(m[None] * m[:, None], axis=(0, 2))


def reduce_axes_big(x):
    b = big(x).reshape(4, 3, 12)
    return b.sum(axis=1).max(axis=1) + b.sum(axis=(1, 2))


def mekf_matmul(x):
    # h.T @ p written as a broadcast multiply and a last-axis sum.
    a = jnp.concatenate([outer(x)[:, :3], outer(x)[:2, 1:]])
    b = outer(x)[1:, :3]
    return jnp.sum(a[:, None, :] * b.T[None], -1)[:4].sum(axis=1)


def motor_thrust(x):
    # The per-motor sum is a middle-axis reduce once mapped over entities.
    arms = jnp.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]])
    thrust = jnp.abs(x)[:, None] * jnp.array([0.0, 0.0, 1.0])
    torque = jnp.cross(arms, thrust)
    return jnp.concatenate([jnp.sum(thrust, axis=0)[2:], jnp.sum(torque, axis=0)])


def matvec(x):
    return outer(x) @ jnp.arange(1.0, 5.0) + jnp.arange(4.0, 0.0, -1.0) @ outer(x)


def reduce_window(x):
    pairs = jax.lax.reduce_window(outer(x), 0.0, jax.lax.add, (1, 2), (1, 2), "VALID")
    peaks = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (3,), (1,), "SAME")
    return jnp.cumsum(x) + pairs.sum(axis=1) + peaks


def max_pool_grad(x):
    def pool(m):
        return jax.lax.reduce_window(m, -jnp.inf, jax.lax.max, (2, 2), (2, 2), "VALID").sum()

    return weighted_rows(jax.grad(pool)(outer(x)))


def history(fn, backend: str, entities: int = 1) -> np.ndarray:
    @el.map
    def step(x: X) -> tuple[X, Y]:
        return jnp.array([-1.3, -0.9, -0.6, 1.2]) * jnp.roll(x, 1), fn(x)

    w = el.World()
    names = [f"e{i}" for i in range(entities)]
    for i, name in enumerate(names):
        x = jnp.array([0.3, 0.7, 0.2, 0.9]) * (1.0 - 0.4 * i)
        w.spawn(Body(x=x, y=jnp.zeros(4)), name)
    exec = w.build(step, backend=backend)
    exec.run(6, show_progress=False)
    col = [f"{name}.parity_y" for name in names]
    return np.stack([np.stack(exec.history(c)[c].to_numpy()) for c in col])


CASES = [
    cond_reads_max,
    cond_nested,
    switch,
    fori_loop,
    drone_mixer,
    scatter_column,
    scatter_dynamic_column,
    scatter_element,
    scatter_slice,
    scatter_add,
    scatter_vector,
    egm08_roll,
    prod,
    argmax,
    argmax_axis,
    argmax_nan,
    select,
    erfc,
    where_scalar,
    reduce_axes,
    reduce_axes_big,
    mekf_matmul,
    motor_thrust,
    matvec,
    reduce_window,
    max_pool_grad,
]


@pytest.mark.parametrize("fn", CASES, ids=lambda fn: fn.__name__)
def test_cranelift_matches_jax_cpu(fn):
    expected = history(fn, "jax-cpu")
    np.testing.assert_allclose(history(fn, "cranelift"), expected, rtol=1e-12, atol=1e-12)


# Several entities run the system under `jax.vmap`.
@pytest.mark.parametrize("fn", CASES, ids=lambda fn: fn.__name__)
def test_cranelift_matches_jax_cpu_batched(fn):
    expected = history(fn, "jax-cpu", entities=3)
    np.testing.assert_allclose(
        history(fn, "cranelift", entities=3), expected, rtol=1e-12, atol=1e-12
    )


def test_matvec_on_the_pointer_abi():
    # Five entities make the batched matrix 80 elements, past the scalar ABI's 64.
    expected = history(matvec, "jax-cpu", entities=5)
    np.testing.assert_allclose(
        history(matvec, "cranelift", entities=5), expected, rtol=1e-12, atol=1e-12
    )
