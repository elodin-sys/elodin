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

pytestmark = pytest.mark.parity

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


# The #922 repros as `prog(x, k) -> (y, yi)` over a float and an integer state.
VEC4_I64 = el.ComponentType(el.PrimitiveType.I64, (4,))
GX = ty.Annotated[jax.Array, el.Component("gist_x", VEC4)]
GK = ty.Annotated[jax.Array, el.Component("gist_k", VEC4_I64)]
GY = ty.Annotated[jax.Array, el.Component("gist_y", VEC4)]
GYI = ty.Annotated[jax.Array, el.Component("gist_yi", VEC4_I64)]
GP = ty.Annotated[jax.Array, el.Component("gist_p", el.ComponentType(el.PrimitiveType.F64, (80,)))]


@dataclass
class Gist(el.Archetype):
    x: GX
    k: GK
    y: GY
    yi: GYI


@dataclass
class PaddedGist(el.Archetype):
    x: GX
    k: GK
    y: GY
    yi: GYI
    p: GP


Z = jnp.zeros(4, jnp.int64)
M = jnp.array(
    [[0.5, -0.2, 0.1, 0.9], [0.3, 0.8, -0.4, 0.0], [-0.7, 0.1, 0.6, 0.2], [0.2, 0.2, -0.3, 1.1]]
)
double = jax.jit(lambda v: v * 2.0)


def i1cmp(x, k):
    a, b = x > 0.25, jnp.roll(x, 1) > 0.25
    return (a != b) + 10.0 * (a == jnp.zeros(4, bool)), Z


def c5(x, k):
    def more(c):
        return (c[0] < 3) & (jnp.abs(c[1]) < 3.0 + jnp.abs(jnp.sum(x)))

    def body(c):
        r = jax.lax.cond(x[0] > -10.0, lambda: c[1] * 0.5, lambda: x[1])
        return c[0] + 1, 0.5 * c[1] + r

    w = jnp.array([0.4, -1.1, 0.7, 0.2])
    out = jax.lax.while_loop(more, body, (0, jnp.dot(w, x) * x[0]))[1]
    return jnp.stack([out, x[1], x[2], x[3]]), Z


def window(x, init, op, stride, pad):
    return jnp.concatenate([jax.lax.reduce_window(x, init, op, (2,), (stride,), "VALID"), pad])


def sas(x, k, dtype=jnp.float64):
    def pooled(u):
        peaks = jax.lax.reduce_window(u, -jnp.inf, jax.lax.max, (2,), (1,), "VALID")
        return jnp.sum(peaks * jnp.array([1.0, 2.0, 3.0], dtype))

    return jax.grad(pooled)(x.astype(dtype)).astype(jnp.float64), Z


def f32_reduced(x, reduce):
    x32 = x.astype(jnp.float32)
    return (x32 * reduce(x32)).astype(jnp.float64), Z


def empty_case(x, k):
    # Both branches return the same capture, so JAX emits a case with no results.
    s = jnp.max(x)
    return x * jax.lax.cond(x[0] > 0, lambda: s, lambda: s), Z


def argmax_nan_gist(x, k):
    y = jnp.where(x > 0.5, jnp.nan, x)
    return x * jnp.argmax(y) + jnp.argmin(y), Z


def nan_max_min(x, k):
    y = jnp.where(x > 0.5, jnp.nan, x)
    return jnp.maximum(y, 0.1), jnp.isnan(jnp.minimum(y, 0.2))


def nan_reductions(x, k):
    y = jnp.where(x > 0.5, jnp.nan, x)
    return jnp.stack([jnp.max(y), jnp.min(y), jnp.max(y[:2]), jnp.min(y[2:])]), Z


def loop_dot_sum(x, k):
    # The loop body is a separate function, which can get the other ABI than its caller.
    w = jnp.array([0.4, -1.1, 0.7, 0.2])
    dot, total = jnp.dot(w, x), jnp.sum(x)

    def body(i, a):
        return a + (dot != jnp.dot(w, x)) + 10 * (total != jnp.sum(x))

    return x, jax.lax.fori_loop(0, 2, body, Z)


GIST_CASES = {
    "i1cmp": i1cmp,
    "c5": c5,
    "clamp_i64": lambda x, k: (x, jax.lax.clamp(jnp.int64(-1), k, jnp.int64(2))),
    "sign_i64": lambda x, k: (x, jnp.sign(k * 5 + 3)),
    "mod_i64": lambda x, k: (x, jnp.mod(k, 3) + 10 * jnp.floor_divide(k, 3)),
    "cumsum": lambda x, k: (jnp.cumsum(x), Z),
    "rw_add_s2": lambda x, k: (window(x, 0.0, jax.lax.add, 2, jnp.zeros(2)), Z),
    "rw_max_s1": lambda x, k: (window(x, -jnp.inf, jax.lax.max, 1, jnp.zeros(1)), Z),
    "sas": sas,
    "sort_i64": lambda x, k: (x, jnp.sort(k)),
    "argsort": lambda x, k: (x, jnp.argsort(k)),
    "select": lambda x, k: (jnp.select([x < -0.2, x < 0.2], [x * 2.0, x * 3.0], x), Z),
    "f32_max": lambda x, k: f32_reduced(x, jnp.max),
    "f32_ptr": lambda x, k: f32_reduced(x, jnp.sum),
    "i32_max": lambda x, k: (x, (k.astype(jnp.int32) * jnp.max(k.astype(jnp.int32)) * 3)),
    "jit_lambda": lambda x, k: (double(x), Z),
    "empty_case": empty_case,
    "argmax_axis0": lambda x, k: (x, jnp.argmax(jnp.outer(x, x[::-1] + 1.0), axis=0)),
    "argmax_nan": argmax_nan_gist,
    "matvec_batched": lambda x, k: ((jnp.stack([M, M.T]) @ x).sum(axis=0), Z),
    "loop_dot_sum": loop_dot_sum,
    "nan_max_min": nan_max_min,
    "nan_reductions": nan_reductions,
    "cumsum_i64": lambda x, k: (x, jnp.cumsum(k)),
    "cumsum_f32": lambda x, k: (jnp.cumsum(x.astype(jnp.float32)).astype(jnp.float64), Z),
    "cummax_i32": lambda x, k: (x, jax.lax.cummax(k.astype(jnp.int32))),
    "sas_f32": lambda x, k: sas(x, k, jnp.float32),
}


def gist_history(prog, backend: str, padded: bool) -> list[np.ndarray]:
    def outputs(x, k):
        y, yi = prog(x, k)
        r = jnp.roll(k, 1)
        return (
            jnp.array([-1.3, -0.9, -0.6, 1.2]) * jnp.roll(x, 1),
            jnp.where(r > 3, r - 9, r + 2),
            jnp.asarray(y, jnp.float64),
            jnp.asarray(yi, jnp.int64),
        )

    x0, k0 = jnp.array([0.3, 0.7, 0.2, 0.9]), jnp.array([1, -2, 3, 0], jnp.int64)
    if padded:

        @el.map
        def step(x: GX, k: GK, p: GP) -> tuple[GX, GK, GY, GYI, GP]:
            return *outputs(x, k), 0.999 * jnp.roll(p, 1) + 0.001 * jnp.sum(x)

        body = PaddedGist(x=x0, k=k0, y=jnp.zeros(4), yi=Z, p=jnp.arange(80.0))
    else:

        @el.map
        def step(x: GX, k: GK) -> tuple[GX, GK, GY, GYI]:
            return outputs(x, k)

        body = Gist(x=x0, k=k0, y=jnp.zeros(4), yi=Z)
    w = el.World()
    w.spawn(body, "g")
    exec = w.build(step, backend=backend)
    exec.run(5, show_progress=False)
    return [np.stack(exec.history(c)[c].to_numpy()) for c in ("g.gist_y", "g.gist_yi")]


# The padded world's 80-element component puts the system on the pointer ABI.
@pytest.mark.parametrize("padded", [False, True], ids=["small", "padded"])
@pytest.mark.parametrize("name", GIST_CASES)
def test_gist_prog_matches_jax_cpu(name, padded):
    want_y, want_yi = gist_history(GIST_CASES[name], "jax-cpu", padded)
    y, yi = gist_history(GIST_CASES[name], "cranelift", padded)
    np.testing.assert_allclose(y, want_y, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(yi, want_yi)
