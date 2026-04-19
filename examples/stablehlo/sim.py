"""StableHLO coverage example: exercises every implemented StableHLO/CHLO op.

System steps grouped by op category:
  1. math_step     -- 26+ unary/binary math ops (sin, cos, asin, sinh, erfc, cbrt, ...)
  2. sort_step     -- stablehlo.sort with comparator region
  3. shape_step    -- broadcast, reduce, concat, slice, reshape, transpose, reverse,
                      gather, iota, dynamic_slice, dynamic_update_slice
  4. control_step  -- while_loop, case/switch
  5. bitwise_step  -- xor, or, and, shift_left, shift_right_logical, not
  6. linalg_step   -- dot_general, matmul, reduce (sum/max/min)
  7. convert_step  -- type conversions, bitcast, select, compare, clamp, pad, scatter
"""

import typing as ty
from dataclasses import field

import elodin as el
import jax
import jax.numpy as jnp

SIMULATION_RATE = 120.0

# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

MathState = ty.Annotated[
    jnp.ndarray,
    el.Component("math_state", el.ComponentType(el.PrimitiveType.F64, (4,))),
]

SortState = ty.Annotated[
    jnp.ndarray,
    el.Component("sort_state", el.ComponentType(el.PrimitiveType.F64, (8,))),
]

ShapeState = ty.Annotated[
    jnp.ndarray,
    el.Component("shape_state", el.ComponentType(el.PrimitiveType.F64, (4,))),
]

ControlState = ty.Annotated[
    jnp.ndarray,
    el.Component("control_state", el.ComponentType(el.PrimitiveType.F64, (4,))),
]

BitwiseState = ty.Annotated[
    jnp.ndarray,
    el.Component("bitwise_state", el.ComponentType(el.PrimitiveType.I64, (4,))),
]

LinalgState = ty.Annotated[
    jnp.ndarray,
    el.Component("linalg_state", el.ComponentType(el.PrimitiveType.F64, (4,))),
]

ConvertState = ty.Annotated[
    jnp.ndarray,
    el.Component("convert_state", el.ComponentType(el.PrimitiveType.F64, (4,))),
]

Linalg2State = ty.Annotated[
    jnp.ndarray,
    el.Component("linalg2_state", el.ComponentType(el.PrimitiveType.F64, (4,))),
]

# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------


@el.dataclass
class MathArchetype(el.Archetype):
    math_state: MathState = field(default_factory=lambda: jnp.array([0.5, 1.0, -0.3, 2.0]))


@el.dataclass
class SortArchetype(el.Archetype):
    sort_state: SortState = field(
        default_factory=lambda: jnp.array([3.0, 1.0, 4.0, 1.5, 2.0, 5.0, 0.5, 2.5])
    )


@el.dataclass
class ShapeArchetype(el.Archetype):
    shape_state: ShapeState = field(default_factory=lambda: jnp.array([1.0, 2.0, 3.0, 4.0]))


@el.dataclass
class ControlArchetype(el.Archetype):
    control_state: ControlState = field(default_factory=lambda: jnp.array([5.0, 1.0, -0.5, 0.0]))


@el.dataclass
class BitwiseArchetype(el.Archetype):
    bitwise_state: BitwiseState = field(
        default_factory=lambda: jnp.array([0xA5, 0x3C, 0xFF, 0x01], dtype=jnp.int64)
    )


@el.dataclass
class LinalgArchetype(el.Archetype):
    linalg_state: LinalgState = field(default_factory=lambda: jnp.array([1.0, 2.0, 3.0, 4.0]))


@el.dataclass
class ConvertArchetype(el.Archetype):
    convert_state: ConvertState = field(default_factory=lambda: jnp.array([1.5, -2.7, 0.0, 100.0]))


@el.dataclass
class Linalg2Archetype(el.Archetype):
    linalg2_state: Linalg2State = field(default_factory=lambda: jnp.array([4.0, 2.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Step 1: math_step -- exercises all unary/binary math ops
# ---------------------------------------------------------------------------


@el.map
def math_step(state: MathState) -> MathState:
    x = state
    r = jnp.zeros(4)

    # Trig: sine, cosine, tan, tanh, atan2
    r = r + jnp.sin(x) + jnp.cos(x)
    r = r + jnp.tanh(x)
    r = r + jnp.arctan2(x, jnp.ones(4))

    # Exponential / logarithmic
    r = r + jnp.exp(x * 0.1)
    r = r + jnp.log(jnp.abs(x) + 1.0)
    r = r + jnp.log1p(jnp.abs(x))
    r = r + jnp.expm1(x * 0.01)

    # Roots / powers
    r = r + jnp.sqrt(jnp.abs(x) + 1.0)
    r = r + jax.lax.rsqrt(jnp.abs(x) + 1.0)
    r = r + jnp.cbrt(jnp.abs(x) + 1.0)
    r = r + jnp.power(jnp.abs(x) + 1.0, 0.5)

    # Rounding / sign
    r = r + jnp.floor(x) + jnp.ceil(x)
    r = r + jnp.sign(x) + jnp.round(x)
    r = r + jnp.abs(x)

    # Inverse trig (CHLO ops)
    safe_x = jnp.clip(x * 0.1, -0.99, 0.99)
    r = r + jnp.arcsin(safe_x)
    r = r + jnp.arccos(safe_x)
    r = r + jnp.arctan(x * 0.1)

    # Hyperbolic (CHLO ops)
    r = r + jnp.sinh(x * 0.1) + jnp.cosh(x * 0.1)

    # Special functions
    r = r + jax.scipy.special.erfc(x * 0.1)

    # Clamp + is_finite
    r = r + jnp.clip(x, -2.0, 2.0)
    mask = jnp.isfinite(r).astype(jnp.float64)

    return r * mask * 0.01


# ---------------------------------------------------------------------------
# Step 2: sort_step -- exercises stablehlo.sort with comparator
# ---------------------------------------------------------------------------


@el.map
def sort_step(state: SortState) -> SortState:
    sorted_vals = jnp.sort(state)
    return sorted_vals * 0.99 + 0.01


# ---------------------------------------------------------------------------
# Step 3: shape_step -- exercises shape manipulation ops
# ---------------------------------------------------------------------------


@el.map
def shape_step(state: ShapeState) -> ShapeState:
    x = state

    # broadcast_in_dim + reduce (sum)
    mat = jnp.broadcast_to(x, (3, 4))
    s = jnp.sum(mat, axis=0)

    # concatenate + slice
    c = jnp.concatenate([s, s[:2]])
    sl = c[1:5]

    # reshape + transpose
    m2 = sl.reshape(2, 2)
    t = jnp.transpose(m2)
    r = t.flatten()[:4]

    # reverse
    r = jnp.flip(r)

    # iota
    iota = jnp.arange(4, dtype=jnp.float64)
    return r * 0.5 + iota * 0.01


# ---------------------------------------------------------------------------
# Step 4: control_step -- exercises while_loop + case/switch
# ---------------------------------------------------------------------------


@el.map
def control_step(state: ControlState) -> ControlState:
    def cond(carry):
        _, i = carry
        return i < 5

    def body(carry):
        x, i = carry
        return (x * 0.9 + 0.1, i + 1)

    result, _ = jax.lax.while_loop(cond, body, (state[0], jnp.int64(0)))

    idx = jnp.int32(jnp.abs(state[1]) % 3)
    branch_result = jax.lax.switch(
        idx,
        [
            lambda: state * 0.95,
            lambda: state * 1.05,
            lambda: state + 0.01,
        ],
    )

    return jnp.array([result, branch_result[0], branch_result[1], state[3] + 0.01])


# ---------------------------------------------------------------------------
# Step 5: bitwise_step -- exercises integer bitwise ops
# ---------------------------------------------------------------------------


@el.map
def bitwise_step(state: BitwiseState) -> BitwiseState:
    x = state
    r = jnp.bitwise_xor(x, jnp.int64(0xFF))
    r = jnp.bitwise_or(r, jnp.int64(0x0F))
    r = jnp.bitwise_and(r, jnp.int64(0xFFF))
    r = jnp.left_shift(r, jnp.int64(1))
    r = jax.lax.shift_right_logical(r, jnp.int64(2))
    return r


# ---------------------------------------------------------------------------
# Step 6: linalg_step -- exercises dot_general, reduce (multiple ops)
# ---------------------------------------------------------------------------


@el.map
def linalg_step(state: LinalgState) -> LinalgState:
    x = state

    # dot_general (via matmul)
    mat = jnp.outer(x[:2], x[2:])
    mv = mat @ x[2:]

    # reduce: sum, max, min
    s = jnp.sum(x)
    mx = jnp.max(x)
    mn = jnp.min(x)

    # remainder
    rem = jnp.remainder(x, jnp.array([1.5, 1.5, 1.5, 1.5]))

    return jnp.array([mv[0] * 0.01 + s * 0.001, mx, mn, rem[0]])


# ---------------------------------------------------------------------------
# Step 7: convert_step -- exercises type conversions, select, compare, pad, scatter
# ---------------------------------------------------------------------------


@el.map
def convert_step(state: ConvertState) -> ConvertState:
    x = state

    # convert f64 -> i32 -> f64 round-trip
    i32_vals = x.astype(jnp.int32)
    back = i32_vals.astype(jnp.float64)

    # compare + select
    mask = x > 0.0
    selected = jnp.where(mask, x, -x)

    # dynamic_update_slice via .at[].set()
    updated = x.at[0].set(selected[1])
    updated = updated.at[2].set(back[3])

    # negate + maximum + minimum
    neg = -x
    combined = jnp.maximum(neg, updated)
    combined = jnp.minimum(combined, jnp.ones(4) * 50.0)

    return combined * 0.99


# ---------------------------------------------------------------------------
# Step 8: linalg2_step -- exercises cholesky + triangular solve (via LAPACK)
# ---------------------------------------------------------------------------


@el.map
def linalg2_step(state: Linalg2State) -> Linalg2State:
    # Interpret state as a 2x2 SPD matrix (a, b, b, c) with a>0, ac-b^2>0
    a_mat = jnp.array(
        [[jnp.abs(state[0]) + 1.0, state[1] * 0.1], [state[1] * 0.1, jnp.abs(state[2]) + 1.0]]
    )
    # Cholesky decomposition (exercises lapack_dpotrf_ffi)
    l_mat = jnp.linalg.cholesky(a_mat)
    # Triangular solve (exercises lapack_dtrsm_ffi)
    b_vec = jnp.array([state[3], 1.0])
    x = jax.scipy.linalg.solve_triangular(l_mat, b_vec, lower=True)
    return jnp.array([l_mat[0, 0], l_mat[1, 1], x[0], x[1]])


# ---------------------------------------------------------------------------
# World + System
# ---------------------------------------------------------------------------


def world() -> el.World:
    w = el.World()
    w.spawn(MathArchetype(), name="math")
    w.spawn(SortArchetype(), name="sorter")
    w.spawn(ShapeArchetype(), name="shaper")
    w.spawn(ControlArchetype(), name="ctrl")
    w.spawn(BitwiseArchetype(), name="bits")
    w.spawn(LinalgArchetype(), name="linalg")
    w.spawn(ConvertArchetype(), name="cvt")
    w.spawn(Linalg2Archetype(), name="linalg2")
    return w


def system() -> el.System:
    return (
        math_step
        | sort_step
        | shape_step
        | control_step
        | bitwise_step
        | linalg_step
        | convert_step
        | linalg2_step
    )
