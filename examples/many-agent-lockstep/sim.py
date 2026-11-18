"""The world: agents that follow moving objects, stepped at 1 kHz."""

import math
import os
import typing as ty

import elodin as el
import jax
import jax.numpy as jnp


RATE_HZ = 1000.0
DT = 1.0 / RATE_HZ
LAYOUT_SLOTS = 24
# Extra components and compute rounds give each entity the size and cost of a real vehicle model.
AUX_COMPONENT_COUNT = int(os.environ.get("ELODIN_MANY_AGENT_AUX_COMPONENTS", "32"))
COMPUTE_ROUNDS = int(os.environ.get("ELODIN_MANY_AGENT_COMPUTE_ROUNDS", "64"))


def component(name: str, shape: tuple[int, ...] = (3,), metadata=None):
    return ty.Annotated[
        jax.Array,
        el.Component(
            name,
            el.ComponentType(el.PrimitiveType.F64, shape),
            metadata=metadata or {},
        ),
    ]


Position = component("position")
Velocity = component("velocity")
Command = component("command", metadata={"external_control": "true"})
Acceleration = component("acceleration")
SensorFrame = component("sensor_frame", (16,))
DiagnosticBank = component("diagnostic_bank", (32,))
Trace0 = component("trace_0", (4,))
Trace1 = component("trace_1", (4,))
Trace2 = component("trace_2", (4,))
Trace3 = component("trace_3", (4,))
Trace4 = component("trace_4", (4,))
Trace5 = component("trace_5", (4,))
Trace6 = component("trace_6", (4,))
Trace7 = component("trace_7", (4,))

ObjectPosition = component("object_position")
ObjectVelocity = component("object_velocity")
ObjectRate = component("object_rate", (1,))
ObjectSignal = component("object_signal", (16,))
AUX_COMPONENTS = tuple(component(f"aux_{index:02d}", (4,)) for index in range(AUX_COMPONENT_COUNT))


def compute_bank(seed: jax.Array) -> jax.Array:
    def compute_round(index, value):
        offset = jnp.asarray(index, dtype=jnp.float64) * 1.0e-4
        return 0.5 * value + 0.25 * jnp.sin(value + offset) + 0.25 * jnp.cos(value - offset)

    return jax.lax.fori_loop(0, COMPUTE_ROUNDS, compute_round, seed)


@el.map
def step_agent(
    position: Position,
    velocity: Velocity,
    command: Command,
) -> tuple[
    Position,
    Velocity,
    Acceleration,
    SensorFrame,
    DiagnosticBank,
    Trace0,
    Trace1,
    Trace2,
    Trace3,
    Trace4,
    Trace5,
    Trace6,
    Trace7,
]:
    speed_sq = jnp.dot(velocity, velocity)
    drag = velocity * (0.03 + 0.001 * speed_sq)
    acceleration = jnp.clip(command, -12.0, 12.0) - drag
    next_velocity = velocity + acceleration * DT
    next_position = position + next_velocity * DT
    phase = jnp.concatenate(
        [
            next_position,
            next_velocity,
            acceleration,
            jnp.sin(next_position),
            jnp.cos(next_velocity),
            jnp.array([speed_sq]),
        ]
    )
    diagnostics = compute_bank(jnp.tile(phase, 2)[:32])
    trace = jnp.array(
        [
            jnp.dot(next_position, next_position),
            jnp.dot(next_velocity, next_velocity),
            jnp.dot(acceleration, acceleration),
            jnp.dot(next_position, next_velocity),
        ]
    )
    return (
        next_position,
        next_velocity,
        acceleration,
        phase,
        diagnostics,
        trace,
        trace * 1.01,
        trace * 1.02,
        trace * 1.03,
        trace * 1.04,
        trace * 1.05,
        trace * 1.06,
        trace * 1.07,
    )


@el.map
def step_object(
    position: ObjectPosition,
    velocity: ObjectVelocity,
    rate: ObjectRate,
) -> tuple[ObjectPosition, ObjectVelocity, ObjectSignal]:
    acceleration = -(rate[0] * rate[0]) * position
    next_velocity = velocity + acceleration * DT
    next_position = position + next_velocity * DT
    signal = compute_bank(
        jnp.tile(
            jnp.concatenate([next_position, next_velocity, acceleration, jnp.sin(next_position)]),
            2,
        )[:16]
    )
    return next_position, next_velocity, signal


def _ring(index: int, count: int, radius: float) -> tuple[jax.Array, jax.Array]:
    angle = 2.0 * math.pi * index / max(count, LAYOUT_SLOTS)
    position = jnp.array(
        [radius * math.cos(angle), radius * math.sin(angle), 0.25 * index],
        dtype=jnp.float64,
    )
    velocity = jnp.array(
        [-math.sin(angle), math.cos(angle), 0.0],
        dtype=jnp.float64,
    )
    return position, velocity


def build(
    agent_count: int, object_count: int
) -> tuple[el.World, el.System, list[el.EntityId], list[el.EntityId]]:
    if agent_count < 1 or object_count < 1:
        raise ValueError("agent and object counts must be positive")
    world = el.World()
    agents = []
    for index in range(agent_count):
        position, velocity = _ring(index, agent_count, 20.0)
        zeros3 = jnp.zeros(3, dtype=jnp.float64)
        zeros4 = jnp.zeros(4, dtype=jnp.float64)
        agent = world.spawn(
            [
                el.C(Position, position),
                el.C(Velocity, velocity),
                el.C(Command, zeros3),
                el.C(Acceleration, zeros3),
                el.C(SensorFrame, jnp.zeros(16, dtype=jnp.float64)),
                el.C(DiagnosticBank, jnp.zeros(32, dtype=jnp.float64)),
                el.C(Trace0, zeros4),
                el.C(Trace1, zeros4),
                el.C(Trace2, zeros4),
                el.C(Trace3, zeros4),
                el.C(Trace4, zeros4),
                el.C(Trace5, zeros4),
                el.C(Trace6, zeros4),
                el.C(Trace7, zeros4),
                *(el.C(auxiliary, zeros4) for auxiliary in AUX_COMPONENTS),
            ],
            name=f"agent_{index:03d}",
        )
        agents.append(agent)
    objects = []
    for index in range(object_count):
        position, tangent = _ring(index, object_count, 32.0)
        rate = 0.08 + index * 0.0005
        observed = world.spawn(
            [
                el.C(ObjectPosition, position),
                el.C(ObjectVelocity, tangent * (32.0 * rate)),
                el.C(ObjectRate, jnp.array([rate], dtype=jnp.float64)),
                el.C(ObjectSignal, jnp.zeros(16, dtype=jnp.float64)),
                *(el.C(auxiliary, jnp.zeros(4, dtype=jnp.float64)) for auxiliary in AUX_COMPONENTS),
            ],
            name=f"object_{index:03d}",
        )
        objects.append(observed)
    return world, step_agent | step_object, agents, objects
