"""Elodin adapters for course scene entities and referee telemetry.

The course/referee domain logic remains in dependency-light modules so geometry
and scoring tests do not need Elodin, Betaflight, an editor, or a GPU.
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp
from course import Course, course_bars
from referee import (
    CORE_COURSE_GATE_CAPACITY,
    LAST_GATE_UNSET,
    PASS_TIME_UNSET_S,
)

LastGatePassed = ty.Annotated[
    jax.Array,
    el.Component(
        "last_gate_passed",
        el.ComponentType(el.PrimitiveType.I64, (1,)),
        metadata={
            "priority": 90,
            "element_names": "ordered_gate_index",
            "external_control": "true",
            "unset_sentinel": str(LAST_GATE_UNSET),
        },
    ),
]

GatePassTimes = ty.Annotated[
    jax.Array,
    el.Component(
        "gate_pass_times",
        el.ComponentType(el.PrimitiveType.F64, (CORE_COURSE_GATE_CAPACITY,)),
        metadata={
            "priority": 89,
            "element_names": "gate_0_s,gate_1_s,gate_2_s",
            "external_control": "true",
            "unset_sentinel": str(PASS_TIME_UNSET_S),
        },
    ),
]


@dataclass
class RaceTelemetry(el.Archetype):
    """Referee-owned telemetry attached to the ``drone`` entity."""

    last_gate_passed: LastGatePassed = field(
        default_factory=lambda: jnp.array([LAST_GATE_UNSET], dtype=jnp.int64)
    )
    gate_pass_times: GatePassTimes = field(
        default_factory=lambda: jnp.full(
            CORE_COURSE_GATE_CAPACITY, PASS_TIME_UNSET_S, dtype=jnp.float64
        )
    )


@el.dataclass
class StaticSceneObject(el.Archetype):
    """A kinematic scene pose intentionally excluded from six-DOF integration."""

    world_pos: el.WorldPos


def spawn_course(world: el.World, course: Course) -> tuple[el.EntityId, ...]:
    """Spawn exactly four static bar entities for each selected gate."""

    entities = []
    for bar in course_bars(course):
        entities.append(
            world.spawn(
                StaticSceneObject(
                    el.WorldPos(
                        angular=el.Quaternion(jnp.array(bar.quaternion_xyzw)),
                        linear=jnp.array(bar.center),
                    )
                ),
                name=bar.name,
            )
        )
    return tuple(entities)
