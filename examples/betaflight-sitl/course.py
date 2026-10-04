"""Pure course geometry for the Betaflight SITL racing example.

World coordinates are ENU. Gate yaw is a right-handed rotation about world +Z;
yaw zero gives a gate-local +X normal along world +X. The gate opening spans
local Y/Z and is approached from negative local X.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Vec3 = tuple[float, float, float]
SINGLE_GATE_INNER_SIZE_M = 2.5
GATE_BAR_THICKNESS_M = 0.2
SATURATED_ORANGE_RGB = (255, 128, 0)


@dataclass(frozen=True, slots=True)
class Gate:
    """One ordered, static square gate in ENU coordinates."""

    index: int
    center: Vec3
    yaw: float
    inner_size: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("gate index must be non-negative")
        if len(self.center) != 3 or not all(math.isfinite(value) for value in self.center):
            raise ValueError("gate center must contain three finite ENU coordinates")
        if not math.isfinite(self.yaw):
            raise ValueError("gate yaw must be finite")
        if not math.isfinite(self.inner_size) or self.inner_size <= 0.0:
            raise ValueError("gate inner size must be positive and finite")

    @property
    def normal(self) -> Vec3:
        """Gate-local +X (the required crossing direction), expressed in ENU."""

        return (math.cos(self.yaw), math.sin(self.yaw), 0.0)

    @property
    def lateral(self) -> Vec3:
        """Gate-local +Y, expressed in ENU."""

        return (-math.sin(self.yaw), math.cos(self.yaw), 0.0)

    def world_to_local(self, position: Sequence[float]) -> Vec3:
        """Transform an ENU point into this gate's local X/Y/Z coordinates."""

        if len(position) != 3:
            raise ValueError("position must contain exactly three coordinates")
        dx = float(position[0]) - self.center[0]
        dy = float(position[1]) - self.center[1]
        dz = float(position[2]) - self.center[2]
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        return (
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy,
            dz,
        )

    def local_to_world(self, position: Sequence[float]) -> Vec3:
        """Transform a gate-local point into ENU coordinates."""

        if len(position) != 3:
            raise ValueError("position must contain exactly three coordinates")
        local_x, local_y, local_z = (float(value) for value in position)
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        return (
            self.center[0] + cos_yaw * local_x - sin_yaw * local_y,
            self.center[1] + sin_yaw * local_x + cos_yaw * local_y,
            self.center[2] + local_z,
        )


@dataclass(frozen=True, slots=True)
class Course:
    """An immutable ordered course definition owned by the simulation/referee."""

    name: str
    gates: tuple[Gate, ...]

    def __post_init__(self) -> None:
        expected = tuple(range(len(self.gates)))
        actual = tuple(gate.index for gate in self.gates)
        if actual != expected:
            raise ValueError(f"gate indices must be contiguous and ordered; got {actual}")

    @property
    def inner_size(self) -> float | None:
        """Public opening rule, or ``None`` when no course is enabled."""

        if not self.gates:
            return None
        first = self.gates[0].inner_size
        if any(gate.inner_size != first for gate in self.gates[1:]):
            raise ValueError("a course must use one public inner-opening size")
        return first


@dataclass(frozen=True, slots=True)
class GateBar:
    """One frame bar, represented as a static oriented box."""

    name: str
    center: Vec3
    yaw: float
    size: Vec3

    @property
    def quaternion_xyzw(self) -> tuple[float, float, float, float]:
        half_yaw = self.yaw * 0.5
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def course_from_name(name: str) -> Course:
    """Resolve a supported Package C course name.

    ``c1_straight`` is a stable reserved configuration value, but its geometry
    intentionally does not land until Package F.
    """

    if name == "none":
        return Course(name="none", gates=())
    if name == "single":
        return Course(
            name="single",
            gates=(
                Gate(
                    index=0,
                    center=(10.0, 0.0, 1.8),
                    yaw=0.0,
                    inner_size=SINGLE_GATE_INNER_SIZE_M,
                ),
            ),
        )
    if name == "c1_straight":
        raise ValueError(
            "RACE_COURSE='c1_straight' is reserved for Package F and is not implemented"
        )
    raise ValueError(
        f"unknown RACE_COURSE={name!r}; expected 'none', 'single', or reserved 'c1_straight'"
    )


def course_from_env(env: Mapping[str, str] | None = None) -> Course:
    """Read and validate ``RACE_COURSE``, defaulting to ``none``."""

    values = os.environ if env is None else env
    return course_from_name(values.get("RACE_COURSE", "none"))


def gate_bars(gate: Gate, thickness: float = GATE_BAR_THICKNESS_M) -> tuple[GateBar, ...]:
    """Return four boxes whose clear local-Y/Z opening is exactly ``inner_size``.

    Box dimensions are in gate-local X/Y/Z. Horizontal bars overlap the side
    bars at the corners, avoiding visual seams without intruding into the inner
    opening.
    """

    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("gate bar thickness must be positive and finite")

    half_inner = gate.inner_size * 0.5
    half_bar = thickness * 0.5
    outer_span = gate.inner_size + 2.0 * thickness
    local_bars = (
        ("top", (0.0, 0.0, half_inner + half_bar), (thickness, outer_span, thickness)),
        ("bottom", (0.0, 0.0, -half_inner - half_bar), (thickness, outer_span, thickness)),
        ("left", (0.0, half_inner + half_bar, 0.0), (thickness, thickness, gate.inner_size)),
        ("right", (0.0, -half_inner - half_bar, 0.0), (thickness, thickness, gate.inner_size)),
    )
    return tuple(
        GateBar(
            name=f"gate_{gate.index}_{suffix}",
            center=gate.local_to_world(offset),
            yaw=gate.yaw,
            size=size,
        )
        for suffix, offset, size in local_bars
    )


def course_bars(course: Course) -> tuple[GateBar, ...]:
    """Return all static visual bars for a course."""

    return tuple(bar for gate in course.gates for bar in gate_bars(gate))


def gate_schematic(course: Course) -> str:
    """Build KDL objects for saturated-orange matte procedural gate bars.

    The editor's procedural boxes use a non-emissive, non-metallic standard
    material with default surface roughness, providing the required matte rather
    than glowing look. ``orientation=absolute`` aligns each box's local XYZ
    dimensions with ENU at yaw zero and then applies the entity's world-pose yaw.
    """

    red, green, blue = SATURATED_ORANGE_RGB
    objects = []
    for bar in course_bars(course):
        x_size, y_size, z_size = bar.size
        objects.append(
            f"""    object_3d {bar.name}.world_pos frame=ENU orientation=absolute {{
        box x={x_size:.6f} y={y_size:.6f} z={z_size:.6f} {{
            color {red} {green} {blue}
        }}
    }}"""
        )
    return "\n".join(objects)
