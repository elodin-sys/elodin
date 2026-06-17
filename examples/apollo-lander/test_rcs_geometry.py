from __future__ import annotations

import re
import unittest
from pathlib import Path

from rcs_geometry import (
    RCS_THRUSTER_AXIS,
    RCS_THRUSTER_SIGN,
    RCS_THRUSTER_VIZ_MIN_RAW_LEVEL,
    rcs_thruster_levels,
)


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _parse_vec3(raw: str) -> tuple[float, float, float]:
    parts = tuple(float(part.strip()) for part in raw.split(","))
    if len(parts) != 3:
        raise ValueError(f"expected vec3, got {raw!r}")
    return parts


def _sign(value: float) -> float:
    if value > 1e-9:
        return 1.0
    if value < -1e-9:
        return -1.0
    return 0.0


def _kdl_rcs_nozzles() -> list[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
]:
    kdl = Path(__file__).with_name("apollo-lander.kdl").read_text()
    matches = re.finditer(
        r'name="rcs_(\d+)"[^\n]*position="\(([^)]*)\)" direction="\(([^)]*)\)"',
        kdl,
    )

    nozzles: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []
    for match in matches:
        index = int(match.group(1))
        position = _parse_vec3(match.group(2))
        exhaust = _parse_vec3(match.group(3))
        reaction_force = tuple(-component for component in exhaust)
        torque = _cross(position, reaction_force)
        if index != len(nozzles):
            raise AssertionError(f"expected rcs_{len(nozzles)}, got rcs_{index}")
        nozzles.append((position, exhaust, torque))
    return nozzles


def _kdl_rcs_torques() -> list[tuple[float, float, float]]:
    return [torque for _, _, torque in _kdl_rcs_nozzles()]


class RcsGeometryTests(unittest.TestCase):
    def test_kdl_rcs_emitters_stay_on_visible_lander_nozzle_band(self) -> None:
        # The GLB is translated by (0, -2.5, 0). Its visible RCS nozzle band
        # lands near +/-1.52 m in X/Z and around 0.5-1.2 m in Y in object space.
        for index, (position, _, _) in enumerate(_kdl_rcs_nozzles()):
            x, y, z = position
            self.assertLessEqual(
                abs(x),
                1.52,
                f"rcs_{index} should not float outside the visual LM X radius",
            )
            self.assertLessEqual(
                abs(z),
                1.52,
                f"rcs_{index} should not float outside the visual LM Z radius",
            )
            self.assertGreaterEqual(
                y,
                0.50,
                f"rcs_{index} should stay in the visible LM RCS nozzle band",
            )
            self.assertLessEqual(
                y,
                1.25,
                f"rcs_{index} should stay in the visible LM RCS nozzle band",
            )

    def test_kdl_rcs_emitters_form_four_lm_quads(self) -> None:
        clusters: dict[tuple[float, float], list[tuple[float, float, float]]] = {}
        for position, exhaust, _ in _kdl_rcs_nozzles():
            x, _, z = position
            clusters.setdefault((_sign(x), _sign(z)), []).append(exhaust)

        self.assertEqual(
            set(clusters),
            {(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)},
        )
        for cluster, exhausts in clusters.items():
            self.assertEqual(len(exhausts), 4, f"quad {cluster} should have 4 nozzles")
            axis_counts = [0, 0, 0]
            y_signs = set()
            for exhaust in exhausts:
                axis = max(range(3), key=lambda item: abs(exhaust[item]))
                axis_counts[axis] += 1
                if axis == 1:
                    y_signs.add(_sign(exhaust[1]))

            self.assertEqual(
                axis_counts,
                [1, 2, 1],
                f"quad {cluster} should be a 4-nozzle star",
            )
            self.assertEqual(
                y_signs,
                {-1.0, 1.0},
                f"quad {cluster} should have opposed Y nozzles",
            )

    def test_thruster_mapping_matches_kdl_nozzle_geometry(self) -> None:
        torques = _kdl_rcs_torques()

        self.assertEqual(len(torques), 16)
        for index, torque in enumerate(torques):
            dominant_axis = max(range(3), key=lambda axis: abs(torque[axis]))
            dominant_sign = _sign(torque[dominant_axis])

            self.assertEqual(
                RCS_THRUSTER_AXIS[index],
                dominant_axis,
                f"rcs_{index} axis should match KDL position x reaction_force",
            )
            self.assertEqual(
                RCS_THRUSTER_SIGN[index],
                dominant_sign,
                f"rcs_{index} sign should match KDL position x reaction_force",
            )

    def test_unit_body_torques_activate_matching_nozzles(self) -> None:
        torques = _kdl_rcs_torques()

        for axis in range(3):
            for sign in (-1.0, 1.0):
                command = [0.0, 0.0, 0.0]
                command[axis] = sign
                active = rcs_thruster_levels(tuple(command))

                self.assertTrue(any(level > 0.0 for level in active))
                for index, level in enumerate(active):
                    if level <= 0.0:
                        continue
                    torque = torques[index]
                    dominant_axis = max(
                        range(3), key=lambda item: abs(torque[item])
                    )
                    self.assertEqual(dominant_axis, axis)
                    self.assertEqual(_sign(torque[dominant_axis]), sign)

    def test_visual_levels_boost_small_real_torque_commands(self) -> None:
        small_command = (0.0, 0.04, 0.0)
        active = rcs_thruster_levels(small_command)

        self.assertGreater(max(active), small_command[1])
        self.assertAlmostEqual(max(active), small_command[1] ** 0.5)

    def test_visual_levels_keep_tiny_torque_commands_off(self) -> None:
        tiny_command = (0.0, RCS_THRUSTER_VIZ_MIN_RAW_LEVEL * 0.5, 0.0)
        self.assertTrue(all(level == 0.0 for level in rcs_thruster_levels(tiny_command)))


if __name__ == "__main__":
    unittest.main()
