from __future__ import annotations

import re
import unittest
from pathlib import Path

from rcs_geometry import RCS_THRUSTER_AXIS, RCS_THRUSTER_SIGN, rcs_thruster_levels


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


def _kdl_rcs_torques() -> list[tuple[float, float, float]]:
    kdl = Path(__file__).with_name("apollo-lander.kdl").read_text()
    matches = re.finditer(
        r'name="rcs_(\d+)"[^\n]*position="\(([^)]*)\)" direction="\(([^)]*)\)"',
        kdl,
    )

    torques: list[tuple[float, float, float]] = []
    for match in matches:
        index = int(match.group(1))
        position = _parse_vec3(match.group(2))
        exhaust = _parse_vec3(match.group(3))
        reaction_force = tuple(-component for component in exhaust)
        torque = _cross(position, reaction_force)
        if index != len(torques):
            raise AssertionError(f"expected rcs_{len(torques)}, got rcs_{index}")
        torques.append(torque)
    return torques


class RcsGeometryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
