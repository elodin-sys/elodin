"""Rebuild examples/drone/rate-control-panel.kdl via elodin.ui (Phase 1 G1)."""

from __future__ import annotations

import elodin.ui as ui


def build() -> ui.Schematic:
    return ui.schematic(
        ui.hsplit(
            ui.vsplit(
                ui.graph("drone.rate_pid_state"),
                ui.component_monitor(component_name="drone.rate_pid_state"),
            ),
            ui.vsplit(
                ui.graph(
                    "drone.gyro, drone.ang_vel_setpoint",
                    name="Drone: rate_control",
                ),
            ),
            name="Rate Control Panel",
        ),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
