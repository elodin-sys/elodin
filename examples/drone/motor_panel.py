"""Rebuild examples/drone/motor-panel.kdl via elodin.ui (Phase 1 G1)."""

from __future__ import annotations

import elodin.ui as ui


def build() -> ui.Schematic:
    return ui.schematic(
        ui.tabs(
            ui.hsplit(
                ui.vsplit(
                    ui.graph("drone.motor_input"),
                    ui.graph("drone.motor_pwm"),
                    ui.graph("drone.motor_rpm"),
                    share=0.4,
                ),
                ui.graph("drone.thrust"),
                name="Motor Panel",
            ),
        ),
    )


if __name__ == "__main__":
    print(build().emit_kdl())
