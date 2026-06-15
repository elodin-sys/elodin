#!/usr/bin/env uv run

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import elodin as el
import numpy as np

from sim import DEFAULT_MAX_TICKS, PARAMS, SIMULATION_RATE_HZ, build

STATE_PORT_ENV = "ELODIN_MONTE_CARLO_STATE_PORT"
COMMAND_PORT_ENV = "ELODIN_MONTE_CARLO_COMMAND_PORT"
DEFAULT_STATE_PORT = 9003
DEFAULT_COMMAND_PORT = 9002
CONTROLLER_ENV = "ELODIN_MONTE_CARLO_CONTROLLER"


class SitlBridge:
    def __init__(self) -> None:
        self.state_port = int(os.environ.get(STATE_PORT_ENV, str(DEFAULT_STATE_PORT)))
        self.command_port = int(os.environ.get(COMMAND_PORT_ENV, str(DEFAULT_COMMAND_PORT)))
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_sock.bind(("127.0.0.1", self.command_port))
        self.command_sock.settimeout(0.02)
        self.last_command = 0.0

    def step(self, position: float, velocity: float, target: float, tick: int) -> float:
        payload = json.dumps(
            {
                "tick": tick,
                "position": position,
                "velocity": velocity,
                "target": target,
            }
        ).encode()
        self.state_sock.sendto(payload, ("127.0.0.1", self.state_port))
        try:
            raw, _ = self.command_sock.recvfrom(1024)
            self.last_command = float(json.loads(raw.decode())["command"])
        except TimeoutError:
            pass
        return self.last_command


params = el.monte_carlo.params(PARAMS)
world, system = build(params)
world.schematic(
    """
    viewport name="Monte Carlo" active=#true hdr=#true show_grid=#true pos="(0,0,0,1, vehicle.position[0] - 8, -10, 5)" look_at="(0,0,0,1, vehicle.position[0], 0, 0)" {
        bloom preset="old_school" intensity=0.35 threshold=0.55 threshold_softness=0.2
    }
    object_3d "(0,0,0,1, vehicle.position[0], 0, 0)" {
        sphere radius=0.25 { color 80 170 255 }
        thruster name="Forward specific force" body_frame=#true position="(-0.35, 0, 0)" direction="(-1, 0, 0)" intensity="vehicle.specific_force[0] / 20.0" emission_rate=420.0
        thruster name="Reverse specific force" body_frame=#true position="(0.35, 0, 0)" direction="(1, 0, 0)" intensity="vehicle.specific_force[0] / -20.0" emission_rate=420.0
    }
    graph "vehicle.specific_force" name="Specific force"
    graph "vehicle.position" name="Position"
    """,
    "monte-carlo.kdl",
)
bridge: SitlBridge | None = None
use_controller = os.environ.get(CONTROLLER_ENV, "1") != "0"

if use_controller:
    controller = el.s10.PyRecipe.process(
        name="Minimal SITL Controller",
        cmd=sys.executable,
        args=[str(Path(__file__).with_name("controller.py"))],
        cwd=str(Path(__file__).parent),
    )
    world.recipe(controller)


def post_step(tick: int, ctx: el.StepContext) -> None:
    global bridge
    if use_controller and bridge is None:
        bridge = SitlBridge()
    position = float(ctx.read_component("vehicle.position")[0])
    velocity = float(ctx.read_component("vehicle.velocity")[0])
    target = float(ctx.read_component("vehicle.target")[0])
    if bridge is not None:
        command = bridge.step(position, velocity, target, tick)
    else:
        command = max(min((target - position) * 1.2 - velocity * 0.35, 20.0), -20.0)
    ctx.write_component("vehicle.command", np.array([command], dtype=np.float64))
    if tick >= DEFAULT_MAX_TICKS - 1:
        el.monte_carlo.result(
            final_position=position,
            target=target,
            error=abs(target - position),
        )


world.run(
    system,
    simulation_rate=SIMULATION_RATE_HZ,
    max_ticks=DEFAULT_MAX_TICKS,
    db_path=params.db_path or os.environ.get("ELODIN_DB_PATH"),
    post_step=post_step,
    interactive=False,
)
