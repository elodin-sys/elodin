"""Cesium viewport smoke test: a position-controlled vehicle over a real site.

The scene is streamed by Cesium Native in the Editor. The controller below
applies forces through Elodin's integrator; there are no terrain collisions.
"""

import os
import typing as ty

import elodin as el
import elodin.ui as ui
import jax
import jax.numpy as jnp


# Both sites are public 3D Tiles sources, selected in the Editor's environment.
AGI = os.environ.get("CESIUM_DEMO", "agi") == "agi"
LAT, LON, ALT = (
    (40.03879587543084, -75.59670696255716, 297.25783960334957) if AGI else (46.948, 7.4474, 550.0)
)
FLIGHT_HEIGHT = 70.0 if AGI else 100.0
RADIUS_X, RADIUS_Y = (100.0, 80.0) if AGI else (180.0, 150.0)

FlightPose = ty.Annotated[
    jax.Array,
    el.Component("flight_pose", el.ComponentType(el.PrimitiveType.F64, (7,))),
]


@el.dataclass
class DisplayPose(el.Archetype):
    flight_pose: FlightPose


@el.map
def display_pose(pos: el.WorldPos, vel: el.WorldVel) -> FlightPose:
    # Display heading follows measured motion; this is not an attitude controller.
    heading = jnp.arctan2(vel.linear()[1], vel.linear()[0])
    return jnp.concatenate(
        [jnp.array([0.0, 0.0, jnp.sin(heading / 2), jnp.cos(heading / 2)]), pos.linear()]
    )


@el.system
def control(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    vehicle: el.Query[el.WorldPos, el.WorldVel, el.Inertia],
) -> el.Query[el.Force]:
    time = tick[0] * dt[0]
    phase = 2.0 * jnp.pi * time / 120.0
    target = jnp.array(
        [
            RADIUS_X * jnp.sin(phase),
            -RADIUS_Y * jnp.cos(phase),
            FLIGHT_HEIGHT + 10.0 * jnp.sin(phase),
        ]
    )

    def force(pos, vel, inertia):
        acceleration = 1.5 * (target - pos.linear()) - 2.5 * vel.linear()
        return el.SpatialForce(linear=inertia.mass() * acceleration)

    return vehicle.map(el.Force, force)


def world():
    w = el.World()
    w.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(linear=jnp.array([0.0, -RADIUS_Y, FLIGHT_HEIGHT])),
                inertia=el.SpatialInertia(mass=1.0),
            ),
            DisplayPose(flight_pose=jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, -RADIUS_Y, FLIGHT_HEIGHT])),
        ],
        name="drone",
    )
    w.schematic(
        ui.schematic(
            ui.vsplit(
                ui.hsplit(
                    ui.viewport(
                        name="Chase",
                        active=True,
                        pos="drone.flight_pose.translate(-2.8,0,2.2)",
                        look_at="drone.world_pos",
                        up="(0,0,1)",
                        frame="ENU",
                        show_grid=False,
                        show_view_cube=True,
                        view_cube_frame="ENU",
                        fov=45.0,
                    ),
                    ui.viewport(
                        name="Onboard / Ground",
                        pos="drone.world_pos.translate_world(0,0,-0.8)",
                        look_at="drone.world_pos.translate_world(0,0,-50)",
                        up="(0,1,0)",
                        frame="ENU",
                        near=0.1,
                        show_grid=False,
                        show_view_cube=False,
                        fov=65.0,
                    ),
                    ui.viewport(
                        name="Follow",
                        pos="drone.flight_pose.translate(-3.2,-2.4,3.5)",
                        look_at="drone.world_pos",
                        up="(0,0,1)",
                        frame="ENU",
                        show_grid=False,
                        show_view_cube=False,
                        fov=55.0,
                    ),
                    share=0.65,
                ),
                ui.hsplit(
                    ui.viewport(
                        name="Site map / North up",
                        pos="(0,0,0,1, 0,0,320)",
                        look_at="(0,0,0,1, 0,0,0)",
                        up="(0,1,0)",
                        frame="ENU",
                        fov=55.0,
                        show_grid=False,
                        show_view_cube=False,
                        effects=False,
                        share=0.25,
                    ),
                    ui.geo_position_gauge(
                        "drone.world_pos",
                        source="ENU",
                        display="LLA",
                        name="Position / WGS84",
                        share=0.25,
                    ),
                    ui.graph(
                        "drone.world_pos[6]",
                        name="Height above origin (m)",
                        colors=["orange"],
                        share=0.25,
                    ),
                    ui.query_plot(
                        "Ground track",
                        # Five samples per second keep the complete five-minute
                        # flight in one query result batch (the plot reads one batch).
                        "SELECT drone_world_pos[5], drone_world_pos[6] FROM "
                        "(SELECT time, drone_world_pos, row_number() OVER (ORDER BY time) AS sample "
                        "FROM drone_world_pos) WHERE sample % 12 = 0 ORDER BY time",
                        query_type="sql",
                        plot_mode="xy",
                        x_label="East (m)",
                        y_label="North (m)",
                        color="orange",
                        refresh_interval_ms=1000.0,
                        share=0.25,
                    ),
                    share=0.35,
                ),
            ),
            ui.object_3d(
                "drone.flight_pose",
                mesh=ui.glb("talon-quad-v2.glb"),
                frame="ENU",
                icon=ui.icon(
                    builtin="flight",
                    color="orange",
                    size=18.0,
                    visibility=ui.visibility_range(min=20.0, max=10000.0, fade_distance=5.0),
                ),
            ),
            ui.line_3d(
                "drone.world_pos",
                line_width=2.0,
                color=ui.color(255, 190, 70, 255),
                perspective=False,
                frame="ENU",
            ),
            timeline=ui.timeline(follow_latest=True),
            coordinate=ui.coordinate(frame="ENU", lat=LAT, lon=LON, alt=ALT),
            environment=ui.environment(
                sun=ui.sun(azimuth=135.0, elevation=45.0, illuminance=12000.0, shadows=True),
                ambient=0.3,
            ),
        ),
        "cesium.kdl",
    )
    return w


if __name__ == "__main__":
    world().run(
        el.six_dof(sys=control) | display_pose,
        simulation_rate=60.0,
        generate_real_time=True,
        max_ticks=18000,
    )
