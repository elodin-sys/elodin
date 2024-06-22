import jax.numpy as np
import elodin as el

TIME_STEP = 1.0 / 60.0

j = np.array([15204079.70002, 14621352.61765, 6237758.3131]) * 1e-9

world = el.World()
sat = world.spawn(
    [
        el.Body(inertia=el.SpatialInertia(2825.2 / 1000.0, j)),
        world.glb(
            "https://storage.googleapis.com/elodin-marketing/models/oresat-low.glb"
        ),
        # Credit to the OreSat program https://www.oresat.org for the model above
    ],
    name="OreSat",
)

world.spawn(
    el.Panel.vsplit(
        [
            el.Panel.hsplit(
                [
                    el.Panel.viewport(
                        track_entity=sat,
                        track_rotation=False,
                        pos=[7.0, 0.0, 2.0],
                        looking_at=[0.0, 0.0, 0.0],
                        show_grid=True,
                    ),
                ]
            ),
            el.Panel.graph(
                [
                    el.GraphEntity(
                        sat,
                        [
                            el.Component.index(el.WorldPos)[:4],
                        ],
                    )
                ]
            ),
        ],
        active=True,
    )
)

world.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform.from_linear(np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion.from_angular(
                np.array([0.0, 0.0, 1.0]) * 7.2921159e-5
            ),
            inertia=el.SpatialInertia(1.0),
        ),
        world.glb("https://storage.googleapis.com/elodin-marketing/models/earth.glb"),
    ],
    name="Earth",
)


@el.map
def noop(pos: el.WorldPos) -> el.WorldPos:
    return pos


exec = world.run(
    system=el.six_dof(TIME_STEP, noop, el.Integrator.SemiImplicit),
    time_step=TIME_STEP,
)
