import jax.numpy as np
import elodin as el
import typing as ty
import jax
from dataclasses import dataclass, field

TIME_STEP = 1.0 / 60.0

j = np.array([15204079.70002, 14621352.61765, 6237758.3131]) * 1e-9

MagRef = ty.Annotated[
    jax.Array,
    el.Component(
        "mag_ref",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
    ),
]
SunRef = ty.Annotated[
    jax.Array,
    el.Component(
        "sun_ref",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
    ),
]

MagValue = ty.Annotated[
    jax.Array,
    el.Component(
        "mag_value",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
    ),
]
SunPos = ty.Annotated[
    jax.Array,
    el.Component(
        "sun_vec_b",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
    ),
]
SunSensors = ty.Annotated[
    jax.Array,
    el.Component(
        "css_value",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
    ),
]


@dataclass
class Determination(el.Archetype):
    mag_ref: MagRef = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    sun_ref: SunRef = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    mag_value: MagValue = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    sun_pos: SunPos = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    sun_sensors: SunSensors = field(default_factory=lambda: np.zeros(6))


world = el.World()
sat = world.spawn(
    [
        el.Body(inertia=el.SpatialInertia(2825.2 / 1000.0, j)),
        world.glb(
            "https://storage.googleapis.com/elodin-marketing/models/oresat-low.glb"
        ),
        Determination(),
        # Credit to the OreSat program https://www.oresat.org for the model above
    ],
    name="OreSat",
)

world.spawn(
    el.Panel.hsplit(
        [
            el.Panel.viewport(
                track_entity=sat,
                track_rotation=False,
                pos=[7.0, 0.0, 2.0],
                looking_at=[0.0, 0.0, 0.0],
                show_grid=True,
            ),
            el.Panel.vsplit(
                [
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
                    el.Panel.graph(
                        [
                            el.GraphEntity(
                                sat,
                                [
                                    el.Component.index(SunPos),
                                ],
                            )
                        ]
                    ),
                    el.Panel.graph(
                        [
                            el.GraphEntity(
                                sat,
                                [
                                    el.Component.index(MagValue),
                                ],
                            )
                        ]
                    ),
                ]
            ),
        ],
        active=True,
    )
)
world.spawn(
    el.Panel.hsplit(
        [
            el.Panel.vsplit(
                [
                    el.Panel.graph(
                        [
                            el.GraphEntity(
                                sat,
                                [
                                    el.Component.index(SunRef),
                                ],
                            )
                        ]
                    ),
                    el.Panel.graph(
                        [
                            el.GraphEntity(
                                sat,
                                [
                                    el.Component.index(MagRef),
                                ],
                            )
                        ]
                    ),
                ]
            ),
            el.Panel.vsplit(
                [
                    el.Panel.graph(
                        [
                            el.GraphEntity(
                                sat,
                                [
                                    el.Component.index(SunSensors),
                                ],
                            )
                        ]
                    ),
                ]
            ),
        ]
    ),
)


@el.map
def noop(
    pos: el.WorldPos,
    mag_ref: MagRef,
    sun_ref: SunRef,
    mag_value: MagValue,
    sun_pos: SunPos,
) -> tuple[el.WorldPos, MagRef, SunRef, MagValue, SunPos]:
    return pos, mag_ref, sun_ref, mag_value, sun_pos


exec = world.run(
    system=el.six_dof(TIME_STEP, noop, el.Integrator.SemiImplicit),
    time_step=TIME_STEP,
)
