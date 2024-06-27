import jax.numpy as np
import elodin as el
import typing as ty
import os
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
MagPostCal = ty.Annotated[
    jax.Array,
    el.Component(
        "mag_postcal_value",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
    ),
]
Gyro = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro_omega",
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
RWSpeed = ty.Annotated[
    jax.Array,
    el.Component(
        "rw_speed",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
    ),
]

RWSpeedSetpoint = ty.Annotated[
    jax.Array,
    el.Component(
        "rw_speed_setpoint",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
    ),
]


@dataclass
class Determination(el.Archetype):
    mag_ref: MagRef = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    sun_ref: SunRef = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    mag_value: MagValue = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    mag_postcal: MagPostCal = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    gyro: Gyro = field(default_factory=lambda: np.zeros(3))
    sun_pos: SunPos = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    sun_sensors: SunSensors = field(default_factory=lambda: np.zeros(6))
    rw_speed: RWSpeed = field(default_factory=lambda: np.zeros(3))

@dataclass
class Control(el.Archetype):
    rw_speed_setpoint: RWSpeedSetpoint = field(default_factory=lambda: np.zeros(3))


world = el.World()
sat = world.spawn(
    [
        el.Body(inertia=el.SpatialInertia(2825.2 / 1000.0, j)),
        world.glb(os.path.abspath("./clients/care-weather/veery.glb")),
        # Credit to the OreSat program https://www.oresat.org for the model above
        Determination(),
        Control(),
    ],
    name="OreSat",
)

world.spawn(
    el.Panel.hsplit(
        [
            el.Panel.viewport(
                track_entity=sat,
                track_rotation=False,
                pos=[0.5, -0.05, 0.1],
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
                                    # el.Component.index(MagValue),
                                    el.Component.index(MagPostCal),
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
                    el.Panel.graph(
                        [
                            el.GraphEntity(
                                sat,
                                [
                                    el.Component.index(Gyro),
                                ],
                            )
                        ]
                    ),
                ]
            ),
        ]
    ),
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
                                    el.Component.index(RWSpeed),
                                ],
                            )
                        ]
                    ),
                    el.Panel.graph(
                        [
                            el.GraphEntity(
                                sat,
                                [
                                    el.Component.index(RWSpeedSetpoint),
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
    mag_postcal: MagPostCal,
    sun_pos: SunPos,
    gyro: Gyro,
    sun_sensor: SunSensors,
    rw_speed: RWSpeed,
    rw_speed_setpoint: RWSpeedSetpoint,
) -> tuple[el.WorldPos, MagRef, SunRef, MagValue, MagPostCal, SunPos, Gyro, SunSensors, RWSpeed, RWSpeedSetpoint]:
    return pos, mag_ref, sun_ref, mag_value, mag_postcal, sun_pos, gyro, sun_sensor, rw_speed, rw_speed_setpoint


exec = world.run(
    system=noop,
    time_step=TIME_STEP,
)
