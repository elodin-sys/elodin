from dataclasses import dataclass
from typing import Annotated

import elodin as el
import jax

TIME_STEP = 1.0 / 30.0


MagRef = Annotated[jax.Array, el.Component("mag_ref", el.ComponentType(el.PrimitiveType.F64, (3,)))]

MagReading = Annotated[
    jax.Array, el.Component("chamber_mag_reading", el.ComponentType(el.PrimitiveType.F64, (3,)))
]


@dataclass
class Mag(el.Archetype):
    mag_ref: MagRef
    chamber_mag_reading: MagReading


w = el.World()

w.spawn(Mag(jax.numpy.array([200.0, 0.0, 0.0]), jax.numpy.array([0.0, 0.0, 0.0])), name="chamber")


@el.map
def nop(ref: MagRef, mag_reading: MagReading) -> MagRef:
    return ref


sim = w.run(nop, TIME_STEP)
