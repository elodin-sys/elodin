from paracosm_py import SimBuilder, RigidBody, Mesh, Material, Joint, editor
import numpy as np


def sim() -> SimBuilder:
    builder = SimBuilder()
    root = builder.body(
        RigidBody(
            mass=1.0,
            mesh=Mesh.box(0.05, 0.05, 0.05),
            material=Material.hex_color("#FFF"),
            joint=Joint.fixed(),
        )
    )
    builder.body(
        RigidBody(
            mass=1.0,
            mesh=Mesh.box(0.2, 1.0, 0.2),
            material=Material.hex_color("#FFF"),
            joint=Joint.revolute(np.array([0.0, 0.0, 1.0]), pos=3.14 / 2.0),
            parent=root,
            body_pos=np.array([0.0, -0.5, 0.0]),
        )
    )

    return builder


editor(sim)
