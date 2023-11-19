from paracosm_py import SimBuilder, RigidBody, Mesh, Material, Joint
import numpy as np


def to_rad(deg):
    return deg * 3.14 / 180

def sim() -> SimBuilder:
    builder = SimBuilder()
    # builder.zero_g()
    root = builder.body(
        RigidBody(
            mass=10.0,
            mesh=Mesh.box(0.05, 0.05, 0.5),
            material=Material.hex_color("#FFF"),
            joint=Joint.fixed(),
            body_pos=np.array([0.0, 1.0, 0.0])
        )
    )
    length = 1.2;
    rod_a = builder.body(
        RigidBody(
            mass=1.0,
            mesh=Mesh.box(0.2, length, 0.2),
            material=Material.hex_color("#000"),
            joint=Joint.revolute(np.array([0.0, 0.0, length / 2]), pos=to_rad(-95)),
            parent=root,
            body_pos=np.array([0.0, -1 * length / 2.0, 0.0]),
        )
    )

    builder.body(
        RigidBody(
            mass=1.0,
            mesh=Mesh.box(0.2, length, 0.2),
            material=Material.hex_color("#F00"),
            joint=Joint.revolute(
                np.array([0.0, 0.0, length / 2]),
                anchor = np.array([0.0, -1 * length / 2.0, 0.0]),
                pos=to_rad(-0)
            ),
            parent=rod_a,
            body_pos=np.array([-0.0, -1 * length / 2.0, -0.0]),
        )
    )

    return builder
