from elodin_py import SimBuilder, RigidBody, Mesh, Material, Joint, Effector, ComponentRef, ComponentType, editor
import jax.numpy as np

def sim() -> SimBuilder:
    builder = SimBuilder()
    builder.zero_g()

    a = builder.body(
        RigidBody(
            mass=1.0 / 6.649e-11,
            mesh=Mesh.sphere(0.2),
            material=Material.hex_color("#FFB800").emissive(20.0, 188.0 / 255.0 * 20.0, 0.0),
            joint=Joint.free(
                pos = np.array([0.8822391241, 0, 0]),
                vel = np.array([0, 1.0042424155, 0])
            ),
            trace_anchor = np.array([0,0,0])
        )
    )

    b = builder.body(
        RigidBody(
            mass=1.0 / 6.649e-11,
            mesh=Mesh.sphere(0.2),
            material=Material.hex_color("#FFB800").emissive(65.0 / 255.0 * 20, 187 / 255.0 * 20, 20.0),
            joint=Joint.free(
                pos = np.array([-0.6432718586,0, 0]),
                vel = np.array([0, -1.6491842814, 0])
            ),
            trace_anchor = np.array([0,0,0])
        )
    )

    c = builder.body(
        RigidBody(
            mass=1.0 / 6.649e-11,
            mesh=Mesh.sphere(0.2),
            material=Material.hex_color("#FFB800").emissive(20., 15.0 / 255.0 * 20., 0.0),
            joint=Joint.free(
                pos = np.array([-0.2389672654, 0, 0]),
                vel = np.array([0,0.6449418659, 0.0])
            ),
            trace_anchor = np.array([0,0,0])
        )
    )

    builder.gravity(a, b)
    builder.gravity(a, c)
    builder.gravity(b, c)

    return builder
