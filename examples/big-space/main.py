import elodin as el
from jax import numpy as jnp

SIM_TIME_STEP = 1.0 / 120.0

DISTANCE = 2_000_000.0
SPHERE_RADIUS = 0.2
CAMERA_Y = -10.0
A_X = DISTANCE / 2
B_X = -DISTANCE / 2

w = el.World()

a = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([A_X, 0.0, 0.0])),
            world_vel=el.WorldVel(linear=jnp.array([1.0, 0.0, 0.0])),
            inertia=el.Inertia(1.0),
        ),
    ],
    name="A",
)
b = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([B_X, 0.0, 0.0])),
            world_vel=el.WorldVel(linear=jnp.array([-1.0, 0.0, 0.0])),
            inertia=el.Inertia(1.0),
        ),
    ],
    name="B",
)

w.schematic(f"""
    coordinate frame=ECEF // This is not exactly correct. HCI would be the right system for this perhaps.
    hsplit {{
        tabs share=0.5 {{
            viewport name=A pos="(0,0,0,1, {A_X},{CAMERA_Y},0)" look_at="(0,0,0,1, {A_X},0,0)" near=0.01 far=100.0 hdr=#true
            graph "a.world_pos" name=Graph
        }}
        tabs share=0.5 {{
            viewport name=B pos="(0,0,0,1, {B_X},{CAMERA_Y},0)" look_at="(0,0,0,1, {B_X},0,0)" near=0.01 far=100.0 hdr=#true
            graph "b.world_pos" name=Graph
        }}
    }}
    object_3d a.world_pos {{
        sphere radius={SPHERE_RADIUS} emissivity=1.0 {{
            color cyan
        }}
    }}
    object_3d b.world_pos {{
        sphere radius={SPHERE_RADIUS} emissivity=1.0 {{
            color pink
        }}
    }}
""")

sys = el.six_dof()
sim = w.run(
    sys,
    simulation_rate=1.0 / SIM_TIME_STEP,
    generate_real_time=True,
)
