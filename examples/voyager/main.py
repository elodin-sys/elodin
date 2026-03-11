import elodin as el
from jax import numpy as jnp
from jax.numpy import linalg as la
import spiceypy as spice
import numpy as np

#SIM_TIME_STEP = 1.0 / 120.0
#SIM_TIME_STEP = 3600.0
SIM_TIME_STEP = 86400.0
# Set the gravitational constant for Newton's law of universal gravitation
G = 6.6743e-11

spice.furnsh('spice/kernels.tm')

start_time_et = spice.utc2et('1977-09-05T13:58:37')

PLANETS = [
    {"spice_name": "MERCURY BARYCENTER", "entity_name": "mercury", "radius": 2000000000.0, "color": "bone", "mass": 3.3011e23},
    {"spice_name": "VENUS BARYCENTER", "entity_name": "venus", "radius": 3000000000.0, "color": "peach", "mass": 4.8675e24},
    {"spice_name": "EARTH", "entity_name": "earth", "radius": 6000000000.0, "color": "blue", "mass": 5.97219e24},
    {"spice_name": "MARS BARYCENTER", "entity_name": "mars", "radius": 4000000000.0, "color": "red", "mass": 6.4171e23},
    {"spice_name": "JUPITER BARYCENTER", "entity_name": "jupiter", "radius": 12000000000.0, "color": "yolk", "mass": 1.898125e27},
    {"spice_name": "SATURN BARYCENTER", "entity_name": "saturn", "radius": 10000000000.0, "color": "yellow", "mass": 5.6834e26},
    {"spice_name": "URANUS BARYCENTER", "entity_name": "uranus", "radius": 8000000000.0, "color": "mint", "mass": 8.6813e25},
    {"spice_name": "NEPTUNE BARYCENTER", "entity_name": "neptune", "radius": 8000000000.0, "color": "hyperblue", "mass": 1.02413e26},
]
PROBES = [
    {"spice_name": "VOYAGER 1", "entity_name": "voyager1", "radius": 4000000000.0, "color": "white", "mass": 825.0},
    {"spice_name": "VOYAGER 2", "entity_name": "voyager2", "radius": 4000000000.0, "color": "turquoise", "mass": 825.0},
]
TRUTH_PROBES = [
    {"spice_name": "VOYAGER 1", "entity_name": "voyager1_truth", "radius": 2500000000.0, "color": "bone", "mass": 825.0},
    {"spice_name": "VOYAGER 2", "entity_name": "voyager2_truth", "radius": 2500000000.0, "color": "mint", "mass": 825.0},
]
EPHEMERIS_BODIES = PLANETS
DISPLAY_BODIES = PLANETS + PROBES + TRUTH_PROBES
SUN_MASS = 1.9885e30


w = el.World()

sun = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel(linear=jnp.array([0.0, 0.0, 0.0])),
            inertia=el.Inertia(SUN_MASS),
        ),
    ],
    name="Sun",
)

body_entity_ids = {"Sun": sun}

for body in EPHEMERIS_BODIES + PROBES + TRUTH_PROBES:
    init_state, _ = spice.spkezr(body["spice_name"], start_time_et, 'ECLIPJ2000', 'NONE', 'SUN')

    init_pos = jnp.array(init_state[:3]) * 1000.0
    init_vel = jnp.array(init_state[3:]) * 1000.0

    print(body["spice_name"])
    print(init_pos)
    print(init_vel)

    body_entity_ids[body["entity_name"]] = w.spawn(
        [
            el.Body(
                world_pos=el.WorldPos(linear=init_pos),
                world_vel=el.WorldVel(linear=init_vel),
                inertia=el.Inertia(body["mass"]),
            ),
        ],
        name=body["entity_name"],
    )


def pre_step(tick: int, ctx: el.StepContext):
    current_time_et = start_time_et + tick * SIM_TIME_STEP

    for body in EPHEMERIS_BODIES + TRUTH_PROBES:
        state, _ = spice.spkezr(body["spice_name"], current_time_et, 'ECLIPJ2000', 'NONE', 'SUN')
        pos_m = np.asarray(state[:3], dtype=np.float64) * 1000.0
        vel_ms = np.asarray(state[3:], dtype=np.float64) * 1000.0

        ctx.write_component(
            f"{body['entity_name']}.world_pos",
            np.array([0.0, 0.0, 0.0, 1.0, pos_m[0], pos_m[1], pos_m[2]], dtype=np.float64),
        )
        ctx.write_component(
            f"{body['entity_name']}.world_vel",
            np.array([0.0, 0.0, 0.0, vel_ms[0], vel_ms[1], vel_ms[2]], dtype=np.float64),
        )


GravityEdge = el.Annotated[el.Edge, el.Component("gravity_edge", el.ComponentType.Edge)]


@el.dataclass
class GravityConstraint(el.Archetype):
    a: GravityEdge

    def __init__(self, a: el.EntityId, b: el.EntityId):
        self.a = GravityEdge(a, b)


@el.system
def gravity(
    graph: el.GraphQuery[GravityEdge],
    query: el.Query[el.WorldPos, el.Inertia],
) -> el.Query[el.Force]:
    def gravity_fn(force, a_pos, a_inertia, b_pos, b_inertia):
        r = a_pos.linear() - b_pos.linear()
        m = a_inertia.mass()
        M = b_inertia.mass()
        norm = la.norm(r)
        f = G * M * m * r / (norm * norm * norm)
        return el.Force(linear=force.force() - f)

    return graph.edge_fold(
        left_query=query,
        right_query=query,
        return_type=el.Force,
        init_value=el.Force(),
        fold_fn=gravity_fn,
    )


for probe in PROBES:
    probe_id = body_entity_ids[probe["entity_name"]]
    for source_name in ["Sun", *[planet["entity_name"] for planet in PLANETS]]:
        w.spawn(
            GravityConstraint(probe_id, body_entity_ids[source_name]),
            name=f"{probe['entity_name']} -> {source_name}",
        )

body_objects = "\n".join(
    f"""    object_3d {body["entity_name"]}.world_pos {{
        sphere radius={body["radius"]} emissivity=1.0 {{
            color {body["color"]}
        }}
    }}
    line_3d {body["entity_name"]}.world_pos line_width=1.0 perspective=#false {{
        color yolk
    }}"""
    for body in DISPLAY_BODIES
)

w.schematic("""
    hsplit {{
        tabs share=0.2 {{
            hierarchy
            schematic_tree
        }}
        tabs share=0.6 {{
            //viewport name=Viewport pos="(0,0,0,0,0,0,100)" look_at="(0,0,0,0,0,0,0)" hdr=#true
            viewport name=Viewport pos="(0,0,0,0, 0,0,2000000000000.0)" look_at="(0,0,0,0, 0,0,0)" fov=45.0 near=1000000.0

            graph "sun.world_pos" name=Graph
        }}
        tabs share=0.2 {{
            inspector
        }}
    }}
    object_3d sun.world_pos {{
        sphere radius=40000000000.0 emissivity=1.0 {{
            color yellow
        }}
    }}
{body_objects}
""".format(body_objects=body_objects))

sys = el.six_dof(sys=gravity)
sim = w.run(sys, SIM_TIME_STEP, run_time_step=1 / 120.0, pre_step=pre_step, max_ticks=10000)
