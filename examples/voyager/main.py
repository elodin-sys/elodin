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
    {"spice_name": "MERCURY BARYCENTER", "entity_name": "mercury", "radius": 2000000000.0, "color": "bone"},
    {"spice_name": "VENUS BARYCENTER", "entity_name": "venus", "radius": 3000000000.0, "color": "peach"},
    {"spice_name": "EARTH", "entity_name": "earth", "radius": 6000000000.0, "color": "blue"},
    {"spice_name": "MARS BARYCENTER", "entity_name": "mars", "radius": 4000000000.0, "color": "red"},
    {"spice_name": "JUPITER BARYCENTER", "entity_name": "jupiter", "radius": 12000000000.0, "color": "yolk"},
    {"spice_name": "SATURN BARYCENTER", "entity_name": "saturn", "radius": 10000000000.0, "color": "yellow"},
    {"spice_name": "URANUS BARYCENTER", "entity_name": "uranus", "radius": 8000000000.0, "color": "mint"},
    {"spice_name": "NEPTUNE BARYCENTER", "entity_name": "neptune", "radius": 8000000000.0, "color": "hyperblue"},
]


w = el.World()

sun = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel(linear=jnp.array([0.0, 0.0, 0.0])),
            inertia=el.Inertia(1.0 / G),
        ),
    ],
    name="Sun",
)


for planet in PLANETS:
    init_state, _ = spice.spkezr(planet["spice_name"], start_time_et, 'ECLIPJ2000', 'NONE', 'SUN')

    init_pos = jnp.array(init_state[:3]) * 1000.0
    init_vel = jnp.array(init_state[3:]) * 1000.0

    print(planet["spice_name"])
    print(init_pos)
    print(init_vel)

    planet = w.spawn(
        [
            el.Body(
                world_pos=el.WorldPos(linear=init_pos),
                world_vel=el.WorldVel(linear=init_vel),
                inertia=el.Inertia(1.0 / G),
            ),
        ],
        name=planet["entity_name"],
    )


def pre_step(tick: int, ctx: el.StepContext):
    current_time_et = start_time_et + tick * SIM_TIME_STEP

    for planet in PLANETS:
        state, _ = spice.spkezr(planet["spice_name"], current_time_et, 'ECLIPJ2000', 'NONE', 'SUN')
        pos_m = np.asarray(state[:3], dtype=np.float64) * 1000.0
        vel_ms = np.asarray(state[3:], dtype=np.float64) * 1000.0

        ctx.write_component(
            f"{planet['entity_name']}.world_pos",
            np.array([0.0, 0.0, 0.0, 1.0, pos_m[0], pos_m[1], pos_m[2]], dtype=np.float64),
        )
        ctx.write_component(
            f"{planet['entity_name']}.world_vel",
            np.array([0.0, 0.0, 0.0, vel_ms[0], vel_ms[1], vel_ms[2]], dtype=np.float64),
        )

#c = w.spawn(
#    [
#        el.Body(
#            world_pos=el.WorldPos(linear=jnp.array([-0.2291782474, 0, 0])),
#            world_vel=el.WorldVel(linear=jnp.array([0, 0.6233673964, 0.0])),
#            inertia=el.Inertia(1.0 / G),
#        ),
#    ],
#    name="C",
#)
#
## Define a new "gravity edge" component type
#GravityEdge = el.Annotated[el.Edge, el.Component("gravity_edge", el.ComponentType.Edge)]
#
#
## Define a new "gravity constraint" archetype using the gravity edge component
#@el.dataclass
#class GravityConstraint(el.Archetype):
#    a: GravityEdge
#
#    def __init__(self, a: el.EntityId, b: el.EntityId):
#        self.a = GravityEdge(a, b)
#
#
## Define a new system to apply gravity by iterating over all gravity edge components
#@el.system
#def gravity(
#    graph: el.GraphQuery[GravityEdge],
#    query: el.Query[el.WorldPos, el.Inertia],
#) -> el.Query[el.Force]:
#    # Create a fold function to take an accumulator and the query results for the
#    # left and right entities, and apply Newton's law of universal gravitation:
#    def gravity_fn(force, a_pos, a_inertia, b_pos, b_inertia):
#        r = a_pos.linear() - b_pos.linear()
#        m = a_inertia.mass()
#        M = b_inertia.mass()
#        norm = la.norm(r)
#        f = G * M * m * r / (norm * norm * norm)
#        # returns the updated force value applied to the left entity this tick
#        return el.Force(linear=force.force() - f)
#
#    return graph.edge_fold(
#        left_query=query,  # i.e. fetches WorldPos and Inertia components from A
#        right_query=query,  # reusing the same query for B
#        return_type=el.Force,  # matching the system query return type
#        init_value=el.Force(),  # initial value for the fold function accumulator
#        fold_fn=gravity_fn,  # the fold function to apply to each edge, that you defined above
#    )


## Add the gravity constraint entities to the world
#w.spawn(GravityConstraint(a, b), name="A -> B")
#w.spawn(GravityConstraint(b, a), name="B -> A")
#
#w.spawn(GravityConstraint(a, c), name="A -> C")
#w.spawn(GravityConstraint(b, c), name="B -> C")
#
#w.spawn(GravityConstraint(c, a), name="C -> A")
#w.spawn(GravityConstraint(c, b), name="C -> B")

planet_objects = "\n".join(
    f"""    object_3d {planet["entity_name"]}.world_pos {{
        sphere radius={planet["radius"]} emissivity=1.0 {{
            color {planet["color"]}
        }}
    }}
    line_3d {planet["entity_name"]}.world_pos line_width=1.0 perspective=#false {{
        color yolk
    }}"""
    for planet in PLANETS
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
{planet_objects}
""".format(planet_objects=planet_objects))

sys = el.six_dof()
sim = w.run(sys, SIM_TIME_STEP, run_time_step=1 / 120.0, pre_step=pre_step, max_ticks=1000)
