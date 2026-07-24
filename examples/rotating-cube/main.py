import elodin as el
import jax.numpy as jnp

SIM_TIME_STEP = 1.0 / 120.0
# Constant spin about world +X (rad/s).
SPIN_RATE_RAD_S = jnp.pi / 2.0

w = el.World()
w.spawn(
    el.Body(
        world_pos=el.SpatialTransform(
            angular=el.Quaternion.identity(),
            linear=jnp.array([0.0, 0.0, 1.0]),
        ),
        world_vel=el.SpatialMotion(angular=jnp.array([SPIN_RATE_RAD_S, 0.0, 0.0])),
        inertia=el.SpatialInertia(1.0),
    ),
    name="Cube",
    id="cube",
)
w.schematic(
    """
    // Geodetic origin so ENU world_pos maps to real lat/lon/alt for the gauges.
    coordinate frame=ENU lat=28.6084 lon=-80.6043 alt=3.0
    hsplit {
        // Left column: geo-position gauge stacked directly above a component
        // monitor on the same data. Drag the divider between this column and the
        // viewport to widen/narrow both at once — the gauge value cards should
        // reflow (row -> wrap) exactly like the monitor's number cards.
        vsplit share=0.32 {
            geo_position_gauge name="GEO NED" eql="cube.world_pos" source="ENU" display="NED"
            component_monitor name="MONITOR world_pos" component_name="cube.world_pos"
        }
        tabs share=0.44 {
            viewport name=Viewport pos="cube.world_pos + (0.0,0.0,0.0,0.0, 3.0, 0.0, 1.5)" look_at="cube.world_pos" hdr=#true show_grid=#true
        }
        // Attitude gimbals from the same ENU world_pos, in two display frames.
        vsplit share=0.24 {
            orientation_gauge name="ATT NED" eql="cube.world_pos" source="ENU" display="NED"
            orientation_gauge name="ATT ECEF" eql="cube.world_pos" source="ENU" display="ECEF"
        }
    }

    object_3d "(0,0,0,1, cube.world_pos[4],cube.world_pos[5],cube.world_pos[6])" {
        glb path="compass.glb"
    }
    object_3d cube.world_pos {
        box x=0.5 y=0.5 z=0.5 {
            color 76 175 80
        }
    }
""",
    "rotating-cube.kdl",
)

sys = el.six_dof(integrator=el.Integrator.Rk4)
w.run(
    sys,
    simulation_rate=1.0 / SIM_TIME_STEP,
    default_playback_speed=1.0,
    max_ticks=2400,
)
