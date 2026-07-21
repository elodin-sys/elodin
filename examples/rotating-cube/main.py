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
        // Position values (top) and attitude gimbals (bottom) from the same
        // ENU world_pos, shown in different display frames.
        vsplit share=0.28 {
            hsplit {
                geo_position_gauge name="NED" eql="cube.world_pos" source="ENU" display="NED"
                geo_position_gauge name="LLA" eql="cube.world_pos" source="ENU" display="LLA"
            }
            hsplit {
                orientation_gauge name="ATT NED" eql="cube.world_pos" source="ENU" display="NED"
                orientation_gauge name="ATT ECEF" eql="cube.world_pos" source="ENU" display="ECEF"
            }
        }
        tabs share=0.5 {
            viewport name=Viewport pos="cube.world_pos + (0.0,0.0,0.0,0.0, 3.0, 0.0, 1.5)" look_at="cube.world_pos" hdr=#true show_grid=#true
        }
        vsplit share=0.22 {
            graph "cube.world_pos" name="World Pos"
            graph "cube.world_vel" name="World Vel"
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
