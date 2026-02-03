#!/usr/bin/env python3
"""
Video Streaming Example

Demonstrates streaming video from GStreamer into Elodin DB and
displaying it in the Elodin Editor's video stream tile.

Usage:
    elodin editor examples/video-stream/main.py

The video stream tile is automatically created via the schematic.
"""

from pathlib import Path

import elodin as el
import jax.numpy as jnp

SIM_TIME_STEP = 1.0 / 120.0

world = el.World()

# Create a simple rotating body for visual reference
# Based on the three-body example structure
body = world.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(
                linear=jnp.array([0.0, 0.0, 0.0]),
            ),
            world_vel=el.SpatialMotion(
                angular=jnp.array([0.0, 0.5, 0.0]),  # Slow rotation around Y axis
                linear=jnp.array([0.0, 0.0, 0.0]),
            ),
            inertia=el.SpatialInertia(mass=1.0),
        ),
    ],
    name="body",
)

# Register the video streaming process via S10 recipe
# The shell script builds the GStreamer plugin and runs the pipeline
stream_script = Path(__file__).parent / "stream-video.sh"
video_streamer = el.s10.PyRecipe.process(
    name="video-stream",
    cmd="bash",
    args=[str(stream_script)],
    cwd=str(Path(__file__).parent),
)
world.recipe(video_streamer)

# Define schematic with 3D viewport and video stream tile
world.schematic("""
    hsplit {
        tabs share=0.5 {
            viewport name=Viewport pos="(0,0,0,0,0,0,5)" look_at="(0,0,0,0,0,0,0)" show_grid=#true
        }
        tabs share=0.5 {
            video_stream "test-video" name="Test Pattern"
        }
    }
    object_3d body.world_pos {
        sphere radius=0.5 {
            color cyan
        }
    }
""")

print("Video Streaming Example")
print("=======================")
print()
print("The video stream tile will appear automatically.")
print("A rotating sphere is shown in the 3D viewport for reference.")
print()

# Run simulation with the standard six_dof system
sys = el.six_dof()
sim = world.run(sys, SIM_TIME_STEP, run_time_step=1 / 60.0)

