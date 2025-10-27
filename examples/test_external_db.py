#!/usr/bin/env python3
"""
Test script for external database connection feature.

Usage:
    # Terminal 1: Start external database
    elodin-db run "[::]:2240" /tmp/test-db

    # Terminal 2: Run this script
    python examples/test_external_db.py

This will connect to the existing database instead of starting an embedded one.
"""

import elodin as el
import jax.numpy as jnp

# Create a simple simulation
w = el.World()

# Spawn a body
w.spawn(
    el.Body(
        world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 100.0])),
        world_vel=el.SpatialMotion(linear=jnp.array([0.0, 0.0, 0.0])),
        inertia=el.SpatialInertia(mass=1.0),
    ),
    name="test_body",
)

# Simple gravity system
@el.map
def gravity(pos: el.WorldPos, mass: el.Inertia) -> el.Force:
    g = jnp.array([0.0, 0.0, -9.81])
    return el.Force(linear=mass.mass() * g)

system = gravity | el.six_dof()

# Run with external database connection
# Make sure elodin-db is running at 127.0.0.1:2240 first!
print("Connecting to external database at 127.0.0.1:2240")
print("(Make sure 'elodin-db run \"[::]:2240\" /tmp/test-db' is running)")

w.run(
    system,
    sim_time_step=1 / 120.0,
    max_ticks=1000,  # Run for ~8 seconds
    db_addr="127.0.0.1:2240",  # <-- Connect to existing database
)

print("\nSimulation complete!")
print("Check the database for component data (once streaming is implemented)")

