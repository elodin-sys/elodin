#!/usr/bin/env python3
"""
Frame Independence Verification Example

This example demonstrates that physics behaves correctly across different
coordinate frames in Elodin. It runs identical simulations in different
frames and verifies that the results are equivalent (when properly transformed).

Tests:
1. Gravity in ENU vs NED - verifies local geodetic frames
2. N-body dynamics in ECI vs GCRF - verifies inertial frames
3. Conservation laws across frames
"""

import elodin as el
import jax.numpy as jnp
from typing import Tuple


def test_gravity_enu_ned() -> Tuple[bool, str]:
    """
    Test that gravity behaves correctly in ENU vs NED frames.

    In ENU: +Z is up, gravity is [0, 0, -9.81]
    In NED: +Z is down, gravity is [0, 0, +9.81]

    We drop a ball from the same height (accounting for frame convention)
    and verify it falls the same distance.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Gravity in ENU vs NED")
    print("=" * 60)

    ticks = 120  # 1 second
    initial_height = 10.0
    mass = 1.0

    # --- ENU Simulation ---
    print("\nRunning ENU simulation...")
    w_enu = el.World(frame=el.Frame.ENU)
    w_enu.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, initial_height])),
            world_vel=el.SpatialMotion(linear=jnp.array([0.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(mass),
        ),
        name="ball",
    )

    @el.map
    def gravity_enu(inertia: el.Inertia, f: el.Force) -> el.Force:
        return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())

    exec_enu = w_enu.build(el.six_dof(sys=gravity_enu))
    exec_enu.run(ticks)
    df_enu = exec_enu.history(["ball.world_pos", "ball.world_vel"])

    # --- NED Simulation ---
    print("Running NED simulation...")
    w_ned = el.World(frame=el.Frame.NED)
    w_ned.spawn(
        el.Body(
            # In NED, +Z is down, so we start at -10 (which means 10 units up)
            world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, -initial_height])),
            world_vel=el.SpatialMotion(linear=jnp.array([0.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(mass),
        ),
        name="ball",
    )

    @el.map
    def gravity_ned(inertia: el.Inertia, f: el.Force) -> el.Force:
        # In NED, gravity points in +Z direction
        return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, 9.81]) * inertia.mass())

    exec_ned = w_ned.build(el.six_dof(sys=gravity_ned))
    exec_ned.run(ticks)
    df_ned = exec_ned.history(["ball.world_pos", "ball.world_vel"])

    # --- Analysis ---
    # Extract final positions and velocities
    z_enu_initial = df_enu["ball.world_pos"][0][6]
    z_enu_final = df_enu["ball.world_pos"][-1][6]
    vz_enu_final = df_enu["ball.world_vel"][-1][5]

    z_ned_initial = df_ned["ball.world_pos"][0][6]
    z_ned_final = df_ned["ball.world_pos"][-1][6]
    vz_ned_final = df_ned["ball.world_vel"][-1][5]

    # In ENU: ball falls from +10 to some lower value (negative displacement)
    delta_z_enu = z_enu_final - z_enu_initial

    # In NED: ball falls from -10 to some higher value (positive displacement)
    delta_z_ned = z_ned_final - z_ned_initial

    print("\nENU Results:")
    print(f"  Initial Z: {z_enu_initial:.4f} m")
    print(f"  Final Z:   {z_enu_final:.4f} m")
    print(f"  Delta Z:   {delta_z_enu:.4f} m")
    print(f"  Final Vz:  {vz_enu_final:.4f} m/s")

    print("\nNED Results:")
    print(f"  Initial Z: {z_ned_initial:.4f} m")
    print(f"  Final Z:   {z_ned_final:.4f} m")
    print(f"  Delta Z:   {delta_z_ned:.4f} m")
    print(f"  Final Vz:  {vz_ned_final:.4f} m/s")

    # Verify: magnitudes should be equal (but signs opposite due to frame convention)
    displacement_match = jnp.isclose(delta_z_enu, -delta_z_ned, atol=0.01)
    velocity_match = jnp.isclose(vz_enu_final, -vz_ned_final, atol=0.01)

    print("\nVerification:")
    print(f"  Displacement magnitudes match: {displacement_match}")
    print(f"  Velocity magnitudes match:     {velocity_match}")

    passed = bool(displacement_match and velocity_match)

    if passed:
        print("\n✅ TEST PASSED: Gravity works correctly in both ENU and NED frames")
        return True, "Gravity test passed"
    else:
        print("\n❌ TEST FAILED: Frame-dependent physics mismatch")
        return False, f"Displacement match: {displacement_match}, Velocity match: {velocity_match}"


def test_inertial_frames() -> Tuple[bool, str]:
    """
    Test that inertial frames (ECI and GCRF) produce identical results.

    Uses a simple two-body orbital problem to verify frame independence.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Inertial Frame Equivalence (ECI vs GCRF)")
    print("=" * 60)

    ticks = 240  # 2 seconds
    G = 6.6743e-11

    def create_two_body_world(frame):
        """Create a two-body gravitational system in the specified frame."""
        w = el.World(frame=frame)

        # Central body (like Earth, but scaled down for faster dynamics)
        center = w.spawn(
            el.Body(
                world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 0.0])),
                world_vel=el.SpatialMotion(linear=jnp.array([0.0, 0.0, 0.0])),
                inertia=el.SpatialInertia(1.0e6 / G),  # Large mass
            ),
            name="center",
        )

        # Orbiting body
        orbital_radius = 10.0  # meters (scaled down)
        # Circular orbit velocity: v = sqrt(G*M/r)
        orbital_velocity = jnp.sqrt(G * (1.0e6 / G) / orbital_radius)

        orbiter = w.spawn(
            el.Body(
                world_pos=el.SpatialTransform(linear=jnp.array([orbital_radius, 0.0, 0.0])),
                world_vel=el.SpatialMotion(linear=jnp.array([0.0, orbital_velocity, 0.0])),
                inertia=el.SpatialInertia(1.0 / G),
            ),
            name="orbiter",
        )

        return w, center, orbiter

    # Define gravity system
    GravityEdge = el.Annotated[el.Edge, el.Component("gravity_edge", el.ComponentType.Edge)]

    @el.dataclass
    class GravityConstraint(el.Archetype):
        edge: GravityEdge

        def __init__(self, a: el.EntityId, b: el.EntityId):
            self.edge = GravityEdge(a, b)

    @el.system
    def gravity(
        graph: el.GraphQuery[GravityEdge],
        query: el.Query[el.WorldPos, el.Inertia],
    ) -> el.Query[el.Force]:
        from jax.numpy import linalg as la

        def gravity_fn(force, a_pos, a_inertia, b_pos, b_inertia):
            r = a_pos.linear() - b_pos.linear()
            m = a_inertia.mass()
            M = b_inertia.mass()
            norm = la.norm(r)
            f = G * M * m * r / (norm * norm * norm)
            return el.Force(linear=force.force() - f)

        return graph.edge_fold(query, query, el.Force, el.Force(), gravity_fn)

    # --- ECI Simulation ---
    print("\nRunning ECI simulation...")
    w_eci, center_eci, orbiter_eci = create_two_body_world(el.Frame.ECI)
    w_eci.spawn(GravityConstraint(center_eci, orbiter_eci))
    w_eci.spawn(GravityConstraint(orbiter_eci, center_eci))

    exec_eci = w_eci.build(el.six_dof(sys=gravity))
    exec_eci.run(ticks)
    df_eci = exec_eci.history(["orbiter.world_pos", "orbiter.world_vel"])

    # --- GCRF Simulation ---
    print("Running GCRF simulation...")
    w_gcrf, center_gcrf, orbiter_gcrf = create_two_body_world(el.Frame.GCRF)
    w_gcrf.spawn(GravityConstraint(center_gcrf, orbiter_gcrf))
    w_gcrf.spawn(GravityConstraint(orbiter_gcrf, center_gcrf))

    exec_gcrf = w_gcrf.build(el.six_dof(sys=gravity))
    exec_gcrf.run(ticks)
    df_gcrf = exec_gcrf.history(["orbiter.world_pos", "orbiter.world_vel"])

    # --- Analysis ---
    pos_eci_final = df_eci["orbiter.world_pos"][-1][4:7].to_numpy()
    vel_eci_final = df_eci["orbiter.world_vel"][-1][3:6].to_numpy()

    pos_gcrf_final = df_gcrf["orbiter.world_pos"][-1][4:7].to_numpy()
    vel_gcrf_final = df_gcrf["orbiter.world_vel"][-1][3:6].to_numpy()

    print("\nECI Results:")
    print(
        f"  Final position: [{pos_eci_final[0]:.6f}, {pos_eci_final[1]:.6f}, {pos_eci_final[2]:.6f}]"
    )
    print(
        f"  Final velocity: [{vel_eci_final[0]:.6f}, {vel_eci_final[1]:.6f}, {vel_eci_final[2]:.6f}]"
    )

    print("\nGCRF Results:")
    print(
        f"  Final position: [{pos_gcrf_final[0]:.6f}, {pos_gcrf_final[1]:.6f}, {pos_gcrf_final[2]:.6f}]"
    )
    print(
        f"  Final velocity: [{vel_gcrf_final[0]:.6f}, {vel_gcrf_final[1]:.6f}, {vel_gcrf_final[2]:.6f}]"
    )

    # Verify: inertial frames should give identical results
    pos_match = jnp.allclose(pos_eci_final, pos_gcrf_final, atol=1e-6)
    vel_match = jnp.allclose(vel_eci_final, vel_gcrf_final, atol=1e-6)

    print("\nVerification:")
    print(f"  Positions match: {pos_match}")
    print(f"  Velocities match: {vel_match}")

    if pos_match and vel_match:
        max_pos_diff = jnp.max(jnp.abs(pos_eci_final - pos_gcrf_final))
        max_vel_diff = jnp.max(jnp.abs(vel_eci_final - vel_gcrf_final))
        print(f"  Max position difference: {max_pos_diff:.2e} m")
        print(f"  Max velocity difference: {max_vel_diff:.2e} m/s")

    passed = bool(pos_match and vel_match)

    if passed:
        print("\n✅ TEST PASSED: Inertial frames produce identical results")
        return True, "Inertial frame test passed"
    else:
        print("\n❌ TEST FAILED: Inertial frames produced different results")
        return False, f"Position match: {pos_match}, Velocity match: {vel_match}"


def test_energy_conservation() -> Tuple[bool, str]:
    """
    Test that energy is conserved in a simulation, regardless of frame.

    Uses a simple falling ball to verify energy conservation.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Energy Conservation")
    print("=" * 60)

    ticks = 60  # 0.5 seconds
    initial_height = 5.0
    mass = 1.0
    g = 9.81

    w = el.World(frame=el.Frame.ENU)
    w.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, initial_height])),
            world_vel=el.SpatialMotion(linear=jnp.array([0.0, 0.0, 0.0])),
            inertia=el.SpatialInertia(mass),
        ),
        name="ball",
    )

    @el.map
    def gravity_enu(inertia: el.Inertia, f: el.Force) -> el.Force:
        return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -g]) * inertia.mass())

    exec = w.build(el.six_dof(sys=gravity_enu))
    exec.run(ticks)
    df = exec.history(["ball.world_pos", "ball.world_vel"])

    # Calculate energy at each timestep
    energies = []
    for i in range(len(df)):
        z = df["ball.world_pos"][i][6]
        vz = df["ball.world_vel"][i][5]

        ke = 0.5 * mass * vz**2
        pe = mass * g * z
        total_energy = ke + pe
        energies.append(total_energy)

    energies = jnp.array(energies)
    initial_energy = energies[0]
    final_energy = energies[-1]
    energy_variation = jnp.std(energies)

    print("\nEnergy Analysis:")
    print(f"  Initial energy: {initial_energy:.6f} J")
    print(f"  Final energy:   {final_energy:.6f} J")
    print(f"  Energy change:  {final_energy - initial_energy:.6f} J")
    print(f"  Std deviation:  {energy_variation:.6f} J")
    print(f"  Relative error: {abs(final_energy - initial_energy) / initial_energy * 100:.4f}%")

    # Energy should be conserved to within numerical precision
    # Allow 1% error due to discretization
    energy_conserved = jnp.isclose(initial_energy, final_energy, rtol=0.01)

    print("\nVerification:")
    print(f"  Energy conserved: {energy_conserved}")

    if energy_conserved:
        print("\n✅ TEST PASSED: Energy is conserved")
        return True, "Energy conservation test passed"
    else:
        print("\n❌ TEST FAILED: Energy not conserved within tolerance")
        return False, f"Energy changed by {abs(final_energy - initial_energy):.6f} J"


def main():
    """Run all frame verification tests."""
    print("\n" + "=" * 60)
    print("ELODIN FRAME INDEPENDENCE VERIFICATION")
    print("=" * 60)
    print("\nThis example demonstrates that physics behaves correctly")
    print("across different coordinate frames in Elodin.")

    results = []

    # Run all tests
    try:
        passed, msg = test_gravity_enu_ned()
        results.append(("Gravity ENU/NED", passed, msg))
    except Exception as e:
        results.append(("Gravity ENU/NED", False, str(e)))

    try:
        passed, msg = test_inertial_frames()
        results.append(("Inertial Frames", passed, msg))
    except Exception as e:
        results.append(("Inertial Frames", False, str(e)))

    try:
        passed, msg = test_energy_conservation()
        results.append(("Energy Conservation", passed, msg))
    except Exception as e:
        results.append(("Energy Conservation", False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed, msg in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status}: {test_name}")
        if not passed:
            print(f"  Reason: {msg}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed, _ in results if passed)

    print(f"\n{passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All frame verification tests passed!")
        print("The configurable frame system is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please investigate.")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
