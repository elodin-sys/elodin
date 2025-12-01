#!/usr/bin/env python3
"""
Streamlit UI for Rocket Simulation
Create, run, and visualize rocket simulations with Elodin integration
"""

import sys
import os
from pathlib import Path

# Add the rocket-barrowman directory to path (but not if it's already there)
_rocket_dir = os.path.dirname(os.path.abspath(__file__))
if _rocket_dir not in sys.path:
    sys.path.insert(0, _rocket_dir)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any, List, Tuple

# Import rocket simulation components
from environment import Environment
from motor_model import Motor
from rocket_model import Rocket as RocketModel
from flight_solver import FlightSolver, FlightResult
from calisto_builder import build_calisto
from rocket_visualizer import visualize_rocket_3d, visualize_rocket_2d_side_view
from openrocket_components import (
    Rocket,
    NoseCone,
    BodyTube,
    TrapezoidFinSet,
    Transition,
    InnerTube,
    CenteringRing,
    MassComponent,
    Parachute,
    MATERIALS,
)
from openrocket_motor import Motor as ORMotor
from ai_rocket_builder import RocketDesigner, RocketRequirements
from motor_scraper import ThrustCurveScraper, MotorData
import os

# Page config
st.set_page_config(
    page_title="Rocket Simulator", page_icon="🚀", layout="wide", initial_sidebar_state="expanded"
)

# Initialize session state
if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None
if "rocket_config" not in st.session_state:
    st.session_state.rocket_config = None
if "motor_config" not in st.session_state:
    st.session_state.motor_config = None
if "motor_database" not in st.session_state:
    st.session_state.motor_database = []
if "ai_designer" not in st.session_state:
    st.session_state.ai_designer = None

# Auto-load motor database on startup
if len(st.session_state.motor_database) == 0:
    try:
        scraper = ThrustCurveScraper()
        motor_db = scraper.load_motor_database()
        if motor_db and len(motor_db) >= 50:
            # Good database loaded
            st.session_state.motor_database = motor_db
            # Initialize AI designer with loaded motors
            openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
            st.session_state.ai_designer = RocketDesigner(motor_db, openai_api_key=openai_key)
        elif motor_db and len(motor_db) < 50:
            # Small database - use it but warn user
            st.session_state.motor_database = motor_db
            openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
            st.session_state.ai_designer = RocketDesigner(motor_db, openai_api_key=openai_key)
            st.warning(
                f"⚠️ Only {len(motor_db)} motors in database. Download full database from sidebar for better motor selection."
            )
    except Exception:
        # If loading fails, that's okay - user can download manually
        pass

if "motor_database" not in st.session_state:
    st.session_state.motor_database = []
if "ai_designer" not in st.session_state:
    st.session_state.ai_designer = None


def build_custom_rocket(config: Dict[str, Any]) -> Rocket:
    """Build a custom rocket from configuration."""
    rocket = Rocket(config.get("name", "Custom Rocket"))

    # Nose cone
    if config.get("has_nose", True):
        # Get nosecone shape from config, default to VON_KARMAN
        nose_shape_str = config.get("nose_shape", "VON_KARMAN").upper()
        nose_shape_map = {
            "CONICAL": NoseCone.Shape.CONICAL,
            "OGIVE": NoseCone.Shape.OGIVE,
            "ELLIPSOID": NoseCone.Shape.ELLIPSOID,
            "PARABOLIC": NoseCone.Shape.PARABOLIC,
            "POWER_SERIES": NoseCone.Shape.POWER_SERIES,
            "HAACK": NoseCone.Shape.HAACK,
            "VON_KARMAN": NoseCone.Shape.VON_KARMAN,
        }
        nose_shape = nose_shape_map.get(nose_shape_str, NoseCone.Shape.VON_KARMAN)

        nose = NoseCone(
            name="Nose Cone",
            length=config.get("nose_length", 0.5),
            base_radius=config.get("body_radius", 0.0635),
            thickness=config.get("nose_thickness", 0.003),
            shape=nose_shape,
        )
        nose.material = MATERIALS.get(
            config.get("nose_material", "Fiberglass"), MATERIALS["Fiberglass"]
        )
        nose.position.x = 0.0
        rocket.add_child(nose)

    # Body tube
    body = BodyTube(
        name="Body Tube",
        length=config.get("body_length", 1.5),
        outer_radius=config.get("body_radius", 0.0635),
        thickness=config.get("body_thickness", 0.003),
    )
    body.material = MATERIALS.get(
        config.get("body_material", "Fiberglass"), MATERIALS["Fiberglass"]
    )
    body.position.x = config.get("nose_length", 0.5) if config.get("has_nose", True) else 0.0
    rocket.add_child(body)

    # Fins
    if config.get("has_fins", True):
        fins = TrapezoidFinSet(
            name="Fins",
            fin_count=config.get("fin_count", 4),
            root_chord=config.get("fin_root_chord", 0.12),
            tip_chord=config.get("fin_tip_chord", 0.06),
            span=config.get("fin_span", 0.11),
            sweep=config.get("fin_sweep", 0.06),
            thickness=config.get("fin_thickness", 0.005),
        )
        fins.material = MATERIALS.get(
            config.get("fin_material", "Fiberglass"), MATERIALS["Fiberglass"]
        )
        body_length = config.get("body_length", 1.5)
        nose_length = config.get("nose_length", 0.5) if config.get("has_nose", True) else 0.0
        fins.position.x = nose_length + body_length - config.get("fin_root_chord", 0.12)
        body.add_child(fins)

    # Motor mount
    if config.get("has_motor_mount", True):
        motor_mount = InnerTube(
            name="Motor Mount",
            length=config.get("motor_mount_length", 0.5),
            outer_radius=config.get("motor_mount_radius", 0.041),
            thickness=config.get("motor_mount_thickness", 0.003),
        )
        motor_mount.material = MATERIALS.get(
            config.get("motor_mount_material", "Fiberglass"), MATERIALS["Fiberglass"]
        )
        body_length = config.get("body_length", 1.5)
        nose_length = config.get("nose_length", 0.5) if config.get("has_nose", True) else 0.0
        motor_mount.position.x = nose_length + body_length - config.get("motor_mount_length", 0.5)
        motor_mount.motor_mount = True
        body.add_child(motor_mount)

    # Parachutes
    if config.get("has_main_chute", True):
        main_chute = Parachute(
            name="Main",
            diameter=config.get("main_chute_diameter", 2.91),
            cd=config.get("main_chute_cd", 1.5),
        )
        main_chute.deployment_event = config.get("main_deployment_event", "ALTITUDE")
        main_chute.deployment_altitude = config.get("main_deployment_altitude", 800.0)
        main_chute.deployment_delay = config.get("main_deployment_delay", 1.5)
        if config.get("has_nose", True):
            nose.add_child(main_chute)
        else:
            body.add_child(main_chute)

    if config.get("has_drogue", True):
        drogue = Parachute(
            name="Drogue",
            diameter=config.get("drogue_diameter", 0.99),
            cd=config.get("drogue_cd", 1.3),
        )
        drogue.deployment_event = config.get("drogue_deployment_event", "APOGEE")
        drogue.deployment_altitude = config.get("drogue_deployment_altitude", 0.0)
        drogue.deployment_delay = config.get("drogue_deployment_delay", 1.5)
        if config.get("has_nose", True):
            nose.add_child(drogue)
        else:
            body.add_child(drogue)

    rocket.calculate_reference_values()
    return rocket


def build_custom_motor(config: Dict[str, Any]) -> ORMotor:
    """Build a custom motor from configuration."""
    # Check if we have a thrust curve from scraped motor
    if "thrust_curve" in config and config["thrust_curve"]:
        # Use actual thrust curve from database
        thrust_curve = config["thrust_curve"]
        burn_time = config.get(
            "burn_time", max([t for t, _ in thrust_curve]) if thrust_curve else 3.9
        )
        total_impulse = config.get(
            "total_impulse",
            sum(
                (thrust_curve[i][1] + thrust_curve[i + 1][1])
                / 2
                * (thrust_curve[i + 1][0] - thrust_curve[i][0])
                for i in range(len(thrust_curve) - 1)
            )
            if len(thrust_curve) > 1
            else 0,
        )
    else:
        # Generate simple thrust curve from max thrust and burn time
        burn_time = config.get("burn_time", 3.9)
        max_thrust = config.get("max_thrust", 2200.0)
        avg_thrust = config.get("avg_thrust", max_thrust * 0.7)

        # Generate simple thrust curve
        times = np.linspace(0, burn_time, 20).tolist()
        # Simple bell curve approximation
        thrusts = []
        for t in times:
            if t < burn_time * 0.1:
                # Startup
                thrust = max_thrust * (t / (burn_time * 0.1))
            elif t < burn_time * 0.9:
                # Sustained
                thrust = avg_thrust
            else:
                # Tailoff
                thrust = avg_thrust * (1 - (t - burn_time * 0.9) / (burn_time * 0.1))
            thrusts.append(max(0, thrust))

        thrust_curve = list(zip(times, thrusts))
        total_impulse = sum(
            (thrusts[i] + thrusts[i + 1]) / 2 * (times[i + 1] - times[i])
            for i in range(len(times) - 1)
        )

    motor = ORMotor(
        designation=config.get("designation", config.get("motor_name", "Custom Motor")),
        manufacturer=config.get("manufacturer", config.get("motor_manufacturer", "Custom")),
        diameter=config.get("diameter", config.get("motor_diameter", 0.075)),
        length=config.get("length", config.get("motor_length", 0.64)),
        total_mass=config.get("total_mass", config.get("motor_total_mass", 4.771)),
        propellant_mass=config.get("propellant_mass", config.get("motor_propellant_mass", 2.956)),
        thrust_curve=thrust_curve,
        burn_time=burn_time,
        total_impulse=config.get("total_impulse", total_impulse),
    )

    motor.cg_position = config.get("motor_cg_position", 0.317)
    motor.propellant_cg = config.get("motor_propellant_cg", 0.397)
    motor.inertia_axial = config.get("motor_inertia_axial", 0.002)
    motor.inertia_lateral = config.get("motor_inertia_lateral", 0.125)

    return motor


def visualize_results(result: FlightResult):
    """Create visualizations of simulation results."""
    if not result or not result.history:
        st.error("No simulation data to visualize")
        return

    history = result.history

    # Extract data
    times = [s.time for s in history]
    altitudes = [s.z for s in history]
    velocities = [np.linalg.norm(s.velocity) for s in history]
    downrange = [np.linalg.norm([s.x, s.y]) for s in history]
    machs = [s.mach for s in history]
    aoas = [s.angle_of_attack for s in history]
    dynamic_pressures = [s.dynamic_pressure for s in history]

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Trajectory", "Performance", "Aerodynamics", "3D Path"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Altitude vs Time
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times, y=altitudes, mode="lines", name="Altitude", line=dict(color="blue")
                )
            )
            fig.update_layout(
                title="Altitude vs Time",
                xaxis_title="Time (s)",
                yaxis_title="Altitude (m)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Velocity vs Time
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times, y=velocities, mode="lines", name="Velocity", line=dict(color="red")
                )
            )
            fig.update_layout(
                title="Velocity vs Time",
                xaxis_title="Time (s)",
                yaxis_title="Velocity (m/s)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Downrange vs Altitude
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=downrange, y=altitudes, mode="lines", name="Trajectory", line=dict(color="green")
            )
        )
        fig.update_layout(
            title="Trajectory (Downrange vs Altitude)",
            xaxis_title="Downrange (m)",
            yaxis_title="Altitude (m)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Key metrics
            max_alt = max(altitudes)
            max_v = max(velocities)
            apogee_time = times[altitudes.index(max_alt)]
            flight_time = times[-1]

            st.metric("Max Altitude", f"{max_alt:.1f} m")
            st.metric("Max Velocity", f"{max_v:.1f} m/s")
            st.metric("Apogee Time", f"{apogee_time:.2f} s")
            st.metric("Flight Time", f"{flight_time:.2f} s")

        with col2:
            # Mach number
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=times, y=machs, mode="lines", name="Mach", line=dict(color="purple"))
            )
            fig.update_layout(
                title="Mach Number vs Time",
                xaxis_title="Time (s)",
                yaxis_title="Mach Number",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            # Angle of Attack
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=times, y=aoas, mode="lines", name="AoA", line=dict(color="orange"))
            )
            fig.update_layout(
                title="Angle of Attack vs Time",
                xaxis_title="Time (s)",
                yaxis_title="Angle of Attack (rad)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Dynamic Pressure
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=dynamic_pressures,
                    mode="lines",
                    name="Dynamic Pressure",
                    line=dict(color="cyan"),
                )
            )
            fig.update_layout(
                title="Dynamic Pressure vs Time",
                xaxis_title="Time (s)",
                yaxis_title="Dynamic Pressure (Pa)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # 3D trajectory
        x_coords = [s.x for s in history]
        y_coords = [s.y for s in history]
        z_coords = [s.z for s in history]

        fig = go.Figure(
            data=go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="lines",
                line=dict(color=times, colorscale="Viridis", width=6),
                marker=dict(size=2),
            )
        )
        fig.update_layout(
            title="3D Flight Path",
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Altitude (m)"),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)


def launch_elodin_editor(result: FlightResult, solver: FlightSolver):
    """Launch Elodin editor with simulation results in a separate process."""
    import subprocess
    import sys
    import tempfile
    import pickle
    from pathlib import Path

    try:
        # Save the result and solver to temporary files so the subprocess can load them
        temp_dir = Path(tempfile.gettempdir()) / "elodin_rocket_sim"
        temp_dir.mkdir(exist_ok=True)

        result_file = temp_dir / "simulation_result.pkl"
        solver_file = temp_dir / "solver.pkl"

        # Save the data - convert to dict to avoid pickling issues
        # Extract essential data from FlightResult
        # FlightResult has history and summary dict
        summary = result.summary if hasattr(result, "summary") and result.summary else {}

        # Calculate max values from history if not in summary
        if not summary or "max_altitude" not in summary:
            max_alt = max(s.z for s in result.history) if result.history else 0.0
            max_v = (
                max(np.linalg.norm(s.velocity) for s in result.history) if result.history else 0.0
            )
            apogee_state = (
                next((s for s in result.history if s.z == max_alt), result.history[-1])
                if result.history
                else None
            )
            apogee_time = apogee_state.time if apogee_state else 0.0
            landing_time = result.history[-1].time if result.history else 0.0
        else:
            max_alt = summary.get("max_altitude", 0.0)
            max_v = summary.get("max_velocity", 0.0)
            apogee_time = summary.get("apogee_time", 0.0)
            landing_time = summary.get("landing_time", 0.0)

        result_data = {
            "history": [
                {
                    "time": s.time,
                    "position": s.position.tolist()
                    if isinstance(s.position, np.ndarray)
                    else s.position,
                    "velocity": s.velocity.tolist()
                    if isinstance(s.velocity, np.ndarray)
                    else s.velocity,
                    "quaternion": s.quaternion.tolist()
                    if isinstance(s.quaternion, np.ndarray)
                    else s.quaternion,
                    "angular_velocity": s.angular_velocity.tolist()
                    if isinstance(s.angular_velocity, np.ndarray)
                    else s.angular_velocity,
                    "motor_mass": s.motor_mass,
                    "angle_of_attack": s.angle_of_attack,
                    "sideslip": s.sideslip,
                    "mach": getattr(s, "mach", 0.0),
                    "dynamic_pressure": getattr(s, "dynamic_pressure", 0.0),
                    "drag_force": getattr(s, "drag_force", np.array([0.0, 0.0, 0.0])).tolist()
                    if isinstance(getattr(s, "drag_force", None), np.ndarray)
                    else [0.0, 0.0, 0.0],
                    "lift_force": getattr(s, "lift_force", np.array([0.0, 0.0, 0.0])).tolist()
                    if isinstance(getattr(s, "lift_force", None), np.ndarray)
                    else [0.0, 0.0, 0.0],
                    "parachute_drag": getattr(
                        s, "parachute_drag", np.array([0.0, 0.0, 0.0])
                    ).tolist()
                    if isinstance(getattr(s, "parachute_drag", None), np.ndarray)
                    else [0.0, 0.0, 0.0],
                    "moment_world": getattr(s, "moment_world", np.array([0.0, 0.0, 0.0])).tolist()
                    if isinstance(getattr(s, "moment_world", None), np.ndarray)
                    else [0.0, 0.0, 0.0],
                    "total_aero_force": s.total_aero_force.tolist()
                    if isinstance(s.total_aero_force, np.ndarray)
                    else s.total_aero_force,
                }
                for s in result.history
            ],
            "max_altitude": max_alt,
            "max_velocity": max_v,
            "apogee_time": apogee_time,
            "landing_time": landing_time,
        }

        # Extract essential data from FlightSolver
        # We need mass_model data for visualization
        mass_model = solver.mass_model
        solver_data = {
            "rocket": {
                "dry_mass": solver.rocket.dry_mass,
                "dry_cg": solver.rocket.dry_cg,
                "reference_diameter": solver.rocket.reference_diameter,
                "structural_mass": mass_model.structural_mass,
                "structural_cg": mass_model.structural_cg,
                "structural_inertia": mass_model.structural_inertia.tolist()
                if isinstance(mass_model.structural_inertia, np.ndarray)
                else list(mass_model.structural_inertia),
            },
            "motor": {
                "total_mass": solver.motor.total_mass,
                "propellant_mass": solver.motor.propellant_mass,
            },
            "environment": {
                "elevation": solver.environment.elevation,
            },
            "mass_model": {
                "times": mass_model.times.tolist()
                if isinstance(mass_model.times, np.ndarray)
                else list(mass_model.times),
                "total_mass_values": mass_model.total_mass_values.tolist()
                if isinstance(mass_model.total_mass_values, np.ndarray)
                else list(mass_model.total_mass_values),
                "inertia_values": mass_model.inertia_values.tolist()
                if isinstance(mass_model.inertia_values, np.ndarray)
                else [[0.0, 0.0, 0.0]],
            },
        }

        with open(result_file, "wb") as f:
            pickle.dump(result_data, f)
        with open(solver_file, "wb") as f:
            pickle.dump(solver_data, f)

        # Get paths
        script_dir = Path(__file__).parent
        main_py = script_dir / "main.py"

        # Verify files exist
        if not main_py.exists():
            st.error(f"❌ main.py not found at {main_py}")
            return

        # Verify pickle files exist
        if not result_file.exists() or not solver_file.exists():
            st.error(f"❌ Simulation data not found!")
            st.info(f"💡 Result file: {result_file} (exists: {result_file.exists()})")
            st.info(f"💡 Solver file: {solver_file} (exists: {solver_file.exists()})")
            st.info("💡 Please run a simulation first before launching Elodin editor.")
            return

        # Launch in a separate terminal window so we can see errors
        with st.spinner("Launching Elodin editor..."):
            import platform
            import os

            # Build the command - use elodin editor CLI
            # The elodin module provides a CLI: elodin editor <file>
            # Make sure we use the full path and run from the script directory
            cmd = ["elodin", "editor", str(main_py)]

            if platform.system() == "Windows":
                # Windows: use cmd.exe to open new window
                full_cmd = f'start cmd /k "cd /d {script_dir} && {" ".join(cmd)}"'
                process = subprocess.Popen(full_cmd, shell=True)
            else:
                # Linux/Unix: try to open in a new terminal window
                # First, try to find a terminal emulator
                terminals = [
                    (["gnome-terminal", "--"], "bash -c"),
                    (["xterm", "-e"], "bash -c"),
                    (["x-terminal-emulator", "-e"], "bash -c"),
                    (["konsole", "-e"], "bash -c"),
                    (["terminator", "-e"], "bash -c"),
                ]

                # Build command string for terminal
                # Use elodin editor which will automatically detect pickle files
                cmd_str = f'cd {script_dir} && elodin editor {main_py.name}; echo ""; echo "Press Enter to close..."; read'

                process = None
                terminal_used = None

                for term_base, shell_prefix in terminals:
                    try:
                        # Check if terminal exists
                        which_result = subprocess.run(
                            ["which", term_base[0]], capture_output=True, timeout=1
                        )
                        if which_result.returncode == 0:
                            # Found terminal, use it
                            if shell_prefix == "bash -c":
                                full_cmd = term_base + ["bash", "-c", cmd_str]
                            else:
                                full_cmd = term_base + [cmd_str]

                            process = subprocess.Popen(
                                full_cmd, cwd=str(script_dir), start_new_session=True
                            )
                            terminal_used = term_base[0]
                            break
                    except:
                        continue

                # Fallback: try using the wrapper script
                if process is None:
                    wrapper_script = script_dir / "launch_elodin_wrapper.sh"
                    if wrapper_script.exists():
                        # Try to run wrapper script in a terminal
                        cmd_str = f"bash {wrapper_script}"
                        for term_base, shell_prefix in terminals[:3]:  # Try first 3 terminals
                            try:
                                which_result = subprocess.run(
                                    ["which", term_base[0]], capture_output=True, timeout=1
                                )
                                if which_result.returncode == 0:
                                    if shell_prefix == "bash -c":
                                        full_cmd = term_base + [
                                            "bash",
                                            "-c",
                                            cmd_str + '; read -p "Press Enter to close..."',
                                        ]
                                    else:
                                        full_cmd = term_base + [cmd_str]

                                    process = subprocess.Popen(
                                        full_cmd, cwd=str(script_dir), start_new_session=True
                                    )
                                    terminal_used = term_base[0]
                                    break
                            except:
                                continue

                    # Last resort: run in background with logging
                    if process is None:
                        log_file = script_dir / "elodin_launch.log"
                        try:
                            with open(log_file, "w") as log:
                                process = subprocess.Popen(
                                    cmd,
                                    cwd=str(script_dir),
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                )
                            st.warning("⚠️ Could not open terminal window. Running in background.")
                            st.info(f"💡 Check log file: `{log_file}`")
                            st.info(f"💡 Or run manually: `bash {wrapper_script}`")
                        except Exception as e:
                            st.error(f"❌ Failed to launch: {str(e)}")
                            st.info(f"💡 Try running manually: `bash {wrapper_script}`")
                            return

            # Give it a moment to start
            import time

            time.sleep(1.5)

            # Check if process started successfully
            if process:
                if process.poll() is None:  # Process is still running
                    if terminal_used:
                        st.success(f"✅ Elodin editor launched in {terminal_used}!")
                    else:
                        st.success("✅ Elodin editor launched in background!")
                    st.info("💡 The Elodin editor window should appear shortly.")
                else:
                    # Process exited immediately - there was an error
                    return_code = process.returncode
                    st.error(f"❌ Elodin editor exited immediately (code: {return_code})")

                    # Try to get error output
                    if log_file and log_file.exists():
                        try:
                            with open(log_file, "r") as f:
                                error_output = f.read()[:1000]
                            if error_output:
                                st.code(error_output)
                        except:
                            pass

                    st.info("💡 Common issues:")
                    st.info("   • Make sure you're in the nix shell: `nix develop`")
                    st.info("   • Ensure Elodin is installed")
                    st.info("   • Check that simulation data exists")
                    st.info(
                        f"💡 Try running manually: `cd {script_dir} && python3 main.py --visualize`"
                    )
            else:
                st.error("❌ Could not launch Elodin editor.")
                st.info(
                    f"💡 Try running manually: `cd {script_dir} && python3 main.py --visualize`"
                )
                st.code(f"Try running manually: python3 {main_py} --visualize")

    except Exception as e:
        st.error(f"❌ Error launching Elodin: {str(e)}")
        import traceback

        st.code(traceback.format_exc())
        # Also try to show what command we tried to run
        st.info(f"💡 Tried to run: `python3 {main_py} --visualize`")
        st.info(f"💡 Working directory: {script_dir}")
        st.info(f"💡 Temp files location: {temp_dir}")
        st.info(
            f"💡 Result file exists: {result_file.exists() if 'result_file' in locals() else 'N/A'}"
        )
        st.info(
            f"💡 Solver file exists: {solver_file.exists() if 'solver_file' in locals() else 'N/A'}"
        )


def main():
    """Main Streamlit app."""
    st.title("🚀 Rocket Flight Simulator")
    st.markdown("Create, simulate, and visualize rocket flights with 6-DOF dynamics")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Rocket selection
        rocket_type = st.radio(
            "Rocket Type",
            ["Calisto (Default)", "Custom Rocket", "AI Builder"],
            help="Use the default Calisto rocket, build custom, or let AI design it",
        )

        # AI Builder section
        if rocket_type == "AI Builder":
            st.subheader("🤖 AI Rocket Designer")
            st.markdown("Describe your rocket requirements in natural language:")

            ai_input = st.text_area(
                "Requirements",
                placeholder="e.g., 'I want a rocket that goes to 10000 ft, carries a 6U payload that weighs 10 lbs'",
                height=100,
            )

            if st.button("✨ Generate Rocket Design", type="primary"):
                if ai_input:
                    with st.spinner("Analyzing requirements and designing rocket..."):
                        try:
                            # Initialize designer if needed
                            if st.session_state.ai_designer is None:
                                # Use existing motor database or try to load
                                motor_db = st.session_state.motor_database
                                if not motor_db:
                                    scraper = ThrustCurveScraper()
                                    motor_db = scraper.load_motor_database()
                                    st.session_state.motor_database = motor_db

                                # Get OpenAI API key from environment or session state
                                openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get(
                                    "openai_api_key"
                                )
                                st.session_state.ai_designer = RocketDesigner(
                                    motor_db, openai_api_key=openai_key
                                )

                            designer = st.session_state.ai_designer

                            # Parse and design
                            req = designer.parse_requirements(ai_input)
                            config, motor_config = designer.build_rocket_config(req)

                            # Store in session state
                            st.session_state.rocket_config = config
                            if motor_config:
                                st.session_state.motor_config = motor_config

                            # Mark that we just generated an AI design (to hide duplicate config display)
                            st.session_state.ai_design_just_generated = True

                            # Display comprehensive design summary
                            st.success("✅ Rocket design generated!")

                            # Calculate total mass
                            dry_mass = config.get("dry_mass", 0)
                            motor_mass = motor_config.get("total_mass", 0) if motor_config else 0
                            total_mass = dry_mass + motor_mass

                            # Display in expandable sections
                            with st.expander("📊 Rocket Design Summary", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Dimensions:**")
                                    st.write(
                                        f"• Total Length: {config['nose_length'] + config['body_length']:.2f} m"
                                    )
                                    st.write(
                                        f"• Body Diameter: {config['body_radius'] * 2 * 1000:.1f} mm"
                                    )
                                    st.write(f"• Nose Length: {config['nose_length']:.2f} m")
                                    st.write(f"• Body Length: {config['body_length']:.2f} m")

                                    st.markdown("**Mass:**")
                                    # Try multiple sources for dry mass
                                    dry_mass_display = config.get("dry_mass", 0)
                                    if dry_mass_display == 0 or dry_mass_display is None:
                                        # Calculate from mass breakdown if available
                                        mass_breakdown = config.get("mass_breakdown", {})
                                        if mass_breakdown and isinstance(mass_breakdown, dict):
                                            dry_mass_display = mass_breakdown.get(
                                                "total_dry_mass", 0
                                            )
                                    # If still 0, recalculate on the fly (fallback)
                                    if dry_mass_display == 0 or dry_mass_display is None:
                                        try:
                                            from ai_rocket_builder import RocketDesigner

                                            if st.session_state.get("ai_designer"):
                                                designer = st.session_state.ai_designer
                                                req = (
                                                    designer.parse_requirements(ai_input)
                                                    if "ai_input" in locals()
                                                    else None
                                                )
                                                if req:
                                                    mass_breakdown = (
                                                        designer._calculate_comprehensive_mass(
                                                            config, req
                                                        )
                                                    )
                                                    dry_mass_display = mass_breakdown.get(
                                                        "total_dry_mass", 0
                                                    )
                                                    # Update config for future use
                                                    config["dry_mass"] = dry_mass_display
                                                    config["mass_breakdown"] = mass_breakdown
                                        except:
                                            pass
                                    st.write(f"• Dry Mass: {dry_mass_display:.2f} kg")
                                    st.write(f"• Motor Mass: {motor_mass:.2f} kg")
                                    st.write(
                                        f"• Total Mass: {dry_mass_display + motor_mass:.2f} kg"
                                    )

                                with col2:
                                    st.markdown("**Performance Targets:**")
                                    if req.target_altitude_m:
                                        st.write(
                                            f"• Target Altitude: {req.target_altitude_m:.1f} m ({req.target_altitude_m * 3.28084:.0f} ft)"
                                        )
                                    if req.payload_mass_kg:
                                        st.write(f"• Payload Mass: {req.payload_mass_kg:.2f} kg")
                                    if req.payload_size:
                                        st.write(f"• Payload Size: {req.payload_size}")

                                    st.markdown("**Motor:**")
                                    if motor_config:
                                        st.write(f"• Designation: {motor_config['designation']}")
                                        st.write(f"• Manufacturer: {motor_config['manufacturer']}")
                                        st.write(
                                            f"• Total Impulse: {motor_config['total_impulse']:.0f} N·s"
                                        )
                                        st.write(
                                            f"• Avg Thrust: {motor_config['avg_thrust']:.0f} N"
                                        )
                                        st.write(f"• Burn Time: {motor_config['burn_time']:.2f} s")

                            with st.expander("🔧 Detailed Configuration"):
                                st.json(
                                    {
                                        "Nose Cone": {
                                            "Length": f"{config['nose_length']:.3f} m",
                                            "Shape": config.get("nose_shape", "VON_KARMAN"),
                                            "Material": config["nose_material"],
                                            "Thickness": f"{config['nose_thickness']:.3f} m",
                                        },
                                        "Body Tube": {
                                            "Length": f"{config['body_length']:.3f} m",
                                            "Radius": f"{config['body_radius']:.3f} m",
                                            "Diameter": f"{config['body_radius'] * 2 * 1000:.1f} mm",
                                            "Material": config["body_material"],
                                            "Thickness": f"{config['body_thickness']:.3f} m",
                                        },
                                        "Fins": {
                                            "Count": config["fin_count"],
                                            "Root Chord": f"{config['fin_root_chord']:.3f} m",
                                            "Tip Chord": f"{config['fin_tip_chord']:.3f} m",
                                            "Span": f"{config['fin_span']:.3f} m",
                                            "Sweep": f"{config['fin_sweep']:.3f} m",
                                            "Material": config["fin_material"],
                                        },
                                        "Recovery": {
                                            "Main Chute": f"{config.get('main_chute_diameter', 0):.2f} m"
                                            if config.get("has_main_chute")
                                            else "None",
                                            "Drogue": f"{config.get('drogue_diameter', 0):.2f} m"
                                            if config.get("has_drogue")
                                            else "None",
                                            "Main Deployment": config.get(
                                                "main_deployment_event", "ALTITUDE"
                                            ),
                                            "Main Deployment Altitude": f"{config.get('main_deployment_altitude', 0):.0f} m",
                                        },
                                    }
                                )

                            # Space Claims
                            if config.get("space_claims"):
                                with st.expander(
                                    "📐 Space Claims & Component Positions", expanded=False
                                ):
                                    space_claims = config["space_claims"]

                                    st.markdown("### Component Positions")
                                    positions = space_claims.get("component_positions", {})

                                    # Create a table for component positions
                                    position_data = []
                                    for comp_name, comp_info in positions.items():
                                        if comp_info.get("start") is not None:
                                            position_data.append(
                                                {
                                                    "Component": comp_name.replace(
                                                        "_", " "
                                                    ).title(),
                                                    "Start (m)": f"{comp_info.get('start', 0):.3f}",
                                                    "End (m)": f"{comp_info.get('end', 0):.3f}",
                                                    "Length (m)": f"{comp_info.get('length', comp_info.get('end', 0) - comp_info.get('start', 0)):.3f}",
                                                    "Diameter (mm)": f"{comp_info.get('diameter', 0) * 1000:.1f}"
                                                    if comp_info.get("diameter")
                                                    else "N/A",
                                                    "Clearance": comp_info.get(
                                                        "clearance_required", "N/A"
                                                    ),
                                                }
                                            )

                                    if position_data:
                                        st.dataframe(
                                            pd.DataFrame(position_data), use_container_width=True
                                        )

                                    st.markdown("### Parachute Locations")
                                    if positions.get("main_parachute"):
                                        main_chute = positions["main_parachute"]
                                        st.write(f"**Main Parachute:**")
                                        st.write(
                                            f"  - Location: {main_chute.get('location', 'N/A')}"
                                        )
                                        st.write(
                                            f"  - Diameter: {main_chute.get('diameter', 0) * 1000:.0f} mm"
                                        )
                                        st.write(
                                            f"  - Deployment: {main_chute.get('deployment_event', 'N/A')} at {main_chute.get('deployment_altitude', 0):.0f} m"
                                        )
                                        st.write(
                                            f"  - Clearance: {main_chute.get('clearance_required', 'N/A')}"
                                        )

                                    if (
                                        positions.get("drogue_parachute")
                                        and positions["drogue_parachute"].get("diameter", 0) > 0
                                    ):
                                        drogue_chute = positions["drogue_parachute"]
                                        st.write(f"**Drogue Parachute:**")
                                        st.write(
                                            f"  - Location: {drogue_chute.get('location', 'N/A')}"
                                        )
                                        st.write(
                                            f"  - Diameter: {drogue_chute.get('diameter', 0) * 1000:.0f} mm"
                                        )
                                        st.write(
                                            f"  - Deployment: {drogue_chute.get('deployment_event', 'N/A')} at {drogue_chute.get('deployment_altitude', 0):.0f} m"
                                        )
                                        st.write(
                                            f"  - Clearance: {drogue_chute.get('clearance_required', 'N/A')}"
                                        )

                                    st.markdown("### Clearances & Fit Requirements")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Clearances:**")
                                        for clearance, desc in space_claims.get(
                                            "clearances", {}
                                        ).items():
                                            st.write(
                                                f"  • {clearance.replace('_', ' ').title()}: {desc}"
                                            )
                                    with col2:
                                        st.markdown("**Fit Requirements:**")
                                        for fit, desc in space_claims.get(
                                            "fit_requirements", {}
                                        ).items():
                                            st.write(f"  • {fit.replace('_', ' ').title()}: {desc}")

                                    st.markdown("### Assembly Order")
                                    for step in space_claims.get("assembly_order", []):
                                        st.write(step)

                                    st.markdown("### Volume Claims")
                                    volumes = space_claims.get("volume_claims", {})
                                    for vol_name, vol_value in volumes.items():
                                        st.write(
                                            f"  • {vol_name.replace('_', ' ').title()}: {vol_value}"
                                        )

                            # Bill of Materials
                            if config.get("bom"):
                                with st.expander("📋 Bill of Materials (BOM)", expanded=False):
                                    bom = config["bom"]
                                    bom_df = pd.DataFrame(bom)
                                    st.dataframe(
                                        bom_df[
                                            [
                                                "part_number",
                                                "description",
                                                "quantity",
                                                "material",
                                                "dimensions",
                                                "mass_kg",
                                                "notes",
                                            ]
                                        ],
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                            # Rocket Visualization
                            st.subheader("🎨 Rocket Visualization")
                            viz_tab1, viz_tab2 = st.tabs(["3D View", "2D Side View"])

                            with viz_tab1:
                                try:
                                    fig_3d = visualize_rocket_3d(config, motor_config)
                                    st.plotly_chart(fig_3d, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error generating 3D visualization: {str(e)}")
                                    st.exception(e)

                            with viz_tab2:
                                try:
                                    fig_2d = visualize_rocket_2d_side_view(config, motor_config)
                                    st.plotly_chart(fig_2d, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error generating 2D visualization: {str(e)}")
                                    st.exception(e)

                            # Switch to custom rocket view
                            rocket_type = "Custom Rocket"

                        except Exception as e:
                            st.error(f"Error generating design: {str(e)}")
                            st.exception(e)
                else:
                    st.warning("Please enter your rocket requirements")

            st.markdown("---")

            # OpenAI API Key configuration
            with st.expander("🔑 OpenAI API Configuration"):
                st.markdown("Enter your OpenAI API key for better natural language understanding:")
                openai_key = st.text_input(
                    "OpenAI API Key",
                    value=st.session_state.get("openai_api_key", ""),
                    type="password",
                    help="Get your API key from https://platform.openai.com/api-keys",
                )
                if openai_key:
                    st.session_state.openai_api_key = openai_key
                    if st.session_state.ai_designer:
                        st.session_state.ai_designer.openai_api_key = openai_key
                        st.session_state.ai_designer.use_openai = True
                    st.success("✅ API key saved (not stored permanently)")
                elif os.getenv("OPENAI_API_KEY"):
                    st.info("Using OpenAI API key from environment variable")

            # Motor database management
            with st.expander("🔧 Motor Database"):
                st.markdown("Download motors from ThrustCurve.org API")

                # Show current database status
                if st.session_state.motor_database:
                    # Check if motors have valid impulse data
                    motors_with_impulse = [
                        m for m in st.session_state.motor_database if m.total_impulse > 0
                    ]
                    if len(motors_with_impulse) < len(st.session_state.motor_database) * 0.5:
                        st.warning(
                            f"⚠️ {len(st.session_state.motor_database)} motors loaded, but {len(st.session_state.motor_database) - len(motors_with_impulse)} have invalid impulse data. Please re-download motors to fix this."
                        )
                    else:
                        st.success(
                            f"✅ {len(st.session_state.motor_database)} motors loaded in database ({len(motors_with_impulse)} with valid impulse data)"
                        )
                else:
                    st.warning("⚠️ No motors in database. Click 'Download Motors' to get started.")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📥 Load from Cache", use_container_width=True):
                        with st.spinner("Loading motor database from cache..."):
                            try:
                                scraper = ThrustCurveScraper()
                                motor_db = scraper.load_motor_database()
                                if motor_db:
                                    st.session_state.motor_database = motor_db
                                    if st.session_state.ai_designer:
                                        st.session_state.ai_designer.motor_database = motor_db
                                    st.success(f"✅ Loaded {len(motor_db)} motors from cache")
                                else:
                                    st.info("No cached database found. Download motors first.")
                            except Exception as e:
                                st.error(f"Error loading database: {str(e)}")

                with col2:
                    if st.button(
                        "🌐 Download/Update Motors", type="primary", use_container_width=True
                    ):
                        with st.spinner(
                            "Downloading full motor database from ThrustCurve.org API (this may take several minutes)..."
                        ):
                            try:
                                scraper = ThrustCurveScraper()
                                # Clear old cache to ensure fresh data with calculated impulse
                                import shutil

                                if scraper.cache_dir.exists():
                                    # Clear individual motor caches but keep the database file
                                    for cache_file in scraper.cache_dir.glob("*.json"):
                                        if cache_file.name != "motor_database.json":
                                            cache_file.unlink()

                                # Download ALL motors - no limit, all impulse classes
                                motors = scraper.scrape_motor_list(
                                    max_motors=10000
                                )  # Get all motors
                                if motors:
                                    scraper.save_motor_database(motors)
                                    st.session_state.motor_database = motors
                                    if st.session_state.ai_designer:
                                        st.session_state.ai_designer.motor_database = motors
                                    else:
                                        # Initialize AI designer if not already done
                                        openai_key = os.getenv(
                                            "OPENAI_API_KEY"
                                        ) or st.session_state.get("openai_api_key")
                                        st.session_state.ai_designer = RocketDesigner(
                                            motors, openai_api_key=openai_key
                                        )

                                    # Count motors with valid impulse
                                    valid_motors = [m for m in motors if m.total_impulse > 0]
                                    st.success(
                                        f"✅ Downloaded {len(motors)} motors from API ({len(valid_motors)} with valid impulse data)"
                                    )
                                    st.balloons()  # Celebrate!
                                else:
                                    st.warning("No motors found. Check your internet connection.")
                            except Exception as e:
                                st.error(f"Error downloading motors: {str(e)}")
                                st.exception(e)

                # Show motor classes in database
                if st.session_state.motor_database:
                    motor_classes = {}
                    for motor in st.session_state.motor_database:
                        # Extract impulse class from designation (e.g., "M1670" -> "M")
                        if motor.designation:
                            first_char = motor.designation[0]
                            if first_char.isalpha():
                                motor_classes[first_char] = motor_classes.get(first_char, 0) + 1

                    if motor_classes:
                        st.markdown("**Motor classes in database:**")
                        classes_str = ", ".join(
                            [f"{cls}: {count}" for cls, count in sorted(motor_classes.items())]
                        )
                        st.caption(classes_str)

        # Regular rocket builder (only show if not AI Builder)
        if rocket_type != "AI Builder":
            if rocket_type == "Custom Rocket":
                st.subheader("Rocket Components")

            # Basic rocket parameters
            rocket_name = st.text_input("Rocket Name", value="Custom Rocket")

            # Nose cone
            has_nose = st.checkbox("Nose Cone", value=True)
            if has_nose:
                nose_length = st.number_input(
                    "Nose Length (m)", min_value=0.1, max_value=2.0, value=0.55829, step=0.01
                )
                nose_thickness = st.number_input(
                    "Nose Thickness (m)", min_value=0.001, max_value=0.01, value=0.003, step=0.001
                )
                nose_shape = st.selectbox(
                    "Nose Shape",
                    ["VON_KARMAN", "OGIVE", "CONICAL", "ELLIPSOID", "PARABOLIC", "HAACK"],
                    index=0,
                    help="VON_KARMAN: Most efficient (low drag), OGIVE: Classic shape, CONICAL: Simple",
                )
                nose_material = st.selectbox("Nose Material", list(MATERIALS.keys()), index=3)

            # Body tube
            body_length = st.number_input(
                "Body Length (m)", min_value=0.1, max_value=5.0, value=1.5, step=0.1
            )
            body_radius = st.number_input(
                "Body Radius (m)", min_value=0.01, max_value=0.5, value=0.0635, step=0.001
            )
            body_thickness = st.number_input(
                "Body Thickness (m)", min_value=0.001, max_value=0.01, value=0.003, step=0.001
            )
            body_material = st.selectbox("Body Material", list(MATERIALS.keys()), index=3)

            # Fins
            has_fins = st.checkbox("Fins", value=True)
            if has_fins:
                fin_count = st.number_input("Fin Count", min_value=2, max_value=8, value=4, step=1)
                fin_root_chord = st.number_input(
                    "Fin Root Chord (m)", min_value=0.01, max_value=0.5, value=0.12, step=0.01
                )
                fin_tip_chord = st.number_input(
                    "Fin Tip Chord (m)", min_value=0.01, max_value=0.5, value=0.06, step=0.01
                )
                fin_span = st.number_input(
                    "Fin Span (m)", min_value=0.01, max_value=0.5, value=0.11, step=0.01
                )
                fin_sweep = st.number_input(
                    "Fin Sweep (m)", min_value=0.0, max_value=0.5, value=0.06, step=0.01
                )
                fin_thickness = st.number_input(
                    "Fin Thickness (m)", min_value=0.001, max_value=0.01, value=0.005, step=0.001
                )
                fin_material = st.selectbox("Fin Material", list(MATERIALS.keys()), index=3)

            # Motor mount
            has_motor_mount = st.checkbox("Motor Mount", value=True)
            if has_motor_mount:
                motor_mount_length = st.number_input(
                    "Motor Mount Length (m)", min_value=0.1, max_value=2.0, value=0.5, step=0.1
                )
                motor_mount_radius = st.number_input(
                    "Motor Mount Radius (m)", min_value=0.01, max_value=0.1, value=0.041, step=0.001
                )
                motor_mount_thickness = st.number_input(
                    "Motor Mount Thickness (m)",
                    min_value=0.001,
                    max_value=0.01,
                    value=0.003,
                    step=0.001,
                )
                motor_mount_material = st.selectbox(
                    "Motor Mount Material", list(MATERIALS.keys()), index=3
                )

            # Parachutes
            st.subheader("Parachutes")
            has_main_chute = st.checkbox("Main Parachute", value=True)
            if has_main_chute:
                main_chute_diameter = st.number_input(
                    "Main Chute Diameter (m)", min_value=0.1, max_value=10.0, value=2.91, step=0.1
                )
                main_chute_cd = st.number_input(
                    "Main Chute CD", min_value=0.1, max_value=2.0, value=1.5, step=0.1
                )
                main_deployment_event = st.selectbox(
                    "Main Deployment Event", ["APOGEE", "ALTITUDE", "TIME"], index=1
                )
                if main_deployment_event == "ALTITUDE":
                    main_deployment_altitude = st.number_input(
                        "Main Deployment Altitude (m)",
                        min_value=0.0,
                        max_value=10000.0,
                        value=800.0,
                        step=10.0,
                    )
                else:
                    main_deployment_altitude = 0.0
                main_deployment_delay = st.number_input(
                    "Main Deployment Delay (s)", min_value=0.0, max_value=10.0, value=1.5, step=0.1
                )

            has_drogue = st.checkbox("Drogue Parachute", value=True)
            if has_drogue:
                drogue_diameter = st.number_input(
                    "Drogue Diameter (m)", min_value=0.1, max_value=5.0, value=0.99, step=0.1
                )
                drogue_cd = st.number_input(
                    "Drogue CD", min_value=0.1, max_value=2.0, value=1.3, step=0.1
                )
                drogue_deployment_event = st.selectbox(
                    "Drogue Deployment Event", ["APOGEE", "ALTITUDE", "TIME"], index=0
                )
                if drogue_deployment_event == "ALTITUDE":
                    drogue_deployment_altitude = st.number_input(
                        "Drogue Deployment Altitude (m)",
                        min_value=0.0,
                        max_value=10000.0,
                        value=1000.0,
                        step=10.0,
                    )
                else:
                    drogue_deployment_altitude = 0.0
                drogue_deployment_delay = st.number_input(
                    "Drogue Deployment Delay (s)",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.5,
                    step=0.1,
                )

            # Store rocket config
            rocket_config = {
                "name": rocket_name,
                "has_nose": has_nose,
                "nose_length": nose_length if has_nose else 0.0,
                "nose_thickness": nose_thickness if has_nose else 0.003,
                "nose_shape": nose_shape if has_nose else "VON_KARMAN",
                "nose_material": nose_material if has_nose else "Fiberglass",
                "body_length": body_length,
                "body_radius": body_radius,
                "body_thickness": body_thickness,
                "body_material": body_material,
                "has_fins": has_fins,
                "fin_count": fin_count if has_fins else 0,
                "fin_root_chord": fin_root_chord if has_fins else 0.12,
                "fin_tip_chord": fin_tip_chord if has_fins else 0.06,
                "fin_span": fin_span if has_fins else 0.11,
                "fin_sweep": fin_sweep if has_fins else 0.06,
                "fin_thickness": fin_thickness if has_fins else 0.005,
                "fin_material": fin_material if has_fins else "Fiberglass",
                "has_motor_mount": has_motor_mount,
                "motor_mount_length": motor_mount_length if has_motor_mount else 0.5,
                "motor_mount_radius": motor_mount_radius if has_motor_mount else 0.041,
                "motor_mount_thickness": motor_mount_thickness if has_motor_mount else 0.003,
                "motor_mount_material": motor_mount_material if has_motor_mount else "Fiberglass",
                "has_main_chute": has_main_chute,
                "main_chute_diameter": main_chute_diameter if has_main_chute else 0.0,
                "main_chute_cd": main_chute_cd if has_main_chute else 1.5,
                "main_deployment_event": main_deployment_event if has_main_chute else "ALTITUDE",
                "main_deployment_altitude": main_deployment_altitude if has_main_chute else 800.0,
                "main_deployment_delay": main_deployment_delay if has_main_chute else 1.5,
                "has_drogue": has_drogue,
                "drogue_diameter": drogue_diameter if has_drogue else 0.0,
                "drogue_cd": drogue_cd if has_drogue else 1.3,
                "drogue_deployment_event": drogue_deployment_event if has_drogue else "APOGEE",
                "drogue_deployment_altitude": drogue_deployment_altitude if has_drogue else 0.0,
                "drogue_deployment_delay": drogue_deployment_delay if has_drogue else 1.5,
            }
            st.session_state.rocket_config = rocket_config

            # Motor configuration
            st.subheader("Motor")

            # Check if we have motors in database
            if st.session_state.motor_database and len(st.session_state.motor_database) > 0:
                # Show motor selection from database
                motor_options = ["Cesaroni M1670 (Default)", "Select from Database", "Custom Motor"]
                motor_type = st.radio("Motor Type", motor_options)

                if motor_type == "Select from Database":
                    # Group motors by class for easier selection
                    motors_by_class = {}
                    for motor in st.session_state.motor_database:
                        if motor.total_impulse > 0:  # Only show motors with valid impulse
                            first_char = motor.designation[0] if motor.designation else "?"
                            if first_char.isalpha():
                                if first_char not in motors_by_class:
                                    motors_by_class[first_char] = []
                                motors_by_class[first_char].append(motor)

                    # Create selection interface
                    if motors_by_class:
                        selected_class = st.selectbox("Motor Class", sorted(motors_by_class.keys()))
                        motors_in_class = sorted(
                            motors_by_class[selected_class], key=lambda m: m.total_impulse
                        )

                        # Create display strings
                        motor_display = [
                            f"{m.designation} ({m.manufacturer}) - {m.total_impulse:.0f} N·s"
                            for m in motors_in_class
                        ]

                        selected_idx = st.selectbox(
                            "Select Motor",
                            range(len(motor_display)),
                            format_func=lambda i: motor_display[i],
                        )

                        selected_motor = motors_in_class[selected_idx]

                        # Convert to motor config format
                        motor_config = {
                            "designation": selected_motor.designation,
                            "manufacturer": selected_motor.manufacturer,
                            "total_impulse": selected_motor.total_impulse,
                            "max_thrust": selected_motor.max_thrust,
                            "avg_thrust": selected_motor.avg_thrust,
                            "burn_time": selected_motor.burn_time,
                            "diameter": selected_motor.diameter,
                            "length": selected_motor.length,
                            "total_mass": selected_motor.total_mass,
                            "propellant_mass": selected_motor.propellant_mass,
                            "case_mass": selected_motor.case_mass,
                            "thrust_curve": selected_motor.thrust_curve,
                        }
                        st.session_state.motor_config = motor_config

                        # Automatically update motor mount dimensions based on selected motor
                        if st.session_state.rocket_config:
                            motor_diameter = selected_motor.diameter
                            motor_length = selected_motor.length

                            # Update motor mount to fit the motor
                            st.session_state.rocket_config["motor_mount_radius"] = (
                                motor_diameter / 2.0
                            ) + 0.005  # 5mm clearance
                            st.session_state.rocket_config["motor_mount_length"] = (
                                motor_length + 0.1
                            )  # 10cm extra

                            st.success(
                                f"✅ Motor mount automatically updated: "
                                f"Radius: {st.session_state.rocket_config['motor_mount_radius'] * 1000:.1f} mm, "
                                f"Length: {st.session_state.rocket_config['motor_mount_length'] * 1000:.0f} mm"
                            )

                        # Display motor info
                        st.info(
                            f"**Selected:** {selected_motor.designation} | "
                            f"Impulse: {selected_motor.total_impulse:.0f} N·s | "
                            f"Avg Thrust: {selected_motor.avg_thrust:.0f} N | "
                            f"Burn Time: {selected_motor.burn_time:.2f} s"
                        )
                    else:
                        st.warning("No motors with valid impulse data in database")
                        motor_type = "Cesaroni M1670 (Default)"
            else:
                motor_type = st.radio("Motor Type", ["Cesaroni M1670 (Default)", "Custom Motor"])

            if motor_type == "Custom Motor":
                motor_name = st.text_input("Motor Name", value="Custom Motor")
                motor_manufacturer = st.text_input("Manufacturer", value="Custom")
                motor_diameter = st.number_input(
                    "Motor Diameter (m)", min_value=0.01, max_value=0.5, value=0.075, step=0.001
                )
                motor_length = st.number_input(
                    "Motor Length (m)", min_value=0.1, max_value=2.0, value=0.64, step=0.01
                )
                motor_total_mass = st.number_input(
                    "Total Mass (kg)", min_value=0.1, max_value=50.0, value=4.771, step=0.1
                )
                motor_propellant_mass = st.number_input(
                    "Propellant Mass (kg)", min_value=0.1, max_value=50.0, value=2.956, step=0.1
                )
                max_thrust = st.number_input(
                    "Max Thrust (N)", min_value=100.0, max_value=10000.0, value=2200.0, step=100.0
                )
                avg_thrust = st.number_input(
                    "Avg Thrust (N)", min_value=100.0, max_value=10000.0, value=1545.0, step=100.0
                )
                burn_time = st.number_input(
                    "Burn Time (s)", min_value=0.1, max_value=30.0, value=3.9, step=0.1
                )

                motor_config = {
                    "motor_name": motor_name,
                    "motor_manufacturer": motor_manufacturer,
                    "motor_diameter": motor_diameter,
                    "motor_length": motor_length,
                    "diameter": motor_diameter,  # Also add for compatibility
                    "length": motor_length,  # Also add for compatibility
                    "motor_total_mass": motor_total_mass,
                    "total_mass": motor_total_mass,  # Also add for compatibility
                    "motor_propellant_mass": motor_propellant_mass,
                    "propellant_mass": motor_propellant_mass,  # Also add for compatibility
                    "max_thrust": max_thrust,
                    "avg_thrust": avg_thrust,
                    "burn_time": burn_time,
                    "motor_cg_position": 0.317,
                    "motor_propellant_cg": 0.397,
                    "motor_inertia_axial": 0.002,
                    "motor_inertia_lateral": 0.125,
                }
                st.session_state.motor_config = motor_config

                # Automatically update motor mount dimensions based on custom motor
                if st.session_state.rocket_config:
                    # Update motor mount to fit the motor
                    st.session_state.rocket_config["motor_mount_radius"] = (
                        motor_diameter / 2.0
                    ) + 0.005  # 5mm clearance
                    st.session_state.rocket_config["motor_mount_length"] = (
                        motor_length + 0.1
                    )  # 10cm extra

                    st.success(
                        f"✅ Motor mount automatically updated: "
                        f"Radius: {st.session_state.rocket_config['motor_mount_radius'] * 1000:.1f} mm, "
                        f"Length: {st.session_state.rocket_config['motor_mount_length'] * 1000:.0f} mm"
                    )
            elif motor_type == "Cesaroni M1670 (Default)":
                # Default motor - clear any custom config
                st.session_state.motor_config = None

            # Environment configuration
            st.subheader("Environment")
            elevation = st.number_input(
                "Launch Site Elevation (m)",
                min_value=0.0,
                max_value=5000.0,
                value=1400.0,
                step=10.0,
            )

            # Launch conditions
            st.subheader("Launch Conditions")
            rail_length = st.number_input(
                "Rail Length (m)", min_value=0.5, max_value=20.0, value=5.2, step=0.1
            )
            inclination_deg = st.number_input(
                "Launch Angle (deg from vertical)",
                min_value=0.0,
                max_value=90.0,
                value=5.0,
                step=1.0,
            )
            heading_deg = st.number_input(
                "Heading (deg, 0=North)", min_value=0.0, max_value=360.0, value=0.0, step=1.0
            )

            # Simulation parameters
            st.subheader("Simulation")
            max_time = st.number_input(
                "Max Simulation Time (s)", min_value=10.0, max_value=1000.0, value=200.0, step=10.0
            )
            dt = st.number_input(
                "Time Step (s)",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                step=0.001,
                format="%.3f",
            )
        else:
            # AI Builder mode - use defaults or AI-generated values
            elevation = st.number_input(
                "Launch Site Elevation (m)",
                min_value=0.0,
                max_value=5000.0,
                value=1400.0,
                step=10.0,
            )
            rail_length = st.number_input(
                "Rail Length (m)", min_value=0.5, max_value=20.0, value=5.2, step=0.1
            )
            inclination_deg = st.number_input(
                "Launch Angle (deg from vertical)",
                min_value=0.0,
                max_value=90.0,
                value=5.0,
                step=1.0,
            )
            heading_deg = st.number_input(
                "Heading (deg, 0=North)", min_value=0.0, max_value=360.0, value=0.0, step=1.0
            )
            max_time = st.number_input(
                "Max Simulation Time (s)", min_value=10.0, max_value=1000.0, value=200.0, step=10.0
            )
            dt = st.number_input(
                "Time Step (s)",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                step=0.001,
                format="%.3f",
            )

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("Simulation Control")

        # Show current rocket configuration if available (but not if we're in AI Builder mode and just generated)
        # Only show if it's a custom rocket or if we haven't just generated an AI design
        show_config = st.session_state.rocket_config and (
            rocket_type != "AI Builder"
            or not st.session_state.get("ai_design_just_generated", False)
        )

        if show_config:
            with st.expander("📋 Current Rocket Configuration", expanded=False):
                config = st.session_state.rocket_config
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Dimensions:**")
                    st.write(
                        f"• Length: {config.get('nose_length', 0) + config.get('body_length', 0):.2f} m"
                    )
                    st.write(f"• Diameter: {config.get('body_radius', 0) * 2 * 1000:.1f} mm")
                    st.write(f"• Nose Shape: {config.get('nose_shape', 'VON_KARMAN')}")
                    st.write(f"• Fin Count: {config.get('fin_count', 0)}")
                with col_b:
                    st.markdown("**Materials:**")
                    st.write(f"• Body: {config.get('body_material', 'N/A')}")
                    st.write(f"• Nose: {config.get('nose_material', 'N/A')}")
                    st.write(f"• Fins: {config.get('fin_material', 'N/A')}")
                if st.session_state.motor_config:
                    motor = st.session_state.motor_config
                    st.markdown("**Motor:**")
                    st.write(
                        f"• {motor.get('designation', 'N/A')} ({motor.get('manufacturer', 'N/A')})"
                    )
                    st.write(f"• Impulse: {motor.get('total_impulse', 0):.0f} N·s")

                # Add visualization
                st.markdown("---")
                st.markdown("**🎨 Rocket Visualization:**")
                viz_tab1, viz_tab2 = st.tabs(["3D View", "2D Side View"])

                with viz_tab1:
                    try:
                        motor_config = (
                            st.session_state.motor_config if st.session_state.motor_config else None
                        )
                        fig_3d = visualize_rocket_3d(config, motor_config)
                        st.plotly_chart(fig_3d, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate 3D visualization: {str(e)}")

                with viz_tab2:
                    try:
                        motor_config = (
                            st.session_state.motor_config if st.session_state.motor_config else None
                        )
                        fig_2d = visualize_rocket_2d_side_view(config, motor_config)
                        st.plotly_chart(fig_2d, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate 2D visualization: {str(e)}")

        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Building rocket and running simulation..."):
                try:
                    # Build rocket
                    if rocket_type == "Calisto (Default)":
                        rocket_raw, motor_raw = build_calisto()
                    elif rocket_type == "AI Builder":
                        # Use AI-generated config if available
                        if st.session_state.rocket_config is None:
                            st.error("Please generate a rocket design first using the AI Builder")
                            st.stop()
                        rocket_raw = build_custom_rocket(st.session_state.rocket_config)
                        # Use AI-selected motor or default
                        if st.session_state.motor_config:
                            motor_raw = build_custom_motor(st.session_state.motor_config)
                        else:
                            _, motor_raw = build_calisto()
                    else:
                        # Custom Rocket mode
                        if st.session_state.rocket_config is None:
                            st.error("Please configure your custom rocket in the sidebar")
                            st.stop()
                        rocket_raw = build_custom_rocket(st.session_state.rocket_config)
                        # Use motor from session state (set in sidebar)
                        if st.session_state.motor_config is None:
                            # Default motor
                            _, motor_raw = build_calisto()
                        else:
                            motor_raw = build_custom_motor(st.session_state.motor_config)

                    rocket = RocketModel(rocket_raw)
                    motor = Motor.from_openrocket(motor_raw)

                    # Create environment
                    env = Environment(elevation=elevation)

                    # Run simulation
                    solver = FlightSolver(
                        rocket=rocket,
                        motor=motor,
                        environment=env,
                        rail_length=rail_length,
                        inclination_deg=inclination_deg,
                        heading_deg=heading_deg,
                        dt=dt,
                    )

                    result = solver.run(max_time=max_time)

                    if len(result.history) == 0:
                        st.error("Simulation failed - no history recorded")
                    else:
                        st.session_state.simulation_result = result
                        st.session_state.solver = solver
                        st.success("Simulation completed successfully!")

                        # Display quick stats
                        max_alt = max(s.z for s in result.history)
                        max_v = max(np.linalg.norm(s.velocity) for s in result.history)
                        st.metric("Max Altitude", f"{max_alt:.1f} m")
                        st.metric("Max Velocity", f"{max_v:.1f} m/s")

                except Exception as e:
                    st.error(f"Simulation error: {str(e)}")
                    st.exception(e)

    with col2:
        st.header("Actions")
        if st.session_state.simulation_result is not None:
            if st.button("📊 View Results", use_container_width=True):
                st.session_state.show_results = True
            if st.button("🎮 Launch Elodin Editor", use_container_width=True):
                try:
                    if "solver" in st.session_state and st.session_state.solver is not None:
                        launch_elodin_editor(
                            st.session_state.simulation_result, st.session_state.solver
                        )
                    else:
                        st.error("❌ Solver not available. Please run a simulation first.")
                        st.info("💡 Click '🚀 Run Simulation' to generate simulation data.")
                except Exception as e:
                    st.error(f"❌ Error launching Elodin editor: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

            # Add manual launch instructions
            if st.session_state.simulation_result is not None:
                with st.expander("🔧 Manual Launch Instructions", expanded=False):
                    st.markdown("""
                    **If the button doesn't work, try these options:**
                    
                    **Option 1: Use the wrapper script (recommended):**
                    ```bash
                    cd /home/kush-mahajan/elodin/examples/rocket-barrowman
                    bash launch_elodin_wrapper.sh
                    ```
                    
                    **Option 2: Run directly:**
                    ```bash
                    cd /home/kush-mahajan/elodin/examples/rocket-barrowman
                    python3 main.py --visualize
                    ```
                    
                    **Option 3: Use the test script:**
                    ```bash
                    cd /home/kush-mahajan/elodin/examples/rocket-barrowman
                    python3 test_elodin_launch.py
                    ```
                    
                    **Make sure you're in the nix shell first:**
                    ```bash
                    nix develop
                    ```
                    """)

    # Display results
    if st.session_state.simulation_result is not None:
        st.header("Simulation Results")
        visualize_results(st.session_state.simulation_result)

        # Data export
        st.subheader("Export Data")
        if st.button("📥 Download CSV"):
            history = st.session_state.simulation_result.history
            df = pd.DataFrame(
                {
                    "time": [s.time for s in history],
                    "x": [s.x for s in history],
                    "y": [s.y for s in history],
                    "z": [s.z for s in history],
                    "vx": [s.vx for s in history],
                    "vy": [s.vy for s in history],
                    "vz": [s.vz for s in history],
                    "velocity": [np.linalg.norm(s.velocity) for s in history],
                    "mach": [s.mach for s in history],
                    "angle_of_attack": [s.angle_of_attack for s in history],
                    "dynamic_pressure": [s.dynamic_pressure for s in history],
                }
            )
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV", data=csv, file_name="rocket_simulation.csv", mime="text/csv"
            )


if __name__ == "__main__":
    main()
