"""
Advanced Rocket Renderer - High-quality 3D visualization.

Creates photorealistic-style rocket renders using Plotly with:
- Metallic materials and lighting
- Proper surface meshes
- Dimension callouts
- Professional color schemes
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple
import math


# Color themes
THEMES = {
    "aerospace": {
        "body": "rgb(240, 240, 245)",  # White/silver
        "nose": "rgb(200, 50, 50)",  # Red nose
        "fins": "rgb(50, 50, 60)",  # Dark gray fins
        "motor": "rgb(255, 140, 0)",  # Orange motor
        "mount": "rgb(80, 80, 90)",  # Gray mount
        "chute_main": "rgb(255, 200, 0)",  # Yellow
        "chute_drogue": "rgb(255, 100, 50)",  # Orange
        "background": "rgb(10, 15, 25)",
    },
    "stealth": {
        "body": "rgb(40, 45, 50)",
        "nose": "rgb(30, 35, 40)",
        "fins": "rgb(25, 30, 35)",
        "motor": "rgb(80, 60, 40)",
        "mount": "rgb(50, 55, 60)",
        "chute_main": "rgb(60, 60, 70)",
        "chute_drogue": "rgb(50, 50, 60)",
        "background": "rgb(5, 8, 12)",
    },
    "racing": {
        "body": "rgb(255, 50, 50)",
        "nose": "rgb(30, 30, 35)",
        "fins": "rgb(255, 200, 0)",
        "motor": "rgb(200, 200, 210)",
        "mount": "rgb(60, 60, 65)",
        "chute_main": "rgb(255, 255, 255)",
        "chute_drogue": "rgb(200, 200, 200)",
        "background": "rgb(15, 18, 25)",
    },
    "blueprint": {
        "body": "rgba(100, 150, 255, 0.3)",
        "nose": "rgba(100, 150, 255, 0.4)",
        "fins": "rgba(100, 150, 255, 0.5)",
        "motor": "rgba(255, 200, 100, 0.5)",
        "mount": "rgba(150, 150, 200, 0.4)",
        "chute_main": "rgba(255, 255, 100, 0.4)",
        "chute_drogue": "rgba(255, 200, 100, 0.4)",
        "background": "rgb(20, 30, 50)",
    },
}


def create_cylinder_mesh(
    x_start: float,
    length: float,
    radius: float,
    n_theta: int = 32,
    n_length: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create mesh for a cylinder with proper triangulation."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    x = np.linspace(x_start, x_start + length, n_length)

    theta_grid, x_grid = np.meshgrid(theta, x)
    y_grid = radius * np.cos(theta_grid)
    z_grid = radius * np.sin(theta_grid)

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()

    # Create triangulation
    i_list, j_list, k_list = [], [], []
    for row in range(n_length - 1):
        for col in range(n_theta - 1):
            idx = row * n_theta + col
            # Two triangles per quad
            i_list.extend([idx, idx])
            j_list.extend([idx + 1, idx + n_theta])
            k_list.extend([idx + n_theta, idx + n_theta + 1])

    return x_flat, y_flat, z_flat, np.array(i_list), np.array(j_list), np.array(k_list)


def create_nose_cone_mesh(
    length: float,
    base_radius: float,
    shape: str = "VON_KARMAN",
    n_theta: int = 32,
    n_length: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create mesh for nose cone with proper shape."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    x = np.linspace(0, length, n_length)

    # Calculate radius at each x position
    radii = []
    for xi in x:
        t = xi / length  # Normalized position (0 at tip, 1 at base)
        if shape == "CONICAL":
            r = base_radius * t
        elif shape == "OGIVE":
            rho = (base_radius**2 + length**2) / (2 * base_radius)
            if rho**2 >= (length - xi) ** 2:
                r = math.sqrt(rho**2 - (length - xi) ** 2) - (rho - base_radius)
            else:
                r = base_radius * t
        elif shape == "VON_KARMAN":
            theta_vk = math.acos(1 - 2 * t) if t <= 1 else math.pi
            r = base_radius * math.sqrt(
                (theta_vk - math.sin(theta_vk) * math.cos(theta_vk)) / math.pi
            )
        elif shape == "HAACK":
            theta_h = math.acos(1 - 2 * t) if t <= 1 else math.pi
            r = base_radius * math.sqrt((theta_h - math.sin(2 * theta_h) / 2) / math.pi)
        else:  # PARABOLIC
            r = base_radius * math.sqrt(t) if t > 0 else 0
        radii.append(max(r, 0.001))  # Prevent zero radius

    radii = np.array(radii)

    theta_grid, x_grid = np.meshgrid(theta, x)
    radii_grid = np.tile(radii.reshape(-1, 1), (1, n_theta))
    y_grid = radii_grid * np.cos(theta_grid)
    z_grid = radii_grid * np.sin(theta_grid)

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()

    # Triangulation
    i_list, j_list, k_list = [], [], []
    for row in range(n_length - 1):
        for col in range(n_theta - 1):
            idx = row * n_theta + col
            i_list.extend([idx, idx])
            j_list.extend([idx + 1, idx + n_theta])
            k_list.extend([idx + n_theta, idx + n_theta + 1])

    return x_flat, y_flat, z_flat, np.array(i_list), np.array(j_list), np.array(k_list)


def create_fin_mesh(
    start_x: float,
    root_chord: float,
    tip_chord: float,
    span: float,
    sweep: float,
    body_radius: float,
    angle: float,
    thickness: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create mesh for a trapezoidal fin with thickness."""
    # Fin vertices (local coords, before rotation)
    # Top surface
    vertices = [
        # Root leading edge (inner)
        [start_x, body_radius, -thickness / 2],
        [start_x, body_radius, thickness / 2],
        # Root trailing edge (inner)
        [start_x + root_chord, body_radius, -thickness / 2],
        [start_x + root_chord, body_radius, thickness / 2],
        # Tip leading edge (outer)
        [start_x + sweep, body_radius + span, -thickness / 2],
        [start_x + sweep, body_radius + span, thickness / 2],
        # Tip trailing edge (outer)
        [start_x + sweep + tip_chord, body_radius + span, -thickness / 2],
        [start_x + sweep + tip_chord, body_radius + span, thickness / 2],
    ]

    # Rotate around x-axis by angle
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated = []
    for v in vertices:
        y_rot = v[1] * cos_a - v[2] * sin_a
        z_rot = v[1] * sin_a + v[2] * cos_a
        rotated.append([v[0], y_rot, z_rot])

    vertices = np.array(rotated)

    # Define faces (triangles)
    # Front face (leading edge side)
    faces = [
        [0, 4, 5],
        [0, 5, 1],  # Leading edge
        [2, 3, 7],
        [2, 7, 6],  # Trailing edge
        [0, 2, 6],
        [0, 6, 4],  # Bottom
        [1, 5, 7],
        [1, 7, 3],  # Top
        [0, 1, 3],
        [0, 3, 2],  # Root
        [4, 6, 7],
        [4, 7, 5],  # Tip
    ]

    i_list = [f[0] for f in faces]
    j_list = [f[1] for f in faces]
    k_list = [f[2] for f in faces]

    return (
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        np.array(i_list),
        np.array(j_list),
        np.array(k_list),
    )


def render_rocket_3d(
    config: Dict[str, Any],
    motor: Optional[Dict[str, Any]] = None,
    theme: str = "aerospace",
    show_dimensions: bool = True,
    show_internals: bool = False,
) -> go.Figure:
    """
    Create high-quality 3D rocket render.

    Args:
        config: Rocket configuration dictionary
        motor: Optional motor configuration
        theme: Color theme ("aerospace", "stealth", "racing", "blueprint")
        show_dimensions: Whether to show dimension callouts
        show_internals: Whether to show internal components

    Returns:
        Plotly figure with rendered rocket
    """
    colors = THEMES.get(theme, THEMES["aerospace"])

    # Extract dimensions
    nose_length = config.get("nose_length", 0.5)
    body_length = config.get("body_length", 1.5)
    body_radius = config.get("body_radius", 0.0635)
    nose_shape = config.get("nose_shape", "VON_KARMAN")

    fin_count = config.get("fin_count", 4)
    fin_root = config.get("fin_root_chord", 0.12)
    fin_tip = config.get("fin_tip_chord", 0.06)
    fin_span = config.get("fin_span", 0.11)
    fin_sweep = config.get("fin_sweep", 0.06)
    fin_thickness = config.get("fin_thickness", 0.005)

    motor_length = motor.get("length", 0.64) if motor else 0.64
    motor_diameter = motor.get("diameter", 0.075) if motor else 0.075
    motor_radius = motor_diameter / 2.0

    total_length = nose_length + body_length

    fig = go.Figure()

    # ═══════════════════════════════════════════════════════════════════
    # NOSE CONE
    # ═══════════════════════════════════════════════════════════════════
    x, y, z, i, j, k = create_nose_cone_mesh(nose_length, body_radius, nose_shape)
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color=colors["nose"],
            opacity=0.95,
            name="Nose Cone",
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.5,
                roughness=0.3,
                fresnel=0.2,
            ),
            lightposition=dict(x=100, y=200, z=300),
            showlegend=True,
        )
    )

    # ═══════════════════════════════════════════════════════════════════
    # BODY TUBE
    # ═══════════════════════════════════════════════════════════════════
    x, y, z, i, j, k = create_cylinder_mesh(nose_length, body_length, body_radius)
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color=colors["body"],
            opacity=0.9,
            name="Body Tube",
            lighting=dict(
                ambient=0.4,
                diffuse=0.7,
                specular=0.6,
                roughness=0.2,
                fresnel=0.3,
            ),
            lightposition=dict(x=100, y=200, z=300),
            showlegend=True,
        )
    )

    # ═══════════════════════════════════════════════════════════════════
    # FINS
    # ═══════════════════════════════════════════════════════════════════
    fin_start_x = nose_length + body_length - fin_root
    for fi in range(fin_count):
        angle = 2 * math.pi * fi / fin_count
        x, y, z, i, j, k = create_fin_mesh(
            fin_start_x, fin_root, fin_tip, fin_span, fin_sweep, body_radius, angle, fin_thickness
        )
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=colors["fins"],
                opacity=0.95,
                name="Fins" if fi == 0 else None,
                showlegend=(fi == 0),
                lighting=dict(
                    ambient=0.3,
                    diffuse=0.6,
                    specular=0.8,
                    roughness=0.1,
                ),
            )
        )

    # ═══════════════════════════════════════════════════════════════════
    # MOTOR (visible at rear)
    # ═══════════════════════════════════════════════════════════════════
    if motor or show_internals:
        motor_start = total_length - motor_length - 0.02
        x, y, z, i, j, k = create_cylinder_mesh(
            motor_start, motor_length, motor_radius * 0.95, n_theta=24
        )
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=colors["motor"],
                opacity=0.9,
                name="Motor",
                showlegend=True,
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.8,
                    specular=0.3,
                ),
            )
        )

        # Motor nozzle (cone at rear)
        nozzle_length = 0.03
        nozzle_start = total_length - 0.01
        theta = np.linspace(0, 2 * np.pi, 16)
        x_nozzle = [nozzle_start, nozzle_start + nozzle_length]
        r_nozzle = [motor_radius * 0.5, motor_radius * 0.3]

        for xi, ri in zip(x_nozzle, r_nozzle):
            y_ring = ri * np.cos(theta)
            z_ring = ri * np.sin(theta)
            fig.add_trace(
                go.Scatter3d(
                    x=[xi] * len(theta),
                    y=y_ring.tolist(),
                    z=z_ring.tolist(),
                    mode="lines",
                    line=dict(color="rgb(50, 50, 55)", width=3),
                    showlegend=False,
                )
            )

    # ═══════════════════════════════════════════════════════════════════
    # MOTOR MOUNT (if showing internals)
    # ═══════════════════════════════════════════════════════════════════
    if show_internals:
        mount_radius = motor_radius + 0.008
        mount_length = motor_length + 0.1
        mount_start = total_length - mount_length
        x, y, z, i, j, k = create_cylinder_mesh(mount_start, mount_length, mount_radius, n_theta=20)
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=colors["mount"],
                opacity=0.5,
                name="Motor Mount",
                showlegend=True,
            )
        )

    # ═══════════════════════════════════════════════════════════════════
    # DIMENSION CALLOUTS
    # ═══════════════════════════════════════════════════════════════════
    if show_dimensions:
        offset = body_radius * 2

        # Total length
        fig.add_trace(
            go.Scatter3d(
                x=[0, total_length],
                y=[offset, offset],
                z=[0, 0],
                mode="lines+text",
                line=dict(color="rgba(255, 255, 255, 0.6)", width=2, dash="dot"),
                text=["", f"{total_length * 1000:.0f}mm"],
                textposition="middle right",
                textfont=dict(color="white", size=12),
                showlegend=False,
            )
        )

        # Diameter
        fig.add_trace(
            go.Scatter3d(
                x=[nose_length + body_length / 2, nose_length + body_length / 2],
                y=[body_radius, -body_radius],
                z=[offset * 0.5, offset * 0.5],
                mode="lines+text",
                line=dict(color="rgba(255, 255, 255, 0.6)", width=2, dash="dot"),
                text=["", f"⌀{body_radius * 2 * 1000:.0f}mm"],
                textposition="bottom center",
                textfont=dict(color="white", size=11),
                showlegend=False,
            )
        )

        # Nose length
        fig.add_trace(
            go.Scatter3d(
                x=[0, nose_length],
                y=[-offset * 0.7, -offset * 0.7],
                z=[0, 0],
                mode="lines+text",
                line=dict(color="rgba(200, 100, 100, 0.6)", width=2),
                text=["", f"Nose: {nose_length * 1000:.0f}mm"],
                textposition="middle right",
                textfont=dict(color="rgb(255, 150, 150)", size=10),
                showlegend=False,
            )
        )

        # Body length
        fig.add_trace(
            go.Scatter3d(
                x=[nose_length, total_length],
                y=[-offset * 0.7, -offset * 0.7],
                z=[0, 0],
                mode="lines+text",
                line=dict(color="rgba(100, 150, 255, 0.6)", width=2),
                text=["", f"Body: {body_length * 1000:.0f}mm"],
                textposition="middle right",
                textfont=dict(color="rgb(150, 180, 255)", size=10),
                showlegend=False,
            )
        )

    # ═══════════════════════════════════════════════════════════════════
    # LAYOUT
    # ═══════════════════════════════════════════════════════════════════
    name = config.get("name", "Rocket")
    motor_name = motor.get("designation", "") if motor else ""
    title = f"{name}"
    if motor_name:
        title += f" | Motor: {motor_name}"

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=20, color="white", family="Arial Black"),
        ),
        scene=dict(
            xaxis=dict(
                title="Length (m)",
                showbackground=True,
                backgroundcolor=colors["background"],
                gridcolor="rgba(100, 100, 120, 0.3)",
                showspikes=False,
                color="rgba(200, 200, 220, 0.8)",
            ),
            yaxis=dict(
                title="",
                showbackground=True,
                backgroundcolor=colors["background"],
                gridcolor="rgba(100, 100, 120, 0.3)",
                showspikes=False,
                color="rgba(200, 200, 220, 0.8)",
            ),
            zaxis=dict(
                title="",
                showbackground=True,
                backgroundcolor=colors["background"],
                gridcolor="rgba(100, 100, 120, 0.3)",
                showspikes=False,
                color="rgba(200, 200, 220, 0.8)",
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.8),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
            bgcolor=colors["background"],
        ),
        paper_bgcolor=colors["background"],
        plot_bgcolor=colors["background"],
        font=dict(color="white", family="Arial"),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(30, 40, 60, 0.8)",
            bordercolor="rgba(100, 120, 150, 0.5)",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        width=900,
        height=650,
    )

    return fig


def render_rocket_profile(
    config: Dict[str, Any],
    motor: Optional[Dict[str, Any]] = None,
    show_dimensions: bool = True,
) -> go.Figure:
    """
    Create 2D side profile view of rocket.
    """
    nose_length = config.get("nose_length", 0.5)
    body_length = config.get("body_length", 1.5)
    body_radius = config.get("body_radius", 0.0635)
    nose_shape = config.get("nose_shape", "VON_KARMAN")

    fin_root = config.get("fin_root_chord", 0.12)
    fin_tip = config.get("fin_tip_chord", 0.06)
    fin_span = config.get("fin_span", 0.11)
    fin_sweep = config.get("fin_sweep", 0.06)

    motor_length = motor.get("length", 0.64) if motor else 0.64
    motor_radius = (motor.get("diameter", 0.075) if motor else 0.075) / 2

    total_length = nose_length + body_length
    fin_start = nose_length + body_length - fin_root

    fig = go.Figure()

    # Nose cone profile
    n = 100
    x_nose = np.linspace(0, nose_length, n)
    r_nose = []
    for xi in x_nose:
        t = xi / nose_length
        if nose_shape == "VON_KARMAN":
            theta = math.acos(1 - 2 * t) if t <= 1 else math.pi
            r = body_radius * math.sqrt((theta - math.sin(theta) * math.cos(theta)) / math.pi)
        else:
            r = body_radius * t
        r_nose.append(r)
    r_nose = np.array(r_nose)

    # Upper profile
    x_upper = np.concatenate([x_nose, [nose_length, total_length, total_length]])
    y_upper = np.concatenate([r_nose, [body_radius, body_radius, 0]])

    # Lower profile (mirror)
    x_lower = np.concatenate([[total_length, total_length, nose_length], x_nose[::-1]])
    y_lower = np.concatenate([[0, -body_radius, -body_radius], -r_nose[::-1]])

    # Complete outline
    x_outline = np.concatenate([x_upper, x_lower])
    y_outline = np.concatenate([y_upper, y_lower])

    # Body fill
    fig.add_trace(
        go.Scatter(
            x=x_outline.tolist() + [x_outline[0]],
            y=y_outline.tolist() + [y_outline[0]],
            fill="toself",
            fillcolor="rgba(200, 210, 230, 0.5)",
            line=dict(color="rgb(100, 120, 180)", width=2),
            name="Body",
        )
    )

    # Fin (upper)
    fig.add_trace(
        go.Scatter(
            x=[
                fin_start,
                fin_start + fin_root,
                fin_start + fin_sweep + fin_tip,
                fin_start + fin_sweep,
                fin_start,
            ],
            y=[
                body_radius,
                body_radius,
                body_radius + fin_span,
                body_radius + fin_span,
                body_radius,
            ],
            fill="toself",
            fillcolor="rgba(80, 90, 100, 0.8)",
            line=dict(color="rgb(50, 60, 70)", width=2),
            name="Fins",
        )
    )

    # Fin (lower, mirrored)
    fig.add_trace(
        go.Scatter(
            x=[
                fin_start,
                fin_start + fin_root,
                fin_start + fin_sweep + fin_tip,
                fin_start + fin_sweep,
                fin_start,
            ],
            y=[
                -body_radius,
                -body_radius,
                -body_radius - fin_span,
                -body_radius - fin_span,
                -body_radius,
            ],
            fill="toself",
            fillcolor="rgba(80, 90, 100, 0.8)",
            line=dict(color="rgb(50, 60, 70)", width=2),
            showlegend=False,
        )
    )

    # Motor
    motor_start = total_length - motor_length - 0.02
    fig.add_trace(
        go.Scatter(
            x=[
                motor_start,
                motor_start + motor_length,
                motor_start + motor_length,
                motor_start,
                motor_start,
            ],
            y=[motor_radius, motor_radius, -motor_radius, -motor_radius, motor_radius],
            fill="toself",
            fillcolor="rgba(255, 150, 50, 0.8)",
            line=dict(color="rgb(200, 100, 0)", width=2),
            name="Motor",
        )
    )

    # Dimensions
    if show_dimensions:
        # Total length
        dim_y = (body_radius + fin_span) * 1.3
        fig.add_trace(
            go.Scatter(
                x=[0, total_length],
                y=[dim_y, dim_y],
                mode="lines+text",
                line=dict(color="rgba(100, 150, 255, 0.8)", width=1, dash="dot"),
                text=["", f"{total_length * 1000:.0f}mm"],
                textposition="top center",
                textfont=dict(size=12, color="rgb(100, 150, 255)"),
                showlegend=False,
            )
        )

        # Add arrows
        arrow_size = 0.02
        fig.add_trace(
            go.Scatter(
                x=[0, arrow_size, 0],
                y=[dim_y, dim_y, dim_y],
                mode="lines",
                line=dict(color="rgba(100, 150, 255, 0.8)", width=1),
                showlegend=False,
            )
        )

    max_extent = max(body_radius + fin_span, total_length * 0.1)

    fig.update_layout(
        title=dict(
            text=f"Profile View - {config.get('name', 'Rocket')}",
            x=0.5,
            font=dict(size=18, color="white"),
        ),
        xaxis=dict(
            title="Length (m)",
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor="rgba(100, 100, 120, 0.2)",
            color="rgba(200, 200, 220, 0.8)",
        ),
        yaxis=dict(
            title="Radius (m)",
            showgrid=True,
            gridcolor="rgba(100, 100, 120, 0.2)",
            color="rgba(200, 200, 220, 0.8)",
            range=[-max_extent * 1.2, max_extent * 1.5],
        ),
        paper_bgcolor="rgb(20, 25, 35)",
        plot_bgcolor="rgb(15, 20, 30)",
        font=dict(color="white"),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(30, 40, 60, 0.8)",
        ),
        width=900,
        height=400,
    )

    return fig


if __name__ == "__main__":
    # Test render
    config = {
        "name": "Test Rocket",
        "nose_length": 0.4,
        "nose_shape": "VON_KARMAN",
        "body_length": 1.2,
        "body_radius": 0.05,
        "fin_count": 4,
        "fin_root_chord": 0.1,
        "fin_tip_chord": 0.05,
        "fin_span": 0.08,
        "fin_sweep": 0.04,
        "fin_thickness": 0.004,
    }

    motor = {
        "designation": "J350",
        "diameter": 0.054,
        "length": 0.35,
    }

    fig = render_rocket_3d(config, motor, theme="aerospace", show_dimensions=True)
    fig.show()
