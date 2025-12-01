#!/usr/bin/env python3
"""
Rocket Visualizer - 3D visualization of rocket designs
Uses Plotly for interactive 3D visualization
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import math


def visualize_rocket_3d(config: Dict[str, Any], motor: Optional[Dict] = None) -> go.Figure:
    """
    Create a 3D visualization of the rocket design.
    
    Args:
        config: Rocket configuration dictionary
        motor: Optional motor configuration dictionary
        
    Returns:
        Plotly figure object
    """
    # Extract dimensions
    nose_length = config.get('nose_length', 0.5)
    body_length = config.get('body_length', 1.5)
    body_radius = config.get('body_radius', 0.0635)
    fin_count = config.get('fin_count', 4)
    fin_root_chord = config.get('fin_root_chord', 0.12)
    fin_tip_chord = config.get('fin_tip_chord', 0.06)
    fin_span = config.get('fin_span', 0.11)
    fin_sweep = config.get('fin_sweep', 0.06)
    fin_thickness = config.get('fin_thickness', 0.005)
    
    # Motor dimensions
    motor_length = motor.get('length', 0.64) if motor else 0.64
    motor_diameter = motor.get('diameter', 0.075) if motor else 0.075
    motor_radius = motor_diameter / 2.0
    
    # Calculate positions
    total_length = nose_length + body_length
    fin_start_x = nose_length + body_length - fin_root_chord
    
    # Create figure
    fig = go.Figure()
    
    # 1. Nose Cone
    nose_shape = config.get('nose_shape', 'VON_KARMAN')
    nose_points = _generate_nose_cone_points(nose_length, body_radius, nose_shape)
    fig.add_trace(go.Scatter3d(
        x=nose_points['x'],
        y=nose_points['y'],
        z=nose_points['z'],
        mode='lines',
        name='Nose Cone',
        line=dict(color='blue', width=4),
        showlegend=True
    ))
    
    # 2. Body Tube (cylinder)
    body_cylinder = _generate_cylinder_points(
        start_x=nose_length,
        length=body_length,
        radius=body_radius,
        segments=32
    )
    fig.add_trace(go.Mesh3d(
        x=body_cylinder['x'],
        y=body_cylinder['y'],
        z=body_cylinder['z'],
        color='lightblue',
        opacity=0.7,
        name='Body Tube',
        showlegend=True
    ))
    
    # 3. Fins
    for i in range(fin_count):
        angle = 2 * math.pi * i / fin_count
        fin_points = _generate_fin_points(
            start_x=fin_start_x,
            root_chord=fin_root_chord,
            tip_chord=fin_tip_chord,
            span=fin_span,
            sweep=fin_sweep,
            angle=angle,
            body_radius=body_radius
        )
        fig.add_trace(go.Mesh3d(
            x=fin_points['x'],
            y=fin_points['y'],
            z=fin_points['z'],
            color='red',
            opacity=0.8,
            name=f'Fin {i+1}' if i == 0 else '',
            showlegend=(i == 0)
        ))
    
    # 4. Payload Bay (if present)
    payload_length = config.get('payload_length', 0.3)
    payload_start_x = nose_length + 0.1
    if payload_length > 0:
        payload_cylinder = _generate_cylinder_points(
            start_x=payload_start_x,
            length=payload_length,
            radius=body_radius - 0.002,  # Slightly smaller than body
            segments=16
        )
        fig.add_trace(go.Mesh3d(
            x=payload_cylinder['x'],
            y=payload_cylinder['y'],
            z=payload_cylinder['z'],
            color='green',
            opacity=0.5,
            name='Payload Bay',
            showlegend=True
        ))
    
    # 5. Avionics Bay (if dual deploy)
    has_drogue = config.get('has_drogue', False)
    if has_drogue:
        avionics_length = 0.15  # 15cm avionics bay
        avionics_start_x = nose_length + body_length * 0.3
        avionics_cylinder = _generate_cylinder_points(
            start_x=avionics_start_x,
            length=avionics_length,
            radius=body_radius - 0.002,
            segments=16
        )
        fig.add_trace(go.Mesh3d(
            x=avionics_cylinder['x'],
            y=avionics_cylinder['y'],
            z=avionics_cylinder['z'],
            color='purple',
            opacity=0.6,
            name='Avionics Bay',
            showlegend=True
        ))
    
    # 6. Motor Mount (inside body) - use config if available, otherwise calculate
    motor_mount_radius = config.get('motor_mount_radius', motor_radius + 0.005)
    motor_mount_length = config.get('motor_mount_length', motor_length + 0.1)
    motor_start_x = nose_length + body_length - motor_mount_length
    motor_cylinder = _generate_cylinder_points(
        start_x=motor_start_x,
        length=motor_mount_length,
        radius=motor_mount_radius,
        segments=16
    )
    fig.add_trace(go.Mesh3d(
        x=motor_cylinder['x'],
        y=motor_cylinder['y'],
        z=motor_cylinder['z'],
        color='orange',
        opacity=0.6,
        name='Motor Mount',
        showlegend=True
    ))
    
    # 7. Motor (if provided)
    if motor:
        motor_actual_length = motor.get('length', motor_length)
        motor_actual_radius = motor.get('diameter', motor_diameter) / 2.0
        motor_cylinder = _generate_cylinder_points(
            start_x=motor_start_x + 0.05,
            length=motor_actual_length,
            radius=motor_actual_radius,
            segments=16
        )
        fig.add_trace(go.Mesh3d(
            x=motor_cylinder['x'],
            y=motor_cylinder['y'],
            z=motor_cylinder['z'],
            color='darkorange',
            opacity=0.9,
            name='Motor',
            showlegend=True
        ))
    
    # 8. Centering Rings (visualize as thin disks)
    ring_positions = [
        motor_start_x,
        motor_start_x + motor_mount_length * 0.5,
        motor_start_x + motor_mount_length
    ]
    for ring_x in ring_positions:
        ring_points = _generate_ring_points(ring_x, body_radius, motor_mount_radius, 0.006)
        fig.add_trace(go.Mesh3d(
            x=ring_points['x'],
            y=ring_points['y'],
            z=ring_points['z'],
            color='gray',
            opacity=0.5,
            name='Centering Ring' if ring_x == ring_positions[0] else '',
            showlegend=(ring_x == ring_positions[0])
        ))
    
    # 9. Parachutes (visualize inside their bays as packed volumes)
    # Parachutes pack much smaller - typically 1/20th to 1/30th of deployed diameter
    if config.get('has_main_chute', False):
        main_chute_dia = config.get('main_chute_diameter', 2.91)
        # Main chute is stored in nose cone - show as packed volume
        # Realistic packed size: parachute packs to ~1/25th of deployed diameter, ~1/10th diameter length
        packed_dia = main_chute_dia / 25.0  # Much smaller packed size
        chute_x = nose_length * 0.3  # Inside nose cone
        chute_length = packed_dia * 2.0  # Packed length is about 2x diameter
        
        # Make sure it fits inside nose cone
        if packed_dia > body_radius * 0.8:
            packed_dia = body_radius * 0.8
        
        chute_cylinder = _generate_cylinder_points(
            start_x=chute_x,
            length=chute_length,
            radius=packed_dia / 2.0,
            segments=16
        )
        fig.add_trace(go.Mesh3d(
            x=chute_cylinder['x'],
            y=chute_cylinder['y'],
            z=chute_cylinder['z'],
            color='yellow',
            opacity=0.7,
            name='Main Parachute (Packed)',
            showlegend=True
        ))
    
    if config.get('has_drogue', False):
        drogue_chute_dia = config.get('drogue_diameter', 0.99)
        # Drogue is stored in avionics bay - show as packed volume
        packed_dia = drogue_chute_dia / 25.0  # Much smaller packed size
        chute_x = avionics_start_x + 0.05  # Inside avionics bay
        chute_length = packed_dia * 2.0
        
        # Make sure it fits inside avionics bay
        if packed_dia > body_radius * 0.8:
            packed_dia = body_radius * 0.8
        
        chute_cylinder = _generate_cylinder_points(
            start_x=chute_x,
            length=chute_length,
            radius=packed_dia / 2.0,
            segments=16
        )
        fig.add_trace(go.Mesh3d(
            x=chute_cylinder['x'],
            y=chute_cylinder['y'],
            z=chute_cylinder['z'],
            color='orange',
            opacity=0.7,
            name='Drogue Parachute (Packed)',
            showlegend=True
        ))
    
    # Set layout
    fig.update_layout(
        title={
            'text': f"3D Rocket Visualization - {config.get('name', 'Custom Rocket')}",
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title='Length (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def _generate_nose_cone_points(length: float, base_radius: float, shape: str) -> Dict:
    """Generate points for nose cone based on shape"""
    n_points = 50
    x = np.linspace(0, length, n_points)
    y = []
    z = []
    
    for xi in x:
        if shape == 'CONICAL':
            r = base_radius * (1 - xi / length)
        elif shape == 'OGIVE':
            # Tangent ogive
            rho = (base_radius**2 + length**2) / (2 * base_radius)
            r = math.sqrt(rho**2 - (length - xi)**2) - (rho - base_radius)
        elif shape == 'VON_KARMAN':
            # Von Karman ogive
            theta = math.acos(1 - 2 * xi / length)
            r = base_radius * math.sqrt((1 - math.cos(theta)) / 2)
        else:
            # Generic approximation
            r = base_radius * math.sqrt(1 - xi / length)
        
        # Create circle at this radius
        angles = np.linspace(0, 2 * math.pi, 20)
        for angle in angles:
            y.append(r * math.cos(angle))
            z.append(r * math.sin(angle))
    
    # Create x array matching y/z
    x_full = np.repeat(x, 20)
    
    return {'x': x_full.tolist(), 'y': y, 'z': z}


def _generate_cylinder_points(start_x: float, length: float, radius: float, segments: int = 32) -> Dict:
    """Generate points for a cylinder"""
    angles = np.linspace(0, 2 * math.pi, segments)
    x = []
    y = []
    z = []
    
    # Top and bottom circles
    for angle in angles:
        y.append(radius * math.cos(angle))
        z.append(radius * math.sin(angle))
        x.append(start_x)
        
        y.append(radius * math.cos(angle))
        z.append(radius * math.sin(angle))
        x.append(start_x + length)
    
    # Connect with vertical lines
    for i in range(segments):
        angle = angles[i]
        y.append(radius * math.cos(angle))
        z.append(radius * math.sin(angle))
        x.append(start_x)
        
        y.append(radius * math.cos(angle))
        z.append(radius * math.sin(angle))
        x.append(start_x + length)
    
    return {'x': x, 'y': y, 'z': z}


def _generate_fin_points(start_x: float, root_chord: float, tip_chord: float,
                        span: float, sweep: float, angle: float,
                        body_radius: float) -> Dict:
    """Generate points for a trapezoidal fin"""
    # Fin coordinates in fin-local frame
    # Root chord at body, tip chord at span distance
    fin_points_local = [
        [0, body_radius, 0],  # Root leading edge
        [root_chord, body_radius, 0],  # Root trailing edge
        [root_chord - sweep, body_radius + span, 0],  # Tip trailing edge
        [-sweep, body_radius + span, 0],  # Tip leading edge
    ]
    
    # Transform to rocket frame (rotate by angle, translate by start_x)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    x = []
    y = []
    z = []
    
    for point in fin_points_local:
        # Rotate around x-axis
        y_rot = point[1] * cos_a - point[2] * sin_a
        z_rot = point[1] * sin_a + point[2] * cos_a
        
        x.append(start_x + point[0])
        y.append(y_rot)
        z.append(z_rot)
    
    # Close the fin
    x.append(x[0])
    y.append(y[0])
    z.append(z[0])
    
    return {'x': x, 'y': y, 'z': z}


def _generate_ring_points(x: float, outer_radius: float, inner_radius: float, thickness: float) -> Dict:
    """Generate points for a centering ring (annulus)"""
    segments = 32
    angles = np.linspace(0, 2 * math.pi, segments)
    
    x_coords = []
    y_coords = []
    z_coords = []
    
    # Outer circle
    for angle in angles:
        x_coords.append(x)
        y_coords.append(outer_radius * math.cos(angle))
        z_coords.append(outer_radius * math.sin(angle))
        
        x_coords.append(x + thickness)
        y_coords.append(outer_radius * math.cos(angle))
        z_coords.append(outer_radius * math.sin(angle))
    
    # Inner circle
    for angle in angles:
        x_coords.append(x)
        y_coords.append(inner_radius * math.cos(angle))
        z_coords.append(inner_radius * math.sin(angle))
        
        x_coords.append(x + thickness)
        y_coords.append(inner_radius * math.cos(angle))
        z_coords.append(inner_radius * math.sin(angle))
    
    return {'x': x_coords, 'y': y_coords, 'z': z_coords}


def visualize_rocket_2d_side_view(config: Dict[str, Any], motor: Optional[Dict] = None) -> go.Figure:
    """
    Create a 2D side view of the rocket.
    
    Args:
        config: Rocket configuration dictionary
        motor: Optional motor configuration dictionary
        
    Returns:
        Plotly figure object
    """
    # Extract dimensions
    nose_length = config.get('nose_length', 0.5)
    body_length = config.get('body_length', 1.5)
    body_radius = config.get('body_radius', 0.0635)
    fin_count = config.get('fin_count', 4)
    fin_root_chord = config.get('fin_root_chord', 0.12)
    fin_tip_chord = config.get('fin_tip_chord', 0.06)
    fin_span = config.get('fin_span', 0.11)
    fin_sweep = config.get('fin_sweep', 0.06)
    
    # Motor dimensions
    motor_length = motor.get('length', 0.64) if motor else 0.64
    motor_radius = (motor.get('diameter', 0.075) if motor else 0.075) / 2.0
    
    # Calculate positions
    total_length = nose_length + body_length
    fin_start_x = nose_length + body_length - fin_root_chord
    motor_start_x = nose_length + body_length - motor_length - 0.1
    
    fig = go.Figure()
    
    # Nose cone outline
    nose_shape = config.get('nose_shape', 'VON_KARMAN')
    n_points = 50
    x_nose = np.linspace(0, nose_length, n_points)
    y_nose = []
    for xi in x_nose:
        if nose_shape == 'CONICAL':
            r = body_radius * (1 - xi / nose_length)
        elif nose_shape == 'OGIVE':
            rho = (body_radius**2 + nose_length**2) / (2 * body_radius)
            r = math.sqrt(rho**2 - (nose_length - xi)**2) - (rho - body_radius)
        elif nose_shape == 'VON_KARMAN':
            theta = math.acos(1 - 2 * xi / nose_length)
            r = body_radius * math.sqrt((1 - math.cos(theta)) / 2)
        else:
            r = body_radius * math.sqrt(1 - xi / nose_length)
        y_nose.append(r)
    
    # Mirror for bottom
    x_nose_full = np.concatenate([x_nose, x_nose[::-1]])
    y_nose_full = np.concatenate([y_nose, [-r for r in y_nose[::-1]]])
    
    fig.add_trace(go.Scatter(
        x=x_nose_full,
        y=y_nose_full,
        fill='toself',
        fillcolor='lightblue',
        line=dict(color='blue', width=2),
        name='Nose Cone'
    ))
    
    # Body tube
    x_body = [nose_length, nose_length + body_length, nose_length + body_length, nose_length]
    y_body = [body_radius, body_radius, -body_radius, -body_radius]
    fig.add_trace(go.Scatter(
        x=x_body,
        y=y_body,
        fill='toself',
        fillcolor='lightblue',
        line=dict(color='blue', width=2),
        name='Body Tube'
    ))
    
    # Fins (show one fin)
    fin_x = [fin_start_x, fin_start_x + fin_root_chord, 
             fin_start_x + fin_root_chord - fin_sweep, fin_start_x - fin_sweep]
    fin_y = [body_radius, body_radius, body_radius + fin_span, body_radius + fin_span]
    fig.add_trace(go.Scatter(
        x=fin_x,
        y=fin_y,
        fill='toself',
        fillcolor='red',
        line=dict(color='darkred', width=2),
        name=f'Fins ({fin_count})'
    ))
    
    # Motor mount
    x_mount = [motor_start_x, motor_start_x + motor_length + 0.1,
               motor_start_x + motor_length + 0.1, motor_start_x]
    y_mount = [motor_radius + 0.005, motor_radius + 0.005,
               -(motor_radius + 0.005), -(motor_radius + 0.005)]
    fig.add_trace(go.Scatter(
        x=x_mount,
        y=y_mount,
        fill='toself',
        fillcolor='orange',
        line=dict(color='darkorange', width=2),
        name='Motor Mount'
    ))
    
    # Motor
    if motor:
        x_motor = [motor_start_x + 0.05, motor_start_x + 0.05 + motor_length,
                   motor_start_x + 0.05 + motor_length, motor_start_x + 0.05]
        y_motor = [motor_radius, motor_radius, -motor_radius, -motor_radius]
        fig.add_trace(go.Scatter(
            x=x_motor,
            y=y_motor,
            fill='toself',
            fillcolor='darkorange',
            line=dict(color='orange', width=2),
            name='Motor'
        ))
    
    # Add dimension labels
    fig.add_annotation(
        x=nose_length / 2,
        y=body_radius * 1.5,
        text=f"Nose: {nose_length*1000:.0f}mm",
        showarrow=False,
        font=dict(size=10)
    )
    fig.add_annotation(
        x=nose_length + body_length / 2,
        y=body_radius * 1.5,
        text=f"Body: {body_length*1000:.0f}mm",
        showarrow=False,
        font=dict(size=10)
    )
    fig.add_annotation(
        x=total_length / 2,
        y=-body_radius * 1.5,
        text=f"Total: {total_length*1000:.0f}mm",
        showarrow=False,
        font=dict(size=10, color='black')
    )
    
    fig.update_layout(
        title={
            'text': f"2D Side View - {config.get('name', 'Custom Rocket')}",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Length (m)',
        yaxis_title='Radius (m)',
        width=800,
        height=400,
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
        showlegend=True
    )
    
    return fig

