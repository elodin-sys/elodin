"""
Real 3D Rocket Mesh Renderer using Trimesh.

Generates actual 3D meshes for:
- Proper visualization with lighting and materials
- Export to STL/OBJ/GLTF for 3D printing or CAD
- Integration with Three.js via GLTF export
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple, List
import io
import base64

try:
    import numpy as np
    import trimesh
    from trimesh.creation import cylinder, cone
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    np = None  # Will cause errors if used without trimesh

# Colors for rocket components (RGB 0-255)
COLORS = {
    "nose": [0, 212, 255, 230],      # Cyan
    "body": [123, 47, 255, 220],      # Purple
    "fins": [255, 107, 53, 255],      # Orange
    "mount": [255, 184, 0, 180],      # Amber
    "motor": [255, 51, 102, 255],     # Red
    "chute": [200, 200, 200, 150],    # Light gray
}


def create_nose_cone(
    length: float,
    base_radius: float,
    shape: str = "VON_KARMAN",
    resolution: int = 32,
) -> "trimesh.Trimesh":
    """Create a nose cone mesh."""
    if not TRIMESH_AVAILABLE:
        return None
    
    # Generate nose cone profile
    n_rings = 20
    z_values = np.linspace(0, length, n_rings)
    
    if shape == "VON_KARMAN" or shape == "HAACK":
        # Von Karman profile (optimal for supersonic)
        theta = np.arccos(1 - 2 * z_values / length)
        radii = base_radius * np.sqrt((theta - np.sin(2 * theta) / 2) / np.pi)
    elif shape == "OGIVE":
        # Tangent ogive
        rho = (base_radius**2 + length**2) / (2 * base_radius)
        radii = np.sqrt(rho**2 - (length - z_values)**2) - rho + base_radius
    elif shape == "CONICAL":
        # Simple cone
        radii = base_radius * z_values / length
    elif shape == "ELLIPSOID":
        # Elliptical
        radii = base_radius * np.sqrt(1 - (1 - z_values / length)**2)
    else:
        # Default to parabolic
        radii = base_radius * np.sqrt(z_values / length)
    
    # Ensure tip starts at 0
    radii[0] = 0.001
    
    # Create vertices
    vertices = []
    faces = []
    
    for i, (z, r) in enumerate(zip(z_values, radii)):
        for j in range(resolution):
            theta = 2 * np.pi * j / resolution
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            vertices.append([x, y, z])
    
    # Create faces
    for i in range(n_rings - 1):
        for j in range(resolution):
            v0 = i * resolution + j
            v1 = i * resolution + (j + 1) % resolution
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + (j + 1) % resolution
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.face_colors = COLORS["nose"]
    return mesh


def create_body_tube(
    length: float,
    outer_radius: float,
    resolution: int = 32,
) -> "trimesh.Trimesh":
    """Create a body tube mesh."""
    if not TRIMESH_AVAILABLE:
        return None
    
    mesh = cylinder(radius=outer_radius, height=length, sections=resolution)
    # Center is at 0, rotate so it extends in +Z
    mesh.apply_translation([0, 0, length / 2])
    mesh.visual.face_colors = COLORS["body"]
    return mesh


def create_fin(
    root_chord: float,
    tip_chord: float,
    span: float,
    sweep: float,
    thickness: float,
) -> "trimesh.Trimesh":
    """Create a single trapezoidal fin."""
    if not TRIMESH_AVAILABLE:
        return None
    
    # Fin vertices (looking from above, fin extends in +Y)
    # XZ plane is the plane of the fin
    half_t = thickness / 2
    
    vertices = [
        # Bottom (outer surface of body tube side)
        [0, 0, -half_t],            # Root leading edge
        [root_chord, 0, -half_t],   # Root trailing edge
        [sweep + tip_chord, span, -half_t],  # Tip trailing edge
        [sweep, span, -half_t],     # Tip leading edge
        # Top
        [0, 0, half_t],
        [root_chord, 0, half_t],
        [sweep + tip_chord, span, half_t],
        [sweep, span, half_t],
    ]
    
    faces = [
        # Bottom face
        [0, 1, 2], [0, 2, 3],
        # Top face
        [4, 6, 5], [4, 7, 6],
        # Leading edge
        [0, 3, 7], [0, 7, 4],
        # Trailing edge
        [1, 5, 6], [1, 6, 2],
        # Root
        [0, 4, 5], [0, 5, 1],
        # Tip
        [3, 2, 6], [3, 6, 7],
    ]
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.face_colors = COLORS["fins"]
    return mesh


def create_fin_set(
    root_chord: float,
    tip_chord: float,
    span: float,
    sweep: float,
    thickness: float,
    fin_count: int,
    body_radius: float,
) -> "trimesh.Trimesh":
    """Create a set of fins around the body."""
    if not TRIMESH_AVAILABLE:
        return None
    
    meshes = []
    
    for i in range(fin_count):
        fin = create_fin(root_chord, tip_chord, span, sweep, thickness)
        if fin is None:
            continue
        
        # Rotate to radial position
        angle = 2 * np.pi * i / fin_count
        
        # Transform: rotate around Z, then translate to body surface
        rotation = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        
        # The fin is in the XY plane, we need it radial
        # First rotate 90° around X to make it vertical
        vertical = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        fin.apply_transform(vertical)
        
        # Translate to body surface
        fin.apply_translation([body_radius, 0, root_chord])
        
        # Rotate around Z axis
        fin.apply_transform(rotation)
        
        meshes.append(fin)
    
    if not meshes:
        return None
    
    return trimesh.util.concatenate(meshes)


def create_motor(
    diameter: float,
    length: float,
    resolution: int = 24,
) -> "trimesh.Trimesh":
    """Create motor mesh."""
    if not TRIMESH_AVAILABLE:
        return None
    
    mesh = cylinder(radius=diameter / 2, height=length, sections=resolution)
    mesh.apply_translation([0, 0, length / 2])
    mesh.visual.face_colors = COLORS["motor"]
    return mesh


def build_rocket_mesh(config: Dict[str, Any], motor: Optional[Dict] = None) -> "trimesh.Trimesh":
    """
    Build complete rocket mesh from config.
    
    Config should have:
    - nose_length, body_length, body_radius
    - fin_count, fin_root_chord, fin_tip_chord, fin_span, fin_sweep
    - Optional: motor diameter/length
    """
    if not TRIMESH_AVAILABLE:
        raise RuntimeError("trimesh not installed")
    
    meshes = []
    
    # Get dimensions
    body_radius = config.get("body_radius", 0.05)
    nose_length = config.get("nose_length", body_radius * 4)
    body_length = config.get("body_length", 1.0)
    
    # Nose cone
    nose_shape = config.get("nose_shape", "VON_KARMAN")
    nose = create_nose_cone(nose_length, body_radius, nose_shape)
    if nose:
        meshes.append(nose)
    
    # Body tube (starts at end of nose)
    body = create_body_tube(body_length, body_radius)
    if body:
        body.apply_translation([0, 0, nose_length])
        meshes.append(body)
    
    # Fins
    if config.get("fin_count", 0) > 0:
        fin_root = config.get("fin_root_chord", body_radius * 1.5)
        fin_tip = config.get("fin_tip_chord", fin_root * 0.4)
        fin_span = config.get("fin_span", body_radius * 1.2)
        fin_sweep = config.get("fin_sweep", fin_root * 0.25)
        fin_thickness = config.get("fin_thickness", 0.004)
        fin_count = config.get("fin_count", 4)
        
        fins = create_fin_set(
            fin_root, fin_tip, fin_span, fin_sweep, fin_thickness,
            fin_count, body_radius
        )
        if fins:
            # Position at rear of body
            fins.apply_translation([0, 0, nose_length + body_length - fin_root])
            meshes.append(fins)
    
    # Motor (if provided)
    if motor:
        motor_dia = motor.get("diameter", 0.054)
        motor_len = motor.get("length", 0.5)
        motor_mesh = create_motor(motor_dia, motor_len)
        if motor_mesh:
            # Position inside motor mount at aft end
            motor_mesh.apply_translation([0, 0, nose_length + body_length - motor_len])
            meshes.append(motor_mesh)
    
    if not meshes:
        raise RuntimeError("No meshes created")
    
    return trimesh.util.concatenate(meshes)


def export_stl(config: Dict, motor: Optional[Dict] = None) -> bytes:
    """Export rocket to STL bytes."""
    mesh = build_rocket_mesh(config, motor)
    buffer = io.BytesIO()
    mesh.export(buffer, file_type='stl')
    return buffer.getvalue()


def export_gltf(config: Dict, motor: Optional[Dict] = None) -> bytes:
    """Export rocket to GLTF bytes (for web viewers)."""
    mesh = build_rocket_mesh(config, motor)
    buffer = io.BytesIO()
    mesh.export(buffer, file_type='glb')
    return buffer.getvalue()


def export_obj(config: Dict, motor: Optional[Dict] = None) -> str:
    """Export rocket to OBJ string."""
    mesh = build_rocket_mesh(config, motor)
    return mesh.export(file_type='obj')


def render_to_plotly(config: Dict, motor: Optional[Dict] = None):
    """Render rocket mesh to Plotly figure with proper lighting."""
    import plotly.graph_objects as go
    
    mesh = build_rocket_mesh(config, motor)
    
    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Get per-face colors
    if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
        face_colors = mesh.visual.face_colors
        # Convert to intensity (use average of RGB)
        intensity = np.mean(face_colors[:, :3] / 255.0, axis=1)
    else:
        intensity = np.ones(len(faces)) * 0.5
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=intensity,
            colorscale=[
                [0.0, '#FF6B35'],   # Orange (fins)
                [0.3, '#7B2FFF'],   # Purple (body)
                [0.6, '#00D4FF'],   # Cyan (nose)
                [1.0, '#FF3366'],   # Red (motor)
            ],
            flatshading=True,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
            ),
            lightposition=dict(x=1000, y=1000, z=2000),
        )
    ])
    
    # Dark theme styling
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor="#0A0E17",
                gridcolor="#1A2540",
                showbackground=True,
                zerolinecolor="#2A3A5C",
                showticklabels=False,
            ),
            yaxis=dict(
                backgroundcolor="#0A0E17",
                gridcolor="#1A2540",
                showbackground=True,
                zerolinecolor="#2A3A5C",
                showticklabels=False,
            ),
            zaxis=dict(
                backgroundcolor="#0A0E17",
                gridcolor="#1A2540",
                showbackground=True,
                zerolinecolor="#2A3A5C",
                title="Length (m)",
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="#0A0E17",
        plot_bgcolor="#0A0E17",
        font=dict(color="#E8ECF4", family="Inter"),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text=config.get("name", "Rocket Design"),
            font=dict(size=18, color="#00D4FF"),
        ),
    )
    
    return fig


def get_rocket_preview_html(config: Dict, motor: Optional[Dict] = None) -> str:
    """
    Generate HTML with embedded 3D viewer using Three.js.
    Returns self-contained HTML that can be embedded in Streamlit.
    """
    try:
        glb_bytes = export_gltf(config, motor)
        glb_base64 = base64.b64encode(glb_bytes).decode('utf-8')
    except Exception as e:
        return f"<div style='color: red;'>Error generating 3D model: {e}</div>"
    
    html = f'''
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ margin: 0; background: #0A0E17; }}
        #container {{ width: 100%; height: 400px; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.157.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.157.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.157.0/examples/js/loaders/GLTFLoader.js"></script>
    <script>
        const container = document.getElementById('container');
        
        // Scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0A0E17);
        
        // Camera
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 100);
        camera.position.set(2, 2, 2);
        
        // Renderer
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 1.0;
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(5, 5, 5);
        scene.add(directionalLight);
        
        const backLight = new THREE.DirectionalLight(0x00D4FF, 0.3);
        backLight.position.set(-5, -5, 5);
        scene.add(backLight);
        
        // Load model
        const glbData = "{glb_base64}";
        const binaryString = atob(glbData);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {{
            bytes[i] = binaryString.charCodeAt(i);
        }}
        
        const loader = new THREE.GLTFLoader();
        loader.parse(bytes.buffer, '', function(gltf) {{
            const model = gltf.scene;
            
            // Center model
            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());
            model.position.sub(center);
            
            // Rotate so rocket is vertical
            model.rotation.x = -Math.PI / 2;
            
            scene.add(model);
            
            // Fit camera
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            camera.position.set(maxDim * 1.5, maxDim * 1.5, maxDim * 1.5);
            controls.target.set(0, 0, 0);
        }});
        
        // Animation
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        // Resize
        window.addEventListener('resize', () => {{
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }});
    </script>
</body>
</html>
'''
    return html


def generate_elodin_assets(config: Dict, motor: Optional[Dict] = None, output_dir: str = ".") -> str:
    """
    Generate assets for Elodin editor visualization.
    
    Creates:
    - rocket.glb - The rocket 3D model
    
    Returns: Path to the generated GLB file
    """
    import os
    
    if not TRIMESH_AVAILABLE:
        raise RuntimeError("trimesh required for Elodin asset generation")
    
    # Build the mesh
    mesh = build_rocket_mesh(config, motor)
    
    # Rotate mesh to align with Elodin/glTF coordinate expectations
    # Testing +90° around X axis (opposite direction)
    rotation = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    mesh.apply_transform(rotation)
    
    # Center the mesh at origin
    mesh.vertices -= mesh.centroid
    
    # Export to GLB
    glb_path = os.path.join(output_dir, "rocket.glb")
    mesh.export(glb_path, file_type='glb')
    
    print(f"✓ Generated Elodin asset: {glb_path}")
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    
    return glb_path


def generate_rocket_glb_from_solver(solver, output_dir: str = ".") -> str:
    """
    Generate rocket.glb from a FlightSolver's rocket configuration.
    
    Args:
        solver: FlightSolver instance with rocket model
        output_dir: Directory to write rocket.glb
        
    Returns: Path to generated GLB
    """
    if not TRIMESH_AVAILABLE:
        raise RuntimeError("trimesh required")
    
    # Extract config from solver's rocket model
    rocket = solver.rocket
    
    # Get dimensions from rocket model
    config = {
        "name": "Rocket",
        "body_radius": rocket.reference_diameter / 2,
        "body_length": 1.5,  # Default, will try to get from components
        "nose_length": 0.5,
        "nose_shape": "VON_KARMAN",
        "fin_count": 4,
        "fin_root_chord": 0.12,
        "fin_tip_chord": 0.05,
        "fin_span": 0.1,
        "fin_sweep": 0.03,
        "fin_thickness": 0.004,
    }
    
    # Try to extract from OpenRocket components if available
    if hasattr(rocket, '_rocket') and hasattr(rocket._rocket, 'children'):
        for child in rocket._rocket.children:
            if hasattr(child, '__class__'):
                cls_name = child.__class__.__name__
                if cls_name == "NoseCone":
                    config["nose_length"] = getattr(child, 'length', 0.5)
                    if hasattr(child, 'shape'):
                        config["nose_shape"] = child.shape.name if hasattr(child.shape, 'name') else "VON_KARMAN"
                elif cls_name == "BodyTube":
                    config["body_length"] = getattr(child, 'length', 1.5)
                    config["body_radius"] = getattr(child, 'outer_radius', 0.05)
                    
                    # Check for fins in body tube children
                    if hasattr(child, 'children'):
                        for subchild in child.children:
                            if "Fin" in subchild.__class__.__name__:
                                config["fin_count"] = getattr(subchild, 'fin_count', 4)
                                config["fin_root_chord"] = getattr(subchild, 'root_chord', 0.12)
                                config["fin_tip_chord"] = getattr(subchild, 'tip_chord', 0.05)
                                config["fin_span"] = getattr(subchild, 'span', 0.1)
                                config["fin_sweep"] = getattr(subchild, 'sweep', 0.03)
    
    # Motor config from solver
    motor_config = None
    if hasattr(solver, 'motor'):
        motor = solver.motor
        motor_config = {
            "diameter": getattr(motor, 'diameter', 0.054),
            "length": getattr(motor, 'length', 0.5),
        }
    
    return generate_elodin_assets(config, motor_config, output_dir)


if __name__ == "__main__":
    # Test
    config = {
        "name": "Test Rocket",
        "nose_length": 0.25,
        "nose_shape": "VON_KARMAN",
        "body_length": 1.2,
        "body_radius": 0.05,
        "fin_count": 4,
        "fin_root_chord": 0.12,
        "fin_tip_chord": 0.05,
        "fin_span": 0.08,
        "fin_sweep": 0.03,
        "fin_thickness": 0.004,
    }
    
    motor = {
        "diameter": 0.054,
        "length": 0.4,
    }
    
    print("Building mesh...")
    mesh = build_rocket_mesh(config, motor)
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    
    # Export test
    stl_data = export_stl(config, motor)
    print(f"STL size: {len(stl_data)} bytes")
    
    # Save test STL
    with open("/tmp/test_rocket.stl", "wb") as f:
        f.write(stl_data)
    print("Saved to /tmp/test_rocket.stl")
    
    # Generate Elodin assets
    print("\nGenerating Elodin assets...")
    generate_elodin_assets(config, motor, "/tmp")

