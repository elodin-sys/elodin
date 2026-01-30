//! Test example for axes-cube.glb with hover highlighting and click-to-rotate
//!
//! Run with:
//!   cargo run --example axes_cube -p elodin-editor
//!
//! Features:
//! - Hover over faces, edges, or corners to highlight them in yellow
//! - Click on any element to rotate the camera to look at the cube from that direction

use bevy::asset::AssetPlugin;
use bevy::picking::prelude::*;
use bevy::prelude::*;
use std::path::PathBuf;

fn main() {
    // Compute path to repo root's assets/ folder
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let assets_path = manifest_dir
        .parent() // libs/
        .unwrap()
        .parent() // repo root
        .unwrap()
        .join("assets");

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Axes Cube - Hover & Click Test".to_string(),
                        resolution: (800, 600).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: assets_path.to_string_lossy().to_string(),
                    ..default()
                }),
        )
        .add_plugins(MeshPickingPlugin)
        .init_resource::<HoveredElement>()
        .init_resource::<OriginalMaterials>()
        .add_systems(Startup, setup)
        .add_systems(Update, (setup_cube_elements, animate_camera))
        .add_observer(on_hover_start)
        .add_observer(on_hover_end)
        .add_observer(on_click)
        .run();
}

// ============================================================================
// Components & Resources
// ============================================================================

#[derive(Component)]
struct AxesCube;

#[derive(Component)]
struct MainCamera;

/// Identifies what type of cube element this is
#[derive(Component, Clone, Debug)]
enum CubeElement {
    Face(FaceDirection),
    Edge(EdgeDirection),
    Corner(CornerPosition),
}

/// Face directions using ENU (East-North-Up) convention
/// - X axis: East (+X) / West (-X) - Red
/// - Y axis: North (+Y) / South (-Y) - Green  
/// - Z axis: Up (+Z) / Down (-Z) - Blue
#[derive(Clone, Copy, Debug)]
enum FaceDirection {
    // X axis (Red)
    East, // +X (Right)
    West, // -X (Left)
    // Y axis (Green)
    North, // +Y (Top)
    South, // -Y (Bottom)
    // Z axis (Blue)
    Up,   // +Z (Front)
    Down, // -Z (Back)
}

#[derive(Clone, Copy, Debug)]
enum EdgeDirection {
    // X-axis edges (horizontal, front-back)
    XTopFront,
    XTopBack,
    XBottomFront,
    XBottomBack,
    // Y-axis edges (vertical)
    YFrontLeft,
    YFrontRight,
    YBackLeft,
    YBackRight,
    // Z-axis edges (horizontal, left-right)
    ZTopLeft,
    ZTopRight,
    ZBottomLeft,
    ZBottomRight,
}

#[derive(Clone, Copy, Debug)]
enum CornerPosition {
    TopFrontLeft,
    TopFrontRight,
    TopBackLeft,
    TopBackRight,
    BottomFrontLeft,
    BottomFrontRight,
    BottomBackLeft,
    BottomBackRight,
}

/// Tracks the currently hovered element
#[derive(Resource, Default)]
struct HoveredElement {
    entity: Option<Entity>,
}

/// Stores original material colors for restoration after hover
#[derive(Resource, Default)]
struct OriginalMaterials {
    colors: std::collections::HashMap<Entity, Color>,
}

/// Camera animation state with start and target for smooth interpolation
#[derive(Resource)]
struct CameraAnimation {
    /// Starting position when animation began
    start_position: Vec3,
    /// Starting rotation when animation began
    start_rotation: Quat,
    /// Target position to animate to
    target_position: Vec3,
    /// Target rotation to animate to
    target_rotation: Quat,
    /// Animation progress from 0.0 to 1.0
    progress: f32,
    /// Whether animation is currently active
    animating: bool,
}

impl Default for CameraAnimation {
    fn default() -> Self {
        Self {
            start_position: Vec3::new(3.0, 2.5, 3.0),
            start_rotation: Quat::IDENTITY,
            target_position: Vec3::ZERO,
            target_rotation: Quat::IDENTITY,
            progress: 0.0,
            animating: false,
        }
    }
}

/// Marker for elements that have been set up
#[derive(Component)]
struct CubeElementSetup;

// ============================================================================
// Setup
// ============================================================================

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.init_resource::<CameraAnimation>();

    // Load the axes-cube.glb
    let scene = asset_server.load("axes-cube.glb#Scene0");

    // Spawn the cube
    commands.spawn((
        SceneRoot(scene),
        Transform::from_xyz(0.0, 0.0, 0.0),
        AxesCube,
        Name::new("axes_cube_root"),
    ));

    // Spawn RGB axes extending from the corner of the cube (like OnShape)
    spawn_axes(&mut commands, &mut meshes, &mut materials);

    // Camera - positioned to see the cube from an isometric-ish angle
    commands.spawn((
        Transform::from_xyz(3.0, 2.5, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        Camera3d::default(),
        MainCamera,
        Name::new("main_camera"),
    ));

    // Directional light
    commands.spawn((
        DirectionalLight {
            illuminance: 15000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(5.0, 10.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ambient light for visibility
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 800.0,
        affects_lightmapped_meshes: true,
    });

    println!("=== Axes Cube Interactive Test ===");
    println!("Hover over faces, edges, or corners to highlight them.");
    println!("Click to rotate the camera toward that element.");
    println!("Press ESC to quit.");
}

/// Spawn RGB axes extending from the bottom-left-back corner of the cube
fn spawn_axes(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    // Axis configuration - long enough to be clearly visible beyond the cube
    const AXIS_LENGTH: f32 = 1.4;
    const AXIS_RADIUS: f32 = 0.035;
    const TIP_RADIUS: f32 = 0.08;
    const TIP_LENGTH: f32 = 0.2;

    // Origin point - slightly outside the bottom-left-back corner of cube
    let origin = Vec3::new(-0.55, -0.55, -0.55);

    // Axis colors (ENU convention)
    // X = East/West (Red), Y = North/South (Green), Z = Up/Down (Blue)
    let colors = [
        (Vec3::X, Color::srgb(0.9, 0.2, 0.2), "X"), // Red for X (East/West)
        (Vec3::Y, Color::srgb(0.2, 0.8, 0.2), "Y"), // Green for Y (North/South)
        (Vec3::Z, Color::srgb(0.2, 0.4, 0.9), "Z"), // Blue for Z (Up/Down)
    ];

    // Create shared meshes
    let shaft_mesh = meshes.add(Cylinder::new(AXIS_RADIUS, AXIS_LENGTH));
    let tip_mesh = meshes.add(Cone::new(TIP_RADIUS, TIP_LENGTH));

    for (direction, color, name) in colors {
        let material = materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            ..default()
        });

        // Calculate rotation to align cylinder with axis direction
        // Cylinder is created along Y axis by default
        let rotation = if direction == Vec3::X {
            Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2)
        } else if direction == Vec3::Z {
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
        } else {
            Quat::IDENTITY // Y axis, no rotation needed
        };

        // Shaft position (center of the cylinder)
        let shaft_pos = origin + direction * (AXIS_LENGTH / 2.0);

        // Spawn shaft
        commands.spawn((
            Mesh3d(shaft_mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(shaft_pos).with_rotation(rotation),
            Name::new(format!("axis_{}_shaft", name)),
        ));

        // Tip position (at the end of the shaft)
        let tip_pos = origin + direction * (AXIS_LENGTH + TIP_LENGTH / 2.0);

        // Tip rotation (cone points along axis direction)
        let tip_rotation = if direction == Vec3::X {
            Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2)
        } else if direction == Vec3::Z {
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
        } else {
            Quat::IDENTITY
        };

        // Spawn tip (cone/arrow)
        commands.spawn((
            Mesh3d(tip_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(tip_pos).with_rotation(tip_rotation),
            Name::new(format!("axis_{}_tip", name)),
        ));
    }
}

/// Get color for a cube element (faces are grey/transparent)
fn get_element_color(element: &CubeElement) -> Color {
    match element {
        // Faces: grey and transparent so axes are visible through
        CubeElement::Face(_) => Color::srgba(0.6, 0.6, 0.65, 0.4),
        // Edges: darker, more opaque
        CubeElement::Edge(_) => Color::srgba(0.4, 0.4, 0.45, 0.85),
        // Corners: visible spheres
        CubeElement::Corner(_) => Color::srgba(0.5, 0.5, 0.55, 0.9),
    }
}

/// Set up cube elements after the GLB is loaded
/// Also clones materials so each element can be highlighted independently
fn setup_cube_elements(
    mut commands: Commands,
    query: Query<(Entity, &Name), (With<Name>, Without<CubeElementSetup>)>,
    parents: Query<&ChildOf>,
    children_query: Query<&Children>,
    axes_cube: Query<Entity, With<AxesCube>>,
    material_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut original_materials: ResMut<OriginalMaterials>,
) {
    let Ok(root) = axes_cube.single() else {
        return;
    };

    for (entity, name) in query.iter() {
        // Check if this entity is in the axes cube subtree
        let mut in_subtree = false;
        let mut current = entity;
        for _ in 0..10 {
            if current == root {
                in_subtree = true;
                break;
            }
            if let Ok(parent) = parents.get(current) {
                current = parent.0;
            } else {
                break;
            }
        }

        if !in_subtree {
            continue;
        }

        let name_str = name.as_str();

        // Parse the name to determine element type
        let element = if name_str.starts_with("Face_") {
            Some(parse_face(name_str))
        } else if name_str.starts_with("Border_") {
            Some(parse_edge(name_str))
        } else if name_str.starts_with("Corner_") {
            Some(parse_corner(name_str))
        } else {
            None
        };

        if let Some(elem) = element {
            // Get the color for this element type
            let element_color = get_element_color(&elem);

            commands
                .entity(entity)
                .insert((elem.clone(), CubeElementSetup));

            // Clone materials for children (mesh nodes) so they can be highlighted independently
            if let Ok(children) = children_query.get(entity) {
                for child in children.iter() {
                    if let Ok(mat_handle) = material_query.get(child) {
                        if let Some(_original_mat) = materials.get(&mat_handle.0) {
                            // Store the custom color as "original"
                            original_materials.colors.insert(child, element_color);

                            // Create a new material with the element color and transparency
                            let new_mat = StandardMaterial {
                                base_color: element_color,
                                alpha_mode: AlphaMode::Blend,
                                unlit: false,
                                double_sided: true,
                                cull_mode: None,
                                ..default()
                            };
                            let new_handle = materials.add(new_mat);
                            commands.entity(child).insert(MeshMaterial3d(new_handle));
                        }
                    }
                }
            }
        }
    }
}

fn parse_face(name: &str) -> CubeElement {
    // Map GLB face names to ENU directions
    let dir = match name {
        "Face_Front" => FaceDirection::Up,     // +Z
        "Face_Back" => FaceDirection::Down,    // -Z
        "Face_Left" => FaceDirection::West,    // -X
        "Face_Right" => FaceDirection::East,   // +X
        "Face_Top" => FaceDirection::North,    // +Y
        "Face_Bottom" => FaceDirection::South, // -Y
        _ => FaceDirection::Up,
    };
    CubeElement::Face(dir)
}

fn parse_edge(name: &str) -> CubeElement {
    // Border_X_y{-1|1}_z{-1|1} -> edges along X axis
    // Border_Y_x{-1|1}_z{-1|1} -> edges along Y axis (vertical)
    // Border_Z_x{-1|1}_y{-1|1} -> edges along Z axis
    let dir = match name {
        "Border_X_y1_z1" => EdgeDirection::XTopFront,
        "Border_X_y1_z-1" => EdgeDirection::XTopBack,
        "Border_X_y-1_z1" => EdgeDirection::XBottomFront,
        "Border_X_y-1_z-1" => EdgeDirection::XBottomBack,
        "Border_Y_x-1_z1" => EdgeDirection::YFrontLeft,
        "Border_Y_x1_z1" => EdgeDirection::YFrontRight,
        "Border_Y_x-1_z-1" => EdgeDirection::YBackLeft,
        "Border_Y_x1_z-1" => EdgeDirection::YBackRight,
        "Border_Z_x-1_y1" => EdgeDirection::ZTopLeft,
        "Border_Z_x1_y1" => EdgeDirection::ZTopRight,
        "Border_Z_x-1_y-1" => EdgeDirection::ZBottomLeft,
        "Border_Z_x1_y-1" => EdgeDirection::ZBottomRight,
        _ => EdgeDirection::XTopFront,
    };
    CubeElement::Edge(dir)
}

fn parse_corner(name: &str) -> CubeElement {
    // Corner_x{-1|1}_y{-1|1}_z{-1|1}
    let pos = match name {
        "Corner_x-1_y1_z1" => CornerPosition::TopFrontLeft,
        "Corner_x1_y1_z1" => CornerPosition::TopFrontRight,
        "Corner_x-1_y1_z-1" => CornerPosition::TopBackLeft,
        "Corner_x1_y1_z-1" => CornerPosition::TopBackRight,
        "Corner_x-1_y-1_z1" => CornerPosition::BottomFrontLeft,
        "Corner_x1_y-1_z1" => CornerPosition::BottomFrontRight,
        "Corner_x-1_y-1_z-1" => CornerPosition::BottomBackLeft,
        "Corner_x1_y-1_z-1" => CornerPosition::BottomBackRight,
        _ => CornerPosition::TopFrontRight,
    };
    CubeElement::Corner(pos)
}

// ============================================================================
// Hover Handlers
// ============================================================================

/// Highlight color for hovered elements
const HIGHLIGHT_COLOR: Color = Color::srgb(1.0, 0.9, 0.2); // Yellow

/// Camera distance from origin (used for both click target and animation)
const CAMERA_DISTANCE: f32 = 4.5;

fn on_hover_start(
    trigger: On<Pointer<Over>>,
    mut commands: Commands,
    cube_elements: Query<&CubeElement>,
    parents_query: Query<&ChildOf>,
    children_query: Query<&Children>,
    material_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut hovered: ResMut<HoveredElement>,
    mut original_materials: ResMut<OriginalMaterials>,
) {
    let entity = trigger.event().event_target();

    // Check if this is a cube element or find the parent that is
    let element_entity = if cube_elements.get(entity).is_ok() {
        Some(entity)
    } else {
        // Check if parent is a cube element (for mesh children like "Cube.003.Mat_Back")
        if let Ok(parent) = parents_query.get(entity) {
            if cube_elements.get(parent.0).is_ok() {
                Some(parent.0)
            } else {
                None
            }
        } else {
            None
        }
    };

    let Some(target) = element_entity else {
        return;
    };

    // Already hovering this element
    if hovered.entity == Some(target) {
        return;
    }

    // Reset previous hovered element
    if let Some(prev) = hovered.entity {
        reset_highlight(
            prev,
            &children_query,
            &material_query,
            &mut materials,
            &original_materials,
        );
    }

    // Apply highlight to target and its mesh children
    apply_highlight(
        target,
        &children_query,
        &material_query,
        &mut materials,
        &mut original_materials,
        &mut commands,
    );

    hovered.entity = Some(target);

    if let Ok(element) = cube_elements.get(target) {
        println!("HOVER: {:?}", element);
    }
}

fn on_hover_end(
    trigger: On<Pointer<Out>>,
    cube_elements: Query<&CubeElement>,
    parents_query: Query<&ChildOf>,
    children_query: Query<&Children>,
    material_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut hovered: ResMut<HoveredElement>,
    original_materials: Res<OriginalMaterials>,
) {
    let entity = trigger.event().event_target();

    // Check if this is the currently hovered element
    let element_entity = if cube_elements.get(entity).is_ok() {
        Some(entity)
    } else {
        // Check if parent is a cube element
        if let Ok(parent) = parents_query.get(entity) {
            if cube_elements.get(parent.0).is_ok() {
                Some(parent.0)
            } else {
                None
            }
        } else {
            None
        }
    };

    let Some(target) = element_entity else {
        return;
    };

    if hovered.entity != Some(target) {
        return;
    }

    reset_highlight(
        target,
        &children_query,
        &material_query,
        &mut materials,
        &original_materials,
    );
    hovered.entity = None;
}

fn apply_highlight(
    entity: Entity,
    children_query: &Query<&Children>,
    material_query: &Query<&MeshMaterial3d<StandardMaterial>>,
    materials: &mut Assets<StandardMaterial>,
    original_materials: &mut OriginalMaterials,
    _commands: &mut Commands,
) {
    // Apply to entity itself if it has a material
    if let Ok(mat_handle) = material_query.get(entity) {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            // Store original color if not already stored
            if !original_materials.colors.contains_key(&entity) {
                original_materials.colors.insert(entity, mat.base_color);
            }
            mat.base_color = HIGHLIGHT_COLOR;
            mat.emissive = LinearRgba::new(0.5, 0.45, 0.1, 1.0);
        }
    }

    // Apply to children (mesh nodes)
    if let Ok(children) = children_query.get(entity) {
        for child in children.iter() {
            if let Ok(mat_handle) = material_query.get(child) {
                if let Some(mat) = materials.get_mut(&mat_handle.0) {
                    // Store original color if not already stored
                    if !original_materials.colors.contains_key(&child) {
                        original_materials.colors.insert(child, mat.base_color);
                    }
                    mat.base_color = HIGHLIGHT_COLOR;
                    mat.emissive = LinearRgba::new(0.5, 0.45, 0.1, 1.0);
                }
            }
        }
    }
}

fn reset_highlight(
    entity: Entity,
    children_query: &Query<&Children>,
    material_query: &Query<&MeshMaterial3d<StandardMaterial>>,
    materials: &mut Assets<StandardMaterial>,
    original_materials: &OriginalMaterials,
) {
    // Reset entity's material
    if let Ok(mat_handle) = material_query.get(entity) {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            // Restore original color
            if let Some(&original_color) = original_materials.colors.get(&entity) {
                mat.base_color = original_color;
            }
            mat.emissive = LinearRgba::BLACK;
        }
    }

    // Reset children
    if let Ok(children) = children_query.get(entity) {
        for child in children.iter() {
            if let Ok(mat_handle) = material_query.get(child) {
                if let Some(mat) = materials.get_mut(&mat_handle.0) {
                    // Restore original color
                    if let Some(&original_color) = original_materials.colors.get(&child) {
                        mat.base_color = original_color;
                    }
                    mat.emissive = LinearRgba::BLACK;
                }
            }
        }
    }
}

// ============================================================================
// Click Handler
// ============================================================================

fn on_click(
    trigger: On<Pointer<Click>>,
    cube_elements: Query<&CubeElement>,
    parents_query: Query<&ChildOf>,
    camera_query: Query<&Transform, With<MainCamera>>,
    mut camera_anim: ResMut<CameraAnimation>,
) {
    let entity = trigger.event().event_target();

    // Find the cube element (or parent if this is a mesh child)
    let element = if let Ok(elem) = cube_elements.get(entity) {
        elem
    } else if let Ok(parent) = parents_query.get(entity) {
        if let Ok(elem) = cube_elements.get(parent.0) {
            elem
        } else {
            return;
        }
    } else {
        return;
    };

    // Get current camera state
    let Ok(current_transform) = camera_query.single() else {
        return;
    };

    // Calculate target rotation based on element
    let look_dir = get_look_direction(element);
    let up_dir = get_up_direction(element);

    // Calculate target: camera should look FROM the opposite direction
    let target_pos = -look_dir * CAMERA_DISTANCE;
    let target_transform = Transform::from_translation(target_pos).looking_at(Vec3::ZERO, up_dir);

    // Store start state and target for smooth interpolation
    camera_anim.start_position = current_transform.translation;
    camera_anim.start_rotation = current_transform.rotation;
    camera_anim.target_position = target_pos;
    camera_anim.target_rotation = target_transform.rotation;
    camera_anim.progress = 0.0;
    camera_anim.animating = true;

    println!("CLICK: {:?} -> rotating camera", element);
}

fn get_look_direction(element: &CubeElement) -> Vec3 {
    match element {
        CubeElement::Face(dir) => match dir {
            FaceDirection::East => Vec3::X,      // +X
            FaceDirection::West => Vec3::NEG_X,  // -X
            FaceDirection::North => Vec3::Y,     // +Y
            FaceDirection::South => Vec3::NEG_Y, // -Y
            FaceDirection::Up => Vec3::Z,        // +Z
            FaceDirection::Down => Vec3::NEG_Z,  // -Z
        },
        CubeElement::Edge(dir) => match dir {
            EdgeDirection::XTopFront => Vec3::new(0.0, 1.0, 1.0).normalize(),
            EdgeDirection::XTopBack => Vec3::new(0.0, 1.0, -1.0).normalize(),
            EdgeDirection::XBottomFront => Vec3::new(0.0, -1.0, 1.0).normalize(),
            EdgeDirection::XBottomBack => Vec3::new(0.0, -1.0, -1.0).normalize(),
            EdgeDirection::YFrontLeft => Vec3::new(-1.0, 0.0, 1.0).normalize(),
            EdgeDirection::YFrontRight => Vec3::new(1.0, 0.0, 1.0).normalize(),
            EdgeDirection::YBackLeft => Vec3::new(-1.0, 0.0, -1.0).normalize(),
            EdgeDirection::YBackRight => Vec3::new(1.0, 0.0, -1.0).normalize(),
            EdgeDirection::ZTopLeft => Vec3::new(-1.0, 1.0, 0.0).normalize(),
            EdgeDirection::ZTopRight => Vec3::new(1.0, 1.0, 0.0).normalize(),
            EdgeDirection::ZBottomLeft => Vec3::new(-1.0, -1.0, 0.0).normalize(),
            EdgeDirection::ZBottomRight => Vec3::new(1.0, -1.0, 0.0).normalize(),
        },
        CubeElement::Corner(pos) => match pos {
            CornerPosition::TopFrontLeft => Vec3::new(-1.0, 1.0, 1.0).normalize(),
            CornerPosition::TopFrontRight => Vec3::new(1.0, 1.0, 1.0).normalize(),
            CornerPosition::TopBackLeft => Vec3::new(-1.0, 1.0, -1.0).normalize(),
            CornerPosition::TopBackRight => Vec3::new(1.0, 1.0, -1.0).normalize(),
            CornerPosition::BottomFrontLeft => Vec3::new(-1.0, -1.0, 1.0).normalize(),
            CornerPosition::BottomFrontRight => Vec3::new(1.0, -1.0, 1.0).normalize(),
            CornerPosition::BottomBackLeft => Vec3::new(-1.0, -1.0, -1.0).normalize(),
            CornerPosition::BottomBackRight => Vec3::new(1.0, -1.0, -1.0).normalize(),
        },
    }
}

fn get_up_direction(element: &CubeElement) -> Vec3 {
    match element {
        CubeElement::Face(dir) => match dir {
            FaceDirection::North => Vec3::NEG_Z, // Looking down from above, Z is "forward"
            FaceDirection::South => Vec3::Z,     // Looking up from below
            _ => Vec3::Y,                        // For other faces, Y is up
        },
        CubeElement::Edge(dir) => match dir {
            EdgeDirection::XTopFront | EdgeDirection::XTopBack => Vec3::new(0.0, 1.0, 0.0),
            EdgeDirection::XBottomFront | EdgeDirection::XBottomBack => Vec3::new(0.0, 1.0, 0.0),
            _ => Vec3::Y,
        },
        CubeElement::Corner(_) => Vec3::Y,
    }
}

// ============================================================================
// Camera Animation
// ============================================================================

fn animate_camera(
    mut camera_query: Query<&mut Transform, With<MainCamera>>,
    mut camera_anim: ResMut<CameraAnimation>,
    time: Res<Time>,
) {
    if !camera_anim.animating {
        return;
    }

    let Ok(mut transform) = camera_query.single_mut() else {
        return;
    };

    // Animation speed (complete in ~0.4 seconds)
    let speed = 2.5;
    camera_anim.progress = (camera_anim.progress + speed * time.delta_secs()).min(1.0);

    // Apply ease-out cubic for smooth deceleration: t' = 1 - (1-t)^3
    let t = camera_anim.progress;
    let eased_t = 1.0 - (1.0 - t).powi(3);

    // Interpolate position and rotation independently
    // Position: linear interpolation (lerp) for smooth straight-line movement
    transform.translation = camera_anim
        .start_position
        .lerp(camera_anim.target_position, eased_t);

    // Rotation: spherical interpolation (slerp) for smooth rotation
    transform.rotation = camera_anim
        .start_rotation
        .slerp(camera_anim.target_rotation, eased_t);

    // Check if animation is complete
    if camera_anim.progress >= 1.0 {
        // Snap to exact target to avoid floating point drift
        transform.translation = camera_anim.target_position;
        transform.rotation = camera_anim.target_rotation;
        camera_anim.animating = false;
    }
}
