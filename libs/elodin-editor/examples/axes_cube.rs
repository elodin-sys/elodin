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
use bevy_fontmesh::prelude::*;
use std::f32::consts::{FRAC_PI_2, PI};
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
        .add_plugins(FontMeshPlugin)
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

/// Face directions using ENU (East-North-Up) convention for Bevy coordinates
/// In Bevy: X=right, Y=up, Z=forward
/// - X axis: East (+X) / West (-X) - Red
/// - Z axis: North (+Z) / South (-Z) - Blue
/// - Y axis: Up (+Y) / Down (-Y) - Green
#[derive(Clone, Copy, Debug)]
enum FaceDirection {
    // X axis (Red) - horizontal
    East, // +X (Right)
    West, // -X (Left)
    // Z axis (Blue) - horizontal (forward/back)
    North, // +Z (Front)
    South, // -Z (Back)
    // Y axis (Green) - vertical
    Up,   // +Y (Top)
    Down, // -Y (Bottom)
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

    // Spawn 3D text labels for ENU faces
    spawn_face_labels(&mut commands, &asset_server, &mut materials);

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

    // Axis colors (ENU convention for Bevy: E=X, N=Z, U=Y)
    // X = East (Red), Y = Up (Green), Z = North (Blue)
    let colors = [
        (Vec3::X, Color::srgb(0.9, 0.2, 0.2), "X"), // Red for X (East)
        (Vec3::Y, Color::srgb(0.2, 0.8, 0.2), "Y"), // Green for Y (Up)
        (Vec3::Z, Color::srgb(0.2, 0.4, 0.9), "Z"), // Blue for Z (North)
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

/// Spawn 3D text labels on cube faces using bevy_fontmesh
fn spawn_face_labels(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    // Load the font
    let font: Handle<FontMesh> = asset_server.load("fonts/Roboto-Bold.ttf");

    // Label configuration
    const LABEL_SCALE: f32 = 0.25; // Scale controls the size
    const LABEL_DEPTH: f32 = 0.08; // Extrusion depth
    const FACE_OFFSET: f32 = 0.52; // Slightly outside the cube face

    // Define face labels - ENU only (East, North, Up)
    // In Bevy: X=right, Y=up, Z=forward
    // ENU mapping: E=+X (red), N=+Z (blue), U=+Y (green)
    let face_labels = [
        // East (+X) - Red axis - on right face
        (
            "E",
            Vec3::new(FACE_OFFSET, 0.0, 0.0),
            Quat::from_rotation_y(FRAC_PI_2),
            Color::srgb(0.9, 0.2, 0.2),
            FaceDirection::East,
        ),
        // North (+Z) - Blue axis - on front face
        (
            "N",
            Vec3::new(0.0, 0.0, FACE_OFFSET),
            Quat::from_rotation_y(PI),
            Color::srgb(0.2, 0.4, 0.9),
            FaceDirection::North,
        ),
        // Up (+Y) - Green axis - on top face
        (
            "U",
            Vec3::new(0.0, FACE_OFFSET, 0.0),
            Quat::from_rotation_x(-FRAC_PI_2),
            Color::srgb(0.2, 0.8, 0.2),
            FaceDirection::Up,
        ),
    ];

    for (text, position, rotation, color, direction) in face_labels {
        let material = materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            // Cull back faces so text is only visible from the front
            // This prevents seeing reversed text through transparent cube faces
            cull_mode: Some(bevy::render::render_resource::Face::Back),
            ..default()
        });

        commands.spawn((
            TextMeshBundle {
                text_mesh: TextMesh {
                    text: text.to_string(),
                    font: font.clone(),
                    style: TextMeshStyle {
                        depth: LABEL_DEPTH,
                        anchor: TextAnchor::Center,
                        ..default()
                    },
                    ..default()
                },
                material: MeshMaterial3d(material),
                transform: Transform::from_translation(position)
                    .with_rotation(rotation)
                    .with_scale(Vec3::splat(LABEL_SCALE)),
                ..default()
            },
            // Add CubeElement so clicking on labels triggers camera rotation
            CubeElement::Face(direction),
            Name::new(format!("label_{}", text)),
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
        // Check if this entity is in the axes cube subtree by traversing parents
        let mut in_subtree = false;
        let mut current = entity;
        loop {
            if current == root {
                in_subtree = true;
                break;
            }
            if let Ok(parent) = parents.get(current) {
                current = parent.0;
            } else {
                // Reached top of hierarchy without finding root
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
    // GLB uses Front/Back/Left/Right/Top/Bottom
    // ENU in Bevy: E=+X, N=+Z, U=+Y
    let dir = match name {
        "Face_Front" => FaceDirection::North, // +Z (front)
        "Face_Back" => FaceDirection::South,  // -Z (back)
        "Face_Left" => FaceDirection::West,   // -X (left)
        "Face_Right" => FaceDirection::East,  // +X (right)
        "Face_Top" => FaceDirection::Up,      // +Y (top)
        "Face_Bottom" => FaceDirection::Down, // -Y (bottom)
        _ => FaceDirection::North,
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

/// Find the nearest ancestor (including self) that has a CubeElement component.
/// Traverses the full parent hierarchy to handle deeply nested mesh entities.
fn find_cube_element_ancestor(
    entity: Entity,
    cube_elements: &Query<&CubeElement>,
    parents_query: &Query<&ChildOf>,
) -> Option<Entity> {
    let mut current = entity;
    loop {
        if cube_elements.get(current).is_ok() {
            return Some(current);
        }
        if let Ok(parent) = parents_query.get(current) {
            current = parent.0;
        } else {
            // Reached top of hierarchy without finding CubeElement
            return None;
        }
    }
}

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

    // Find the CubeElement ancestor (traverses full parent hierarchy)
    let Some(target) = find_cube_element_ancestor(entity, &cube_elements, &parents_query) else {
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

    // Find the CubeElement ancestor (traverses full parent hierarchy)
    let Some(target) = find_cube_element_ancestor(entity, &cube_elements, &parents_query) else {
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

    // Find the CubeElement ancestor (traverses full parent hierarchy)
    let Some(target_entity) = find_cube_element_ancestor(entity, &cube_elements, &parents_query)
    else {
        return;
    };

    let Ok(element) = cube_elements.get(target_entity) else {
        return;
    };

    // Get current camera state
    let Ok(current_transform) = camera_query.single() else {
        return;
    };

    // Calculate target rotation based on element
    let look_dir = get_look_direction(element);
    let up_dir = get_up_direction(element);

    // For faces: check if the face is actually facing the camera
    // This prevents clicking "through" transparent faces to hit the back face
    if let CubeElement::Face(_) = element {
        let camera_dir = current_transform.translation.normalize();
        let face_facing_camera = camera_dir.dot(look_dir) > 0.0;

        if !face_facing_camera {
            // This face is facing away from camera - we clicked through to the back
            // Ignore this click
            println!(
                "CLICK: {:?} -> face not visible from camera, ignoring",
                element
            );
            return;
        }
    }

    // Calculate target: camera should be positioned in the direction of the clicked face
    // so that the face label is visible and facing the camera
    let target_pos = look_dir * CAMERA_DISTANCE;
    let target_transform = Transform::from_translation(target_pos).looking_at(Vec3::ZERO, up_dir);

    // If camera is already at target position/rotation, don't animate
    // Use both position distance AND rotation similarity for robust detection
    let position_similar = current_transform.translation.distance(target_pos) < 0.5;
    let rotation_similar = current_transform
        .rotation
        .dot(target_transform.rotation)
        .abs()
        > 0.99;

    if position_similar && rotation_similar {
        println!("CLICK: {:?} -> already at this view, skipping", element);
        return;
    }

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
            FaceDirection::East => Vec3::X,      // +X (right)
            FaceDirection::West => Vec3::NEG_X,  // -X (left)
            FaceDirection::North => Vec3::Z,     // +Z (front)
            FaceDirection::South => Vec3::NEG_Z, // -Z (back)
            FaceDirection::Up => Vec3::Y,        // +Y (top)
            FaceDirection::Down => Vec3::NEG_Y,  // -Y (bottom)
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

/// Get the up direction for a camera view, ensuring consistent orientation like OnShape
fn get_up_direction(element: &CubeElement) -> Vec3 {
    match element {
        CubeElement::Face(dir) => match dir {
            // Looking from above (+Y), "up" is towards -Z (back of scene)
            FaceDirection::Up => Vec3::NEG_Z,
            // Looking from below (-Y), "up" is towards +Z (front of scene)
            FaceDirection::Down => Vec3::Z,
            // For horizontal faces (E, W, N, S), Y is always up
            FaceDirection::East
            | FaceDirection::West
            | FaceDirection::North
            | FaceDirection::South => Vec3::Y,
        },
        CubeElement::Edge(dir) => {
            // For edges, use Y as up unless it would be parallel to view direction
            let look_dir = match dir {
                EdgeDirection::XTopFront => Vec3::new(0.0, 1.0, 1.0),
                EdgeDirection::XTopBack => Vec3::new(0.0, 1.0, -1.0),
                EdgeDirection::XBottomFront => Vec3::new(0.0, -1.0, 1.0),
                EdgeDirection::XBottomBack => Vec3::new(0.0, -1.0, -1.0),
                EdgeDirection::YFrontLeft => Vec3::new(-1.0, 0.0, 1.0),
                EdgeDirection::YFrontRight => Vec3::new(1.0, 0.0, 1.0),
                EdgeDirection::YBackLeft => Vec3::new(-1.0, 0.0, -1.0),
                EdgeDirection::YBackRight => Vec3::new(1.0, 0.0, -1.0),
                EdgeDirection::ZTopLeft => Vec3::new(-1.0, 1.0, 0.0),
                EdgeDirection::ZTopRight => Vec3::new(1.0, 1.0, 0.0),
                EdgeDirection::ZBottomLeft => Vec3::new(-1.0, -1.0, 0.0),
                EdgeDirection::ZBottomRight => Vec3::new(1.0, -1.0, 0.0),
            }
            .normalize();

            // If Y is nearly parallel to view, use Z as up
            if look_dir.y.abs() > 0.9 {
                if look_dir.y > 0.0 {
                    Vec3::NEG_Z
                } else {
                    Vec3::Z
                }
            } else {
                Vec3::Y
            }
        }
        CubeElement::Corner(pos) => {
            // For corners, check if we're looking mostly up or down
            let look_dir = match pos {
                CornerPosition::TopFrontLeft => Vec3::new(-1.0, 1.0, 1.0),
                CornerPosition::TopFrontRight => Vec3::new(1.0, 1.0, 1.0),
                CornerPosition::TopBackLeft => Vec3::new(-1.0, 1.0, -1.0),
                CornerPosition::TopBackRight => Vec3::new(1.0, 1.0, -1.0),
                CornerPosition::BottomFrontLeft => Vec3::new(-1.0, -1.0, 1.0),
                CornerPosition::BottomFrontRight => Vec3::new(1.0, -1.0, 1.0),
                CornerPosition::BottomBackLeft => Vec3::new(-1.0, -1.0, -1.0),
                CornerPosition::BottomBackRight => Vec3::new(1.0, -1.0, -1.0),
            }
            .normalize();

            // Corners are at 45°, Y works well as up
            // But adjust slightly for top/bottom corners
            if look_dir.y > 0.3 {
                // Top corners: tilt up slightly towards back
                Vec3::new(0.0, 0.7, -0.7).normalize()
            } else if look_dir.y < -0.3 {
                // Bottom corners: tilt up slightly towards front
                Vec3::new(0.0, 0.7, 0.7).normalize()
            } else {
                Vec3::Y
            }
        }
    }
}

// ============================================================================
// Camera Animation
// ============================================================================

/// Spherical linear interpolation for Vec3 directions (for orbital camera movement)
fn slerp_vec3(start: Vec3, end: Vec3, t: f32) -> Vec3 {
    let start_norm = start.normalize();
    let end_norm = end.normalize();

    let dot = start_norm.dot(end_norm).clamp(-1.0, 1.0);
    let theta = dot.acos();

    // If vectors are nearly parallel, use linear interpolation
    if theta.abs() < 0.0001 {
        return start.lerp(end, t);
    }

    let sin_theta = theta.sin();
    let a = ((1.0 - t) * theta).sin() / sin_theta;
    let b = (t * theta).sin() / sin_theta;

    // Interpolate direction and maintain distance
    let start_len = start.length();
    let end_len = end.length();
    let interpolated_len = start_len + (end_len - start_len) * t;

    (start_norm * a + end_norm * b) * interpolated_len
}

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

    // Animation speed (complete in ~0.5 seconds for smoother CAD-like feel)
    let speed = 2.0;
    camera_anim.progress = (camera_anim.progress + speed * time.delta_secs()).min(1.0);

    // Apply ease-in-out cubic for smooth CAD-like movement
    // t' = t < 0.5 ? 4t³ : 1 - (-2t + 2)³ / 2
    let t = camera_anim.progress;
    let eased_t = if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    };

    // Spherical interpolation for orbital movement (like rotating around a sphere)
    transform.translation = slerp_vec3(
        camera_anim.start_position,
        camera_anim.target_position,
        eased_t,
    );

    // Recalculate rotation to always look at center with proper up vector
    // Interpolate the up vector for smooth transitions
    let start_up = camera_anim.start_rotation * Vec3::Y;
    let target_up = camera_anim.target_rotation * Vec3::Y;
    let current_up = slerp_vec3(start_up, target_up, eased_t).normalize();

    // Make camera look at origin with interpolated up vector
    transform.look_at(Vec3::ZERO, current_up);

    // Check if animation is complete
    if camera_anim.progress >= 1.0 {
        // Snap to exact target to avoid floating point drift
        transform.translation = camera_anim.target_position;
        transform.rotation = camera_anim.target_rotation;
        camera_anim.animating = false;
    }
}
