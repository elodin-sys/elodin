//! Test example for axes-cube.glb with hover highlighting and click-to-rotate
//!
//! Run with:
//!   cargo run --example axes_cube -p elodin-editor
//!
//! Features:
//! - Hover over faces, edges, or corners to highlight them in yellow
//! - Click on any element to rotate the camera to look at the cube from that direction

use bevy::asset::{AssetPlugin, RenderAssetUsages};
use bevy::mesh::{Indices, PrimitiveTopology};
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
        // Configure coordinate system here - change to NED if needed
        .insert_resource(CoordinateConfig {
            system: CoordinateSystem::ENU, // <- Change to NED mode here if necessary
            scale: 0.95,                   // <- Adjust global scale here
        })
        .init_resource::<HoveredElement>()
        .init_resource::<OriginalMaterials>()
        .add_systems(Startup, setup)
        .add_systems(Update, (setup_cube_elements, animate_camera))
        .add_observer(on_hover_start)
        .add_observer(on_hover_end)
        .add_observer(on_click)
        .add_observer(on_arrow_hover_start)
        .add_observer(on_arrow_hover_end)
        .add_observer(on_arrow_click)
        .run();
}

// ============================================================================
// Coordinate System Configuration
// ============================================================================

/// Supported coordinate systems
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoordinateSystem {
    /// East-North-Up: X=East, Y=Up, Z=North (aviation/robotics)
    #[default]
    ENU,
    /// North-East-Down: X=North, Y=East, Z=Down (aerospace/navigation)
    NED,
}

/// Resource holding the active coordinate system
#[derive(Resource)]
pub struct CoordinateConfig {
    pub system: CoordinateSystem,
    pub scale: f32, // Global scale factor for the widget
}

impl Default for CoordinateConfig {
    fn default() -> Self {
        Self {
            system: CoordinateSystem::NED,
            scale: 1.5,
        }
    }
}

/// Axis definition with label, direction, and color
#[derive(Clone, Debug)]
pub struct AxisDefinition {
    pub positive_label: &'static str,
    pub negative_label: &'static str,
    pub direction: Vec3,  // Unit vector for positive direction
    pub color: Color,     // Primary color
    pub color_dim: Color, // Dimmed color for negative side
}

impl CoordinateSystem {
    /// Get the three axis definitions for this coordinate system
    /// Returns (primary, secondary, vertical) axes
    pub fn get_axes(&self) -> [AxisDefinition; 3] {
        match self {
            CoordinateSystem::ENU => [
                // X axis = East/West (Red)
                AxisDefinition {
                    positive_label: "East",
                    negative_label: "West",
                    direction: Vec3::X,
                    color: Color::srgb(0.9, 0.2, 0.2),
                    color_dim: Color::srgb(0.6, 0.15, 0.15),
                },
                // Y axis = Up/Down (Green) - Bevy's vertical
                AxisDefinition {
                    positive_label: "Up",
                    negative_label: "Down",
                    direction: Vec3::Y,
                    color: Color::srgb(0.2, 0.8, 0.2),
                    color_dim: Color::srgb(0.15, 0.5, 0.15),
                },
                // Z axis = North/South (Blue) - Bevy's forward
                AxisDefinition {
                    positive_label: "North",
                    negative_label: "South",
                    direction: Vec3::Z,
                    color: Color::srgb(0.2, 0.4, 0.9),
                    color_dim: Color::srgb(0.15, 0.3, 0.6),
                },
            ],
            CoordinateSystem::NED => [
                // NED X axis = North/South (Red) - maps to Bevy X
                AxisDefinition {
                    positive_label: "North",
                    negative_label: "South",
                    direction: Vec3::X,
                    color: Color::srgb(0.9, 0.2, 0.2),
                    color_dim: Color::srgb(0.6, 0.15, 0.15),
                },
                // NED Y axis = East/West (Green) - maps to Bevy Y
                AxisDefinition {
                    positive_label: "East",
                    negative_label: "West",
                    direction: Vec3::Y,
                    color: Color::srgb(0.2, 0.8, 0.2),
                    color_dim: Color::srgb(0.15, 0.5, 0.15),
                },
                // NED Z axis = Down/Up (Blue) - maps to Bevy Z
                AxisDefinition {
                    positive_label: "Down",
                    negative_label: "Up",
                    direction: Vec3::Z,
                    color: Color::srgb(0.2, 0.4, 0.9),
                    color_dim: Color::srgb(0.15, 0.3, 0.6),
                },
            ],
        }
    }

    /// Get all 6 face labels with their properties
    pub fn get_face_labels(&self, face_offset: f32) -> Vec<FaceLabelConfig> {
        let axes = self.get_axes();
        let mut labels = Vec::new();

        for axis in &axes {
            // Positive direction face
            labels.push(FaceLabelConfig {
                text: axis.positive_label,
                position: axis.direction * face_offset,
                rotation: Self::get_rotation_for_direction(axis.direction),
                color: axis.color,
                direction: Self::direction_to_face(axis.direction),
            });
            // Negative direction face
            labels.push(FaceLabelConfig {
                text: axis.negative_label,
                position: -axis.direction * face_offset,
                rotation: Self::get_rotation_for_direction(-axis.direction),
                color: axis.color_dim,
                direction: Self::direction_to_face(-axis.direction),
            });
        }
        labels
    }

    /// Get rotation quaternion to make text face outward from the given direction
    fn get_rotation_for_direction(dir: Vec3) -> Quat {
        // Text mesh faces -Z by default (readable from +Z)
        if dir.x.abs() > 0.9 {
            // X axis faces
            if dir.x > 0.0 {
                Quat::from_rotation_y(FRAC_PI_2)
            } else {
                Quat::from_rotation_y(-FRAC_PI_2)
            }
        } else if dir.y.abs() > 0.9 {
            // Y axis faces
            if dir.y > 0.0 {
                Quat::from_rotation_x(-FRAC_PI_2)
            } else {
                Quat::from_rotation_x(FRAC_PI_2)
            }
        } else {
            // Z axis faces
            if dir.z > 0.0 {
                Quat::IDENTITY
            } else {
                Quat::from_rotation_y(std::f32::consts::PI)
            }
        }
    }

    /// Convert a direction vector to FaceDirection enum
    fn direction_to_face(dir: Vec3) -> FaceDirection {
        if dir.x > 0.5 {
            FaceDirection::East
        } else if dir.x < -0.5 {
            FaceDirection::West
        } else if dir.y > 0.5 {
            FaceDirection::Up
        } else if dir.y < -0.5 {
            FaceDirection::Down
        } else if dir.z > 0.5 {
            FaceDirection::North
        } else {
            FaceDirection::South
        }
    }
}

/// Configuration for a face label
#[derive(Clone, Debug)]
pub struct FaceLabelConfig {
    pub text: &'static str,
    pub position: Vec3,
    pub rotation: Quat,
    pub color: Color,
    pub direction: FaceDirection,
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

/// Face directions - axis-agnostic, based on Bevy coordinates
/// In Bevy: X=right, Y=up, Z=forward
#[derive(Clone, Copy, Debug)]
pub enum FaceDirection {
    East,  // +X (Right)
    West,  // -X (Left)
    North, // +Z (Front)
    South, // -Z (Back)
    Up,    // +Y (Top)
    Down,  // -Y (Bottom)
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

// ============================================================================
// Rotation Arrows
// ============================================================================

/// Direction for rotation arrows
#[derive(Clone, Copy, Debug, Component)]
pub enum RotationArrow {
    Left,
    Right,
}

/// Rotation increment per click (15 degrees)
const ROTATION_INCREMENT: f32 = 15.0 * PI / 180.0;

/// Create a triangle mesh for rotation arrows
fn create_arrow_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    // Triangle pointing up (+Y)
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [0.0, 0.07, 0.0],    // tip
            [-0.04, -0.03, 0.0], // bottom left
            [0.04, -0.03, 0.0],  // bottom right
        ],
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    );
    mesh.insert_indices(Indices::U32(vec![0, 1, 2]));

    mesh
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
    coord_config: Res<CoordinateConfig>,
) {
    commands.init_resource::<CameraAnimation>();

    // Load the axes-cube.glb
    let scene = asset_server.load("axes-cube.glb#Scene0");

    // Spawn the cube with scale from config
    let cube_entity = commands
        .spawn((
            SceneRoot(scene),
            Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(coord_config.scale)),
            AxesCube,
            Name::new("axes_cube_root"),
        ))
        .id();

    // Spawn RGB axes extending from the corner of the cube
    spawn_axes(&mut commands, &mut meshes, &mut materials, &coord_config);

    // Spawn 3D text labels based on coordinate system
    spawn_face_labels(&mut commands, &asset_server, &mut materials, &coord_config);

    // Note: rotation arrows are spawned as children of camera below

    println!("Coordinate System: {:?}", coord_config.system);
    println!("Scale: {}", coord_config.scale);

    // Camera - positioned to see the cube from an isometric-ish angle
    let camera_entity = commands
        .spawn((
            Transform::from_xyz(3.0, 2.5, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
            Camera3d::default(),
            MainCamera,
            Name::new("main_camera"),
        ))
        .id();

    // Spawn rotation arrows as children of camera (fixed on screen)
    spawn_rotation_arrows_on_camera(&mut commands, &mut meshes, &mut materials, camera_entity);

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
    config: &CoordinateConfig,
) {
    // Axis configuration - scaled by global config
    let axis_length = 1.4 * config.scale;
    let axis_radius = 0.035 * config.scale;
    let tip_radius = 0.08 * config.scale;
    let tip_length = 0.2 * config.scale;

    // Origin point - slightly outside the bottom-left-back corner of cube
    let origin = Vec3::new(-0.55, -0.55, -0.55) * config.scale;

    // Get axis colors from coordinate system
    let axes = config.system.get_axes();
    let axis_configs: [(Vec3, Color, &str); 3] = [
        (
            Vec3::X,
            axes.iter()
                .find(|a| a.direction == Vec3::X)
                .map(|a| a.color)
                .unwrap_or(Color::srgb(0.9, 0.2, 0.2)),
            "X",
        ),
        (
            Vec3::Y,
            axes.iter()
                .find(|a| a.direction == Vec3::Y || a.direction == Vec3::NEG_Y)
                .map(|a| a.color)
                .unwrap_or(Color::srgb(0.2, 0.8, 0.2)),
            "Y",
        ),
        (
            Vec3::Z,
            axes.iter()
                .find(|a| a.direction == Vec3::Z)
                .map(|a| a.color)
                .unwrap_or(Color::srgb(0.2, 0.4, 0.9)),
            "Z",
        ),
    ];

    // Create shared meshes
    let shaft_mesh = meshes.add(Cylinder::new(axis_radius, axis_length));
    let tip_mesh = meshes.add(Cone::new(tip_radius, tip_length));

    for (direction, color, name) in axis_configs {
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
        let shaft_pos = origin + direction * (axis_length / 2.0);

        // Spawn shaft
        commands.spawn((
            Mesh3d(shaft_mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(shaft_pos).with_rotation(rotation),
            Pickable::IGNORE, // Don't block picking on the cube
            Name::new(format!("axis_{}_shaft", name)),
        ));

        // Tip position (at the end of the shaft)
        let tip_pos = origin + direction * (axis_length + tip_length / 2.0);

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
            Pickable::IGNORE, // Don't block picking on the cube
            Name::new(format!("axis_{}_tip", name)),
        ));
    }
}

/// Spawn 3D text labels on cube faces using bevy_fontmesh
#[allow(clippy::needless_update)]
fn spawn_face_labels(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    config: &CoordinateConfig,
) {
    // Load the font
    let font: Handle<FontMesh> = asset_server.load("fonts/Roboto-Bold.ttf");

    // Label configuration - scaled by global config
    let label_scale = 0.12 * config.scale;
    let label_depth = 0.05 * config.scale;
    let face_offset = 0.52 * config.scale;

    // Get face labels from coordinate system configuration
    let face_labels = config.system.get_face_labels(face_offset);

    for label in face_labels {
        let material = materials.add(StandardMaterial {
            base_color: label.color,
            unlit: true,
            // Cull back faces so text is only visible from the front
            // This prevents seeing reversed text through transparent cube faces
            cull_mode: Some(bevy::render::render_resource::Face::Back),
            ..default()
        });

        commands.spawn((
            TextMeshBundle {
                text_mesh: TextMesh {
                    text: label.text.to_string(),
                    font: font.clone(),
                    style: TextMeshStyle {
                        depth: label_depth,
                        anchor: TextAnchor::Center,
                        ..default()
                    },
                    ..default()
                },
                material: MeshMaterial3d(material),
                transform: Transform::from_translation(label.position)
                    .with_rotation(label.rotation)
                    .with_scale(Vec3::splat(label_scale)),
                ..default()
            },
            // Add CubeElement so clicking on labels triggers camera rotation
            CubeElement::Face(label.direction),
            Name::new(format!("label_{}", label.text)),
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

/// Spawn left and right rotation arrows as children of camera (fixed on screen)
fn spawn_rotation_arrows_on_camera(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    camera_entity: Entity,
) {
    // Use a simple sphere for visibility testing
    let arrow_mesh = meshes.add(Sphere::new(0.15));

    // Semi-transparent white
    let arrow_color = Color::srgba(1.0, 1.0, 1.0, 0.6);

    // Position in camera local space:
    // X = left/right on screen
    // Y = up/down on screen  
    // Z = depth (negative = in front of camera)
    let screen_distance = 0.8; // Left/right distance
    let depth = -2.0; // In front of camera

    let arrows = [
        (RotationArrow::Left, Vec3::new(-screen_distance, 0.0, depth)),
        (RotationArrow::Right, Vec3::new(screen_distance, 0.0, depth)),
    ];

    for (direction, position) in arrows {
        let material = materials.add(StandardMaterial {
            base_color: arrow_color,
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            ..default()
        });

        // Spawn as child of camera (stays fixed on screen)
        commands
            .spawn((
                Mesh3d(arrow_mesh.clone()),
                MeshMaterial3d(material),
                Transform::from_translation(position),
                direction,
                Name::new(format!("rotation_arrow_{:?}", direction)),
            ))
            .insert(ChildOf(camera_entity));
    }

    println!("Spawned rotation arrows on camera (fixed on screen)");
}

/// Set up cube elements after the GLB is loaded
/// Also clones materials so each element can be highlighted independently
#[allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::collapsible_if
)]
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
                        if materials.get(&mat_handle.0).is_some() {
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

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::collapsible_if)]
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
            original_materials
                .colors
                .entry(entity)
                .or_insert(mat.base_color);
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
                    original_materials
                        .colors
                        .entry(child)
                        .or_insert(mat.base_color);
                    mat.base_color = HIGHLIGHT_COLOR;
                    mat.emissive = LinearRgba::new(0.5, 0.45, 0.1, 1.0);
                }
            }
        }
    }
}

#[allow(clippy::collapsible_if)]
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
    let mut look_dir = get_look_direction(element);

    // For faces: if we clicked on a face that's NOT facing the camera,
    // it means we clicked "through" the transparent cube to the back face.
    // In that case, flip to the opposite face (what the user actually intended).
    if let CubeElement::Face(dir) = element {
        let camera_dir = current_transform.translation.normalize();
        let face_facing_camera = camera_dir.dot(look_dir) > 0.0;

        if !face_facing_camera {
            // Clicked through to back face - go to opposite face instead
            look_dir = -look_dir;
            println!(
                "CLICK: {:?} -> clicked through, going to opposite face",
                dir
            );
        }
    }

    let up_dir = get_up_direction_for_look(look_dir);

    // Calculate target: camera should be positioned in the direction of the clicked element
    // so that the element is visible and facing the camera
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

/// Get the up direction for a given look direction (used when look_dir might be flipped)
fn get_up_direction_for_look(look_dir: Vec3) -> Vec3 {
    // If looking up or down (Y axis), use Z for up
    if look_dir.y.abs() > 0.9 {
        if look_dir.y > 0.0 {
            Vec3::NEG_Z // Looking up, "up" is towards -Z
        } else {
            Vec3::Z // Looking down, "up" is towards +Z
        }
    } else {
        // For horizontal views, Y is always up
        Vec3::Y
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

// ============================================================================
// Rotation Arrow Handlers
// ============================================================================

fn on_arrow_hover_start(
    trigger: On<Pointer<Over>>,
    arrows: Query<&RotationArrow>,
    materials_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let entity = trigger.event().event_target();

    // Only handle rotation arrows
    if arrows.get(entity).is_err() {
        return;
    }

    if let Ok(mat_handle) = materials_query.get(entity)
        && let Some(mat) = materials.get_mut(&mat_handle.0)
    {
        mat.base_color = Color::srgba(1.0, 1.0, 0.3, 0.9); // Bright yellow
    }
}

fn on_arrow_hover_end(
    trigger: On<Pointer<Out>>,
    arrows: Query<&RotationArrow>,
    materials_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let entity = trigger.event().event_target();

    // Only handle rotation arrows
    if arrows.get(entity).is_err() {
        return;
    }

    if let Ok(mat_handle) = materials_query.get(entity)
        && let Some(mat) = materials.get_mut(&mat_handle.0)
    {
        mat.base_color = Color::srgba(1.0, 1.0, 1.0, 0.5); // Back to semi-transparent white
    }
}

fn on_arrow_click(
    trigger: On<Pointer<Click>>,
    arrows: Query<&RotationArrow>,
    mut camera_query: Query<&mut Transform, With<MainCamera>>,
) {
    let entity = trigger.event().event_target();

    // Only handle rotation arrows
    let Ok(arrow) = arrows.get(entity) else {
        return;
    };

    // Only left click
    if trigger.button != PointerButton::Primary {
        return;
    }

    let Ok(mut transform) = camera_query.single_mut() else {
        return;
    };

    // Yaw rotation around world Y axis
    let sign = match arrow {
        RotationArrow::Left => 1.0,
        RotationArrow::Right => -1.0,
    };

    let rotation = Quat::from_rotation_y(sign * ROTATION_INCREMENT);

    // Rotate around the origin (where the cube is)
    transform.rotate_around(Vec3::ZERO, rotation);

    println!("Arrow click: {:?}, rotating {}°", arrow, sign * 15.0);
}
