use bevy::prelude::*;
use bevy::pbr::MaterialPlugin;
use bevy::mesh::{SphereKind, SphereMeshBuilder};

use bevy_lower_tri_material::{params_from_linear, LowerTriMaterial, LowerTriTransformExt};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(bevy_inspector_egui::bevy_egui::EguiPlugin::default())
        .add_plugins(bevy_inspector_egui::quick::WorldInspectorPlugin::new())
        .add_plugins(bevy_editor_cam::DefaultEditorCamPlugins)
        .add_plugins(MaterialPlugin::<LowerTriMaterial>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, draw_axes_gizmos)
        .run();
}

fn draw_axes_gizmos(mut gizmos: Gizmos) {
    let origin = Vec3::ZERO;
    let len = 2.0;
    gizmos.line(origin, origin + Vec3::X * len, Color::srgb(1.0, 0.0, 0.0)); // X = red
    gizmos.line(origin, origin + Vec3::Y * len, Color::srgb(0.0, 1.0, 0.0)); // Y = green
    gizmos.line(origin, origin + Vec3::Z * len, Color::srgb(0.0, 0.0, 1.0)); // Z = blue
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LowerTriMaterial>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 1.8, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        bevy_editor_cam::controller::component::EditorCam::default(),
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(2.0, 4.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // --- Unit sphere mesh via SphereMeshBuilder (radius = 1.0) ---
    let sphere_mesh = SphereMeshBuilder::new(1.0, SphereKind::Uv { sectors: 64, stacks: 32 }).build();
    let sphere = meshes.add(sphere_mesh);

    // --- Lower-triangular 3x3 (example) ---
    // [ 1   0   0 ]
    // [ a   1   0 ]
    // [ b   c   1 ]
    fn sqrt(x: f32) -> f32 {
        x.sqrt()
    }
    let a = 1.35;
    let b = 0.40;
    let c = 0.25;
    let d = sqrt(a*a + b*b);
    let e = sqrt(2.0);
    // let linear = Mat3::from_cols_array(&[
    //     d/e, 0.0, 0.0,
    //     (a*a - b*b)/(e * d),   a*b*e/d, 0.0,
    //     0.0,   0.0,   c,
    // ]);
    let linear = Mat3::from_cols_array(&[
        a, 0.0, 0.0,
        0.0,   b, 0.0,
        0.0,   0.0,   c,
    ]);

    let params = params_from_linear(linear);

    // StandardMaterial base controls PBR appearance; our extension only modifies the vertex positions/normals.
    let material = materials.add(LowerTriMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.2, 0.6, 0.9),
            perceptual_roughness: 0.35,
            metallic: 0.05,
            ..default()
        },
        extension: LowerTriTransformExt { params },
    });

    let regular_material = standard_materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.4, 0.2),
        perceptual_roughness: 0.35,
        metallic: 0.05,
        ..default()
    });

    commands.spawn((
        Mesh3d(sphere.clone()),
        MeshMaterial3d(material),
        Transform::from_xyz(-1.2, 0.0, 0.0),
    ));

    // Control sphere: no deformation (plain StandardMaterial).
    commands.spawn((
        Mesh3d(sphere),
        MeshMaterial3d(regular_material),
        Transform::from_xyz(1.2, 0.0, 0.0),
    ));
}
