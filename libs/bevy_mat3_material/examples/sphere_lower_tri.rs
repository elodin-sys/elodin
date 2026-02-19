use bevy::prelude::*;
use bevy::pbr::MaterialPlugin;
use bevy::mesh::{SphereKind, SphereMeshBuilder};

use bevy_lower_tri_material::{params_from_linear, LowerTriMaterial, LowerTriTransformExt};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(MaterialPlugin::<LowerTriMaterial>::default())
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LowerTriMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 1.8, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
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
    let a = 0.35;
    let b = -0.20;
    let c = 0.45;
    let linear = Mat3::from_cols_array(&[
        1.0, 0.0, 0.0,
        a,   1.0, 0.0,
        b,   c,   1.0,
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

    commands.spawn((
        Mesh3d(sphere),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}
