use bevy::prelude::*;
use bevy::pbr::MaterialPlugin;
use bevy::mesh::{SphereKind, SphereMeshBuilder, VertexAttributeValues};

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
        .add_systems(Update, draw_normals)
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
    let sphere = meshes.add(sphere_mesh.clone());

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
    let linear = Mat3::from_cols_array(&[
        d/e, 0.0, 0.0,
        (a*a - b*b)/(e * d),   a*b*e/d, 0.0,
        0.0,   0.0,   c,
    ]);
    // let linear = Mat3::from_cols_array(&[
    //     a, 0.0, 0.0,
    //     0.0,   b, 0.0,
    //     0.0,   0.0,   c,
    // ]);
    let deform = Mat4::from_mat3(linear);

    // New sphere with matrix baked into the mesh (for normals / CPU-deformed geometry).
    let mut deformed_sphere_mesh =
        SphereMeshBuilder::new(1.0, SphereKind::Uv { sectors: 64, stacks: 32 }).build();
    apply_matrix_to_mesh(&mut deformed_sphere_mesh, deform);
    let deformed_sphere = meshes.add(deformed_sphere_mesh);

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

    // Deformed by material (shader) — same unit sphere, deformed at render time.
    commands.spawn((
        Mesh3d(sphere.clone()),
        MeshMaterial3d(material),
        Transform::from_xyz(-1.2, 0.0, 0.0),
    ));

    // Deformed by apply_matrix_to_mesh — normals reflect actual mesh geometry.
    commands.spawn((
        Mesh3d(deformed_sphere),
        MeshMaterial3d(regular_material.clone()),
        Transform::from_xyz(4.2, 0.0, 0.0),
    ));

    // Control sphere: no deformation (plain StandardMaterial).
    commands.spawn((
        Mesh3d(sphere),
        MeshMaterial3d(regular_material),
        Transform::from_xyz(1.2, 0.0, 0.0),
    ));
}

fn draw_normals(
    mut gizmos: Gizmos,
    query: Query<(&Mesh3d, &GlobalTransform)>,
    meshes: Res<Assets<Mesh>>,
) {
    for (mesh_3d, transform) in &query {
        let mesh = if let Some(mesh) = meshes.get(&mesh_3d.0) {
            mesh
        } else {
            continue;
        };

        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(pos)) => pos,
            _ => continue,
        };

        let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(norm)) => norm,
            _ => continue,
        };

        for (position, normal) in positions.iter().zip(normals.iter()) {
            let world_pos = transform.transform_point(Vec3::from(*position));
            let world_normal = transform.rotation() * Vec3::from(*normal);

            gizmos.line(
                world_pos,
                world_pos + world_normal * 0.2,
                Color::srgb(0.0, 1.0, 0.0),
            );
        }
    }
}

fn apply_matrix_to_mesh(mesh: &mut Mesh, m: Mat4) {
    // Positions
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for p in positions.iter_mut() {
            let v = Vec3::from(*p);
            *p = m.transform_point3(v).to_array();
        }
    } else {
        panic!("mesh has no POSITION attribute");
    }

    // Normals (important if your matrix is not pure rotation/scale)
    // For a general linear transform, normals should use inverse-transpose of the 3x3 part.
    let linear = Mat3::from_mat4(m);
    let normal_xform = linear.inverse().transpose();

    if let Some(VertexAttributeValues::Float32x3(normals)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
    {
        for n in normals.iter_mut() {
            let v = Vec3::from(*n);
            *n = normal_xform.mul_vec3(v).normalize().to_array();
        }
    }

    // If you use normal maps/tangents, you may need to recompute tangents after this.
}
