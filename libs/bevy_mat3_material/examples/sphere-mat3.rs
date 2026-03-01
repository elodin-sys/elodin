use bevy::mesh::{Mesh, SphereKind, SphereMeshBuilder, VertexAttributeValues};
use bevy::prelude::*;
use bevy::render::alpha::AlphaMode;

use bevy_mat3_material::{
    uv_sphere_grid_line_mesh, Mat3Material, Mat3MaterialPlugin, Mat3Params, Mat3TransformExt,
    Mat3Uniforms,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Mat3MaterialPlugin)
        .add_plugins(bevy_inspector_egui::bevy_egui::EguiPlugin::default())
        .add_plugins(bevy_inspector_egui::quick::WorldInspectorPlugin::new())
        .add_plugins(bevy_editor_cam::DefaultEditorCamPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, draw_axes_gizmos)
        .add_systems(Update, draw_mesh_normals)
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
    mut materials: ResMut<Assets<Mat3Material>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
    // let sectors = 64;
    // let stacks = 32;

    let sectors = 20;
    let stacks = 10;
    let transparent = true;

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(1.2, 0.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
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
    let sphere_mesh = SphereMeshBuilder::new(1.0, SphereKind::Uv { sectors, stacks }).build();
    let sphere = meshes.add(sphere_mesh.clone());

    fn sqrt(x: f32) -> f32 {
        x.sqrt()
    }

    let a = 1.35;
    let b = 0.40;
    let c = 0.25;
    let d = sqrt(a * a + b * b);
    let e = sqrt(2.0);
    #[rustfmt::skip]
    let linear = Mat3::from_cols_array(&[
        d/e,                 0.0,     0.0,
        (a*a - b*b)/(e * d), a*b*e/d, 0.0,
        0.0,                 0.0,     c,
    ]);
    #[rustfmt::skip]
    let linear2 = Mat3::from_cols_array(&[
        a, 0.0, 0.0,
        0.0,   b, 0.0,
        0.0,   0.0,   c,
    ]);

    // New sphere with matrix baked into the mesh (for normals / CPU-deformed geometry).
    let mut deformed_sphere_mesh =
        SphereMeshBuilder::new(1.0, SphereKind::Uv { sectors, stacks }).build();
    apply_matrix_to_mesh(&mut deformed_sphere_mesh, Mat4::from_mat3(linear2));
    let deformed_sphere = meshes.add(deformed_sphere_mesh);

    let params = linear.into();
    let params2 = linear2.into();

    let (base_color_deformed, regular_color, deformed_color, alpha_mode) = if transparent {
        let alpha = 0.35;
        (
            Color::srgba(0.2, 0.6, 0.9, alpha),
            Color::srgba(0.9, 0.4, 0.2, alpha),
            Color::srgba(0.4, 0.9, 0.2, alpha),
            AlphaMode::Blend,
        )
    } else {
        (
            Color::srgb(0.2, 0.6, 0.9),
            Color::srgb(0.9, 0.4, 0.2),
            Color::srgb(0.4, 0.9, 0.2),
            AlphaMode::Opaque,
        )
    };

    // Material for "deformed by shader" — unique handle so Mat3Params can drive it.
    let material = materials.add(Mat3Material {
        base: StandardMaterial {
            base_color: base_color_deformed,
            perceptual_roughness: 0.35,
            metallic: 0.05,
            alpha_mode,
            ..default()
        },
        extension: Mat3TransformExt { params },
    });

    let regular_material = standard_materials.add(StandardMaterial {
        base_color: regular_color,
        perceptual_roughness: 0.35,
        metallic: 0.05,
        alpha_mode,
        ..default()
    });

    let deformed_material = standard_materials.add(StandardMaterial {
        base_color: deformed_color,
        perceptual_roughness: 0.35,
        metallic: 0.05,
        alpha_mode,
        ..default()
    });

    // Single unit-sphere grid mesh; deformation is done in the vertex shader via Mat3Material.
    let grid_mesh = meshes.add(uv_sphere_grid_line_mesh(1.0, sectors, stacks));

    let grid_material_deformed = materials.add(Mat3Material {
        base: StandardMaterial {
            base_color: Color::srgba(0., 0., 0., 1.0),
            unlit: true,
            ..default()
        },
        extension: Mat3TransformExt { params },
    });

    let grid_material_deformed2 = materials.add(Mat3Material {
        base: StandardMaterial {
            base_color: Color::srgba(0., 0., 0., 1.0),
            unlit: true,
            ..default()
        },
        extension: Mat3TransformExt { params: params2 },
    });
    let grid_material_unit = materials.add(Mat3Material {
        base: StandardMaterial {
            base_color: Color::srgba(0., 0., 0., 1.0),
            unlit: true,
            ..default()
        },
        extension: Mat3TransformExt {
            params: Mat3Uniforms::default(),
        },
    });

    let shadow_receiver = true;
    // Deformed by material (shader) — params editable via Mat3Params in the inspector.
    commands
        .spawn((
            Mesh3d(sphere.clone()),
            MeshMaterial3d(material),
            Mat3Params { linear },
            Transform::from_xyz(-1.2, 0.0, 0.0),
            Name::new("deformed by shader"),
        ))
        .insert_if(bevy::light::NotShadowReceiver, || !shadow_receiver)
        .with_children(|commands| {
            commands.spawn((
                Mesh3d(grid_mesh.clone()),
                MeshMaterial3d(grid_material_deformed),
                bevy::light::NotShadowReceiver,
                bevy::light::NotShadowCaster,
                Name::new("grid (deformed by shader)"),
            ));
        });

    // Deformed by apply_matrix_to_mesh — normals reflect actual mesh geometry.
    commands
        .spawn((
            Mesh3d(deformed_sphere),
            MeshMaterial3d(deformed_material),
            Transform::from_xyz(4.2, 0.0, 0.0),
            Name::new("deformed mesh"),
        ))
        .insert_if(bevy::light::NotShadowReceiver, || !shadow_receiver)
        .with_children(|commands| {
            commands.spawn((
                Mesh3d(grid_mesh.clone()),
                MeshMaterial3d(grid_material_deformed2.clone()),
                bevy::light::NotShadowReceiver,
                bevy::light::NotShadowCaster,
                Name::new("grid (deformed mesh)"),
            ));
        });

    // Control sphere: no deformation (plain StandardMaterial).
    commands
        .spawn((
            Mesh3d(sphere),
            MeshMaterial3d(regular_material),
            Transform::from_xyz(1.2, 0.0, 0.0),
            Name::new("control"),
        ))
        .insert_if(bevy::light::NotShadowReceiver, || !shadow_receiver)
        .with_children(|commands| {
            commands.spawn((
                Mesh3d(grid_mesh),
                MeshMaterial3d(grid_material_unit),
                bevy::light::NotShadowReceiver,
                bevy::light::NotShadowCaster,
                Name::new("grid (control)"),
            ));
        });
}

#[allow(dead_code)]
fn draw_mesh_normals(
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

/// Transform a mesh's positions and normals.
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
    let normal_transform = linear.inverse().transpose();

    if let Some(VertexAttributeValues::Float32x3(normals)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
    {
        for n in normals.iter_mut() {
            let v = Vec3::from(*n);
            *n = normal_transform.mul_vec3(v).normalize().to_array();
        }
    }

    // If you use normal maps/tangents, you may need to recompute tangents after this.
}
