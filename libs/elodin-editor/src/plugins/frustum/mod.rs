use crate::{MainCamera, ui::tiles::ViewportConfig};
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::transform::TransformSystems;
use std::collections::{HashMap, HashSet};

/// Squared distance below which two cameras are considered overlapping.
/// Frustum pairs are skipped when cameras are this close, which prevents the
/// visual glitch at startup when all viewports share the same default position.
const MIN_FRUSTUM_CAMERA_DISTANCE_SQ: f32 = 0.01;

type MainViewportQueryItem = (
    Entity,
    &'static Camera,
    &'static Projection,
    &'static GlobalTransform,
    Option<&'static ViewportConfig>,
);

pub struct FrustumPlugin;

impl Plugin for FrustumPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrustumMaterialCache>()
            .add_systems(Startup, frustum_mesh_setup)
            .add_systems(
                PostUpdate,
                draw_viewport_frustums.after(TransformSystems::Propagate),
            );
    }
}

#[derive(Resource, Clone)]
struct FrustumLineAssets {
    edge_mesh: Handle<Mesh>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct FrustumMaterialKey {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

#[derive(Resource, Default)]
struct FrustumMaterialCache {
    materials: HashMap<FrustumMaterialKey, Handle<StandardMaterial>>,
}

#[derive(Component, Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CameraFrustumRootVisual {
    source: Entity,
    target: Entity,
}

#[derive(Component, Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CameraFrustumLineVisual {
    source: Entity,
    target: Entity,
    segment: u8,
}

#[derive(Component, Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CameraFrustumFaceVisual {
    source: Entity,
    target: Entity,
}

const FRUSTUM_FACE_ALPHA: u8 = 10;

fn frustum_face_material_for_color(
    color: impeller2_wkt::Color,
    materials: &mut Assets<StandardMaterial>,
    cache: &mut FrustumMaterialCache,
) -> Handle<StandardMaterial> {
    let key = FrustumMaterialKey {
        r: color_component_to_u8(color.r),
        g: color_component_to_u8(color.g),
        b: color_component_to_u8(color.b),
        a: FRUSTUM_FACE_ALPHA,
    };
    if let Some(handle) = cache.materials.get(&key) {
        return handle.clone();
    }

    let material = materials.add(StandardMaterial {
        base_color: Color::srgba_u8(key.r, key.g, key.b, FRUSTUM_FACE_ALPHA),
        emissive: Color::srgb_u8(key.r, key.g, key.b).into(),
        cull_mode: None,
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        depth_bias: -4.0,
        ..Default::default()
    });
    cache.materials.insert(key, material.clone());
    material
}

fn build_frustum_face_mesh(points: &[Vec3; 8]) -> Mesh {
    let quads: [[usize; 4]; 6] = [
        [0, 1, 2, 3], // near
        [5, 4, 7, 6], // far
        [4, 0, 3, 7], // left
        [1, 5, 6, 2], // right
        [4, 5, 1, 0], // top
        [3, 2, 6, 7], // bottom
    ];

    let mut positions = Vec::with_capacity(36);
    let mut normals = Vec::with_capacity(36);

    for quad in &quads {
        let a = points[quad[0]];
        let b = points[quad[1]];
        let c = points[quad[2]];
        let d = points[quad[3]];
        let n = (b - a).cross(d - a).normalize_or_zero();
        let nn = [n.x, n.y, n.z];
        positions.push([a.x, a.y, a.z]);
        positions.push([b.x, b.y, b.z]);
        positions.push([c.x, c.y, c.z]);
        normals.push(nn);
        normals.push(nn);
        normals.push(nn);
        positions.push([a.x, a.y, a.z]);
        positions.push([c.x, c.y, c.z]);
        positions.push([d.x, d.y, d.z]);
        normals.push(nn);
        normals.push(nn);
        normals.push(nn);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh
}

#[derive(SystemParam)]
struct FrustumDrawParams<'w, 's> {
    main_viewports: Query<'w, 's, MainViewportQueryItem, With<MainCamera>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    meshes: ResMut<'w, Assets<Mesh>>,
    material_cache: ResMut<'w, FrustumMaterialCache>,
    line_assets: Option<Res<'w, FrustumLineAssets>>,
    existing_roots: Query<'w, 's, (Entity, &'static CameraFrustumRootVisual)>,
    existing_lines: Query<'w, 's, (Entity, &'static CameraFrustumLineVisual)>,
    existing_faces: Query<'w, 's, (Entity, &'static CameraFrustumFaceVisual, &'static Mesh3d)>,
}

fn frustum_mesh_setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let edge_mesh = meshes.add(Mesh::from(Cylinder::new(1.0, 1.0)));
    commands.insert_resource(FrustumLineAssets { edge_mesh });
}

fn color_component_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn frustum_material_for_color(
    color: impeller2_wkt::Color,
    materials: &mut Assets<StandardMaterial>,
    cache: &mut FrustumMaterialCache,
) -> Handle<StandardMaterial> {
    let key = FrustumMaterialKey {
        r: color_component_to_u8(color.r),
        g: color_component_to_u8(color.g),
        b: color_component_to_u8(color.b),
        a: color_component_to_u8(color.a),
    };
    if let Some(handle) = cache.materials.get(&key) {
        return handle.clone();
    }

    let alpha = key.a as f32 / 255.0;
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba_u8(key.r, key.g, key.b, key.a),
        emissive: Color::srgb_u8(key.r, key.g, key.b).into(),
        cull_mode: None,
        unlit: true,
        alpha_mode: if alpha < 1.0 {
            AlphaMode::Blend
        } else {
            AlphaMode::Opaque
        },
        depth_bias: -8.0,
        ..Default::default()
    });
    cache.materials.insert(key, material.clone());
    material
}

fn frustum_local_points(perspective: &PerspectiveProjection) -> Option<[Vec3; 8]> {
    let near = perspective.near;
    let far = perspective.far;
    let fov = perspective.fov;
    let aspect = perspective.aspect_ratio;
    if !(near > 0.0 && far > near && fov > 0.0 && aspect > 0.0) {
        return None;
    }

    let tan_half = (fov * 0.5).tan();
    let near_half_height = tan_half * near;
    let near_half_width = near_half_height * aspect;
    let far_half_height = tan_half * far;
    let far_half_width = far_half_height * aspect;

    Some([
        Vec3::new(-near_half_width, near_half_height, -near),
        Vec3::new(near_half_width, near_half_height, -near),
        Vec3::new(near_half_width, -near_half_height, -near),
        Vec3::new(-near_half_width, -near_half_height, -near),
        Vec3::new(-far_half_width, far_half_height, -far),
        Vec3::new(far_half_width, far_half_height, -far),
        Vec3::new(far_half_width, -far_half_height, -far),
        Vec3::new(-far_half_width, -far_half_height, -far),
    ])
}

fn frustum_segments(points: [Vec3; 8]) -> [(Vec3, Vec3); 12] {
    [
        (points[0], points[1]),
        (points[1], points[2]),
        (points[2], points[3]),
        (points[3], points[0]),
        (points[4], points[5]),
        (points[5], points[6]),
        (points[6], points[7]),
        (points[7], points[4]),
        (points[0], points[4]),
        (points[1], points[5]),
        (points[2], points[6]),
        (points[3], points[7]),
    ]
}

fn frustum_segment_transform_local(
    start_local: Vec3,
    end_local: Vec3,
    thickness: f32,
) -> Option<Transform> {
    let delta = end_local - start_local;
    let length = delta.length();
    if length <= 1.0e-5_f32 || thickness <= 0.0 {
        return None;
    }

    let rotation = Quat::from_rotation_arc(Vec3::Y, delta / length);
    let midpoint_local = start_local + delta * 0.5;

    Some(Transform {
        translation: midpoint_local,
        rotation,
        scale: Vec3::new(thickness, length, thickness),
    })
}

fn cleanup_frustum_entities(params: &FrustumDrawParams<'_, '_>, commands: &mut Commands) {
    for (entity, _) in params.existing_lines.iter() {
        commands.entity(entity).despawn();
    }
    for (entity, _, _) in params.existing_faces.iter() {
        commands.entity(entity).despawn();
    }
    for (entity, _) in params.existing_roots.iter() {
        commands.entity(entity).despawn();
    }
}

fn draw_viewport_frustums(mut params: FrustumDrawParams<'_, '_>, mut commands: Commands) {
    let Some(line_assets) = params.line_assets.as_ref() else {
        cleanup_frustum_entities(&params, &mut commands);
        return;
    };

    let mut sources = Vec::new();
    let mut targets = Vec::new();
    let mut camera_positions: HashMap<Entity, Vec3> = HashMap::new();

    for (camera_entity, camera, projection, global_transform, config) in
        params.main_viewports.iter()
    {
        if !camera.is_active {
            continue;
        }

        let Some(config) = config else {
            continue;
        };

        let Some(viewport_layer) = config.viewport_layer else {
            continue;
        };

        camera_positions.insert(camera_entity, global_transform.translation());
        if config.show_frustums {
            targets.push((camera_entity, RenderLayers::layer(viewport_layer)));
        }

        if !config.create_frustum {
            continue;
        }

        let Projection::Perspective(perspective) = projection else {
            continue;
        };
        let Some(points) = frustum_local_points(perspective) else {
            continue;
        };
        sources.push((
            camera_entity,
            points,
            config.frustums_color,
            config.frustums_thickness,
        ));
    }

    if sources.is_empty() || targets.is_empty() {
        cleanup_frustum_entities(&params, &mut commands);
        return;
    }

    let mut desired_roots: HashSet<CameraFrustumRootVisual> = HashSet::new();
    let mut desired_segments: HashMap<
        CameraFrustumLineVisual,
        (
            CameraFrustumRootVisual,
            Transform,
            RenderLayers,
            MeshMaterial3d<StandardMaterial>,
        ),
    > = HashMap::new();

    struct DesiredFace {
        key: CameraFrustumFaceVisual,
        root_key: CameraFrustumRootVisual,
        mesh: Mesh,
        render_layers: RenderLayers,
        material: MeshMaterial3d<StandardMaterial>,
    }
    let mut desired_faces: Vec<DesiredFace> = Vec::new();

    for (source_camera, points, color, thickness) in sources {
        let material =
            frustum_material_for_color(color, &mut params.materials, &mut params.material_cache);
        let face_material = frustum_face_material_for_color(
            color,
            &mut params.materials,
            &mut params.material_cache,
        );
        let segments = frustum_segments(points);
        for (target_camera, render_layers) in &targets {
            if source_camera == *target_camera {
                continue;
            }
            if let (Some(&src_pos), Some(&tgt_pos)) = (
                camera_positions.get(&source_camera),
                camera_positions.get(target_camera),
            ) && (src_pos - tgt_pos).length_squared() < MIN_FRUSTUM_CAMERA_DISTANCE_SQ
            {
                continue;
            }
            let root_key = CameraFrustumRootVisual {
                source: source_camera,
                target: *target_camera,
            };
            desired_roots.insert(root_key);

            for (segment_idx, (start_local, end_local)) in segments.iter().enumerate() {
                let Some(local_transform) =
                    frustum_segment_transform_local(*start_local, *end_local, thickness)
                else {
                    continue;
                };
                desired_segments.insert(
                    CameraFrustumLineVisual {
                        source: source_camera,
                        target: *target_camera,
                        segment: segment_idx as u8,
                    },
                    (
                        root_key,
                        local_transform,
                        render_layers.clone(),
                        MeshMaterial3d(material.clone()),
                    ),
                );
            }

            desired_faces.push(DesiredFace {
                key: CameraFrustumFaceVisual {
                    source: source_camera,
                    target: *target_camera,
                },
                root_key,
                mesh: build_frustum_face_mesh(&points),
                render_layers: render_layers.clone(),
                material: MeshMaterial3d(face_material.clone()),
            });
        }
    }

    let mut existing_roots_by_key: HashMap<CameraFrustumRootVisual, Entity> = HashMap::new();
    for (entity, key) in params.existing_roots.iter() {
        existing_roots_by_key.insert(*key, entity);
    }

    let mut root_entities: HashMap<CameraFrustumRootVisual, Entity> = HashMap::new();
    for key in &desired_roots {
        if let Some(entity) = existing_roots_by_key.remove(key) {
            commands.entity(entity).insert(ChildOf(key.source));
            root_entities.insert(*key, entity);
            continue;
        }

        let root_entity = commands
            .spawn((
                Transform::default(),
                GlobalTransform::default(),
                CameraFrustumRootVisual {
                    source: key.source,
                    target: key.target,
                },
                ChildOf(key.source),
                Name::new("viewport_frustum_root"),
            ))
            .id();
        root_entities.insert(*key, root_entity);
    }

    for entity in existing_roots_by_key.into_values() {
        commands.entity(entity).despawn();
    }

    let mut existing_lines_by_key: HashMap<CameraFrustumLineVisual, Entity> = HashMap::new();
    for (entity, key) in params.existing_lines.iter() {
        existing_lines_by_key.insert(*key, entity);
    }

    for (key, (root_key, transform, render_layers, material)) in desired_segments {
        let Some(&root_entity) = root_entities.get(&root_key) else {
            continue;
        };

        if let Some(entity) = existing_lines_by_key.remove(&key) {
            commands.entity(entity).insert((
                transform,
                render_layers,
                material,
                ChildOf(root_entity),
            ));
            continue;
        }

        commands.spawn((
            Mesh3d(line_assets.edge_mesh.clone()),
            material,
            transform,
            GlobalTransform::default(),
            render_layers,
            NoFrustumCulling,
            CameraFrustumLineVisual {
                source: key.source,
                target: key.target,
                segment: key.segment,
            },
            ChildOf(root_entity),
            Name::new("viewport_frustum_segment"),
        ));
    }

    for entity in existing_lines_by_key.into_values() {
        commands.entity(entity).despawn();
    }

    let mut existing_faces_by_key: HashMap<CameraFrustumFaceVisual, (Entity, Handle<Mesh>)> =
        HashMap::new();
    for (entity, key, mesh3d) in params.existing_faces.iter() {
        existing_faces_by_key.insert(*key, (entity, mesh3d.0.clone()));
    }

    for face in desired_faces {
        let Some(&root_entity) = root_entities.get(&face.root_key) else {
            continue;
        };

        if let Some((entity, mesh_handle)) = existing_faces_by_key.remove(&face.key) {
            if let Some(mesh_asset) = params.meshes.get_mut(&mesh_handle) {
                *mesh_asset = face.mesh;
            } else {
                let new_mesh = params.meshes.add(face.mesh);
                commands.entity(entity).insert(Mesh3d(new_mesh));
            }
            commands.entity(entity).insert((
                face.render_layers,
                face.material,
                ChildOf(root_entity),
            ));
            continue;
        }

        let mesh_handle = params.meshes.add(face.mesh);
        commands.spawn((
            Mesh3d(mesh_handle),
            face.material,
            Transform::IDENTITY,
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
            face.render_layers,
            NoFrustumCulling,
            face.key,
            ChildOf(root_entity),
            Name::new("viewport_frustum_face"),
        ));
    }

    for (entity, _) in existing_faces_by_key.into_values() {
        commands.entity(entity).despawn();
    }
}
