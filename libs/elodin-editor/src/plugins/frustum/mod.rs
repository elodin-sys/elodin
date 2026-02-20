use crate::{MainCamera, ui::tiles::ViewportConfig};
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use std::collections::HashMap;

pub struct FrustumPlugin;

impl Plugin for FrustumPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrustumMaterialCache>()
            .add_systems(Startup, frustum_mesh_setup)
            .add_systems(PreUpdate, draw_viewport_frustums);
    }
}

const SELF_VIEW_XY_INSET: f32 = 0.985;
const SELF_VIEW_NEAR_DEPTH_SCALE: f32 = 1.02;
const SELF_VIEW_FAR_DEPTH_SCALE: f32 = 0.998;

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

#[derive(SystemParam)]
struct FrustumDrawParams<'w, 's> {
    main_viewports: Query<
        'w,
        's,
        (
            Entity,
            &'static Camera,
            &'static Projection,
            Option<&'static ViewportConfig>,
        ),
        With<MainCamera>,
    >,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    material_cache: ResMut<'w, FrustumMaterialCache>,
    line_assets: Option<Res<'w, FrustumLineAssets>>,
    existing_roots: Query<'w, 's, (Entity, &'static CameraFrustumRootVisual)>,
    existing_lines: Query<'w, 's, (Entity, &'static CameraFrustumLineVisual)>,
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

fn frustum_points_for_target(points: [Vec3; 8], is_self_view: bool) -> [Vec3; 8] {
    if !is_self_view {
        return points;
    }

    let mut adjusted = points;
    for point in adjusted.iter_mut().take(4) {
        point.x *= SELF_VIEW_XY_INSET;
        point.y *= SELF_VIEW_XY_INSET;
        point.z *= SELF_VIEW_NEAR_DEPTH_SCALE;
    }
    for point in adjusted.iter_mut().skip(4) {
        point.x *= SELF_VIEW_XY_INSET;
        point.y *= SELF_VIEW_XY_INSET;
        point.z *= SELF_VIEW_FAR_DEPTH_SCALE;
    }
    adjusted
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

    for (camera_entity, camera, projection, config) in params.main_viewports.iter() {
        if !camera.is_active {
            continue;
        }

        let Some(config) = config else {
            continue;
        };

        let Some(viewport_layer) = config.viewport_layer else {
            continue;
        };

        targets.push((camera_entity, RenderLayers::layer(viewport_layer)));

        if !config.show_frustums {
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

    let mut desired_roots: HashMap<CameraFrustumRootVisual, ()> = HashMap::new();
    let mut desired_segments: HashMap<
        CameraFrustumLineVisual,
        (
            CameraFrustumRootVisual,
            Transform,
            RenderLayers,
            MeshMaterial3d<StandardMaterial>,
        ),
    > = HashMap::new();

    for (source_camera, points, color, thickness) in sources {
        let material =
            frustum_material_for_color(color, &mut params.materials, &mut params.material_cache);
        for (target_camera, render_layers) in &targets {
            let segments = frustum_segments(frustum_points_for_target(
                points,
                source_camera == *target_camera,
            ));
            let root_key = CameraFrustumRootVisual {
                source: source_camera,
                target: *target_camera,
            };
            desired_roots.insert(root_key, ());

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
        }
    }

    let mut existing_roots_by_key: HashMap<CameraFrustumRootVisual, Entity> = HashMap::new();
    for (entity, key) in params.existing_roots.iter() {
        existing_roots_by_key.insert(*key, entity);
    }

    let mut root_entities: HashMap<CameraFrustumRootVisual, Entity> = HashMap::new();
    for key in desired_roots.keys() {
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
}
