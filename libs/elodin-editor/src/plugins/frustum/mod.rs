use crate::{MainCamera, ui::tiles::ViewportConfig};
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use std::collections::HashMap;

pub struct FrustumPlugin;

impl Plugin for FrustumPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, frustum_mesh_setup)
            .add_systems(PreUpdate, draw_viewport_frustums);
    }
}

const FRUSTUM_EDGE_RADIUS: f32 = 0.006;

#[derive(Resource, Clone)]
struct FrustumLineAssets {
    edge_mesh: Handle<Mesh>,
    edge_material: Handle<StandardMaterial>,
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
    line_assets: Option<Res<'w, FrustumLineAssets>>,
    existing_roots: Query<'w, 's, (Entity, &'static CameraFrustumRootVisual)>,
    existing_lines: Query<'w, 's, (Entity, &'static CameraFrustumLineVisual)>,
}

fn frustum_mesh_setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let edge_mesh = meshes.add(Mesh::from(Cylinder::new(1.0, 1.0)));
    let edge_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 0.0, 0.9),
        emissive: Color::srgb(1.0, 1.0, 0.0).into(),
        cull_mode: None,
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        depth_bias: -8.0,
        ..Default::default()
    });
    commands.insert_resource(FrustumLineAssets {
        edge_mesh,
        edge_material,
    });
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

fn frustum_segment_transform_local(start_local: Vec3, end_local: Vec3) -> Option<Transform> {
    let delta = end_local - start_local;
    let length = delta.length();
    if length <= 1.0e-5_f32 {
        return None;
    }

    let rotation = Quat::from_rotation_arc(Vec3::Y, delta / length);
    let midpoint_local = start_local + delta * 0.5;

    Some(Transform {
        translation: midpoint_local,
        rotation,
        scale: Vec3::new(FRUSTUM_EDGE_RADIUS, length, FRUSTUM_EDGE_RADIUS),
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

fn draw_viewport_frustums(params: FrustumDrawParams<'_, '_>, mut commands: Commands) {
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

        if !config.show_frustum {
            continue;
        }

        let Projection::Perspective(perspective) = projection else {
            continue;
        };
        let Some(points) = frustum_local_points(perspective) else {
            continue;
        };
        sources.push((camera_entity, points));
    }

    if sources.is_empty() || targets.is_empty() {
        cleanup_frustum_entities(&params, &mut commands);
        return;
    }

    let mut desired_roots: HashMap<CameraFrustumRootVisual, ()> = HashMap::new();
    let mut desired_segments: HashMap<
        CameraFrustumLineVisual,
        (CameraFrustumRootVisual, Transform, RenderLayers),
    > = HashMap::new();

    for (source_camera, points) in sources {
        let segments = frustum_segments(points);
        for (target_camera, render_layers) in &targets {
            let root_key = CameraFrustumRootVisual {
                source: source_camera,
                target: *target_camera,
            };
            desired_roots.insert(root_key, ());

            for (segment_idx, (start_local, end_local)) in segments.iter().enumerate() {
                let Some(local_transform) =
                    frustum_segment_transform_local(*start_local, *end_local)
                else {
                    continue;
                };
                desired_segments.insert(
                    CameraFrustumLineVisual {
                        source: source_camera,
                        target: *target_camera,
                        segment: segment_idx as u8,
                    },
                    (root_key, local_transform, render_layers.clone()),
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

    for (key, (root_key, transform, render_layers)) in desired_segments {
        let Some(&root_entity) = root_entities.get(&root_key) else {
            continue;
        };

        if let Some(entity) = existing_lines_by_key.remove(&key) {
            commands
                .entity(entity)
                .insert((transform, render_layers, ChildOf(root_entity)));
            continue;
        }

        commands.spawn((
            Mesh3d(line_assets.edge_mesh.clone()),
            MeshMaterial3d(line_assets.edge_material.clone()),
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
