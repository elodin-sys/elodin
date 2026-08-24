use bevy::{
    ecs::query::Or,
    math::{DQuat, DVec3},
    pbr::wireframe::{Wireframe, WireframeColor},
    prelude::*,
};
use bevy_geo_frames::{GeoContext, GeoPosition, GeoRotation, OrDefault};
use bevy_world_mesh::prelude::WorldMeshPlugin as BevyWorldMeshRendererPlugin;
use bevy_world_mesh::terrain::{
    math::TerrainModel,
    terrain::{TerrainBundle, TerrainConfig},
    terrain_data::{
        AttachmentConfig, AttachmentFormat,
        tile_atlas::TileAtlas,
        tile_tree::{TerrainViewPosition, TileTree},
    },
    terrain_view::{TerrainViewComponents, TerrainViewConfig},
};

use crate::{MainCamera, sensor_camera::SensorCamera};

type WorldMeshViewFilter = Or<(With<MainCamera>, With<SensorCamera>)>;
type WorldMeshTerrainQuery<'w, 's> =
    Query<'w, 's, (Entity, &'static ChildOf), (With<WorldMeshTerrain>, With<TileAtlas>)>;
#[cfg(feature = "big_space")]
type WorldMeshViewPositionQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Transform,
        Option<&'static crate::spatial::GridCell>,
        Option<&'static ChildOf>,
    ),
    WorldMeshViewFilter,
>;

const PLANAR_TEXTURE_SIZE: u32 = 512;
const SPHERICAL_TEXTURE_SIZE: u32 = 512;
const PREPROCESSED_PLANAR_LOD_COUNT: u32 = 5;
const DEFAULT_PLANAR_LOD_COUNT: u32 = PREPROCESSED_PLANAR_LOD_COUNT;
const DEFAULT_SPHERICAL_LOD_COUNT: u32 = 5;
const SPHERICAL_ATLAS_SIZE: u32 = 2048;
const SPHERICAL_PATH: &str = "terrains/spherical";
const WGS84_MAJOR_AXIS_M: f64 = 6_378_137.0;
const WGS84_MINOR_AXIS_M: f64 = 6_356_752.314_245;
const SPHERICAL_MIN_HEIGHT_M: f32 = -12_000.0;
const SPHERICAL_MAX_HEIGHT_M: f32 = 9_000.0;
const SPHERICAL_FALLBACK_GRID_SECTORS: u32 = 64;
const SPHERICAL_FALLBACK_GRID_STACKS: u32 = 32;

/// Marker for terrain renderer entities spawned from a schematic `world_mesh` element.
#[derive(Component)]
pub struct WorldMeshTerrain;

/// Spatial anchor for a real terrain renderer.
///
/// Geo-frame and big-space systems own this entity's transform. The renderer
/// stays below it so its [`TerrainBundle`] model-local transform is preserved.
#[derive(Component)]
struct WorldMeshTerrainAnchor;

/// Editor integration layer for the real `bevy_world_mesh` terrain renderer.
///
/// The renderer/material plugin lives in `bevy_world_mesh`; this editor plugin
/// only adds Elodin-specific dynamic viewport wiring.
pub struct EditorWorldMeshPlugin;

impl Plugin for EditorWorldMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(BevyWorldMeshRendererPlugin).add_systems(
            Update,
            (sync_terrain_view_components, sync_terrain_view_positions).chain(),
        );
    }
}

pub(crate) fn spawn_world_mesh_terrain(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    world_mesh_materials: &mut Assets<bevy_world_mesh::prelude::WorldMeshMaterial>,
    world_mesh: &impeller2_wkt::WorldMesh,
    geo_context: &GeoContext,
) -> Entity {
    let region = world_mesh.region.clone();
    let config = if region == "globe" {
        spherical_terrain_config(world_mesh.lod_count)
    } else {
        planar_terrain_config(&region, world_mesh.lod_count)
    };

    match config {
        WorldMeshConfig::Terrain(config) => {
            let tile_atlas = TileAtlas::new(&config);
            let mut terrain_bundle = TerrainBundle::new(tile_atlas);
            terrain_bundle.visibility = world_mesh_visibility(world_mesh);

            let material =
                world_mesh_materials.add(bevy_world_mesh::prelude::WorldMeshMaterial::default());

            spawn_world_mesh_terrain_bundle(
                commands,
                terrain_bundle,
                material,
                world_mesh,
                &region,
                region != "globe",
                geo_context,
            )
        }
        WorldMeshConfig::Fallback(fallback) => spawn_world_mesh_fallback(
            commands,
            meshes,
            materials,
            world_mesh,
            &region,
            fallback,
            geo_context,
        ),
    }
}

enum WorldMeshConfig {
    Terrain(Box<TerrainConfig>),
    Fallback(WorldMeshFallback),
}

enum WorldMeshFallback {
    PlanarGrid,
    Globe,
}

fn spawn_world_mesh_terrain_bundle(
    commands: &mut Commands,
    terrain_bundle: TerrainBundle,
    material: Handle<bevy_world_mesh::prelude::WorldMeshMaterial>,
    world_mesh: &impeller2_wkt::WorldMesh,
    region: &str,
    y_up_surface: bool,
    geo_context: &GeoContext,
) -> Entity {
    let anchor = commands
        .spawn((
            WorldMeshTerrainAnchor,
            world_mesh_transform(world_mesh),
            Visibility::Visible,
            Name::new(format!("world_mesh terrain ({region})")),
        ))
        .id();

    commands.spawn((
        terrain_bundle,
        MeshMaterial3d(material),
        WorldMeshTerrain,
        ChildOf(anchor),
        Name::new(format!("world_mesh terrain renderer ({region})")),
    ));

    insert_geo_components(commands, anchor, world_mesh, y_up_surface, geo_context);
    insert_big_space_cell(commands, anchor);
    anchor
}

fn world_mesh_transform(world_mesh: &impeller2_wkt::WorldMesh) -> Transform {
    let mut transform = Transform::default();
    if world_mesh.frame.or_default().is_some() {
        return transform;
    }
    if let Some((tx, ty, tz)) = world_mesh.translate {
        transform.translation += Vec3::new(tx as f32, ty as f32, tz as f32);
    }
    transform
}

fn insert_geo_components(
    commands: &mut Commands,
    entity: Entity,
    world_mesh: &impeller2_wkt::WorldMesh,
    y_up_surface: bool,
    geo_context: &GeoContext,
) {
    let Some(frame) = world_mesh.frame.or_default() else {
        return;
    };
    let (x, y, z) = world_mesh.translate.unwrap_or_default();
    // Planar heightfields are Bevy Y-up (normal +Y), same as InfiniteGrid.
    // Cancel `bevy_R` so `to_bevy` leaves the ground a Bevy XZ plane in
    // every frame — `y_up_to_schematic` only cancels ENU.
    let att = if y_up_surface {
        GeoRotation::y_up_level(frame, geo_context)
    } else {
        DQuat::IDENTITY
    };
    commands.entity(entity).insert((
        GeoPosition(frame, DVec3::new(x, y, z)),
        GeoRotation::absolute(frame, att),
    ));
}

fn world_mesh_visibility(world_mesh: &impeller2_wkt::WorldMesh) -> Visibility {
    if world_mesh.visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    }
}

fn insert_big_space_cell(commands: &mut Commands, entity: Entity) {
    #[cfg(feature = "big_space")]
    commands
        .entity(entity)
        .insert(crate::spatial::GridCell::default());

    #[cfg(not(feature = "big_space"))]
    let _ = (commands, entity);
}

fn planar_terrain_config(region: &str, lod_count: Option<u32>) -> WorldMeshConfig {
    let manifest_path =
        bevy_world_mesh::terrain::util::asset_path(format!("terrains/planar/{region}/region.toml"));
    let Some(manifest) = std::fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|text| toml::from_str::<bevy_world_mesh::regions::RegionManifest>(&text).ok())
        .or_else(|| {
            bevy_world_mesh::regions::lookup(region)
                .map(bevy_world_mesh::regions::RegionManifest::from)
        })
    else {
        bevy::log::warn!(
            "schematic world_mesh region={region:?} is not a built-in preset and could not load a valid manifest from {}; showing fallback grid",
            manifest_path.display()
        );
        return WorldMeshConfig::Fallback(WorldMeshFallback::PlanarGrid);
    };

    let terrain_size = manifest.terrain_size_m();
    let height = manifest.height_m();
    let terrain_path = format!("terrains/planar/{region}");
    let atlas_ready = terrain_atlas_ready(
        &terrain_path,
        &format!("{region:?}"),
        "planar",
        "fetch_real_terrain and preprocess",
    );

    let config = TerrainConfig {
        lod_count: planar_lod_count(lod_count),
        model: TerrainModel::planar(
            bevy::math::DVec3::new(0.0, -(height as f64) * 0.4, 0.0),
            terrain_size,
            0.0,
            height,
        ),
        path: terrain_path,
        ..default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: PLANAR_TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::R16,
    })
    .add_attachment(AttachmentConfig {
        name: "albedo".to_string(),
        texture_size: PLANAR_TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::Rgba8,
    });

    if atlas_ready {
        WorldMeshConfig::Terrain(Box::new(config))
    } else {
        WorldMeshConfig::Fallback(WorldMeshFallback::PlanarGrid)
    }
}

fn planar_lod_count(lod_count: Option<u32>) -> u32 {
    lod_count
        .unwrap_or(DEFAULT_PLANAR_LOD_COUNT)
        .min(PREPROCESSED_PLANAR_LOD_COUNT)
}

fn terrain_atlas_ready(terrain_path: &str, region: &str, kind: &str, hint: &str) -> bool {
    let atlas_config_path =
        bevy_world_mesh::terrain::util::asset_path(format!("{terrain_path}/config.tc"));
    let Ok(tile_config) = bevy_world_mesh::terrain::formats::TC::load_file(&atlas_config_path)
    else {
        bevy::log::warn!(
            "schematic world_mesh region={region} has no prepared {kind} atlas at {}; showing fallback visual; run {hint} first",
            atlas_config_path.display()
        );
        return false;
    };
    let Some(tile) = tile_config.tiles.first() else {
        bevy::log::warn!(
            "schematic world_mesh region={region} has an empty {kind} atlas at {}; showing fallback visual; run {hint} first",
            atlas_config_path.display()
        );
        return false;
    };

    for attachment in ["height", "albedo"] {
        let tile_path = bevy_world_mesh::terrain::util::asset_path(format!(
            "{terrain_path}/data/{attachment}/{tile}.bin"
        ));
        if !tile_path.is_file() {
            bevy::log::warn!(
                "schematic world_mesh region={region} has no prepared {attachment} tile data at {}; showing fallback visual; run {hint} first",
                tile_path.display()
            );
            return false;
        }
    }

    true
}

#[derive(serde::Deserialize)]
struct GlobeManifest {
    min_height_m: f32,
    max_height_m: f32,
    lod_count: u32,
}

impl Default for GlobeManifest {
    fn default() -> Self {
        Self {
            min_height_m: SPHERICAL_MIN_HEIGHT_M,
            max_height_m: SPHERICAL_MAX_HEIGHT_M,
            lod_count: DEFAULT_SPHERICAL_LOD_COUNT,
        }
    }
}

fn spherical_terrain_config(lod_count: Option<u32>) -> WorldMeshConfig {
    let manifest_path =
        bevy_world_mesh::terrain::util::asset_path(format!("{SPHERICAL_PATH}/globe.toml"));
    let manifest = std::fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|text| toml::from_str::<GlobeManifest>(&text).ok())
        .unwrap_or_else(|| {
            bevy::log::warn!(
                "schematic world_mesh region=\"globe\" could not load a valid manifest from {}; showing fallback globe visual",
                manifest_path.display()
            );
            GlobeManifest::default()
        });

    let atlas_ready = terrain_atlas_ready(
        SPHERICAL_PATH,
        "\"globe\"",
        "spherical",
        "preprocess_global",
    );

    let config = TerrainConfig {
        lod_count: lod_count
            .unwrap_or(manifest.lod_count)
            .min(DEFAULT_SPHERICAL_LOD_COUNT),
        model: TerrainModel::ellipsoid(
            bevy::math::DVec3::ZERO,
            WGS84_MAJOR_AXIS_M,
            WGS84_MINOR_AXIS_M,
            manifest.min_height_m,
            manifest.max_height_m,
        ),
        path: SPHERICAL_PATH.to_string(),
        atlas_size: SPHERICAL_ATLAS_SIZE,
        ..default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: SPHERICAL_TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::R16,
    })
    .add_attachment(AttachmentConfig {
        name: "albedo".to_string(),
        texture_size: SPHERICAL_TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::Rgba8,
    });

    if atlas_ready {
        WorldMeshConfig::Terrain(Box::new(config))
    } else {
        WorldMeshConfig::Fallback(WorldMeshFallback::Globe)
    }
}

fn spawn_world_mesh_fallback(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    world_mesh: &impeller2_wkt::WorldMesh,
    region: &str,
    fallback: WorldMeshFallback,
    geo_context: &GeoContext,
) -> Entity {
    let entity = match fallback {
        WorldMeshFallback::PlanarGrid => spawn_planar_fallback_grid(commands, world_mesh, region),
        WorldMeshFallback::Globe => {
            spawn_globe_fallback(commands, meshes, materials, world_mesh, region)
        }
    };

    insert_geo_components(
        commands,
        entity,
        world_mesh,
        matches!(fallback, WorldMeshFallback::PlanarGrid),
        geo_context,
    );
    insert_big_space_cell(commands, entity);
    entity
}

fn spawn_planar_fallback_grid(
    commands: &mut Commands,
    world_mesh: &impeller2_wkt::WorldMesh,
    region: &str,
) -> Entity {
    commands
        .spawn((
            bevy::dev_tools::infinite_grid::InfiniteGrid,
            fallback_grid_settings(world_mesh.frame),
            world_mesh_transform(world_mesh),
            world_mesh_visibility(world_mesh),
            WorldMeshTerrain,
            Name::new(format!("world_mesh fallback grid ({region})")),
        ))
        .id()
}

fn fallback_grid_settings(
    frame: Option<bevy_geo_frames::GeoFrame>,
) -> bevy::dev_tools::infinite_grid::InfiniteGridSettings {
    let (x_axis_color, z_axis_color) = if frame == Some(bevy_geo_frames::GeoFrame::NED) {
        (crate::ui::colors::bevy::GREEN, crate::ui::colors::bevy::RED)
    } else {
        (crate::ui::colors::bevy::RED, crate::ui::colors::bevy::GREEN)
    };

    bevy::dev_tools::infinite_grid::InfiniteGridSettings {
        minor_line_color: Color::srgba(1.0, 1.0, 1.0, 0.02),
        major_line_color: Color::srgba(1.0, 1.0, 1.0, 0.05),
        z_axis_color,
        x_axis_color,
        fadeout_distance: 50_000.0,
        scale: 0.1,
        ..Default::default()
    }
}

fn spawn_globe_fallback(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    world_mesh: &impeller2_wkt::WorldMesh,
    region: &str,
) -> Entity {
    let mut transform = world_mesh_transform(world_mesh);
    transform.scale = Vec3::new(
        WGS84_MAJOR_AXIS_M as f32,
        WGS84_MINOR_AXIS_M as f32,
        WGS84_MAJOR_AXIS_M as f32,
    );

    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.20, 0.26, 0.32),
        perceptual_roughness: 0.8,
        ..default()
    });

    commands
        .spawn((
            Mesh3d(meshes.add(Sphere::new(1.0).mesh().uv(
                SPHERICAL_FALLBACK_GRID_SECTORS,
                SPHERICAL_FALLBACK_GRID_STACKS,
            ))),
            MeshMaterial3d(material),
            transform,
            world_mesh_visibility(world_mesh),
            Wireframe,
            WireframeColor {
                color: Color::srgba(0.70, 0.78, 0.86, 0.85),
            },
            WorldMeshTerrain,
            bevy::light::NotShadowCaster,
            bevy::light::NotShadowReceiver,
            Name::new(format!("world_mesh fallback globe ({region})")),
        ))
        .id()
}

/// The terrain renderer needs one [`TileTree`] per `(terrain, camera)` pair.
/// Editor viewports are spawned dynamically from KDL, so wire the pairs after
/// both the terrain entity and viewport cameras exist.
fn sync_terrain_view_components(
    terrains: Query<(Entity, &TileAtlas), With<WorldMeshTerrain>>,
    cameras: Query<Entity, WorldMeshViewFilter>,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
) {
    tile_trees
        .retain(|(terrain, view), _| terrains.get(*terrain).is_ok() && cameras.get(*view).is_ok());

    let view_config = TerrainViewConfig::default();
    for (terrain, tile_atlas) in &terrains {
        for view in &cameras {
            tile_trees
                .entry((terrain, view))
                .or_insert_with(|| TileTree::new(tile_atlas, &view_config));
        }
    }
}

/// A world-absolute position expressed in a terrain's model space.
///
/// The terrain renderer's model coordinates live under its anchor entity, so
/// tile LOD selection needs the camera position pulled back through the
/// anchor's global pose. With the anchor at the origin this is the identity,
/// which is all the pre-geo-anchor code supported.
fn terrain_model_view_position(
    anchor_translation: DVec3,
    anchor_rotation: DQuat,
    view_absolute: DVec3,
) -> DVec3 {
    anchor_rotation.inverse() * (view_absolute - anchor_translation)
}

#[cfg(feature = "big_space")]
fn sync_terrain_view_positions(
    terrains: WorldMeshTerrainQuery,
    anchors: Query<(&Transform, Option<&crate::spatial::GridCell>), With<WorldMeshTerrainAnchor>>,
    cameras: WorldMeshViewPositionQuery,
    parents: Query<(&Transform, &crate::spatial::GridCell)>,
    floating_origin: Res<crate::spatial::FloatingOriginSettings>,
    mut view_positions: ResMut<TerrainViewComponents<TerrainViewPosition>>,
) {
    view_positions
        .retain(|(terrain, view), _| terrains.get(*terrain).is_ok() && cameras.get(*view).is_ok());
    for (camera, transform, cell, parent) in &cameras {
        let absolute = cell
            .map(|cell| floating_origin.grid_position_double(cell, transform))
            .or_else(|| {
                let parent = parent?;
                let (parent_transform, parent_cell) = parents.get(parent.parent()).ok()?;
                let combined = parent_transform.mul_transform(*transform);
                Some(floating_origin.grid_position_double(parent_cell, &combined))
            })
            .unwrap_or_else(|| transform.translation.as_dvec3());

        for (terrain, anchor) in &terrains {
            let Ok((anchor_transform, anchor_cell)) = anchors.get(anchor.parent()) else {
                continue;
            };
            let anchor_translation = anchor_cell
                .map(|cell| floating_origin.grid_position_double(cell, anchor_transform))
                .unwrap_or_else(|| anchor_transform.translation.as_dvec3());
            let local = terrain_model_view_position(
                anchor_translation,
                anchor_transform.rotation.as_dquat(),
                absolute,
            );
            view_positions.insert((terrain, camera), TerrainViewPosition(local));
        }
    }
}

#[cfg(not(feature = "big_space"))]
fn sync_terrain_view_positions(
    terrains: WorldMeshTerrainQuery,
    anchors: Query<&Transform, With<WorldMeshTerrainAnchor>>,
    cameras: Query<(Entity, &Transform), WorldMeshViewFilter>,
    mut view_positions: ResMut<TerrainViewComponents<TerrainViewPosition>>,
) {
    view_positions
        .retain(|(terrain, view), _| terrains.get(*terrain).is_ok() && cameras.get(*view).is_ok());
    for (camera, transform) in &cameras {
        let absolute = transform.translation.as_dvec3();
        for (terrain, anchor) in &terrains {
            let Ok(anchor_transform) = anchors.get(anchor.parent()) else {
                continue;
            };
            let local = terrain_model_view_position(
                anchor_transform.translation.as_dvec3(),
                anchor_transform.rotation.as_dquat(),
                absolute,
            );
            view_positions.insert((terrain, camera), TerrainViewPosition(local));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;
    use bevy_geo_frames::{GeoContext, GeoFrame};
    use impeller2_wkt::{NodeId, WorldMesh};

    #[test]
    fn terrain_model_view_position_is_identity_at_origin() {
        let view = DVec3::new(10.0, 20.0, 30.0);
        let local = terrain_model_view_position(DVec3::ZERO, DQuat::IDENTITY, view);
        assert!((local - view).length() < 1e-12);
    }

    #[test]
    fn terrain_model_view_position_pulls_back_through_the_anchor_pose() {
        // A terrain anchored far from the origin (the ECEF case): a camera
        // sitting exactly at the anchor must select tiles as if it were at
        // the terrain model origin, not 6,371 km away.
        let anchor = DVec3::new(-2.0e6, -4.5e6, 3.8e6);
        let rotation = DQuat::from_rotation_z(0.7) * DQuat::from_rotation_x(-0.3);
        let local = terrain_model_view_position(anchor, rotation, anchor);
        assert!(
            local.length() < 1e-6,
            "camera at anchor => model origin, got {local:?}"
        );

        // A point offset from the anchor along a rotated axis lands on that
        // axis in model space with the offset preserved.
        let offset = rotation * DVec3::new(0.0, 123.0, 0.0);
        let local = terrain_model_view_position(anchor, rotation, anchor + offset);
        assert!(
            (local - DVec3::new(0.0, 123.0, 0.0)).length() < 1e-6,
            "got {local:?}"
        );
    }

    #[test]
    fn sync_writes_terrain_relative_view_positions() {
        let translate = (1_000.0, 2_000.0, 50.0);
        let (mut app, anchor, renderer) = spawn_model_terrain(
            TerrainModel::planar(DVec3::ZERO, 250.0, 0.0, 100.0),
            world_mesh(Some(GeoFrame::ENU), Some(translate)),
        );
        apply_geo_transforms(&mut app);

        app.init_resource::<TerrainViewComponents<TerrainViewPosition>>();
        #[cfg(feature = "big_space")]
        app.insert_resource(crate::spatial::FloatingOriginSettings::new(10_000.0, 100.0));

        let anchor_transform = *app.world().get::<Transform>(anchor).unwrap();
        let camera_pos = anchor_transform.translation + Vec3::new(3.0, 4.0, 5.0);
        let camera = app
            .world_mut()
            .spawn((Transform::from_translation(camera_pos), MainCamera))
            .id();
        #[cfg(feature = "big_space")]
        app.world_mut()
            .entity_mut(camera)
            .insert(crate::spatial::GridCell::default());

        app.world_mut()
            .run_system_once(sync_terrain_view_positions)
            .unwrap();

        let view_positions = app
            .world()
            .resource::<TerrainViewComponents<TerrainViewPosition>>();
        let local = view_positions
            .get(&(renderer, camera))
            .expect("keyed view position for the (terrain, camera) pair")
            .0;
        let expected = terrain_model_view_position(
            anchor_transform.translation.as_dvec3(),
            anchor_transform.rotation.as_dquat(),
            camera_pos.as_dvec3(),
        );
        assert!(
            (local - expected).length() < 1e-4,
            "keyed position {local:?} != expected {expected:?}"
        );
        // The pulled-back position must be near the model origin, not out at
        // the anchor's world offset.
        assert!(local.length() < 10.0, "not terrain-relative: {local:?}");
    }

    #[test]
    fn planar_lod_count_defaults_to_preprocessed_depth() {
        assert_eq!(planar_lod_count(None), PREPROCESSED_PLANAR_LOD_COUNT);
    }

    #[test]
    fn planar_lod_count_caps_values_above_preprocessed_depth() {
        assert_eq!(planar_lod_count(Some(7)), PREPROCESSED_PLANAR_LOD_COUNT);
    }

    #[test]
    fn planar_lod_count_keeps_lower_values() {
        assert_eq!(planar_lod_count(Some(3)), 3);
    }

    #[test]
    fn planar_model_transform_survives_geo_and_big_space_integration() {
        let model_translation = Vec3::new(0.0, -40.0, 0.0);
        let model_scale = Vec3::splat(250.0);
        let (mut app, anchor, renderer) = spawn_model_terrain(
            TerrainModel::planar(model_translation.as_dvec3(), 250.0, 0.0, 100.0),
            world_mesh(Some(GeoFrame::NED), Some((1.0, 2.0, 3.0))),
        );

        apply_geo_transforms(&mut app);

        let transform = app.world().get::<Transform>(renderer).unwrap();
        assert_eq!(transform.translation, model_translation);
        assert_eq!(transform.scale, model_scale);
        assert_eq!(
            app.world().get::<ChildOf>(renderer).unwrap().parent(),
            anchor
        );
        assert!(app.world().get::<GeoPosition>(renderer).is_none());
        assert!(app.world().get::<GeoRotation>(renderer).is_none());
        assert!(app.world().get::<GeoPosition>(anchor).is_some());
        assert!(app.world().get::<TileAtlas>(renderer).is_some());
        #[cfg(feature = "big_space")]
        {
            assert!(
                app.world()
                    .get::<crate::spatial::GridCell>(anchor)
                    .is_some()
            );
            assert!(
                app.world()
                    .get::<crate::spatial::GridCell>(renderer)
                    .is_none()
            );
        }

        assert!(app.world_mut().despawn(anchor));
        assert!(
            app.world().get_entity(renderer).is_err(),
            "despawning the schematic root must clean up its renderer child"
        );
    }

    #[test]
    fn globe_ellipsoid_scale_survives_geo_and_big_space_integration() {
        let major_axis = 6_378_137.0;
        let minor_axis = 6_356_752.0;
        let (mut app, _, renderer) = spawn_model_terrain(
            TerrainModel::ellipsoid(
                DVec3::ZERO,
                major_axis,
                minor_axis,
                SPHERICAL_MIN_HEIGHT_M,
                SPHERICAL_MAX_HEIGHT_M,
            ),
            world_mesh(Some(GeoFrame::ECEF), Some((10.0, 20.0, 30.0))),
        );

        apply_geo_transforms(&mut app);

        let transform = app.world().get::<Transform>(renderer).unwrap();
        assert_eq!(transform.translation, Vec3::ZERO);
        assert_eq!(
            transform.scale,
            Vec3::new(major_axis as f32, minor_axis as f32, major_axis as f32)
        );
    }

    #[test]
    fn planar_enu_anchor_stays_level_in_bevy() {
        let (mut app, anchor, _) = spawn_model_terrain(
            TerrainModel::planar(DVec3::ZERO, 250.0, 0.0, 100.0),
            world_mesh(Some(GeoFrame::ENU), None),
        );
        apply_geo_transforms(&mut app);
        let rotation = app.world().get::<Transform>(anchor).unwrap().rotation;
        assert!(
            rotation.dot(Quat::IDENTITY).abs() > 1.0 - 1e-5,
            "planar ENU terrain should sit level in Bevy, got {rotation:?}"
        );
    }

    #[test]
    fn planar_ned_anchor_stays_level_in_bevy() {
        let (mut app, anchor, _) = spawn_model_terrain(
            TerrainModel::planar(DVec3::ZERO, 250.0, 0.0, 100.0),
            world_mesh(Some(GeoFrame::NED), None),
        );
        apply_geo_transforms(&mut app);
        let rotation = app.world().get::<Transform>(anchor).unwrap().rotation;
        assert!(
            rotation.dot(Quat::IDENTITY).abs() > 1.0 - 1e-5,
            "planar NED terrain should sit level in Bevy, got {rotation:?}"
        );
    }

    #[test]
    fn framed_world_mesh_anchor_stays_at_origin_for_geo_pipeline() {
        let world_mesh = world_mesh(Some(GeoFrame::NED), Some((1.0, 2.0, 3.0)));

        assert_eq!(world_mesh_transform(&world_mesh).translation, Vec3::ZERO);
    }

    #[test]
    fn unframed_world_mesh_anchor_uses_default_geo_frame() {
        let world_mesh = world_mesh(None, Some((1.0, 2.0, 3.0)));

        assert_eq!(world_mesh_transform(&world_mesh).translation, Vec3::ZERO);
    }

    fn spawn_model_terrain(model: TerrainModel, world_mesh: WorldMesh) -> (App, Entity, Entity) {
        let config = TerrainConfig {
            model,
            path: "terrain-transform-regression-test".to_string(),
            ..default()
        };
        let terrain_bundle = TerrainBundle::new(TileAtlas::new(&config));
        let mut app = App::new();
        app.insert_resource(GeoContext::default());

        let anchor = {
            let mut commands = app.world_mut().commands();
            spawn_world_mesh_terrain_bundle(
                &mut commands,
                terrain_bundle,
                Handle::default(),
                &world_mesh,
                &world_mesh.region,
                true,
                &GeoContext::default(),
            )
        };
        app.world_mut().flush();

        let renderer = {
            let world = app.world_mut();
            let mut renderers =
                world.query_filtered::<Entity, (With<WorldMeshTerrain>, With<TileAtlas>)>();
            renderers.single(world).expect("one terrain renderer")
        };
        (app, anchor, renderer)
    }

    fn apply_geo_transforms(app: &mut App) {
        app.world_mut()
            .run_system_once(bevy_geo_frames::apply_transforms)
            .unwrap();
        app.world_mut()
            .run_system_once(bevy_geo_frames::apply_geo_rotation)
            .unwrap();
    }

    fn world_mesh(frame: Option<GeoFrame>, translate: Option<(f64, f64, f64)>) -> WorldMesh {
        WorldMesh {
            region: "no_such_region".to_string(),
            lod_count: None,
            translate,
            frame,
            visible: true,
            node_id: NodeId::default(),
        }
    }
}
