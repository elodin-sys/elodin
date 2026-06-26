use bevy::{
    ecs::query::Or,
    math::{DQuat, DVec3},
    pbr::wireframe::{Wireframe, WireframeColor},
    prelude::*,
};
use bevy_geo_frames::{GeoPosition, GeoRotation, OrDefault};
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

/// Marker for terrain entities spawned from a schematic `world_mesh` element.
#[derive(Component)]
pub struct WorldMeshTerrain;

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
            apply_world_mesh_transform_and_visibility(&mut terrain_bundle, world_mesh);

            let material =
                world_mesh_materials.add(bevy_world_mesh::prelude::WorldMeshMaterial::default());

            let entity = commands
                .spawn((
                    terrain_bundle,
                    MeshMaterial3d(material),
                    WorldMeshTerrain,
                    Name::new(format!("world_mesh terrain ({region})")),
                ))
                .id();

            insert_geo_components(commands, entity, world_mesh);
            insert_big_space_cell(commands, entity);
            entity
        }
        WorldMeshConfig::Fallback(fallback) => {
            spawn_world_mesh_fallback(commands, meshes, materials, world_mesh, &region, fallback)
        }
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

fn apply_world_mesh_transform_and_visibility(
    terrain_bundle: &mut TerrainBundle,
    world_mesh: &impeller2_wkt::WorldMesh,
) {
    terrain_bundle.transform = world_mesh_transform(world_mesh);
    terrain_bundle.visibility = world_mesh_visibility(world_mesh);
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
) {
    let Some(frame) = world_mesh.frame.or_default() else {
        return;
    };
    let (x, y, z) = world_mesh.translate.unwrap_or_default();
    commands.entity(entity).insert((
        GeoPosition(frame, DVec3::new(x, y, z)),
        GeoRotation::new(frame, DQuat::IDENTITY),
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
) -> Entity {
    let entity = match fallback {
        WorldMeshFallback::PlanarGrid => spawn_planar_fallback_grid(commands, world_mesh, region),
        WorldMeshFallback::Globe => {
            spawn_globe_fallback(commands, meshes, materials, world_mesh, region)
        }
    };

    insert_geo_components(commands, entity, world_mesh);
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
            bevy_infinite_grid::InfiniteGridBundle {
                settings: fallback_grid_settings(world_mesh.frame),
                transform: world_mesh_transform(world_mesh),
                visibility: world_mesh_visibility(world_mesh),
                ..default()
            },
            WorldMeshTerrain,
            Name::new(format!("world_mesh fallback grid ({region})")),
        ))
        .id()
}

fn fallback_grid_settings(
    frame: Option<bevy_geo_frames::GeoFrame>,
) -> bevy_infinite_grid::InfiniteGridSettings {
    let (x_axis_color, z_axis_color) = if frame == Some(bevy_geo_frames::GeoFrame::NED) {
        (crate::ui::colors::bevy::GREEN, crate::ui::colors::bevy::RED)
    } else {
        (crate::ui::colors::bevy::RED, crate::ui::colors::bevy::GREEN)
    };

    bevy_infinite_grid::InfiniteGridSettings {
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

#[cfg(feature = "big_space")]
fn sync_terrain_view_positions(
    mut commands: Commands,
    cameras: WorldMeshViewPositionQuery,
    parents: Query<(&Transform, &crate::spatial::GridCell)>,
    floating_origin: Res<crate::spatial::FloatingOriginSettings>,
) {
    for (entity, transform, cell, parent) in &cameras {
        let absolute = cell
            .map(|cell| floating_origin.grid_position_double(cell, transform))
            .or_else(|| {
                let parent = parent?;
                let (parent_transform, parent_cell) = parents.get(parent.parent()).ok()?;
                let combined = parent_transform.mul_transform(*transform);
                Some(floating_origin.grid_position_double(parent_cell, &combined))
            })
            .unwrap_or_else(|| transform.translation.as_dvec3());

        commands
            .entity(entity)
            .insert(TerrainViewPosition(absolute));
    }
}

#[cfg(not(feature = "big_space"))]
fn sync_terrain_view_positions(
    mut commands: Commands,
    cameras: Query<(Entity, &Transform), WorldMeshViewFilter>,
) {
    for (entity, transform) in &cameras {
        commands
            .entity(entity)
            .insert(TerrainViewPosition(transform.translation.as_dvec3()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_geo_frames::GeoFrame;
    use impeller2_wkt::{NodeId, WorldMesh};

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
    fn framed_world_mesh_transform_stays_at_origin_for_geo_pipeline() {
        let world_mesh = world_mesh(Some(GeoFrame::NED), Some((1.0, 2.0, 3.0)));

        assert_eq!(world_mesh_transform(&world_mesh).translation, Vec3::ZERO);
    }

    #[test]
    fn unframed_world_mesh_transform_uses_default_geo_frame() {
        let world_mesh = world_mesh(None, Some((1.0, 2.0, 3.0)));

        assert_eq!(world_mesh_transform(&world_mesh).translation, Vec3::ZERO);
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
