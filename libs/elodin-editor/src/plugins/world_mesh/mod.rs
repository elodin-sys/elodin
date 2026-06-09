use bevy::ecs::query::Or;
use bevy::prelude::*;
use bevy_world_mesh::prelude::WorldMeshPlugin as BevyWorldMeshRendererPlugin;
use bevy_world_mesh::terrain::{
    math::TerrainModel,
    terrain::{TerrainBundle, TerrainConfig},
    terrain_data::{
        AttachmentConfig, AttachmentFormat, tile_atlas::TileAtlas, tile_tree::TileTree,
    },
    terrain_view::{TerrainViewComponents, TerrainViewConfig},
};

use crate::{MainCamera, sensor_camera::SensorCamera};

type WorldMeshViewFilter = Or<(With<MainCamera>, With<SensorCamera>)>;

const PLANAR_TEXTURE_SIZE: u32 = 512;
const SPHERICAL_TEXTURE_SIZE: u32 = 512;
const DEFAULT_PLANAR_LOD_COUNT: u32 = 5;
const DEFAULT_SPHERICAL_LOD_COUNT: u32 = 5;
const SPHERICAL_ATLAS_SIZE: u32 = 2048;
const SPHERICAL_PATH: &str = "terrains/spherical";
const WGS84_MAJOR_AXIS_M: f64 = 6_378_137.0;
const WGS84_MINOR_AXIS_M: f64 = 6_356_752.314_245;
const SPHERICAL_MIN_HEIGHT_M: f32 = -12_000.0;
const SPHERICAL_MAX_HEIGHT_M: f32 = 9_000.0;

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
        app.add_plugins(BevyWorldMeshRendererPlugin)
            .add_systems(Update, sync_terrain_view_components);
    }
}

pub(crate) fn spawn_world_mesh_terrain(
    commands: &mut Commands,
    world_mesh_materials: &mut Assets<bevy_world_mesh::prelude::WorldMeshMaterial>,
    world_mesh: &impeller2_wkt::WorldMesh,
) -> Option<Entity> {
    let region = world_mesh.region.clone();
    let config = if region == "globe" {
        spherical_terrain_config(world_mesh.lod_count)
    } else {
        planar_terrain_config(&region, world_mesh.lod_count)?
    };

    let tile_atlas = TileAtlas::new(&config);
    let mut terrain_bundle = TerrainBundle::new(tile_atlas);
    if let Some((tx, ty, tz)) = world_mesh.translate {
        terrain_bundle.transform.translation += Vec3::new(tx as f32, ty as f32, tz as f32);
    }
    terrain_bundle.visibility = if world_mesh.visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };

    let material = world_mesh_materials.add(bevy_world_mesh::prelude::WorldMeshMaterial::default());

    let entity = commands
        .spawn((
            terrain_bundle,
            MeshMaterial3d(material),
            WorldMeshTerrain,
            Name::new(format!("world_mesh terrain ({region})")),
        ))
        .id();

    #[cfg(feature = "big_space")]
    commands
        .entity(entity)
        .insert(crate::spatial::GridCell::default());

    Some(entity)
}

fn planar_terrain_config(region: &str, lod_count: Option<u32>) -> Option<TerrainConfig> {
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
            "schematic world_mesh region={region:?} is not a built-in preset and could not load a valid manifest from {}",
            manifest_path.display()
        );
        return None;
    };

    let terrain_size = manifest.terrain_size_m();
    let height = manifest.height_m();
    let terrain_path = format!("terrains/planar/{region}");

    Some(
        TerrainConfig {
            lod_count: lod_count.unwrap_or(DEFAULT_PLANAR_LOD_COUNT),
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
        }),
    )
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

fn spherical_terrain_config(lod_count: Option<u32>) -> TerrainConfig {
    let manifest_path = bevy_world_mesh::terrain::util::asset_path("terrains/spherical/globe.toml");
    let manifest = std::fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|text| toml::from_str::<GlobeManifest>(&text).ok())
        .unwrap_or_default();

    TerrainConfig {
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
    })
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
