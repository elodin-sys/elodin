use bevy::math::DVec3;
use bevy::pbr::MeshMaterial3d;
use bevy::shader::ShaderRef;
use bevy::{prelude::*, reflect::TypePath, render::render_resource::*};
use bevy_world_mesh::prelude::*;

const PATH: &str = "terrains/planar";
const TERRAIN_SIZE: f64 = 2000.0;
const HEIGHT: f32 = 500.0;
const TEXTURE_SIZE: u32 = 512;
const LOD_COUNT: u32 = 8;

#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct TerrainMaterial {
    #[texture(0, dimension = "1d")]
    #[sampler(1)]
    gradient: Handle<Image>,
}

impl Material for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/planar.wgsl".into()
    }
    fn enable_prepass() -> bool {
        false
    }
    fn enable_shadows() -> bool {
        false
    }
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.build().disable::<TransformPlugin>(),
            TerrainPlugin,
            TerrainDebugPlugin, // enable debug settings and controls
            TerrainMaterialPlugin::<TerrainMaterial>::default(),
            EnvScreenshotPlugin,
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<LoadingImages>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
    asset_server: Res<AssetServer>,
) {
    let gradient = asset_server.load("textures/gradient2.png");
    images.load_image(
        &gradient,
        TextureDimension::D1,
        TextureFormat::Rgba8UnormSrgb,
    );

    // Configure all the important properties of the terrain, as well as its attachments.
    let config = TerrainConfig {
        lod_count: LOD_COUNT,
        model: TerrainModel::planar(DVec3::new(0.0, -100.0, 0.0), TERRAIN_SIZE, 0.0, HEIGHT),
        path: PATH.to_string(),
        ..default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::R16,
    })
    .add_attachment(AttachmentConfig {
        name: "albedo".to_string(),
        texture_size: TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::Rgba8,
    });

    // Configure the quality settings of the terrain view. Adapt the settings to your liking.
    let view_config = TerrainViewConfig::default();

    let tile_atlas = TileAtlas::new(&config);
    let tile_tree = TileTree::new(&tile_atlas, &view_config);

    commands.spawn_big_space(Grid::default(), |root| {
        let frame = root.grid().clone();

        let terrain = root
            .spawn_spatial((
                TerrainBundle::new(tile_atlas, &frame),
                MeshMaterial3d(materials.add(TerrainMaterial { gradient })),
            ))
            .id();

        let view = root
            .spawn_spatial((
                Camera3d::default(),
                Projection::Perspective(PerspectiveProjection {
                    near: 0.1,
                    ..default()
                }),
                Transform::default(),
                FloatingOrigin,
                BigSpaceCameraController::default().with_slowing(true),
            ))
            .id();

        tile_trees.insert((terrain, view), tile_tree);
    });
}
