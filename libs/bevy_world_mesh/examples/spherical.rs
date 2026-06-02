use bevy::pbr::MeshMaterial3d;
use bevy::shader::ShaderRef;
use bevy::{math::DVec3, prelude::*, reflect::TypePath, render::render_resource::*};
use bevy_world_mesh::prelude::*;

const PATH: &str = "terrains/spherical";
const RADIUS: f64 = 6371000.0;
const MAJOR_AXES: f64 = 6378137.0;
const MINOR_AXES: f64 = 6356752.314245;
const MIN_HEIGHT: f32 = -12000.0;
const MAX_HEIGHT: f32 = 9000.0;
const TEXTURE_SIZE: u32 = 512;
const LOD_COUNT: u32 = 16;

#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct TerrainMaterial {
    #[texture(0, dimension = "1d")]
    #[sampler(1)]
    gradient: Handle<Image>,
}

impl Material for TerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/spherical.wgsl".into()
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
            TerrainMaterialPlugin::<TerrainMaterial>::default(),
            TerrainDebugPlugin, // enable debug settings and controls
            EnvScreenshotPlugin,
        ))
        // .insert_resource(ClearColor(Color::WHITE))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<LoadingImages>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
    asset_server: Res<AssetServer>,
) {
    let gradient = asset_server.load("textures/gradient.png");
    images.load_image(
        &gradient,
        TextureDimension::D1,
        TextureFormat::Rgba8UnormSrgb,
    );

    // Configure all the important properties of the terrain, as well as its attachments.
    let config = TerrainConfig {
        lod_count: LOD_COUNT,
        model: TerrainModel::ellipsoid(DVec3::ZERO, MAJOR_AXES, MINOR_AXES, MIN_HEIGHT, MAX_HEIGHT),
        // model: TerrainModel::ellipsoid(
        //     DVec3::ZERO,
        //     6378137.0,
        //     6378137.0 * 0.5,
        //     MIN_HEIGHT,
        //     MAX_HEIGHT,
        // ),
        // model: TerrainModel::sphere(DVec3::ZERO, RADIUS),
        path: PATH.to_string(),
        ..default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::R16,
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
                MeshMaterial3d(materials.add(TerrainMaterial {
                    gradient: gradient.clone(),
                })),
            ))
            .id();

        let (cell, translation) = frame.translation_to_grid(-DVec3::X * RADIUS * 3.0);
        // `translation` is the within-cell offset, not the absolute
        // position — `looking_to(Vec3::X, Vec3::Y)` sets forward = +X
        // directly (frame-independent), matching the old
        // `DebugCameraBundle::new` behaviour.
        let view = root
            .spawn_spatial((
                Camera3d::default(),
                Projection::Perspective(PerspectiveProjection {
                    near: 0.1,
                    ..default()
                }),
                Transform::from_translation(translation).looking_to(Vec3::X, Vec3::Y),
                cell,
                FloatingOrigin,
                BigSpaceCameraController::default().with_slowing(true),
            ))
            .id();

        tile_trees.insert((terrain, view), tile_tree);

        let sun_position = DVec3::new(-1.0, 1.0, -1.0) * RADIUS * 10.0;
        let (sun_cell, sun_translation) = frame.translation_to_grid(sun_position);

        root.spawn_spatial((
            Mesh3d(meshes.add(Sphere::new(RADIUS as f32 * 2.0).mesh().build())),
            Transform::from_translation(sun_translation),
            sun_cell,
        ));

        root.spawn_spatial(Mesh3d(meshes.add(Cuboid::from_length(RADIUS as f32 * 0.1))));
    });
}
