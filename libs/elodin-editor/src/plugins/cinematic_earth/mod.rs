//! Built-in camera-driven Earth for ECEF scenes.

pub mod curves;
pub mod earth_night_material;
pub mod effects;
pub mod modifiers;

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::image::{
    ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor,
};
use bevy::light::atmosphere::ScatteringMedium;
use bevy::light::{Atmosphere, NotShadowCaster, Skybox, SunDisk};
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use bevy::render::render_resource::{TextureViewDescriptor, TextureViewDimension};
use bevy::transform::TransformSystems;
use bevy::world_serialization::{WorldAsset, WorldAssetRoot};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};
use bevy_hanabi::{EffectAsset, EffectMaterial, EffectProperties, ParticleEffect};

use crate::plugins::render_layer_alloc::CINEMATIC_EARTH_RENDER_LAYER;
use crate::plugins::scene_environment::{
    CinematicViewport, SceneEnvironment, SchematicAtmosphere, SchematicSun, SpaceVisibility,
};
use crate::plugins::thruster_particles::images_ready;

use earth_night_material::{EarthNightExt, EarthNightMaterial, EarthNightParams};

/// Shared property-name contract with pyrotechnique / thruster effects.
pub const INTENSITY_PROPERTY: &str = "intensity";
pub const SUN_DIR_PROPERTY: &str = "sun_dir";
pub const VIEW_POS_PROPERTY: &str = "view_pos";
pub const SIZE_PROPERTY: &str = "size";
pub const HEIGHT_PROPERTY: &str = "height";

const EMBEDDED_GLOBE: &str = "embedded://elodin_editor/assets/earth/earth_v5.glb#Scene0";
const EMBEDDED_SKYBOX: &str = "embedded://elodin_editor/assets/earth/milky_way.cubemap.ktx2";
const EMBEDDED_COLOR: &str = "embedded://elodin_editor/assets/earth/color.ktx2";
const EMBEDDED_NIGHT: &str = "embedded://elodin_editor/assets/earth/night.ktx2";
const EMBEDDED_CLOUDS: &str = "embedded://elodin_editor/assets/earth/clouds.ktx2";
const EMBEDDED_NORMAL: &str = "embedded://elodin_editor/assets/earth/normal.ktx2";
const EMBEDDED_METALLIC_ROUGHNESS: &str =
    "embedded://elodin_editor/assets/earth/metallic_roughness.ktx2";
const EMBEDDED_SUN_FLARE: &str = "embedded://elodin_editor/assets/earth/sun_flare.png";

/// WGS84 polar flattening for the spherical globe mesh.
const WGS84_POLAR_SCALE: f32 = (curves::WGS84_B_M / curves::WGS84_A_M) as f32;

fn earth_map_sampler() -> ImageSamplerDescriptor {
    ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 8,
        ..default()
    }
}

fn load_earth_map(asset_server: &AssetServer, path: &'static str, srgb: bool) -> Handle<Image> {
    asset_server
        .load_builder()
        .with_settings(move |settings: &mut ImageLoaderSettings| {
            settings.is_srgb = srgb;
            settings.sampler = ImageSampler::Descriptor(earth_map_sampler());
            settings.asset_usage = RenderAssetUsages::RENDER_WORLD;
        })
        .load(path)
}

/// Gain for the deliberately dim Milky Way texture.
const SKYBOX_NIGHT_BRIGHTNESS: f32 = 4000.0;
const EARTH_EMISSIVE_NIGHT: f32 = 120.0;
const CLOUD_NIGHT_ALPHA: f32 = 0.05;
/// Warm fill for the camera-facing night hemisphere.
const NIGHT_GLOBE_ILLUMINANCE: f32 = 50.0;
/// Camera-riding fill whose range keeps it off the globe.
const EARTHSHINE_OFFSET_M: f32 = 300.0;
const EARTHSHINE_RANGE_M: f32 = 5_000.0;
/// Luminous flux for ~20 klx at [`EARTHSHINE_OFFSET_M`].
const EARTHSHINE_LUMENS: f32 = 20_000.0 * 4.0 * std::f32::consts::PI * 300.0 * 300.0;
/// Sun flare: additive 16:9 billboard riding the chosen camera.
const SUN_FLARE_DIST_M: f32 = 2_000.0;
/// Quad width for ~20° angular size at [`SUN_FLARE_DIST_M`]. Height is 9/16.
const SUN_FLARE_SIZE_M: f32 = 700.0;
/// Unlit colors skip exposure, so this gain must stay order-1.
const SUN_FLARE_GAIN: f32 = 3.0;

/// Globe root: ECEF origin, model→ECEF alignment rotation, no scale.
#[derive(Component)]
pub struct CinematicEarthRoot;

#[derive(Component)]
struct CinematicEarthEllipsoid;

/// Star-sphere root, Earth-centered. ECEF-fixed galactic frame via [`sky_geo_rotation`].
#[derive(Component)]
struct CinematicSkyRoot;

/// IAU galactic north pole, ICRS J2000 (Reid & Brunthaler 2004).
const GALACTIC_NORTH_RA_DEG: f64 = 192.85948;
const GALACTIC_NORTH_DEC_DEG: f64 = 27.12825;
/// IAU galactic center, ICRS J2000.
const GALACTIC_CENTER_RA_DEG: f64 = 266.4051;
const GALACTIC_CENTER_DEC_DEG: f64 = -28.936175;

fn ra_dec_to_dir(ra_deg: f64, dec_deg: f64) -> DVec3 {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let (sin_ra, cos_ra) = ra.sin_cos();
    let (sin_dec, cos_dec) = dec.sin_cos();
    DVec3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec)
}

/// Sky-local +Y/+X onto IAU galactic north/center. ICRS as ECEF (θ = 0); no sidereal spin.
fn sky_geo_rotation(ctx: &GeoContext) -> GeoRotation {
    let bevy_r = GeoFrame::bevy_R_(&GeoFrame::ECEF, ctx);
    let pole = (bevy_r * ra_dec_to_dir(GALACTIC_NORTH_RA_DEG, GALACTIC_NORTH_DEC_DEG)).normalize();
    let center =
        (bevy_r * ra_dec_to_dir(GALACTIC_CENTER_RA_DEG, GALACTIC_CENTER_DEC_DEG)).normalize();
    let y = pole;
    let x = (center - y * center.dot(y)).normalize();
    let z = x.cross(y);
    GeoRotation::from_bevy(
        GeoFrame::ECEF,
        DQuat::from_mat3(&DMat3::from_cols(x, y, z)),
        ctx,
    )
}

/// Teardown marker for every entity this plugin spawns.
#[derive(Component)]
struct CinematicEarthEntity;

/// Globe mesh whose emissive night sheet is masked per-pixel by local sun.
#[derive(Component)]
struct EarthGlobeMaterial;

/// Cloud mesh whose alpha thins at night so city lights read through.
#[derive(Component)]
struct EarthCloudsMaterial;

/// Star-field emitter (intensity = star visibility).
#[derive(Component)]
struct SkyEmitter;

#[derive(Component)]
struct CinematicParticleEmitter;

#[derive(Component)]
struct PendingEffectMaterial {
    effect: Handle<EffectAsset>,
    images: Vec<Handle<Image>>,
}

/// Globe-attached emitter (city lights / airglow).
#[derive(Clone, Copy, PartialEq, Eq)]
enum EarthEmitterKind {
    CityLights,
    Airglow,
}

#[derive(Component)]
struct EarthEmitter {
    kind: EarthEmitterKind,
}

/// Camera-riding earthshine.
#[derive(Component)]
struct CinematicEarthshine;

/// Additive starburst billboard toward the sun, on the chosen camera.
#[derive(Component)]
struct CinematicSunFlare;

/// Dim warm directional on the night globe.
#[derive(Component)]
struct NightGlobeFill;

/// Camera carrying the built-in Milky Way [`Skybox`].
///
/// Distinct from `bevy_ai_skybox`'s [`PrimarySkybox`]: the render-server
/// skybox gate must ignore this cubemap or it never emits frames.
#[derive(Component)]
pub(crate) struct CinematicSkybox;

/// Strong handles retained across Earth activation cycles.
#[derive(Resource, Clone)]
struct CinematicEarthAssets {
    globe_scene: Handle<WorldAsset>,
    skybox: Handle<Image>,
    color: Handle<Image>,
    night: Handle<Image>,
    clouds: Handle<Image>,
    normal: Handle<Image>,
    metallic_roughness: Handle<Image>,
    soft_circle: Handle<Image>,
    glow_veil: Handle<Image>,
    sun_flare: Handle<Image>,
    stars_dim: Handle<EffectAsset>,
    stars_bright: Handle<EffectAsset>,
    milky_way: Handle<EffectAsset>,
    city_lights: Handle<EffectAsset>,
    airglow_green: Handle<EffectAsset>,
    airglow_red: Handle<EffectAsset>,
}

/// Camera-relative Earth frame, computed once per frame after transform propagation.
#[derive(Resource, Clone, Debug)]
pub(crate) struct ViewerFrame {
    active: bool,
    /// Camera driving the look.
    camera: Option<Entity>,
    /// Camera radial up in render space.
    up: Vec3,
    altitude_m: f32,
    /// WGS84 geocentric surface radius under the camera [m].
    surface_radius_m: f32,
    space_vis: f32,
    star_vis: f32,
    nightglow_vis: f32,
    /// Limb-relative sinE (sun vs the visible horizon, not local horizontal).
    sun_elevation: f32,
    to_sun_world: Vec3,
}

impl Default for ViewerFrame {
    fn default() -> Self {
        Self {
            active: false,
            camera: None,
            up: Vec3::Y,
            altitude_m: 0.0,
            surface_radius_m: curves::WGS84_A_M as f32,
            space_vis: 0.0,
            star_vis: 0.0,
            nightglow_vis: 0.0,
            sun_elevation: 1.0,
            to_sun_world: Vec3::Y,
        }
    }
}

/// Last density used to regenerate the scattering LUT.
#[derive(Resource, Default)]
struct DensityTune {
    last_density: Option<f32>,
}

#[derive(Clone, Copy, PartialEq)]
struct EarthLookFingerprint {
    star_density: f32,
    star_brightness: f32,
    city_density: f32,
    city_brightness: f32,
    airglow_density: f32,
    airglow_brightness: f32,
}

impl EarthLookFingerprint {
    fn from_earth(earth: &impeller2_wkt::EarthConfig) -> Self {
        let earth = earth.clamp();
        Self {
            star_density: earth.stars.density,
            star_brightness: earth.stars.brightness,
            city_density: earth.city_lights.density,
            city_brightness: earth.city_lights.brightness,
            airglow_density: earth.airglow.density,
            airglow_brightness: earth.airglow.brightness,
        }
    }
}

#[derive(Resource, Default)]
struct EarthLookTune {
    last: Option<EarthLookFingerprint>,
    seen: Option<EarthLookFingerprint>,
    idle_s: f32,
}

pub struct CinematicEarthPlugin;

impl Plugin for CinematicEarthPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewerFrame>()
            .init_resource::<DensityTune>()
            .init_resource::<EarthLookTune>()
            // The chain flushes emitter swaps before Hanabi runs in PostUpdate.
            .add_systems(
                Update,
                (
                    sync_cinematic_earth,
                    bind_ready_emitters,
                    sync_earth_look,
                    tag_globe_meshes,
                )
                    .chain(),
            );
        app.add_systems(
            PostUpdate,
            (compute_viewer_frame, apply_cinematic_earth, tune_atmosphere)
                .chain()
                .after(TransformSystems::Propagate),
        );
    }
}

fn ensure_assets(
    commands: &mut Commands,
    assets: Option<&CinematicEarthAssets>,
    asset_server: &AssetServer,
    effects: &mut Assets<EffectAsset>,
    images: &mut Assets<Image>,
    earth: &impeller2_wkt::EarthConfig,
) -> CinematicEarthAssets {
    if let Some(assets) = assets {
        write_effects(assets, effects, earth);
        return assets.clone();
    }
    let created = CinematicEarthAssets {
        globe_scene: asset_server.load(EMBEDDED_GLOBE),
        skybox: asset_server.load(EMBEDDED_SKYBOX),
        color: load_earth_map(asset_server, EMBEDDED_COLOR, true),
        night: load_earth_map(asset_server, EMBEDDED_NIGHT, true),
        clouds: load_earth_map(asset_server, EMBEDDED_CLOUDS, true),
        normal: load_earth_map(asset_server, EMBEDDED_NORMAL, false),
        metallic_roughness: load_earth_map(asset_server, EMBEDDED_METALLIC_ROUGHNESS, false),
        soft_circle: images.add(effects::build_soft_circle_image()),
        glow_veil: images.add(effects::build_glow_veil_image()),
        sun_flare: asset_server.load(EMBEDDED_SUN_FLARE),
        stars_dim: effects.add(effects::stars_dim(earth)),
        stars_bright: effects.add(effects::stars_bright(earth)),
        milky_way: effects.add(effects::milky_way(earth)),
        city_lights: effects.add(effects::city_lights(earth)),
        airglow_green: effects.add(effects::airglow_green(earth)),
        airglow_red: effects.add(effects::airglow_red(earth)),
    };
    commands.insert_resource(created.clone());
    created
}

fn write_effects(
    assets: &CinematicEarthAssets,
    effects: &mut Assets<EffectAsset>,
    earth: &impeller2_wkt::EarthConfig,
) {
    if let Some(mut slot) = effects.get_mut(&assets.stars_dim) {
        *slot = effects::stars_dim(earth);
    }
    if let Some(mut slot) = effects.get_mut(&assets.stars_bright) {
        *slot = effects::stars_bright(earth);
    }
    if let Some(mut slot) = effects.get_mut(&assets.milky_way) {
        *slot = effects::milky_way(earth);
    }
    if let Some(mut slot) = effects.get_mut(&assets.city_lights) {
        *slot = effects::city_lights(earth);
    }
    if let Some(mut slot) = effects.get_mut(&assets.airglow_green) {
        *slot = effects::airglow_green(earth);
    }
    if let Some(mut slot) = effects.get_mut(&assets.airglow_red) {
        *slot = effects::airglow_red(earth);
    }
}

fn spawn_earth_emitters(parent: &mut ChildSpawnerCommands, assets: &CinematicEarthAssets) {
    parent.spawn((
        EarthEmitter {
            kind: EarthEmitterKind::CityLights,
        },
        CinematicEarthEntity,
        sky_emitter_bundle(
            "cinematic city_lights",
            assets.city_lights.clone(),
            vec![assets.glow_veil.clone(), assets.night.clone()],
        ),
    ));
    parent.spawn((
        EarthEmitter {
            kind: EarthEmitterKind::Airglow,
        },
        CinematicEarthEntity,
        sky_emitter_bundle(
            "cinematic airglow_green",
            assets.airglow_green.clone(),
            vec![assets.glow_veil.clone()],
        ),
    ));
    parent.spawn((
        EarthEmitter {
            kind: EarthEmitterKind::Airglow,
        },
        CinematicEarthEntity,
        sky_emitter_bundle(
            "cinematic airglow_red",
            assets.airglow_red.clone(),
            vec![assets.glow_veil.clone()],
        ),
    ));
}

fn spawn_sky_emitters(parent: &mut ChildSpawnerCommands, assets: &CinematicEarthAssets) {
    parent.spawn((
        SkyEmitter,
        CinematicEarthEntity,
        sky_emitter_bundle(
            "cinematic stars_dim",
            assets.stars_dim.clone(),
            vec![assets.soft_circle.clone()],
        ),
    ));
    parent.spawn((
        SkyEmitter,
        CinematicEarthEntity,
        sky_emitter_bundle(
            "cinematic stars_bright",
            assets.stars_bright.clone(),
            vec![assets.soft_circle.clone()],
        ),
    ));
    parent.spawn((
        SkyEmitter,
        CinematicEarthEntity,
        sky_emitter_bundle(
            "cinematic milky_way",
            assets.milky_way.clone(),
            vec![assets.soft_circle.clone()],
        ),
    ));
}

fn sky_emitter_bundle(
    name: &'static str,
    effect: Handle<EffectAsset>,
    images: Vec<Handle<Image>>,
) -> impl Bundle {
    (
        Name::new(name),
        CinematicParticleEmitter,
        PendingEffectMaterial { effect, images },
        EffectProperties::default(),
        Transform::default(),
        Visibility::default(),
        NoFrustumCulling,
        RenderLayers::layer(CINEMATIC_EARTH_RENDER_LAYER),
    )
}

fn bind_ready_emitters(
    mut commands: Commands,
    pending: Query<(Entity, &PendingEffectMaterial)>,
    asset_server: Res<AssetServer>,
) {
    for (entity, pending) in &pending {
        if !images_ready(&pending.images, &asset_server) {
            continue;
        }
        commands
            .entity(entity)
            .insert((
                ParticleEffect::new(pending.effect.clone()),
                EffectMaterial {
                    images: pending.images.clone(),
                },
            ))
            .remove::<PendingEffectMaterial>();
    }
}

/// Spawns/despawns the whole rig when `environment { earth }` toggles.
#[allow(clippy::too_many_arguments)]
fn sync_cinematic_earth(
    mut commands: Commands,
    environment: Res<SceneEnvironment>,
    coordinate: Res<crate::Coordinate>,
    geo_context: Res<GeoContext>,
    asset_server: Res<AssetServer>,
    assets: Option<Res<CinematicEarthAssets>>,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut images: ResMut<Assets<Image>>,
    roots: Query<Entity, With<CinematicEarthRoot>>,
    spawned: Query<Entity, With<CinematicEarthEntity>>,
    skybox_cameras: Query<Entity, With<CinematicSkybox>>,
    mut tune: ResMut<DensityTune>,
    mut look: ResMut<EarthLookTune>,
) {
    let active = environment
        .0
        .as_ref()
        .is_some_and(|env| env.earth.is_some());

    if !active {
        if !spawned.is_empty() || !skybox_cameras.is_empty() {
            for entity in &spawned {
                commands.entity(entity).despawn();
            }
            for camera in &skybox_cameras {
                commands
                    .entity(camera)
                    .remove::<Skybox>()
                    .remove::<CinematicSkybox>();
            }
            tune.last_density = None;
            look.last = None;
            look.seen = None;
            look.idle_s = 0.0;
        }
        return;
    }

    if !roots.is_empty() {
        return;
    }

    if coordinate.0 != Some(GeoFrame::ECEF) {
        warn_once!(
            "environment `earth` expects `coordinate frame=\"ECEF\"`; the globe \
             spawns at the ECEF origin regardless, but positions authored in \
             other frames will not sit on it correctly"
        );
    }
    let earth = environment
        .0
        .as_ref()
        .and_then(|env| env.earth)
        .unwrap_or_default();
    let assets = ensure_assets(
        &mut commands,
        assets.as_deref(),
        &asset_server,
        &mut effects,
        &mut images,
        &earth,
    );

    // The GLB's model-to-ECEF alignment is a 180° rotation about X.
    commands
        .spawn((
            CinematicEarthRoot,
            CinematicEarthEntity,
            Name::new("cinematic earth"),
            Transform::default(),
            GlobalTransform::default(),
            Visibility::default(),
            #[cfg(feature = "big_space")]
            crate::spatial::GridCell::default(),
            GeoPosition(GeoFrame::ECEF, DVec3::ZERO),
            GeoRotation::absolute(GeoFrame::ECEF, DQuat::from_rotation_x(std::f64::consts::PI)),
            NoFrustumCulling,
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    CinematicEarthEllipsoid,
                    Name::new("earth model"),
                    WorldAssetRoot(assets.globe_scene.clone()),
                    // Scale the model and Earth-attached effects together.
                    Transform::from_scale(Vec3::new(1.0, 1.0, WGS84_POLAR_SCALE)),
                    Visibility::default(),
                ))
                .with_children(|parent| spawn_earth_emitters(parent, &assets));
        });

    commands
        .spawn((
            CinematicSkyRoot,
            CinematicEarthEntity,
            Name::new("cinematic sky"),
            Transform::default(),
            GlobalTransform::default(),
            Visibility::default(),
            #[cfg(feature = "big_space")]
            crate::spatial::GridCell::default(),
            GeoPosition(GeoFrame::ECEF, DVec3::ZERO),
            sky_geo_rotation(&geo_context),
            NoFrustumCulling,
        ))
        .with_children(|parent| {
            spawn_sky_emitters(parent, &assets);
        });

    commands.spawn((
        NightGlobeFill,
        CinematicEarthEntity,
        Name::new("cinematic night globe fill"),
        DirectionalLight {
            color: Color::srgb(0.55, 0.48, 0.40),
            illuminance: 0.0,
            shadow_maps_enabled: false,
            ..default()
        },
        SunDisk::OFF,
        Transform::from_rotation(Quat::from_rotation_arc(Vec3::Z, Vec3::Y)),
        RenderLayers::layer(CINEMATIC_EARTH_RENDER_LAYER),
    ));

    tune.last_density = None;
    look.last = Some(EarthLookFingerprint::from_earth(&earth));
    look.seen = look.last;
    look.idle_s = 0.0;
    info!("cinematic earth spawned (embedded globe + sky)");
}

#[derive(bevy::ecs::system::SystemParam)]
struct EarthLookEntities<'w, 's> {
    ellipsoid: Query<'w, 's, Entity, With<CinematicEarthEllipsoid>>,
    sky_root: Query<'w, 's, Entity, With<CinematicSkyRoot>>,
    emitters: Query<'w, 's, Entity, With<CinematicParticleEmitter>>,
}

fn sync_earth_look(
    mut commands: Commands,
    environment: Res<SceneEnvironment>,
    assets: Option<Res<CinematicEarthAssets>>,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut look: ResMut<EarthLookTune>,
    time: Res<Time>,
    entities: EarthLookEntities,
) {
    let Some(earth) = environment.0.as_ref().and_then(|env| env.earth) else {
        return;
    };
    let Ok(ellipsoid) = entities.ellipsoid.single() else {
        return;
    };
    let Ok(sky_root) = entities.sky_root.single() else {
        return;
    };
    let Some(assets) = assets else {
        return;
    };
    let fingerprint = EarthLookFingerprint::from_earth(&earth);
    if look.seen != Some(fingerprint) {
        look.seen = Some(fingerprint);
        look.idle_s = 0.0;
        return;
    }
    look.idle_s += time.delta_secs();
    if look.last == Some(fingerprint) || look.idle_s < 0.15 {
        return;
    }
    for entity in &entities.emitters {
        commands.entity(entity).despawn();
    }
    write_effects(&assets, &mut effects, &earth);
    commands
        .entity(ellipsoid)
        .with_children(|parent| spawn_earth_emitters(parent, &assets));
    commands
        .entity(sky_root)
        .with_children(|parent| spawn_sky_emitters(parent, &assets));
    look.last = Some(fingerprint);
}

/// Tags asynchronously loaded globe meshes and disables limb-unsafe culling.
#[derive(bevy::ecs::system::SystemParam)]
struct GlobeMeshParams<'w, 's> {
    roots: Query<'w, 's, Entity, With<CinematicEarthEllipsoid>>,
    children: Query<'w, 's, &'static Children>,
    names: Query<'w, 's, &'static Name>,
    untagged: Query<
        'w,
        's,
        &'static MeshMaterial3d<StandardMaterial>,
        (Without<EarthGlobeMaterial>, Without<EarthCloudsMaterial>),
    >,
    unculled: Query<'w, 's, Entity, (With<Mesh3d>, Without<NoFrustumCulling>)>,
    unlayered: Query<'w, 's, Entity, (With<Mesh3d>, Without<RenderLayers>)>,
    assets: Option<Res<'w, CinematicEarthAssets>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    night_materials: ResMut<'w, Assets<EarthNightMaterial>>,
}

fn tag_globe_meshes(mut meshes: GlobeMeshParams, mut commands: Commands) {
    let Ok(root) = meshes.roots.single() else {
        return;
    };
    let Some(assets) = meshes.assets.as_deref() else {
        return;
    };
    for descendant in meshes.children.iter_descendants(root) {
        if meshes.unculled.contains(descendant) {
            commands.entity(descendant).insert(NoFrustumCulling);
        }
        if meshes.unlayered.contains(descendant) {
            commands
                .entity(descendant)
                .insert(RenderLayers::layer(CINEMATIC_EARTH_RENDER_LAYER));
        }
        let Ok(handle) = meshes.untagged.get(descendant) else {
            continue;
        };
        let name = meshes
            .names
            .get(descendant)
            .map(|n| n.as_str())
            .unwrap_or("");
        if name.contains("Cloud") {
            let Some(mut material) = meshes.materials.get_mut(&handle.0) else {
                continue;
            };
            material.base_color = Color::WHITE;
            material.base_color_texture = Some(assets.clouds.clone());
            material.alpha_mode = AlphaMode::Blend;
            commands.entity(descendant).insert(EarthCloudsMaterial);
        } else {
            let Some(mut base) = meshes.materials.get(&handle.0).cloned() else {
                continue;
            };
            base.base_color = Color::WHITE;
            base.base_color_texture = Some(assets.color.clone());
            base.emissive = LinearRgba::WHITE;
            base.emissive_texture = Some(assets.night.clone());
            base.normal_map_texture = Some(assets.normal.clone());
            base.metallic_roughness_texture = Some(assets.metallic_roughness.clone());
            let night = meshes.night_materials.add(EarthNightMaterial {
                base,
                extension: EarthNightExt {
                    params: EarthNightParams::default(),
                },
            });
            commands
                .entity(descendant)
                .remove::<MeshMaterial3d<StandardMaterial>>()
                .insert(MeshMaterial3d(night))
                .insert(EarthGlobeMaterial);
        }
    }
}

/// Cinematic Earth camera: `viewport cinematic=#true`.
type ViewportCameraQuery<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static Camera, &'static GlobalTransform),
    (With<CinematicViewport>, With<Camera3d>),
>;

/// Material access grouped under Bevy's 16-parameter system limit.
#[derive(bevy::ecs::system::SystemParam)]
struct EarthMaterialParams<'w, 's> {
    globe: Query<'w, 's, &'static MeshMaterial3d<EarthNightMaterial>, With<EarthGlobeMaterial>>,
    clouds: Query<'w, 's, &'static MeshMaterial3d<StandardMaterial>, With<EarthCloudsMaterial>>,
    standard: ResMut<'w, Assets<StandardMaterial>>,
    night: ResMut<'w, Assets<EarthNightMaterial>>,
    meshes: ResMut<'w, Assets<Mesh>>,
}

fn compute_viewer_frame(
    environment: Res<SceneEnvironment>,
    cameras: ViewportCameraQuery,
    earth: Query<&GlobalTransform, With<CinematicEarthRoot>>,
    sun: Query<&GlobalTransform, With<SchematicSun>>,
    mut frame: ResMut<ViewerFrame>,
    mut space_visibility: ResMut<SpaceVisibility>,
) {
    space_visibility.0 = 0.0;
    let active = environment
        .0
        .as_ref()
        .is_some_and(|env| env.earth.is_some());
    let Ok(earth_gt) = earth.single() else {
        *frame = ViewerFrame::default();
        return;
    };
    if !active {
        *frame = ViewerFrame::default();
        return;
    }
    let Some((camera, _, cam_gt)) = cameras.iter().next() else {
        *frame = ViewerFrame::default();
        return;
    };

    let center = earth_gt.translation();
    let radial = cam_gt.translation() - center;
    let distance = radial.length();
    let up = if distance > 1.0 {
        radial / distance
    } else {
        Vec3::Y
    };
    // Model -Z is the globe's polar axis.
    let north = (earth_gt.rotation() * Vec3::NEG_Z).normalize_or(Vec3::Y);
    let sin_lat = up.dot(north).clamp(-1.0, 1.0);
    let surface_radius = curves::geocentric_surface_radius_m(f64::from(sin_lat)) as f32;
    let altitude = distance - surface_radius;

    let to_sun = sun.single().map(|gt| gt.rotation() * Vec3::Z).unwrap_or(up);
    let elevation = curves::limb_relative_elevation(
        curves::sun_elevation(to_sun, up),
        altitude,
        surface_radius,
    );
    let space_vis = curves::space_visibility(altitude);
    space_visibility.0 = space_vis;

    *frame = ViewerFrame {
        active: true,
        camera: Some(camera),
        up,
        altitude_m: altitude,
        surface_radius_m: surface_radius,
        space_vis,
        star_vis: curves::star_visibility(elevation) * space_vis,
        nightglow_vis: curves::nightglow_visibility(elevation) * space_vis,
        sun_elevation: elevation,
        to_sun_world: to_sun,
    };
}

/// Reinterprets the loaded skybox as a cube view.
fn configure_cubemap_image(handle: &Handle<Image>, images: &mut Assets<Image>) -> bool {
    let Some(image) = images.get(handle) else {
        return false;
    };
    if image
        .texture_view_descriptor
        .as_ref()
        .is_some_and(|descriptor| descriptor.dimension == Some(TextureViewDimension::Cube))
    {
        return true;
    }
    if image.width() == 0 {
        return false;
    }

    let array_layers = image.texture_descriptor.array_layer_count();
    if array_layers == 6 {
        let Some(mut image) = images.get_mut(handle) else {
            return false;
        };
        image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        });
        return true;
    }

    if array_layers == 1 {
        let layers = image.height() / image.width();
        if layers == 6 {
            let Some(mut image) = images.get_mut(handle) else {
                return false;
            };
            let _ = image.reinterpret_stacked_2d_as_array(layers);
            image.texture_view_descriptor = Some(TextureViewDescriptor {
                dimension: Some(TextureViewDimension::Cube),
                ..default()
            });
            return true;
        }
    }

    false
}

/// Applies the per-frame look from [`ViewerFrame`].
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn apply_cinematic_earth(
    mut commands: Commands,
    environment: Res<SceneEnvironment>,
    frame: Res<ViewerFrame>,
    assets: Option<Res<CinematicEarthAssets>>,
    mut images: ResMut<Assets<Image>>,
    ellipsoid: Query<&GlobalTransform, With<CinematicEarthEllipsoid>>,
    sky: Query<&GlobalTransform, With<CinematicSkyRoot>>,
    mut cameras: Query<
        (Entity, &GlobalTransform, Option<&mut Skybox>),
        (With<CinematicViewport>, With<Camera3d>),
    >,
    mut earthshine: Query<
        (Entity, &mut Transform, &mut PointLight, &ChildOf),
        (
            With<CinematicEarthshine>,
            Without<NightGlobeFill>,
            Without<CinematicSunFlare>,
        ),
    >,
    mut globe_fill: Query<
        (&mut Transform, &mut DirectionalLight),
        (
            With<NightGlobeFill>,
            Without<CinematicEarthshine>,
            Without<CinematicSunFlare>,
        ),
    >,
    mut sun_flare: Query<
        (
            Entity,
            &mut Transform,
            &mut Visibility,
            &MeshMaterial3d<StandardMaterial>,
            &ChildOf,
        ),
        (
            With<CinematicSunFlare>,
            Without<CinematicEarthshine>,
            Without<NightGlobeFill>,
        ),
    >,
    mut mats: EarthMaterialParams,
    mut sky_emitters: Query<&mut EffectProperties, (With<SkyEmitter>, Without<EarthEmitter>)>,
    mut earth_emitters: Query<(&EarthEmitter, &mut EffectProperties), Without<SkyEmitter>>,
) {
    if !frame.active {
        return;
    }
    let Some(assets) = assets else {
        return;
    };
    let Ok(earth_gt) = ellipsoid.single() else {
        return;
    };
    let night_w = (1.0 - frame.sun_elevation.max(0.0)) * frame.space_vis;
    let skybox_ready = configure_cubemap_image(&assets.skybox, &mut images);
    let skybox_rotation = sky
        .single()
        .map(|gt| gt.rotation())
        .unwrap_or(Quat::IDENTITY);

    let mut chosen_cam_pos = None;
    for (entity, cam_gt, skybox) in &mut cameras {
        let chosen = frame.camera == Some(entity);
        if chosen {
            chosen_cam_pos = Some(cam_gt.translation());
        }
        // Star visibility keeps the Milky Way out of daylight.
        if skybox_ready {
            let brightness = SKYBOX_NIGHT_BRIGHTNESS * frame.star_vis;
            match skybox {
                Some(mut skybox) => {
                    if skybox.image.as_ref() != Some(&assets.skybox) {
                        skybox.image = Some(assets.skybox.clone());
                    }
                    if (skybox.brightness - brightness).abs() > 1e-3 {
                        skybox.brightness = brightness;
                    }
                    if skybox.rotation != skybox_rotation {
                        skybox.rotation = skybox_rotation;
                    }
                }
                None => {
                    commands.entity(entity).insert((
                        Skybox {
                            image: Some(assets.skybox.clone()),
                            brightness,
                            rotation: skybox_rotation,
                        },
                        CinematicSkybox,
                    ));
                }
            }
        }
    }

    // Earthshine: point light riding below the chosen camera.
    if let Some(chosen) = frame.camera {
        let mut found = false;
        for (entity, mut transform, mut light, child_of) in &mut earthshine {
            found = true;
            if child_of.parent() != chosen {
                commands.entity(entity).insert(ChildOf(chosen));
            }
            // Convert the radial earthshine offset into camera-local space.
            if let Ok((_, cam_gt, ..)) = cameras.get(chosen) {
                let translation = cam_gt.rotation().inverse() * (-frame.up * EARTHSHINE_OFFSET_M);
                if transform.translation != translation {
                    transform.translation = translation;
                }
            }
            let intensity = EARTHSHINE_LUMENS * night_w;
            if light.intensity != intensity {
                light.intensity = intensity;
            }
        }
        if !found {
            commands.spawn((
                CinematicEarthshine,
                CinematicEarthEntity,
                Name::new("cinematic earthshine"),
                PointLight {
                    color: Color::srgb(0.72, 0.78, 0.88),
                    intensity: 0.0,
                    range: EARTHSHINE_RANGE_M,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::default(),
                RenderLayers::layer(CINEMATIC_EARTH_RENDER_LAYER),
                ChildOf(chosen),
            ));
        }
    }

    // Show the sun flare only above the limb in space.
    if let Some(chosen) = frame.camera {
        let flare_vis = (frame.sun_elevation / 0.02).clamp(0.0, 1.0) * frame.space_vis;
        let cam_rot = cameras
            .get(chosen)
            .map(|(_, cam_gt, ..)| cam_gt.rotation())
            .unwrap_or(Quat::IDENTITY);
        let sun_cam = (cam_rot.inverse() * frame.to_sun_world).normalize_or(Vec3::Z);
        let c = SUN_FLARE_GAIN * flare_vis;
        let color = Color::from(LinearRgba::rgb(c, c, c));
        let translation = sun_cam * SUN_FLARE_DIST_M;
        let rotation = Quat::from_rotation_arc(Vec3::Z, -sun_cam);
        let shown = if flare_vis > 0.0 {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        let mut found = false;
        for (entity, mut transform, mut visibility, material, child_of) in &mut sun_flare {
            found = true;
            if child_of.parent() != chosen {
                commands.entity(entity).insert(ChildOf(chosen));
            }
            if transform.translation != translation {
                transform.translation = translation;
            }
            if transform.rotation != rotation {
                transform.rotation = rotation;
            }
            if *visibility != shown {
                *visibility = shown;
            }
            if let Some(mut material) = mats.standard.get_mut(&material.0)
                && material.base_color != color
            {
                material.base_color = color;
            }
        }
        if !found {
            commands.spawn((
                CinematicSunFlare,
                CinematicEarthEntity,
                Name::new("cinematic sun flare"),
                Mesh3d(mats.meshes.add(Rectangle::new(
                    SUN_FLARE_SIZE_M,
                    SUN_FLARE_SIZE_M * 720.0 / 1280.0,
                ))),
                MeshMaterial3d(mats.standard.add(StandardMaterial {
                    base_color: Color::BLACK,
                    base_color_texture: Some(assets.sun_flare.clone()),
                    unlit: true,
                    alpha_mode: AlphaMode::Add,
                    ..default()
                })),
                Transform::from_translation(translation).with_rotation(rotation),
                Visibility::Hidden,
                NotShadowCaster,
                NoFrustumCulling,
                RenderLayers::layer(CINEMATIC_EARTH_RENDER_LAYER),
                ChildOf(chosen),
            ));
        }
    }

    // Aim the night fill at the camera-facing hemisphere.
    for (mut transform, mut light) in &mut globe_fill {
        let rotation = Quat::from_rotation_arc(Vec3::Z, frame.up);
        if transform.rotation != rotation {
            transform.rotation = rotation;
        }
        let illuminance = NIGHT_GLOBE_ILLUMINANCE * night_w;
        if light.illuminance != illuminance {
            light.illuminance = illuminance;
        }
    }

    let earth = environment
        .0
        .as_ref()
        .and_then(|env| env.earth)
        .unwrap_or_default()
        .clamp();

    // Globe night sheet: constant emissive × space_vis; per-pixel twilight mask.
    let e = EARTH_EMISSIVE_NIGHT * earth.night_map.brightness * frame.space_vis;
    let emissive = LinearRgba::rgb(e, e, e);
    let to_sun = frame.to_sun_world.normalize_or(Vec3::Y).extend(0.0);
    for handle in &mats.globe {
        if let Some(mut material) = mats.night.get_mut(&handle.0) {
            if material.base.emissive != emissive {
                material.base.emissive = emissive;
            }
            if material.extension.params.to_sun_world != to_sun {
                material.extension.params.to_sun_world = to_sun;
            }
        }
    }
    let cloud_alpha = 1.0 - (1.0 - CLOUD_NIGHT_ALPHA) * frame.nightglow_vis;
    for handle in &mats.clouds {
        if let Some(mut material) = mats.standard.get_mut(&handle.0)
            && (material.base_color.alpha() - cloud_alpha).abs() > 1e-3
        {
            material.base_color = material.base_color.with_alpha(cloud_alpha);
        }
    }

    // Particle-field properties, in the globe's local (model) frame.
    let earth_inv = earth_gt.affine().inverse();
    let sun_local = earth_inv
        .transform_vector3(frame.to_sun_world)
        .normalize_or(Vec3::Y);
    let view_pos_local = chosen_cam_pos
        .map(|pos| earth_inv.transform_point3(pos))
        .unwrap_or(Vec3::Y * 6_778_140.0);

    for properties in &mut sky_emitters {
        let properties =
            EffectProperties::set_if_changed(properties, INTENSITY_PROPERTY, frame.star_vis.into());
        let _ =
            EffectProperties::set_if_changed(properties, SIZE_PROPERTY, earth.stars.size.into());
    }
    for (emitter, properties) in &mut earth_emitters {
        let vis = match emitter.kind {
            EarthEmitterKind::Airglow => frame.nightglow_vis,
            EarthEmitterKind::CityLights => frame.space_vis,
        };
        let properties =
            EffectProperties::set_if_changed(properties, INTENSITY_PROPERTY, vis.into());
        let properties =
            EffectProperties::set_if_changed(properties, SUN_DIR_PROPERTY, sun_local.into());
        let properties =
            EffectProperties::set_if_changed(properties, VIEW_POS_PROPERTY, view_pos_local.into());
        match emitter.kind {
            EarthEmitterKind::CityLights => {
                let properties = EffectProperties::set_if_changed(
                    properties,
                    SIZE_PROPERTY,
                    earth.city_lights.size.into(),
                );
                let _ = EffectProperties::set_if_changed(
                    properties,
                    HEIGHT_PROPERTY,
                    earth.city_lights.height.into(),
                );
            }
            EarthEmitterKind::Airglow => {
                let _ = EffectProperties::set_if_changed(
                    properties,
                    SIZE_PROPERTY,
                    earth.airglow.size.into(),
                );
            }
        }
    }
}

/// Tunes atmosphere density and local WGS84 surface radius.
fn tune_atmosphere(
    frame: Res<ViewerFrame>,
    mut tune: ResMut<DensityTune>,
    mut atmospheres: Query<&mut Atmosphere, With<SchematicAtmosphere>>,
    mut media: ResMut<Assets<ScatteringMedium>>,
) {
    if !frame.active {
        return;
    }
    let Ok(mut atmosphere) = atmospheres.single_mut() else {
        return;
    };

    let inner = frame.surface_radius_m;
    if (atmosphere.inner_radius - inner).abs() > 0.5 {
        atmosphere.inner_radius = inner;
        atmosphere.outer_radius = inner + 100_000.0;
    }

    let density = curves::quantize_density(frame.altitude_m);
    if tune.last_density.map(|d| (d - density).abs() > 1e-4) != Some(false) {
        if let Some(mut medium) = media.get_mut(&atmosphere.medium) {
            *medium = ScatteringMedium::earth(256, 256).with_density_multiplier(density);
        }
        tune.last_density = Some(density);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_dir_near(got: DVec3, expect: DVec3, label: &str) {
        assert!(
            got.dot(expect) > 1.0 - 1e-9,
            "{label}: got {got:?} expected {expect:?} (dot {})",
            got.dot(expect)
        );
    }

    #[test]
    fn ellipsoid_scale_matches_wgs84_axes() {
        let scale = Vec3::new(1.0, 1.0, WGS84_POLAR_SCALE);
        let equator = (scale * (Vec3::X * effects::EARTH_R)).length();
        let pole = (scale * (Vec3::Z * effects::EARTH_R)).length();
        assert!((equator - effects::EARTH_R).abs() < 1.0);
        assert!((pole - curves::WGS84_B_M as f32).abs() < 5.0);
    }

    #[test]
    fn galactic_pole_and_center_are_nearly_orthogonal() {
        let pole = ra_dec_to_dir(GALACTIC_NORTH_RA_DEG, GALACTIC_NORTH_DEC_DEG);
        let center = ra_dec_to_dir(GALACTIC_CENTER_RA_DEG, GALACTIC_CENTER_DEC_DEG);
        assert!((pole.length() - 1.0).abs() < 1e-12);
        assert!(
            pole.dot(center).abs() < 0.02,
            "IAU pole·center = {}",
            pole.dot(center)
        );
    }

    #[test]
    fn sky_root_maps_local_axes_onto_galactic_frame() {
        let ctx = GeoContext::default();
        let q = sky_geo_rotation(&ctx).to_bevy(&ctx);
        let bevy_r = GeoFrame::bevy_R_(&GeoFrame::ECEF, &ctx);
        let pole =
            (bevy_r * ra_dec_to_dir(GALACTIC_NORTH_RA_DEG, GALACTIC_NORTH_DEC_DEG)).normalize();
        let center =
            (bevy_r * ra_dec_to_dir(GALACTIC_CENTER_RA_DEG, GALACTIC_CENTER_DEC_DEG)).normalize();
        assert_dir_near(q * DVec3::Y, pole, "local +Y");
        let x = q * DVec3::X;
        assert!(
            x.dot(pole).abs() < 1e-9,
            "local +X must sit in the galactic plane"
        );
        assert!(
            x.dot(center) > 0.99,
            "local +X toward galactic center, dot {}",
            x.dot(center)
        );
    }
}
