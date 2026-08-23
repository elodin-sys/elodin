//! Applies the schematic's sun, atmosphere, ambient light, and sky color.

use bevy::camera::ClearColorConfig;
use bevy::camera::visibility::RenderLayers;
use bevy::light::SunDisk;
use bevy::light::atmosphere::ScatteringMedium;
use bevy::math::DVec3;
use bevy::pbr::{AtmosphereMode, AtmosphereSettings};
// EnvironmentMapLight comes in via the prelude (bevy_light).
use bevy::prelude::*;
use bevy_geo_frames::solar::sun_direction_ecef;
use bevy_geo_frames::{GeoContext, GeoFrame};
use impeller2::types::Timestamp;
use impeller2_wkt::{AtmosphereConfig, CurrentTimestamp, EnvironmentConfig, SunConfig};

use crate::MainCamera;
use crate::plugins::cinematic_earth::CinematicEarthRoot;
use crate::plugins::render_layer_alloc::CINEMATIC_EARTH_RENDER_LAYER;

/// Marks the viewport that owns the cinematic Earth camera pipeline.
#[derive(Component)]
pub struct CinematicViewport;

/// Active schematic environment.
#[derive(Resource, Default, Clone)]
pub struct SceneEnvironment(pub Option<EnvironmentConfig>);

#[derive(Resource, Default)]
pub(crate) struct SpaceVisibility(pub f32);

/// Baked IBL intensity before environment scaling.
pub const BASE_ENVIRONMENT_MAP_INTENSITY: f32 = 2000.0;

/// Sun spawned from the schematic environment.
#[derive(Component)]
pub(crate) struct SchematicSun;

/// Atmosphere spawned from the schematic environment.
#[derive(Component)]
pub(crate) struct SchematicAtmosphere(AtmosphereConfig);

/// Returns the explicit atmosphere or cinematic Earth's derived atmosphere.
fn effective_atmosphere(env: &EnvironmentConfig) -> Option<AtmosphereConfig> {
    if env.earth.is_some() {
        if env.atmosphere.is_some() {
            warn_once!(
                "environment has both `earth` and `atmosphere`; `earth` \
                 supersedes it — remove the `atmosphere` child"
            );
        }
        return Some(AtmosphereConfig {
            origin: (0.0, 0.0, 0.0),
            inner_radius: 6_371_000.0,
            outer_radius: 6_471_000.0,
            ground_albedo: (0.3, 0.3, 0.3),
            raymarched: true,
        });
    }
    env.atmosphere
}

/// Returns the explicit sun or cinematic Earth's default sun.
fn effective_sun(env: &EnvironmentConfig) -> Option<SunConfig> {
    env.sun
        .or_else(|| env.earth.is_some().then(SunConfig::default))
}

fn atmosphere_local_rotation(earth_transform: &GlobalTransform) -> Quat {
    earth_transform.rotation().inverse()
}

pub struct SceneEnvironmentPlugin;

impl Plugin for SceneEnvironmentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SceneEnvironment>()
            .init_resource::<SpaceVisibility>()
            .add_systems(Update, (sync_sun, sync_atmosphere, sync_camera_environment));
    }
}

/// Builds the sun rotation from Bevy Y-up azimuth and elevation.
fn sun_rotation(sun: &SunConfig) -> Quat {
    let az = sun.azimuth_deg.unwrap_or(SunConfig::default_azimuth_deg());
    let el = sun
        .elevation_deg
        .unwrap_or(SunConfig::default_elevation_deg());
    Quat::from_euler(EulerRot::YXZ, -az.to_radians(), -el.to_radians(), 0.0)
}

/// Converts a frame-relative sun direction to Bevy light rotation.
fn rotation_from_frame_direction(to_sun: DVec3, frame: GeoFrame, ctx: &GeoContext) -> Option<Quat> {
    let to_sun = (GeoFrame::bevy_R_(&frame, ctx) * to_sun).try_normalize()?;
    Some(Quat::from_rotation_arc(Vec3::NEG_Z, (-to_sun).as_vec3()))
}

/// Resolves sun rotation from direction, angles, or ephemeris.
fn sun_rotation_world(
    sun: &SunConfig,
    frame: GeoFrame,
    ctx: &GeoContext,
    unix_micros: i64,
) -> Quat {
    if let Some((x, y, z)) = sun.direction {
        return rotation_from_frame_direction(
            DVec3::new(f64::from(x), f64::from(y), f64::from(z)),
            frame,
            ctx,
        )
        .unwrap_or_else(|| sun_rotation(sun));
    }
    if sun.tracks_ephemeris() {
        return rotation_from_frame_direction(sun_direction_ecef(unix_micros), GeoFrame::ECEF, ctx)
            .unwrap_or_else(|| sun_rotation(sun));
    }
    sun_rotation(sun)
}

fn playhead_unix_micros(current: &CurrentTimestamp) -> i64 {
    match current.0 {
        Timestamp::EPOCH => Timestamp::now().0,
        Timestamp(us) => us,
    }
}

fn cinematic_sun_layers() -> RenderLayers {
    RenderLayers::from_layers(&[0, CINEMATIC_EARTH_RENDER_LAYER])
}

fn sync_sun(
    mut commands: Commands,
    environment: Res<SceneEnvironment>,
    coordinate: Res<crate::Coordinate>,
    geo_ctx: Res<GeoContext>,
    current_ts: Res<CurrentTimestamp>,
    mut suns: Query<
        (
            Entity,
            &mut DirectionalLight,
            &mut Transform,
            Option<&RenderLayers>,
        ),
        With<SchematicSun>,
    >,
) {
    let earth = environment
        .0
        .as_ref()
        .is_some_and(|env| env.earth.is_some());
    let config = environment.0.as_ref().and_then(effective_sun);
    let frame = coordinate.0.unwrap_or_default();
    let unix_micros = playhead_unix_micros(&current_ts);
    match (config, suns.iter_mut().next()) {
        (Some(sun), Some((entity, mut light, mut transform, layers))) => {
            // Compare before writing: mutations dirty render extraction.
            if light.illuminance != sun.illuminance {
                light.illuminance = sun.illuminance;
            }
            if light.shadow_maps_enabled != sun.shadows {
                light.shadow_maps_enabled = sun.shadows;
            }
            let rotation = sun_rotation_world(&sun, frame, &geo_ctx, unix_micros);
            // ~1e-4 rad: paused playheads do not dirty extraction every frame.
            if transform.rotation.angle_between(rotation) > 1e-4 {
                transform.rotation = rotation;
            }
            let cine_layers = cinematic_sun_layers();
            match (earth, layers) {
                (true, Some(current)) if *current == cine_layers => {}
                (true, _) => {
                    commands.entity(entity).insert(cine_layers);
                }
                (false, Some(_)) => {
                    commands.entity(entity).remove::<RenderLayers>();
                }
                (false, None) => {}
            }
        }
        (Some(sun), None) => {
            let mut entity = commands.spawn((
                SchematicSun,
                Name::new("environment sun"),
                DirectionalLight {
                    illuminance: sun.illuminance,
                    shadow_maps_enabled: sun.shadows,
                    ..default()
                },
                SunDisk::EARTH,
                Transform::from_rotation(sun_rotation_world(&sun, frame, &geo_ctx, unix_micros)),
            ));
            if earth {
                entity.insert(cinematic_sun_layers());
            }
        }
        (None, Some((entity, ..))) => {
            commands.entity(entity).despawn();
        }
        (None, None) => {}
    }
}

/// Synchronizes the schematic atmosphere entity.
fn sync_atmosphere(
    mut commands: Commands,
    environment: Res<SceneEnvironment>,
    earth: Option<Single<(Entity, &GlobalTransform), With<CinematicEarthRoot>>>,
    mut media: ResMut<Assets<ScatteringMedium>>,
    mut existing: Query<(Entity, &SchematicAtmosphere, &mut Transform)>,
) {
    let Some((earth, earth_transform)) = earth.map(|x| *x) else {
        return;
    };
    let config = environment.0.as_ref().and_then(effective_atmosphere);
    let current = existing.iter_mut().next();
    match (config, current) {
        (Some(config), Some((entity, spawned, mut transform))) if spawned.0 == config => {
            let _ = entity;
            let rotation = atmosphere_local_rotation(earth_transform);
            if transform.rotation.angle_between(rotation) > 1e-6 {
                transform.rotation = rotation;
            }
        }
        (Some(config), current) => {
            if let Some((entity, ..)) = current {
                commands.entity(entity).despawn();
            }
            let medium = media.add(ScatteringMedium::earth(256, 256));
            let (r, g, b) = config.ground_albedo;
            commands.spawn((
                Transform::from_rotation(atmosphere_local_rotation(earth_transform)),
                *earth_transform,
                #[cfg(feature = "big_space")]
                crate::spatial::LowPrecisionRoot,
                SchematicAtmosphere(config),
                Name::new("environment atmosphere"),
                bevy::light::Atmosphere {
                    inner_radius: config.inner_radius,
                    outer_radius: config.outer_radius,
                    ground_albedo: Vec3::new(r, g, b),
                    medium,
                },
                ChildOf(earth),
            ));
        }
        (None, Some((entity, ..))) => {
            commands.entity(entity).despawn();
        }
        (None, None) => {}
    }
}

fn clear_color_matches(current: &ClearColorConfig, desired: &ClearColorConfig) -> bool {
    match (current, desired) {
        (ClearColorConfig::Default, ClearColorConfig::Default) => true,
        (ClearColorConfig::Custom(a), ClearColorConfig::Custom(b)) => a == b,
        _ => false,
    }
}

/// Synchronizes viewport IBL, clear color, and atmosphere settings.
fn atmosphere_settings_for(config: AtmosphereConfig) -> AtmosphereSettings {
    if config.raymarched {
        // A larger sky-view LUT preserves distant planetary limbs.
        AtmosphereSettings {
            aerial_view_lut_max_distance: 3.2e5,
            rendering_method: AtmosphereMode::Raymarched,
            sky_max_samples: 48,
            sky_view_lut_samples: 32,
            sky_view_lut_size: UVec2::new(800, 400),
            ..AtmosphereSettings::default()
        }
    } else {
        AtmosphereSettings {
            // Extend aerial perspective for long-range chase cameras.
            aerial_view_lut_max_distance: 3.2e5,
            ..AtmosphereSettings::default()
        }
    }
}

type EnvironmentCameraQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static mut Camera,
        &'static mut EnvironmentMapLight,
        Option<&'static AtmosphereSettings>,
        Has<CinematicViewport>,
    ),
    With<MainCamera>,
>;

fn sync_camera_environment(
    mut commands: Commands,
    environment: Res<SceneEnvironment>,
    space_visibility: Res<SpaceVisibility>,
    cinematic: Query<Entity, With<CinematicViewport>>,
    mut cameras: EnvironmentCameraQuery,
) {
    let (ambient_scale, sky_color, atmosphere, earth) = match &environment.0 {
        Some(config) => (
            config.ambient_scale.max(0.0),
            config.sky_color,
            effective_atmosphere(config),
            config.earth.is_some(),
        ),
        None => (1.0, None, None, false),
    };
    let intensity = BASE_ENVIRONMENT_MAP_INTENSITY * ambient_scale;
    let environment_clear = match sky_color {
        Some(color) => ClearColorConfig::Custom(Color::srgba(color.r, color.g, color.b, color.a)),
        None if earth => ClearColorConfig::Custom(Color::BLACK),
        None => ClearColorConfig::Default,
    };
    let regular_clear = match sky_color {
        Some(color) if !earth => {
            ClearColorConfig::Custom(Color::srgba(color.r, color.g, color.b, color.a))
        }
        _ => ClearColorConfig::Default,
    };
    // Bevy 0.19 permits AtmosphereSettings on only one active view.
    let cinematic_cam = cinematic.iter().next();
    let chosen = if earth {
        atmosphere.and(cinematic_cam)
    } else {
        let active_count = cameras
            .iter()
            .filter(|(_, camera, ..)| camera.is_active)
            .count();
        let active = cameras
            .iter()
            .filter(|(_, camera, ..)| camera.is_active)
            .map(|(entity, ..)| entity)
            .min();
        if atmosphere.is_some() && active_count > 1 {
            warn_once!(
                "schematic atmosphere renders on only one active main viewport \
                 (Bevy 0.19: several cameras with AtmosphereSettings trip wgpu \
                 bind-group validation and quit the editor). Other viewports keep \
                 clear-color/IBL; switch tabs to move the sky to another pane."
            );
        }
        atmosphere.and(active)
    };
    // Fade studio IBL out in space.
    let space_ibl_fade = 1.0 - space_visibility.0;
    for (entity, mut camera, mut light, current_settings, is_cinematic) in &mut cameras {
        let (target_intensity, target_clear) = if earth {
            if is_cinematic {
                (intensity * space_ibl_fade, environment_clear)
            } else {
                (BASE_ENVIRONMENT_MAP_INTENSITY, regular_clear)
            }
        } else {
            (intensity, environment_clear)
        };
        if light.intensity != target_intensity {
            light.intensity = target_intensity;
        }
        if !clear_color_matches(&camera.clear_color, &target_clear) {
            camera.clear_color = target_clear;
        }
        let wants_atmosphere = chosen == Some(entity);
        if wants_atmosphere {
            let desired = atmosphere_settings_for(atmosphere.unwrap());
            let needs_write = match current_settings {
                None => true,
                Some(s) => {
                    !std::mem::discriminant(&s.rendering_method)
                        .eq(&std::mem::discriminant(&desired.rendering_method))
                        || s.sky_max_samples != desired.sky_max_samples
                        || s.aerial_view_lut_max_distance != desired.aerial_view_lut_max_distance
                }
            };
            if needs_write {
                commands.entity(entity).insert(desired);
            }
        } else if current_settings.is_some() {
            commands.entity(entity).remove::<AtmosphereSettings>();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_rotation_points_light_downward_at_positive_elevation() {
        let sun = SunConfig {
            azimuth_deg: Some(0.0),
            elevation_deg: Some(45.0),
            illuminance: 100_000.0,
            shadows: true,
            direction: None,
        };
        // Positive elevation must tilt light forward below the horizon.
        let forward = sun_rotation(&sun) * Vec3::NEG_Z;
        assert!(forward.y < -0.5, "sun should shine downward, got {forward}");
    }

    #[test]
    fn earth_implies_default_ephemeris_sun() {
        let env = EnvironmentConfig {
            earth: Some(impeller2_wkt::EarthConfig::default()),
            ..Default::default()
        };
        let sun = effective_sun(&env).expect("earth implies a sun");
        assert!(sun.tracks_ephemeris());
        assert_eq!(sun.illuminance, SunConfig::default_illuminance());
        assert!(sun.shadows);
    }

    #[test]
    fn explicit_sun_wins_over_earth_default() {
        let env = EnvironmentConfig {
            sun: Some(SunConfig {
                illuminance: 12_000.0,
                ..Default::default()
            }),
            earth: Some(impeller2_wkt::EarthConfig::default()),
            ..Default::default()
        };
        assert_eq!(effective_sun(&env).unwrap().illuminance, 12_000.0);
    }

    #[test]
    fn atmosphere_local_rotation_cancels_earth_rotation() {
        for rotation in [
            Quat::IDENTITY,
            Quat::from_rotation_x(std::f32::consts::PI),
            Quat::from_euler(EulerRot::XYZ, 0.3, -0.8, 1.1),
        ] {
            let earth = GlobalTransform::from(Transform::from_rotation(rotation));
            let rendered = earth.rotation() * atmosphere_local_rotation(&earth);
            assert!(
                rendered.angle_between(Quat::IDENTITY) < 1e-6,
                "{rotation:?} rendered as {rendered:?}"
            );
        }
    }

    #[test]
    fn atmosphere_spawns_at_cinematic_earth_transform() {
        let mut app = App::new();
        app.insert_resource(SceneEnvironment(Some(EnvironmentConfig {
            earth: Some(impeller2_wkt::EarthConfig::default()),
            ..default()
        })))
        .init_resource::<Assets<ScatteringMedium>>()
        .add_systems(Update, sync_atmosphere);

        let earth_transform = GlobalTransform::from(
            Transform::from_translation(Vec3::new(4.0, -8.0, 15.0))
                .with_rotation(Quat::from_euler(EulerRot::XYZ, 0.3, -0.8, 1.1)),
        );
        let earth = app
            .world_mut()
            .spawn((CinematicEarthRoot, Transform::default(), earth_transform))
            .id();

        app.update();

        let mut query = app
            .world_mut()
            .query_filtered::<
                (Entity, &Transform, &GlobalTransform, &ChildOf),
                With<SchematicAtmosphere>,
            >();
        let (atmosphere, local, transform, parent) = query.single(app.world()).unwrap();
        assert_eq!(parent.parent(), earth);
        assert_eq!(transform.translation(), earth_transform.translation());
        assert!((earth_transform.rotation() * local.rotation).angle_between(Quat::IDENTITY) < 1e-6);
        #[cfg(feature = "big_space")]
        assert!(
            app.world()
                .get::<crate::spatial::LowPrecisionRoot>(atmosphere)
                .is_some()
        );
    }

    #[test]
    fn ephemeris_rotation_is_finite_at_j2000() {
        let sun = SunConfig::default();
        let ctx = GeoContext::default();
        let rot = sun_rotation_world(&sun, GeoFrame::ECEF, &ctx, 946_728_000_000_000);
        assert!(rot.is_finite());
        assert!(rot.is_normalized());
    }

    #[test]
    fn ephemeris_rotation_is_finite_at_crs12() {
        let sun = SunConfig::default();
        let ctx = GeoContext::default();
        // 2017-08-14T16:31:37Z
        let rot = sun_rotation_world(&sun, GeoFrame::ECEF, &ctx, 1_502_728_297_000_000);
        assert!(rot.is_finite(), "{rot:?}");
        assert!(rot.is_normalized(), "{rot:?}");
    }
}
