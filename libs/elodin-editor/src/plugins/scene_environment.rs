//! Applies the schematic's top-level `environment` node: a directional sun
//! (with optional shadow maps), ambient/IBL scaling, and viewport sky color.
//!
//! Without an `environment` node the editor renders exactly as before: baked
//! `EnvironmentMapLight` IBL, no sun, theme-colored background. See
//! docs/design-thruster-effects-port.md §4.2 for the schema and rationale.

use bevy::camera::ClearColorConfig;
use bevy::camera::visibility::RenderLayers;
use bevy::light::SunDisk;
use bevy::light::atmosphere::ScatteringMedium;
use bevy::math::{DQuat, DVec3};
use bevy::pbr::{AtmosphereMode, AtmosphereSettings};
// EnvironmentMapLight comes in via the prelude (bevy_light).
use bevy::prelude::*;
use bevy_geo_frames::solar::sun_direction_ecef;
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};
use impeller2::types::Timestamp;
use impeller2_wkt::{AtmosphereConfig, CurrentTimestamp, EnvironmentConfig, SunConfig};

use crate::MainCamera;
use crate::plugins::render_layer_alloc::CINEMATIC_EARTH_RENDER_LAYER;

/// Marker for the viewport that owns the cinematic Earth camera pipeline
/// (`viewport cinematic=#true`). Atmosphere, HDR, skybox, and earth-layer
/// content attach only to this camera.
#[derive(Component)]
pub struct CinematicViewport;

/// The active schematic's `environment` node, set on schematic load
/// (`None` = default editor look).
#[derive(Resource, Default, Clone)]
pub struct SceneEnvironment(pub Option<EnvironmentConfig>);

/// Baked IBL intensity viewport cameras spawn with (see `ViewportPane::spawn`);
/// `environment { ambient scale=… }` multiplies this.
pub const BASE_ENVIRONMENT_MAP_INTENSITY: f32 = 2000.0;

/// Marker for the sun spawned from the schematic `environment` node.
/// `pub(crate)`: the cinematic-earth plugin reads its direction.
#[derive(Component)]
pub(crate) struct SchematicSun;

/// Marker + source config for the atmosphere spawned from the schematic
/// `environment` node; the stored config detects edits (hot reload).
/// `pub(crate)`: the cinematic-earth plugin tunes density and surface radius.
#[derive(Component)]
pub(crate) struct SchematicAtmosphere(AtmosphereConfig);

/// The atmosphere the scene should actually run: the explicit `atmosphere`
/// child, or the built-in cinematic Earth's derived one (`environment {
/// earth }` owns its atmosphere: raymarched, ECEF origin; the cinematic-earth
/// plugin then drives density and surface radius from the camera).
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

/// Explicit `sun`, or a default (ephemeris, 100 klx, shadows) when `earth`
/// is set with no sun of its own.
fn effective_sun(env: &EnvironmentConfig) -> Option<SunConfig> {
    env.sun
        .or_else(|| env.earth.is_some().then(SunConfig::default))
}

fn atmosphere_geo_rotation(frame: GeoFrame, ctx: &GeoContext) -> GeoRotation {
    // Since #774, `GeoRotation::to_bevy` composes the frame basis for both
    // rotation kinds. Store its inverse so a spherical atmosphere stays
    // unrotated in Bevy, matching its pre-#774 placement.
    GeoRotation::from_bevy(frame, DQuat::IDENTITY, ctx)
}

pub struct SceneEnvironmentPlugin;

impl Plugin for SceneEnvironmentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SceneEnvironment>()
            .add_systems(Update, (sync_sun, sync_atmosphere, sync_camera_environment));
    }
}

/// Sun rotation from azimuth/elevation degrees (Bevy Y-up frame; matches the
/// pyrotechnique authoring convention so scene values transcribe directly).
/// A missing angle uses the historical default so a partially-specified sun
/// behaves as it did before az/el became optional.
fn sun_rotation(sun: &SunConfig) -> Quat {
    let az = sun.azimuth_deg.unwrap_or(SunConfig::default_azimuth_deg());
    let el = sun
        .elevation_deg
        .unwrap_or(SunConfig::default_elevation_deg());
    Quat::from_euler(EulerRot::YXZ, -az.to_radians(), -el.to_radians(), 0.0)
}

/// Rotate a "toward the sun" vector in `frame` into a Bevy directional-light
/// orientation (light travels −Z).
fn rotation_from_frame_direction(to_sun: DVec3, frame: GeoFrame, ctx: &GeoContext) -> Option<Quat> {
    let to_sun = (GeoFrame::bevy_R_(&frame, ctx) * to_sun).try_normalize()?;
    Some(Quat::from_rotation_arc(Vec3::NEG_Z, (-to_sun).as_vec3()))
}

/// World-frame sun rotation. Most explicit wins: `direction`, then az/el,
/// then the ECEF ephemeris at `unix_micros`.
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
    RenderLayers::layer(CINEMATIC_EARTH_RENDER_LAYER)
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

/// Spawns/despawns the planetary atmosphere declared by the schematic. The
/// entity lives in the same high-precision space as world objects (GeoPosition
/// in the schematic frame + big_space grid cell), so the planet center stays
/// put through floating-origin rebases whether the scene is a local ENU pad or
/// full ECEF Earth.
fn sync_atmosphere(
    mut commands: Commands,
    environment: Res<SceneEnvironment>,
    coordinate: Res<crate::Coordinate>,
    geo_ctx: Res<GeoContext>,
    mut media: ResMut<Assets<ScatteringMedium>>,
    existing: Query<(Entity, &SchematicAtmosphere)>,
    #[cfg(feature = "big_space")] root: Option<Res<crate::spatial::BigSpaceRootEntity>>,
) {
    let config = environment.0.as_ref().and_then(effective_atmosphere);
    let current = existing.iter().next();
    match (config, current) {
        (Some(config), Some((entity, spawned))) if spawned.0 == config => {
            let _ = entity;
        }
        (Some(config), current) => {
            if let Some((entity, _)) = current {
                commands.entity(entity).despawn();
            }
            let frame = coordinate.0.unwrap_or_default();
            let medium = media.add(ScatteringMedium::earth(256, 256));
            let (r, g, b) = config.ground_albedo;
            let mut entity = commands.spawn((
                SchematicAtmosphere(config),
                Name::new("environment atmosphere"),
                bevy::light::Atmosphere {
                    inner_radius: config.inner_radius,
                    outer_radius: config.outer_radius,
                    ground_albedo: Vec3::new(r, g, b),
                    medium,
                },
                Transform::default(),
                GlobalTransform::default(),
                #[cfg(feature = "big_space")]
                crate::spatial::GridCell::default(),
                GeoPosition(
                    frame,
                    DVec3::new(config.origin.0, config.origin.1, config.origin.2),
                ),
                atmosphere_geo_rotation(frame, &geo_ctx),
            ));
            #[cfg(feature = "big_space")]
            crate::spatial::parent_under_big_space(&mut entity, root.as_deref());
            let _ = &mut entity;
        }
        (None, Some((entity, _))) => {
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

/// Keeps main viewport cameras in sync with the environment: IBL intensity
/// scaling, sky (clear) color, and the per-camera `AtmosphereSettings` that
/// activates the schematic atmosphere. Runs every frame because cameras can
/// spawn at any time; writes are change-gated.
fn atmosphere_settings_for(config: AtmosphereConfig) -> AtmosphereSettings {
    if config.raymarched {
        // Distant planet views (apollo Earth-from-Moon, ~2° disk): raymarch
        // with a larger sky-view LUT so the blue limb stays resolvable.
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
            // Ground ECEF scenes (falcon9): default LookupTexture; longer
            // aerial-view span for chase cams watching multi-km plumes.
            aerial_view_lut_max_distance: 3.2e5,
            ..AtmosphereSettings::default()
        }
    }
}

fn sync_camera_environment(
    mut commands: Commands,
    environment: Res<SceneEnvironment>,
    viewer_frame: Res<crate::plugins::cinematic_earth::ViewerFrame>,
    cinematic: Query<Entity, With<CinematicViewport>>,
    mut cameras: Query<
        (
            Entity,
            &mut Camera,
            &mut EnvironmentMapLight,
            Option<&AtmosphereSettings>,
            Has<CinematicViewport>,
        ),
        With<MainCamera>,
    >,
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
    let cinematic_clear = match sky_color {
        Some(color) => ClearColorConfig::Custom(Color::srgba(color.r, color.g, color.b, color.a)),
        None if earth => ClearColorConfig::Custom(Color::BLACK),
        None => ClearColorConfig::Default,
    };
    let default_clear = match sky_color {
        Some(color) if !earth => {
            ClearColorConfig::Custom(Color::srgba(color.r, color.g, color.b, color.a))
        }
        _ => ClearColorConfig::Default,
    };
    // Bevy 0.19 fatally fails wgpu validation when several active views carry
    // AtmosphereSettings. Earth scenes pin it to the cinematic camera. Plain
    // `atmosphere` (apollo-lander) still elects the lowest-id active camera.
    let cinematic_cam = cinematic.iter().next();
    let chosen = if earth {
        atmosphere.and(cinematic_cam)
    } else {
        let active: Vec<Entity> = cameras
            .iter()
            .filter(|(_, camera, ..)| camera.is_active)
            .map(|(entity, ..)| entity)
            .collect();
        if atmosphere.is_some() && active.len() > 1 {
            warn_once!(
                "schematic atmosphere renders on only one active main viewport \
                 (Bevy 0.19: several cameras with AtmosphereSettings trip wgpu \
                 bind-group validation and quit the editor). Other viewports keep \
                 clear-color/IBL; switch tabs to move the sky to another pane."
            );
        }
        atmosphere.and(active.into_iter().min())
    };
    // Pyrotechnique kills ambient in space (`ambient × (1 − space_vis)`): the
    // studio IBL has no business lighting the night globe from LEO.
    let space_ibl_fade = 1.0 - viewer_frame.space_vis;
    for (entity, mut camera, mut light, current_settings, is_cinematic) in &mut cameras {
        let (target_intensity, target_clear) = if earth {
            if is_cinematic {
                (intensity * space_ibl_fade, cinematic_clear)
            } else {
                (BASE_ENVIRONMENT_MAP_INTENSITY, default_clear)
            }
        } else {
            (intensity, cinematic_clear)
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
        // Light forward is -Z rotated by the sun rotation; positive elevation
        // must tilt it below the horizon (negative Y).
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
    fn atmosphere_rotation_is_bevy_identity_in_every_frame() {
        let ctx = GeoContext::default();
        for frame in [GeoFrame::ENU, GeoFrame::NED, GeoFrame::ECEF] {
            let rendered = atmosphere_geo_rotation(frame, &ctx).to_bevy(&ctx);
            assert!(
                rendered.dot(DQuat::IDENTITY).abs() > 1.0 - 1e-9,
                "{frame:?} rendered as {rendered:?}"
            );
        }
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
