//! Applies the schematic's top-level `environment` node: a directional sun
//! (with optional shadow maps), ambient/IBL scaling, and viewport sky color.
//!
//! Without an `environment` node the editor renders exactly as before: baked
//! `EnvironmentMapLight` IBL, no sun, theme-colored background. See
//! docs/design-thruster-effects-port.md §4.2 for the schema and rationale.

use bevy::camera::ClearColorConfig;
use bevy::light::SunDisk;
use bevy::light::atmosphere::ScatteringMedium;
use bevy::math::{DQuat, DVec3};
use bevy::pbr::{AtmosphereMode, AtmosphereSettings};
// EnvironmentMapLight comes in via the prelude (bevy_light).
use bevy::prelude::*;
use bevy_geo_frames::{GeoPosition, GeoRotation};
use impeller2_wkt::{AtmosphereConfig, EnvironmentConfig, SunConfig};

use crate::MainCamera;

/// The active schematic's `environment` node, set on schematic load
/// (`None` = default editor look).
#[derive(Resource, Default, Clone)]
pub struct SceneEnvironment(pub Option<EnvironmentConfig>);

/// Baked IBL intensity viewport cameras spawn with (see `ViewportPane::spawn`);
/// `environment { ambient scale=… }` multiplies this.
pub const BASE_ENVIRONMENT_MAP_INTENSITY: f32 = 2000.0;

/// Marker for the sun spawned from the schematic `environment` node.
#[derive(Component)]
struct SchematicSun;

/// Marker + source config for the atmosphere spawned from the schematic
/// `environment` node; the stored config detects edits (hot reload).
#[derive(Component)]
struct SchematicAtmosphere(AtmosphereConfig);

pub struct SceneEnvironmentPlugin;

impl Plugin for SceneEnvironmentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SceneEnvironment>()
            .add_systems(Update, (sync_sun, sync_atmosphere, sync_camera_environment));
    }
}

/// Sun rotation from azimuth/elevation degrees (Bevy Y-up frame; matches the
/// pyrotechnique authoring convention so scene values transcribe directly).
fn sun_rotation(sun: &SunConfig) -> Quat {
    Quat::from_euler(
        EulerRot::YXZ,
        -sun.azimuth_deg.to_radians(),
        -sun.elevation_deg.to_radians(),
        0.0,
    )
}

fn sync_sun(
    mut commands: Commands,
    environment: Res<SceneEnvironment>,
    mut suns: Query<(Entity, &mut DirectionalLight, &mut Transform), With<SchematicSun>>,
) {
    let config = environment.0.as_ref().and_then(|env| env.sun);
    match (config, suns.iter_mut().next()) {
        (Some(sun), Some((_, mut light, mut transform))) => {
            // Compare before writing: mutations dirty render extraction.
            if light.illuminance != sun.illuminance {
                light.illuminance = sun.illuminance;
            }
            if light.shadow_maps_enabled != sun.shadows {
                light.shadow_maps_enabled = sun.shadows;
            }
            let rotation = sun_rotation(&sun);
            if transform.rotation != rotation {
                transform.rotation = rotation;
            }
        }
        (Some(sun), None) => {
            commands.spawn((
                SchematicSun,
                Name::new("environment sun"),
                DirectionalLight {
                    illuminance: sun.illuminance,
                    shadow_maps_enabled: sun.shadows,
                    ..default()
                },
                SunDisk::EARTH,
                Transform::from_rotation(sun_rotation(&sun)),
            ));
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
    mut media: ResMut<Assets<ScatteringMedium>>,
    existing: Query<(Entity, &SchematicAtmosphere)>,
    #[cfg(feature = "big_space")] root: Option<Res<crate::spatial::BigSpaceRootEntity>>,
) {
    let config = environment.0.as_ref().and_then(|env| env.atmosphere);
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
                GeoRotation::relative(frame, DQuat::IDENTITY),
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
    mut cameras: Query<
        (
            Entity,
            &mut Camera,
            &mut EnvironmentMapLight,
            Option<&AtmosphereSettings>,
        ),
        With<MainCamera>,
    >,
) {
    let (ambient_scale, sky_color, atmosphere) = match &environment.0 {
        Some(config) => (
            config.ambient_scale.max(0.0),
            config.sky_color,
            config.atmosphere,
        ),
        None => (1.0, None, None),
    };
    let intensity = BASE_ENVIRONMENT_MAP_INTENSITY * ambient_scale;
    let clear = match sky_color {
        Some(color) => ClearColorConfig::Custom(Color::srgba(color.r, color.g, color.b, color.a)),
        None => ClearColorConfig::Default,
    };
    // The atmosphere entity is scene-global; Bevy still requires per-camera
    // `AtmosphereSettings` to render it. Putting settings on several active
    // views in one frame trips wgpu bind-group validation and the editor's
    // fatal render-error handler. Pick the lowest-id active main camera —
    // deterministic across frames; tab switches hand the sky over a frame
    // later. Multi-viewport layouts get the procedural sky in one pane only.
    let active: Vec<Entity> = cameras
        .iter()
        .filter(|(_, camera, _, _)| camera.is_active)
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
    let chosen = atmosphere.and(active.into_iter().min());
    for (entity, mut camera, mut light, current_settings) in &mut cameras {
        if light.intensity != intensity {
            light.intensity = intensity;
        }
        if !clear_color_matches(&camera.clear_color, &clear) {
            camera.clear_color = clear;
        }
        // `AtmosphereSettings` requires `Hdr`, which main viewport cameras get
        // from the global HDR toggle; cinematic schematics declare hdr=#true.
        let wants_atmosphere = chosen == Some(entity);
        if wants_atmosphere {
            let desired = atmosphere_settings_for(atmosphere.unwrap());
            // AtmosphereMode is not PartialEq; compare the fields we set.
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
            azimuth_deg: 0.0,
            elevation_deg: 45.0,
            illuminance: 100_000.0,
            shadows: true,
        };
        // Light forward is -Z rotated by the sun rotation; positive elevation
        // must tilt it below the horizon (negative Y).
        let forward = sun_rotation(&sun) * Vec3::NEG_Z;
        assert!(forward.y < -0.5, "sun should shine downward, got {forward}");
    }
}
