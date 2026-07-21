//! Applies the schematic's top-level `environment` node: a directional sun
//! (with optional shadow maps), ambient/IBL scaling, and viewport sky color.
//!
//! Without an `environment` node the editor renders exactly as before: baked
//! `EnvironmentMapLight` IBL, no sun, theme-colored background. See
//! docs/design-thruster-effects-port.md §4.2 for the schema and rationale.

use bevy::camera::ClearColorConfig;
use bevy::light::SunDisk;
// EnvironmentMapLight comes in via the prelude (bevy_light).
use bevy::prelude::*;
use impeller2_wkt::{EnvironmentConfig, SunConfig};

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

pub struct SceneEnvironmentPlugin;

impl Plugin for SceneEnvironmentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SceneEnvironment>()
            .add_systems(Update, (sync_sun, sync_camera_environment));
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

fn clear_color_matches(current: &ClearColorConfig, desired: &ClearColorConfig) -> bool {
    match (current, desired) {
        (ClearColorConfig::Default, ClearColorConfig::Default) => true,
        (ClearColorConfig::Custom(a), ClearColorConfig::Custom(b)) => a == b,
        _ => false,
    }
}

/// Keeps main viewport cameras in sync with the environment: IBL intensity
/// scaling and sky (clear) color. Runs every frame because cameras can spawn
/// at any time; writes are change-gated.
fn sync_camera_environment(
    environment: Res<SceneEnvironment>,
    mut cameras: Query<(&mut Camera, &mut EnvironmentMapLight), With<MainCamera>>,
) {
    let (ambient_scale, sky_color) = match &environment.0 {
        Some(config) => (config.ambient_scale.max(0.0), config.sky_color),
        None => (1.0, None),
    };
    let intensity = BASE_ENVIRONMENT_MAP_INTENSITY * ambient_scale;
    let clear = match sky_color {
        Some(color) => ClearColorConfig::Custom(Color::srgba(color.r, color.g, color.b, color.a)),
        None => ClearColorConfig::Default,
    };
    for (mut camera, mut light) in &mut cameras {
        if light.intensity != intensity {
            light.intensity = intensity;
        }
        if !clear_color_matches(&camera.clear_color, &clear) {
            camera.clear_color = clear;
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
