use std::time::Duration;

use bevy::{
    app::{App, Plugin, ScheduleRunnerPlugin, Startup},
    asset::{AssetPlugin, UnapprovedPathMode},
    log::LogPlugin,
    prelude::*,
    window::{ExitCondition, WindowPlugin},
    winit::WinitPlugin,
};
use big_space::{FloatingOrigin, GridCell};

use crate::{PositionSync, sensor_camera::SensorCameraPlugin, sync_pos};
use impeller2_wkt::{CurrentTimestamp, LastUpdated};

/// A headless variant of the editor that provides rendering without a window.
///
/// Used by `elodin run` to render sensor camera frames into the DB
/// without opening a GUI. Includes the same scene loading, entity
/// transform sync, and sensor camera rendering as the full editor,
/// but skips all UI (egui, tiles, graphs, video streams, gizmos, etc.).
pub struct HeadlessEditorPlugin;

impl Plugin for HeadlessEditorPlugin {
    fn build(&self, app: &mut App) {
        // Asset sources must be registered before DefaultPlugins/AssetPlugin
        app.add_plugins(crate::plugins::WebAssetPlugin)
            .add_plugins(crate::plugins::env_asset_source::plugin)
            .add_plugins(
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: None,
                        exit_condition: ExitCondition::DontExit,
                        ..default()
                    })
                    .disable::<WinitPlugin>()
                    .disable::<LogPlugin>()
                    .set(AssetPlugin {
                        unapproved_path_mode: UnapprovedPathMode::Allow,
                        ..default()
                    }),
            )
            .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
                1.0 / 120.0,
            )))
            .add_plugins(impeller2_bevy::Impeller2Plugin)
            .add_plugins(big_space::FloatingOriginPlugin::<i128>::new(16_000., 100.))
            .add_plugins(bevy_mat3_material::Mat3MaterialPlugin)
            .add_plugins(crate::object_3d::Object3DPlugin)
            .add_plugins(SensorCameraPlugin)
            .add_systems(
                PreUpdate,
                advance_headless_timestamp.after(impeller2_bevy::sink),
            )
            .add_systems(
                PreUpdate,
                (
                    impeller2_bevy::apply_cached_data,
                    crate::object_3d::update_object_3d_system,
                    crate::sync_object_3d,
                    sync_pos,
                )
                    .chain()
                    .after(advance_headless_timestamp)
                    .in_set(PositionSync),
            )
            .add_systems(Startup, setup_floating_origin)
            .init_resource::<crate::EqlContext>()
            .init_resource::<crate::SyncedObject3d>()
            .add_systems(Update, crate::update_eql_context);
    }
}

/// In headless mode there is no playback UI, so `CurrentTimestamp` must
/// track the latest simulation tick directly. Without this, entity data
/// would stay pinned to tick 0 and sensor frame timestamps would be stale.
fn advance_headless_timestamp(
    last_updated: Res<LastUpdated>,
    mut current_ts: ResMut<CurrentTimestamp>,
) {
    if last_updated.0 .0 > i64::MIN {
        current_ts.0 = last_updated.0;
    }
}

fn setup_floating_origin(mut commands: Commands) {
    commands.spawn((
        FloatingOrigin,
        GridCell::<i128>::default(),
        Transform::default(),
        GlobalTransform::default(),
    ));
}
