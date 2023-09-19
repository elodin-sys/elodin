use self::{traces::TracesPlugin, ui::*};
use crate::{
    xpbd::builder::{Assets, AssetsInner, Env, FromEnv, XpbdBuilder},
    ObservableNum, SharedNum,
};
use bevy::{
    core_pipeline::{
        bloom::BloomSettings,
        experimental::taa::{TemporalAntiAliasBundle, TemporalAntiAliasPlugin},
        tonemapping::Tonemapping,
    },
    pbr::{
        DirectionalLightShadowMap, ScreenSpaceAmbientOcclusionBundle,
        ScreenSpaceAmbientOcclusionQualityLevel, ScreenSpaceAmbientOcclusionSettings,
    },
    prelude::*,
    window::WindowTheme,
    DefaultPlugins,
};
use bevy_atmosphere::prelude::*;
use bevy_egui::{
    egui::{self, Ui},
    EguiPlugin,
};
use bevy_infinite_grid::{GridShadowCamera, InfiniteGrid, InfiniteGridBundle, InfiniteGridPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_polyline::PolylinePlugin;
use paracosm_macros::Editable;
use std::ops::DerefMut;

use super::runner::{IntoSimRunner, SimRunnerEnv};

pub(crate) mod traces;
mod ui;

impl<'a> FromEnv<SimRunnerEnv> for Assets<'a> {
    type Item<'e> = Assets<'e>;

    fn from_env(env: <SimRunnerEnv as Env>::Param<'_>) -> Self::Item<'_> {
        let unsafe_world_cell = env.app.world.as_unsafe_world_cell_readonly();
        let meshes = unsafe { unsafe_world_cell.get_resource_mut().unwrap() };
        let materials = unsafe { unsafe_world_cell.get_resource_mut().unwrap() };

        Assets(Some(AssetsInner { meshes, materials }))
    }

    fn init(_: &mut SimRunnerEnv) {}
}

pub fn editor<'a, T>(func: impl IntoSimRunner<'a, T>) {
    let runner = func.into_runner();
    let mut app = runner.build_with_plugins(EditorPlugin);
    app.run()
}

pub struct EditorPlugin;
impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                window_theme: Some(WindowTheme::Dark),
                title: "Paracosm Editor".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(TemporalAntiAliasPlugin)
        .add_plugins(EguiPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(AtmospherePlugin)
        .add_plugins(PolylinePlugin)
        .add_plugins(TracesPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, ui_system)
        .insert_resource(AtmosphereModel::new(Gradient {
            sky: Color::hex("1B2642").unwrap(),
            horizon: Color::hex("00081E").unwrap(),
            ground: Color::hex("#00081E").unwrap(),
        }))
        .insert_resource(AmbientLight {
            color: Color::hex("#FFF").unwrap(),
            brightness: 1.0,
        })
        .insert_resource(Editables::default())
        .insert_resource(ClearColor(Color::hex("#16161A").unwrap()))
        .insert_resource(DirectionalLightShadowMap { size: 8192 })
        .insert_resource(Msaa::Off);
    }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(ScreenSpaceAmbientOcclusionSettings {
        quality_level: ScreenSpaceAmbientOcclusionQualityLevel::High,
    });

    commands.spawn(InfiniteGridBundle {
        grid: InfiniteGrid {
            minor_line_color: Color::hex("#00081E").unwrap(),
            major_line_color: Color::hex("#00081E").unwrap(),
            x_axis_color: Color::hex("F46E22").unwrap(),
            ..Default::default()
        },
        ..default()
    });

    commands
        .spawn(Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(5.0, 5.0, 10.0)),
            camera: Camera {
                hdr: true,
                ..Default::default()
            },
            tonemapping: Tonemapping::TonyMcMapface,
            ..default()
        })
        .insert(BloomSettings { ..default() })
        .insert(AtmosphereCamera::default())
        .insert(PanOrbitCamera::default())
        .insert(GridShadowCamera)
        .insert(EnvironmentMapLight {
            diffuse_map: asset_server.load("diffuse.ktx2"),
            specular_map: asset_server.load("specular.ktx2"),
        })
        .insert(ScreenSpaceAmbientOcclusionBundle {
            settings: ScreenSpaceAmbientOcclusionSettings {
                quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Ultra,
            },
            ..Default::default()
        })
        .insert(TemporalAntiAliasBundle::default());
}

#[derive(Resource, Clone, Debug, Default)]
pub struct Input(pub SharedNum<f64>);

impl<'a> FromEnv<SimRunnerEnv> for XpbdBuilder<'a> {
    type Item<'t> = XpbdBuilder<'t>;

    fn init(_env: &mut SimRunnerEnv) {}

    fn from_env(env: <SimRunnerEnv as Env>::Param<'_>) -> Self::Item<'_> {
        XpbdBuilder {
            queue: env.command_queue.borrow_mut(),
            entities: env.app.world.entities(),
        }
    }
}

#[derive(Editable, Resource, Clone, Debug, Default)]
#[editable(slider, range_min = "-1.25", range_max = 1.25, name = "input")]
pub struct ObservableInput(pub ObservableNum<f64>);
