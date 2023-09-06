pub(crate) use self::sealed::EditorEnv;
use self::{traces::TracesPlugin, ui::*};
use crate::{
    xpbd::{
        builder::{Assets, AssetsInner, Env, FromEnv, SimBuilder, XpbdBuilder},
        plugin::XpbdPlugin,
    },
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
use std::ops::DerefMut;

pub(crate) mod traces;
mod ui;

pub(crate) mod sealed {
    use std::cell::RefCell;

    use bevy::prelude::App;
    use bevy_ecs::system::CommandQueue;

    pub struct EditorEnv {
        pub app: App,
        pub command_queue: RefCell<CommandQueue>,
    }

    impl EditorEnv {
        pub(crate) fn new(app: App) -> EditorEnv {
            EditorEnv {
                app,
                command_queue: Default::default(),
            }
        }
    }
}

impl Env for EditorEnv {
    type Param<'e> = &'e EditorEnv;

    fn param(&mut self) -> Self::Param<'_> {
        self
    }
}

impl<'a> FromEnv<EditorEnv> for Assets<'a> {
    type Item<'e> = Assets<'e>;

    fn from_env(env: <EditorEnv as Env>::Param<'_>) -> Self::Item<'_> {
        let unsafe_world_cell = env.app.world.as_unsafe_world_cell_readonly();
        let meshes = unsafe { unsafe_world_cell.get_resource_mut().unwrap() };
        let materials = unsafe { unsafe_world_cell.get_resource_mut().unwrap() };

        Assets(Some(AssetsInner { meshes, materials }))
    }

    fn init(_: &mut EditorEnv) {}
}

pub fn editor<T>(sim_builder: impl SimBuilder<T, EditorEnv>) {
    let mut app = App::new();
    app.add_plugins(EditorPlugin);
    let mut editor_env = EditorEnv::new(app);
    sim_builder.build(&mut editor_env);
    let EditorEnv {
        mut app,
        command_queue,
    } = editor_env;
    let mut command_queue = command_queue.into_inner();
    command_queue.apply(&mut app.world);
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
        .add_plugins(XpbdPlugin)
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
        .insert_resource(Msaa::Off)
        .insert_resource(crate::Time(0.0))
        .insert_resource(super::components::Config::default());
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

impl<'a> FromEnv<EditorEnv> for XpbdBuilder<'a> {
    type Item<'t> = XpbdBuilder<'t>;

    fn init(_env: &mut EditorEnv) {}

    fn from_env(env: <EditorEnv as Env>::Param<'_>) -> Self::Item<'_> {
        XpbdBuilder {
            queue: env.command_queue.borrow_mut(),
            entities: env.app.world.entities(),
        }
    }
}

#[derive(Resource, Clone, Debug, Default)]
pub struct ObservableInput(pub ObservableNum<f64>);
impl Editable for ObservableInput {
    fn build(&mut self, ui: &mut Ui) {
        let mut num = self.0.load();
        ui.add(egui::Slider::new(num.deref_mut(), -1.25..=1.25).text("input"));
    }
}
