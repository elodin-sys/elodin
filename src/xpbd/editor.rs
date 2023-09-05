use std::ops::DerefMut;

use bevy::{
    core_pipeline::experimental::taa::{TemporalAntiAliasBundle, TemporalAntiAliasPlugin},
    pbr::{
        DirectionalLightShadowMap, ScreenSpaceAmbientOcclusionBundle,
        ScreenSpaceAmbientOcclusionQualityLevel, ScreenSpaceAmbientOcclusionSettings,
    },
    prelude::*,
    DefaultPlugins,
};
use bevy_atmosphere::prelude::*;
use bevy_egui::{
    egui::{self, Ui},
    EguiContexts, EguiPlugin,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use crate::{ObservableNum, SharedNum};

use self::sealed::EditorEnv;

use super::{
    builder::{Assets, AssetsInner, XpbdBuilder},
    plugin::XpbdPlugin,
    Env, FromEnv, SimBuilder,
};

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

    app.add_plugins((DefaultPlugins, TemporalAntiAliasPlugin))
        .add_plugins(EguiPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(AtmospherePlugin)
        .add_plugins(XpbdPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, ui_system)
        .insert_resource(AtmosphereModel::new(Gradient {
            horizon: Color::hex("1B2642").unwrap(),
            sky: Color::hex("1B2642").unwrap(),
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

fn ui_system(mut contexts: EguiContexts, mut editables: ResMut<Editables>) {
    egui::Window::new("Hello").show(contexts.ctx_mut(), |ui| {
        for editable in &mut editables.0 {
            editable.build(ui);
        }
        ui.label("world");
    });
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(ScreenSpaceAmbientOcclusionSettings {
        quality_level: ScreenSpaceAmbientOcclusionQualityLevel::High,
    });

    // camera
    commands
        .spawn(Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 10.0)),
            ..default()
        })
        .insert(AtmosphereCamera::default())
        .insert(PanOrbitCamera::default())
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

impl Editable for Input {
    fn build(&mut self, ui: &mut Ui) {
        let mut num = self.0.load();
        ui.add(egui::Slider::new(num.deref_mut(), -1.25..=1.25));
    }
}

pub trait Editable: Send + Sync {
    fn build(&mut self, ui: &mut Ui);
}

impl<F: Editable + Clone + Resource + Default> FromEnv<EditorEnv> for F {
    type Item<'a> = F;

    fn from_env(env: <EditorEnv as Env>::Param<'_>) -> Self::Item<'_> {
        env.app
            .world
            .get_resource::<F>()
            .expect("missing resource")
            .clone()
    }

    fn init(env: &mut EditorEnv) {
        let f = F::default();
        let mut editables = env
            .app
            .world
            .get_resource_or_insert_with(|| Editables(vec![]));
        editables.0.push(Box::new(f.clone()));
        env.app.world.insert_resource(f);
    }
}

#[derive(Resource, Default)]
pub struct Editables(Vec<Box<dyn Editable>>);

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
        ui.add(egui::Slider::new(num.deref_mut(), -1.25..=1.25));
    }
}
