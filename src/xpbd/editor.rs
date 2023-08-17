use std::ops::DerefMut;

use bevy::{prelude::*, DefaultPlugins};
use bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell;
use bevy_egui::{
    egui::{self, Ui},
    EguiContexts, EguiPlugin,
};

use crate::{Att, Pos, SharedNum};

use self::sealed::EditorEnv;

use super::{
    builder::{Assets, AssetsInner},
    systems::{self, SubstepSchedule},
    Env, FromEnv, SimBuilder,
};

pub(crate) mod sealed {
    use bevy::prelude::App;

    pub struct EditorEnv {
        pub app: App,
    }

    impl EditorEnv {
        pub(crate) fn new(app: App) -> EditorEnv {
            EditorEnv { app }
        }
    }
}

impl Env for EditorEnv {
    type Param<'e> = UnsafeWorldCell<'e>;

    fn param(&mut self) -> Self::Param<'_> {
        self.app.world.as_unsafe_world_cell()
    }
}

impl<'a> FromEnv<EditorEnv> for Assets<'a> {
    type Item<'e> = Assets<'e>;

    fn from_env(env: <EditorEnv as Env>::Param<'_>) -> Self::Item<'_> {
        let meshes = unsafe { env.get_resource_mut().unwrap() };
        let materials = unsafe { env.get_resource_mut().unwrap() };

        Assets(Some(AssetsInner { meshes, materials }))
    }

    fn init(_: &mut EditorEnv) {}
}

pub fn editor<T>(sim_builder: impl SimBuilder<T, EditorEnv>) {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, ui_system)
        .add_systems(Update, (tick).in_set(TickSet::TickPhysics))
        .add_schedule(SubstepSchedule, systems::schedule())
        .add_systems(
            Update,
            (sync_pos)
                .in_set(TickSet::SyncPos)
                .after(TickSet::TickPhysics),
        )
        .insert_resource(crate::Time(0.0))
        .insert_resource(super::components::Config { dt: 1.0 / 60.0 });
    let mut editor_env = EditorEnv::new(app);
    let queue = sim_builder.build(&mut editor_env);
    queue.apply(&mut editor_env.app.world);
    editor_env.app.run()
}

fn ui_system(mut contexts: EguiContexts, mut editables: ResMut<Editables>) {
    egui::Window::new("Hello").show(contexts.ctx_mut(), |ui| {
        for editable in &mut editables.0 {
            editable.build(ui);
        }
        ui.label("world");
    });
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<bevy::prelude::Assets<Mesh>>,
    mut materials: ResMut<bevy::prelude::Assets<StandardMaterial>>,
) {
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

#[derive(SystemSet, Debug, PartialEq, Eq, Hash, Clone)]
enum TickSet {
    TickPhysics,
    SyncPos,
}

pub fn sync_pos(mut query: Query<(&mut Transform, &Pos, &Att)>) {
    query
        .par_iter_mut()
        .for_each_mut(|(mut transform, Pos(pos), Att(att))| {
            transform.translation = Vec3 {
                x: pos.x as f32,
                y: pos.y as f32,
                z: pos.z as f32,
            };
            transform.rotation = Quat {
                x: att.i as f32,
                y: att.j as f32,
                z: att.k as f32,
                w: att.w as f32,
            }; // TODO: Is `Quat` a JPL quat who knows?!
        });
}

pub fn tick(world: &mut World) {
    world.run_schedule(SubstepSchedule)
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
        unsafe { env.get_resource::<F>().expect("missing resource").clone() }
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

#[derive(Resource)]
pub struct Editables(Vec<Box<dyn Editable>>);
