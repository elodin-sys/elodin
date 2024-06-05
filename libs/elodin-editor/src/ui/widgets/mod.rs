use bevy::{
    ecs::{
        query::With,
        system::{Resource, SystemParam, SystemState},
        world::{Mut, World},
    },
    window::PrimaryWindow,
};
use bevy_egui::{egui, EguiContext};
use std::collections::HashMap;

pub mod button;
pub mod command_palette;
pub mod hierarchy;
pub mod inspector;
pub mod label;
pub mod modal;
pub mod plot;
pub mod status_bar;
pub mod timeline;

/// world.RootWidget

pub trait RootWidgetSystemExt {
    fn add_root_widget<S: RootWidgetSystem<Args = ()> + 'static>(&mut self, id: &str) -> S::Output {
        self.add_root_widget_with::<S>(id, ())
    }

    fn add_root_widget_with<S: RootWidgetSystem + 'static>(
        &mut self,
        id: &str,
        args: S::Args,
    ) -> S::Output;

    fn egui_context_scope<R>(&mut self, f: impl FnOnce(&mut Self, egui::Context) -> R) -> R;
}

impl RootWidgetSystemExt for World {
    fn add_root_widget_with<S: RootWidgetSystem + 'static>(
        &mut self,
        id: &str,
        args: S::Args,
    ) -> S::Output {
        self.egui_context_scope(|world, mut ctx| {
            let id = WidgetId::new(id);

            if !world.contains_resource::<StateInstances<S>>() {
                world.insert_resource(StateInstances::<S> {
                    instances: HashMap::new(),
                });
            }

            world.resource_scope(|world, mut states: Mut<StateInstances<S>>| {
                let cached_state = states
                    .instances
                    .entry(id)
                    .or_insert_with(|| SystemState::new(world));
                let output = S::ctx_system(world, cached_state, &mut ctx, args);
                cached_state.apply(world);
                output
            })
        })
    }

    fn egui_context_scope<R>(&mut self, f: impl FnOnce(&mut Self, egui::Context) -> R) -> R {
        let mut state =
            self.query_filtered::<&mut EguiContext, (With<EguiContext>, With<PrimaryWindow>)>();
        let mut egui_ctx = state.single_mut(self);
        let ctx = egui_ctx.get_mut().clone();
        f(self, ctx)
    }
}

/// ui.Widget

pub trait WidgetSystemExt {
    fn add_widget<S: WidgetSystem<Args = ()> + 'static>(
        &mut self,
        world: &mut World,
        id: &str,
    ) -> S::Output {
        self.add_widget_with::<S>(world, id, ())
    }

    fn add_widget_with<S: WidgetSystem + 'static>(
        &mut self,
        world: &mut World,
        id: &str,
        args: S::Args,
    ) -> S::Output;
}

impl WidgetSystemExt for egui::Ui {
    fn add_widget_with<S: WidgetSystem + 'static>(
        &mut self,
        world: &mut World,
        id: &str,
        args: S::Args,
    ) -> S::Output {
        let id = WidgetId::new(id);

        if !world.contains_resource::<StateInstances<S>>() {
            world.insert_resource(StateInstances::<S> {
                instances: HashMap::new(),
            });
        }

        world.resource_scope(|world, mut states: Mut<StateInstances<S>>| {
            let cached_state = states
                .instances
                .entry(id)
                .or_insert_with(|| SystemState::new(world));
            let output = S::ui_system(world, cached_state, self, args);
            cached_state.apply(world);
            output
        })
    }
}

/// Widget Traits

pub trait RootWidgetSystem: SystemParam {
    type Args;
    type Output;

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        args: Self::Args,
    ) -> Self::Output;
}

pub trait WidgetSystem: SystemParam {
    type Args;
    type Output;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) -> Self::Output;
}

#[derive(Resource, Default)]
struct StateInstances<T: SystemParam + 'static> {
    instances: HashMap<WidgetId, SystemState<T>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WidgetId(pub u64);

impl WidgetId {
    pub const fn new(str: &str) -> Self {
        Self(conduit::const_fnv1a_hash::fnv1a_hash_str_64(str))
    }
}
