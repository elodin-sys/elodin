use super::Input;
use crate::xpbd::{
    builder::{Env, FromEnv},
    editor::EditorEnv,
};
use bevy::prelude::*;
use bevy_egui::{
    egui::{self, Ui},
    EguiContexts,
};
use std::ops::DerefMut;

pub(crate) fn ui_system(mut contexts: EguiContexts, mut editables: ResMut<Editables>) {
    egui::Window::new("Inputs").show(contexts.ctx_mut(), |ui| {
        for editable in &mut editables.0 {
            editable.build(ui);
        }
    });
}

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
