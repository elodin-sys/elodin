use crate::{
    builder::{Env, FromEnv},
    runner::SimRunnerEnv,
};
use bevy::{
    asset::{Asset, AssetPath},
    prelude::*,
};
use bevy_ecs::world::Mut;

pub struct Assets<'a>(pub Option<AssetsInner<'a>>);

pub struct AssetsInner<'a> {
    pub meshes: Mut<'a, bevy::prelude::Assets<Mesh>>,
    pub materials: Mut<'a, bevy::prelude::Assets<StandardMaterial>>,
    pub server: Mut<'a, bevy::prelude::AssetServer>,
}

impl<'a> Assets<'a> {
    pub fn load<'p, T: Asset>(&mut self, file: impl Into<AssetPath<'p>>) -> AssetHandle<T> {
        AssetHandle(if let Some(inner) = self.0.as_mut() {
            Some(inner.server.load(file))
        } else {
            None
        })
    }

    pub fn mesh(&mut self, mesh: Mesh) -> AssetHandle<Mesh> {
        AssetHandle(if let Some(inner) = self.0.as_mut() {
            Some(inner.meshes.add(mesh))
        } else {
            None
        })
    }

    pub fn material(&mut self, material: StandardMaterial) -> AssetHandle<StandardMaterial> {
        AssetHandle(if let Some(inner) = self.0.as_mut() {
            Some(inner.materials.add(material))
        } else {
            None
        })
    }
}

pub struct AssetHandle<T: Asset>(pub(crate) Option<Handle<T>>);

impl<'a> FromEnv<SimRunnerEnv> for Assets<'a> {
    type Item<'e> = Assets<'e>;

    fn from_env(env: <SimRunnerEnv as Env>::Param<'_>) -> Self::Item<'_> {
        let unsafe_world_cell = env.app.world.as_unsafe_world_cell_readonly();
        let meshes = unsafe { unsafe_world_cell.get_resource_mut().unwrap() };
        let materials = unsafe { unsafe_world_cell.get_resource_mut().unwrap() };
        let server = unsafe { unsafe_world_cell.get_resource_mut().unwrap() };

        Assets(Some(AssetsInner {
            meshes,
            materials,
            server,
        }))
    }

    fn init(_: &mut SimRunnerEnv) {}
}
