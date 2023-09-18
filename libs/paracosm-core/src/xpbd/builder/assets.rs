use bevy::{asset::Asset, prelude::*};
use bevy_ecs::world::Mut;

pub struct Assets<'a>(pub(crate) Option<AssetsInner<'a>>);

pub(crate) struct AssetsInner<'a> {
    pub(crate) meshes: Mut<'a, bevy::prelude::Assets<Mesh>>,
    pub(crate) materials: Mut<'a, bevy::prelude::Assets<StandardMaterial>>,
}

impl<'a> Assets<'a> {
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
