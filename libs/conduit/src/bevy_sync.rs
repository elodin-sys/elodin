use std::{marker::PhantomData, ops::DerefMut};

use crate::{
    bevy::{
        AppExt, AssetAdapter, ComponentValueMap, ConduitSubscribePlugin, EntityMap, SimPeer,
        Subscriptions,
    },
    client::MsgPair,
    well_known::{
        self, BodyAxes, EntityMetadata, Glb, Material, Mesh as ConduitMesh, Panel, VectorArrow,
        WorldPos,
    },
    EntityId,
};
use bevy::prelude::*;
use bevy::{ecs::system::SystemId, utils::HashMap};
use big_space::GridCell;
use serde::de::DeserializeOwned;

pub struct SyncPlugin {
    pub plugin: ConduitSubscribePlugin,
    pub subscriptions: Subscriptions,
    pub enable_pbr: bool,
}

impl SyncPlugin {
    pub fn new(rx: flume::Receiver<MsgPair>) -> Self {
        Self {
            plugin: ConduitSubscribePlugin::new(rx),
            subscriptions: Subscriptions::default(),
            enable_pbr: true,
        }
    }
}

impl Plugin for SyncPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        let sync_mesh = app.world_mut().register_system(sync_mesh_to_bevy);
        let sync_material = app.world_mut().register_system(sync_material_to_bevy);
        let sync_glb = app.world_mut().register_system(sync_glb_to_bevy);
        app.add_plugins(self.plugin.clone())
            .insert_resource(SimPeer::default())
            .insert_resource(self.subscriptions.clone())
            .insert_resource(EntityMap::default())
            .add_conduit_component::<WorldPos>()
            .add_conduit_component::<well_known::Camera>()
            .add_conduit_asset::<VectorArrow>(Box::new(SyncPostcardAdapter::<VectorArrow>::new(
                None,
            )))
            .add_conduit_asset::<BodyAxes>(Box::new(SyncPostcardAdapter::<BodyAxes>::new(None)))
            .add_conduit_asset::<Panel>(Box::new(SyncPostcardAdapter::<Panel>::new(None)))
            .add_conduit_asset::<EntityMetadata>(Box::new(
                SyncPostcardAdapter::<EntityMetadata>::new(None),
            ))
            .add_conduit_asset::<ConduitMesh>(Box::new(SyncPostcardAdapter::<ConduitMesh>::new(
                self.enable_pbr.then_some(sync_mesh),
            )))
            .add_conduit_asset::<Material>(Box::new(SyncPostcardAdapter::<Material>::new(
                self.enable_pbr.then_some(sync_material),
            )))
            .add_conduit_asset::<Glb>(Box::new(SyncPostcardAdapter::<Glb>::new(
                self.enable_pbr.then_some(sync_glb),
            )));
    }
}

struct SyncPostcardAdapter<T: DeserializeOwned> {
    sync_system: Option<SystemId>,
    phantom_data: PhantomData<T>,
}

impl<T: DeserializeOwned> SyncPostcardAdapter<T> {
    fn new(sync_system: Option<SystemId>) -> Self {
        Self {
            sync_system,
            phantom_data: PhantomData,
        }
    }
}

impl<T: DeserializeOwned + Component + crate::Asset> AssetAdapter for SyncPostcardAdapter<T> {
    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        asset_index: u64,
        value: &[u8],
    ) {
        let Ok(comp) = postcard::from_bytes::<T>(value) else {
            warn!("failed to deserialize asset");
            return;
        };
        let mut e = if let Some(entity) = entity_map.0.get(&entity_id) {
            let Some(e) = commands.get_entity(*entity) else {
                return;
            };
            e
        } else {
            let e = commands.spawn((
                EntityMetadata {
                    name: format!("{:?}", entity_id),
                    color: Color::WHITE.into(),
                },
                entity_id,
                ComponentValueMap::default(),
                Transform::default(),
                GridCell::<i128>::default(),
            ));
            entity_map.0.insert(entity_id, e.id());
            e
        };
        e.insert(comp).insert(AssetHandle::<T> {
            id: asset_index,
            phantom_data: PhantomData,
        });
        if let Some(sync_system) = self.sync_system {
            commands.run_system(sync_system);
        }
    }
}

#[derive(bevy::prelude::Component, Debug, Clone)]
struct AssetHandle<T> {
    id: u64,
    phantom_data: PhantomData<T>,
}

impl<T> PartialEq for AssetHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for AssetHandle<T> {}

impl<T> core::hash::Hash for AssetHandle<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Default)]
struct SyncedMeshes(HashMap<AssetHandle<ConduitMesh>, Handle<Mesh>>);
#[derive(Default)]
struct SyncedMaterials(HashMap<AssetHandle<Material>, Handle<StandardMaterial>>);
#[derive(Default)]
struct SyncedGlbs(HashMap<AssetHandle<Glb>, Handle<Scene>>);
#[derive(Component)]
struct SyncedPbr;

fn sync_mesh_to_bevy(
    mesh: Query<(
        Entity,
        &ConduitMesh,
        &AssetHandle<ConduitMesh>,
        Option<&SyncedPbr>,
    )>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut commands: Commands,
    mut cache: Local<SyncedMeshes>,
) {
    for (entity, mesh, handle, synced_pbr) in mesh.iter() {
        let mut entity = commands.entity(entity);
        let mesh = if let Some(mesh) = cache.0.get(handle) {
            mesh.clone()
        } else {
            let mesh = Mesh::from(mesh.clone());
            let mesh = mesh_assets.add(mesh);
            cache.0.insert(handle.clone(), mesh.clone());
            mesh
        };
        if synced_pbr.is_some() {
            entity.insert(mesh);
        } else {
            entity.insert((
                PbrBundle {
                    mesh,
                    ..Default::default()
                },
                SyncedPbr,
            ));
        }
    }
}

fn sync_material_to_bevy(
    material: Query<(
        Entity,
        &Material,
        &AssetHandle<Material>,
        Option<&SyncedPbr>,
    )>,
    mut material_assets: ResMut<Assets<StandardMaterial>>,
    mut image_assets: ResMut<Assets<Image>>,
    mut commands: Commands,
    mut cache: Local<SyncedMaterials>,
) {
    for (entity, material, handle, synced_pbr) in material.iter() {
        let mut entity = commands.entity(entity);
        let material = if let Some(material) = cache.0.get(handle) {
            material.clone()
        } else {
            let material = material.clone().into_material(image_assets.deref_mut());
            let material = material_assets.add(material);
            cache.0.insert(handle.clone(), material.clone());
            material
        };
        if synced_pbr.is_some() {
            entity.insert(material);
        } else {
            entity.insert((
                PbrBundle {
                    material,
                    ..Default::default()
                },
                SyncedPbr,
            ));
        }
    }
}
fn sync_glb_to_bevy(
    mut commands: Commands,
    mut cache: Local<SyncedGlbs>,
    glb: Query<(Entity, &Glb, &AssetHandle<Glb>)>,
    assets: Res<AssetServer>,
) {
    for (entity, glb, handle) in glb.iter() {
        let Glb(u) = glb;
        let mut entity = commands.entity(entity);
        let scene = if let Some(glb) = cache.0.get(handle) {
            glb.clone()
        } else {
            let url = format!("{u}#Scene0");
            let scene = assets.load(&url);
            cache.0.insert(handle.clone(), scene.clone());
            scene
        };
        entity.insert((SceneBundle {
            scene,
            ..Default::default()
        },));
    }
}
