use std::ops::DerefMut;

use crate::{
    bevy::{AppExt, AssetAdapter, ConduitSubscribePlugin, EntityMap, Subscriptions},
    client::MsgPair,
    well_known::{Material as SyncMaterial, Mesh as SyncMesh, TraceAnchor, WorldPos},
    EntityId,
};
use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use bytes::Bytes;

pub struct SyncPlugin {
    pub plugin: ConduitSubscribePlugin,
    pub subscriptions: Subscriptions,
}

impl SyncPlugin {
    pub fn new(rx: flume::Receiver<MsgPair>) -> Self {
        Self {
            plugin: ConduitSubscribePlugin::new(rx),
            subscriptions: Subscriptions::default(),
        }
    }
}

impl Plugin for SyncPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        let sync_pbr = app.world.register_system(sync_pbr_to_bevy);
        app.add_plugins(self.plugin.clone())
            .insert_resource(self.subscriptions.clone())
            .insert_resource(EntityMap::default())
            .add_conduit_component::<WorldPos>()
            .add_conduit_component::<TraceAnchor>()
            .add_conduit_asset::<SyncMaterial>(
                SyncMaterial::ASSET_ID,
                Box::new(SyncMaterialAdaptor { sync_pbr }),
            )
            .add_conduit_asset::<SyncMesh>(
                SyncMesh::ASSET_ID,
                Box::new(SyncMeshAdapter { sync_pbr }),
            );
    }
}

struct SyncMaterialAdaptor {
    sync_pbr: SystemId,
}

impl AssetAdapter for SyncMaterialAdaptor {
    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: Bytes,
    ) {
        let Ok(mat) = postcard::from_bytes::<SyncMaterial>(&value) else {
            warn!("failed to deserialize material");
            return;
        };
        let mut e = if let Some(entity) = entity_map.0.get(&entity_id) {
            let Some(e) = commands.get_entity(*entity) else {
                return;
            };
            e
        } else {
            let e = commands.spawn_empty();
            entity_map.0.insert(entity_id, e.id());
            e
        };
        e.insert(mat);
        commands.run_system(self.sync_pbr);
    }
}

struct SyncMeshAdapter {
    sync_pbr: SystemId,
}

impl AssetAdapter for SyncMeshAdapter {
    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: Bytes,
    ) {
        let Ok(mat) = postcard::from_bytes::<SyncMesh>(&value) else {
            warn!("failed to deserialize mesh");
            return;
        };
        let mut e = if let Some(entity) = entity_map.0.get(&entity_id) {
            let Some(e) = commands.get_entity(*entity) else {
                return;
            };
            e
        } else {
            let e = commands.spawn_empty();
            entity_map.0.insert(entity_id, e.id());
            e
        };
        e.insert(mat);
        commands.run_system(self.sync_pbr);
    }
}

pub struct SendPlbPlugin;

impl Plugin for SendPlbPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, sync_pbr);
    }
}

pub fn sync_pbr(
    mesh_query: Query<(Entity, &Handle<Mesh>), Changed<Handle<Mesh>>>,
    mat_query: Query<(Entity, &Handle<StandardMaterial>), Changed<Handle<Mesh>>>,
    mesh_assets: ResMut<Assets<Mesh>>,
    mat_assets: ResMut<Assets<StandardMaterial>>,
    image_assets: Res<Assets<Image>>,
    mut commands: Commands,
) {
    for (entity, mesh) in mesh_query.iter() {
        let mesh = SyncMesh::from(mesh_assets.get(mesh).unwrap().clone());
        commands.entity(entity).insert(mesh);
    }

    for (entity, mat) in mat_query.iter() {
        let mat = SyncMaterial::from_bevy(mat_assets.get(mat).unwrap().clone(), &image_assets);
        commands.entity(entity).insert(mat);
    }
}

fn sync_pbr_to_bevy(
    query: Query<(Entity, &SyncMaterial, &SyncMesh)>,
    mut mat_assets: ResMut<Assets<StandardMaterial>>,
    mut image_assets: ResMut<Assets<Image>>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut commands: Commands,
) {
    for (entity, material, mesh) in query.iter() {
        let material = material.clone().into_material(image_assets.deref_mut());
        let material = mat_assets.add(material);
        let asset = Mesh::from(mesh.clone());
        let mesh = mesh_assets.add(asset);
        let mut entity = commands.entity(entity);
        entity.insert((PbrBundle {
            mesh,
            material,
            ..Default::default()
        },));
    }
}
