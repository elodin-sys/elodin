use crate::{
    types::{SyncMaterial, SyncMeshData},
    ReflectSerde, SimState,
};
use bevy::prelude::*;
use elo_conduit::{
    bevy::{AppExt, ConduitSubscribePlugin, EntityMap, Msg, Subscriptions},
    Component, EntityId,
};

use crate::WorldPos;

pub struct SyncPlugin {
    pub plugin: ConduitSubscribePlugin,
    pub subscriptions: Subscriptions,
}

impl SyncPlugin {
    pub fn new(rx: flume::Receiver<Msg<'static>>) -> Self {
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
            .add_conduit_component::<SimState>()
            .add_conduit_component::<WorldPos>()
            .add_conduit_component_with_insert_fn::<SyncMaterial>(Box::new(
                move |commands, entity_map, entity_id, value| {
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

                    if let Some(c) = SyncMaterial::from_component_value(value) {
                        e.insert(c);
                    }
                    commands.run_system(sync_pbr);
                },
            ))
            .add_conduit_component_with_insert_fn::<SyncMeshData>(Box::new(
                move |commands, entity_map, entity_id, value| {
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

                    if let Some(c) = SyncMeshData::from_component_value(value) {
                        e.insert(c);
                    }
                    commands.run_system(sync_pbr);
                },
            ))
            .add_conduit_resource::<SimState>();
    }
}

pub struct SendPlbPlugin;

impl Plugin for SendPlbPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, sync_pbr);
    }
}

pub fn setup_entity_map(
    query: Query<(&EntityId, Entity), Changed<EntityId>>,
    mut map: ResMut<EntityMap>,
) {
    *map = EntityMap(query.iter().map(|(id, e)| (*id, e)).collect());
}

pub fn sync_pbr(
    mesh_query: Query<(Entity, &Handle<Mesh>), Changed<Handle<Mesh>>>,
    mat_query: Query<(Entity, &Handle<StandardMaterial>), Changed<Handle<Mesh>>>,
    mesh_assets: ResMut<Assets<Mesh>>,
    mat_assets: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    for (entity, mesh) in mesh_query.iter() {
        //let material = ReflectSerde(material_assets.get(material).unwrap().clone());
        let mesh = SyncMeshData::from(mesh_assets.get(mesh).unwrap().clone());
        commands.entity(entity).insert(mesh);
    }

    for (entity, mat) in mat_query.iter() {
        let mat = ReflectSerde(mat_assets.get(mat).unwrap().clone());
        commands.entity(entity).insert(SyncMaterial(mat));
    }
}

fn sync_pbr_to_bevy(
    query: Query<(Entity, &SyncMaterial, &SyncMeshData)>,
    mut mat_assets: ResMut<Assets<StandardMaterial>>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut commands: Commands,
) {
    for (entity, material, mesh) in query.iter() {
        let material = mat_assets.add(material.0 .0.clone());
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
