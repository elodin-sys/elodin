use crate::{
    types::{SyncMaterial, SyncMeshData},
    ReflectSerde, SimState, TraceAnchor,
};
use bevy::prelude::*;
use bevy_ecs::system::SystemId;
use elodin_conduit::{
    bevy::{AppExt, ComponentAdapter, ConduitSubscribePlugin, EntityMap, Msg, Subscriptions},
    Component, ComponentFilter, EntityId,
};

use crate::WorldPos;

pub const DEFAULT_SUB_FILTERS: &[ComponentFilter] = &[
    ComponentFilter::from_str("world_pos"),
    ComponentFilter::from_str("sync_mesh_data"),
    ComponentFilter::from_str("sync_material"),
    ComponentFilter::from_str("trace_anchor"),
];

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
            .add_conduit_component::<TraceAnchor>()
            .add_conduit_component_with_adapter::<SyncMaterial>(Box::new(SyncMaterialAdaptor {
                sync_pbr,
            }))
            .add_conduit_component_with_adapter::<SyncMeshData>(Box::new(SyncMeshAdapter {
                sync_pbr,
            }))
            .add_conduit_resource::<SimState>();
    }
}

struct SyncMaterialAdaptor {
    sync_pbr: SystemId,
}

impl ComponentAdapter for SyncMaterialAdaptor {
    fn get(&self, _world: &World, _entity: Entity) -> Option<elodin_conduit::ComponentValue> {
        None
    }

    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: elodin_conduit::ComponentValue,
    ) {
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
        commands.run_system(self.sync_pbr);
    }
}

struct SyncMeshAdapter {
    sync_pbr: SystemId,
}

impl ComponentAdapter for SyncMeshAdapter {
    fn get(&self, _world: &World, _entity: Entity) -> Option<elodin_conduit::ComponentValue> {
        None
    }

    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: elodin_conduit::ComponentValue,
    ) {
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
        commands.run_system(self.sync_pbr);
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
