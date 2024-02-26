use std::ops::DerefMut;

use crate::{
    bevy::{AppExt, AssetAdapter, ConduitSubscribePlugin, EntityMap, SimPeer, Subscriptions},
    client::MsgPair,
    well_known::{Pbr, TraceAnchor, WorldPos},
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
            .insert_resource(SimPeer::default())
            .insert_resource(self.subscriptions.clone())
            .insert_resource(EntityMap::default())
            .add_conduit_component::<WorldPos>()
            .add_conduit_component::<TraceAnchor>()
            .add_conduit_asset::<Pbr>(Pbr::ASSET_ID, Box::new(SyncPbrAdapter { sync_pbr }));
    }
}

struct SyncPbrAdapter {
    sync_pbr: SystemId,
}

impl AssetAdapter for SyncPbrAdapter {
    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: Bytes,
    ) {
        let Ok(mat) = postcard::from_bytes::<Pbr>(&value) else {
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

fn sync_pbr_to_bevy(
    query: Query<(Entity, &Pbr)>,
    mut mat_assets: ResMut<Assets<StandardMaterial>>,
    mut image_assets: ResMut<Assets<Image>>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut commands: Commands,
    assets: Res<AssetServer>,
) {
    for (entity, pbr) in query.iter() {
        let mut entity = commands.entity(entity);
        match pbr {
            Pbr::Url(u) => {
                let scene = assets.load(format!("{u}#Scene0"));
                entity.insert(SceneBundle {
                    scene,
                    ..Default::default()
                });
            }
            Pbr::Bundle { mesh, material } => {
                let material = material.clone().into_material(image_assets.deref_mut());
                let material = mat_assets.add(material);
                let asset = Mesh::from(mesh.clone());
                let mesh = mesh_assets.add(asset);
                entity.insert((PbrBundle {
                    mesh,
                    material,
                    ..Default::default()
                },));
            }
        }
    }
}
