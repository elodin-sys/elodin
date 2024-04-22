use std::{marker::PhantomData, ops::DerefMut};

use crate::{
    bevy::{
        AppExt, AssetAdapter, ComponentValueMap, ConduitSubscribePlugin, EntityMap, SimPeer,
        Subscriptions,
    },
    client::MsgPair,
    well_known::{self, EntityMetadata, Gizmo, Panel, Pbr, TraceAnchor, WorldPos},
    EntityId,
};
use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use big_space::GridCell;
use bytes::Bytes;
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
        let sync_pbr = app.world.register_system(sync_pbr_to_bevy);
        app.add_plugins(self.plugin.clone())
            .insert_resource(SimPeer::default())
            .insert_resource(self.subscriptions.clone())
            .insert_resource(EntityMap::default())
            .add_conduit_component::<WorldPos>()
            .add_conduit_component::<well_known::Camera>()
            .add_conduit_asset::<Gizmo>(Box::new(SyncPostcardAdapter::<Gizmo>::new(None)))
            .add_conduit_asset::<Panel>(Box::new(SyncPostcardAdapter::<Panel>::new(None)))
            .add_conduit_component::<TraceAnchor>()
            .add_conduit_asset::<EntityMetadata>(Box::new(
                SyncPostcardAdapter::<EntityMetadata>::new(None),
            ))
            .add_conduit_asset::<Pbr>(Box::new(SyncPostcardAdapter::<Pbr>::new(
                self.enable_pbr.then_some(sync_pbr),
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

impl<T: DeserializeOwned + Component> AssetAdapter for SyncPostcardAdapter<T> {
    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: Bytes,
    ) {
        let Ok(comp) = postcard::from_bytes::<T>(&value) else {
            warn!("failed to deserialize material");
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
        e.insert(comp);
        if let Some(sync_system) = self.sync_system {
            commands.run_system(sync_system);
        }
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
