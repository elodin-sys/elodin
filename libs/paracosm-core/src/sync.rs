use std::collections::HashMap;

use bevy::prelude::{
    shape, Assets, Color, Deref, DerefMut, Handle, Mesh, PbrBundle, StandardMaterial,
};
use bevy_ecs::{
    entity::Entity,
    system::{Commands, Query, Res, ResMut, Resource},
};

use crate::{ClientMsg, MeshData, ModelData, ReflectSerde, Synced, Uuid, WorldPos};
use flume::{Receiver, Sender};

#[derive(Resource, DerefMut, Deref, Default)]
pub struct EntityMap(HashMap<Uuid, Entity>);

pub fn recv_data<S: ClientRx>(
    mut commands: Commands,
    mut world_pos: Query<&mut WorldPos>,
    mut entity_map: ResMut<EntityMap>,
    client: Res<S>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut material_assets: ResMut<Assets<StandardMaterial>>,
) {
    while let Some(msg) = client.try_recv_msg() {
        match msg {
            ClientMsg::Clear => todo!(),
            ClientMsg::SyncWorldPos(msg) => {
                if let Some(e) = entity_map.get(&msg.body_id) {
                    if let Ok(mut world_pos) = world_pos.get_mut(*e) {
                        **world_pos = msg.pos;
                    }
                } else {
                    let entity = commands.spawn(WorldPos(msg.pos)).id();
                    entity_map.insert(msg.body_id, entity);
                }
            }
            ClientMsg::ModelDataResp(ModelData::Glb { .. }) => {
                todo!()
            }
            ClientMsg::ModelDataResp(ModelData::Pbr {
                body_id,
                material,
                mesh,
            }) => {
                let material = material_assets.add(material.0);
                let mesh = mesh_assets.add(Mesh::from(mesh));
                // let material = material_assets.add(Color::WHITE.into());
                // let mesh = mesh_assets.add(Mesh::from(shape::Box::new(1.0, 1.0, 1.0)));

                let entity = entity_map
                    .entry(body_id)
                    .or_insert_with(|| commands.spawn_empty().id());
                let mut entity = commands.entity(*entity);
                entity.insert((
                    PbrBundle {
                        mesh,
                        material,
                        ..Default::default()
                    },
                    WorldPos(Default::default()),
                ));
            }
        }
    }
}

pub trait ClientRx: Clone + Send + Sync + Resource {
    fn try_recv_msg(&self) -> Option<ClientMsg>;
}

pub trait ServerTx: Send + Sync + Resource + Clone {
    fn send_msg(&self, msg: ClientMsg);
}

pub fn send_pos<Tx: ServerTx>(query: Query<(&WorldPos, &Uuid)>, mut tx: ResMut<Tx>) {
    for (pos, body_id) in query.iter() {
        tx.send_msg(ClientMsg::SyncWorldPos(crate::SyncWorldPos {
            body_id: *body_id,
            pos: pos.0,
        }))
    }
}

pub fn send_model<Tx: ServerTx>(
    mut pbr_query: Query<(&Handle<Mesh>, &Handle<StandardMaterial>, &Uuid, &mut Synced)>,
    mesh_assets: ResMut<Assets<Mesh>>,
    material_assets: ResMut<Assets<StandardMaterial>>,
    tx: Res<Tx>,
) {
    for (mesh, material, body_id, mut synced) in pbr_query.iter_mut() {
        if !**synced {
            let material = ReflectSerde(material_assets.get(material).unwrap().clone());
            let mesh = MeshData::from(mesh_assets.get(mesh).unwrap().clone());
            tx.send_msg(ClientMsg::ModelDataResp(ModelData::Pbr {
                body_id: *body_id,
                material,
                mesh,
            }));
            **synced = true;
        }
    }
}

#[derive(Debug, Clone, Resource, Deref, DerefMut)]
pub struct ChannelSender(Sender<ClientMsg>);

impl ServerTx for ChannelSender {
    fn send_msg(&self, msg: ClientMsg) {
        let _ = self.send(msg);
    }
}

#[derive(Debug, Clone, Resource, Deref, DerefMut)]
pub struct ChannelReceiver(Receiver<ClientMsg>);

pub fn channel_pair() -> (ChannelSender, ChannelReceiver) {
    let (tx, rx) = flume::unbounded();
    (ChannelSender(tx), ChannelReceiver(rx))
}

impl ClientRx for ChannelReceiver {
    fn try_recv_msg(&self) -> Option<ClientMsg> {
        self.try_recv().ok()
    }
}
