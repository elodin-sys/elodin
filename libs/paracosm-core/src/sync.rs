use std::collections::HashMap;

use bevy::{
    app::AppExit,
    prelude::{Assets, Deref, DerefMut, Handle, Mesh, PbrBundle, StandardMaterial},
};
use bevy_ecs::{
    entity::Entity,
    event::{EventReader, EventWriter},
    system::{Commands, NonSend, Query, Res, ResMut, Resource},
};

use crate::{
    history::{HistoryStore, RollbackEvent},
    ClientMsg, MeshData, ModelData, Paused, ReflectSerde, ServerMsg, SimState, SyncModels, Synced,
    Uuid, WorldPos,
};
use flume::{Receiver, Sender};

#[derive(Resource, DerefMut, Deref, Default)]
pub struct EntityMap(HashMap<Uuid, Entity>);

pub fn recv_data<S: ClientTransport>(
    mut commands: Commands,
    mut world_pos: Query<&mut WorldPos>,
    mut entity_map: ResMut<EntityMap>,
    client: NonSend<S>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut material_assets: ResMut<Assets<StandardMaterial>>,
    mut sim_state: ResMut<SimState>,
) {
    while let Some(msg) = client.try_recv_msg() {
        match msg {
            ClientMsg::Clear => todo!(),
            ClientMsg::SimSate(state) => {
                *sim_state = state;
            }
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

pub trait ClientTransport: Clone + 'static {
    fn try_recv_msg(&self) -> Option<ClientMsg>;
    fn send_msg(&self, msg: ServerMsg);
}

pub trait ServerTransport: Send + Sync + Resource + Clone {
    fn send_msg(&self, msg: ClientMsg);
    fn try_recv_msg(&self) -> Option<ServerMsg>;
}

pub fn send_pos<Tx: ServerTransport>(
    query: Query<(&WorldPos, &Uuid)>,
    tx: ResMut<Tx>,
    history: Res<HistoryStore>,
    paused: Res<Paused>,
) {
    for (pos, body_id) in query.iter() {
        tx.send_msg(ClientMsg::SyncWorldPos(crate::SyncWorldPos {
            body_id: *body_id,
            pos: pos.0,
        }))
    }
    tx.send_msg(ClientMsg::SimSate(SimState {
        paused: **paused,
        history_count: history.count(),
        history_index: history.current_index(),
    }))
}

pub fn startup_sync_model(mut event_writer: EventWriter<SyncModels>) {
    event_writer.send(SyncModels);
}

pub fn send_model<Tx: ServerTransport>(
    mut pbr_query: Query<(&Handle<Mesh>, &Handle<StandardMaterial>, &Uuid)>,
    mesh_assets: ResMut<Assets<Mesh>>,
    material_assets: ResMut<Assets<StandardMaterial>>,
    tx: Res<Tx>,
    mut event_reader: EventReader<SyncModels>,
) {
    for _ in &mut event_reader.read() {
        for (mesh, material, body_id) in pbr_query.iter_mut() {
            let material = ReflectSerde(material_assets.get(material).unwrap().clone());
            let mesh = MeshData::from(mesh_assets.get(mesh).unwrap().clone());
            tx.send_msg(ClientMsg::ModelDataResp(ModelData::Pbr {
                body_id: *body_id,
                material,
                mesh,
            }));
        }
    }
}

pub fn recv_server<T: ServerTransport>(
    transport: Res<T>,
    mut event: EventWriter<AppExit>,
    mut paused: ResMut<Paused>,
    mut rollback: EventWriter<RollbackEvent>,
    mut sync_models: EventWriter<SyncModels>,
) {
    while let Some(msg) = transport.try_recv_msg() {
        match msg {
            ServerMsg::Exit => event.send(AppExit),
            ServerMsg::RequestModels => {
                sync_models.send(SyncModels);
            }
            ServerMsg::Pause(pause) => paused.0 = pause,
            ServerMsg::Rollback(time) => rollback.send(RollbackEvent(time)),
        }
    }
}

#[derive(Debug, Clone, Resource)]
pub struct ServerChannel {
    pub tx: Sender<ClientMsg>,
    pub rx: Receiver<ServerMsg>,
}

impl ServerTransport for ServerChannel {
    fn send_msg(&self, msg: ClientMsg) {
        self.tx.send(msg).unwrap();
    }

    fn try_recv_msg(&self) -> Option<ServerMsg> {
        self.rx.try_recv().ok()
    }
}

#[derive(Debug, Clone, Resource)]
pub struct ClientChannel {
    pub rx: Receiver<ClientMsg>,
    pub tx: Sender<ServerMsg>,
}

pub fn channel_pair() -> (ServerChannel, ClientChannel) {
    let (client_tx, client_rx) = flume::unbounded();
    let (server_tx, server_rx) = flume::unbounded();
    (
        ServerChannel {
            tx: client_tx,
            rx: server_rx,
        },
        ClientChannel {
            rx: client_rx,
            tx: server_tx,
        },
    )
}

impl ClientTransport for ClientChannel {
    fn try_recv_msg(&self) -> Option<ClientMsg> {
        self.rx.try_recv().ok()
    }

    fn send_msg(&self, msg: ServerMsg) {
        self.tx.send(msg).unwrap();
    }
}
