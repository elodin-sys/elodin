use std::collections::HashMap;

use bevy::{
    ecs::{
        entity::Entity,
        system::{Commands, EntityCommands, Resource},
    },
    prelude::{Deref, DerefMut, Plugin, Res, ResMut, Update},
};

use crate::{Component, ComponentId, ComponentValue, DataMsg, EntityId};

impl<'a> DataMsg<'a> {
    pub fn load_into_bevy(
        self,
        entity_map: &mut EntityMap,
        component_map: &ComponentMap,
        commands: &mut Commands,
    ) {
        for component in self.components.into_iter() {
            'value: for (entity_id, value) in component.storage.into_iter() {
                let Some(insert_fn) = component_map.0.get(&component.component_id) else {
                    continue 'value;
                };
                let mut e = if let Some(entity) = entity_map.0.get(&entity_id) {
                    let Some(e) = commands.get_entity(*entity) else {
                        continue 'value;
                    };
                    e
                } else {
                    let e = commands.spawn_empty();
                    entity_map.0.insert(entity_id, e.id());
                    e
                };
                (insert_fn)(&mut e, value);
            }
        }
    }
}

#[derive(Resource, Default)]
pub struct EntityMap(HashMap<EntityId, Entity>);

#[derive(Resource, Default)]
pub struct ComponentMap(HashMap<ComponentId, InsertConduitComponentFn>);

type InsertConduitComponentFn = Box<
    dyn for<'b, 'w, 's, 'a> Fn(&'b mut EntityCommands<'w, 's, 'a>, ComponentValue) + Sync + Send,
>;

pub trait AppExt {
    fn add_conduit_component<C: Component + bevy::prelude::Component>(&mut self) -> &mut Self;
}

impl AppExt for bevy::app::App {
    fn add_conduit_component<C: Component + bevy::prelude::Component>(&mut self) -> &mut Self {
        let mut map = self
            .world
            .get_resource_or_insert_with(|| ComponentMap(HashMap::default()));
        map.0.insert(
            C::component_id(),
            Box::new(|commands, value| {
                if let Some(c) = C::from_component_value(value) {
                    commands.insert(c);
                }
            }),
        );
        self
    }
}

#[derive(Debug, Resource, Deref, DerefMut)]
struct ConduitRx(flume::Receiver<DataMsg<'static>>);

fn recv_system(
    rx: Res<ConduitRx>,
    mut entity_map: ResMut<EntityMap>,
    component_map: Res<ComponentMap>,
    mut commands: Commands,
) {
    while let Ok(msg) = rx.try_recv() {
        msg.load_into_bevy(entity_map.as_mut(), component_map.as_ref(), &mut commands);
    }
}

pub struct ConduitSubscribePlugin;

impl ConduitSubscribePlugin {
    #[cfg(feature = "tokio")]
    pub fn tcp(addr: std::net::SocketAddr) -> ConduitTcpSubscriber {
        ConduitTcpSubscriber { addr }
    }
}

#[cfg(feature = "tokio")]
pub struct ConduitTcpSubscriber {
    addr: std::net::SocketAddr,
}

#[cfg(feature = "tokio")]
impl Plugin for ConduitTcpSubscriber {
    fn build(&self, app: &mut bevy::prelude::App) {
        use std::thread;
        #[derive(Resource)]
        struct ConduitTokioHandle(thread::JoinHandle<()>);
        let (tx, rx) = flume::unbounded();
        let addr = self.addr;
        let thread = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("tokio runtime failed to start");
            rt.block_on(async move {
                let mut socket = crate::tokio::TcpClient::connect(addr, 0)
                    .await
                    .expect("tcp client failed");
                while let Some(msg) = socket.recv().await.expect("recv failed") {
                    tx.send_async(msg).await.unwrap();
                }
            });
        });
        app.insert_resource(ConduitRx(rx))
            .insert_resource(ConduitTokioHandle(thread));
    }
}

impl Plugin for ConduitSubscribePlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.insert_resource(EntityMap::default());
        app.insert_resource(ComponentMap::default());
        app.add_systems(Update, recv_system);
    }
}
