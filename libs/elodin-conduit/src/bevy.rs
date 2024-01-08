use crate::{
    eid, Component, ComponentData, ComponentFilter, ComponentId, ComponentValue, EntityId,
};
use bevy::ecs::system::EntityCommand;
use bevy::prelude::{DetectChanges, Event, EventReader, Without, World};
use bevy::ptr::OwningPtr;
use ip_network_table_deps_treebitmap::address::Address;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tracing::info;

use bevy::{
    app::AppExit,
    ecs::query::Changed,
    prelude::{
        Commands, Deref, DerefMut, Entity, EventWriter, Plugin, Query, Res, ResMut, Resource,
        Update,
    },
};

#[derive(bevy::prelude::Component)]
pub struct Received;

#[derive(Debug)]
pub enum Msg<'a> {
    Data(ComponentData<'a>),
    Subscribe(SubscribeEvent<'a>),
    Exit,
}

impl<'a> ComponentData<'a> {
    pub fn load_into_bevy(
        self,
        entity_map: &mut EntityMap,
        component_map: &ComponentMap,
        commands: &mut Commands,
    ) {
        let Some(adapter) = component_map.0.get(&self.component_id) else {
            tracing::warn!(?self.component_id, "unknown insert fn");
            return;
        };

        for (entity_id, value) in self.storage.into_iter() {
            adapter.insert(commands, entity_map, entity_id, value);
        }
    }
}

#[derive(Event, Debug, Clone)]
pub struct SubscribeEvent<'a> {
    pub filters: Vec<ComponentFilter>,
    pub tx: flume::Sender<Msg<'a>>,
}

#[derive(Resource, Default, Deref)]
pub struct EntityMap(pub HashMap<EntityId, Entity>);

#[derive(Resource, Default)]
pub struct ComponentMap(pub HashMap<ComponentId, Box<dyn ComponentAdapter + Send + Sync>>);

pub trait ComponentAdapter {
    fn get(&self, world: &World, entity: Entity) -> Option<ComponentValue>;
    fn insert(
        &self,
        commands: &mut Commands,
        map: &mut EntityMap,
        entity_id: EntityId,
        value: ComponentValue,
    );
}

struct StaticComponentAdapter<C> {
    _phantom: PhantomData<C>,
}

impl<C> Default for StaticComponentAdapter<C> {
    fn default() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }
}

impl<C> ComponentAdapter for StaticComponentAdapter<C>
where
    C: Component + bevy::prelude::Component + std::fmt::Debug,
{
    fn get(&self, world: &World, entity: Entity) -> Option<ComponentValue> {
        Some(world.get_entity(entity)?.get::<C>()?.component_value())
    }

    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: ComponentValue,
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

        if let Some(c) = C::from_component_value(value) {
            e.insert((c, Received));
        }
    }
}

struct StaticResourceAdapter<C> {
    _phantom: PhantomData<C>,
}

impl<C> Default for StaticResourceAdapter<C> {
    fn default() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }
}

impl<C: Resource + Component + std::fmt::Debug> ComponentAdapter for StaticResourceAdapter<C> {
    fn get(&self, world: &World, _entity_id: Entity) -> Option<ComponentValue> {
        Some(world.get_resource::<C>()?.component_value())
    }

    fn insert(
        &self,
        commands: &mut Commands,
        _map: &mut EntityMap,
        _entity_id: EntityId,
        value: ComponentValue,
    ) {
        if let Some(r) = C::from_component_value(value) {
            commands.insert_resource(r)
        }
    }
}

struct DynamicComponentAdapter {
    component_id: bevy::ecs::component::ComponentId,
}

impl ComponentAdapter for DynamicComponentAdapter {
    fn get(&self, world: &World, entity: Entity) -> Option<ComponentValue> {
        unsafe {
            Some(
                world
                    .get_by_id(entity, self.component_id)?
                    .deref::<ComponentValue>()
                    .to_owned(),
            )
        }
    }

    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: ComponentValue,
    ) {
        let e = if let Some(entity) = entity_map.0.get(&entity_id) {
            let Some(e) = commands.get_entity(*entity) else {
                return;
            };
            e
        } else {
            let e = commands.spawn_empty();
            entity_map.0.insert(entity_id, e.id());
            e
        };
        let id = e.id();
        commands.add(
            InsertValueById {
                component_id: self.component_id,
                value: value.into_owned(),
            }
            .with_entity(id),
        )
    }
}

struct InsertValueById {
    component_id: bevy::ecs::component::ComponentId,
    value: ComponentValue<'static>,
}

impl EntityCommand for InsertValueById {
    fn apply(self, id: Entity, world: &mut World) {
        if let Some(mut e) = world.get_or_spawn(id) {
            OwningPtr::make(self.value, |ptr| unsafe {
                e.insert_by_id(self.component_id, ptr);
            })
        }
    }
}

pub trait AppExt {
    fn add_conduit_component<C>(&mut self) -> &mut Self
    where
        C: Component + bevy::prelude::Component + std::fmt::Debug;

    fn add_conduit_component_with_adapter<C>(
        &mut self,
        adapter: Box<dyn ComponentAdapter + Send + Sync>,
    ) -> &mut Self
    where
        C: Component + bevy::prelude::Component + std::fmt::Debug;

    fn add_conduit_resource<C>(&mut self) -> &mut Self
    where
        C: Component + bevy::prelude::Resource + std::fmt::Debug;
}

impl AppExt for bevy::app::App {
    fn add_conduit_component<C>(&mut self) -> &mut Self
    where
        C: Component + bevy::prelude::Component + std::fmt::Debug,
    {
        self.add_conduit_component_with_adapter::<C>(Box::<StaticComponentAdapter<C>>::default())
    }

    fn add_conduit_component_with_adapter<C>(
        &mut self,
        adapter: Box<dyn ComponentAdapter + Send + Sync>,
    ) -> &mut Self
    where
        C: Component + bevy::prelude::Component + std::fmt::Debug,
    {
        let mut map = self
            .world
            .get_resource_or_insert_with(|| ComponentMap(HashMap::default()));
        map.0.insert(C::component_id(), adapter);
        self.add_systems(Update, sync_component::<C>);
        self.add_systems(Update, query_component::<C>);
        self
    }

    fn add_conduit_resource<R>(&mut self) -> &mut Self
    where
        R: Component + bevy::prelude::Resource + std::fmt::Debug,
    {
        let mut map = self
            .world
            .get_resource_or_insert_with(|| ComponentMap(HashMap::default()));
        map.0.insert(
            R::component_id(),
            Box::<StaticResourceAdapter<R>>::default(),
        );

        self.add_systems(Update, sync_resource::<R>);
        self
    }
}

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct ConduitRx(pub flume::Receiver<Msg<'static>>);

fn recv_system(
    rx: Res<ConduitRx>,
    mut entity_map: ResMut<EntityMap>,
    component_map: Res<ComponentMap>,
    mut commands: Commands,
    mut subscriptions: ResMut<Subscriptions>,
    mut exit: EventWriter<AppExit>,
    mut subscribe_event: EventWriter<SubscribeEvent<'static>>,
) {
    while let Ok(msg) = rx.try_recv() {
        match msg {
            Msg::Data(data) => {
                data.load_into_bevy(entity_map.as_mut(), component_map.as_ref(), &mut commands);
            }
            Msg::Subscribe(event) => {
                subscribe_event.send(event.clone());
                for filter in event.filters.into_iter() {
                    subscriptions.subscribe(filter, event.tx.clone());
                }
            }
            Msg::Exit => {
                exit.send(AppExit);
            }
        }
    }
}

#[derive(Clone)]
pub struct ConduitSubscribePlugin {
    rx: flume::Receiver<Msg<'static>>,
}

impl ConduitSubscribePlugin {
    pub fn new(rx: flume::Receiver<Msg<'static>>) -> Self {
        Self { rx }
    }

    pub fn pair() -> (ConduitSubscribePlugin, flume::Sender<Msg<'static>>) {
        let (tx, rx) = flume::unbounded();
        (ConduitSubscribePlugin { rx }, tx)
    }

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
        use tokio::net::TcpStream;
        #[derive(Resource)]
        struct ConduitTokioHandle(thread::JoinHandle<()>);
        let (sub_plugin, bevy_tx) = ConduitSubscribePlugin::pair();
        let addr = self.addr;
        let thread = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("tokio runtime failed to start");
            rt.block_on(async move {
                let (rx_socket, tx_socket) = TcpStream::connect(addr)
                    .await
                    .expect("tcp client failed")
                    .into_split();
                handle_socket(bevy_tx, tx_socket, rx_socket, &[]).await;
            });
        });
        app.add_plugins(sub_plugin)
            .insert_resource(ConduitTokioHandle(thread));
    }
}

#[cfg(feature = "tokio")]
pub async fn handle_socket(
    server_tx: flume::Sender<Msg<'static>>,
    tx_socket: impl tokio::io::AsyncWrite + Unpin,
    rx_socket: impl tokio::io::AsyncRead + Unpin,
    default_filters: &[ComponentFilter],
) {
    use crate::{cid_mask, Error};
    use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};
    use tracing::error;
    let mut tx_socket =
        crate::tokio::Client::new(0, FramedWrite::new(tx_socket, LengthDelimitedCodec::new()));
    let mut rx_socket =
        crate::tokio::Client::new(0, FramedRead::new(rx_socket, LengthDelimitedCodec::new()));

    let (out_tx, out_rx) = flume::unbounded();
    if let Err(err) = server_tx
        .send_async(Msg::Subscribe(SubscribeEvent {
            tx: out_tx.clone(),
            filters: default_filters.to_vec(),
        }))
        .await
    {
        error!(?err, "initial subscribe error");
    }

    if let Err(err) = out_tx
        .send_async(Msg::Data(ComponentData::subscribe(cid_mask!(32;sim_state))))
        .await
    {
        error!(?err, "initial subscribe error");
    }
    let tx = async move {
        while let Ok(Msg::Data(msg)) = out_rx.recv_async().await {
            tx_socket.send_data(0, msg).await?;
        }
        Ok::<(), Error>(())
    };
    let rx = async move {
        while let Some(batch) = rx_socket.recv().await.expect("recv failed") {
            let filters: Vec<_> = batch
                .components
                .iter()
                .filter(|c| c.component_id == crate::SUB_COMPONENT_ID)
                .flat_map(|c| c.storage.iter())
                .filter_map(|(_, c)| match c {
                    ComponentValue::Filter(filter) => Some(filter),
                    _ => None,
                })
                .cloned()
                .collect();
            if filters.is_empty() {
                for data in batch.components.into_iter() {
                    server_tx
                        .send_async(Msg::Data(data))
                        .await
                        .map_err(|_| Error::SendError)?;
                }
            } else {
                server_tx
                    .send_async(Msg::Subscribe(SubscribeEvent {
                        tx: out_tx.clone(),
                        filters,
                    }))
                    .await
                    .map_err(|_| Error::SendError)?;
            }
        }
        Ok::<(), Error>(())
    };
    tokio::select! {
        res = tx => res.unwrap(),
        res = rx => res.unwrap()
    }
}

impl Plugin for ConduitSubscribePlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_event::<SubscribeEvent>();
        app.insert_resource(EntityMap::default());
        app.insert_resource(ComponentMap::default());
        app.insert_resource(ConduitRx(self.rx.clone()));
        app.add_systems(Update, recv_system);
    }
}

pub fn sync_resource<T: bevy::prelude::Resource + Component + Debug>(
    res: Res<T>,
    mut subs: ResMut<Subscriptions>,
) {
    if !res.is_changed() {
        return;
    }
    let component_id = T::component_id();
    let storage = vec![(eid!(31;resource), res.component_value())];
    let data = ComponentData {
        component_id,
        storage,
    };
    let _ = subs.send(data);
}

#[allow(clippy::type_complexity)]
pub fn sync_component<T>(
    query: Query<(&EntityId, &T), (Changed<T>, Without<Received>)>,
    mut subs: ResMut<Subscriptions>,
) where
    T: bevy::prelude::Component + Component + Debug,
{
    let component_id = T::component_id();
    if query.is_empty() {
        return;
    }
    let storage = query
        .iter()
        .map(|(entity_id, val)| (*entity_id, val.component_value()))
        .collect();
    let data = ComponentData {
        component_id,
        storage,
    };
    let _ = subs.send(data);
}

pub fn query_component<T: bevy::prelude::Component + Component + Debug>(
    mut events: EventReader<SubscribeEvent<'static>>,
    query: Query<(&EntityId, &T)>,
) {
    let component_id = T::component_id();
    for event in events.read() {
        if !event
            .filters
            .iter()
            .any(|filter| filter.apply(component_id))
        {
            continue;
        }
        let storage = query
            .iter()
            .map(|(entity_id, val)| (*entity_id, val.component_value()))
            .collect();
        let data = ComponentData {
            component_id,
            storage,
        };
        let _ = event.tx.send(Msg::Data(data));
    }
}

#[derive(Resource, Default, Clone)]
pub struct Subscriptions {
    tree: Arc<
        Mutex<
            ip_network_table_deps_treebitmap::IpLookupTable<
                ComponentId,
                Vec<flume::Sender<Msg<'static>>>,
            >,
        >,
    >,
}

impl Subscriptions {
    pub fn send(&mut self, data: ComponentData<'static>) -> Result<(), SendError> {
        let mut tree = self.tree.lock().unwrap();
        for (_, _, txes) in tree.matches_mut(data.component_id) {
            let mut dead_sockets = vec![];
            for (i, tx) in txes.iter().enumerate() {
                if tx.send(Msg::Data(data.clone())).is_err() {
                    dead_sockets.push(i);
                }
            }
            for i in dead_sockets.into_iter().rev() {
                txes.swap_remove(i);
            }
        }
        Ok(())
    }

    pub fn subscribe(&mut self, filter: ComponentFilter, tx: flume::Sender<Msg<'static>>) {
        info!("add filter {:?}", filter);
        let mut tree = self.tree.lock().unwrap();
        if let Some(vec) = tree.exact_match_mut(ComponentId(filter.id), filter.mask_len as u32) {
            vec.push(tx);
        } else {
            tree.insert(ComponentId(filter.id), filter.mask_len as u32, vec![tx]);
        }
    }
}

#[derive(Debug)]
pub struct SendError;

impl Address for ComponentId {
    type Nibbles = [u8; 16];

    fn nibbles(self) -> Self::Nibbles {
        let mut ret: Self::Nibbles = [0; 16];
        let bytes: [u8; 8] = self.0.to_be_bytes();
        for (i, byte) in bytes.iter().enumerate() {
            ret[i * 2] = byte >> 4;
            ret[i * 2 + 1] = byte & 0xf;
        }
        ret
    }

    fn from_nibbles(nibbles: &[u8]) -> Self {
        let mut ret: [u8; 8] = [0; 8];
        for (i, nibble) in nibbles.iter().enumerate().take(ret.len() * 2) {
            match i % 2 {
                0 => {
                    ret[i / 2] = *nibble << 4;
                }
                _ => {
                    ret[i / 2] |= *nibble;
                }
            }
        }
        ComponentId(u64::from_be_bytes(ret))
    }

    fn mask(self, masklen: u32) -> Self {
        let masked = match masklen {
            0 => 0,
            n => self.0 & (!0 << (64 - n)),
        };
        Self(masked)
    }
}

#[cfg(test)]
mod tests {
    use crate::{cid, cid_mask};

    use super::*;

    #[test]
    fn test_subscription() {
        let mut subs = Subscriptions::default();
        let (tx, _rx) = flume::unbounded();
        subs.subscribe(cid_mask!(31), tx);
        assert!(subs.tree.lock().unwrap().matches(cid!(31:12)).any(|_| true));
    }
}
