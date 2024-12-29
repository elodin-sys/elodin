use arc_swap::ArcSwap;
use control::{
    ComponentMetadata, EntityMetadata, GetComponentMetadata, GetEntityMetadata, GetSchema,
    GetTimeSeries, Metadata, SchemaMsg, SetComponentMetadata, SetEntityMetadata, SetStreamState,
    Stream, StreamFilter, StreamId, VTableMsg,
};
use dashmap::DashMap;
use impeller2::{
    com_de::Decomponentize,
    schema::Schema,
    table::{VTable, VTableBuilder},
    types::{ComponentId, EntityId, PacketId, PrimType},
};
use impeller2_stella::{LenPacket, Msg, Packet, PacketSink};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::ffi::OsStr;
use std::{
    hash::{DefaultHasher, Hash, Hasher},
    net::SocketAddr,
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        atomic::{self, AtomicBool, AtomicU64},
        Arc,
    },
    time::Instant,
};
use std::{sync::Mutex as SyncMutex, time::Duration};
use stellarator::{
    io::{AsyncWrite, SplitExt},
    net::{TcpListener, TcpStream},
    rent,
    sync::{Mutex, RwLock, WaitCell, WaitQueue},
};
use time_series::{TimeSeries, TimeSeriesWriter};
use tracing::{info, trace, warn};

pub use error::Error;

pub mod control;
mod error;
mod registry;
pub(crate) mod time_series;

pub struct DB {
    pub path: PathBuf,
    pub latest_tick: AtomicU64,
    pub vtable_gen: AtomicU64,
    pub vtable_registry: RwLock<registry::VTableRegistry>,
    pub components: DashMap<ComponentId, Component>,
    pub tick_waker: WaitQueue,
    pub streams: DashMap<StreamId, Arc<StreamState>>,
    pub entity_metadata: DashMap<EntityId, Metadata>,
}

impl DB {
    pub fn create(path: PathBuf) -> Self {
        info!(?path, "created db");

        DB {
            path,
            latest_tick: AtomicU64::new(0),
            vtable_registry: Default::default(),
            components: Default::default(),
            vtable_gen: AtomicU64::new(0),
            tick_waker: WaitQueue::new(),
            streams: Default::default(),
            entity_metadata: Default::default(),
        }
    }

    pub fn open(path: PathBuf) -> Result<Self, Error> {
        let components = DashMap::new();
        let mut latest_tick = 0;
        for elem in std::fs::read_dir(&path)? {
            let Ok(elem) = elem else { continue };
            let path = elem.path();
            if !path.is_dir() || path.file_name() == Some(OsStr::new("entity_metadata")) {
                continue;
            }

            let component_id = ComponentId(
                path.file_name()
                    .and_then(|p| p.to_str())
                    .and_then(|p| p.parse().ok())
                    .ok_or(Error::InvalidComponentId)?,
            );

            let schema = ComponentSchema::read(path.join("schema"))?;
            let metadata = Metadata::read(path.join("metadata"))?;

            let component = Component {
                schema,
                entities: DashMap::default(),
                metadata: arc_swap::ArcSwap::new(Arc::new(metadata)),
            };

            for elem in std::fs::read_dir(path)? {
                let Ok(elem) = elem else { continue };

                let path = elem.path();
                if path
                    .file_name()
                    .and_then(|p| p.to_str())
                    .map(|s| s == "schema" || s == "metadata")
                    .unwrap_or(false)
                {
                    continue;
                }

                let entity_id = EntityId(
                    path.file_name()
                        .and_then(|p| p.to_str())
                        .and_then(|p| p.parse().ok())
                        .ok_or(Error::InvalidComponentId)?,
                );

                let entity = Entity::open(path, entity_id, component.schema.clone())?;

                latest_tick = latest_tick.max(entity.latest_tick());
                component.entities.insert(entity_id, entity);
            }

            components.insert(component_id, component);
        }

        let entity_metadata = DashMap::new();
        for elem in std::fs::read_dir(path.join("entity_metadata"))? {
            let Ok(elem) = elem else { continue };

            let path = elem.path();
            let entity_id = EntityId(
                path.file_name()
                    .and_then(|p| p.to_str())
                    .and_then(|p| p.parse().ok())
                    .ok_or(Error::InvalidComponentId)?,
            );
            let metadata = Metadata::read(path)?;
            entity_metadata.insert(entity_id, metadata);
        }
        info!(db.path = ?path, db.latest_tick = ?latest_tick, "opened db");
        Ok(DB {
            path,
            latest_tick: AtomicU64::new(latest_tick),
            vtable_gen: Default::default(),
            vtable_registry: Default::default(),
            components,
            tick_waker: WaitQueue::new(),
            streams: Default::default(),
            entity_metadata,
        })
    }

    // pub fn try_get_or_insert_entity(
    //     &mut self,
    //     component_id: ComponentId,
    //     entity_id: EntityId,
    //     prim_type: PrimType,
    //     shape: &[u64],
    // ) -> Result<dashmap::mapref::one::MappedRefMut<'_, EntityId, Entity>, Error> {
    //     let component = self.components.entry(component_id).or_try_insert_with(|| {
    //         self.vtable_gen.fetch_add(1, atomic::Ordering::SeqCst);
    //         let schema = ComponentSchema {
    //             component_id,
    //             prim_type,
    //             shape: shape.iter().map(|&x| x as u64).collect(),
    //         };
    //         let component_dir = self.path.join(component_id.to_string());
    //         std::fs::create_dir_all(&component_dir)?;
    //         let schema_path = component_dir.join("schema");
    //         schema.write(schema_path)?;
    //         let metadata = Metadata::default();
    //         metadata.write(component_dir.join("metadata"))?;
    //         Ok::<_, Error>(Component {
    //             schema,
    //             entities: Default::default(),
    //             metadata: ArcSwap::new(Arc::new(metadata)),
    //         })
    //     })?;
    //     let entity = component.entities.entry(entity_id).or_try_insert_with(|| {
    //         self.vtable_gen.fetch_add(1, atomic::Ordering::SeqCst);
    //         let component_dir = self.path.join(component_id.to_string());
    //         std::fs::create_dir_all(&component_dir)?;
    //         let path = component_dir.join(entity_id.to_string());
    //         let start_tick = self.latest_tick.load(atomic::Ordering::SeqCst);
    //         Entity::create(path, start_tick, entity_id, component.schema.clone())
    //     })?;
    //     Ok(entity)
    // }
}

pub struct Component {
    pub schema: ComponentSchema,
    pub metadata: arc_swap::ArcSwap<Metadata>,
    pub entities: DashMap<EntityId, Entity>,
}

impl Component {
    pub fn try_create(
        component_id: ComponentId,
        prim_type: PrimType,
        shape: &[usize],
        db_path: &Path,
    ) -> Result<Self, Error> {
        let schema = ComponentSchema {
            component_id,
            prim_type,
            shape: shape.iter().map(|&x| x as u64).collect(),
        };
        let component_dir = db_path.join(component_id.to_string());
        std::fs::create_dir_all(&component_dir)?;
        let schema_path = component_dir.join("schema");
        schema.write(schema_path)?;
        let metadata = Metadata::default();
        metadata.write(component_dir.join("metadata"))?;
        Ok(Component {
            schema,
            entities: Default::default(),
            metadata: ArcSwap::new(Arc::new(metadata)),
        })
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ComponentSchema {
    pub component_id: ComponentId,
    pub prim_type: PrimType,
    pub shape: SmallVec<[u64; 4]>,
}

impl ComponentSchema {
    fn size(&self) -> usize {
        self.shape.iter().product::<u64>() as usize * self.prim_type.size()
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, Error> {
        let data = std::fs::read(path)?;
        Ok(postcard::from_bytes(&data)?)
    }

    fn write(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let data = postcard::to_allocvec(&self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn to_schema(&self) -> Schema<Vec<u8>> {
        Schema::new(self.component_id, self.prim_type, &self.shape).expect("failed to create shape")
    }
}

impl Metadata {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, Error> {
        let data = std::fs::read(path)?;
        Ok(postcard::from_bytes(&data)?)
    }

    fn write(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let data = postcard::to_allocvec(&self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}

impl Component {
    fn add_to_vtable(
        &self,
        vtable: &mut VTableBuilder<Vec<impeller2::table::Entry>, Vec<u8>>,
    ) -> Result<(), Error> {
        vtable.column(
            self.schema.component_id,
            self.schema.prim_type,
            &self.schema.shape,
            self.entities
                .iter()
                .map(|kv| *kv.key())
                .collect::<Vec<_>>()
                .into_iter(),
        )?;
        Ok(())
    }
}

pub struct Entity {
    pub entity_id: EntityId,
    pub time_series: TimeSeries,
    pub writer: SyncMutex<TimeSeriesWriter>,
    pub schema: ComponentSchema,
}

impl Entity {
    pub fn create(
        path: impl AsRef<Path>,
        start_tick: u64,
        entity_id: EntityId,
        schema: ComponentSchema,
    ) -> Result<Self, Error> {
        let (time_series, writer) = TimeSeries::create(path, start_tick)?;
        let writer = SyncMutex::new(writer);
        Ok(Entity {
            entity_id,
            time_series,
            writer,
            schema,
        })
    }

    pub fn open(
        path: impl AsRef<Path>,
        entity_id: EntityId,
        schema: ComponentSchema,
    ) -> Result<Self, Error> {
        let (time_series, writer) = TimeSeries::open(path)?;
        let writer = SyncMutex::new(writer);
        Ok(Entity {
            entity_id,
            time_series,
            writer,
            schema,
        })
    }

    fn add_to_vtable(
        &self,
        vtable: &mut VTableBuilder<Vec<impeller2::table::Entry>, Vec<u8>>,
    ) -> Result<(), Error> {
        vtable.entity(
            self.entity_id,
            &[(
                self.schema.component_id,
                self.schema.prim_type,
                &self.schema.shape,
            )],
        )?;
        Ok(())
    }

    fn tick_to_offset(&self, tick: u64) -> Option<usize> {
        let start_tick = self.time_series.start_tick();
        let schema_size = self.schema.size();
        (tick.checked_sub(start_tick)? as usize).checked_mul(schema_size)
    }

    fn get(&self, tick: u64) -> Option<&[u8]> {
        let buf_start = self.tick_to_offset(tick)?;
        let schema_size = self.schema.size();
        let buf_end = buf_start.checked_add(schema_size)?;
        self.time_series.get(buf_start..buf_end)
    }

    fn get_range(&self, range: Range<u64>) -> Option<&[u8]> {
        let buf_start = self.tick_to_offset(range.start)?;
        let buf_end = self.tick_to_offset(range.end)?;
        self.time_series.get(buf_start..buf_end)
    }

    fn len(&self) -> u64 {
        self.time_series.len() / self.schema.size() as u64
    }

    fn latest_tick(&self) -> u64 {
        self.len() + self.time_series.start_tick()
    }
}

struct DBSink<'a>(&'a Arc<DB>);

impl Decomponentize for DBSink<'_> {
    fn apply_value(
        &mut self,
        component_id: impeller2::types::ComponentId,
        entity_id: EntityId,
        value: impeller2::types::ComponentView<'_>,
    ) {
        let component = match self
            .0
            .components
            .entry(component_id)
            .or_try_insert_with(|| {
                self.0.vtable_gen.fetch_add(1, atomic::Ordering::SeqCst);
                Component::try_create(component_id, value.prim_type(), value.shape(), &self.0.path)
            }) {
            Ok(entity) => entity,
            Err(err) => {
                warn!(
                    component.id = ?component_id,
                    ?err,
                    "failed to create new component db"
                );
                return;
            }
        };
        let entity = match component.entities.entry(entity_id).or_try_insert_with(|| {
            self.0.vtable_gen.fetch_add(1, atomic::Ordering::SeqCst);
            let component_dir = self.0.path.join(component_id.to_string());
            let path = component_dir.join(entity_id.to_string());
            let start_tick = self.0.latest_tick.load(atomic::Ordering::SeqCst);
            Entity::create(path, start_tick, entity_id, component.schema.clone())
        }) {
            Ok(entity) => entity,
            Err(err) => {
                warn!(
                    entity.id = ?entity_id,
                    component.id = ?component_id,
                    ?err,
                    "failed to create new entity db"
                );
                return;
            }
        };
        let value_buf = value.as_bytes();
        let mut writer = entity.writer.lock().expect("poisoned lock");
        let head = match writer.head_mut(value_buf.len()) {
            Ok(h) => h,
            Err(err) => {
                warn!(?err, "failed to write head value");
                return;
            }
        };
        head.copy_from_slice(value_buf);
    }
}

pub struct Server {
    pub listener: TcpListener,
    pub db: Arc<DB>,
    pub time_step: Duration,
}

impl Server {
    pub fn new(path: impl AsRef<Path>, addr: SocketAddr) -> Result<Server, Error> {
        let listener = TcpListener::bind(addr)?;
        let path = path.as_ref().to_path_buf();
        let db = if path.exists() {
            DB::open(path)?
        } else {
            DB::create(path)
        };
        Ok(Server {
            listener,
            db: Arc::new(db),
            time_step: Duration::from_millis(20),
        })
    }

    pub async fn run(self) -> Result<(), Error> {
        let Self {
            listener,
            db,
            time_step,
        } = self;
        stellarator::spawn(tick(time_step, db.clone()));
        loop {
            let stream = listener.accept().await?;
            stellarator::spawn(handle_conn(stream, db.clone()));
        }
    }
}

async fn tick(time_step: Duration, db: Arc<DB>) {
    let mut start = Instant::now();
    loop {
        for kv in db.components.iter() {
            for entity_kv in kv.value().entities.iter() {
                let entity = entity_kv.value();
                let mut writer = entity.writer.lock().unwrap();
                if let Err(err) = writer.commit_head_copy() {
                    warn!(?err, "failed to commit head")
                }
            }
        }
        db.latest_tick.fetch_add(1, atomic::Ordering::Release);
        db.tick_waker.wake_all();
        let sleep_time = time_step.saturating_sub(start.elapsed());
        stellarator::sleep(sleep_time).await;
        start += time_step;
    }
}

pub async fn handle_conn(stream: TcpStream, db: Arc<DB>) {
    if let Err(err) = handle_conn_inner(stream, db).await {
        warn!(?err, "error handling stream")
    }
}

async fn handle_conn_inner(stream: TcpStream, db: Arc<DB>) -> Result<(), Error> {
    let (rx, tx) = stream.split();
    let mut rx = impeller2_stella::PacketStream::new(rx);
    let tx = Arc::new(Mutex::new(impeller2_stella::PacketSink::new(tx)));
    let mut buf = vec![0u8; 1024 * 64];
    loop {
        let pkt = rx.next(buf).await?;
        match &pkt {
            Packet::Msg(m) if m.id == VTableMsg::ID => {
                let vtable = m.parse::<VTableMsg>()?;
                let mut registry = db.vtable_registry.write().await;
                registry.map.insert(vtable.id, vtable.vtable);
            }
            Packet::Msg(m) if m.id == Stream::ID => {
                let stream = m.parse::<Stream>()?;
                let stream_id = stream.id;
                let state = Arc::new(StreamState::from_state(
                    stream,
                    db.latest_tick.load(atomic::Ordering::Relaxed),
                ));
                trace!(stream.id = ?stream_id, "inserting stream");
                db.streams.insert(stream_id, state.clone());
                let db = db.clone();
                let tx = tx.clone();
                stellarator::spawn(async move {
                    if let Err(err) = handle_stream(tx, state, db).await {
                        warn!(?err, "error streaming data");
                    }
                });
            }
            Packet::Msg(m) if m.id == SetStreamState::ID => {
                let set_stream_state = m.parse::<SetStreamState>()?;
                let stream_id = set_stream_state.id;
                let Some(state) = db.streams.get(&stream_id) else {
                    warn!(stream.id = stream_id, "stream not found");
                    buf = pkt.into_buf();
                    continue;
                };
                trace!(msg = ?set_stream_state, "set_stream_state received");
                if let Some(playing) = set_stream_state.playing {
                    state.set_playing(playing);
                }
                if let Some(tick) = set_stream_state.tick {
                    state.set_tick(tick);
                }
            }
            Packet::Msg(m) if m.id == GetSchema::ID => {
                let get_schema = m.parse::<GetSchema>()?;
                if let Some(component) = db.components.get(&get_schema.component_id) {
                    let schema = component.schema.to_schema();
                    let tx = tx.lock().await;
                    if let Err(err) = tx.send(SchemaMsg(schema).to_len_packet()).await.0 {
                        warn!(?err, "failed to send packet");
                    }
                } else {
                    todo!("send error here")
                }
            }
            Packet::Msg(m) if m.id == GetTimeSeries::ID => {
                let get_time_series = m.parse::<GetTimeSeries>()?;
                let Some(component) = db.components.get(&get_time_series.component_id) else {
                    buf = pkt.into_buf();
                    continue;
                };
                let Some(entity) = component.entities.get(&get_time_series.entity_id) else {
                    buf = pkt.into_buf();
                    continue;
                };
                let Some(time_series) = entity.get_range(get_time_series.range) else {
                    buf = pkt.into_buf();
                    continue;
                };
                let mut pkt = LenPacket::time_series(get_time_series.id, time_series.len());
                pkt.extend_from_slice(time_series); // TODO: make this zero copy
                let tx = tx.lock().await;
                if let Err(err) = tx.send(pkt).await.0 {
                    warn!(?err, "failed to send packet");
                }
            }
            Packet::Msg(m) if m.id == SetComponentMetadata::ID => {
                let set_component_metadata = m.parse::<SetComponentMetadata>()?;
                trace!(msg = ?set_component_metadata, "set component metadata");
                let Some(component) = db.components.get(&set_component_metadata.component_id)
                else {
                    trace!(component.id = ?set_component_metadata.component_id, "component not found");
                    buf = pkt.into_buf();
                    continue;
                };
                let metadata = set_component_metadata.metadata;
                trace!(?metadata, component.id = ?set_component_metadata.component_id, "set component metadata");
                if let Err(err) = metadata.write(
                    db.path
                        .join(set_component_metadata.component_id.to_string())
                        .join("metadata"),
                ) {
                    warn!(?err, "failed to write metadata");
                }
                component.metadata.store(Arc::new(metadata));
            }
            Packet::Msg(m) if m.id == GetComponentMetadata::ID => {
                let GetComponentMetadata { component_id } = m.parse::<GetComponentMetadata>()?;
                let Some(component) = db.components.get(&component_id) else {
                    buf = pkt.into_buf();
                    continue;
                };
                let metadata = component.metadata.load();
                let msg = ComponentMetadata {
                    component_id,
                    metadata: (**metadata).clone(),
                };
                let tx = tx.lock().await;
                if let Err(err) = tx.send(msg.to_len_packet()).await.0 {
                    warn!(?err, "failed to send packet");
                }
            }

            Packet::Msg(m) if m.id == SetEntityMetadata::ID => {
                let set_entity_metadata = m.parse::<SetEntityMetadata>()?;
                trace!(msg = ?set_entity_metadata, "set entity metadata");
                let entity_metadata_path = db.path.join("entity_metadata");
                if let Err(err) = std::fs::create_dir_all(&entity_metadata_path) {
                    warn!(?err, "failed to create entity dir");
                    buf = pkt.into_buf();
                    continue;
                }
                let path = entity_metadata_path.join(set_entity_metadata.entity_id.to_string());

                let metadata = set_entity_metadata.metadata;
                trace!(?metadata, entity.id = ?set_entity_metadata.entity_id, "set entity metadata");
                if let Err(err) = metadata.write(&path) {
                    warn!(?err, "failed to write metadata");
                    buf = pkt.into_buf();
                    continue;
                }
                db.entity_metadata
                    .insert(set_entity_metadata.entity_id, metadata);
            }

            Packet::Msg(m) if m.id == GetEntityMetadata::ID => {
                let GetEntityMetadata { entity_id } = m.parse::<GetEntityMetadata>()?;
                let Some(metadata) = db.entity_metadata.get(&entity_id) else {
                    buf = pkt.into_buf();
                    continue;
                };
                let msg = EntityMetadata {
                    entity_id,
                    metadata: metadata.clone(),
                };
                let tx = tx.lock().await;
                if let Err(err) = tx.send(msg.to_len_packet()).await.0 {
                    warn!(?err, "failed to send packet");
                }
            }
            Packet::Table(table) => {
                let registry = db.vtable_registry.read().await;
                let mut sink = DBSink(&db);
                if let Err(err) = table.sink(&*registry, &mut sink) {
                    warn!(?err, "failed to sink table into db")
                }
            }
            _ => {}
        }
        buf = pkt.into_buf();
    }
}

pub struct StreamState {
    time_step: AtomicU64,
    is_playing: AtomicBool,
    current_tick: AtomicU64,
    playing_cell: WaitCell,
    filter: StreamFilter,
}

impl StreamState {
    fn from_state(stream: Stream, latest_tick: u64) -> StreamState {
        StreamState {
            time_step: AtomicU64::new(stream.time_step.as_nanos() as u64),
            is_playing: AtomicBool::new(true),
            current_tick: AtomicU64::new(stream.start_tick.unwrap_or(latest_tick)),
            playing_cell: WaitCell::new(),
            filter: stream.filter,
        }
    }

    fn set_playing(&self, playing: bool) {
        self.is_playing.store(playing, atomic::Ordering::Relaxed);
        self.playing_cell.wake();
    }

    fn set_tick(&self, tick: u64) {
        self.current_tick.store(tick, atomic::Ordering::SeqCst);
    }

    fn is_playing(&self) -> bool {
        self.is_playing.load(atomic::Ordering::Relaxed)
    }

    fn time_step(&self) -> Duration {
        Duration::from_nanos(self.time_step.load(atomic::Ordering::Relaxed))
    }

    fn current_tick(&self) -> u64 {
        self.current_tick.load(atomic::Ordering::Relaxed)
    }

    fn try_increment_tick(&self, last_tick: u64) {
        let _ = self.current_tick.compare_exchange(
            last_tick,
            last_tick.saturating_add(1),
            atomic::Ordering::Acquire,
            atomic::Ordering::Relaxed,
        );
    }
}

async fn handle_stream<A: AsyncWrite>(
    stream: Arc<Mutex<PacketSink<A>>>,
    state: Arc<StreamState>,
    db: Arc<DB>,
) -> Result<(), Error> {
    let mut start = Instant::now();
    let mut current_gen = u64::MAX;
    let mut table = LenPacket::table([0; 3], 2048 - 16);
    loop {
        if state
            .playing_cell
            .wait_for(|| state.is_playing())
            .await
            .is_err()
        {
            return Ok(());
        }
        let tick = state.current_tick();
        if db
            .tick_waker
            .wait_for(|| db.latest_tick.load(atomic::Ordering::Relaxed) > tick)
            .await
            .is_err()
        {
            return Ok(());
        }

        let gen = db.vtable_gen.load(atomic::Ordering::SeqCst);
        let stream = stream.lock().await;
        if gen != current_gen {
            let id = state.filter.vtable_id(&db)?;
            table = LenPacket::table(id, 2048 - 16);
            let vtable = state.filter.vtable(&db)?;
            let msg = VTableMsg { id, vtable };
            stream.send(msg.to_len_packet()).await.0?;
            current_gen = gen;
        }
        table.clear();
        if let Err(err) = state.filter.populate_table(&db, &mut table, tick) {
            warn!(?err, "failed to populate table");
        }

        rent!(stream.send(table).await, table)?;

        let time_step = state.time_step();
        let sleep_time = time_step.saturating_sub(start.elapsed());
        stellarator::sleep(sleep_time).await; // TODO: select on current_tick change
        start += time_step;
        state.try_increment_tick(tick);
    }
}

impl StreamFilter {
    fn vtable_id(&self, db: &DB) -> Result<PacketId, Error> {
        let mut hasher = DefaultHasher::new();
        match (self.component_id, self.entity_id) {
            (None, None) => {
                hasher.write_u8(0b00u8);
                for kv in &db.components {
                    for entity_kv in &kv.value().entities {
                        let id = entity_kv.key();
                        id.hash(&mut hasher);
                    }
                }
            }
            (None, Some(id)) => {
                hasher.write_u8(0b01u8);
                for kv in &db.components {
                    let component = kv.value();
                    if component.entities.get(&id).is_some() {
                        continue;
                    }
                    id.hash(&mut hasher);
                }
            }
            (Some(id), None) => {
                hasher.write_u8(0b10u8);
                let component = db.components.get(&id).ok_or(Error::ComponentNotFound(id))?;
                for entity_kv in &component.entities {
                    let entity = entity_kv.key();
                    entity.hash(&mut hasher);
                }
            }
            (Some(component_id), Some(entity_id)) => {
                hasher.write_u8(0b11u8);
                let component = db
                    .components
                    .get(&component_id)
                    .ok_or(Error::ComponentNotFound(component_id))?;
                let _ = component
                    .entities
                    .get(&entity_id)
                    .ok_or(Error::EntityNotFound(entity_id))?;
                component_id.hash(&mut hasher);
                entity_id.hash(&mut hasher);
            }
        }
        let id = hasher.finish();
        Ok(id.to_le_bytes()[..3].try_into().unwrap())
    }

    fn vtable(&self, db: &DB) -> Result<VTable<Vec<impeller2::table::Entry>, Vec<u8>>, Error> {
        let mut vtable = VTableBuilder::default();
        match (self.component_id, self.entity_id) {
            (None, None) => {
                for kv in &db.components {
                    let (_, component) = kv.pair();
                    component.add_to_vtable(&mut vtable)?;
                }
            }
            (None, Some(id)) => {
                for kv in &db.components {
                    let component = kv.value();
                    let Some(entity) = component.entities.get(&id) else {
                        continue;
                    };
                    entity.add_to_vtable(&mut vtable)?;
                }
            }
            (Some(component_id), None) => {
                let component = db
                    .components
                    .get(&component_id)
                    .ok_or_else(|| Error::ComponentNotFound(component_id))?;
                component.add_to_vtable(&mut vtable)?;
            }
            (Some(component_id), Some(entity_id)) => {
                let component = db
                    .components
                    .get(&component_id)
                    .ok_or_else(|| Error::ComponentNotFound(component_id))?;

                let entity = component
                    .entities
                    .get(&entity_id)
                    .ok_or_else(|| Error::EntityNotFound(entity_id))?;
                entity.add_to_vtable(&mut vtable)?;
            }
        }
        Ok(vtable.build())
    }

    fn populate_table(&self, db: &DB, table: &mut LenPacket, tick: u64) -> Result<(), Error> {
        match (self.component_id, self.entity_id) {
            (None, None) => {
                for kv in &db.components {
                    for entity_kv in &kv.value().entities {
                        let entity = entity_kv.value();
                        let Some(buf) = entity.get(tick) else {
                            continue;
                        };
                        table.extend_from_slice(buf);
                    }
                }
            }
            (None, Some(id)) => {
                for kv in &db.components {
                    let component = kv.value();
                    let Some(entity) = component.entities.get(&id) else {
                        continue;
                    };
                    let Some(buf) = entity.get(tick) else {
                        continue;
                    };
                    table.extend_from_slice(buf);
                }
            }
            (Some(id), None) => {
                let component = db.components.get(&id).ok_or(Error::ComponentNotFound(id))?;
                for entity_kv in &component.entities {
                    let entity = entity_kv.value();
                    let Some(buf) = entity.get(tick) else {
                        continue;
                    };
                    table.extend_from_slice(buf);
                }
            }
            (Some(component_id), Some(entity_id)) => {
                let component = db
                    .components
                    .get(&component_id)
                    .ok_or(Error::ComponentNotFound(component_id))?;
                let entity = component
                    .entities
                    .get(&entity_id)
                    .ok_or(Error::EntityNotFound(entity_id))?;
                let Some(buf) = entity.get(tick) else {
                    return Ok(());
                };
                table.extend_from_slice(buf);
            }
        }
        Ok(())
    }
}
