use arc_swap::ArcSwap;
use assets::Assets;
use dashmap::DashMap;
use futures_lite::future;
use impeller2::types::OwnedPacket as Packet;
use impeller2::{
    com_de::Decomponentize,
    component::Component as _,
    registry,
    schema::Schema,
    table::{VTable, VTableBuilder},
    types::{ComponentId, ComponentView, EntityId, LenPacket, Msg, MsgExt, PacketId, PrimType},
};
use impeller2_stella::PacketSink;
use impeller2_wkt::{
    Asset, ComponentMetadata, DbSettings, DumpAssets, DumpMetadata, DumpMetadataResp,
    EntityMetadata, GetAsset, GetComponentMetadata, GetDbSettings, GetEntityMetadata, GetSchema,
    GetTimeSeries, SchemaMsg, SetAsset, SetComponentMetadata, SetDbSettings, SetEntityMetadata,
    SetStreamState, Stream, StreamFilter, StreamId, SubscribeMaxTick, VTableMsg,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use smallvec::SmallVec;
use std::{borrow::Cow, ffi::OsStr};
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
    io::{AsyncWrite, OwnedWriter, SplitExt},
    net::{TcpListener, TcpStream},
    rent,
    sync::{Mutex, RwLock, WaitQueue},
};
use time_series::{TimeSeries, TimeSeriesWriter};
use tracing::{debug, info, trace, warn};

pub use error::Error;

mod assets;
mod error;
pub(crate) mod time_series;

pub struct DB {
    pub path: PathBuf,
    pub latest_tick: AtomicU64,
    pub vtable_gen: AtomicU64,
    pub vtable_registry: RwLock<registry::HashMapRegistry>,
    pub components: DashMap<ComponentId, Component>,
    pub tick_waker: WaitQueue,
    pub streams: DashMap<StreamId, Arc<StreamState>>,
    pub entity_metadata: DashMap<EntityId, EntityMetadata>,
    pub assets: Assets,
    pub recording_cell: PlayingCell,
    pub time_step: AtomicU64,
}

impl DB {
    pub fn create(path: PathBuf) -> Result<Self, Error> {
        Self::with_time_step(path, Duration::from_secs_f64(1.0 / 120.0))
    }

    pub fn with_time_step(path: PathBuf, time_step: Duration) -> Result<Self, Error> {
        info!(?path, "created db");

        let assets = Assets::open(path.join("assets"))?;
        std::fs::create_dir_all(path.join("entity_metadata"))?;

        let db = DB {
            path,
            latest_tick: AtomicU64::new(0),
            vtable_registry: Default::default(),
            components: Default::default(),
            vtable_gen: AtomicU64::new(0),
            tick_waker: WaitQueue::new(),
            streams: Default::default(),
            entity_metadata: Default::default(),
            recording_cell: PlayingCell::new(true),
            assets,
            time_step: AtomicU64::new(time_step.as_nanos() as u64),
        };
        db.init_globals();
        db.save_db_state()?;
        Ok(db)
    }

    fn init_globals(&self) {
        let mut sink = DBSink(self);
        let arr = nox::array![0u64];
        sink.apply_value(
            impeller2_wkt::Tick::COMPONENT_ID,
            EntityId(0),
            ComponentView::U64(arr.view()),
        );
    }

    fn db_settings(&self) -> DbSettings {
        DbSettings {
            recording: self.recording_cell.is_playing(),
            time_step: Duration::from_nanos(self.time_step.load(atomic::Ordering::SeqCst)),
        }
    }

    fn save_db_state(&self) -> Result<(), Error> {
        let db_state = self.db_settings();
        db_state.write(self.path.join("db_state"))
    }

    pub fn open(path: PathBuf) -> Result<Self, Error> {
        let components = DashMap::new();
        let mut latest_tick = 0;
        for elem in std::fs::read_dir(&path)? {
            let Ok(elem) = elem else { continue };
            let path = elem.path();
            if !path.is_dir()
                || (path.file_name() == Some(OsStr::new("entity_metadata"))
                    || path.file_name() == Some(OsStr::new("assets")))
            {
                continue;
            }

            let component_id = ComponentId(
                path.file_name()
                    .and_then(|p| p.to_str())
                    .and_then(|p| p.parse().ok())
                    .ok_or(Error::InvalidComponentId)?,
            );

            let schema = ComponentSchema::read(path.join("schema"))?;
            let metadata = ComponentMetadata::read(path.join("metadata"))?;

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
            let metadata = EntityMetadata::read(path)?;
            entity_metadata.insert(entity_id, metadata);
        }
        info!(db.path = ?path, db.latest_tick = ?latest_tick, "opened db");
        let assets = Assets::open(path.join("assets"))?;
        let db_state = DbSettings::read(path.join("db_state"))?;
        Ok(DB {
            path,
            latest_tick: AtomicU64::new(latest_tick),
            vtable_gen: Default::default(),
            vtable_registry: Default::default(),
            components,
            tick_waker: WaitQueue::new(),
            streams: Default::default(),
            entity_metadata,
            assets,
            recording_cell: PlayingCell::new(db_state.recording),
            time_step: AtomicU64::new(db_state.time_step.as_nanos() as u64),
        })
    }

    pub fn time_step(&self) -> Duration {
        Duration::from_nanos(self.time_step.load(atomic::Ordering::Relaxed))
    }

    pub fn vtable_registry(&self) -> &RwLock<registry::HashMapRegistry> {
        &self.vtable_registry
    }
}

pub struct Component {
    pub schema: ComponentSchema,
    pub metadata: arc_swap::ArcSwap<ComponentMetadata>,
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
            dim: shape.iter().copied().collect(),
        };
        let component_dir = db_path.join(component_id.to_string());
        std::fs::create_dir_all(&component_dir)?;
        let schema_path = component_dir.join("schema");
        schema.write(schema_path)?;
        let metadata = ComponentMetadata {
            component_id,
            name: Cow::Owned(component_id.to_string()),
            metadata: Default::default(),
            asset: false,
        };
        metadata.write(component_dir.join("metadata"))?;
        Ok(Component {
            schema,
            entities: Default::default(),
            metadata: ArcSwap::new(Arc::new(metadata)),
        })
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct ComponentSchema {
    pub component_id: ComponentId,
    pub prim_type: PrimType,
    pub shape: SmallVec<[u64; 4]>,
    pub dim: SmallVec<[usize; 4]>,
}

impl ComponentSchema {
    pub fn size(&self) -> usize {
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

    pub fn to_schema(&self) -> Schema<Vec<u64>> {
        Schema::new(self.component_id, self.prim_type, &self.dim).expect("failed to create shape")
    }

    pub fn parse_value<'a>(&'a self, buf: &'a [u8]) -> Result<(usize, ComponentView<'a>), Error> {
        let size = self.size();
        let buf = buf
            .get(..size)
            .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?;
        let view = match self.prim_type {
            PrimType::U8 => ComponentView::U8(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::U16 => ComponentView::U16(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::U32 => ComponentView::U32(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::U64 => ComponentView::U64(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I8 => ComponentView::I8(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I16 => ComponentView::I16(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I32 => ComponentView::I32(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I64 => ComponentView::I64(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::Bool => ComponentView::Bool(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::F32 => ComponentView::F32(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::F64 => ComponentView::F64(
                nox::ArrayView::from_bytes_shape_unchecked(buf, &self.dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
        };
        Ok((size, view))
    }
}

impl From<Schema<Vec<u64>>> for ComponentSchema {
    fn from(value: Schema<Vec<u64>>) -> Self {
        let component_id = value.component_id();
        let prim_type = value.prim_type();
        let shape = value.shape().iter().map(|&x| x as u64).collect();
        let dim = value.shape().iter().copied().collect();
        ComponentSchema {
            component_id,
            prim_type,
            shape,
            dim,
        }
    }
}

pub trait MetadataExt: Sized + Serialize + DeserializeOwned {
    fn read(path: impl AsRef<Path>) -> Result<Self, Error> {
        let data = std::fs::read(path)?;
        Ok(postcard::from_bytes(&data)?)
    }
    fn write(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let data = postcard::to_allocvec(&self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}

impl MetadataExt for EntityMetadata {}
impl MetadataExt for ComponentMetadata {}

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

struct DBSink<'a>(&'a DB);

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
        let _ = self
            .0
            .entity_metadata
            .entry(entity_id)
            .or_insert_with(|| EntityMetadata {
                entity_id,
                name: entity_id.to_string(),
                metadata: Default::default(),
            });
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
}

impl Server {
    pub fn new(path: impl AsRef<Path>, addr: SocketAddr) -> Result<Server, Error> {
        let listener = TcpListener::bind(addr)?;
        let path = path.as_ref().to_path_buf();
        let db = if path.exists() {
            DB::open(path)?
        } else {
            DB::create(path)?
        };
        Ok(Server {
            listener,
            db: Arc::new(db),
        })
    }

    pub async fn run(self) -> Result<(), Error> {
        let Self { listener, db } = self;
        stellarator::spawn(tick(db.clone()));
        loop {
            let stream = listener.accept().await?;
            let conn_db = db.clone();
            stellarator::struc_con::stellar(move || handle_conn(stream, conn_db));
        }
    }
}

async fn tick(db: Arc<DB>) {
    let mut start = Instant::now();
    loop {
        if !db.recording_cell.wait().await {
            return;
        }
        for kv in db.components.iter() {
            for entity_kv in kv.value().entities.iter() {
                let entity = entity_kv.value();
                let mut writer = entity.writer.lock().unwrap();
                if let Err(err) = writer.commit_head_copy() {
                    warn!(?err, "failed to commit head")
                }
            }
        }

        let last_tick = db.latest_tick.fetch_add(1, atomic::Ordering::Release);
        if let Some(tick) = db.components.get(&impeller2_wkt::Tick::COMPONENT_ID) {
            if let Some(entity) = tick.entities.get(&EntityId(0)) {
                let mut writer = entity.writer.lock().unwrap();
                let latest_tick = last_tick + 1; // NOTE: there might be an off by one error here
                writer
                    .head_mut(8)
                    .unwrap()
                    .copy_from_slice(&latest_tick.to_le_bytes());
            }
        }

        db.tick_waker.wake_all();
        let time_step = db.time_step();
        let sleep_time = time_step.saturating_sub(start.elapsed());
        stellarator::sleep(sleep_time).await;
        let now = Instant::now();
        while start < now {
            start += time_step;
        }
    }
}

pub async fn handle_conn(stream: TcpStream, db: Arc<DB>) {
    match handle_conn_inner(stream, db).await {
        Ok(_) => {}
        Err(err) if err.is_stream_closed() => {}
        Err(err) => {
            warn!(?err, "error handling stream")
        }
    }
}

async fn handle_conn_inner(stream: TcpStream, db: Arc<DB>) -> Result<(), Error> {
    let (rx, tx) = stream.split();
    let mut rx = impeller2_stella::PacketStream::new(rx);
    let tx = Arc::new(Mutex::new(impeller2_stella::PacketSink::new(tx)));
    let mut buf = vec![0u8; 1024 * 64];
    loop {
        let pkt = rx.next(buf).await?;
        match handle_packet(&pkt, &db, &tx).await {
            Ok(_) => {}
            Err(err) if err.is_stream_closed() => {}
            Err(err) => {
                warn!(?err, "error handling packet");
            }
        }
        buf = pkt.into_buf();
    }
}

async fn handle_packet(
    pkt: &Packet<Vec<u8>>,
    db: &Arc<DB>,
    tx: &Arc<Mutex<impeller2_stella::PacketSink<OwnedWriter<TcpStream>>>>,
) -> Result<(), Error> {
    trace!(?pkt, "handling pkt");
    match &pkt {
        Packet::Msg(m) if m.id == VTableMsg::ID => {
            let vtable = m.parse::<VTableMsg>()?;
            debug!(id = ?vtable.id, "inserting vtable");
            let mut registry = db.vtable_registry.write().await;
            registry.map.insert(vtable.id, vtable.vtable);
        }
        Packet::Msg(m) if m.id == Stream::ID => {
            let stream = m.parse::<Stream>()?;
            let stream_id = stream.id;
            let state = Arc::new(StreamState::from_state(
                stream,
                db.latest_tick.load(atomic::Ordering::Relaxed),
                &db.time_step,
            ));
            debug!(stream.id = ?stream_id, "inserting stream");
            db.streams.insert(stream_id, state.clone());
            let db = db.clone();
            let tx = tx.clone();
            stellarator::spawn(async move {
                match handle_stream(tx, state, db).await {
                    Ok(_) => {}
                    Err(err) if err.is_stream_closed() => {}
                    Err(err) => {
                        warn!(?err, "error streaming data");
                    }
                }
            });
        }
        Packet::Msg(m) if m.id == SetStreamState::ID => {
            let set_stream_state = m.parse::<SetStreamState>()?;
            let stream_id = set_stream_state.id;
            let Some(state) = db.streams.get(&stream_id) else {
                warn!(stream.id = stream_id, "stream not found");
                return Ok(());
            };
            debug!(msg = ?set_stream_state, "set_stream_state received");
            if let Some(playing) = set_stream_state.playing {
                state.set_playing(playing);
            }
            if let Some(tick) = set_stream_state.tick {
                state.set_tick(tick);
            }
            if let Some(time_step) = set_stream_state.time_step {
                state.set_time_step(time_step);
            }
        }
        Packet::Msg(m) if m.id == GetSchema::ID => {
            let get_schema = m.parse::<GetSchema>()?;
            let schema = if let Some(component) = db.components.get(&get_schema.component_id) {
                component.schema.to_schema()
            } else {
                todo!("send error here")
            };
            let tx = tx.lock().await;
            tx.send(SchemaMsg(schema).to_len_packet()).await.0?;
        }
        Packet::Msg(m) if m.id == GetTimeSeries::ID => {
            let get_time_series = m.parse::<GetTimeSeries>()?;
            debug!(msg = ?get_time_series, "get time series");

            let tx = tx.clone();
            let db = db.clone();
            stellarator::spawn(async move {
                let pkt = {
                    let Some(component) = db.components.get(&get_time_series.component_id) else {
                        return Ok(());
                    };
                    let Some(entity) = component.entities.get(&get_time_series.entity_id) else {
                        return Ok(());
                    };
                    let Some(time_series) = entity.get_range(get_time_series.range) else {
                        return Ok(());
                    };
                    let mut pkt = LenPacket::time_series(get_time_series.id, time_series.len());
                    pkt.extend_from_slice(time_series); // TODO: make this zero copy
                    pkt
                };
                let tx = tx.lock().await;
                tx.send(pkt).await.0?;
                Ok::<_, Error>(())
            });
        }
        Packet::Msg(m) if m.id == SetComponentMetadata::ID => {
            let SetComponentMetadata {
                component_id,
                metadata,
                name,
                asset,
            } = m.parse::<SetComponentMetadata>()?;
            debug!(component.id = ?component_id, name = ?name, metadata = ?metadata, "set component metadata");
            let Some(component) = db.components.get(&component_id) else {
                debug!(component.id = ?component_id, "component not found");
                return Ok(());
            };
            let metadata = ComponentMetadata {
                component_id,
                name: name.into(),
                metadata,
                asset,
            };
            debug!(?metadata, component.id = ?component_id, "set component metadata");
            metadata.write(db.path.join(component_id.to_string()).join("metadata"))?;
            component.metadata.store(Arc::new(metadata));
        }
        Packet::Msg(m) if m.id == GetComponentMetadata::ID => {
            let GetComponentMetadata { component_id } = m.parse::<GetComponentMetadata>()?;
            let metadata = {
                let Some(component) = db.components.get(&component_id) else {
                    return Ok(());
                };
                component.metadata.load()
            };
            let tx = tx.lock().await;
            tx.send(metadata.to_len_packet()).await.0?;
        }

        Packet::Msg(m) if m.id == SetEntityMetadata::ID => {
            let set_entity_metadata = m.parse::<SetEntityMetadata>()?;
            debug!(msg = ?set_entity_metadata, "set entity metadata");
            let entity_metadata_path = db.path.join("entity_metadata");
            std::fs::create_dir_all(&entity_metadata_path)?;
            let path = entity_metadata_path.join(set_entity_metadata.entity_id.to_string());

            let metadata = EntityMetadata {
                entity_id: set_entity_metadata.entity_id,
                name: set_entity_metadata.name,
                metadata: set_entity_metadata.metadata,
            };
            debug!(?metadata, entity.id = ?set_entity_metadata.entity_id, "set entity metadata");
            metadata.write(&path)?;
            db.entity_metadata
                .insert(set_entity_metadata.entity_id, metadata);
        }

        Packet::Msg(m) if m.id == GetEntityMetadata::ID => {
            let GetEntityMetadata { entity_id } = m.parse::<GetEntityMetadata>()?;
            let msg = {
                let Some(metadata) = db.entity_metadata.get(&entity_id) else {
                    return Ok(());
                };
                metadata.to_len_packet()
            };
            let tx = tx.lock().await;
            tx.send(msg).await.0?;
        }

        Packet::Msg(m) if m.id == SetAsset::ID => {
            let set_asset = dbg!(m.parse::<SetAsset<'_>>())?;
            db.assets.insert(set_asset.id, &set_asset.buf)?;
        }
        Packet::Msg(m) if m.id == GetAsset::ID => {
            let GetAsset { id } = m.parse::<GetAsset>()?;
            let Some(mmap) = db.assets.get(id) else {
                warn!(?id, "asset not found");
                return Ok(());
            };
            let asset = Asset {
                id,
                buf: Cow::Borrowed(&mmap[..]),
            };
            let tx = tx.lock().await;
            tx.send(asset.to_len_packet()).await.0?;
        }
        Packet::Msg(m) if m.id == DumpMetadata::ID => {
            let entity_metadata: Vec<_> = db
                .entity_metadata
                .iter()
                .map(|kv| kv.value().clone())
                .collect();
            let component_metadata: Vec<_> = db
                .components
                .iter()
                .map(|kv| kv.value().metadata.load().as_ref().clone())
                .collect();
            let msg = DumpMetadataResp {
                component_metadata,
                entity_metadata,
            };
            let tx = tx.lock().await;
            tx.send(msg.to_len_packet()).await.0?;
        }
        Packet::Msg(m) if m.id == DumpAssets::ID => {
            let tx = tx.lock().await;
            for kv in db.assets.items.iter() {
                let (id, mmap) = kv.pair();

                let asset = Asset {
                    id: *id,
                    buf: Cow::Borrowed(&mmap[..]),
                };
                tx.send(asset.to_len_packet()).await.0?;
            }
        }
        Packet::Msg(m) if m.id == SubscribeMaxTick::ID => {
            let tx = tx.clone();
            let db = db.clone();
            stellarator::spawn(async move {
                loop {
                    let latest_tick = db.latest_tick.load(atomic::Ordering::Relaxed);
                    {
                        let tx = tx.lock().await;
                        match tx
                            .send(impeller2_wkt::MaxTick(latest_tick).to_len_packet())
                            .await
                            .0
                            .map_err(Error::from)
                        {
                            Err(err) if err.is_stream_closed() => {}
                            Err(err) => {
                                warn!(?err, "failed to send packet");
                                return;
                            }
                            _ => (),
                        }
                    }
                    if db
                        .tick_waker
                        .wait_for(|| db.latest_tick.load(atomic::Ordering::Relaxed) > latest_tick)
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            });
        }
        Packet::Msg(m) if m.id == SetDbSettings::ID => {
            let SetDbSettings {
                recording,
                time_step,
            } = m.parse::<SetDbSettings>()?;
            if let Some(recording) = recording {
                db.recording_cell.set_playing(recording);
            }
            if let Some(time_step) = time_step {
                db.time_step
                    .store(time_step.as_nanos() as u64, atomic::Ordering::Release);
            }
            db.save_db_state()?;
            let tx = tx.lock().await;
            tx.send(db.db_settings().to_len_packet()).await.0?;
        }
        Packet::Msg(m) if m.id == GetDbSettings::ID => {
            let tx = tx.lock().await;
            let settings = db.db_settings();
            tx.send(settings.to_len_packet()).await.0?;
        }
        Packet::Table(table) => {
            debug!(table.len = table.buf.len(), "sinking table");
            let registry = db.vtable_registry.read().await;
            let mut sink = DBSink(db);
            table.sink(&*registry, &mut sink)?;
        }
        _ => {}
    }
    Ok(())
}

pub struct StreamState {
    stream_id: StreamId,
    time_step: AtomicU64,
    is_scrubbed: AtomicBool,
    current_tick: AtomicU64,
    playing_cell: PlayingCell,
    filter: StreamFilter,
}

impl StreamState {
    fn from_state(stream: Stream, latest_tick: u64, db_timestep: &AtomicU64) -> StreamState {
        StreamState {
            stream_id: stream.id,
            time_step: AtomicU64::new(
                stream
                    .time_step
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or_else(|| db_timestep.load(atomic::Ordering::Relaxed)),
            ),
            is_scrubbed: AtomicBool::new(false),
            current_tick: AtomicU64::new(stream.start_tick.unwrap_or(latest_tick)),
            playing_cell: PlayingCell::new(true),
            filter: stream.filter,
        }
    }

    fn is_scrubbed(&self) -> bool {
        self.is_scrubbed.swap(false, atomic::Ordering::Relaxed)
    }

    fn set_tick(&self, tick: u64) {
        self.is_scrubbed.store(true, atomic::Ordering::SeqCst);
        self.current_tick.store(tick, atomic::Ordering::SeqCst);
        self.playing_cell.wait_cell.wake_all();
    }

    fn set_time_step(&self, time_step: Duration) {
        self.time_step
            .store(time_step.as_nanos() as u64, atomic::Ordering::SeqCst);
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

    fn is_playing(&self) -> bool {
        self.playing_cell.is_playing()
    }

    fn set_playing(&self, playing: bool) {
        self.playing_cell.set_playing(playing)
    }
}

async fn handle_stream<A: AsyncWrite>(
    stream: Arc<Mutex<PacketSink<A>>>,
    state: Arc<StreamState>,
    db: Arc<DB>,
) -> Result<(), Error> {
    let mut current_gen = u64::MAX;
    let mut table = LenPacket::table([0; 3], 2048 - 16);
    loop {
        if state
            .playing_cell
            .wait_cell
            .wait_for(|| state.is_playing() || state.is_scrubbed())
            .await
            .is_err()
        {
            return Ok(());
        }
        if future::race(
            db.tick_waker
                .wait_for(|| db.latest_tick.load(atomic::Ordering::Relaxed) > state.current_tick()),
            state
                .playing_cell
                .wait_cell
                .wait_for(|| db.latest_tick.load(atomic::Ordering::Relaxed) > state.current_tick()),
        )
        .await
        .is_err()
        {
            return Ok(());
        }
        let start = Instant::now();
        let tick = state.current_tick();
        let gen = db.vtable_gen.load(atomic::Ordering::SeqCst);
        if gen != current_gen {
            let stream = stream.lock().await;
            let id: PacketId = state.stream_id.to_le_bytes()[..3].try_into().unwrap();
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
        {
            let stream = stream.lock().await;
            rent!(stream.send(table).await, table)?;
        }

        let time_step = state.time_step();
        let sleep_time = time_step.saturating_sub(start.elapsed());
        futures_lite::future::race(
            async {
                state.playing_cell.wait_for_change().await;
            },
            stellarator::sleep(sleep_time),
        )
        .await;
        state.try_increment_tick(tick);
    }
}

pub trait StreamFilterExt {
    fn vtable_id(&self, db: &DB) -> Result<PacketId, Error>;
    fn vtable(&self, db: &DB) -> Result<VTable<Vec<impeller2::table::Entry>, Vec<u8>>, Error>;
    fn populate_table(&self, db: &DB, table: &mut LenPacket, tick: u64) -> Result<(), Error>;
}

impl StreamFilterExt for StreamFilter {
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
                    .ok_or(Error::ComponentNotFound(component_id))?;
                component.add_to_vtable(&mut vtable)?;
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
                        let tick = entity.time_series.start_tick().max(tick);
                        let Some(buf) = entity.get(tick) else {
                            continue;
                        };
                        table.pad_for_type(kv.schema.prim_type);
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
                    let tick = entity.time_series.start_tick().max(tick);
                    let Some(buf) = entity.get(tick) else {
                        continue;
                    };
                    table.pad_for_type(component.schema.prim_type);
                    table.extend_from_slice(buf);
                }
            }
            (Some(id), None) => {
                let component = db.components.get(&id).ok_or(Error::ComponentNotFound(id))?;
                for entity_kv in &component.entities {
                    let entity = entity_kv.value();
                    let tick = entity.time_series.start_tick().max(tick);
                    let Some(buf) = entity.get(tick) else {
                        continue;
                    };
                    table.pad_for_type(component.schema.prim_type);
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
                let tick = entity.time_series.start_tick().max(tick);
                let Some(buf) = entity.get(tick) else {
                    return Ok(());
                };
                table.pad_for_type(component.schema.prim_type);
                table.extend_from_slice(buf);
            }
        }
        Ok(())
    }
}

pub struct PlayingCell {
    is_playing: AtomicBool,
    wait_cell: WaitQueue,
}

impl PlayingCell {
    fn new(is_playing: bool) -> Self {
        Self {
            is_playing: AtomicBool::new(is_playing),
            wait_cell: WaitQueue::new(),
        }
    }

    fn set_playing(&self, playing: bool) {
        self.is_playing.store(playing, atomic::Ordering::SeqCst);
        self.wait_cell.wake_all();
    }

    fn is_playing(&self) -> bool {
        self.is_playing.load(atomic::Ordering::Relaxed)
    }

    pub async fn wait(&self) -> bool {
        self.wait_cell.wait_for(|| self.is_playing()).await.is_ok()
    }

    async fn wait_for_change(&self) {
        let _ = self.wait_cell.wait().await;
    }
}

impl MetadataExt for DbSettings {}
