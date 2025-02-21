use arc_swap::ArcSwap;
use assets::Assets;
use dashmap::DashMap;
use impeller2::{
    com_de::Decomponentize,
    component::Component as _,
    registry,
    schema::Schema,
    table::{Entry, VTable, VTableBuilder},
    types::{
        ComponentId, ComponentView, EntityId, LenPacket, Msg, MsgExt, OwnedPacket as Packet,
        PacketId, PrimType, Timestamp,
    },
};
use impeller2_stella::PacketSink;
use impeller2_wkt::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use smallvec::SmallVec;
use std::{borrow::Cow, collections::HashSet, ffi::OsStr, sync::atomic::AtomicI64};
use std::{
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
    buf::Slice,
    io::{AsyncWrite, OwnedWriter, SplitExt},
    net::{TcpListener, TcpStream},
    rent,
    sync::{Mutex, RwLock, WaitQueue},
    util::AtomicCell,
};
use time_series::{TimeSeries, TimeSeriesWriter};
use tracing::{debug, info, trace, warn};
use zerocopy::IntoBytes;

pub use error::Error;

pub mod append_log;
mod assets;
mod error;
pub(crate) mod time_series;

pub struct DB {
    // storage
    pub components: DashMap<ComponentId, Component>,
    pub entity_metadata: DashMap<EntityId, EntityMetadata>,
    pub vtable_gen: AtomicCell<u64>,
    pub assets: Assets,

    // state
    pub vtable_registry: RwLock<registry::HashMapRegistry>,
    pub streams: DashMap<StreamId, Arc<FixedRateStreamState>>,
    pub recording_cell: PlayingCell,

    // metadata
    pub path: PathBuf,
    pub time_step: AtomicU64,
    pub default_stream_time_step: AtomicU64,
    pub last_updated: AtomicCell<Timestamp>,
    pub earliest_timestamp: Timestamp,
}

impl DB {
    pub fn create(path: PathBuf) -> Result<Self, Error> {
        Self::with_time_step(path, Duration::from_secs_f64(1.0 / 120.0))
    }

    pub fn with_time_step(path: PathBuf, time_step: Duration) -> Result<Self, Error> {
        info!(?path, "created db");

        let assets = Assets::open(path.join("assets"))?;
        std::fs::create_dir_all(path.join("entity_metadata"))?;

        let default_stream_time_step = AtomicU64::new(time_step.as_nanos() as u64);
        let time_step = AtomicU64::new(time_step.as_nanos() as u64);
        let db = DB {
            path,
            vtable_registry: Default::default(),
            components: Default::default(),
            vtable_gen: AtomicCell::new(0),
            streams: Default::default(),
            entity_metadata: Default::default(),
            recording_cell: PlayingCell::new(true),
            assets,
            time_step,
            default_stream_time_step,
            last_updated: AtomicCell::new(Timestamp(i64::MIN)),
            earliest_timestamp: Timestamp::now(),
        };
        db.init_globals()?;

        db.save_db_state()?;
        Ok(db)
    }

    fn init_globals(&self) -> Result<(), Error> {
        let mut sink = DBSink(self);
        let arr = nox::array![0u64];
        let globals_id = EntityId(0);
        sink.apply_value(
            impeller2_wkt::Tick::COMPONENT_ID,
            globals_id,
            ComponentView::U64(arr.view()),
            None,
        );
        self.set_entity_metadata(EntityMetadata {
            entity_id: globals_id,
            name: "Globals".to_string(),
            metadata: Default::default(),
        })?;
        self.set_component_metadata(ComponentMetadata {
            component_id: impeller2_wkt::Tick::COMPONENT_ID,
            name: "Tick".to_string(),
            metadata: [("element_names".to_string(), "tick".to_string())]
                .into_iter()
                .collect(),
            asset: false,
        })?;
        Ok(())
    }

    fn set_entity_metadata(&self, metadata: EntityMetadata) -> Result<(), Error> {
        info!(entity.name = ?metadata.name, entity.id = ?metadata.entity_id.0, "setting entity metadata");
        let entity_metadata_path = self.path.join("entity_metadata");
        std::fs::create_dir_all(&entity_metadata_path)?;
        metadata.write(entity_metadata_path.join(metadata.entity_id.to_string()))?;
        self.entity_metadata.insert(metadata.entity_id, metadata);
        Ok(())
    }

    fn set_component_metadata(&self, metadata: ComponentMetadata) -> Result<(), Error> {
        let component_id = metadata.component_id;
        let Some(component) = self.components.get(&component_id) else {
            warn!(component.id = ?component_id, "component not found");
            return Ok(());
        };
        info!(component.name= ?metadata.name, component.id = ?component_id.0, "setting component metadata");
        let component_metadata_path = self.path.join(component_id.to_string());
        std::fs::create_dir_all(&component_metadata_path)?;
        metadata.write(component_metadata_path.join("metadata"))?;
        component.metadata.store(Arc::new(metadata));
        Ok(())
    }

    fn db_settings(&self) -> DbSettings {
        DbSettings {
            recording: self.recording_cell.is_playing(),
            time_step: Duration::from_nanos(self.time_step.load(atomic::Ordering::SeqCst)),
            default_stream_time_step: Duration::from_nanos(
                self.default_stream_time_step.load(atomic::Ordering::SeqCst),
            ),
        }
    }

    pub fn save_db_state(&self) -> Result<(), Error> {
        let db_state = self.db_settings();
        db_state.write(self.path.join("db_state"))
    }

    pub fn open(path: PathBuf) -> Result<Self, Error> {
        let components = DashMap::new();
        let mut last_updated = i64::MIN;
        let mut start_timestamp = i64::MAX;

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
                if let Some((timestamp, _)) = entity.time_series.latest() {
                    last_updated = timestamp.0.max(last_updated);
                };
                start_timestamp = start_timestamp.min(entity.time_series.start_timestamp().0);
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
        info!(db.path = ?path, "opened db");
        let assets = Assets::open(path.join("assets"))?;
        let db_state = DbSettings::read(path.join("db_state"))?;
        Ok(DB {
            path,
            vtable_gen: AtomicCell::new(0),
            components,
            streams: Default::default(),
            entity_metadata,
            assets,
            recording_cell: PlayingCell::new(db_state.recording),
            time_step: AtomicU64::new(db_state.time_step.as_nanos() as u64),
            default_stream_time_step: AtomicU64::new(
                db_state.default_stream_time_step.as_nanos() as u64
            ),
            last_updated: AtomicCell::new(Timestamp(last_updated)),
            earliest_timestamp: Timestamp(start_timestamp),
            vtable_registry: Default::default(),
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
            name: component_id.to_string(),
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

pub struct Entity {
    pub entity_id: EntityId,
    pub time_series: TimeSeries,
    pub writer: SyncMutex<TimeSeriesWriter>,
    pub schema: ComponentSchema,
}

impl Entity {
    pub fn create(
        path: impl AsRef<Path>,
        entity_id: EntityId,
        schema: ComponentSchema,
        start_timestamp: Timestamp,
    ) -> Result<Self, Error> {
        let (time_series, writer) =
            TimeSeries::create(path, start_timestamp, schema.size() as u64)?;
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
        let timestamp = vtable.timestamp()?;
        vtable.entity_with_timestamp(
            self.entity_id,
            &[(
                self.schema.component_id,
                self.schema.prim_type,
                &self.schema.shape,
            )],
            Some(timestamp),
        )?;
        Ok(())
    }

    fn get_nearest(&self, timestamp: Timestamp) -> Option<(Timestamp, &[u8])> {
        self.time_series.get_nearest(timestamp)
    }

    fn get_range(&self, range: Range<Timestamp>) -> Option<(&[Timestamp], &[u8])> {
        self.time_series.get_range(range)
    }
}

struct DBSink<'a>(&'a DB);

impl Decomponentize for DBSink<'_> {
    fn apply_value(
        &mut self,
        component_id: impeller2::types::ComponentId,
        entity_id: EntityId,
        value: impeller2::types::ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) {
        let timestamp = timestamp.unwrap_or_else(Timestamp::now);
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
            Entity::create(path, entity_id, component.schema.clone(), timestamp)
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
        if let Err(err) = writer.push_with_buf(timestamp, value_buf.len(), |buf| {
            buf.copy_from_slice(value_buf);
        }) {
            warn!(?err, "failed to write head value");
        }
        self.0.last_updated.update_max(timestamp);
    }
}

pub struct Server {
    pub listener: TcpListener,
    pub db: Arc<DB>,
}

impl Server {
    pub fn new(path: impl AsRef<Path>, addr: SocketAddr) -> Result<Server, Error> {
        info!(?addr, "listening");
        let listener = TcpListener::bind(addr)?;
        let path = path.as_ref().to_path_buf();
        let db = if path.exists() && path.join("entity_metadata").exists() {
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
        loop {
            let stream = listener.accept().await?;
            let conn_db = db.clone();
            stellarator::struc_con::stellar(move || handle_conn(stream, conn_db));
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
        buf = pkt.into_buf().into_inner();
    }
}

async fn handle_packet(
    pkt: &Packet<Slice<Vec<u8>>>,
    db: &Arc<DB>,
    tx: &Arc<Mutex<impeller2_stella::PacketSink<OwnedWriter<TcpStream>>>>,
) -> Result<(), Error> {
    trace!(?pkt, "handling pkt");
    match &pkt {
        Packet::Msg(m) if m.id == VTableMsg::ID => {
            let vtable = m.parse::<VTableMsg>()?;
            info!(id = ?vtable.id, "inserting vtable");
            let mut registry = db.vtable_registry.write().await;
            for (component_id, prim_ty, shape) in vtable.vtable.component_iter() {
                db.components.entry(component_id).or_try_insert_with(|| {
                    Component::try_create(component_id, prim_ty, shape, &db.path)
                })?;
                db.vtable_gen.fetch_add(1, atomic::Ordering::SeqCst);
            }
            registry.map.insert(vtable.id, vtable.vtable);
        }
        Packet::Msg(m) if m.id == Stream::ID => {
            let stream = m.parse::<Stream>()?;
            let stream_id = stream.id;
            match stream.behavior {
                StreamBehavior::RealTime => {
                    let tx = tx.clone();
                    let db = db.clone();
                    stellarator::spawn(async move {
                        match handle_real_time_stream(tx, stream_id, stream.filter, db).await {
                            Ok(_) => {}
                            Err(err) if err.is_stream_closed() => {}
                            Err(err) => {
                                warn!(?err, "error streaming data");
                            }
                        }
                    });
                }
                StreamBehavior::FixedRate(fixed_rate) => {
                    let state = Arc::new(FixedRateStreamState::from_state(
                        stream.id,
                        fixed_rate.timestep.unwrap_or_else(|| {
                            Duration::from_nanos(
                                db.default_stream_time_step.load(atomic::Ordering::Relaxed),
                            )
                        }),
                        match fixed_rate.initial_timestamp {
                            InitialTimestamp::Earliest => db.earliest_timestamp,
                            InitialTimestamp::Latest => db.last_updated.latest(),
                            InitialTimestamp::Manual(timestamp) => timestamp,
                        },
                        stream.filter,
                    ));
                    debug!(stream.id = ?stream_id, "inserting stream");
                    db.streams.insert(stream_id, state.clone());
                    let db = db.clone();
                    let tx = tx.clone();
                    stellarator::spawn(async move {
                        match handle_fixed_stream(tx, state, db).await {
                            Ok(_) => {}
                            Err(err) if err.is_stream_closed() => {}
                            Err(err) => {
                                warn!(?err, "error streaming data");
                            }
                        }
                    });
                }
            }
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
            if let Some(timestamp) = set_stream_state.timestamp {
                state.set_timestamp(timestamp);
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
                    let Some((timestamps, data)) = entity.get_range(get_time_series.range) else {
                        return Ok(());
                    };
                    let size = component.schema.size();
                    let (timestamps, data) = if let Some(limit) = get_time_series.limit {
                        let len = timestamps.len().min(limit);
                        (&timestamps[..len], &data[..len * size])
                    } else {
                        (timestamps, data)
                    };
                    let mut pkt = LenPacket::time_series(
                        get_time_series.id,
                        data.len() + timestamps.as_bytes().len() + 8,
                    );
                    pkt.extend_from_slice(&(timestamps.len() as u64).to_le_bytes());
                    pkt.extend_from_slice(timestamps.as_bytes());
                    pkt.extend_from_slice(data);
                    pkt
                };
                let tx = tx.lock().await;
                tx.send(pkt).await.0?;
                Ok::<_, Error>(())
            });
        }
        Packet::Msg(m) if m.id == SetComponentMetadata::ID => {
            let SetComponentMetadata(metadata) = m.parse::<SetComponentMetadata>()?;
            db.set_component_metadata(metadata)?;
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
            let SetEntityMetadata(metadata) = m.parse::<SetEntityMetadata>()?;
            db.set_entity_metadata(metadata)?;
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
            let set_asset = m.parse::<SetAsset<'_>>()?;
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
        Packet::Msg(m) if m.id == DumpSchema::ID => {
            let schemas = db
                .components
                .iter()
                .map(|c| c.schema.to_schema())
                .collect::<Vec<_>>();
            let msg = DumpSchemaResp { schemas };

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
        Packet::Msg(m) if m.id == SubscribeLastUpdated::ID => {
            let tx = tx.clone();
            let db = db.clone();
            stellarator::spawn(async move {
                loop {
                    let last_updated = db.last_updated.latest();
                    {
                        let tx = tx.lock().await;
                        match tx
                            .send(LastUpdated(last_updated).to_len_packet())
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
                    db.last_updated.wait_for(|time| time > last_updated).await;
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
        Packet::Msg(m) if m.id == GetEarliestTimestamp::ID => {
            let tx = tx.lock().await;
            tx.send(EarliestTimestamp(db.earliest_timestamp).to_len_packet())
                .await
                .0?;
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

pub struct FixedRateStreamState {
    stream_id: StreamId,
    time_step: AtomicU64,
    is_scrubbed: AtomicBool,
    current_tick: AtomicI64,
    playing_cell: PlayingCell,
    filter: StreamFilter,
}

impl FixedRateStreamState {
    fn from_state(
        stream_id: StreamId,
        time_step: Duration,
        current_tick: Timestamp,
        filter: StreamFilter,
    ) -> FixedRateStreamState {
        FixedRateStreamState {
            stream_id,
            time_step: AtomicU64::new(time_step.as_nanos() as u64),
            is_scrubbed: AtomicBool::new(false),
            current_tick: AtomicI64::new(current_tick.0),
            playing_cell: PlayingCell::new(true),
            filter,
        }
    }

    fn is_scrubbed(&self) -> bool {
        self.is_scrubbed.swap(false, atomic::Ordering::Relaxed)
    }

    fn set_timestamp(&self, timestamp: Timestamp) {
        self.is_scrubbed.store(true, atomic::Ordering::SeqCst);
        self.current_tick
            .store(timestamp.0, atomic::Ordering::SeqCst);
        self.playing_cell.wait_cell.wake_all();
    }

    fn set_time_step(&self, time_step: Duration) {
        self.time_step
            .store(time_step.as_nanos() as u64, atomic::Ordering::SeqCst);
    }

    fn time_step(&self) -> Duration {
        Duration::from_nanos(self.time_step.load(atomic::Ordering::Relaxed))
    }

    fn current_timestamp(&self) -> Timestamp {
        Timestamp(self.current_tick.load(atomic::Ordering::Relaxed))
    }

    fn try_increment_tick(&self, last_timestamp: Timestamp) {
        let time_step = self.time_step();
        let _ = self.current_tick.compare_exchange(
            last_timestamp.0,
            (last_timestamp + time_step).0,
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

async fn handle_real_time_stream<A: AsyncWrite + 'static>(
    stream: Arc<Mutex<PacketSink<A>>>,
    _stream_id: StreamId,
    filter: StreamFilter,
    db: Arc<DB>,
) -> Result<(), Error> {
    let mut visited_ids = HashSet::new();
    loop {
        filter.visit(&db, |component_id, _, entity| {
            if visited_ids.contains(&(component_id, entity.entity_id)) {
                return Ok(());
            }
            visited_ids.insert((component_id, entity.entity_id));
            let stream = stream.clone();
            let entity_id = entity.entity_id;

            let mut vtable = VTableBuilder::default();
            entity.add_to_vtable(&mut vtable)?;
            let vtable = vtable.build();
            let waiter = entity.time_series.waiter();
            let db = db.clone();
            stellarator::spawn(async move {
                match handle_real_time_entity(stream, component_id, entity_id, waiter, vtable, db)
                    .await
                {
                    Ok(_) => {}
                    Err(err) if err.is_stream_closed() => {}
                    Err(err) => warn!(?err, "failed to handle real time stream"),
                }
            });
            Ok(())
        })?;
        db.vtable_gen.wait().await;
    }
}

async fn handle_real_time_entity<A: AsyncWrite>(
    stream: Arc<Mutex<PacketSink<A>>>,
    component_id: ComponentId,
    entity_id: EntityId,
    waiter: Arc<WaitQueue>,
    vtable: VTable<Vec<Entry>, Vec<u8>>,
    db: Arc<DB>,
) -> Result<(), Error> {
    let vtable_id: PacketId = fastrand::u64(..).to_le_bytes()[..3].try_into().unwrap();
    {
        let stream = stream.lock().await;
        stream
            .send(
                VTableMsg {
                    id: vtable_id,
                    vtable,
                }
                .to_len_packet(),
            )
            .await
            .0?;
    }
    let (time_series, prim_type) = {
        let component = db
            .components
            .get(&component_id)
            .ok_or(Error::ComponentNotFound(component_id))?;
        let entity = component
            .value()
            .entities
            .get(&entity_id)
            .ok_or(Error::EntityNotFound(entity_id))?;
        (entity.time_series.clone(), component.schema.prim_type)
    };

    let mut table = LenPacket::table(vtable_id, 2048 - 16);
    loop {
        let _ = waiter.wait().await;
        let Some((&timestamp, buf)) = time_series.latest() else {
            continue;
        };
        table.push_aligned(timestamp);
        table.pad_for_type(prim_type);
        table.extend_from_slice(buf);
        {
            let stream = stream.lock().await;
            rent!(stream.send(table).await, table)?;
        }
        table.clear();
    }
}

async fn handle_fixed_stream<A: AsyncWrite>(
    stream: Arc<Mutex<PacketSink<A>>>,
    state: Arc<FixedRateStreamState>,
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
        let start = Instant::now();
        let current_timestamp = state.current_timestamp();
        let gen = db.vtable_gen.latest();
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
        if let Err(err) = state
            .filter
            .populate_table(&db, &mut table, current_timestamp)
        {
            warn!(?err, "failed to populate table");
        }
        {
            let stream = stream.lock().await;
            stream
                .send(
                    StreamTimestamp {
                        timestamp: current_timestamp,
                        stream_id: state.stream_id,
                    }
                    .to_len_packet(),
                )
                .await
                .0?;
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
        state.try_increment_tick(current_timestamp);
    }
}

pub trait StreamFilterExt {
    fn visit(
        &self,
        db: &DB,
        f: impl for<'a> FnMut(ComponentId, &'a Component, &'a Entity) -> Result<(), Error>,
    ) -> Result<(), Error>;
    fn vtable(&self, db: &DB) -> Result<VTable<Vec<impeller2::table::Entry>, Vec<u8>>, Error>;
    fn populate_table(&self, db: &DB, table: &mut LenPacket, tick: Timestamp) -> Result<(), Error>;
}

impl StreamFilterExt for StreamFilter {
    fn vtable(&self, db: &DB) -> Result<VTable<Vec<impeller2::table::Entry>, Vec<u8>>, Error> {
        let mut vtable = VTableBuilder::default();
        self.visit(db, |_, _, entity| {
            entity.add_to_vtable(&mut vtable)?;
            Ok(())
        })?;
        Ok(vtable.build())
    }

    fn populate_table(
        &self,
        db: &DB,
        table: &mut LenPacket,
        timestamp: Timestamp,
    ) -> Result<(), Error> {
        self.visit(db, |_, component, entity| {
            let tick = entity.time_series.start_timestamp().max(timestamp);
            let Some((timestamp, buf)) = entity.get_nearest(tick) else {
                return Ok(());
            };
            table.push_aligned(timestamp);
            table.pad_for_type(component.schema.prim_type);
            table.extend_from_slice(buf);
            Ok(())
        })
    }

    fn visit(
        &self,
        db: &DB,
        mut f: impl for<'a> FnMut(ComponentId, &'a Component, &'a Entity) -> Result<(), Error>,
    ) -> Result<(), Error> {
        match (self.component_id, self.entity_id) {
            (None, None) => {
                for kv in &db.components {
                    for entity_kv in &kv.value().entities {
                        let entity = entity_kv.value();
                        f(*kv.key(), kv.value(), entity)?;
                    }
                }
                Ok(())
            }
            (None, Some(id)) => {
                for kv in &db.components {
                    let component = kv.value();
                    let Some(entity) = component.entities.get(&id) else {
                        continue;
                    };
                    f(*kv.key(), component, entity.value())?;
                }
                Ok(())
            }
            (Some(id), None) => {
                let component = db.components.get(&id).ok_or(Error::ComponentNotFound(id))?;
                for entity in &component.entities {
                    f(id, component.value(), entity.value())?;
                }
                Ok(())
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
                f(component_id, component.value(), entity.value())
            }
        }
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

    pub fn set_playing(&self, playing: bool) {
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

pub trait AtomicTimestampExt {
    fn update_max(&self, val: Timestamp);
}

impl AtomicTimestampExt for AtomicCell<Timestamp> {
    fn update_max(&self, val: Timestamp) {
        self.value.fetch_max(val.0, atomic::Ordering::AcqRel);
        self.wait_queue.wake_all();
    }
}
