use assets::open_assets;
use datafusion::common::HashSet;
use futures_lite::StreamExt;
use impeller2::registry::VTableRegistry;
use impeller2::types::{PacketHeader, PacketTy};
use impeller2::vtable::builder::{
    OpBuilder, pair, raw_field, raw_table, schema, timestamp, vtable,
};
use impeller2::vtable::{RealizedField, builder};
use impeller2::{
    com_de::Decomponentize,
    registry,
    schema::Schema,
    types::{
        ComponentId, ComponentView, EntityId, IntoLenPacket, LenPacket, Msg, OwnedPacket as Packet,
        PacketId, PrimType, RequestId, Timestamp,
    },
    vtable::VTable,
};
use impeller2_stellar::{PacketSink, PacketStream};
use impeller2_wkt::*;
use msg_log::MsgLog;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::OsStr,
    net::{SocketAddr, ToSocketAddrs},
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        Arc, RwLock,
        atomic::{self, AtomicBool, AtomicI64, AtomicU64},
    },
    time::{Duration, Instant},
};
use stellarator::{
    buf::Slice,
    io::{AsyncRead, AsyncWrite, OwnedReader, OwnedWriter, SplitExt},
    net::{TcpListener, TcpStream, UdpSocket},
    rent,
    struc_con::Joinable,
    sync::{Mutex, WaitQueue},
    util::AtomicCell,
};
use time_series::TimeSeries;
use tracing::{debug, info, trace, warn};
use vtable_stream::handle_vtable_stream;
use zerocopy::IntoBytes;

pub use error::Error;

pub mod append_log;
mod arrow;
mod assets;
pub mod axum;
mod error;
mod msg_log;
pub(crate) mod time_series;
mod vtable_stream;

pub struct DB {
    pub vtable_gen: AtomicCell<u64>,
    state: RwLock<State>,
    pub recording_cell: PlayingCell,

    // metadata
    pub path: PathBuf,
    pub time_step: AtomicU64,
    pub default_stream_time_step: AtomicU64,
    pub last_updated: AtomicCell<Timestamp>,
    pub earliest_timestamp: Timestamp,
}

#[derive(Default)]
pub struct State {
    components: HashMap<(EntityId, ComponentId), Component>,
    component_metadata: HashMap<ComponentId, ComponentMetadata>,
    entity_metadata: HashMap<EntityId, EntityMetadata>,
    assets: HashMap<AssetId, memmap2::Mmap>,

    msg_logs: HashMap<PacketId, MsgLog>,

    vtable_registry: registry::HashMapRegistry,
    streams: HashMap<StreamId, Arc<FixedRateStreamState>>,

    udp_vtable_streams: HashSet<(SocketAddr, [u8; 2])>,
}

impl DB {
    pub fn create(path: PathBuf) -> Result<Self, Error> {
        Self::with_time_step(path, Duration::from_secs_f64(1.0 / 120.0))
    }

    pub fn with_time_step(path: PathBuf, time_step: Duration) -> Result<Self, Error> {
        info!(?path, "created db");

        std::fs::create_dir_all(path.join("entity_metadata"))?;
        let default_stream_time_step = AtomicU64::new(time_step.as_nanos() as u64);
        let time_step = AtomicU64::new(time_step.as_nanos() as u64);
        let state = State {
            assets: open_assets(&path)?,
            ..Default::default()
        };
        let db = DB {
            state: RwLock::new(state),
            recording_cell: PlayingCell::new(true),
            path,
            vtable_gen: AtomicCell::new(0),
            time_step,
            default_stream_time_step,
            last_updated: AtomicCell::new(Timestamp(i64::MIN)),
            earliest_timestamp: Timestamp::now(),
        };
        db.save_db_state()?;
        Ok(db)
    }

    pub fn with_state<O, F: FnOnce(&State) -> O>(&self, f: F) -> O {
        let state = self.state.read().unwrap();
        f(&state)
    }

    pub fn with_state_mut<O, F: FnOnce(&mut State) -> O>(&self, f: F) -> O {
        let mut state = self.state.write().unwrap();
        f(&mut state)
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
        let mut entity_metadata = HashMap::new();
        let mut component_metadata = HashMap::new();
        let mut entity_components = HashMap::new();
        let mut msg_logs = HashMap::new();
        let mut last_updated = i64::MIN;
        let mut start_timestamp = i64::MAX;

        info!("Opening database: {}", path.display());
        for elem in std::fs::read_dir(&path)? {
            let Ok(elem) = elem else { continue };
            let path = elem.path();
            if !path.is_dir()
                || (path.file_name() == Some(OsStr::new("entity_metadata"))
                    || path.file_name() == Some(OsStr::new("assets")))
                || path.file_name() == Some(OsStr::new("msgs"))
            {
                trace!("Skipping non-component directory: {}", path.display());
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
            trace!("Read component metadata for {}", metadata.name);
            component_metadata.insert(component_id, metadata);

            for elem in std::fs::read_dir(&path)? {
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

                trace!("Opening entity file {}", path.display());

                let entity = Component::open(path, component_id, entity_id, schema.clone())?;
                if let Some((timestamp, _)) = entity.time_series.latest() {
                    last_updated = timestamp.0.max(last_updated);
                };
                start_timestamp = start_timestamp.min(entity.time_series.start_timestamp().0);
                entity_components.insert((entity_id, component_id), entity);
            }
        }
        if let Ok(msgs_dir) = std::fs::read_dir(path.join("msgs")) {
            for elem in msgs_dir {
                let Ok(elem) = elem else { continue };

                let path = elem.path();
                let msg_id: u16 = path
                    .file_name()
                    .and_then(|p| p.to_str())
                    .and_then(|p| p.parse().ok())
                    .ok_or(Error::InvalidMsgId)?;
                let msg_log = MsgLog::open(path)?;
                msg_logs.insert(msg_id.to_le_bytes(), msg_log);
            }
        }

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
        let db_state = DbSettings::read(path.join("db_state"))?;
        let state = State {
            components: entity_components,
            entity_metadata,
            component_metadata,
            assets: open_assets(&path)?,
            msg_logs,
            ..Default::default()
        };
        Ok(DB {
            state: RwLock::new(state),
            path,
            vtable_gen: AtomicCell::new(0),
            recording_cell: PlayingCell::new(db_state.recording),
            time_step: AtomicU64::new(db_state.time_step.as_nanos() as u64),
            default_stream_time_step: AtomicU64::new(
                db_state.default_stream_time_step.as_nanos() as u64
            ),
            last_updated: AtomicCell::new(Timestamp(last_updated)),
            earliest_timestamp: Timestamp(start_timestamp),
        })
    }

    pub fn time_step(&self) -> Duration {
        Duration::from_nanos(self.time_step.load(atomic::Ordering::Relaxed))
    }

    pub fn insert_vtable(&self, vtable: VTableMsg) -> Result<(), Error> {
        info!(id = ?vtable.id, "inserting vtable");
        self.with_state_mut(|state| {
            for res in vtable.vtable.realize_fields(None) {
                let RealizedField {
                    entity_id,
                    component_id,
                    shape,
                    ty,
                    ..
                } = res?;
                let component_schema = ComponentSchema::new(ty, shape);
                state.insert_component(component_id, component_schema, entity_id, &self.path)?;
                self.vtable_gen.fetch_add(1, atomic::Ordering::SeqCst);
            }
            state.vtable_registry.map.insert(vtable.id, vtable.vtable);
            Ok::<_, Error>(())
        })?;
        Ok(())
    }

    pub fn insert_asset(&self, id: AssetId, buf: &[u8]) -> Result<(), Error> {
        self.with_state_mut(|state| assets::insert_asset(&self.path, &mut state.assets, id, buf))
    }

    pub fn push_msg(&self, timestamp: Timestamp, id: PacketId, msg: &[u8]) -> Result<(), Error> {
        let exists = self.with_state(|s| {
            if let Some(msg_log) = s.msg_logs.get(&id) {
                msg_log.push(timestamp, msg)?;
                Ok::<_, Error>(true)
            } else {
                Ok(false)
            }
        })?;
        if !exists {
            self.with_state_mut(move |s| {
                let msg_log = s.get_or_insert_msg_log(id, &self.path)?;
                msg_log.push(timestamp, msg)?;
                Ok::<_, Error>(())
            })?;
        }
        Ok(())
    }
}

impl State {
    pub fn insert_component(
        &mut self,
        component_id: ComponentId,
        schema: ComponentSchema,
        entity_id: EntityId,
        db_path: &Path,
    ) -> Result<(), Error> {
        if let Some(existing_component) = self.components.get(&(entity_id, component_id)) {
            if existing_component.schema != schema {
                warn!( ?existing_component.schema, new_component.schema = ?schema,
                       ?existing_component.component_id,
                       ?existing_component.entity_id,
                      "schema mismatch");
                return Err(Error::SchemaMismatch);
            }
            return Ok(());
        }
        info!(entity.id = ?entity_id.0, component.id = ?component_id.0, "inserting");
        let component_metadata = ComponentMetadata {
            component_id,
            name: component_id.to_string(),
            metadata: Default::default(),
            asset: false,
        };
        let entity_metadata = EntityMetadata {
            entity_id,
            name: entity_id.to_string(),
            metadata: Default::default(),
        };
        let entity = Component::create(db_path, component_id, entity_id, schema, Timestamp::now())?;
        if !self.entity_metadata.contains_key(&entity_id) {
            self.set_entity_metadata(entity_metadata.clone(), db_path)?;
        }
        if !self.component_metadata.contains_key(&component_id) {
            self.set_component_metadata(component_metadata, db_path)?;
        }
        self.components.insert((entity_id, component_id), entity);
        Ok(())
    }

    pub fn get_component_metadata(&self, component_id: ComponentId) -> Option<&ComponentMetadata> {
        self.component_metadata.get(&component_id)
    }

    pub fn get_entity_metadata(&self, entity_id: EntityId) -> Option<&EntityMetadata> {
        self.entity_metadata.get(&entity_id)
    }

    pub fn get_component(
        &self,
        entity_id: EntityId,
        component_id: ComponentId,
    ) -> Option<&Component> {
        self.components.get(&(entity_id, component_id))
    }

    pub fn set_entity_metadata(
        &mut self,
        metadata: EntityMetadata,
        db_path: &Path,
    ) -> Result<(), Error> {
        let entity_metadata_path = db_path.join("entity_metadata");
        std::fs::create_dir_all(&entity_metadata_path)?;
        let entity_metadata_path = entity_metadata_path.join(metadata.entity_id.to_string());
        if entity_metadata_path.exists() && EntityMetadata::read(&entity_metadata_path)? == metadata
        {
            return Ok(());
        }
        info!(entity.name = ?metadata.name, entity.id = ?metadata.entity_id.0, "setting entity metadata");
        metadata.write(entity_metadata_path)?;
        self.entity_metadata.insert(metadata.entity_id, metadata);
        Ok(())
    }

    pub fn set_component_metadata(
        &mut self,
        metadata: ComponentMetadata,
        db_path: &Path,
    ) -> Result<(), Error> {
        let component_metadata_path = db_path.join(metadata.component_id.to_string());
        std::fs::create_dir_all(&component_metadata_path)?;
        let component_metadata_path = component_metadata_path.join("metadata");
        if component_metadata_path.exists()
            && ComponentMetadata::read(&component_metadata_path)? == metadata
        {
            return Ok(());
        }
        info!(component.name= ?metadata.name, component.id = ?metadata.component_id.0, "setting component metadata");
        metadata.write(component_metadata_path)?;
        self.component_metadata
            .insert(metadata.component_id, metadata);
        Ok(())
    }

    pub fn get_or_insert_msg_log(
        &mut self,
        id: PacketId,
        db_path: &Path,
    ) -> Result<&mut MsgLog, Error> {
        Ok(match self.msg_logs.entry(id) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let msg_log = MsgLog::create(
                    db_path
                        .join("msgs")
                        .join(u16::from_le_bytes(id).to_string()),
                )?;
                entry.insert(msg_log)
            }
        })
    }

    pub fn set_msg_metadata(
        &mut self,
        id: PacketId,
        metadata: MsgMetadata,
        db_path: &Path,
    ) -> Result<(), Error> {
        let msg_log = self.get_or_insert_msg_log(id, db_path)?;
        msg_log.set_metadata(metadata)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComponentSchema {
    pub prim_type: PrimType,
    pub dim: SmallVec<[usize; 4]>,
}

impl Serialize for ComponentSchema {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let schema = self.to_schema();
        schema.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ComponentSchema {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let schema = Schema::<Vec<u64>>::deserialize(deserializer)?;
        Ok(schema.into())
    }
}

impl ComponentSchema {
    pub fn new(prim_type: PrimType, shape: &[usize]) -> Self {
        let dim = shape.into();
        ComponentSchema { prim_type, dim }
    }

    pub fn size(&self) -> usize {
        self.dim.iter().product::<usize>() * self.prim_type.size()
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
        Schema::new(self.prim_type, self.shape()).expect("failed to create shape")
    }

    pub fn shape(&self) -> SmallVec<[u64; 4]> {
        self.dim.iter().map(|&x| x as u64).collect()
    }

    pub fn parse_value<'a>(&'a self, buf: &'a [u8]) -> Result<(usize, ComponentView<'a>), Error> {
        let size = self.size();
        let buf = buf
            .get(..size)
            .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?;
        let dim = &self.dim;
        let view = match self.prim_type {
            PrimType::U8 => ComponentView::U8(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::U16 => ComponentView::U16(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::U32 => ComponentView::U32(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::U64 => ComponentView::U64(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I8 => ComponentView::I8(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I16 => ComponentView::I16(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I32 => ComponentView::I32(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I64 => ComponentView::I64(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::Bool => ComponentView::Bool(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::F32 => ComponentView::F32(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::F64 => ComponentView::F64(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
        };
        Ok((size, view))
    }
}

impl From<Schema<Vec<u64>>> for ComponentSchema {
    fn from(value: Schema<Vec<u64>>) -> Self {
        let prim_type = value.prim_type();
        ComponentSchema {
            prim_type,
            dim: value.shape().into(),
        }
    }
}

impl From<ComponentSchema> for Schema<Vec<u64>> {
    fn from(value: ComponentSchema) -> Self {
        value.to_schema()
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
impl MetadataExt for MsgMetadata {}

#[derive(Clone)]
pub struct Component {
    pub component_id: ComponentId,
    pub entity_id: EntityId,
    pub time_series: TimeSeries,
    pub schema: ComponentSchema,
}

impl Component {
    pub fn create(
        db_path: &Path,
        component_id: ComponentId,
        entity_id: EntityId,
        schema: ComponentSchema,
        start_timestamp: Timestamp,
    ) -> Result<Self, Error> {
        let component_path = db_path.join(component_id.to_string());
        let entity_path = component_path.join(entity_id.to_string());
        std::fs::create_dir_all(&entity_path)?;
        let component_schema_path = component_path.join("schema");
        if !component_schema_path.exists() {
            schema.write(component_schema_path)?;
        }
        let time_series = TimeSeries::create(entity_path, start_timestamp, schema.size() as u64)?;
        Ok(Component {
            component_id,
            entity_id,
            time_series,
            schema,
        })
    }

    pub fn open(
        path: impl AsRef<Path>,
        component_id: ComponentId,
        entity_id: EntityId,
        schema: ComponentSchema,
    ) -> Result<Self, Error> {
        let time_series = TimeSeries::open(path)?;
        Ok(Component {
            component_id,
            entity_id,
            time_series,
            schema,
        })
    }

    fn as_vtable_op(&self) -> Arc<OpBuilder> {
        schema(
            self.schema.prim_type,
            &self.schema.shape(),
            pair(self.entity_id, self.component_id),
        )
    }

    fn get_nearest(&self, timestamp: Timestamp) -> Option<(Timestamp, &[u8])> {
        self.time_series.get_nearest(timestamp)
    }

    fn get_range(&self, range: Range<Timestamp>) -> Option<(&[Timestamp], &[u8])> {
        self.time_series.get_range(range)
    }
}

struct DBSink<'a> {
    components: &'a HashMap<(EntityId, ComponentId), Component>,
    last_updated: &'a AtomicCell<Timestamp>,
    sunk_new_time_series: bool,
    table_received: Timestamp,
}

impl Decomponentize for DBSink<'_> {
    type Error = Error;
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        entity_id: EntityId,
        value: impeller2::types::ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) -> Result<(), Error> {
        let timestamp = timestamp.unwrap_or(self.table_received);
        let value_buf = value.as_bytes();
        let Some(component) = self.components.get(&(entity_id, component_id)) else {
            return Err(Error::ComponentNotFound(component_id));
        };
        let time_series_empty = component.time_series.index().is_empty();
        component.time_series.push_buf(timestamp, value_buf)?;
        if time_series_empty {
            debug!(
                "sunk new time series for component {} entity {}",
                component_id, entity_id
            );
            self.sunk_new_time_series = true;
        }
        self.last_updated.update_max(timestamp);
        Ok(())
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
        Server::from_listener(listener, path)
    }

    pub fn from_listener(listener: TcpListener, path: impl AsRef<Path>) -> Result<Server, Error> {
        let path = path.as_ref().to_path_buf();
        let db = if path.exists() && path.join("entity_metadata").exists() {
            DB::open(path)?
        } else {
            DB::create(path)?
        };
        let db = Arc::new(db);
        Ok(Server { listener, db })
    }

    pub async fn run(self) -> Result<(), Error> {
        let Self { listener, db } = self;
        let addr = listener.local_addr()?;
        let udp_db = db.clone();
        stellarator::struc_con::stellar(move || Self::handle_udp(addr, udp_db));
        loop {
            let stream = listener.accept().await?;
            let conn_db = db.clone();
            stellarator::struc_con::stellar(move || handle_conn(stream, conn_db));
        }
    }

    pub async fn handle_udp(addr: SocketAddr, db: Arc<DB>) -> Result<(), Error> {
        let socket = UdpSocket::bind(addr)?;
        let (rx, tx) = socket.split();
        let rx = PacketStream::new(rx);
        let tx = Arc::new(Mutex::new(PacketSink::new(tx)));
        handle_conn_inner(tx, rx, db).await?;
        Ok(())
    }
}

pub async fn handle_conn(stream: TcpStream, db: Arc<DB>) {
    let (rx, tx) = stream.split();
    let rx = PacketStream::new(rx);
    let tx = Arc::new(Mutex::new(PacketSink::new(tx)));
    match handle_conn_inner(tx, rx, db).await {
        Ok(_) => {}
        Err(err) if err.is_stream_closed() => {}
        Err(err) => {
            warn!(?err, "error handling stream")
        }
    }
}

async fn handle_conn_inner<A: AsyncRead + AsyncWrite + 'static>(
    mut tx: Arc<Mutex<PacketSink<OwnedWriter<A>>>>,
    mut rx: PacketStream<OwnedReader<A>>,
    db: Arc<DB>,
) -> Result<(), Error> {
    let mut buf = vec![0u8; 1024 * 64];
    let mut resp_pkt = LenPacket::new(PacketTy::Msg, [0, 0], 1024 * 64);
    loop {
        let pkt = rx.next(buf).await?;
        let req_id = pkt.req_id();
        let mut pkt_tx = PacketTx {
            req_id,
            tx,
            pkt: Some(resp_pkt),
        };
        let result = handle_packet(&pkt, &db, &mut pkt_tx).await;
        buf = pkt.into_buf().into_inner();
        match result {
            Ok(_) => {}
            Err(err) if err.is_stream_closed() => {}
            Err(err) => {
                warn!(?err, "error handling packet");
                if let Err(err) = pkt_tx
                    .send_msg(&ErrorResponse {
                        description: err.to_string(),
                    })
                    .await
                {
                    warn!(?err, "error sending err resp");
                }
            }
        }
        resp_pkt = pkt_tx.pkt.expect("len pkt taken and not given back");
        tx = pkt_tx.tx;
    }
}

pub struct PacketTx<A: AsyncWrite + 'static> {
    req_id: RequestId,
    tx: Arc<Mutex<PacketSink<OwnedWriter<A>>>>,
    pkt: Option<LenPacket>,
}

impl<A: AsyncWrite + 'static> Clone for PacketTx<A> {
    fn clone(&self) -> Self {
        Self {
            req_id: self.req_id,
            tx: self.tx.clone(),
            pkt: self.pkt.clone(),
        }
    }
}

impl<A: AsyncWrite + 'static> PacketTx<A> {
    pub async fn send_msg<M: Msg>(&mut self, msg: &M) -> Result<(), Error> {
        let req_id = self.req_id;
        self.send_with_builder(|pkt| {
            let header = PacketHeader {
                packet_ty: impeller2::types::PacketTy::Msg,
                id: M::ID,
                req_id,
            };
            pkt.as_mut_packet().header = header;
            pkt.clear();
            postcard::serialize_with_flavor(&msg, pkt).map_err(Error::from)
        })
        .await
    }

    pub async fn send_with_builder(
        &mut self,
        builder: impl FnOnce(&mut LenPacket) -> Result<(), Error> + '_,
    ) -> Result<(), Error> {
        let mut pkt = self.pkt.take().expect("missing len pkt");
        pkt.clear();
        if let Err(err) = builder(&mut pkt) {
            self.pkt = Some(pkt);
            return Err(err);
        }
        pkt.as_mut_packet().header.req_id = self.req_id;
        let tx = self.tx.lock().await;
        let res = rent!(tx.send(pkt).await, pkt);
        self.pkt = Some(pkt);
        res.map_err(Error::from)
    }

    pub async fn send_time_series(
        &mut self,
        id: PacketId,
        timestamps: &[Timestamp],
        data: &[u8],
    ) -> Result<(), Error> {
        let req_id = self.req_id;
        self.send_with_builder(|pkt| {
            let header = PacketHeader {
                packet_ty: impeller2::types::PacketTy::TimeSeries,
                id,
                req_id,
            };
            pkt.as_mut_packet().header = header;
            pkt.clear();
            pkt.extend_from_slice(&(timestamps.len() as u64).to_le_bytes());
            pkt.extend_from_slice(timestamps.as_bytes());
            pkt.extend_from_slice(data);
            Ok(())
        })
        .await
    }
}

async fn handle_packet<A: AsyncWrite + 'static>(
    pkt: &Packet<Slice<Vec<u8>>>,
    db: &Arc<DB>,
    tx: &mut PacketTx<A>,
) -> Result<(), Error> {
    trace!(?pkt, "handling pkt");
    match &pkt {
        Packet::Msg(m) if m.id == VTableMsg::ID => {
            let vtable = m.parse::<VTableMsg>()?;
            db.insert_vtable(vtable)?;
        }
        Packet::Msg(m) if m.id == UdpUnicast::ID => {
            let udp_broadcast = m.parse::<UdpUnicast>()?;
            let db = db.clone();
            let addr = udp_broadcast
                .addr
                .to_socket_addrs()?
                .next()
                .ok_or_else(|| {
                    Error::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "missing socket ip",
                    ))
                })?;
            handle_unicast_stream(addr, udp_broadcast.stream, db);
        }
        Packet::Msg(m) if m.id == Stream::ID => {
            let stream = m.parse::<Stream>()?;
            let tx = tx.tx.clone();
            let db = db.clone();
            handle_stream(tx, stream, db, m.req_id);
        }
        Packet::Msg(m) if m.id == SetStreamState::ID => {
            let set_stream_state = m.parse::<SetStreamState>()?;
            let stream_id = set_stream_state.id;
            db.with_state(|s| {
                let Some(state) = s.streams.get(&stream_id) else {
                    return Err(Error::StreamNotFound(stream_id));
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
                if let Some(frequency) = set_stream_state.frequency {
                    state.set_frequency(frequency);
                }
                Ok(())
            })?;
        }
        Packet::Msg(m) if m.id == GetSchema::ID => {
            let get_schema = m.parse::<GetSchema>()?;
            let schema = db.with_state(|state| {
                state
                    .components
                    .iter()
                    .filter(|((_, component_id), _)| component_id == &get_schema.component_id)
                    .map(|(_, component)| component.schema.to_schema())
                    .next()
                    .ok_or(Error::ComponentNotFound(get_schema.component_id))
            })?;
            tx.send_msg(&SchemaMsg(schema)).await?;
        }
        Packet::Msg(m) if m.id == GetTimeSeries::ID => {
            let get_time_series = m.parse::<GetTimeSeries>()?;
            debug!(msg = ?get_time_series, "get time series");
            let db = db.clone();
            let GetTimeSeries {
                component_id,
                entity_id,
                range,
                limit,
                id,
            } = get_time_series;
            let entity = db.with_state(|state| {
                let Some(component) = state.components.get(&(entity_id, component_id)) else {
                    if state.component_metadata.contains_key(&component_id) {
                        return Err(Error::EntityNotFound(entity_id));
                    } else {
                        return Err(Error::ComponentNotFound(component_id));
                    }
                };
                Ok(component.clone())
            })?;
            let Some((timestamps, data)) = entity.get_range(range) else {
                return Err(Error::TimeRangeOutOfBounds);
            };
            let size = entity.schema.size();
            let (timestamps, data) = if let Some(limit) = limit {
                let len = timestamps.len().min(limit);
                (&timestamps[..len], &data[..len * size])
            } else {
                (timestamps, data)
            };
            tx.send_time_series(id, timestamps, data).await?;
        }
        Packet::Msg(m) if m.id == SetComponentMetadata::ID => {
            let SetComponentMetadata(metadata) = m.parse::<SetComponentMetadata>()?;
            db.with_state_mut(|state| state.set_component_metadata(metadata, &db.path))?;
        }
        Packet::Msg(m) if m.id == GetComponentMetadata::ID => {
            let GetComponentMetadata { component_id } = m.parse::<GetComponentMetadata>()?;

            tx.send_with_builder(|pkt| {
                let header = PacketHeader {
                    packet_ty: impeller2::types::PacketTy::Msg,
                    id: ComponentMetadata::ID,
                    req_id: m.req_id,
                };
                pkt.as_mut_packet().header = header;
                pkt.clear();
                db.with_state(|state| {
                    let Some(metadata) = state.component_metadata.get(&component_id) else {
                        return Err(Error::ComponentNotFound(component_id));
                    };
                    postcard::serialize_with_flavor(&metadata, pkt).map_err(Error::from)
                })
            })
            .await?;
        }

        Packet::Msg(m) if m.id == SetEntityMetadata::ID => {
            let SetEntityMetadata(metadata) = m.parse::<SetEntityMetadata>()?;
            db.with_state_mut(|state| state.set_entity_metadata(metadata, &db.path))?;
        }

        Packet::Msg(m) if m.id == GetEntityMetadata::ID => {
            let GetEntityMetadata { entity_id } = m.parse::<GetEntityMetadata>()?;

            tx.send_with_builder(|pkt| {
                let header = PacketHeader {
                    packet_ty: impeller2::types::PacketTy::Msg,
                    id: EntityMetadata::ID,
                    req_id: m.req_id,
                };
                pkt.as_mut_packet().header = header;
                pkt.clear();
                db.with_state(|state| {
                    let Some(metadata) = state.entity_metadata.get(&entity_id) else {
                        return Err(Error::EntityNotFound(entity_id));
                    };
                    postcard::serialize_with_flavor(&metadata, pkt).map_err(Error::from)
                })
            })
            .await?;
        }

        Packet::Msg(m) if m.id == SetAsset::ID => {
            let set_asset = m.parse::<SetAsset<'_>>()?;
            db.insert_asset(set_asset.id, &set_asset.buf)?;
        }
        Packet::Msg(m) if m.id == GetAsset::ID => {
            let GetAsset { id } = m.parse::<GetAsset>()?;

            tx.send_with_builder(|pkt| {
                let header = PacketHeader {
                    packet_ty: impeller2::types::PacketTy::Msg,
                    id: Asset::ID,
                    req_id: m.req_id,
                };
                pkt.as_mut_packet().header = header;
                pkt.clear();
                db.with_state(|state| {
                    let Some(mmap) = state.assets.get(&id) else {
                        return Err(Error::AssetNotFound(id));
                    };
                    let asset = Asset {
                        id,
                        buf: Cow::Borrowed(&mmap[..]),
                    };
                    postcard::serialize_with_flavor(&asset, pkt).map_err(Error::from)
                })
            })
            .await?;
        }
        Packet::Msg(m) if m.id == DumpMetadata::ID => {
            let msg = db.with_state(|state| {
                let component_metadata = state.component_metadata.values().cloned().collect();
                let entity_metadata = state.entity_metadata.values().cloned().collect();
                let msg_metadata = state
                    .msg_logs
                    .values()
                    .flat_map(|m| m.metadata())
                    .cloned()
                    .collect();
                DumpMetadataResp {
                    component_metadata,
                    entity_metadata,
                    msg_metadata,
                }
            });
            tx.send_msg(&msg).await?;
        }
        Packet::Msg(m) if m.id == DumpSchema::ID => {
            let msg = db.with_state(|state| {
                let schemas = state
                    .components
                    .values()
                    .map(|c| (c.component_id, c.schema.to_schema()))
                    .collect();
                DumpSchemaResp { schemas }
            });

            tx.send_msg(&msg).await?;
        }

        Packet::Msg(m) if m.id == DumpAssets::ID => {
            // TODO(sphw): Implement dump assets in a no-alloc fashion
            let packets = db.with_state(|state| {
                state
                    .assets
                    .iter()
                    .map(|(id, mmap)| Asset {
                        id: *id,
                        buf: Cow::Owned(mmap[..].to_vec()),
                    })
                    .collect::<Vec<_>>()
            });

            for packet in packets {
                tx.send_msg(&packet).await?;
            }
        }
        Packet::Msg(m) if m.id == SubscribeLastUpdated::ID => {
            let mut tx = tx.clone();
            let db = db.clone();
            stellarator::spawn(async move {
                loop {
                    let last_updated = db.last_updated.latest();
                    {
                        match tx.send_msg(&LastUpdated(last_updated)).await {
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
            tx.send_msg(&db.db_settings()).await?;
        }
        Packet::Msg(m) if m.id == GetEarliestTimestamp::ID => {
            tx.send_msg(&EarliestTimestamp(db.earliest_timestamp))
                .await?;
        }
        Packet::Msg(m) if m.id == GetDbSettings::ID => {
            let settings = db.db_settings();
            tx.send_msg(&settings).await?;
        }
        Packet::Table(table) => {
            trace!(table.len = table.buf.len(), "sinking table");
            db.with_state(|state| {
                let mut sink = DBSink {
                    components: &state.components,
                    last_updated: &db.last_updated,
                    sunk_new_time_series: false,
                    table_received: Timestamp::now(),
                };
                table
                    .sink(&state.vtable_registry, &mut sink)?
                    .inspect_err(|err| {
                        dbg!(err);
                    })?;
                if sink.sunk_new_time_series {
                    db.vtable_gen.fetch_add(1, atomic::Ordering::SeqCst);
                }
                Ok::<_, Error>(())
            })?;
        }

        Packet::Msg(m) if m.id == GetDbSettings::ID => {
            let settings = db.db_settings();
            tx.send_msg(&settings).await?;
        }
        Packet::Msg(m) if m.id == SQLQuery::ID => {
            let SQLQuery(query) = m.parse::<SQLQuery>()?;
            let db = db.clone();
            let (tokio_tx, rx) = thingbuf::mpsc::channel::<Vec<u8>>(4);
            let res = stellarator::struc_con::tokio(move |_| async move {
                let mut ctx = db.as_session_context()?;
                db.insert_views(&mut ctx).await?;
                let df = ctx.sql(&query).await?;
                let mut stream = df.execute_stream().await?;

                while let Some(batch) = stream.next().await {
                    let batch = batch?;
                    let mut buf = vec![];
                    let mut writer =
                        ::arrow::ipc::writer::StreamWriter::try_new(&mut buf, batch.schema_ref())?;
                    writer.write(&batch)?;
                    writer.finish()?;
                    let _ = tokio_tx.send(buf).await;
                }
                Ok::<_, Error>(())
            })
            .join();
            while let Some(batch) = rx.recv().await {
                tx.send_msg(&ArrowIPC {
                    batch: Some(Cow::Owned(batch)),
                })
                .await?;
            }
            res.await??;
            tx.send_msg(&ArrowIPC { batch: None }).await?;
        }
        Packet::Msg(m) if m.id == SetMsgMetadata::ID => {
            let SetMsgMetadata { id, metadata } = m.parse::<SetMsgMetadata>()?;
            db.with_state_mut(|s| s.set_msg_metadata(id, metadata, &db.path))?;
        }
        Packet::Msg(m) if m.id == MsgStream::ID => {
            let MsgStream { msg_id } = m.parse::<MsgStream>()?;
            let msg_log =
                db.with_state_mut(|s| s.get_or_insert_msg_log(msg_id, &db.path).cloned())?;
            let req_id = m.req_id;
            stellarator::spawn(handle_msg_stream(msg_id, req_id, msg_log, tx.tx.clone()));
        }
        Packet::Msg(m) if m.id == GetMsgMetadata::ID => {
            let GetMsgMetadata { msg_id } = m.parse::<GetMsgMetadata>()?;
            let Some(metadata) =
                db.with_state(|s| s.msg_logs.get(&msg_id).and_then(|m| m.metadata().cloned()))
            else {
                return Err(Error::MsgNotFound(msg_id));
            };
            tx.send_msg(&metadata).await?;
        }
        Packet::Msg(m) if m.id == GetMsgs::ID => {
            let GetMsgs {
                msg_id,
                range,
                limit,
            } = m.parse::<GetMsgs>()?;
            let msg_log = db.with_state_mut(|s| {
                s.msg_logs
                    .get(&msg_id)
                    .ok_or(Error::MsgNotFound(msg_id))
                    .cloned()
            })?;
            let iter = msg_log.get_range(range).map(|(t, b)| (t, b.to_vec()));
            let data = if let Some(limit) = limit {
                iter.take(limit).collect()
            } else {
                iter.collect()
            };
            tx.send_msg(&MsgBatch { data }).await?;
        }
        Packet::Msg(m) if m.id == SaveArchive::ID => {
            let SaveArchive { path, format } = m.parse()?;
            db.save_archive(&path, format)?;
            tx.send_msg(&ArchiveSaved { path }).await?;
        }
        Packet::Msg(m) if m.id == VTableStream::ID => {
            let VTableStream { id } = m.parse::<VTableStream>()?;
            let vtable = db
                .with_state(|state| state.vtable_registry.get(&id).cloned())
                .ok_or(Error::InvalidMsgId)?;
            stellarator::spawn(handle_vtable_stream(
                id,
                vtable,
                db.clone(),
                tx.tx.clone(),
                m.req_id,
            ));
        }
        Packet::Msg(m) if m.id == UdpVTableStream::ID => {
            let UdpVTableStream { id, addr } = m.parse::<UdpVTableStream>()?;
            let addr = addr.to_socket_addrs()?.next().ok_or_else(|| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "missing socket ip",
                ))
            })?;
            db.with_state_mut(|state| {
                // ensure idempotency
                if state.udp_vtable_streams.contains(&(addr, id)) {
                    return Ok(());
                }
                let vtable = state
                    .vtable_registry
                    .get(&id)
                    .ok_or(Error::InvalidMsgId)?
                    .clone();
                let db = db.clone();
                let req_id = m.req_id;
                stellarator::struc_con::stellar(move || async move {
                    info!("VTable streaming to udp://{}", addr);
                    let mut socket = UdpSocket::ephemeral()?;
                    socket.connect(addr);
                    let (_, tx) = socket.split();
                    let tx = Arc::new(Mutex::new(PacketSink::new(tx)));
                    handle_vtable_stream(id, vtable, db, tx, req_id).await
                });
                state.udp_vtable_streams.insert((addr, id));
                Ok::<_, Error>(())
            })?;
        }
        Packet::Msg(m) => db.push_msg(Timestamp::now(), m.id, &m.buf)?,
        _ => {}
    }
    Ok(())
}

pub async fn handle_msg_stream<A: AsyncWrite>(
    msg_id: PacketId,
    req_id: RequestId,
    msg_log: MsgLog,
    tx: Arc<Mutex<PacketSink<A>>>,
) -> Result<(), Error> {
    let mut pkt = LenPacket::msg(msg_id, 64).with_request_id(req_id);
    loop {
        msg_log.wait().await;
        let Some((_timestamp, msg)) = msg_log.latest() else {
            continue;
        };
        let tx = tx.lock().await;
        pkt.extend_from_slice(msg);
        rent!(tx.send(pkt).await, pkt)?;
        pkt.clear();
    }
}
pub struct FixedRateStreamState {
    stream_id: StreamId,
    req_id: RequestId,
    time_step: AtomicU64,
    frequency: AtomicU64,
    is_scrubbed: AtomicBool,
    current_tick: AtomicI64,
    playing_cell: PlayingCell,
    filter: StreamFilter,
}

impl FixedRateStreamState {
    fn from_state(
        stream_id: StreamId,
        req_id: RequestId,
        time_step: Duration,
        current_tick: Timestamp,
        filter: StreamFilter,
        frequency: u64,
    ) -> FixedRateStreamState {
        FixedRateStreamState {
            stream_id,
            req_id,
            time_step: AtomicU64::new(time_step.as_nanos() as u64),
            is_scrubbed: AtomicBool::new(false),
            current_tick: AtomicI64::new(current_tick.0),
            playing_cell: PlayingCell::new(true),
            filter,
            frequency: AtomicU64::new(frequency),
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

    fn set_frequency(&self, frequency: u64) {
        self.frequency.store(frequency, atomic::Ordering::SeqCst);
    }

    fn time_step(&self) -> Duration {
        Duration::from_nanos(self.time_step.load(atomic::Ordering::Relaxed))
    }

    fn sleep_time(&self) -> Duration {
        Duration::from_micros(1_000_000 / self.frequency.load(atomic::Ordering::Relaxed))
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

fn handle_unicast_stream(addr: SocketAddr, stream: Stream, db: Arc<DB>) {
    stellarator::struc_con::stellar(move || async move {
        info!("UDP unicasting to {}", addr);
        let mut socket = UdpSocket::ephemeral()?;
        socket.connect(addr);
        let (_, tx) = socket.split();
        let tx = Arc::new(Mutex::new(PacketSink::new(tx)));
        handle_stream(tx.clone(), stream.clone(), db.clone(), 0)
            .await
            .unwrap();
        Ok::<_, Error>(())
    });
}

fn handle_stream<A: AsyncWrite + 'static>(
    tx: Arc<Mutex<PacketSink<A>>>,
    stream: Stream,
    db: Arc<DB>,
    req_id: RequestId,
) -> stellarator::JoinHandle<()> {
    match stream.behavior {
        StreamBehavior::RealTime => stellarator::spawn(async move {
            match handle_real_time_stream(tx, req_id, stream.filter, db).await {
                Ok(_) => {}
                Err(err) if err.is_stream_closed() => {}
                Err(err) => {
                    warn!(?err, "error streaming data");
                }
            }
        }),
        StreamBehavior::FixedRate(fixed_rate) => {
            let state = Arc::new(FixedRateStreamState::from_state(
                stream.id,
                req_id,
                Duration::from_nanos(fixed_rate.timestep.unwrap_or_else(|| {
                    db.default_stream_time_step.load(atomic::Ordering::Relaxed)
                })),
                match fixed_rate.initial_timestamp {
                    InitialTimestamp::Earliest => db.earliest_timestamp,
                    InitialTimestamp::Latest => db.last_updated.latest(),
                    InitialTimestamp::Manual(timestamp) => timestamp,
                },
                stream.filter,
                fixed_rate.frequency.unwrap_or(60),
            ));
            debug!(stream.id = ?stream.id, "inserting stream");
            db.with_state_mut(|s| s.streams.insert(stream.id, state.clone()));
            stellarator::spawn(async move {
                match handle_fixed_stream(tx, state, db).await {
                    Ok(_) => {}
                    Err(err) if err.is_stream_closed() => {}
                    Err(err) => {
                        warn!(?err, "error streaming data");
                    }
                }
            })
        }
    }
}

async fn handle_real_time_stream<A: AsyncWrite + 'static>(
    sink: Arc<Mutex<PacketSink<A>>>,
    req_id: RequestId,
    filter: StreamFilter,
    db: Arc<DB>,
) -> Result<(), Error> {
    let mut visited_ids = HashSet::new();
    loop {
        db.with_state(|state| {
            filter.visit(&state.components, |component| {
                if visited_ids.contains(&(component.component_id, component.entity_id)) {
                    return Ok(());
                }
                visited_ids.insert((component.component_id, component.entity_id));
                let sink = sink.clone();
                let component = component.clone();
                stellarator::spawn(handle_real_time_component(sink, component, req_id));
                Ok(())
            })
        })?;
        db.vtable_gen.wait().await;
    }
}

async fn handle_real_time_component<A: AsyncWrite>(
    stream: Arc<Mutex<PacketSink<A>>>,
    component: Component,
    req_id: RequestId,
) -> Result<(), Error> {
    let timestamp_loc = raw_table(0, size_of::<Timestamp>() as u16);
    let prim_type = component.schema.prim_type;
    let vtable = vtable([raw_field(
        (prim_type.padding(8) + size_of::<Timestamp>()) as u16,
        component.schema.size() as u16,
        timestamp(timestamp_loc, component.as_vtable_op()),
    )]);
    let waiter = component.time_series.waiter();
    let vtable_id: PacketId = fastrand::u16(..).to_le_bytes();
    {
        let stream = stream.lock().await;
        stream
            .send(
                VTableMsg {
                    id: vtable_id,
                    vtable,
                }
                .with_request_id(req_id),
            )
            .await
            .0?;
    }

    let mut table = LenPacket::table(vtable_id, 2048 - 16);
    loop {
        let _ = waiter.wait().await;
        let Some((&timestamp, buf)) = component.time_series.latest() else {
            continue;
        };
        table.push_aligned(timestamp);
        table.pad_for_type(prim_type);
        table.extend_from_slice(buf);
        {
            let stream = stream.lock().await;
            if let Err(err) = rent!(stream.send(table.with_request_id(req_id)).await, table) {
                debug!(%err, "error sending table");
                if Error::from(err).is_stream_closed() {
                    return Ok(());
                }
            }
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
    let mut table = LenPacket::table([0; 2], 2048 - 16);
    let mut components = db.with_state(|state| state.components.clone());
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
        let vtable_gen = db.vtable_gen.latest();
        if vtable_gen != current_gen {
            components = db.with_state(|state| state.components.clone());
            let stream = stream.lock().await;
            let id: PacketId = state.stream_id.to_le_bytes()[..2].try_into().unwrap();
            table = LenPacket::table(id, 2048 - 16);
            let vtable = state.filter.vtable(&components)?;
            let msg = VTableMsg { id, vtable };
            stream.send(msg.with_request_id(state.req_id)).await.0?;
            current_gen = vtable_gen;
        }
        table.clear();
        if let Err(err) = state
            .filter
            .populate_table(&components, &mut table, current_timestamp)
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
                    .with_request_id(state.req_id),
                )
                .await
                .0?;
            rent!(
                stream.send(table.with_request_id(state.req_id)).await,
                table
            )?;
        }
        let sleep_time = state.sleep_time().saturating_sub(start.elapsed());
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
        components: &HashMap<(EntityId, ComponentId), Component>,
        f: impl for<'a> FnMut(&'a Component) -> Result<(), Error>,
    ) -> Result<(), Error>;
    fn vtable(
        &self,
        components: &HashMap<(EntityId, ComponentId), Component>,
    ) -> Result<VTable, Error>;
    fn populate_table(
        &self,
        components: &HashMap<(EntityId, ComponentId), Component>,
        table: &mut LenPacket,
        tick: Timestamp,
    ) -> Result<(), Error>;
}

impl StreamFilterExt for StreamFilter {
    fn vtable(
        &self,
        components: &HashMap<(EntityId, ComponentId), Component>,
    ) -> Result<VTable<Vec<impeller2::vtable::Op>, Vec<u8>, Vec<impeller2::vtable::Field>>, Error>
    {
        let mut fields = vec![];
        let mut offset = 0;
        self.visit(components, |entity| {
            if !entity.time_series.index().is_empty() {
                offset += PrimType::U64.padding(offset);
                let op = timestamp(builder::raw_table(offset as u16, 8), entity.as_vtable_op());
                offset += size_of::<Timestamp>();
                offset += entity.schema.prim_type.padding(offset);
                let len = entity.schema.size();
                let size = len as u16;
                fields.push(raw_field(offset as u16, size, op));
                offset += len;
            }
            Ok(())
        })?;
        Ok(vtable(fields))
    }

    fn populate_table(
        &self,
        components: &HashMap<(EntityId, ComponentId), Component>,
        table: &mut LenPacket,
        timestamp: Timestamp,
    ) -> Result<(), Error> {
        self.visit(components, |entity| {
            let tick = entity.time_series.start_timestamp().max(timestamp);
            let Some((timestamp, buf)) = entity.get_nearest(tick) else {
                return Ok(());
            };
            table.push_aligned(timestamp);
            table.pad_for_type(entity.schema.prim_type);
            table.extend_from_slice(buf);
            Ok(())
        })
    }

    fn visit(
        &self,
        components: &HashMap<(EntityId, ComponentId), Component>,
        mut f: impl for<'a> FnMut(&'a Component) -> Result<(), Error>,
    ) -> Result<(), Error> {
        components
            .iter()
            .filter(|((_, id), _)| self.component_id.is_none() || self.component_id == Some(*id))
            .filter(|((id, _), _)| self.entity_id.is_none() || self.entity_id == Some(*id))
            .try_for_each(|(_, component)| f(component))
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
