use assets::open_assets;
use impeller2::{
    com_de::Decomponentize,
    component::Component as _,
    registry,
    schema::Schema,
    table::{Entry, VTable, VTableBuilder},
    types::{
        ComponentId, ComponentView, EntityId, IntoLenPacket, LenPacket, Msg, OwnedPacket as Packet,
        PacketId, PrimType, RequestId, Timestamp,
    },
};
use impeller2_stella::PacketSink;
use impeller2_wkt::*;
use msg_log::MsgLog;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    ffi::OsStr,
    sync::atomic::AtomicI64,
};
use std::{
    net::SocketAddr,
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{self, AtomicBool, AtomicU64},
    },
    time::Instant,
};
use std::{sync::RwLock, time::Duration};
use stellarator::{
    buf::Slice,
    io::{AsyncWrite, OwnedWriter, SplitExt},
    net::{TcpListener, TcpStream},
    rent,
    struc_con::Joinable,
    sync::{Mutex, WaitQueue},
    util::AtomicCell,
};
use time_series::TimeSeries;
use tracing::{debug, info, trace, warn};
use zerocopy::IntoBytes;

pub use error::Error;

pub mod append_log;
mod arrow;
mod assets;
mod error;
mod msg_log;
pub(crate) mod time_series;

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
        let mut db = DB {
            state: RwLock::new(state),
            recording_cell: PlayingCell::new(true),
            path,
            vtable_gen: AtomicCell::new(0),
            time_step,
            default_stream_time_step,
            last_updated: AtomicCell::new(Timestamp(i64::MIN)),
            earliest_timestamp: Timestamp::now(),
        };
        db.init_globals()?;
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

    fn init_globals(&mut self) -> Result<(), Error> {
        let arr = nox::array![0u64];
        let globals_id = EntityId(0);
        let globals_metadata = EntityMetadata {
            entity_id: globals_id,
            name: "Globals".to_string(),
            metadata: Default::default(),
        };
        let tick_component_id = impeller2_wkt::Tick::COMPONENT_ID;
        let tick_component_view = ComponentView::U64(arr.view());
        let tick_component_schema =
            ComponentSchema::new(tick_component_view.prim_type(), tick_component_view.shape());
        let tick_component_metadata = ComponentMetadata {
            component_id: tick_component_id,
            name: impeller2_wkt::Tick::NAME.to_string(),
            metadata: [("element_names".to_string(), "tick".to_string())]
                .into_iter()
                .collect(),
            asset: false,
        };

        self.with_state_mut(|state| {
            state.set_entity_metadata(globals_metadata, &self.path)?;
            state.set_component_metadata(tick_component_metadata, &self.path)?;
            state.insert_component(
                tick_component_id,
                tick_component_schema,
                globals_id,
                &self.path,
            )?;
            let mut sink = DBSink {
                components: &state.components,
                last_updated: &self.last_updated,
            };
            sink.apply_value(tick_component_id, globals_id, tick_component_view, None);
            Ok(())
        })
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

        for elem in std::fs::read_dir(&path)? {
            let Ok(elem) = elem else { continue };
            let path = elem.path();
            if !path.is_dir()
                || (path.file_name() == Some(OsStr::new("entity_metadata"))
                    || path.file_name() == Some(OsStr::new("assets")))
                || path.file_name() == Some(OsStr::new("msgs"))
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

                let entity = Component::open(path, component_id, entity_id, schema.clone())?;
                if let Some((timestamp, _)) = entity.time_series.latest() {
                    last_updated = timestamp.0.max(last_updated);
                };
                start_timestamp = start_timestamp.min(entity.time_series.start_timestamp().0);
                entity_components.insert((entity_id, component_id), entity);
            }
        }
        for elem in std::fs::read_dir(path.join("msgs"))? {
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
            for (entity_id, component_id, prim_ty, shape) in vtable.vtable.column_iter() {
                let component_schema = ComponentSchema::new(prim_ty, shape);
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
        if self.components.contains_key(&(entity_id, component_id)) {
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

    fn add_to_vtable(
        &self,
        vtable: &mut VTableBuilder<Vec<impeller2::table::Entry>, Vec<u8>>,
    ) -> Result<(), Error> {
        let timestamp = vtable.timestamp()?;
        vtable.entity_with_timestamp(
            self.entity_id,
            &[(
                self.component_id,
                self.schema.prim_type,
                &self.schema.shape(),
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

struct DBSink<'a> {
    components: &'a HashMap<(EntityId, ComponentId), Component>,
    last_updated: &'a AtomicCell<Timestamp>,
}

impl Decomponentize for DBSink<'_> {
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        entity_id: EntityId,
        value: impeller2::types::ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) {
        let timestamp = timestamp.unwrap_or_else(Timestamp::now);
        let value_buf = value.as_bytes();
        let Some(component) = self.components.get(&(entity_id, component_id)) else {
            warn!(component.id = ?component_id, entity.id = ?entity_id, "component not found when sinking");
            return;
        };
        if let Err(err) = component.time_series.push_buf(timestamp, value_buf) {
            warn!(?err, "failed to write head value");
            return;
        }
        self.last_updated.update_max(timestamp);
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
        let req_id = pkt.req_id();
        match handle_packet(&pkt, &db, &tx).await {
            Ok(_) => {}
            Err(err) if err.is_stream_closed() => {}
            Err(err) => {
                warn!(?err, "error handling packet");
                let tx = tx.lock().await;
                if let Err(err) = tx
                    .send(
                        ErrorResponse {
                            description: err.to_string(),
                        }
                        .with_request_id(req_id),
                    )
                    .await
                    .0
                {
                    warn!(?err, "error sending err resp");
                }
                trace!(?err, "error handling packet");
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
            db.insert_vtable(vtable)?;
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
                        fixed_rate.frequency.unwrap_or(60),
                    ));
                    debug!(stream.id = ?stream_id, "inserting stream");
                    db.with_state_mut(|s| s.streams.insert(stream_id, state.clone()));
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
            let tx = tx.lock().await;
            tx.send(&SchemaMsg(schema)).await.0?;
        }
        Packet::Msg(m) if m.id == GetTimeSeries::ID => {
            let get_time_series = m.parse::<GetTimeSeries>()?;
            debug!(msg = ?get_time_series, "get time series");

            let tx = tx.clone();
            let db = db.clone();
            let result = (|| {
                let GetTimeSeries {
                    component_id,
                    entity_id,
                    range,
                    limit,
                    id,
                } = get_time_series;
                let entity = db.with_state(|state| {
                    let Some(component) = state.components.get(&(entity_id, component_id)) else {
                        return Err(Error::ComponentNotFound(component_id));
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
                let mut pkt =
                    LenPacket::time_series(id, data.len() + timestamps.as_bytes().len() + 8);
                pkt.extend_from_slice(&(timestamps.len() as u64).to_le_bytes());
                pkt.extend_from_slice(timestamps.as_bytes());
                pkt.extend_from_slice(data);
                Ok(pkt)
            })();
            stellarator::spawn(async move {
                let tx = tx.lock().await;
                match result {
                    Ok(pkt) => {
                        tx.send(pkt).await.0?;
                    }
                    Err(err) => tx.send(&ErrorResponse::from(err)).await.0?,
                }
                Ok::<_, Error>(())
            });
        }
        Packet::Msg(m) if m.id == SetComponentMetadata::ID => {
            let SetComponentMetadata(metadata) = m.parse::<SetComponentMetadata>()?;
            db.with_state_mut(|state| state.set_component_metadata(metadata, &db.path))?;
        }
        Packet::Msg(m) if m.id == GetComponentMetadata::ID => {
            let GetComponentMetadata { component_id } = m.parse::<GetComponentMetadata>()?;
            let metadata = db.with_state(|state| {
                let Some(metadata) = state.component_metadata.get(&component_id) else {
                    return Err(Error::ComponentNotFound(component_id));
                };
                Ok(metadata.into_len_packet())
            })?;
            let tx = tx.lock().await;
            tx.send(metadata).await.0?;
        }

        Packet::Msg(m) if m.id == SetEntityMetadata::ID => {
            let SetEntityMetadata(metadata) = m.parse::<SetEntityMetadata>()?;
            db.with_state_mut(|state| state.set_entity_metadata(metadata, &db.path))?;
        }

        Packet::Msg(m) if m.id == GetEntityMetadata::ID => {
            let GetEntityMetadata { entity_id } = m.parse::<GetEntityMetadata>()?;
            let msg = db.with_state(|state| {
                let Some(metadata) = state.entity_metadata.get(&entity_id) else {
                    return Err(Error::EntityNotFound(entity_id));
                };
                Ok(metadata.into_len_packet())
            })?;
            let tx = tx.lock().await;
            tx.send(msg).await.0?;
        }

        Packet::Msg(m) if m.id == SetAsset::ID => {
            let set_asset = m.parse::<SetAsset<'_>>()?;
            db.insert_asset(set_asset.id, &set_asset.buf)?;
        }
        Packet::Msg(m) if m.id == GetAsset::ID => {
            let GetAsset { id } = m.parse::<GetAsset>()?;
            let packet = db.with_state(|state| {
                let Some(mmap) = state.assets.get(&id) else {
                    return Err(Error::AssetNotFound(id));
                };
                let asset = Asset {
                    id,
                    buf: Cow::Borrowed(&mmap[..]),
                };
                Ok(asset.into_len_packet())
            })?;
            let tx = tx.lock().await;
            tx.send(packet).await.0?;
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
            let tx = tx.lock().await;
            tx.send(&msg).await.0?;
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

            let tx = tx.lock().await;
            tx.send(&msg).await.0?;
        }

        Packet::Msg(m) if m.id == DumpAssets::ID => {
            let packets = db.with_state(|state| {
                state
                    .assets
                    .iter()
                    .map(|(id, mmap)| Asset {
                        id: *id,
                        buf: Cow::Borrowed(&mmap[..]),
                    })
                    .map(|asset| asset.into_len_packet())
                    .collect::<Vec<_>>()
            });

            let tx = tx.lock().await;
            for packet in packets {
                tx.send(packet).await.0?;
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
                            .send(&LastUpdated(last_updated))
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
            tx.send(&db.db_settings()).await.0?;
        }
        Packet::Msg(m) if m.id == GetEarliestTimestamp::ID => {
            let tx = tx.lock().await;
            tx.send(&EarliestTimestamp(db.earliest_timestamp)).await.0?;
        }
        Packet::Msg(m) if m.id == GetDbSettings::ID => {
            let tx = tx.lock().await;
            let settings = db.db_settings();
            tx.send(&settings).await.0?;
        }
        Packet::Table(table) => {
            trace!(table.len = table.buf.len(), "sinking table");
            db.with_state(|state| {
                let mut sink = DBSink {
                    components: &state.components,
                    last_updated: &db.last_updated,
                };
                table.sink(&state.vtable_registry, &mut sink)
            })?;
        }

        Packet::Msg(m) if m.id == GetDbSettings::ID => {
            let tx = tx.lock().await;
            let settings = db.db_settings();
            tx.send(&settings).await.0?;
        }
        Packet::Msg(m) if m.id == SQLQuery::ID => {
            let SQLQuery(query) = m.parse::<SQLQuery>()?;
            let inner_tx = tx.clone();
            let db = db.clone();
            let req_id = m.req_id;
            let msg = stellarator::struc_con::tokio(move |_| async move {
                let mut ctx = db.as_session_context()?;
                db.insert_views(&mut ctx).await?;
                let df = ctx.sql(&query).await?;
                let results = df.collect().await?;
                let mut batches = vec![];
                for batch in results.into_iter() {
                    let mut buf = vec![];
                    let mut writer =
                        ::arrow::ipc::writer::StreamWriter::try_new(&mut buf, batch.schema_ref())?;
                    writer.write(&batch)?;
                    writer.finish()?;
                    batches.push(Cow::Owned(buf));
                }
                let msg = ArrowIPC { batches };
                Ok::<_, Error>(msg)
            })
            .join()
            .await??;

            let tx = inner_tx.lock().await;
            tx.send(msg.with_request_id(req_id)).await.0?;
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
            let tx = tx.clone();
            stellarator::spawn(handle_msg_stream(msg_id, req_id, msg_log, tx));
        }
        Packet::Msg(m) if m.id == GetMsgMetadata::ID => {
            let GetMsgMetadata { msg_id } = m.parse::<GetMsgMetadata>()?;
            let Some(metadata) =
                db.with_state(|s| s.msg_logs.get(&msg_id).and_then(|m| m.metadata().cloned()))
            else {
                return Err(Error::MsgNotFound(msg_id));
            };
            let tx = tx.lock().await;
            tx.send(metadata.with_request_id(m.req_id)).await.0.unwrap();
        }
        Packet::Msg(m) if m.id == GetMsgs::ID => {
            let GetMsgs {
                msg_id,
                range,
                limit,
            } = m.parse::<GetMsgs>()?;
            let msg_log =
                db.with_state_mut(|s| s.get_or_insert_msg_log(msg_id, &db.path).cloned())?;
            let iter = msg_log.get_range(range).map(|(t, b)| (t, b.to_vec()));
            let data = if let Some(limit) = limit {
                iter.take(limit).collect()
            } else {
                iter.collect()
            };
            let tx = tx.lock().await;
            tx.send(MsgBatch { data }.with_request_id(m.req_id))
                .await
                .0
                .unwrap();
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
        time_step: Duration,
        current_tick: Timestamp,
        filter: StreamFilter,
        frequency: u64,
    ) -> FixedRateStreamState {
        FixedRateStreamState {
            stream_id,
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

async fn handle_real_time_stream<A: AsyncWrite + 'static>(
    stream: Arc<Mutex<PacketSink<A>>>,
    _stream_id: StreamId,
    filter: StreamFilter,
    db: Arc<DB>,
) -> Result<(), Error> {
    let mut visited_ids = HashSet::new();
    loop {
        db.with_state(|state| {
            filter.visit(&state.components, |entity| {
                if visited_ids.contains(&(entity.component_id, entity.entity_id)) {
                    return Ok(());
                }
                visited_ids.insert((entity.component_id, entity.entity_id));
                let stream = stream.clone();

                let mut vtable = VTableBuilder::default();
                entity.add_to_vtable(&mut vtable)?;
                let vtable = vtable.build();
                let waiter = entity.time_series.waiter();
                let entity = entity.clone();
                stellarator::spawn(async move {
                    match handle_real_time_entity(stream, entity, waiter, vtable).await {
                        Ok(_) => {}
                        Err(err) if err.is_stream_closed() => {}
                        Err(err) => warn!(?err, "failed to handle real time stream"),
                    }
                });
                Ok(())
            })
        })?;
        db.vtable_gen.wait().await;
    }
}

async fn handle_real_time_entity<A: AsyncWrite>(
    stream: Arc<Mutex<PacketSink<A>>>,
    entity: Component,
    waiter: Arc<WaitQueue>,
    vtable: VTable<Vec<Entry>, Vec<u8>>,
) -> Result<(), Error> {
    let vtable_id: PacketId = fastrand::u16(..).to_le_bytes();
    {
        let stream = stream.lock().await;
        stream
            .send(&VTableMsg {
                id: vtable_id,
                vtable,
            })
            .await
            .0?;
    }

    let prim_type = entity.schema.prim_type;
    let mut table = LenPacket::table(vtable_id, 2048 - 16);
    loop {
        let _ = waiter.wait().await;
        let Some((&timestamp, buf)) = entity.time_series.latest() else {
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
            stream.send(msg.into_len_packet()).await.0?;
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
                    .into_len_packet(),
                )
                .await
                .0?;
            rent!(stream.send(table).await, table)?;
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
    ) -> Result<VTable<Vec<impeller2::table::Entry>, Vec<u8>>, Error>;
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
    ) -> Result<VTable<Vec<impeller2::table::Entry>, Vec<u8>>, Error> {
        let mut vtable = VTableBuilder::default();
        self.visit(components, |entity| {
            entity.add_to_vtable(&mut vtable)?;
            Ok(())
        })?;
        Ok(vtable.build())
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
