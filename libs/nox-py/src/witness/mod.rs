//! Deterministic witnesses of lockstep runs: the exact bytes of chosen batches and hook channels
//! at fixed points of every tick, digested into a tree, compared with another run, and replayed
//! through the `pre_commit` hooks without the systems or the DB.

mod compare;
mod format;
mod record;
mod replay;

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use impeller::types::ComponentId;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::World;
use crate::native_hook::{Field, NativeStepHook};
use crate::prepared::{BatchHandle, DeferredWrite, PreCommit};

pub use compare::Mismatch;
pub use format::{HookLabel, Reader};
pub use record::{Plan, Recorder, WitnessReport};

use format::{BatchSchema, ChannelSchema, Digest, FieldSchema, Kind, RecordSchema, Schema};

/// Ticks kept exact before and after the first mismatch with a reference.
const WINDOW: u64 = 16;

/// Bytes of one tick's records, back to back, and their digests once hashed.
#[derive(Default)]
pub struct Block {
    pub tick: u64,
    pub timestamp: i64,
    pub bytes: Vec<u8>,
    /// Where each record ends in `bytes`, in schema order.
    pub ends: Vec<usize>,
    pub digests: Vec<Digest>,
}

impl Block {
    pub fn record(&self, index: usize) -> &[u8] {
        let start = index.checked_sub(1).map_or(0, |i| self.ends[i]);
        &self.bytes[start..self.ends[index]]
    }

    fn clear(&mut self) {
        self.bytes.clear();
        self.ends.clear();
    }

    fn end_record(&mut self) {
        self.ends.push(self.bytes.len());
    }
}

/// Where a batch field's rows are in the world.
pub struct FieldSource {
    component_id: ComponentId,
    rows: Vec<usize>,
    row_bytes: usize,
    /// The first row when the rows are consecutive.
    first_row: Option<usize>,
}

impl FieldSource {
    fn new(component_id: ComponentId, rows: Vec<usize>, row_bytes: usize) -> Self {
        let consecutive = rows.windows(2).all(|pair| pair[1] == pair[0] + 1);
        Self {
            component_id,
            first_row: rows.first().copied().filter(|_| consecutive),
            rows,
            row_bytes,
        }
    }

    fn gather(&self, world: &World, out: &mut Vec<u8>) {
        let column = &world.host[&self.component_id].buffer;
        let size = self.row_bytes;
        match self.first_row {
            Some(row) => out.extend_from_slice(&column[row * size..][..self.rows.len() * size]),
            None => {
                for &row in &self.rows {
                    out.extend_from_slice(&column[row * size..][..size]);
                }
            }
        }
    }

    fn scatter(&self, world: &mut World, bytes: &[u8]) {
        let column = &mut world.host.get_mut(&self.component_id).unwrap().buffer;
        let size = self.row_bytes;
        for (&row, value) in self.rows.iter().zip(bytes.chunks_exact(size)) {
            column[row * size..][..size].copy_from_slice(value);
        }
    }
}

/// A witness resolved for one run: its schema, and where each batch field's rows are.
pub struct Layout {
    pub schema: Schema,
    /// Per batch, per field.
    sources: Vec<Vec<FieldSource>>,
    /// The records of each stage.
    stages: Vec<Range<usize>>,
    after_systems: Option<usize>,
    after_hook: Vec<Option<usize>>,
    after_commit: Option<usize>,
}

impl Layout {
    fn new(schema: Schema, sources: Vec<Vec<FieldSource>>, hooks: usize) -> Self {
        let stages = (0..schema.stages.len())
            .map(|stage| {
                let start = schema.records.partition_point(|r| r.stage < stage);
                start..schema.records.partition_point(|r| r.stage <= stage)
            })
            .collect();
        let stage = |name: &str| schema.stages.iter().position(|s| s == name);
        Self {
            after_systems: stage("after_systems"),
            after_hook: (0..hooks)
                .map(|n| stage(&format!("after_hook:{n}")))
                .collect(),
            after_commit: stage("after_commit"),
            schema,
            sources,
            stages,
        }
    }

    /// The bytes of a tick whose channels are all filled and whose hooks defer nothing.
    fn block_bytes(&self) -> usize {
        let records = self.schema.records.iter().filter_map(|record| {
            let batch = record.batch?;
            let rows = self.sources[batch]
                .first()
                .map_or(0, |field| field.rows.len());
            Some(rows * record.row_bytes as usize)
        });
        records.sum()
    }

    fn records(&self, stage: usize) -> impl Iterator<Item = (usize, &RecordSchema)> {
        let range = self.stages[stage].clone();
        range.clone().zip(&self.schema.records[range])
    }

    fn source(&self, record: &RecordSchema) -> &FieldSource {
        &self.sources[record.batch.expect("field records have a batch")][record.index]
    }

    pub fn begin(&self, block: &mut Block, tick: u64, timestamp: i64) {
        block.clear();
        block.tick = tick;
        block.timestamp = timestamp;
    }

    /// Appends the `after_systems` records: every row of every batch field.
    pub fn capture_systems(&self, block: &mut Block, world: &World) {
        let Some(stage) = self.after_systems else {
            return;
        };
        for (_, record) in self.records(stage) {
            self.source(record).gather(world, &mut block.bytes);
            block.end_record();
        }
    }

    /// Appends hook `hook`'s records: the channels it filled, every write field, and its
    /// deferred writes.
    pub fn capture_hook(
        &self,
        hook: usize,
        block: &mut Block,
        world: &World,
        channels: &Channels,
        deferred: &[DeferredWrite],
        start: i64,
    ) {
        let Some(stage) = self.after_hook[hook] else {
            return;
        };
        for (_, record) in self.records(stage) {
            match record.kind {
                Kind::Field => self.source(record).gather(world, &mut block.bytes),
                Kind::Channel | Kind::External => {
                    if let Some(bytes) = channels.bytes(record.index) {
                        block.bytes.extend_from_slice(bytes);
                    }
                }
                Kind::Deferred => {
                    for write in deferred {
                        block
                            .bytes
                            .extend_from_slice(&(write.timestamp.0 - start).to_le_bytes());
                        for (_, id, value) in &write.rows {
                            block.bytes.extend_from_slice(&id.0.to_le_bytes());
                            block
                                .bytes
                                .extend_from_slice(&(value.len() as u32).to_le_bytes());
                            block.bytes.extend_from_slice(value);
                        }
                    }
                }
                Kind::Recorded => unreachable!("commit records are in after_commit"),
            }
            block.end_record();
        }
    }

    /// Appends the `after_commit` records: whether the commit recorded each row of each field.
    pub fn capture_commit(&self, block: &mut Block, recorded: impl Fn(usize) -> bool) {
        let Some(stage) = self.after_commit else {
            return;
        };
        let mut row = 0;
        for (_, record) in self.records(stage) {
            for _ in 0..self.source(record).rows.len() {
                block.bytes.push(recorded(row).into());
                row += 1;
            }
            block.end_record();
        }
    }

    /// The rows of the `after_commit` records, in order: each one's component and column row.
    fn commit_rows(&self) -> impl Iterator<Item = (ComponentId, usize)> + '_ {
        let records = self.after_commit.into_iter().flat_map(|s| self.records(s));
        records.flat_map(|(_, record)| {
            let source = self.source(record);
            source.rows.iter().map(|&row| (source.component_id, row))
        })
    }

    /// Hashes every hashed record of `block`; a channel no hook filled hashes to zeros.
    pub fn digest(&self, block: &mut Block) {
        block.digests.clear();
        for (index, record) in self.schema.records.iter().enumerate() {
            if !record.hashed() {
                continue;
            }
            let bytes = block.record(index);
            block
                .digests
                .push(if record.kind == Kind::Channel && bytes.is_empty() {
                    [0; 32]
                } else {
                    *blake3::hash(bytes).as_bytes()
                });
        }
    }

    fn load_systems(&self, world: &mut World, block: &Block) {
        for (index, record) in self.after_systems.into_iter().flat_map(|s| self.records(s)) {
            self.source(record).scatter(world, block.record(index));
        }
    }

    fn load_writes(&self, stage: usize, world: &mut World, block: &Block) {
        for (index, record) in self.records(stage) {
            if record.kind == Kind::Field {
                self.source(record).scatter(world, block.record(index));
            }
        }
    }

    fn load_external(&self, stage: usize, channels: &mut Channels, block: &Block) {
        for (index, record) in self.records(stage) {
            let bytes = block.record(index);
            if record.kind == Kind::External && !bytes.is_empty() {
                channels.buffers[record.index].copy_from_slice(bytes);
                channels.filled[record.index] = true;
            }
        }
    }
}

/// The stages of a run with `hooks` hooks, in order; `selected` keeps only those it names.
fn stages(hooks: usize, selected: Option<&[String]>) -> Result<Vec<String>, String> {
    let all: Vec<String> = std::iter::once("after_systems".to_string())
        .chain((0..hooks).map(|n| format!("after_hook:{n}")))
        .chain(std::iter::once("after_commit".to_string()))
        .collect();
    let Some(selected) = selected else {
        return Ok(all);
    };
    if let Some(unknown) = selected.iter().find(|stage| !all.contains(stage)) {
        return Err(format!(
            "this run has no stage {unknown:?}; its stages are {all:?}"
        ));
    }
    Ok(all.into_iter().filter(|s| selected.contains(s)).collect())
}

/// The records of each stage: after the systems, every batch field; after a hook, its channels,
/// the write fields and its deferred writes, then the external channels; after the commit, a
/// recorded flag per row of every field.
fn records(
    batches: &[BatchSchema],
    channels: &[ChannelSchema],
    stages: &[String],
) -> Vec<RecordSchema> {
    let mut records = Vec::new();
    for (stage, name) in stages.iter().enumerate() {
        let fields = batches.iter().enumerate().flat_map(|(b, batch)| {
            let fields = batch.fields.iter().enumerate();
            fields.map(move |(f, field)| (b, f, batch, field))
        });
        let field_record =
            |kind, (b, f, batch, field): (usize, usize, &BatchSchema, &FieldSchema)| RecordSchema {
                stage,
                kind,
                name: format!("{}.{}", batch.name, field.name),
                batch: Some(b),
                index: f,
                row_bytes: if kind == Kind::Recorded {
                    1
                } else {
                    field.row_bytes
                },
            };
        let channel_record = |(c, channel): (usize, &ChannelSchema)| RecordSchema {
            stage,
            kind: if channel.external {
                Kind::External
            } else {
                Kind::Channel
            },
            name: channel.name.clone(),
            batch: Some(channel.batch),
            index: c,
            row_bytes: channel.row_bytes,
        };
        match name.as_str() {
            "after_systems" => records.extend(fields.map(|f| field_record(Kind::Field, f))),
            "after_commit" => records.extend(fields.map(|f| field_record(Kind::Recorded, f))),
            _ => {
                let channels = channels.iter().enumerate();
                records.extend(
                    channels
                        .clone()
                        .filter(|(_, c)| !c.external)
                        .map(channel_record),
                );
                let writes = fields.filter(|(.., field)| field.access == "write");
                records.extend(writes.map(|f| field_record(Kind::Field, f)));
                records.push(RecordSchema {
                    stage,
                    kind: Kind::Deferred,
                    name: "deferred".into(),
                    batch: None,
                    index: 0,
                    row_bytes: 0,
                });
                records.extend(channels.filter(|(_, c)| c.external).map(channel_record));
            }
        }
    }
    records
}

/// The witness channels of a run: one buffer per channel, a row per row of its batch, which
/// hooks fill.
#[derive(Default)]
pub struct Channels {
    defs: Vec<ChannelDef>,
    buffers: Vec<Vec<u8>>,
    filled: Vec<bool>,
    /// External channels hold recorded bytes, which hooks read and cannot write.
    replay: bool,
}

pub struct ChannelDef {
    pub name: String,
    /// `BatchHandle::id` of the batch whose rows it has.
    pub batch: u64,
    pub rows: usize,
    pub row_bytes: usize,
    pub external: bool,
}

impl Channels {
    pub fn new(defs: Vec<ChannelDef>, replay: bool) -> Self {
        Self {
            buffers: defs.iter().map(|c| vec![0; c.rows * c.row_bytes]).collect(),
            filled: vec![false; defs.len()],
            defs,
            replay,
        }
    }

    pub fn defs(&self) -> &[ChannelDef] {
        &self.defs
    }

    /// The index of channel `name`; none when the run records no witness.
    pub fn find(&self, name: &str) -> Result<Option<usize>, String> {
        if self.defs.is_empty() {
            return Ok(None);
        }
        let names = || {
            self.defs
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
        };
        match self.defs.iter().position(|c| c.name == name) {
            Some(index) => Ok(Some(index)),
            None => Err(format!(
                "no witness channel named {name:?}; the channels are {:?}",
                names()
            )),
        }
    }

    pub fn writable(&self, index: usize) -> bool {
        !(self.replay && self.defs[index].external)
    }

    /// Fills channel `index` with `bytes`, every row of it.
    pub fn write(&mut self, index: usize, bytes: &[u8]) -> Result<(), String> {
        let def = &self.defs[index];
        if !self.writable(index) {
            return Err(format!(
                "{:?} is external, so a replay reads it and cannot write it",
                def.name
            ));
        }
        if bytes.len() != def.rows * def.row_bytes {
            return Err(format!(
                "{:?} takes {} rows of {} bytes, {} in all; got {} bytes",
                def.name,
                def.rows,
                def.row_bytes,
                def.rows * def.row_bytes,
                bytes.len()
            ));
        }
        self.buffers[index].copy_from_slice(bytes);
        self.filled[index] = true;
        Ok(())
    }

    /// The recorded bytes of external channel `index`, in a replay that has them.
    pub fn replayed(&self, index: usize) -> Option<&[u8]> {
        let replayed = self.replay && self.defs[index].external && self.filled[index];
        replayed.then_some(self.buffers[index].as_slice())
    }

    /// Channel `index`'s buffer, for a native hook's column; zeroed when the hook may write it.
    pub fn column(&mut self, index: usize) -> *mut u8 {
        if self.writable(index) {
            self.buffers[index].fill(0);
        }
        self.buffers[index].as_mut_ptr()
    }

    pub fn set_filled(&mut self, index: usize) {
        self.filled[index] = true;
    }

    /// Channel `index`'s bytes, once filled.
    pub fn bytes(&self, index: usize) -> Option<&[u8]> {
        self.filled[index].then_some(self.buffers[index].as_slice())
    }

    fn clear_filled(&mut self) {
        self.filled.fill(false);
    }
}

static NEXT_BATCH_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_batch_id() -> u64 {
    NEXT_BATCH_ID.fetch_add(1, Ordering::Relaxed)
}

/// Batches and hook channels to witness, from `World.prepare_witness`.
#[pyclass(frozen)]
pub struct Witness {
    batches: Vec<(String, Py<BatchHandle>)>,
    channels: Vec<ChannelSchema>,
    stages: Option<Vec<String>>,
}

impl Witness {
    pub fn new(
        batches: &Bound<'_, PyDict>,
        channels: Option<&Bound<'_, PyDict>>,
        external: Option<&Bound<'_, PyDict>>,
        stages: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let batches = batches
            .iter()
            .map(|(name, batch)| Ok((name.extract()?, batch.extract()?)))
            .collect::<PyResult<Vec<(String, Py<BatchHandle>)>>>()?;
        if batches.is_empty() {
            return Err(PyValueError::new_err("a witness needs at least one batch"));
        }
        let mut parsed: Vec<ChannelSchema> = Vec::new();
        for (declared, external) in [(channels, false), (external, true)] {
            for (name, value) in declared.iter().flat_map(|d| d.iter()) {
                let name: String = name.extract()?;
                let (batch, row_bytes): (Bound<'_, BatchHandle>, u64) =
                    value.extract().map_err(|_| {
                        PyValueError::new_err(format!(
                            "channel {name:?} takes a (batch, row bytes) pair"
                        ))
                    })?;
                let id = batch.get().id;
                let Some(batch) = batches.iter().position(|(_, b)| b.get().id == id) else {
                    return Err(PyValueError::new_err(format!(
                        "the batch of channel {name:?} must be one of the witnessed batches"
                    )));
                };
                if row_bytes == 0 {
                    return Err(PyValueError::new_err(format!(
                        "channel {name:?} needs rows of at least one byte"
                    )));
                }
                if parsed.iter().any(|channel| channel.name == name) {
                    return Err(PyValueError::new_err(format!(
                        "channel {name:?} is declared twice"
                    )));
                }
                parsed.push(ChannelSchema {
                    name,
                    batch,
                    row_bytes,
                    external,
                });
            }
        }
        for stage in stages.iter().flatten() {
            let hook = stage.strip_prefix("after_hook:");
            let known = matches!(stage.as_str(), "after_systems" | "after_commit")
                || hook.is_some_and(|n| n.parse::<usize>().is_ok());
            if !known {
                return Err(PyValueError::new_err(format!(
                    "unknown stage {stage:?}; stages are after_systems, after_hook:N and after_commit"
                )));
            }
        }
        Ok(Self {
            batches,
            channels: parsed,
            stages,
        })
    }

    pub fn channels(&self, replay: bool) -> Channels {
        let defs = self.channels.iter().map(|channel| {
            let batch = self.batches[channel.batch].1.get();
            ChannelDef {
                name: channel.name.clone(),
                batch: batch.id,
                rows: batch.rows(),
                row_bytes: channel.row_bytes as usize,
                external: channel.external,
            }
        });
        Channels::new(defs.collect(), replay)
    }

    /// The witnessed batch named for each native hook, with the rows each of its calls owns.
    pub fn labels(
        &self,
        hooks: &[PyObject],
        runners: &[Box<dyn PreCommit + Send>],
    ) -> Vec<HookLabel> {
        Python::with_gil(|py| {
            hooks
                .iter()
                .zip(runners)
                .map(|(hook, runner)| {
                    let batch = hook
                        .downcast_bound::<NativeStepHook>(py)
                        .ok()
                        .and_then(|native| {
                            let id = native.get().batch();
                            let mut batches = self.batches.iter();
                            batches
                                .find(|(_, b)| b.get().id == id)
                                .map(|(name, _)| name.clone())
                        });
                    HookLabel {
                        name: runner.name().to_string(),
                        batch,
                        partitions: runner.partitions(),
                    }
                })
                .collect()
        })
    }

    /// The witness laid out over `world` for a run of `hooks` hooks.
    pub fn layout(&self, world: &World, hooks: usize) -> Result<Layout, String> {
        let stages = stages(hooks, self.stages.as_deref())?;
        let mut entities = BTreeMap::new();
        let mut sources = Vec::new();
        let batches = self
            .batches
            .iter()
            .map(|(name, batch)| {
                let mut batch_sources = Vec::new();
                let fields = batch.get().fields.iter().map(|(field, handle)| {
                    let handle = handle.get();
                    let (schema, _) = &world.metadata.component_map[&handle.component_id()];
                    for entity in handle.entities() {
                        let id = entity.inner;
                        let name = world
                            .metadata
                            .entity_metadata
                            .get(&id)
                            .map(|e| e.name.clone());
                        entities.insert(id.0, name.unwrap_or_else(|| format!("entity{}", id.0)));
                    }
                    batch_sources.push(FieldSource::new(
                        handle.component_id(),
                        handle.rows.clone(),
                        handle.row_bytes,
                    ));
                    FieldSchema {
                        name: field.clone(),
                        component: handle.name().to_string(),
                        dtype: crate::prepared::numpy_dtype(schema.prim_type).to_string(),
                        shape: schema.shape().to_vec(),
                        row_bytes: handle.row_bytes as u64,
                        access: if handle.writes() { "write" } else { "read" }.to_string(),
                        entities: handle.entities().iter().map(|e| e.inner.0).collect(),
                    }
                });
                let fields = fields.collect();
                sources.push(batch_sources);
                BatchSchema {
                    name: name.clone(),
                    fields,
                }
            })
            .collect::<Vec<_>>();
        let records = records(&batches, &self.channels, &stages);
        let schema = Schema {
            entities: entities.into_iter().collect(),
            batches,
            channels: self.channels.clone(),
            stages,
            records,
        };
        Ok(Layout::new(schema, sources, hooks))
    }
}

/// When a witness keeps each tick's exact bytes, besides the window around a first mismatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Raw {
    Never,
    Always,
    /// Ticks `start..stop`.
    Window(u64, u64),
}

impl Raw {
    fn keeps(self, tick: u64) -> bool {
        match self {
            Raw::Never => false,
            Raw::Always => true,
            Raw::Window(start, stop) => (start..stop).contains(&tick),
        }
    }
}

/// Where and how a run records its witness, for `World.run(determinism_trace=...)`.
#[pyclass(frozen)]
pub struct DeterminismTrace {
    #[pyo3(get)]
    path: PathBuf,
    witness: Py<Witness>,
    raw: Raw,
    #[pyo3(get)]
    reference: Option<PathBuf>,
    #[pyo3(get)]
    stop_after_first_mismatch: bool,
}

#[pymethods]
impl DeterminismTrace {
    #[new]
    #[pyo3(signature = (path, witness, raw = None, reference = None, stop_after_first_mismatch = false))]
    fn new(
        path: PathBuf,
        witness: Py<Witness>,
        raw: Option<&Bound<'_, PyAny>>,
        reference: Option<PathBuf>,
        stop_after_first_mismatch: bool,
    ) -> PyResult<Self> {
        let invalid =
            || PyValueError::new_err("raw must be True, False or a (start, stop) tick window");
        let raw = match raw {
            None => Raw::Never,
            Some(raw) => match raw.extract::<bool>() {
                Ok(true) => Raw::Always,
                Ok(false) => Raw::Never,
                Err(_) => {
                    let (start, stop): (u64, u64) = raw.extract().map_err(|_| invalid())?;
                    if start >= stop {
                        return Err(invalid());
                    }
                    Raw::Window(start, stop)
                }
            },
        };
        Ok(Self {
            path,
            witness,
            raw,
            reference,
            stop_after_first_mismatch,
        })
    }
}

impl DeterminismTrace {
    pub fn witness(&self) -> &Witness {
        self.witness.get()
    }

    /// Lays the witness out over `world`, checks it against the reference, and creates the file.
    pub fn plan(
        &self,
        world: &World,
        channels: Channels,
        labels: Vec<HookLabel>,
    ) -> PyResult<Plan> {
        let layout = self
            .witness()
            .layout(world, labels.len())
            .map_err(PyValueError::new_err)?;
        let reference = match &self.reference {
            Some(path) => {
                let reader = Reader::open(path).map_err(PyValueError::new_err)?;
                if reader.header.schema != layout.schema {
                    let difference =
                        compare::schema_difference(&reader.header.schema, &layout.schema);
                    return Err(PyValueError::new_err(format!(
                        "the reference witness {} has a different schema: {difference}",
                        path.display()
                    )));
                }
                Some(reader)
            }
            None => None,
        };
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(&self.path)?;
        Ok(Plan {
            file,
            path: self.path.clone(),
            layout,
            labels,
            channels,
            raw: self.raw,
            reference,
            stop_after_first_mismatch: self.stop_after_first_mismatch,
        })
    }
}

/// Replays the witnessed hooks of `trace` through `hooks`, one per recorded hook, over `world`,
/// and compares what each writes with the recording.
pub fn replay_world(
    trace: &std::path::Path,
    witness: &Witness,
    world: &mut World,
    hooks: Vec<Option<Box<dyn PreCommit + Send>>>,
    channels: Channels,
) -> Result<replay::ReplayReport, String> {
    let mut reader = Reader::open(trace)?;
    let recorded = reader.header.labels.hooks.len();
    if hooks.len() != recorded {
        return Err(format!(
            "{} recorded {recorded} pre_commit hooks; pass the same list, got {}",
            trace.display(),
            hooks.len()
        ));
    }
    let layout = witness.layout(world, recorded)?;
    if reader.header.schema != layout.schema {
        let difference = compare::schema_difference(&reader.header.schema, &layout.schema);
        return Err(format!(
            "{} was recorded with a different witness: {difference}",
            trace.display()
        ));
    }
    replay::replay(&mut reader, &layout, world, hooks, channels)
}

/// `elodin.determinism.compare`: the first difference between two witnesses, as JSON.
#[pyfunction]
#[pyo3(name = "_witness_compare", signature = (left, right, stop_at_first = false))]
fn py_compare(
    py: Python<'_>,
    left: PathBuf,
    right: PathBuf,
    stop_at_first: bool,
) -> PyResult<String> {
    let comparison = py
        .allow_threads(|| compare::compare(&left, &right, stop_at_first))
        .map_err(PyValueError::new_err)?;
    Ok(serde_json::to_string(&comparison).expect("comparison serializes"))
}

/// `elodin.determinism.replay_hook`: replays the native hook `symbol` of `library` from `trace`
/// on the recorded rows alone, as JSON.
#[pyfunction]
#[pyo3(name = "_witness_replay_hook", signature = (trace, library, symbol, live = false))]
fn py_replay_hook(trace: PathBuf, library: &str, symbol: &str, live: bool) -> PyResult<String> {
    let mut reader = Reader::open(&trace).map_err(PyValueError::new_err)?;
    let schema = reader.header.schema.clone();
    let labels = &reader.header.labels.hooks;
    let natives = || {
        let natives = labels.iter().filter(|hook| !hook.partitions.is_empty());
        natives.map(|hook| hook.name.as_str()).collect::<Vec<_>>()
    };
    let Some(index) = labels
        .iter()
        .position(|hook| hook.name == symbol && !hook.partitions.is_empty())
    else {
        return Err(PyValueError::new_err(format!(
            "{} has no native hook {symbol:?}; its native hooks are {:?}",
            trace.display(),
            natives()
        )));
    };
    let label = &labels[index];
    let Some(batch) = label
        .batch
        .as_ref()
        .and_then(|name| schema.batches.iter().position(|b| &b.name == name))
    else {
        return Err(PyValueError::new_err(format!(
            "the batch {symbol:?} was loaded with is not in the witness"
        )));
    };
    let workers = label.partitions.len();
    let hook_count = labels.len();
    let mut world = World::default();
    let sources = schema
        .batches
        .iter()
        .map(|batch| {
            let fields = batch.fields.iter().map(|field| {
                let id = ComponentId::new(&format!("{}.{}", batch.name, field.name));
                let rows = field.entities.len();
                let row_bytes = field.row_bytes as usize;
                world.host.insert(
                    id,
                    crate::world::Column {
                        buffer: vec![0; rows * row_bytes],
                        entity_ids: Vec::new(),
                    },
                );
                FieldSource::new(id, (0..rows).collect(), row_bytes)
            });
            fields.collect()
        })
        .collect::<Vec<Vec<FieldSource>>>();
    let fields = schema.batches[batch]
        .fields
        .iter()
        .zip(&sources[batch])
        .map(|(field, source)| {
            Field::new(
                &field.name,
                source.component_id,
                source.rows.clone(),
                source.row_bytes,
                field.access == "write",
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    let defs = schema.channels.iter().map(|channel| ChannelDef {
        name: channel.name.clone(),
        batch: channel.batch as u64,
        rows: schema.batches[channel.batch]
            .fields
            .first()
            .map_or(0, |f| f.entities.len()),
        row_bytes: channel.row_bytes as usize,
        external: channel.external,
    });
    let channels = Channels::new(defs.collect(), !live);
    let hook = NativeStepHook::open(library, symbol, fields, batch as u64, workers)?;
    let mut runner = Some(hook.runner(&channels));
    let hooks = (0..hook_count).map(|n| if n == index { runner.take() } else { None });
    let layout = Layout::new(schema, sources, hook_count);
    let report = replay::replay(&mut reader, &layout, &mut world, hooks.collect(), channels)
        .map_err(PyValueError::new_err)?;
    Ok(serde_json::to_string(&report).expect("replay report serializes"))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Witness>()?;
    m.add_class::<DeterminismTrace>()?;
    m.add_function(wrap_pyfunction!(py_compare, m)?)?;
    m.add_function(wrap_pyfunction!(py_replay_hook, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::compare::compare;
    use super::format::{Header, HookLabel, Labels, Writer};
    use super::*;
    use crate::prepared::HookWrites;

    /// Two entities, `a` and `b`, with a read field `x` and a write field `u`; one hook, which
    /// fills the `ingress` channel and the external `reply` channel.
    fn layout(entities: [u64; 2]) -> Layout {
        let field = |name: &str, access: &str| FieldSchema {
            name: name.into(),
            component: name.into(),
            dtype: "float64".into(),
            shape: vec![1],
            row_bytes: 8,
            access: access.into(),
            entities: entities.to_vec(),
        };
        let batches = vec![BatchSchema {
            name: "pair".into(),
            fields: vec![field("x", "read"), field("u", "write")],
        }];
        let channel = |name: &str, external| ChannelSchema {
            name: name.into(),
            batch: 0,
            row_bytes: 8,
            external,
        };
        let channels = vec![channel("ingress", false), channel("reply", true)];
        let stages = stages(1, None).unwrap();
        let schema = Schema {
            entities: vec![(1, "a".into()), (2, "b".into())],
            records: records(&batches, &channels, &stages),
            batches,
            channels,
            stages,
        };
        let row = |name: &str| FieldSource::new(ComponentId::new(name), vec![0, 1], 8);
        Layout::new(schema, vec![vec![row("x"), row("u")]], 1)
    }

    fn channels() -> Channels {
        let def = |name: &str, external| ChannelDef {
            name: name.into(),
            batch: 1,
            rows: 2,
            row_bytes: 8,
            external,
        };
        Channels::new(vec![def("ingress", false), def("reply", true)], false)
    }

    fn header(layout: &Layout) -> Header {
        let hook = HookLabel {
            name: "exchange".into(),
            batch: Some("pair".into()),
            partitions: vec![(0, 1), (1, 2)],
        };
        let labels = Labels {
            build: "test".into(),
            start_timestamp: 1_000,
            hooks: vec![hook],
        };
        Header::new(layout.schema.clone(), labels)
    }

    fn f64s(values: [f64; 2]) -> Vec<u8> {
        values.map(f64::to_le_bytes).concat()
    }

    /// Captures one tick the way a run does: `x = [t, 2t]`, the hook sends `x` and writes
    /// `u = x / 2`, and the commit records every row. `flip` XORs the low bit of `ingress` row 1.
    fn capture(layout: &Layout, block: &mut Block, tick: u64, flip: bool) {
        let mut world = World::default();
        let t = tick as f64;
        for (name, values) in [("x", [t, 2.0 * t]), ("u", [t / 2.0, t])] {
            let column = crate::world::Column {
                buffer: f64s(values),
                entity_ids: Vec::new(),
            };
            world.host.insert(ComponentId::new(name), column);
        }
        let mut channels = channels();
        let mut ingress = f64s([t, 2.0 * t]);
        ingress[8] ^= u8::from(flip);
        channels.write(0, &ingress).unwrap();
        channels.write(1, &f64s([-t, -t])).unwrap();
        layout.begin(block, tick, 1_000 + 1_000 * tick as i64);
        layout.capture_systems(block, &world);
        layout.capture_hook(
            0,
            block,
            &world,
            &channels,
            &HookWrites::default().deferred,
            1_000,
        );
        layout.capture_commit(block, |_| true);
        layout.digest(block);
    }

    /// Writes `ticks` ticks to `path`, keeping the exact bytes of those `raw` keeps.
    fn record(path: &Path, layout: &Layout, ticks: u64, raw: Raw, flip_at: Option<u64>) -> String {
        let file = std::fs::File::create(path).unwrap();
        let mut writer = Writer::new(file, &header(layout)).unwrap();
        let mut block = Block::default();
        for tick in 0..ticks {
            capture(layout, &mut block, tick, flip_at == Some(tick));
            writer.tick(&block, raw.keeps(tick)).unwrap();
        }
        writer.finish(None).unwrap().root
    }

    #[test]
    fn a_witness_reads_back_what_was_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.witness");
        let layout = layout([1, 2]);
        let root = record(&path, &layout, 5, Raw::Window(1, 3), None);

        let mut reader = Reader::open(&path).unwrap();
        let mut tick = format::Tick::default();
        let mut expected = Block::default();
        let mut raw = Vec::new();
        while reader.next(&mut tick).unwrap() {
            capture(&layout, &mut expected, tick.block.tick, false);
            assert_eq!(tick.block.digests, expected.digests);
            assert_eq!(tick.block.timestamp, expected.timestamp);
            if tick.raw {
                assert_eq!(tick.block.bytes, expected.bytes);
                assert_eq!(tick.block.ends, expected.ends);
                raw.push(tick.block.tick);
            }
        }
        assert_eq!(raw, [1, 2]);
        assert_eq!(reader.trailer.unwrap().root, root);
        // after_systems: x, u; after_hook:0: ingress, u, deferred, reply; after_commit: x, u.
        assert_eq!(layout.schema.hashed_records().len(), 7);
    }

    #[test]
    fn the_v1_format_is_stable() {
        let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/witness/golden-v1.bin");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.witness");
        record(&path, &layout([1, 2]), 3, Raw::Window(1, 2), None);
        if std::env::var_os("ELODIN_BLESS_WITNESS").is_some() {
            std::fs::copy(&path, &golden).unwrap();
        }
        assert_eq!(
            std::fs::read(&path).unwrap(),
            std::fs::read(&golden).unwrap()
        );
        let mut reader = Reader::open(&golden).unwrap();
        while reader.next(&mut format::Tick::default()).unwrap() {}
        assert_eq!((reader.ticks, reader.raw_ticks), (3, 1));
    }

    #[test]
    fn other_format_versions_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.witness");
        record(&path, &layout([1, 2]), 1, Raw::Never, None);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        let err = Reader::open(&path).err().unwrap();
        assert!(
            err.ends_with("has witness format version 2; this build reads version 1"),
            "{err}"
        );
    }

    #[test]
    fn compare_descends_to_the_first_differing_byte() {
        let dir = tempfile::tempdir().unwrap();
        let (left, right) = (dir.path().join("left"), dir.path().join("right"));
        let layout = layout([1, 2]);
        record(&left, &layout, 6, Raw::Always, None);
        record(&right, &layout, 6, Raw::Window(3, 4), Some(3));

        let same = compare(&left, &left, false).unwrap();
        assert!(same.identical && same.ticks == 6);

        let found = compare(&left, &right, true).unwrap();
        let mismatch = found.first_mismatch.unwrap();
        let expected = Mismatch {
            tick: 3,
            stage: Some("after_hook:0".into()),
            hook: Some("exchange".into()),
            kind: Some("channel"),
            record: Some("ingress".into()),
            entity: Some("b".into()),
            row: Some(1),
            byte: Some(0),
            left: Some(f64s([3.0, 6.0])[8]),
            right: Some(f64s([3.0, 6.0])[8] ^ 1),
            note: None,
        };
        assert_eq!(mismatch, expected);
        assert_eq!((found.ticks, found.mismatched_ticks), (4, 1));

        // Without exact bytes on both sides, the record is all a comparison can name.
        let digests_only = dir.path().join("digests");
        record(&digests_only, &layout, 6, Raw::Never, Some(3));
        let mismatch = compare(&left, &digests_only, false)
            .unwrap()
            .first_mismatch
            .unwrap();
        assert_eq!(
            (mismatch.record.as_deref(), mismatch.row),
            (Some("ingress"), None)
        );
    }

    #[test]
    fn missing_ticks_and_different_schemas_fail_comparison() {
        let dir = tempfile::tempdir().unwrap();
        let (full, short) = (dir.path().join("full"), dir.path().join("short"));
        record(&full, &layout([1, 2]), 4, Raw::Never, None);
        record(&short, &layout([1, 2]), 2, Raw::Never, None);
        let comparison = compare(&full, &short, false).unwrap();
        let mismatch = comparison.first_mismatch.unwrap();
        assert_eq!(mismatch.tick, 2);
        assert_eq!(
            mismatch.note.unwrap(),
            "tick 2 and those after it are missing from right"
        );

        let swapped = dir.path().join("swapped");
        record(&swapped, &layout([2, 1]), 4, Raw::Never, None);
        let err = compare(&full, &swapped, false).unwrap_err();
        assert!(
            err.contains("pair.x has a different entity order: row 0 is a and b"),
            "{err}"
        );
    }

    #[test]
    fn a_witness_thread_that_falls_behind_fails_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let layout = layout([1, 2]);
        let reference = dir.path().join("reference");
        record(&reference, &layout, 1, Raw::Never, None);
        // A reference that stops after its header holds the witness thread at its first tick.
        let header = std::fs::read(&reference).unwrap();
        let header_len = 16 + u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let fifo = dir.path().join("fifo");
        let made = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(made.success());
        let feeder = fifo.clone();
        std::thread::spawn(move || {
            use std::io::Write;
            let mut pipe = std::fs::OpenOptions::new()
                .write(true)
                .open(feeder)
                .unwrap();
            pipe.write_all(&header[..header_len]).unwrap();
            std::thread::park();
        });
        let plan = Plan {
            file: std::fs::File::create(dir.path().join("run.witness")).unwrap(),
            path: dir.path().join("run.witness"),
            layout,
            labels: header_labels(),
            channels: channels(),
            raw: Raw::Never,
            reference: Some(Reader::open(&fifo).unwrap()),
            stop_after_first_mismatch: false,
        };
        let mut recorder = plan.start(impeller::types::Timestamp(1_000), |_, _| None);
        let world = {
            let mut world = World::default();
            for name in ["x", "u"] {
                let column = crate::world::Column {
                    buffer: vec![0; 16],
                    entity_ids: Vec::new(),
                };
                world.host.insert(ComponentId::new(name), column);
            }
            world
        };
        let err = (0..)
            .find_map(|tick| {
                recorder.after_systems(tick, impeller::types::Timestamp(1_000), &world);
                recorder.after_hook(0, &world, &HookWrites::default());
                recorder.after_commit(&[]).err()
            })
            .unwrap();
        assert_eq!(
            err,
            "the witness thread fell 1024 ticks behind the simulation"
        );
        std::mem::forget(recorder);
    }

    fn header_labels() -> Vec<HookLabel> {
        vec![HookLabel {
            name: "exchange".into(),
            batch: Some("pair".into()),
            partitions: vec![(0, 1), (1, 2)],
        }]
    }
}
