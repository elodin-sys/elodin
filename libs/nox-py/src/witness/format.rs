//! The witness file: a JSON header, one frame per tick, and a JSON trailer, all little-endian.
//!
//! ```text
//! file    := "ELWITNES" version:u32 header_len:u32 header_json frame* trailer
//! frame   := 0x01 tick:u64 timestamp:i64 digest:[u8; 32]* raw:u8 (len:u32 bytes)*
//! trailer := 0x02 json_len:u32 trailer_json
//! ```
//!
//! A frame has one BLAKE3-256 digest per hashed record of the schema, in record order, and with
//! `raw` set, the exact bytes of every record. A stage's digest is the hash of its records'
//! digests; a tick's is the hash of the tick, its simulation time and its stages' digests; the
//! root is the hash of the schema digest and every tick's digest.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::Block;

pub const MAGIC: [u8; 8] = *b"ELWITNES";
pub const VERSION: u32 = 1;
const TICK: u8 = 1;
const TRAILER: u8 = 2;

pub type Digest = [u8; 32];

/// What a witness records. Two witnesses compare only when their schemas are equal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    /// Every entity a field has a row of, by id.
    pub entities: Vec<(u64, String)>,
    pub batches: Vec<BatchSchema>,
    pub channels: Vec<ChannelSchema>,
    pub stages: Vec<String>,
    pub records: Vec<RecordSchema>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchSchema {
    pub name: String,
    pub fields: Vec<FieldSchema>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldSchema {
    pub name: String,
    pub component: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub row_bytes: u64,
    pub access: String,
    /// The entity of each row.
    pub entities: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelSchema {
    pub name: String,
    /// Index of the batch whose rows it has.
    pub batch: usize,
    pub row_bytes: u64,
    /// Bytes from outside the simulation: kept for replay, never hashed.
    pub external: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Field,
    Channel,
    External,
    Deferred,
    Recorded,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordSchema {
    pub stage: usize,
    pub kind: Kind,
    pub name: String,
    /// Index of the batch whose rows it has; none for deferred writes.
    pub batch: Option<usize>,
    /// Index of the field in its batch, or of the channel.
    pub index: usize,
    pub row_bytes: u64,
}

impl RecordSchema {
    pub fn hashed(&self) -> bool {
        self.kind != Kind::External
    }
}

impl Schema {
    pub fn digest(&self) -> Digest {
        *blake3::hash(&serde_json::to_vec(self).expect("schema serializes")).as_bytes()
    }

    /// Hashed records per stage.
    pub fn stage_leaves(&self) -> Vec<usize> {
        (0..self.stages.len())
            .map(|stage| {
                let records = self.records.iter();
                records.filter(|r| r.stage == stage && r.hashed()).count()
            })
            .collect()
    }

    /// Index in `records` of each digest.
    pub fn hashed_records(&self) -> Vec<usize> {
        let records = self.records.iter().enumerate();
        records
            .filter(|(_, r)| r.hashed())
            .map(|(i, _)| i)
            .collect()
    }

    pub fn entity_name(&self, id: u64) -> String {
        match self.entities.binary_search_by_key(&id, |(id, _)| *id) {
            Ok(index) => self.entities[index].1.clone(),
            Err(_) => format!("entity{id}"),
        }
    }
}

/// Stored and printed, never hashed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Labels {
    pub build: String,
    pub start_timestamp: i64,
    pub hooks: Vec<HookLabel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HookLabel {
    pub name: String,
    /// The witnessed batch a native hook was loaded with.
    pub batch: Option<String>,
    /// The rows each call of a native hook owns; empty for a Python hook.
    pub partitions: Vec<(u64, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub format: String,
    pub version: u32,
    pub hash: String,
    pub schema_digest: String,
    pub schema: Schema,
    pub labels: Labels,
}

impl Header {
    pub fn new(schema: Schema, labels: Labels) -> Self {
        Self {
            format: "elodin.witness".into(),
            version: VERSION,
            hash: "blake3-256 over little-endian bytes".into(),
            schema_digest: hex(&schema.digest()),
            schema,
            labels,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trailer {
    pub ticks: u64,
    pub root: String,
    pub failure: Option<String>,
}

pub fn hex(digest: &Digest) -> String {
    blake3::Hash::from_bytes(*digest).to_hex().to_string()
}

pub fn tick_digest(stage_leaves: &[usize], tick: u64, sim_time: i64, digests: &[Digest]) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&tick.to_le_bytes());
    hasher.update(&sim_time.to_le_bytes());
    let mut leaves = digests;
    for &count in stage_leaves {
        let (stage, rest) = leaves.split_at(count);
        hasher.update(blake3::hash(stage.as_flattened()).as_bytes());
        leaves = rest;
    }
    *hasher.finalize().as_bytes()
}

pub struct Writer {
    out: BufWriter<File>,
    stage_leaves: Vec<usize>,
    start: i64,
    root: blake3::Hasher,
    pub ticks: u64,
    pub raw_ticks: u64,
    pub bytes: u64,
}

impl Writer {
    pub fn new(file: File, header: &Header) -> io::Result<Self> {
        let json = serde_json::to_vec(header)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(&MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&(json.len() as u32).to_le_bytes())?;
        out.write_all(&json)?;
        let mut root = blake3::Hasher::new();
        root.update(&header.schema.digest());
        Ok(Self {
            out,
            stage_leaves: header.schema.stage_leaves(),
            start: header.labels.start_timestamp,
            root,
            ticks: 0,
            raw_ticks: 0,
            bytes: (MAGIC.len() + 8 + json.len()) as u64,
        })
    }

    /// Appends `block`'s tick with its digests, and with `raw`, its records' bytes.
    pub fn tick(&mut self, block: &Block, raw: bool) -> io::Result<()> {
        let sim_time = block.timestamp - self.start;
        let digest = tick_digest(&self.stage_leaves, block.tick, sim_time, &block.digests);
        self.root.update(&digest);
        self.out.write_all(&[TICK])?;
        self.out.write_all(&block.tick.to_le_bytes())?;
        self.out.write_all(&block.timestamp.to_le_bytes())?;
        self.out.write_all(block.digests.as_flattened())?;
        self.out.write_all(&[raw.into()])?;
        self.bytes += 18 + 32 * block.digests.len() as u64;
        if raw {
            for index in 0..block.ends.len() {
                let record = block.record(index);
                self.out.write_all(&(record.len() as u32).to_le_bytes())?;
                self.out.write_all(record)?;
                self.bytes += 4 + record.len() as u64;
            }
            self.raw_ticks += 1;
        }
        self.ticks += 1;
        Ok(())
    }

    pub fn finish(&mut self, failure: Option<String>) -> io::Result<Trailer> {
        let trailer = Trailer {
            ticks: self.ticks,
            root: self.root.finalize().to_hex().to_string(),
            failure,
        };
        let json = serde_json::to_vec(&trailer)?;
        self.out.write_all(&[TRAILER])?;
        self.out.write_all(&(json.len() as u32).to_le_bytes())?;
        self.out.write_all(&json)?;
        self.out.flush()?;
        self.bytes += 5 + json.len() as u64;
        Ok(trailer)
    }
}

/// One tick read back; `block` has its records' bytes only when `raw`.
#[derive(Default)]
pub struct Tick {
    pub block: Block,
    pub raw: bool,
}

pub struct Reader {
    input: BufReader<File>,
    pub path: PathBuf,
    pub header: Header,
    pub trailer: Option<Trailer>,
    stage_leaves: Vec<usize>,
    digests: usize,
    records: usize,
    root: blake3::Hasher,
    pub ticks: u64,
    pub raw_ticks: u64,
}

impl Reader {
    pub fn open(path: &Path) -> Result<Self, String> {
        let name = path.display();
        let file = File::open(path).map_err(|err| format!("cannot open {name}: {err}"))?;
        let mut input = BufReader::with_capacity(1 << 20, file);
        let mut magic = [0; 8];
        let mut word = [0; 4];
        input
            .read_exact(&mut magic)
            .and_then(|()| input.read_exact(&mut word))
            .map_err(|_| format!("{name} is not an Elodin witness"))?;
        if magic != MAGIC {
            return Err(format!("{name} is not an Elodin witness"));
        }
        let version = u32::from_le_bytes(word);
        if version != VERSION {
            return Err(format!(
                "{name} has witness format version {version}; this build reads version {VERSION}"
            ));
        }
        let mut json = vec![0; read_u32(&mut input).map_err(|err| truncated(path, err))? as usize];
        input
            .read_exact(&mut json)
            .map_err(|err| truncated(path, err))?;
        let header: Header = serde_json::from_slice(&json)
            .map_err(|err| format!("{name} has an unreadable header: {err}"))?;
        if header.schema_digest != hex(&header.schema.digest()) {
            return Err(format!(
                "{name} has a header whose schema digest does not match"
            ));
        }
        let mut root = blake3::Hasher::new();
        root.update(&header.schema.digest());
        Ok(Self {
            input,
            path: path.to_path_buf(),
            stage_leaves: header.schema.stage_leaves(),
            digests: header.schema.hashed_records().len(),
            records: header.schema.records.len(),
            header,
            trailer: None,
            root,
            ticks: 0,
            raw_ticks: 0,
        })
    }

    /// Reads the next tick into `tick`; false at the trailer or the end of the file.
    pub fn next(&mut self, tick: &mut Tick) -> Result<bool, String> {
        let mut tag = [0];
        match self.input.read(&mut tag) {
            Ok(0) => return Ok(false),
            Ok(_) => {}
            Err(err) => return Err(truncated(&self.path, err)),
        }
        match tag[0] {
            TICK => self
                .read_tick(tick)
                .map_err(|err| truncated(&self.path, err))?,
            TRAILER => {
                self.read_trailer()?;
                return Ok(false);
            }
            other => {
                return Err(format!(
                    "{} has an unknown frame {other} after tick {}",
                    self.path.display(),
                    self.ticks
                ));
            }
        }
        Ok(true)
    }

    fn read_tick(&mut self, tick: &mut Tick) -> io::Result<()> {
        let block = &mut tick.block;
        block.clear();
        block.tick = read_u64(&mut self.input)?;
        block.timestamp = read_u64(&mut self.input)? as i64;
        block.digests.resize(self.digests, [0; 32]);
        self.input.read_exact(block.digests.as_flattened_mut())?;
        let mut raw = [0];
        self.input.read_exact(&mut raw)?;
        tick.raw = raw[0] != 0;
        if tick.raw {
            for _ in 0..self.records {
                let len = read_u32(&mut self.input)? as usize;
                let start = block.bytes.len();
                block.bytes.resize(start + len, 0);
                self.input.read_exact(&mut block.bytes[start..])?;
                block.ends.push(block.bytes.len());
            }
            self.raw_ticks += 1;
        }
        let sim_time = block.timestamp - self.header.labels.start_timestamp;
        let digest = tick_digest(&self.stage_leaves, block.tick, sim_time, &block.digests);
        self.root.update(&digest);
        self.ticks += 1;
        Ok(())
    }

    fn read_trailer(&mut self) -> Result<(), String> {
        let read = |input: &mut BufReader<File>| -> io::Result<Vec<u8>> {
            let mut json = vec![0; read_u32(input)? as usize];
            input.read_exact(&mut json)?;
            Ok(json)
        };
        let json = read(&mut self.input).map_err(|err| truncated(&self.path, err))?;
        let trailer: Trailer = serde_json::from_slice(&json)
            .map_err(|err| format!("{} has an unreadable trailer: {err}", self.path.display()))?;
        if trailer.ticks != self.ticks || trailer.root != self.root.finalize().to_hex().as_str() {
            return Err(format!(
                "{} is corrupt: its trailer does not match its ticks",
                self.path.display()
            ));
        }
        self.trailer = Some(trailer);
        Ok(())
    }
}

fn truncated(path: &Path, err: io::Error) -> String {
    format!("{} is truncated or unreadable: {err}", path.display())
}

fn read_u32(input: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(input: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}
