use std::{collections::HashMap, path::PathBuf, sync::Mutex};

use impeller2::types::{PrimType as DbPrimType, Timestamp};
use impeller2_wkt::{
    ComponentMetadata as DbComponentMetadata, DbConfig as DbDbConfig,
    MsgMetadata as DbMessageMetadata,
};
use sha2::{Digest, Sha256};
use tonic::{Code, Status};
use tonic_types::{ErrorDetails, StatusExt};

use super::v1;
use crate::Error;

pub(super) const ERROR_DOMAIN: &str = "db.elodin.systems";

pub(super) fn status_with_reason(code: Code, message: String, reason: &str) -> Status {
    Status::with_error_details(
        code,
        message,
        ErrorDetails::with_error_info(reason, ERROR_DOMAIN, HashMap::<String, String>::new()),
    )
}

// Record time lives on a microsecond grid: writes floor into their bucket,
// half-open [start_ns, end_ns) reads select buckets whose full span is inside.
pub(super) fn record_timestamp(ns: i64) -> Timestamp {
    Timestamp(ns.div_euclid(1000))
}

// Signed div_ceil is unstable (int_roundings); this form cannot overflow.
fn ceil_us(ns: i64) -> i64 {
    ns.div_euclid(1000) + i64::from(ns.rem_euclid(1000) != 0)
}

pub(super) fn range_start(start_ns: Option<i64>) -> Timestamp {
    Timestamp(start_ns.map_or(i64::MIN, ceil_us))
}

pub(super) fn range_end_exclusive(end_ns: Option<i64>) -> Timestamp {
    Timestamp(end_ns.map_or(i64::MAX, ceil_us))
}

pub(super) fn row_limit(limit: Option<u64>) -> Result<usize, Status> {
    match limit {
        None => Ok(usize::MAX),
        Some(0) => Err(Status::invalid_argument(
            "limit must be >= 1; omit it for unlimited",
        )),
        Some(value) => Ok(usize::try_from(value).unwrap_or(usize::MAX)),
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct SessionKey {
    pub(super) client_name: String,
    pub(super) client_instance_id: Vec<u8>,
}

// Per-session resume sequences, cached in memory and persisted as dotfiles
// beside the database so resume positions survive server restarts.
pub(super) struct SessionResume {
    db_path: PathBuf,
    prefix: &'static str,
    cache: Mutex<HashMap<SessionKey, u64>>,
}

impl SessionResume {
    pub(super) fn new(db_path: PathBuf, prefix: &'static str) -> Self {
        Self {
            db_path,
            prefix,
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn get(&self, key: &SessionKey) -> u64 {
        if let Some(sequence) = self.cache.lock().unwrap().get(key).copied() {
            return sequence;
        }
        let sequence = std::fs::read_to_string(self.path(key))
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        self.cache.lock().unwrap().insert(key.clone(), sequence);
        sequence
    }

    pub(super) fn remember(&self, key: &SessionKey, sequence: u64) {
        let mut cache = self.cache.lock().unwrap();
        let stored = cache.entry(key.clone()).or_default();
        *stored = (*stored).max(sequence);
    }

    pub(super) fn persist(&self, key: &SessionKey, sequence: u64) -> std::io::Result<()> {
        let path = self.path(key);
        std::fs::create_dir_all(path.parent().unwrap())?;
        let temporary = path.with_extension("tmp");
        std::fs::write(&temporary, sequence.to_string())?;
        std::fs::rename(temporary, path)?;
        self.remember(key, sequence);
        Ok(())
    }

    fn path(&self, key: &SessionKey) -> PathBuf {
        let mut hash = Sha256::new();
        hash.update(key.client_name.len().to_le_bytes());
        hash.update(key.client_name.as_bytes());
        hash.update(key.client_instance_id.len().to_le_bytes());
        hash.update(&key.client_instance_id);
        self.db_path
            .join(format!("{}{:x}", self.prefix, hash.finalize()))
    }
}

pub(super) fn prim_type(value: DbPrimType) -> v1::PrimType {
    match value {
        DbPrimType::U8 => v1::PrimType::U8,
        DbPrimType::U16 => v1::PrimType::U16,
        DbPrimType::U32 => v1::PrimType::U32,
        DbPrimType::U64 => v1::PrimType::U64,
        DbPrimType::I8 => v1::PrimType::I8,
        DbPrimType::I16 => v1::PrimType::I16,
        DbPrimType::I32 => v1::PrimType::I32,
        DbPrimType::I64 => v1::PrimType::I64,
        DbPrimType::Bool => v1::PrimType::Bool,
        DbPrimType::F32 => v1::PrimType::F32,
        DbPrimType::F64 => v1::PrimType::F64,
    }
}

pub(super) fn db_config(value: &DbDbConfig) -> v1::DbConfig {
    v1::DbConfig {
        recording: value.recording,
        default_stream_time_step_ns: value.default_stream_time_step.as_nanos() as u64,
        metadata: value.metadata.clone(),
    }
}

pub(super) fn component_metadata(value: &DbComponentMetadata) -> v1::ComponentMetadata {
    v1::ComponentMetadata {
        name: value.name.clone(),
        metadata: value.metadata.clone(),
    }
}

pub(super) fn element_names(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

pub(super) fn message_metadata(value: &DbMessageMetadata) -> Result<v1::MessageMetadata, Status> {
    Ok(v1::MessageMetadata {
        name: value.name.clone(),
        postcard_schema: postcard::to_allocvec(&value.schema).map_err(internal)?,
        metadata: value.metadata.clone(),
    })
}

pub(super) fn internal(error: impl std::fmt::Display) -> Status {
    Status::internal(error.to_string())
}

pub(super) fn db_error(error: Error) -> Status {
    match error {
        Error::ComponentNotFound(_) => {
            status_with_reason(Code::NotFound, error.to_string(), "COMPONENT_NOT_FOUND")
        }
        Error::MsgNotFound(_) => {
            status_with_reason(Code::NotFound, error.to_string(), "MESSAGE_NOT_FOUND")
        }
        Error::Io(ref io) if io.kind() == std::io::ErrorKind::PermissionDenied => {
            Status::permission_denied(error.to_string())
        }
        _ => internal(error),
    }
}
