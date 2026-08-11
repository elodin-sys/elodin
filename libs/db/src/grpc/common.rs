use std::time::Duration;

use impeller2::types::PrimType as DbPrimType;
use impeller2_wkt::{
    ComponentMetadata as DbComponentMetadata, DbConfig as DbDbConfig,
    MsgMetadata as DbMessageMetadata,
};
use tonic::Status;

use super::v1;
use crate::Error;

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

pub(super) fn duration(value: u64) -> Result<Duration, Status> {
    if value == 0 {
        return Err(Status::invalid_argument("duration must be non-zero"));
    }
    Ok(Duration::from_nanos(value))
}

pub(super) fn internal(error: impl std::fmt::Display) -> Status {
    Status::internal(error.to_string())
}

pub(super) fn db_error(error: Error) -> Status {
    match error {
        Error::ComponentNotFound(_) | Error::MsgNotFound(_) => Status::not_found(error.to_string()),
        Error::Io(ref io) if io.kind() == std::io::ErrorKind::PermissionDenied => {
            Status::permission_denied(error.to_string())
        }
        _ => internal(error),
    }
}
