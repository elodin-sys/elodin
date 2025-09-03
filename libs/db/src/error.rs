use std::io;

use impeller2::types::{ComponentId, PacketId};
use impeller2_wkt::{ErrorResponse, StreamId};
use thiserror::Error;
#[derive(Debug, Error)]
pub enum Error {
    #[error("map overflow")]
    MapOverflow,
    #[error("stellarator {0}")]
    Stellar(stellarator::Error),
    #[error("io {0}")]
    Io(#[from] std::io::Error),
    #[error("impeller_stella {0}")]
    ImpellerStella(impeller2_stellar::Error),
    #[error("impeller {0}")]
    Impeller(#[from] impeller2::error::Error),
    #[error("component not found {0}")]
    ComponentNotFound(ComponentId),
    #[error("postcard error {0}")]
    Postcard(#[from] postcard::Error),
    #[error("invalid component id")]
    InvalidComponentId,
    #[error("time travel - you tried to push a time stamp in the past")]
    TimeTravel,
    #[error("datafusion {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),
    #[error("arrow  {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("stream not found {0}")]
    StreamNotFound(StreamId),
    #[error("time range out of bounds")]
    TimeRangeOutOfBounds,
    #[error("invalid msg id")]
    InvalidMsgId,
    #[error("msg not found {0:?}")]
    MsgNotFound(PacketId),
    #[error("bad message")]
    BadMessage,
    #[error("unsupported archive format")]
    UnsupportedArchiveFormat,
    #[cfg(feature = "parquet")]
    #[error("parquet {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("schema mismatch")]
    SchemaMismatch,
}

impl From<impeller2_stellar::Error> for Error {
    fn from(value: impeller2_stellar::Error) -> Self {
        match value {
            impeller2_stellar::Error::Stellar(error) => Error::from(error),
            err => Error::ImpellerStella(err),
        }
    }
}

impl From<stellarator::Error> for Error {
    fn from(value: stellarator::Error) -> Self {
        match value {
            stellarator::Error::Io(err) => Error::from(err),
            err => Error::Stellar(err),
        }
    }
}

impl Error {
    pub fn is_stream_closed(&self) -> bool {
        match self {
            Error::Stellar(stellarator::Error::EOF) => true,
            Error::Io(err)
                if err.kind() == io::ErrorKind::BrokenPipe
                    || err.kind() == io::ErrorKind::ConnectionReset =>
            {
                true
            }
            _ => false,
        }
    }
}

impl From<Error> for ErrorResponse {
    fn from(val: Error) -> Self {
        ErrorResponse {
            description: val.to_string(),
        }
    }
}
