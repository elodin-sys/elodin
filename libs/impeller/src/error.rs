use crate::StreamId;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[cfg(feature = "std")]
    #[error("io {0}")]
    Io(#[from] std::io::Error),
    #[error("buffer overflow")]
    BufferOverflow,
    #[error("parsing error")]
    ParsingError,
    #[error("send error")]
    SendError,
    #[error("component values need contiguous memory to be encoded")]
    NonContiguousMemory,
    #[error("shape must have less than 255 dims")]
    TooManyDims,
    #[error("eof")]
    EOF,
    #[error("checked cast error")]
    CheckedCast,
    #[error("shape error {0}")]
    ShapeError(#[from] ndarray::ShapeError),
    #[error("unknown primitive type")]
    UnknownPrimitiveTy,
    #[error("postcard {0}")]
    Postcard(#[from] postcard::Error),
    #[error("stream not found {0:?}")]
    StreamNotFound(StreamId),
    #[error("invalid alignment")]
    InvalidAlignment,
    #[error("value size mismatch")]
    ValueSizeMismatch,
    #[error("connection closed")]
    ConnectionClosed,
    #[error("non utf8 path")]
    NonUtf8Path,
    #[cfg(feature = "polars")]
    #[error("polars {0}")]
    Polars(#[from] ::polars::error::PolarsError),
    #[error("serde_json {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid query")]
    InvalidQuery,
    #[error("component not found")]
    ComponentNotFound,
    #[error("asset not found")]
    AssetNotFound,
}

impl From<try_buf::ErrorKind> for Error {
    fn from(value: try_buf::ErrorKind) -> Self {
        match value {
            try_buf::ErrorKind::EOF => Error::EOF,
        }
    }
}

#[cfg(all(feature = "std", feature = "flume"))]
impl<T> From<flume::SendError<T>> for Error {
    fn from(_: flume::SendError<T>) -> Self {
        Self::SendError
    }
}
