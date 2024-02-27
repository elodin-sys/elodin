use thiserror::Error;

use crate::StreamId;
#[derive(Debug, Error)]
pub enum Error {
    #[error("io {0}")]
    Io(#[from] std::io::Error),
    #[error("entity and component iters length must match")]
    EntityComponentLengthMismatch,
    #[error("buffer overflow")]
    BufferOverflow,
    #[error("parsing error")]
    ParsingError,
    #[error("send error")]
    SendError,
    #[error("component values need contigous memory to be encoded")]
    NonContigousMemory,
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
    #[error("connection closed")]
    ConnectionClosed,
    #[error("non utf8 path")]
    NonUtf8Path,
}

impl From<try_buf::ErrorKind> for Error {
    fn from(value: try_buf::ErrorKind) -> Self {
        match value {
            try_buf::ErrorKind::EOF => Error::EOF,
        }
    }
}

impl<T> From<flume::SendError<T>> for Error {
    fn from(_: flume::SendError<T>) -> Self {
        Self::SendError
    }
}
