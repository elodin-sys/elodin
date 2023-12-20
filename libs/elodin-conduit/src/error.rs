use thiserror::Error;
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
}
