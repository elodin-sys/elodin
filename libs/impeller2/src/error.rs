use thiserror::Error;
#[derive(Error, Debug, Clone)]
#[cfg_attr(feature = "std", derive(miette::Diagnostic))]
pub enum Error {
    #[error("buffer underflow")]
    #[cfg_attr(
        feature = "std",
        diagnostic(
            code(impeller::buf_underflow),
            help("ran out of room when reading from buffer")
        )
    )]
    BufferUnderflow,

    #[error("buffer overflow")]
    #[cfg_attr(
        feature = "std",
        diagnostic(
            code(impeller::buf_overflow),
            help("ran out of room while writing to buffer")
        )
    )]
    BufferOverflow,

    #[error("offset overflow")]
    #[cfg_attr(
        feature = "std",
        diagnostic(
            code(impeller::offset_overflowe),
            help("offset was larger than platform's usize")
        )
    )]
    OffsetOverflow,

    #[error("incorrect aligned input")]
    #[cfg_attr(
        feature = "std",
        diagnostic(code(impeller::alignment), help("alignment was incorrect for buf"))
    )]
    Alignment,

    #[error("incorrect component data")]
    #[cfg_attr(
        feature = "std",
        diagnostic(
            code(impeller::invalid_component_data),
            help("component data was invalid")
        )
    )]
    InvalidComponentData,
}

impl<A, B: ?Sized> From<zerocopy::CastError<A, B>> for Error {
    fn from(value: zerocopy::CastError<A, B>) -> Self {
        match value {
            zerocopy::ConvertError::Alignment(_) => Error::Alignment,
            zerocopy::ConvertError::Size(_) => Error::BufferUnderflow,
            zerocopy::ConvertError::Validity(_) => unreachable!(),
        }
    }
}

impl<A, B: ?Sized + zerocopy::TryFromBytes> From<zerocopy::TryCastError<A, B>> for Error {
    fn from(value: zerocopy::TryCastError<A, B>) -> Self {
        match value {
            zerocopy::TryCastError::Alignment(_) => Error::Alignment,
            zerocopy::TryCastError::Size(_) => Error::BufferUnderflow,
            zerocopy::TryCastError::Validity(_) => Error::InvalidComponentData,
        }
    }
}
