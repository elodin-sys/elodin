use cpp::{cpp, cpp_class};
use cxx::{CxxString, UniquePtr};

cpp! {{
    #include "xla/statusor.h"
    using namespace xla;
}}

/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Incorrect number of elements.
    #[error("wrong element count {element_count} for dims {dims:?}")]
    WrongElementCount {
        dims: Vec<i64>,
        element_count: usize,
    },

    /// Error from the xla C++ library.
    #[error("xla error {msg}\n{backtrace}")]
    XlaError { msg: String, backtrace: String },

    #[error("unexpected element type {0}")]
    UnexpectedElementType(i32),

    #[error("unexpected number of dimensions, expected: {expected}, got: {got} ({dims:?})")]
    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        dims: Vec<i64>,
    },

    #[error("not an element type, got: {got:?}")]
    NotAnElementType { got: crate::PrimitiveType },

    #[error("not an array, expected: {expected:?}, got: {got:?}")]
    NotAnArray {
        expected: Option<usize>,
        got: crate::Shape,
    },

    #[error("cannot handle unsupported shapes {shape:?}")]
    UnsupportedShape { shape: crate::Shape },

    #[error("unexpected number of tuple elements, expected: {expected}, got: {got}")]
    UnexpectedNumberOfElemsInTuple { expected: usize, got: usize },

    #[error("element type mismatch, on-device: {on_device:?}, on-host: {on_host:?}")]
    ElementTypeMismatch {
        on_device: crate::ElementType,
        on_host: crate::ElementType,
    },

    #[error("unsupported element type for {op}: {ty:?}")]
    UnsupportedElementType {
        ty: crate::PrimitiveType,
        op: &'static str,
    },

    #[error(
        "target buffer is too large, offset {offset}, shape {shape:?}, buffer_len: {buffer_len}"
    )]
    TargetBufferIsTooLarge {
        offset: usize,
        shape: crate::ArrayShape,
        buffer_len: usize,
    },

    #[error("binary buffer is too large, element count {element_count}, buffer_len: {buffer_len}")]
    BinaryBufferIsTooLarge {
        element_count: usize,
        buffer_len: usize,
    },

    #[error("empty literal")]
    EmptyLiteral,

    #[error("index out of bounds {index}, rank {rank}")]
    IndexOutOfBounds { index: i64, rank: usize },

    #[error("npy/npz error {0}")]
    Npy(String),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    #[error("cannot create literal with shape {ty:?} {dims:?} from bytes data with len {data_len_in_bytes}")]
    CannotCreateLiteralWithData {
        data_len_in_bytes: usize,
        ty: crate::PrimitiveType,
        dims: Vec<usize>,
    },

    #[error("invalid dimensions in matmul, lhs: {lhs_dims:?}, rhs: {rhs_dims:?}, {msg}")]
    MatMulIncorrectDims {
        lhs_dims: Vec<i64>,
        rhs_dims: Vec<i64>,
        msg: &'static str,
    },
    #[error("podcast error {0}")]
    PodCastError(bytemuck::PodCastError),
}

pub type Result<T> = std::result::Result<T, Error>;

cpp_class!(pub unsafe struct Status as "Status");

impl Status {
    pub fn ok() -> Self {
        unsafe {
            cpp!([] -> Status as "Status" {
                return Status();
            })
        }
    }

    pub fn is_ok(&self) -> bool {
        unsafe {
            cpp!([self as "const Status*"] -> bool as "bool" {
                return self->ok();
            })
        }
    }

    pub fn to_result(&self) -> Result<()> {
        if self.is_ok() {
            Ok(())
        } else {
            let msg = unsafe {
                cpp!([self as "Status*"] -> UniquePtr<CxxString> as "std::unique_ptr<std::string>" {
                    return std::make_unique<std::string>(std::string(self->message()));
                })
            };
            let msg = msg
                .as_ref()
                .and_then(|msg| msg.to_str().ok())
                .map(|msg| msg.to_string())
                .unwrap_or_default();
            let backtrace = std::backtrace::Backtrace::capture().to_string();
            Err(Error::XlaError { msg, backtrace })
        }
    }
}
