use std::fmt;

use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;

/// Browser `DOMException`, kept as the JS object until displayed.
#[derive(Debug, Clone)]
pub struct DomException(web_sys::DomException);

impl DomException {
    pub fn into_inner(self) -> web_sys::DomException {
        self.0
    }

    pub fn as_inner(&self) -> &web_sys::DomException {
        &self.0
    }
}

impl fmt::Display for DomException {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.0.name();
        let message = self.0.message();
        if message.is_empty() {
            f.write_str(&name)
        } else if name.is_empty() {
            f.write_str(&message)
        } else {
            write!(f, "{name}: {message}")
        }
    }
}

impl std::error::Error for DomException {}

impl From<web_sys::DomException> for DomException {
    fn from(value: web_sys::DomException) -> Self {
        Self(value)
    }
}

/// Typed failure from WebCodecs setup, decode, or upload.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Js(#[from] DomException),
    #[error("access unit has no NAL units")]
    EmptyAccessUnit,
    #[error("first keyframe is missing SPS or PPS, so avcC cannot be built")]
    MissingParameterSets,
    #[error("SPS is too short to build an avc1 codec string")]
    ShortSps,
    #[error("access unit has no picture NAL")]
    NoPictureNal,
    #[error("keyframe has no IDR NAL")]
    NoIdr,
    #[error("codec string is not known yet")]
    UnknownCodec,
    #[error("VideoDecoder.configure did not stick")]
    ConfigureFailed,
    #[error("decoder unavailable")]
    DecoderUnavailable,
    #[error("no video texture")]
    NoVideoTexture,
    #[error("no gpu image")]
    NoGpuImage,
    #[error("{0}")]
    Message(&'static str),
    #[error("{0}")]
    Context(String),
}

impl From<JsValue> for Error {
    fn from(err: JsValue) -> Self {
        match err.dyn_into::<web_sys::DomException>() {
            Ok(exc) => Self::Js(DomException(exc)),
            Err(other) => Self::Context(
                other
                    .as_string()
                    .unwrap_or_else(|| "unknown error".to_string()),
            ),
        }
    }
}

impl From<Error> for JsValue {
    fn from(err: Error) -> Self {
        match err {
            Error::Js(DomException(exc)) => exc.into(),
            other => JsValue::from_str(&other.to_string()),
        }
    }
}
