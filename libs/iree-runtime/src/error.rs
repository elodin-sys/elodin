use crate::ffi;
use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    code: u32,
    message: String,
}

impl Error {
    pub fn code(&self) -> u32 {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IREE error (code {}): {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

pub(crate) fn check(status: ffi::iree_status_t) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }

    // Extract full error message before consuming the status
    let message = status_to_string(status);

    let code = unsafe { ffi::iree_status_consume_code(status) }.0;

    Err(Error { code, message })
}

fn status_to_string(status: ffi::iree_status_t) -> String {
    let mut buffer = [0i8; 1024];
    let mut length: usize = 0;
    let ok =
        unsafe { ffi::iree_status_format(status, buffer.len(), buffer.as_mut_ptr(), &mut length) };
    if ok && length > 0 {
        let len = length.min(buffer.len());
        let bytes: Vec<u8> = buffer[..len].iter().map(|&b| b as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    } else {
        "unknown error".to_string()
    }
}
