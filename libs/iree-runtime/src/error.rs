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

    pub(crate) fn invalid_argument(message: impl Into<String>) -> Self {
        Self {
            code: 2,
            message: message.into(),
        }
    }

    pub fn is_overlap_copy_error(&self) -> bool {
        let message = self.message();
        self.code == 3
            && (message.contains("source and target ranges must not overlap")
                || message.contains("source and target ranges overlap within the same buffer"))
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

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn detects_known_overlap_error_variants() {
        let transfer_error = Error {
            code: 3,
            message: "INVALID_ARGUMENT; source and target ranges must not overlap within the same buffer".to_string(),
        };
        assert!(transfer_error.is_overlap_copy_error());

        let command_buffer_error = Error {
            code: 3,
            message: "INVALID_ARGUMENT; source and target ranges overlap within the same buffer".to_string(),
        };
        assert!(command_buffer_error.is_overlap_copy_error());
    }

    #[test]
    fn does_not_match_unrelated_errors() {
        let unrelated = Error {
            code: 3,
            message: "INTERNAL; command buffer submission failed".to_string(),
        };
        assert!(!unrelated.is_overlap_copy_error());

        let wrong_code = Error {
            code: 2,
            message: "INVALID_ARGUMENT; source and target ranges must not overlap within the same buffer".to_string(),
        };
        assert!(!wrong_code.is_overlap_copy_error());
    }
}
