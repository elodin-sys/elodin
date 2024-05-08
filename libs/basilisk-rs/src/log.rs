use std::ffi::c_char;

use crate::sys::logLevel_t;
#[allow(non_snake_case)]
#[no_mangle]
unsafe extern "C" fn _bskLog(_logger: *const (), level: logLevel_t, msg: *const c_char) {
    let msg = std::ffi::CStr::from_ptr(msg).to_string_lossy();
    println!("{}: {}", level.0, msg);
    match level {
        logLevel_t::BSK_DEBUG => {
            tracing::debug!("{}", msg);
        }
        logLevel_t::BSK_INFORMATION => {
            tracing::info!("{}", msg);
        }
        logLevel_t::BSK_WARNING => {
            tracing::warn!("{}", msg);
        }
        logLevel_t::BSK_ERROR => {
            tracing::error!("{}", msg);
        }
        level => {
            tracing::trace!(?level, "{}", msg);
        }
    };
}
