use std::ffi::{CStr, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use libloading::Library;

use super::{ELODIN_MODEL_ERR_STATE, ELODIN_MODEL_ERR_STEP, fail, ffi_status, panic_message};

const HOOK_ABI_VERSION: u32 = 1;
const ERROR_CAPACITY: usize = 512;

#[repr(C)]
pub struct ElodinColumn {
    pub name: *const c_char,
    pub data: *mut u8,
    pub row_bytes: u64,
    pub writable: u32,
}

#[repr(C)]
pub struct ElodinStep {
    pub abi_version: u32,
    pub tick: u64,
    pub timestamp: i64,
    pub rows: u64,
    pub row_start: u64,
    pub row_end: u64,
    pub columns: *const ElodinColumn,
    pub column_count: u64,
    pub state: *mut c_void,
    pub error: *mut c_char,
    pub error_capacity: u64,
}

type StepFn = unsafe extern "C" fn(*const ElodinStep) -> i32;
type InitFn = unsafe extern "C" fn(*const ElodinStep, *mut *mut c_void) -> i32;
type DestroyFn = unsafe extern "C" fn(*mut c_void);

pub struct ElodinHook {
    _library: Library,
    step: StepFn,
    init: Option<InitFn>,
    destroy: Option<DestroyFn>,
    state: *mut c_void,
    rows: u64,
    initialized: bool,
}

fn hook_error(buffer: &[u8]) -> String {
    CStr::from_bytes_until_nul(buffer)
        .ok()
        .and_then(|message| message.to_str().ok())
        .filter(|message| !message.is_empty())
        .unwrap_or("native hook failed")
        .to_string()
}

fn call_step(
    hook: &ElodinHook,
    tick: u64,
    timestamp: i64,
    columns: *const ElodinColumn,
    column_count: u64,
) -> Result<(), String> {
    if columns.is_null() && column_count != 0 {
        return Err("hook columns are null".into());
    }
    let mut error = [0u8; ERROR_CAPACITY];
    let step = ElodinStep {
        abi_version: HOOK_ABI_VERSION,
        tick,
        timestamp,
        rows: hook.rows,
        row_start: 0,
        row_end: hook.rows,
        columns,
        column_count,
        state: hook.state,
        error: error.as_mut_ptr().cast(),
        error_capacity: error.len() as u64,
    };
    // Safety: the caller selected a library that implements the documented ABI, and keeps every
    // column and its data valid for this synchronous call.
    let status = unsafe { (hook.step)(&raw const step) };
    if status == 0 {
        Ok(())
    } else {
        Err(hook_error(&error))
    }
}

/// # Safety
/// `library` and `symbol` are non-null, NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_hook_open(
    library: *const c_char,
    symbol: *const c_char,
) -> *mut ElodinHook {
    let opened = catch_unwind(AssertUnwindSafe(|| -> Result<ElodinHook, String> {
        if library.is_null() || symbol.is_null() {
            return Err("hook library and symbol are required".into());
        }
        let library = unsafe { CStr::from_ptr(library) }
            .to_str()
            .map_err(|error| error.to_string())?;
        let symbol = unsafe { CStr::from_ptr(symbol) }
            .to_str()
            .map_err(|error| error.to_string())?;
        // Safety: loading executes the initializers of the caller-selected library.
        let loaded = unsafe { Library::new(library) }
            .map_err(|error| format!("cannot load {library:?}: {error}"))?;
        // Safety: the caller asserts these symbols implement elodin_step_hook.h.
        let step = unsafe { loaded.get::<StepFn>(symbol.as_bytes()) }
            .map_err(|error| format!("{library:?} has no {symbol:?}: {error}"))?;
        let step = *step;
        let init = unsafe { loaded.get::<InitFn>(format!("{symbol}_init").as_bytes()) }
            .ok()
            .map(|function| *function);
        let destroy = unsafe { loaded.get::<DestroyFn>(format!("{symbol}_destroy").as_bytes()) }
            .ok()
            .map(|function| *function);
        Ok(ElodinHook {
            _library: loaded,
            step,
            init,
            destroy,
            state: ptr::null_mut(),
            rows: 0,
            initialized: false,
        })
    }));
    match opened {
        Ok(Ok(hook)) => Box::into_raw(Box::new(hook)),
        Ok(Err(error)) => {
            fail(error);
            ptr::null_mut()
        }
        Err(payload) => {
            fail(panic_message(payload));
            ptr::null_mut()
        }
    }
}

/// Runs the hook's init symbol with the tick and timestamp of the first call, as Elodin does.
///
/// # Safety
/// `hook` is live; `columns` and their data remain valid for this synchronous call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_hook_initialize(
    hook: *mut ElodinHook,
    rows: u64,
    tick: u64,
    timestamp: i64,
    columns: *const ElodinColumn,
    column_count: u64,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_STATE, || {
        let hook = unsafe { hook.as_mut() }.ok_or("hook is null")?;
        if hook.initialized {
            return Err("hook is already initialized".into());
        }
        if rows == 0 {
            return Err("hook rows must be positive".into());
        }
        if columns.is_null() && column_count != 0 {
            return Err("hook columns are null".into());
        }
        hook.rows = rows;
        if let Some(init) = hook.init {
            let mut error = [0u8; ERROR_CAPACITY];
            let step = ElodinStep {
                abi_version: HOOK_ABI_VERSION,
                tick,
                timestamp,
                rows,
                row_start: 0,
                row_end: rows,
                columns,
                column_count,
                state: ptr::null_mut(),
                error: error.as_mut_ptr().cast(),
                error_capacity: error.len() as u64,
            };
            // Safety: the loaded function implements the documented ABI.
            let status = unsafe { init(&raw const step, &raw mut hook.state) };
            if status != 0 {
                return Err(hook_error(&error));
            }
        }
        hook.initialized = true;
        Ok(())
    })
}

/// # Safety
/// `hook` is live and initialized; columns and their data remain valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_hook_call(
    hook: *mut ElodinHook,
    tick: u64,
    timestamp: i64,
    columns: *const ElodinColumn,
    column_count: u64,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_STEP, || {
        let hook = unsafe { hook.as_ref() }.ok_or("hook is null")?;
        if !hook.initialized {
            return Err("hook is not initialized".into());
        }
        call_step(hook, tick, timestamp, columns, column_count)
    })
}

/// # Safety
/// `hook` is null or from `elodin_hook_open` and has not been closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_hook_close(hook: *mut ElodinHook) {
    if hook.is_null() {
        return;
    }
    // Safety: ownership returns exactly once.
    let hook = unsafe { Box::from_raw(hook) };
    if hook.initialized
        && let Some(destroy) = hook.destroy
    {
        // Safety: the loaded destroy function owns the state returned by its init function.
        unsafe { destroy(hook.state) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_hook_calls_fail_without_panicking() {
        unsafe {
            assert_eq!(
                elodin_hook_initialize(ptr::null_mut(), 1, 0, 0, ptr::null(), 0),
                ELODIN_MODEL_ERR_STATE
            );
            assert_eq!(
                elodin_hook_call(ptr::null_mut(), 0, 0, ptr::null(), 0),
                ELODIN_MODEL_ERR_STEP
            );
            elodin_hook_close(ptr::null_mut());
        }
    }
}
