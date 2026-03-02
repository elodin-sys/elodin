#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unused_imports,
    clippy::all,
    unsafe_op_in_unsafe_fn
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe extern "C" {
    pub fn iree_allocator_libc_ctl(
        self_: *mut ::std::os::raw::c_void,
        command: iree_allocator_command_e,
        params: *const ::std::os::raw::c_void,
        inout_ptr: *mut *mut ::std::os::raw::c_void,
    ) -> iree_status_t;
}

pub fn iree_allocator_system() -> iree_allocator_t {
    iree_allocator_t {
        self_: std::ptr::null_mut(),
        ctl: Some(iree_allocator_libc_ctl),
    }
}

pub fn iree_allocator_null() -> iree_allocator_t {
    iree_allocator_t {
        self_: std::ptr::null_mut(),
        ctl: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_test_linking() {
        let code_str = unsafe { iree_status_code_string(iree_status_code_e(0)) };
        assert!(!code_str.is_null());
        let s = unsafe { std::ffi::CStr::from_ptr(code_str) };
        assert_eq!(s.to_str().unwrap(), "OK");
    }
}
