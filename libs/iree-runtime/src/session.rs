use crate::call::Call;
use crate::device::Device;
use crate::error::{self, Result};
use crate::ffi;
use crate::instance::Instance;
use std::ptr;

pub struct Session {
    pub(crate) ptr: *mut ffi::iree_runtime_session_t,
}

impl Session {
    pub fn new(instance: &Instance, device: &Device) -> Result<Self> {
        let mut options = ffi::iree_runtime_session_options_t::default();
        unsafe { ffi::iree_runtime_session_options_initialize(&mut options) };

        let mut session: *mut ffi::iree_runtime_session_t = ptr::null_mut();
        let status = unsafe {
            ffi::iree_runtime_session_create_with_device(
                instance.ptr,
                &options,
                device.ptr,
                instance.allocator,
                &mut session,
            )
        };
        error::check(status)?;

        Ok(Self { ptr: session })
    }

    pub fn load_vmfb(&self, vmfb_data: &[u8]) -> Result<()> {
        let byte_span = ffi::iree_const_byte_span_t {
            data: vmfb_data.as_ptr(),
            data_length: vmfb_data.len(),
        };
        let status = unsafe {
            ffi::iree_runtime_session_append_bytecode_module_from_memory(
                self.ptr,
                byte_span,
                ffi::iree_allocator_null(),
            )
        };
        error::check(status)
    }

    pub fn call(&self, function_name: &str) -> Result<Call> {
        Call::new(self, function_name)
    }

    pub(crate) fn device(&self) -> *mut ffi::iree_hal_device_t {
        unsafe { ffi::iree_runtime_session_device(self.ptr) }
    }

    pub(crate) fn device_allocator(&self) -> *mut ffi::iree_hal_allocator_t {
        unsafe { ffi::iree_runtime_session_device_allocator(self.ptr) }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::iree_runtime_session_release(self.ptr) };
        }
    }
}

unsafe impl Send for Session {}
