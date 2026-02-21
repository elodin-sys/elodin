use crate::device::Device;
use crate::error::{self, Result};
use crate::ffi;
use std::ffi::CString;
use std::ptr;

pub struct Instance {
    pub(crate) ptr: *mut ffi::iree_runtime_instance_t,
    pub(crate) allocator: ffi::iree_allocator_t,
}

impl Instance {
    pub fn new() -> Result<Self> {
        let allocator = ffi::iree_allocator_system();

        let mut options = ffi::iree_runtime_instance_options_t::default();
        unsafe { ffi::iree_runtime_instance_options_initialize(&mut options) };
        unsafe { ffi::iree_runtime_instance_options_use_all_available_drivers(&mut options) };

        let mut instance: *mut ffi::iree_runtime_instance_t = ptr::null_mut();
        let status =
            unsafe { ffi::iree_runtime_instance_create(&options, allocator, &mut instance) };
        error::check(status)?;

        Ok(Self {
            ptr: instance,
            allocator,
        })
    }

    pub fn create_device(&self, device_uri: &str) -> Result<Device> {
        let uri = CString::new(device_uri).expect("device URI must not contain null bytes");
        let uri_view = ffi::iree_string_view_t {
            data: uri.as_ptr(),
            size: device_uri.len(),
        };
        let mut device: *mut ffi::iree_hal_device_t = ptr::null_mut();
        let status = unsafe {
            ffi::iree_runtime_instance_try_create_default_device(self.ptr, uri_view, &mut device)
        };
        error::check(status)?;

        Ok(Device { ptr: device })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::iree_runtime_instance_release(self.ptr) };
        }
    }
}

unsafe impl Send for Instance {}
