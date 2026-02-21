use crate::call::Call;
use crate::device::Device;
use crate::error::{self, Result};
use crate::ffi;
use crate::instance::Instance;
use std::ffi::CString;
use std::path::Path;
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

    /// Load a compiled VMFB module from bytes.
    ///
    /// The data is copied into IREE-managed aligned memory, so the caller
    /// does not need to worry about alignment of the input slice.
    pub fn load_vmfb(&self, vmfb_data: &[u8]) -> Result<()> {
        let allocator = ffi::iree_allocator_system();

        // Allocate an IREE-owned buffer and copy the data into it.
        // IREE takes ownership of this buffer (frees it via the allocator).
        let mut iree_buf: *mut std::ffi::c_void = ptr::null_mut();
        let alloc_status =
            unsafe { ffi::iree_allocator_malloc(allocator, vmfb_data.len(), &mut iree_buf) };
        error::check(alloc_status)?;

        unsafe {
            ptr::copy_nonoverlapping(vmfb_data.as_ptr(), iree_buf as *mut u8, vmfb_data.len());
        }

        let byte_span = ffi::iree_const_byte_span_t {
            data: iree_buf as *const u8,
            data_length: vmfb_data.len(),
        };

        // Pass the system allocator so IREE owns (and eventually frees) the buffer.
        let status = unsafe {
            ffi::iree_runtime_session_append_bytecode_module_from_memory(
                self.ptr, byte_span, allocator,
            )
        };
        error::check(status)
    }

    /// Load a compiled VMFB module from a file path.
    pub fn load_vmfb_file(&self, path: &Path) -> Result<()> {
        let path_str =
            CString::new(path.to_str().expect("path must be valid UTF-8")).expect("no null bytes");
        let status = unsafe {
            ffi::iree_runtime_session_append_bytecode_module_from_file(self.ptr, path_str.as_ptr())
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
