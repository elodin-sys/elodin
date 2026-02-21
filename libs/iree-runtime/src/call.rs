use crate::buffer_view::BufferView;
use crate::error::{self, Result};
use crate::ffi;
use crate::session::Session;
use std::ffi::CString;
use std::mem::MaybeUninit;

pub struct Call {
    inner: ffi::iree_runtime_call_t,
}

impl Call {
    pub(crate) fn new(session: &Session, function_name: &str) -> Result<Self> {
        let name = CString::new(function_name).expect("function name must not contain null bytes");
        let name_view = ffi::iree_string_view_t {
            data: name.as_ptr(),
            size: function_name.len(),
        };

        let mut call = MaybeUninit::<ffi::iree_runtime_call_t>::uninit();
        let status = unsafe {
            ffi::iree_runtime_call_initialize_by_name(session.ptr, name_view, call.as_mut_ptr())
        };
        error::check(status)?;

        Ok(Self {
            inner: unsafe { call.assume_init() },
        })
    }

    pub fn push_input(&mut self, buffer_view: &BufferView) -> Result<()> {
        let status = unsafe {
            ffi::iree_runtime_call_inputs_push_back_buffer_view(&mut self.inner, buffer_view.ptr)
        };
        error::check(status)
    }

    pub fn invoke(&mut self) -> Result<()> {
        let status = unsafe { ffi::iree_runtime_call_invoke(&mut self.inner, 0) };
        error::check(status)
    }

    pub fn pop_output(&mut self) -> Result<BufferView> {
        let mut view: *mut ffi::iree_hal_buffer_view_t = std::ptr::null_mut();
        let status = unsafe {
            ffi::iree_runtime_call_outputs_pop_front_buffer_view(&mut self.inner, &mut view)
        };
        error::check(status)?;
        Ok(BufferView { ptr: view })
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        unsafe { ffi::iree_runtime_call_deinitialize(&mut self.inner) };
    }
}
