use crate::ffi;

pub struct Device {
    pub(crate) ptr: *mut ffi::iree_hal_device_t,
}

impl Drop for Device {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::iree_hal_device_release(self.ptr) };
        }
    }
}

unsafe impl Send for Device {}
