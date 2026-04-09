mod ffi;

mod buffer_view;
mod call;
mod device;
mod device_buffer;
mod element_type;
mod error;
pub mod hal_modules;
mod instance;
pub mod lapack;
mod session;

pub use buffer_view::BufferView;
pub use call::Call;
pub use device::Device;
pub use device_buffer::{BufferMapping, BufferSpec, DeviceArena, DeviceBuffer, MappedArena};
pub use element_type::ElementType;
pub use error::{Error, Result};
pub use instance::Instance;
pub use session::Session;

/// Release a VM module reference obtained from a `*_create` call.
///
/// # Safety
/// `module` must be a valid, non-null IREE VM module pointer with at least one
/// outstanding reference.
pub unsafe fn vm_module_release(module: *mut ffi::iree_vm_module_t) {
    unsafe { ffi::iree_vm_module_release(module) }
}
