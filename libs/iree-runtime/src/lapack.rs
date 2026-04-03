use crate::{Result, error, ffi};

unsafe extern "C" {
    fn elodin_lapack_module_create(
        instance: *mut ffi::iree_vm_instance_t,
        device: *mut ffi::iree_hal_device_t,
        host_allocator: ffi::iree_allocator_t,
        out_module: *mut *mut ffi::iree_vm_module_t,
    ) -> ffi::iree_status_t;
}

/// Creates an IREE VM module named "elodin_lapack" that provides
/// LAPACK functions (SVD, Cholesky, LU, QR, solve, eigh) via OpenBLAS.
///
/// # Safety
/// `instance` and `device` must be valid, non-null IREE pointers.
pub unsafe fn create_module(
    instance: *mut ffi::iree_vm_instance_t,
    device: *mut ffi::iree_hal_device_t,
) -> Result<*mut ffi::iree_vm_module_t> {
    let mut module: *mut ffi::iree_vm_module_t = std::ptr::null_mut();
    let status = unsafe {
        elodin_lapack_module_create(instance, device, ffi::iree_allocator_system(), &mut module)
    };
    error::check(status)?;
    Ok(module)
}
