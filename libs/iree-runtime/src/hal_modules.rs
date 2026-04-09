use crate::{Result, error, ffi};
use std::ptr;

/// Register the `hal_inline` and `hal_loader` VM modules on a session.
/// Required for VMFBs compiled with `--iree-execution-model=inline-dynamic`.
///
/// # Safety
/// All pointer arguments must be valid, non-null IREE runtime pointers.
pub unsafe fn register_inline_modules(
    instance: *mut ffi::iree_vm_instance_t,
    device: *mut ffi::iree_hal_device_t,
    session: *mut ffi::iree_runtime_session_t,
) -> Result<()> {
    let allocator = ffi::iree_allocator_system();
    let device_allocator = unsafe { ffi::iree_hal_device_allocator(device) };

    let mut inline_module: *mut ffi::iree_vm_module_t = ptr::null_mut();
    let status = unsafe {
        ffi::iree_hal_inline_module_create(
            instance,
            0,
            ffi::iree_hal_module_debug_sink_t::default(),
            device_allocator,
            allocator,
            &mut inline_module,
        )
    };
    error::check(status)?;
    let status = unsafe { ffi::iree_runtime_session_append_module(session, inline_module) };
    error::check(status)?;
    unsafe { ffi::iree_vm_module_release(inline_module) };

    let mut loaders: Vec<*mut ffi::iree_hal_executable_loader_t> = Vec::new();

    if cfg!(target_os = "macos") {
        let mut sys_loader: *mut ffi::iree_hal_executable_loader_t = ptr::null_mut();
        let status = unsafe {
            ffi::iree_hal_system_library_loader_create(ptr::null_mut(), allocator, &mut sys_loader)
        };
        error::check(status)?;
        loaders.push(sys_loader);
    }

    {
        let mut elf_loader: *mut ffi::iree_hal_executable_loader_t = ptr::null_mut();
        let status = unsafe {
            ffi::iree_hal_embedded_elf_loader_create(ptr::null_mut(), allocator, &mut elf_loader)
        };
        error::check(status)?;
        loaders.push(elf_loader);
    }

    if cfg!(not(target_os = "macos")) {
        let mut sys_loader: *mut ffi::iree_hal_executable_loader_t = ptr::null_mut();
        let status = unsafe {
            ffi::iree_hal_system_library_loader_create(ptr::null_mut(), allocator, &mut sys_loader)
        };
        error::check(status)?;
        loaders.push(sys_loader);
    }

    let mut loader_module: *mut ffi::iree_vm_module_t = ptr::null_mut();
    let status = unsafe {
        ffi::iree_hal_loader_module_create(
            instance,
            0,
            loaders.len(),
            loaders.as_mut_ptr(),
            allocator,
            &mut loader_module,
        )
    };
    error::check(status)?;
    for &loader in &loaders {
        unsafe { ffi::iree_hal_executable_loader_release(loader) };
    }

    let status = unsafe { ffi::iree_runtime_session_append_module(session, loader_module) };
    error::check(status)?;
    unsafe { ffi::iree_vm_module_release(loader_module) };

    Ok(())
}
