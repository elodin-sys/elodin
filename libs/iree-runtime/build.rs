use std::env;
use std::path::PathBuf;

fn main() {
    let iree_dir = env::var("IREE_RUNTIME_DIR").unwrap_or_else(|_| {
        panic!(
            "IREE_RUNTIME_DIR is not set. \
             Enter the nix develop shell to set it automatically."
        )
    });
    let iree_dir = PathBuf::from(iree_dir);

    println!("cargo:rerun-if-env-changed=IREE_RUNTIME_DIR");

    let include_dir = iree_dir.join("include");
    let lib_dir = iree_dir.join("lib");

    assert!(
        include_dir.join("iree/runtime/api.h").exists(),
        "IREE_RUNTIME_DIR={} does not contain include/iree/runtime/api.h",
        iree_dir.display()
    );
    assert!(
        lib_dir.exists(),
        "IREE_RUNTIME_DIR={} does not contain lib/",
        iree_dir.display()
    );

    // Link search path
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // The unified runtime library contains the high-level API
    println!("cargo:rustc-link-lib=static=iree_runtime_unified");
    println!("cargo:rustc-link-lib=static=iree_runtime_impl");

    // VM (bytecode interpreter)
    println!("cargo:rustc-link-lib=static=iree_vm_bytecode_module");
    println!("cargo:rustc-link-lib=static=iree_vm_impl");

    // HAL (hardware abstraction layer)
    println!("cargo:rustc-link-lib=static=iree_hal_hal");
    println!("cargo:rustc-link-lib=static=iree_hal_local_local");
    println!("cargo:rustc-link-lib=static=iree_hal_drivers_drivers");

    // HAL drivers: local-sync and local-task
    println!("cargo:rustc-link-lib=static=iree_hal_drivers_local_sync_sync_driver");
    println!("cargo:rustc-link-lib=static=iree_hal_drivers_local_sync_registration_registration");
    println!("cargo:rustc-link-lib=static=iree_hal_drivers_local_task_task_driver");
    println!("cargo:rustc-link-lib=static=iree_hal_drivers_local_task_registration_registration");

    // Executable loaders
    println!("cargo:rustc-link-lib=static=iree_hal_local_loaders_embedded_elf_loader");
    println!("cargo:rustc-link-lib=static=iree_hal_local_loaders_system_library_loader");
    println!("cargo:rustc-link-lib=static=iree_hal_local_loaders_vmvx_module_loader");
    println!("cargo:rustc-link-lib=static=iree_hal_local_loaders_static_library_loader");
    println!("cargo:rustc-link-lib=static=iree_hal_local_loaders_registration_registration");
    println!("cargo:rustc-link-lib=static=iree_hal_local_executable_loader");
    println!("cargo:rustc-link-lib=static=iree_hal_local_executable_format");
    println!("cargo:rustc-link-lib=static=iree_hal_local_executable_environment");
    println!("cargo:rustc-link-lib=static=iree_hal_local_executable_library_util");
    println!("cargo:rustc-link-lib=static=iree_hal_local_executable_plugin_manager");
    println!("cargo:rustc-link-lib=static=iree_hal_local_elf_elf_module");
    println!("cargo:rustc-link-lib=static=iree_hal_local_elf_arch");
    println!("cargo:rustc-link-lib=static=iree_hal_local_elf_platform");
    println!("cargo:rustc-link-lib=static=iree_hal_local_plugins_embedded_elf_plugin");
    println!("cargo:rustc-link-lib=static=iree_hal_local_plugins_registration_registration");
    println!("cargo:rustc-link-lib=static=iree_hal_local_plugins_static_plugin");
    println!("cargo:rustc-link-lib=static=iree_hal_local_plugins_system_library_plugin");

    // HAL utilities
    println!("cargo:rustc-link-lib=static=iree_hal_utils_allocators");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_caching_allocator");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_debug_allocator");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_deferred_command_buffer");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_deferred_work_queue");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_executable_debug_info");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_executable_header");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_file_cache");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_file_transfer");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_files");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_libmpi");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_mpi_channel_provider");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_queue_emulation");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_queue_host_call_emulation");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_resource_set");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_semaphore_base");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_stream_tracing");
    println!("cargo:rustc-link-lib=static=iree_hal_utils_collective_batch");

    // Modules
    println!("cargo:rustc-link-lib=static=iree_modules_hal_hal");
    println!("cargo:rustc-link-lib=static=iree_modules_hal_types");
    println!("cargo:rustc-link-lib=static=iree_modules_hal_inline_inline");
    println!("cargo:rustc-link-lib=static=iree_modules_hal_loader_loader");
    println!("cargo:rustc-link-lib=static=iree_modules_hal_utils_buffer_diagnostics");
    println!("cargo:rustc-link-lib=static=iree_modules_hal_debugging");
    println!("cargo:rustc-link-lib=static=iree_modules_vmvx_vmvx");
    println!("cargo:rustc-link-lib=static=iree_modules_io_parameters_parameters");

    // Base libraries
    println!("cargo:rustc-link-lib=static=iree_base_base");
    println!("cargo:rustc-link-lib=static=iree_base_internal_arena");
    println!("cargo:rustc-link-lib=static=iree_base_internal_atomic_slist");
    println!("cargo:rustc-link-lib=static=iree_base_internal_bitmap");
    println!("cargo:rustc-link-lib=static=iree_base_internal_cpu");
    println!("cargo:rustc-link-lib=static=iree_base_internal_dynamic_library");
    println!("cargo:rustc-link-lib=static=iree_base_internal_event_pool");
    println!("cargo:rustc-link-lib=static=iree_base_internal_flags");
    println!("cargo:rustc-link-lib=static=iree_base_internal_fpu_state");
    println!("cargo:rustc-link-lib=static=iree_base_internal_memory");
    println!("cargo:rustc-link-lib=static=iree_base_internal_path");
    println!("cargo:rustc-link-lib=static=iree_base_internal_synchronization");
    println!("cargo:rustc-link-lib=static=iree_base_internal_threading");
    println!("cargo:rustc-link-lib=static=iree_base_internal_time");
    println!("cargo:rustc-link-lib=static=iree_base_internal_wait_handle");
    println!("cargo:rustc-link-lib=static=iree_base_loop_sync");

    // IO
    println!("cargo:rustc-link-lib=static=iree_io_file_handle");
    println!("cargo:rustc-link-lib=static=iree_io_parameter_index");
    println!("cargo:rustc-link-lib=static=iree_io_parameter_index_provider");
    println!("cargo:rustc-link-lib=static=iree_io_parameter_provider");
    println!("cargo:rustc-link-lib=static=iree_io_scope_map");
    println!("cargo:rustc-link-lib=static=iree_io_stream");
    println!("cargo:rustc-link-lib=static=iree_io_formats_parser_registry");
    println!("cargo:rustc-link-lib=static=iree_io_formats_gguf_gguf");
    println!("cargo:rustc-link-lib=static=iree_io_formats_irpa_irpa");
    println!("cargo:rustc-link-lib=static=iree_io_formats_safetensors_safetensors");

    // Task system (for local-task driver)
    println!("cargo:rustc-link-lib=static=iree_task_task");
    println!("cargo:rustc-link-lib=static=iree_task_api");

    // Builtins
    println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_ukernel");
    println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_fallback");
    println!("cargo:rustc-link-lib=static=iree_builtins_device_device");
    println!("cargo:rustc-link-lib=static=iree_builtins_musl_bin_libmusl");

    // Architecture-specific micro-kernels.
    // Use CARGO_CFG_TARGET_ARCH (not #[cfg]) so cross-compilation links
    // the correct target libraries rather than the host's.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if target_arch == "aarch64" {
        println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_arm_64_arm_64");
        println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_arm_64_arm_64_bf16");
        println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_arm_64_arm_64_dotprod");
        println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_arm_64_arm_64_fp16fml");
        println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_arm_64_arm_64_fullfp16");
        println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_arm_64_arm_64_i8mm");
    } else if target_arch == "x86_64" {
        println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_x86_64_x86_64");
        println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_x86_64_common_x86_64");
        println!("cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_x86_64_x86_64_avx2_fma");
        println!(
            "cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_x86_64_x86_64_avx512_base"
        );
        println!(
            "cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_x86_64_x86_64_avx512_vnni"
        );
        println!(
            "cargo:rustc-link-lib=static=iree_builtins_ukernel_arch_x86_64_x86_64_avx512_bf16"
        );
    }

    // Third-party deps
    println!("cargo:rustc-link-lib=static=flatcc_parsing");
    println!("cargo:rustc-link-lib=static=flatcc_runtime");
    println!("cargo:rustc-link-lib=static=benchmark");

    // System libraries.
    // Use CARGO_CFG_TARGET_OS (not #[cfg]) for cross-compilation correctness.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=m");
    } else if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Security");
        println!("cargo:rustc-link-lib=dl");
    }

    // Run bindgen
    let bindings = bindgen::Builder::default()
        .header(include_dir.join("iree/runtime/api.h").to_str().unwrap())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("iree_.*")
        .allowlist_type("iree_.*")
        .allowlist_var("IREE_.*")
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: true,
            is_global: true,
        })
        .derive_default(true)
        .generate_comments(true)
        .generate()
        .expect("Failed to generate IREE bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");
}
