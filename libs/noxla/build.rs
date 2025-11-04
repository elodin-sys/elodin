use std::io::Read;
use std::path::{Path, PathBuf};
use std::{env, io};

use anyhow::Context;
use flate2::read::GzDecoder;
use tar::Archive;

#[derive(Clone, Copy, Eq, PartialEq)]
#[allow(clippy::enum_variant_names)]
enum OS {
    Linux,
    MacOS,
    Windows,
}

impl OS {
    fn get() -> Self {
        let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
        match os.as_str() {
            "linux" => Self::Linux,
            "macos" => Self::MacOS,
            "windows" => Self::Windows,
            os => panic!("Unsupported system {os}"),
        }
    }
}

fn env_var_rerun(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name).ok()
}

fn main() -> anyhow::Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("missing out dir"));
    let os = OS::get();

    let xla_dir = env_var_rerun("XLA_EXTENSION_DIR")
        .map_or_else(|| out_dir.join("xla_extension"), PathBuf::from);
    if !xla_dir.exists() {
        download_xla(&out_dir)?;
    }

    // The CPU kernels are now compiled directly with cpp_build below
    // This ensures their global constructors run when the library is loaded
    let kernels_path = out_dir.join("libnoxla_kernels.a");
    println!("cargo:noxla_kernels={}", kernels_path.display());

    let mut config = cpp_build::Config::new();
    config
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-w") // Suppress all warnings from third-party XLA C++ code
        .include(xla_dir.join("include"))
        .include("./vendor")
        // Add the CPU kernel files directly to the cpp_build
        .file("./vendor/jaxlib/cpu/cpu_kernels.cc")
        .file("./vendor/jaxlib/cpu/sparse_kernels.cc")
        .file("./vendor/jaxlib/cpu/lapack_kernels.cc")
        .file("./vendor/jaxlib/cpu/lapack_kernels_using_lapack.cc")
        // .include(format!("{out_dir}/xla_extension/include"))
        .build("src/lib.rs");

    if cfg!(feature = "cuda") {
        let cuda = find_cuda_helper::find_cuda_root().expect("No CUDA found!");
        find_cuda_helper::include_cuda();
        config
            .include(cuda.join("include"))
            .include(&cuda)
            .define("JAX_GPU_CUDA", Some("1"))
            .define("EL_CUDA", Some("1"))
            .file("./vendor/jaxlib/gpu/cholesky_update_kernel.cc")
            .file("./vendor/jaxlib/gpu/lu_pivot_kernels.cc")
            .file("./vendor/jaxlib/gpu/prng_kernels.cc");
        let mut cuda_config = cc::Build::new();
        cuda_config
            .flag("-std=c++17")
            .flag("-DLLVM_ON_UNIX=1")
            .flag("-DLLVM_VERSION_STRING=")
            .flag_if_supported("-w") // Suppress all warnings from third-party XLA C++ code
            .include(xla_dir.join("include"))
            .include("./vendor");
        cuda_config
            .flag("--disable-warnings")
            .cuda(true)
            .cudart("static")
            .include(cuda)
            .flag("-gencode")
            .flag("arch=compute_89,code=sm_89")
            .file("./vendor/jaxlib/gpu/blas_kernels.cc")
            .file("./vendor/jaxlib/gpu/cholesky_update_kernel.cu")
            .file("./vendor/jaxlib/gpu/gpu_kernel_helpers.cc")
            .file("./vendor/jaxlib/gpu/gpu_kernels.cc")
            .file("./vendor/jaxlib/gpu/lu_pivot_kernels.cu")
            .file("./vendor/jaxlib/gpu/prng_kernels.cu")
            .file("./vendor/jaxlib/gpu/rnn_kernels.cc")
            .file("./vendor/jaxlib/gpu/solver_kernels.cc")
            .file("./vendor/jaxlib/gpu/sparse_kernels.cc")
            .define("JAX_GPU_CUDA", Some("1"));
        cuda_config.compile("jaxlib_cuda");
        config.build("src/lib.rs")
    }

    println!("cargo:rerun-if-changed=vendor/jaxlib/cpu/cpu_kernels.cc");
    println!("cargo:rerun-if-changed=vendor/jaxlib/cpu/sparse_kernels.cc");
    println!("cargo:rerun-if-changed=vendor/jaxlib/cpu/lapack_kernels.cc");
    println!("cargo:rerun-if-changed=vendor/jaxlib/cpu/lapack_kernels_using_lapack.cc");
    println!("cargo:rerun-if-changed=src/executable.rs");
    println!("cargo:rerun-if-changed=src/literal.rs");
    println!("cargo:rerun-if-changed=src/op.rs");
    println!("cargo:rerun-if-changed=src/shape.rs");
    println!("cargo:rerun-if-changed=src/native_type.rs");
    println!("cargo:rerun-if-changed=src/builder.rs");
    println!("cargo:rerun-if-changed=src/error.rs");
    println!("cargo:rerun-if-changed=src/client.rs");
    println!("cargo:rerun-if-changed=src/buffer.rs");
    println!("cargo:rerun-if-changed=src/computation.rs");
    println!("cargo:rerun-if-changed=src/hlo_module.rs");

    let jax_metal_dir =
        env_var_rerun("JAX_METAL_DIR").map_or_else(|| out_dir.join("jax_metal"), PathBuf::from);
    if !jax_metal_dir.exists() && cfg!(target_os = "macos") {
        download_jax_metal(&jax_metal_dir)?;
    }

    // Exit early on docs.rs as the C++ library would not be available.
    if std::env::var("DOCS_RS").is_ok() {
        return Ok(());
    }

    // Explicitly link libstdc++ and BLAS/LAPACK on Linux to avoid "DSO missing from command line" errors.
    if os == OS::Linux {
        println!("cargo:rustc-link-lib=dylib=lapack");
        println!("cargo:rustc-link-lib=dylib=blas");
        println!("cargo:rustc-link-lib=dylib=gfortran");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    if cfg!(feature = "shared") {
        println!("cargo:rustc-link-search={}", xla_dir.join("lib").display());
        println!("cargo:rustc-link-lib=dylib=xla_extension");
    } else {
        println!(
            "cargo:rustc-link-search=native={}",
            xla_dir.join("lib").display()
        );
        println!("cargo:rustc-link-lib=static=xla_extension");
    }
    if os == OS::MacOS {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=SystemConfiguration");
        println!("cargo:rustc-link-lib=framework=Security");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    Ok(())
}

fn download_jax_metal(jax_dir: &Path) -> anyhow::Result<()> {
    let url = "https://files.pythonhosted.org/packages/7e/59/ff91dc65e7f945479b08509185d07de0c947e81c07705367b018cb072ee9/jax_metal-0.0.4-py3-none-macosx_11_0_arm64.whl";
    let buf = download_file(url)?;
    let mut archive = zip::ZipArchive::new(io::Cursor::new(buf))?;
    archive.extract(jax_dir)?;
    Ok(())
}

fn download_xla(xla_dir: &Path) -> anyhow::Result<()> {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    let url = match (os.as_str(), arch.as_str()) {
        ("macos", arch) => format!(
            "https://github.com/elodin-sys/xla-next/releases/download/v0.9.1/xla_extension-0.9.1-{}-darwin-cpu.tar.gz",
            arch
        ),
        ("linux", arch) => format!(
            "https://github.com/elodin-sys/xla-next/releases/download/v0.9.1/xla_extension-0.9.1-{}-linux-gnu-cpu.tar.gz",
            arch
        ),
        (os, arch) => panic!("{}-{} is an unsupported platform", os, arch),
    };
    let buf = download_file(&url)?;
    let mut bytes = io::Cursor::new(buf);

    let tar = GzDecoder::new(&mut bytes);
    let mut archive = Archive::new(tar);
    archive.unpack(xla_dir)?;

    Ok(())
}

fn download_file(url: &str) -> anyhow::Result<Vec<u8>> {
    let res = ureq::get(url).call()?;
    let content_length = res
        .header("Content-Length")
        .context("Content-Length header not found")?
        .parse::<usize>()?;
    let mut buf = Vec::with_capacity(content_length);
    res.into_reader()
        .take(content_length as u64)
        .read_to_end(&mut buf)
        .context("Failed to read response")?;
    Ok(buf)
}
