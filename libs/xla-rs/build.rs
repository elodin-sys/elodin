use std::path::{Path, PathBuf};
use std::{env, io};

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

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("missing out dir"));
    let os = OS::get();

    let xla_dir = env_var_rerun("XLA_EXTENSION_DIR")
        .map_or_else(|| out_dir.join("xla_extension"), PathBuf::from);
    if !xla_dir.exists() {
        if cfg!(feature = "shared") {
            download_shared_xla(&out_dir).await?;
        } else {
            download_xla(&out_dir).await?;
        }
    }

    let mut config = cpp_build::Config::new();
    config
        .flag("-std=c++17")
        .flag("-DLLVM_ON_UNIX=1")
        .flag("-DLLVM_VERSION_STRING=")
        .flag(&format!("-isystem{}", xla_dir.join("include").display()))
        .file("./vendor/jaxlib/cpu/cpu_kernels.cc")
        .file("./vendor/jaxlib/cpu/lapack_kernels.cc")
        .include("./vendor");
    if cfg!(feature = "cuda") {
        find_cuda_helper::include_cuda();
        let cuda = find_cuda_helper::find_cuda_root().unwrap();
        config
            .include(&cuda.join("include"))
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
    }
    config.build("src/lib.rs");
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
        download_jax_metal(&jax_metal_dir).await?;
    }

    // Exit early on docs.rs as the C++ library would not be available.
    if std::env::var("DOCS_RS").is_ok() {
        return Ok(());
    }

    // The --copy-dt-needed-entries -lstdc++ are helpful to get around some
    // "DSO missing from command line" error
    // undefined reference to symbol '_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE@@GLIBCXX_3.4.21'
    if os == OS::Linux {
        println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
        println!("cargo:rustc-link-arg=-Wl,-lstdc++");
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
    }

    Ok(())
}

async fn download_jax_metal(jax_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://files.pythonhosted.org/packages/7e/59/ff91dc65e7f945479b08509185d07de0c947e81c07705367b018cb072ee9/jax_metal-0.0.4-py3-none-macosx_11_0_arm64.whl";

    let res = reqwest::get(url).await?;
    let bytes = io::Cursor::new(res.bytes().await?);
    zip_extract::extract(bytes, jax_dir, true)?;
    Ok(())
}

async fn download_shared_xla(xla_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    let url = match (os.as_str(), arch.as_str()) {
        ("macos", arch) => format!("https://github.com/elixir-nx/xla/releases/download/v0.7.0/xla_extension-{}-darwin-cpu.tar.gz", arch),
        ("linux", arch) => {
            if true {
              format!("https://github.com/elixir-nx/xla/releases/download/v0.7.0/xla_extension-{}-linux-gnu-cuda12.tar.gz", arch)
            }else{
              format!("https://github.com/elixir-nx/xla/releases/download/v0.7.0/xla_extension-{}-linux-gnu-cpu.tar.gz", arch)
            }
        }

        (os, arch) => panic!("{}-{} is an unsupported platform", os, arch)
    };

    let res = reqwest::get(url).await?;
    let mut bytes = io::Cursor::new(res.bytes().await?);

    let tar = GzDecoder::new(&mut bytes);
    let mut archive = Archive::new(tar);
    archive.unpack(xla_dir)?;

    Ok(())
}

async fn download_xla(xla_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    let url = match (os.as_str(), arch.as_str()) {
        ("macos", arch) => format!("https://github.com/elodin-sys/xla/releases/download/v0.5.4/xla_extension-{}-darwin-cpu.tar.gz", arch),
        ("linux", arch) => format!("https://github.com/elodin-sys/xla/releases/download/v0.5.4/xla_extension-{}-linux-gnu-cpu.tar.gz", arch),
        (os, arch) => panic!("{}-{} is an unsupported platform", os, arch)
    };
    let res = reqwest::get(url).await?;
    let mut bytes = io::Cursor::new(res.bytes().await?);

    let tar = GzDecoder::new(&mut bytes);
    let mut archive = Archive::new(tar);
    archive.unpack(xla_dir)?;

    Ok(())
}
