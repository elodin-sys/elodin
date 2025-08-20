use flate2::read::GzDecoder;
use fs_extra::dir as fs_dir;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, io};
use tar::Archive;

const XLA_REV: &str = "2a6015f068e4285a69ca9a535af63173ba92995b";

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

fn main() -> anyhow::Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("missing out dir"));
    let os = OS::get();

    let xla_dir = out_dir.join("xla_extension");
    if !xla_dir.exists() {
        let xla_src_dir = out_dir.join(format!("xla-{XLA_REV}"));
        if !xla_src_dir.exists() {
            download_xla(&out_dir)?;
        }
        build_xla(&out_dir, &xla_src_dir)?;
    }

    let mut config = cpp_build::Config::new();
    config
        .flag("-std=c++17")
        .flag("-Wno-unused-parameter")
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

    let jax_metal_dir = out_dir.join("jax_metal");
    if !jax_metal_dir.exists() && cfg!(target_os = "macos") {
        download_jax_metal(&jax_metal_dir)?;
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

    println!(
        "cargo:rustc-link-search=native={}",
        xla_dir.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=xla_extension");

    if os == OS::MacOS {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=SystemConfiguration");
        println!("cargo:rustc-link-lib=framework=Security");
    }

    Ok(())
}

fn build_xla(out_dir: &Path, xla_dir: &Path) -> anyhow::Result<()> {
    // thread::sleep(Duration::from_secs(20));
    // std::intrinsics::breakpoint();
    let cwd = env::current_dir()?;
    let src_dir = cwd.join("extension");
    let dst_dir = xla_dir.join("xla");
    copy_dir(src_dir, dst_dir)?;
    env::set_current_dir(xla_dir)?;
    Command::new("bazelisk")
        .args([
            "build",
            "--enable_workspace",
            "--experimental_cc_static_library",
            "//xla/extension:tarball",
        ])
        .status()?;
    let tarball = xla_dir.join("bazel-bin/xla/extension/xla_extension.tar.gz");
    let file = File::open(tarball)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    archive.unpack(out_dir)?;
    env::set_current_dir(cwd)?;
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
    let url = format!("https://github.com/openxla/xla/archive/{XLA_REV}.tar.gz");
    let buf = download_file(&url)?;
    let mut bytes = io::Cursor::new(buf);
    let tar = GzDecoder::new(&mut bytes);
    let mut archive = Archive::new(tar);
    archive.unpack(xla_dir)?;
    Ok(())
}

fn download_file(url: &str) -> anyhow::Result<Vec<u8>> {
    let res = ureq::get(url).call()?;
    let mut buf = Vec::new();
    res.into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

fn copy_dir(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> fs_extra::error::Result<u64> {
    let mut options = fs_dir::CopyOptions::new();
    options.overwrite = true; // Overwrite existing files in destination
    fs_dir::copy(src, dst, &options)
}
