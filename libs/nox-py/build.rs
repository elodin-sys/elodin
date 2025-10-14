use std::env;
use std::path::Path;

fn main() {
    let kernels_path = env::var("DEP_NOXLA_NOXLA_KERNELS")
        .expect("noxla did not export path to libnoxla_kernels.a");

    #[cfg(target_os = "linux")]
    link_linux_libraries(&kernels_path);

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}", kernels_path);
        println!("cargo:rustc-link-arg=-Wl,-no_dead_strip_inits_and_terms");
        println!("cargo:rustc-link-lib=c++");
    }
}

#[cfg(target_os = "linux")]
fn link_linux_libraries(kernels_path: &str) {
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-Wl,--whole-archive");
    println!("cargo:rustc-link-arg={}", kernels_path);
    println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

    let mut use_pkg_config = false;

    match env::var("DEP_BLAS_ROOT") {
        Ok(blas_root) => {
            let lapack_path = format!("{}/lib/liblapack-netlib.a", blas_root);
            let blas_path = format!("{}/lib/libblas-netlib.a", blas_root);

            if Path::new(&lapack_path).exists() && Path::new(&blas_path).exists() {
                println!("cargo:rustc-link-arg=-Wl,--start-group");
                println!("cargo:rustc-link-arg={}", lapack_path);
                println!("cargo:rustc-link-arg={}", blas_path);
                println!("cargo:rustc-link-arg=-Wl,--end-group");
            } else {
                println!(
                    "cargo:warning=Expected static Netlib archives at {lapack_path} and {blas_path}, falling back to pkg-config lookup."
                );
                use_pkg_config = true;
            }
        }
        Err(err) => {
            println!(
                "cargo:warning=DEP_BLAS_ROOT not provided ({}); falling back to pkg-config lookup for BLAS/LAPACK.",
                err
            );
            use_pkg_config = true;
        }
    }

    if use_pkg_config {
        let mut pkg_errors = Vec::new();
        let mut have_lapack = false;
        let mut have_blas = false;

        match pkg_config::Config::new().probe("lapack") {
            Ok(_) => have_lapack = true,
            Err(err) => pkg_errors.push(format!("lapack: {err}")),
        }

        match pkg_config::Config::new().probe("blas") {
            Ok(_) => have_blas = true,
            Err(err) => pkg_errors.push(format!("blas: {err}")),
        }

        if !(have_lapack && have_blas) {
            match pkg_config::Config::new().probe("openblas") {
                Ok(_) => {
                    have_lapack = true;
                    have_blas = true;
                    pkg_errors.clear();
                }
                Err(err) => pkg_errors.push(format!("openblas: {err}")),
            }
        }

        if !(have_lapack && have_blas) {
            panic!(
                "Failed to locate BLAS/LAPACK via pkg-config. Set DEP_BLAS_ROOT via netlib-src or install lapack/blas with pkg-config support. Errors: {}",
                pkg_errors.join(", ")
            );
        }
    }

    println!("cargo:rustc-link-lib=dylib=gfortran");
    println!("cargo:rustc-link-lib=stdc++");
}
