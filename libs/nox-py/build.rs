use std::env;

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

    match env::var("DEP_BLAS_ROOT") {
        Ok(blas_root) => {
            let lapack_path = format!("{}/lib/liblapack-netlib.a", blas_root);
            let blas_path = format!("{}/lib/libblas-netlib.a", blas_root);

            println!("cargo:rustc-link-arg=-Wl,--start-group");
            println!("cargo:rustc-link-arg={}", lapack_path);
            println!("cargo:rustc-link-arg={}", blas_path);
            println!("cargo:rustc-link-arg=-Wl,--end-group");
        }
        Err(err) => {
            println!(
                "cargo:warning=DEP_BLAS_ROOT not provided ({}); falling back to pkg-config lookup for BLAS/LAPACK.",
                err
            );

            let mut pkg_errors = Vec::new();
            let mut found = false;
            for lib in ["lapack", "openblas"] {
                match pkg_config::Config::new().probe(lib) {
                    Ok(_) => {
                        found = true;
                        break;
                    }
                    Err(probe_err) => {
                        pkg_errors.push(format!("{lib}: {probe_err}"));
                    }
                }
            }

            if !found {
                panic!(
                    "Failed to locate a BLAS/LAPACK implementation. Set DEP_BLAS_ROOT via netlib-src or install lapack/openblas with pkg-config support. Errors: {}",
                    pkg_errors.join(", ")
                );
            }
        }
    }

    println!("cargo:rustc-link-lib=dylib=gfortran");
    println!("cargo:rustc-link-lib=stdc++");
}
