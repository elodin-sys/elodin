use std::env;

fn main() {
    let kernels_path = env::var("DEP_NOXLA_NOXLA_KERNELS")
        .expect("noxla did not export path to libnoxla_kernels.a");

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        // println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        // println!("cargo:rustc-link-arg={}", kernels_path);
        // println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
        // Resolve cyclic/late dependencies between kernels <-> lapack/blas.
        println!("cargo:rustc-link-arg=-Wl,--start-group");
        // `link-kernels` feature on `xla` will emit -lnoxla_kernels for us.
        // We also add the math stack explicitly; `lapack-src` has already emitted -L to its OUT_DIR.
        println!("cargo:rustc-link-lib=static=lapack");
        println!("cargo:rustc-link-lib=static=blas");
        println!("cargo:rustc-link-lib=gfortran");
        println!("cargo:rustc-link-lib=quadmath");
        println!("cargo:rustc-link-arg=-Wl,--end-group");
        println!("cargo:rustc-link-lib=stdc++");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}", kernels_path);
        println!("cargo:rustc-link-arg=-Wl,-no_dead_strip_inits_and_terms");
        println!("cargo:rustc-link-lib=c++");
    }
}
