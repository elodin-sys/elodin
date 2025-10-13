use std::env;

fn main() {
    let kernels_path = env::var("DEP_NOXLA_NOXLA_KERNELS")
        .expect("noxla did not export path to libnoxla_kernels.a");

    #[cfg(target_os = "linux")]
    {
        let blas_root =
            env::var("DEP_BLAS_ROOT").expect("netlib-src crate is not a direct dependency!");
        let lapack_path = format!("{}/lib/liblapack-netlib.a", blas_root);
        let blas_path = format!("{}/lib/libblas-netlib.a", blas_root);

        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg={}", kernels_path);
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

        println!("cargo:rustc-link-arg=-Wl,--start-group");
        println!("cargo:rustc-link-arg={}", lapack_path);
        println!("cargo:rustc-link-arg={}", blas_path);
        println!("cargo:rustc-link-lib=dylib=gfortran");
        println!("cargo:rustc-link-arg=-Wl,--end-group");
        // println!("cargo:rustc-link-arg=-Wl,-u,_gfortran_stop_string");
        println!("cargo:rustc-link-lib=stdc++");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}", kernels_path);
        println!("cargo:rustc-link-arg=-Wl,-no_dead_strip_inits_and_terms");
        println!("cargo:rustc-link-lib=c++");
    }
}
