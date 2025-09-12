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
        println!("cargo:rustc-link-lib=stdc++");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}", kernels_path);
        println!("cargo:rustc-link-arg=-Wl,-no_dead_strip_inits_and_terms");
        println!("cargo:rustc-link-lib=c++");
    }
}
