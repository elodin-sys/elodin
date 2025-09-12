fn main() {
	let noxla = "libnoxla.a";

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg={}", noxla);
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
        println!("cargo:rustc-link-lib=stdc++");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}", noxla);
        println!("cargo:rustc-link-arg=-Wl,-no_dead_strip_inits_and_terms");
        println!("cargo:rustc-link-lib=c++");
    }
}
