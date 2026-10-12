fn main() {
    println!("cargo:rerun-if-env-changed=CESIUM_NATIVE_DIR");
    #[cfg(feature = "native")]
    {
        let root = std::path::PathBuf::from(std::env::var_os("CESIUM_NATIVE_DIR").expect(
            "cesium requires CESIUM_NATIVE_DIR pointing to a Cesium Native 0.65 installation",
        ));
        cxx_build::bridge("src/lib.rs")
            .file("src/bridge.cpp")
            .include("include")
            .include(root.join("include"))
            .std("c++20")
            .compile("elodin-cesium-bridge");
        println!("cargo:rerun-if-changed=src");
        println!("cargo:rerun-if-changed=include");
        println!(
            "cargo:rustc-link-search=native={}",
            root.join("lib").display()
        );
        let mut libraries: Vec<_> = std::fs::read_dir(root.join("lib"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "a"))
            .collect();
        libraries.sort();
        for library in libraries {
            let name = library
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .strip_prefix("lib")
                .unwrap()
                .to_owned();
            println!("cargo:rustc-link-lib=static={name}");
        }
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
            for framework in ["Security", "SystemConfiguration", "CoreFoundation"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
            println!("cargo:rustc-link-lib=iconv");
            println!("cargo:rustc-link-lib=resolv");
        }
    }
}
