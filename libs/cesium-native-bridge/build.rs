fn main() {
    println!("cargo:rerun-if-env-changed=CESIUM_NATIVE_DIR");
    {
        let root = native_install();
        assert!(
            root.join("include/Cesium3DTilesSelection/Tileset.h")
                .is_file(),
            "Cesium Native is required by the Editor. Run `just prepare-cesium` inside nix develop, or set CESIUM_NATIVE_DIR to an existing 0.65 installation."
        );
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

fn native_install() -> std::path::PathBuf {
    use std::{env, path::PathBuf};
    for key in ["CESIUM_BUILD_ROOT", "XDG_CACHE_HOME", "HOME"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    if let Some(path) = env::var_os("CESIUM_NATIVE_DIR") {
        return PathBuf::from(path);
    }
    let root = env::var_os("CESIUM_BUILD_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let cache = env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(
                        env::var_os("HOME").expect("Set CESIUM_NATIVE_DIR when HOME is unset"),
                    )
                    .join(".cache")
                });
            cache.join("elodin/cesium-0.65.0")
        });
    root.join("install")
}
