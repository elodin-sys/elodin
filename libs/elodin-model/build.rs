//! Embeds the model `ELODIN_MODEL_DIR` holds, as written by `JaxSim.export_model`, compiled by
//! `ELODIN_MODEL_BACKEND`: `aot` (default) links its `step.o`, `jit` compiles its StableHLO when
//! the library is opened. Without `ELODIN_MODEL_DIR` the library holds no model.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(elodin_model, values(\"aot\", \"jit\"))");
    println!("cargo::rerun-if-env-changed=ELODIN_MODEL_DIR");
    println!("cargo::rerun-if-env-changed=ELODIN_MODEL_BACKEND");
    println!("cargo::rerun-if-env-changed=ELODIN_MODEL_BUILD_INFO_PATH");
    // Otherwise executables record the library's absolute build path.
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => println!("cargo::rustc-cdylib-link-arg=-Wl,-soname,libelodin_model.so"),
        Ok("macos") => {
            println!("cargo::rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libelodin_model.dylib")
        }
        _ => {}
    }

    let Some(dir) = env::var_os("ELODIN_MODEL_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let dir = dir.canonicalize().unwrap_or_else(|e| {
        panic!(
            "ELODIN_MODEL_DIR {} (relative to libs/elodin-model): {e}",
            dir.display()
        )
    });
    let backend = env::var("ELODIN_MODEL_BACKEND").unwrap_or_else(|_| "aot".to_string());
    let code = match backend.as_str() {
        "aot" => "step.o",
        "jit" => "step.stablehlo.mlir",
        other => panic!("ELODIN_MODEL_BACKEND must be aot or jit, not {other:?}"),
    };
    for file in [code, "state.bin", "schema.json"] {
        let path = dir.join(file);
        assert!(path.is_file(), "{} is missing", path.display());
        println!("cargo::rerun-if-changed={}", path.display());
    }
    let supplied_manifest = env::var_os("ELODIN_MODEL_BUILD_INFO_PATH").map(PathBuf::from);
    let manifest = supplied_manifest.unwrap_or_else(|| dir.join("model-manifest.json"));
    let manifest = if manifest.is_file() {
        println!("cargo::rerun-if-changed={}", manifest.display());
        manifest
    } else {
        let fallback =
            PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("model-manifest.json");
        fs::write(&fallback, "{}").expect("write model-manifest fallback");
        fallback
    };

    println!("cargo::rustc-env=ELODIN_MODEL_DIR={}", dir.display());
    println!(
        "cargo::rustc-env=ELODIN_MODEL_BUILD_INFO={}",
        manifest.display()
    );
    println!("cargo::rustc-cfg=elodin_model=\"{backend}\"");
    if backend == "aot" {
        cc::Build::new()
            .object(dir.join(code))
            .cargo_metadata(true)
            .compile("elodin_model_step");
    }
}
