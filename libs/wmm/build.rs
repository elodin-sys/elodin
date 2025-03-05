use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/basilisk.h");

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("vendor/GeomagnetismHeader.h")
        .clang_arg("-Ivendor")
        .newtype_enum("logLevel_t")
        .derive_partialeq(true)
        .derive_default(true)
        .derive_debug(true)
        .no_copy("(.*)Msg_C")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_path.join("bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write bindings!");

    // Post-process the bindings to make extern blocks unsafe for Rust 2024 edition
    let mut content = String::new();
    fs::File::open(&bindings_path)
        .expect("Failed to open bindings.rs")
        .read_to_string(&mut content)
        .expect("Failed to read bindings.rs");

    let processed_content = content.replace("extern \"C\" {", "unsafe extern \"C\" {");

    fs::File::create(&bindings_path)
        .expect("Failed to create bindings.rs")
        .write_all(processed_content.as_bytes())
        .expect("Failed to write to bindings.rs");

    cc::Build::new()
        .file("vendor/GeomagnetismLibrary.c")
        .cargo_warnings(false)
        .include("src")
        .compile("geomagnetism");
}
