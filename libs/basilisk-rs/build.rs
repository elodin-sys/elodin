use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/basilisk.h");

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/basilisk.h")
        .clang_arg("-Ivendor")
        .newtype_enum("logLevel_t")
        .derive_partialeq(true)
        .derive_default(true)
        .derive_debug(true)
        .no_copy("(.*)Msg_C")
        .blocklist_item(".*MaybeUninit.*")
        .wrap_unsafe_ops(true)
        .default_macro_constant_type(bindgen::MacroTypeVariation::Signed)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_file = out_path.join("bindings.rs");
    bindings
        .write_to_file(&bindings_file)
        .expect("Couldn't write bindings!");

    // Post-process the generated bindings to fix extern blocks
    let contents = fs::read_to_string(&bindings_file).expect("Unable to read bindings file");
    let updated_contents = contents.replace("extern \"C\" {", "unsafe extern \"C\" {");
    fs::write(&bindings_file, updated_contents).expect("Unable to write updated bindings file");

    cc::Build::new()
        .file("vendor/fswAlgorithms/attControl/mrpSteering/mrpSteering.c")
        .file("vendor/fswAlgorithms/attControl/mrpFeedback/mrpFeedback.c")
        .file("vendor/fswAlgorithms/attControl/mrpPD/mrpPD.c")
        .file("vendor/architecture/utilities/rigidBodyKinematics.c")
        .file("vendor/architecture/utilities/linearAlgebra.c")
        .file("vendor/fswAlgorithms/attGuidance/hillPoint/hillPoint.c")
        .file("vendor/fswAlgorithms/effectorInterfaces/rwMotorVoltage/rwMotorVoltage.c")
        .file("vendor/fswAlgorithms/effectorInterfaces/rwMotorTorque/rwMotorTorque.c")
        .file("vendor/fswAlgorithms/attDetermination/sunlineEKF/sunlineEKF.c")
        .file("vendor/fswAlgorithms/attGuidance/attTrackingError/attTrackingError.c")
        .cargo_warnings(false)
        .include("vendor")
        .compile("basilisk");
}
