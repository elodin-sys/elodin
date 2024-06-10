use std::env;
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
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

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
