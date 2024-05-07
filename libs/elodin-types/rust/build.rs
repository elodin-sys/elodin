use std::{env, path::PathBuf};

fn protobuf_dir() -> PathBuf {
    std::env::var("ELODIN_PROTOBUFS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../protobufs"))
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let protos = ["api.proto", "sandbox.proto"].map(|p| protobuf_dir().join(p));
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .type_attribute(
            "elodin.types.api.LicenseType",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        .type_attribute(
            "elodin.types.api.OnboardingData",
            "#[derive(serde::Serialize, serde::Deserialize)]",
        )
        // only needed for libprotoc versions <= 3.14, optional keyword is supported by default in >= 3.15
        .protoc_arg("--experimental_allow_proto3_optional")
        .file_descriptor_set_path(out_dir.join("elodin_types.bin"))
        .compile(&protos, &[protobuf_dir()])
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
}
