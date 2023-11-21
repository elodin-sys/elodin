use std::{env, path::PathBuf};

fn protobuf_dir() -> PathBuf {
    std::env::var("ELODIN_PROTOBUFS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../protobufs"))
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .file_descriptor_set_path(out_dir.join("paracosm_types.bin"))
        .compile(&[protobuf_dir().join("api.proto")], &[protobuf_dir()])
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
}
