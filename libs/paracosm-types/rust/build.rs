use std::path::PathBuf;

fn protobuf_dir() -> PathBuf {
    std::env::var("ELODIN_PROTOBUFS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../protobufs"))
}

fn main() {
    tonic_build::compile_protos(protobuf_dir().join("api.proto"))
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
}
