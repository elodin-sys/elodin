#[path = "../build-common/git_inspect.rs"]
mod git_inspect;
fn main() {
    #[cfg(feature = "grpc")]
    {
        let protos = [
            "proto/elodin/db/v1/common.proto",
            "proto/elodin/db/v1/ingest.proto",
            "proto/elodin/db/v1/query.proto",
            "proto/elodin/db/v1/stream.proto",
            "proto/elodin/db/v1/msg.proto",
            "proto/elodin/db/v1/admin.proto",
        ];
        for proto in protos {
            println!("cargo:rerun-if-changed={proto}");
        }
        let descriptor = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
            .join("elodin_db_descriptor.bin");
        tonic_prost_build::configure()
            .file_descriptor_set_path(descriptor)
            .compile_protos(&protos, &["proto"])
            .expect("failed to compile gRPC protobufs");
    }

    let hash = git_inspect::short_hash();
    println!(
        "cargo:rustc-env=GIT_HASH={}",
        hash.unwrap_or_else(|| "unknown".to_string())
    );
    if let Some(git_head_path) = git_inspect::head_path() {
        println!("cargo:rerun-if-changed={}", git_head_path);
    }
}
