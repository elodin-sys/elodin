#[path = "../build-common/git_inspect.rs"]
mod git_inspect;
fn main() {
    #[cfg(feature = "grpc")]
    {
        println!("cargo:rerun-if-changed=proto/elodin/db/v1/ingest.proto");
        tonic_prost_build::configure()
            .compile_protos(&["proto/elodin/db/v1/ingest.proto"], &["proto"])
            .expect("failed to compile gRPC protobufs");
    }

    let hash = git_inspect::short_hash();
    println!(
        "cargo:rustc-env=GIT_HASH={}",
        hash.unwrap_or_else(|| "unknown".to_string())
    );
    if let Some(git_head_path) = git_inspect::head_path() {
        println!("cargo:rerun-if-changed={}", &git_head_path);
    }
}
