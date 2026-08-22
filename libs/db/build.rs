#[path = "../build-common/git_inspect.rs"]
mod git_inspect;

fn main() {
    #[cfg(feature = "grpc")]
    compile_grpc_protos();

    let hash = git_inspect::short_hash();
    println!(
        "cargo:rustc-env=GIT_HASH={}",
        hash.unwrap_or_else(|| "unknown".to_string())
    );
    if let Some(git_head_path) = git_inspect::head_path() {
        println!("cargo:rerun-if-changed={}", &git_head_path);
    }
}

#[cfg(feature = "grpc")]
fn compile_grpc_protos() {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    println!("cargo:rerun-if-env-changed=PROTOC");
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
    let descriptor =
        PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("elodin_db_descriptor.bin");
    let mut config = tonic_prost_build::Config::new();
    config.protoc_executable(resolve_protoc());
    tonic_prost_build::configure()
        .file_descriptor_set_path(descriptor)
        .compile_with_config(config, &protos, &["proto"])
        .expect("failed to compile gRPC protobufs");

    fn resolve_protoc() -> PathBuf {
        if let Some(path) = std::env::var_os("PROTOC")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
        {
            if proto3_optional_supported(&path) {
                return path;
            }
            println!(
                "cargo:warning=PROTOC ({}) is too old for proto3 optional; using vendored protoc",
                path.display()
            );
        } else if let Some(path) = find_protoc_on_path() {
            if proto3_optional_supported(&path) {
                return path;
            }
            println!(
                "cargo:warning={} is too old for proto3 optional; using vendored protoc",
                path.display()
            );
        }
        protoc_bin_vendored::protoc_bin_path().expect("vendored protoc for this host")
    }

    fn find_protoc_on_path() -> Option<PathBuf> {
        let path_var = std::env::var_os("PATH")?;
        std::env::split_paths(&path_var).find_map(|dir| {
            let unix = dir.join("protoc");
            if unix.is_file() {
                return Some(unix);
            }
            let windows = dir.join("protoc.exe");
            windows.is_file().then_some(windows)
        })
    }

    fn proto3_optional_supported(protoc: &Path) -> bool {
        let Ok(output) = Command::new(protoc).arg("--version").output() else {
            return false;
        };
        if !output.status.success() {
            return false;
        }
        let Ok(stdout) = String::from_utf8(output.stdout) else {
            return false;
        };
        // proto3 optional is stable in 3.15+. After 3.21 Google jumped to 22+.
        let version = stdout.split_whitespace().last().unwrap_or("");
        let mut parts = version.split('.');
        let major: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        major > 3 || (major == 3 && minor >= 15)
    }
}
