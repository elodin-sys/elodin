#[path = "../../libs/build-common/git_inspect.rs"]
mod git_inspect;
fn main() {
    let hash = git_inspect::short_hash();
    println!(
        "cargo:rustc-env=GIT_HASH={}",
        hash.unwrap_or_else(|| "unknown".to_string())
    );
    git_inspect::emit_rerun_directives();
}
