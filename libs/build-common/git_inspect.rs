#![allow(clippy::needless_doctest_main)]
//! The `git_inspect` module provides git helpers for build.rs scripts.
//!
//! Standard usage might look like this:
//!
//! ```rust, ignore
//! mod git_inspect;
//! fn main() {
//!     let hash = git_inspect::short_hash();
//!     println!(
//!         "cargo:rustc-env=GIT_HASH={}",
//!         hash.unwrap_or_else(|| "unknown".to_string())
//!     );
//!     git_inspect::emit_rerun_directives();
//! }
//! ```

/// Returns the short git hash and, if the tree is dirty, appends `.dirty`.
/// Suffix is PEP 440 / SemVer build-metadata safe (no `*`).
///
/// ```rust, ignore
/// assert_eq!(hash(), Some("123456"));       // Clean tree
/// assert_eq!(hash(), Some("123456.dirty")); // Dirty tree
/// assert_eq!(hash(), None);                 // Could not get hash.
/// ```
pub fn short_hash() -> Option<String> {
    let mut hash = match std::env::var("GIT_HASH") {
        Ok(s) if !s.is_empty() => Some(s),
        _ => std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout)
                        .ok()
                        .map(|s| s.trim().to_string())
                } else {
                    None
                }
            }),
    };

    if let Some(ref mut hash) = hash
        && let Ok(status) = std::process::Command::new("git")
            .args(["diff", "--quiet", "--exit-code", "HEAD"])
            .status()
        && !status.success()
    {
        hash.push_str(".dirty");
    }
    hash
}

/// Rerun on `GIT_HASH` changes; watch `.git/HEAD` only when it is unset or empty.
pub fn emit_rerun_directives() {
    println!("cargo:rerun-if-env-changed=GIT_HASH");
    if std::env::var("GIT_HASH").unwrap_or_default().is_empty()
        && let Some(head) = head_path()
    {
        println!("cargo:rerun-if-changed={head}");
    }
}

/// Returns the git HEAD path if available.
///
/// In a regular repository, this path is `$GIT_ROOT/.git/HEAD` but in a git
/// worktree, it is `$GIT_WORKTREE_SOURCE_ROOT/.git/HEAD`. This function calls
/// out to `git rev-parse --git-path HEAD`, so that it works in all cases.
pub fn head_path() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--git-path", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}
