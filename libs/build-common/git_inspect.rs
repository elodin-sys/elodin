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
//!     if let Some(git_head_path) = git_inspect::head_path() {
//!         println!("cargo:rerun-if-changed={}", &git_head_path);
//!     }
//! }
//! ```

/// Returns the short git hash and marks whether its a dirty tree.
///
/// ```rust, ignore
/// assert_eq!(hash(), Some("123456"));  // Clean tree
/// assert_eq!(hash(), Some("123456*")); // Dirty tree
/// assert_eq!(hash(), None);            // Could not get hash.
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
        hash.push('*');
    }
    hash
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
