fn main() {
    let mut hash = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let is_dirty = std::process::Command::new("git")
        .args(["diff", "--quiet", "--exit-code", "HEAD"])
        .status()
        .map(|status| {
            if status.success() {
                false
            } else {
                true
            }
        });
    match is_dirty {
        Ok(is_dirty) => {
            if is_dirty {
                hash.push('*');
            }
        }
        Err(e) => {
            eprintln!("error: Failed to get git hash {e}");
        }
    }
    println!("cargo:rustc-env=GIT_HASH={}", hash);
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
