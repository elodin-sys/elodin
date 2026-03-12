fn main() {
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

    println!(
        "cargo:rustc-env=GIT_HASH={}",
        hash.unwrap_or_else(|| "unknown".to_string())
    );
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
