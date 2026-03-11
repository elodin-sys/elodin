fn main() {
    let mut hash = match std::env::var("GIT_HASH") {
        Ok(s) if !s.is_empty() => s,
        _ => std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "unknown".to_string()),
    };

    if std::env::var("GIT_HASH").is_err() {
        if let Ok(status) = std::process::Command::new("git")
            .args(["diff", "--quiet", "--exit-code", "HEAD"])
            .status()
        {
            if !status.success() {
                hash.push('*');
            }
        }
    }

    println!("cargo:rustc-env=GIT_HASH={}", hash);
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
