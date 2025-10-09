use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn main() {
    // We're picking one file out of the handful of git LFS files present. It is
    // the smallest.
    let path = Path::new("src/assets/fonts/IBMPlexMono-Medium_ss04.ttf");

    // You can create an LFS pointer file to test this code. You can't check it
    // in though without upsetting git-lfs.
    //
    // ```sh
    // $ cat test_lfs_pointer.txt
    // version https://git-lfs.github.com/spec/v1
    // oid sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
    // size 133872
    // ```
    // let path = Path::new("test_lfs_pointer.txt");
    check_lfs(path);
}

fn check_lfs(path: &Path) {
    if std::env::var_os("BUILDKITE")
        .map(|value| value == OsStr::new("true"))
        .unwrap_or(false)
    {
        eprintln!("WARN: Skipping LFS check on buildkite. See issue #208.");
        eprintln!("https://github.com/elodin-sys/elodin/issues/208");
        return;
    }
    // Check if the file exists.
    if !path.exists() {
        panic!("LFS file not found: {}", path.display());
    }

    // Read only the first few bytes of the file.
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) => {
            panic!("Failed to open LFS file {:?}: {}", path.display(), e);
        }
    };

    // Git LFS pointers start with "version https://git-lfs.github.com/spec/".
    let git_lfs_prefix = b"version https://git-lfs.github.com/spec/";
    const HEADER_LENGTH: usize = 40;
    assert_eq!(HEADER_LENGTH, git_lfs_prefix.len());
    let mut buffer = [0u8; HEADER_LENGTH]; // Read exactly enough bytes for Git LFS check.
    let bytes_read = match file.read(&mut buffer) {
        Ok(bytes_read) => bytes_read,
        Err(e) => {
            panic!("Failed to read LFS file {:?}: {}", path.display(), e);
        }
    };

    // Check if we read enough bytes.
    if bytes_read < HEADER_LENGTH {
        panic!(
            "LFS file {:?} is too small ({} bytes). This might be a Git LFS pointer file. Please run 'git lfs pull' to download the actual font file.",
            path.display(),
            bytes_read
        );
    }

    // Check if the first bytes look like a Git LFS pointer.
    if buffer.starts_with(git_lfs_prefix) {
        panic!(
            "LFS file {:?} appears to be a Git LFS pointer file. The actual file was not downloaded. Please run 'git lfs pull' to download the font file.",
            path.display()
        );
    }

    println!("cargo:rerun-if-changed={}", path.display());
}
