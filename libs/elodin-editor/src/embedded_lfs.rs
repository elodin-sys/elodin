//! [`embedded_lfs_asset`] wraps Bevy's [`bevy::asset::embedded_asset!`] with a compile-time
//! check that the file is not a Git LFS pointer stub.

#[doc(hidden)]
pub const fn bytes_look_like_git_lfs_pointer(bytes: &[u8]) -> bool {
    const PREFIX: &[u8] = b"version https://git-lfs.github.com/spec/v1";
    if bytes.len() < PREFIX.len() {
        return false;
    }
    let mut i = 0;
    while i < PREFIX.len() {
        if bytes[i] != PREFIX[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Same as [`bevy::asset::embedded_asset!`], but compilation fails if the file is a Git LFS
/// pointer (text stub) instead of the real binary.
macro_rules! embedded_lfs_asset {
    ($app:expr, $path:literal) => {{
        const _: () = ::core::assert!(
            !$crate::embedded_lfs::bytes_look_like_git_lfs_pointer(::core::include_bytes!($path)),
            concat!(
                "embedded asset is a Git LFS pointer stub, not real binary data: ",
                $path,
                " (run `git lfs pull`)"
            )
        );
        ::bevy::asset::embedded_asset!($app, $path);
    }};
    ($app:expr, $source_path:literal, $path:literal) => {{
        const _: () = ::core::assert!(
            !$crate::embedded_lfs::bytes_look_like_git_lfs_pointer(::core::include_bytes!($path)),
            concat!(
                "embedded asset is a Git LFS pointer stub, not real binary data: ",
                $path,
                " (run `git lfs pull`)"
            )
        );
        ::bevy::asset::embedded_asset!($app, $source_path, $path);
    }};
}

pub(crate) use embedded_lfs_asset;

#[cfg(test)]
mod tests {
    use super::bytes_look_like_git_lfs_pointer;

    #[test]
    fn detects_lfs_pointer_prefix() {
        let s = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n";
        assert!(bytes_look_like_git_lfs_pointer(s));
    }

    #[test]
    fn binary_header_not_lfs() {
        let glb_header: [u8; 12] = [
            0x67, 0x6C, 0x54, 0x46, 0x01, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00,
        ];
        assert!(!bytes_look_like_git_lfs_pointer(&glb_header));
    }
}
