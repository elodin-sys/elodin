/// # Safety
/// The caller must ensure that the input buffer is valid utf8.
pub const unsafe fn buf_and_len_to_str<const MAX_LEN: usize>(
    buf_len: &([u8; MAX_LEN], usize),
) -> &str {
    let buf = &buf_len.0;
    let len = buf_len.1;
    let buf = buf.split_at(len).0;
    core::str::from_utf8_unchecked(buf)
}

pub const fn concat_buf<const MAX_LEN: usize>(left: &str, right: &str) -> ([u8; MAX_LEN], usize) {
    let mut buf = [0u8; MAX_LEN];
    let mut i = 0;
    while i < left.len() {
        buf[i] = left.as_bytes()[i];
        i += 1;
    }
    while i - left.len() < right.len() {
        buf[i] = right.as_bytes()[i - left.len()];
        i += 1;
    }
    (buf, i)
}

#[allow(unused)]
macro_rules! concat_str {
    ($left:expr, $right:expr) => {
        // # Safety: the concatenation of two valid utf8 strings is valid utf8.
        unsafe { crate::buf_and_len_to_str(&crate::concat_buf::<64>($left, $right)) }
    };
}

#[allow(unused)]
pub(crate) use concat_str;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_str() {
        const CONCAT: &str = concat_str!("hello", "world");
        assert_eq!(CONCAT, "helloworld");
    }
}
