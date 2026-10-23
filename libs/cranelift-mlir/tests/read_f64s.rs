mod common;

use std::borrow::Cow;

use common::{f64_buf, read_f64s};

fn sample_f64s() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.5,
        -2.25,
        f64::MIN,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_0000_0000_0001),
    ]
}

fn assert_same_bits(got: &[f64], expected: &[f64]) {
    assert_eq!(got.len(), expected.len());
    for (i, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "element {i}: {got} vs {expected}"
        );
    }
}

/// Little-endian bytes placed at an offset that is either 8-aligned or one byte off.
fn placed_le_bytes(vals: &[f64], misaligned: bool) -> (Vec<u8>, usize) {
    let nbytes = vals.len() * 8;
    let mut storage = vec![0u8; nbytes + 16];
    let aligned = (8 - (storage.as_ptr().addr() % 8)) % 8;
    let start = aligned + usize::from(misaligned);
    for (i, value) in vals.iter().enumerate() {
        let at = start + i * 8;
        storage[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }
    (storage, start)
}

#[test]
fn f64_buf_round_trip_preserves_bits() {
    let vals = sample_f64s();
    let bytes = f64_buf(&vals);
    assert_same_bits(&read_f64s(&bytes), &vals);
}

#[test]
fn aligned_buffer_borrows_on_little_endian() {
    let vals = sample_f64s();
    let (storage, start) = placed_le_bytes(&vals, false);
    let end = start + vals.len() * 8;
    let bytes = &storage[start..end];
    assert_eq!(bytes.as_ptr().addr() % 8, 0);

    let back = read_f64s(bytes);
    assert_same_bits(&back, &vals);
    if cfg!(target_endian = "little") {
        let Cow::Borrowed(slice) = &back else {
            panic!("aligned little-endian buffer should borrow");
        };
        assert_eq!(slice.as_ptr(), bytes.as_ptr().cast());
    } else {
        assert!(
            matches!(back, Cow::Owned(_)),
            "big-endian must copy little-endian bytes"
        );
    }
}

#[test]
fn unaligned_buffer_is_owned() {
    let vals = sample_f64s();
    let (storage, start) = placed_le_bytes(&vals, true);
    let end = start + vals.len() * 8;
    let bytes = &storage[start..end];
    assert_ne!(bytes.as_ptr().addr() % 8, 0);

    let back = read_f64s(bytes);
    assert_same_bits(&back, &vals);
    assert!(
        matches!(back, Cow::Owned(_)),
        "misaligned buffer cannot be cast in place"
    );
}

#[test]
fn trailing_byte_forces_an_owned_copy() {
    let vals = [1.25, -4.5];
    let mut bytes = f64_buf(&vals);
    bytes.push(0xab);
    let back = read_f64s(&bytes);
    assert_same_bits(&back, &vals);
    assert!(
        matches!(back, Cow::Owned(_)),
        "a length that is not a multiple of 8 cannot be borrowed"
    );
}
