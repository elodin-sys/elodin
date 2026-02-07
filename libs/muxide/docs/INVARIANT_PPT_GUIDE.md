# Invariant PPT Testing Guide for Muxide

## Overview

This project uses **Predictive Property-Based Testing (PPT)** combined with **runtime invariant enforcement**. This methodology is preferred over traditional TDD for high-change, AI-assisted development.

## Core Philosophy

1. **Properties over Implementations** - Test what must always be true, not specific outputs
2. **Invariants in Code** - Critical rules live next to the code they protect
3. **Contract Tests** - Permanent tests that verify invariants are checked
4. **Exploration Tests** - Temporary tests during development

---

## Test Layers

| Layer | Description | Stability |
|-------|-------------|-----------|
| **E-Test** | Exploration (temporary) | Deleted after feature complete |
| **P-Test** | Property tests (proptest) | Stable, covers edge cases |
| **C-Test** | Contract tests | Permanent, must-pass |

---

## Muxide-Specific Invariants

### MP4 Container Invariants

```rust
// INV-001: Box sizes must match content
assert_invariant!(
    box_size == header_size + payload.len(),
    "MP4 box size must equal header + payload",
    "mp4::box_building"
);

// INV-002: Width/height must be 16-bit in sample entry
assert_invariant!(
    width <= u16::MAX as u32 && height <= u16::MAX as u32,
    "Video dimensions must fit in 16 bits for sample entry",
    "mp4::sample_entry"
);

// INV-003: Duration must match sample count
assert_invariant!(
    duration == samples.iter().map(|s| s.duration).sum::<u64>(),
    "mdhd duration must equal sum of sample durations",
    "mp4::duration"
);

// INV-004: No empty samples in stsz
assert_invariant!(
    samples.iter().all(|s| s.size > 0),
    "All samples must have non-zero size",
    "mp4::stsz"
);
```

### Codec Invariants

```rust
// INV-010: Annex B must have start codes
assert_invariant!(
    data.windows(4).any(|w| w == [0,0,0,1]) || data.windows(3).any(|w| w == [0,0,1]),
    "Annex B data must contain start codes",
    "codec::h264"
);

// INV-011: AVCC must have length prefixes
assert_invariant!(
    !data.windows(4).any(|w| w == [0,0,0,1]),
    "AVCC data must not contain start codes",
    "codec::h264"
);

// INV-012: SPS must precede IDR in keyframe
assert_invariant!(
    keyframe_has_sps_before_idr(data),
    "H.264 keyframe must have SPS before IDR slice",
    "codec::h264"
);
```

---

## Property Tests with proptest

### Example: Codec Roundtrip

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn annexb_avcc_roundtrip(nal_data in prop::collection::vec(any::<u8>(), 1..1000)) {
        // Property: Converting Annex B -> AVCC -> Annex B preserves NAL content
        let annexb = wrap_as_annexb(&nal_data);
        let avcc = annexb_to_avcc(&annexb);
        let back = avcc_to_annexb(&avcc);
        
        // The NAL data (without start codes/lengths) should be preserved
        prop_assert_eq!(extract_nals(&annexb), extract_nals(&back));
    }
}
```

### Example: Timing Monotonicity

```rust
proptest! {
    #[test]
    fn pts_always_increases(
        frame_count in 1..100usize,
        fps in 24.0..60.0f64
    ) {
        let mut muxer = create_test_muxer(fps);
        let mut prev_pts = 0u64;
        
        for i in 0..frame_count {
            let pts = muxer.next_pts();
            prop_assert!(pts > prev_pts, "PTS must monotonically increase");
            prev_pts = pts;
        }
    }
}
```

---

## Contract Tests

Contract tests verify that invariants are actively being checked:

```rust
#[test]
fn contract_mp4_box_building() {
    // This test fails if invariant checks are removed from production code
    contract_test("mp4 box building", &[
        "MP4 box size must equal header + payload",
        "Video dimensions must fit in 16 bits for sample entry",
    ]);
}

#[test]
fn contract_duration_calculation() {
    contract_test("duration calculation", &[
        "mdhd duration must equal sum of sample durations",
    ]);
}
```

---

## Running Tests

```bash
# All tests
cargo test

# Property tests only (more iterations)
cargo test --test '*_props' -- --nocapture
PROPTEST_CASES=1000 cargo test

# With coverage
cargo tarpaulin --out Html --output-dir coverage/

# Specific contract tests
cargo test contract_
```

---

## Adding New Invariants

1. **Identify the invariant** - What must ALWAYS be true?
2. **Add assert_invariant!** in production code
3. **Add property test** exploring edge cases
4. **Add contract test** verifying the invariant is checked
5. **Document** in this file

---

## Coverage Goals

| Module | Target | Current |
|--------|--------|---------|
| `codec::h264` | 90% | TBD |
| `codec::h265` | 90% | TBD |
| `codec::av1` | 85% | TBD |
| `muxer::mp4` | 95% | TBD |
| `api` | 95% | TBD |
| `fragmented` | 85% | TBD |

---

## References

- proptest crate: https://docs.rs/proptest
- Original PPT Guide: See `ppt_invariant_guide.md`
