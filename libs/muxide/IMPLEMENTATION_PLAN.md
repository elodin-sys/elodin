# Muxide Implementation Plan: Audit Fixes + Fragmented MP4 Multi-Codec Support

**Date Created:** January 1, 2026  
**Status:** Active  
**Version:** v0.1.5 → v0.2.0  

This document tracks the deterministic implementation plan for fixing remaining audit issues and adding fragmented MP4 support for H.265, AV1, and VP9. Each step has clear deliverables and verification criteria. Check off completed items with `[x]`.

## Phase 1: Fix Remaining Audit Issues

### 1. Fix Opus Code-3 Panic on Malformed Packets
- **Description**: In `opus_frame_count()`, code 3 still panics if `count == 0`. Replace with graceful `return None`.
- **Files**: `src/codec/opus.rs`
- **Deliverable**: Change `assert_invariant!(count > 0, ...)` to `if count == 0 { return None; }`
- **Test**: Extend `test_opus_functions_handle_bad_input_gracefully` to include `[0x03, 0x00]` packet and assert `None` return.
- **Verification**: `cargo test --lib test_opus_functions_handle_bad_input_gracefully` passes.
- [x] **Checkbox**: Implement and test the fix.

### 2. Fix Fragmented Muxer DTS Underflow
- **Description**: Prevent DTS underflow in `ready_to_flush()` and `flush_segment()` by enforcing monotonic DTS.
- **Files**: `src/fragmented.rs`, `src/api.rs` (for DTS validation in `write_video_with_dts`)
- **Deliverable**: Add `last_dts` tracking in `FragmentedMuxer`, reject decreasing DTS with error, or use `checked_sub` and return `None` on underflow.
- **Test**: Add test in `tests/fragmented.rs` that writes samples with decreasing DTS and asserts rejection.
- **Verification**: Test passes, no underflow possible.
- [x] **Checkbox**: Implement monotonic DTS enforcement.

### 3. Fix MP4 MDAT Size Overflow
- **Description**: Prevent `u32` overflow in payload size accumulation.
- **Files**: `src/muxer/mp4.rs`
- **Deliverable**: Accumulate into `u64`, error if exceeds `u32::MAX`, or implement largesize boxes.
- **Test**: Add test with synthetic large samples hitting overflow boundary.
- **Verification**: Test passes, overflow handled deterministically.
- [x] **Checkbox**: Implement overflow-safe size accumulation.

### 4. Fix Documentation Drift (README + Fragmented.rs Examples)
- **Description**: Update examples to match actual `FragmentedMuxer` API (u64 timestamps, no `?` on `()` return).
- **Files**: `README.md`, `src/fragmented.rs`
- **Deliverable**: Correct examples to use proper signatures and units.
- **Test**: Make doc-tests compile (remove `ignore` if applicable).
- **Verification**: Examples compile and run as doc-tests.
- [x] **Checkbox**: Align documentation with API.

### 5. Fix DTS Error Variants (Semantic Drift)
- **Description**: Use DTS-specific error variants instead of PTS ones for DTS validation.
- **Files**: `src/api.rs`
- **Deliverable**: Add `InvalidVideoDts`/`NegativeVideoDts` variants and use them.
- **Test**: Add tests for DTS NaN/Inf/negative expecting DTS-specific errors.
- **Verification**: Tests pass with correct error types.
- [x] **Checkbox**: Implement DTS-specific error variants.

### 6. Add Missing Audio/DTS Non-Finite Tests
- **Description**: Add tests for audio PTS and DTS non-finite rejection.
- **Files**: `tests/input_contract.rs`
- **Deliverable**: `audio_pts_nan_inf_is_rejected` and `dts_nan_inf_is_rejected` tests.
- **Test**: Mirror video test structure.
- **Verification**: Tests pass, coverage confirmed.
- [x] **Checkbox**: Add and verify tests.

## Phase 2: Implement Fragmented MP4 Multi-Codec Support

### 7. Analyze Codec-Specific Requirements
- **Description**: Document what each codec (H.265, AV1, VP9) needs for fragmented MP4 (e.g., parameter sets, timing).
- **Files**: Research in `src/codec/` modules.
- **Deliverable**: Document per-codec requirements (e.g., H.265 needs VPS/SPS/PPS in init segment).
- **Test**: N/A (planning step).
- **Verification**: Clear requirements list.
- [x] **Checkbox**: Complete codec analysis.

### 8. Extend FragmentedMuxer Struct for Multi-Codec
- **Description**: Add codec-specific fields to `FragmentedMuxer` (e.g., parameter set storage).
- **Files**: `src/fragmented.rs`
- **Deliverable**: Update struct to handle H.265/AV1/VP9 init segments.
- **Test**: Struct compiles.
- **Verification**: No breaking changes to H.264 path.
- [x] **Checkbox**: Extend struct.

### 9. Implement H.265 Fragmented Support
- **Description**: Add H.265 logic to init segment (VPS/SPS/PPS) and sample writing.
- **Files**: `src/fragmented.rs`, integrate with `src/codec/h265.rs`
- **Deliverable**: H.265 fragmented muxing works end-to-end.
- **Test**: Add `test_fragmented_h265_basic` test.
- **Verification**: Test passes, generates valid fMP4.
- [x] **Checkbox**: Implement H.265 support.

### 10. Implement AV1 Fragmented Support
- **Description**: Add AV1 logic (OBU parsing for init segment).
- **Files**: `src/fragmented.rs`, integrate with `src/codec/av1.rs`
- **Deliverable**: AV1 fragmented muxing works.
- **Test**: Add `test_fragmented_av1_basic` test.
- **Verification**: Test passes.
- [x] **Checkbox**: Implement AV1 support.

### 11. Implement VP9 Fragmented Support
- **Description**: Add VP9 logic (frame header parsing).
- **Files**: `src/fragmented.rs`, integrate with `src/codec/vp9.rs`
- **Deliverable**: VP9 fragmented muxing works.
- **Test**: Add `test_fragmented_vp9_basic` test.
- **Verification**: Test passes.
- [x] **Checkbox**: Implement VP9 support.

### 12. Update API and Builder
- **Description**: Remove H.264 restriction in `MuxerBuilder::new_with_fragmented()`.
- **Files**: `src/api.rs`
- **Deliverable**: Accept all video codecs for fragmented.
- **Test**: Update existing test to accept other codecs.
- **Verification**: Builder works for all codecs.
- [x] **Checkbox**: Update API.

### 13. Update Documentation and Examples
- **Description**: Update README and docs for multi-codec fragmented support.
- **Files**: `README.md`, `src/fragmented.rs`
- **Deliverable**: Accurate examples for all codecs.
- **Test**: Doc-tests compile.
- **Verification**: Documentation matches implementation.
- [x] **Checkbox**: Update docs.

### 14. Full Integration Testing
- **Description**: End-to-end tests for all codecs in fragmented mode.
- **Files**: `tests/` (new integration tests).
- **Deliverable**: Comprehensive test suite.
- **Test**: All fragmented tests pass.
- **Verification**: `cargo test` clean.
- [x] **Checkbox**: Complete testing.

### 15. Update Roadmap and Version
- **Description**: Mark fragmented multi-codec as complete in ROADMAP.md, prepare for v0.2.0.
- **Files**: `ROADMAP.md`, `Cargo.toml`
- **Deliverable**: Updated docs and version bump.
- **Test**: N/A.
- **Verification**: Roadmap reflects completion.
- [x] **Checkbox**: Finalize versioning.

## Progress Tracking
- **Total Steps:** 15
- **Completed:** 15
- **Remaining:** 0
- **Current Phase:** Complete - Ready for v0.2.0 Release

## Notes
- Start with Phase 1 to fix existing issues before adding new features.
- Run `cargo test` after each step to ensure no regressions.
- Update this document as steps are completed.</content>
<parameter name="filePath">c:\Users\micha\repos\muxide\IMPLEMENTATION_PLAN.md