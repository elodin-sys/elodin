# Release Audit Checklist (Muxide)

This checklist is meant to support a "clean as a whistle" release: consistent docs, no obvious drift, no lint/style debt, and interoperable output for at least one mainstream toolchain.

## 1) Tooling Gates (must pass)
- `cargo fmt --check`
- `cargo clippy --all-targets` (no warnings)
- `cargo test` (unit + integration + doctests)

## 2) API/Docs Consistency
- README examples compile against current public API.
- Version references in docs match `Cargo.toml` (or explicitly state historical context).
- Avoid misleading comments like "stub" when implementation is real.

## 3) fMP4 / Fragmented MP4 Correctness (minimum bar)
- Sample format requirements are explicit (Annex B vs length-prefixed) and enforced or converted.
- Codec configuration boxes are structurally consistent (e.g., HEVC `hvcC` array count matches emitted arrays).
- Interop sanity check: ffprobe parses generated H.264 MP4 and concatenated init+segment fMP4.

## 4) CLI Output Integrity
- CLI commands should not claim codec detection unless parsing is actually meaningful.
- Errors must be actionable and not panic on malformed input.

## 5) Regression Safety
- Tests cover the invariants we rely on (timestamps monotonicity, bounds, required codec headers).
- Any previously "green" benchmarks/tests remain green.
