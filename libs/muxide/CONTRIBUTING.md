# Contributing to Muxide

Thank you for your interest in Muxide!

## Open Source, Not Open Contribution

Muxide is **open source** but **not open contribution**.

- The code is freely available under the MIT license
- You can fork, modify, use, and learn from it without restriction
- **Pull requests are not accepted by default**
- All architectural, roadmap, and merge decisions are made by the project maintainer

This model keeps the project coherent, maintains clear ownership, and ensures consistent quality. It's the same approach used by SQLite and many infrastructure projects.

## How to Contribute

If you believe you can contribute meaningfully to Muxide:

1. **Email the maintainer first**: [michaelallenkuykendall@gmail.com](mailto:michaelallenkuykendall@gmail.com)
2. Describe your background and proposed contribution
3. If there is alignment, a scoped collaboration may be discussed privately
4. Only after discussion will PRs be considered

**Unsolicited PRs will be closed without merge.** This isn't personal — it's how this project operates.

## What We Welcome (via email first)

- Bug reports with detailed reproduction steps (Issues are fine)
- Security vulnerability reports (please email directly)
- Documentation improvements (discuss first)
- Codec-specific bug fixes (discuss first)

## What We Handle Internally

- New features and architectural changes
- API design decisions
- New codec support
- Performance optimizations
- Container format compatibility work

## Bug Reports

Bug reports via GitHub Issues are welcome! Please include:
- Rust version and muxide version
- OS and version
- Minimal reproduction case
- Expected vs actual behavior
- Sample files if relevant (or instructions to generate them)

## Code Style (for reference)

If a contribution is discussed and approved:
- Rust 2021 edition with `cargo fmt` and `cargo clippy`
- Minimal runtime dependencies (only std and essential crates)
- MSRV 1.74 compatibility
- Property-based tests for new functionality
- All public APIs must have documentation

## Muxide Philosophy

Any accepted work must align with:
- **Minimal dependencies**: Only std and essential crates at runtime
- **Pure Rust**: No unsafe, no FFI
- **Strict validation**: Garbage in, error out
- **Standards compliance**: Valid ISO-BMFF output
- **MIT licensed**: No GPL contamination

## Why This Model?

Building reliable multimedia infrastructure requires tight architectural control. This ensures:
- Consistent API design
- No ownership disputes or governance overhead
- Quality control without committee delays
- Clear direction for the project's future

The code is open. The governance is centralized. This is intentional.

## Recognition

Helpful bug reports and community members are acknowledged in release notes.
If email collaboration leads to merged work, attribution will be given appropriately.

## Release Process

Releases are handled by the maintainer using automated tooling:

1. **Version bump**: Update `Cargo.toml` version
2. **Changelog**: Update `CHANGELOG.md` with release notes
3. **Tag creation**: `git tag vX.Y.Z && git push --tags`
4. **Automated publishing**:
   - GitHub Actions builds cross-platform binaries
   - Binaries are attached to the GitHub release
   - Crate is published to crates.io

Pre-built binaries are available for Linux (x86_64) in GitHub releases.

---

**Maintainer**: Michael A. Kuykendall
**Contact**: [michaelallenkuykendall@gmail.com](mailto:michaelallenkuykendall@gmail.com)
