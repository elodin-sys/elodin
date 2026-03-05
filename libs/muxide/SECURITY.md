# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Muxide, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please email: **michaelallenkuykendall@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to Expect

- **Acknowledgment:** Within 48 hours
- **Initial assessment:** Within 7 days
- **Resolution timeline:** Depends on severity, typically 30-90 days

### Scope

Security issues we care about:
- Memory safety issues (buffer overflows, use-after-free)
- Malformed input causing crashes or hangs
- Resource exhaustion (memory, CPU)
- Output files that could exploit media players

### Out of Scope

- Denial of service via extremely large files (expected behavior)
- Issues in dev-dependencies (proptest, lazy_static)
- Theoretical issues without practical exploit

## Security Design

Muxide is designed with security in mind:

1. **Pure Rust** - Memory safety enforced by Rust’s guarantees
2. **Minimal runtime dependencies** - Limited third-party runtime dependency supply chain
3. **No unsafe code** - All code is safe Rust
4. **Input validation** - All inputs are validated before processing
5. **Bounded operations** - Avoids unbounded allocations from user input where practical

## Acknowledgments

We thank security researchers who help keep Muxide safe. Contributors will be acknowledged here (with permission).
