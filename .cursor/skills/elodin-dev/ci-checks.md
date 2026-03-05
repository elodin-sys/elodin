# CI Checks Reference

All of these checks are enforced by CI. Run the relevant ones locally before pushing.

## Rust

```bash
# Format all Rust code (applies fixes in-place)
cargo fmt

# Run all tests
cargo test

# Lint — warnings are treated as errors
cargo clippy -- -Dwarnings
```

### Common clippy fixes

- Unused imports/variables: remove them or prefix with `_`
- Missing `#[must_use]`: add the attribute to functions returning values that shouldn't be ignored
- Redundant closures: simplify `|x| foo(x)` to `foo`

## Python

```bash
# Check formatting (dry-run)
ruff format --check

# Apply formatting
ruff format

# Lint + auto-fix
ruff check --fix
```

Ruff configuration is in `ruff.toml` at the repo root.

## Nix

```bash
# Format all .nix files
alejandra
```

Alejandra is an opinionated Nix formatter with no configuration. It formats all `.nix` files in the repository.

## Running All Checks

Quick one-liner to run everything:

```bash
cargo fmt && cargo test && cargo clippy -- -Dwarnings && ruff format --check && ruff check --fix && alejandra
```

## Notes

- `cargo test` may skip some tests that require hardware or specific system configuration
- `cargo clippy` uses the workspace's `rust-toolchain.toml` (edition 2024)
- Python checks apply to all `.py` files in the repository
- Nix formatting applies to `flake.nix`, `aleph/**/*.nix`, and `nix/**/*.nix`
