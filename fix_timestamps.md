# fix-timestamps

This document explains how to use the `fix-timestamps` command in `elodin-db` to realign
database timestamps and (by default) remove empty components.

## Usage

```bash
cargo run --bin elodin-db --manifest-path /chemin/vers/elodin/libs/db/Cargo.toml -- \
  fix-timestamps [OPTIONS] <PATH>
```

`<PATH>` is the path to the database directory (e.g. `~/Downloads/my_db`).

## Options

- `--dry-run`
  - Shows what would be changed without modifying files.
  - Useful to validate the offset and the list of affected components.
- `-y`, `--yes`
  - Skips the interactive confirmation prompt.
  - Handy for scripts.
- `--reference <wall-clock|monotonic>`
  - Sets the reference clock used to compute the offset.
  - `wall-clock` (default): epoch timestamps (>= 2020) are the reference,
    monotonic timestamps (< 2000) are corrected.
  - `monotonic`: monotonic timestamps (< 2000) are the reference,
    wall-clock timestamps (>= 2020) are corrected.
- `--no-prune`
  - Disables pruning empty components.
  - By default, empty components are removed (when `--dry-run` is not used).

## Examples

### Dry run with wall-clock reference (default)

```bash
cargo run --bin elodin-db --manifest-path /chemin/vers/elodin/libs/db/Cargo.toml -- \
  fix-timestamps --dry-run ~/Downloads/my_db
```

### Convert to monotonic (wall-clock -> monotonic)

```bash
cargo run --bin elodin-db --manifest-path /chemin/vers/elodin/libs/db/Cargo.toml -- \
  fix-timestamps --dry-run --reference monotonic ~/Downloads/my_db
```

### Apply changes without prompt and without pruning

```bash
cargo run --bin elodin-db --manifest-path /chemin/vers/elodin/libs/db/Cargo.toml -- \
  fix-timestamps --yes --no-prune ~/Downloads/my_db
```
