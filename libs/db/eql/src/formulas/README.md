# EQL Formulas

This directory centralises every formula exposed by the Elodin Query Language (EQL).  
Each formula is implemented in its own `*.rs` file, while `mod.rs` wires them together and re-exports the module documentation via `#![doc = include_str!("README.md")]`.

## Layout
- `mod.rs` – registers the formula modules and exposes this guide in Rustdoc.
- `fft.rs`, `fftfreq.rs`, `first.rs`, `last.rs`, `norm.rs` – reference implementations you can mimic.

## Adding a Formula (cheatsheet)
1. **Create a file**: `<formula>.rs`.
2. **Register the module** in `mod.rs` with `pub mod <formula>;`.
3. **Implement the `EqlFormula` trait** for a zero-sized marker struct:
   - `parse` receives the shared `Arc<dyn EqlFormula>` and must return the EQL expression for this invocation (usually `Expr::Formula(formula, Box::new(recv))`, but it can also construct specialised enums such as `Expr::First`).
   - Override `to_field`, `to_qualified_field`, or `to_column_name` when the defaults (wrapping the inner expression) are not enough.
   - Override `suggestions` when the formula should appear in autocompletion for particular receivers.
4. **Register the formula** in `create_default_registry()` so it is available to the parser and completion engine.

## Scenario A — Formula backed by existing SQL primitives
When PostgreSQL already provides the underlying function (e.g. `sqrt`, `pow`, aggregation functions), only the EQL layer needs to change.

Steps:
1. Implement the transformation logic inside the new module (see `norm.rs` for a complete template).
2. Return the appropriate `Expr` from `parse` (typically `Expr::Formula`) and override the rendering helpers if needed.
3. Cover the behaviour with unit tests (parsing, SQL generation, autocompletion).

## Scenario B — Formula requiring a DataFusion UDF
If the feature does not exist in PostgreSQL/DataFusion, you must implement and register a User Defined Function first.

1. Build the UDF in `libs/db/src/arrow/<formula>.rs` (cf. `fft.rs`).
2. Register it inside `libs/db/src/arrow/mod.rs`.
3. Wire the formula in this directory just like Scenario A, reusing the UDF when generating SQL.
4. Add tests as above to guarantee the plumbing works.
