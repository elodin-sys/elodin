---
name: rust
description: >-
  Elodin Rust style and anti-patterns. Use when writing or reviewing Rust in this
  repo, adding error types, designing APIs, or when the user mentions Rust
  conventions or DONTs.
---

# Rust

## DON'Ts

1. Do not use `String` as an error type. Use `thiserror` crate and an enum in general.

2. Do not use accessory functions that do nothing, e.g.,

```rust
    pub fn stats(&self) -> &Snapshot {
        &self.snapshot
    }
```

This is an old object-oriented pattern that we do not need. Expose fields as
`pub` or `pub(crate)` and access them directly.

3. Do not allocate strings if you can help it. Be suspicious of `format!`, `to_string()`, `to_owned()`, `into()`.

## DOs

1. Consider using `Cow<'static, str>` if static strings are common, but there are exceptional cases that require owned `String`s, e.g., for error enum variants: `Message(Cow<'static, str>)`.
