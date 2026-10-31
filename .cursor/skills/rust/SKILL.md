---
name: rust
description: >-
  Elodin Rust style and anti-patterns. Use when writing or reviewing Rust in this
  repo, adding error types, designing APIs, or when the user mentions Rust
  conventions or DONTs.
---

# Rust

## DON'Ts

1. Do not use `String` as an error type. Use `thiserror` crate and an enum in
   general.

2. Do not use accessory functions that do nothing, e.g.,

```rust
    pub fn stats(&self) -> &Snapshot {
        &self.snapshot
    }
```

This is an old object-oriented pattern that we do not need. Expose fields as
`pub` or `pub(crate)` and access them directly.

3. Do not allocate strings if you can help it. Be suspicious of `format!`,
   `to_string()`, `to_owned()`, `into()`, `str::to_lowercase()`,
   `str::to_uppercase()`, `str::replace*()`, `str::repeat()`,
   `Path::to_string_lossy()`, `PathBuf::to_path_buf()`.

4. Avoid `unwrap()` and `expect()` except for mutex locks, logical invariants
   (guaranteed success), prototypes, tests, and examples.
   
5. Don't call `format!` multiple times in succession; it allocates a string each
   time. Use `write!` to build upon one `String`.

## DOs

1. Consider using `Cow<'static, str>` for a field instead of `String` or
   `&'static str` if static strings are the rule but exceptional cases require
   an owned `String`s, e.g., for error enum variants: `Message(Cow<'static,
   str>)`.
