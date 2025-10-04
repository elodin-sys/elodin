# Code Review Fix - NSApplication Memory Safety

## Issue Identified by Reviewer

**Critical Memory Safety Bug** in `set_icon_mac()` function.

### The Problem

```rust
// ❌ INCORRECT - causes use-after-free
let app: Retained<NSApplication> = msg_send_id![NSApplication::class(), sharedApplication];
```

**Why this is wrong:**
1. `sharedApplication` returns the global singleton with **+0 retain count** (non-owning reference)
2. `Retained<T>` assumes **ownership** and calls `release` on drop
3. When the function returns, `Retained` drops and releases the singleton
4. This potentially deallocates the global NSApplication object
5. AppKit is left with a **dangling pointer** → crash on next use

### The Fix

```rust
// ✅ CORRECT - uses raw pointer for non-owning reference
let app: *mut NSApplication = msg_send![NSApplication::class(), sharedApplication];
```

**Why this is correct:**
1. `msg_send!` (not `msg_send_id!`) returns a raw pointer without ownership semantics
2. No `Retained` wrapper = no automatic `release` call
3. The singleton remains valid for the lifetime of the application
4. Proper memory safety preserved

### Code Changes

**Before:**
```rust
unsafe {
    let app: Retained<NSApplication> = msg_send_id![NSApplication::class(), sharedApplication];
    // ... use app ...
    let _: () = msg_send![&app, setApplicationIconImage: &*app_icon];
}
// ❌ app drops here, calling release on the singleton
```

**After:**
```rust
unsafe {
    // Get unowned reference to shared app singleton (+0 retain count)
    // Do NOT wrap in Retained - sharedApplication returns a non-owning reference
    let app: *mut NSApplication = msg_send![NSApplication::class(), sharedApplication];
    if app.is_null() {
        return;
    }
    // ... use app ...
    let _: () = msg_send![app, setApplicationIconImage: &*app_icon];
}
// ✅ No automatic release - singleton remains valid
```

### Key Learnings

**objc2 Memory Management Rules:**
- `msg_send_id!` → Returns `Retained<T>` (owned, will release on drop)
- `msg_send!` → Returns raw pointer (unowned, caller manages lifetime)

**Objective-C Naming Conventions:**
- Methods with `alloc`, `new`, `copy`, `mutableCopy` → **+1 retain count** (ownership transfer)
- Other methods (like `sharedApplication`) → **+0 retain count** (non-owning reference)

### Testing

- ✅ Code compiles successfully
- ⚠️ Runtime testing on macOS still needed to verify the app doesn't crash

### Files Modified

- `libs/elodin-editor/src/lib.rs` (line ~555)

---

**Impact:** Critical fix preventing potential crash
**Risk:** Low - correct pattern for singleton access
**Testing Required:** Runtime verification on macOS

