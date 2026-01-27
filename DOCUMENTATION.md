# Kjarni Documentation Style Guide

**Author:** Ólafur Aron <olafurjohannss@gmail.com>  
**License:** MIT OR Apache-2.0

This guide follows the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/documentation.html). When in doubt, refer there.

---

## Crate Header

Place this at the top of `lib.rs`:

```rust
//! # Kjarni
//!
//! Brief description of what the crate does.
//!
//! ## Quick Start
//!
//! ```ignore
//! use kjarni::Something;
//!
//! let thing = Something::new();
//! ```
```

---

## Core Rules

1. **Start with a verb** — "Returns", "Computes", "Creates" (not "This function...")
2. **End with a period** — Every doc comment is a sentence
3. **Explain the why** — Code shows what; docs explain why
4. **Keep it short** — One line for simple items, expand only when needed

---

## Examples

**Simple function** — one line is enough:

```rust
/// Returns the number of elements.
pub fn len(&self) -> usize { ... }
```

**Function with errors:**

```rust
/// Loads a tensor from the model weights.
///
/// # Errors
///
/// Returns an error if the tensor doesn't exist or has an unsupported dtype.
pub fn load_tensor(&self, name: &str) -> Result<Tensor> { ... }
```

**Unsafe function** — always document safety:

```rust
/// Dereferences a raw pointer to read a value.
///
/// # Safety
///
/// `ptr` must be valid, aligned, and point to an initialized `T`.
pub unsafe fn read_ptr<T>(ptr: *const T) -> T { ... }
```

**Struct:**

```rust
/// A linear transformation layer.
///
/// Stores weights and an optional bias for computing `y = xW^T + b`.
pub struct LinearLayer {
    weights: Array2<f32>,
    bias: Option<Array1<f32>>,
}
```

**Enum:**

```rust
/// Supported tensor data types.
pub enum DType {
    /// 32-bit floating point.
    F32,
    /// 16-bit brain float.
    BF16,
}
```

---

## Checklist

Before committing, verify:

- [ ] All public items have doc comments
- [ ] Comments start with a verb and end with a period
- [ ] Functions returning `Result` have `# Errors`
- [ ] Functions that panic have `# Panics`
- [ ] Unsafe functions have `# Safety`

---

## Lints

Add to `Cargo.toml` or `lib.rs`:

```rust
#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
```

Verify with:

```bash
cargo doc --no-deps
cargo clippy -- -W missing_docs
```