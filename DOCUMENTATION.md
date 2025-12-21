# Kjarni Documentation Style Guide

This document defines the documentation standards for the Kjarni project.
All code—whether written by humans or AI—must follow these conventions.

## Table of Contents

1. [Philosophy](#philosophy)
2. [Module Documentation](#module-documentation)
3. [Function Documentation](#function-documentation)
4. [Struct Documentation](#struct-documentation)
5. [Enum Documentation](#enum-documentation)
6. [Error Documentation](#error-documentation)
7. [Examples](#examples)
8. [Safety Documentation](#safety-documentation)
9. [Performance Notes](#performance-notes)
10. [Links and References](#links-and-references)
11. [Forbidden Patterns](#forbidden-patterns)

---

## Philosophy

1. **Clarity over brevity** — A longer, clear explanation beats a short, cryptic one.
2. **Future-you is the audience** — Write for someone who hasn't seen this code in 6 months.
3. **Explain the "why"** — The code shows "what"; docs explain "why".
4. **Be precise** — Use exact terminology. "Returns" not "gives back".
5. **Be consistent** — Same structure, same tone, everywhere.

---

## Module Documentation

Every module (`mod.rs` or `filename.rs`) starts with a module-level doc comment.

### Structure
```rust
//! Brief one-line description ending with a period.
//!
//! Longer description explaining the module's purpose, how it fits into
//! the larger system, and any important design decisions.
//!
//! # Overview
//!
//! Explain the main components and their relationships.
//!
//! # Example
//!
//! ```rust
//! use kjarni::module_name::MainType;
//!
//! let instance = MainType::new();
//! ```
//!
//! # Architecture
//!
//! Optional section for complex modules explaining internal design.
//!
//! # See Also
//!
//! - [`RelatedModule`] — Brief description of relationship
//! - [`OtherType`] — Why you might also need this
```

### Example
```rust
//! CPU-based linear layer with multi-dtype support.
//!
//! This module provides `LinearLayer`, the core building block for neural network
//! weight matrices. It supports F32, BF16, and quantized (Q4_K, Q8_0) storage
//! with automatic dispatch to optimized SIMD kernels.
//!
//! # Overview
//!
//! The [`LinearLayer`] struct wraps weight data in various formats and provides
//! a unified `matmul` interface that dispatches to the optimal kernel based on
//! the underlying data type.
//!
//! # Example
//!
//! ```rust
//! use kjarni_transformers::linear_layer::LinearLayer;
//! use kjarni_transformers::weights::ModelWeights;
//!
//! let weights = ModelWeights::new(Path::new("./model"))?;
//! let layer = LinearLayer::from_weights(&weights, "model.layer.weight", None, None, None)?;
//! let output = layer.matmul(&input.view());
//! ```
//!
//! # Performance
//!
//! - F32: Uses `faer` library or custom AVX2/FMA kernels (~50 GFLOPS on modern CPUs)
//! - BF16: Mixed-precision with F32 accumulation (~40 GFLOPS)
//! - Q4_K: 4-bit quantized with on-the-fly dequantization (~30 GFLOPS)
//!
//! # See Also
//!
//! - [`ModelWeights`] — Loading weights from safetensors/GGUF files
//! - [`DType`] — Supported data types
```

---

## Function Documentation

### Structure
```rust
/// Brief one-line description ending with a period.
///
/// Longer description explaining what the function does, when to use it,
/// and any important behavior. Use present tense ("Returns", "Computes").
///
/// # Arguments
///
/// * `param_name` - Description starting with capital, ending with period.
/// * `another_param` - Keep descriptions concise but complete.
///
/// # Returns
///
/// Description of the return value. For `Result`, describe the `Ok` case here.
///
/// # Errors
///
/// * [`ErrorType::Variant`] - When this error occurs.
/// * [`anyhow::Error`] - General conditions that cause failure.
///
/// # Panics
///
/// Conditions that cause a panic. Omit section if function never panics.
///
/// # Example
///
/// ```rust
/// let result = function_name(arg1, arg2)?;
/// assert_eq!(result, expected);
/// ```
///
/// # Performance
///
/// Optional. Include for hot-path functions with specific characteristics.
///
/// # See Also
///
/// * [`related_function`] - When you might use this instead.
```

### Example
```rust
/// Computes matrix multiplication with automatic kernel dispatch.
///
/// Performs `C = A @ W^T` where `A` is the input activation and `W` is the
/// weight matrix stored in this layer. The kernel is selected based on the
/// weight's data type (F32, BF16, or quantized).
///
/// # Arguments
///
/// * `input` - Input tensor of shape `[batch, in_features]`. Must be contiguous.
///
/// # Returns
///
/// Output tensor of shape `[batch, out_features]`.
///
/// # Panics
///
/// * If `input` is not contiguous in memory.
/// * If `input.shape()[1] != self.in_features()`.
///
/// # Example
///
/// ```rust
/// let layer = LinearLayer::new_f32(weights, Some(bias));
/// let input = Array2::<f32>::zeros((1, 2048));
/// let output = layer.matmul(&input.view());
/// assert_eq!(output.shape(), &[1, 4096]);
/// ```
///
/// # Performance
///
/// For single-token decode (`batch=1`), consider using [`project_logits`] for
/// the final vocabulary projection, which is optimized for this case.
#[inline]
pub fn matmul(&self, input: &ArrayView2<f32>) -> Array2<f32> {
    // ...
}
```

### Short Functions

For trivial getters/setters, a single line suffices:
```rust
/// Returns the number of output features.
pub fn out_features(&self) -> usize {
    self.out_features
}

/// Returns `true` if this layer has a bias term.
pub fn has_bias(&self) -> bool {
    self.bias.is_some()
}

/// Returns the data type of the weight matrix.
pub fn dtype(&self) -> DType {
    self.data.dtype()
}
```

---

## Struct Documentation

### Structure
```rust
/// Brief one-line description ending with a period.
///
/// Longer description explaining the struct's purpose, ownership semantics,
/// and typical usage patterns.
///
/// # Fields
///
/// Document public fields. Private fields are documented inline.
///
/// # Example
///
/// ```rust
/// let instance = StructName::new(args);
/// instance.method();
/// ```
///
/// # Thread Safety
///
/// Explain `Send`/`Sync` bounds if relevant.
///
/// # See Also
///
/// * [`RelatedStruct`] - Relationship explanation.
pub struct StructName {
    /// Description of public field.
    pub field: Type,
    
    // Private fields use regular comments
    private_field: Type,
}
```

### Example
```rust
/// A CPU-based linear transformation layer supporting multiple data types.
///
/// `LinearLayer` wraps a weight matrix and optional bias vector, providing
/// efficient matrix multiplication with automatic dispatch to SIMD kernels.
/// Weights can be stored in F32, BF16, or quantized formats (Q4_K, Q8_0).
///
/// # Example
///
/// ```rust
/// use kjarni_transformers::linear_layer::LinearLayer;
/// use ndarray::Array2;
///
/// // Create from raw arrays
/// let weights = Array2::<f32>::zeros((4096, 2048));
/// let layer = LinearLayer::new_f32(weights, None);
///
/// // Or load from model files
/// let layer = LinearLayer::from_weights(&model_weights, "layer.weight", None, None, None)?;
///
/// // Forward pass
/// let input = Array2::<f32>::zeros((1, 2048));
/// let output = layer.matmul(&input.view());
/// ```
///
/// # Thread Safety
///
/// `LinearLayer` is `Send + Sync` and can be safely shared across threads.
/// The weight data is immutable after construction.
///
/// # See Also
///
/// * [`LinearData`] — The underlying weight storage enum.
/// * [`ModelWeights`] — Loading weights from files.
pub struct LinearLayer {
    /// The weight matrix in one of several supported formats.
    pub data: LinearData,
    
    /// Optional bias vector, always stored as F32.
    pub bias: Option<Array1<f32>>,
    
    // Strategy for F32 matmul (Faer vs CustomSimd)
    f32_strategy: F32MatmulStrategy,
}
```

---

## Enum Documentation

### Structure
```rust
/// Brief one-line description ending with a period.
///
/// Longer description explaining when to use each variant and the enum's
/// role in the system.
///
/// # Variants
///
/// Variants are documented individually below.
///
/// # Example
///
/// ```rust
/// let value = EnumName::Variant;
/// match value {
///     EnumName::Variant => { /* ... */ }
/// }
/// ```
pub enum EnumName {
    /// Description of this variant.
    ///
    /// Additional details if needed.
    Variant,
    
    /// Description of variant with data.
    VariantWithData(Type),
}
```

### Example
```rust
/// Supported numerical data types for tensor storage.
///
/// Kjarni supports multiple precision levels to balance memory usage,
/// computation speed, and numerical accuracy. Lower precision formats
/// reduce memory bandwidth requirements at the cost of some precision.
///
/// # Example
///
/// ```rust
/// use kjarni_transformers::tensor::DType;
///
/// let dtype = DType::BF16;
/// assert_eq!(dtype.size_of(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit IEEE 754 floating point.
    ///
    /// Highest precision, largest memory footprint. Use for accuracy-critical
    /// computations or when loading F32 model weights.
    F32,
    
    /// 16-bit IEEE 754 floating point (half precision).
    ///
    /// Good precision for inference with 2x memory savings over F32.
    F16,
    
    /// 16-bit brain floating point.
    ///
    /// Same exponent range as F32 with reduced mantissa. Preferred over F16
    /// for neural network inference due to better dynamic range.
    BF16,
    
    /// 8-bit block-quantized format (32 elements per block).
    ///
    /// Each block stores 32 int8 values with a shared F16 scale factor.
    /// Provides ~4x compression with minimal quality loss.
    Q8_0,
    
    /// 4-bit block-quantized format with K-quants (256 elements per block).
    ///
    /// Each block stores 256 4-bit values with multiple scale factors.
    /// Provides ~8x compression, suitable for large models on limited memory.
    Q4_K,
}
```

---

## Error Documentation

### Structure for Error Types
```rust
/// Errors that can occur during model loading.
///
/// These errors indicate problems with model files, configuration,
/// or unsupported features.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// The model file was not found at the specified path.
    #[error("Model file not found: {path}")]
    NotFound {
        /// The path that was searched.
        path: PathBuf,
    },
    
    /// The model format is not supported.
    #[error("Unsupported model format: {format}")]
    UnsupportedFormat {
        /// The detected format.
        format: String,
    },
}
```

### Error Section in Functions
```rust
/// # Errors
///
/// Returns an error if:
///
/// * The tensor `name` does not exist in the model.
/// * The tensor has an unsupported data type.
/// * The tensor shape does not match expectations.
pub fn get_tensor(&self, name: &str) -> Result<CpuTensor> {
    // ...
}
```

---

## Safety Documentation

For `unsafe` code, safety documentation is **mandatory**.

### Structure
```rust
/// Brief description.
///
/// # Safety
///
/// This function is unsafe because:
///
/// * Reason 1 — what invariant must the caller maintain.
/// * Reason 2 — what could go wrong if violated.
///
/// The caller must ensure:
///
/// * Requirement 1.
/// * Requirement 2.
///
/// # Example
///
/// ```rust
/// // SAFETY: We verified that ptr is valid and aligned.
/// unsafe {
///     function_name(ptr);
/// }
/// ```
pub unsafe fn function_name(ptr: *const f32) {
    // ...
}
```

### Example
```rust
/// Computes a vector-matrix product using AVX2 and FMA instructions.
///
/// # Safety
///
/// This function is unsafe because:
///
/// * It dereferences raw pointers without bounds checking.
/// * It requires AVX2 and FMA CPU features to be present.
///
/// The caller must ensure:
///
/// * `a_ptr` points to at least `k` contiguous f32 values.
/// * `b_ptr` points to at least `out_len * k` contiguous u16 values (BF16).
/// * `out` has length `out_len`.
/// * The CPU supports AVX2 and FMA (check with `is_x86_feature_detected!`).
///
/// # Example
///
/// ```rust
/// #[cfg(target_arch = "x86_64")]
/// if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
///     // SAFETY: We checked CPU features and ensured pointer validity.
///     unsafe {
///         matmul_vec_bf16(out, a.as_ptr(), b.as_ptr() as *const u16, k);
///     }
/// }
/// ```
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matmul_vec_bf16(
    out: &mut [f32],
    a_ptr: *const f32,
    b_ptr: *const u16,
    k: usize,
) {
    // ...
}
```

---

## Performance Notes

Include for hot-path code.

### Structure
```rust
/// # Performance
///
/// Time complexity: O(n * m)
/// Space complexity: O(n)
///
/// Benchmarks on Intel i7-13700 (single thread):
///
/// | Input Size | Time     | Throughput |
/// |------------|----------|------------|
/// | 1K × 1K    | 0.5ms    | 4 GFLOPS   |
/// | 4K × 4K    | 8ms      | 8 GFLOPS   |
///
/// Optimizations:
///
/// * Uses 4x loop unrolling to hide FMA latency.
/// * Prefetches next cache line to reduce memory stalls.
/// * Parallelizes over output rows for batch > 1.
```

---

## Examples Section

### Guidelines

1. **Every public item gets an example** — No exceptions for non-trivial items.
2. **Examples must compile** — Use `cargo test --doc` to verify.
3. **Show common use cases** — Not edge cases.
4. **Keep examples minimal** — Show one concept at a time.
5. **Use meaningful values** — Not `foo`, `bar`, `x`, `y`.

### Multiple Examples
```rust
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// let layer = LinearLayer::new_f32(weights, None);
/// let output = layer.matmul(&input.view());
/// ```
///
/// With bias:
///
/// ```rust
/// let layer = LinearLayer::new_f32(weights, Some(bias));
/// let output = layer.matmul(&input.view());
/// // output already includes bias
/// ```
///
/// Loading from model files:
///
/// ```rust
/// let layer = LinearLayer::from_weights(&model, "layer.weight", None, None, None)?;
/// ```
```

### Hiding Boilerplate
```rust
/// # Example
///
/// ```rust
/// # use anyhow::Result;
/// # fn main() -> Result<()> {
/// use kjarni_transformers::linear_layer::LinearLayer;
///
/// let layer = LinearLayer::from_weights(&weights, "name", None, None, None)?;
/// # Ok(())
/// # }
/// ```
```

---

## Links and References

### Intra-doc Links
```rust
/// Returns a [`CpuTensor`] loaded from the model.
///
/// See [`ModelWeights::get_raw`] for the low-level byte access.
///
/// For GPU tensors, use [`GpuTensor::from_cpu`] instead.
```

### External Links
```rust
/// Implements the SwiGLU activation function.
///
/// SwiGLU was introduced in [GLU Variants Improve Transformer][paper].
///
/// [paper]: https://arxiv.org/abs/2002.05202
```

---

## Forbidden Patterns

### Never Do These
```rust
// ❌ Empty doc comment
///
pub fn foo() {}

// ❌ Obvious/redundant documentation
/// Returns the length.
pub fn len(&self) -> usize { self.len }

// ❌ Starting with "This function" or "This method"
/// This function computes the sum.
pub fn sum() {}

// ❌ Using "I" or "we"
/// We return the value here.
pub fn get() {}

// ❌ Passive voice
/// The value is returned.
pub fn get() {}

// ❌ Missing period at end
/// Returns the length
pub fn len() {}

// ❌ Using abbreviations without definition
/// Computes the FFN output.  // What is FFN?
pub fn ffn() {}

// ❌ Vague descriptions
/// Does stuff with the data.
pub fn process() {}
```

### Always Do These
```rust
// ✅ Clear, active voice, complete sentence
/// Returns the number of elements in the tensor.
pub fn len(&self) -> usize { self.len }

// ✅ Starts with verb in third person
/// Computes the matrix product of input and weights.
pub fn matmul() {}

// ✅ Defines abbreviations on first use
/// Computes the feed-forward network (FFN) output.
pub fn ffn() {}

// ✅ Specific about behavior
/// Loads the tensor, converting BF16 to F32 if necessary.
pub fn load() {}
```

---

## Crate-Level Documentation

The main `lib.rs` must have comprehensive crate documentation.
```rust
//! # Kjarni Transformers
//!
//! A high-performance transformer inference engine for CPU and GPU.
//!
//! ## Features
//!
//! - **Multi-format weight loading**: SafeTensors, GGUF
//! - **Mixed-precision inference**: F32, BF16, F16, Q4_K, Q8_0
//! - **Optimized kernels**: AVX2/FMA (x86), NEON (ARM), WebGPU
//! - **Model support**: Llama, BERT, GPT-2, BART, T5
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use kjarni_transformers::prelude::*;
//!
//! // Load a model
//! let model = LlamaModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//!
//! // Generate text
//! let output = model.generate("Hello, world!", GenerationConfig::default())?;
//! println!("{}", output);
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                     kjarni-cli                          │
//! ├─────────────────────────────────────────────────────────┤
//! │                     kjarni (public API)                 │
//! ├─────────────────────────────────────────────────────────┤
//! │  kjarni-models  │  kjarni-rag  │  kjarni-search         │
//! ├─────────────────────────────────────────────────────────┤
//! │                 kjarni-transformers                     │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Modules
//!
//! - [`linear_layer`] — Linear transformation with multi-dtype support
//! - [`weights`] — Model weight loading (SafeTensors, GGUF)
//! - [`tensor`] — Tensor types and data formats
//! - [`kernels`] — Optimized SIMD compute kernels
//!
//! ## Feature Flags
//!
//! - `cuda` — Enable CUDA backend (requires CUDA toolkit)
//! - `metal` — Enable Metal backend (macOS only)
//! - `wgpu` — Enable WebGPU backend (default)
//!
//! ## License
//!
//! MIT OR Apache-2.0
```

---

## Checklist for AI/Human Writers

Before submitting documentation, verify:

- [ ] Module has `//!` doc comment explaining purpose
- [ ] All public functions have `///` doc comments
- [ ] All public structs/enums have `///` doc comments
- [ ] All doc comments start with third-person verb
- [ ] All doc comments end with periods
- [ ] `# Arguments` section for functions with parameters
- [ ] `# Returns` section for non-`()` return types
- [ ] `# Errors` section for `Result` return types
- [ ] `# Panics` section if function can panic
- [ ] `# Safety` section for all `unsafe` functions
- [ ] `# Example` section with compilable code
- [ ] No forbidden patterns used
- [ ] Links use `[`backtick`]` syntax
- [ ] Abbreviations defined on first use

---

## Integration with Tools

### rustdoc
```bash
# Generate and open documentation
cargo doc --open --no-deps

# Check for broken links
cargo doc --no-deps 2>&1 | grep "warning"
```

### Doctests
```bash
# Run all documentation examples
cargo test --doc
```

### clippy
```bash
# Enable documentation lints
cargo clippy -- -W missing_docs -W rustdoc::broken_intra_doc_links
```

### CI Integration
```yaml
# .github/workflows/docs.yml
- name: Check documentation
  run: |
    cargo doc --no-deps
    cargo test --doc
    cargo clippy -- -D missing_docs
```
```

---

**Usage for AI assistants:**

When you give this to any AI for documentation, include:
```
Please document this code following the style guide in DOCUMENTATION.md.
Ensure all public items have proper documentation with Examples sections.