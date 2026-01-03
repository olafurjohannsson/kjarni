//! Rotary Position Embedding (RoPE) for single tensor.
//!
//! Applies rotary position embeddings to a single tensor (either Q or K).
//! Identical to `rope.wgsl` but operates on one tensor instead of two.
//!
//! # Algorithm
//!
//! For each pair of dimensions (i, i + head_dim/2):
//! ```text
//! x[i]              = x_old[i] * cos(θ) - x_old[i+d/2] * sin(θ)
//! x[i + head_dim/2] = x_old[i+d/2] * cos(θ) + x_old[i] * sin(θ)
//! ```
//! Where θ = position / (10000^(2i/d))
//!
//! # Performance
//!
//! - **Current**: ~0.025ms for [1, 32, 128, 128] on RTX 3090
//! - **Memory bound**: Simple read-modify-write pattern
//! - Half the work of dual-tensor RoPE
//!
//! # Use Cases
//!
//! - When Q and K are processed in separate passes
//! - When only one of Q/K needs RoPE (rare)
//! - For cross-attention where only Q gets position encoding
//!
//! # TODO / Improvements
//!
//! - Increase workgroup size from 16 to 64 or 128 for better occupancy
//! - Add vec2 loads/stores for 2x bandwidth
//! - Consider adding in-place variant to save memory bandwidth
//!
//! # Limitations
//!
//! - Assumes head_dim is even (required for pair rotation)
//! - Out-of-place only (requires separate input/output buffers)
//! - No support for RoPE scaling variants (YaRN, NTK, etc.)
//!
//! # See Also
//!
//! - [`rope.wgsl`] — Dual tensor variant for applying RoPE to Q and K together
//! - [`crate::gpu_ops::blocks::rope::GpuRoPE`] — Rust dispatch

/// Uniform parameters for single-tensor RoPE operation.
struct RoPEUniforms {
    /// Batch size.
    batch_size: u32,
    /// Number of attention heads.
    num_heads: u32,
    /// Sequence length (number of tokens).
    seq_len: u32,
    /// Head dimension (must be even).
    head_dim: u32,
    /// Starting position in the sequence (for decode with KV cache).
    position_offset: u32,
    /// Stride in cos cache (typically head_dim/2).
    cos_stride: u32,
    /// Stride in sin cache (typically head_dim/2).
    sin_stride: u32,
    /// Padding to ensure struct is multiple of 16 bytes.
    _padding: u32,
};

@group(0) @binding(0) var<uniform> uniforms: RoPEUniforms;
@group(0) @binding(1) var<storage, read> tensor_in: array<f32>;
@group(0) @binding(2) var<storage, read> cos_cache: array<f32>; // Precomputed cosines
@group(0) @binding(3) var<storage, read> sin_cache: array<f32>; // Precomputed sines
@group(0) @binding(4) var<storage, read_write> tensor_out: array<f32>;

/// Applies RoPE to a single tensor.
///
/// Each thread handles one dimension in the first half of head_dim.
/// The paired dimension (dim + head_dim/2) is rotated together.
///
/// # Workgroup Size
/// 16x1x1 - Small workgroup for simplicity. Could be increased to 64 or 128.
@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim_idx = global_id.x;        // Dimension index [0, head_dim/2)
    let seq_idx = global_id.y;        // Token position [0, seq_len)
    let head_batch_idx = global_id.z; // Combined batch*heads index

    let half_dim = uniforms.head_dim / 2u;

    // Bounds check
    if (dim_idx >= half_dim || seq_idx >= uniforms.seq_len || head_batch_idx >= uniforms.batch_size * uniforms.num_heads) {
        return;
    }

    // Compute absolute position (for decode with cache, offset != 0)
    let pos = uniforms.position_offset + seq_idx;

    // Fetch precomputed sin/cos for this position and dimension
    let cos_val = cos_cache[pos * uniforms.cos_stride + dim_idx];
    let sin_val = sin_cache[pos * uniforms.sin_stride + dim_idx];

    let base_idx = head_batch_idx * uniforms.seq_len * uniforms.head_dim + seq_idx * uniforms.head_dim;

    // Load the dimension pair
    let v0 = tensor_in[base_idx + dim_idx];
    let v1 = tensor_in[base_idx + dim_idx + half_dim];

    // Rotate: [v0, v1] → [v0*cos - v1*sin, v1*cos + v0*sin]
    tensor_out[base_idx + dim_idx]            = v0 * cos_val - v1 * sin_val;
    tensor_out[base_idx + dim_idx + half_dim] = v1 * cos_val + v0 * sin_val;
}