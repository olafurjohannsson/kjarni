//! Rotary Position Embedding (RoPE) for dual tensors (Q and K).

/// Uniform parameters for RoPE operation.
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
@group(0) @binding(1) var<storage, read> q_in: array<f32>;
@group(0) @binding(2) var<storage, read> k_in: array<f32>;
@group(0) @binding(3) var<storage, read> cos_cache: array<f32>; // Precomputed cosines
@group(0) @binding(4) var<storage, read> sin_cache: array<f32>; // Precomputed sines
@group(0) @binding(5) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> k_out: array<f32>;

/// Applies RoPE to both Q and K tensors
@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim_idx = global_id.x;        // Dimension index [0, head_dim/2)
    let seq_idx = global_id.y;        // Token position [0, seq_len)
    let head_batch_idx = global_id.z; // Combined batch*heads index

    let half_dim = uniforms.head_dim / 2u;

    if (dim_idx >= half_dim || seq_idx >= uniforms.seq_len || head_batch_idx >= uniforms.batch_size * uniforms.num_heads) {
        return;
    }

    // Compute absolute position (for decode with cache, offset != 0)
    let pos = uniforms.position_offset + seq_idx;

    // Fetch precomputed sin/cos for this position and dimension
    let cos_val = cos_cache[pos * uniforms.cos_stride + dim_idx];
    let sin_val = sin_cache[pos * uniforms.sin_stride + dim_idx];
    // RoPE Q Tensor
    {
        let base_idx = head_batch_idx * uniforms.seq_len * uniforms.head_dim + seq_idx * uniforms.head_dim;

        let q0 = q_in[base_idx + dim_idx];
        let q1 = q_in[base_idx + dim_idx + half_dim];

        // Rotate: [q0, q1] -> [q0*cos - q1*sin, q1*cos + q0*sin]
        q_out[base_idx + dim_idx]            = q0 * cos_val - q1 * sin_val;
        q_out[base_idx + dim_idx + half_dim] = q1 * cos_val + q0 * sin_val;
    }

    // RoPE K Tensor
    {
        let base_idx = head_batch_idx * uniforms.seq_len * uniforms.head_dim + seq_idx * uniforms.head_dim;

        let k0 = k_in[base_idx + dim_idx];
        let k1 = k_in[base_idx + dim_idx + half_dim];

        // Rotate: [k0, k1] -> [k0*cos - k1*sin, k1*cos + k0*sin]
        k_out[base_idx + dim_idx]            = k0 * cos_val - k1 * sin_val;
        k_out[base_idx + dim_idx + half_dim] = k1 * cos_val + k0 * sin_val;
    }
}