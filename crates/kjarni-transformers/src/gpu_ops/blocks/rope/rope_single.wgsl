//! Rotary Position Embedding 

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
@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim_idx = global_id.x;       
    let seq_idx = global_id.y;        
    let head_batch_idx = global_id.z; 

    let half_dim = uniforms.head_dim / 2u;

    if (dim_idx >= half_dim || seq_idx >= uniforms.seq_len || head_batch_idx >= uniforms.batch_size * uniforms.num_heads) {
        return;
    }

    // Compute absolute position 
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