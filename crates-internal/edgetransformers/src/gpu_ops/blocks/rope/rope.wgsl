struct RoPEUniforms {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    position_offset: u32,
    cos_stride: u32,
    sin_stride: u32,
};

@group(0) @binding(0) var<uniform> uniforms: RoPEUniforms;
@group(0) @binding(1) var<storage, read_write> q_tensor: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_tensor: array<f32>;
@group(0) @binding(3) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(4) var<storage, read> sin_cache: array<f32>;

@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim_pair_idx = global_id.x; // Index for the dimension pair (0 to head_dim/2 - 1)
    let seq_idx = global_id.y;      // Index in the sequence (0 to seq_len - 1)
    let head_batch_idx = global_id.z; // Flattened (batch * num_heads) index

    // Bounds check
    if (dim_pair_idx >= uniforms.head_dim / 2u || seq_idx >= uniforms.seq_len || head_batch_idx >= uniforms.batch_size * uniforms.num_heads) {
        return;
    }

    let pos = uniforms.position_offset + seq_idx;
    let dim1 = dim_pair_idx * 2u;
    let dim2 = dim1 + 1u;

    // Get cos and sin for this position from the precomputed cache
    let cos = cos_cache[pos * uniforms.cos_stride + dim1];
    let sin = sin_cache[pos * uniforms.sin_stride + dim1];

    // --- ✅ FIX: Inlined logic for Q tensor ---
    {
        let base_idx = head_batch_idx * uniforms.seq_len * uniforms.head_dim + seq_idx * uniforms.head_dim;
        let x0 = q_tensor[base_idx + dim1];
        let x1 = q_tensor[base_idx + dim2];
        q_tensor[base_idx + dim1] = x0 * cos - x1 * sin;
        q_tensor[base_idx + dim2] = x0 * sin + x1 * cos;
    }

    // --- ✅ FIX: Inlined logic for K tensor ---
    {
        let base_idx = head_batch_idx * uniforms.seq_len * uniforms.head_dim + seq_idx * uniforms.head_dim;
        let x0 = k_tensor[base_idx + dim1];
        let x1 = k_tensor[base_idx + dim2];
        k_tensor[base_idx + dim1] = x0 * cos - x1 * sin;
        k_tensor[base_idx + dim2] = x0 * sin + x1 * cos;
    }
}