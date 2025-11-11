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
    // Each thread now handles one dimension in the FIRST HALF of the head_dim.
    let dim_idx = global_id.x;
    let seq_idx = global_id.y;
    let head_batch_idx = global_id.z;

    let half_dim = uniforms.head_dim / 2u;

    // Bounds check
    if (dim_idx >= half_dim || seq_idx >= uniforms.seq_len || head_batch_idx >= uniforms.batch_size * uniforms.num_heads) {
        return;
    }

    let pos = uniforms.position_offset + seq_idx;
    
    // Get cos and sin. The precomputation logic makes cos[i] == cos[i + half_dim].
    let cos = cos_cache[pos * uniforms.cos_stride + dim_idx];
    let sin = sin_cache[pos * uniforms.sin_stride + dim_idx];

    // --- Inlined logic for Q tensor ---
    {
        let base_idx = head_batch_idx * uniforms.seq_len * uniforms.head_dim + seq_idx * uniforms.head_dim;
        
        // Get the pair of values from the first and second halves of the vector
        let q0 = q_tensor[base_idx + dim_idx];
        let q1 = q_tensor[base_idx + dim_idx + half_dim];

        // Apply the rotation based on the `rotate_half` formula:
        // q_rot[i]              = q[i] * cos[i] - q[i + half_dim] * sin[i]
        // q_rot[i + half_dim]   = q[i + half_dim] * cos[i] + q[i] * sin[i]
        q_tensor[base_idx + dim_idx]            = q0 * cos - q1 * sin;
        q_tensor[base_idx + dim_idx + half_dim] = q1 * cos + q0 * sin;
    }

    // --- Inlined logic for K tensor ---
    {
        let base_idx = head_batch_idx * uniforms.seq_len * uniforms.head_dim + seq_idx * uniforms.head_dim;

        // Get the pair of values from the first and second halves of the vector
        let k0 = k_tensor[base_idx + dim_idx];
        let k1 = k_tensor[base_idx + dim_idx + half_dim];

        // Apply the same rotation
        k_tensor[base_idx + dim_idx]            = k0 * cos - k1 * sin;
        k_tensor[base_idx + dim_idx + half_dim] = k1 * cos + k0 * sin;
    }
}