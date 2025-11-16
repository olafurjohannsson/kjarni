struct RoPEUniforms {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    position_offset: u32,
    cos_stride: u32,
    sin_stride: u32,
    _padding: u32, // Padding to ensure struct is multiple of 16 bytes
};

// --- BINDINGS ---
// FIX: Add the missing uniform binding
@group(0) @binding(0) var<uniform> uniforms: RoPEUniforms;

@group(0) @binding(1) var<storage, read> q_in: array<f32>;
@group(0) @binding(2) var<storage, read> k_in: array<f32>;
@group(0) @binding(3) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(4) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(5) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> k_out: array<f32>;


@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread handles one dimension in the FIRST HALF of the head_dim.
    let dim_idx = global_id.x;
    let seq_idx = global_id.y;
    let head_batch_idx = global_id.z;

    let half_dim = uniforms.head_dim / 2u;

    // Bounds check
    if (dim_idx >= half_dim || seq_idx >= uniforms.seq_len || head_batch_idx >= uniforms.batch_size * uniforms.num_heads) {
        return;
    }

    let pos = uniforms.position_offset + seq_idx;

    // Get cos and sin.
    let cos_val = cos_cache[pos * uniforms.cos_stride + dim_idx];
    let sin_val = sin_cache[pos * uniforms.sin_stride + dim_idx];

    // --- Logic for Q tensor (Out-of-Place) ---
    {
        let base_idx = head_batch_idx * uniforms.seq_len * uniforms.head_dim + seq_idx * uniforms.head_dim;

        let q0 = q_in[base_idx + dim_idx];
        let q1 = q_in[base_idx + dim_idx + half_dim];

        q_out[base_idx + dim_idx]            = q0 * cos_val - q1 * sin_val;
        q_out[base_idx + dim_idx + half_dim] = q1 * cos_val + q0 * sin_val;
    }

    // --- Logic for K tensor (Out-of-Place) ---
    {
        let base_idx = head_batch_idx * uniforms.seq_len * uniforms.head_dim + seq_idx * uniforms.head_dim;

        let k0 = k_in[base_idx + dim_idx];
        let k1 = k_in[base_idx + dim_idx + half_dim];

        k_out[base_idx + dim_idx]            = k0 * cos_val - k1 * sin_val;
        k_out[base_idx + dim_idx + half_dim] = k1 * cos_val + k0 * sin_val;
    }
}