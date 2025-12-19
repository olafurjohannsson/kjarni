struct UpdateCacheUniforms {
    b: u32,
    s_new: u32,
    h: u32,
    d: u32,
    s_total: u32,
    offset: u32,
    _padding0: u32,
    _padding1: u32,
};

@group(0) @binding(0) var<uniform> uniforms: UpdateCacheUniforms;
@group(0) @binding(1) var<storage, read> k_new_in: array<f32>;
@group(0) @binding(2) var<storage, read> v_new_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> k_cache_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_cache_out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let d_idx = global_id.x;
    let s_idx_new = global_id.y;
    let bh_idx = global_id.z;

    let b_idx = bh_idx / uniforms.h;
    let h_idx = bh_idx % uniforms.h;

    if (b_idx >= uniforms.b || h_idx >= uniforms.h || s_idx_new >= uniforms.s_new || d_idx >= uniforms.d) {
        return;
    }

    let hidden_size = uniforms.h * uniforms.d;
    let src_idx = b_idx * (uniforms.s_new * hidden_size) + // Offset for batch
                    s_idx_new * hidden_size +               // Offset for sequence
                    h_idx * uniforms.d + d_idx;             // Offset within the hidden dimension
    let k_val = k_new_in[src_idx];
    let v_val = v_new_in[src_idx];

    let s_out = uniforms.offset + s_idx_new;

    // FIXED: K cache layout is now [B, H, S_total, D] to match V cache
    let k_out_idx = b_idx * (uniforms.h * uniforms.s_total * uniforms.d) +
                    h_idx * (uniforms.s_total * uniforms.d) +
                    s_out * uniforms.d +
                    d_idx;
    k_cache_out[k_out_idx] = k_val;

    // V cache layout: [B, H, S_total, D] (already correct)
    let v_out_idx = b_idx * (uniforms.h * uniforms.s_total * uniforms.d) +
                    h_idx * (uniforms.s_total * uniforms.d) +
                    s_out * uniforms.d +
                    d_idx;
    v_cache_out[v_out_idx] = v_val;
}