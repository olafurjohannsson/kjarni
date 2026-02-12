struct ReorderUniforms {
    num_beams: u32,
    num_heads: u32,
    capacity: u32,
    head_dim: u32,
    current_seq_len: u32,
    // Explicit padding to match Rust's [u32; 3] and ensure 16-byte alignment for the struct
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(0) var<uniform> uniforms: ReorderUniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> src_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_cache: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let d_idx = global_id.x;
    let h_idx = global_id.y;
    let bs_idx = global_id.z;

    if (d_idx >= uniforms.head_dim || h_idx >= uniforms.num_heads) {
        return;
    }

    // Decompose z into beam and seq
    // If current_seq_len is 0 (due to layout bug), this would crash/produce garbage.
    let b_idx = bs_idx / uniforms.current_seq_len;
    let s_idx = bs_idx % uniforms.current_seq_len;

    if (b_idx >= uniforms.num_beams) {
        return;
    }

    let src_beam_idx = indices[b_idx];

    // Strides for [Beams, Heads, Capacity, Dim]
    let stride_beam = uniforms.num_heads * uniforms.capacity * uniforms.head_dim;
    let stride_head = uniforms.capacity * uniforms.head_dim;
    let stride_seq  = uniforms.head_dim;

    let src_flat_idx = (src_beam_idx * stride_beam) + 
                       (h_idx * stride_head) + 
                       (s_idx * stride_seq) + 
                       d_idx;

    let dst_flat_idx = (b_idx * stride_beam) + 
                       (h_idx * stride_head) + 
                       (s_idx * stride_seq) + 
                       d_idx;

    dst_cache[dst_flat_idx] = src_cache[src_flat_idx];
}