@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;

struct Uniforms {
    num_beams: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
};
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear_id = global_id.x;
    
    // Total number of elements in a single beam's cache for one layer
    let elements_per_beam = uniforms.num_heads * uniforms.seq_len * uniforms.head_dim;

    if (linear_id >= uniforms.num_beams * elements_per_beam) {
        return;
    }

    // --- Calculate Destination ---
    let dest_beam_idx = linear_id / elements_per_beam;
    let idx_in_beam = linear_id % elements_per_beam;
    let dest_idx = linear_id;

    // --- Calculate Source ---
    // 1. Find the parent beam for our destination
    let parent_beam_idx = indices[dest_beam_idx];

    // 2. The source index is the same offset within the parent beam's data
    let src_idx = parent_beam_idx * elements_per_beam + idx_in_beam;
    
    // --- Perform the Copy ---
    dst[dest_idx] = src[src_idx];
}