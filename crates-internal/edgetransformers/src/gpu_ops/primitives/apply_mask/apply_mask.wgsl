struct Uniforms {
    batch_size: u32,
    num_heads: u32,
    query_len: u32,
    logical_key_len: u32, // The number of actual tokens (cache + new)
    key_stride: u32,      // The physical width of the buffer (e.g., max_len)
    is_causal: u32,
    position_offset: u32, // The starting position of the query tokens
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>; // The padding mask

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let query_pos = global_id.x; // Index along the query sequence (0 to query_len-1)
    let key_pos = global_id.y;   // Index along the key sequence (0 to key_stride-1)
    let head_batch_idx = global_id.z;
    
    // Bounds check against PHYSICAL buffer dimensions
    if (query_pos >= uniforms.query_len || key_pos >= uniforms.key_stride || head_batch_idx >= (uniforms.batch_size * uniforms.num_heads)) {
        return;
    }
    
    let batch_idx = head_batch_idx / uniforms.num_heads;
    
    // Index into scores using the PHYSICAL key_stride
    let score_idx = head_batch_idx * uniforms.query_len * uniforms.key_stride + query_pos * uniforms.key_stride + key_pos;
    
    var should_mask = false;
    
    // 1. Apply causal mask
    // This logic uses the LOGICAL length.
    let absolute_query_pos = uniforms.position_offset + query_pos;
    if (uniforms.is_causal != 0u && key_pos > absolute_query_pos) {
        should_mask = true;
    }
    
    // 2. Apply padding mask
    // This logic uses the PHYSICAL key_stride for indexing.
    let mask_idx = batch_idx * uniforms.key_stride + key_pos;
    if (mask[mask_idx] == 0.0) {
        should_mask = true;
    }
    
    if (should_mask) {
        scores[score_idx] = -1e9;
    }
}