struct Uniforms {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    is_causal: u32,  // 1 = causal, 0 = padding only
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let query_pos = global_id.x;
    let key_pos = global_id.y;
    let head_batch_idx = global_id.z;
    
    let seq_len = uniforms.seq_len;
    
    // Bounds check
    if (query_pos >= seq_len || key_pos >= seq_len) {
        return;
    }
    
    let batch_idx = head_batch_idx / uniforms.num_heads;
    
    // Calculate index into scores [B, H, S, S]
    let score_idx = head_batch_idx * seq_len * seq_len + query_pos * seq_len + key_pos;
    
    var should_mask = false;
    
    // Apply causal mask (mask future tokens)
    if (uniforms.is_causal != 0u && key_pos > query_pos) {
        should_mask = true;
    }
    
    // Apply padding mask (mask padding tokens)
    let mask_idx = batch_idx * seq_len + key_pos;
    if (mask[mask_idx] == 0.0) {
        should_mask = true;
    }
    
    // Set masked positions to large negative value
    if (should_mask) {
        scores[score_idx] = -1e9;
    }
}