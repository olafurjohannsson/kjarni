//! Attention mask

struct Uniforms {
    /// Batch size.
    batch_size: u32,
    /// Number of attention heads.
    num_heads: u32,
    /// Query sequence length.
    query_len: u32,
    /// Logical key length (actual tokens including cache).
    logical_key_len: u32,
    /// Physical buffer width (max_seq_len for KV cache).
    key_stride: u32,
    /// Whether to apply causal mask (1 = yes, 0 = no).
    is_causal: u32,
    /// Starting position of query tokens (for decode with cache).
    position_offset: u32,
    /// Padding to ensure struct is multiple of 16 bytes.
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>; // Attention scores [batch, heads, query_len, key_stride]
@group(0) @binding(2) var<storage, read> mask: array<f32>;         // Padding mask [batch, key_stride]

/// Applies padding and causal masks to attention scores
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let query_pos = global_id.x;      
    let key_pos = global_id.y;        
    let head_batch_idx = global_id.z; 

    if (query_pos >= uniforms.query_len || key_pos >= uniforms.key_stride || head_batch_idx >= (uniforms.batch_size * uniforms.num_heads)) {
        return;
    }

    let batch_idx = head_batch_idx / uniforms.num_heads;

    let score_idx = head_batch_idx * uniforms.query_len * uniforms.key_stride + query_pos * uniforms.key_stride + key_pos;

    var should_mask = false;

    // Fast path: Beyond logical sequence length (unused KV cache slots)
    if (key_pos >= uniforms.logical_key_len) {
        should_mask = true;
    } else {
        // Prevent attending to future tokens: key_pos must be <= query_pos
        let absolute_query_pos = uniforms.position_offset + query_pos;
        if (uniforms.is_causal != 0u && key_pos > absolute_query_pos) {
            should_mask = true;
        }

        let mask_idx = batch_idx * uniforms.key_stride + key_pos;
        if (mask[mask_idx] == 0.0) {
            should_mask = true;
        }
    }

    if (should_mask) {
        scores[score_idx] = -1e9;
    }
}