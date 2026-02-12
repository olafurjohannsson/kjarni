//! Softmax activation

/// Uniform parameters for softmax kernel.
struct SoftmaxUniforms {
    /// Number of rows to process (batch * num_heads * seq_len).
    rows: u32,
    /// Physical width of each row (including padding).
    cols: u32,
    /// Logical number of valid elements (unmasked positions).
    valid_cols: u32,
    /// Pre-attention scaling factor (typically 1/sqrt(head_dim)).
    scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: SoftmaxUniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
var<workgroup> s_max: array<f32, 256>;  // For max reduction
var<workgroup> s_sum: array<f32, 256>;  // For sum reduction

/// Softmax kernel
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row_idx = group_id.x;  // One workgroup per row
    let tid = local_id.x;      // Thread ID within workgroup (0-255)

    if (row_idx >= uniforms.rows) { return; }

    let row_offset = row_idx * uniforms.cols;
    var local_max = -3.402823e+38; // f32 min

    for (var i = tid; i < uniforms.valid_cols; i += 256u) {
        let idx = row_offset + i;
        let val = data[idx];
        local_max = max(local_max, val * uniforms.scale);
    }

    // Store partial max
    s_max[tid] = local_max;
    workgroupBarrier();

    // Binary tree reduction for max (7 iterations: 128..1)
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_max[tid] = max(s_max[tid], s_max[tid + s]);
        }
        workgroupBarrier();
    }

    let max_val = s_max[0];
    workgroupBarrier();

    var local_sum = 0.0;

    for (var i = tid; i < uniforms.valid_cols; i += 256u) {
        let idx = row_offset + i;
        let val = data[idx];
        let exp_val = exp((val * uniforms.scale) - max_val);
        
        data[idx] = exp_val;
        local_sum += exp_val;
    }

    s_sum[tid] = local_sum;
    workgroupBarrier();

    // reduction for sum
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        workgroupBarrier();
    }

    let sum_val = s_sum[0];
    workgroupBarrier();

    // Optimization: Compute inverse once. Multiplication is faster than division.
    let inv_sum = 1.0 / sum_val;

    for (var i = tid; i < uniforms.valid_cols; i += 256u) {
        let idx = row_offset + i;
        data[idx] = data[idx] * inv_sum;
    }

    for (var i = uniforms.valid_cols + tid; i < uniforms.cols; i += 256u) {
        data[row_offset + i] = 0.0;
    }
}