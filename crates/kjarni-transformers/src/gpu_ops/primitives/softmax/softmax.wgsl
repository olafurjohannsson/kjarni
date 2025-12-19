struct SoftmaxUniforms {
    rows: u32,
    cols: u32,        // Physical width of the row (padded size)
    valid_cols: u32,  // Logical number of elements to process
    scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: SoftmaxUniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    if (row_idx >= uniforms.rows) { return; }

    let row_offset = row_idx * uniforms.cols;
    
    // 1. Apply scale first to the valid columns
    for (var i = 0u; i < uniforms.valid_cols; i = i + 1u) {
        data[row_offset + i] = data[row_offset + i] * uniforms.scale;
    }
    
    // 2. Find max for numerical stability across valid columns
    var max_val = -3.4e+38; // f32.min approximation
    for (var i = 0u; i < uniforms.valid_cols; i = i + 1u) {
        max_val = max(max_val, data[row_offset + i]);
    }

    // 3. Exp and sum across valid columns
    var exp_sum = 0.0;
    for (var i = 0u; i < uniforms.valid_cols; i = i + 1u) {
        let val = exp(data[row_offset + i] - max_val);
        data[row_offset + i] = val; // Store intermediate exp value
        exp_sum += val;
    }

    // 4. Normalize the valid columns
    for (var i = 0u; i < uniforms.valid_cols; i = i + 1u) {
        data[row_offset + i] = data[row_offset + i] / exp_sum;
    }
    
    // 5. Zero out the padding area
    for (var i = uniforms.valid_cols; i < uniforms.cols; i = i + 1u) {
        data[row_offset + i] = 0.0;
    }
}