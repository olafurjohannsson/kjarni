//! Layer Normalization

/// Uniform parameters for layer normalization.
struct NormUniforms {
    /// Number of rows to normalize (batch * seq_len).
    m: u32,
    /// Elements per row (hidden dimension).
    n: u32,
    /// Epsilon
    eps: f32,
    /// Padding for 16-byte alignment.
    _padding1: f32
};

@group(0) @binding(0) var<uniform> uniforms: NormUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>; // Scale (weight)
@group(0) @binding(3) var<storage, read> beta: array<f32>;  // Shift (bias)
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    // Launched over a 2D grid when one dimension would exceed 65535
    // workgroups; see gpu_ops::utils::grid_1d.
    let row_idx = global_id.y * grid.x * 256u + global_id.x;
    if (row_idx >= uniforms.m) {
        return;
    }

    let hidden_size = uniforms.n;
    let row_offset = row_idx * hidden_size;

    var mean = 0.0;
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        mean += input[row_offset + i];
    }
    mean /= f32(hidden_size);
    var variance = 0.0;
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        let val = input[row_offset + i] - mean;
        variance += val * val;
    }
    variance /= f32(hidden_size);

    let inv_std = 1.0 / sqrt(variance + uniforms.eps);

    for (var i = 0u; i < hidden_size; i = i + 1u) {
        let normalized = (input[row_offset + i] - mean) * inv_std;
        output[row_offset + i] = normalized * gamma[i] + beta[i];
    }
}

var<workgroup> wide_sum: array<f32, 64>;

/// Adds up every thread's value in `wide_sum`, which holds it afterwards in slot 0.
fn wide_reduce(tid: u32) {
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wide_sum[tid] += wide_sum[tid + s];
        }
        workgroupBarrier();
    }
}

/// The same normalization for a few rows: one workgroup per row, 64 threads across
/// its columns. `main` gives each row one thread, which is what many rows want, but
/// with a handful of rows only those threads run, walking the row one value at a
/// time.
@compute @workgroup_size(64)
fn wide(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let row = group_id.y * grid.x + group_id.x;
    let tid = local_id.x;
    if (row >= uniforms.m) { return; }

    let hidden_size = uniforms.n;
    let row_offset = row * hidden_size;

    var partial = 0.0;
    for (var i = tid; i < hidden_size; i += 64u) {
        partial += input[row_offset + i];
    }
    wide_sum[tid] = partial;
    workgroupBarrier();
    wide_reduce(tid);
    let mean = wide_sum[0] / f32(hidden_size);
    workgroupBarrier();

    partial = 0.0;
    for (var i = tid; i < hidden_size; i += 64u) {
        let val = input[row_offset + i] - mean;
        partial += val * val;
    }
    wide_sum[tid] = partial;
    workgroupBarrier();
    wide_reduce(tid);
    let variance = wide_sum[0] / f32(hidden_size);

    let inv_std = 1.0 / sqrt(variance + uniforms.eps);
    for (var i = tid; i < hidden_size; i += 64u) {
        let normalized = (input[row_offset + i] - mean) * inv_std;
        output[row_offset + i] = normalized * gamma[i] + beta[i];
    }
}
