//! Second feedforward layer (FC2): bias, added in place after the matmul back
//! to the hidden dimension, which runs as a tiled GEMM in GpuLinearLayer.



struct FfnUniforms {
    /// Number of input rows (batch * seq_len).
    m: u32,
    /// Input dimension (intermediate_size).
    k: u32,
    /// Output dimension (hidden_size).
    n: u32,
    /// Padding for 16-byte alignment.
    _padding: u32,
};

@group(0) @binding(0) var<uniform> info: FfnUniforms;
@group(0) @binding(1) var<storage, read> fc2_weight: array<f32>;
@group(0) @binding(2) var<storage, read> fc2_bias: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    // Launched over a 2D grid when one dimension would exceed 65535
    // workgroups; see gpu_ops::utils::grid_1d.
    let idx = global_id.y * grid.x * 512u + global_id.x;

    let total_outputs = info.m * info.n;
    if (idx >= total_outputs) {
        return;
    }

    // `output` already holds input * fc2_weight^T, from the tiled GEMM in
    // GpuLinearLayer. This pass adds the bias.
    let col = idx % info.n;
    output[idx] = output[idx] + fc2_bias[col];
}