struct AddBroadcastOffsetUniforms {
    output_size: u32,
    b_row_offset: u32,
    seq_len: u32,
    hidden_size: u32,
    b_stride_0: u32,
};

@group(0) @binding(0) var<uniform> uniforms: AddBroadcastOffsetUniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    // Launched over a 2D grid when one dimension would exceed 65535
    // workgroups; see gpu_ops::utils::grid_1d.
    let idx = global_id.y * grid.x * 256u + global_id.x;

    if (idx >= uniforms.output_size) {
        return;
    }

    let hid = idx % uniforms.hidden_size;
    let seq = (idx / uniforms.hidden_size) % uniforms.seq_len;
    let b_row = seq + uniforms.b_row_offset;

    //let b_idx = b_row * uniforms.hidden_size + hid;
    let b_idx = b_row * uniforms.b_stride_0 + hid;

    // Ensure the addition is present
    output[idx] = a[idx] + b[b_idx];
}