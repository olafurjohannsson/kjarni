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
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

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