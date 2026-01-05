struct ScaleUniforms {
    size: u32,
    scale: f32,
    _padding1: u32,
    _padding2: u32,
};

@group(0) @binding(0) var<uniform> uniforms: ScaleUniforms;
@group(0) @binding(1) var<storage, read> input_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>; // The new output buffer

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }
    // Read from input, write to output
    output_data[idx] = input_data[idx] * uniforms.scale;
}