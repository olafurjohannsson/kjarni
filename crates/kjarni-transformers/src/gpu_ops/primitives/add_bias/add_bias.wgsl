struct Uniforms {
    size: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) { return; }

    let bias_dim = arrayLength(&bias);
    if (bias_dim == 0u) { return; } // Safety check

    output[idx] = input[idx] + bias[idx % bias_dim];
}