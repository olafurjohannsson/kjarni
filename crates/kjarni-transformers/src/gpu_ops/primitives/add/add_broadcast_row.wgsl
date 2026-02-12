struct Uniforms {
    m: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>; // The larger tensor [M, N]
@group(0) @binding(2) var<storage, read> b: array<f32>; // The row to broadcast [1, N]
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.m * uniforms.n) {
        return;
    }

    let col = idx % uniforms.n;
    
    output[idx] = a[idx] + b[col];
}