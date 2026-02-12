struct ScaleUniforms {
    size: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: ScaleUniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }
    data[idx] = data[idx] * uniforms.scale;
}