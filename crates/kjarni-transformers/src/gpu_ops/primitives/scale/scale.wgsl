struct ScaleUniforms {
    size: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: ScaleUniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    // Launched over a 2D grid when one dimension would exceed 65535
    // workgroups; see gpu_ops::utils::grid_1d.
    let idx = global_id.y * grid.x * 256u + global_id.x;
    if (idx >= uniforms.size) {
        return;
    }
    data[idx] = data[idx] * uniforms.scale;
}