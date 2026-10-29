/// A uniform struct to pass metadata about the tensor size.
struct AddUniforms {
    size: u32,
};

/// @group(0) @binding(0): Uniforms containing metadata.
@group(0) @binding(0) var<uniform> uniforms: AddUniforms;

/// @group(0) @binding(1): Read-only input tensor 'a'.
@group(0) @binding(1) var<storage, read> a: array<f32>;

/// @group(0) @binding(2): Read-only input tensor 'b'.
@group(0) @binding(2) var<storage, read> b: array<f32>;

/// @group(0) @binding(3): Writable output tensor.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;


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

    output[idx] = a[idx] + b[idx];
}