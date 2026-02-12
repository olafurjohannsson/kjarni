@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Uniforms {
    // Strides for the OUTPUT tensor
    out_stride_0: u32,
    out_stride_1: u32,
    out_stride_2: u32,
    out_stride_3: u32,

    // Strides for input A
    a_stride_0: u32,
    a_stride_1: u32,
    a_stride_2: u32,
    a_stride_3: u32,

    // Strides for input B
    b_stride_0: u32,
    b_stride_1: u32,
    b_stride_2: u32,
    b_stride_3: u32,

    // Shape of input A
    a_shape_0: u32,
    a_shape_1: u32,
    a_shape_2: u32,
    a_shape_3: u32,

    // Shape of the OUTPUT tensor (for bounds checks)
    out_shape_0: u32,
    out_shape_1: u32,
    out_shape_2: u32,
    out_shape_3: u32,

    // The axis along which to concatenate
    concat_axis: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // global_id represents the index in the OUTPUT tensor.
    // For simplicity, we map this to a 4D tensor.
    let i3 = global_id.x;
    let i2 = global_id.y;
    let i1 = global_id.z % uniforms.out_shape_1;
    let i0 = global_id.z / uniforms.out_shape_1;

    if (i0 >= uniforms.out_shape_0 || i1 >= uniforms.out_shape_1 || i2 >= uniforms.out_shape_2 || i3 >= uniforms.out_shape_3) {
        return;
    }

    let out_idx = i0 * uniforms.out_stride_0 +
                  i1 * uniforms.out_stride_1 +
                  i2 * uniforms.out_stride_2 +
                  i3 * uniforms.out_stride_3;

    var src_idx: u32;

    var axis_idx: u32;
    if (uniforms.concat_axis == 0u) { axis_idx = i0; }
    else if (uniforms.concat_axis == 1u) { axis_idx = i1; }
    else if (uniforms.concat_axis == 2u) { axis_idx = i2; }
    else { axis_idx = i3; }

    if (axis_idx < uniforms.a_shape_2) { // Using a_shape_2 as placeholder for concat axis size
        src_idx = i0 * uniforms.a_stride_0 +
                  i1 * uniforms.a_stride_1 +
                  i2 * uniforms.a_stride_2 +
                  i3 * uniforms.a_stride_3;
        output[out_idx] = a[src_idx];
    } else {
        var b_i0 = i0;
        var b_i1 = i1;
        var b_i2 = i2;
        var b_i3 = i3;

        if (uniforms.concat_axis == 0u) { b_i0 = i0 - uniforms.a_shape_0; }
        else if (uniforms.concat_axis == 1u) { b_i1 = i1 - uniforms.a_shape_1; }
        else if (uniforms.concat_axis == 2u) { b_i2 = i2 - uniforms.a_shape_2; }
        else { b_i3 = i3 - uniforms.a_shape_3; }

        src_idx = b_i0 * uniforms.b_stride_0 +
                  b_i1 * uniforms.b_stride_1 +
                  b_i2 * uniforms.b_stride_2 +
                  b_i3 * uniforms.b_stride_3;
        output[out_idx] = b[src_idx];
    }
}