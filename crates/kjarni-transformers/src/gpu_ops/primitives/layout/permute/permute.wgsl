struct PermuteUniforms {
    // Shape of the OUTPUT tensor
    out_shape: vec4<u32>,
    // Strides of the OUTPUT tensor
    out_strides: vec4<u32>,
    // The permutation mapping: perm[dst_axis] = src_axis
    // e.g., for [0, 2, 1, 3], perm is vec4<u32>(0, 2, 1, 3)
    perm: vec4<u32>,
};

@group(0) @binding(0) var<uniform> uniforms: PermuteUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Helper to get the total number of elements from shape
fn num_elements(shape: vec4<u32>) -> u32 {
    return shape[0] * shape[1] * shape[2] * shape[3];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    
    if (out_idx >= num_elements(uniforms.out_shape)) {
        return;
    }

    // Calculate the 4D coordinate
    var out_coords: vec4<u32>;
    var temp_idx = out_idx;
    out_coords[0] = temp_idx / uniforms.out_strides[0];
    temp_idx = temp_idx % uniforms.out_strides[0];
    out_coords[1] = temp_idx / uniforms.out_strides[1];
    temp_idx = temp_idx % uniforms.out_strides[1];
    out_coords[2] = temp_idx / uniforms.out_strides[2];
    out_coords[3] = temp_idx % uniforms.out_strides[2]; // Equivalent to temp_idx / strides[3] where strides[3] is 1

    var in_coords: vec4<u32>;
    in_coords[uniforms.perm[0]] = out_coords[0];
    in_coords[uniforms.perm[1]] = out_coords[1];
    in_coords[uniforms.perm[2]] = out_coords[2];
    in_coords[uniforms.perm[3]] = out_coords[3];
    
    // assume the tensor is contiguous
    let in_idx = in_coords[0] * (uniforms.out_shape[uniforms.perm[1]] * uniforms.out_shape[uniforms.perm[2]] * uniforms.out_shape[uniforms.perm[3]]) +
                 in_coords[1] * (uniforms.out_shape[uniforms.perm[2]] * uniforms.out_shape[uniforms.perm[3]]) +
                 in_coords[2] * (uniforms.out_shape[uniforms.perm[3]]) +
                 in_coords[3];

    output[out_idx] = input[in_idx];
}