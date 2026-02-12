@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct Uniforms {
    // Strides for the 4D source tensor
    src_stride_b: u32,
    src_stride_h: u32,
    src_stride_s: u32,
    src_stride_d: u32,

    // Strides for the 4D destination tensor
    dst_stride_b: u32,
    dst_stride_h: u32,
    dst_stride_s: u32,
    dst_stride_d: u32,

    // Slice offsets for each dimension
    offset_b: u32,
    offset_h: u32,
    offset_s: u32,
    offset_d: u32,

    // Shape of the destination tensor for bounds checking
    dst_shape_b: u32,
    dst_shape_h: u32,
    dst_shape_s: u32,
    dst_shape_d: u32,
};
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // These IDs map directly to the *destination* tensor's indices
    let d_idx = global_id.x;
    let s_idx = global_id.y;
    let bh_idx = global_id.z; // Flattened batch and head

    let b_idx = bh_idx / uniforms.dst_shape_h;
    let h_idx = bh_idx % uniforms.dst_shape_h;

    // Bounds check against the destination shape
    if (b_idx >= uniforms.dst_shape_b || h_idx >= uniforms.dst_shape_h || s_idx >= uniforms.dst_shape_s || d_idx >= uniforms.dst_shape_d) {
        return;
    }

    // Calculate the linear index for the destination buffer
    let dst_idx = b_idx * uniforms.dst_stride_b + 
                  h_idx * uniforms.dst_stride_h + 
                  s_idx * uniforms.dst_stride_s + 
                  d_idx * uniforms.dst_stride_d;

    // Calculate the corresponding source index by adding the offset to each dimension's index
    let src_b_idx = b_idx + uniforms.offset_b;
    let src_h_idx = h_idx + uniforms.offset_h;
    let src_s_idx = s_idx + uniforms.offset_s;
    let src_d_idx = d_idx + uniforms.offset_d;

    let src_idx = src_b_idx * uniforms.src_stride_b +
                  src_h_idx * uniforms.src_stride_h +
                  src_s_idx * uniforms.src_stride_s +
                  src_d_idx * uniforms.src_stride_d;

    dst[dst_idx] = src[src_idx];
}