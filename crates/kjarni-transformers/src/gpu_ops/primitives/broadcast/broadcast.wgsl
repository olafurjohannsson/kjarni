struct Uniforms {
    src_stride: u32,
    dst_stride: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_idx = global_id.x;
    if (dst_idx >= arrayLength(&dst)) {
        return;
    }

    let outer_idx = dst_idx / uniforms.dst_stride;
    let inner_idx = dst_idx % uniforms.src_stride;

    let src_idx = outer_idx * uniforms.src_stride + inner_idx;

    dst[dst_idx] = src[src_idx];
}