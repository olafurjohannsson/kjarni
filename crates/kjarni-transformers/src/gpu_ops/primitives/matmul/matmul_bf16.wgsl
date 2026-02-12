//! Tiled matrix multiplication for BF16 weights

const TILE_DIM = 32u;
const TILE_DIM_PADDED = 33u; // Avoid bank conflicts in shared memory

/// Uniform parameters for matrix dimensions.
struct MatmulInfo {
    /// Number of rows in A (and C).
    m: u32,
    /// Inner dimension (A cols = B cols due to implicit transpose).
    k: u32,
    /// Number of rows in B (columns in C).
    n: u32,
}

var<workgroup> a_tile: array<f32, 1024>; // 32 * 32 = 4KB
var<workgroup> b_tile: array<f32, 1056>; // 32 * 33 = 4.2KB (padded for bank conflicts)

@group(0) @binding(0) var<uniform> info: MatmulInfo;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;      // [M, K]
@group(0) @binding(2) var<storage, read> b_in: array<u32>;      // [N, K/2] packed BF16
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>; // [M, N]

/// Manually unpacks two BF16 values from a packed u32.
fn unpack_bf16_manual(packed: u32) -> vec2<f32> {
    // Low 16 bits -> high bits of F32
    let v1 = bitcast<f32>(packed << 16u);
    // High 16 bits already in position
    let v2 = bitcast<f32>(packed & 0xFFFF0000u);
    return vec2<f32>(v1, v2);
}

/// Fetches a single BF16 value from the packed weight buffer.
fn get_b_value(index: u32) -> f32 {
    let vec_idx = index / 2u;
    let packed = b_in[vec_idx];
    let vals = unpack_bf16_manual(packed);

    // Select based on odd/even index
    if (index % 2u == 0u) {
        return vals.x;
    } else {
        return vals.y;
    }
}

/// matmul kerne
@compute @workgroup_size(32, 32, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tx = local_id.x;
    let ty = local_id.y;

    let global_row = group_id.y * TILE_DIM + ty; // M dimension
    let global_col = group_id.x * TILE_DIM + tx; // N dimension

    var acc = 0.0;
    let num_tiles = (info.k + TILE_DIM - 1u) / TILE_DIM;

    // Iterate over K dimension in tiles
    for (var t = 0u; t < num_tiles; t = t + 1u) {


        // Load A tile [TILE_DIM, TILE_DIM] from [M, K]
        let a_col = t * TILE_DIM + tx;
        if (global_row < info.m && a_col < info.k) {
            a_tile[ty * TILE_DIM + tx] = a_in[global_row * info.k + a_col];
        } else {
            a_tile[ty * TILE_DIM + tx] = 0.0;
        }

        let b_phys_row = group_id.x * TILE_DIM + ty; // N dimension
        let b_phys_col = t * TILE_DIM + tx;          // K dimension

        if (b_phys_row < info.n && b_phys_col < info.k) {
            let val = get_b_value(b_phys_row * info.k + b_phys_col);
            b_tile[tx * TILE_DIM_PADDED + ty] = val;
        } else {
            b_tile[tx * TILE_DIM_PADDED + ty] = 0.0;
        }

        workgroupBarrier();

        for (var i = 0u; i < TILE_DIM; i = i + 1u) {
            acc = acc + a_tile[ty * TILE_DIM + i] * b_tile[i * TILE_DIM_PADDED + tx];
        }

        workgroupBarrier();
    }

    if (global_row < info.m && global_col < info.n) {
        c_out[global_row * info.n + global_col] = acc;
    }
}
