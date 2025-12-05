const TILE_DIM = 32u;
const TILE_DIM_PADDED = 33u; // Avoid bank conflicts

struct MatmulInfo {
    m: u32,
    k: u32,
    n: u32,
}

var<workgroup> a_tile: array<f32, 1024>; // 32 * 32
var<workgroup> b_tile: array<f32, 1056>; // 32 * 33 (Padded)

@group(0) @binding(0) var<uniform> info: MatmulInfo;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;
@group(0) @binding(2) var<storage, read> b_in: array<u32>; // Packed BF16
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>;

// --- FIXED BF16 UNPACKER ---
fn unpack_bf16_manual(packed: u32) -> vec2<f32> {
    // Shift left 16 to get low part into high F32 bits
    let v1 = bitcast<f32>(packed << 16u);
    // Mask to keep high part in high F32 bits
    let v2 = bitcast<f32>(packed & 0xFFFF0000u);
    return vec2<f32>(v1, v2);
}

fn get_b_value(index: u32) -> f32 {
    let vec_idx = index / 2u;
    let packed = b_in[vec_idx];
    let vals = unpack_bf16_manual(packed);
    
    // Select based on LSB
    if (index % 2u == 0u) {
        return vals.x;
    } else {
        return vals.y;
    }
}

@compute @workgroup_size(32, 32, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tx = local_id.x;
    let ty = local_id.y;
    
    let global_row = group_id.y * TILE_DIM + ty; // M
    let global_col = group_id.x * TILE_DIM + tx; // N

    var acc = 0.0;
    let num_tiles = (info.k + TILE_DIM - 1u) / TILE_DIM;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        
        // 1. Load A
        let a_col = t * TILE_DIM + tx;
        if (global_row < info.m && a_col < info.k) {
            a_tile[ty * TILE_DIM + tx] = a_in[global_row * info.k + a_col];
        } else {
            a_tile[ty * TILE_DIM + tx] = 0.0;
        }

        // 2. Load B (Physically [N, K]) -> Transpose to Shared [K, N]
        let b_phys_row = group_id.x * TILE_DIM + ty; 
        let b_phys_col = t * TILE_DIM + tx;          
        
        if (b_phys_row < info.n && b_phys_col < info.k) {
            let val = get_b_value(b_phys_row * info.k + b_phys_col);
            b_tile[tx * TILE_DIM_PADDED + ty] = val; 
        } else {
            b_tile[tx * TILE_DIM_PADDED + ty] = 0.0;
        }

        workgroupBarrier();

        // 3. Compute
        for (var i = 0u; i < TILE_DIM; i = i + 1u) {
            acc = acc + a_tile[ty * TILE_DIM + i] * b_tile[i * TILE_DIM_PADDED + tx];
        }
        
        workgroupBarrier();
    }
    
    if (global_row < info.m && global_col < info.n) {
        c_out[global_row * info.n + global_col] = acc;
    }
}