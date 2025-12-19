// matmul_bf16_fast.wgsl - A @ B^T with register blocking
// A: [M, K] F32, B: [N, K] packed BF16, C: [M, N] F32

const TILE_M = 128u;
const TILE_N = 128u;
const TILE_K = 32u;
const THREAD_M = 8u;  // Each thread computes 8x8 output
const THREAD_N = 8u;
const THREADS_X = 16u; // TILE_N / THREAD_N
const THREADS_Y = 16u; // TILE_M / THREAD_M

struct MatmulInfo {
    m: u32,
    k: u32,
    n: u32,
}

var<workgroup> a_shared: array<f32, TILE_M * TILE_K>;      // [128, 32]
var<workgroup> b_shared: array<f32, TILE_K * TILE_N>;      // [32, 128] (transposed during load)

@group(0) @binding(0) var<uniform> info: MatmulInfo;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;
@group(0) @binding(2) var<storage, read> b_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>;

fn unpack_bf16(packed: u32) -> vec2<f32> {
    return vec2<f32>(
        bitcast<f32>(packed << 16u),
        bitcast<f32>(packed & 0xFFFF0000u)
    );
}

@compute @workgroup_size(THREADS_X, THREADS_Y, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    let tx = local_id.x;
    let ty = local_id.y;
    
    // Output tile position
    let tile_m = group_id.y * TILE_M;
    let tile_n = group_id.x * TILE_N;
    
    // Each thread's output region
    let thread_m = ty * THREAD_M;
    let thread_n = tx * THREAD_N;
    
    // Accumulators in registers (8x8 per thread)
    var acc: array<array<f32, THREAD_N>, THREAD_M>;
    for (var i = 0u; i < THREAD_M; i++) {
        for (var j = 0u; j < THREAD_N; j++) {
            acc[i][j] = 0.0;
        }
    }
    
    let num_k_tiles = (info.k + TILE_K - 1u) / TILE_K;
    
    for (var kt = 0u; kt < num_k_tiles; kt++) {
        let k_offset = kt * TILE_K;
        
        // === Cooperative Load A [TILE_M, TILE_K] ===
        // 256 threads, 128*32 = 4096 elements, 16 per thread
        let a_loads_per_thread = (TILE_M * TILE_K) / (THREADS_X * THREADS_Y);
        for (var i = 0u; i < a_loads_per_thread; i++) {
            let flat_idx = local_idx * a_loads_per_thread + i;
            let a_m = flat_idx / TILE_K;
            let a_k = flat_idx % TILE_K;
            
            let global_m = tile_m + a_m;
            let global_k = k_offset + a_k;
            
            if (global_m < info.m && global_k < info.k) {
                a_shared[a_m * TILE_K + a_k] = a_in[global_m * info.k + global_k];
            } else {
                a_shared[a_m * TILE_K + a_k] = 0.0;
            }
        }
        
        // === Cooperative Load B [N, K] -> shared [K, N] (transpose) ===
        let b_loads_per_thread = (TILE_K * TILE_N) / (THREADS_X * THREADS_Y);
        for (var i = 0u; i < b_loads_per_thread; i++) {
            let flat_idx = local_idx * b_loads_per_thread + i;
            let s_k = flat_idx / TILE_N;  // shared memory K index
            let s_n = flat_idx % TILE_N;  // shared memory N index
            
            let global_n = tile_n + s_n;
            let global_k = k_offset + s_k;
            
            if (global_n < info.n && global_k < info.k) {
                // B physical: [N, K], access B[global_n, global_k]
                let b_idx = global_n * info.k + global_k;
                let packed = b_in[b_idx / 2u];
                let vals = unpack_bf16(packed);
                let val = select(vals.y, vals.x, b_idx % 2u == 0u);
                b_shared[s_k * TILE_N + s_n] = val;
            } else {
                b_shared[s_k * TILE_N + s_n] = 0.0;
            }
        }
        
        workgroupBarrier();
        
        // === Compute 8x8 output per thread ===
        for (var k = 0u; k < TILE_K; k++) {
            // Load A values for this k into registers
            var a_reg: array<f32, THREAD_M>;
            for (var i = 0u; i < THREAD_M; i++) {
                a_reg[i] = a_shared[(thread_m + i) * TILE_K + k];
            }
            
            // Load B values for this k into registers
            var b_reg: array<f32, THREAD_N>;
            for (var j = 0u; j < THREAD_N; j++) {
                b_reg[j] = b_shared[k * TILE_N + thread_n + j];
            }
            
            // Outer product
            for (var i = 0u; i < THREAD_M; i++) {
                for (var j = 0u; j < THREAD_N; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        workgroupBarrier();
    }
    
    // === Write results ===
    for (var i = 0u; i < THREAD_M; i++) {
        for (var j = 0u; j < THREAD_N; j++) {
            let global_m = tile_m + thread_m + i;
            let global_n = tile_n + thread_n + j;
            
            if (global_m < info.m && global_n < info.n) {
                c_out[global_m * info.n + global_n] = acc[i][j];
            }
        }
    }
}