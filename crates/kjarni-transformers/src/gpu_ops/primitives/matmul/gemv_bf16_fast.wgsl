// gemv_bf16_fast.wgsl - Optimized for single-token decode
// A: [1, K] F32, B: [N, K] packed BF16, C: [1, N] F32

const WORKGROUP_SIZE = 256u;
const ELEMENTS_PER_THREAD = 8u;  // Process 8 K elements per iteration

struct MatmulInfo {
    m: u32,
    k: u32,
    n: u32,
}

var<workgroup> partial_sums: array<f32, WORKGROUP_SIZE>;

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

// Each workgroup computes one output element
@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    let n_idx = group_id.x;
    if (n_idx >= info.n) { return; }
    
    let tid = local_idx;
    
    // B row start (in BF16 elements)
    let b_row_start = n_idx * info.k;
    
    // Each thread processes a strided portion of K
    var acc = 0.0;
    
    // Process 4 BF16 pairs (8 elements) per iteration for better throughput
    let k_per_thread = (info.k + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    let k_start = tid * k_per_thread;
    let k_end = min(k_start + k_per_thread, info.k);
    
    var k = k_start;
    
    // Main loop - process 8 elements at a time
    for (; k + 8u <= k_end; k += 8u) {
        // Load 4 packed u32s (8 BF16 values)
        let b_base = (b_row_start + k) / 2u;
        let p0 = b_in[b_base];
        let p1 = b_in[b_base + 1u];
        let p2 = b_in[b_base + 2u];
        let p3 = b_in[b_base + 3u];
        
        let b01 = unpack_bf16(p0);
        let b23 = unpack_bf16(p1);
        let b45 = unpack_bf16(p2);
        let b67 = unpack_bf16(p3);
        
        let a0 = a_in[k];
        let a1 = a_in[k + 1u];
        let a2 = a_in[k + 2u];
        let a3 = a_in[k + 3u];
        let a4 = a_in[k + 4u];
        let a5 = a_in[k + 5u];
        let a6 = a_in[k + 6u];
        let a7 = a_in[k + 7u];
        
        acc += a0 * b01.x + a1 * b01.y;
        acc += a2 * b23.x + a3 * b23.y;
        acc += a4 * b45.x + a5 * b45.y;
        acc += a6 * b67.x + a7 * b67.y;
    }
    
    // Remainder loop
    for (; k < k_end; k += 2u) {
        let packed = b_in[(b_row_start + k) / 2u];
        let bv = unpack_bf16(packed);
        acc += a_in[k] * bv.x;
        if (k + 1u < k_end) {
            acc += a_in[k + 1u] * bv.y;
        }
    }
    
    partial_sums[tid] = acc;
    workgroupBarrier();
    
    // Parallel reduction
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        c_out[n_idx] = partial_sums[0];
    }
}