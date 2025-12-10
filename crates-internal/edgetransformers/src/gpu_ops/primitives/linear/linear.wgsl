struct LinearInfo {
    M: u32, // Batch * Seq
    K: u32, // Hidden Size
    N: u32, // Vocab Size
}

@group(0) @binding(0) var<uniform> info: LinearInfo;
@group(0) @binding(1) var<storage, read> Input: array<f32>;

// We use two bindings for weights. Only one is active per pipeline.
@group(0) @binding(2) var<storage, read> W_F32: array<f32>;  // [N, K]
@group(0) @binding(3) var<storage, read> W_BF16: array<u32>; // [N, K/2]

@group(0) @binding(4) var<storage, read_write> Output: array<f32>; // [M, N]

// Reduction cache for Wide kernel
var<workgroup> wg_sum: array<f32, 256>;

fn unpack_bf16(packed: u32) -> vec2<f32> {
    let x = bitcast<f32>(packed << 16u);
    let y = bitcast<f32>(packed & 0xFFFF0000u);
    return vec2<f32>(x, y);
}

// ------------------------------------------------------------------
// KERNEL 1: Wide GEMV (BF16) - Optimized for Large N (lm_head)
// Dispatch: (N, 1, 1) -> One Workgroup (256 threads) per Output Neuron
// ------------------------------------------------------------------
@compute @workgroup_size(256)
fn gemv_bf16_wide(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32> 
) {
    let n = wg_id.y * grid.x + wg_id.x; 
    
    let tid = local_id.x;

    if (n >= info.N) { return; }

    // 1. Collaborative Dot Product
    // The warp reads contiguous blocks of Input and Weights.
    // Thread 0 reads W[n, 0..1], Thread 1 reads W[n, 2..3]...
    // This creates perfectly coalesced 128-byte transactions.
    
    let k_pairs = info.K / 2u;
    let weight_base = n * k_pairs;
    var partial_sum = 0.0;

    for (var k = tid; k < k_pairs; k = k + 256u) {
        let in0 = Input[k * 2u];
        let in1 = Input[k * 2u + 1u];
        
        let w_packed = W_BF16[weight_base + k];
        let w_vec = unpack_bf16(w_packed);
        
        partial_sum += in0 * w_vec.x + in1 * w_vec.y;
    }

    wg_sum[tid] = partial_sum;
    workgroupBarrier();

    // 2. Tree Reduction in Shared Memory
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum[tid] += wg_sum[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        Output[n] = wg_sum[0];
    }
}

// ------------------------------------------------------------------
// KERNEL 2: Standard GEMV (BF16) - 1 Thread per Output
// Good for small N (e.g. QKV projections) where launching 128k blocks is bad.
// ------------------------------------------------------------------
@compute @workgroup_size(256)
fn gemv_bf16(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if (n >= info.N) { return; }

    var sum = vec4<f32>(0.0);
    let k_pairs = info.K / 2u;
    let weight_offset = n * k_pairs;

    // Manual unroll x4 for instruction-level parallelism
    // Process 4 pairs (8 floats) per iteration
    var k = 0u;
    for (; k + 3u < k_pairs; k += 4u) {
        let w0 = unpack_bf16(W_BF16[weight_offset + k]);
        let w1 = unpack_bf16(W_BF16[weight_offset + k + 1u]);
        let w2 = unpack_bf16(W_BF16[weight_offset + k + 2u]);
        let w3 = unpack_bf16(W_BF16[weight_offset + k + 3u]);

        // Input index is k*2
        let idx = k * 2u;
        let in0 = Input[idx];      let in1 = Input[idx + 1u];
        let in2 = Input[idx + 2u]; let in3 = Input[idx + 3u];
        let in4 = Input[idx + 4u]; let in5 = Input[idx + 5u];
        let in6 = Input[idx + 6u]; let in7 = Input[idx + 7u];

        sum.x += in0 * w0.x + in1 * w0.y;
        sum.y += in2 * w1.x + in3 * w1.y;
        sum.z += in4 * w2.x + in5 * w2.y;
        sum.w += in6 * w3.x + in7 * w3.y;
    }

    // Cleanup tail
    var tail_sum = 0.0;
    for (; k < k_pairs; k += 1u) {
        let w = unpack_bf16(W_BF16[weight_offset + k]);
        let in0 = Input[k * 2u];
        let in1 = Input[k * 2u + 1u];
        tail_sum += in0 * w.x + in1 * w.y;
    }

    Output[n] = sum.x + sum.y + sum.z + sum.w + tail_sum;
}

// ------------------------------------------------------------------
// KERNEL 3: BMM (BF16) - 2D Tiled
// Used for Prefill (M > 1).
// ------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn bmm_bf16(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    let m = id.y;
    
    if (m >= info.M || n >= info.N) { return; }

    var sum = 0.0;
    let input_offset = m * info.K;
    let k_pairs = info.K / 2u;
    let weight_offset = n * k_pairs;
    
    for (var k = 0u; k < k_pairs; k = k + 1u) {
        let w_vec = unpack_bf16(W_BF16[weight_offset + k]);
        let in0 = Input[input_offset + k * 2u];
        let in1 = Input[input_offset + k * 2u + 1u];
        sum += in0 * w_vec.x + in1 * w_vec.y;
    }
    Output[m * info.N + n] = sum;
}

// ------------------------------------------------------------------
// KERNEL 4: GEMV (F32)
// ------------------------------------------------------------------
@compute @workgroup_size(256)
fn gemv_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if (n >= info.N) { return; }

    var sum = 0.0;
    let weight_offset = n * info.K;
    
    for (var k = 0u; k < info.K; k = k + 1u) {
        sum += Input[k] * W_F32[weight_offset + k];
    }
    Output[n] = sum;
}

// ------------------------------------------------------------------
// KERNEL 5: BMM (F32)
// ------------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn bmm_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    let m = id.y;
    if (m >= info.M || n >= info.N) { return; }

    var sum = 0.0;
    let input_offset = m * info.K;
    let weight_offset = n * info.K;

    for (var k = 0u; k < info.K; k = k + 1u) {
        sum += Input[input_offset + k] * W_F32[weight_offset + k];
    }
    Output[m * info.N + n] = sum;
}