struct Info {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> input: array<f32>;       // [M, K]

// BF16 Weights (Packed u32)
@group(0) @binding(2) var<storage, read> gate_w_bf16: array<u32>; // [N, K/2]
@group(0) @binding(3) var<storage, read> up_w_bf16: array<u32>;   // [N, K/2]

@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [M, N]

// F32 Weights
@group(0) @binding(5) var<storage, read> gate_w_f32: array<f32>;  // [N, K]
@group(0) @binding(6) var<storage, read> up_w_f32: array<f32>;    // [N, K]

// Shared Memory Cache for GEMV (M=1)
// Size 8192 floats = 32KB. Supports hidden_size up to 8192 (Llama-70B).
// If your GPU has strict 16KB limits, reduce this to 4096.
var<workgroup> sh_input: array<f32, 8192>;

var<workgroup> wg_sum_gate: array<f32, 256>;
var<workgroup> wg_sum_up: array<f32, 256>;


fn unpack_bf16(packed: u32) -> vec2<f32> {
    let lo = bitcast<f32>(packed << 16u);
    let hi = bitcast<f32>(packed & 0xFFFF0000u);
    return vec2<f32>(lo, hi);
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// ========================================================================
// BF16 KERNELS
// ========================================================================

// --------------------------------------------------------------------------
// WIDE Fused GEMV (BF16)
// Dispatch: (N, 1, 1) -> 1 Workgroup per Output Neuron
// --------------------------------------------------------------------------
@compute @workgroup_size(256)
fn fused_gemv_bf16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let n = wg_id.x; // Output neuron index
    let tid = local_id.x;

    if (n >= info.N) { return; }

    // 1. Collaborative Dot Products (Gate & Up)
    // Coalesced Reads: Threads read contiguous weights W[n, tid], W[n, tid+1]...
    
    let k_pairs = info.K / 2u;
    let weight_base = n * k_pairs;
    
    var partial_gate = 0.0;
    var partial_up = 0.0;

    // Stride by 256
    for (var k = tid; k < k_pairs; k = k + 256u) {
        // Read Input (Broadcast read from L2)
        let in0 = input[k * 2u];
        let in1 = input[k * 2u + 1u];
        
        // Read Weights (Coalesced global read)
        let g_packed = gate_w_bf16[weight_base + k];
        let u_packed = up_w_bf16[weight_base + k];
        
        let g_vec = unpack_bf16(g_packed);
        let u_vec = unpack_bf16(u_packed);
        
        partial_gate += in0 * g_vec.x + in1 * g_vec.y;
        partial_up   += in0 * u_vec.x + in1 * u_vec.y;
    }

    wg_sum_gate[tid] = partial_gate;
    wg_sum_up[tid] = partial_up;
    workgroupBarrier();

    // 2. Parallel Reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum_gate[tid] += wg_sum_gate[tid + s];
            wg_sum_up[tid]   += wg_sum_up[tid + s];
        }
        workgroupBarrier();
    }

    // 3. Activation & Write
    if (tid == 0u) {
        let gate_val = wg_sum_gate[0];
        let up_val = wg_sum_up[0];
        output[n] = silu(gate_val) * up_val;
    }
}


@compute @workgroup_size(16, 16)
fn fused_bmm_bf16(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x; // Output dim
    let m = id.y; // Batch/Seq dim
    
    if (m >= info.M || n >= info.N) { return; }
    
    var gate_sum = 0.0;
    var up_sum = 0.0;
    
    let input_offset = m * info.K;
    let k_pairs = info.K / 2u;
    let weight_offset = n * k_pairs;
    
    // Naive global read for BMM (Prefill phase is compute bound anyway)
    for (var k = 0u; k < k_pairs; k = k + 1u) {
        let in0 = input[input_offset + k * 2u];
        let in1 = input[input_offset + k * 2u + 1u];
        
        let g_vec = unpack_bf16(gate_w_bf16[weight_offset + k]);
        let u_vec = unpack_bf16(up_w_bf16[weight_offset + k]);
        
        gate_sum += in0 * g_vec.x + in1 * g_vec.y;
        up_sum   += in0 * u_vec.x + in1 * u_vec.y;
    }
    
    output[m * info.N + n] = silu(gate_sum) * up_sum;
}

// ========================================================================
// F32 KERNELS
// ========================================================================

@compute @workgroup_size(256)
fn fused_gemv_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let n = global_id.x;

    // 1. Collaborative Load
    let num_tiles = (info.K + 255u) / 256u;
    for (var i = 0u; i < num_tiles; i = i + 1u) {
        let load_idx = i * 256u + tid;
        if (load_idx < info.K) {
            sh_input[load_idx] = input[load_idx];
        }
    }

    workgroupBarrier();

    if (n >= info.N) { return; }
    
    var gate_sum = 0.0;
    var up_sum = 0.0;
    
    let weight_offset = n * info.K;
    
    // 2. Compute
    for (var k = 0u; k < info.K; k = k + 1u) {
        let val = sh_input[k];
        gate_sum += val * gate_w_f32[weight_offset + k];
        up_sum   += val * up_w_f32[weight_offset + k];
    }
    
    output[n] = silu(gate_sum) * up_sum;
}

@compute @workgroup_size(16, 16)
fn fused_bmm_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    let m = id.y;
    
    if (m >= info.M || n >= info.N) { return; }
    
    var gate_sum = 0.0;
    var up_sum = 0.0;
    
    let input_offset = m * info.K;
    let weight_offset = n * info.K;
    
    for (var k = 0u; k < info.K; k = k + 1u) {
        let val = input[input_offset + k];
        gate_sum += val * gate_w_f32[weight_offset + k];
        up_sum   += val * up_w_f32[weight_offset + k];
    }
    
    output[m * info.N + n] = silu(gate_sum) * up_sum;
}