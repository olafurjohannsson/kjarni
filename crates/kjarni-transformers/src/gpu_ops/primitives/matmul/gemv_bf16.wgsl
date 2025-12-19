// GEMV for M=1, BF16 Weights (Implicit Transpose)
struct MatmulInfo {
    m: u32,
    k: u32,
    n: u32,
}

@group(0) @binding(0) var<uniform> info: MatmulInfo;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;
@group(0) @binding(2) var<storage, read> b_in: array<u32>; // Packed BF16 pairs
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>;

// Manual BF16 Unpack
// BF16 is just the upper 16 bits of an F32.
// To convert BF16 -> F32, we shift the bits left by 16 and pad with zeros.
fn unpack_bf16(packed: u32) -> vec2<f32> {
    // Little Endian: Low 16 bits are the first value, High 16 are the second.
    
    // 1. First Value (Low 16 bits) -> Shift left to fill top 16 bits
    let v1_bits = packed << 16u;
    let v1 = bitcast<f32>(v1_bits);
    
    // 2. Second Value (High 16 bits) -> Mask them (they are already at top)
    let v2_bits = packed & 0xFFFF0000u;
    let v2 = bitcast<f32>(v2_bits);
    
    return vec2<f32>(v1, v2);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n_idx = global_id.x; // The output neuron we are computing
    
    if (n_idx >= info.n) { return; }

    var acc = 0.0;
    
    // B is physically [N, K]. We want row n_idx.
    // Row start index in ELEMENTS (not u32s)
    let b_row_start_idx = n_idx * info.k; 
    
    // We iterate K.
    // Optimization: Unroll by 2 to process one u32 at a time.
    let k_u32_start = b_row_start_idx / 2u;
    let k_u32_count = info.k / 2u; // Assumes K is even (valid for Llama)

    var a_idx = 0u;

    for (var i = 0u; i < k_u32_count; i = i + 1u) {
        let packed_b = b_in[k_u32_start + i];
        let b_vals = unpack_bf16(packed_b);
        
        let a0 = a_in[a_idx];
        let a1 = a_in[a_idx + 1u];
        
        acc = acc + (a0 * b_vals.x) + (a1 * b_vals.y);
        a_idx = a_idx + 2u;
    }

    c_out[n_idx] = acc;
}