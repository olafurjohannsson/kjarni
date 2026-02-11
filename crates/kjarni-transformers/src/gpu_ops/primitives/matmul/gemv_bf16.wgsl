//! Optimized vector-matrix multiplication (GEMV) for BF16 weights

struct MatmulInfo {
    /// Number of rows in input (always 1 for GEMV).
    m: u32,
    /// Inner dimension (length of input vector).
    k: u32,
    /// Number of output elements (number of weight matrix rows).
    n: u32,
}

@group(0) @binding(0) var<uniform> info: MatmulInfo;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;      // [1, K] input vector
@group(0) @binding(2) var<storage, read> b_in: array<u32>;      // [N, K/2] packed BF16 weights
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>; // [1, N] output vector

/// Unpacks two BF16 values from a u32.
fn unpack_bf16(packed: u32) -> vec2<f32> {
    // First value (low 16 bits) → shift to high position
    let v1_bits = packed << 16u;
    let v1 = bitcast<f32>(v1_bits);

    // Second value (high 16 bits) → already in position
    let v2_bits = packed & 0xFFFF0000u;
    let v2 = bitcast<f32>(v2_bits);

    return vec2<f32>(v1, v2);
}

/// GEMV kernel
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n_idx = global_id.x; // Output element index

    if (n_idx >= info.n) { return; }

    var acc = 0.0;

    let b_row_start_idx = n_idx * info.k; // Element offset

    let k_u32_start = b_row_start_idx / 2u;
    let k_u32_count = info.k / 2u; 

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
