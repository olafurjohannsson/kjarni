use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_0};

/// Dequantize a Q4_K block to 256 f32 values.
pub fn dequantize_q4_k_block(b: &BlockQ4_K, out: &mut [f32]) {
    let d = b.d.to_f32();
    let dmin = b.dmin.to_f32();

    let mut is = 0; // Sub-block index (0..7)
    let mut q_idx = 0; // Index into qs array

    // Process 4 pairs of sub-blocks (each pair processes 64 values)
    for j in 0..4 {
        // QK_K / 64 = 256 / 64 = 4
        // Get scales and mins for two consecutive sub-blocks
        let (sc1, m1) = get_scale_min_k4(is, &b.scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, &b.scales);

        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;

        // First 32 elements: lower nibbles with first scale/min
        for l in 0..32 {
            out[j * 64 + l] = d1 * (b.qs[q_idx + l] & 0xF) as f32 - min1;
        }

        // Next 32 elements: upper nibbles with second scale/min
        for l in 0..32 {
            out[j * 64 + 32 + l] = d2 * (b.qs[q_idx + l] >> 4) as f32 - min2;
        }

        q_idx += 32;
        is += 2;
    }
}

/// Dequantize a Q5_K block to 256 f32 values, as ggml's `dequantize_row_q5_K`.
pub fn dequantize_q5_k_block(b: &BlockQ5_K, out: &mut [f32]) {
    let d = b.d.to_f32();
    let dmin = b.dmin.to_f32();
    for j in 0..4 {
        let (sc1, m1) = get_scale_min_k4(2 * j, &b.scales);
        let (sc2, m2) = get_scale_min_k4(2 * j + 1, &b.scales);
        let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
        let ql = &b.qs[j * 32..j * 32 + 32];
        for l in 0..32 {
            let lo = (ql[l] & 0xF) | (((b.qh[l] >> (2 * j)) & 1) << 4);
            let hi = (ql[l] >> 4) | (((b.qh[l] >> (2 * j + 1)) & 1) << 4);
            out[j * 64 + l] = d1 * lo as f32 - min1;
            out[j * 64 + 32 + l] = d2 * hi as f32 - min2;
        }
    }
}

// Q8_0 Dequantization

/// Dequantize a Q8_0 block to 32 f32 values.
pub fn dequantize_q8_0_block(block: &BlockQ8_0, output: &mut [f32]) {
    let scale = block.d.to_f32();
    for (i, &q) in block.qs.iter().enumerate() {
        output[i] = (q as f32) * scale;
    }
}

// Q6_K Dequantization

//
// Q6_K block structure (256 values):
// - ql: 128 bytes (lower 4 bits of each 6-bit value)
// - qh: 64 bytes (upper 2 bits, packed)
// - scales: 16 x i8 sub-block scales
// - d: f16 super-block scale
//
// Each value is: ((ql & 0xF) | ((qh_bits) << 4)) - 32, then scaled

/// Dequantize a Q6_K block to 256 f32 values.
#[inline(always)]
pub fn dequantize_q6_k_block(b: &BlockQ6_K, out: &mut [f32]) {
    let d = b.d.to_f32();

    // Loop over two halves (0..128 and 128..256)
    for i in 0..2 {
        let ql = &b.ql[i * 64..];
        let qh = &b.qh[i * 32..];
        let sc = &b.scales[i * 8..];
        let out_ptr = &mut out[i * 128..];

        for j in 0..32 {
            let is = j / 16; // 0 or 1

            // Extract the byte containing high bits for 4 values
            let qh_val = qh[j];

            // 1. Lower 4 bits from ql[j], Upper 2 bits from qh[j] (bits 0-1)
            // Scale: sc[is + 0]
            let q0 = ((ql[j] & 0xF) as i8) | (((qh_val & 0x03) << 4) as i8);
            let val0 = (q0.wrapping_sub(32)) as f32;
            out_ptr[j] = d * val0 * sc[is] as f32;

            // 2. Lower 4 bits from ql[j+32], Upper 2 bits from qh[j] (bits 2-3)
            // Scale: sc[is + 2]
            let q1 = ((ql[j + 32] & 0xF) as i8) | (((qh_val & 0x0C) << 2) as i8);
            let val1 = (q1.wrapping_sub(32)) as f32;
            out_ptr[j + 32] = d * val1 * sc[is + 2] as f32;

            // 3. Upper 4 bits from ql[j], Upper 2 bits from qh[j] (bits 4-5)
            // Scale: sc[is + 4]
            let q2 = ((ql[j] >> 4) as i8) | ((qh_val & 0x30) as i8);
            let val2 = (q2.wrapping_sub(32)) as f32;
            out_ptr[j + 64] = d * val2 * sc[is + 4] as f32;

            // 4. Upper 4 bits from ql[j+32], Upper 2 bits from qh[j] (bits 6-7)
            // Scale: sc[is + 6]
            let q3 = ((ql[j + 32] >> 4) as i8) | (((qh_val & 0xC0) >> 2) as i8);
            let val3 = (q3.wrapping_sub(32)) as f32;
            out_ptr[j + 96] = d * val3 * sc[is + 6] as f32;
        }
    }
}

//
// Q4_K block structure (256 values in 144 bytes):
// - d: f16 super-block scale (2 bytes)
// - dmin: f16 super-block minimum (2 bytes)
// - scales: 12 bytes encoding 8 x 6-bit scales + 8 x 6-bit mins
// - qs: 128 bytes of 4-bit quantized values
//
// The 8 sub-blocks each have 32 values. Scales/mins are 6-bit values packed as:
// - bytes 0-3: lower 6 bits of scales 0-3
// - bytes 4-7: lower 6 bits of mins 0-3
// - bytes 8-11: packed upper bits and scales/mins 4-7
//
// For each sub-block j:
//   value[i] = d * scale[j] * q[i] - dmin * min[j]

/// Decode the 6-bit scale and min for sub-block j from the packed scales array.
///
/// This matches llama.cpp's get_scale_min_k4 function exactly.
#[inline]
pub fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        // Sub-blocks 0-3: simple 6-bit values in first 8 bytes
        let sc = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (sc, m)
    } else {
        // Sub-blocks 4-7: reconstruct 6-bit values from bytes 8-11 + high bits
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

// Unit Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_scale_min_k4_low_indices() {
        // Test sub-blocks 0-3 use simple 6-bit extraction
        let scales: [u8; 12] = [
            0b00_111111,
            0b00_101010,
            0b00_010101,
            0b00_000000, // scales 0-3 (lower 6 bits)
            0b00_110011,
            0b00_001100,
            0b00_110000,
            0b00_001111, // mins 0-3 (lower 6 bits)
            0,
            0,
            0,
            0, // unused for j < 4
        ];

        let (sc0, m0) = get_scale_min_k4(0, &scales);
        assert_eq!(sc0, 63);
        assert_eq!(m0, 51);

        let (sc1, m1) = get_scale_min_k4(1, &scales);
        assert_eq!(sc1, 42);
        assert_eq!(m1, 12);
    }

    #[test]
    fn test_get_scale_min_k4_high_indices() {
        // Test sub-blocks 4-7 reconstruct from packed bits
        let scales: [u8; 12] = [
            0b11_000000,
            0b10_000000,
            0b01_000000,
            0b00_000000, // high bits in bits 6-7
            0b11_000000,
            0b10_000000,
            0b01_000000,
            0b00_000000, // high bits in bits 6-7
            0b0101_0011,
            0b0110_0100,
            0b0111_0101,
            0b1000_0110, // bytes 8-11
        ];

        // j=4: sc = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
        //        = 0x3 | (3 << 4) = 0x3 | 0x30 = 0x33 = 51
        //      m = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
        //        = 0x5 | (3 << 4) = 0x5 | 0x30 = 0x35 = 53
        let (sc4, m4) = get_scale_min_k4(4, &scales);
        assert_eq!(sc4, 51);
        assert_eq!(m4, 53);
    }

    #[test]
    fn test_q4k_dequantize_not_all_zeros() {
        // Create a test block with known non-zero values
        let block = BlockQ4_K {
            d: half::f16::from_f32(1.0),
            dmin: half::f16::from_f32(0.0),
            scales: [63, 63, 63, 63, 63, 63, 63, 63, 0xFF, 0xFF, 0xFF, 0xFF],
            qs: [0x88; 128], // All values = 8
        };

        let mut out = [0.0f32; 256];
        dequantize_q4_k_block(&block, &mut out);

        // With scale=63, d=1.0, q=8: value = 1.0 * 63 * 8 = 504
        assert!(out[0] > 0.0, "First value should be positive");
        assert!(
            out.iter().all(|&v| v.is_finite()),
            "All values should be finite"
        );
    }

    /// The first block of `blk.0.attn_qkv.weight` in unsloth's
    /// Qwen3.5-0.8B-Q4_K_M.gguf, which stores that tensor as Q5_K.
    const REAL_Q5_K: &str = "65037511e59ca3aa6cde69ffdf85a2cb8ccb0969294b8da88220098de0222e29a76c26bf2c082c4a64cae92ba9afec8c794274309a9037406b0750a1e7465da7376e5836ed9c96fd8c7def2c64d7d7cd31777b5a4ba1935db8da5b30ae0b2158f8208c704b0c35165360ed9a9e361e82b2f6467623aaf8d450d8d93980476997471ae34f4b76c5b73604f608794f75d5872fc7b08ac30a5c0e02e173b4fff06b05a09aedcd98d6f5f12115e5ae9704b0";

    fn real_q5_k() -> crate::cpu::kernels::q_common::BlockQ5_K {
        let bytes: Vec<u8> = (0..REAL_Q5_K.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&REAL_Q5_K[i..i + 2], 16).unwrap())
            .collect();
        *bytemuck::from_bytes(&bytes)
    }

    /// Values from gguf-py's `dequantize` (llama.cpp's reference) on the same block.
    #[test]
    fn q5_k_dequantizes_as_ggml_does() {
        let mut out = [0f32; 256];
        dequantize_q5_k_block(&real_q5_k(), &mut out);
        for (i, want) in [
            (0, -0.012062013149261475f32),
            (1, 0.005186200141906738),
            (31, -0.0043961405754089355),
            (32, -0.009832143783569336),
            (63, -0.0025806427001953125),
            (64, 0.003507077693939209),
            (100, 0.0015420913696289062),
            (127, 0.010243892669677734),
            (128, -0.012791752815246582),
            (191, -0.012389838695526123),
            (200, 0.0073354244232177734),
            (255, 0.020167171955108643),
        ] {
            assert!(
                (out[i] - want).abs() < 1e-7,
                "value {i}: {} vs {want}",
                out[i]
            );
        }
        let sum: f64 = out.iter().map(|&v| v as f64).sum();
        assert!((sum - -0.129644513130188).abs() < 1e-5, "sum {sum}");
    }

    /// The integer dot product equals the dequantized weights against the
    /// dequantized activations.
    #[test]
    fn q5_k_dot_matches_its_dequantized_values() {
        let w = real_q5_k();
        let a: Vec<f32> = (0..256)
            .map(|i| ((i * 37) % 101) as f32 / 25.0 - 2.0)
            .collect();
        let q = crate::cpu::kernels::quantize::quantize_row_q8_k(&a);
        let mut wf = [0f32; 256];
        dequantize_q5_k_block(&w, &mut wf);
        let want: f64 = wf
            .iter()
            .zip(q[0].qs.iter())
            .map(|(&x, &y)| x as f64 * y as f64 * q[0].d as f64)
            .sum();
        let got = crate::cpu::kernels::scalar::vec_dot_q5k_q8k_scalar(256, &[w], &q);
        assert!(
            (got as f64 - want).abs() < 1e-5 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }
}
