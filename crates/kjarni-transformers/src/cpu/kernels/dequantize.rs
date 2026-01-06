use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0};


/// Dequantize a Q4_K block to 256 f32 values.
pub fn dequantize_q4_k_block(b: &BlockQ4_K, out: &mut [f32]) {
    let d = b.d.to_f32();
    let dmin = b.dmin.to_f32();

    let mut is = 0;      // Sub-block index (0..7)
    let mut q_idx = 0;   // Index into qs array

    // Process 4 pairs of sub-blocks (each pair processes 64 values)
    for j in 0..4 {  // QK_K / 64 = 256 / 64 = 4
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

// =============================================================================
// Q8_0 Dequantization
// =============================================================================

/// Dequantize a Q8_0 block to 32 f32 values.
pub fn dequantize_q8_0_block(block: &BlockQ8_0, output: &mut [f32]) {
    let scale = block.d.to_f32();
    for (i, &q) in block.qs.iter().enumerate() {
        output[i] = (q as f32) * scale;
    }
}

// =============================================================================
// Q6_K Dequantization
// =============================================================================
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
            let q2 = ((ql[j] >> 4) as i8) | (((qh_val & 0x30) << 0) as i8);
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

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_scale_min_k4_low_indices() {
        // Test sub-blocks 0-3 use simple 6-bit extraction
        let scales: [u8; 12] = [
            0b00_111111, 0b00_101010, 0b00_010101, 0b00_000000, // scales 0-3 (lower 6 bits)
            0b00_110011, 0b00_001100, 0b00_110000, 0b00_001111, // mins 0-3 (lower 6 bits)
            0, 0, 0, 0, // unused for j < 4
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
            0b11_000000, 0b10_000000, 0b01_000000, 0b00_000000, // high bits in bits 6-7
            0b11_000000, 0b10_000000, 0b01_000000, 0b00_000000, // high bits in bits 6-7
            0b0101_0011, 0b0110_0100, 0b0111_0101, 0b1000_0110, // bytes 8-11
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
        let mut block = BlockQ4_K {
            d: half::f16::from_f32(1.0),
            dmin: half::f16::from_f32(0.0),
            scales: [63, 63, 63, 63, 63, 63, 63, 63, 0xFF, 0xFF, 0xFF, 0xFF],
            qs: [0x88; 128], // All values = 8
        };

        let mut out = [0.0f32; 256];
        dequantize_q4_k_block(&block, &mut out);

        // With scale=63, d=1.0, q=8: value = 1.0 * 63 * 8 = 504
        assert!(out[0] > 0.0, "First value should be positive");
        assert!(out.iter().all(|&v| v.is_finite()), "All values should be finite");
    }
}