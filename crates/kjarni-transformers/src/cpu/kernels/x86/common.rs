#![allow(unsafe_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Helper function to horizontally sum a `__m256` vector.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) unsafe fn hsum_ps_avx(v: __m256) -> f32 {
    unsafe {
        let vlow = _mm256_castps256_ps128(v);
        let vhigh = _mm256_extractf128_ps(v, 1);
        let vsum = _mm_add_ps(vlow, vhigh);
        let vsum = _mm_hadd_ps(vsum, vsum);
        let vsum = _mm_hadd_ps(vsum, vsum);
        _mm_cvtss_f32(vsum)
    }
}
