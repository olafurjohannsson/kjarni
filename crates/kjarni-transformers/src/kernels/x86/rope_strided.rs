use std::arch::x86_64::*;

/// Fully optimized AVX2 RoPE for GGUF layout (Split Complex).
///
/// This handles the [Real, Real... | Imag, Imag...] layout using strided loads.
///
/// # Arguments
/// * `head_ptr` - Pointer to the start of the head for the specific token.
/// * `cos_ptr` - Pointer to the cosine cache for this position.
/// * `sin_ptr` - Pointer to the sine cache for this position.
/// * `head_dim` - The dimension of the head (e.g., 128 for 3B).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rope_strided_avx2(
    head_ptr: *mut f32,
    cos_ptr: *const f32,
    sin_ptr: *const f32,
    head_dim: usize,
) {
    let half_dim = head_dim / 2;
    unsafe {
        // Process 8 pairs (16 floats) per iteration
        let mut i = 0;
        while i + 8 <= half_dim {
            // Pointers to Real (x) and Imag (y) parts
            let x_ptr = head_ptr.add(i);
            let y_ptr = head_ptr.add(i + half_dim);

            // Load Data
            let x_vec = _mm256_loadu_ps(x_ptr);
            let y_vec = _mm256_loadu_ps(y_ptr);

            // Load Cache
            let cos_vec = _mm256_loadu_ps(cos_ptr.add(i));
            let sin_vec = _mm256_loadu_ps(sin_ptr.add(i));

            // Calculate RoPE:
            // out_x = x * cos - y * sin
            // out_y = x * sin + y * cos

            // let out_x = _mm256_fmadd_ps(
            //     x_vec, cos_vec,                 // x * cos
            //     _mm256_mul_ps(y_vec, sin_vec).neg() // - (y * sin) (negate manually if fnmadd not avail)
            // );
            // Note: For strict correctness with FMA: x*cos - y*sin.
            // _mm256_fnmadd_ps(a, b, c) does -(a*b) + c.
            // So we want _mm256_fnmadd_ps(y, sin, x*cos).
            // But straight sub is often cleaner to read unless maximizing precision.
            let term1 = _mm256_mul_ps(x_vec, cos_vec);
            let term2 = _mm256_mul_ps(y_vec, sin_vec);
            let final_x = _mm256_sub_ps(term1, term2);

            let out_y = _mm256_fmadd_ps(
                x_vec,
                sin_vec,                       // x * sin
                _mm256_mul_ps(y_vec, cos_vec), // + y * cos
            );

            // Store back
            _mm256_storeu_ps(x_ptr, final_x);
            _mm256_storeu_ps(y_ptr, out_y);

            i += 8;
        }

        // Handle remainder (scalar fallback)
        while i < half_dim {
            let x = *head_ptr.add(i);
            let y = *head_ptr.add(i + half_dim);
            let c = *cos_ptr.add(i);
            let s = *sin_ptr.add(i);

            *head_ptr.add(i) = x * c - y * s;
            *head_ptr.add(i + half_dim) = x * s + y * c;
            i += 1;
        }
    }
}
