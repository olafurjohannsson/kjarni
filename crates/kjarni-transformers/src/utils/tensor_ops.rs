use faer::Parallelism;
use ndarray::{Array2, ArrayView2};

/// Computes matrix multiplication `C = A @ B`
pub fn matmul_2d_faer(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k, k2, "Matmul dimension mismatch: A[k] != B[k]");

    let mut c = Array2::<f32>::zeros((m, n));

    let a_s = a.as_standard_layout();
    let b_s = b.as_standard_layout();
    let a_sl = a_s.as_slice().unwrap();
    let b_sl = b_s.as_slice().unwrap();
    let c_sl = c.as_slice_mut().unwrap();

    faer::linalg::matmul::matmul(
        faer::mat::from_row_major_slice_mut(c_sl, m, n),
        faer::mat::from_row_major_slice(a_sl, m, k),
        faer::mat::from_row_major_slice(b_sl, k, n),
        None,
        1.0,
        Parallelism::Rayon(0),
    );
    c
}