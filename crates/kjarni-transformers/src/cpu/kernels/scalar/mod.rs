mod scalar;


pub use scalar::{
    matmul_vec_bf16_scalar,
    matmul_vec_f32_scalar,
    matmul_vec_q4_k_scalar,
    matmul_vec_q6_k_scalar,
    matmul_vec_q8_0_scalar,
    vec_dot_q6k_q8k_scalar,
};

// todo: rope
// todo softmax?