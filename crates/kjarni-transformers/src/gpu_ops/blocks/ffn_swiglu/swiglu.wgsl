//! SwiGLU activation function 

/// Input from gate projection (W_gate @ x).
@group(0) @binding(0) var<storage, read> a_in: array<f32>;
/// Input from up projection (W_up @ x).
@group(0) @binding(1) var<storage, read> b_in: array<f32>;
/// Output: silu(gate) * up.
@group(0) @binding(2) var<storage, read_write> c_out: array<f32>;

/// ReLU activation
fn relu(x: f32) -> f32 {
    return max(x, 0.0);
}

/// SiLU (Swish) activation: x * sigmoid(x).
fn silu(x: f32) -> f32 {
    // Clamping prevents exp overflow and underflow
    if (x <= -20.0) { return 0.0; }
    if (x >= 20.0) { return x; }
    return x / (1.0 + exp(-x));
}

/// Fast SiLU without branches
fn silu_fast(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

/// Applies SwiGLU activation 
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let gate_val = a_in[idx];
    let up_val = b_in[idx];

    // SwiGLU: silu(gate) * up
    c_out[idx] = silu(gate_val) * up_val;
}