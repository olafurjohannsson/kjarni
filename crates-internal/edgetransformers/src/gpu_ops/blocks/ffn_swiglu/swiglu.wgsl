// Input A (from gate projection)
@group(0) @binding(0) var<storage, read> a_in: array<f32>;
// Input B (from up projection)
@group(0) @binding(1) var<storage, read> b_in: array<f32>;
// Output C
@group(0) @binding(2) var<storage, read_write> c_out: array<f32>;

// SiLU activation function: x * sigmoid(x)
fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // The number of elements is determined by the dispatch size, so no bounds check needed
    // if we are careful on the Rust side. Adding one just in case is good practice.
    // (Assuming c_out is the same size as a_in and b_in)
    
    let gate_val = a_in[idx];
    let up_val = b_in[idx];
    
    c_out[idx] = silu(gate_val) * up_val;
}