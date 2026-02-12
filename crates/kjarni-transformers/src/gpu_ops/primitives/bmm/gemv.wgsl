struct Uniforms {
    M: u32, // Batch size (1 for token generation)
    K: u32, // Input features
    N: u32, // Output features
}

@group(0) @binding(0) var<uniform> info: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;  // [M, K]
@group(0) @binding(2) var<storage, read> weight: array<f32>; // [N, K] (Native Layout)
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x; // Output feature index (0..N)
    
    if (row >= info.N) {
        return;
    }

    var sum = 0.0;
    
    // Weight offset: row * K (jump to the start of the row)
    let weight_offset = row * info.K;
    
    // Loop over K (inner dimension)
    // Both input and weight are read sequentially!
    // input[k] and weight[row*K + k]
    for (var k = 0u; k < info.K; k = k + 1u) {
        sum = sum + input[k] * weight[weight_offset + k];
    }

    output[row] = sum;
}