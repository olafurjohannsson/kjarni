struct FfnUniforms {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32, // keep alignment to 16 bytes
}

@group(0) @binding(0) var<uniform> info: FfnUniforms;
@group(0) @binding(1) var<storage, read> fc1_weight: array<f32>;
@group(0) @binding(2) var<storage, read> fc1_bias: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;


// GELU (exact)
fn gelu_2(x: f32) -> f32 {
    let SQRT_2_INV: f32 = 0.7071067811865475;
    // Note: WGSL doesn't have erf, need to approximate or use a precomputed LUT
    // For now, use tanh approximation which is close
    let t = x * SQRT_2_INV;
    let erf_approx = tanh(1.41421356 * t); // Rough approximation
    return 0.5 * x * (1.0 + erf_approx);
}

// GELU_NEW (tanh approximation) - RECOMMENDED
fn gelu(x: f32) -> f32 {
    let SQRT_2_OVER_PI: f32 = 0.7978845608;
    let GELU_COEFF: f32 = 0.044715;

    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(512, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_outputs = info.m * info.n;
    
    if (idx >= total_outputs) {
        return;
    }
    
    let row = idx / info.n;
    let col = idx % info.n;
    
    var sum = 0.0;
    let k_vec = info.k / 4u;
    let input_offset = row * info.k;
    
    for (var k = 0u; k < k_vec; k = k + 1u) {
        let k4 = k * 4u;
        
        let input_vec = vec4<f32>(
            input[input_offset + k4],
            input[input_offset + k4 + 1u],
            input[input_offset + k4 + 2u],
            input[input_offset + k4 + 3u]
        );
        
        let weight_vec = vec4<f32>(
            fc1_weight[k4 * info.n + col],
            fc1_weight[(k4 + 1u) * info.n + col],
            fc1_weight[(k4 + 2u) * info.n + col],
            fc1_weight[(k4 + 3u) * info.n + col]
        );
        
        sum = sum + dot(input_vec, weight_vec);
    }
    
    let remainder_start = k_vec * 4u;
    for (var k = remainder_start; k < info.k; k = k + 1u) {
        sum = sum + input[input_offset + k] * fc1_weight[k * info.n + col];
    }
    
    sum = sum + fc1_bias[col];
    output[idx] = gelu(sum);
}