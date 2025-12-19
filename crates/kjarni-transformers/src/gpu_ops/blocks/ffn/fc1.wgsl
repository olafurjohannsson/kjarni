struct FfnUniforms {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
};
// 0: GELU (default), 1: GELU_NEW, 2: RELU
@id(0) override COMPUTE_ACT_TYPE: u32 = 0;

@group(0) @binding(0) var<uniform> info: FfnUniforms;
// Weight is now transposed, shape [n, k] for coalesced memory access
@group(0) @binding(1) var<storage, read> fc1_weight: array<f32>;
@group(0) @binding(2) var<storage, read> fc1_bias: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// GELU (tanh approximation)
fn gelu_new(x: f32) -> f32 {
    let SQRT_2_OVER_PI: f32 = 0.7978845608;
    let GELU_COEFF: f32 = 0.044715;
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    return 0.5 * x * (1.0 + tanh(inner));
}

// Abramowitz and Stegun approximation (accurate to ~1.5e-7)
fn erf_approx(x: f32) -> f32 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;
    
    let sign = select(-1.0, 1.0, x >= 0.0);
    let abs_x = abs(x);
    let t = 1.0 / (1.0 + p * abs_x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp(-abs_x * abs_x);
    return sign * y;
}

fn gelu(x: f32) -> f32 {
    let SQRT_2_INV: f32 = 0.7071067811865476;  // 1/sqrt(2)
    return 0.5 * x * (1.0 + erf_approx(x * SQRT_2_INV));
}

@compute @workgroup_size(512, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_outputs = info.m * info.n;
    if (idx >= total_outputs) { return; }

    let row = idx / info.n;
    let col = idx % info.n;

    var sum: f32 = 0.0;

    let input_offset = row * info.k;
    // The weight for a given output column `col` is now a contiguous row.
    let weight_offset = col * info.k;

    let k_vec = info.k / 4u;
    for (var t = 0u; t < k_vec; t = t + 1u) {
        let base = t * 4u;
        let input_vec = vec4<f32>(
            input[input_offset + base + 0u],
            input[input_offset + base + 1u],
            input[input_offset + base + 2u],
            input[input_offset + base + 3u]
        );
        // Read a contiguous vec4 from the transposed weight matrix.
        let weight_vec = vec4<f32>(
            fc1_weight[weight_offset + base + 0u],
            fc1_weight[weight_offset + base + 1u],
            fc1_weight[weight_offset + base + 2u],
            fc1_weight[weight_offset + base + 3u]
        );
        sum = sum + dot(input_vec, weight_vec);
    }

    let remainder_start = k_vec * 4u;
    for (var kk = remainder_start; kk < info.k; kk = kk + 1u) {
        sum = sum + input[input_offset + kk] * fc1_weight[weight_offset + kk];
    }

    sum = sum + fc1_bias[col];
    //output[idx] = gelu(sum);
    // The driver will DELETE the branches that don't match the constant.
    if (COMPUTE_ACT_TYPE == 0u) {
        output[idx] = gelu(sum);
    } else if (COMPUTE_ACT_TYPE == 1u) {
        output[idx] = gelu_new(sum);
    } else if (COMPUTE_ACT_TYPE == 2u) { // Example
        output[idx] = max(0.0, sum);
    }
}