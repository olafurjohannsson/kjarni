struct FfnUniforms {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32, // keep alignment to 16 bytes
};


@group(0) @binding(0) var<uniform> info: FfnUniforms;
@group(0) @binding(1) var<storage, read> fc2_weight: array<f32>; // shape [k, n], row-major: index = i*n + j
@group(0) @binding(2) var<storage, read> fc2_bias: array<f32>;   // length n
@group(0) @binding(3) var<storage, read> input: array<f32>;      // shape [m, k], row-major: index = row*k + col
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // shape [m, n], row-major: index = row*n + col

@compute @workgroup_size(512, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // total outputs are m * n (rows * hidden_size)
    let total_outputs = info.m * info.n;
    if (idx >= total_outputs) {
        return;
    }

    // compute row and column in the output matrix [m, n]
    let row = idx / info.n;
    let col = idx % info.n;

    var sum: f32 = 0.0;

    // vectorized loop over k (the inner dimension)
    let k_vec = info.k / 4u;
    let input_offset = row * info.k; // row stride in input is k

    // For weight: layout is [k, n] row-major -> element (i, j) is at i*info.n + j
    for (var t = 0u; t < k_vec; t = t + 1u) {
        let base = t * 4u;

        let input_vec = vec4<f32>(
            input[input_offset + base + 0u],
            input[input_offset + base + 1u],
            input[input_offset + base + 2u],
            input[input_offset + base + 3u]
        );

        let weight_vec = vec4<f32>(
            fc2_weight[(base + 0u) * info.n + col],
            fc2_weight[(base + 1u) * info.n + col],
            fc2_weight[(base + 2u) * info.n + col],
            fc2_weight[(base + 3u) * info.n + col]
        );

        sum = sum + dot(input_vec, weight_vec);
    }

    // remainder
    let remainder_start = k_vec * 4u;
    for (var kk = remainder_start; kk < info.k; kk = kk + 1u) {
        sum = sum + input[input_offset + kk] * fc2_weight[kk * info.n + col];
    }

    // add bias and write output
    sum = sum + fc2_bias[col];
    // output index is row * n + col
    output[row * info.n + col] = sum;
}
