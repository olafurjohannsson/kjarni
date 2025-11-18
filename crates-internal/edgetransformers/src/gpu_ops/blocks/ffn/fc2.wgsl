struct FfnUniforms {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32, // keep alignment to 16 bytes
};

@group(0) @binding(0) var<uniform> info: FfnUniforms;
// --- MODIFICATION ---
// Weight is now transposed, with shape [n, k]
@group(0) @binding(1) var<storage, read> fc2_weight: array<f32>;
@group(0) @binding(2) var<storage, read> fc2_bias: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let total_outputs = info.m * info.n;
    if (idx >= total_outputs) {
        return;
    }

    let row = idx / info.n;
    let col = idx % info.n;

    var sum: f32 = 0.0;

    let input_offset = row * info.k;
    // --- MODIFICATION ---
    // The weight for a given output column `col` is now a contiguous row in the transposed matrix.
    // The offset to this row is `col * k`.
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

        // --- MODIFICATION ---
        // Read a contiguous vec4 from the transposed weight matrix. This is much faster.
        let weight_vec = vec4<f32>(
            fc2_weight[weight_offset + base + 0u],
            fc2_weight[weight_offset + base + 1u],
            fc2_weight[weight_offset + base + 2u],
            fc2_weight[weight_offset + base + 3u]
        );

        sum = fma(input_vec.x, weight_vec.x, sum);
        sum = fma(input_vec.y, weight_vec.y, sum);
        sum = fma(input_vec.z, weight_vec.z, sum);
        sum = fma(input_vec.w, weight_vec.w, sum);
    }

    // Remainder loop
    let remainder_start = k_vec * 4u;
    for (var kk = remainder_start; kk < info.k; kk = kk + 1u) {
        sum = fma(input[input_offset + kk], fc2_weight[weight_offset + kk], sum);
    }

    sum = sum + fc2_bias[col];
    output[row * info.n + col] = sum;
}