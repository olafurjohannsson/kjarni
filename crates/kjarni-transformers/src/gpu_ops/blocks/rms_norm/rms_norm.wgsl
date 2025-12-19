struct NormUniforms {
    m: u32, // number of rows (batch * seq)
    n: u32, // hidden_size (e.g. 2048)
    eps: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: NormUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Shared memory for reduction. Size 256 matches workgroup size.
var<workgroup> s_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = group_id.x;
    let tid = local_id.x;

    if (row >= uniforms.m) { return; }

    let row_offset = row * uniforms.n;

    // 1. Calculate Sum of Squares (Thread-Local)
    // Each thread processes every 256th element
    var sum_sq = 0.0;
    for (var i = tid; i < uniforms.n; i += 256u) {
        let val = input[row_offset + i];
        sum_sq += val * val;
    }

    // 2. Parallel Reduction in Shared Memory
    s_sum[tid] = sum_sq;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        workgroupBarrier();
    }

    // 3. Compute Inverse RMS (Broadcast via Shared Memory or re-read)
    // Only thread 0 computes the final scalar, but all threads need it.
    // Optimization: Store it in s_sum[0] so all threads can read it.
    if (tid == 0u) {
        let mean = s_sum[0] / f32(uniforms.n);
        s_sum[0] = 1.0 / sqrt(mean + uniforms.eps);
    }
    workgroupBarrier();
    
    let inv_rms = s_sum[0];

    // 4. Normalize and Write (Coalesced)
    // Threads write the same elements they read in step 1.
    for (var i = tid; i < uniforms.n; i += 256u) {
        let idx = row_offset + i;
        output[idx] = input[idx] * inv_rms * gamma[i];
    }
}