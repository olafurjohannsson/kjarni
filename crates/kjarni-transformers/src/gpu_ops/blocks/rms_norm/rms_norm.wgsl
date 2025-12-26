// rms_norm.wgsl - RMSNorm with BF16 gamma support

struct NormUniforms {
    m: u32,
    n: u32,
    eps: f32,
    is_bf16: u32,  // 0 = F32 gamma, 1 = BF16 gamma
}

@group(0) @binding(0) var<uniform> uniforms: NormUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<u32>;  // Can be F32 or packed BF16
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> s_sum: array<f32, 256>;

fn get_gamma(index: u32) -> f32 {
    if (uniforms.is_bf16 == 1u) {
        // BF16: 2 values packed per u32
        let packed = gamma[index / 2u];
        if (index % 2u == 0u) {
            return bitcast<f32>(packed << 16u);
        } else {
            return bitcast<f32>(packed & 0xFFFF0000u);
        }
    } else {
        // F32: direct read (reinterpret u32 as f32)
        return bitcast<f32>(gamma[index]);
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = group_id.x;
    let tid = local_id.x;

    if (row >= uniforms.m) { return; }

    let row_offset = row * uniforms.n;

    // 1. Calculate Sum of Squares
    var sum_sq = 0.0;
    for (var i = tid; i < uniforms.n; i += 256u) {
        let val = input[row_offset + i];
        sum_sq += val * val;
    }

    // 2. Parallel Reduction
    s_sum[tid] = sum_sq;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        workgroupBarrier();
    }

    // 3. Compute Inverse RMS
    if (tid == 0u) {
        let mean = s_sum[0] / f32(uniforms.n);
        s_sum[0] = 1.0 / sqrt(mean + uniforms.eps);
    }
    workgroupBarrier();
    
    let inv_rms = s_sum[0];

    // 4. Normalize and Scale
    for (var i = tid; i < uniforms.n; i += 256u) {
        let idx = row_offset + i;
        output[idx] = input[idx] * inv_rms * get_gamma(i);
    }
}