//! Gated DeltaNet's output norm: RMSNorm per head, scaled by its weight as
//! stored, times SiLU of the gate z.

struct Info {
    rows: u32,
    heads: u32,
    dim: u32,
    eps: f32,
}

const THREADS: u32 = 64u;

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> z: array<f32>;            // [B * S, H * D]
@group(0) @binding(2) var<storage, read> weight: array<f32>;       // [D]
@group(0) @binding(3) var<storage, read_write> core: array<f32>;   // [B * S, H * D]

var<workgroup> sums: array<f32, THREADS>;

/// One workgroup per (row, head); its threads share the head's values.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = wg.y * grid.x + wg.x;
    if (idx >= info.rows * info.heads) { return; }
    let base = idx * info.dim;
    var sum = 0.0;
    for (var j = lid.x; j < info.dim; j += THREADS) {
        let x = core[base + j];
        sum += x * x;
    }
    sums[lid.x] = sum;
    workgroupBarrier();
    for (var s = THREADS / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            sums[lid.x] += sums[lid.x + s];
        }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(sums[0] / f32(info.dim) + info.eps);
    for (var j = lid.x; j < info.dim; j += THREADS) {
        let gate = z[base + j];
        core[base + j] = core[base + j] * inv * weight[j] * (gate / (1.0 + exp(-gate)));
    }
}
