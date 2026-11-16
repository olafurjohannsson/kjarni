//! Gated DeltaNet's output norm: RMSNorm per head, scaled by its weight as
//! stored, times SiLU of the gate z.

struct Info {
    rows: u32,
    heads: u32,
    dim: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> z: array<f32>;            // [B * S, H * D]
@group(0) @binding(2) var<storage, read> weight: array<f32>;       // [D]
@group(0) @binding(3) var<storage, read_write> core: array<f32>;   // [B * S, H * D]

/// One thread per (row, head).
@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = (wg.y * grid.x + wg.x) * 32u + lid.x;
    if (idx >= info.rows * info.heads) { return; }
    let base = idx * info.dim;
    var sum = 0.0;
    for (var j = 0u; j < info.dim; j++) {
        let x = core[base + j];
        sum += x * x;
    }
    let inv = 1.0 / sqrt(sum / f32(info.dim) + info.eps);
    for (var j = 0u; j < info.dim; j++) {
        let gate = z[base + j];
        core[base + j] = core[base + j] * inv * weight[j] * (gate / (1.0 + exp(-gate)));
    }
}
