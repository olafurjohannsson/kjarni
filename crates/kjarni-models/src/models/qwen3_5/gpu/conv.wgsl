//! Gated DeltaNet's causal depthwise convolution with SiLU, continuing from
//! the inputs held in the state and leaving the newest ones there.

struct Info {
    batch: u32,
    seq: u32,
    channels: u32,
    kernel: u32,
    /// Also keep the window as it stood after this many inputs; 0 for none.
    capture: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// The widest kernel the private window holds.
const MAX_KERNEL: u32 = 8u;

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> x: array<f32>;              // [B * S, C]
@group(0) @binding(2) var<storage, read> weight: array<f32>;         // [C, kernel]
@group(0) @binding(3) var<storage, read_write> state: array<f32>;    // [B, C, kernel - 1]
@group(0) @binding(4) var<storage, read_write> captured: array<f32>; // [B, C, kernel - 1]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;      // [B * S, C]

/// One thread per (batch, channel), walking the tokens in order.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = (wg.y * grid.x + wg.x) * 64u + lid.x;
    if (idx >= info.batch * info.channels) { return; }
    let batch = idx / info.channels;
    let c = idx % info.channels;
    let window = info.kernel - 1u;
    let held_base = (batch * info.channels + c) * window;

    var held: array<f32, MAX_KERNEL>;
    for (var j = 0u; j < window; j++) {
        held[j] = state[held_base + j];
    }
    for (var t = 0u; t < info.seq; t++) {
        let r = batch * info.seq + t;
        let input = x[r * info.channels + c];
        // Oldest first, as the CPU sums them.
        var acc = 0.0;
        for (var j = 0u; j < window; j++) {
            acc += weight[c * info.kernel + j] * held[j];
        }
        acc += weight[c * info.kernel + window] * input;
        out[r * info.channels + c] = acc / (1.0 + exp(-acc));
        for (var j = 1u; j < window; j++) {
            held[j - 1u] = held[j];
        }
        if (window > 0u) {
            held[window - 1u] = input;
        }
        if (t + 1u == info.capture) {
            for (var j = 0u; j < window; j++) {
                captured[held_base + j] = held[j];
            }
        }
    }
    for (var j = 0u; j < window; j++) {
        state[held_base + j] = held[j];
    }
}
