struct MatmulInfo {
    b: u32,
    m: u32,
    k: u32,
    n: u32,
    a_stride_batch: u32,
    b_stride_batch: u32,
    b_stride_row: u32,
    b_stride_col: u32,
}

@group(0) @binding(0) var<uniform> info: MatmulInfo;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;
@group(0) @binding(2) var<storage, read> b_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>;

var<workgroup> a_tile: array<array<f32, 32>, 32>;
var<workgroup> b_tile: array<array<f32, 32>, 32>;

@compute @workgroup_size(32, 32, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let b_idx = group_id.z;

    // Use strides from uniforms to calculate the base offset for this batch item.
    // If b_stride_batch is 0, b_batch_offset will always be 0 (broadcasting).
    let a_batch_offset = b_idx * info.a_stride_batch;
    let b_batch_offset = b_idx * info.b_stride_batch;
    let c_batch_offset = b_idx * info.m * info.n;

    let global_row = group_id.y * 32u + local_id.y;
    let global_col = group_id.x * 32u + local_id.x;

    var acc = 0.0;
    let num_tiles = (info.k + 31u) / 32u;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * 32u + local_id.x;
        let b_row = t * 32u + local_id.y;

        // Load tile for A from its batch
        if (global_row < info.m && a_col < info.k) {
            a_tile[local_id.y][local_id.x] = a_in[a_batch_offset + global_row * info.k + a_col];
        } else {
            a_tile[local_id.y][local_id.x] = 0.0;
        }

        // Load tile for B from its (potentially broadcasted) batch
        // if (b_row < info.k && global_col < info.n) {
        //     b_tile[local_id.y][local_id.x] = b_in[b_batch_offset + b_row * info.n + global_col];
        // } else {
        //     b_tile[local_id.y][local_id.x] = 0.0;
        // }
        if (b_row < info.k && global_col < info.n) {
            let b_idx = b_batch_offset + 
                        b_row * info.b_stride_row +     // Row stride
                        global_col * info.b_stride_col;  // Column stride
            b_tile[local_id.y][local_id.x] = b_in[b_idx];
        } else {
            b_tile[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        // todo: unroll loop
         for (var i = 0u; i < 32u; i = i + 1u) {
            acc = acc + a_tile[local_id.y][i] * b_tile[i][local_id.x];
        }
        
        workgroupBarrier();
    }
    
    if (global_row < info.m && global_col < info.n) {
        c_out[c_batch_offset + global_row * info.n + global_col] = acc;
    }
}

/// The same multiply for few rows and a short inner dimension: one thread per output,
/// reading both operands directly. See FEW_ROWS.
@compute @workgroup_size(32)
fn main_few_rows(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = (wg.y * grid.x + wg.x) * 32u + lid.x;
    if (idx >= info.b * info.m * info.n) { return; }

    let col = idx % info.n;
    let row = (idx / info.n) % info.m;
    let batch = idx / (info.n * info.m);

    let a_base = batch * info.a_stride_batch + row * info.k;
    let b_base = batch * info.b_stride_batch + col * info.b_stride_col;

    var acc = 0.0;
    for (var k = 0u; k < info.k; k++) {
        acc += a_in[a_base + k] * b_in[b_base + k * info.b_stride_row];
    }
    c_out[batch * info.m * info.n + row * info.n + col] = acc;
}
