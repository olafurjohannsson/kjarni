@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_index: u32;


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    
    let vocab_size = arrayLength(&logits);
    
    var max_val = -3.40282347e+38; // f32 min
    var max_idx: u32 = 0u;

    let stride = 256u; // workgroup_size
    
    // reduction
    for (var i = global_id.x; i < vocab_size; i += stride) {
        let val = logits[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    if (global_id.x == 0u) {
        var final_max = -3.40282347e+38;
        var final_idx = 0u;
        
        for (var i = 0u; i < vocab_size; i++) {
            let val = logits[i];
            if (val > final_max) {
                final_max = val;
                final_idx = i;
            }
        }
        out_index = final_idx;
    }
}