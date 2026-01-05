@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_index: u32;

// Shared memory for reduction. 
// Assuming Max Vocab ~32k-128k. 
// A single workgroup cannot hold all this.
// We will use a simple serialized loop for the single batch item for absolute robustness 
// and zero synchronization bugs, as typically VocabSize is not large enough 
// to justify complex multi-stage reduction overhead for a single result.

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // NOTE: This shader implementation assumes BatchSize = 1 for simplicity
    // and scans the entire array.
    
    let vocab_size = arrayLength(&logits);
    
    // Each thread finds the max in its stride
    var max_val = -3.40282347e+38; // f32 min
    var max_idx: u32 = 0u;

    let stride = 256u; // workgroup_size
    
    // 1. Thread-local reduction
    for (var i = global_id.x; i < vocab_size; i += stride) {
        let val = logits[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // 2. Workgroup reduction using shared memory
    // (Simplification: For stability on all hardware, we utilize a persistent atomic-like 
    // approach or just a single thread if perf allows. 
    // But below is a robust "Leader Selection" pattern).
    
    // Store results in shared memory? 
    // Actually, for 32k items, a single thread loop is extremely fast on GPU 
    // compared to PCIe overhead.
    
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