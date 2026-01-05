// kjarni-transformers/src/pipeline/tests.rs

use super::*;
use crate::execution::ExecutionPlan;
use crate::prelude::Device;

#[test]
fn test_execution_plan_validation() {
    // We'd need mock components to fully test this
    // For now, just test the plan logic
    
    let plan = ExecutionPlan::full_gpu();
    assert!(plan.needs_gpu());
    assert!(!plan.needs_cpu());
    
    let plan = ExecutionPlan::gpu_offload_ends();
    assert!(plan.needs_gpu());
    assert!(plan.needs_cpu());
}

#[test]
fn test_pipeline_config() {
    let config = DecoderPipelineConfig {
        num_layers: 32,
        hidden_size: 4096,
        vocab_size: 128256,
    };
    
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.vocab_size, 128256);
}