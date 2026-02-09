
use crate::prelude::Device;
use crate::execution::ExecutionPlan;

#[test]
fn test_execution_plan_full_gpu() {
    let plan = ExecutionPlan::full_gpu();

    assert_eq!(plan.embeddings, Device::Wgpu);
    assert_eq!(plan.layers, Device::Wgpu);
    assert_eq!(plan.lm_head, Device::Wgpu);

    assert!(plan.needs_gpu());
    assert!(!plan.needs_cpu());
}

#[test]
fn test_execution_plan_full_cpu() {
    let plan = ExecutionPlan::full_cpu();

    assert_eq!(plan.embeddings, Device::Cpu);
    assert_eq!(plan.layers, Device::Cpu);
    assert_eq!(plan.lm_head, Device::Cpu);

    assert!(!plan.needs_gpu());
    assert!(plan.needs_cpu());
}

#[test]
fn test_execution_plan_gpu_offload_ends() {
    let plan = ExecutionPlan::gpu_offload_ends();

    assert_eq!(plan.embeddings, Device::Cpu);
    assert_eq!(plan.layers, Device::Wgpu);
    assert_eq!(plan.lm_head, Device::Cpu);

    assert!(plan.needs_gpu());
    assert!(plan.needs_cpu());
}

#[test]
fn test_execution_plan_gpu_offload_head() {
    let plan = ExecutionPlan::gpu_offload_head();

    assert_eq!(plan.embeddings, Device::Wgpu);
    assert_eq!(plan.layers, Device::Wgpu);
    assert_eq!(plan.lm_head, Device::Cpu);

    assert!(plan.needs_gpu());
    assert!(plan.needs_cpu());
}

#[test]
fn test_execution_plan_custom() {
    let plan = ExecutionPlan::custom(Device::Cpu, Device::Wgpu, Device::Wgpu);

    assert_eq!(plan.embeddings, Device::Cpu);
    assert_eq!(plan.layers, Device::Wgpu);
    assert_eq!(plan.lm_head, Device::Wgpu);
}

#[test]
fn test_execution_plan_default_is_full_gpu() {
    let plan = ExecutionPlan::default();
    let full_gpu = ExecutionPlan::full_gpu();

    assert_eq!(plan, full_gpu);
}

#[test]
fn test_execution_plan_equality() {
    let plan1 = ExecutionPlan::full_gpu();
    let plan2 = ExecutionPlan::full_gpu();
    let plan3 = ExecutionPlan::full_cpu();

    assert_eq!(plan1, plan2);
    assert_ne!(plan1, plan3);
}

#[test]
fn test_execution_plan_clone() {
    let plan1 = ExecutionPlan::gpu_offload_ends();
    let plan2 = plan1.clone();

    assert_eq!(plan1, plan2);
}
