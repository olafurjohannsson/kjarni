// kjarni-transformers/src/execution/tests.rs

use super::*;
use crate::prelude::Device;

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

// ============================================================================
// HiddenState Tests
// ============================================================================

use ndarray::Array3;

#[test]
fn test_hidden_state_from_cpu() {
    let arr = Array3::<f32>::zeros((1, 10, 512));
    let hidden = HiddenState::from(arr.clone());

    assert_eq!(hidden.device(), Device::Cpu);
    assert_eq!(hidden.shape(), (1, 10, 512));
    assert!(hidden.as_cpu().is_some());
    assert!(hidden.as_gpu().is_none());
}

#[test]
fn test_hidden_state_shape() {
    let arr = Array3::<f32>::zeros((2, 5, 256));
    let hidden = HiddenState::Cpu(arr);

    let (batch, seq, dim) = hidden.shape();
    assert_eq!(batch, 2);
    assert_eq!(seq, 5);
    assert_eq!(dim, 256);
}

#[tokio::test]
async fn test_hidden_state_into_cpu_from_cpu() {
    let arr = Array3::<f32>::ones((1, 4, 128));
    let hidden = HiddenState::Cpu(arr.clone());

    let result = hidden.into_cpu().await.unwrap();

    assert_eq!(result.dim(), (1, 4, 128));
    assert_eq!(result[[0, 0, 0]], 1.0);
}

use super::*;
use crate::WgpuContext;

#[tokio::test]
async fn test_hidden_state_from_gpu() {
    let ctx = WgpuContext::new().await.unwrap();
    let arr = Array3::<f32>::zeros((1, 10, 512));
    let gpu_tensor = crate::gpu_ops::GpuTensor::from_ndarray(&ctx, &arr).unwrap();

    let hidden = HiddenState::from(gpu_tensor);

    assert_eq!(hidden.device(), Device::Wgpu);
    assert_eq!(hidden.shape(), (1, 10, 512));
    assert!(hidden.as_gpu().is_some());
    assert!(hidden.as_cpu().is_none());
}

#[tokio::test]
async fn test_hidden_state_cpu_to_gpu() {
    let ctx = std::sync::Arc::new(WgpuContext::new().await.unwrap());
    let arr = Array3::<f32>::ones((1, 4, 128));
    let hidden = HiddenState::Cpu(arr);

    let gpu_hidden = hidden.into_gpu(&ctx).unwrap();

    assert_eq!(gpu_hidden.shape(), &[1, 4, 128]);
}

#[tokio::test]
async fn test_hidden_state_gpu_to_cpu() {
    let ctx = std::sync::Arc::new(WgpuContext::new().await.unwrap());
    let arr = Array3::<f32>::ones((1, 4, 128));
    let gpu_tensor = crate::gpu_ops::GpuTensor::from_ndarray(&ctx, &arr).unwrap();
    let hidden = HiddenState::Gpu(gpu_tensor);

    let cpu_arr = hidden.into_cpu().await.unwrap();

    assert_eq!(cpu_arr.dim(), (1, 4, 128));
    assert_eq!(cpu_arr[[0, 0, 0]], 1.0);
}

#[tokio::test]
async fn test_hidden_state_roundtrip() {
    let ctx = std::sync::Arc::new(WgpuContext::new().await.unwrap());

    // Create CPU array with specific values
    let mut arr = Array3::<f32>::zeros((2, 3, 64));
    arr[[0, 0, 0]] = 1.5;
    arr[[1, 2, 63]] = -2.5;

    // CPU -> GPU
    let hidden_cpu = HiddenState::Cpu(arr.clone());
    let hidden_gpu = HiddenState::Gpu(hidden_cpu.into_gpu(&ctx).unwrap());

    // GPU -> CPU
    let result = hidden_gpu.into_cpu().await.unwrap();

    assert_eq!(result.dim(), (2, 3, 64));
    assert!((result[[0, 0, 0]] - 1.5).abs() < 1e-6);
    assert!((result[[1, 2, 63]] - (-2.5)).abs() < 1e-6);
}
