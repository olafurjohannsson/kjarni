// kjarni-transformers/src/execution/hidden_state.rs

use crate::gpu::GpuTensor;
use crate::WgpuContext;
use anyhow::Result;
use ndarray::Array3;
use std::sync::Arc;

/// Represents hidden states during pipeline execution.
/// Owned variant for internal pipeline state.
#[derive(Debug)]
pub enum HiddenState {
    Cpu(Array3<f32>),
    Gpu(GpuTensor),
}

impl HiddenState {
    /// Convert to CPU, downloading from GPU if necessary.
    pub async fn into_cpu(self) -> Result<Array3<f32>> {
        match self {
            HiddenState::Cpu(arr) => Ok(arr),
            HiddenState::Gpu(tensor) => tensor.to_ndarray_3d::<f32>().await,
        }
    }

    /// Convert to GPU, uploading from CPU if necessary.
    pub fn into_gpu(self, ctx: &Arc<WgpuContext>) -> Result<GpuTensor> {
        match self {
            HiddenState::Gpu(tensor) => Ok(tensor),
            HiddenState::Cpu(arr) => GpuTensor::from_ndarray(ctx, &arr),
        }
    }

    /// Get reference as CPU array (only if already CPU)
    pub fn as_cpu(&self) -> Option<&Array3<f32>> {
        match self {
            HiddenState::Cpu(arr) => Some(arr),
            HiddenState::Gpu(_) => None,
        }
    }

    /// Get reference as GPU tensor (only if already GPU)
    pub fn as_gpu(&self) -> Option<&GpuTensor> {
        match self {
            HiddenState::Gpu(tensor) => Some(tensor),
            HiddenState::Cpu(_) => None,
        }
    }

    /// Check which device the hidden state is on
    pub fn device(&self) -> crate::prelude::Device {
        match self {
            HiddenState::Cpu(_) => crate::prelude::Device::Cpu,
            HiddenState::Gpu(_) => crate::prelude::Device::Wgpu,
        }
    }

    /// Get the shape
    pub fn shape(&self) -> (usize, usize, usize) {
        match self {
            HiddenState::Cpu(arr) => arr.dim(),
            HiddenState::Gpu(tensor) => tensor.dims3(),
        }
    }
}

impl From<Array3<f32>> for HiddenState {
    fn from(arr: Array3<f32>) -> Self {
        HiddenState::Cpu(arr)
    }
}

impl From<GpuTensor> for HiddenState {
    fn from(tensor: GpuTensor) -> Self {
        HiddenState::Gpu(tensor)
    }
}