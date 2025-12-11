//! Classification head traits for encoder models

use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use anyhow::Result;
use ndarray::{Array2, Array3};


pub trait GpuClassificationHead {
    fn project(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor>;
}


/// CPU sequence classification head.
pub trait CpuSequenceClassifier: Send + Sync {
    /// Classify sequences.
    ///
    /// # Returns
    /// Logits `[batch_size, num_classes]`
    fn classify(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
    ) -> Result<Array2<f32>>;

    /// Number of output classes.
    fn num_classes(&self) -> usize;
}

/// GPU sequence classification head.
pub trait GpuSequenceClassifier: Send + Sync {
    /// Classify sequences.
    ///
    /// # Returns
    /// Logits `[batch_size, num_classes]` on GPU
    fn classify(
        &self,
        cmd: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
    ) -> Result<GpuTensor>;

    /// Number of output classes.
    fn num_classes(&self) -> usize;
}

/// CPU token classification head (NER, POS tagging).
pub trait CpuTokenClassifier: Send + Sync {
    /// Classify each token.
    ///
    /// # Returns
    /// Logits `[batch_size, seq_len, num_labels]`
    fn classify(
        &self,
        hidden_states: &Array3<f32>,
    ) -> Result<Array3<f32>>;

    /// Number of output labels.
    fn num_labels(&self) -> usize;
}

/// GPU token classification head.
pub trait GpuTokenClassifier: Send + Sync {
    fn classify(
        &self,
        cmd: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor>;

    fn num_labels(&self) -> usize;
}