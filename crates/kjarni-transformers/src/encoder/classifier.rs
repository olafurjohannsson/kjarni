//! Concrete classification heads for encoder models.
//!
//! These components take the final hidden states from an encoder and project them
//! into task-specific outputs, such as sequence-level logits (for sentiment analysis)
//! or token-level logits (for NER).

use anyhow::{anyhow, Result};
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::primitives::{
    linear::GpuLinearLayer,
    tanh::GpuTanh,
};
use crate::gpu_ops::{GpuFrameContext, GpuTensor, Kernel};
use crate::linear_layer_old::LinearLayer;
use ndarray::{Array1, Array2, Array3, s};

// ============================================================================
//  CPU Sequence Classification Head
// ============================================================================

/// A CPU-based head for sequence classification tasks (e.g., sentiment, NLI, reranking).
pub struct CpuSequenceClassificationHead {
    // --- REFACTOR: Use LinearLayer for both components ---
    pooler: Option<LinearLayer>,
    classifier: LinearLayer,
}

impl CpuSequenceClassificationHead {
    /// Creates a new classification head from LinearLayer components.
    pub fn new(pooler: Option<LinearLayer>, classifier: LinearLayer) -> Result<Self> {
        // You could add validation here, e.g., check that pooler output dim matches classifier input dim
        if let Some(p) = &pooler {
            if p.out_features() != classifier.in_features() {
                return Err(anyhow!(
                    "Dimension mismatch: Pooler output ({}) does not match Classifier input ({})",
                    p.out_features(), classifier.in_features()
                ));
            }
        }
        Ok(Self { pooler, classifier })
    }

    /// Takes the full sequence of hidden states and produces final logits.
    ///
    /// # Arguments
    /// * `encoder_hidden_states`: Shape `[batch, seq_len, hidden_size]`.
    ///
    /// # Returns
    /// * Logits with shape `[batch, num_classes]`.
    pub fn forward(&self, encoder_hidden_states: &Array3<f32>) -> Result<Array2<f32>> {
        // 1. Extract the [CLS] token's hidden state. Shape: [batch, hidden_size]
        let cls_embedding = encoder_hidden_states.slice(s![.., 0, ..]).to_owned();

        // 2. Pass through the pooler layer, if it exists.
        let pooled_output = if let Some(p) = &self.pooler {
            // The matmul method handles the dot product and bias addition.
            let mut pooled = p.matmul(&cls_embedding.view());
            // The pooler has a Tanh activation which is separate from the linear layer itself.
            pooled.mapv_inplace(f32::tanh);
            pooled
        } else {
            cls_embedding
        };

        // 3. Pass through the final classifier. Its matmul method handles the final dot+bias.
        let logits = self.classifier.matmul(&pooled_output.view());

        Ok(logits)
    }

    pub fn num_classes(&self) -> usize {
        self.classifier.out_features()
    }
}

// ============================================================================
//  GPU Sequence Classification Head
// ============================================================================

/// A GPU-accelerated head for sequence classification tasks.
///
/// Mirrors the logic of `CpuSequenceClassificationHead` using GPU kernels.
/// A GPU-accelerated head for sequence classification tasks.
pub struct GpuSequenceClassificationHead {
    // Kernels for operations
    slicer: GpuSlice,
    linear: GpuLinearLayer,
    add_bias: GpuAdd,
    tanh: GpuTanh,
    // For many models, omitting it from the pooler might have a minor impact.

    // Weights on GPU
    pooler_weight: Option<GpuTensor>,
    pooler_bias: Option<GpuTensor>,
    classifier_weight: GpuTensor,
    classifier_bias: GpuTensor,
}

impl GpuSequenceClassificationHead {
    pub fn new(
        context: &std::sync::Arc<crate::WgpuContext>,
        pooler_weight: Option<GpuTensor>,
        pooler_bias: Option<GpuTensor>,
        classifier_weight: GpuTensor,
        classifier_bias: GpuTensor,
    ) -> Result<Self> {
        if pooler_weight.is_some() != pooler_bias.is_some() {
            return Err(anyhow!("GPU Pooler weight and bias must both be Some or None."));
        }
        Ok(Self {
            slicer: GpuSlice::new(context),
            linear: GpuLinearLayer::new(context),
            add_bias: GpuAdd::new(context),
            tanh: GpuTanh::new(context),
            pooler_weight,
            pooler_bias,
            classifier_weight,
            classifier_bias,
        })
    }

     pub fn forward(
        &self,
        frame: &mut GpuFrameContext,
        encoder_hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        let (batch, seq_len, hidden) = encoder_hidden_states.dims3();
        let (encoder_cmd, pool) = frame.resources();

        // 1. Slice the [CLS] token.
        let hidden_states_4d = encoder_hidden_states.view(vec![batch, 1, seq_len, hidden]);
        let cls_embedding_4d = pool.get(vec![batch, 1, 1, hidden]);
        self.slicer.encode(encoder_cmd, &hidden_states_4d, &cls_embedding_4d, &[0, 0, 0, 0]);
        let cls_embedding_2d = cls_embedding_4d.view(vec![batch, hidden]);

        // 2. Pass through the pooler layer, if it exists.
        let pooled_output = if let (Some(weight), Some(bias)) = (&self.pooler_weight, &self.pooler_bias) {
            let pooled_linear_out = pool.get(vec![batch, weight.shape()[0]]);
            self.linear.encode(encoder_cmd, &cls_embedding_2d, weight, &pooled_linear_out);

            let pooled_with_bias = pool.get(pooled_linear_out.shape().to_vec());
            self.add_bias.encode_broadcast_row(encoder_cmd, &pooled_linear_out, bias, &pooled_with_bias);
            
            // --- FIX: Apply Tanh activation in-place ---
            self.tanh.encode_inplace(encoder_cmd, &pooled_with_bias);
            
            pooled_with_bias
        } else {
            cls_embedding_2d
        };

        // 3. Pass through the final classifier linear layer.
        let logits_linear_out = pool.get(vec![batch, self.classifier_weight.shape()[0]]);
        self.linear.encode(encoder_cmd, &pooled_output, &self.classifier_weight, &logits_linear_out);

        let final_logits = pool.get(logits_linear_out.shape().to_vec());
        self.add_bias.encode_broadcast_row(encoder_cmd, &logits_linear_out, &self.classifier_bias, &final_logits);

        Ok(final_logits)
    }

    pub fn num_classes(&self) -> usize {
        self.classifier_bias.shape()[0]
    }
}