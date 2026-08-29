//! Concrete classification heads for encoder models.

use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::primitives::{linear::GpuLinearLayer, tanh::GpuTanh};
use crate::gpu::{GpuFrameContext, GpuTensor};
use crate::linear_layer::LinearLayer;
use crate::models::base::ModelLoadConfig;
use crate::weights::ModelWeights;
use crate::{PoolingStrategy, last_token_pool};
use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, s};


/// A GPU-accelerated head for sequence classification tasks.
pub struct GpuSequenceClassificationHead {
    // Kernels for operations
    slicer: GpuSlice,
    linear: GpuLinearLayer,
    add_bias: GpuAdd,
    tanh: GpuTanh,
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
            return Err(anyhow!(
                "GPU Pooler weight and bias must both be Some or None."
            ));
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

        // Slice the CLS token
        let hidden_states_4d = encoder_hidden_states.view(vec![batch, 1, seq_len, hidden]);
        let cls_embedding_4d = pool.get(vec![batch, 1, 1, hidden]);
        self.slicer.encode(
            encoder_cmd,
            &hidden_states_4d,
            &cls_embedding_4d,
            &[0, 0, 0, 0],
        );
        let cls_embedding_2d = cls_embedding_4d.view(vec![batch, hidden]);

        // pooler layer, if it exists.
        let pooled_output =
            if let (Some(weight), Some(bias)) = (&self.pooler_weight, &self.pooler_bias) {
                let pooled_linear_out = pool.get(vec![batch, weight.shape()[0]]);
                self.linear
                    .encode(encoder_cmd, &cls_embedding_2d, weight, &pooled_linear_out);

                let pooled_with_bias = pool.get(pooled_linear_out.shape().to_vec());
                self.add_bias.encode_broadcast_row(
                    encoder_cmd,
                    &pooled_linear_out,
                    bias,
                    &pooled_with_bias,
                );

                self.tanh.encode_inplace(encoder_cmd, &pooled_with_bias);

                pooled_with_bias
            } else {
                cls_embedding_2d
            };

        // final classifier linear layer.
        let logits_linear_out = pool.get(vec![batch, self.classifier_weight.shape()[0]]);
        self.linear.encode(
            encoder_cmd,
            &pooled_output,
            &self.classifier_weight,
            &logits_linear_out,
        );

        let final_logits = pool.get(logits_linear_out.shape().to_vec());
        self.add_bias.encode_broadcast_row(
            encoder_cmd,
            &logits_linear_out,
            &self.classifier_bias,
            &final_logits,
        );

        Ok(final_logits)
    }

    pub fn num_classes(&self) -> usize {
        self.classifier_bias.shape()[0]
    }
}
