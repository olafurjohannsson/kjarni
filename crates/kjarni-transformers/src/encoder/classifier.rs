//! Concrete classification heads for encoder models.
//!
//! These components take the final hidden states from an encoder and project them
//! into task-specific outputs, such as sequence-level logits (for sentiment analysis)
//! or token-level logits (for NER).

use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::primitives::{linear::GpuLinearLayer, tanh::GpuTanh};
use crate::gpu_ops::{GpuFrameContext, GpuTensor};
use crate::linear_layer::LinearLayer;
use anyhow::{anyhow, Result};
use ndarray::{s, Array2, Array3};

// ============================================================================
//  CPU Sequence Classification Head
// ============================================================================

/// A CPU-based head for sequence classification tasks (e.g., sentiment, NLI, reranking).
pub struct CpuSequenceClassificationHead {
    pooler: Option<LinearLayer>,
    classifier: LinearLayer,
}

impl CpuSequenceClassificationHead {
    pub fn new(pooler: Option<LinearLayer>, classifier: LinearLayer) -> Result<Self> {
        if let Some(p) = &pooler {
            if p.out_features() != classifier.in_features() {
                return Err(anyhow!(
                    "Dimension mismatch: Pooler output ({}) does not match Classifier input ({})",
                    p.out_features(),
                    classifier.in_features()
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
        let (batch, seq_len, _hidden_size) = encoder_hidden_states.dim();

        if batch == 0 || seq_len == 0 {
            return Ok(Array2::<f32>::zeros((batch, self.num_classes())));
        }

        let cls_embedding = encoder_hidden_states.slice(s![.., 0, ..]).to_owned();

        let pooled_output = if let Some(p) = &self.pooler {
            let mut pooled = p.matmul(&cls_embedding.view());
            pooled.mapv_inplace(f32::tanh);
            pooled
        } else {
            cls_embedding
        };

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

        // 1. Slice the [CLS] token.
        let hidden_states_4d = encoder_hidden_states.view(vec![batch, 1, seq_len, hidden]);
        let cls_embedding_4d = pool.get(vec![batch, 1, 1, hidden]);
        self.slicer.encode(
            encoder_cmd,
            &hidden_states_4d,
            &cls_embedding_4d,
            &[0, 0, 0, 0],
        );
        let cls_embedding_2d = cls_embedding_4d.view(vec![batch, hidden]);

        // 2. Pass through the pooler layer, if it exists.
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

                // --- FIX: Apply Tanh activation in-place ---
                self.tanh.encode_inplace(encoder_cmd, &pooled_with_bias);

                pooled_with_bias
            } else {
                cls_embedding_2d
            };

        // 3. Pass through the final classifier linear layer.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_layer::LinearLayer;
    use ndarray::{array, Array1, Array2, Array3};

    fn make_linear_layer(in_features: usize, out_features: usize) -> LinearLayer {
        let mut weight = Array2::<f32>::zeros((out_features, in_features));
        // Fill diagonal-ish to make it act like identity
        for i in 0..out_features.min(in_features) {
            weight[[i, i]] = 1.0;
        }
        let bias = Array1::<f32>::zeros(out_features);
        LinearLayer::new_f32(weight, bias)
    }

    #[test]
    fn test_cpu_classification_head_no_pooler() {
        // Input: batch=2, seq_len=3, hidden_size=3
        let hidden_states = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        ];
        let classifier = make_linear_layer(3, 2); // 3 hidden -> 2 classes
        let head = CpuSequenceClassificationHead::new(None, classifier).unwrap();

        let logits = head.forward(&hidden_states).unwrap();

        // Should be [batch, num_classes] = [2,2]
        assert_eq!(logits.shape(), &[2, 2]);

        // Check first row: matmul with identity-truncated weight (2x3) and zero bias
        // Weight: [[1,0,0],[0,1,0]] -> first 2 hidden dims
        assert_eq!(logits[[0, 0]], 1.0); // corresponds to first hidden dim
        assert_eq!(logits[[0, 1]], 2.0);
        assert_eq!(logits[[1, 0]], 10.0);
        assert_eq!(logits[[1, 1]], 11.0);

        // num_classes()
        assert_eq!(head.num_classes(), 2);
    }

    #[test]
    fn test_cpu_classification_head_with_pooler() {
        // Input: batch=2, seq_len=3, hidden_size=3
        let hidden_states = array![
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        ];

        // Pooler: maps hidden_size->4
        let pooler = make_linear_layer(3, 4);

        // Classifier: maps 4->2 classes
        let classifier = make_linear_layer(4, 2);

        let head = CpuSequenceClassificationHead::new(Some(pooler), classifier).unwrap();

        let logits = head.forward(&hidden_states).unwrap();
        assert_eq!(logits.shape(), &[2, 2]);

        // Values are deterministic based on our identity-like linear layers
        // The Tanh activation is applied to pooler output
        for b in 0..2 {
            for c in 0..2 {
                assert!(logits[[b, c]].is_finite());
            }
        }

        // num_classes()
        assert_eq!(head.num_classes(), 2);
    }

    #[test]
    fn test_cpu_classification_head_pooler_dimension_mismatch() {
        let pooler = make_linear_layer(3, 5);
        let classifier = make_linear_layer(4, 2); // mismatch: pooler out 5 != classifier in 4

        let err = CpuSequenceClassificationHead::new(Some(pooler), classifier);
        assert!(err.is_err());
    }

    #[test]
    fn test_forward_empty_batch() {
        // batch=0, seq_len=3, hidden_size=3
        let hidden_states = Array3::<f32>::zeros((0, 3, 3));
        let classifier = make_linear_layer(3, 2);
        let head = CpuSequenceClassificationHead::new(None, classifier).unwrap();

        let logits = head.forward(&hidden_states).unwrap();
        assert_eq!(logits.shape(), &[0, 2]);
    }
}
