//! Concrete classification heads for encoder models.
//!
//! These components take the final hidden states from an encoder and project them
//! into task-specific outputs, such as sequence-level logits (for sentiment analysis)
//! or token-level logits (for NER).

use crate::{PoolingStrategy, last_token_pool};
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::primitives::{linear::GpuLinearLayer, tanh::GpuTanh};
use crate::gpu_ops::{GpuFrameContext, GpuTensor};
use crate::linear_layer::LinearLayer;
use crate::models::base::ModelLoadConfig;
use crate::weights::ModelWeights;
use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, s};

// ============================================================================
//  CPU Sequence Classification Head
// ============================================================================

/// Layout for sequence classification head.
#[derive(Debug, Clone)]
pub struct ClassificationHeadLayout {
    /// Pre-classifier projection (hidden_size -> hidden_size)
    pub pre_classifier_weight: Option<String>,
    pub pre_classifier_bias: Option<String>,

    /// Final classifier (hidden_size -> num_labels)
    pub classifier_weight: String,
    pub classifier_bias: Option<String>,
}

/// Activation function for the classification head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeadActivation {
    #[default]
    Tanh,
    Relu,
    Gelu,
    None,
}

/// CPU classification head supporting multiple architectures.
///
/// Supports:
/// - BERT/RoBERTa: pooler (Tanh) → classifier  
/// - DistilBERT: pre_classifier (ReLU) → classifier
/// - Simple: classifier only
pub struct CpuSequenceClassificationHead {
    /// Pooler layer (BERT-style, uses Tanh)
    pooler: Option<LinearLayer>,
    /// Pre-classifier layer (DistilBERT-style)
    pre_classifier: Option<LinearLayer>,
    /// Final classifier
    classifier: LinearLayer,
    /// Activation after pooler/pre_classifier
    activation: HeadActivation,
    pooling_strategy: PoolingStrategy,
    /// Label names (optional)
    labels: Option<Vec<String>>,
}

impl CpuSequenceClassificationHead {
    /// Create a new classification head.
    ///
    /// # Arguments
    /// * `pooler` - Optional pooler layer (BERT-style)
    /// * `classifier` - Final classification layer
    pub fn new(pooler: Option<LinearLayer>, classifier: LinearLayer) -> Result<Self> {
        Self::with_config(
            pooler,
            None,
            classifier,
            HeadActivation::Tanh,
            PoolingStrategy::Cls,
            None,
        )
    }

    /// Create with full configuration.
    pub fn with_config(
        pooler: Option<LinearLayer>,
        pre_classifier: Option<LinearLayer>,
        classifier: LinearLayer,
        activation: HeadActivation,
        pooling_strategy: PoolingStrategy,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        // Validate dimensions
        if let Some(ref p) = pooler {
            if p.out_features() != classifier.in_features() {
                return Err(anyhow!(
                    "Pooler output ({}) != Classifier input ({})",
                    p.out_features(),
                    classifier.in_features()
                ));
            }
        }
        if let Some(ref pc) = pre_classifier {
            if pc.out_features() != classifier.in_features() {
                return Err(anyhow!(
                    "Pre-classifier output ({}) != Classifier input ({})",
                    pc.out_features(),
                    classifier.in_features()
                ));
            }
        }

        Ok(Self {
            pooler,
            pre_classifier,
            classifier,
            activation,
            pooling_strategy,
            labels,
        })
    }

    /// Create from model weights with auto-detection.
    pub fn from_weights(
        weights: &ModelWeights,
        load_config: &ModelLoadConfig,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        let pooling_strategy = if weights.config_json.contains("\"model_type\": \"bart\"") {
            // BART for classification uses the EOS token, which is the last token.
            PoolingStrategy::LastToken
        } else {
            // BERT, RoBERTa, DistilBERT, etc., all default to using the CLS token.
            PoolingStrategy::Cls
        };
        println!("Pooling {:?}", pooling_strategy);

        // Case 1: BART-style head (for Zero-Shot NLI)
        // Checks for "classification_head.dense.weight"
        if weights.contains("classification_head.dense.weight") {
            let pre_classifier = LinearLayer::builder(weights, "classification_head.dense.weight")
                .with_optional_bias(Some("classification_head.dense.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            let classifier = LinearLayer::builder(weights, "classification_head.out_proj.weight")
                .with_optional_bias(Some("classification_head.out_proj.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            // BART's head uses a Tanh activation between layers
            return Self::with_config(
                None,
                Some(pre_classifier),
                classifier,
                HeadActivation::Tanh,
                pooling_strategy,
                labels,
            );
        }

        // Case 2: RoBERTa-style sequence classification head
        // Checks for "classifier.dense.weight"
        if weights.contains("classifier.dense.weight") {
            let pre_classifier = LinearLayer::builder(weights, "classifier.dense.weight")
                .with_optional_bias(Some("classifier.dense.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            let classifier = LinearLayer::builder(weights, "classifier.out_proj.weight")
                .with_optional_bias(Some("classifier.out_proj.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            return Self::with_config(
                None,
                Some(pre_classifier),
                classifier,
                HeadActivation::Tanh,
                pooling_strategy,
                labels,
            );
        }

        // Case 3: DistilBERT-style head
        // Checks for "pre_classifier.weight"
        if weights.contains("pre_classifier.weight") {
            let pre_classifier = LinearLayer::builder(weights, "pre_classifier.weight")
                .with_optional_bias(Some("pre_classifier.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            // The final layer is named "classifier.weight"
            let classifier = LinearLayer::builder(weights, "classifier.weight")
                .with_optional_bias(Some("classifier.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            return Self::with_config(
                None,
                Some(pre_classifier),
                classifier,
                HeadActivation::Relu,
                pooling_strategy,
                labels,
            );
        }

        // Case 4: Standard BERT-style head (with a pooler)
        // Checks for "bert.pooler.dense.weight"
        if weights.contains("bert.pooler.dense.weight") {
            let pooler = LinearLayer::builder(weights, "bert.pooler.dense.weight")
                .with_optional_bias(Some("bert.pooler.dense.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            let classifier = LinearLayer::builder(weights, "classifier.weight")
                .with_optional_bias(Some("classifier.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            return Self::with_config(
                Some(pooler),
                None,
                classifier,
                HeadActivation::Tanh,
                pooling_strategy,
                labels,
            );
        }

        // Fallback Case: A simple, single-layer head named "classifier.weight"
        if weights.contains("classifier.weight") {
            let classifier = LinearLayer::builder(weights, "classifier.weight")
                .with_optional_bias(Some("classifier.bias"))
                .with_target_dtype(load_config.target_dtype)
                .build()?;
            return Self::with_config(
                None,
                None,
                classifier,
                HeadActivation::None,
                pooling_strategy,
                labels,
            );
        }

        // If none of the known structures are found, return an error.
        Err(anyhow!(
            "Could not auto-detect a valid classification head structure from the model weights. Checked for 'classification_head.dense', 'classifier.dense', 'pre_classifier', 'bert.pooler', and 'classifier' weights."
        ))
    }

    /// Forward pass: hidden_states → logits.
    ///
    /// # Arguments
    /// * `encoder_hidden_states`: Shape `[batch, seq_len, hidden_size]`
    ///
    /// # Returns
    /// * Logits with shape `[batch, num_classes]`
    pub fn forward(&self, encoder_hidden_states: &Array3<f32>, attention_mask: Option<&Array2<f32>>,) -> Result<Array2<f32>> {
        let (batch, seq_len, _hidden_size) = encoder_hidden_states.dim();
        if batch == 0 || seq_len == 0 {
            return Ok(Array2::<f32>::zeros((batch, self.num_classes())));
        }

        // Extract [CLS] token (index 0)
        // let cls_embedding = encoder_hidden_states.slice(s![.., 0, ..]).to_owned();

        let sequence_embedding = match self.pooling_strategy {
            PoolingStrategy::Cls => {
                // Extract [CLS] token (index 0)
                encoder_hidden_states.slice(s![.., 0, ..]).to_owned()
            }
            PoolingStrategy::LastToken => {
                crate::utils::last_token_pool(encoder_hidden_states, attention_mask.unwrap())
            }
            // You can add Mean/Max later if needed.
            _ => return Err(anyhow!("Unsupported pooling strategy for classification head: {:?}", self.pooling_strategy)),
        };

        // Apply pooler OR pre_classifier (not both)
        let features = if let Some(ref pooler) = self.pooler {
            let mut pooled = pooler.matmul(&sequence_embedding.view());
            self.apply_activation_inplace(&mut pooled);
            pooled
        } else if let Some(ref pre_classifier) = self.pre_classifier {
            let mut pre_out = pre_classifier.matmul(&sequence_embedding.view());
            self.apply_activation_inplace(&mut pre_out);
            pre_out
        } else {
            sequence_embedding
        };

        // Final classifier
        let logits = self.classifier.matmul(&features.view());
        Ok(logits)
    }

    fn apply_activation_inplace(&self, x: &mut Array2<f32>) {
        match self.activation {
            HeadActivation::Tanh => x.mapv_inplace(f32::tanh),
            HeadActivation::Relu => x.mapv_inplace(|v| v.max(0.0)),
            HeadActivation::Gelu => x.mapv_inplace(|v| {
                0.5 * v
                    * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3))).tanh())
            }),
            HeadActivation::None => {}
        }
    }

    pub fn num_classes(&self) -> usize {
        self.classifier.out_features()
    }

    pub fn labels(&self) -> Option<&[String]> {
        self.labels.as_deref()
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
    use ndarray::{Array1, Array2, Array3, array};

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

        let logits = head.forward(&hidden_states, None).unwrap();

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

        let logits = head.forward(&hidden_states, None).unwrap();
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

        let logits = head.forward(&hidden_states, None).unwrap();
        assert_eq!(logits.shape(), &[0, 2]);
    }
}
