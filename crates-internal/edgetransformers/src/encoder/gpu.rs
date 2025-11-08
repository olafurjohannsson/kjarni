use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3, s};
use std::sync::Arc;

use crate::gpu_ops::GpuTensor;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::encoder::GpuEncoderLayer; // The new layer we just built
use crate::gpu_ops::blocks::attention::{GpuAttentionWeights, TempStorage};
use crate::gpu_ops::blocks::ffn::GpuFeedForwardWeights;
use crate::gpu_ops::blocks::layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
use crate::traits::{Device, Encoder, EncoderArchitecture, EncoderOutput, TransformerModel};
use crate::weights::ModelWeights;

/// The GPU backend for a generic Transformer Encoder.
/// It holds the GPU-native weights and orchestrates the layer-by-layer forward pass.
pub struct GpuTransformerEncoder {
    // CPU-side embeddings for the initial lookup.
    word_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,
    token_type_embeddings: Array2<f32>,

    // GPU-side components
    embedding_layer_norm: GpuLayerNorm,
    embedding_ln_weights: GpuLayerNormWeights,
    layers: Vec<GpuEncoderLayer>,
    
    // Configuration and context
    config: Arc<dyn EncoderArchitecture + Send + Sync>,
    context: Arc<WgpuContext>,
}

impl GpuTransformerEncoder {
    /// Constructs a new `GpuTransformerEncoder`.
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn EncoderArchitecture + Send + Sync>,
        context: Arc<WgpuContext>,
    ) -> Result<Self> {
        // --- Load CPU embeddings ---
        let (word_w, pos_w, type_w) = config.get_embedding_weight_names();
        let word_embeddings = weights.get_array2(word_w)?;
        let position_embeddings = weights.get_array2(pos_w)?;
        let token_type_embeddings = if let Some(name) = type_w {
            weights.get_array2(name)?
        } else {
            Array2::zeros((0, 0)) // Placeholder for models without token types
        };

        // --- Load and create embedding LayerNorm components ---
        let (norm_w_name, norm_b_name) = config.get_embedding_layer_norm_names();
        let embed_ln_gamma_cpu = weights.get_array1(norm_w_name)?;
        let embed_ln_beta_cpu = weights.get_array1(norm_b_name)?;

        let embedding_ln_weights = GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(&context, &embed_ln_gamma_cpu)?,
            GpuTensor::from_ndarray(&context, &embed_ln_beta_cpu)?,
        )?;
        let embedding_layer_norm = GpuLayerNorm::new(&context, config.layer_norm_eps());

        // --- Load weights and create layers ---
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i);
            let ffn_names = config.get_feed_forward_names(i);
            
            // Note: For many encoders (like BERT), Q/K/V are not combined in one tensor.
            // This logic assumes separate weights are provided by the config.
            let transpose_attn = config.transpose_attention_weights();

            let prep_attn_w = |name: &str| -> Result<Array2<f32>> {
                let raw = weights.get_array2(name)?;
                if transpose_attn {
                    Ok(raw) // Assume raw is [in, out]
                } else {
                    Ok(raw.t().as_standard_layout().to_owned()) // Raw is [out, in], so transpose
                }
            };

            let self_attn_weights = GpuAttentionWeights::new(
                GpuTensor::from_ndarray(&context, &prep_attn_w(&attn_names.q_weight)?)?,
                GpuTensor::from_ndarray(&context, &weights.get_array1(&attn_names.q_bias)?)?,
                GpuTensor::from_ndarray(&context, &prep_attn_w(&attn_names.k_weight)?)?,
                GpuTensor::from_ndarray(&context, &weights.get_array1(&attn_names.k_bias)?)?,
                GpuTensor::from_ndarray(&context, &prep_attn_w(&attn_names.v_weight)?)?,
                GpuTensor::from_ndarray(&context, &weights.get_array1(&attn_names.v_bias)?)?,
                GpuTensor::from_ndarray(&context, &prep_attn_w(&attn_names.output_weight)?)?,
                GpuTensor::from_ndarray(&context, &weights.get_array1(&attn_names.output_bias)?)?,
            )?;
            
            let self_attn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_ndarray(&context, &weights.get_array1(&attn_names.norm_weight)?)?,
                GpuTensor::from_ndarray(&context, &weights.get_array1(&attn_names.norm_bias)?)?,
            )?;
            
            let intermediate_w = weights.get_array2(&ffn_names.intermediate_weight)?;
            let fc1_w_cpu = if config.transpose_ffn_weights() { intermediate_w.t().as_standard_layout().to_owned() } else { intermediate_w };
            let output_w = weights.get_array2(&ffn_names.output_weight)?;
            let fc2_w_cpu = if config.transpose_ffn_weights() { output_w.t().as_standard_layout().to_owned() } else { output_w };
            
            let ff_weights = GpuFeedForwardWeights::new(
                GpuTensor::from_ndarray(&context, &fc1_w_cpu)?,
                GpuTensor::from_ndarray(&context, &weights.get_array1(&ffn_names.intermediate_bias)?)?,
                GpuTensor::from_ndarray(&context, &fc2_w_cpu)?,
                GpuTensor::from_ndarray(&context, &weights.get_array1(&ffn_names.output_bias)?)?,
            )?;

            let ffn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_ndarray(&context, &weights.get_array1(&ffn_names.norm_weight)?)?,
                GpuTensor::from_ndarray(&context, &weights.get_array1(&ffn_names.norm_bias)?)?,
            )?;
            
            layers.push(GpuEncoderLayer::new(
                &context,
                self_attn_weights,
                self_attn_ln_weights,
                ff_weights,
                ffn_ln_weights,
                config.as_ref(),
            )?);
        }

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embedding_layer_norm,
            embedding_ln_weights,
            layers,
            config,
            context,
        })
    }
    pub fn config(&self) -> &Arc<dyn EncoderArchitecture + Send + Sync> {
        &self.config
    }
    fn perform_cpu_embedding(
        &self,
        input_ids: &Array2<f32>,
        token_type_ids: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        // This is a simplified version of the CPU embedding logic from your old encoder.
        let cpu_embedding_layer = crate::Embeddings::new(
            self.word_embeddings.clone(),
            self.position_embeddings.clone(),
            Some(self.token_type_embeddings.clone()),
        );
        Ok(cpu_embedding_layer.forward(input_ids, token_type_ids))
    }
}

impl TransformerModel for GpuTransformerEncoder {
    fn device(&self) -> Device { Device::Wgpu }
    fn context(&self) -> Option<Arc<WgpuContext>> { Some(self.context.clone()) }
}

#[async_trait]
impl Encoder for GpuTransformerEncoder {
    type Input = Array2<f32>;
    type Output = EncoderOutput;

    async fn forward(
        &self,
        input_ids: &Self::Input,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<f32>>,
    ) -> Result<Self::Output> {
        // --- 1. Setup ---
        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Encoder Forward") });
        let mut temp = TempStorage::new(self.context.clone());

        // --- 2. Embeddings (CPU) & Upload ---
        let initial_embeddings_cpu = self.perform_cpu_embedding(input_ids, token_type_ids)?;
        let mut hidden_states = GpuTensor::from_ndarray(&self.context, &initial_embeddings_cpu)?;
        let attention_mask_gpu = GpuTensor::from_ndarray(&self.context, attention_mask)?;

        // --- 3. Initial Layer Normalization (for Post-Norm models like BERT) ---
        if !self.config.is_prenorm() {
            let ln_output = temp.get(hidden_states.shape().to_vec());
            self.embedding_layer_norm.encode(&mut encoder, &self.embedding_ln_weights, &hidden_states, &ln_output);
            hidden_states = ln_output;
        }

        // --- 4. Layer-by-Layer Execution Loop ---
        for layer in self.layers.iter() {
            hidden_states = layer.forward(
                &mut encoder,
                &hidden_states,
                &attention_mask_gpu,
                self.config.as_ref(),
                &mut temp,
            )?;
        }

        // --- 5. Finalize and Return ---
        temp.reclaim();
        self.context.queue.submit(Some(encoder.finish()));
        let last_hidden_state_cpu = hidden_states.to_ndarray_3d().await?;

        Ok(EncoderOutput { last_hidden_state: last_hidden_state_cpu })
    }

    async fn get_hidden_states(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let output = self.forward(input, attention_mask, token_type_ids).await?;
        Ok(output.last_hidden_state)
    }
}