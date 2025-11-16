use anyhow::{Result};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::sync::Arc;

use crate::gpu_ops::GpuTensor;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::encoder::GpuEncoderLayer; 
use crate::gpu_ops::blocks::attention::{GpuAttentionWeights, TempStorage};
use crate::gpu_ops::blocks::ffn::GpuFeedForwardWeights;
use crate::gpu_ops::blocks::layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::traits::{Device, Encoder, EncoderArchitecture, EncoderOutput, TransformerModel};
use crate::weights::ModelWeights;
use crate::Embeddings;

pub struct GpuTransformerEncoder {
    embedding_weights: GpuEmbeddingWeights,
    embeddings: GpuEmbeddings,
    embedding_layer_norm: GpuLayerNorm,
    embedding_ln_weights: GpuLayerNormWeights,
    layers: Vec<GpuEncoderLayer>,
    config: Arc<dyn EncoderArchitecture + Send + Sync>,
    context: Arc<WgpuContext>,
    cpu_embeddings: Embeddings,
}

impl GpuTransformerEncoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn EncoderArchitecture + Send + Sync>,
        context: Arc<WgpuContext>,
    ) -> Result<Self> {
        let (word_w, pos_w, type_w) = config.get_embedding_weight_names();
        let word_embeddings = weights.get_array2(word_w)?;
        let position_embeddings = weights.get_array2(pos_w)?;
        let token_type_embeddings = if let Some(name) = type_w {
            weights.get_array2(name)?
        } else {
            Array2::zeros((0, 0)) // Placeholder for models without token types
        };

        let cpu_embeddings = Embeddings::new(
            word_embeddings.clone(),
            Some(position_embeddings.clone()),
            Some(token_type_embeddings.clone()),
        );

        let embedding_weights: GpuEmbeddingWeights = GpuEmbeddingWeights::new(&context, weights, config.as_ref())?;
        let embeddings: GpuEmbeddings = GpuEmbeddings::new(&context)?;

        // --- Load and create embedding LayerNorm components ---
        let (norm_w_name, norm_b_name) = config.get_embedding_layer_norm_names();
        let embed_ln_gamma_cpu = weights.get_array1(norm_w_name)?;
        let embed_ln_beta_cpu = weights.get_array1(norm_b_name)?;

        let embedding_ln_weights = GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(&context, &embed_ln_gamma_cpu)?,
            GpuTensor::from_ndarray(&context, &embed_ln_beta_cpu)?,
        )?;
        let embedding_layer_norm = GpuLayerNorm::new(&context, config.layer_norm_eps());

        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i);
            let ffn_names = config.get_feed_forward_names(i);
            let transpose_attn = config.transpose_attention_weights();

            let prep_attn_w = |name: &str| -> Result<Array2<f32>> {
                let raw = weights.get_array2(name)?;
                // todo: this is backwards, but it works for now
                if transpose_attn {
                    Ok(raw)
                } else {
                    Ok(raw.t().as_standard_layout().to_owned())
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
            embedding_weights,
            embeddings,
            embedding_layer_norm,
            embedding_ln_weights,
            layers,
            config,
            context,
            cpu_embeddings
        })
    }

    pub fn config(&self) -> &Arc<dyn EncoderArchitecture + Send + Sync> {
        &self.config
    }

}

impl TransformerModel for GpuTransformerEncoder {
    fn device(&self) -> Device { Device::Wgpu }
    fn context(&self) -> Option<Arc<WgpuContext>> { Some(self.context.clone()) }
}

#[async_trait]
impl Encoder for GpuTransformerEncoder {
    type Input = Array2<u32>;
    type Output = EncoderOutput;

    async fn forward(
        &self,
        input_ids: &Self::Input,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Result<Self::Output> {
        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Encoder Forward") });
        let mut temp = TempStorage::new(self.context.clone());
        let input_ids_gpu = GpuTensor::from_ndarray(&self.context, input_ids)?;
        
        // Handle optional token_type_ids upload
        let token_type_ids_gpu = if let Some(ids) = token_type_ids {
            Some(GpuTensor::from_ndarray(&self.context, ids)?)
        } else {
            None
        };

        let mut hidden_states = self.embeddings.encode(
            &mut encoder,
            &self.embedding_weights,
            &input_ids_gpu,
            token_type_ids_gpu.as_ref(), // Pass as Option<&GpuTensor>
            0, // encoder doesnt need a dynamic position offset
            self.config.as_ref(),
            &mut temp,
        )?;

        let attention_mask_gpu = GpuTensor::from_ndarray(&self.context, attention_mask)?;

        if !self.config.is_prenorm() {
            let ln_output = temp.get(hidden_states.shape().to_vec());
            self.embedding_layer_norm.encode(&mut encoder, &self.embedding_ln_weights, &hidden_states, &ln_output);
            hidden_states = ln_output;
        }

        for layer in self.layers.iter() {
            hidden_states = layer.forward(
                &mut encoder,
                &hidden_states,
                &attention_mask_gpu,
                self.config.as_ref(),
                &mut temp,
            )?;
        }

        temp.reclaim();
        self.context.queue.submit(Some(encoder.finish()));
        let last_hidden_state_cpu = hidden_states.to_ndarray_3d().await?;

        Ok(EncoderOutput { last_hidden_state: last_hidden_state_cpu })
    }

    async fn get_hidden_states(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Result<Array3<f32>> {
        let output = self.forward(input, attention_mask, token_type_ids).await?;
        Ok(output.last_hidden_state)
    }
}