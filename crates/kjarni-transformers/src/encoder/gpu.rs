use anyhow::Result;
use ndarray::{Array2};
use std::sync::Arc;

use crate::encoder::traits::EncoderArchitecture;
use crate::gpu_context::WgpuContext;
use crate::encoder::traits::{GpuEncoder, GpuEncoderInput};
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::gpu_ops::blocks::encoder::GpuEncoderLayer;
use crate::gpu_ops::blocks::ffn::GpuFeedForwardWeights;
use crate::gpu_ops::blocks::layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use crate::traits::{Device, TransformerModel};
use crate::weights::ModelWeights;
use crate::{activations, Embeddings};

pub struct GpuTransformerEncoder {
    embedding_weights: GpuEmbeddingWeights,
    embeddings: GpuEmbeddings,
    embedding_layer_norm: GpuLayerNorm,
    embedding_ln_weights: GpuLayerNormWeights,
    layers: Vec<GpuEncoderLayer>,
    config: Arc<dyn EncoderArchitecture + Send + Sync>,
    context: Arc<WgpuContext>,
    cpu_embeddings: Embeddings,
    // pool: Mutex<GpuTensorPool>,
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

        let embedding_weights: GpuEmbeddingWeights =
            GpuEmbeddingWeights::new(&context, weights, config.as_ref())?;
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
                Ok(raw)
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
            let output_w = weights.get_array2(&ffn_names.output_weight)?;

            let fc1_w_normalized = if config.transpose_ffn_weights() {
                intermediate_w.t().as_standard_layout().to_owned()
            } else {
                intermediate_w
            };
            let fc2_w_normalized = if config.transpose_ffn_weights() {
                output_w.t().as_standard_layout().to_owned()
            } else {
                output_w
            };
            let fc1_b_cpu = weights.get_array1(&ffn_names.intermediate_bias)?;
            let fc2_b_cpu = weights.get_array1(&ffn_names.output_bias)?;

            // 4. Call the smart constructor with the now-guaranteed-standard arrays.
            //    The smart constructor will then handle the internal GPU-specific transposition.
            let ff_weights = GpuFeedForwardWeights::from_ndarrays(
                &context,
                &fc1_w_normalized,
                &fc1_b_cpu,
                &fc2_w_normalized,
                &fc2_b_cpu,
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
                activations::Activation::Gelu,
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
            context: context.clone(),
            cpu_embeddings,
            // pool: Mutex::new(GpuTensorPool::new(context)),
        })
    }

    pub fn config(&self) -> &Arc<dyn EncoderArchitecture + Send + Sync> {
        &self.config
    }
}

impl TransformerModel for GpuTransformerEncoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        Some(self.context.clone())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl GpuEncoder for GpuTransformerEncoder {
    fn embed(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: GpuEncoderInput,
        token_type_ids: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        match input {
            GpuEncoderInput::TokensGpu(input_ids) => {
                // Standard pure-GPU path
                self.embeddings.encode(
                    cmd_encoder,
                    &self.embedding_weights,
                    input_ids,
                    token_type_ids,
                    0, // Encoders don't use a rolling position offset
                    self.config.as_ref(),
                    pool,
                )
            }
            GpuEncoderInput::TokensCpu(input_ids_cpu) => {
                // Hybrid path: embeddings on CPU, layers on GPU.
                let hidden_cpu = self.cpu_embeddings.forward(
                    input_ids_cpu, 
                    None, // Assuming CPU path doesn't get token_type_ids for now
                    self.config.extra_pos_embeddings(), 
                    self.config.scale_embeddings()
                );
                GpuTensor::from_ndarray(&self.context, &hidden_cpu)
            }
            GpuEncoderInput::HiddenGpu(hidden_states) => Ok(hidden_states.clone()),
            GpuEncoderInput::HiddenCpu(hidden_states_cpu) => GpuTensor::from_ndarray(&self.context, hidden_states_cpu),
        }
    }

    fn embed_and_normalize(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: GpuEncoderInput,
        token_type_ids: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        // This logic is taken directly from your old `forward` method.
        let hidden_states = self.embed(cmd_encoder, pool, input, token_type_ids)?;
        
        // This logic correctly handles post-norm models like BERT/BART.
        // For a pre-norm model, this would just return `hidden_states`.
        if !self.config.is_prenorm() {
            let ln_output = pool.get(hidden_states.shape().to_vec());
            self.embedding_layer_norm.encode(
                cmd_encoder,
                &self.embedding_ln_weights,
                &hidden_states,
                &ln_output,
            );
            Ok(ln_output)
        } else {
            Ok(hidden_states)
        }
    }

    fn forward_layers(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor> {
        // This logic is also taken directly from your old `forward` method.
        let mut current_states = hidden_states.clone();
        for layer in &self.layers[start_layer..end_layer] {
            current_states = layer.forward(
                cmd_encoder,
                &current_states,
                attention_mask,
                self.config.as_ref(),
                pool,
            )?;
        }
        Ok(current_states)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size()
    }

    // The default `forward` method in the `GpuEncoder` trait will automatically
    // chain these three methods together, so you don't need to implement it here.
}

// #[async_trait]
// impl Encoder for GpuTransformerEncoder {
//     type Input = Array2<u32>;
//     type Output = EncoderOutput;

//     async fn forward(
//         &self,
//         input_ids: &Self::Input,
//         attention_mask: &Array2<f32>,
//         token_type_ids: Option<&Array2<u32>>,
//     ) -> Result<Self::Output> {
//         let pool_guard = self.pool.lock().await;
//         let mut frame = GpuFrameContext::new(&self.context, pool_guard);
//         let (encoder, pool) = frame.resources();

//         let input_ids_gpu = GpuTensor::from_ndarray(&self.context, input_ids)?;

//         // Handle optional token_type_ids upload
//         let token_type_ids_gpu = if let Some(ids) = token_type_ids {
//             Some(GpuTensor::from_ndarray(&self.context, ids)?)
//         } else {
//             None
//         };

//         let mut hidden_states = self.embeddings.encode(
//             encoder,
//             &self.embedding_weights,
//             &input_ids_gpu,
//             token_type_ids_gpu.as_ref(), // Pass as Option<&GpuTensor>
//             0,                           // encoder doesnt need a dynamic position offset
//             self.config.as_ref(),
//             pool,
//         )?;

//         let attention_mask_gpu = GpuTensor::from_ndarray(&self.context, attention_mask)?;

//         if !self.config.is_prenorm() {
//             let ln_output = pool.get(hidden_states.shape().to_vec());
//             self.embedding_layer_norm.encode(
//                 encoder,
//                 &self.embedding_ln_weights,
//                 &hidden_states,
//                 &ln_output,
//             );
//             hidden_states = ln_output;
//         }

//         for layer in self.layers.iter() {
//             hidden_states = layer.forward(
//                 encoder,
//                 &hidden_states,
//                 &attention_mask_gpu,
//                 self.config.as_ref(),
//                 pool,
//             )?;
//         }

//         frame.finish();
//         let last_hidden_state_cpu = hidden_states.to_ndarray_3d().await?;

//         Ok(EncoderOutput {
//             last_hidden_state: last_hidden_state_cpu,
//         })
//     }

//     async fn get_hidden_states(
//         &self,
//         input: &Self::Input,
//         attention_mask: &Array2<f32>,
//         token_type_ids: Option<&Array2<u32>>,
//     ) -> Result<Array3<f32>> {
//         let output = self.forward(input, attention_mask, token_type_ids).await?;
//         Ok(output.last_hidden_state)
//     }
// }


