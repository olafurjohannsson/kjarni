use crate::cache::GpuKVCache;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::attention::TempStorage;
use crate::gpu_ops::blocks::decoder::GpuPreNormDecoderLayer;
use crate::gpu_ops::blocks::ffn::GpuFeedForwardWeights;
use crate::gpu_ops::blocks::layer_norm::GpuLayerNorm;
use crate::gpu_ops::blocks::layer_norm::GpuLayerNormWeights;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::traits::{Cache, Decoder, DecoderArchitecture, DecoderOutput, Device, TransformerModel};
use crate::weights::ModelWeights;
use anyhow::Result;
use async_trait::async_trait;
use log::{debug, info};
use ndarray::{Array2, Array3, s};
use std::sync::Arc;

/// The GPU backend for a generic Transformer Decoder.
pub struct GpuTransformerDecoder {
    // CPU-side embeddings
    word_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,

    // GPU-side weight buffers
    layers: Vec<GpuPreNormDecoderLayer>,

    // Final LayerNorm components
    final_layer_norm: GpuLayerNorm,
    final_ln_weights: GpuLayerNormWeights,

    // Essential GPU primitives needed by the forward pass
    slicer: GpuSlice,

    config: Arc<dyn DecoderArchitecture + Send + Sync>,

    context: Arc<WgpuContext>,
}

impl GpuTransformerDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
        context: Arc<WgpuContext>,
    ) -> Result<Self> {
        // --- Initialize stateless primitives needed by the decoder ---
        let slicer = GpuSlice::new(&context);

        // Load CPU-side embeddings
        let (word_w, pos_w) = config.get_embedding_weight_names();
        let word_embeddings = weights.get_array2(word_w)?;
        let position_embeddings = weights.get_array2(pos_w)?;

        // --- Load and create final LayerNorm components ---
        let (norm_w_name, norm_b_name) = config.get_final_layer_norm_names();
        let final_ln_gamma_cpu = weights.get_array1(norm_w_name)?;
        let final_ln_beta_cpu = weights.get_array1(norm_b_name)?;

        let final_ln_weights = GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(&context, &final_ln_gamma_cpu)?,
            GpuTensor::from_ndarray(&context, &final_ln_beta_cpu)?,
        )?;
        let final_layer_norm = GpuLayerNorm::new(&context, config.layer_norm_eps());

        // --- Load weights and create layers ---
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i);
            let ffn_names = config.get_feed_forward_names(i);
            let hidden_size = config.hidden_size();

            // 1. Load Attention weights from CPU tensors
            let qkv_weight = weights.get_array2(&attn_names.qkv_weight)?;
            let qkv_bias = weights.get_array1(&attn_names.qkv_bias)?;
            let q_weight = qkv_weight.slice(s![.., 0..hidden_size]).to_owned();
            let k_weight = qkv_weight
                .slice(s![.., hidden_size..2 * hidden_size])
                .to_owned();
            let v_weight = qkv_weight
                .slice(s![.., 2 * hidden_size..3 * hidden_size])
                .to_owned();
            let q_bias = qkv_bias.slice(s![0..hidden_size]).to_owned();
            let k_bias = qkv_bias.slice(s![hidden_size..2 * hidden_size]).to_owned();
            let v_bias = qkv_bias
                .slice(s![2 * hidden_size..3 * hidden_size])
                .to_owned();
            let attn_output_w = weights.get_array2(&attn_names.output_weight)?;
            let attn_output_b = weights.get_array1(&attn_names.output_bias)?;

            let self_attn_weights = GpuAttentionWeights::new(
                GpuTensor::from_ndarray(&context, &q_weight)?,
                GpuTensor::from_ndarray(&context, &q_bias)?,
                GpuTensor::from_ndarray(&context, &k_weight)?,
                GpuTensor::from_ndarray(&context, &k_bias)?,
                GpuTensor::from_ndarray(&context, &v_weight)?,
                GpuTensor::from_ndarray(&context, &v_bias)?,
                GpuTensor::from_ndarray(&context, &attn_output_w)?,
                GpuTensor::from_ndarray(&context, &attn_output_b)?,
            )?;

            // 2. Load Attention LayerNorm weights
            let attn_ln_gamma = weights.get_array1(&attn_names.norm_weight)?;
            let attn_ln_beta = weights.get_array1(&attn_names.norm_bias)?;
            let self_attn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_ndarray(&context, &attn_ln_gamma)?,
                GpuTensor::from_ndarray(&context, &attn_ln_beta)?,
            )?;

            // 3. Load Feed-Forward weights, applying transposition logic first
            let intermediate_w = weights.get_array2(&ffn_names.intermediate_weight)?;
            let fc1_w_cpu = if config.transpose_ffn_weights() {
                intermediate_w.t().as_standard_layout().to_owned()
            } else {
                intermediate_w
            };
            let output_w = weights.get_array2(&ffn_names.output_weight)?;
            let fc2_w_cpu = if config.transpose_ffn_weights() {
                output_w.t().as_standard_layout().to_owned()
            } else {
                output_w
            };

            let ff_weights = GpuFeedForwardWeights::new(
                GpuTensor::from_ndarray(&context, &fc1_w_cpu)?,
                GpuTensor::from_ndarray(
                    &context,
                    &weights.get_array1(&ffn_names.intermediate_bias)?,
                )?,
                GpuTensor::from_ndarray(&context, &fc2_w_cpu)?,
                GpuTensor::from_ndarray(&context, &weights.get_array1(&ffn_names.output_bias)?)?,
            )?;

            // 4. Load FFN LayerNorm weights
            let ffn_ln_gamma = weights.get_array1(&ffn_names.norm_weight)?;
            let ffn_ln_beta = weights.get_array1(&ffn_names.norm_bias)?;
            let ffn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_ndarray(&context, &ffn_ln_gamma)?,
                GpuTensor::from_ndarray(&context, &ffn_ln_beta)?,
            )?;

            // 5. Create the specialized GPU decoder layer
            layers.push(GpuPreNormDecoderLayer::new(
                &context,
                self_attn_weights,
                self_attn_ln_weights,
                ff_weights,
                ffn_ln_weights,
                config.clone(),
            )?);
        }

        Ok(Self {
            word_embeddings,
            position_embeddings,
            layers,
            final_layer_norm,
            final_ln_weights,
            slicer,
            config: config.clone(),
            context,
        })
    }

    fn perform_cpu_embedding(
        &self,
        input_ids: &Array2<f32>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.config.hidden_size();
        let mut cpu_hidden_states = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));

        // Word embeddings
        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                if token_id < self.word_embeddings.shape()[0] {
                    cpu_hidden_states
                        .slice_mut(s![i, j, ..])
                        .assign(&self.word_embeddings.row(token_id));
                }
            }
        }

        // Position embeddings (offset by cache position for incremental decoding)
        let pos_start = position_offset;
        let pos_end = position_offset + seq_len;
        let pos_embeddings_to_add = self.position_embeddings.slice(s![pos_start..pos_end, ..]);
        cpu_hidden_states += &pos_embeddings_to_add;

        Ok(cpu_hidden_states)
    }
}

impl TransformerModel for GpuTransformerDecoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        Some(self.context.clone())
    }
}

#[async_trait]
impl Decoder for GpuTransformerDecoder {
    type Input = Array2<f32>;
    type Output = DecoderOutput;

    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        // --- 1. Setup ---
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Decoder Forward"),
                });
        let mut temp = TempStorage::new(self.context.clone());
        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
        let seq_len = input.shape()[1];
        debug!(
            "[GPU Decoder] Forward pass started. position_offset: {}",
            position_offset
        );

        // --- 2. Embeddings & Upload ---
        let initial_embeddings_cpu = self.perform_cpu_embedding(input, position_offset)?;
        let mut hidden_states = GpuTensor::from_ndarray(&self.context, &initial_embeddings_cpu)?;
        let attention_mask_gpu = GpuTensor::from_ndarray(&self.context, attention_mask)?;

        let mut gpu_cache = cache.and_then(|c| c.as_any_mut().downcast_mut::<GpuKVCache>());
        let mut new_key_values = Vec::with_capacity(self.layers.len());

        // --- 3. Layer-by-Layer Execution Loop ---
        info!(
            "[GPU Decoder] Executing {} decoder layers...",
            self.layers.len()
        );
        for (i, layer) in self.layers.iter().enumerate() {
            debug!("[GPU Decoder] --- Layer {} ---", i);

            // ✅ OPTIMIZED PATH: Pass PHYSICAL cache buffers directly
            let physical_past_kv_3d = if position_offset > 0 {
                let (k_4d, v_4d) = gpu_cache.as_ref().unwrap().get(i).unwrap();

                // Reshape [batch, heads, max_seq, head_dim] -> [batch*heads, max_seq, head_dim]
                let (b, h, max_seq, d) = k_4d.dims4();

                // View as 3D (no copy, just different shape interpretation)
                let k_3d = k_4d.view_as_3d(b * h, max_seq, d)?;
                let v_3d = v_4d.view_as_3d(b * h, max_seq, d)?;
                println!(
                    "[GPU Decoder] Layer {} cache shapes => k_4d: {:?} rank={}, v_4d: {:?} rank={} | k_3d: {:?} rank={}, v_3d: {:?} rank={}",
                    i,
                    k_4d.shape(),
                    k_4d.rank(),
                    v_4d.shape(),
                    v_4d.rank(),
                    k_3d.shape(),
                    k_3d.rank(),
                    v_3d.shape(),
                    v_3d.rank(),
                );
                Some((k_3d, v_3d))
            } else {
                None
            };

            if let Some((k, _)) = &physical_past_kv_3d {
                debug!(
                    "[GPU Decoder] Layer {} using physical cache with shape: {:?}",
                    i,
                    k.shape()
                );
            } else {
                debug!("[GPU Decoder] Layer {} priming pass (no cache)", i);
            }

            // Call layer with physical buffers (no slicing/concatenating!)
            let (output, (new_k, new_v)) = layer.forward(
                &mut encoder,
                &hidden_states,
                &attention_mask_gpu,
                position_offset,
                physical_past_kv_3d.as_ref().map(|(k, v)| (k, v)), // Convert Option<(T, T)> to Option<(&T, &T)>
                &mut temp,
            )?;

            hidden_states = output;
            new_key_values.push((new_k, new_v));
        }
        // for (i, layer) in self.layers.iter().enumerate() {
        //     debug!("[GPU Decoder] --- Layer {} ---", i);

        //     // ========================= THE FINAL, CORRECT LOGIC =========================

        //     // This will hold the correctly prepared past_kv view for the layer.
        //     // We need to declare the variables that will own the temporary sliced tensors here.
        //     let (mut temp_k, mut temp_v);
        //     let past_kv_for_layer: Option<(&GpuTensor, &GpuTensor)>;

        //     if position_offset > 0 {
        //         // --- GENERATION PASS ---
        //         // 1. Get the PHYSICAL cache buffers.
        //         let (physical_k, physical_v) = gpu_cache.as_ref().unwrap().get(i).unwrap();

        //         // 2. SLICE the physical buffers to the current logical length (`position_offset`).
        //         let (b, h, _, d) = physical_k.dims4();

        //         // TODO: remove and use strides for cache in GpuBatchedMatMul and GpuApplyMask
        //         temp_k = physical_k.slice(
        //             &mut encoder,
        //             &self.slicer,
        //             &[0, 0, 0, 0],
        //             &[b, h, position_offset, d], // Slice to the length of what's already in the cache
        //         )?;
        //         temp_v = physical_v.slice(
        //             &mut encoder,
        //             &self.slicer,
        //             &[0, 0, 0, 0],
        //             &[b, h, position_offset, d],
        //         )?;

        //         // The layer will receive a view of these new, temporary sliced tensors.
        //         past_kv_for_layer = Some((&temp_k, &temp_v));
        //     } else {
        //         // --- PRIMING PASS ---
        //         // There is no past state, so we pass None.
        //         past_kv_for_layer = None;
        //     }

        //     if let Some((k, _)) = past_kv_for_layer {
        //         debug!(
        //             "[GPU Decoder] Layer {} past_kv shape for forward: {:?}",
        //             i,
        //             k.shape()
        //         );
        //     } else {
        //         debug!("[GPU Decoder] Layer {} past_kv is None (priming pass).", i);
        //     }

        //     // Call the layer's forward method. It now receives the correct data in all cases.
        //     let (output, (new_k, new_v)) = layer.forward(
        //         &mut encoder,
        //         &hidden_states,
        //         &attention_mask_gpu,
        //         position_offset,
        //         past_kv_for_layer,
        //         &mut temp,
        //     )?;

        //     // ==============================================================================

        //     hidden_states = output;
        //     new_key_values.push((new_k, new_v));
        // }

        // --- 4. Cache Update ---
        if let Some(ref mut cache) = gpu_cache {
            for (i, (k, v)) in new_key_values.iter().enumerate() {
                cache.update(&mut encoder, i, k, v)?;
            }
        }

        // --- 5. Final Layer Normalization ---
        let final_ln_output = temp.get(hidden_states.shape().to_vec());
        self.final_layer_norm.encode(
            &mut encoder,
            &self.final_ln_weights,
            &hidden_states,
            &final_ln_output,
        );
        hidden_states = final_ln_output;

        // --- 6. Finalize and Return ---
        temp.reclaim();
        self.context.queue.submit(Some(encoder.finish()));
        let last_hidden_state_cpu = hidden_states.to_ndarray_3d().await?;

        if let Some(cache) = gpu_cache {
            cache.increment_len(seq_len);
        }

        Ok(DecoderOutput {
            last_hidden_state: last_hidden_state_cpu,
            past_key_values: None,
        })
    }

    async fn get_hidden_states(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        // Forward without cache
        let output = self.forward(input, attention_mask, None).await?;
        Ok(output.last_hidden_state)
    }
}
