use anyhow::Result;
use async_trait::async_trait;
use bytemuck;
use ndarray::{Array2, Array3, s};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::gpu_ops::GpuTensor;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::ffn::GpuFeedForward;
use crate::gpu_ops::blocks::attention::AttentionWeights;
use crate::gpu_ops::blocks::ffn_old::FFNWeights;
use crate::gpu_pipeline::{GpuTransformerLayer, GpuTransformerPipeline};
use crate::traits::{Device, Encoder, EncoderArchitecture, EncoderOutput, TransformerModel};
use crate::weights::ModelWeights;

pub struct GpuEncoderLayer {
    pub ffn: GpuFeedForward,
}

/// The GPU backend for a generic Transformer Encoder.
/// It holds the GPU-native weights and the generic pipeline to execute them.
pub struct GpuTransformerEncoder {
    pipeline: GpuTransformerPipeline,
    // CPU-side embeddings for the initial lookup.
    word_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,
    token_type_embeddings: Array2<f32>,
    // GPU-side weight buffers, specific to this model instance.
    embedding_norm_weights: (Arc<wgpu::Buffer>, Arc<wgpu::Buffer>),
    layers: Vec<GpuTransformerLayer>,
    config: Arc<dyn EncoderArchitecture + Send + Sync>,
}

impl GpuTransformerEncoder {
    /// Constructs a new `GpuTransformerEncoder`.
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn EncoderArchitecture + Send + Sync>,
        context: Arc<WgpuContext>,
    ) -> Result<Self> {
        let pipeline = GpuTransformerPipeline::new(context.clone())?;

        let device = &context.device;

        let upload_1d = |name: &str| -> Result<Arc<wgpu::Buffer>> {
            let tensor = weights.get_array1(name)?;
            Ok(Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytemuck::cast_slice(tensor.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
                },
            )))
        };

        let upload_2d = |name: &str| -> Result<Arc<wgpu::Buffer>> {
            let tensor = weights.get_array2(name)?;

            let transposed_tensor = tensor.t().to_owned();

            Ok(Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytemuck::cast_slice(
                        transposed_tensor.as_standard_layout().as_slice().unwrap(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            )))
        };

        // Load CPU embeddings
        let (word_w, pos_w, type_w) = config.get_embedding_weight_names();
        let word_embeddings = weights.get_array2(word_w)?;
        let position_embeddings = weights.get_array2(pos_w)?;
        let token_type_embeddings = match type_w {
            Some(name) => weights.get_array2(name)?, // Load if present
            None => Array2::zeros((0, config.hidden_size())), // Empty for RoBERTa
        };

        // Upload embedding norm weights
        let (norm_w, norm_b) = config.get_embedding_layer_norm_names();
        let embedding_norm_weights = (upload_1d(norm_w)?, upload_1d(norm_b)?);

        // Upload layer weights
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i);
            let ffn_names = config.get_feed_forward_names(i);
            let q_weight_buf = upload_2d(&attn_names.q_weight)?;
            let q_bias_buf = upload_1d(&attn_names.q_bias)?;
            let k_weight_buf = upload_2d(&attn_names.k_weight)?;
            let k_bias_buf = upload_1d(&attn_names.k_bias)?;
            let v_weight_buf = upload_2d(&attn_names.v_weight)?;
            let v_bias_buf = upload_1d(&attn_names.v_bias)?;
            let attention_weights = AttentionWeights {
                q_weight: q_weight_buf,
                q_bias: q_bias_buf,
                k_weight: k_weight_buf,
                k_bias: k_bias_buf,
                v_weight: v_weight_buf,
                v_bias: v_bias_buf,
                output_weight: upload_2d(&attn_names.output_weight)?,
                output_bias: upload_1d(&attn_names.output_bias)?,
                norm_weight: upload_1d(&attn_names.norm_weight)?,
                norm_bias: upload_1d(&attn_names.norm_bias)?,
            };

            let intermediate_w = weights.get_array2(&ffn_names.intermediate_weight)?;
            let intermediate_b = weights.get_array1(&ffn_names.intermediate_bias)?;
            let output_w = weights.get_array2(&ffn_names.output_weight)?;
            let output_b = weights.get_array1(&ffn_names.output_bias)?;
            let fc1_weight_data = if config.transpose_ffn_weights() {
                intermediate_w.t().as_standard_layout().to_owned()
            } else {
                intermediate_w.as_standard_layout().to_owned()
            };

            let fc2_weight_data = if config.transpose_ffn_weights() {
                output_w.t().as_standard_layout().to_owned()
            } else {
                output_w.as_standard_layout().to_owned()
            };

            // Upload FC1 weights
            let fc1_weight_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} FC1 Weight", i)),
                    contents: bytemuck::cast_slice(fc1_weight_data.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));

            let fc1_bias_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} FC1 Bias", i)),
                    contents: bytemuck::cast_slice(intermediate_b.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));

            // Upload FC2 weights
            let fc2_weight_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} FC2 Weight", i)),
                    contents: bytemuck::cast_slice(fc2_weight_data.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));

            let fc2_bias_buf = Arc::new(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Layer {} FC2 Bias", i)),
                    contents: bytemuck::cast_slice(output_b.as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));

            let ffn_weights = FFNWeights {
                fc1_weight: fc1_weight_buf,
                fc1_bias: fc1_bias_buf,
                fc2_weight: fc2_weight_buf,
                fc2_bias: fc2_bias_buf,
                norm_weight: upload_1d(&ffn_names.norm_weight)?,
                norm_bias: upload_1d(&ffn_names.norm_bias)?,
            };
            // TODO: move to this
            // let fc1_w = GpuTensor::from_ndarray(&context, &intermediate_w.clone())?;
            // let fc1_b = GpuTensor::from_ndarray(&context, &intermediate_b.clone())?;
            // let fc2_w = GpuTensor::from_ndarray(&context, &output_w.clone())?;
            // let fc2_b = GpuTensor::from_ndarray(&context, &output_b.clone())?;

            // let ffn = GpuFeedForward::new(
            //     &context.clone(),
            //     fc1_w,
            //     fc1_b,
            //     fc2_w,
            //     fc2_b,
            //     crate::activations::Activation::Gelu,
            //     config.transpose_ffn_weights()
            // )?;

            layers.push(GpuTransformerLayer {
                attention_weights,
                ffn_weights,
                ffn: None,
            });
        }

        Ok(Self {
            pipeline,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embedding_norm_weights,
            layers,
            config,
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
        let cpu_embedding_layer = crate::Embeddings::new(
            self.word_embeddings.clone(),
            self.position_embeddings.clone(),
            Some(self.token_type_embeddings.clone()),
        );

        let cpu_hidden_states = cpu_embedding_layer.forward(input_ids, token_type_ids);

        Ok(cpu_hidden_states)
    }
}

impl TransformerModel for GpuTransformerEncoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
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
        let initial_embeddings = self.perform_cpu_embedding(input_ids, token_type_ids)?;

        let last_hidden_state = self
            .pipeline
            .forward(
                self.config.as_ref(),
                &initial_embeddings,
                &attention_mask,
                (
                    &self.embedding_norm_weights.0,
                    &self.embedding_norm_weights.1,
                ),
                &self.layers,
            )
            .await?;

        Ok(EncoderOutput { last_hidden_state })
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
