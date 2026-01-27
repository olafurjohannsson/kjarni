//! Unified LM head supporting both CPU and GPU execution.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array3, s};

use crate::gpu_ops::primitives::linear::GpuLinearLayer;
use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use crate::linear_layer::LinearLayer;
use crate::tensor::DType;
use crate::weights::ModelWeights;
use crate::WgpuContext;

/// Configuration for the LM head.
pub struct LMHeadConfig {
    pub weight_name: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl LMHeadConfig {
    pub fn new(weight_name: impl Into<String>, vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            weight_name: weight_name.into(),
            vocab_size,
            hidden_size,
        }
    }
}

/// Unified LM head supporting both CPU and GPU execution.
pub struct LoadedLMHead {
    pub cpu_weights: Option<LinearLayer>,
    pub gpu_weights: Option<GpuTensor>,
    pub gpu_kernel: Option<GpuLinearLayer>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub context: Option<Arc<WgpuContext>>,
}

impl LoadedLMHead {
    /// Loads LM head from model weights.
    pub fn new(
        ctx: Option<&Arc<WgpuContext>>,
        weights: &ModelWeights,
        config: LMHeadConfig,
        load_cpu: bool,
        load_gpu: bool,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let cpu_weights = if load_cpu {
            log::info!("loading LM head to CPU");
            Some(
                LinearLayer::builder(weights, &config.weight_name)
                    .with_target_dtype(target_dtype)
                    .build()?,
            )
        } else {
            None
        };

        let (gpu_weights, gpu_kernel) = if load_gpu {
            let ctx = ctx.ok_or_else(|| anyhow!("GPU context required for GPU LM head"))?;
            log::info!("loading LM head to GPU");

            let tensor = GpuTensor::from_model_weights(
                ctx,
                weights,
                &config.weight_name,
                target_dtype,
                "lm_head",
            )?;
            let kernel = GpuLinearLayer::new(ctx);

            (Some(tensor), Some(kernel))
        } else {
            (None, None)
        };

        Ok(Self {
            cpu_weights,
            gpu_weights,
            gpu_kernel,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            context: ctx.cloned(),
        })
    }

    /// Creates an LM head by aliasing existing weights (for tied embedding models).
    pub fn from_shared_weights(
        ctx: Option<&Arc<WgpuContext>>,
        cpu_weights: Option<LinearLayer>,
        gpu_weights: Option<GpuTensor>,
        config: LMHeadConfig,
        quantize_dtype: Option<DType>,
    ) -> Result<Self> {
        let gpu_kernel = if let (Some(context), Some(_)) = (ctx, &gpu_weights) {
            Some(GpuLinearLayer::new(context))
        } else {
            None
        };

        let final_cpu_weights = if let Some(cpu_emb) = cpu_weights {
            if let Some(dtype) = quantize_dtype {
                log::info!("creating quantized ({:?}) copy of tied LM head weights", dtype);
                Some(cpu_emb.to_quantized(dtype)?)
            } else {
                Some(cpu_emb.clone())
            }
        } else {
            None
        };

        Ok(Self {
            cpu_weights: final_cpu_weights,
            gpu_weights,
            gpu_kernel,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            context: ctx.cloned(),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn has_cpu(&self) -> bool {
        self.cpu_weights.is_some()
    }

    pub fn has_gpu(&self) -> bool {
        self.gpu_weights.is_some()
    }

    /// Projects hidden states to logits on CPU.
    pub fn forward_cpu(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let t_total = Instant::now();

        let cpu_weights = self
            .cpu_weights
            .as_ref()
            .ok_or_else(|| anyhow!("CPU LM head not loaded"))?;

        let (batch, seq, hidden) = hidden_states.dim();

        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch * seq, hidden))?;

        let t_matmul = Instant::now();
        let logits_2d = cpu_weights.matmul(&hidden_2d);
        let d_matmul = t_matmul.elapsed();

        let result = logits_2d
            .into_shape_with_order((batch, seq, self.vocab_size))
            .map_err(|e| anyhow!("shape error: {}", e));

        let d_total = t_total.elapsed();

        if d_total.as_millis() > 1 {
            log::info!(
                "[LM head] total: {:?}, matmul: {:?}, overhead: {:?}",
                d_total,
                d_matmul,
                d_total.saturating_sub(d_matmul)
            );
        }

        result
    }

    /// Projects hidden states to logits on GPU.
    pub fn forward_gpu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        let gpu_weights = self
            .gpu_weights
            .as_ref()
            .ok_or_else(|| anyhow!("GPU LM head not loaded"))?;
        let gpu_kernel = self.gpu_kernel.as_ref().unwrap();

        let (batch, seq, _hidden) = hidden_states.dims3();

        let logits = pool.get(vec![batch, seq, self.vocab_size]);

        gpu_kernel.encode(encoder, hidden_states, gpu_weights, &logits);

        Ok(logits)
    }

    /// Returns last-token logits on CPU.
    pub fn forward_cpu_last_token(&self, hidden_states: &Array3<f32>) -> Result<Array1<f32>> {
        let logits = self.forward_cpu(hidden_states)?;
        Ok(logits.slice(s![0, -1, ..]).to_owned())
    }
}