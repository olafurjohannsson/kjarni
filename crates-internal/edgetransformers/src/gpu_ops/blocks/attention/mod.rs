//! Attention Block
//!
//! Multi-head self-attention mechanism configurable for:
//! - Encoder (bidirectional)
//! - Decoder (causal)
//! - Cross-attention (encoder-decoder)

use crate::gpu_ops::primitives::{
    add_bias::{run_gpu_add_bias, compile_add_bias_pipeline},
    apply_mask::{run_gpu_apply_mask, compile_apply_mask_pipeline},
    matmul_old::{run_gpu_bmm, run_gpu_matmul, compile_bmm_pipeline, compile_matmul_pipeline},
    reshape::{run_gpu_reshape, run_gpu_unreshape, compile_reshape_pipeline, compile_unreshape_pipeline},
    softmax::{run_gpu_softmax, compile_softmax_pipeline},
};
use crate::gpu_context::WgpuContext;
use std::sync::Arc;
use wgpu::{Buffer, CommandEncoder, ComputePipeline};

/// Configuration for attention computation
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub is_causal: bool,
}

/// GPU buffers for attention weights
pub struct AttentionWeights {
    pub q_weight: Arc<Buffer>,
    pub q_bias: Arc<Buffer>,
    pub k_weight: Arc<Buffer>,
    pub k_bias: Arc<Buffer>,
    pub v_weight: Arc<Buffer>,
    pub v_bias: Arc<Buffer>,
    pub output_weight: Arc<Buffer>,
    pub output_bias: Arc<Buffer>,
    pub norm_weight: Arc<Buffer>,
    pub norm_bias: Arc<Buffer>,
}

/// Temporary buffers for attention computation
pub struct AttentionTempBuffers {
    pub q_proj: Buffer,
    pub k_proj: Buffer,
    pub v_proj: Buffer,
    pub proj_biased: Buffer,
    pub q_permuted: Buffer,
    pub k_permuted_t: Buffer,
    pub v_permuted: Buffer,
    pub scores: Buffer,
    pub context_vectors: Buffer,
    pub ffn_intermediate: Buffer,
}

/// Pre-compiled pipelines for attention operations
pub struct AttentionPipelines {
    pub matmul: Arc<ComputePipeline>,
    pub add_bias: Arc<ComputePipeline>,
    pub reshape: Arc<ComputePipeline>,
    pub unreshape: Arc<ComputePipeline>,
    pub bmm: Arc<ComputePipeline>,
    pub softmax: Arc<ComputePipeline>,
    pub apply_mask: Arc<ComputePipeline>,
}

impl AttentionPipelines {
    pub fn new(context: &WgpuContext) -> Self {
        Self {
            matmul: Arc::new(compile_matmul_pipeline(context)),
            add_bias: Arc::new(compile_add_bias_pipeline(context)),
            reshape: Arc::new(compile_reshape_pipeline(context)),
            unreshape: Arc::new(compile_unreshape_pipeline(context)),
            bmm: Arc::new(compile_bmm_pipeline(context)),
            softmax: Arc::new(compile_softmax_pipeline(context)),
            apply_mask: Arc::new(compile_apply_mask_pipeline(context)),
        }
    }
}

// First 10 embeddings for GPU (WGPU) - embeddings: [0.0035336413, -0.040116385, 0.012812932, -0.0039921585,
/// Run complete attention block on GPU
///
/// This orchestrates:
/// 1. Q, K, V projections
/// 2. Reshape to [batch, heads, seq, head_dim]
/// 3. Compute scores: Q @ K^T / sqrt(d)
/// 4. Apply mask (causal or padding)
/// 5. Softmax over scores
/// 6. Apply to values: Softmax @ V
/// 7. Reshape back and output projection
///
/// # Arguments
/// * `input` - Input hidden states [batch, seq, hidden]
/// * `output` - Output buffer [batch, seq, hidden]
/// * `mask` - Attention mask
/// * `config` - Attention configuration
/// * `weights` - Pre-uploaded GPU weights
/// * `pipelines` - Pre-compiled compute pipelines
/// * `temp` - Temporary buffers
pub fn run_attention_block(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipelines: &AttentionPipelines,
    input: &Buffer,
    output: &Buffer,
    mask: &Buffer,
    config: &AttentionConfig,
    weights: &AttentionWeights,
    temp: &AttentionTempBuffers,
) {
    let m = (config.batch_size * config.seq_len) as u32;
    let k = config.hidden_size as u32;

    // === Q Projection: input @ W_q + b_q ===
    run_gpu_matmul(
        context,
        encoder,
        &pipelines.matmul,
        input,
        &weights.q_weight,
        &temp.q_proj,
        m,
        k,
        k,
    );
    run_gpu_add_bias(
        context,
        encoder,
        &pipelines.add_bias,
        &temp.q_proj,
        &weights.q_bias,
        &temp.proj_biased,
        m * k,
    );

    // Reshape to [batch, heads, seq, head_dim]
    run_gpu_reshape(
        context,
        encoder,
        &pipelines.reshape,
        &temp.proj_biased,
        &temp.q_permuted,
        config.batch_size as u32,
        config.seq_len as u32,
        config.num_heads as u32,
        config.head_dim as u32,
        false,
    );

    // === K Projection: input @ W_k + b_k ===
    run_gpu_matmul(
        context,
        encoder,
        &pipelines.matmul,
        input,
        &weights.k_weight,
        &temp.k_proj,
        m,
        k,
        k,
    );

    run_gpu_add_bias(
        context,
        encoder,
        &pipelines.add_bias,
        &temp.k_proj,
        &weights.k_bias,
        &temp.proj_biased,
        m * k,
    );

    // Reshape and transpose for K
    run_gpu_reshape(
        context,
        encoder,
        &pipelines.reshape,
        &temp.proj_biased,
        &temp.k_permuted_t,
        config.batch_size as u32,
        config.seq_len as u32,
        config.num_heads as u32,
        config.head_dim as u32,
        true, // Transpose K
    );

    // === V Projection: input @ W_v + b_v ===
    run_gpu_matmul(
        context,
        encoder,
        &pipelines.matmul,
        input,
        &weights.v_weight,
        &temp.v_proj,
        m,
        k,
        k,
    );

    run_gpu_add_bias(
        context,
        encoder,
        &pipelines.add_bias,
        &temp.v_proj,
        &weights.v_bias,
        &temp.proj_biased,
        m * k,
    );

    run_gpu_reshape(
        context,
        encoder,
        &pipelines.reshape,
        &temp.proj_biased,
        &temp.v_permuted,
        config.batch_size as u32,
        config.seq_len as u32,
        config.num_heads as u32,
        config.head_dim as u32,
        false,
    );

    // === Attention Scores: Q @ K^T ===
    run_gpu_bmm(
        context,
        encoder,
        &pipelines.bmm,
        &temp.q_permuted,
        &temp.k_permuted_t,
        &temp.scores,
        (config.batch_size * config.num_heads) as u32,
        config.seq_len as u32,
        config.head_dim as u32,
        config.seq_len as u32,
    );

    // === Apply Mask ===
    run_gpu_apply_mask(
        context,
        encoder,
        &pipelines.apply_mask,
        &temp.scores,
        mask,
        config.batch_size as u32,
        config.num_heads as u32,
        config.seq_len as u32,
        config.is_causal,
    );

    // === Softmax with Scaling ===
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    run_gpu_softmax(
        context,
        encoder,
        &pipelines.softmax,
        &temp.scores,
        (config.batch_size * config.num_heads * config.seq_len) as u32,
        config.seq_len as u32,
        scale,
    );

    // === Apply to Values: Attention @ V ===
    run_gpu_bmm(
        context,
        encoder,
        &pipelines.bmm,
        &temp.scores,
        &temp.v_permuted,
        &temp.context_vectors,
        (config.batch_size * config.num_heads) as u32,
        config.seq_len as u32,
        config.seq_len as u32,
        config.head_dim as u32,
    );

    // === Reshape Back to [batch, seq, hidden] ===
    run_gpu_unreshape(
        context,
        encoder,
        &pipelines.unreshape,
        &temp.context_vectors,
        &temp.proj_biased,
        config.batch_size as u32,
        config.seq_len as u32,
        config.num_heads as u32,
        config.head_dim as u32,
    );

    // === Output Projection: context @ W_o + b_o ===
    run_gpu_matmul(
        context,
        encoder,
        &pipelines.matmul,
        &temp.proj_biased,
        &weights.output_weight,
        &temp.q_proj, // Reuse as temporary
        m,
        k,
        k,
    );

    run_gpu_add_bias(
        context,
        encoder,
        &pipelines.add_bias,
        &temp.q_proj,
        &weights.output_bias,
        output,
        m * k,
    );
}

#[cfg(test)]
mod tests;
