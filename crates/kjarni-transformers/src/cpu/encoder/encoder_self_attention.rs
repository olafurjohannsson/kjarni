//! Bidirectional self-attention for encoder models.
//!
//! This module provides `EncoderSelfAttention`, the CPU implementation of
//! self-attention for encoder-only and encoder portions of encoder-decoder models.
//!
//! # Characteristics
//!
//! - **Bidirectional**: Each token can attend to all other tokens (no causal mask).
//! - **No KV Cache**: Processes the full sequence at once.
//! - **Optional Position Bias**: Supports T5-style relative position bias.
//!
//! # Used By
//!
//! - BERT, RoBERTa, DistilBERT (encoder-only)
//! - BART encoder, T5 encoder (encoder-decoder)
//! - Sentence transformers
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::cpu_ops::attention::EncoderSelfAttention;
//!
//! let attn = EncoderSelfAttention::new(768, 12, q_proj, k_proj, v_proj, o_proj);
//!
//! let output = attn.forward(&hidden_states, &attention_mask, None)?;
//! ```

use crate::activations::softmax_4d_inplace;
use crate::linear_layer::LinearLayer;
use crate::rope::RoPE;
use crate::utils::linear_algebra::{apply_attention_mask, matmul_4d};
use anyhow::Result;
use ndarray::{Array2, Array3, Array4, s};

/// Large negative value for masking (avoids NaN in softmax).
const MASK_VALUE: f32 = -1e9;

/// Bidirectional self-attention for encoder models.
///
/// Computes multi-head self-attention where every token can attend to every
/// other token in the sequence. This is the standard attention used in
/// encoder-only models like BERT.
///
/// # Architecture
///
/// ```text
/// Input [B, S, H]
///     │
///     ├──► Q = input @ W_q + b_q
///     ├──► K = input @ W_k + b_k
///     └──► V = input @ W_v + b_v
///           │
///           ▼
///     Split into heads: [B, S, H] -> [B, num_heads, S, head_dim]
///           │
///           ▼
///     Scores = Q @ K^T / sqrt(head_dim)
///           │
///           ▼
///     Apply padding mask (optional)
///           │
///           ▼
///     Add position bias (optional, for T5)
///           │
///           ▼
///     Softmax
///           │
///           ▼
///     Context = Scores @ V
///           │
///           ▼
///     Merge heads: [B, num_heads, S, head_dim] -> [B, S, H]
///           │
///           ▼
///     Output = Context @ W_o + b_o
/// ```
pub struct EncoderSelfAttention {
    /// Query projection layer.
    pub q_proj: LinearLayer,
    /// Key projection layer.
    pub k_proj: LinearLayer,
    /// Value projection layer.
    pub v_proj: LinearLayer,
    /// Output projection layer.
    pub out_proj: LinearLayer,

    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Scaling factor: 1 / sqrt(head_dim).
    pub scale_factor: f32,
    /// Whether to scale Q@K by sqrt(head_dim). False for T5.
    pub scale_qk: bool,
}

impl EncoderSelfAttention {
    /// Creates a new encoder self-attention module.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - The model's hidden dimension.
    /// * `num_heads` - Number of attention heads.
    /// * `q` - Query projection weights.
    /// * `k` - Key projection weights.
    /// * `v` - Value projection weights.
    /// * `o` - Output projection weights.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // BERT-base: 768 hidden, 12 heads
    /// use kjarni_transformers::encoder::EncoderSelfAttention;
    /// use kjarni_transformers::linear_layer::LinearLayer;
    /// use kjarni_transformers::tensor::DType;
    /// let q_proj = LinearLayer::new(768, 768, DType::F32);
    /// let k_proj = LinearLayer::new(768, 768, DType::F32);
    /// let v_proj = LinearLayer::new(768, 768, DType::F32);
    /// let o_proj = LinearLayer::new(768, 768, DType::F32);
    /// let attn = EncoderSelfAttention::new(768, 12, q_proj, k_proj, v_proj, o_proj);
    /// ```
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        q: LinearLayer,
        k: LinearLayer,
        v: LinearLayer,
        o: LinearLayer,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            out_proj: o,
            num_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
            scale_qk: true,
        }
    }

    /// Disables Q@K scaling (for T5-style attention).
    ///
    /// T5 uses relative position bias instead of scaled dot-product attention.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kjarni_transformers::encoder::EncoderSelfAttention;
    /// use kjarni_transformers::linear_layer::LinearLayer;
    /// use kjarni_transformers::tensor::DType;
    /// let q = LinearLayer::new(768, 768, DType::F32);
    /// let k = LinearLayer::new(768, 768, DType::F32);
    /// let v = LinearLayer::new(768, 768, DType::F32);
    /// let o = LinearLayer::new(768, 768, DType::F32);
    /// let attn = EncoderSelfAttention::new(768, 12, q, k, v, o)
    ///     .with_no_qk_scaling();
    /// ```
    pub fn with_no_qk_scaling(mut self) -> Self {
        self.scale_qk = false;
        self
    }

    /// Performs the forward pass of encoder self-attention.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor of shape `[batch, seq, hidden]`.
    /// * `attention_mask` - Padding mask of shape `[batch, seq]` where 1.0 = valid, 0.0 = masked.
    /// * `position_bias` - Optional relative position bias `[1, heads, seq, seq]` (T5/ALiBi).
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, seq, hidden]`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Standard attention (BERT/BART)
    /// use kjarni_transformers::encoder::EncoderSelfAttention;
    /// use kjarni_transformers::tensor::DType;
    /// use kjarni_transformers::linear_layer::LinearLayer;
    /// let q_proj = LinearLayer::new(768, 768, DType::F32);
    /// let k_proj = LinearLayer::new(768, 768, DType::F32);
    /// let v_proj = LinearLayer::new(768, 768, DType::F32);
    /// let o_proj = LinearLayer::new(768, 768, DType::F32);
    /// let hidden = Array3::<f32>::zeros((2, 128, 768)); // [batch, seq, hidden]
    /// let mask = Array2::<f32>::ones((2, 128)); // [batch, seq]
    /// let position_bias = Array4::<f32>::zeros((1, 12, 128, 128)); // [1, heads, seq, seq]
    /// let attn = EncoderSelfAttention::new(768, 12, q_proj, k_proj, v_proj, o_proj);
    /// let output = attn.forward(&hidden, &mask, None)?;
    ///
    /// // With position bias (T5)
    /// let output = attn.forward(&hidden, &mask, Some(&position_bias))?;
    /// ```
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&Array4<f32>>,
        rope: Option<&RoPE>,
    ) -> Result<Array3<f32>> {
        let (batch, seq_len, _) = hidden_states.dim();
        let hidden_dim = self.num_heads * self.head_dim;

        // 1. Flatten & Project
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch * seq_len, hidden_dim))?;

        let q = self.q_proj.matmul(&hidden_2d);
        let k = self.k_proj.matmul(&hidden_2d);
        let v = self.v_proj.matmul(&hidden_2d);

        let mut q_3d = q.into_shape_with_order((batch, seq_len, hidden_dim))?;
        let mut k_3d = k.into_shape_with_order((batch, seq_len, hidden_dim))?;

        if let Some(r) = rope {
            // Apply to the full sequence (offset 0 for Encoder)
            // Note: Encoders (like Nomic) don't use GQA, so kv_heads == num_heads
            let (q_rot, k_rot) = r.apply_3d(
                &q_3d,
                &k_3d,
                self.num_heads,
                self.num_heads, // kv_heads same as heads
                0,              // position_offset
            )?;
            q_3d = q_rot;
            k_3d = k_rot;
        }

        // 2. Reshape & Permute to [B, H, S, D]
        let q_heads = q_3d
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        // K: [Batch, Seq, Hidden] -> [Batch, Seq, Heads, Dim] -> [Batch, Heads, Dim, Seq] (Transposed)
        let k_heads_t = k_3d
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 3, 1]) // 3 (Dim) before 1 (Seq) = Transpose
            .to_owned();

        let v_heads = v
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        // 3. Compute attention scores: Q @ K^T
        let mut scores = matmul_4d(&q_heads, &k_heads_t);

        // 4. Scale (BERT/BART) or not (T5)
        if self.scale_qk {
            scores.mapv_inplace(|x| x * self.scale_factor);
        }

        // 5. Add position bias if provided (T5/ALiBi)
        if let Some(bias) = position_bias {
            // bias is [1, heads, seq_q, seq_k], broadcasts to [batch, heads, seq_q, seq_k]
            scores = scores + bias;
        }

        // 6. Apply padding mask
        // scores = apply_attention_mask(scores, attention_mask);
        scores = crate::utils::apply_padding_mask(scores, attention_mask)?;

        // 7. Softmax
        softmax_4d_inplace(&mut scores);

        // 8. Compute context: Scores @ V
        let context = matmul_4d(&scores, &v_heads);

        // 9. Merge heads and output projection
        let context_contigous = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .to_owned();
        
        let context_flat =
            context_contigous.into_shape_with_order((batch * seq_len, hidden_dim))?;
        let output = self.out_proj.matmul(&context_flat.view());

        let output_3d = output
            .as_standard_layout()
            .into_owned() // Ensures we own the data
            .into_shape_with_order((batch, seq_len, hidden_dim))?;

        Ok(output_3d)
    }
}
