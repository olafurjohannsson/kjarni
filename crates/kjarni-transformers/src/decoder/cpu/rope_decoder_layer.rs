use std::sync::Arc;
use anyhow::Result;
use ndarray::{Array2, Array3};

use crate::{Normalization, decoder::cpu::DecoderAttention, feedforward::SwiGluFeedForward, rope::RoPE};





pub struct CpuRoPEDecoderLayer {
    pub attention: DecoderAttention,
    pub feed_forward: SwiGluFeedForward,
    pub attention_norm: Normalization,
    pub ffn_norm: Normalization,
    pub rope: Arc<RoPE>,
}

impl CpuRoPEDecoderLayer {
    pub fn forward_2(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        // Changed: Explicit split views instead of Option<Tuple>
        past_k: ndarray::ArrayView3<f32>,
        past_v: ndarray::ArrayView3<f32>,
        dest_k: ndarray::ArrayViewMut3<f32>,
        dest_v: ndarray::ArrayViewMut3<f32>,
    ) -> Result<Array3<f32>> { // Changed: Returns only hidden states
        let t_start = std::time::Instant::now();

        // 1. Pre-Norm (RMS)
        let norm_1 = self.attention_norm.forward(hidden_states);

        let t_norm1 = t_start.elapsed();

        // 2. Attention (In-Place Write)
        // We pass the destinations; attention writes to them and returns only attn_out
        let attn_out = self.attention.forward_2(
            &norm_1,
            Some(attention_mask),
            past_k,
            past_v,
            dest_k,
            dest_v,
            Some(&self.rope),
        )?;

        let t_attn = t_start.elapsed() - t_norm1;

        let residual_1 = hidden_states + &attn_out;

        // 3. Pre-Norm (RMS)
        let norm_2 = self.ffn_norm.forward(&residual_1);

        let t_norm2 = t_start.elapsed() - t_attn - t_norm1;

        // 4. FeedForward
        let ffn_out = self.feed_forward.forward(&norm_2)?;

        let t_ffn = t_start.elapsed() - t_norm2 - t_attn - t_norm1;

        let output = residual_1 + ffn_out;

        // Log if slow (> 10ms)
        if t_start.elapsed().as_millis() > 10 {
            log::info!(
                "Layer Perf: Norm1: {:?}, Attn: {:?}, Norm2: {:?}, FFN: {:?}",
                t_norm1,
                t_attn,
                t_norm2,
                t_ffn
            );
        }

        Ok(output)
    }
    // Optimized forward pass
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let t_start = std::time::Instant::now();

        // 1. Pre-Norm (RMS)
        let norm_1 = self.attention_norm.forward(hidden_states);

        let t_norm1 = t_start.elapsed();

        // 2. Attention
        let (attn_out, new_k, new_v) =
            self.attention
                .forward(&norm_1, Some(attention_mask), past_kv, Some(&self.rope))?;

        let t_attn = t_start.elapsed() - t_norm1;

        let residual_1 = hidden_states + &attn_out;

        // 3. Pre-Norm (RMS)
        let norm_2 = self.ffn_norm.forward(&residual_1);

        let t_norm2 = t_start.elapsed() - t_attn - t_norm1;

        // 4. FeedForward
        let ffn_out = self.feed_forward.forward(&norm_2)?;

        let t_ffn = t_start.elapsed() - t_norm2 - t_attn - t_norm1;

        let output = residual_1 + ffn_out;

        // Log if slow (> 10ms)
        if t_start.elapsed().as_millis() > 10 {
            log::info!(
                "Layer Perf: Norm1: {:?}, Attn: {:?}, Norm2: {:?}, FFN: {:?}",
                t_norm1,
                t_attn,
                t_norm2,
                t_ffn
            );
        }

        Ok((output, new_k, new_v))
    }
}