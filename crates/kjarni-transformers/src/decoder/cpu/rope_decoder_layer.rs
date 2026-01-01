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
    pub fn forward(
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
   pub fn forward_new(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        mut k_cache: ndarray::ArrayViewMut3<f32>,
        mut v_cache: ndarray::ArrayViewMut3<f32>,
    ) -> Result<Array3<f32>> { // Changed: Returns only hidden states
        let t_start = std::time::Instant::now();

        // 1. Pre-Norm (RMS)
        let norm_1 = self.attention_norm.forward(hidden_states);

        let t_norm1 = t_start.elapsed();

        // 2. Attention (In-Place Write)
        // We pass the destinations; attention writes to them and returns only attn_out
        let attn_out = self.attention.forward_new(
            &norm_1,
            Some(attention_mask),
            k_cache,
            v_cache,
            position_offset,
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
}