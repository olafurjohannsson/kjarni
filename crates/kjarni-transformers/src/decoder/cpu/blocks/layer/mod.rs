use crate::decoder::cpu::{
    DecoderAttention,
};
use crate::feedforward::FeedForward;
use crate::normalization::Normalization;
use crate::rope::RoPE;
use anyhow::Result;
use ndarray::{Array2, Array3};
use std::sync::Arc;

#[cfg(test)]
mod tests;

pub struct DecoderLayer {
    pub self_attn: DecoderAttention,
    pub self_attn_layer_norm: Normalization,
    pub feedforward: FeedForward,
    pub ffn_layer_norm: Normalization,
    pub is_prenorm: bool,
    pub rope: Option<Arc<RoPE>>,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        if self.is_prenorm {
            self.forward_prenorm(hidden_states, attention_mask, past_kv)
        } else {
            self.forward_postnorm(hidden_states, attention_mask, past_kv)
        }
    }
     /// Llama / Mistral / Phi (Pre-Norm)
    /// Flow: x = x + Attn(Norm(x))
    fn forward_prenorm(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        
        let residual = hidden_states.clone();
        
        // 1. Norm before Attn
        let ln1_out = self.self_attn_layer_norm.forward(hidden_states);

        // 2. Attn
        // REFACTOR: Direct call. DecoderAttention handles RoPE and Cache.
        let (attn_out, new_k, new_v) = self.self_attn.forward(
            &ln1_out,
            Some(attention_mask),
            past_kv,
            self.rope.as_deref(), 
        )?;

        // 3. Residual connection
        let attn_block_output = residual + attn_out;

        let residual = attn_block_output.clone();
        
        // 4. Norm before FFN
        let ln2_out = self.ffn_layer_norm.forward(&attn_block_output);

        // 5. FFN
        let ffn_out = self.feedforward.forward(&ln2_out)?;

        // 6. Final Residual
        let final_output = residual + ffn_out;

        Ok((final_output, (new_k, new_v)))
    }

    /// GPT-2 (Post-Norm)
    /// Flow: x = Norm(x + Attn(x))
    fn forward_postnorm(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        let residual = hidden_states.clone();

        // 1. Attn (No Norm before)
        // REFACTOR: We removed the 60 lines of manual projection logic.
        // DecoderAttention::forward does exactly what that manual code did.
        let (attn_out, new_k, new_v) = self.self_attn.forward(
            hidden_states,
            Some(attention_mask),
            past_kv,
            self.rope.as_deref(), // Pass RoPE if it exists (usually None for GPT-2)
        )?;

        // 2. Residual + Norm
        let hidden = residual + attn_out;
        let hidden = self.self_attn_layer_norm.forward(&hidden);

        // 3. FFN
        let residual = hidden.clone();
        let ffn_out = self.feedforward.forward(&hidden)?;
        
        // 4. Residual + Norm
        let hidden = residual + ffn_out;
        let output = self.ffn_layer_norm.forward(&hidden);

        Ok((output, (new_k, new_v)))
    }
}

