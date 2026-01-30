//! Pre-allocated buffers for encoder computations.
//!
//! These buffers eliminate allocation overhead in the hot path by reusing
//! memory across forward passes.

use ndarray::{Array2, Array4};

/// Pre-allocated buffers for encoder layer computations.
///
/// Allocated once and reused across all layers and forward passes.
/// Eliminates allocation overhead in the hot path.
///
/// # Memory Layout
///
/// | Buffer | Shape | Usage |
/// |--------|-------|-------|
/// | q, k, v | [max_tokens, hidden] | QKV projection outputs |
/// | qkv_scratch | [max_tokens, 3*hidden] | Fused QKV intermediate (optional) |
/// | attn_scores | [max_batch, num_heads, max_seq, max_seq] | Attention scores & probs |
/// | attn_context | [max_batch, num_heads, max_seq, head_dim] | Attention context |
/// | attn_output | [max_tokens, hidden] | Attention output |
/// | ffn_intermediate | [max_tokens, intermediate_dim] | FFN hidden activations |
/// | ffn_output | [max_tokens, hidden] | FFN output |
///
/// # Example
///
/// ```ignore
/// let mut buffers = EncoderBuffers::new(
///     32,    // max_batch
///     128,   // max_seq
///     768,   // hidden
///     12,    // num_heads
///     3072,  // intermediate_dim
///     false, // use_fused_qkv (hidden > 512)
/// );
///
/// // Use in forward pass
/// encoder_layer.forward_noalloc(hidden, mask, &mut buffers)?;
/// ```
pub struct EncoderBuffers {
    // =========================================================================
    // QKV Projection Outputs
    // =========================================================================
    
    /// Q projection output [max_tokens, hidden]
    pub q: Array2<f32>,
    /// K projection output [max_tokens, hidden]
    pub k: Array2<f32>,
    /// V projection output [max_tokens, hidden]
    pub v: Array2<f32>,
    /// Scratch buffer for fused QKV intermediate [max_tokens, 3*hidden]
    /// Only allocated if use_fused_qkv=true at construction
    pub qkv_scratch: Option<Array2<f32>>,

    // =========================================================================
    // Attention Intermediates
    // =========================================================================
    
    /// Attention scores [max_batch, num_heads, max_seq, max_seq]
    /// Also used for attention probabilities after in-place softmax
    pub attn_scores: Array4<f32>,
    
    /// Attention context [max_batch, num_heads, max_seq, head_dim]
    pub attn_context: Array4<f32>,

    // =========================================================================
    // Layer Outputs
    // =========================================================================
    
    /// Attention output [max_tokens, hidden]
    pub attn_output: Array2<f32>,

    /// FFN intermediate activations [max_tokens, intermediate_dim]
    pub ffn_intermediate: Array2<f32>,
    
    /// FFN output [max_tokens, hidden]
    pub ffn_output: Array2<f32>,

    /// Layer norm scratch buffer [max_tokens, hidden]
    pub norm_scratch: Array2<f32>,

    /// Scratch buffer for merged heads [max_tokens, hidden]
    pub merge_scratch: Array2<f32>,

    // =========================================================================
    // Configuration (stored for validation and resizing)
    // =========================================================================
    
    max_batch: usize,
    max_seq: usize,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    use_fused_qkv: bool,
}

impl EncoderBuffers {
    /// Creates new encoder buffers.
    ///
    /// # Arguments
    ///
    /// * `max_batch` - Maximum batch size
    /// * `max_seq` - Maximum sequence length
    /// * `hidden` - Hidden dimension
    /// * `num_heads` - Number of attention heads
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `use_fused_qkv` - Whether to allocate scratch for fused QKV projection
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For BERT-base with batch=32, seq=128
    /// let buffers = EncoderBuffers::new(
    ///     32,    // max_batch
    ///     128,   // max_seq
    ///     768,   // hidden
    ///     12,    // num_heads
    ///     3072,  // intermediate_dim
    ///     false, // separate QKV (hidden > 512)
    /// );
    /// ```
    pub fn new(
        max_batch: usize,
        max_seq: usize,
        hidden: usize,
        num_heads: usize,
        intermediate_dim: usize,
        use_fused_qkv: bool,
    ) -> Self {
        let max_tokens = max_batch * max_seq;
        let head_dim = hidden / num_heads;

        Self {
            // QKV outputs
            q: Array2::zeros((max_tokens, hidden)),
            k: Array2::zeros((max_tokens, hidden)),
            v: Array2::zeros((max_tokens, hidden)),
            qkv_scratch: if use_fused_qkv {
                Some(Array2::zeros((max_tokens, 3 * hidden)))
            } else {
                None
            },
            
            // Attention intermediates
            attn_scores: Array4::zeros((max_batch, num_heads, max_seq, max_seq)),
            attn_context: Array4::zeros((max_batch, num_heads, max_seq, head_dim)),
            
            // Layer outputs
            attn_output: Array2::zeros((max_tokens, hidden)),
            ffn_intermediate: Array2::zeros((max_tokens, intermediate_dim)),
            ffn_output: Array2::zeros((max_tokens, hidden)),
            norm_scratch: Array2::zeros((max_tokens, hidden)),
            merge_scratch: Array2::zeros((max_tokens, hidden)),

            // Config
            max_batch,
            max_seq,
            hidden,
            num_heads,
            head_dim,
            intermediate_dim,
            use_fused_qkv,
        }
    }

    /// Creates buffers with automatic fused/separate selection based on hidden size.
    ///
    /// Uses fused QKV for hidden <= 512, separate for larger models.
    pub fn new_auto(
        max_batch: usize,
        max_seq: usize,
        hidden: usize,
        num_heads: usize,
        intermediate_dim: usize,
    ) -> Self {
        let use_fused_qkv = hidden <= 512;
        Self::new(max_batch, max_seq, hidden, num_heads, intermediate_dim, use_fused_qkv)
    }

    /// Ensures buffers have capacity for the given dimensions.
    ///
    /// # Behavior
    ///
    /// - **Debug mode**: Panics if capacity exceeded (catches bugs during development)
    /// - **Release mode**: Logs warning and resizes buffers
    ///
    /// # Panics
    ///
    /// In debug mode, panics if batch > max_batch or seq > max_seq.
    pub fn ensure_capacity(&mut self, batch: usize, seq: usize) {
        if batch > self.max_batch || seq > self.max_seq {
            #[cfg(debug_assertions)]
            panic!(
                "EncoderBuffers capacity exceeded: need (batch={}, seq={}), have (batch={}, seq={}). \
                 Initialize with larger capacity.",
                batch, seq, self.max_batch, self.max_seq
            );

            #[cfg(not(debug_assertions))]
            {
                log::warn!(
                    "EncoderBuffers resizing: (batch={}, seq={}) -> (batch={}, seq={})",
                    self.max_batch, self.max_seq, batch, seq
                );
                *self = Self::new(
                    batch.max(self.max_batch),
                    seq.max(self.max_seq),
                    self.hidden,
                    self.num_heads,
                    self.intermediate_dim,
                    self.use_fused_qkv,
                );
            }
        }
    }

    /// Ensures buffers have capacity for the given token count.
    ///
    /// This is a convenience method when you only have the flattened token count.
    /// Assumes batch=1 and seq=tokens for capacity check.
    pub fn ensure_capacity_tokens(&mut self, tokens: usize) {
        let max_tokens = self.max_batch * self.max_seq;
        if tokens > max_tokens {
            #[cfg(debug_assertions)]
            panic!(
                "EncoderBuffers token capacity exceeded: need {}, have {}. \
                 Initialize with larger capacity.",
                tokens, max_tokens
            );

            #[cfg(not(debug_assertions))]
            {
                // Resize assuming batch=1
                self.ensure_capacity(1, tokens);
            }
        }
    }

    /// Validates that buffer dimensions match expected model configuration.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if dimensions don't match.
    #[inline]
    pub fn validate_dimensions(
        &self,
        batch: usize,
        seq: usize,
        hidden: usize,
        num_heads: usize,
        intermediate_dim: usize,
    ) {
        #[cfg(debug_assertions)]
        {
            if batch > self.max_batch {
                panic!(
                    "Batch size {} exceeds buffer capacity {}",
                    batch, self.max_batch
                );
            }
            if seq > self.max_seq {
                panic!(
                    "Sequence length {} exceeds buffer capacity {}",
                    seq, self.max_seq
                );
            }
            if hidden != self.hidden {
                panic!(
                    "Hidden dimension mismatch: expected {}, buffer has {}",
                    hidden, self.hidden
                );
            }
            if num_heads != self.num_heads {
                panic!(
                    "Number of heads mismatch: expected {}, buffer has {}",
                    num_heads, self.num_heads
                );
            }
            if intermediate_dim != self.intermediate_dim {
                panic!(
                    "Intermediate dimension mismatch: expected {}, buffer has {}",
                    intermediate_dim, self.intermediate_dim
                );
            }
        }
        // In release mode, skip validation for performance
        let _ = (batch, seq, hidden, num_heads, intermediate_dim);
    }

    /// Returns whether fused QKV scratch is available.
    #[inline]
    pub fn has_qkv_scratch(&self) -> bool {
        self.qkv_scratch.is_some()
    }

    /// Returns the maximum batch size.
    #[inline]
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Returns the maximum sequence length.
    #[inline]
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Returns the maximum token capacity (max_batch * max_seq).
    #[inline]
    pub fn max_tokens(&self) -> usize {
        self.max_batch * self.max_seq
    }

    /// Returns the hidden dimension.
    #[inline]
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Returns the number of attention heads.
    #[inline]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Returns the head dimension.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Returns the intermediate dimension.
    #[inline]
    pub fn intermediate_dim(&self) -> usize {
        self.intermediate_dim
    }

    /// Returns whether fused QKV is enabled.
    #[inline]
    pub fn use_fused_qkv(&self) -> bool {
        self.use_fused_qkv
    }
    
    /// Returns total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let max_tokens = self.max_batch * self.max_seq;
        
        // 2D buffers
        let qkv = max_tokens * self.hidden * 4 * 3; // q, k, v
        let outputs = max_tokens * self.hidden * 4 * 2; // attn_output, ffn_output
        let ffn = max_tokens * self.intermediate_dim * 4;
        let norm = max_tokens * self.hidden * 4;
        let qkv_scratch = if self.use_fused_qkv {
            max_tokens * 3 * self.hidden * 4
        } else {
            0
        };
        
        // 4D buffers
        let attn_scores = self.max_batch * self.num_heads * self.max_seq * self.max_seq * 4;
        let attn_context = self.max_batch * self.num_heads * self.max_seq * self.head_dim * 4;

        qkv + outputs + ffn + qkv_scratch + attn_scores + attn_context + norm
    }
    
    /// Returns memory usage breakdown as a formatted string.
    pub fn memory_breakdown(&self) -> String {
        let max_tokens = self.max_batch * self.max_seq;
        
        let qkv = max_tokens * self.hidden * 4 * 3;
        let qkv_scratch = if self.use_fused_qkv {
            max_tokens * 3 * self.hidden * 4
        } else {
            0
        };
        let attn_scores = self.max_batch * self.num_heads * self.max_seq * self.max_seq * 4;
        let attn_context = self.max_batch * self.num_heads * self.max_seq * self.head_dim * 4;
        let attn_output = max_tokens * self.hidden * 4;
        let ffn_inter = max_tokens * self.intermediate_dim * 4;
        let ffn_out = max_tokens * self.hidden * 4;
        let norm = max_tokens * self.hidden * 4;

        format!(
            "Q/K/V: {:.2} MB, QKV scratch: {:.2} MB, Attn scores: {:.2} MB, \
             Attn context: {:.2} MB, Attn output: {:.2} MB, FFN inter: {:.2} MB, FFN out: {:.2} MB, Norm: {:.2} MB",
            qkv as f64 / 1024.0 / 1024.0,
            qkv_scratch as f64 / 1024.0 / 1024.0,
            attn_scores as f64 / 1024.0 / 1024.0,
            attn_context as f64 / 1024.0 / 1024.0,
            attn_output as f64 / 1024.0 / 1024.0,
            ffn_inter as f64 / 1024.0 / 1024.0,
            ffn_out as f64 / 1024.0 / 1024.0,
            norm as f64 / 1024.0 / 1024.0,
        )
    }
}

impl std::fmt::Debug for EncoderBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderBuffers")
            .field("max_batch", &self.max_batch)
            .field("max_seq", &self.max_seq)
            .field("hidden", &self.hidden)
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .field("intermediate_dim", &self.intermediate_dim)
            .field("use_fused_qkv", &self.use_fused_qkv)
            .field("memory_mb", &(self.memory_usage() as f64 / 1024.0 / 1024.0))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buffers = EncoderBuffers::new(32, 128, 768, 12, 3072, false);
        
        assert_eq!(buffers.q.dim(), (32 * 128, 768));
        assert_eq!(buffers.k.dim(), (32 * 128, 768));
        assert_eq!(buffers.v.dim(), (32 * 128, 768));
        assert!(buffers.qkv_scratch.is_none());
        assert_eq!(buffers.attn_scores.dim(), (32, 12, 128, 128));
        assert_eq!(buffers.attn_context.dim(), (32, 12, 128, 64)); // head_dim = 768/12 = 64
        assert_eq!(buffers.attn_output.dim(), (32 * 128, 768));
        assert_eq!(buffers.ffn_intermediate.dim(), (32 * 128, 3072));
        assert_eq!(buffers.ffn_output.dim(), (32 * 128, 768));
    }

    #[test]
    fn test_buffer_creation_fused() {
        let buffers = EncoderBuffers::new(32, 128, 384, 6, 1536, true);
        
        assert!(buffers.qkv_scratch.is_some());
        assert_eq!(buffers.qkv_scratch.as_ref().unwrap().dim(), (32 * 128, 3 * 384));
    }

    #[test]
    fn test_auto_selection() {
        // Small model -> fused
        let small = EncoderBuffers::new_auto(32, 128, 384, 6, 1536);
        assert!(small.use_fused_qkv());
        assert!(small.has_qkv_scratch());

        // Large model -> separate
        let large = EncoderBuffers::new_auto(32, 128, 768, 12, 3072);
        assert!(!large.use_fused_qkv());
        assert!(!large.has_qkv_scratch());
    }

    #[test]
    fn test_memory_usage() {
        let buffers = EncoderBuffers::new(1, 128, 768, 12, 3072, false);
        
        // Calculate expected
        let max_tokens = 128;
        let qkv = max_tokens * 768 * 4 * 3;
        let outputs = max_tokens * 768 * 4 * 2; // attn_output, ffn_output
        let ffn = max_tokens * 3072 * 4;
        let norm = max_tokens * 768 * 4; // norm_scratch - THIS WAS MISSING
        let attn_scores = 1 * 12 * 128 * 128 * 4;
        let attn_context = 1 * 12 * 128 * 64 * 4;
        let expected = qkv + outputs + ffn + norm + attn_scores + attn_context;
        
        assert_eq!(buffers.memory_usage(), expected);
        
        println!("Memory for batch=1, seq=128, BERT-base: {:.2} MB", 
                buffers.memory_usage() as f64 / 1024.0 / 1024.0);
        println!("Breakdown: {}", buffers.memory_breakdown());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "capacity exceeded")]
    fn test_capacity_panic_debug() {
        let mut buffers = EncoderBuffers::new(8, 64, 768, 12, 3072, false);
        buffers.ensure_capacity(16, 64); // batch exceeds
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "capacity exceeded")]
    fn test_seq_capacity_panic_debug() {
        let mut buffers = EncoderBuffers::new(8, 64, 768, 12, 3072, false);
        buffers.ensure_capacity(8, 128); // seq exceeds
    }
}