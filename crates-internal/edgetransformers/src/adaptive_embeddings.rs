// gpu_ops/blocks/embeddings/adaptive.rs
use anyhow::Result;
use ndarray::{Array2, Array3};
use std::sync::Arc;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;

#[derive(Debug, Clone)]
pub enum EmbeddingMode {
    AlwaysCpu,
    AlwaysGpu,
    Adaptive {
        threshold: usize,
        memory_pressure_threshold: f32,
    },
}

pub struct EmbeddingSelector {
    mode: EmbeddingMode,
    embedding_size_bytes: usize,
    max_buffer_binding_size: u32,
}

impl EmbeddingSelector {
    pub fn new(
        context: &WgpuContext,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Self {
        let embedding_size_bytes = vocab_size * hidden_size * 4;
        let max_buffer_binding_size = context.memory_info.max_storage_buffer_binding_size;

        // Determine mode based on size and available memory
        let mode = if embedding_size_bytes > max_buffer_binding_size as usize {
            log::warn!(
                "Embeddings ({:.0}MB) exceed GPU buffer limit ({:.0}MB) - forcing CPU mode",
                embedding_size_bytes as f32 / 1_048_576.0,
                max_buffer_binding_size as f32 / 1_048_576.0
            );
            EmbeddingMode::AlwaysCpu
        } else if embedding_size_bytes > 256_000_000 {
            // Large embeddings: adaptive mode
            EmbeddingMode::Adaptive {
                threshold: 256,
                memory_pressure_threshold: 0.8,
            }
        } else {
            // Small embeddings: always GPU
            EmbeddingMode::AlwaysGpu
        };

        log::info!(
            "Embedding mode: {:?} (size: {:.0}MB)",
            mode,
            embedding_size_bytes as f32 / 1_048_576.0
        );

        Self {
            mode,
            embedding_size_bytes,
            max_buffer_binding_size,
        }
    }

    pub fn should_use_gpu(
        &self,
        batch_size: usize,
        seq_len: usize,
        context: &WgpuContext,
    ) -> bool {
        let token_count = batch_size * seq_len;

        match &self.mode {
            EmbeddingMode::AlwaysCpu => false,
            EmbeddingMode::AlwaysGpu => true,
            EmbeddingMode::Adaptive { threshold, memory_pressure_threshold } => {
                // Check token count threshold
                if token_count < *threshold {
                    return false; // Too small, CPU is faster
                }

                // Check memory pressure
                if let Some(available) = context.memory_info.available_memory {
                    let used = context.get_allocated_memory() as u64;
                    let pressure = used as f32 / available as f32;

                    if pressure > *memory_pressure_threshold {
                        log::debug!("Memory pressure {:.1}% - using CPU embeddings", pressure * 100.0);
                        return false;
                    }
                }

                true
            }
        }
    }
}