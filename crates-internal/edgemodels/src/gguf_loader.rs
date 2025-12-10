// src/models/llama/gguf_loader.rs

use anyhow::{anyhow, Result};
use gguf_rs::{get_gguf_container, GGUFModel, Tensor};
use ndarray::Array2;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Simple Q4_0 quantization support only
#[derive(Debug)]
pub struct GgufTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub kind: u32,  // GGML type
    pub data: Vec<u8>,
}

impl GgufTensor {
    /// Dequantize Q4_0 tensor to f32 array
    pub fn dequantize_q4_0(&self) -> Result<Array2<f32>> {
        if self.shape.len() != 2 {
            return Err(anyhow!("Expected 2D tensor, got shape {:?}", self.shape));
        }

        let rows = self.shape[0] as usize;
        let cols = self.shape[1] as usize;
        
        const BLOCK_SIZE: usize = 32;
        let mut result = vec![0.0f32; rows * cols];
        
        let num_blocks = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let block_bytes = 2 + BLOCK_SIZE / 2; // 2 bytes scale + 16 bytes data

        for row in 0..rows {
            for block_idx in 0..num_blocks {
                let block_offset = (row * num_blocks + block_idx) * block_bytes;
                
                if block_offset + block_bytes > self.data.len() {
                    break;
                }

                // Read scale (f16)
                let scale_bits = u16::from_le_bytes([
                    self.data[block_offset],
                    self.data[block_offset + 1],
                ]);
                let scale = half::f16::from_bits(scale_bits).to_f32();

                // Read 4-bit values (packed 2 per byte)
                for i in 0..BLOCK_SIZE {
                    let col = block_idx * BLOCK_SIZE + i;
                    if col >= cols {
                        break;
                    }
                    
                    let byte_idx = i / 2;
                    let byte = self.data[block_offset + 2 + byte_idx];
                    
                    // Extract 4-bit value and convert to signed
                    let q_val = if i % 2 == 0 {
                        (byte & 0x0F) as i8 - 8  // Low nibble
                    } else {
                        ((byte >> 4) & 0x0F) as i8 - 8  // High nibble
                    };
                    
                    result[row * cols + col] = (q_val as f32) * scale;
                }
            }
        }

        Array2::from_shape_vec((rows, cols), result)
            .map_err(|e| anyhow!("Shape error: {}", e))
    }
}

pub struct GgufLoader {
    model: GGUFModel,
    file_path: PathBuf,
    tensors_cache: HashMap<String, GgufTensor>,
}

impl GgufLoader {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        log::info!("Loading GGUF file: {}", path.display());

        // Fix: Convert Path to &str
        let path_str = path.to_str()
            .ok_or_else(|| anyhow!("Invalid UTF-8 in path"))?;

        let mut container = get_gguf_container(path_str)
            .map_err(|e| anyhow!("Failed to load GGUF: {}", e))?;

        let model = container.decode()
            .map_err(|e| anyhow!("Failed to decode GGUF: {}", e))?;

        log::info!("GGUF Model Info:");
        log::info!("  Family: {}", model.model_family());
        log::info!("  Parameters: {}", model.model_parameters());
        log::info!("  File Type: {}", model.file_type());
        log::info!("  Tensors: {}", model.num_tensor());

        Ok(Self {
            model,
            file_path: path.to_path_buf(),
            tensors_cache: HashMap::new(),
        })
    }

    /// Load a tensor from the GGUF file
    fn load_tensor(&mut self, tensor_info: &Tensor) -> Result<GgufTensor> {
        let mut file = File::open(&self.file_path)?;
        
        // Seek to tensor data
        file.seek(SeekFrom::Start(tensor_info.offset))?;
        
        // Read tensor data
        let mut data = vec![0u8; tensor_info.size as usize];
        file.read_exact(&mut data)?;

        Ok(GgufTensor {
            name: tensor_info.name.clone(),
            shape: tensor_info.shape.clone(),
            kind: tensor_info.kind,
            data,
        })
    }

    /// Get tensor by name and dequantize
    pub fn get_array2(&mut self, name: &str) -> Result<Array2<f32>> {
        // Check cache first
        if let Some(tensor) = self.tensors_cache.get(name) {
            return tensor.dequantize_q4_0();
        }

        // Find tensor info
        let tensor_info = self.model.tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;

        // Load and cache
        let tensor = self.load_tensor(&tensor_info.clone())?;
        let result = tensor.dequantize_q4_0()?;
        self.tensors_cache.insert(name.to_string(), tensor);
        
        Ok(result)
    }

    /// Map HuggingFace Llama names to GGUF names
    pub fn map_llama_tensor(&mut self, llama_name: &str) -> Result<Array2<f32>> {
        // Fix: Make all arms return String
        let gguf_name = match llama_name {
            // Embeddings
            "model.embed_tokens.weight" => "token_embd.weight".to_string(),
            
            // Output
            "model.norm.weight" => "output_norm.weight".to_string(),
            "lm_head.weight" => "output.weight".to_string(),
            
            // Layer mappings
            name if name.contains("layers") => {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() < 3 {
                    return Err(anyhow!("Invalid layer name: {}", name));
                }
                
                let layer_num = parts[2];
                
                // Build GGUF tensor name
                let component = if name.contains("self_attn.q_proj") {
                    format!("blk.{}.attn_q.weight", layer_num)
                } else if name.contains("self_attn.k_proj") {
                    format!("blk.{}.attn_k.weight", layer_num)
                } else if name.contains("self_attn.v_proj") {
                    format!("blk.{}.attn_v.weight", layer_num)
                } else if name.contains("self_attn.o_proj") {
                    format!("blk.{}.attn_output.weight", layer_num)
                } else if name.contains("mlp.gate_proj") {
                    format!("blk.{}.ffn_gate.weight", layer_num)
                } else if name.contains("mlp.up_proj") {
                    format!("blk.{}.ffn_up.weight", layer_num)
                } else if name.contains("mlp.down_proj") {
                    format!("blk.{}.ffn_down.weight", layer_num)
                } else if name.contains("input_layernorm") {
                    format!("blk.{}.attn_norm.weight", layer_num)
                } else if name.contains("post_attention_layernorm") {
                    format!("blk.{}.ffn_norm.weight", layer_num)
                } else {
                    return Err(anyhow!("Unknown component: {}", name));
                };
                
                component
            }
            
            _ => return Err(anyhow!("Unknown mapping: {}", llama_name)),
        };

        self.get_array2(&gguf_name)
    }

    /// List all tensors in the GGUF file
    pub fn list_tensors(&self) {
        println!("\n=== GGUF Tensors ===");
        for tensor in self.model.tensors() {
            println!("  {} - shape: {:?}, kind: {}, size: {} bytes",
                     tensor.name, tensor.shape, tensor.kind, tensor.size);
        }
        println!("====================\n");
    }
}