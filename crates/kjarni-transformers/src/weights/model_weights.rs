//! Model weights loader with dtype-aware loading

use anyhow::{anyhow, Result};
use half::{bf16, f16};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use std::path::Path;

use crate::tensor::{DType, RawTensor, TypedCpuTensor};
use super::{WeightLoader, SafeTensorsLoader};

/// High-level interface for loading model weights
pub struct ModelWeights {
    loader: Box<dyn WeightLoader>,
    pub config_json: String,
}

impl ModelWeights {
    pub fn new(path: &Path) -> Result<Self> {
        let loader: Box<dyn WeightLoader> = if path.join("model.safetensors").exists()
            || path.join("model.safetensors.index.json").exists()
        {
            Box::new(SafeTensorsLoader::new(path)?)
        } else {
            return Err(anyhow!("No supported weight format found in {:?}", path));
        };

        let config_json = std::fs::read_to_string(path.join("config.json"))?;

        Ok(Self { loader, config_json })
    }

    /// Check if tensor exists
    pub fn contains(&self, name: &str) -> bool {
        self.loader.contains(name)
    }

    // =========================================================================
    // Low-level access (for advanced use cases)
    // =========================================================================

    /// Get raw tensor (for GPU DMA or custom processing)
    pub fn get_raw(&self, name: &str) -> Result<RawTensor<'_>> {
        self.loader.get_raw(name)
    }

    /// Get typed tensor (preserves original dtype)
    pub fn get_typed_tensor(&self, name: &str) -> Result<TypedCpuTensor> {
        let raw = self.get_raw(name)?;
        raw_to_typed(raw)
    }

    // =========================================================================
    // High-level access (always returns F32, converts if needed)
    // =========================================================================

    /// Load as F32 Array1 (for biases, layer norm weights)
    /// Converts from F16/BF16 if needed
    pub fn get_array1(&self, name: &str) -> Result<Array1<f32>> {
        let typed = self.get_typed_tensor(name)?;
        typed.to_array1_f32()
            .map_err(|e| anyhow!("Failed to load '{}' as Array1<f32>: {}", name, e))
    }

    /// Load as F32 Array2 (for weight matrices)
    /// Converts from F16/BF16 if needed
    pub fn get_array2(&self, name: &str) -> Result<Array2<f32>> {
        let typed = self.get_typed_tensor(name)?;
        typed.to_array2_f32()
            .map_err(|e| anyhow!("Failed to load '{}' as Array2<f32>: {}", name, e))
    }

    /// Load as F32 ArrayD (generic shape)
    pub fn get_arrayd(&self, name: &str) -> Result<ArrayD<f32>> {
        let typed = self.get_typed_tensor(name)?;
        typed.to_arrayd_f32()
            .map_err(|e| anyhow!("Failed to load '{}' as ArrayD<f32>: {}", name, e))
    }

    // =========================================================================
    // Dtype-preserving loading (for memory-constrained scenarios)
    // =========================================================================

    /// Load tensor keeping original dtype
    /// Use this for large tensors where memory matters
    pub fn get_typed_array2(&self, name: &str) -> Result<TypedCpuTensor> {
        let typed = self.get_typed_tensor(name)?;
        // Validate it's 2D
        if typed.shape().len() != 2 {
            return Err(anyhow!(
                "Tensor '{}' has {} dimensions, expected 2",
                name,
                typed.shape().len()
            ));
        }
        Ok(typed)
    }

    /// Load with optional dtype override
    /// - None: keep original dtype
    /// - Some(DType::F32): convert to F32
    /// - Some(DType::BF16): convert to BF16 (useful for downcasting F32 models)
    pub fn get_array2_as(&self, name: &str, dtype: Option<DType>) -> Result<TypedCpuTensor> {
        let typed = self.get_typed_tensor(name)?;

        match dtype {
            None => Ok(typed),
            Some(DType::F32) => Ok(TypedCpuTensor::F32(typed.to_arrayd_f32()?)),
            Some(DType::BF16) => {
                let f32_arr = typed.to_arrayd_f32()?;
                let bf16_arr = f32_arr.mapv(|v| bf16::from_f32(v));
                Ok(TypedCpuTensor::BF16(bf16_arr))
            }
            Some(DType::F16) => {
                let f32_arr = typed.to_arrayd_f32()?;
                let f16_arr = f32_arr.mapv(|v| f16::from_f32(v));
                Ok(TypedCpuTensor::F16(f16_arr))
            }
            Some(other) => Err(anyhow!("Cannot convert to {:?}", other)),
        }
    }
}

/// Convert RawTensor to TypedCpuTensor
fn raw_to_typed(raw: RawTensor<'_>) -> Result<TypedCpuTensor> {
    match raw.dtype {
        DType::F32 => {
            let data: Vec<f32> = if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(&raw.bytes) {
                slice.to_vec()
            } else {
                raw.bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            Ok(TypedCpuTensor::F32(ArrayD::from_shape_vec(IxDyn(&raw.shape), data)?))
        }
        DType::F16 => {
            let data: Vec<f16> = bytemuck::try_cast_slice::<u8, f16>(&raw.bytes)
                .map_err(|e| anyhow!("Failed to cast to f16: {}", e))?
                .to_vec();
            Ok(TypedCpuTensor::F16(ArrayD::from_shape_vec(IxDyn(&raw.shape), data)?))
        }
        DType::BF16 => {
            let data: Vec<bf16> = bytemuck::try_cast_slice::<u8, bf16>(&raw.bytes)
                .map_err(|e| anyhow!("Failed to cast to bf16: {}", e))?
                .to_vec();
            Ok(TypedCpuTensor::BF16(ArrayD::from_shape_vec(IxDyn(&raw.shape), data)?))
        }
        DType::Q4_K => {
            // TODO: Parse Q4_K format properly
            Err(anyhow!("Q4_K loading not yet implemented"))
        }
        other => Err(anyhow!("Unsupported dtype: {:?}", other)),
    }
}