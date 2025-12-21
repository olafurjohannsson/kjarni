
use super::{safetensors_loader::SafeTensorsLoader, WeightLoader};
use crate::kernels::q_common::{BlockQ4_K, BlockQ8_0};
use crate::tensor::{
    dtype::DType,
    raw_tensor::RawTensor,
    {QuantizedMatrix, TypedCpuTensor},
};
use anyhow::{anyhow, Result};
use half::{bf16, f16};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use std::path::Path;

/// High-level interface for loading model weights from a directory.
///
/// This struct detects the weight format (`.safetensors`, `.gguf`, etc.) and provides
/// a consistent API for accessing tensors in various typed formats.
pub struct ModelWeights {
    loader: Box<dyn WeightLoader>,
    pub config_json: String,
}

impl ModelWeights {
    /// Creates a new `ModelWeights` loader from a given model directory path.
    pub fn new(path: &Path) -> Result<Self> {
        let loader: Box<dyn WeightLoader> = if path.join("model.safetensors").exists()
            || path.join("model.safetensors.index.json").exists()
        {
            Box::new(SafeTensorsLoader::new(path)?)
        } else {
            // Placeholder for GGUF or other formats
            return Err(anyhow!("No supported weight format found in {:?}", path));
        };

        let config_json = std::fs::read_to_string(path.join("config.json"))?;

        Ok(Self { loader, config_json })
    }

    /// Checks if a tensor with the given name exists in the model files.
    pub fn contains(&self, name: &str) -> bool {
        self.loader.contains(name)
    }

    /// Gets a raw, untyped view of a tensor's bytes.
    ///
    /// This is a low-level function intended for advanced use cases like direct
    /// GPU DMA transfers where you need a pointer to the raw data in the mmap'd file.
    pub fn get_raw(&self, name: &str) -> Result<RawTensor<'_>> {
        self.loader.get_raw(name)
    }

    /// Gets a typed tensor, preserving its original, memory-efficient dtype.
    ///
    /// This is the primary and most efficient method for loading tensors for CPU computation,
    /// as it avoids unnecessary type conversions and allocations.
    pub fn get_typed_tensor(&self, name: &str) -> Result<TypedCpuTensor> {
        let raw = self.loader.get_raw(name)?;
        raw_to_typed(raw)
    }

    // --- High-level accessors (use with care, as they may convert and allocate) ---

    /// Loads a tensor and converts it to `Array1<f32>`.
    ///
    /// This is a convenience method for small 1D tensors like biases or norms.
    /// It will fail if the tensor is not 1D or is a quantized matrix type.
    pub fn get_array1(&self, name: &str) -> Result<Array1<f32>> {
        let typed = self.get_typed_tensor(name)?;
        typed
            .to_array1_f32()
            .map_err(|e| anyhow!("Failed to load '{}' as Array1<f32>: {}", name, e))
    }

    /// Loads a tensor and converts it to `Array2<f32>`.
    ///
    /// **Warning:** This will perform a slow, full dequantization for quantized types
    /// and should only be used for debugging or for models that are entirely F32/BF16/F16.
    pub fn get_array2(&self, name: &str) -> Result<Array2<f32>> {
        let typed = self.get_typed_tensor(name)?;
        typed
            .to_array2_f32()
            .map_err(|e| anyhow!("Failed to load '{}' as Array2<f32>: {}", name, e))
    }
}

/// Converts a `RawTensor` into a `TypedCpuTensor`, performing the necessary parsing.
fn raw_to_typed(raw: RawTensor<'_>) -> Result<TypedCpuTensor> {
    let context_err = |e| anyhow!("Failed to cast bytes for tensor '{}' (dtype {:?}): {}", raw.name, raw.dtype, e);

    match raw.dtype {
        DType::F32 => {
            let data: Vec<f32> = bytemuck::try_cast_slice(&raw.bytes).map_err(context_err)?.to_vec();
            Ok(TypedCpuTensor::F32(ArrayD::from_shape_vec(IxDyn(&raw.shape), data)?))
        }
        DType::F16 => {
            let data: Vec<f16> = bytemuck::try_cast_slice(&raw.bytes).map_err(context_err)?.to_vec();
            Ok(TypedCpuTensor::F16(ArrayD::from_shape_vec(IxDyn(&raw.shape), data)?))
        }
        DType::BF16 => {
            let data: Vec<bf16> = bytemuck::try_cast_slice(&raw.bytes).map_err(context_err)?.to_vec();
            Ok(TypedCpuTensor::BF16(ArrayD::from_shape_vec(IxDyn(&raw.shape), data)?))
        }
        DType::Q8_0 => {
            if raw.shape.len() != 2 {
                return Err(anyhow!("Q8_0 tensor '{}' must be 2D", raw.name));
            }
            let blocks: Vec<BlockQ8_0> = bytemuck::try_cast_slice(&raw.bytes).map_err(context_err)?.to_vec();
            Ok(TypedCpuTensor::Q8_0(QuantizedMatrix {
                blocks,
                shape: [raw.shape[0], raw.shape[1]],
            }))
        }
        DType::Q4_K => {
            if raw.shape.len() != 2 {
                return Err(anyhow!("Q4_K tensor '{}' must be 2D", raw.name));
            }
            let blocks: Vec<BlockQ4_K> = bytemuck::try_cast_slice(&raw.bytes).map_err(context_err)?.to_vec();
            Ok(TypedCpuTensor::Q4_K(QuantizedMatrix {
                blocks,
                shape: [raw.shape[0], raw.shape[1]],
            }))
        }
        _ => Err(anyhow!("Unsupported dtype for typed conversion: {:?}", raw.dtype)),
    }
}