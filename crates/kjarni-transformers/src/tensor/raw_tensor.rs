use crate::tensor::dtype::DType;
use std::borrow::Cow;
use anyhow::{anyhow, Result};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};

/// A raw, untyped view into a tensor's bytes, shape, and dtype.
/// This is the primary output of a file loader before any CPU/GPU-specific processing.
pub struct TensorView<'a> {
    pub name: String,
    pub bytes: Cow<'a, [u8]>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl<'a> TensorView<'a> {
    /// **CPU Compatibility Layer**
    /// Converts raw bytes into a standard `ndarray::ArrayD<f32>`.
    /// This is a slow, memory-intensive operation and should only be used for debugging
    /// or for components that have not yet been updated to handle multiple dtypes.
    #[deprecated(
        since = "0.1.0",
        note = "This is a slow compatibility layer. Use ModelWeights::get_typed_tensor and handle specific dtypes instead."
    )]
    pub fn to_ndarray_f32(&self) -> Result<ArrayD<f32>> {
        let data: Vec<f32> = match self.dtype {
            DType::F32 => {
                // --- THIS IS THE FIX ---
                // Fast Path: Try a zero-copy cast if the memory is aligned.
                if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(&self.bytes) {
                    slice.to_vec()
                }
                // Safe Path: If not aligned, copy the bytes element by element.
                // This is slower but guaranteed to be safe.
                else {
                    self.bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect()
                }
            }
            DType::F16 => {
                let f16_data: &[f16] = bytemuck::try_cast_slice(&self.bytes)
                    .map_err(|e| anyhow!("Failed to cast bytes to f16 for tensor '{}': {}", self.name, e))?;
                f16_data.iter().map(|x| x.to_f32()).collect()
            }
            DType::BF16 => {
                let bf16_data: &[bf16] = bytemuck::try_cast_slice(&self.bytes)
                    .map_err(|e| anyhow!("Failed to cast bytes to bf16 for tensor '{}': {}", self.name, e))?;
                bf16_data.iter().map(|x| x.to_f32()).collect()
            }
            _ => {
                return Err(anyhow!(
                    "Automatic f32 conversion not supported for dtype {:?} on tensor '{}'",
                    self.dtype, self.name
                ));
            }
        };

        Ok(ArrayD::from_shape_vec(IxDyn(&self.shape), data)?)
    }
}