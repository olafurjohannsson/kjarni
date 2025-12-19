use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// Standard 32-bit float
    F32,
    /// 16-bit float (IEEE 754 half-precision)
    F16,
    /// 16-bit brain float (more range, less precision than F16)
    BF16,
    /// 32-bit unsigned integer, primarily for token IDs
    U32,
    /// 4-bit block-quantized with K-quants (from GGUF)
    Q4_K,
}


impl DType {
    /// Maps a safetensors::Dtype to our internal DType.
    pub fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self> {
        match dtype {
            safetensors::Dtype::F32 => Ok(DType::F32),
            safetensors::Dtype::F16 => Ok(DType::F16),
            safetensors::Dtype::BF16 => Ok(DType::BF16),
            safetensors::Dtype::U32 => Ok(DType::U32),
            _ => Err(anyhow!("Unsupported or unknown safetensors DType: {:?}", dtype)),
        }
    }

    /// Calculates the required buffer size in bytes for a tensor of a given shape.
    ///
    /// This is superior to a simple `size_of()` method because it correctly handles
    /// block-quantized types like Q4_K.
    pub fn buffer_size_for_shape(&self, shape: &[usize]) -> usize {
        let num_elements = shape.iter().product::<usize>();
        match self {
            DType::F32 => num_elements * 4,
            DType::F16 => num_elements * 2,
            DType::BF16 => num_elements * 2,
            DType::U32 => num_elements * 4,
            DType::Q4_K => {
                // This logic is specific to the Q4_K GGUF quantization scheme.
                // It has a block size of 256 elements.
                const QK_K: usize = 256;
                // Each block has 2 bytes for scale/min, plus 128 bytes for the 256 4-bit weights.
                // (256 * 4 / 8) + 2 = 128 + 2 = 130 bytes per block
                let size_of_block = 130;
                let num_blocks = num_elements / QK_K;
                num_blocks * size_of_block
            }
        }
    }
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::U32 => 4,
            DType::Q4_K => {
                panic!("DType::size_of() is not defined for block-quantized types like Q4_K. Use buffer_size_for_shape() instead.");
            }
        }
    }
}