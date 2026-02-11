use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ8_0, QK_K};
use anyhow::{Result, anyhow};

#[allow(non_camel_case_types)]
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
    /// 8-bit block-quantized (from GGUF)
    Q8_0,
    /// 4-bit block-quantized with K-quants (from GGUF)
    Q4_K,
    Q5_K,
    Q6_K,
}

impl DType {
    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::Q8_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K)
    }
    /// Maps a `safetensors::Dtype` to our internal `DType`.
    pub fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self> {
        match dtype {
            safetensors::Dtype::F32 => Ok(DType::F32),
            safetensors::Dtype::F16 => Ok(DType::F16),
            safetensors::Dtype::BF16 => Ok(DType::BF16),
            safetensors::Dtype::U32 => Ok(DType::U32),
            _ => Err(anyhow!(
                "Unsupported or unknown safetensors DType: {:?}",
                dtype
            )),
        }
    }
    pub fn element_size(&self) -> Option<usize> {
        match self {
            DType::F32 => Some(4),
            DType::F16 => Some(2),
            DType::BF16 => Some(2),
            DType::U32 => Some(4),
            DType::Q8_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K => None,
        }
    }
    

    pub fn from_gguf(dtype: gguf::GGMLType) -> Result<Self> {
        match dtype {
            gguf::GGMLType::F32 => Ok(DType::F32),
            gguf::GGMLType::F16 => Ok(DType::F16),
            gguf::GGMLType::Q8_0 => Ok(DType::Q8_0),
            gguf::GGMLType::Q4K => Ok(DType::Q4_K),
            gguf::GGMLType::Q6K => Ok(DType::Q6_K),
            _ => Err(anyhow!("Unsupported or unknown GGUF DType: {:?}", dtype)),
        }
    }
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::U32 => 4,
            // DType::I32 => 4,
            DType::Q8_0 => panic!("Q8_0 has variable block size"),
            DType::Q4_K => panic!("Q4_K has variable block size"),
            DType::Q5_K => panic!("Q5_K has variable block size"),
            DType::Q6_K => panic!("Q6_K has variable block size"),
        }
    }
    /// Calculates the required buffer size in bytes for a tensor of a given shape.
    pub fn buffer_size_for_shape(&self, shape: &[usize]) -> Result<usize> {
        let num_elements = shape.iter().product::<usize>();
        match self {
            DType::F32 => Ok(num_elements * 4),
            DType::F16 => Ok(num_elements * 2),
            DType::BF16 => Ok(num_elements * 2),
            DType::U32 => Ok(num_elements * 4),
            DType::Q8_0 => {
                let k_per_block = std::mem::size_of::<[i8; 32]>(); // 32
                if num_elements % k_per_block != 0 {
                    return Err(anyhow!(
                        "For Q8_0, the number of elements must be a multiple of {}",
                        k_per_block
                    ));
                }
                let num_blocks = num_elements / k_per_block;
                Ok(num_blocks * std::mem::size_of::<BlockQ8_0>())
            }
            DType::Q4_K => {
                if num_elements % QK_K != 0 {
                    return Err(anyhow!(
                        "For Q4_K, the number of elements must be a multiple of {}",
                        QK_K
                    ));
                }
                let num_blocks = num_elements / QK_K;
                Ok(num_blocks * std::mem::size_of::<BlockQ4_K>())
            }
            DType::Q6_K => {
                unimplemented!()
            }
            DType::Q5_K => {
                unimplemented!()
            }
        }
    }

    /// Returns the size in bytes of a single element of the type.
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::U32 => 4,
            DType::Q8_0 | DType::Q4_K | DType::Q6_K | DType::Q5_K => {
                panic!(
                    "DType::size_of() is not defined for block-quantized types like {:?}. Use buffer_size_for_shape() instead.",
                    self
                );
            }
        }
    }
}
