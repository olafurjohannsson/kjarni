use crate::tensor::dtype::DType;
use half::{bf16, f16};
use ndarray::{Array1, Array2, ArrayD, ArrayView2, Ix1, Ix2, IxDyn};
use anyhow::{anyhow, Result};

/// A struct to hold a block-quantized tensor's data.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// The raw bytes containing the 4-bit weights, usually grouped in blocks.
    pub data: Vec<u8>,
    /// The scale factors for each block, usually bf16 or f16.
    pub scales: Vec<bf16>,
    /// The shape of the tensor *after* dequantization.
    pub shape: [usize; 2],
    /// The number of elements per block (e.g., 256 for Q4_K).
    pub block_size: usize,
}

impl QuantizedTensor {
    /// Dequantize to F32 (expensive, avoid in hot paths)
    pub fn dequantize(&self) -> Array2<f32> {
        let mut output = Array2::<f32>::zeros((self.shape[0], self.shape[1]));
        // TODO: Implement actual Q4_K dequantization
        // For now, return zeros
        output
    }

    /// Dequantize a single row (for incremental matmul)
    pub fn dequantize_row(&self, row: usize) -> Array1<f32> {
        let mut output = Array1::<f32>::zeros(self.shape[1]);
        // TODO: Implement row-wise dequantization
        output
    }
}

/// A generic, typed tensor for CPU computation.
/// This enum holds an owned ndarray of a specific type.
pub enum TypedCpuTensor {
    F32(ArrayD<f32>),
    BF16(ArrayD<bf16>),
    F16(ArrayD<f16>),
    Q4_K(QuantizedTensor),
}

impl TypedCpuTensor {
    pub fn dtype(&self) -> DType {
        match self {
            TypedCpuTensor::F32(_) => DType::F32,
            TypedCpuTensor::BF16(_) => DType::BF16,
            TypedCpuTensor::F16(_) => DType::F16,
            TypedCpuTensor::Q4_K(_) => DType::Q4_K,
        }
    }
    pub fn shape(&self) -> &[usize] {
        match self {
            TypedCpuTensor::F32(arr) => arr.shape(),
            TypedCpuTensor::F16(arr) => arr.shape(),
            TypedCpuTensor::BF16(arr) => arr.shape(),
            TypedCpuTensor::Q4_K(q) => &q.shape,
        }
    }

    /// Total number of elements
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            TypedCpuTensor::F32(arr) => arr.len() * 4,
            TypedCpuTensor::F16(arr) => arr.len() * 2,
            TypedCpuTensor::BF16(arr) => arr.len() * 2,
            TypedCpuTensor::Q4_K(q) => q.data.len() + q.scales.len() * 2,
        }
    }
    // =========================================================================
    // Conversion methods - used at load time for small tensors
    // =========================================================================

    /// Convert to F32 Array1 (for biases, layer norm weights)
    pub fn to_array1_f32(&self) -> Result<Array1<f32>> {
        match self {
            TypedCpuTensor::F32(arr) => {
                Ok(arr.clone().into_dimensionality::<Ix1>()?)
            }
            TypedCpuTensor::F16(arr) => {
                Ok(arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix1>()?)
            }
            TypedCpuTensor::BF16(arr) => {
                Ok(arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix1>()?)
            }
            TypedCpuTensor::Q4_K(_) => {
                Err(anyhow!("Cannot convert Q4_K to Array1"))
            }
        }
    }

    /// Convert to F32 Array2 (for weight matrices)
    pub fn to_array2_f32(&self) -> Result<Array2<f32>> {
        match self {
            TypedCpuTensor::F32(arr) => {
                Ok(arr.clone().into_dimensionality::<Ix2>()?)
            }
            TypedCpuTensor::F16(arr) => {
                Ok(arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix2>()?)
            }
            TypedCpuTensor::BF16(arr) => {
                Ok(arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix2>()?)
            }
            TypedCpuTensor::Q4_K(q) => {
                Ok(q.dequantize())
            }
        }
    }

    /// Get as F32 ArrayD (generic shape)
    pub fn to_arrayd_f32(&self) -> Result<ArrayD<f32>> {
        match self {
            TypedCpuTensor::F32(arr) => Ok(arr.clone()),
            TypedCpuTensor::F16(arr) => Ok(arr.mapv(|v| v.to_f32())),
            TypedCpuTensor::BF16(arr) => Ok(arr.mapv(|v| v.to_f32())),
            TypedCpuTensor::Q4_K(q) => Ok(q.dequantize().into_dyn()),
        }
    }

    // =========================================================================
    // View methods - zero-copy access when dtype matches
    // =========================================================================

    /// Try to get a view as F32 (zero-copy if already F32)
    pub fn as_array2_f32(&self) -> Option<ArrayView2<f32>> {
        match self {
            TypedCpuTensor::F32(arr) => arr.view().into_dimensionality::<Ix2>().ok(),
            _ => None,
        }
    }

    /// Try to get a view as BF16
    pub fn as_array2_bf16(&self) -> Option<ArrayView2<bf16>> {
        match self {
            TypedCpuTensor::BF16(arr) => arr.view().into_dimensionality::<Ix2>().ok(),
            _ => None,
        }
    }
}

// =========================================================================
// From implementations for easy construction
// =========================================================================

impl From<ArrayD<f32>> for TypedCpuTensor {
    fn from(arr: ArrayD<f32>) -> Self {
        TypedCpuTensor::F32(arr)
    }
}

impl From<ArrayD<f16>> for TypedCpuTensor {
    fn from(arr: ArrayD<f16>) -> Self {
        TypedCpuTensor::F16(arr)
    }
}

impl From<ArrayD<bf16>> for TypedCpuTensor {
    fn from(arr: ArrayD<bf16>) -> Self {
        TypedCpuTensor::BF16(arr)
    }
}

impl From<Array2<f32>> for TypedCpuTensor {
    fn from(arr: Array2<f32>) -> Self {
        TypedCpuTensor::F32(arr.into_dyn())
    }
}

impl From<Array1<f32>> for TypedCpuTensor {
    fn from(arr: Array1<f32>) -> Self {
        TypedCpuTensor::F32(arr.into_dyn())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_f32_to_array1() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let tensor = TypedCpuTensor::F32(arr);
        
        let result = tensor.to_array1_f32().unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_bf16_to_array1() {
        let arr = ArrayD::from_shape_vec(
            IxDyn(&[4]),
            vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(4.0)]
        ).unwrap();
        let tensor = TypedCpuTensor::BF16(arr);
        
        let result = tensor.to_array1_f32().unwrap();
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_size_bytes() {
        let f32_tensor = TypedCpuTensor::F32(ArrayD::zeros(IxDyn(&[100, 100])));
        let bf16_tensor = TypedCpuTensor::BF16(ArrayD::from_elem(IxDyn(&[100, 100]), bf16::ZERO));
        
        assert_eq!(f32_tensor.size_bytes(), 100 * 100 * 4);
        assert_eq!(bf16_tensor.size_bytes(), 100 * 100 * 2);
    }
}