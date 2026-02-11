//! Defines the tensor

pub mod dtype;
pub mod raw_tensor;

pub use dtype::DType;

use crate::cpu::kernels::{
    dequantize::{dequantize_q4_k_block, dequantize_q6_k_block, dequantize_q8_0_block},
    q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0},
};
use anyhow::{anyhow, Result};
use half::{bf16, f16};
use ndarray::{Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};


/// A wrapper for quantized matrix dat
#[derive(Debug, Clone)]
pub struct QuantizedMatrix<T> {
    pub blocks: Vec<T>,
    pub shape: [usize; 2], // [out_features, in_features]
}

impl<B: Dequantizable> QuantizedMatrix<B> {
    /// Dequantize the entire matrix to f32.
    pub fn dequantize(&self) -> Result<Array2<f32>> {
        let [rows, cols] = self.shape;
        let mut output = Array2::zeros((rows, cols));
        B::dequantize_blocks(&self.blocks, &mut output, rows, cols)?;
        Ok(output)
    }
}

/// Trait for types that can be dequantized to f32.
pub trait Dequantizable {
    /// Block size (number of elements per block)
    const BLOCK_SIZE: usize;
    
    /// Dequantize a single block into the output slice
    fn dequantize_block(block: &Self, output: &mut [f32]);
    
    /// Dequantize all blocks into a 2D array
    fn dequantize_blocks(
        blocks: &[Self],
        output: &mut Array2<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()>
    where
        Self: Sized,
    {
        let blocks_per_row = cols / Self::BLOCK_SIZE;
        
        for (row_idx, mut row) in output.axis_iter_mut(ndarray::Axis(0)).enumerate() {
            let row_blocks = &blocks[row_idx * blocks_per_row..(row_idx + 1) * blocks_per_row];
            let row_slice = row.as_slice_mut()
                .ok_or_else(|| anyhow!("Non-contiguous row during dequantization"))?;
            
            for (b_idx, block) in row_blocks.iter().enumerate() {
                let start = b_idx * Self::BLOCK_SIZE;
                let end = start + Self::BLOCK_SIZE;
                Self::dequantize_block(block, &mut row_slice[start..end]);
            }
        }
        Ok(())
    }
}

impl Dequantizable for BlockQ8_0 {
    const BLOCK_SIZE: usize = 32;
    
    fn dequantize_block(block: &Self, output: &mut [f32]) {
        dequantize_q8_0_block(block, output);
    }
}

impl Dequantizable for BlockQ4_K {
    const BLOCK_SIZE: usize = 256;
    
    fn dequantize_block(block: &Self, output: &mut [f32]) {
        dequantize_q4_k_block(block, output);
    }
}

impl Dequantizable for BlockQ6_K {
    const BLOCK_SIZE: usize = 256;
    
    fn dequantize_block(block: &Self, output: &mut [f32]) {
        dequantize_q6_k_block(block, output);
    }
}


/// A generic, typed tensor for CPU computation.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub enum CpuTensor {
    F32(ArrayD<f32>),
    BF16(ArrayD<bf16>),
    F16(ArrayD<f16>),
    Q8_0(QuantizedMatrix<BlockQ8_0>),
    Q4_K(QuantizedMatrix<BlockQ4_K>),
    Q6_K(QuantizedMatrix<BlockQ6_K>),
}

impl CpuTensor {
    pub fn dtype(&self) -> DType {
        match self {
            CpuTensor::F32(_) => DType::F32,
            CpuTensor::BF16(_) => DType::BF16,
            CpuTensor::F16(_) => DType::F16,
            CpuTensor::Q8_0(_) => DType::Q8_0,
            CpuTensor::Q4_K(_) => DType::Q4_K,
            CpuTensor::Q6_K(_) => DType::Q6_K,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            CpuTensor::Q8_0(_) | CpuTensor::Q4_K(_) | CpuTensor::Q6_K(_)
        )
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        match self {
            CpuTensor::F32(arr) => arr.shape(),
            CpuTensor::F16(arr) => arr.shape(),
            CpuTensor::BF16(arr) => arr.shape(),
            CpuTensor::Q8_0(q) => &q.shape,
            CpuTensor::Q4_K(q) => &q.shape,
            CpuTensor::Q6_K(q) => &q.shape,
        }
    }

    /// Returns the rank (number of dimensions) of the tensor.
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Returns the total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape().iter().product()
    }

    /// Returns the size in bytes (approximate for quantized).
    pub fn size_bytes(&self) -> usize {
        match self {
            CpuTensor::F32(arr) => arr.len() * 4,
            CpuTensor::F16(arr) => arr.len() * 2,
            CpuTensor::BF16(arr) => arr.len() * 2,
            CpuTensor::Q8_0(q) => q.blocks.len() * std::mem::size_of::<BlockQ8_0>(),
            CpuTensor::Q4_K(q) => q.blocks.len() * std::mem::size_of::<BlockQ4_K>(),
            CpuTensor::Q6_K(q) => q.blocks.len() * std::mem::size_of::<BlockQ6_K>(),
        }
    }

    /// Converts to a 1D f32 array, consuming self.
    ///
    /// Works for F32, F16, BF16. Fails for quantized matrix types.
    pub fn to_array1_f32(self) -> Result<Array1<f32>> {
        let len = self.num_elements();
        
        match self {
            CpuTensor::F32(arr) => Ok(arr
                .into_shape_with_order(len)?
                .into_dimensionality::<Ix1>()?),
            CpuTensor::F16(arr) => Ok(arr
                .mapv(|v| v.to_f32())
                .into_shape_with_order(len)?
                .into_dimensionality::<Ix1>()?),
            CpuTensor::BF16(arr) => Ok(arr
                .mapv(|v| v.to_f32())
                .into_shape_with_order(len)?
                .into_dimensionality::<Ix1>()?),
            _ => Err(anyhow!(
                "Cannot convert quantized matrix type {:?} to Array1",
                self.dtype()
            )),
        }
    }

    /// Converts to a 2D f32 array, consuming self.
    pub fn to_array2_f32(self) -> Result<Array2<f32>> {
        let shape = self.shape().to_vec();
        if shape.len() < 2 {
            return Err(anyhow!("Cannot convert rank-{} tensor to Array2", shape.len()));
        }
        let (rows, cols) = (shape[0], shape[1]);

        match self {
            CpuTensor::F32(arr) => Ok(arr
                .into_shape_with_order((rows, cols))?
                .into_dimensionality::<Ix2>()?),
            CpuTensor::F16(arr) => Ok(arr
                .mapv(|v| v.to_f32())
                .into_shape_with_order((rows, cols))?
                .into_dimensionality::<Ix2>()?),
            CpuTensor::BF16(arr) => Ok(arr
                .mapv(|v| v.to_f32())
                .into_shape_with_order((rows, cols))?
                .into_dimensionality::<Ix2>()?),
            CpuTensor::Q8_0(matrix) => matrix.dequantize(),
            CpuTensor::Q4_K(matrix) => matrix.dequantize(),
            CpuTensor::Q6_K(matrix) => matrix.dequantize(),
        }
    }

    /// Converts to a 3D f32 array, consuming self.
    pub fn to_array3_f32(self) -> Result<Array3<f32>> {
        let shape = self.shape().to_vec();
        if shape.len() != 3 {
            return Err(anyhow!("Cannot convert rank-{} tensor to Array3", shape.len()));
        }
        let (d0, d1, d2) = (shape[0], shape[1], shape[2]);

        match self {
            CpuTensor::F32(arr) => Ok(arr
                .into_shape_with_order((d0, d1, d2))?
                .into_dimensionality::<Ix3>()?),
            CpuTensor::F16(arr) => Ok(arr
                .mapv(|v| v.to_f32())
                .into_shape_with_order((d0, d1, d2))?
                .into_dimensionality::<Ix3>()?),
            CpuTensor::BF16(arr) => Ok(arr
                .mapv(|v| v.to_f32())
                .into_shape_with_order((d0, d1, d2))?
                .into_dimensionality::<Ix3>()?),
            _ => Err(anyhow!(
                "Cannot convert quantized matrix type {:?} to Array3",
                self.dtype()
            )),
        }
    }

    /// Converts to a 1D f32 array without consuming self.
    ///
    /// This clones the data internally. Prefer `to_array1_f32()`
    pub fn as_array1_f32(&self) -> Result<Array1<f32>> {
        self.clone().to_array1_f32()
    }

    /// Converts to a 2D f32 array without consuming self.
    ///
    /// This clones the data internally. Prefer `to_array2_f32()`
    pub fn as_array2_f32(&self) -> Result<Array2<f32>> {
        self.clone().to_array2_f32()
    }

    /// Converts to a 3D f32 array without consuming self.
    ///
    /// This clones the data internally. Prefer `to_array3_f32()`
    pub fn as_array3_f32(&self) -> Result<Array3<f32>> {
        self.clone().to_array3_f32()
    }

    /// Get the raw f32 data if this is an F32 tensor.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            CpuTensor::F32(arr) => arr.as_slice(),
            _ => None,
        }
    }

    /// Get the raw f16 data if this is an F16 tensor.
    pub fn as_f16_slice(&self) -> Option<&[f16]> {
        match self {
            CpuTensor::F16(arr) => arr.as_slice(),
            _ => None,
        }
    }

    /// Get the raw bf16 data if this is a BF16 tensor.
    pub fn as_bf16_slice(&self) -> Option<&[bf16]> {
        match self {
            CpuTensor::BF16(arr) => arr.as_slice(),
            _ => None,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_cpu_tensor_f32_metadata() {
        let arr = ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor = CpuTensor::F32(arr);

        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.num_elements(), 6);
        assert_eq!(tensor.size_bytes(), 24); // 6 * 4 bytes
        assert!(!tensor.is_quantized());
    }

    #[test]
    fn test_cpu_tensor_f16_metadata() {
        let arr = ArrayD::from_shape_vec(
            vec![2, 2],
            vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)],
        ).unwrap();
        let tensor = CpuTensor::F16(arr);

        assert_eq!(tensor.dtype(), DType::F16);
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.size_bytes(), 8); // 4 * 2 bytes
        assert!(!tensor.is_quantized());
    }

    #[test]
    fn test_cpu_tensor_bf16_metadata() {
        let arr = ArrayD::from_shape_vec(
            vec![4],
            vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(4.0)],
        ).unwrap();
        let tensor = CpuTensor::BF16(arr);

        assert_eq!(tensor.dtype(), DType::BF16);
        assert_eq!(tensor.shape(), &[4]);
        assert_eq!(tensor.rank(), 1);
        assert!(!tensor.is_quantized());
    }

    #[test]
    fn test_f32_to_array1() {
        let arr = ArrayD::from_shape_vec(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let tensor = CpuTensor::F32(arr);

        let result = tensor.to_array1_f32().unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_f32_to_array2() {
        let arr = ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor = CpuTensor::F32(arr);

        let result = tensor.to_array2_f32().unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[1, 2]], 6.0);
    }

    #[test]
    fn test_f32_to_array3() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let arr = ArrayD::from_shape_vec(vec![2, 3, 4], data).unwrap();
        let tensor = CpuTensor::F32(arr);

        let result = tensor.to_array3_f32().unwrap();
        assert_eq!(result.shape(), &[2, 3, 4]);
        assert_eq!(result[[0, 0, 0]], 0.0);
        assert_eq!(result[[1, 2, 3]], 23.0);
    }

    #[test]
    fn test_f16_to_array1() {
        let arr = ArrayD::from_shape_vec(
            vec![3],
            vec![f16::from_f32(1.5), f16::from_f32(2.5), f16::from_f32(3.5)],
        ).unwrap();
        let tensor = CpuTensor::F16(arr);

        let result = tensor.to_array1_f32().unwrap();
        assert_eq!(result.shape(), &[3]);
        assert!((result[0] - 1.5).abs() < 0.01);
        assert!((result[1] - 2.5).abs() < 0.01);
        assert!((result[2] - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_bf16_to_array2() {
        let arr = ArrayD::from_shape_vec(
            vec![2, 2],
            vec![
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
            ],
        ).unwrap();
        let tensor = CpuTensor::BF16(arr);

        let result = tensor.to_array2_f32().unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert!((result[[0, 0]] - 1.0).abs() < 0.01);
        assert!((result[[1, 1]] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_array1_from_2d_flattens() {
        let arr = ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor = CpuTensor::F32(arr);

        // Should flatten to 1D
        let result = tensor.to_array1_f32().unwrap();
        assert_eq!(result.shape(), &[6]);
    }

    #[test]
    fn test_array3_wrong_rank_error() {
        let arr = ArrayD::from_shape_vec(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor = CpuTensor::F32(arr);

        let result = tensor.to_array3_f32();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rank"));
    }
    #[test]
    fn test_as_array_keeps_original() {
        let arr = ArrayD::from_shape_vec(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let tensor = CpuTensor::F32(arr);

        let result1 = tensor.as_array1_f32().unwrap();
        let result2 = tensor.as_array1_f32().unwrap(); // Can call again!

        assert_eq!(result1, result2);
        assert_eq!(tensor.shape(), &[4]);
    }
    #[test]
    fn test_as_f32_slice() {
        let arr = ArrayD::from_shape_vec(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let tensor = CpuTensor::F32(arr);

        let slice = tensor.as_f32_slice().unwrap();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_as_f32_slice_wrong_type() {
        let arr = ArrayD::from_shape_vec(
            vec![2],
            vec![f16::from_f32(1.0), f16::from_f32(2.0)],
        ).unwrap();
        let tensor = CpuTensor::F16(arr);

        assert!(tensor.as_f32_slice().is_none());
        assert!(tensor.as_f16_slice().is_some());
    }

    #[test]
    fn test_quantized_matrix_shape() {
        let matrix = QuantizedMatrix::<BlockQ8_0> {
            blocks: vec![],
            shape: [64, 128],
        };

        let tensor = CpuTensor::Q8_0(matrix);
        assert!(tensor.is_quantized());
        assert_eq!(tensor.dtype(), DType::Q8_0);
        assert_eq!(tensor.shape(), &[64, 128]);
    }

    #[test]
    fn test_quantized_cannot_to_array1() {
        let matrix = QuantizedMatrix::<BlockQ8_0> {
            blocks: vec![],
            shape: [64, 128],
        };
        let tensor = CpuTensor::Q8_0(matrix);

        let result = tensor.to_array1_f32();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("quantized"));
    }

    #[test]
    fn test_quantized_cannot_to_array3() {
        let matrix = QuantizedMatrix::<BlockQ4_K> {
            blocks: vec![],
            shape: [64, 256],
        };
        let tensor = CpuTensor::Q4_K(matrix);

        let result = tensor.to_array3_f32();
        assert!(result.is_err());
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(BlockQ8_0::BLOCK_SIZE, 32);
        assert_eq!(BlockQ4_K::BLOCK_SIZE, 256);
        assert_eq!(BlockQ6_K::BLOCK_SIZE, 256);
    }
}