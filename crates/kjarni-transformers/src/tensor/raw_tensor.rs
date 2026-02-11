use crate::tensor::dtype::DType;
use anyhow::{Result, anyhow};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};
use std::borrow::Cow;

/// A raw, untyped view into a tensor's bytes, shape, and dtype.
/// This is the primary output of a file loader before any CPU/GPU-specific processing.
/// 
/// Important to note is that TensorView borrows and is only valid while
/// mmap pages are alive and loader is alive
/// IMPORTANT: The view must be fully consumed synchronously
#[derive(Debug)] 
pub(crate) struct TensorView<'a> {
    pub name: String,
    pub bytes: Cow<'a, [u8]>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl<'a> TensorView<'a> {
    #[deprecated(
        since = "0.1.0",
        note = "This is a slow compatibility layer. Use ModelWeights::get_typed_tensor and handle specific dtypes instead."
    )]
    pub fn to_ndarray_f32(&self) -> Result<ArrayD<f32>> {
        let data: Vec<f32> = match self.dtype {
            DType::F32 => {
                if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(&self.bytes) {
                    slice.to_vec()
                }
                // if not aligned, copy the bytes element by element.
                else {
                    self.bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect()
                }
            }
            DType::F16 => {
                let f16_data: &[f16] = bytemuck::try_cast_slice(&self.bytes).map_err(|e| {
                    anyhow!(
                        "Failed to cast bytes to f16 for tensor '{}': {}",
                        self.name,
                        e
                    )
                })?;
                f16_data.iter().map(|x| x.to_f32()).collect()
            }
            DType::BF16 => {
                let bf16_data: &[bf16] = bytemuck::try_cast_slice(&self.bytes).map_err(|e| {
                    anyhow!(
                        "Failed to cast bytes to bf16 for tensor '{}': {}",
                        self.name,
                        e
                    )
                })?;
                bf16_data.iter().map(|x| x.to_f32()).collect()
            }
            _ => {
                return Err(anyhow!(
                    "Automatic f32 conversion not supported for dtype {:?} on tensor '{}'",
                    self.dtype,
                    self.name
                ));
            }
        };

        Ok(ArrayD::from_shape_vec(IxDyn(&self.shape), data)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};

    #[test]
    fn test_tensorview_f32_to_ndarray() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let tensor = TensorView {
            name: "f32_tensor".to_string(),
            bytes: Cow::Owned(bytes),
            shape: vec![2, 2],
            dtype: DType::F32,
        };

        let arr = tensor.to_ndarray_f32().unwrap();
        let flat: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(flat, values);
    }

    #[test]
    fn test_tensorview_f16_to_ndarray() {
        let values: Vec<f32> = vec![1.5, 2.5, 3.5, 4.5];
        let f16_values: Vec<f16> = values.iter().map(|v| f16::from_f32(*v)).collect();
        let bytes: Vec<u8> = f16_values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let tensor = TensorView {
            name: "f16_tensor".to_string(),
            bytes: Cow::Owned(bytes),
            shape: vec![2, 2],
            dtype: DType::F16,
        };

        let arr = tensor.to_ndarray_f32().unwrap();
        let flat: Vec<f32> = arr.iter().copied().collect();
        for (a, b) in flat.iter().zip(values.iter()) {
            let diff = (a - b).abs();
            assert!(diff < 0.001, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_tensorview_f32_safe_path_aligned_length_unaligned_start() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Add 1-byte offset at the start to misalign
        let mut unaligned_bytes = vec![0u8];
        unaligned_bytes.extend_from_slice(&bytes);

        let tensor = TensorView {
            name: "unaligned_f32".to_string(),
            bytes: Cow::Owned(unaligned_bytes[1..].to_vec()), // skip first byte
            shape: vec![2, 2],
            dtype: DType::F32,
        };

        let arr = tensor.to_ndarray_f32().unwrap();
        let flat: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(flat, values);
    }

    #[test]
    fn test_tensorview_f32_safe_path_unaligned() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut unaligned = vec![0u8];
        unaligned.extend_from_slice(&bytes);

        let tensor = TensorView {
            name: "unaligned_f32".to_string(),
            bytes: Cow::Owned(unaligned[1..].to_vec()),
            shape: vec![2, 2],
            dtype: DType::F32,
        };

        let arr = tensor
            .to_ndarray_f32()
            .expect("Safe path conversion failed");
        let flat: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(flat, values);
    }

    #[test]
    fn test_tensorview_bf16_to_ndarray() {
        let values: Vec<f32> = vec![5.5, 6.5, 7.5, 8.5];
        let bf16_values: Vec<bf16> = values.iter().map(|v| bf16::from_f32(*v)).collect();
        let bytes: Vec<u8> = bf16_values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let tensor = TensorView {
            name: "bf16_tensor".to_string(),
            bytes: Cow::Owned(bytes),
            shape: vec![2, 2],
            dtype: DType::BF16,
        };

        let arr = tensor.to_ndarray_f32().unwrap();
        let flat: Vec<f32> = arr.iter().copied().collect();
        for (a, b) in flat.iter().zip(values.iter()) {
            let diff = (a - b).abs();
            assert!(diff < 0.01, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_tensorview_invalid_dtype() {
        let tensor = TensorView {
            name: "invalid".to_string(),
            bytes: Cow::Owned(vec![0u8; 4]),
            shape: vec![1],
            dtype: DType::Q8_0,
        };

        let result = tensor.to_ndarray_f32();
        assert!(result.is_err());
    }
}
