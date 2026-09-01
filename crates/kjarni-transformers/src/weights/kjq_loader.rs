//! Serves a `KJQ8` container's tensors without expanding them to f32.
//!
//! `KJQ1` is unpacked into f32 safetensors and read by `SafeTensorsLoader`,
//! which is right for small encoders. `KJQ8` cannot go that route: safetensors
//! has no dtype that describes `BlockQ8_0`, and materialising the weights as f32
//! is the thing the encoding exists to avoid. Qwen2.5 0.5B is 494M parameters,
//! which is 1.98GB as f32 against a 2GB cap on a single wasm32 allocation.
//!
//! So the blocks are handed straight through as `DType::Q8_0`, and
//! `raw_to_typed` already knows how to turn those into `CpuTensor::Q8_0`.

use std::any::Any;
use std::collections::HashMap;

use anyhow::{Result, anyhow};

use crate::tensor::DType;
use crate::tensor::raw_tensor::TensorView;
use crate::weights::WeightLoader;
use crate::weights::kjq::KjqUnpacked;

/// A loader over the block tensors of a `KJQ8` file.
pub struct KjqLoader {
    /// Block-quantised tensors, kept as the bytes they were written as.
    blocks: HashMap<String, (Vec<u8>, [usize; 2])>,
    /// Tensors the writer kept in f32: biases, layer norms, anything whose
    /// shape does not divide into blocks.
    plain: HashMap<String, (Vec<u8>, Vec<usize>)>,
}

impl KjqLoader {
    /// Builds a loader from an unpacked `KJQ8` container.
    ///
    /// The f32 remainder still arrives as a safetensors buffer, because that is
    /// what the writer produced for those tensors and they are small.
    pub fn new(unpacked: &KjqUnpacked) -> Result<Self> {
        let mut blocks = HashMap::with_capacity(unpacked.blocks.len());
        for t in &unpacked.blocks {
            blocks.insert(t.name.clone(), (t.bytes.clone(), t.shape));
        }

        let mut plain = HashMap::new();
        if !unpacked.safetensors.is_empty() {
            let st = safetensors::SafeTensors::deserialize(&unpacked.safetensors)
                .map_err(|e| anyhow!("the f32 half of the .kjq is not valid safetensors: {e}"))?;
            for (name, view) in st.tensors() {
                plain.insert(name, (view.data().to_vec(), view.shape().to_vec()));
            }
        }

        Ok(Self { blocks, plain })
    }
}

impl WeightLoader for KjqLoader {
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>> {
        if let Some((bytes, shape)) = self.blocks.get(name) {
            return Ok(TensorView {
                name: name.to_string(),
                bytes: std::borrow::Cow::Borrowed(bytes),
                shape: shape.to_vec(),
                dtype: DType::Q8_0,
            });
        }
        if let Some((bytes, shape)) = self.plain.get(name) {
            return Ok(TensorView {
                name: name.to_string(),
                bytes: std::borrow::Cow::Borrowed(bytes),
                shape: shape.clone(),
                dtype: DType::F32,
            });
        }
        Err(anyhow!("tensor '{name}' is not in this .kjq file"))
    }

    fn contains(&self, name: &str) -> bool {
        self.blocks.contains_key(name) || self.plain.contains_key(name)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
