use crate::Config;
use anyhow::Result;
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

pub struct ModelWeights {
    tensors: HashMap<String, Vec<f32>>,
    shapes: HashMap<String, Vec<usize>>,
    pub config: Config,
}

/// Result of loading a .kjq file. weights and tokenizer json
pub struct QuantizedModel {
    pub weights: ModelWeights,
    pub tokenizer_json: String,
}

impl ModelWeights {
    pub fn load(path: &Path) -> Result<Self> {
        let weights_file = path.join("model.safetensors");
        let data = std::fs::read(weights_file)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let mut tensor_data = HashMap::new();
        let mut shapes = HashMap::new();

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let data: Vec<f32> = view
                .data()
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            shapes.insert(name.to_string(), shape);
            tensor_data.insert(name.to_string(), data);
        }

        // Load config
        let config_file = path.join("config.json");
        let config_str = std::fs::read_to_string(config_file)?;
        let config: Config = serde_json::from_str(&config_str)?;

        Ok(Self {
            tensors: tensor_data,
            shapes,
            config,
        })
    }

    pub fn get_array1(&self, name: &str) -> Result<Array1<f32>> {
        let data = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;
        let shape = &self.shapes[name];
        anyhow::ensure!(
            shape.len() == 1,
            "Expected 1D tensor, got shape {:?}",
            shape
        );
        Ok(Array1::from_vec(data.clone()))
    }

    pub fn get_array2(&self, name: &str) -> Result<Array2<f32>> {
        let data = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;
        let shape = &self.shapes[name];
        Array2::from_shape_vec((shape[0], shape[1]), data.clone())
            .map_err(|e| anyhow::anyhow!("Shape error: {}", e))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_bytes(data: &[u8], config_json: &str) -> Result<Self> {
        let tensors = SafeTensors::deserialize(data)?;

        let mut tensor_data = HashMap::new();
        let mut shapes = HashMap::new();

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let data: Vec<f32> = view
                .data()
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            shapes.insert(name.to_string(), shape);
            tensor_data.insert(name.to_string(), data);
        }

        let config: Config = serde_json::from_str(config_json)?;

        Ok(Self {
            tensors: tensor_data,
            shapes,
            config,
        })
    }

    pub fn from_quantized_bytes(data: &[u8]) -> Result<QuantizedModel> {
        let mut cursor = 0;

        // Magic check
        if data.len() < 4 || &data[0..4] != b"KJQ1" {
            return Err(anyhow::anyhow!("Invalid .kjq file: bad magic bytes"));
        }
        cursor += 4;

        // Config JSON
        let config_json_len = read_u32_le(data, &mut cursor)? as usize;
        if cursor + config_json_len > data.len() {
            return Err(anyhow::anyhow!("Invalid .kjq file: config truncated"));
        }
        let config_json = std::str::from_utf8(&data[cursor..cursor + config_json_len])
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in config: {}", e))?;
        cursor += config_json_len;

        let config: crate::Config = serde_json::from_str(config_json)
            .map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))?;

        // Tokenizer JSON
        let tokenizer_json_len = read_u32_le(data, &mut cursor)? as usize;
        if cursor + tokenizer_json_len > data.len() {
            return Err(anyhow::anyhow!("Invalid .kjq file: tokenizer truncated"));
        }
        let tokenizer_json = std::str::from_utf8(&data[cursor..cursor + tokenizer_json_len])
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in tokenizer: {}", e))?
            .to_string();
        cursor += tokenizer_json_len;

        // Number of tensors
        let num_tensors = read_u32_le(data, &mut cursor)? as usize;

        let mut tensors = HashMap::with_capacity(num_tensors);
        let mut shapes = HashMap::with_capacity(num_tensors);

        for _ in 0..num_tensors {
            // Tensor name
            let name_len = read_u32_le(data, &mut cursor)? as usize;
            if cursor + name_len > data.len() {
                return Err(anyhow::anyhow!("Invalid .kjq file: name truncated"));
            }
            let name = std::str::from_utf8(&data[cursor..cursor + name_len])
                .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in tensor name: {}", e))?
                .to_string();
            cursor += name_len;

            // Shape
            let ndim = read_u32_le(data, &mut cursor)? as usize;
            let mut shape = Vec::with_capacity(ndim);
            let mut numel: usize = 1;
            for _ in 0..ndim {
                let dim = read_u32_le(data, &mut cursor)? as usize;
                numel *= dim;
                shape.push(dim);
            }

            // Quantized flag
            if cursor >= data.len() {
                return Err(anyhow::anyhow!("Invalid .kjq file: missing quantized flag"));
            }
            let is_quantized = data[cursor] != 0;
            cursor += 1;

            let tensor_data = if is_quantized {
                // Read scale (f32)
                if cursor + 4 > data.len() {
                    return Err(anyhow::anyhow!("Invalid .kjq file: scale truncated"));
                }
                let scale = f32::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]);
                cursor += 4;

                // Read int8 data and dequantize to f32
                if cursor + numel > data.len() {
                    return Err(anyhow::anyhow!(
                        "Invalid .kjq file: tensor '{}' data truncated (need {} bytes, have {})",
                        name,
                        numel,
                        data.len() - cursor
                    ));
                }

                let mut f32_data = Vec::with_capacity(numel);
                for i in 0..numel {
                    let q_val = data[cursor + i] as i8;
                    f32_data.push(q_val as f32 * scale);
                }
                cursor += numel;

                f32_data
            } else {
                // Read f32 data directly
                let byte_len = numel * 4;
                if cursor + byte_len > data.len() {
                    return Err(anyhow::anyhow!(
                        "Invalid .kjq file: tensor '{}' f32 data truncated",
                        name
                    ));
                }

                let f32_data: Vec<f32> = data[cursor..cursor + byte_len]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                cursor += byte_len;

                f32_data
            };
            let name = if let Some(pos) = name.find("embeddings.") {
                name[pos..].to_string()
            } else if let Some(pos) = name.find("encoder.") {
                name[pos..].to_string()
            } else if let Some(pos) = name.find("pooler.") {
                name[pos..].to_string()
            } else if let Some(pos) = name.find("classifier.") {
                name[pos..].to_string()
            } else {
                name
            };

            shapes.insert(name.clone(), shape);
            tensors.insert(name, tensor_data);
        }

        Ok(QuantizedModel {
            weights: Self {
                tensors,
                shapes,
                config,
            },
            tokenizer_json,
        })
    }
}

/// Read a little-endian u32 from a byte slice at the given cursor position.
fn read_u32_le(data: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > data.len() {
        return Err(anyhow::anyhow!(
            "Invalid .kjq file: unexpected end of data at offset {}",
            cursor
        ));
    }
    let val = u32::from_le_bytes([
        data[*cursor],
        data[*cursor + 1],
        data[*cursor + 2],
        data[*cursor + 3],
    ]);
    *cursor += 4;
    Ok(val)
}
