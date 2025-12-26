use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Copy, Serialize)]
#[allow(non_snake_case)] // To allow serde to match the camelCase keys
pub struct SummarizationParams {
    pub early_stopping: bool,
    pub length_penalty: f32,
    pub max_length: usize,
    pub min_length: usize,
    pub no_repeat_ngram_size: usize,
    pub num_beams: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(non_snake_case)]
pub struct TaskSpecificParams {
    pub summarization: SummarizationParams,
}
