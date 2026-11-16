//! Configuration of the Qwen 3.5 text decoder.
//!
//! A Qwen 3.5 checkpoint is usually the multimodal `Qwen3_5ForConditionalGeneration`,
//! whose `config.json` nests the decoder under `text_config` and whose weights put it
//! under `model.language_model.`. A text-only `Qwen3_5ForCausalLM` export has the same
//! fields at the top level and its weights under `model.`. Both are read here; the
//! vision tower and the multi-token-prediction head are never loaded.

use anyhow::{Context, Result, anyhow, bail};
use kjarni_transformers::{
    activations::Activation,
    traits::{
        AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, ModelConfig,
        ModelLayout, ModelMetadata, NormalizationStrategy,
    },
    weights::ModelWeights,
};
use serde::Deserialize;
use std::sync::Arc;

/// What mixes information across positions in one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Softmax attention with a KV cache.
    FullAttention,
    /// Gated DeltaNet: a fixed-size recurrent state per head.
    LinearAttention,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    partial_rotary_factor: f32,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}
fn default_partial_rotary_factor() -> f32 {
    0.25
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_true() -> bool {
    true
}
fn default_full_attention_interval() -> usize {
    4
}
fn default_conv_kernel() -> usize {
    4
}

#[derive(Debug, Clone, Deserialize)]
struct TextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: Option<usize>,
    // A mixture-of-experts config has expert sizes instead.
    #[serde(default)]
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default = "default_full_attention_interval")]
    full_attention_interval: usize,
    #[serde(default = "default_conv_kernel")]
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    // Older exports state these at the top level instead of in rope_parameters.
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
    #[serde(default = "default_true")]
    attn_output_gate: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    pad_token_id: Option<u32>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    num_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    #[serde(default)]
    shared_expert_intermediate_size: Option<usize>,
}

/// The feed-forward of a mixture-of-experts model, used in every layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeDims {
    pub num_experts: usize,
    /// Experts each token goes through.
    pub top_k: usize,
    /// Each expert's intermediate width.
    pub expert_size: usize,
    /// The shared expert's intermediate width; 0 for none.
    pub shared_size: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen3_5Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    /// Set for a mixture-of-experts model, whose layers have experts in place of
    /// a dense feed-forward.
    pub moe: Option<MoeDims>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub layer_kinds: Vec<LayerKind>,
    pub conv_kernel: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rope_theta: f32,
    /// How many leading dimensions of each attention head are rotated.
    pub rotary_dim: usize,
    pub eos_token_id: Vec<u32>,
    pub bos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
    /// Where the decoder's tensors start: `model.language_model.` in a multimodal
    /// checkpoint, `model.` in a text-only one.
    pub prefix: String,
    /// The weights are as llama.cpp's converter writes them: the norms already
    /// include their +1 offset (except the DeltaNet's gated norm), `A_log` is
    /// stored as -exp(A_log), and when value heads outnumber key heads they are
    /// in tiled order, so value head `h` reads key head `h % num_key_heads`
    /// rather than `h / (num_value_heads / num_key_heads)`.
    pub llama_cpp_layout: bool,
}

impl Qwen3_5Config {
    /// Reads `config.json`, taking the weight prefix from what the checkpoint holds,
    /// or a GGUF's own metadata.
    pub fn from_weights(weights: &ModelWeights) -> Result<Arc<Self>> {
        if weights.is_gguf() {
            return Ok(Arc::new(Self::from_gguf(weights)?));
        }
        let prefix = if weights.contains("model.language_model.embed_tokens.weight") {
            "model.language_model."
        } else {
            "model."
        };
        Ok(Arc::new(Self::from_json(weights.config_json(), prefix)?))
    }

    pub fn from_json(json: &str, prefix: &str) -> Result<Self> {
        let root: serde_json::Value =
            serde_json::from_str(json).context("config.json is not valid JSON")?;
        let text_value = root.get("text_config").unwrap_or(&root);
        let text: TextConfig = serde_json::from_value(text_value.clone())
            .context("config.json does not describe a Qwen 3.5 text decoder")?;

        let moe = match text.num_experts.filter(|&n| n > 0) {
            Some(num_experts) => {
                let field = |v: Option<usize>, name: &str| {
                    v.filter(|&n| n > 0)
                        .with_context(|| format!("a mixture-of-experts config needs {name}"))
                };
                let top_k = field(text.num_experts_per_tok, "num_experts_per_tok")?;
                if top_k > num_experts {
                    bail!("each token cannot go through {top_k} of {num_experts} experts");
                }
                Some(MoeDims {
                    num_experts,
                    top_k,
                    expert_size: field(text.moe_intermediate_size, "moe_intermediate_size")?,
                    shared_size: text.shared_expert_intermediate_size.unwrap_or(0),
                })
            }
            None if text.intermediate_size == 0 => bail!("config.json has no intermediate_size"),
            None => None,
        };
        if !text.attn_output_gate {
            bail!(
                "this Qwen 3.5 config turns off the attention output gate, which Kjarni does not implement"
            );
        }
        if text.attention_bias {
            bail!("this Qwen 3.5 config has attention biases, which Kjarni does not implement");
        }
        if let Some(act) = &text.hidden_act
            && act != "silu"
        {
            bail!("Qwen 3.5 uses silu; this config asks for '{act}'");
        }

        let layer_kinds = match &text.layer_types {
            Some(types) => types
                .iter()
                .map(|t| match t.as_str() {
                    "full_attention" => Ok(LayerKind::FullAttention),
                    "linear_attention" => Ok(LayerKind::LinearAttention),
                    other => Err(anyhow!("unknown Qwen 3.5 layer type '{other}'")),
                })
                .collect::<Result<Vec<_>>>()?,
            // Every `interval`-th layer is full attention, counting from one.
            None => (0..text.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % text.full_attention_interval == 0 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
                })
                .collect(),
        };
        if layer_kinds.len() != text.num_hidden_layers {
            bail!(
                "config.json lists {} layer types for {} layers",
                layer_kinds.len(),
                text.num_hidden_layers
            );
        }
        if text.linear_num_key_heads == 0
            || !text
                .linear_num_value_heads
                .is_multiple_of(text.linear_num_key_heads)
        {
            bail!(
                "{} value heads cannot share {} key heads evenly",
                text.linear_num_value_heads,
                text.linear_num_key_heads
            );
        }

        let head_dim = text
            .head_dim
            .unwrap_or(text.hidden_size / text.num_attention_heads);
        let rope = text.rope_parameters.clone();
        let rope_theta = rope
            .as_ref()
            .map(|r| r.rope_theta)
            .or(text.rope_theta)
            .unwrap_or_else(default_rope_theta);
        let factor = rope
            .as_ref()
            .map(|r| r.partial_rotary_factor)
            .or(text.partial_rotary_factor)
            .unwrap_or_else(default_partial_rotary_factor);
        // The same truncation transformers applies: int(head_dim * factor).
        let rotary_dim = (head_dim as f32 * factor) as usize;
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > head_dim {
            bail!("a rotary width of {rotary_dim} does not fit a head of {head_dim}");
        }

        let eos_token_id = match text
            .eos_token_id
            .or_else(|| root.get("eos_token_id").cloned())
        {
            Some(serde_json::Value::Number(n)) => vec![n.as_u64().context("eos_token_id")? as u32],
            Some(serde_json::Value::Array(ids)) => ids
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect(),
            _ => Vec::new(),
        };
        let tie_word_embeddings = text
            .tie_word_embeddings
            .or_else(|| root.get("tie_word_embeddings").and_then(|v| v.as_bool()))
            .unwrap_or(false);

        Ok(Self {
            hidden_size: text.hidden_size,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim,
            intermediate_size: text.intermediate_size,
            moe,
            vocab_size: text.vocab_size,
            max_position_embeddings: text.max_position_embeddings,
            rms_norm_eps: text.rms_norm_eps,
            layer_kinds,
            conv_kernel: text.linear_conv_kernel_dim,
            linear_key_head_dim: text.linear_key_head_dim,
            linear_value_head_dim: text.linear_value_head_dim,
            linear_num_key_heads: text.linear_num_key_heads,
            linear_num_value_heads: text.linear_num_value_heads,
            rope_theta,
            rotary_dim,
            eos_token_id,
            bos_token_id: text.bos_token_id,
            pad_token_id: text.pad_token_id,
            tie_word_embeddings,
            prefix: prefix.to_string(),
            llama_cpp_layout: false,
        })
    }

    /// Reads the `qwen35.*` metadata llama.cpp's converter writes.
    pub fn from_gguf(weights: &ModelWeights) -> Result<Self> {
        let loader = weights.loader();
        let arch = loader.get_string("general.architecture").unwrap_or("");
        // The dense models are `qwen35`, the mixture-of-experts ones `qwen35moe`,
        // and each names its metadata after itself.
        if !matches!(arch, "qwen35" | "qwen35moe") {
            bail!("this GGUF holds a '{arch}' model, not Qwen 3.5");
        }
        let int = |key: &str| {
            loader
                .get_u32(&format!("{arch}.{key}"))
                .map(|v| v as usize)
                .with_context(|| format!("the GGUF has no {arch}.{key}"))
        };
        let float = |key: &str| loader.get_f32(&format!("{arch}.{key}"));

        // Converters that keep the multi-token-prediction head count it as blocks.
        let mtp = loader
            .get_u32(&format!("{arch}.nextn_predict_layers"))
            .unwrap_or(0) as usize;
        let num_hidden_layers = int("block_count")?
            .checked_sub(mtp)
            .context("qwen35.nextn_predict_layers exceeds the block count")?;
        // The converter lists which layers are recurrent when the checkpoint
        // names its layer types; otherwise every `interval`-th is attention.
        // GGUF files are read natively only, so wasm never has the list.
        #[cfg(not(target_arch = "wasm32"))]
        let listed = loader
            .as_any()
            .downcast_ref::<kjarni_transformers::weights::GgufLoader>()
            .and_then(|g| {
                g.metadata()
                    .get(&format!("{arch}.attention.recurrent_layers"))
            });
        #[cfg(target_arch = "wasm32")]
        let listed: Option<&serde_json::Value> = None;
        let recurrent: Option<Vec<bool>> = listed
            .and_then(|v| v.as_array())
            .map(|flags| {
                flags
                    .iter()
                    .map(|f| f.as_bool().or_else(|| f.as_u64().map(|n| n != 0)))
                    .collect::<Option<Vec<bool>>>()
                    .context("attention.recurrent_layers is not a list of booleans")
            })
            .transpose()?;
        let interval = int("full_attention_interval").unwrap_or(4);
        let layer_kinds: Vec<LayerKind> = (0..num_hidden_layers)
            .map(|i| {
                let linear = match &recurrent {
                    Some(flags) => flags.get(i).copied().unwrap_or(false),
                    None => (i + 1) % interval != 0,
                };
                if linear {
                    LayerKind::LinearAttention
                } else {
                    LayerKind::FullAttention
                }
            })
            .collect();

        let num_k = int("ssm.group_count")?;
        let num_v = int("ssm.time_step_rank")?;
        if num_k == 0 || num_v == 0 || !num_v.is_multiple_of(num_k) {
            bail!("{num_v} value heads cannot share {num_k} key heads evenly");
        }
        let head_dim = int("attention.key_length")?;
        let rotary_dim = int("rope.dimension_count")?;
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > head_dim {
            bail!("a rotary width of {rotary_dim} does not fit a head of {head_dim}");
        }
        let vocab_size = *weights
            .tensor_shape("model.embed_tokens.weight")?
            .first()
            .context("the token embedding has no shape")?;
        let eos = loader.get_u32("tokenizer.ggml.eos_token_id");
        let moe = if arch == "qwen35moe" {
            let num_experts = int("expert_count")?;
            let top_k = int("expert_used_count")?;
            if top_k == 0 || top_k > num_experts {
                bail!("each token cannot go through {top_k} of {num_experts} experts");
            }
            Some(MoeDims {
                num_experts,
                top_k,
                expert_size: int("expert_feed_forward_length")?,
                shared_size: int("expert_shared_feed_forward_length").unwrap_or(0),
            })
        } else {
            None
        };

        Ok(Self {
            hidden_size: int("embedding_length")?,
            num_hidden_layers,
            num_attention_heads: int("attention.head_count")?,
            num_key_value_heads: int("attention.head_count_kv")?,
            head_dim,
            intermediate_size: match moe {
                Some(_) => 0,
                None => int("feed_forward_length")?,
            },
            moe,
            vocab_size,
            max_position_embeddings: int("context_length")?,
            rms_norm_eps: float("attention.layer_norm_rms_epsilon")
                .unwrap_or_else(default_rms_norm_eps),
            layer_kinds,
            conv_kernel: int("ssm.conv_kernel")?,
            linear_key_head_dim: int("ssm.state_size")?,
            linear_value_head_dim: int("ssm.inner_size")? / num_v,
            linear_num_key_heads: num_k,
            linear_num_value_heads: num_v,
            rope_theta: float("rope.freq_base").unwrap_or_else(default_rope_theta),
            rotary_dim,
            eos_token_id: eos.into_iter().collect(),
            bos_token_id: None,
            pad_token_id: loader.get_u32("tokenizer.ggml.padding_token_id"),
            // `output.weight` is absent when the head shares the embedding.
            tie_word_embeddings: !weights.contains("lm_head.weight"),
            prefix: "model.".to_string(),
            llama_cpp_layout: true,
        })
    }

    /// The full name of tensor `name` in layer `i`, or of a model-level one.
    pub fn tensor(&self, name: &str) -> String {
        format!("{}{name}", self.prefix)
    }

    pub fn layer_tensor(&self, i: usize, name: &str) -> String {
        format!("{}layers.{i}.{name}", self.prefix)
    }

    /// Width of the channels the depthwise convolution runs over: q, k and v.
    pub fn conv_dim(&self) -> usize {
        2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn num_full_attention_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count()
    }
}

impl ModelConfig for Qwen3_5Config {
    fn model_type(&self) -> &str {
        "qwen3_5"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_position_embeddings,
            norm_eps: self.rms_norm_eps,
            activation: Activation::SilU,
            rope_theta: Some(self.rope_theta),
            rope_scaling: None,
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: true,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::RMSNorm,
            no_scale_qk: false,
            problem_type: None,
            decoder_layers: None,
            intermediate_size: self.intermediate_size,
        }
    }

    /// The names the shared pipeline reads: the embedding table, the LM head and
    /// the final norm. The decoder builds its layers itself, because half of what
    /// a Qwen 3.5 layer holds has no slot in this layout.
    fn layout(&self) -> ModelLayout {
        let layer = |name: &str| format!("{}layers.{{}}.{name}", self.prefix);
        let embed = self.tensor("embed_tokens.weight");
        ModelLayout {
            token_embedding: embed.clone(),
            lm_head: if self.tie_word_embeddings {
                embed
            } else {
                "lm_head.weight".to_string()
            },
            encoder: None,
            decoder: Some(DecoderLayout {
                position_embedding: None,
                token_type_embedding: None,
                embedding_norm_weight: None,
                embedding_norm_bias: None,
                final_norm_weight: Some(self.tensor("norm.weight")),
                final_norm_bias: None,
                layer: DecoderLayerLayout {
                    self_attn: AttentionLayout {
                        q_weight: layer("self_attn.q_proj.weight"),
                        q_bias: None,
                        k_weight: layer("self_attn.k_proj.weight"),
                        k_bias: None,
                        v_weight: layer("self_attn.v_proj.weight"),
                        v_bias: None,
                        o_weight: layer("self_attn.o_proj.weight"),
                        o_bias: None,
                        norm_weight: layer("input_layernorm.weight"),
                        norm_bias: None,
                    },
                    cross_attn: None,
                    ffn: FeedForwardLayout {
                        up_weight: layer("mlp.up_proj.weight"),
                        up_bias: None,
                        down_weight: layer("mlp.down_proj.weight"),
                        down_bias: None,
                        gate_weight: Some(layer("mlp.gate_proj.weight")),
                        gate_bias: None,
                        norm_weight: layer("post_attention_layernorm.weight"),
                        norm_bias: None,
                    },
                },
            }),
        }
    }
}
