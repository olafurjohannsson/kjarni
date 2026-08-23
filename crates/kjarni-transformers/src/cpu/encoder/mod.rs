pub mod classifier;
pub mod config;
mod transformer_encoder;
pub mod encoder_layer;
pub mod encoder_self_attention;
pub mod pooler;
pub mod traits;
pub mod buffers;
pub mod qkv_projection;
pub use encoder_self_attention::EncoderSelfAttention;
pub use traits::{CpuEncoder, CpuEncoderOps, SentenceEncoderModel};
#[cfg(not(target_arch = "wasm32"))]
pub use traits::{GpuEncoder, GpuEncoderOps};
pub use transformer_encoder::CpuTransformerEncoder;

pub mod prelude {
    pub use crate::cpu::encoder::{
        classifier::CpuSequenceClassificationHead,
        config::{EncodingConfig, PoolingStrategy},
        encoder_self_attention::EncoderSelfAttention,
        pooler::{CpuPooler, StandardCpuPooler},
        traits::{CpuEncoder, CpuEncoderOutput, EncoderLanguageModel},
    };

    #[cfg(not(target_arch = "wasm32"))]
    pub use crate::cpu::encoder::{
        classifier::GpuSequenceClassificationHead,
        pooler::GpuPooler,
        traits::{GpuEncoder, GpuEncoderOutput},
    };
}
