pub mod classifier;
pub mod config;
mod transformer_encoder;
pub mod encoder_layer;
pub mod encoder_self_attention;
mod gpu;
pub mod pooler;
pub mod traits;
pub mod buffers;
pub mod qkv_projection;
pub use encoder_self_attention::EncoderSelfAttention;
pub use gpu::GpuTransformerEncoder;
pub use traits::{CpuEncoder, CpuEncoderOps, GpuEncoder, GpuEncoderOps, SentenceEncoderModel};
pub use transformer_encoder::CpuTransformerEncoder;

pub mod prelude {
    pub use crate::cpu::encoder::{
        classifier::{CpuSequenceClassificationHead, GpuSequenceClassificationHead},
        config::{EncodingConfig, PoolingStrategy},
        encoder_self_attention::EncoderSelfAttention,
        pooler::{CpuPooler, GpuPooler, StandardCpuPooler},
        traits::{
            CpuEncoder, CpuEncoderOutput, EncoderLanguageModel, GpuEncoder,
            GpuEncoderOutput,
        },
    };
}
