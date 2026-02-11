pub mod traits;
pub mod generator;
pub mod backend;

mod gpu;


pub mod prelude {
    pub use crate::cpu::decoder::{
        CpuDecoderBackend,
        CpuRoPEDecoderLayer,
        DecoderAttention,
        DecoderLayer,
    };
    pub use crate::gpu::{
        decoder::backend::GpuDecoderBackend,
    };
    pub use crate::decoder::{
        backend::AnyDecoderBackend,
        generator::DecoderGenerator,
        gpu::{
            GpuPreNormDecoderLayer,
            GpuRoPEDecoderLayer,
        },
        traits::{
            CpuDecoder,
            CpuDecoderOps,
            DecoderGenerationBackend,
            DecoderLanguageModel,
            GpuDecoder,
            GpuDecoderOps,
        },
    };
}

#[cfg(test)]
mod test_generator;