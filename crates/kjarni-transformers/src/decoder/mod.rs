pub mod backend;
pub mod generator;
pub mod traits;

#[cfg(not(target_arch = "wasm32"))]
mod gpu;

pub mod prelude {
    pub use crate::cpu::decoder::{
        CpuDecoderBackend, CpuRoPEDecoderLayer, DecoderAttention, DecoderLayer,
    };
    #[cfg(not(target_arch = "wasm32"))]
    pub use crate::decoder::gpu::{GpuPreNormDecoderLayer, GpuRoPEDecoderLayer};
    pub use crate::decoder::{
        backend::AnyDecoderBackend,
        generator::DecoderGenerator,
        traits::{
            CpuDecoder, CpuDecoderOps, DecoderGenerationBackend, DecoderLanguageModel, GpuDecoder,
            GpuDecoderOps,
        },
    };
    #[cfg(not(target_arch = "wasm32"))]
    pub use crate::gpu::decoder::backend::GpuDecoderBackend;
}

#[cfg(test)]
mod test_generator;
