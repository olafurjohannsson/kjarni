// mod beams;
mod generator;
mod gpu_backend;
mod cpu_backend;
// mod traits;

// pub use beams::{find_best_beams_and_get_indices, run_beam_search, run_beam_search_stream, BeamHypothesis};
pub use cpu_backend::CpuBackend;
pub use generator::Seq2SeqGenerator;
pub use gpu_backend::GpuBackend;
// pub use traits::{GenerationBackend, HasShape, StepInput};
