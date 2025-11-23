
mod beams;
mod generator;
mod gpu_backend;
mod cpu_backend;
mod traits;

pub use generator::Seq2SeqGenerator;
pub use traits::{GenerationBackend, StepInput, HasShape};
pub use gpu_backend::GpuBackend;
pub use beams::{BeamHypothesis, find_best_beams_and_get_indices, run_beam_search};