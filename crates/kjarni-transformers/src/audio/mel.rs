//! Audio preprocessing frontend for speech models (Whisper).
//!
//! This module handles all audio-specific processing:
//! - Mel spectrogram computation
//! - Convolutional frontend
//! - Position encoding for audio
//!
//! The output is hidden states that can be fed into any Seq2SeqEncoder.

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Array3, s};

use crate::weights::ModelWeights;

// =============================================================================
// Mel Spectrogram
// =============================================================================

/// Configuration for mel spectrogram computation.
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub chunk_length_secs: usize, // 30 seconds for Whisper
    /// Use Whisper-style normalization
    pub whisper_normalize: bool,
    /// Use centered STFT (pads signal)
    pub center: bool,
    /// Use power spectrum (magnitude^2) vs magnitude
    pub power: bool,
    /// Maximum frequency for mel filterbank (None = sr/2)
    pub fmax: Option<f32>,
    /// Minimum frequency for mel filterbank
    pub fmin: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            chunk_length_secs: 30,
            whisper_normalize: false,
            center: false,
            power: false,
            fmax: None,
            fmin: 0.0,
        }
    }
}
impl MelConfig {
    /// Config matching Whisper's exact implementation
    pub fn whisper() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            chunk_length_secs: 30,
            whisper_normalize: true,
            center: true,
            power: true,
            fmax: Some(8000.0), // Whisper uses 8000 Hz max
            fmin: 0.0,
        }
    }
}
/// Compute mel spectrogram from raw audio.
pub fn compute_mel_spectrogram(audio: &[f32], config: &MelConfig) -> Result<Array2<f32>> {
    let fmax = config.fmax.unwrap_or(config.sample_rate as f32 / 2.0);

    // Optionally pad for centered STFT
    let audio_processed: Vec<f32> = if config.center {
        pad_reflect(audio, config.n_fft / 2)
    } else {
        audio.to_vec()
    };

    let n_samples = audio_processed.len();
    // let n_frames = 1 + (n_samples - config.n_fft) / config.hop_length;
    // let n_frames = (n_samples - config.n_fft) / config.hop_length + 1;
    // Or for exact Whisper behavior (fixed 3000 frames for 30s):
    let n_frames = if config.whisper_normalize {
        3000
    } else {
        1 + (n_samples - config.n_fft) / config.hop_length
    };

    // Create mel filterbank (librosa-compatible)
    let mel_filters = create_mel_filterbank_librosa(
        config.sample_rate,
        config.n_fft,
        config.n_mels,
        config.fmin,
        fmax,
    )?;

    // Compute STFT
    let mut spectrogram = Array2::<f32>::zeros((config.n_fft / 2 + 1, n_frames));
    let window = hann_window(config.n_fft);

    for frame_idx in 0..n_frames {
        let start = frame_idx * config.hop_length;
        let end = start + config.n_fft;

        if end > n_samples {
            break;
        }

        let windowed: Vec<f32> = audio_processed[start..end]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();

        let magnitudes = compute_fft_magnitude(&windowed)?;

        for (i, &mag) in magnitudes.iter().enumerate() {
            spectrogram[[i, frame_idx]] = if config.power { mag * mag } else { mag };
        }
    }

    // Apply mel filterbank
    let mel_spec = mel_filters.dot(&spectrogram);

    // Log scale with appropriate normalization
    let log_mel = if config.whisper_normalize {
        whisper_log_mel(&mel_spec)
    } else {
        mel_spec.mapv(|x| (x.max(1e-10)).ln())
    };

    Ok(log_mel)
}

/// Whisper's specific log mel normalization
fn whisper_log_mel(mel_spec: &Array2<f32>) -> Array2<f32> {
    // Log10 scale
    let log_spec = mel_spec.mapv(|x| (x.max(1e-10)).log10());

    // Find max value
    let max_val = log_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Clamp to max - 8.0
    let clamped = log_spec.mapv(|x| x.max(max_val - 8.0));

    // Normalize to roughly [-1, 1] range
    clamped.mapv(|x| (x + 4.0) / 4.0)
}

/// Reflect padding (matches numpy's 'reflect' mode)
fn pad_reflect(audio: &[f32], pad_len: usize) -> Vec<f32> {
    let n = audio.len();
    let mut padded = Vec::with_capacity(n + 2 * pad_len);

    // Left padding (reflect)
    for i in (1..=pad_len).rev() {
        let idx = if i < n { i } else { n - 1 };
        padded.push(audio[idx]);
    }

    // Original audio
    padded.extend_from_slice(audio);

    // Right padding (reflect)
    for i in 0..pad_len {
        let idx = if n >= 2 + i { n - 2 - i } else { 0 };
        padded.push(audio[idx]);
    }

    padded
}

/// Create mel filterbank matrix.
fn create_mel_filterbank(config: &MelConfig) -> Result<Array2<f32>> {
    let n_fft_bins = config.n_fft / 2 + 1;
    let mut filters = Array2::<f32>::zeros((config.n_mels, n_fft_bins));

    // Convert Hz to Mel scale
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(config.sample_rate as f32 / 2.0);

    // Create mel points
    let mel_points: Vec<f32> = (0..=config.n_mels + 1)
        .map(|i| mel_low + (mel_high - mel_low) * (i as f32) / (config.n_mels + 1) as f32)
        .collect();

    // Convert back to Hz and then to FFT bins
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((config.n_fft as f32 + 1.0) * hz / config.sample_rate as f32) as usize)
        .collect();

    // Create triangular filters
    for m in 0..config.n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in left..center {
            if k < n_fft_bins {
                filters[[m, k]] = (k - left) as f32 / (center - left).max(1) as f32;
            }
        }

        for k in center..right {
            if k < n_fft_bins {
                filters[[m, k]] = (right - k) as f32 / (right - center).max(1) as f32;
            }
        }
    }

    Ok(filters)
}

/// Create mel filterbank matching librosa/HuggingFace exactly.
/// Uses Slaney mel scale and Slaney normalization.
pub fn create_mel_filterbank_librosa(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Result<Array2<f32>> {
    let n_fft_bins = n_fft / 2 + 1;
    let sr = sample_rate as f32;

    // Slaney mel scale (NOT HTK!) - this is what librosa uses by default
    let f_sp = 200.0 / 3.0; // ~66.67
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp; // 15.0
    let logstep = 6.4_f32.ln() / 27.0;

    let hz_to_mel = |hz: f32| -> f32 {
        if hz < min_log_hz {
            hz / f_sp
        } else {
            min_log_mel + (hz / min_log_hz).ln() / logstep
        }
    };

    let mel_to_hz = |mel: f32| -> f32 {
        if mel < min_log_mel {
            mel * f_sp
        } else {
            min_log_hz * (logstep * (mel - min_log_mel)).exp()
        }
    };

    // Get n_mels + 2 mel center frequencies
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / (n_mels + 1) as f32)
        .collect();

    // Convert to Hz
    let mel_f: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Differences between adjacent mel frequencies
    let fdiff: Vec<f32> = mel_f.windows(2).map(|w| w[1] - w[0]).collect();

    // FFT bin center frequencies
    let fft_freqs: Vec<f32> = (0..n_fft_bins)
        .map(|i| sr * i as f32 / n_fft as f32)
        .collect();

    let mut weights = Array2::<f32>::zeros((n_mels, n_fft_bins));

    // Build triangular filters
    for i in 0..n_mels {
        for k in 0..n_fft_bins {
            let lower = (fft_freqs[k] - mel_f[i]) / fdiff[i];
            let upper = (mel_f[i + 2] - fft_freqs[k]) / fdiff[i + 1];
            weights[[i, k]] = 0.0_f32.max(lower.min(upper));
        }
    }

    // Slaney normalization
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for k in 0..n_fft_bins {
            weights[[i, k]] *= enorm;
        }
    }

    Ok(weights)
}

/// Hann window function.
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos()))
        .collect()
}

/// Compute FFT magnitude using DFT (replace with rustfft for production).
fn compute_fft_magnitude(samples: &[f32]) -> Result<Vec<f32>> {
    let n = samples.len();
    let n_bins = n / 2 + 1;
    let mut magnitudes = vec![0.0f32; n_bins];

    for k in 0..n_bins {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;

        for (i, &sample) in samples.iter().enumerate() {
            let angle = -2.0 * std::f32::consts::PI * (k * i) as f32 / n as f32;
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }

        magnitudes[k] = (real * real + imag * imag).sqrt();
    }

    Ok(magnitudes)
}

// =============================================================================
// Convolutional Frontend
// =============================================================================

/// Whisper's convolutional frontend that converts mel spectrograms to hidden states.
///
/// Architecture:
/// - Conv1D(n_mels, hidden_size, kernel=3, stride=1, padding=1) + GELU
/// - Conv1D(hidden_size, hidden_size, kernel=3, stride=2, padding=1) + GELU
///
/// Input: [batch, n_mels, time]
/// Output: [batch, time/2, hidden_size]
pub struct AudioConvFrontend {
    conv1_weight: Array3<f32>, // [out_ch, in_ch, kernel]
    conv1_bias: Array1<f32>,
    conv2_weight: Array3<f32>,
    conv2_bias: Array1<f32>,
    position_embeddings: Array2<f32>, // [max_positions, hidden_size]
}

impl AudioConvFrontend {
    /// Load from weights.
    pub fn from_weights(
        weights: &ModelWeights,
        prefix: &str,
        max_positions: usize,
    ) -> Result<Self> {
        let conv1_weight = weights.get_array3(&format!("{}.conv1.weight", prefix))?;
        let conv1_bias = weights.get_array1(&format!("{}.conv1.bias", prefix))?;
        let conv2_weight = weights.get_array3(&format!("{}.conv2.weight", prefix))?;
        let conv2_bias = weights.get_array1(&format!("{}.conv2.bias", prefix))?;

        // Try to load learned positions, fallback to sinusoidal
        let position_embeddings = weights
            .get_array2(&format!("{}.embed_positions.weight", prefix))
            .unwrap_or_else(|_| {
                let hidden_size = conv2_weight.dim().0;
                create_sinusoidal_embeddings(max_positions, hidden_size)
            });

        Ok(Self {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            position_embeddings,
        })
    }

    /// Process mel spectrogram to hidden states.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, n_mels, time]
    ///
    /// # Returns
    /// Hidden states [batch, time/2, hidden_size] ready for transformer encoder
    pub fn forward(&self, mel: &Array3<f32>) -> Result<Array3<f32>> {
        // Conv1: kernel=3, stride=1, padding=1
        let x = self.conv1d(mel, &self.conv1_weight, &self.conv1_bias, 1, 1)?;
        let x = gelu_3d(&x);

        // Conv2: kernel=3, stride=2, padding=1 (downsamples time by 2)
        let x = self.conv1d(&x, &self.conv2_weight, &self.conv2_bias, 2, 1)?;
        let x = gelu_3d(&x);

        // Transpose: [batch, hidden, time] -> [batch, time, hidden]
        let x = x.permuted_axes([0, 2, 1]).as_standard_layout().to_owned();

        // Add position embeddings
        let (batch, seq_len, hidden_size) = x.dim();
        let mut output = x;

        for b in 0..batch {
            for s in 0..seq_len.min(self.position_embeddings.nrows()) {
                for h in 0..hidden_size {
                    output[[b, s, h]] += self.position_embeddings[[s, h]];
                }
            }
        }

        Ok(output)
    }

    /// 1D convolution implementation.
    fn conv1d(
        &self,
        input: &Array3<f32>,
        weight: &Array3<f32>,
        bias: &Array1<f32>,
        stride: usize,
        padding: usize,
    ) -> Result<Array3<f32>> {
        let (batch, in_channels, in_time) = input.dim();
        let (out_channels, weight_in_ch, kernel_size) = weight.dim();

        assert_eq!(in_channels, weight_in_ch, "Channel mismatch");

        let out_time = (in_time + 2 * padding - kernel_size) / stride + 1;
        let mut output = Array3::<f32>::zeros((batch, out_channels, out_time));

        // TODO: Replace with optimized SIMD kernel
        for b in 0..batch {
            for oc in 0..out_channels {
                for t in 0..out_time {
                    let t_start = t * stride;
                    let mut sum = bias[oc];

                    for ic in 0..in_channels {
                        for k in 0..kernel_size {
                            let t_in = t_start + k;
                            if t_in >= padding && t_in < in_time + padding {
                                let actual_t = t_in - padding;
                                sum += input[[b, ic, actual_t]] * weight[[oc, ic, k]];
                            }
                        }
                    }

                    output[[b, oc, t]] = sum;
                }
            }
        }

        Ok(output)
    }
}

/// GELU activation for 3D arrays.
fn gelu_3d(x: &Array3<f32>) -> Array3<f32> {
    x.mapv(|v| v * 0.5 * (1.0 + (v * 0.7978845608 * (1.0 + 0.044715 * v * v)).tanh()))
}

/// Create sinusoidal position embeddings.
fn create_sinusoidal_embeddings(max_len: usize, dim: usize) -> Array2<f32> {
    let mut embeddings = Array2::<f32>::zeros((max_len, dim));

    for pos in 0..max_len {
        for i in 0..dim / 2 {
            let angle = pos as f32 / 10000_f32.powf(2.0 * i as f32 / dim as f32);
            embeddings[[pos, 2 * i]] = angle.sin();
            embeddings[[pos, 2 * i + 1]] = angle.cos();
        }
    }

    embeddings
}

// =============================================================================
// High-Level Audio Pipeline
// =============================================================================

/// Complete audio processing pipeline for Whisper-style models.
///
/// Combines mel spectrogram computation and convolutional frontend.
pub struct AudioPipeline {
    mel_config: MelConfig,
    frontend: AudioConvFrontend,
}

impl AudioPipeline {
    /// Create from weights.
    pub fn from_weights(
        weights: &ModelWeights,
        prefix: &str,
        mel_config: MelConfig,
        max_positions: usize,
    ) -> Result<Self> {
        let frontend = AudioConvFrontend::from_weights(weights, prefix, max_positions)?;
        Ok(Self {
            mel_config,
            frontend,
        })
    }

    /// Process raw audio to hidden states.
    ///
    /// # Arguments
    /// * `audio` - Raw audio samples [n_samples] or [batch, n_samples]
    ///
    /// # Returns
    /// Hidden states ready for Seq2SeqEncoder
    pub fn process(&self, audio: &[f32]) -> Result<Array3<f32>> {
        // 1. Compute mel spectrogram
        let mel = compute_mel_spectrogram(audio, &self.mel_config)?;

        // 2. Add batch dimension: [n_mels, time] -> [1, n_mels, time]
        let mel_batch = mel.insert_axis(ndarray::Axis(0));

        // 3. Run through conv frontend
        self.frontend.forward(&mel_batch)
    }

    /// Process batch of audio samples.
    pub fn process_batch(&self, audio_batch: &[Vec<f32>]) -> Result<Array3<f32>> {
        // Process each audio independently
        let hidden_states: Vec<Array3<f32>> = audio_batch
            .iter()
            .map(|audio| self.process(audio))
            .collect::<Result<Vec<_>>>()?;

        // Stack into batch (assumes same length - pad if needed)
        // For now, simple concatenation along batch axis
        let batch_size = hidden_states.len();
        if batch_size == 0 {
            return Err(anyhow!("Empty audio batch"));
        }

        let (_, seq_len, hidden_size) = hidden_states[0].dim();
        let mut output = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));

        for (b, hs) in hidden_states.iter().enumerate() {
            output
                .slice_mut(s![b, .., ..])
                .assign(&hs.slice(s![0, .., ..]));
        }

        Ok(output)
    }
}

#[cfg(test)]
mod audio_frontend_tests {
    use super::*;
    use crate::weights::ModelWeights;
    use anyhow::Result;
    use ndarray::Array3;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use tempfile::TempDir;

    // Helper to create temporary .safetensors model directory
    fn create_model_weights(
        weights_map: HashMap<String, Vec<f32>>,
        shapes: HashMap<String, Vec<usize>>,
    ) -> Result<(ModelWeights, TempDir)> {
        let dir = tempfile::tempdir()?;
        let stored_data: Vec<(String, Vec<usize>, Vec<u8>)> = weights_map
            .into_iter()
            .map(|(k, v)| {
                let shape = shapes.get(&k).unwrap().clone();
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                (k, shape, bytes)
            })
            .collect();

        let mut tensors = HashMap::new();
        for (k, shape, bytes) in &stored_data {
            tensors.insert(
                k.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            );
        }

        let file_path = dir.path().join("model.safetensors");
        safetensors::serialize_to_file(&tensors, &None, &file_path)?;
        let weights = ModelWeights::new(dir.path())?;

        Ok((weights, dir))
    }

    // ==========================================
    // GOLDEN VALUES
    // ==========================================

    // conv1_weight Shape: [8, 4, 3]
    fn get_conv1_weight_data() -> Vec<f32> {
        vec![
            0.087017, 0.096910, 0.058030, 0.054364, 0.056184, -0.060715, -0.030015, 0.043221,
            0.025469, 0.042495, -0.001893, -0.096370, 0.047400, -0.062899, -0.082637, 0.098893,
            0.044083, 0.049698, -0.043813, 0.036851, 0.008499, 0.068211, 0.011698, 0.089378,
            -0.023422, 0.092281, 0.099036, 0.011828, -0.073319, 0.000039, 0.079599, -0.068322,
            -0.035738, 0.036160, 0.020319, 0.094004, 0.016471, -0.000765, 0.013143, -0.020569,
            -0.092061, -0.035075, -0.090651, 0.073168, 0.006051, 0.011644, -0.006940, -0.051309,
            -0.070549, -0.015306, 0.030256, -0.054724, -0.061868, 0.064970, -0.034720, 0.056338,
            0.095733, 0.093991, -0.077316, 0.007417, -0.078411, 0.004127, 0.013907, -0.043282,
            0.029681, -0.031349, 0.056341, 0.000953, 0.069412, 0.067590, 0.022330, 0.018669,
            -0.045260, -0.066036, 0.067421, 0.065992, 0.015682, 0.083902, 0.052215, 0.065323,
            -0.019181, 0.014295, -0.089909, -0.091371, 0.047547, -0.060279, 0.076162, -0.080690,
            -0.082782, -0.082086, 0.064178, -0.066284, -0.085758, 0.042242, -0.020171, -0.095246,
        ]
    }

    // conv1_bias Shape: [8]
    fn get_conv1_bias_data() -> Vec<f32> {
        vec![
            0.070922, 0.072477, 0.080214, 0.062773, -0.077527, -0.044022, -0.039299, -0.097251,
        ]
    }

    // conv2_weight Shape: [8, 8, 3]
    fn get_conv2_weight_data() -> Vec<f32> {
        vec![
            0.069883, -0.040468, 0.034323, -0.050850, 0.089612, 0.070462, 0.069017, 0.048945,
            0.083706, 0.032695, 0.017088, -0.022548, 0.081260, -0.096491, 0.071420, -0.009462,
            -0.006620, -0.065742, -0.043721, 0.018863, -0.040749, 0.035600, 0.014088, -0.085126,
            0.054654, -0.044029, -0.036130, -0.036804, 0.039512, -0.098606, -0.072068, 0.040244,
            -0.030300, -0.018975, -0.089849, 0.010141, 0.047063, -0.036096, 0.007643, 0.002836,
            -0.021175, 0.080658, 0.025752, 0.019007, -0.003111, -0.002096, 0.053699, 0.005751,
            0.022629, 0.081156, 0.070706, -0.036699, -0.081885, 0.026339, -0.068299, -0.099454,
            0.068669, -0.082882, 0.098703, -0.092126, -0.081424, 0.057493, 0.007716, -0.043586,
            0.037957, -0.064973, -0.055118, -0.096305, -0.051223, -0.085892, 0.085689, -0.043472,
            0.016967, -0.044947, -0.022910, 0.073051, 0.020352, 0.048282, 0.032675, -0.081502,
            0.060056, 0.081022, 0.029885, 0.079659, -0.047647, -0.065427, -0.080320, -0.015812,
            -0.009890, -0.056839, -0.034199, -0.032204, 0.096443, -0.007141, 0.081070, 0.009633,
            0.035634, 0.057626, -0.092042, -0.045075, -0.075999, 0.097147, 0.036697, -0.041913,
            -0.075412, 0.055377, 0.023212, -0.049197, -0.072575, -0.052188, 0.024351, -0.084195,
            -0.010552, 0.059354, 0.053413, 0.010793, 0.010625, 0.023365, 0.039055, 0.078927,
            0.027163, 0.054974, 0.069707, -0.077572, -0.053374, 0.054904, 0.020035, -0.057723,
            0.029550, -0.090910, -0.037208, 0.050426, -0.002635, -0.067578, 0.049430, 0.006949,
            -0.014311, -0.096171, -0.036072, 0.011045, -0.054343, 0.016829, -0.071038, 0.077164,
            0.077002, 0.058901, -0.011880, -0.099653, -0.039048, -0.008101, -0.092166, 0.078730,
            0.028465, 0.075868, -0.089469, -0.051344, 0.022537, -0.001524, -0.058158, 0.073118,
            0.041239, -0.004769, 0.095300, -0.097525, -0.090832, 0.019944, 0.090752, 0.056532,
            -0.055030, 0.071743, 0.051958, 0.005685, 0.063267, 0.019790, 0.000131, -0.027689,
            -0.053128, -0.063036, -0.013146, -0.068388, 0.042384, 0.028238, -0.097500, -0.073557,
            0.071464, -0.052425, -0.001031, -0.064051, -0.021871, 0.049319, 0.025417, 0.049083,
        ]
    }

    // conv2_bias Shape: [8]
    fn get_conv2_bias_data() -> Vec<f32> {
        vec![
            -0.005823, -0.079787, -0.001259, 0.059286, 0.023391, 0.020681, 0.019690, 0.082049,
        ]
    }

    // pos_embed Shape: [1500, 8]
    fn get_pos_embed_data() -> Vec<f32> {
        let pos_embed_data = vec![
            0.050447, -0.016355, -0.054254, 0.081373, -0.052024, -0.066849, 0.031237, -0.063857,
            -0.086287, -0.080977, 0.024659, -0.062840, 0.088463, -0.004013, -0.042471, -0.061319,
            -0.029895, 0.087241, -0.063833, -0.092588, 0.029753, 0.033909, -0.084726, -0.044887,
            0.086678, -0.047268, 0.041957, -0.048003, -0.088281, -0.096466, 0.076280, 0.030657,
            0.028439, 0.034418, 0.038539, 0.066018, 0.076721, 0.077113, 0.032134, 0.017859,
            0.021468, -0.080994, 0.044587, 0.024661, 0.043467, -0.024676, 0.083059, 0.012013,
            -0.023703, -0.002827, -0.039914, -0.014693, 0.084875, 0.033170, -0.011176, 0.003776,
            0.022843, -0.057711, -0.054850, 0.088176, 0.000545, 0.029762, -0.046019, -0.073506,
            0.019838, 0.054011, -0.094186, -0.098902, -0.033351, 0.064668, -0.089353, 0.079892,
            0.090435, -0.022442, 0.006388, -0.004130, 0.097426, 0.011113, 0.024435, -0.034216,
            -0.001221, -0.053031, 0.005028, -0.013876, 0.018638, 0.019442, -0.081820, 0.066809,
            0.055006, 0.070442, 0.073645, 0.025802, 0.054666, 0.049290, 0.059623, -0.060220,
            0.047380, -0.008012, 0.038680, 0.036759, -0.074538, -0.045491, -0.071190, -0.026586,
            0.032275, 0.023325, 0.079400, -0.072199, 0.031294, 0.081788, -0.077037, -0.075273,
            0.024067, -0.040587, 0.025574, -0.048333, 0.012883, -0.084761, 0.079675, -0.063854,
            -0.091765, -0.051791, 0.002124, 0.093974, 0.020908, -0.020908, -0.071974, 0.045645,
            0.007292, -0.028959, -0.028043, 0.014662, 0.058879, -0.060138, -0.024153, 0.047160,
            -0.068495, -0.002471, 0.080989, 0.069827, -0.080231, -0.046935, 0.016089, 0.063165,
            -0.040172, -0.090806, -0.003717, 0.062724, -0.051422, -0.092091, 0.006887, 0.083124,
            -0.081065, 0.021519, 0.063337, 0.003010, -0.053594, -0.036183, -0.065774, 0.007268,
            -0.043514, 0.003375, -0.022011, -0.068481, 0.096833, 0.023558, 0.049462, 0.059212,
            0.040288, -0.023662, -0.030690, -0.055138, -0.088960, -0.076356, -0.038083, 0.094196,
            -0.095427, -0.062985, -0.077790, -0.070067, 0.081777, -0.079099, -0.071555, -0.058496,
            0.026809, -0.031259, -0.066586, 0.082665, 0.015335, -0.073002, -0.037337, 0.061570,
            0.036305, -0.063710, -0.050808, -0.039224, 0.038197, -0.080195, 0.093481, 0.009770,
            0.039730, 0.029488, 0.077842, -0.032697, -0.037786, -0.065227, 0.081811, -0.008082,
            -0.001561, 0.088120, -0.025212, -0.033553, -0.041752, 0.057449, 0.012121, -0.053425,
            0.065711, 0.038986, -0.022910, 0.022968, -0.090633, -0.071060, -0.036686, 0.048161,
            -0.021826, 0.072965, -0.093278, 0.072756, -0.096688, -0.034352, 0.091713, -0.033421,
            -0.043606, 0.054024, -0.053576, 0.020727, -0.035876, 0.039187, -0.061317, -0.052333,
            -0.066429, -0.017737, 0.099296, 0.057495, -0.079351, 0.051060, 0.091597, 0.098964,
            0.009960, 0.026693, 0.072578, 0.096228, 0.003417, -0.017685, 0.081170, -0.026866,
            0.079700, 0.084621, 0.027867, -0.080516, 0.078073, 0.047764, 0.024016, 0.021653,
            0.047056, 0.059039, -0.065577, -0.079174, 0.051796, -0.041623, -0.087837, 0.083543,
            -0.077511, -0.034153, -0.062815, 0.086739, -0.012964, -0.052275, 0.083495, 0.093070,
            0.053013, 0.078330, -0.090047, -0.090375, 0.054647, -0.022453, 0.042521, 0.092932,
            -0.022847, -0.053129, 0.093698, -0.047939, -0.057301, 0.080251, -0.004111, -0.048773,
            0.035666, -0.050533, 0.003645, -0.078782, -0.024140, 0.077579, 0.014405, -0.013546,
            0.075833, 0.054160, 0.056118, 0.047452, -0.034218, -0.089030, -0.059217, -0.059440,
            0.027848, 0.080194, 0.092440, 0.080076, 0.014189, 0.068869, 0.005019, 0.046446,
            -0.066200, 0.096323, 0.046230, 0.000147, -0.039701, 0.017514, -0.011520, -0.064726,
            -0.061957, -0.039889, 0.022956, -0.064479, 0.055110, -0.055075, 0.060122, 0.003450,
            0.000172, 0.013505, -0.000754, 0.049426, -0.007997, -0.035306, 0.010425, -0.058553,
            0.040027, 0.010059, 0.050445, -0.017536, -0.000565, -0.047617, -0.044892, 0.002474,
            0.051190, -0.037835, 0.078589, -0.033762, -0.020558, 0.080131, -0.018151, 0.054387,
            -0.009846, 0.059439, 0.013006, -0.047121, -0.065873, 0.059362, 0.024574, -0.015531,
            -0.019105, -0.041510, -0.069264, 0.022184, -0.085028, -0.001327, 0.071461, 0.045624,
            0.054026, 0.018484, -0.055935, 0.050977, 0.078036, 0.040930, 0.069654, -0.080744,
            -0.087017, -0.023159, 0.026313, 0.064250, -0.029359, -0.044267, 0.064496, -0.003065,
            0.006782, -0.095412, -0.009289, 0.048513, 0.096569, 0.009317, 0.063519, 0.013900,
            -0.078919, -0.003370, -0.034173, -0.013580, -0.038897, 0.074756, -0.081633, -0.060108,
            -0.031051, 0.083781, -0.020463, 0.040806, 0.049028, 0.085592, -0.022976, 0.018681,
            -0.048808, -0.007601, -0.085698, -0.081594, 0.027199, -0.076641, 0.052265, 0.073911,
            0.035088, 0.038281, -0.090855, -0.046244, -0.072203, -0.094822, 0.042512, 0.022897,
            0.095148, -0.027551, -0.066210, -0.086082, 0.006158, -0.054804, 0.017741, 0.066148,
            0.085502, -0.016309, -0.065026, -0.054301, 0.063507, 0.025879, 0.006332, -0.097158,
            0.053863, 0.063262, 0.010320, -0.018019, -0.047454, 0.000817, 0.004968, 0.028209,
            -0.004238, 0.094808, 0.064460, 0.022577, -0.088780, -0.047748, 0.039130, -0.064626,
            0.027879, 0.024558, -0.062870, 0.019836, -0.012934, 0.074483, -0.022062, 0.087269,
            0.086883, -0.033154, 0.016780, -0.027336, 0.080077, 0.047945, 0.000274, 0.060476,
            -0.080295, -0.025573, 0.020413, -0.094741, 0.087404, 0.008216, 0.088344, -0.029105,
            0.005954, -0.027580, -0.040999, -0.037056, 0.069162, 0.089850, 0.028776, 0.055840,
            0.043499, 0.000125, -0.074434, 0.025086, -0.034362, 0.038288, -0.065068, -0.006797,
            -0.039307, -0.092509, -0.092449, 0.092262, -0.000109, -0.086204, 0.016397, -0.006276,
            0.082382, 0.094245, -0.025703, -0.060005, -0.057307, -0.026589, -0.054893, 0.090134,
            -0.038223, 0.051521, 0.045255, 0.023381, 0.048750, 0.084680, 0.042548, 0.044003,
            -0.083906, -0.008862, 0.002309, 0.078316, 0.042730, 0.077770, 0.077349, -0.046646,
            0.014187, 0.095353, 0.035352, 0.056541, 0.040647, -0.031391, 0.023879, -0.069096,
            0.092256, 0.005722, 0.077493, -0.070296, -0.091866, 0.011974, -0.012812, -0.040450,
            0.052395, -0.065052, 0.087629, -0.097621, 0.020508, -0.058470, 0.021694, 0.098758,
            -0.012003, 0.023252, -0.039354, -0.051315, -0.085420, 0.057720, -0.001111, 0.034630,
            -0.076909, -0.083256, -0.090929, -0.057628, -0.013002, -0.015089, 0.043970, -0.048050,
            0.097439, -0.045571, -0.094040, 0.075646, -0.064267, 0.009530, -0.009655, -0.020860,
            0.000065, -0.098336, 0.089924, -0.064918, 0.060970, -0.059132, -0.024625, -0.002365,
            -0.039414, -0.043323, -0.084173, -0.068684, -0.004692, -0.029232, -0.014044, 0.025510,
            -0.012820, -0.094664, -0.036778, 0.030608, 0.035990, 0.048391, 0.005538, 0.068368,
            -0.068431, -0.083920, -0.026663, 0.000674, 0.072607, -0.057669, -0.070036, -0.002705,
            0.025430, -0.091208, -0.040311, -0.078555, 0.023369, 0.009329, -0.088924, -0.066589,
            0.004624, -0.026499, -0.077347, -0.030303, -0.091838, -0.011419, -0.011656, 0.005242,
            0.008494, -0.017419, 0.079804, -0.035803, 0.045191, 0.037755, -0.012257, -0.073284,
            -0.084261, 0.027681, 0.030574, 0.060651, -0.044265, 0.080738, 0.099987, -0.015905,
            -0.038969, -0.031411, -0.086940, 0.057649, 0.010898, 0.046071, 0.064250, -0.042468,
            0.000931, 0.080839, -0.077090, 0.087921, 0.052000, -0.031406, -0.081651, 0.053413,
            0.047711, -0.045286, 0.047608, 0.029679, 0.098436, 0.051152, 0.015439, -0.098982,
            0.081385, 0.085860, -0.027719, -0.007820, -0.003896, -0.006106, 0.054361, 0.013517,
            -0.086970, 0.032174, -0.061426, 0.028551, 0.043528, -0.047750, 0.067742, 0.002554,
            -0.050296, 0.085690, 0.063900, 0.073125, -0.068853, -0.076212, -0.087923, 0.064653,
            0.058322, -0.060958, 0.039203, 0.020100, -0.042567, 0.003432, -0.060330, -0.029278,
            0.039541, 0.014121, -0.074418, 0.025165, -0.019955, -0.073461, -0.092457, 0.004312,
            -0.073597, -0.061128, 0.024097, 0.006166, -0.045274, 0.015503, 0.050377, -0.038615,
            -0.035664, 0.017658, 0.024461, 0.034563, 0.068100, -0.006684, 0.033220, -0.057544,
            -0.012752, 0.074077, -0.096874, -0.065781, -0.086443, 0.083204, -0.011290, 0.091587,
            -0.053898, -0.027193, 0.064364, -0.099598, -0.097457, -0.066967, 0.001701, -0.091998,
            0.018515, 0.084572, -0.020269, -0.017607, -0.088006, -0.021139, 0.008271, 0.020471,
            -0.032930, -0.061110, -0.052391, 0.068952, -0.054401, 0.007774, -0.017577, -0.040210,
            -0.004320, 0.075588, 0.075717, -0.020888, -0.020114, -0.056016, -0.046513, -0.017082,
            0.074203, 0.014246, -0.085588, -0.025705, -0.086389, 0.092413, 0.009258, -0.025381,
            0.040099, -0.093746, -0.033013, 0.009381, 0.072158, 0.081567, -0.003343, -0.099962,
            0.018473, -0.056394, 0.090194, -0.026926, -0.027215, -0.014852, 0.092687, 0.043852,
            -0.068962, 0.017524, 0.031797, -0.025974, 0.033197, -0.089762, -0.051549, 0.063884,
            0.092902, 0.031765, -0.056654, 0.036444, -0.057597, -0.080127, -0.065984, -0.009478,
            0.021235, 0.007446, 0.096111, -0.007095, 0.091992, -0.074318, -0.051481, 0.013711,
            0.059601, 0.029682, 0.098065, -0.053306, 0.093586, 0.031928, 0.026404, -0.002564,
            -0.071977, 0.053496, 0.021690, 0.068934, 0.073591, -0.084932, 0.092635, -0.023027,
            -0.066995, 0.087280, -0.015815, 0.016477, 0.015010, -0.053790, 0.063414, 0.037919,
            0.080549, -0.027793, -0.073376, 0.019587, -0.026869, 0.000052, -0.081332, 0.089685,
            -0.052593, -0.009035, 0.002537, -0.043443, 0.059414, -0.081482, -0.013451, -0.064760,
            0.059299, -0.072800, 0.079172, -0.086491, -0.099944, 0.002439, -0.098446, 0.093151,
            0.093410, -0.045666, 0.074748, -0.038459, 0.024463, -0.070091, -0.006770, -0.030533,
            0.022434, 0.032147, -0.093339, 0.006757, 0.077436, 0.080561, 0.045276, 0.002316,
            0.068609, 0.053066, -0.065778, -0.005704, -0.092640, 0.007041, 0.007594, -0.065836,
            0.023882, 0.069154, -0.085089, -0.089079, -0.048867, 0.012958, -0.097057, -0.019902,
            0.076782, -0.082535, -0.026712, -0.021139, 0.060438, -0.037186, 0.075914, -0.050120,
            0.037101, -0.018483, 0.087050, -0.049032, -0.037572, 0.090192, -0.079219, 0.028273,
            0.049706, -0.008685, 0.087919, -0.093009, -0.014523, -0.052621, 0.039400, 0.097776,
            -0.097887, -0.044576, -0.074054, -0.056090, 0.030232, 0.001474, -0.078324, -0.061322,
            -0.065494, 0.007587, -0.082355, -0.078191, -0.080120, 0.080660, 0.073039, -0.047316,
            0.023473, -0.013966, -0.040190, 0.046628, -0.067263, -0.000330, 0.074733, -0.086837,
            0.082448, 0.066056, 0.044223, -0.013067, -0.038898, -0.054902, -0.072504, -0.097479,
            -0.014345, 0.019882, -0.096798, 0.098437, -0.071452, 0.007894, -0.033892, 0.091009,
            -0.043532, 0.026122, 0.031873, 0.045856, -0.070106, -0.055382, 0.001114, -0.098634,
            -0.039024, -0.083907, 0.044202, 0.010362, -0.093448, 0.097795, 0.040994, -0.057313,
            -0.010068, 0.067816, -0.059431, 0.025010, 0.032520, -0.064287, -0.052553, -0.017734,
            -0.078225, 0.086056, 0.028554, -0.031832, -0.001176, 0.001093, 0.033119, 0.071031,
            -0.084037, -0.035571, 0.036056, -0.055852, -0.004927, -0.029257, 0.057298, -0.053619,
            0.090113, 0.047545, -0.071698, -0.019881, -0.009381, -0.097884, -0.085704, 0.082741,
            0.071850, -0.068101, -0.000984, 0.026869, 0.061018, 0.088578, 0.062735, 0.082933,
            -0.064834, -0.022793, 0.047849, -0.076975, -0.096841, 0.056364, 0.072914, 0.039965,
            0.063516, 0.046788, 0.013539, -0.022514, 0.013788, -0.034879, 0.081190, 0.084466,
            -0.030634, 0.044132, -0.027151, 0.080013, -0.089959, 0.014229, -0.007692, 0.068278,
            0.010597, 0.018973, -0.097919, -0.055015, 0.093344, -0.008962, 0.042747, -0.081757,
            0.051487, -0.043506, -0.010261, 0.094228, 0.070839, -0.087682, -0.044859, 0.059005,
            -0.018376, 0.078962, 0.041086, -0.011092, -0.062347, 0.005739, 0.078876, 0.076848,
            -0.043156, -0.027637, 0.032251, -0.084997, 0.013361, 0.025546, 0.098142, -0.012382,
            0.097292, 0.097260, -0.051457, 0.015547, 0.057317, -0.011613, -0.054774, 0.021323,
            0.044068, -0.081025, -0.080583, -0.075458, -0.076630, 0.087559, -0.044479, -0.011876,
            -0.026612, -0.051401, 0.084608, -0.035706, 0.060726, -0.046211, 0.098085, 0.025274,
            -0.017650, 0.000704, 0.041430, -0.053707, -0.029658, -0.074205, -0.067067, 0.048100,
            -0.008372, -0.049916, 0.004649, 0.057059, 0.091840, -0.058540, 0.040815, 0.009557,
            -0.050751, 0.022164, 0.011236, 0.059098, 0.081827, 0.046969, 0.018920, -0.075131,
            -0.083725, 0.046851, -0.061350, 0.001383, 0.076983, 0.050194, -0.012704, 0.015760,
            0.003440, 0.008632, 0.087203, -0.061179, 0.087050, 0.004800, 0.094841, -0.042636,
            0.061474, -0.054977, -0.081933, 0.072689, 0.034567, 0.098305, 0.040530, -0.042187,
            -0.059473, 0.081851, -0.019824, -0.059414, -0.073859, -0.066941, -0.042006, -0.080400,
            0.096246, 0.011554, 0.032992, -0.049196, 0.002819, 0.068163, -0.033891, 0.080737,
            -0.031090, -0.037387, -0.073605, -0.099108, 0.043599, -0.064513, -0.054825, -0.040907,
            -0.076482, -0.076868, -0.065287, 0.097493, 0.005809, -0.037685, -0.008337, 0.083595,
            -0.058168, 0.040388, -0.018631, -0.084963, 0.066140, 0.046197, 0.073496, 0.045782,
            0.016294, 0.030745, -0.074656, 0.096367, -0.064416, 0.070551, -0.063317, -0.072268,
            0.041212, 0.065537, -0.027237, 0.028008, -0.087163, -0.011229, 0.088230, 0.093439,
            -0.031842, 0.036948, 0.092227, 0.037086, 0.018331, -0.080580, 0.028748, 0.026698,
            -0.056330, 0.049933, -0.065231, 0.095734, -0.005032, 0.068083, -0.026258, -0.050797,
            0.072748, 0.060714, 0.098190, 0.029396, -0.041982, -0.058724, -0.095421, 0.006337,
            0.095732, -0.033148, -0.019301, 0.070436, 0.088955, 0.027166, 0.007942, 0.097790,
            0.083749, -0.063632, -0.069768, -0.022941, 0.039629, -0.061197, -0.042807, 0.008301,
            -0.025265, 0.067087, 0.063342, 0.068867, 0.053189, -0.057854, -0.076965, -0.067196,
            -0.012800, 0.078602, -0.022205, 0.092019, 0.074635, -0.094910, 0.046179, -0.042181,
            -0.095394, 0.035049, -0.034359, -0.088845, -0.070374, 0.073576, -0.023956, 0.078379,
            -0.034702, 0.097688, -0.025031, 0.094651, 0.038790, -0.075139, 0.043698, 0.041990,
            -0.083950, -0.086457, 0.005801, 0.054166, 0.085205, -0.027725, -0.093646, 0.045679,
            0.006973, 0.002550, 0.063134, 0.011941, 0.086685, 0.076769, 0.025605, 0.074489,
            0.032016, -0.094901, 0.089758, 0.003046, -0.012354, 0.082411, 0.087223, -0.079128,
            0.039393, -0.054973, -0.025696, 0.021485, -0.037112, -0.042215, 0.049787, -0.024047,
            0.037767, 0.095884, 0.024259, 0.023256, 0.087733, -0.037597, 0.000144, -0.092560,
            -0.042575, -0.007834, 0.090688, 0.056494, -0.018572, -0.027206, 0.024484, -0.043424,
            -0.014616, 0.026479, -0.030992, 0.073197, 0.039597, -0.010385, -0.017460, 0.009942,
            0.052287, 0.028031, -0.027264, 0.034863, -0.004247, -0.072980, -0.078826, 0.000748,
            0.086532, -0.038873, 0.062754, -0.029700, 0.059785, -0.022411, -0.041460, 0.029322,
            0.052906, -0.076216, -0.007296, 0.020755, 0.017987, 0.097487, -0.087661, 0.085338,
            0.051710, -0.094297, -0.038531, 0.034740, -0.075273, 0.093510, -0.037621, 0.038331,
            0.019301, -0.038929, -0.094109, 0.007312, -0.081711, -0.021842, 0.021026, -0.035017,
            0.057105, -0.089844, -0.078944, 0.053258, -0.052078, 0.082127, -0.020226, -0.017905,
            0.095356, -0.011089, -0.080180, -0.041658, 0.031118, 0.068104, 0.084319, -0.036711,
            0.014718, 0.072849, -0.040911, 0.006686, -0.048463, 0.014249, 0.097285, 0.084499,
            -0.065767, 0.049080, 0.094085, -0.017703, 0.035493, -0.089800, 0.055560, -0.018734,
            0.091847, 0.070044, -0.057483, -0.044222, -0.078633, -0.092383, -0.090727, 0.048409,
            0.066313, -0.063445, 0.069989, -0.031372, 0.029837, 0.083759, 0.077586, -0.079706,
            0.039359, -0.050227, -0.085867, -0.004431, -0.022396, 0.020992, 0.011281, 0.057889,
            0.051583, -0.089655, -0.087389, -0.063192, 0.051497, 0.092350, -0.058537, 0.018738,
            0.005582, 0.037319, 0.038626, 0.015581, 0.066526, -0.033242, 0.095464, -0.002825,
            -0.090631, 0.048732, 0.070401, 0.067047, 0.010994, 0.033833, 0.031035, -0.042698,
            -0.004000, 0.002361, -0.055371, -0.037305, -0.039841, -0.016414, -0.065800, 0.058000,
            -0.002522, 0.076919, -0.064199, 0.027770, -0.032798, 0.052984, -0.039987, -0.049595,
            0.007949, -0.057341, 0.069009, -0.003767, 0.056053, -0.029345, -0.024617, -0.094582,
            0.027945, 0.034107, 0.065654, 0.013253, -0.009564, -0.013436, 0.040783, 0.095648,
            0.006428, -0.084364, -0.089265, 0.038286, 0.089018, -0.052855, 0.050407, 0.083695,
            0.087529, 0.063532, -0.064884, 0.070150, 0.003514, -0.060030, -0.099270, -0.039269,
            0.083193, 0.099917, 0.073458, -0.070680, 0.057675, 0.055283, 0.090247, -0.058848,
            0.082829, -0.013069, 0.039949, -0.085477, -0.015496, -0.063281, -0.028440, -0.010367,
            -0.035827, 0.044370, 0.092667, 0.023223, -0.044105, -0.022307, -0.030944, 0.007417,
            -0.008899, 0.058479, 0.007654, -0.089681, -0.035774, -0.048442, 0.035130, 0.089639,
            0.081271, 0.023669, -0.064560, -0.003425, 0.015616, 0.002011, 0.055148, 0.048286,
            -0.030724, -0.037058, 0.011634, 0.009582, 0.002038, -0.096419, -0.093825, 0.089979,
            0.097574, -0.087655, -0.070056, -0.014537, -0.056603, -0.098046, 0.030948, -0.063971,
            0.032074, -0.064597, -0.005221, -0.031755, -0.018931, -0.043304, -0.083806, 0.080994,
            -0.008328, -0.053854, 0.093402, 0.012765, -0.023562, 0.029054, -0.006115, 0.025077,
            0.083970, 0.036935, 0.070488, 0.052877, -0.040273, -0.008223, -0.018847, 0.099411,
            -0.037885, 0.071486, -0.028951, 0.083632, 0.042649, 0.055126, -0.071510, -0.002961,
            0.093218, -0.079416, -0.021860, -0.098600, -0.092413, 0.073734, -0.059046, 0.009528,
            -0.085259, -0.075720, 0.090269, 0.019561, -0.027431, -0.026529, 0.096479, 0.071259,
            0.070786, 0.089942, 0.003788, 0.014916, 0.005851, -0.039965, -0.086801, 0.089242,
            0.082987, 0.009379, -0.035525, 0.086887, -0.044460, -0.003798, -0.034286, -0.058341,
            -0.076244, 0.097593, 0.087793, 0.049155, -0.036745, -0.032662, 0.058596, 0.075827,
            0.099969, 0.046727, -0.038072, -0.043777, 0.095262, -0.070519, -0.084170, -0.010394,
            -0.016353, 0.049792, -0.067965, -0.022769, -0.077351, 0.068226, 0.022720, -0.049946,
            -0.026896, 0.063024, 0.079298, -0.091986, -0.058152, 0.017170, 0.025324, -0.063395,
            0.038185, 0.004741, -0.055508, -0.076485, -0.000974, 0.031184, 0.045591, 0.037652,
            0.070323, 0.089196, -0.060235, -0.035483, -0.083977, 0.077803, -0.095178, -0.048390,
            -0.090563, -0.015601, -0.039336, 0.058634, -0.080912, 0.060201, -0.037431, 0.019160,
            -0.064296, 0.027529, 0.078343, 0.034048, 0.030322, -0.026046, -0.094225, 0.028821,
            0.071396, 0.056252, 0.085958, -0.052806, 0.057119, 0.068521, 0.074103, -0.051322,
            0.074330, -0.086777, -0.053584, 0.052296, 0.079059, 0.065995, -0.026197, -0.075087,
            0.061588, -0.049883, 0.069520, -0.000940, 0.040302, 0.018019, -0.013289, -0.042521,
            -0.021261, -0.026423, -0.082166, 0.033964, 0.039491, -0.099869, -0.047876, -0.079455,
            -0.000898, -0.055900, 0.079648, -0.076261, -0.033605, 0.088312, 0.003938, 0.080165,
            -0.047440, 0.086478, -0.067710, -0.039755, -0.088076, 0.094713, 0.038129, -0.032548,
            0.011289, -0.002555, -0.072756, -0.066110, 0.041132, 0.072565, 0.002113, 0.096672,
            -0.006292, 0.060106, 0.038464, 0.099370, 0.042061, -0.063799, 0.065592, -0.047733,
            -0.079844, -0.080550, 0.099277, 0.023550, 0.041612, -0.043094, -0.080210, 0.027957,
            -0.081979, 0.021065, -0.013260, 0.009921, 0.063421, -0.090265, 0.043335, -0.045349,
            -0.018853, -0.051573, -0.033395, 0.038013, 0.049640, -0.080265, 0.016771, -0.000137,
            0.066262, -0.000387, 0.092759, 0.079587, -0.089523, 0.080229, 0.007434, -0.009924,
            -0.060928, -0.062212, 0.029439, -0.039343, 0.036144, -0.013216, -0.023433, 0.060054,
            -0.071639, -0.034670, 0.069138, 0.053677, 0.097437, 0.075192, -0.067093, 0.057197,
            0.008314, 0.070502, -0.024069, 0.053119, -0.023313, 0.044896, -0.049440, 0.049751,
            0.058742, 0.076013, -0.053451, -0.054453, 0.033966, -0.066044, -0.060131, 0.012595,
            0.094322, -0.070863, -0.017384, -0.073599, -0.010639, 0.088443, -0.076698, 0.067346,
            0.054706, 0.013107, 0.065806, 0.049402, 0.031710, 0.014152, -0.065032, -0.094973,
            0.067012, -0.036534, -0.090467, -0.032682, -0.070409, 0.081521, 0.073864, -0.089011,
            0.040616, -0.040807, 0.095579, 0.009430, -0.006138, 0.039352, -0.085770, -0.039664,
            0.082596, 0.085667, -0.017603, 0.075249, -0.082067, -0.078806, -0.020293, -0.093612,
            0.085982, -0.014678, -0.053288, 0.030624, -0.016892, 0.058820, 0.057375, -0.073740,
            0.022170, -0.020087, 0.049171, -0.069576, 0.001619, 0.030203, -0.079441, 0.030783,
            -0.038372, 0.037107, -0.042420, -0.021719, -0.095732, -0.013588, 0.075303, 0.035229,
            -0.075233, -0.083293, 0.075750, -0.043019, -0.079537, -0.040962, 0.052500, 0.012137,
            -0.051120, 0.049574, 0.099767, 0.022497, -0.074993, -0.027335, -0.099555, 0.085337,
            -0.065443, -0.002882, -0.058715, -0.003523, -0.070700, -0.078997, -0.072092, 0.042575,
            -0.100000, 0.083325, 0.043120, 0.053619, 0.093549, 0.007409, -0.025471, -0.092208,
            -0.059419, -0.068338, -0.091612, -0.041448, -0.002058, -0.003063, 0.015568, -0.068745,
            0.059368, -0.092919, 0.034162, -0.039767, 0.085523, -0.052518, 0.093286, -0.072792,
            -0.066143, -0.010261, -0.069595, 0.010159, -0.071476, -0.010541, -0.012809, -0.035405,
            0.016236, -0.056816, 0.058353, -0.028516, -0.057002, -0.029243, 0.036584, -0.057082,
            -0.093336, -0.022618, -0.067799, 0.097770, -0.035102, 0.025263, -0.047122, 0.002460,
            -0.022011, -0.014582, 0.082848, 0.042046, -0.050700, 0.051728, -0.064379, 0.076948,
            -0.088319, -0.097449, 0.098134, 0.054630, 0.003516, -0.077330, 0.061661, -0.040571,
            0.027989, 0.017458, 0.064261, -0.023594, 0.099954, -0.072549, -0.045305, 0.098483,
            -0.088252, -0.030480, 0.031245, 0.039999, -0.031921, -0.078466, 0.084510, -0.076749,
            0.095187, 0.007485, -0.057155, 0.014030, -0.003362, -0.083234, 0.086762, -0.058401,
            -0.017876, -0.053567, 0.022003, -0.067038, 0.085719, -0.026537, 0.093319, 0.000434,
            -0.016098, -0.089722, 0.062778, -0.000642, -0.079516, 0.048305, 0.015197, -0.054688,
            -0.060105, -0.020169, 0.046841, 0.032273, -0.061785, -0.070125, 0.083758, 0.019496,
            0.048787, -0.094241, 0.047204, 0.086802, -0.003128, 0.081109, 0.021187, 0.003903,
            0.034913, -0.035010, -0.010604, -0.019581, 0.025803, 0.009932, -0.046675, 0.051221,
            -0.056098, 0.063078, 0.010045, -0.006400, -0.006520, 0.021893, 0.004192, 0.087257,
            0.080097, 0.099425, 0.060768, 0.002121, -0.038751, -0.037124, 0.060664, -0.007916,
            -0.089690, 0.003141, -0.023580, -0.072364, -0.075844, -0.009132, 0.035788, 0.053957,
            -0.071629, -0.084569, -0.076335, -0.004008, -0.008171, 0.046709, -0.081402, 0.090992,
            -0.029926, -0.094719, -0.063056, 0.024260, -0.020717, 0.072405, -0.000471, 0.060856,
            0.082909, -0.038005, 0.008217, -0.007647, -0.045728, -0.091045, -0.048554, 0.051080,
            -0.019224, -0.088129, 0.026476, 0.036504, 0.018372, 0.081147, -0.076631, -0.039452,
            -0.005793, -0.065984, 0.033706, -0.025107, 0.093186, -0.033077, -0.070679, 0.076751,
            0.098318, 0.073677, -0.077381, -0.049455, 0.047086, 0.091846, 0.068175, -0.085511,
            -0.033667, -0.005779, 0.049080, -0.069240, 0.039524, -0.023406, 0.030891, -0.015958,
            0.096767, -0.010520, -0.038584, -0.016241, -0.049368, -0.082697, 0.064588, -0.060651,
            -0.043538, -0.051553, -0.001021, 0.056181, -0.026967, 0.000583, -0.035498, 0.078280,
            -0.014732, 0.071429, -0.027197, -0.028324, -0.055280, -0.066871, 0.054329, -0.094233,
            0.019919, 0.083274, -0.074266, 0.041107, 0.024440, 0.009492, 0.094968, -0.068740,
            0.058759, -0.049624, 0.059504, 0.007883, 0.007276, -0.085150, 0.011400, -0.073453,
            -0.083094, 0.055984, 0.000891, -0.025695, -0.015904, 0.079367, -0.001797, -0.064011,
            0.096621, -0.055572, 0.072371, 0.083434, 0.018275, -0.090552, -0.029537, -0.083392,
            -0.087659, -0.073898, -0.053094, -0.092286, -0.054913, -0.079003, -0.077620, -0.065513,
            -0.096241, 0.001795, 0.008594, -0.019219, 0.098612, 0.048446, 0.062900, 0.052207,
            -0.052805, 0.025427, 0.031858, -0.073595, 0.087274, -0.059399, -0.012330, -0.097661,
            -0.078112, 0.080196, -0.073325, 0.089997, 0.051743, 0.092463, -0.046851, -0.078387,
            0.083199, -0.040906, -0.011839, 0.093177, -0.045315, -0.034349, 0.057254, 0.090354,
            0.004498, 0.001003, 0.071051, -0.040655, 0.080011, 0.059039, 0.013423, -0.052648,
            0.088181, -0.026484, -0.050265, -0.021282, 0.015593, -0.099059, 0.023293, -0.014734,
            0.030260, -0.003005, -0.052681, -0.023850, -0.028836, 0.084839, -0.088456, 0.028740,
            0.090624, 0.030984, -0.053072, 0.029412, 0.040147, 0.043057, 0.008284, -0.087592,
            0.022391, -0.021006, 0.053212, 0.089572, -0.035788, 0.001700, 0.081983, -0.017125,
            0.015127, -0.038790, 0.025056, -0.041269, -0.087720, -0.087664, 0.043921, -0.015018,
            0.017454, 0.067071, 0.020940, 0.088595, 0.030604, -0.091071, 0.063320, -0.089604,
            0.048095, -0.007779, -0.095382, -0.018874, 0.087304, -0.045619, -0.043272, 0.054078,
            0.020363, 0.040379, -0.063033, -0.034021, -0.065045, 0.022771, 0.020362, -0.050082,
            -0.054555, 0.046502, 0.008123, 0.020455, -0.023306, -0.088595, 0.064230, 0.093624,
            0.084815, -0.073645, -0.047205, -0.072472, 0.090412, 0.080284, -0.045035, 0.017939,
            0.071675, 0.076470, -0.046410, -0.037823, -0.076545, -0.089983, -0.026936, -0.031205,
            0.059886, 0.075641, -0.021827, 0.044475, 0.000627, 0.050250, -0.088274, -0.017607,
            -0.033289, 0.080785, 0.047396, 0.038903, 0.049902, -0.066721, -0.075420, -0.059093,
            -0.016991, -0.051385, 0.042870, 0.086466, 0.099022, 0.067611, 0.041314, 0.001153,
            -0.053397, 0.017269, 0.057499, 0.011425, -0.010988, -0.008379, -0.089083, 0.003593,
            0.015830, 0.085340, 0.036266, 0.094496, -0.002076, 0.076851, -0.059833, 0.028826,
            0.095360, 0.025148, -0.055848, 0.041637, -0.095448, 0.043434, -0.055112, 0.070408,
            0.009100, 0.032283, -0.045413, -0.009242, -0.052115, 0.085120, 0.039435, -0.052850,
            -0.037507, 0.047670, -0.061032, 0.019802, -0.049295, -0.012298, 0.068761, -0.014921,
            0.055511, -0.065976, 0.078337, -0.094321, -0.095250, -0.098546, -0.086486, -0.008780,
            -0.095490, 0.022547, -0.052392, 0.016074, -0.006839, 0.037144, -0.013249, -0.080029,
            0.012391, 0.023355, -0.002568, -0.021215, -0.010148, 0.006549, 0.005096, -0.028177,
            0.068101, -0.083778, 0.069438, -0.062192, 0.071222, -0.069071, -0.062056, 0.013421,
            0.092745, -0.011708, -0.016202, 0.038716, 0.034016, -0.030969, 0.006710, 0.099897,
            -0.001382, -0.089979, -0.089866, 0.055327, -0.069331, -0.091291, 0.006219, -0.024560,
            0.098879, -0.077803, 0.069952, 0.029212, 0.018271, 0.035290, 0.056513, 0.071653,
            -0.076093, 0.094738, 0.034293, 0.015286, -0.045096, 0.096863, -0.052106, 0.018405,
            0.041681, -0.079667, 0.090263, -0.000751, 0.012022, -0.092528, 0.032403, -0.078995,
            -0.022237, 0.011708, -0.058443, 0.089576, -0.081515, 0.021360, 0.037075, -0.070258,
            -0.007145, -0.093848, -0.099435, 0.071594, -0.013858, -0.032099, -0.080536, 0.092976,
            -0.042029, 0.079863, -0.095652, 0.031885, -0.036489, 0.069746, 0.002917, -0.037378,
            -0.041340, -0.037169, 0.086131, -0.000427, -0.094580, -0.001238, -0.004446, 0.084778,
            -0.084376, 0.000493, 0.077896, 0.011560, -0.002608, -0.089358, -0.075982, -0.030352,
            -0.083152, 0.037435, 0.030107, -0.084137, 0.099085, 0.011494, 0.065759, -0.037388,
            0.082761, 0.083991, -0.088733, -0.040386, -0.072239, -0.000655, -0.067728, -0.042547,
            0.036798, 0.045848, 0.076643, 0.090947, 0.032127, -0.021056, 0.021429, 0.094728,
            0.095099, -0.099368, 0.090267, -0.030795, -0.034238, -0.089184, 0.052154, 0.029758,
            0.069193, -0.051429, 0.066650, -0.093606, 0.073352, 0.070608, -0.089235, -0.079131,
            0.019350, 0.047522, 0.097821, 0.050832, -0.068768, 0.032590, -0.092091, -0.050653,
            0.015533, 0.028803, -0.088186, -0.023771, -0.081069, -0.088564, -0.000606, -0.005647,
            0.021624, 0.064467, -0.096870, -0.056714, -0.031946, -0.075653, 0.090983, 0.094259,
            -0.097404, -0.086768, -0.053866, -0.029875, 0.007701, -0.097765, -0.013409, -0.080583,
            0.093276, -0.059609, 0.017004, 0.099185, -0.079336, 0.004370, 0.089524, 0.001129,
            -0.073742, -0.045759, -0.086137, -0.051418, -0.065079, 0.046123, 0.056663, -0.023689,
            0.078105, -0.061094, 0.035968, -0.088579, 0.058301, -0.099171, 0.058481, -0.006561,
            -0.088548, 0.090559, 0.014266, -0.074466, 0.026459, -0.064480, -0.072677, 0.001895,
            -0.015090, -0.068864, 0.072189, 0.024173, -0.049534, 0.051267, 0.059884, 0.081128,
            -0.012154, -0.030234, 0.011054, -0.008723, 0.051996, -0.046944, 0.050250, 0.092831,
            0.040294, 0.068885, -0.065747, 0.036424, -0.083428, -0.003282, -0.015027, 0.098768,
            -0.030246, 0.082075, 0.060181, -0.000889, 0.068379, 0.098423, -0.055041, -0.091101,
            0.023042, 0.051828, 0.013699, 0.091857, -0.015736, 0.018749, -0.077129, 0.071401,
            -0.058759, -0.013308, -0.066263, 0.068610, -0.059111, -0.084367, -0.050102, 0.000186,
            -0.066451, 0.005451, -0.015860, 0.077926, 0.046041, 0.038267, -0.052105, 0.056330,
            0.062978, -0.005445, 0.000565, -0.014923, -0.001857, -0.073301, 0.049594, 0.092042,
            0.050175, 0.047207, 0.090647, 0.069746, 0.023353, 0.066427, 0.069622, 0.020456,
            -0.024142, -0.059187, -0.027466, 0.029908, -0.078014, -0.090942, -0.015358, 0.016800,
            0.064111, 0.080902, -0.081022, 0.001459, -0.072662, 0.009449, -0.081652, -0.090179,
            0.004682, 0.066544, -0.095751, -0.089967, 0.011465, 0.096633, -0.028579, 0.092093,
            0.093541, -0.043624, 0.027505, 0.061796, -0.087658, 0.040079, 0.041418, -0.000268,
            -0.098772, -0.015586, -0.076020, 0.000002, 0.002647, -0.054621, 0.087307, -0.065387,
            0.064946, 0.063810, 0.092153, 0.029312, 0.066444, -0.034756, 0.035852, 0.090692,
            -0.097423, -0.070745, 0.029632, -0.022818, -0.016966, -0.057117, 0.000088, -0.031370,
            0.095262, 0.061237, 0.026176, 0.041026, -0.042098, -0.026899, 0.059210, 0.085923,
            -0.020551, -0.087673, -0.002522, 0.059490, 0.004691, -0.096415, 0.012584, -0.054114,
            0.041078, 0.082474, 0.060819, -0.081350, 0.022902, -0.005591, 0.019872, -0.049430,
            -0.081194, -0.046663, 0.004151, 0.055184, -0.088243, 0.050055, 0.095631, -0.083190,
            -0.069816, 0.019244, -0.059540, 0.017549, -0.090337, -0.007573, 0.090830, 0.075172,
            0.017034, 0.044281, 0.090112, -0.080929, 0.074080, 0.034101, -0.083925, 0.051385,
            0.013815, -0.088298, -0.051804, 0.009607, 0.011439, 0.026478, 0.077706, 0.083209,
            0.038004, 0.008835, 0.000376, 0.091502, 0.005671, -0.066091, -0.048460, -0.097519,
            -0.069323, 0.097230, -0.077842, -0.022388, -0.094271, 0.056490, -0.041164, 0.099853,
            0.067789, -0.042027, -0.041357, -0.016635, -0.026132, -0.086622, 0.009827, -0.067922,
            -0.063807, 0.030791, 0.053693, -0.094903, 0.021644, 0.083578, 0.038117, -0.074186,
            -0.095640, -0.010383, 0.070192, -0.007407, -0.094229, 0.097840, -0.074343, 0.031223,
            -0.059577, 0.020326, -0.020827, -0.011635, -0.016183, 0.079817, -0.046889, -0.030008,
            -0.071637, -0.058049, -0.017022, -0.078731, -0.038031, -0.012487, -0.070633, 0.020801,
            0.012263, 0.018918, 0.093304, -0.066393, -0.019178, -0.040085, -0.076958, -0.025486,
            0.025620, 0.022246, -0.009156, 0.071502, 0.069199, 0.077403, 0.071856, 0.003139,
            -0.004815, 0.076670, 0.013439, 0.085552, 0.004092, -0.076128, -0.068376, 0.046157,
            0.063229, 0.033121, 0.078253, 0.093161, -0.043049, 0.029665, 0.016813, -0.012181,
            -0.083808, -0.077218, 0.043926, -0.043298, -0.038805, -0.054867, -0.042204, 0.044148,
            -0.071087, 0.044625, -0.008857, 0.051187, -0.053301, 0.035508, 0.022073, 0.074597,
            0.062470, 0.008041, 0.084674, -0.074897, -0.033908, 0.042064, 0.054877, -0.071503,
            -0.029157, 0.079221, -0.086032, 0.043995, 0.099107, -0.037104, -0.011200, 0.025354,
            0.045908, -0.014346, -0.088763, -0.004352, 0.011825, 0.066085, 0.018011, 0.075240,
            -0.038349, -0.015367, 0.029194, 0.079491, 0.041683, 0.092196, 0.032778, 0.051222,
            0.021687, 0.082577, 0.078220, 0.060745, -0.084738, -0.011100, -0.033877, -0.015034,
            0.071204, -0.066022, -0.052128, -0.040556, 0.007046, -0.055563, 0.019984, -0.084749,
            -0.051181, 0.059723, 0.068954, -0.031252, 0.005345, -0.003401, -0.070795, -0.022424,
            0.059036, 0.003360, 0.095917, -0.081691, -0.070883, -0.085254, -0.063297, 0.020981,
            0.066895, -0.081867, -0.076301, -0.037039, 0.059318, 0.052819, -0.010159, 0.091217,
            0.036391, 0.093994, 0.000269, -0.061927, 0.001116, 0.031674, 0.039097, -0.000376,
            0.003416, 0.018974, -0.022048, 0.024483, -0.063649, -0.022005, 0.053916, 0.030567,
            0.055958, -0.066600, -0.029433, -0.051771, -0.077791, -0.041056, -0.006397, -0.007326,
            -0.006179, -0.063167, -0.006784, 0.026516, 0.071821, 0.060498, -0.083571, -0.063708,
            0.078880, 0.067882, 0.055510, -0.051576, 0.084331, -0.097153, -0.079528, 0.008950,
            0.046508, -0.094256, 0.014720, -0.001505, -0.013163, -0.078326, -0.047939, 0.004694,
            0.023916, 0.062719, -0.035427, 0.094654, 0.037653, -0.067941, -0.099557, 0.067234,
            0.007558, -0.010036, -0.033105, 0.047014, -0.035984, -0.073162, 0.069447, -0.042875,
            0.057579, -0.068768, -0.020103, -0.054798, 0.004256, -0.011827, 0.029834, -0.077565,
            -0.088199, 0.079530, -0.076568, 0.033720, -0.086848, 0.019487, -0.002128, -0.049175,
            -0.002059, 0.012278, 0.002798, 0.075653, -0.057280, 0.045282, 0.092101, -0.071827,
            -0.078325, 0.033798, -0.035500, -0.002255, -0.093282, 0.061083, -0.006602, 0.006148,
            -0.034761, 0.034714, -0.001640, -0.039854, 0.024841, -0.037130, 0.037112, -0.095778,
            -0.038693, -0.068517, 0.042545, -0.070221, 0.077481, 0.024349, 0.061288, -0.065232,
            -0.081233, -0.016410, -0.054946, -0.033872, 0.082200, 0.053448, -0.029546, 0.060724,
            -0.066982, -0.058109, 0.044294, 0.043173, -0.028914, -0.068241, 0.015630, 0.096468,
            0.098421, 0.082040, -0.074990, 0.066289, 0.018313, -0.081166, 0.006423, -0.096572,
            0.021468, -0.054987, -0.067284, 0.004866, 0.067447, 0.056129, -0.061884, 0.056880,
            0.037208, -0.066757, -0.025793, -0.023964, -0.081278, -0.091248, -0.067906, 0.090908,
            -0.026570, 0.009599, -0.083039, -0.017201, 0.037781, -0.002026, -0.007858, -0.001332,
            0.067238, -0.070446, -0.089802, -0.096320, 0.040662, -0.055704, -0.043737, -0.013227,
            0.058625, 0.000869, 0.019574, -0.028089, 0.054424, -0.049888, 0.077083, 0.082314,
            -0.089955, -0.031664, -0.012050, -0.053848, 0.005369, -0.032822, 0.082788, -0.007911,
            -0.055899, -0.049656, -0.093459, -0.061460, 0.019415, 0.030207, -0.043100, 0.001017,
            0.050175, 0.013218, 0.011266, -0.065601, 0.094017, -0.029522, 0.008250, -0.012315,
            0.043445, -0.082830, 0.022758, -0.077620, 0.078931, -0.039417, 0.020414, 0.006618,
            -0.039747, -0.013346, 0.095120, -0.050563, -0.025994, 0.065916, 0.058708, 0.027667,
            -0.087504, -0.016131, -0.098532, -0.049421, -0.010691, -0.037797, 0.041053, 0.001003,
            0.019680, 0.029315, 0.024142, -0.098039, 0.020190, 0.082282, 0.033754, 0.071786,
            -0.054145, -0.025384, -0.091237, -0.091328, -0.075388, 0.051103, -0.058817, 0.033567,
            0.069256, 0.035716, -0.041914, 0.063884, 0.036675, 0.022679, -0.025087, 0.079442,
            0.048328, -0.017577, 0.046119, -0.066637, -0.008659, -0.083022, -0.089238, 0.081454,
            0.053157, -0.033940, -0.055719, -0.038373, -0.023765, -0.017012, 0.085629, 0.095721,
            0.052217, -0.055945, -0.035256, 0.012470, -0.059560, 0.089461, -0.059272, 0.092560,
            -0.000438, 0.090603, 0.065357, 0.008990, 0.098156, 0.008737, 0.038384, -0.025790,
            -0.088060, -0.061301, -0.080490, 0.068502, 0.044471, 0.028173, 0.092718, -0.070595,
            0.028131, 0.004314, -0.087528, -0.096598, 0.097121, 0.084914, 0.018170, -0.064197,
            0.013932, 0.071306, -0.021757, 0.001389, -0.049186, 0.075594, -0.060951, -0.056193,
            -0.025690, -0.003169, 0.025195, 0.077839, 0.061238, 0.014427, 0.026224, 0.084768,
            -0.029365, -0.047606, 0.076585, 0.086753, -0.004096, -0.003923, 0.075888, -0.018650,
            0.098715, -0.073273, 0.024581, 0.075751, -0.096280, -0.026448, 0.085215, 0.072383,
            -0.066729, 0.080002, 0.085695, 0.039307, -0.025342, -0.057465, -0.020646, 0.031488,
            0.021172, 0.013760, -0.070878, 0.081070, 0.011804, 0.020663, -0.083165, -0.028175,
            -0.024150, 0.069095, 0.079489, -0.062291, -0.068112, -0.059280, -0.028035, -0.076751,
            0.092406, -0.066579, 0.082416, 0.040629, -0.036690, -0.049285, 0.010052, -0.098029,
            -0.042601, 0.013137, -0.096457, 0.093567, -0.059398, -0.002656, -0.050195, -0.041864,
            -0.040009, 0.019877, -0.061507, 0.041239, 0.038369, 0.055320, -0.092997, -0.059106,
            0.060125, 0.079898, 0.037436, -0.086510, 0.024518, 0.067565, -0.004596, 0.060945,
            -0.070597, 0.059458, 0.019867, -0.035404, 0.080370, -0.090745, 0.085772, 0.022704,
            0.027384, -0.011018, 0.012832, 0.070420, 0.012002, 0.011700, 0.049392, -0.045646,
            -0.022282, 0.039007, 0.088890, -0.018388, 0.002615, 0.042093, -0.064293, -0.080029,
            0.005777, 0.017826, -0.077253, 0.097314, 0.034775, -0.060788, 0.077112, 0.028107,
            0.007704, 0.009177, -0.066786, -0.008049, 0.077692, -0.049822, 0.046347, 0.063492,
            0.098677, -0.080338, 0.027376, 0.032876, -0.033826, 0.058937, 0.038702, 0.085024,
            -0.071485, 0.027086, -0.072381, 0.026532, 0.063887, -0.085297, 0.077478, 0.057290,
            -0.009098, 0.092514, 0.000606, -0.073188, -0.070775, 0.064806, -0.082153, -0.033577,
            0.080326, 0.051414, 0.091405, 0.034645, 0.002330, 0.046421, -0.045214, 0.059743,
            -0.048859, -0.031554, -0.027846, -0.054616, -0.001953, -0.021535, -0.018867, 0.064187,
            0.076336, -0.052542, 0.095108, 0.070025, -0.027528, 0.071393, -0.032807, 0.092857,
            -0.005272, 0.075474, 0.081602, 0.068750, 0.039227, 0.014098, -0.080078, 0.001138,
            0.062398, 0.095269, 0.035732, -0.061514, -0.001915, -0.041601, -0.000760, -0.064069,
            -0.009491, -0.000714, 0.058546, -0.043405, -0.074832, -0.004924, 0.096155, 0.046358,
            0.054070, -0.097417, -0.000990, -0.021076, 0.018368, 0.075202, 0.094422, 0.041181,
            0.065638, -0.024099, -0.009729, -0.092782, -0.069778, -0.081402, 0.058911, 0.048162,
            -0.080788, 0.079750, -0.094419, 0.010011, -0.090821, 0.081886, -0.046369, -0.064236,
            0.046872, -0.021444, 0.064467, 0.093341, 0.071089, 0.012453, -0.008365, -0.071643,
            -0.058557, -0.059498, 0.060634, 0.004157, -0.066364, 0.089963, -0.031567, 0.098175,
            0.095119, 0.017277, -0.081973, 0.032957, 0.002165, -0.049181, -0.059842, -0.069667,
            0.015439, 0.041718, 0.016865, 0.057865, -0.036510, -0.036065, 0.097900, -0.075822,
            -0.092141, 0.097313, 0.061973, -0.016727, 0.030576, -0.031469, -0.013854, 0.013863,
            -0.034354, 0.007025, -0.037567, -0.082764, 0.046622, -0.073369, -0.097702, 0.019787,
            0.073162, 0.034714, -0.070758, 0.060227, -0.028954, -0.012355, 0.059535, 0.039298,
            -0.035253, 0.060758, -0.026647, -0.026895, 0.068968, -0.022619, -0.001999, 0.093553,
            0.096188, 0.023550, 0.075211, -0.014106, -0.046017, 0.093534, 0.052478, 0.068369,
            -0.048416, 0.052345, 0.081856, 0.009566, 0.066653, -0.090736, 0.039548, -0.024697,
            0.090866, 0.036096, 0.099020, -0.034762, 0.009197, 0.071183, -0.072320, -0.034879,
            -0.080019, -0.075077, 0.044293, -0.030202, 0.065394, 0.009232, 0.035217, 0.010647,
            -0.093716, -0.015078, -0.054917, -0.039614, 0.080463, -0.077380, -0.073016, -0.032069,
            -0.044380, 0.024683, 0.064166, 0.015693, 0.001761, 0.048032, 0.059717, 0.003516,
            -0.067527, -0.086651, 0.099792, 0.032524, 0.091551, -0.038381, 0.027495, -0.052660,
            -0.052646, -0.045759, 0.099943, 0.072907, -0.061540, -0.070544, 0.032985, -0.023580,
            0.004782, 0.012768, -0.087561, 0.022087, -0.000874, -0.069988, -0.098893, 0.051060,
            -0.091442, -0.045049, 0.060782, -0.066634, -0.047359, -0.063010, 0.058865, 0.030938,
            0.010321, 0.035210, 0.028610, -0.021128, -0.009216, -0.062594, -0.046971, 0.001543,
            -0.012136, 0.025174, 0.047464, 0.076710, 0.095821, -0.007600, 0.039430, -0.080991,
            0.057873, -0.026738, -0.079349, -0.077862, -0.083370, -0.046188, 0.045662, -0.081564,
            -0.011803, -0.015541, 0.036784, -0.040766, 0.050661, -0.061842, 0.058758, -0.099644,
            -0.068105, 0.035075, 0.015345, 0.070046, 0.052085, -0.083326, 0.078650, -0.025951,
            -0.018886, -0.042571, -0.017340, -0.093362, 0.007732, 0.075350, -0.019987, 0.076893,
            -0.045193, 0.040393, 0.009887, -0.038811, 0.097487, 0.008220, 0.004453, -0.010803,
            -0.005418, 0.080615, -0.015219, 0.060052, 0.036268, 0.065220, -0.075790, -0.087627,
            -0.032835, -0.027915, 0.095283, 0.098536, -0.023844, -0.047915, -0.000422, 0.038273,
            0.034178, 0.018876, 0.073443, 0.010630, 0.080915, 0.024359, -0.008890, 0.083255,
            0.003251, 0.003990, -0.019326, 0.058736, -0.029652, 0.078014, 0.053389, -0.001604,
            -0.016394, 0.039184, -0.097650, -0.033075, -0.035348, -0.012250, -0.047854, -0.043193,
            0.047555, -0.010600, -0.069309, -0.007224, 0.032289, -0.040514, 0.085081, -0.069417,
            0.087070, 0.038492, -0.057478, -0.094670, 0.039008, -0.018646, 0.029629, 0.042699,
            -0.067796, 0.011754, -0.021313, -0.090042, 0.017643, 0.021372, -0.076498, 0.094059,
            0.052439, 0.056851, -0.087135, -0.086405, 0.043197, 0.057750, -0.085761, 0.096029,
            -0.022037, -0.081071, -0.073365, -0.023201, 0.052627, -0.068969, 0.073679, 0.099932,
            0.094005, 0.045518, -0.040850, -0.038015, 0.071802, -0.020884, 0.015302, 0.099954,
            -0.099405, -0.022835, -0.039534, -0.016187, 0.004119, -0.021661, 0.097074, 0.040554,
            -0.062829, 0.096898, -0.027756, 0.014400, 0.009335, -0.073502, 0.050877, -0.060753,
            -0.089091, 0.083940, 0.059878, -0.041858, -0.071577, 0.090305, 0.035360, -0.066378,
            -0.083092, -0.041373, -0.055574, 0.007089, 0.009573, -0.097853, 0.050174, 0.050944,
            -0.090618, 0.080402, -0.039820, 0.051471, 0.098081, 0.059233, 0.067145, 0.003519,
            0.014648, -0.085065, -0.072550, 0.080548, -0.091391, 0.043067, -0.058920, 0.086692,
            0.073780, -0.022598, -0.012788, -0.069370, 0.093270, -0.060268, 0.035695, 0.070075,
            -0.096288, -0.098355, -0.070183, -0.015012, 0.062167, -0.030434, -0.041299, 0.066544,
            0.080201, 0.052698, 0.086414, -0.090168, -0.089180, 0.016615, -0.083902, -0.027915,
            0.060871, 0.097168, -0.020098, -0.088194, -0.049344, 0.097184, -0.018519, -0.093002,
            -0.052947, -0.062373, -0.026102, -0.072401, -0.033847, 0.091678, -0.047358, -0.021486,
            0.056145, -0.093447, -0.052519, -0.008497, 0.078636, 0.013909, -0.000494, -0.033182,
            0.046311, -0.042983, -0.015647, -0.043098, 0.037377, 0.099743, -0.075450, -0.007046,
            -0.007087, -0.050070, -0.071290, -0.042656, -0.078152, 0.075934, -0.070885, 0.061003,
            0.082697, -0.018376, 0.093044, 0.072260, -0.069732, 0.057088, 0.025710, 0.078685,
            -0.075847, 0.077835, 0.014190, 0.048884, -0.055516, 0.065850, -0.007191, 0.082463,
            0.008536, 0.062076, -0.063192, -0.088625, 0.034731, 0.052974, 0.027677, 0.031417,
            0.065868, -0.051452, 0.053524, 0.047938, -0.042446, -0.059487, -0.099308, 0.073539,
            -0.001542, 0.022866, -0.070775, -0.075186, 0.082063, 0.070456, 0.054081, 0.079967,
            0.080529, 0.059069, 0.072232, -0.012846, 0.026045, -0.085949, -0.035843, -0.009343,
            -0.004875, 0.098627, 0.081593, 0.031490, 0.093246, -0.050982, 0.019443, -0.079324,
            -0.085065, 0.053465, 0.032108, -0.005382, -0.064704, 0.072089, -0.007672, 0.068270,
            -0.069091, 0.076850, -0.004723, -0.031250, -0.008255, -0.077127, -0.018284, 0.005859,
            -0.074038, 0.054002, 0.094466, 0.051557, -0.049123, -0.062144, -0.079251, -0.061648,
            0.059691, 0.046089, -0.037401, 0.001638, -0.015752, 0.001444, -0.099444, 0.084509,
            -0.035096, -0.086142, -0.001809, 0.029891, 0.045798, -0.050510, -0.007303, -0.074906,
            -0.075386, -0.066020, -0.062809, 0.062797, -0.090202, -0.020760, -0.041460, 0.065767,
            -0.006462, -0.055032, 0.081203, -0.075826, -0.037865, -0.091245, -0.007985, 0.028465,
            0.087382, -0.072503, -0.066742, 0.099934, 0.017596, -0.067767, 0.043119, 0.096323,
            0.003680, -0.045856, -0.053462, -0.035971, 0.050977, -0.070770, 0.090959, -0.061879,
            0.028041, 0.099480, -0.044210, -0.066853, -0.051839, -0.041935, -0.061408, -0.098536,
            -0.051393, -0.034765, 0.086638, -0.017066, 0.011652, -0.031381, 0.015588, 0.047639,
            0.017473, -0.012308, 0.008421, 0.094362, -0.066606, 0.098104, -0.022211, 0.071304,
            0.077448, 0.003479, 0.002252, 0.069318, -0.013893, 0.027519, 0.095679, -0.019797,
            -0.035222, 0.077549, 0.089843, 0.095894, -0.094352, 0.065850, 0.049222, -0.063810,
            0.065312, -0.053706, 0.052893, 0.008549, 0.031525, -0.061479, -0.019335, -0.054363,
            0.054405, -0.079438, 0.007002, 0.038633, -0.002327, -0.030087, 0.022811, 0.007899,
            -0.055129, -0.027154, -0.021872, -0.056359, 0.002475, -0.031065, -0.056533, -0.081019,
            -0.065207, -0.021771, 0.079983, -0.011766, -0.053693, -0.076263, 0.017051, 0.085031,
            -0.090116, 0.077112, 0.089324, 0.022624, 0.049889, 0.029695, -0.087425, -0.039888,
            -0.023411, -0.059633, -0.003144, 0.021098, 0.003420, 0.054552, -0.058931, 0.084117,
            -0.040669, 0.080183, 0.090707, -0.081696, -0.095831, 0.074209, 0.076127, -0.010706,
            0.020737, 0.014742, -0.030533, 0.052618, 0.033741, -0.096469, -0.027586, -0.059392,
            0.042927, -0.099846, -0.085940, -0.088148, -0.034348, -0.077119, -0.019933, -0.056906,
            0.064289, 0.087739, 0.038685, -0.038801, 0.040800, -0.085866, -0.050017, -0.020963,
            -0.093497, 0.093773, 0.065528, 0.046326, 0.046999, 0.098408, -0.038023, 0.023988,
            -0.033627, -0.016976, -0.081980, 0.059793, -0.072032, -0.044692, 0.089146, -0.058688,
            0.012854, -0.091778, -0.058072, 0.090727, 0.053781, 0.004237, -0.049857, 0.076265,
            0.094127, 0.020040, -0.064465, 0.003445, 0.048717, 0.029203, 0.089986, 0.090055,
            0.050652, -0.069550, -0.005406, 0.019365, -0.033355, 0.029912, 0.065271, -0.054028,
            -0.030484, -0.076066, 0.015533, 0.059859, 0.090447, 0.005243, 0.065119, -0.096576,
            -0.002173, -0.018866, -0.046051, 0.092574, 0.062444, -0.083335, -0.090811, -0.046847,
            0.033945, 0.047443, 0.051766, 0.077304, -0.064004, 0.008430, -0.044564, -0.048511,
            0.000967, -0.008801, -0.081021, -0.001454, 0.048086, 0.089684, -0.027025, 0.067624,
            -0.022367, -0.073560, 0.086495, -0.023636, -0.023235, -0.016919, 0.090132, -0.042707,
            -0.026555, 0.005825, -0.065473, 0.038997, 0.084192, 0.036156, -0.024048, -0.019134,
            -0.098008, -0.078831, -0.027414, 0.055435, 0.093015, -0.007008, -0.009595, 0.068420,
            -0.020972, -0.068220, 0.056627, -0.043237, -0.071216, 0.000861, -0.066532, -0.077991,
            -0.055122, -0.008676, 0.033284, -0.041434, -0.095636, -0.034037, 0.093014, -0.023093,
            -0.073006, -0.037430, 0.090060, 0.015949, -0.014019, 0.097831, 0.010312, 0.015213,
            0.098556, -0.010266, -0.017615, 0.080014, 0.031701, -0.070152, 0.064009, -0.041917,
            -0.085234, 0.088384, -0.073870, -0.042577, -0.015941, -0.000590, -0.011440, 0.074404,
            0.074821, -0.088974, -0.092779, -0.097862, 0.083221, -0.012129, -0.049499, -0.065973,
            0.013070, 0.028611, 0.057378, 0.080714, 0.088911, -0.066667, -0.005286, -0.063581,
            -0.036081, 0.002877, 0.056076, -0.044891, -0.056378, -0.026493, 0.041339, 0.002780,
            0.074935, -0.090012, 0.036346, 0.075736, 0.062991, -0.087019, -0.066746, 0.001479,
            0.000474, 0.059572, 0.081560, -0.006588, -0.029383, 0.054520, -0.064831, -0.045621,
            0.012305, -0.016829, 0.060447, -0.019828, 0.010623, 0.028405, -0.008550, -0.032154,
            0.062265, 0.055728, -0.015267, -0.064026, -0.032788, 0.069177, 0.058524, -0.032611,
            0.054669, -0.011782, -0.005858, -0.047034, 0.002924, -0.020898, 0.025257, -0.095109,
            -0.005975, -0.078194, -0.092918, -0.099147, -0.016073, 0.018685, 0.010443, 0.023541,
            -0.042951, -0.066231, -0.021636, -0.075059, -0.012471, 0.020991, 0.013952, -0.090833,
            0.094964, 0.086142, -0.044794, -0.085176, 0.014601, 0.022562, -0.026737, -0.045939,
            0.007545, -0.060860, -0.052401, 0.031990, 0.011308, 0.040350, 0.084570, -0.000973,
            0.034033, 0.071390, -0.062600, 0.066615, 0.068542, -0.038190, 0.078161, -0.045847,
            -0.047721, 0.071699, 0.046389, -0.007205, -0.078993, 0.057041, 0.024553, -0.059698,
            -0.003718, 0.072246, 0.042226, -0.045980, -0.001070, -0.035345, 0.066006, 0.046932,
            -0.014808, -0.011721, -0.072837, -0.022131, 0.049024, -0.095908, 0.012763, 0.095146,
            0.043043, 0.049674, -0.027065, -0.008434, -0.085389, -0.048395, 0.025126, 0.034136,
            0.058962, -0.056338, 0.028718, -0.055223, -0.073737, -0.038348, -0.068010, 0.018963,
            0.015747, 0.064178, 0.052221, 0.019495, 0.038586, 0.030639, -0.094562, -0.046209,
            -0.081642, -0.057982, 0.020038, 0.094300, 0.069238, -0.040715, 0.023239, 0.068822,
            0.020836, -0.081781, 0.012261, 0.040726, 0.016372, 0.091340, 0.083452, -0.055877,
            0.076029, -0.089497, -0.056771, 0.040546, 0.003034, 0.086359, 0.059498, -0.021303,
            0.095629, -0.071766, 0.018636, 0.031915, 0.078171, 0.097840, 0.048764, -0.044565,
            0.080408, 0.068845, -0.001971, 0.071789, 0.015931, -0.008696, 0.030895, 0.000279,
            -0.028181, 0.014517, -0.000666, -0.068704, -0.073277, 0.032859, -0.038941, 0.002446,
            0.088310, 0.073774, 0.055557, 0.053250, 0.067713, -0.067062, 0.093054, -0.073737,
            -0.064801, -0.040964, 0.097736, 0.055978, 0.028028, 0.057977, -0.095507, -0.095378,
            0.031156, -0.027343, 0.047855, -0.070972, 0.094680, -0.087237, 0.050966, 0.052683,
            -0.053627, 0.088367, 0.052107, -0.045656, -0.045240, 0.076529, -0.024631, 0.006575,
            0.045837, 0.046658, -0.050074, 0.075277, 0.030326, 0.032197, -0.046466, 0.045080,
            -0.012416, 0.070914, 0.051276, 0.072339, -0.013578, 0.064993, 0.076930, 0.022189,
            -0.006307, -0.065347, -0.051613, 0.057600, -0.061298, -0.049177, 0.051003, -0.050822,
            -0.077350, 0.022224, 0.043159, -0.090648, 0.012785, -0.077117, 0.042191, 0.054705,
            0.033649, 0.097655, -0.019408, -0.039779, 0.027056, 0.030014, -0.044689, -0.065825,
            -0.030447, 0.041480, 0.042257, -0.044284, 0.084579, -0.074324, -0.011606, 0.024421,
            -0.072031, 0.091014, -0.049150, 0.059217, 0.004027, -0.056143, 0.093478, -0.011129,
            0.017800, 0.094057, -0.094171, 0.080461, -0.049349, -0.081164, 0.055620, -0.030597,
            -0.079812, -0.049027, 0.064641, 0.043820, -0.004910, 0.051355, 0.019864, -0.074874,
            0.019208, 0.071206, -0.009371, -0.011427, 0.098198, -0.028590, 0.093214, 0.087557,
            0.080847, 0.026566, 0.027449, 0.025728, -0.091662, -0.039568, -0.055996, -0.014723,
            -0.050017, -0.032205, -0.068087, -0.072716, 0.032323, 0.097613, -0.040557, -0.066083,
            -0.033623, -0.037764, -0.097387, 0.087830, 0.067923, 0.074154, -0.051679, -0.066122,
            0.095854, -0.060497, -0.084020, -0.083298, 0.030437, -0.013345, -0.061012, 0.085152,
            -0.055438, 0.006513, 0.049836, 0.047853, 0.083703, -0.012389, 0.051548, -0.007936,
            0.013211, -0.058729, 0.061893, 0.022757, 0.020045, -0.007663, 0.040470, -0.096589,
            0.081719, 0.078450, 0.053578, -0.003647, -0.093311, -0.027090, -0.080339, -0.047911,
            0.078646, -0.011538, 0.099626, -0.050409, -0.046574, -0.080147, 0.038087, 0.037013,
            0.050497, 0.086490, 0.060124, -0.091709, 0.002870, 0.087233, 0.069788, 0.038702,
            -0.046955, -0.084375, 0.096388, -0.032750, 0.034183, -0.027029, -0.033768, 0.035608,
            0.092512, -0.023752, 0.043318, 0.040261, -0.015974, 0.019898, 0.033226, 0.089278,
            -0.037749, 0.076917, -0.006457, 0.039444, -0.081077, 0.067209, 0.084280, 0.023293,
            0.003069, 0.026097, -0.078981, -0.086200, 0.074483, -0.089798, -0.022399, -0.061246,
            -0.093347, -0.042438, -0.000807, -0.080246, -0.076658, -0.080653, -0.003613, -0.026633,
            0.073489, -0.095728, -0.087347, -0.030482, 0.099996, 0.091083, -0.012930, -0.027257,
            0.013825, 0.004202, 0.009400, -0.043220, -0.004864, 0.019239, 0.059757, -0.063084,
            -0.020343, 0.050211, 0.065824, 0.054169, -0.024254, 0.022803, -0.034229, 0.084931,
            0.056806, -0.090240, 0.048574, 0.098167, 0.026741, -0.079320, 0.064325, 0.005354,
            -0.026831, 0.043643, 0.077641, 0.045380, -0.078847, 0.041735, -0.061708, -0.093403,
            -0.049598, 0.052604, 0.076880, 0.029522, 0.061465, -0.095416, -0.048694, -0.057057,
            -0.084960, -0.062445, -0.071402, -0.083565, 0.014591, -0.046611, 0.004129, -0.016011,
            0.006871, -0.090831, -0.082019, -0.090416, 0.069593, 0.003312, -0.085114, -0.038488,
            -0.001031, -0.066181, 0.080905, 0.071761, 0.024035, -0.017220, 0.058163, 0.026184,
            0.008512, 0.012332, -0.087039, 0.028315, 0.036511, 0.061696, 0.047423, 0.085113,
            -0.053159, 0.008204, -0.073643, 0.085851, -0.004668, -0.086417, 0.064823, 0.011793,
            -0.028824, 0.089464, -0.052668, -0.049614, -0.034330, 0.011455, -0.077709, 0.021624,
            -0.096762, -0.053900, -0.058495, -0.072660, 0.022911, 0.043932, -0.024451, -0.030042,
            0.002118, -0.081054, -0.094916, -0.064771, -0.072496, 0.014103, 0.080711, 0.071354,
            -0.061562, 0.055638, 0.087706, -0.021156, -0.032608, -0.094472, -0.007206, -0.038742,
            0.025234, -0.091813, -0.009576, 0.017588, -0.024876, -0.015370, 0.094919, 0.043502,
            -0.074418, -0.045276, 0.046133, 0.043565, -0.084228, -0.068352, -0.041355, -0.012176,
            0.005978, 0.024651, -0.046626, -0.055838, 0.056004, -0.078017, -0.017638, -0.089690,
            -0.076077, 0.037367, -0.004897, 0.083912, 0.099078, -0.082898, -0.053409, 0.022737,
            -0.012586, -0.091592, -0.059540, -0.053634, 0.010397, 0.042487, 0.061348, 0.084565,
            -0.052818, 0.075689, -0.057339, -0.085907, -0.043075, 0.045861, 0.016576, -0.086959,
            0.026911, -0.003873, 0.072646, -0.056508, -0.082941, -0.006309, -0.040088, -0.017005,
            -0.026776, 0.097534, -0.013106, -0.096750, -0.058832, -0.003161, 0.056414, 0.024189,
            -0.098664, 0.014595, 0.071394, -0.032072, 0.046514, 0.083117, 0.084049, 0.016576,
            -0.095941, -0.085495, 0.074042, -0.032200, 0.085697, 0.064223, -0.089687, 0.045164,
            -0.083094, -0.072495, 0.070394, 0.090044, 0.048404, 0.046444, 0.011677, 0.064668,
            -0.093768, 0.049059, 0.011815, -0.030041, -0.058310, 0.037481, -0.021796, 0.010859,
            -0.020696, -0.002938, -0.035760, -0.095060, 0.023920, -0.014465, -0.029849, 0.065224,
            -0.046962, -0.013559, -0.091302, -0.037321, 0.071145, 0.005350, 0.055260, 0.083685,
            0.025767, 0.073452, 0.023926, 0.059498, -0.086322, 0.074018, 0.033397, 0.015436,
            -0.098083, 0.056455, 0.053923, -0.075259, 0.085957, 0.092708, -0.009419, -0.031788,
            -0.006027, -0.084330, -0.089150, -0.013865, -0.034421, 0.088690, 0.058756, -0.027806,
            -0.098549, 0.030868, -0.008445, -0.053012, 0.088820, 0.062573, -0.038310, 0.085115,
            -0.087308, -0.082917, 0.046626, 0.060781, -0.020880, -0.019067, -0.097033, 0.033389,
            -0.002897, -0.075074, -0.076865, 0.056491, -0.083748, 0.021736, 0.091218, -0.029366,
            0.056502, 0.044232, -0.022533, -0.061633, -0.059802, -0.097729, 0.014479, 0.023070,
            -0.025930, 0.053577, -0.006947, 0.030605, 0.097214, 0.093594, 0.079939, -0.041399,
            0.002138, -0.030157, 0.028822, 0.058779, 0.016605, -0.032159, 0.016892, -0.073977,
            -0.087464, -0.017250, -0.009211, -0.020947, 0.045131, 0.031497, 0.023998, 0.024822,
            -0.005664, 0.018634, -0.068436, -0.063793, 0.011741, 0.062673, 0.004237, 0.090914,
            0.080775, -0.099820, -0.046065, 0.034776, -0.004795, 0.033561, -0.054907, 0.053452,
            -0.057542, 0.006851, -0.079816, -0.028793, 0.063853, 0.051702, -0.072140, 0.057284,
            -0.014697, -0.076817, 0.090653, 0.033975, 0.010539, 0.031767, 0.044276, -0.090174,
            0.057098, 0.073020, 0.012309, -0.028423, 0.014086, 0.006839, 0.016916, -0.062120,
            0.063179, 0.006123, 0.005852, -0.036971, 0.086700, -0.071678, -0.033832, -0.067448,
            0.075023, -0.040117, 0.001677, -0.006915, 0.075397, -0.084672, -0.051480, -0.097214,
            -0.021703, -0.086193, -0.002092, 0.038498, 0.009827, 0.009664, -0.020833, -0.071765,
            -0.015366, 0.081392, 0.018839, 0.017945, -0.029933, 0.067181, 0.066480, 0.021428,
            0.070717, 0.007702, 0.001923, 0.027435, -0.096170, 0.063842, -0.052091, -0.043535,
            -0.005994, -0.075186, -0.064371, 0.080918, -0.016349, 0.036683, 0.019969, 0.098016,
            -0.046868, -0.051197, 0.095758, -0.066619, -0.098440, -0.017568, 0.056543, 0.044641,
            -0.067073, -0.077440, 0.032115, -0.065504, 0.055443, -0.022650, -0.040487, -0.017971,
            0.081405, -0.060143, -0.076790, 0.040497, 0.042137, -0.019199, 0.060212, -0.024113,
            -0.041927, -0.076047, -0.050009, -0.055587, 0.026594, -0.018348, 0.072164, -0.048482,
            0.058229, 0.081123, -0.022798, -0.055886, 0.049790, -0.095189, 0.066186, 0.076756,
            -0.005835, -0.061981, 0.012719, 0.072368, 0.013388, 0.018706, 0.043586, 0.056558,
            0.021819, -0.054501, 0.002793, -0.042648, 0.050320, -0.051423, 0.087407, 0.042083,
            -0.099895, -0.041888, -0.015033, -0.032422, -0.003342, -0.053505, 0.070294, 0.045565,
            -0.009474, -0.043638, -0.076430, -0.057634, -0.055205, 0.093970, -0.000106, -0.057049,
            -0.007716, -0.003920, 0.024312, 0.032564, 0.062717, -0.049806, -0.021612, -0.085822,
            -0.035958, -0.087921, -0.008669, -0.060448, -0.034234, 0.093570, 0.090535, 0.013014,
            0.050885, 0.064611, 0.004398, 0.019916, -0.006723, -0.079763, 0.079655, -0.074091,
            -0.029904, -0.083238, -0.059244, 0.099556, -0.009227, 0.031308, 0.024195, 0.078123,
            0.020932, 0.003535, -0.045893, 0.018344, 0.021719, -0.068872, -0.072325, -0.043238,
            0.086032, 0.039729, 0.055983, 0.066648, -0.000997, -0.066772, 0.095816, 0.016473,
            -0.035972, 0.040246, -0.070215, 0.003525, 0.092860, 0.032967, -0.093730, 0.087194,
            0.021335, -0.026726, 0.078660, 0.044415, 0.058421, 0.048915, -0.056008, -0.072252,
            0.071817, -0.006178, 0.081208, 0.049686, -0.096400, 0.054655, 0.049869, -0.096275,
            0.005909, -0.079898, 0.048971, 0.053229, 0.030439, 0.062594, -0.055487, -0.003488,
            -0.027094, 0.076431, 0.071383, -0.035281, -0.084858, -0.003486, 0.078162, -0.021578,
            0.036805, -0.072691, -0.066486, -0.085826, -0.086016, -0.040648, -0.042647, 0.038617,
            0.082416, 0.012286, 0.040309, -0.063904, -0.043317, -0.093423, 0.093638, 0.010673,
            -0.027401, -0.091372, 0.018741, -0.098130, 0.067215, -0.093805, 0.021406, 0.062051,
            -0.080796, 0.073445, 0.096004, 0.069878, 0.090804, -0.046790, -0.017544, 0.089728,
            0.047372, -0.067877, 0.098495, -0.016902, -0.032529, 0.013789, 0.096219, -0.050392,
            0.008782, -0.011760, -0.029955, 0.046119, 0.006018, -0.082042, -0.002469, 0.063177,
            -0.047416, 0.088845, -0.082387, 0.033866, 0.062011, 0.035065, 0.039498, 0.014707,
            -0.021066, 0.071665, -0.022169, 0.084031, 0.095661, -0.086581, 0.081317, 0.024884,
            -0.068206, -0.055588, -0.013794, 0.093470, 0.061055, -0.026594, -0.059700, -0.064720,
            -0.050527, -0.002268, -0.086814, -0.089253, -0.062797, 0.027558, 0.035498, 0.029489,
            0.083151, 0.024207, -0.038832, 0.039697, -0.046660, 0.088946, -0.038383, -0.062456,
            -0.019180, -0.055769, -0.010043, -0.016713, 0.011717, -0.013234, -0.069670, -0.093682,
            -0.072296, -0.093053, -0.099721, 0.097045, 0.058554, 0.045341, 0.064390, 0.030395,
            -0.033034, -0.012419, -0.041104, 0.045107, 0.025737, -0.083994, 0.015540, -0.076984,
            0.075194, 0.010864, -0.001543, -0.059125, 0.015406, 0.012724, 0.025673, -0.070035,
            -0.042230, -0.069683, -0.062512, -0.054804, -0.080658, 0.098248, -0.094635, -0.026185,
            0.004240, -0.040270, -0.027738, 0.046804, -0.022979, -0.011818, 0.088241, -0.035899,
            -0.004424, 0.063618, -0.023450, -0.079768, 0.077005, -0.086182, -0.098561, -0.051350,
            0.013265, 0.081024, -0.078066, -0.049776, 0.090333, 0.078070, 0.037871, -0.084660,
            0.054957, -0.006355, -0.076861, 0.078338, -0.068395, 0.016297, -0.054485, 0.084777,
            0.003113, -0.024210, 0.056251, 0.091000, -0.007786, 0.067844, -0.035917, -0.054015,
            -0.054086, -0.095995, 0.088461, 0.024764, -0.070591, -0.086707, 0.062744, 0.004766,
            -0.068350, -0.009979, -0.036355, 0.013029, -0.041339, -0.046788, -0.058609, -0.041287,
            0.001605, 0.048660, 0.021302, 0.063777, -0.031239, -0.034492, -0.060101, -0.086208,
            -0.061356, -0.020814, 0.094168, -0.097465, 0.074814, 0.051334, 0.049776, 0.053124,
            0.053713, 0.086478, 0.037035, -0.020803, 0.038050, -0.007453, -0.048415, 0.036627,
            0.099980, 0.075013, -0.043115, -0.008859, 0.038295, 0.055478, -0.087064, 0.013609,
            -0.011807, 0.095954, 0.028423, -0.044883, -0.031865, 0.081575, -0.029589, 0.018588,
            0.087474, -0.088381, -0.008193, -0.090168, -0.030376, 0.036683, 0.000056, 0.049580,
            -0.068057, 0.044554, -0.043493, 0.064831, -0.025968, 0.081835, 0.008038, 0.034211,
            -0.075708, 0.014641, 0.073785, -0.076321, 0.073983, 0.037274, -0.042564, -0.065227,
            0.061423, -0.030432, 0.048691, 0.012212, 0.093387, -0.068845, -0.013996, 0.043779,
            -0.049137, -0.027473, -0.021901, 0.098088, 0.043943, -0.092741, 0.027929, 0.084501,
            0.050384, 0.090836, -0.019798, 0.009933, 0.097720, 0.092179, -0.029378, 0.021065,
            0.004063, 0.088249, -0.043914, -0.012355, 0.036601, -0.015537, -0.047062, 0.085148,
            -0.037100, 0.061438, -0.022517, -0.098628, -0.069129, 0.025636, 0.071619, -0.014057,
            0.065129, 0.018368, 0.042523, 0.099019, 0.068221, 0.077467, -0.063542, -0.073764,
            0.079282, 0.092229, 0.050814, 0.039677, -0.023137, 0.035028, -0.048169, -0.004228,
            0.093575, -0.063000, -0.084042, -0.007790, 0.052159, -0.093004, 0.063653, 0.004457,
            0.011964, 0.061921, 0.050560, -0.005672, -0.095140, 0.086489, -0.032376, -0.055546,
            0.033445, 0.067211, 0.014240, -0.056901, -0.090816, 0.043535, 0.035004, 0.059440,
            0.099641, -0.054472, 0.031062, 0.099204, -0.003122, -0.037129, -0.078063, 0.001602,
            -0.050505, -0.068080, -0.095081, 0.041267, -0.014425, 0.043114, 0.053538, -0.099379,
            -0.036254, 0.089115, -0.017232, 0.067370, -0.086455, 0.091401, 0.042232, -0.070223,
            0.023483, 0.063724, 0.053878, 0.019415, -0.038851, -0.031303, -0.072584, 0.098293,
            0.019079, 0.021669, -0.057761, -0.078085, -0.011754, 0.051628, 0.075433, -0.009804,
            0.042001, -0.017608, -0.029689, 0.099935, 0.024353, 0.098170, -0.098094, 0.007931,
            -0.034242, 0.077447, 0.050126, 0.016683, 0.062270, -0.059346, -0.026220, 0.050753,
            -0.058140, 0.024044, 0.032535, -0.083138, -0.058773, -0.074011, -0.091463, -0.035482,
            0.031434, 0.037136, 0.056674, 0.012535, -0.064007, -0.094288, 0.087138, 0.017780,
            0.076723, -0.093785, -0.054384, -0.002637, 0.082746, -0.082014, -0.053188, -0.075864,
            0.050868, 0.007724, -0.037978, 0.066687, -0.074343, -0.009877, -0.045725, -0.022337,
            0.001032, -0.002261, 0.099064, -0.088268, -0.041880, -0.007617, -0.041848, 0.092891,
            0.051461, 0.075641, 0.073441, -0.005344, 0.065891, -0.008434, 0.071836, 0.014565,
            0.039822, 0.041204, -0.037959, 0.085491, 0.081497, -0.024167, -0.029602, -0.088730,
            0.052608, -0.076650, -0.092382, 0.082288, 0.022161, 0.087089, -0.083654, -0.056332,
            -0.026684, 0.061821, -0.075918, -0.086164, -0.029448, -0.051880, 0.096898, 0.051013,
            -0.067498, -0.043102, -0.003681, -0.062582, -0.075559, 0.089612, -0.066504, 0.057756,
            -0.028788, -0.075966, -0.045309, -0.045820, -0.067918, -0.061256, -0.004542, -0.052655,
            0.081326, -0.095537, -0.016291, 0.065831, -0.066002, -0.022844, -0.081540, -0.096473,
            0.047376, 0.025545, 0.009733, 0.079079, -0.002287, -0.083441, -0.097295, 0.014501,
            -0.059255, 0.093124, -0.012511, 0.086017, -0.016137, -0.046280, -0.099063, 0.095361,
            -0.086430, 0.070709, -0.038827, -0.073811, -0.034380, 0.048415, 0.076314, -0.083640,
            -0.074565, 0.033952, 0.047735, -0.064219, 0.041624, -0.014006, -0.087256, 0.056013,
            -0.083760, -0.005662, -0.021619, -0.045962, -0.016170, -0.089258, 0.088521, -0.074796,
            0.061726, 0.003459, -0.004462, -0.007280, -0.042671, -0.018121, 0.071965, -0.047497,
            0.045615, -0.080680, -0.049445, 0.069152, 0.036588, -0.090814, -0.070038, 0.064721,
            -0.006856, 0.092906, -0.098629, 0.026525, -0.098898, 0.041619, -0.033867, -0.010608,
            -0.012347, 0.012857, -0.067299, -0.055483, 0.043581, -0.057850, 0.027986, -0.011811,
            0.063155, -0.025165, 0.070488, -0.002659, -0.055560, 0.018881, -0.061465, 0.024108,
            -0.048002, 0.012072, -0.018840, -0.025892, -0.086294, -0.033569, 0.083491, 0.056387,
            -0.041817, 0.011727, 0.025022, 0.061431, 0.068242, -0.054892, -0.023743, 0.082081,
            0.010231, -0.055375, -0.050254, -0.049552, -0.094264, -0.067109, 0.099299, -0.012043,
            0.066701, -0.092520, 0.047154, 0.038536, -0.093562, -0.024211, 0.026175, -0.078574,
            0.038449, 0.058214, -0.090991, -0.038497, 0.070142, -0.008904, -0.058481, 0.024993,
            -0.082881, 0.033392, -0.057829, 0.036330, -0.053138, -0.029887, -0.046422, 0.052277,
            -0.091702, 0.060541, 0.051940, -0.070418, 0.067526, 0.074192, -0.062993, 0.009134,
            0.025030, -0.013768, 0.072749, -0.025392, -0.034297, 0.028229, 0.053725, 0.078993,
            -0.099570, -0.024810, -0.044243, -0.036578, -0.027250, -0.056295, -0.059040, -0.054073,
            -0.067808, 0.039669, 0.054822, -0.045545, 0.094363, 0.036704, -0.075886, 0.047536,
            0.008898, 0.085021, -0.008727, -0.040323, 0.077765, -0.091923, 0.042867, -0.085543,
            -0.023934, -0.010040, -0.094073, -0.055505, -0.040016, -0.013702, 0.020484, -0.069478,
            0.081670, 0.015186, 0.012347, 0.022471, -0.063124, -0.001556, -0.088335, 0.096901,
            -0.037476, -0.092779, 0.008829, -0.073215, -0.085723, -0.070362, -0.014797, 0.035252,
            -0.099959, -0.014911, -0.022869, -0.043550, 0.041403, 0.070011, 0.012080, -0.038293,
            0.031482, -0.044699, -0.006201, 0.056377, -0.019538, 0.096350, 0.099000, 0.028693,
            -0.009576, -0.023935, 0.040854, 0.015089, -0.084403, -0.085274, 0.000698, 0.025113,
            0.062310, 0.071044, 0.031778, 0.074041, 0.073692, -0.073383, 0.094976, -0.048070,
            -0.063287, 0.013090, -0.067459, 0.007471, 0.003519, 0.030529, -0.057074, 0.050841,
            0.074311, 0.009295, 0.046865, -0.057127, 0.042566, -0.058460, -0.035516, -0.018869,
            -0.000621, 0.090414, 0.014747, 0.098967, 0.023587, -0.031222, -0.033064, -0.009775,
            -0.065619, 0.038797, 0.089424, 0.057478, 0.061153, 0.097110, 0.012919, -0.006084,
            -0.033809, -0.083710, -0.066224, -0.005410, -0.070518, -0.068959, 0.016512, -0.057468,
            -0.083094, -0.069460, 0.038297, 0.013733, -0.095051, -0.080689, 0.039948, 0.093662,
            0.034639, 0.050331, -0.004641, 0.042288, 0.050681, -0.053699, 0.050727, -0.027077,
            0.035562, -0.094225, -0.054192, -0.058704, -0.040508, 0.091652, -0.049142, 0.026407,
            -0.009209, 0.015789, -0.044678, -0.057859, -0.081304, 0.008133, -0.091844, -0.092756,
            0.089496, 0.041917, -0.016415, -0.046538, 0.088479, -0.004325, -0.083339, 0.071389,
            0.093064, 0.018230, 0.071913, -0.080163, 0.002769, 0.098035, -0.098357, -0.094697,
            0.008870, 0.026691, 0.025122, -0.028584, 0.083503, 0.072377, -0.098773, -0.095042,
            0.084817, -0.040690, 0.019167, 0.035321, 0.023005, -0.051575, -0.068150, -0.077343,
            0.041323, -0.012741, -0.058201, -0.080773, 0.054008, 0.013592, -0.095823, -0.005461,
            -0.027248, -0.002789, 0.024747, 0.034837, -0.094005, 0.014570, 0.051690, -0.023354,
            0.076891, 0.097958, 0.027794, 0.095967, 0.042514, -0.066812, 0.062511, -0.042630,
            -0.071766, -0.090150, 0.008810, 0.089991, -0.034439, 0.016014, 0.032491, 0.061835,
            0.084457, 0.053040, -0.035602, -0.002827, -0.095864, -0.077526, -0.054262, 0.059601,
            0.017260, -0.000316, 0.065968, -0.009180, 0.076217, 0.094037, -0.081199, 0.079419,
            0.063304, -0.011351, 0.030052, -0.067197, -0.031610, -0.056910, -0.069552, -0.016701,
            -0.099670, -0.018714, 0.063734, -0.061695, -0.032725, 0.004062, -0.043434, 0.009190,
            -0.092187, 0.027003, 0.093569, 0.006123, 0.097443, -0.088090, -0.051955, -0.030375,
            0.085485, -0.091601, -0.018176, -0.089557, 0.019235, -0.086849, -0.082128, -0.062238,
            -0.038408, 0.063498, 0.016174, -0.090146, 0.094558, -0.002029, 0.029207, -0.051638,
            -0.047149, -0.074894, -0.078631, -0.051096, 0.055191, -0.038057, -0.084860, 0.029304,
            -0.026271, -0.071668, 0.080980, -0.093317, -0.097252, 0.057796, 0.055568, -0.038273,
            0.070684, -0.065775, -0.047027, -0.014395, -0.033473, 0.064657, -0.092943, 0.080968,
            0.057277, -0.055134, -0.052704, 0.015556, -0.007603, -0.088674, 0.081810, 0.078685,
            0.080021, 0.047483, -0.069614, -0.052619, -0.042220, 0.027738, 0.053905, -0.004724,
            -0.098390, -0.063384, -0.047815, -0.067314, 0.051483, 0.004462, -0.071369, -0.060666,
            -0.073059, -0.069278, 0.037500, -0.052532, -0.085628, 0.007222, -0.005996, -0.082912,
            0.026942, 0.008536, 0.013061, 0.043927, -0.050478, -0.087384, 0.064640, 0.090701,
            0.092876, 0.053369, 0.091918, 0.028226, -0.087249, 0.029745, -0.076469, -0.094202,
            0.038694, -0.052963, -0.072242, 0.094438, 0.031869, 0.042730, -0.029847, 0.037395,
            -0.001574, -0.023356, -0.019080, 0.074213, -0.068375, -0.030831, 0.073021, -0.004659,
            0.049094, -0.042385, 0.067093, -0.071085, -0.053621, -0.087965, -0.079873, 0.062965,
            0.064659, 0.053601, 0.049498, 0.067831, 0.060826, 0.058849, -0.049093, 0.074931,
            0.066404, 0.030982, -0.018014, 0.087421, -0.006645, -0.093211, -0.037571, -0.002555,
            0.047536, 0.026898, 0.017299, 0.085087, 0.065378, -0.056802, -0.036265, -0.009531,
            -0.005699, -0.076245, -0.026793, 0.075746, -0.096949, 0.014407, 0.020252, -0.031943,
            -0.034016, -0.050122, -0.090881, -0.090177, 0.053169, -0.002999, -0.004351, -0.016400,
            -0.094583, -0.016315, -0.092219, 0.049274, -0.035096, -0.033983, 0.035221, -0.068544,
            -0.065074, -0.055493, -0.022552, -0.063183, -0.099626, 0.044718, 0.054370, -0.041268,
            -0.076898, 0.067691, -0.060724, -0.077539, -0.090191, 0.024768, 0.022675, 0.095188,
            -0.039314, -0.062476, 0.097699, 0.068737, -0.088510, -0.093663, 0.037228, -0.049812,
            0.017428, -0.006564, -0.031109, 0.085365, 0.032791, 0.030094, -0.075057, -0.035103,
            0.046743, 0.054968, 0.010634, -0.097944, -0.053793, 0.098539, 0.081201, 0.052659,
            -0.058384, -0.053950, 0.036174, -0.083724, -0.011677, 0.038302, 0.039419, 0.049188,
            0.005982, 0.067112, -0.097787, -0.051601, -0.030829, 0.058469, -0.078064, 0.073746,
            -0.057928, -0.030209, 0.073999, 0.097891, 0.029993, -0.021365, 0.070745, -0.004841,
            0.060264, 0.040501, -0.002180, 0.094368, 0.048209, 0.093910, 0.094438, 0.019332,
            -0.046790, -0.033621, -0.015911, -0.032635, 0.010434, 0.097215, 0.065869, 0.027383,
            0.051979, -0.056459, 0.040201, 0.087596, 0.032343, 0.001539, 0.080755, -0.082232,
            0.056453, -0.019361, -0.092177, 0.056064, -0.019998, -0.022465, -0.045714, -0.033019,
            -0.015372, -0.098192, 0.048631, 0.012911, 0.037136, 0.099505, -0.022315, -0.060015,
            -0.064935, -0.013781, -0.019467, 0.035608, 0.007110, 0.033319, -0.050959, -0.086101,
            0.003729, 0.021152, -0.047834, -0.035687, -0.017635, -0.089417, 0.093336, 0.072046,
            -0.073010, -0.095292, 0.087071, 0.083027, -0.089805, 0.021198, -0.063524, -0.073608,
            -0.069564, 0.003135, 0.092353, 0.046616, -0.092347, 0.053919, 0.090762, 0.070571,
            0.092389, -0.058146, -0.050891, 0.055567, -0.078773, -0.028280, 0.071782, 0.030720,
            -0.026754, 0.039997, -0.046756, -0.050674, 0.030005, -0.048263, 0.069587, 0.044583,
            0.060729, 0.071791, 0.012616, -0.036921, -0.084004, -0.028965, -0.084111, 0.060782,
            0.007784, -0.042478, -0.003111, 0.072160, -0.043808, 0.019279, 0.012025, -0.013544,
            -0.023247, 0.037461, -0.032278, -0.082123, 0.022571, -0.058166, -0.051258, 0.002813,
            0.048110, -0.029017, -0.060061, 0.046208, -0.061779, 0.021588, 0.028456, -0.002370,
            -0.025703, -0.013459, -0.056578, 0.065767, -0.008934, -0.041804, -0.054793, 0.044193,
            -0.061712, 0.041299, 0.000311, -0.092346, 0.063893, 0.039395, 0.042220, -0.044851,
            -0.086964, 0.047139, -0.077921, 0.047430, 0.086631, -0.012422, -0.081037, 0.075016,
            -0.095793, -0.033115, -0.074174, 0.042423, 0.090407, -0.063954, -0.057403, 0.081092,
            0.049094, 0.069996, -0.071680, 0.093465, -0.036313, -0.055280, -0.033427, 0.006462,
            0.058376, -0.025575, -0.067539, 0.016178, -0.096277, -0.079654, 0.021511, 0.089008,
            -0.057793, 0.056683, -0.062266, -0.034341, 0.099180, 0.027150, -0.042209, -0.045020,
            0.021991, 0.062156, 0.075189, 0.007343, -0.050469, 0.049854, -0.083728, -0.003210,
            -0.092154, -0.020991, -0.000592, 0.081898, 0.022901, -0.067498, 0.045907, 0.002628,
            -0.063606, -0.084585, 0.020664, -0.057460, -0.038278, -0.051236, -0.074471, -0.064706,
            -0.014864, -0.002458, 0.067515, -0.082069, 0.085576, 0.092341, -0.096696, 0.064393,
            0.081653, -0.076096, 0.051889, 0.042575, 0.052379, -0.034429, -0.052084, -0.000635,
            -0.030627, 0.044420, -0.074470, 0.084825, -0.031948, 0.057347, -0.032330, -0.015252,
            0.014851, -0.088200, 0.046057, 0.090479, -0.074690, 0.038245, -0.090938, -0.003934,
            -0.045523, -0.086482, 0.043818, 0.062537, 0.013638, 0.033690, -0.055312, 0.066081,
            -0.079589, -0.027477, -0.036006, -0.043142, 0.082332, 0.077402, 0.051398, -0.038533,
            -0.061606, -0.027703, -0.039045, 0.011146, 0.050460, 0.093788, 0.070075, -0.082154,
            0.085899, -0.092781, 0.011062, 0.011875, -0.017787, -0.030815, -0.043128, 0.057340,
            -0.063741, 0.019718, 0.036086, 0.076428, 0.001926, 0.086134, -0.097863, -0.055257,
            -0.086902, -0.002880, -0.061843, 0.066742, -0.047539, 0.008502, 0.084541, -0.098554,
            0.041251, -0.085182, 0.048458, -0.026203, -0.085804, 0.016478, 0.007302, 0.034789,
            0.059964, -0.010323, -0.046374, 0.097733, -0.030881, 0.085558, 0.066683, -0.098073,
            -0.028958, 0.006460, -0.080137, 0.017094, -0.041335, -0.051318, 0.074431, 0.042993,
            0.099521, -0.020456, -0.028442, -0.063868, 0.011313, 0.078459, -0.010945, -0.034643,
            0.063922, 0.003487, 0.014203, 0.094238, 0.095778, 0.095609, -0.090336, 0.016982,
            0.081308, 0.088659, -0.086783, 0.073758, -0.046204, -0.050080, -0.056326, -0.053155,
            -0.015637, -0.001794, -0.043813, 0.064270, -0.057428, 0.089085, -0.090244, 0.032955,
            -0.061255, -0.055609, 0.023708, -0.018721, -0.067091, -0.005908, -0.072081, -0.059474,
            0.059104, 0.006195, -0.028240, 0.080071, 0.012303, 0.072320, 0.041406, 0.008210,
            -0.036849, -0.043939, 0.087558, 0.021380, 0.085254, 0.059756, -0.005647, -0.092149,
            0.040328, -0.022185, 0.020049, -0.036429, 0.078577, 0.096257, -0.059307, -0.023784,
            -0.074441, 0.048406, -0.072457, -0.077324, -0.073102, -0.080653, 0.077249, -0.043056,
            0.061817, -0.027673, -0.030176, -0.027393, -0.081116, 0.028701, -0.040223, 0.029934,
            -0.085207, 0.015642, 0.081221, -0.030739, 0.021316, -0.098724, -0.040845, 0.082827,
            0.039665, 0.036638, 0.043400, -0.020418, -0.042447, 0.094109, -0.036670, -0.007975,
            -0.075787, 0.076653, -0.032670, -0.037283, -0.062198, 0.023276, -0.077431, 0.086139,
            0.073516, -0.076106, -0.091756, 0.026131, -0.059460, -0.099129, 0.062785, -0.025148,
            0.020691, 0.010695, -0.062414, -0.049055, 0.034766, -0.088296, -0.013733, -0.008798,
            -0.055444, 0.098836, -0.088616, -0.085718, -0.072296, 0.056709, -0.085124, 0.060704,
            0.097067, 0.093855, 0.093196, 0.016244, -0.087440, 0.024489, 0.041172, -0.007556,
            -0.087636, 0.075443, 0.019269, 0.067362, -0.011092, -0.089383, 0.085161, -0.097872,
            -0.071441, 0.091366, -0.028781, 0.054301, 0.061255, -0.011743, -0.020108, 0.044433,
            -0.020293, 0.041329, -0.031685, -0.054862, -0.063164, -0.099402, -0.030447, -0.045685,
            0.031650, 0.000868, -0.066149, -0.072475, -0.015018, 0.036048, 0.005884, 0.011640,
            -0.053702, -0.044953, -0.013770, 0.041671, -0.091750, 0.042508, 0.056171, -0.098809,
            0.041970, 0.015084, 0.039813, -0.055514, 0.063016, -0.057747, -0.043504, 0.032187,
            -0.021538, 0.069696, -0.011581, -0.077964, -0.083335, 0.005148, -0.077927, 0.089395,
            0.086930, 0.060201, -0.076303, -0.022988, 0.073781, 0.020365, 0.007867, -0.007759,
            -0.054886, -0.031813, 0.058291, -0.019478, -0.026792, 0.049283, -0.048134, -0.014836,
            0.083072, 0.014314, 0.045551, -0.049335, 0.048342, -0.054177, 0.097891, -0.027686,
            -0.072993, -0.021621, 0.073660, 0.078461, 0.007071, 0.069556, -0.033365, -0.009876,
            0.073550, -0.001086, 0.021104, 0.059477, 0.007735, -0.058263, 0.010834, -0.006026,
            -0.009914, 0.096757, 0.085294, -0.090582, -0.036381, -0.011005, 0.057643, -0.021928,
            -0.015370, -0.025917, -0.062300, -0.085906, -0.079474, 0.018977, 0.046620, -0.006417,
            0.092075, -0.017375, 0.074499, -0.016765, -0.055601, 0.009818, -0.099690, 0.037478,
            0.021183, 0.046599, 0.052162, 0.085872, -0.065604, -0.094220, 0.008519, -0.044155,
            0.000503, 0.086644, 0.010625, 0.012016, -0.083193, 0.000524, -0.080187, -0.025026,
            0.068779, -0.008154, 0.042076, 0.033973, 0.067376, 0.032097, -0.088708, -0.007728,
            0.009140, -0.042068, -0.075400, -0.064514, 0.046175, -0.034456, -0.009054, 0.030316,
            0.066013, 0.051654, 0.044770, 0.090567, 0.018378, -0.044097, 0.045241, -0.009689,
            0.098486, -0.077332, -0.058304, 0.064957, -0.065527, 0.054620, -0.066020, 0.030607,
            0.086009, 0.039980, -0.059543, 0.039992, -0.000871, -0.020647, -0.098444, -0.065915,
            -0.033982, 0.074558, -0.068711, -0.093567, -0.073181, 0.032161, -0.089170, -0.099962,
            0.002767, -0.094318, 0.019794, -0.010925, 0.091015, -0.037238, -0.092393, -0.058390,
            -0.005108, -0.040067, 0.097870, -0.078117, -0.096117, 0.043605, -0.024070, 0.062171,
            0.016744, 0.034420, 0.071772, -0.012304, -0.042215, -0.036000, -0.026157, -0.089342,
            0.082283, 0.043559, -0.034817, 0.078564, 0.040074, 0.041312, 0.058597, 0.015845,
            -0.007857, 0.075289, 0.031972, 0.066936, 0.056049, -0.066682, -0.024888, 0.077930,
            -0.017834, -0.043421, -0.043216, -0.079213, 0.058460, 0.057734, -0.054869, -0.097018,
            0.023434, 0.005963, 0.070442, 0.020816, -0.014748, 0.061140, 0.065439, 0.050162,
            0.068967, -0.090718, 0.041620, 0.055591, 0.060621, -0.040719, 0.023987, -0.081734,
            0.080103, 0.007215, 0.069625, 0.027541, -0.045167, 0.055391, 0.064459, -0.053082,
            0.008473, -0.053292, -0.071365, 0.038566, -0.083849, 0.038109, -0.075908, -0.003788,
            -0.035916, 0.047420, 0.093201, -0.075040, -0.027282, -0.016224, -0.087807, -0.088737,
            -0.046341, -0.021818, 0.086829, 0.031251, 0.087073, -0.081158, -0.053205, 0.010866,
            -0.079408, 0.045546, -0.000385, -0.098320, -0.097817, 0.010054, 0.033019, -0.003540,
            -0.029433, 0.064471, 0.005895, -0.049768, 0.083752, -0.025919, -0.031955, -0.009736,
            -0.019844, -0.006627, -0.042694, 0.059925, 0.004561, 0.056733, 0.011298, 0.066598,
            -0.067147, 0.052007, 0.038700, 0.044533, -0.053725, 0.009923, -0.037958, -0.035964,
            -0.015783, -0.015272, 0.045072, 0.099072, -0.026115, 0.095190, 0.053192, -0.089255,
            0.078189, -0.079215, -0.041041, -0.037531, -0.001980, -0.098482, -0.093894, -0.081785,
            -0.048725, 0.000465, 0.066480, -0.091320, 0.011116, 0.076730, 0.030692, 0.021874,
            -0.088870, 0.063558, -0.022356, 0.041139, -0.066327, -0.016931, 0.058897, -0.064161,
            -0.077099, -0.011822, 0.051429, -0.067965, 0.012855, -0.042036, 0.014652, 0.015234,
            0.067319, -0.050532, -0.091964, 0.011773, 0.055860, -0.044096, 0.032358, -0.091149,
            0.049340, 0.051767, -0.093312, -0.023513, 0.043232, 0.045134, 0.029713, -0.072186,
            0.083266, -0.087414, -0.002037, -0.053242, -0.082040, -0.043877, 0.089285, -0.086961,
            0.068264, 0.088819, -0.056238, 0.030395, -0.038541, -0.094085, -0.087667, -0.017317,
            -0.072050, -0.013125, 0.057685, 0.048571, -0.098864, 0.038925, -0.000693, 0.034307,
            0.063949, -0.083356, -0.051819, 0.042855, 0.016592, -0.028288, -0.050428, 0.014796,
            -0.027020, 0.029249, -0.044174, 0.097201, 0.012549, -0.026684, -0.082515, 0.097355,
            -0.074955, -0.087131, 0.039184, -0.051573, 0.072970, 0.020005, -0.058554, -0.088569,
            -0.066172, 0.096421, 0.002737, -0.058417, 0.084957, 0.034990, 0.074666, 0.068567,
            0.063739, 0.051850, 0.045104, 0.040569, 0.049190, -0.029820, -0.024266, -0.043154,
            -0.054649, 0.084982, 0.059828, 0.076470, 0.039719, 0.070184, 0.060699, -0.009281,
            -0.092981, 0.038775, -0.031072, -0.063964, 0.074671, 0.087796, 0.059919, -0.032415,
            -0.044417, 0.084569, 0.059593, -0.017385, -0.076392, -0.044467, -0.039488, -0.060226,
            -0.085156, -0.022682, 0.079252, -0.016696, 0.041419, 0.001649, -0.086442, 0.079883,
            0.085544, -0.021462, 0.035600, 0.040530, -0.063658, 0.019032, -0.057063, 0.049477,
            -0.078918, 0.003867, -0.033046, -0.085659, 0.009468, -0.026070, -0.006925, 0.005634,
            0.080492, -0.070652, 0.043291, -0.057789, 0.045556, 0.035868, 0.082242, -0.043070,
            0.010016, -0.022652, -0.080689, -0.079408, 0.094429, 0.003245, 0.058924, -0.090518,
            0.079680, 0.056488, 0.051366, 0.065278, -0.066095, 0.005614, -0.087036, -0.008314,
            0.080839, 0.051134, 0.057371, -0.056218, -0.088647, -0.027201, 0.075280, 0.077833,
            0.097978, 0.043579, -0.005649, -0.095269, -0.005424, 0.059060, 0.035741, -0.031087,
            -0.050341, 0.056496, 0.005075, 0.006677, -0.028677, 0.041173, -0.026264, 0.060919,
            0.019970, -0.029898, -0.009731, -0.045198, 0.074467, -0.027873, 0.082970, -0.097841,
            0.064504, 0.040520, 0.081466, -0.036027, -0.070295, 0.035872, -0.078449, 0.080707,
            0.011002, 0.069811, 0.070028, -0.022350, -0.040379, -0.028096, -0.053194, 0.030953,
            0.020850, -0.064314, 0.019385, 0.092785, 0.080247, 0.053208, 0.011987, 0.047534,
            0.052452, -0.041700, -0.028555, -0.004165, 0.051385, -0.087388, 0.017589, 0.071244,
            0.030410, -0.075771, -0.054901, -0.089308, -0.080920, -0.080432, -0.007947, -0.041261,
            -0.096843, 0.088825, 0.023891, -0.065851, -0.055990, -0.006047, 0.066038, -0.031896,
            -0.037017, 0.063252, -0.014288, -0.035967, 0.066987, -0.080383, 0.026527, -0.086957,
            0.003875, -0.017440, -0.067982, -0.053138, 0.064322, 0.079380, 0.020879, 0.041933,
            0.056229, -0.017278, -0.005517, 0.081253, 0.061328, -0.037926, 0.022822, 0.030376,
            -0.035383, 0.060900, -0.039775, 0.040963, 0.056372, -0.010114, -0.078976, -0.082399,
            -0.012594, 0.055734, 0.089972, 0.059728, 0.006157, -0.010841, 0.011152, -0.035758,
            -0.020261, 0.056592, -0.081611, -0.063257, -0.026636, -0.062378, -0.074576, 0.049156,
            0.045029, 0.053049, -0.070561, 0.030783, 0.053320, 0.020249, -0.046909, -0.044768,
            0.054811, 0.033112, 0.025666, -0.076395, -0.089169, 0.078317, 0.006226, 0.084409,
            -0.095872, -0.046825, -0.092552, 0.092496, 0.014716, 0.039144, 0.076469, 0.005882,
            0.021562, -0.072357, 0.010787, 0.070072, -0.014744, -0.069301, -0.051716, 0.002894,
            0.066110, 0.008597, -0.057385, 0.008418, -0.032448, 0.005555, 0.062579, -0.083323,
            -0.027116, -0.052566, 0.084201, -0.083981, -0.023375, -0.089756, -0.070956, -0.039925,
            0.018342, -0.088322, -0.010233, -0.054550, 0.070568, -0.064600, -0.084083, -0.009529,
            -0.082302, -0.038481, 0.032663, -0.081004, -0.040408, 0.056974, 0.074024, 0.082901,
            0.084457, 0.081728, -0.023906, -0.044418, -0.022086, 0.021575, 0.010330, -0.063074,
            0.083137, -0.027307, 0.075596, 0.002215, 0.040749, 0.007171, 0.037392, -0.060756,
            0.042386, -0.058514, 0.050513, -0.045457, 0.026780, -0.094986, -0.064228, 0.084262,
            0.076034, -0.088920, -0.061709, -0.091623, 0.010237, -0.026905, -0.001084, 0.031790,
            0.027798, -0.015975, 0.015947, -0.081716, 0.076769, -0.087585, -0.052389, -0.033239,
            -0.058348, 0.005269, 0.050338, -0.034218, -0.031898, -0.056164, 0.037552, -0.069552,
            0.073533, 0.022216, -0.077534, -0.007163, 0.029361, 0.099402, -0.035450, -0.080856,
            -0.053868, 0.024522, 0.067016, 0.003214, 0.089097, -0.069358, -0.099401, 0.012563,
            0.062879, 0.093560, 0.005795, -0.007282, 0.049787, 0.061602, -0.091176, -0.049015,
            0.071315, -0.028778, -0.081066, -0.074496, -0.051515, -0.010689, -0.006893, -0.082206,
            0.095223, 0.004286, 0.097306, -0.035541, 0.073449, 0.029688, -0.090612, 0.069698,
            0.064448, -0.074417, 0.082224, -0.020501, -0.040508, -0.030116, -0.069682, 0.084634,
            -0.033510, -0.058899, 0.096373, -0.099004, -0.074755, 0.031510, 0.090732, -0.049570,
            -0.050232, -0.053165, 0.064751, -0.004499, -0.076429, -0.044113, -0.080744, 0.038934,
            -0.049841, 0.007159, 0.068040, 0.070538, -0.097866, 0.048437, 0.088804, 0.078911,
            0.035979, -0.004849, -0.058343, -0.050838, -0.056602, -0.028696, 0.055291, 0.096802,
            0.057876, -0.026630, 0.009851, -0.038161, -0.032720, -0.020743, -0.004420, 0.007143,
            0.046013, 0.034538, 0.068455, 0.013257, -0.080060, 0.006067, 0.040897, -0.088886,
            0.004503, 0.062071, -0.066503, 0.046851, -0.061250, 0.038228, 0.035594, 0.040546,
            -0.068934, 0.044505, -0.071687, -0.050340, 0.040688, -0.069559, 0.056704, 0.051573,
            0.096031, -0.056851, 0.026716, 0.078295, 0.036974, -0.024674, 0.087140, 0.063002,
            0.027140, -0.031820, -0.061442, 0.089005, 0.001093, -0.015499, -0.000369, -0.052612,
            0.047080, 0.023406, 0.019776, -0.015494, 0.033790, 0.085157, 0.053228, 0.092040,
            0.030771, 0.051549, -0.032338, -0.069647, 0.051677, 0.013406, -0.003470, 0.014644,
            0.040781, -0.003956, 0.044873, 0.096649, -0.014531, -0.027935, -0.046673, 0.062985,
            -0.027583, 0.022897, -0.068629, 0.030947, -0.075524, 0.015821, -0.058430, -0.091543,
            0.005317, 0.032044, -0.069248, -0.054077, 0.067144, -0.014957, 0.063852, -0.052466,
            -0.027155, 0.051776, -0.088157, 0.029090, -0.062235, -0.049049, 0.019512, 0.057440,
            -0.052189, 0.009350, -0.014974, -0.070647, -0.015843, 0.010373, -0.047221, -0.050123,
            0.081013, -0.060816, -0.010363, 0.075469, -0.039595, -0.077298, -0.050853, -0.077974,
            0.070498, -0.046020, 0.033471, 0.053220, -0.024429, 0.043384, 0.084742, -0.012746,
            0.032692, -0.049769, 0.075163, -0.094317, -0.023472, 0.008373, -0.077242, -0.074999,
            0.050821, -0.085098, -0.001495, 0.090454, -0.082681, 0.099143, 0.079169, -0.026212,
            0.032644, -0.026691, -0.052141, 0.060146, 0.014829, 0.035785, 0.090755, -0.012454,
            -0.011091, -0.086013, -0.088488, -0.055876, -0.061897, 0.076874, -0.041121, -0.016893,
            0.073052, -0.046744, 0.005361, -0.087469, 0.073854, -0.098366, 0.060428, -0.008335,
            -0.042981, -0.021433, -0.077027, -0.032404, 0.070336, 0.081818, 0.058147, -0.036459,
            -0.024828, 0.038858, -0.096169, -0.089670, -0.094665, 0.096290, 0.043963, -0.049009,
            -0.027188, 0.017848, 0.030893, 0.015159, 0.044265, 0.015485, -0.028264, 0.028779,
            0.062006, -0.013085, 0.024916, -0.099666, -0.090381, -0.093750, 0.032345, -0.060220,
            0.044969, 0.005687, 0.035154, -0.083374, 0.087674, -0.087740, -0.031137, -0.009852,
            0.019433, -0.014407, -0.098253, -0.062900, -0.055345, 0.099069, 0.026761, 0.063630,
            -0.041812, 0.060671, -0.043480, 0.029177, 0.038426, 0.097002, 0.027732, -0.007099,
            0.082215, 0.068808, -0.067143, -0.043376, 0.068531, -0.033627, -0.041897, -0.014192,
            0.060259, -0.086052, 0.057164, 0.009057, 0.046958, -0.075780, -0.009430, 0.080525,
            -0.044784, 0.008544, 0.009633, -0.056174, 0.095794, -0.096860, 0.011083, -0.019481,
            0.089246, 0.059317, -0.081044, -0.011184, 0.090204, 0.009199, 0.049084, -0.081497,
            -0.027023, -0.097982, -0.072054, -0.071297, -0.016848, 0.013031, -0.097627, -0.035025,
            -0.000253, 0.099590, -0.028500, -0.065628, 0.017269, 0.008700, 0.000888, -0.021575,
            -0.005867, 0.077411, -0.078384, 0.041358, -0.043103, 0.029017, -0.052474, -0.078142,
            0.042082, 0.028005, 0.049003, -0.036289, 0.019442, 0.071736, 0.075197, 0.064764,
            -0.020462, -0.051558, 0.008440, -0.067058, -0.029258, 0.012719, 0.035495, -0.010390,
            0.038207, -0.085493, 0.038008, -0.065370, -0.081779, 0.010364, 0.088120, 0.014532,
            0.035603, -0.013387, 0.005368, 0.010808, -0.044293, 0.081036, -0.062398, 0.051578,
            0.067346, -0.052483, 0.086228, 0.040365, 0.087101, 0.039966, -0.058790, 0.061321,
            0.080799, 0.061303, 0.035916, 0.016813, 0.006919, 0.098373, -0.048742, 0.022507,
            -0.002433, 0.090194, 0.057011, -0.061883, 0.045511, -0.032502, 0.069661, -0.084286,
            -0.090267, 0.044543, -0.027835, 0.025806, 0.082793, -0.069427, -0.012603, -0.035385,
            0.054114, -0.037874, 0.084424, 0.003323, 0.028911, -0.049492, -0.003934, -0.089621,
            -0.044803, -0.028593, -0.083877, -0.001617, 0.051037, -0.059164, -0.006745, 0.074753,
            -0.060488, 0.029252, -0.020708, 0.083170, -0.008614, 0.082926, -0.032820, -0.077584,
            0.002838, -0.059637, -0.054543, 0.070221, 0.092357, -0.065606, -0.099487, 0.030292,
            -0.084086, -0.028497, 0.041486, 0.035642, 0.014727, 0.043964, -0.055910, -0.032144,
            -0.032694, -0.099683, -0.057091, 0.013248, 0.026504, -0.043481, 0.069446, -0.054751,
            0.005500, 0.086375, 0.066163, 0.094439, 0.027815, -0.069088, -0.034083, -0.028694,
            -0.073912, -0.037458, -0.093697, -0.047173, -0.099727, 0.097756, 0.015789, 0.095280,
            0.057073, 0.098056, -0.029765, -0.047276, -0.042353, -0.091592, -0.005946, 0.014705,
            0.056963, -0.018918, 0.032396, 0.042442, 0.020411, 0.024373, -0.037990, 0.054404,
            -0.097094, 0.011804, 0.052895, 0.062653, 0.044074, -0.015194, 0.071273, -0.094809,
            -0.020924, -0.037423, 0.061018, 0.087894, -0.026445, 0.020124, -0.014453, -0.028453,
            0.040000, 0.023529, -0.048088, 0.021826, 0.090324, -0.069933, -0.015950, -0.088769,
            -0.025241, 0.094583, 0.065040, -0.094325, -0.089550, 0.004018, -0.011437, 0.075329,
            -0.027571, 0.075842, -0.059141, -0.018220, -0.002142, -0.051471, 0.004243, -0.021548,
            -0.047448, -0.078700, -0.000161, 0.093064, 0.038816, -0.076484, -0.050513, 0.065683,
            -0.037275, -0.009266, -0.047020, 0.047679, -0.067630, -0.094784, -0.073481, -0.086290,
            0.009615, 0.015504, -0.031652, 0.041448, 0.009380, 0.024164, 0.052154, 0.020222,
            0.099283, 0.073577, -0.056572, 0.074669, -0.010704, -0.005277, 0.009065, -0.087500,
            0.065291, 0.065709, 0.096951, 0.018232, -0.027296, -0.014712, -0.032781, -0.090429,
            0.033991, 0.014919, -0.030845, -0.058302, -0.022343, 0.027650, 0.075249, 0.040781,
            0.075386, 0.080539, -0.049182, -0.002081, -0.084047, -0.057529, 0.038705, 0.037362,
            0.044306, 0.085547, 0.040618, -0.048995, -0.020671, 0.007359, 0.050975, -0.051450,
            -0.041813, 0.038277, 0.033691, 0.094821, 0.082471, -0.046408, 0.034365, -0.061647,
            -0.004231, -0.008458, 0.022578, 0.087640, -0.078235, 0.092031, 0.069758, -0.048391,
            -0.027000, -0.021939, 0.062705, -0.025672, 0.035158, -0.039612, 0.080740, -0.016058,
            0.095602, -0.074444, -0.029858, 0.064632, 0.033258, -0.050635, -0.081886, -0.075498,
            -0.083558, -0.055263, -0.060592, -0.028861, 0.086646, -0.026543, -0.061809, 0.048890,
            -0.058857, -0.050048, -0.016525, -0.012309, -0.061668, -0.079529, -0.097613, -0.042414,
            0.050667, 0.008022, 0.079715, 0.017344, -0.075021, 0.009420, -0.025622, 0.056623,
            0.027208, 0.098993, -0.089659, -0.047595, -0.084879, 0.008898, -0.063744, -0.041524,
            0.095458, -0.033504, 0.049201, 0.064293, 0.072525, 0.081802, 0.061686, 0.009910,
            0.084375, 0.096774, -0.068981, 0.020883, 0.068805, -0.067606, 0.039246, 0.062437,
            0.003120, -0.028909, -0.008270, 0.086988, -0.079178, 0.039674, -0.054130, 0.067484,
            -0.041099, 0.022106, 0.063193, -0.012993, -0.016001, 0.090274, 0.095287, 0.053166,
            0.002196, -0.000813, -0.062373, 0.067939, 0.044445, 0.079746, -0.065284, 0.024801,
            0.019554, -0.029778, -0.010434, 0.020032, -0.027221, 0.094742, -0.089742, -0.014364,
            -0.047203, 0.075999, -0.036013, -0.022987, 0.086022, -0.087372, -0.033403, 0.085918,
            0.070263, -0.028480, -0.018817, -0.047658, 0.019789, 0.029315, 0.067555, 0.043896,
            0.050441, -0.015925, -0.071468, 0.055216, -0.001737, -0.065435, 0.008947, 0.098944,
            -0.008431, -0.083535, 0.060393, 0.074360, -0.063905, -0.087464, -0.071550, 0.041116,
            -0.041041, -0.098231, -0.007176, -0.050808, -0.029601, -0.039468, -0.033555, -0.065110,
            -0.020306, -0.063022, -0.013215, -0.062305, 0.006556, 0.070907, -0.040715, -0.013149,
            -0.017007, -0.033099, -0.069271, 0.025329, -0.008489, 0.005145, -0.025706, -0.063668,
            -0.036479, 0.051418, 0.069230, -0.031928, 0.052427, -0.057357, -0.005670, -0.090267,
            -0.013976, 0.029143, 0.060964, 0.034995, 0.047177, -0.021455, 0.098358, -0.082950,
            -0.022532, -0.015693, 0.067193, 0.019501, 0.097652, -0.067456, 0.033223, -0.084081,
            0.031709, -0.045876, -0.043862, 0.010985, 0.009939, 0.098570, 0.080159, 0.005154,
            0.034137, 0.020581, -0.021452, -0.072299, -0.080331, 0.009644, -0.038910, 0.091484,
            -0.043863, 0.027193, -0.048311, 0.099597, 0.042911, 0.054925, 0.027134, -0.014484,
            -0.034152, -0.044683, -0.086039, -0.016758, -0.040265, 0.090872, 0.095325, -0.065913,
            0.007399, -0.080703, 0.048984, 0.075559, -0.056817, -0.070636, -0.017903, 0.080839,
            0.070570, -0.052267, 0.004721, -0.076374, 0.093503, 0.043689, -0.040411, 0.045579,
            0.007211, -0.059484, -0.096653, 0.061389, -0.074728, 0.066111, -0.029864, 0.057180,
            0.041254, 0.062200, -0.044445, 0.049278, 0.023671, 0.001724, -0.000762, 0.001630,
            0.058275, 0.059497, -0.036960, -0.002140, -0.022426, -0.036003, 0.043955, 0.092420,
            0.050522, -0.025123, 0.059834, 0.096875, 0.026252, 0.088223, 0.032290, -0.020019,
            0.033469, 0.024060, 0.029376, -0.024498, -0.073244, -0.092515, -0.064453, 0.060444,
            -0.023865, -0.081067, 0.076095, -0.091857, 0.017893, 0.086262, -0.049740, -0.001790,
            0.099288, 0.085533, 0.071682, -0.017959, 0.059071, 0.040182, 0.005437, 0.092273,
            0.028407, 0.062524, -0.075690, 0.030116, -0.092109, 0.016710, 0.026885, 0.059318,
            0.053188, -0.044560, 0.059463, -0.099787, 0.083200, 0.048042, -0.015351, 0.029379,
            0.055107, -0.080196, 0.064902, 0.029849, 0.097326, 0.004405, 0.009986, 0.022997,
            0.080881, -0.083297, 0.055292, 0.057554, -0.008063, -0.067989, 0.077214, 0.060851,
            -0.023171, 0.017378, -0.030625, -0.045063, 0.099334, 0.040251, 0.006365, 0.063568,
            0.066919, 0.053502, 0.044516, -0.074795, -0.040053, 0.083120, 0.030430, 0.027474,
            -0.065562, -0.047810, -0.037540, 0.047624, -0.040196, 0.076445, -0.070950, -0.017176,
            0.084893, -0.041967, 0.001793, -0.029382, 0.035295, -0.050455, -0.075561, -0.037638,
            0.040330, -0.025931, -0.076687, -0.020868, -0.043635, 0.083136, -0.014537, 0.044376,
            0.096170, -0.070012, 0.038405, 0.013604, 0.084738, 0.003092, 0.029331, -0.097894,
            0.095918, -0.075956, 0.074122, -0.065317, 0.021448, 0.001150, 0.074684, -0.093109,
            0.083730, 0.096990, 0.029029, 0.092413, 0.080785, 0.004869, 0.056407, 0.006873,
            0.024476, -0.069406, -0.027064, -0.054434, -0.058869, -0.063698, -0.042657, 0.017442,
            -0.002998, 0.045007, 0.043771, -0.028083, 0.070535, -0.069573, -0.056651, 0.096806,
            0.048162, 0.048397, -0.094117, 0.072451, -0.081886, 0.015125, -0.050361, 0.037802,
            -0.055716, -0.085690, -0.090091, -0.076686, -0.037411, 0.003278, -0.035308, -0.033266,
            0.017455, 0.084344, -0.030756, 0.080907, -0.041237, 0.079696, -0.043685, -0.048675,
            -0.027662, -0.022705, -0.063237, 0.068213, -0.084157, -0.001448, -0.095223, -0.062478,
            0.013923, -0.015158, -0.019630, 0.009080, -0.028401, -0.088124, -0.003936, 0.027967,
            0.071305, -0.059547, -0.068340, -0.080981, 0.044917, -0.088378, 0.078342, 0.037014,
            0.068710, -0.036364, 0.063984, -0.044747, -0.012295, 0.076557, -0.053170, -0.024153,
            0.011196, 0.032545, -0.013914, -0.079113, -0.023727, 0.039031, -0.077866, -0.082633,
            -0.030673, 0.021890, -0.035529, 0.032541, 0.013751, 0.074632, -0.009647, -0.054372,
            -0.033660, -0.076959, -0.035953, -0.049349, -0.079771, -0.094697, -0.086172, 0.044073,
            -0.041991, 0.083631, -0.042509, 0.043643, -0.061794, 0.095535, 0.023307, 0.013223,
            0.029982, 0.023641, -0.039189, 0.023893, -0.058457, 0.048479, 0.095862, 0.096644,
            -0.020272, -0.019570, 0.088308, -0.065167, 0.030123, 0.022210, 0.090016, -0.038759,
            0.080262, -0.011452, 0.096946, -0.099893, -0.057562, -0.021230, -0.098723, -0.009084,
            -0.079399, 0.062498, 0.084946, -0.006824, 0.058422, 0.049526, 0.039790, 0.012544,
            -0.065372, -0.088840, 0.040048, -0.067740, 0.077067, 0.020741, -0.050627, -0.042094,
            0.024399, -0.003751, 0.087552, -0.038012, -0.058947, -0.091390, 0.018698, -0.074549,
            -0.010140, 0.098308, -0.044061, 0.088501, 0.060867, -0.061295, 0.052963, -0.039556,
            0.053915, -0.058429, 0.026415, 0.056245, 0.073458, -0.037192, 0.058853, 0.071559,
            -0.025195, 0.069831, -0.021240, -0.056959, 0.035037, 0.028047, -0.090844, -0.041272,
            -0.024807, 0.065785, -0.006792, -0.095286, 0.015864, 0.026513, 0.016440, 0.080763,
            -0.098232, 0.010522, -0.054870, -0.044723, -0.064161, -0.023507, -0.093767, -0.092853,
            0.067447, 0.061291, -0.037487, -0.032680, -0.035610, 0.024707, -0.039730, 0.041244,
            -0.000251, 0.025438, -0.050216, -0.049419, 0.042896, 0.076279, -0.054216, 0.052166,
            -0.007207, 0.098605, -0.042941, 0.084946, 0.041101, 0.077171, -0.000563, 0.067770,
            0.001046, 0.001506, -0.017633, 0.028629, 0.096394, 0.080081, -0.054971, -0.041689,
            0.054428, -0.038041, -0.031998, 0.021311, -0.057258, -0.072377, 0.080386, -0.090472,
            -0.060346, 0.045166, -0.014512, 0.050163, -0.013651, 0.000183, -0.019315, 0.028281,
            0.027820, -0.013390, 0.067640, -0.061918, -0.025051, 0.007597, 0.049672, -0.034891,
            0.003918, -0.091810, 0.048456, 0.060010, -0.083052, -0.082544, 0.005284, 0.087347,
            0.037014, 0.076103, -0.081931, 0.071629, 0.092742, -0.080047, 0.022871, -0.019345,
            0.098818, -0.050747, -0.094364, -0.081703, 0.071115, 0.027502, -0.092923, -0.080370,
            0.053330, -0.051236, -0.068247, -0.042620, -0.039024, -0.061337, -0.043713, 0.013660,
            0.020406, 0.084402, -0.094992, -0.024311, -0.029590, -0.054677, 0.033674, -0.048191,
            -0.035478, -0.077584, -0.039161, 0.004914, -0.055376, -0.019224, -0.058894, 0.018553,
            0.035496, -0.059797, -0.041726, -0.014705, -0.052846, -0.043708, 0.048510, -0.016812,
            0.038976, 0.000939, 0.004146, -0.070330, 0.007202, 0.094403, 0.092940, 0.069946,
            -0.031474, -0.000550, 0.090653, -0.022768, 0.024144, 0.008689, 0.064084, -0.048580,
            0.019174, -0.088658, -0.023922, 0.045010, 0.010331, 0.092669, -0.071334, 0.044621,
            -0.004905, 0.093471, 0.086752, 0.088437, -0.066077, 0.099537, -0.062891, -0.046824,
            0.057555, -0.000009, 0.048409, 0.085406, -0.011323, 0.026622, 0.002597, 0.067002,
            0.051763, 0.005540, -0.073172, 0.030036, 0.061286, 0.007051, 0.046646, 0.024944,
            -0.049282, -0.051706, -0.017567, -0.086086, -0.002233, -0.024802, -0.084230, 0.012642,
            0.018697, 0.038493, -0.004078, -0.035534, -0.016837, -0.078900, 0.001701, 0.053440,
            0.001515, 0.010420, 0.084284, 0.081344, 0.090060, 0.081570, 0.053642, -0.002785,
            -0.094717, 0.011826, 0.056387, 0.003816, 0.053410, -0.075895, -0.002847, 0.098361,
            -0.090965, -0.014610, 0.044583, -0.095636, -0.038869, -0.043515, 0.066338, -0.015600,
            0.003053, 0.080516, 0.023023, -0.034102, 0.049312, -0.041663, 0.086382, -0.029716,
            0.088591, -0.030243, 0.054233, 0.088776, -0.036454, -0.007995, -0.048681, 0.051124,
            -0.034265, 0.074774, -0.064760, -0.034935, -0.023391, 0.082438, 0.053535, -0.004974,
            0.049888, -0.087883, -0.008250, 0.089302, -0.023294, -0.055553, -0.047121, 0.078651,
            0.014555, 0.024477, -0.046416, 0.050187, -0.001272, 0.099484, 0.063911, -0.086567,
            0.082312, -0.066566, -0.033312, 0.015126, -0.079990, -0.004938, -0.052013, -0.025751,
            -0.000578, -0.016342, 0.034690, -0.027740, 0.013153, -0.001084, -0.033086, -0.091244,
            0.051238, -0.083779, 0.011452, 0.061454, 0.091524, -0.052467, 0.084503, -0.053219,
            -0.059475, 0.014925, -0.039676, -0.064003, 0.027734, 0.081829, 0.059896, 0.035408,
            0.054449, 0.082144, -0.044315, 0.079127, 0.027038, 0.067483, -0.097037, 0.023274,
            -0.005005, -0.002313, -0.072857, -0.010836, -0.022559, -0.034076, 0.083156, -0.089030,
            0.010017, 0.044814, 0.019029, 0.068842, 0.012371, 0.062392, 0.047071, -0.066427,
            -0.085955, -0.049884, -0.057994, -0.074881, 0.021848, -0.077989, -0.072908, -0.034419,
            -0.094305, -0.062661, -0.022072, 0.006329, 0.042589, 0.088825, -0.019844, -0.059189,
            -0.062381, -0.072372, 0.064470, 0.088350, 0.024996, 0.024579, 0.032622, 0.067393,
            0.055907, 0.066557, 0.034047, 0.012559, -0.026238, 0.062100, 0.028731, 0.061984,
            0.095230, 0.074832, 0.092101, 0.075324, -0.024344, -0.037620, 0.021531, 0.062387,
            -0.083247, 0.030094, 0.045773, -0.048007, -0.050372, -0.045933, 0.098185, -0.004816,
            -0.056050, 0.049017, -0.068131, -0.096054, 0.066219, -0.000013, 0.008077, 0.057455,
            -0.013097, -0.008693, -0.017128, 0.052552, -0.019774, 0.006558, -0.067188, -0.043734,
            -0.083683, 0.069266, 0.083881, 0.040090, 0.037072, -0.046607, 0.032862, 0.041305,
            0.093512, -0.081500, -0.060242, -0.050095, 0.080670, -0.001179, 0.079400, -0.037110,
            -0.014757, 0.015375, -0.026623, 0.076530, 0.071119, -0.079962, -0.045054, 0.065692,
            0.063281, 0.001601, 0.081223, 0.010046, -0.096649, -0.033172, 0.070433, -0.021913,
            -0.074059, -0.022771, -0.059362, 0.041229, 0.031872, -0.045912, 0.026680, 0.069098,
            0.076251, 0.015027, 0.035896, -0.040967, -0.070692, 0.043998, 0.030371, 0.081655,
            -0.000983, 0.061010, 0.060549, 0.002997, -0.059222, -0.084264, 0.068479, -0.096219,
            0.044605, -0.041673, -0.009443, 0.053168, -0.099119, 0.070361, 0.005080, 0.062202,
            0.024056, 0.050707, -0.027634, -0.045180, 0.045104, -0.092740, -0.072180, 0.000676,
            0.087839, 0.024442, -0.043964, 0.057530, 0.046115, 0.038538, 0.095182, 0.078465,
            0.036422, -0.099779, 0.044484, 0.055552, -0.058753, 0.059060, 0.099483, 0.031570,
            0.068426, 0.058664, -0.016760, -0.088839, -0.041008, 0.056382, -0.076534, 0.086598,
            0.018901, -0.034730, 0.076023, -0.083076, -0.007088, -0.099371, 0.047579, 0.013247,
            0.066064, -0.051156, 0.073895, -0.099658, 0.083172, 0.042274, 0.028441, -0.076523,
            -0.052913, -0.016717, 0.052786, 0.072287, 0.092200, 0.022265, -0.024693, -0.085672,
            0.000043, 0.014032, -0.040300, 0.040808, -0.069005, 0.015772, -0.000980, -0.071839,
            -0.092306, -0.080266, 0.095388, -0.036250, 0.025055, -0.031708, 0.018861, -0.083422,
            0.046365, 0.080398, -0.005025, 0.006215, -0.082502, -0.060676, -0.078198, 0.091933,
            -0.004889, 0.032647, -0.012521, 0.047182, 0.073706, -0.074120, -0.067450, -0.028696,
            0.006578, -0.018776, 0.042935, 0.045444, 0.021327, -0.086065, -0.061665, 0.051691,
            -0.037454, -0.068563, -0.022237, 0.076972, 0.025335, -0.041082, 0.084563, -0.017208,
            0.082088, 0.097530, -0.075883, -0.021167, 0.076450, -0.095636, 0.070757, 0.080873,
            0.093077, -0.037754, 0.083268, 0.023651, 0.081962, 0.040466, 0.003179, -0.097886,
            -0.061277, 0.082192, 0.039357, -0.029640, 0.096586, -0.073444, 0.068064, 0.097920,
            -0.037196, 0.027877, 0.062682, -0.023699, 0.021776, 0.049059, -0.028647, -0.056785,
            0.030866, -0.086880, -0.087973, 0.001555, -0.048935, 0.028826, 0.019978, -0.080694,
            -0.057500, 0.097100, -0.007050, -0.013693, -0.045158, 0.037350, 0.037769, 0.038839,
            -0.036295, 0.050446, 0.064506, 0.093463, -0.095267, 0.036607, -0.035931, -0.028346,
            -0.073188, 0.028405, -0.079932, -0.058711, -0.004351, 0.036610, 0.044189, 0.062997,
            -0.098670, -0.047939, 0.016825, -0.071461, -0.054261, -0.040645, 0.049963, 0.043816,
            -0.038192, -0.050793, 0.020137, 0.029331, 0.089317, 0.071543, -0.027300, -0.007980,
            0.054150, 0.022303, 0.048551, 0.092190, 0.034488, -0.006802, -0.097685, -0.046165,
            0.046826, 0.090935, 0.033729, -0.069741, -0.080740, 0.096150, -0.030288, -0.025464,
            -0.019980, 0.057236, 0.089666, 0.034933, 0.065034, 0.087460, -0.096415, -0.039870,
            -0.052576, 0.050676, 0.071237, 0.013680, 0.067469, -0.093826, -0.018852, 0.010986,
            -0.008835, 0.076920, -0.039580, 0.081950, -0.028363, 0.005437, 0.017024, 0.084475,
            -0.019743, -0.006924, 0.071195, 0.097943, -0.055693, 0.028138, -0.001751, -0.094335,
            -0.085933, -0.025865, -0.022435, 0.015388, -0.050371, 0.022445, 0.098234, 0.010520,
            0.064919, -0.062990, 0.087876, -0.025287, 0.098825, -0.034623, 0.012784, 0.017349,
            0.025275, 0.005963, -0.019149, -0.028058, 0.071384, 0.090357, 0.094428, -0.055711,
            0.082483, 0.086636, 0.050695, -0.008513, -0.068111, -0.030016, -0.094992, 0.058105,
            0.068742, 0.039332, -0.096910, 0.033836, -0.051511, -0.043670, -0.078698, 0.084840,
            0.053713, 0.089156, 0.092702, -0.048344, 0.073359, 0.076582, 0.025675, -0.097574,
            -0.015581, -0.098489, -0.010246, 0.084216, -0.002882, -0.078512, 0.093716, -0.012727,
            0.055436, -0.020535, -0.063104, 0.031423, 0.016375, 0.032150, -0.003833, 0.083681,
            -0.089262, -0.076963, -0.083221, -0.027655, 0.024998, -0.036265, -0.025426, -0.048973,
            -0.039225, -0.089541, 0.007044, 0.031749, -0.037758, 0.006331, 0.084476, 0.064500,
            -0.096261, -0.078672, 0.042117, -0.004213, 0.081977, 0.071676, -0.070881, -0.085372,
            0.016620, 0.031125, -0.022812, 0.065741, 0.096019, -0.011475, 0.033811, -0.093636,
            -0.011784, -0.078644, 0.081822, 0.031127, 0.052539, -0.039282, 0.043454, 0.037319,
            -0.045089, -0.018133, 0.055548, 0.024614, -0.004032, -0.001805, -0.069021, -0.014218,
            0.010234, 0.041046, -0.036478, 0.036188, 0.015297, 0.091309, 0.038869, -0.098724,
            -0.024404, -0.006989, -0.043796, -0.035942, 0.004538, -0.002328, -0.026240, -0.041120,
            -0.044945, 0.033739, 0.032168, 0.018222, 0.071628, 0.008478, 0.092803, -0.020651,
            0.000957, -0.052047, -0.079263, -0.043082, 0.023751, 0.007493, 0.041223, 0.031994,
            -0.097364, 0.083049, -0.027443, -0.077378, 0.038997, 0.026217, -0.056479, 0.001318,
            0.084204, 0.026657, -0.028088, -0.023970, 0.032391, 0.095157, -0.027796, 0.088385,
            -0.040028, 0.070264, 0.009645, -0.042282, 0.083557, 0.073033, -0.018923, -0.095219,
            -0.051328, -0.005699, -0.085732, -0.076311, -0.008112, -0.024600, 0.003300, -0.051319,
            -0.050397, 0.032515, 0.004981, 0.065614, -0.055131, -0.042802, -0.046136, -0.065392,
            0.054440, -0.016347, 0.099462, 0.057512, -0.048920, -0.025157, 0.043552, -0.070959,
            0.087177, 0.016168, 0.028689, 0.086068, -0.086126, -0.035255, -0.044673, -0.065474,
            0.065257, 0.065495, 0.075798, 0.095482, -0.049056, 0.000719, 0.025694, -0.030795,
            0.056347, 0.097521, -0.098524, -0.020628, 0.059008, -0.071017, -0.017810, 0.064315,
            0.024700, -0.053066, -0.025173, -0.000068, 0.094738, -0.065172, -0.007672, -0.020754,
            0.099406, 0.075563, -0.014162, -0.044559, 0.098514, -0.011871, -0.027526, 0.079204,
            0.029688, 0.010875, 0.090019, -0.050217, 0.079417, 0.049989, -0.063947, 0.091419,
            0.034652, -0.072625, 0.085649, -0.081548, 0.013517, 0.045792, 0.053254, -0.088070,
            0.087576, 0.087554, 0.058565, 0.034241, -0.064258, 0.075100, 0.006154, -0.014302,
            -0.031589, 0.006430, -0.020856, -0.074828, -0.065469, -0.035451, -0.018115, -0.051647,
            -0.028958, -0.076601, -0.070909, 0.059826, -0.031145, -0.031884, 0.000026, 0.015760,
            -0.064863, 0.045618, 0.042913, 0.044462, -0.072747, 0.090101, 0.054116, 0.028485,
            0.055103, -0.056991, 0.038506, 0.089558, 0.041847, -0.004452, -0.093338, -0.036349,
            -0.066990, 0.097510, -0.018157, -0.032742, -0.099514, -0.085108, -0.036204, 0.038384,
            -0.015901, 0.052441, 0.097252, -0.062100, 0.031388, 0.007504, -0.088220, -0.019987,
            0.012392, -0.032241, -0.072769, 0.092347, 0.041683, -0.090491, -0.095571, 0.023940,
            -0.050291, 0.020356, 0.038575, 0.013008, 0.058031, 0.043418, -0.012211, 0.022840,
            -0.025987, -0.052228, -0.098145, 0.042681, 0.078616, -0.028741, 0.061918, 0.079604,
            0.014044, 0.043553, 0.005505, 0.058087, 0.048550, 0.067374, 0.091043, -0.037482,
            0.031193, 0.038724, -0.079082, -0.086528, -0.022987, 0.044607, 0.076272, -0.039266,
            -0.041298, -0.088655, 0.059997, -0.048651, 0.033709, 0.036230, -0.007736, 0.088548,
            0.070740, -0.017503, -0.002180, 0.065847, -0.058101, -0.084535, 0.062935, -0.052178,
            0.009962, 0.007996, 0.072628, 0.014033, 0.010327, 0.020147, -0.043010, -0.096289,
            0.085428, -0.080760, -0.072566, -0.024989, 0.065373, -0.039978, 0.077261, 0.035035,
            0.064572, -0.037076, 0.047085, 0.022281, 0.088680, -0.045837, 0.086385, -0.063418,
            -0.069967, 0.093413, 0.048852, 0.057413, -0.049362, 0.083099, -0.026647, 0.084336,
            0.064654, -0.086065, -0.027783, 0.099459, 0.063972, 0.009478, -0.009461, 0.038542,
            -0.029762, -0.030415, -0.050775, -0.004342, -0.052699, -0.053474, 0.071848, 0.067790,
            -0.050114, -0.017212, -0.048652, 0.044930, -0.066398, -0.052828, 0.035656, 0.013356,
            0.030026, -0.012400, 0.098207, -0.015952, 0.086242, 0.009477, 0.093936, -0.060873,
            0.060153, -0.029527, 0.053910, -0.080989, 0.060212, -0.029073, -0.042972, -0.036790,
            0.071053, -0.017101, -0.012405, 0.091619, 0.074508, 0.009121, -0.030073, 0.030623,
            -0.026480, 0.087300, 0.035383, 0.066893, 0.053830, -0.022918, -0.083689, -0.031196,
            -0.037979, 0.035028, 0.039916, 0.089676, 0.061410, 0.092045, 0.023056, -0.076259,
            -0.009913, -0.006691, -0.038388, -0.057517, -0.020293, 0.053096, 0.007452, 0.089102,
            -0.074523, 0.003046, 0.061696, 0.021358, 0.055756, 0.079694, -0.095530, -0.079700,
            0.083264, -0.017301, 0.093167, -0.050537, -0.012121, -0.006926, 0.078557, -0.078797,
            -0.011698, 0.040845, -0.066633, -0.001152, 0.053226, 0.013937, 0.054834, 0.033719,
            0.028372, -0.021512, 0.069174, -0.044345, -0.080337, 0.013615, 0.065505, 0.096184,
            -0.026355, 0.039016, -0.006019, -0.045209, -0.035098, -0.028578, -0.038592, 0.043079,
            0.081770, -0.061102, -0.005739, 0.048001, 0.050445, 0.012949, -0.057362, -0.054310,
            -0.091176, -0.097570, -0.080575, -0.047863, 0.019008, 0.081165, -0.031304, 0.090352,
            0.053313, 0.035076, 0.042035, -0.016015, -0.025412, -0.033752, -0.093675, -0.031297,
            -0.054753, -0.085078, -0.075362, 0.086923, 0.005325, 0.079576, 0.061561, 0.037314,
            -0.079366, -0.036410, 0.037063, 0.024169, -0.082642, 0.096076, 0.081103, 0.039627,
            -0.060796, 0.045058, 0.059897, 0.044413, -0.051474, 0.001161, -0.078305, -0.052921,
            -0.058309, 0.081302, -0.073900, 0.034785, 0.000957, -0.005519, 0.026133, -0.073193,
            0.046011, 0.071369, -0.095424, 0.049693, -0.011345, 0.018391, -0.021506, 0.024744,
            0.056143, -0.034352, -0.087808, -0.018960, 0.067549, -0.093387, 0.073405, -0.002127,
            -0.046973, 0.015028, 0.050035, -0.089848, -0.085572, -0.099615, 0.059486, -0.099518,
            0.061977, 0.059696, 0.020752, 0.091465, -0.017641, -0.003517, 0.041937, -0.068979,
            0.083071, -0.042857, -0.082060, 0.034675, -0.041968, 0.084296, -0.014946, 0.068762,
            -0.025913, -0.060133, -0.022178, -0.014470, 0.037810, 0.018317, -0.045536, -0.077333,
            -0.098608, 0.098411, 0.020912, -0.001120, 0.075575, 0.020659, 0.015962, -0.061853,
            -0.072386, -0.086056, -0.006032, -0.010780, -0.063659, 0.064451, -0.009118, 0.034199,
            -0.031069, -0.037190, -0.051486, -0.022558, -0.026662, -0.054874, 0.090928, 0.077800,
            0.093655, 0.091890, -0.036755, 0.043814, -0.039442, 0.095411, -0.030435, -0.074566,
            -0.049474, -0.000602, -0.085250, 0.011617, 0.000327, -0.089223, 0.082593, -0.030785,
            0.092083, -0.007106, 0.064359, 0.099144, -0.068883, 0.048177, 0.069881, 0.011644,
            -0.091997, -0.030299, 0.025535, -0.076025, 0.090129, -0.058302, -0.039634, 0.043991,
            -0.094562, 0.061093, -0.001802, 0.012992, -0.087037, -0.086322, -0.073218, -0.099889,
            0.092190, -0.033378, 0.022102, 0.009891, -0.048985, -0.077945, -0.081189, 0.034876,
            -0.021664, 0.048040, 0.068912, -0.007731, 0.009359, -0.014277, -0.076052, 0.074094,
            -0.077763, -0.011077, 0.098604, -0.033861, 0.022418, 0.066260, -0.070503, 0.018746,
            -0.040282, -0.059845, -0.088856, 0.060035, -0.041204, 0.086834, 0.096716, 0.003654,
            0.082364, -0.023630, 0.057343, -0.005632, -0.067066, -0.050498, -0.063687, 0.026567,
            -0.002358, 0.078926, 0.025487, -0.022607, -0.012779, 0.064520, -0.053492, 0.098699,
            -0.006358, 0.019827, 0.092444, -0.088213, 0.067682, 0.081469, 0.082871, 0.034886,
            -0.031076, -0.068008, 0.073415, 0.023666, 0.077266, 0.061065, -0.053637, 0.066622,
            0.008350, -0.091655, 0.008731, -0.049820, -0.036601, -0.034716, -0.093273, -0.066703,
            0.082077, -0.053236, -0.018674, -0.013936, 0.079660, 0.049668, -0.042180, 0.004859,
            -0.044891, 0.008350, -0.050510, -0.057821, -0.026963, -0.075177, -0.042340, 0.035995,
            -0.017521, -0.080819, -0.067792, -0.059844, 0.061339, -0.068533, -0.006617, 0.056536,
            -0.090624, 0.018463, 0.038461, 0.089685, 0.038674, -0.032605, 0.098862, -0.011850,
            -0.002046, 0.030840, -0.075218, 0.083555, 0.028102, 0.068999, 0.028993, -0.083587,
            0.040419, -0.007711, 0.071303, 0.038091, 0.017304, 0.041897, 0.064999, 0.031469,
            0.085815, 0.095790, -0.079028, -0.069071, -0.075912, -0.078244, 0.052640, 0.066142,
            0.006346, -0.018734, -0.089564, -0.096411, 0.025079, 0.024205, 0.035484, -0.006064,
            0.025434, 0.025514, -0.043025, 0.083327, -0.005073, -0.001244, -0.064468, 0.073705,
            0.087869, 0.022587, -0.063863, -0.059035, -0.097664, -0.007767, 0.048673, -0.055344,
            -0.050814, -0.058202, 0.017938, 0.048850, -0.074465, -0.051373, 0.040531, 0.073196,
            0.063792, -0.023648, -0.009660, -0.059123, 0.030821, 0.076070, -0.005964, -0.067490,
            0.097294, 0.040319, 0.035000, 0.087765, -0.028542, -0.005571, 0.021626, 0.078214,
            0.046298, -0.012729, 0.077429, 0.003242, -0.044089, 0.003611, 0.098550, -0.090376,
            0.087010, -0.076949, -0.062317, -0.087312, 0.066252, 0.017161, -0.067016, -0.037701,
            0.068295, 0.095890, 0.004431, 0.026555, -0.036462, 0.040479, 0.028535, -0.086887,
            0.091709, 0.074717, 0.079970, 0.014244, -0.023348, 0.074697, -0.071453, -0.086636,
            0.006467, 0.033412, 0.022821, 0.009795, 0.023069, -0.014492, -0.059643, -0.089611,
            -0.021490, 0.046025, -0.045269, -0.045029, -0.018833, 0.083531, 0.055512, -0.048313,
            -0.006910, -0.095807, 0.026077, -0.045802, 0.040512, 0.020973, 0.010214, 0.001123,
            0.070177, 0.003994, 0.056815, 0.045943, 0.009486, -0.006463, -0.077922, 0.001088,
            -0.089279, -0.011791, -0.055469, 0.086818, -0.069959, -0.062032, 0.042258, -0.074396,
            -0.026623, -0.053093, 0.008489, 0.063467, 0.099853, -0.006175, -0.067323, 0.006045,
            0.026285, 0.026406, 0.073496, -0.036935, 0.091842, -0.083706, 0.052348, -0.054361,
            -0.074721, 0.085241, 0.076630, 0.018068, 0.001204, -0.055578, -0.060468, -0.003608,
            -0.005020, 0.091953, 0.023098, -0.031763, 0.051291, -0.064745, 0.030155, 0.060865,
            -0.024593, 0.026308, -0.080103, 0.090713, 0.083646, 0.083636, 0.026219, 0.067532,
            0.068862, -0.006405, 0.005967, -0.086749, 0.044187, 0.001064, -0.063768, 0.069698,
            -0.015977, 0.002313, -0.007811, 0.095125, 0.056459, 0.098916, -0.092159, -0.084796,
            0.031016, -0.027084, -0.091697, 0.015837, -0.038433, -0.019430, 0.065958, -0.013946,
            0.087779, -0.025531, -0.013777, 0.083952, -0.076833, 0.021248, -0.045080, -0.060348,
            -0.071040, 0.089681, -0.064563, 0.094192, -0.071543, 0.085517, -0.087834, -0.091831,
            0.086145, 0.042526, 0.063427, -0.094552, 0.052694, 0.081075, -0.070227, -0.026084,
            0.066787, 0.006296, 0.036609, 0.022911, -0.010159, 0.014393, 0.038261, 0.061362,
            -0.066785, 0.061909, -0.077739, 0.059574, 0.017381, 0.012798, 0.053737, -0.054651,
            0.015750, -0.013459, 0.015378, 0.070563, -0.028876, 0.065967, 0.057375, -0.017849,
            -0.096232, -0.063370, 0.031657, -0.062299, -0.038346, 0.025054, 0.022896, 0.017609,
            -0.057941, 0.048754, 0.026414, -0.056583, 0.035775, -0.023074, -0.015533, 0.013313,
            0.042670, -0.005348, -0.043542, 0.025366, 0.019491, -0.012419, 0.093661, -0.034381,
            -0.099783, -0.059062, 0.026230, 0.095501, -0.081843, -0.028222, -0.042856, 0.060596,
            -0.062756, 0.043863, -0.078628, -0.079220, -0.024451, 0.040603, -0.058661, -0.032532,
            -0.028081, -0.058254, 0.003667, 0.075389, -0.028367, -0.098289, 0.086589, -0.065562,
            -0.031967, -0.002231, -0.097310, 0.095155, -0.069145, -0.042808, -0.024238, 0.008972,
            0.086664, 0.048318, 0.071396, 0.066292, -0.025579, -0.041218, 0.018414, -0.095263,
            -0.020359, -0.012004, -0.031694, 0.005943, -0.038818, -0.075460, -0.052312, 0.079018,
            -0.099687, -0.025487, 0.089278, -0.010593, 0.032151, 0.033702, 0.019670, 0.080597,
            0.088272, -0.019039, -0.013735, 0.097803, -0.098819, -0.093764, 0.067511, -0.066877,
            0.031687, -0.033947, -0.075951, 0.089181, 0.035047, -0.049012, 0.014657, -0.020457,
            -0.043881, 0.056021, -0.023046, -0.016977, -0.087700, 0.001842, 0.061301, 0.043027,
            0.034569, 0.034392, 0.072204, -0.019611, -0.092358, -0.055171, 0.024310, 0.070861,
            0.079348, -0.016554, -0.062530, 0.092298, -0.099611, 0.099544, -0.086593, -0.018871,
            0.045156, 0.015542, 0.084857, 0.004582, -0.092932, 0.057625, -0.018700, 0.088385,
            0.063407, 0.008143, 0.050118, -0.048371, -0.092411, -0.072866, -0.032308, -0.070872,
            -0.088625, 0.052506, -0.030727, 0.059736, -0.012828, -0.043693, -0.094232, -0.091070,
            0.066966, 0.035862, -0.006381, -0.023925, -0.088341, 0.091996, 0.098873, -0.058997,
            0.014112, -0.082518, -0.072573, -0.002262, -0.062303, -0.094893, -0.051390, -0.090768,
            0.030325, -0.067353, 0.080745, 0.056776, 0.093813, -0.087045, 0.055525, 0.079404,
            0.009732, -0.008023, 0.048696, 0.026732, -0.037764, -0.097828, -0.063887, 0.049224,
            0.073474, 0.007003, -0.045176, 0.046093, -0.019789, 0.034059, -0.067954, 0.054766,
            -0.045858, 0.018410, 0.019384, 0.098473, 0.091490, -0.082581, 0.049688, -0.080169,
            -0.048901, 0.039467, -0.068424, 0.085773, 0.037871, 0.008553, 0.056031, -0.085212,
            0.016503, 0.037275, -0.046009, -0.031096, 0.040576, 0.084735, -0.039010, 0.060398,
            0.056075, -0.058540, -0.079512, -0.011786, 0.062073, -0.083397, -0.069507, 0.072894,
            -0.057364, -0.071420, -0.004262, 0.097884, 0.065979, 0.051607, 0.012797, -0.050116,
            -0.077936, 0.053143, 0.029648, 0.026031, -0.036968, -0.097596, 0.013682, -0.017226,
            -0.007568, 0.009496, -0.058470, -0.062307, -0.029014, 0.038821, -0.062625, 0.093234,
            -0.098441, -0.062084, 0.028670, -0.097301, -0.055532, 0.051618, -0.067793, 0.032037,
            -0.053299, -0.047264, 0.017351, 0.009684, 0.090702, 0.020805, 0.017410, -0.051939,
            0.086915, -0.001313, 0.059837, -0.080047, 0.052319, 0.081244, 0.064802, 0.083548,
            -0.045456, -0.071746, 0.036002, 0.043331, -0.060762, -0.032114, -0.001283, -0.008389,
            0.047830, 0.071028, 0.028905, -0.039590, -0.062369, 0.086368, 0.011205, -0.052320,
            -0.074612, -0.067940, 0.093393, -0.062981, -0.097357, -0.074922, -0.047485, -0.003055,
            -0.068569, 0.044038, -0.016412, -0.041629, -0.060670, -0.031663, 0.097155, 0.005038,
            0.028177, -0.096873, 0.061061, 0.015796, -0.011064, 0.001806, -0.046203, -0.077634,
            -0.050049, 0.031374, 0.084464, -0.069638, 0.046006, -0.048608, 0.075398, 0.012202,
            -0.090858, -0.086262, 0.024060, 0.055851, 0.029517, 0.027519, -0.015697, -0.049896,
            0.017248, -0.079273, 0.049900, -0.038082, -0.089644, 0.010866, 0.038440, -0.076314,
            -0.063121, -0.017912, -0.041367, 0.081023, 0.015232, -0.031219, -0.077920, -0.073953,
            0.034208, -0.023930, 0.014048, 0.018109, 0.069710, 0.000695, 0.044676, -0.093762,
            0.070441, -0.085437, -0.028648, -0.099714, 0.051007, -0.026388, -0.098232, -0.055251,
            0.054807, 0.067354, 0.070070, -0.095270, -0.041093, 0.016154, -0.006259, -0.050762,
            0.097331, -0.062148, 0.065266, 0.064281, -0.085834, -0.066240, -0.067231, 0.073032,
            -0.087654, 0.001204, -0.030553, -0.023239, -0.059096, -0.086258, -0.085942, -0.010655,
            0.005698, 0.025905, 0.032796, -0.042312, 0.054684, 0.058846, -0.079015, 0.085466,
            -0.010833, -0.005451, -0.095154, 0.053090, 0.040425, 0.025388, 0.043969, -0.087943,
            0.099362, 0.066888, -0.029453, 0.052679, -0.035653, -0.031756, -0.006080, 0.061758,
            0.054953, -0.034507, -0.082752, 0.001192, -0.042468, -0.006472, 0.050797, 0.066754,
            0.005281, 0.040974, 0.018338, -0.045514, 0.059426, -0.070124, -0.007544, -0.043433,
            0.006765, 0.082968, 0.031646, -0.006570, -0.093213, -0.071298, -0.020749, -0.048055,
            0.019708, -0.098619, 0.093084, -0.030365, 0.059582, 0.071775, 0.081772, -0.016566,
            0.003466, 0.051049, -0.074515, -0.007107, 0.042063, -0.002259, -0.071181, 0.014867,
            -0.020541, 0.061342, -0.069680, 0.008356, -0.013579, -0.001831, 0.090504, 0.000818,
            -0.003951, -0.094782, -0.048286, -0.080660, 0.006912, -0.021779, 0.029355, 0.081186,
            0.021611, -0.071060, -0.060816, -0.045266, -0.055878, 0.016610, -0.095817, -0.076626,
            -0.017460, -0.068441, 0.063026, 0.032912, 0.077577, 0.045960, -0.026643, -0.014918,
            0.046867, -0.090841, 0.024179, -0.091329, 0.042795, -0.032469, 0.033021, 0.038541,
            0.003904, 0.012459, -0.036614, 0.003258, 0.057680, 0.082205, -0.028125, 0.023134,
            0.023973, 0.040118, 0.034568, 0.033232, -0.072494, -0.017080, -0.049821, 0.082889,
            -0.038178, 0.043659, 0.026542, 0.035320, 0.023985, -0.009055, -0.032487, 0.097995,
            0.023944, 0.074913, 0.032802, 0.096922, 0.027422, -0.033317, 0.016691, -0.073596,
            0.059403, -0.071014, 0.054747, 0.035055, -0.099374, -0.034436, -0.021050, -0.035784,
            0.026769, -0.052109, 0.085494, -0.016564, 0.049399, -0.045570, -0.058448, 0.078945,
            -0.091354, -0.004062, -0.051601, -0.040548, -0.007546, 0.029355, -0.010612, 0.014323,
            0.090247, 0.017764, 0.064161, -0.026431, -0.023252, -0.081166, -0.052078, -0.036537,
            0.022069, -0.022198, -0.088085, -0.034171, 0.088208, -0.043129, -0.072092, 0.045827,
            -0.030665, -0.023148, -0.082259, -0.075642, 0.021368, 0.061593, -0.049656, -0.032262,
            0.034427, 0.091027, -0.022942, -0.097003, 0.069602, -0.088332, -0.050631, -0.068604,
            0.045831, 0.057689, 0.093095, 0.007911, -0.067134, 0.069840, 0.072213, 0.030008,
            0.029105, -0.075859, -0.022376, 0.093807, 0.015396, -0.097563, 0.096006, 0.066146,
            0.068877, 0.054880, 0.045584, 0.093404, 0.039087, -0.092555, -0.050878, 0.032053,
            -0.086876, -0.074484, -0.075798, -0.064374, 0.041181, 0.067286, -0.056505, -0.049412,
            -0.022771, -0.035208, -0.027342, -0.012026, 0.093385, -0.097534, 0.020904, -0.040793,
            -0.065259, 0.026935, 0.058101, 0.089706, -0.052367, -0.007611, -0.078053, -0.055505,
            0.009407, 0.053447, -0.020032, 0.061485, -0.000126, -0.027199, 0.023311, 0.081187,
            0.040875, -0.037644, -0.046145, 0.094996, 0.078400, 0.069695, -0.059941, -0.097596,
            -0.070718, 0.035719, 0.095012, -0.043852, -0.099199, -0.066231, 0.037962, 0.069490,
            -0.019547, 0.002790, -0.072677, -0.045752, -0.066344, 0.065100, -0.019704, -0.041563,
            -0.010834, 0.045100, -0.055441, -0.014168, 0.042413, -0.011821, 0.032402, -0.067183,
            -0.005673, -0.048513, 0.092070, -0.099882, -0.050312, -0.039483, 0.044141, -0.044623,
            -0.063047, -0.005832, -0.032032, 0.097118, 0.061512, 0.078214, -0.083785, -0.025792,
            -0.076640, 0.090237, 0.059304, -0.078315, -0.073915, 0.049408, -0.050743, 0.031948,
            -0.087817, -0.079812, -0.034044, -0.078225, 0.061490, -0.076371, -0.032724, -0.097836,
            0.091755, 0.016320, -0.060760, -0.021904, 0.029982, -0.034221, 0.079810, -0.022999,
            0.093602, 0.054572, 0.070556, 0.002636, -0.028836, 0.042545, 0.062826, 0.013677,
            0.019957, 0.085984, -0.054292, -0.021568, 0.007609, -0.011737, -0.057474, 0.032067,
            0.019216, -0.060520, -0.005644, 0.011465, -0.010160, 0.030785, -0.091923, -0.013800,
            0.073785, 0.002765, -0.011823, -0.027342, -0.015438, -0.011764, -0.048732, -0.071438,
            -0.008551, 0.099159, 0.075412, 0.004852, 0.043308, 0.021992, -0.051301, 0.036102,
            0.045687, -0.006627, 0.059629, 0.039790, -0.047403, 0.029140, 0.045644, 0.078999,
            -0.068163, -0.092219, -0.039178, 0.008441, -0.078888, -0.024716, -0.073150, -0.075690,
            0.030353, 0.048828, 0.013474, 0.069369, -0.023965, 0.012935, 0.073751, 0.082834,
            0.028629, -0.085377, 0.024187, 0.072061, 0.034065, -0.098495, -0.032682, -0.051234,
            -0.090003, -0.083199, 0.064638, -0.064065, -0.060276, -0.043518, -0.010637, 0.059449,
            -0.098518, -0.044718, -0.016305, 0.027747, -0.055486, -0.027188, -0.053771, -0.022447,
            0.051696, -0.042282, -0.013439, -0.065505, 0.027933, -0.064225, 0.023178, -0.099558,
            0.039049, 0.035997, -0.075853, -0.012730, -0.059997, -0.047755, 0.046330, -0.057102,
            0.056457, 0.050542, 0.089660, -0.037421, -0.088690, 0.064732, -0.023530, -0.039366,
            -0.047389, 0.092997, -0.077808, 0.021720, 0.019873, -0.038987, 0.056893, 0.057857,
            0.067273, 0.026946, 0.026569, 0.028324, -0.031877, -0.079902, 0.002928, 0.009245,
            0.061784, 0.016847, 0.002796, -0.035433, 0.083712, 0.088418, 0.098356, -0.006399,
            -0.049085, -0.080959, -0.098410, -0.080034, -0.031977, 0.077913, -0.092182, -0.029765,
            0.040162, 0.046971, 0.004396, -0.022469, 0.004754, -0.062663, 0.049258, 0.058435,
            0.016771, 0.070186, 0.058882, -0.093671, -0.039313, -0.085140, 0.036421, -0.089717,
            0.023750, -0.081273, 0.051021, -0.068487, -0.090040, -0.040399, 0.033057, -0.042734,
            0.037334, 0.003194, -0.054391, -0.005519, 0.063444, -0.061258, 0.087318, 0.027789,
            0.051982, -0.019259, 0.074462, -0.074601, -0.040362, 0.062757, -0.053759, 0.083620,
            -0.036253, -0.094712, 0.094355, 0.081621, 0.041227, -0.085655, 0.021622, -0.023752,
            0.096291, 0.023212, -0.007894, -0.099725, -0.036684, 0.074910, -0.076862, 0.010090,
            -0.077644, -0.009039, -0.023198, -0.013230, -0.003609, -0.028599, -0.037050, 0.048400,
            0.014163, -0.049793, 0.051511, 0.098280, 0.054940, 0.044129, -0.017939, 0.074115,
            -0.038364, 0.049681, 0.091749, -0.011374, -0.053537, 0.030368, -0.013577, 0.057632,
            -0.043538, 0.069807, 0.070714, 0.051373, 0.015118, -0.035159, 0.071484, -0.016468,
            -0.034253, 0.077883, -0.024292, 0.045700, 0.042549, -0.038051, 0.066225, -0.007776,
            0.052648, -0.065625, 0.034552, 0.081453, -0.045224, 0.094150, -0.062203, 0.030963,
            0.031835, 0.000407, 0.035049, 0.027465, -0.044250, -0.081646, -0.072401, -0.002208,
            0.000484, -0.012270, -0.002333, -0.084742, -0.092714, -0.036552, -0.087617, 0.050456,
            -0.031284, -0.046604, 0.012016, 0.020257, 0.052072, 0.004094, 0.034393, 0.023501,
            0.086518, 0.028645, 0.037512, 0.002912, 0.043520, -0.004694, -0.048201, -0.057525,
            0.059292, 0.007924, 0.097702, 0.058130, 0.086836, -0.011731, 0.063669, 0.013402,
            0.021335, 0.029290, -0.060411, 0.089699, 0.036749, -0.021690, -0.074287, -0.055416,
            -0.008782, 0.018230, 0.038413, 0.098157, 0.071748, -0.003993, -0.018782, 0.081286,
            0.002017, 0.019093, -0.005118, 0.084906, 0.054718, 0.044349, 0.036871, -0.010976,
            0.032037, -0.070516, -0.065433, 0.032869, 0.022805, -0.065550, 0.040396, 0.044262,
            0.059007, -0.045936, -0.052643, -0.001690, -0.005394, -0.000312, -0.070552, 0.074671,
            0.099035, -0.042346, 0.075904, 0.025031, 0.029615, -0.007370, 0.090261, -0.024345,
            0.023600, 0.000092, 0.015098, -0.046167, -0.066433, 0.065246, -0.091053, -0.064402,
            0.093797, -0.046913, -0.008546, -0.089593, -0.023554, -0.038631, 0.041961, 0.053622,
            0.093167, -0.029528, 0.088969, 0.039133, -0.078481, -0.085061, -0.013431, 0.086973,
            -0.001284, -0.063761, 0.058132, -0.097669, 0.075762, -0.001187, 0.005552, -0.036219,
            0.014012, 0.003985, 0.078398, 0.086779, 0.042685, -0.012910, -0.000578, -0.064578,
            0.057831, -0.024366, -0.098725, -0.033662, -0.070527, -0.080481, 0.016450, 0.007289,
            -0.055224, -0.072396, 0.031151, 0.020650, 0.067380, 0.051532, -0.065232, 0.050059,
            -0.091712, -0.087770, 0.039831, 0.091209, 0.045567, -0.004788, -0.048000, 0.097386,
            -0.066301, 0.089974, -0.059651, -0.092349, 0.000796, 0.028182, -0.083151, -0.032005,
            -0.004961, -0.008181, -0.049014, 0.091789, -0.073177, 0.060960, 0.028089, -0.042381,
            0.042791, -0.084580, -0.069813, 0.053360, -0.051891, 0.047990, -0.094837, 0.017523,
            -0.094260, 0.046844, 0.092413, 0.057740, 0.024877, 0.067250, 0.088722, 0.054713,
            0.042071, 0.030793, 0.021665, 0.067164, 0.079074, -0.029468, 0.018784, 0.053782,
            -0.082544, -0.076438, -0.027571, 0.014569, 0.063743, 0.049926, 0.049364, -0.021946,
            -0.026880, -0.097334, -0.000577, -0.022488, -0.072180, 0.049261, 0.061233, 0.029698,
            -0.068558, -0.023968, -0.086851, -0.016194, 0.047879, 0.078914, -0.081334, -0.020216,
            -0.048904, 0.072151, 0.095318, -0.088740, -0.042685, 0.003376, -0.089476, -0.094080,
            -0.060494, 0.073276, 0.020816, -0.047229, -0.014115, 0.091743, -0.094584, 0.039161,
            -0.003393, -0.096681, 0.087039, -0.046141, 0.075166, 0.007455, 0.022934, -0.014486,
            -0.080602, 0.050019, -0.034397, 0.037535, 0.094286, -0.028836, 0.011854, 0.035110,
            -0.031459, 0.046150, -0.062714, 0.060681, -0.037620, -0.001670, -0.034242, 0.010872,
            0.070420, 0.019805, -0.035033, 0.093921, 0.091074, -0.041487, -0.059388, -0.070812,
            0.017501, 0.093428, 0.031788, 0.036837, -0.094410, 0.040198, 0.053596, -0.030662,
            -0.062298, -0.012139, -0.086008, 0.092007, -0.060625, 0.090003, -0.099207, 0.063747,
            -0.041799, -0.077012, 0.028124, 0.092069, -0.047671, 0.014038, -0.065490, 0.065160,
            0.090022, -0.027260, 0.095805, 0.051753, 0.013496, -0.043023, 0.068953, 0.005144,
            -0.042292, -0.098375, -0.096923, 0.099699, -0.052847, -0.074606, 0.063405, -0.028606,
            -0.074566, 0.020188, -0.014781, 0.038581, 0.092915, -0.025174, 0.056952, -0.027940,
            -0.032986, 0.033351, -0.022351, 0.026864, 0.022269, -0.080084, -0.023438, 0.090232,
            -0.000614, 0.094802, 0.075573, -0.028063, -0.045228, 0.032987, -0.016513, -0.012909,
            -0.006836, -0.064583, 0.062949, -0.039076, -0.081444, -0.078110, -0.063795, 0.018442,
            -0.096288, 0.036940, -0.061603, -0.002975, -0.024222, -0.029959, 0.064528, -0.034116,
            0.034082, 0.068064, 0.049402, 0.042547, -0.009375, -0.051882, -0.073834, 0.094987,
            -0.090729, -0.024422, 0.084711, -0.019011, -0.083979, 0.098438, -0.025750, 0.041771,
            0.045489, -0.031846, -0.096901, -0.014227, -0.079363, -0.049132, -0.084120, 0.046635,
            -0.070393, 0.048731, 0.084100, 0.049850, 0.059407, 0.016985, 0.066929, 0.094022,
            0.021326, 0.042493, 0.063022, -0.000047, 0.085962, 0.024211, -0.060968, -0.075939,
            0.025448, -0.079029, -0.090239, 0.092921, 0.075480, -0.027036, 0.049847, 0.033846,
            0.092464, 0.030500, 0.046280, -0.071514, 0.033053, -0.077045, -0.076392, 0.055052,
            -0.048939, 0.077468, -0.025258, 0.051964, 0.093184, 0.012995, 0.090811, -0.062992,
            0.053047, 0.031429, 0.076086, 0.091989, -0.020781, 0.021472, -0.050468, 0.042441,
            -0.000392, -0.046081, -0.055258, 0.024937, 0.089044, -0.088518, 0.064568, 0.070356,
            -0.086463, 0.045621, 0.008845, 0.075569, 0.084552, 0.056079, -0.039088, -0.087391,
            0.069806, 0.000217, 0.040658, 0.031538, 0.013258, 0.075684, -0.025030, 0.040429,
            0.046418, -0.045409, 0.042928, 0.007104, -0.099384, 0.055310, -0.088374, 0.045301,
            -0.046061, -0.048626, 0.056071, -0.004176, -0.073031, 0.034667, -0.055033, -0.091245,
            0.080445, -0.084350, -0.042610, -0.060323, 0.076402, -0.009017, -0.083527, 0.058689,
            0.075131, 0.030802, -0.011048, -0.096173, -0.015370, 0.054870, -0.067988, -0.098534,
            -0.000707, 0.016014, -0.039179, -0.041292, -0.077559, -0.015135, 0.093027, 0.083280,
            0.016660, 0.000333, 0.089335, 0.014916, -0.013800, -0.070998, 0.092370, -0.095492,
            -0.034960, 0.021823, 0.012213, -0.032666, -0.093146, 0.009977, -0.024038, -0.070095,
            -0.062030, -0.048842, -0.092516, 0.048521, -0.001483, -0.086571, 0.050787, 0.076275,
            0.022295, -0.033571, -0.032077, 0.032336, 0.094443, 0.009951, -0.038085, -0.013613,
            0.038337, 0.087247, 0.095776, 0.099703, -0.068564, 0.016611, 0.005346, 0.020609,
            0.004760, -0.034648, 0.028837, 0.084454, 0.015640, 0.032696, 0.060796, 0.000180,
            -0.035567, 0.095693, -0.054621, 0.094940, -0.014397, 0.063028, 0.007866, 0.011200,
            0.073766, -0.040331, -0.083454, 0.007587, -0.083224, -0.067493, 0.098204, 0.014445,
            -0.035091, 0.096702, 0.005608, 0.075064, 0.010566, 0.082245, 0.076828, -0.006819,
            -0.083656, 0.089894, -0.094559, -0.006580, -0.018737, 0.095232, -0.093582, -0.002106,
            0.029439, 0.056275, 0.095320, -0.061756, 0.088117, 0.082219, 0.025228, -0.002754,
            0.076050, -0.041554, -0.018785, 0.006487, 0.052851, -0.039630, 0.094237, -0.029871,
            0.041347, 0.051462, 0.073709, -0.088291, 0.017446, 0.044577, 0.073208, 0.055934,
            0.067184, -0.060405, 0.085353, -0.039426, -0.014533, -0.005222, -0.090715, 0.086322,
            -0.094438, 0.028993, -0.046134, 0.043121, 0.028620, -0.027882, -0.069053, 0.040021,
            0.042679, -0.098569, -0.022207, -0.036867, -0.001762, -0.036532, -0.094680, 0.033111,
            0.044252, 0.058183, 0.082676, 0.058989, -0.061505, 0.097124, -0.065840, 0.089148,
            -0.086952, -0.054023, -0.071682, 0.005883, -0.038024, 0.091018, -0.015039, -0.049050,
            -0.073318, -0.022586, -0.053207, -0.021834, -0.038573, 0.091867, 0.099366, -0.054340,
            0.082445, -0.011343, -0.029376, -0.097215, -0.068465, -0.045625, -0.064184, -0.065718,
            -0.013677, -0.050958, -0.065458, 0.073853, -0.074792, -0.062916, -0.033063, 0.072169,
            0.003192, -0.035831, -0.069300, 0.017375, 0.084564, -0.080124, -0.026821, -0.057204,
            -0.010129, -0.006231, -0.048205, 0.056828, 0.025593, -0.090649, 0.065486, -0.028573,
            -0.002300, 0.017160, 0.065038, 0.011789, 0.084440, 0.004518, -0.035558, 0.083645,
        ];
        pos_embed_data
    }

    // mel_input Shape: [1, 4, 10]
    fn get_mel_input_data() -> Vec<f32> {
        vec![
            1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667,
            -0.752135, 1.648723, -0.392479, -1.403607, -0.727881, -0.559430, -0.768839, 0.762445,
            1.642317, -0.159597, -0.497398, 0.439589, -0.758131, 1.078318, 0.800801, 1.680621,
            0.034912, 0.321103, 1.573600, -0.845467, 1.312308, 0.687160, -1.089175, -0.355287,
            -1.418060, 0.896268, 0.049905, 2.266718, 1.178951, -0.434454, -1.386366, -1.286216,
        ]
    }

    // frontend_output Shape: [1, 5, 8]
    fn get_frontend_output_data() -> Vec<f32> {
        vec![
            0.043605, -0.064049, -0.040184, 0.105161, -0.065381, -0.026701, 0.059192, -0.008650,
            -0.076440, -0.117406, 0.015703, -0.005414, 0.122630, 0.005762, -0.005344, -0.026535,
            -0.015208, 0.046031, -0.086211, -0.078521, 0.038325, 0.020003, -0.057400, -0.000514,
            0.080639, -0.095072, 0.030898, -0.001475, -0.090110, -0.093364, 0.050631, 0.093680,
            0.010636, -0.012569, 0.046244, 0.087715, 0.076234, 0.071110, 0.022132, 0.047630,
        ]
    }

    #[test]
    fn test_conv_frontend_golden() -> Result<()> {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        w.insert("model.conv1.weight".into(), get_conv1_weight_data());
        s.insert("model.conv1.weight".into(), vec![8, 4, 3]);

        w.insert("model.conv1.bias".into(), get_conv1_bias_data());
        s.insert("model.conv1.bias".into(), vec![8]);

        w.insert("model.conv2.weight".into(), get_conv2_weight_data());
        s.insert("model.conv2.weight".into(), vec![8, 8, 3]);

        w.insert("model.conv2.bias".into(), get_conv2_bias_data());
        s.insert("model.conv2.bias".into(), vec![8]);

        // Note: You must provide the full vector for this to match shape [1500, 8]
        // If you only paste the short snippet, change shape to match the snippet length!
        // Assuming you pasted the full block:
        let pos_data = get_pos_embed_data();
        let pos_len = pos_data.len() / 8; // Should be 1500
        w.insert("model.embed_positions.weight".into(), pos_data);
        s.insert("model.embed_positions.weight".into(), vec![pos_len, 8]);

        let (weights, _tmp) = create_model_weights(w, s)?;

        // 2. Initialize Frontend
        let frontend = AudioConvFrontend::from_weights(&weights, "model", 1500)?;

        // 3. Prepare Input
        let input_data = get_mel_input_data();
        let input = Array3::from_shape_vec((1, 4, 10), input_data)?;

        // 4. Run
        let output = frontend.forward(&input)?;

        // 5. Validation
        let golden_data = get_frontend_output_data();
        let golden = Array3::from_shape_vec((1, 5, 8), golden_data)?;

        let diff = (&output - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Conv Frontend Max Diff: {}", max_diff);

        // Relax tolerance slightly for GELU/Conv float ops differences
        assert!(max_diff < 1e-4, "Frontend output mismatch");

        Ok(())
    }
    #[test]
    fn test_mel_spectrogram_shape() {
        let config = MelConfig::default(); // n_mels = 80

        // 1 second of audio at 16khz
        let audio = vec![0.0f32; 16000];

        let spec = compute_mel_spectrogram(&audio, &config).unwrap();

        // Check shape: [n_mels, time_frames]
        assert_eq!(spec.nrows(), 80);
        assert!(spec.ncols() > 0);
    }

    #[test]
    fn test_hann_window() {
        let window = hann_window(400);
        assert_eq!(window.len(), 400);
        // Hann window should be near 0 at edges
        assert!(window[0].abs() < 1e-6);
    }
}
