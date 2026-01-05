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
    pub chunk_length: usize, // 30 seconds for Whisper
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,        // Whisper uses 80 or 128 mels
            chunk_length: 30,
        }
    }
}

/// Compute mel spectrogram from raw audio.
///
/// # Arguments
/// * `audio` - Raw audio samples, normalized to [-1, 1]
/// * `config` - Mel spectrogram configuration
///
/// # Returns
/// Mel spectrogram of shape [n_mels, time_frames]
pub fn compute_mel_spectrogram(audio: &[f32], config: &MelConfig) -> Result<Array2<f32>> {
    let n_samples = audio.len();
    let n_frames = (n_samples - config.n_fft) / config.hop_length + 1;
    
    // 1. Create mel filterbank
    let mel_filters = create_mel_filterbank(config)?;
    
    // 2. Compute STFT
    let mut spectrogram = Array2::<f32>::zeros((config.n_fft / 2 + 1, n_frames));
    let window = hann_window(config.n_fft);
    
    for frame_idx in 0..n_frames {
        let start = frame_idx * config.hop_length;
        let end = start + config.n_fft;
        
        if end > n_samples {
            break;
        }
        
        // Apply window and compute FFT magnitude
        let windowed: Vec<f32> = audio[start..end]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();
        
        let magnitudes = compute_fft_magnitude(&windowed)?;
        
        for (i, &mag) in magnitudes.iter().enumerate() {
            spectrogram[[i, frame_idx]] = mag;
        }
    }
    
    // 3. Apply mel filterbank
    let mel_spec = mel_filters.dot(&spectrogram);
    
    // 4. Log scale
    let log_mel = mel_spec.mapv(|x| (x.max(1e-10)).ln());
    
    Ok(log_mel)
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

/// Hann window function.
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
        })
        .collect()
}

/// Compute FFT magnitude (simplified - use rustfft in production).
fn compute_fft_magnitude(samples: &[f32]) -> Result<Vec<f32>> {
    // In production, use rustfft crate
    // This is a placeholder that computes DFT directly (slow but correct)
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
    pub fn from_weights(weights: &ModelWeights, prefix: &str, max_positions: usize) -> Result<Self> {
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
        let x = x.permuted_axes([0, 2, 1]).to_owned();
        
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
        Ok(Self { mel_config, frontend })
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
            output.slice_mut(s![b, .., ..]).assign(&hs.slice(s![0, .., ..]));
        }
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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