// kjarni_transformers/src/audio/loader.rs

use anyhow::{anyhow, Result};
use std::path::Path;

/// Audio sample data with metadata
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Samples normalized to [-1.0, 1.0]
    pub samples: Vec<f32>,
    /// Original sample rate before resampling
    pub original_sample_rate: u32,
    /// Current sample rate (after resampling if applied)
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Duration in seconds
    pub duration_secs: f32,
}

impl AudioData {
    /// Duration in seconds
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
    
    /// Number of samples
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }
}

/// Audio loader configuration
#[derive(Debug, Clone)]
pub struct AudioLoaderConfig {
    /// Target sample rate (Whisper uses 16000)
    pub target_sample_rate: u32,
    /// Convert to mono
    pub mono: bool,
    /// Normalize samples to [-1, 1]
    pub normalize: bool,
    /// Maximum duration in seconds (None = no limit)
    pub max_duration_secs: Option<f32>,
    /// Pad or truncate to exact duration (for Whisper's 30s chunks)
    pub target_duration_secs: Option<f32>,
}

impl Default for AudioLoaderConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16000, // Whisper's sample rate
            mono: true,
            normalize: true,
            max_duration_secs: None,
            target_duration_secs: None,
        }
    }
}

impl AudioLoaderConfig {
    /// Config for Whisper (16kHz, mono, 30s chunks)
    pub fn for_whisper() -> Self {
        Self {
            target_sample_rate: 16000,
            mono: true,
            normalize: false,
            max_duration_secs: Some(30.0),
            target_duration_secs: Some(30.0), // Pad to 30s
        }
    }
}

/// Load audio from file
pub fn load_audio<P: AsRef<Path>>(path: P, config: &AudioLoaderConfig) -> Result<AudioData> {
    let path = path.as_ref();
    
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    
    match extension.as_str() {
        "wav" => load_wav(path, config),
        "mp3" | "flac" | "ogg" => {
            #[cfg(feature = "symphonia")]
            {
                load_with_symphonia(path, config)
            }
            #[cfg(not(feature = "symphonia"))]
            {
                Err(anyhow!(
                    "Format '{}' requires the 'symphonia' feature. Only WAV is supported by default.",
                    extension
                ))
            }
        }
        _ => Err(anyhow!("Unsupported audio format: {}", extension)),
    }
}

/// Load audio from raw bytes (useful for streaming/API)
pub fn load_audio_bytes(bytes: &[u8], format: &str, config: &AudioLoaderConfig) -> Result<AudioData> {
    match format.to_lowercase().as_str() {
        "wav" => load_wav_bytes(bytes, config),
        _ => Err(anyhow!("Unsupported format for bytes: {}", format)),
    }
}

/// Load WAV file using hound
fn load_wav<P: AsRef<Path>>(path: P, config: &AudioLoaderConfig) -> Result<AudioData> {
    let reader = hound::WavReader::open(path.as_ref())
        .map_err(|e| anyhow!("Failed to open WAV file: {}", e))?;
    
    load_wav_reader(reader, config)
}

/// Load WAV from bytes
fn load_wav_bytes(bytes: &[u8], config: &AudioLoaderConfig) -> Result<AudioData> {
    let cursor = std::io::Cursor::new(bytes);
    let reader = hound::WavReader::new(cursor)
        .map_err(|e| anyhow!("Failed to read WAV bytes: {}", e))?;
    
    load_wav_reader(reader, config)
}

/// Common WAV loading logic
fn load_wav_reader<R: std::io::Read>(reader: hound::WavReader<R>, config: &AudioLoaderConfig) -> Result<AudioData> {
    let spec = reader.spec();
    let original_sample_rate = spec.sample_rate;
    let channels = spec.channels;
    let sample_format = spec.sample_format;
    let bits_per_sample = spec.bits_per_sample;
    
    // Read and convert samples to f32
    let samples: Vec<f32> = match sample_format {
        hound::SampleFormat::Int => {
            let max_value = (1u32 << (bits_per_sample - 1)) as f32;
            match bits_per_sample {
                8 => reader
                    .into_samples::<i8>()
                    .map(|s| s.map(|v| v as f32 / max_value))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| anyhow!("Failed to read samples: {}", e))?,
                16 => reader
                    .into_samples::<i16>()
                    .map(|s| s.map(|v| v as f32 / max_value))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| anyhow!("Failed to read samples: {}", e))?,
                24 | 32 => reader
                    .into_samples::<i32>()
                    .map(|s| s.map(|v| v as f32 / max_value))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| anyhow!("Failed to read samples: {}", e))?,
                _ => return Err(anyhow!("Unsupported bit depth: {}", bits_per_sample)),
            }
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("Failed to read float samples: {}", e))?,
    };
    
    // Convert to mono if needed
    let samples = if config.mono && channels > 1 {
        convert_to_mono(&samples, channels as usize)
    } else {
        samples
    };
    
    // Resample if needed
    let (samples, sample_rate) = if original_sample_rate != config.target_sample_rate {
        let resampled = resample(&samples, original_sample_rate, config.target_sample_rate)?;
        (resampled, config.target_sample_rate)
    } else {
        (samples, original_sample_rate)
    };
    
    // Apply max duration
    let samples = if let Some(max_secs) = config.max_duration_secs {
        let max_samples = (max_secs * sample_rate as f32) as usize;
        if samples.len() > max_samples {
            samples[..max_samples].to_vec()
        } else {
            samples
        }
    } else {
        samples
    };
    
    // Pad/truncate to target duration
    let samples = if let Some(target_secs) = config.target_duration_secs {
        let target_samples = (target_secs * sample_rate as f32) as usize;
        pad_or_truncate(&samples, target_samples)
    } else {
        samples
    };
    
    // Normalize if needed
    let samples = if config.normalize {
        normalize_samples(&samples)
    } else {
        samples
    };
    
    let duration_secs = samples.len() as f32 / sample_rate as f32;
    
    Ok(AudioData {
        samples,
        original_sample_rate,
        sample_rate,
        channels: if config.mono { 1 } else { channels },
        duration_secs,
    })
}

/// Convert stereo (or multi-channel) to mono by averaging
fn convert_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    samples
        .chunks(channels)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample audio to target sample rate using linear interpolation
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);
    
    for i in 0..output_len {
        let src_idx = i as f64 / ratio;
        let src_idx_floor = src_idx.floor() as usize;
        let src_idx_ceil = (src_idx_floor + 1).min(samples.len() - 1);
        let frac = src_idx - src_idx_floor as f64;
        
        let sample = if src_idx_floor < samples.len() {
            let a = samples[src_idx_floor];
            let b = samples[src_idx_ceil];
            a + (b - a) * frac as f32
        } else {
            0.0
        };
        
        output.push(sample);
    }
    
    Ok(output)
}

/// High-quality resampling using rubato (sinc interpolation)
#[cfg(feature = "rubato")]
fn resample_hq(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::{FftFixedIn, Resampler};
    
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    
    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        1024,  // chunk size
        2,     // sub chunks
        1,     // channels
    ).map_err(|e| anyhow!("Failed to create resampler: {}", e))?;
    
    let input = vec![samples.to_vec()];
    let mut output = resampler
        .process(&input, None)
        .map_err(|e| anyhow!("Resampling failed: {}", e))?;
    
    Ok(output.pop().unwrap_or_default())
}

/// Pad or truncate samples to exact length
fn pad_or_truncate(samples: &[f32], target_len: usize) -> Vec<f32> {
    if samples.len() >= target_len {
        samples[..target_len].to_vec()
    } else {
        let mut padded = samples.to_vec();
        padded.resize(target_len, 0.0);
        padded
    }
}

fn normalize_samples(samples: &[f32]) -> Vec<f32> {
    let max_abs = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    
    if max_abs > 0.0 && max_abs != 1.0 {
        samples.iter().map(|s| s / max_abs).collect()
    } else {
        samples.to_vec()
    }
}

pub fn load_audio_for_whisper<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    let config = AudioLoaderConfig::for_whisper();
    let audio = load_audio(path, &config)?;
    Ok(audio.samples)
}

pub fn create_sine_wave(frequency: f32, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_convert_to_mono() {
        let stereo = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
        let mono = convert_to_mono(&stereo, 2);
        assert_eq!(mono, vec![0.5, 0.5, 0.5]);
    }
    
    #[test]
    fn test_pad_or_truncate() {
        let samples = vec![1.0, 2.0, 3.0];
        
        let truncated = pad_or_truncate(&samples, 2);
        assert_eq!(truncated, vec![1.0, 2.0]);
        let padded = pad_or_truncate(&samples, 5);
        assert_eq!(padded, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }
    
    #[test]
    fn test_normalize() {
        let samples = vec![0.5, -1.0, 0.25];
        let normalized = normalize_samples(&samples);
        assert_eq!(normalized, vec![0.5, -1.0, 0.25]);
        
        let samples = vec![0.25, -0.5, 0.125];
        let normalized = normalize_samples(&samples);
        assert_eq!(normalized, vec![0.5, -1.0, 0.25]);
    }
    
    #[test]
    fn test_create_sine_wave() {
        let wave = create_sine_wave(440.0, 0.01, 16000);
        assert_eq!(wave.len(), 160);
        assert!(wave[0].abs() < 0.01);
    }
    
    #[test]
    fn test_resample() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let resampled = resample(&samples, 1000, 500).unwrap();
        assert_eq!(resampled.len(), 50);
    }
    
    #[test]
    fn test_audio_loader_config() {
        let config = AudioLoaderConfig::for_whisper();
        assert_eq!(config.target_sample_rate, 16000);
        assert!(config.mono);
        assert_eq!(config.target_duration_secs, Some(30.0));
    }
}