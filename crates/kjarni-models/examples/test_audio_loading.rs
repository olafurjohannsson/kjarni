// test_audio_loading.rs

use anyhow::Result;
use kjarni_transformers::audio::{
    load_audio, AudioLoaderConfig, MelConfig, compute_mel_spectrogram,
    create_mel_filterbank_librosa
};
use ndarray::Array2;

fn main() -> Result<()> {
    let test_wav = std::env::args().nth(1);
    
    if let Some(path) = test_wav {
        println!("=== Load WAV file ===");
        println!("Loading: {}", path);
        
        let config = AudioLoaderConfig::for_whisper();
        let audio_data = load_audio(&path, &config)?;
        
        println!("Sample rate: {}", audio_data.sample_rate);
        println!("Duration: {:.2}s", audio_data.duration_secs);
        println!("Samples: {}", audio_data.num_samples());
        println!("Audio[:10]: {:?}", &audio_data.samples[..10]);
        
        // Test 1: Raw log mel (current implementation)
        println!("\n=== Raw Log Mel (default) ===");
        let mel_config = MelConfig::default();
        let mel = compute_mel_spectrogram(&audio_data.samples, &mel_config)?;
        print_mel_stats(&mel);
        
        // Test 2: Whisper-style mel
        println!("\n=== Whisper-style Mel ===");
        let mel_config = MelConfig::whisper();
        let mel = compute_mel_spectrogram(&audio_data.samples, &mel_config)?;
        print_mel_stats(&mel);
        debug_mel_filterbank();
        
    } else {
        println!("Usage: cargo run --example test_audio_loading <audio.wav>");
    }
    
    Ok(())
}

fn print_mel_stats(mel: &Array2<f32>) {
    println!("Shape: {:?}", mel.dim());
    println!("Min/max: {:.6} / {:.6}", 
        mel.iter().cloned().fold(f32::INFINITY, f32::min),
        mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("Mean: {:.6}", mel.mean().unwrap_or(0.0));
    
    print!("Mel [0, :10]: [");
    for i in 0..10.min(mel.ncols()) {
        print!("{:.6}", mel[[0, i]]);
        if i < 9 { print!(", "); }
    }
    println!("]");
    
    print!("Mel [:5, 0]: [");
    for i in 0..5.min(mel.nrows()) {
        print!("{:.6}", mel[[i, 0]]);
        if i < 4 { print!(", "); }
    }
    println!("]");
}

// Add to your test temporarily
fn debug_mel_filterbank() {
    let mel_filters = create_mel_filterbank_librosa(16000, 400, 80, 0.0, 8000.0).unwrap();
    
    println!("=== Rust Mel Filterbank ===");
    println!("Shape: {:?}", mel_filters.dim());
    
    // Sum per filter (first 5)
    print!("Sum per filter (first 5): [");
    for i in 0..5 {
        let sum: f32 = mel_filters.row(i).sum();
        print!("{:.6}", sum);
        if i < 4 { print!(", "); }
    }
    println!("]");
    
    // Max per filter (first 5)
    print!("Max per filter (first 5): [");
    for i in 0..5 {
        let max = mel_filters.row(i).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        print!("{:.6}", max);
        if i < 4 { print!(", "); }
    }
    println!("]");
    
    // Filter 0 first 10 bins
    print!("Filter 0 [0:10]: [");
    for i in 0..10 {
        print!("{:.6}", mel_filters[[0, i]]);
        if i < 9 { print!(", "); }
    }
    println!("]");
}