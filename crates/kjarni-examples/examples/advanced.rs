//! Advanced classifier API examples.
//!
//! Run with: cargo run --example advanced

use kjarni::classifier::{
    self, Classifier, ClassificationMode, ClassificationOverrides,
    presets::{self, SentimentTier, SENTIMENT_BINARY_V1},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    
    println!("=== Device Selection ===\n");
    
    // Explicit CPU
    let classifier = Classifier::builder("distilbert-sentiment")
        .cpu()
        .build()
        .await?;
    println!("Running on: {:?}", classifier.device());
    
    // GPU (falls back to CPU if unavailable)
    let classifier = Classifier::builder("distilbert-sentiment")
        .gpu()
        .build()
        .await?;
    println!("Running on: {:?}", classifier.device());
    
    // Auto-select best available
    let classifier = Classifier::builder("distilbert-sentiment")
        .auto_device()
        .build()
        .await?;
    println!("Auto-selected: {:?}", classifier.device());
    
    println!("\n=== Precision Control ===\n");
    
    // Use float16 for lower memory
    let _classifier = Classifier::builder("distilbert-sentiment")
        .f16()
        .build()
        .await?;
    println!("Loaded with f16 precision");
    
    // Use bfloat16
    let _classifier = Classifier::builder("distilbert-sentiment")
        .bf16()
        .build()
        .await?;
    println!("Loaded with bf16 precision");
    
    println!("\n=== Custom Labels ===\n");
    
    // Override labels for a model
    let classifier = Classifier::builder("distilbert-sentiment")
        .labels(vec!["👎", "👍"])
        .build()
        .await?;
    
    let result = classifier.classify("This is great!").await?;
    println!("With emoji labels: {}", result.label);
    
    // Labels from comma-separated string
    let classifier = Classifier::builder("distilbert-sentiment")
        .labels_str("bad,good")
        .build()
        .await?;
    
    let result = classifier.classify("This is great!").await?;
    println!("With string labels: {}", result.label);
    
    // Preset label helpers
    let classifier = Classifier::builder("distilbert-sentiment")
        .sentiment_labels()  // ["negative", "positive"]
        .build()
        .await?;
    println!("Using sentiment_labels(): {:?}", classifier.labels());
    
    println!("\n=== Classification Modes ===\n");
    
    let text = "I'm happy but also a bit nervous about this.";
    
    // Single-label (softmax, mutually exclusive) — default
    let single = Classifier::builder("roberta-emotions")
        .single_label()
        .build()
        .await?;
    let result = single.classify(text).await?;
    println!("Single-label: {} ({:.1}%)", result.label, result.score * 100.0);
    
    // Multi-label (sigmoid, independent predictions)
    let multi = Classifier::builder("roberta-emotions")
        .multi_label()
        .threshold(0.25)
        .build()
        .await?;
    let result = multi.classify(text).await?;
    println!("Multi-label (>25%):");
    for (label, score) in &result.all_scores {
        println!("  {}: {:.1}%", label, score * 100.0);
    }
    
    println!("\n=== Default Overrides ===\n");
    
    let classifier = Classifier::builder("roberta-sentiment")
        .top_k(2)               // always return top 2
        .threshold(0.05)        // filter below 5%
        .max_length(128)        // truncate long inputs
        .batch_size(16)         // batch inference size
        .build()
        .await?;
    
    let result = classifier.classify("Pretty good overall.").await?;
    println!("Top {} results (threshold 5%):", result.all_scores.len());
    for (label, score) in &result.all_scores {
        println!("  {}: {:.1}%", label, score * 100.0);
    }
    
    println!("\n=== Runtime Overrides ===\n");
    
    let classifier = Classifier::new("roberta-sentiment").await?;
    
    // Override at call time
    let overrides = ClassificationOverrides {
        top_k: Some(1),
        threshold: Some(0.5),
        ..Default::default()
    };
    
    let result = classifier
        .classify_with_config("Excellent!", &overrides)
        .await?;
    println!("With runtime override: {}", result);
    
    // Convenience constructors
    let result = classifier
        .classify_with_config("Meh.", &ClassificationOverrides::top_1())
        .await?;
    println!("Using top_1(): {}", result);
    
    println!("\n=== Batch Classification ===\n");
    
    let texts = [
        "Amazing product!",
        "Total waste of money.",
        "It works as expected.",
    ];
    
    let overrides = ClassificationOverrides::with_threshold(0.1);
    let results = classifier
        .classify_batch_with_config(&texts, &overrides)
        .await?;
    
    for (text, result) in texts.iter().zip(results.iter()) {
        println!("{:30} → {}", text, result);
    }
    
    println!("\n=== Raw Scores ===\n");
    
    let scores = classifier.classify_scores("Not bad at all.").await?;
    println!("Raw score vector: {:?}", scores);
    
    println!("\n=== Result Methods ===\n");
    
    let result = classifier.classify("This is wonderful!").await?;
    
    // Check confidence threshold
    if result.is_confident(0.9) {
        println!("High confidence: {}", result.label);
    }
    
    // Get predictions above threshold
    let strong = result.above_threshold(0.1);
    println!("Above 10%: {:?}", strong);
    
    // Get top K
    println!("Top 2: {:?}", result.top_k(2));
    
    println!("\n=== Model Introspection ===\n");
    
    let classifier = Classifier::new("roberta-sentiment").await?;
    
    println!("Model ID:        {}", classifier.model_id());
    println!("Model name:      {}", classifier.model_name());
    println!("Device:          {:?}", classifier.device());
    println!("Labels:          {:?}", classifier.labels());
    println!("Num labels:      {}", classifier.num_labels());
    println!("Max seq length:  {}", classifier.max_seq_length());
    println!("Mode:            {:?}", classifier.mode());
    println!("Custom labels:   {}", classifier.has_custom_labels());
    
    println!("\n=== Presets and Tiers ===\n");
    
    // Use a preset directly
    let classifier = Classifier::from_preset(&SENTIMENT_BINARY_V1)
        .build()
        .await?;
    println!("From preset: {}", classifier.model_name());
    
    // Use tier-based selection
    let preset = SentimentTier::Fast.resolve();
    println!("Fast tier:     {} ({}MB)", preset.model, preset.memory_mb);
    
    let preset = SentimentTier::Balanced.resolve();
    println!("Balanced tier: {} ({}MB)", preset.model, preset.memory_mb);
    
    let preset = SentimentTier::Detailed.resolve();
    println!("Detailed tier: {} ({}MB)", preset.model, preset.memory_mb);
    
    // Find preset by name
    if let Some(preset) = presets::find_preset("EMOTION_V1") {
        println!("Found preset:  {} - {}", preset.name, preset.description);
    }
    
    println!("\n=== Download Policy ===\n");
    
    // Never download (fail if not cached)
    let result = Classifier::builder("distilbert-sentiment")
        .offline()
        .build()
        .await;
    
    match result {
        Ok(c) => println!("Loaded from cache: {}", c.model_name()),
        Err(e) => println!("Offline mode error: {}", e),
    }
    
    // Suppress progress output
    let _classifier = Classifier::builder("distilbert-sentiment")
        .quiet(true)
        .build()
        .await?;
    println!("\nLoaded with quiet mode (no progress output)");
    
    println!("\n=== Reusing Classifier ===\n");
    
    // Load once, classify many since model stays in memory
    let classifier = Classifier::new("distilbert-sentiment").await?;
    
    let inputs = vec![
        "First review",
        "Second review", 
        "Third review",
        "Fourth review",
    ];
    
    for text in &inputs {
        let result = classifier.classify(text).await?;
        println!("{}: {}", text, result.label);
    }
    
    let results = classifier.classify_batch(&inputs).await?;
    println!("\nBatch processed {} items", results.len());
    
    Ok(())
}