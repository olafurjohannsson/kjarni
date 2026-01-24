//! Emotion detection examples.
//!
//! Run with: cargo run --example emotion

use kjarni::classifier::{self, Classifier, ClassificationMode};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // =========================================================================
    // 1. Basic emotions (7 classes)
    // =========================================================================
    
    let text = "I can't believe they cancelled the concert!";
    let result = classifier::classify("distilroberta-emotion", text).await?;
    
    println!("Basic emotion detection:");
    println!("  Text: \"{text}\"");
    println!("  Emotion: {} ({:.1}%)", result.label, result.score * 100.0);
    
    // Show top 3
    println!("  Top 3:");
    for (label, score) in result.top_k(3) {
        println!("    {}: {:.1}%", label, score * 100.0);
    }
    
    // =========================================================================
    // 2. Fine-grained emotions (28 classes, multi-label)
    // =========================================================================
    
    let text = "Thank you so much, this means the world to me!";
    
    let classifier = Classifier::builder("roberta-emotions")
        .multi_label()      // multiple emotions can be true
        .threshold(0.3)     // only show emotions above 30%
        .build()
        .await?;
    
    let result = classifier.classify(text).await?;
    
    println!("\nFine-grained emotions (multi-label):");
    println!("  Text: \"{text}\"");
    println!("  Detected:");
    for (label, score) in &result.all_scores {
        println!("    {}: {:.1}%", label, score * 100.0);
    }
    
    // =========================================================================
    // 3. Comparing models on the same text
    // =========================================================================
    
    let texts = [
        "This is the best day of my life!",
        "I'm so worried about the exam tomorrow.",
        "Ugh, the traffic is unbearable.",
    ];
    
    let basic = Classifier::new("distilroberta-emotion").await?;
    let detailed = Classifier::builder("roberta-emotions")
        .multi_label()
        .threshold(0.2)
        .build()
        .await?;
    
    println!("\nBasic vs detailed:");
    for text in texts {
        let basic_result = basic.classify(text).await?;
        let detailed_result = detailed.classify(text).await?;
        
        let detailed_labels: Vec<_> = detailed_result
            .all_scores
            .iter()
            .map(|(l, _)| l.as_str())
            .collect();
        
        println!("  \"{text}\"");
        println!("    Basic:    {}", basic_result.label);
        println!("    Detailed: {}", detailed_labels.join(", "));
    }
    
    Ok(())
}