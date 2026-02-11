//! Sentiment analysis examples.
//!
//! Run with: cargo run --example sentiment

use kjarni::classifier::{self, Classifier};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // One-liner — the simplest API
    
    let result = classifier::classify("distilbert-sentiment", "I love this product!").await?;
    println!("One-liner: {} ({:.1}%)", result.label, result.score * 100.0);
    
    // Builder configuration
    
    let classifier = Classifier::builder("distilbert-sentiment")
        .top_k(2)           // return top 2 results
        .threshold(0.01)    // filter below 1%
        .build()
        .await?;
    
    let result = classifier.classify("This is terrible.").await?;
    println!("\nWith builder:");
    for (label, score) in &result.all_scores {
        println!("  {}: {:.2}%", label, score * 100.0);
    }
    
    // Three sentiment models 
    
    let text = "The food was okay, nothing special.";
    
    // Binary: positive/negative
    let binary = classifier::classify("distilbert-sentiment", text).await?;
    
    // 3-class: positive/neutral/negative  
    let three_class = classifier::classify("roberta-sentiment", text).await?;
    
    // 5-star: 1-5 stars (multilingual)
    let five_star = classifier::classify("bert-sentiment-multilingual", text).await?;
    
    println!("\nSame text, three models:");
    println!("  Text: \"{text}\"");
    println!("  Binary:    {}", binary.label);
    println!("  3-class:   {}", three_class.label);
    println!("  5-star:    {}", five_star.label);
    
    // Batch classification — multiple texts at once
    
    let reviews = [
        "Absolutely fantastic, exceeded expectations!",
        "Worst purchase I've ever made.",
        "It's fine, does what it says.",
        "Not bad for the price.",
    ];
    
    let classifier = Classifier::new("roberta-sentiment").await?;
    let results = classifier.classify_batch(&reviews).await?;
    
    println!("\nBatch classification:");
    for (text, result) in reviews.iter().zip(results.iter()) {
        println!("  {:50} → {}", text, result.label);
    }
    
    Ok(())
}