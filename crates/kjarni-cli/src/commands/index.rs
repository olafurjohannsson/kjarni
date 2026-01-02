//! Index management commands
//!
//! Uses production-grade segmented index that:
//! - Streams documents to disk (no OOM)
//! - Memory-maps vectors for search
//! - Handles 100GB+ datasets

use anyhow::{Result, anyhow};
use kjarni::Chunk;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokio::sync::mpsc;
use tokio::sync::Semaphore;
use std::sync::Arc;
use kjarni::{Device, ModelArchitecture, ModelType, SentenceEncoder, registry};
use kjarni::{DocumentLoader, IndexConfig, IndexReader, IndexWriter, LoaderConfig, SplitterConfig};
use kjarni_cli::IndexCommands;

const ENCODE_BATCH_SIZE: usize = 32;

pub async fn run(action: IndexCommands) -> Result<()> {
    match action {
        IndexCommands::Create {
            output,
            inputs,
            chunk_size,
            chunk_overlap,
            model,
            gpu,
            quiet,
        } => {
            create(
                &output,
                &inputs,
                chunk_size,
                chunk_overlap,
                &model,
                gpu,
                quiet,
            )
            .await
        }
        IndexCommands::Add {
            index_path,
            inputs,
            chunk_size,
            chunk_overlap,
            model,
            gpu,
            quiet,
        } => {
            add(
                &index_path,
                &inputs,
                chunk_size,
                chunk_overlap,
                &model,
                gpu,
                quiet,
            )
            .await
        }
        IndexCommands::Info { index_path } => info(&index_path),
    }
}

/// Create a new index from documents
async fn create(
    output: &str,
    inputs: &[String],
    chunk_size: usize,
    chunk_overlap: usize,
    model: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    if inputs.is_empty() {
        return Err(anyhow!(
            "No input files specified. Provide files or directories to index."
        ));
    }

    // 1. Load encoder and determine dimension
    let encoder = load_encoder(model, gpu, quiet).await?;
    let dimension = {
        let sample = encoder.encode("dimension probe").await?;
        sample.len()
    };

    if !quiet {
        eprintln!("Embedding dimension: {}", dimension);
    }

    // 2. Create index writer
    let config = IndexConfig {
        dimension,
        max_docs_per_segment: 10_000,
        embedding_model: Some(model.to_string()),
        ..Default::default()
    };

    let mut writer = IndexWriter::open(output, config)?;

    // 3. Process documents
    let total_indexed = process_inputs(
        &mut writer,
        &encoder,
        inputs,
        chunk_size,
        chunk_overlap,
        quiet,
    )
    .await?;

    if total_indexed == 0 {
        return Err(anyhow!("No documents found to index."));
    }

    // 4. Commit
    writer.commit()?;

    if !quiet {
        eprintln!();
        eprintln!("✓ Index created: {}", output);
        eprintln!("  Documents: {}", total_indexed);
        eprintln!("  Dimension: {}", dimension);

        // Show size
        if let Ok(size) = calculate_index_size(output) {
            eprintln!("  Size: {}", format_size(size));
        }
    }

    Ok(())
}

/// Add documents to an existing index
async fn add(
    index_path: &str,
    inputs: &[String],
    chunk_size: usize,
    chunk_overlap: usize,
    model: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    if inputs.is_empty() {
        return Err(anyhow!("No input files specified."));
    }

    // 1. Open existing index
    let mut writer = IndexWriter::open_existing(index_path)?;
    let initial_count = writer.len();
    let dimension = writer.dimension();

    if !quiet {
        eprintln!("Opened index with {} documents", initial_count);
    }

    // 2. Load encoder
    let encoder = load_encoder(model, gpu, quiet).await?;

    // Verify dimension matches
    let encoder_dim = {
        let sample = encoder.encode("dimension probe").await?;
        sample.len()
    };

    if encoder_dim != dimension {
        return Err(anyhow!(
            "Dimension mismatch: index has {}, encoder produces {}.\n\
             Use the same model that was used to create the index.",
            dimension,
            encoder_dim
        ));
    }

    // 3. Process new documents
    let added = process_inputs(
        &mut writer,
        &encoder,
        inputs,
        chunk_size,
        chunk_overlap,
        quiet,
    )
    .await?;

    // 4. Commit
    writer.commit()?;

    if !quiet {
        eprintln!();
        eprintln!("✓ Added {} documents to index", added);
        eprintln!("  Total documents: {}", initial_count + added);
    }

    Ok(())
}

async fn process_inputs(
    writer: &mut IndexWriter,
    encoder: &SentenceEncoder,
    inputs: &[String],
    chunk_size: usize,
    chunk_overlap: usize,
    quiet: bool,
) -> Result<usize> {
    // 1. Configure Loader
    let loader_config = LoaderConfig {
        splitter: SplitterConfig {
            chunk_size,
            chunk_overlap,
            separator: "\n\n".to_string(),
        },
        ..Default::default()
    };
    // Wrap loader in Arc so threads can share it
    let loader = Arc::new(DocumentLoader::new(loader_config));

    // 2. Setup Concurrency
    // Channel for sending loaded chunks to the main encoder loop
    let (tx, mut rx) = mpsc::channel::<Vec<Chunk>>(64); 
    
    // Semaphore to limit concurrent file reads (adjust based on your CPU/IO)
    let concurrency_limit = 16; 
    let semaphore = Arc::new(Semaphore::new(concurrency_limit));

    // 3. Spawn Producer Task (Walks files and spawns workers)
    let inputs_owned = inputs.to_vec();
    let loader_ref = loader.clone();
    let quiet_clone = quiet;
    
    tokio::spawn(async move {
        for input in inputs_owned {
            let path = Path::new(&input);
            if !path.exists() { continue; }

            for entry in walkdir::WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
                if !entry.file_type().is_file() { continue; }
                
                // Only process supported extensions
                if !loader_ref.is_supported_extension(entry.path()) { continue; }

                // Acquire permit (backpressure)
                let permit = match semaphore.clone().acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => break,
                };

                let loader = loader_ref.clone();
                let tx = tx.clone();
                let file_path = entry.path().to_owned();
                let quiet = quiet_clone;

                // Spawn a worker for this file
                tokio::spawn(async move {
                    // Move permit into task (it drops when task finishes)
                    let _p = permit; 
                    
                    if !quiet {
                        // eprintln!("Processing: {:?}", file_path); // Optional: verbose logging
                    }

                    // Run CPU-intensive split in blocking thread to avoid stalling async runtime
                    let result = tokio::task::spawn_blocking(move || {
                        loader.load_file(&file_path)
                    }).await;

                    if let Ok(Ok(chunks)) = result {
                        if !chunks.is_empty() {
                            let _ = tx.send(chunks).await;
                        }
                    }
                });
            }
        }
        // tx drops here, which eventually closes the channel and stops the receiver loop
    });

    // 4. Consumer Loop (Main thread: Batching -> Encoding -> Writing)
    let mut batch_texts: Vec<String> = Vec::with_capacity(ENCODE_BATCH_SIZE);
    let mut batch_metadata: Vec<HashMap<String, String>> = Vec::with_capacity(ENCODE_BATCH_SIZE);
    let mut total_indexed = 0;

    // Receive chunks as they finish processing
    while let Some(chunks) = rx.recv().await {
        for chunk in chunks {
            batch_texts.push(chunk.text);
            batch_metadata.push(chunk.metadata.to_hashmap());

            if batch_texts.len() >= ENCODE_BATCH_SIZE {
                total_indexed += flush_batch(
                    writer,
                    encoder,
                    &mut batch_texts,
                    &mut batch_metadata,
                ).await?;

                if !quiet {
                    eprint!("\r  Indexed {} documents", total_indexed);
                }
            }
        }
    }

    // Process remaining batch
    if !batch_texts.is_empty() {
        total_indexed += flush_batch(
            writer,
            encoder,
            &mut batch_texts,
            &mut batch_metadata,
        ).await?;
    }

    if !quiet && total_indexed > 0 {
        eprintln!("\r  Indexed {} documents", total_indexed);
    }

    Ok(total_indexed)
}

/// Helper function to process a single file and handle batch flushing
/// This replaces the closure and allows async/await
async fn process_file_step(
    writer: &mut IndexWriter,
    encoder: &SentenceEncoder,
    loader: &DocumentLoader,
    path: &Path,
    batch_texts: &mut Vec<String>,
    batch_metadata: &mut Vec<HashMap<String, String>>,
    total_indexed: &mut usize,
    quiet: bool,
) -> Result<()> {
    // Optional: Check extension before loading to save IO
    if !loader.is_supported_extension(path) {
        return Ok(());
    }
    if !quiet {
        eprintln!("Reading file: {:?}", path); 
    }

    // Load only THIS file into memory
    let chunks = match loader.load_file(path) {
        Ok(c) => c,
        Err(_) => return Ok(()), // Skip failed files
    };

    for chunk in chunks {
        batch_texts.push(chunk.text);
        batch_metadata.push(chunk.metadata.to_hashmap());

        // Check if batch is full
        if batch_texts.len() >= ENCODE_BATCH_SIZE {
            *total_indexed += flush_batch(
                writer,
                encoder,
                batch_texts,
                batch_metadata,
            ).await?;

            if !quiet {
                eprint!("\r  Indexed {} documents", *total_indexed);
            }
        }
    }

    Ok(())
}

/// Encode a batch and write to index
async fn flush_batch(
    writer: &mut IndexWriter,
    encoder: &SentenceEncoder,
    texts: &mut Vec<String>,
    metadata: &mut Vec<HashMap<String, String>>,
) -> Result<usize> {
    let count = texts.len();

    // Encode batch
    let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let embeddings = encoder.encode_batch(&texts_ref).await?;

    // Write to index
    for ((text, meta), embedding) in texts.drain(..).zip(metadata.drain(..)).zip(embeddings) {
        writer.add(&text, &embedding, Some(&meta))?;
    }

    Ok(count)
}

/// Show index information
fn info(index_path: &str) -> Result<()> {
    let reader = IndexReader::open(index_path)?;

    println!();
    println!("Index: {}", index_path);
    println!("{}", "-".repeat(50));
    println!("Documents:      {}", reader.len());
    println!("Segments:       {}", reader.segment_count());
    println!("Dimension:      {}", reader.dimension());

    if let Ok(size) = calculate_index_size(index_path) {
        println!("Total size:     {}", format_size(size));
    }

    // Show sample documents
    if !reader.is_empty() {
        println!();
        println!("Sample documents:");
        for i in 0..3.min(reader.len()) {
            if let Ok(doc) = reader.get_document(i) {
                println!("  [{}] {}", i, truncate(&doc, 60));
            }
        }
    }

    // Show segment details
    println!();
    println!("Segments:");
    let segments_dir = Path::new(index_path).join("segments");
    if segments_dir.exists() {
        let mut entries: Vec<_> = fs::read_dir(&segments_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let meta_path = entry.path().join("segment.json");
            if let Ok(content) = fs::read_to_string(&meta_path) {
                if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&content) {
                    let doc_count = meta["doc_count"].as_u64().unwrap_or(0);
                    let name = entry.file_name();
                    println!("  {:?}: {} docs", name, doc_count);
                }
            }
        }
    }

    println!();
    Ok(())
}

/// Load and prepare encoder model
async fn load_encoder(model: &str, gpu: bool, quiet: bool) -> Result<SentenceEncoder> {
    let device = if gpu { Device::Wgpu } else { Device::Cpu };

    let model_type = ModelType::from_cli_name(model).ok_or_else(|| {
        let mut msg = format!("Unknown model: '{}'.", model);
        let suggestions = ModelType::find_similar(model);
        if !suggestions.is_empty() {
            msg.push_str("\n\nDid you mean?");
            for (name, _) in suggestions.iter().take(3) {
                msg.push_str(&format!("\n  - {}", name));
            }
        }
        anyhow!(msg)
    })?;

    if model_type.architecture() != ModelArchitecture::Bert {
        return Err(anyhow!(
            "Model '{}' is not an encoder. Use an encoder model like minilm-l6-v2.",
            model
        ));
    }

    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("Downloading model '{}'...", model);
        }
        registry::download_model(model, false).await?;
    }

    if !quiet {
        eprintln!("Loading encoder '{}'...", model);
    }

    SentenceEncoder::from_registry(model_type, None, device, None, None).await
}

/// Calculate total size of index directory
fn calculate_index_size(path: &str) -> Result<u64> {
    let mut total = 0u64;

    for entry in walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            total += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }

    Ok(total)
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} bytes", bytes)
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ").replace('\t', " ");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
