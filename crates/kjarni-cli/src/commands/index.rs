//! Index management commands
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::Path;

use kjarni::embedder::Embedder;
use kjarni::indexer::Indexer;
#[cfg(test)]
use kjarni::{DocumentLoader, LoaderConfig};
use kjarni::{
    IndexBuild, IndexConfig, IndexReader, IndexWriter, ProgressStage, SplitterConfig, is_index_dir,
};
use kjarni_cli::IndexCommands;

const ENCODE_BATCH_SIZE: usize = 32;

/// For indexes that did not record their model.
const DEFAULT_EMBEDDING_MODEL: &str = "minilm-l6-v2";

pub async fn run(action: IndexCommands) -> Result<()> {
    match action {
        IndexCommands::Create {
            output,
            from_chunks,
            inputs,
            chunk_size,
            chunk_overlap,
            model,
            gpu,
            quiet,
        } => {
            create(
                &output,
                from_chunks,
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
                model.as_deref(),
                gpu,
                quiet,
            )
            .await
        }
        IndexCommands::Remove {
            index_path,
            paths,
            quiet,
        } => remove(&index_path, &paths, quiet),
        IndexCommands::Info { index_path } => info(&index_path),
    }
}

/// Create a new index from documents
async fn create(
    output: &str,
    from_chunks: Option<String>,
    inputs: &[String],
    chunk_size: usize,
    chunk_overlap: usize,
    model: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    if inputs.is_empty() && from_chunks.is_none() {
        return Err(anyhow!(
            "No input files specified. Provide files or directories to index, \
             or --from-chunks with a JSONL file."
        ));
    }

    validate_chunking(chunk_size, chunk_overlap)?;
    check_output_is_free(output)?;
    check_inputs_exist(inputs)?;

    match &from_chunks {
        Some(path) => create_from_chunks(output, path, model, gpu, quiet).await,
        None => {
            create_from_inputs(output, inputs, chunk_size, chunk_overlap, model, gpu, quiet).await
        }
    }
}

async fn create_from_inputs(
    output: &str,
    inputs: &[String],
    chunk_size: usize,
    chunk_overlap: usize,
    model: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    let indexer = build_indexer(model, gpu, chunk_size, chunk_overlap, quiet).await?;
    if !quiet {
        eprintln!("Model: {}", indexer.model_name());
        eprintln!("Dimension: {}", indexer.dimension());
    }

    let input_refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
    let stats = indexer.create(output, &input_refs).await?;
    finish_progress_line(quiet);

    if stats.documents_indexed == 0 {
        Indexer::delete(output)?;
        return Err(anyhow!("No documents found to index."));
    }

    print_created(output, stats.documents_indexed, stats.dimension, quiet);
    Ok(())
}

async fn create_from_chunks(
    output: &str,
    chunks_path: &str,
    model: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    let embedder = load_embedder(model, gpu, quiet).await?;
    let dimension = embedder.dimension();

    if !quiet {
        eprintln!("Model: {}", embedder.model_name());
        eprintln!("Dimension: {}", dimension);
    }

    let config = IndexConfig {
        dimension,
        max_docs_per_segment: 10_000,
        embedding_model: Some(embedder.model_name().to_string()),
        created_at: Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        ),
        ..Default::default()
    };

    // Built beside the output and moved there once complete, so a failed run leaves
    // nothing behind.
    let build = IndexBuild::new(output)?;
    let mut writer = IndexWriter::open(build.path(), config)?;

    let total_indexed = process_chunk_file(&mut writer, &embedder, chunks_path, quiet).await?;
    if total_indexed == 0 {
        return Err(anyhow!("No documents found to index."));
    }

    writer.commit()?;
    build.install()?;

    print_created(output, total_indexed, dimension, quiet);
    Ok(())
}

fn print_created(output: &str, documents: usize, dimension: usize, quiet: bool) {
    if quiet {
        return;
    }
    eprintln!();
    eprintln!("✓ Index created: {}", output);
    eprintln!("  Documents: {}", documents);
    eprintln!("  Dimension: {}", dimension);
    if let Ok(size) = calculate_index_size(output) {
        eprintln!("  Size: {}", format_size(size));
    }
}

async fn build_indexer(
    model: &str,
    gpu: bool,
    chunk_size: usize,
    chunk_overlap: usize,
    quiet: bool,
) -> Result<Indexer> {
    let mut builder = Indexer::builder(model)
        .chunk_size(chunk_size)
        .chunk_overlap(chunk_overlap)
        .max_file_size(usize::MAX)
        .quiet(quiet);
    for pattern in super::mcp::EXCLUDED {
        builder = builder.exclude(pattern);
    }
    if gpu {
        builder = builder.gpu();
    }
    // Carriage-return progress only makes sense on a terminal. Redirected it would run
    // every update together on one line in the log.
    if !quiet && std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        builder = builder.on_progress(|progress, _| {
            if matches!(progress.stage, ProgressStage::Embedding) {
                eprint!("\r  Indexed {} documents", progress.current);
            }
        });
    }
    Ok(builder.build().await?)
}

fn finish_progress_line(quiet: bool) {
    if !quiet && std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        eprintln!();
    }
}

fn validate_chunking(chunk_size: usize, chunk_overlap: usize) -> Result<()> {
    SplitterConfig {
        chunk_size,
        chunk_overlap,
        clean_markdown: true,
    }
    .validate()
    .map_err(|e| anyhow!("{e} (--chunk-size {chunk_size}, --chunk-overlap {chunk_overlap})"))
}

fn check_output_is_free(output: &str) -> Result<()> {
    let path = Path::new(output);
    if is_index_dir(path) {
        return Err(anyhow!(
            "{output} already holds an index. Use `kjarni index add` to add to it, \
             or delete it to start again."
        ));
    }
    if path.exists() && !std::fs::read_dir(path).is_ok_and(|mut d| d.next().is_none()) {
        return Err(anyhow!("{output} exists and is not an empty folder."));
    }
    Ok(())
}

fn check_inputs_exist(inputs: &[String]) -> Result<()> {
    match inputs.iter().find(|input| !Path::new(input).exists()) {
        Some(input) => Err(anyhow!("Input not found: {input}")),
        None => Ok(()),
    }
}

/// Add files to an existing index, replacing any it already holds
async fn add(
    index_path: &str,
    inputs: &[String],
    chunk_size: usize,
    chunk_overlap: usize,
    model: Option<&str>,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    if inputs.is_empty() {
        return Err(anyhow!("No input files specified."));
    }
    validate_chunking(chunk_size, chunk_overlap)?;
    check_inputs_exist(inputs)?;

    let reader = IndexReader::open(index_path)?;
    let initial_count = reader.len();
    let recorded = reader.embedding_model().map(str::to_string);
    drop(reader);

    if !quiet {
        eprintln!("Opened index with {} documents", initial_count);
    }

    let model = model
        .map(str::to_string)
        .or(recorded)
        .unwrap_or_else(|| DEFAULT_EMBEDDING_MODEL.to_string());
    let indexer = build_indexer(&model, gpu, chunk_size, chunk_overlap, quiet).await?;

    let input_refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
    let added = indexer.add(index_path, &input_refs).await?;
    finish_progress_line(quiet);

    if !quiet {
        let total = IndexReader::open(index_path)?.len();
        eprintln!();
        eprintln!("✓ Indexed {} documents", added);
        eprintln!("  Total documents: {}", total);
    }

    Ok(())
}

/// Remove files or folders from an index
fn remove(index_path: &str, paths: &[String], quiet: bool) -> Result<()> {
    let path_refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    let removed = Indexer::remove(index_path, &path_refs)?;
    if !quiet {
        eprintln!("✓ Removed {} documents", removed);
    }
    Ok(())
}

/// Indexes a JSONL file of already-chunked documents, one object per line.
///
/// `{"text": "...", "metadata": {"k": "v"}}`, where `metadata` is optional and its
/// values are strings. Chunking is skipped entirely, which is the point: a caller
/// who has already split documents their own way should not have them re-split.
///
/// The flag existed and was accepted for some time while the handler ignored it,
/// so `--from-chunks` parsed and then failed with "No input files specified".
async fn process_chunk_file(
    writer: &mut IndexWriter,
    embedder: &Embedder,
    path: &str,
    quiet: bool,
) -> Result<usize> {
    use std::io::BufRead;

    let file =
        std::fs::File::open(path).map_err(|e| anyhow!("could not open chunk file {path}: {e}"))?;
    let reader = std::io::BufReader::new(file);

    let mut texts: Vec<String> = Vec::new();
    let mut metadata: Vec<HashMap<String, String>> = Vec::new();
    let mut total = 0usize;

    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| anyhow!("reading {path} line {}: {e}", lineno + 1))?;
        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| anyhow!("{path} line {}: {e}", lineno + 1))?;

        // A line without text is a mistake worth naming rather than skipping: it
        // would otherwise show up later as an index that is quietly short.
        let text = value
            .get("text")
            .and_then(|t| t.as_str())
            .ok_or_else(|| anyhow!("{path} line {}: no \"text\" field", lineno + 1))?;

        let mut meta = HashMap::new();
        if let Some(obj) = value.get("metadata").and_then(|m| m.as_object()) {
            for (k, v) in obj {
                if let Some(s) = v.as_str() {
                    meta.insert(k.clone(), s.to_string());
                }
            }
        }

        texts.push(text.to_string());
        metadata.push(meta);

        if texts.len() >= ENCODE_BATCH_SIZE {
            total += flush_batch(writer, embedder, &mut texts, &mut metadata).await?;
            if !quiet && std::io::IsTerminal::is_terminal(&std::io::stderr()) {
                eprint!("\r  Indexed {total} documents");
            }
        }
    }

    if !texts.is_empty() {
        total += flush_batch(writer, embedder, &mut texts, &mut metadata).await?;
    }

    if !quiet && total > 0 {
        if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            eprintln!("\r  Indexed {total} documents");
        } else {
            eprintln!("  Indexed {total} documents");
        }
    }
    Ok(total)
}

/// Encode a batch and write to index
async fn flush_batch(
    writer: &mut IndexWriter,
    embedder: &Embedder,
    texts: &mut Vec<String>,
    metadata: &mut Vec<HashMap<String, String>>,
) -> Result<usize> {
    let count = texts.len();

    let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed_batch(&texts_ref).await?;

    for ((text, meta), embedding) in texts.drain(..).zip(metadata.drain(..)).zip(embeddings) {
        writer.add(&text, &embedding, Some(&meta))?;
    }

    Ok(count)
}

/// Show index information
fn info(index_path: &str) -> Result<()> {
    let reader = IndexReader::open(index_path)?;
    let output = format_index_info(index_path, &reader)?;
    print!("{}", output);
    Ok(())
}

/// Format index information as a string
fn format_index_info(index_path: &str, reader: &IndexReader) -> Result<String> {
    let mut output = String::new();

    output.push('\n');
    output.push_str(&format!("Index: {}\n", index_path));
    output.push_str(&format!("{}\n", "-".repeat(50)));
    output.push_str(&format!("Documents:      {}\n", reader.len()));
    output.push_str(&format!("Segments:       {}\n", reader.segment_count()));
    output.push_str(&format!("Dimension:      {}\n", reader.dimension()));

    if let Ok(size) = calculate_index_size(index_path) {
        output.push_str(&format!("Total size:     {}\n", format_size(size)));
    }

    if !reader.is_empty() {
        output.push('\n');
        output.push_str("Sample documents:\n");
        for i in 0..3.min(reader.len()) {
            if let Ok(doc) = reader.get_document(i) {
                output.push_str(&format!("  [{}] {}\n", i, truncate(&doc, 60)));
            }
        }
    }

    // Show segment details
    output.push('\n');
    output.push_str("Segments:\n");
    let segments_dir = Path::new(index_path).join("segments");
    if segments_dir.exists() {
        let mut entries: Vec<_> = std::fs::read_dir(&segments_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let meta_path = entry.path().join("segment.json");
            if let Ok(content) = std::fs::read_to_string(&meta_path)
                && let Ok(meta) = serde_json::from_str::<serde_json::Value>(&content)
            {
                let doc_count = meta["doc_count"].as_u64().unwrap_or(0);
                let name = entry.file_name();
                output.push_str(&format!("  {:?}: {} docs\n", name, doc_count));
            }
        }
    }

    output.push('\n');
    Ok(output)
}

async fn load_embedder(model: &str, gpu: bool, quiet: bool) -> Result<Embedder> {
    let mut builder = Embedder::builder(model).quiet(quiet);

    if gpu {
        builder = builder.gpu();
    } else {
        builder = builder.cpu();
    }

    builder.build().await.map_err(|e| anyhow!(e))
}

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
    let s = s.replace(['\n', '\t'], " ");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// A loader with the exclusions `build_indexer` applies.
#[cfg(test)]
fn document_loader(chunk_size: usize, chunk_overlap: usize, quiet: bool) -> DocumentLoader {
    DocumentLoader::new(LoaderConfig {
        splitter: SplitterConfig {
            chunk_size,
            chunk_overlap,
            clean_markdown: true,
        },
        quiet,
        exclude_patterns: super::mcp::EXCLUDED.iter().map(|p| p.to_string()).collect(),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folders_skip_hidden_and_build_output() {
        let dir = tempfile::tempdir().unwrap();
        for file in [
            "keep/note.md",
            ".hidden/secret.md",
            "target/build.md",
            "node_modules/pkg/readme.md",
        ] {
            let path = dir.path().join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "some indexable text").unwrap();
        }

        let files = document_loader(100, 10, true).files_in(dir.path());

        assert_eq!(files, [dir.path().join("keep/note.md")]);
    }

    #[test]
    fn a_hidden_file_named_directly_is_indexed() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(".notes.md");
        std::fs::write(&file, "some indexable text").unwrap();

        assert_eq!(document_loader(100, 10, true).files_in(&file), [file]);
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 bytes");
        assert_eq!(format_size(1), "1 bytes");
        assert_eq!(format_size(512), "512 bytes");
        assert_eq!(format_size(999), "999 bytes");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1_000), "1.00 KB");
        assert_eq!(format_size(1_500), "1.50 KB");
        assert_eq!(format_size(10_000), "10.00 KB");
        assert_eq!(format_size(999_999), "1000.00 KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1_000_000), "1.00 MB");
        assert_eq!(format_size(1_500_000), "1.50 MB");
        assert_eq!(format_size(100_000_000), "100.00 MB");
        assert_eq!(format_size(999_999_999), "1000.00 MB");
    }

    #[test]
    fn test_format_size_gigabytes() {
        assert_eq!(format_size(1_000_000_000), "1.00 GB");
        assert_eq!(format_size(1_500_000_000), "1.50 GB");
        assert_eq!(format_size(10_000_000_000), "10.00 GB");
        assert_eq!(format_size(100_000_000_000), "100.00 GB");
    }

    #[test]
    fn test_format_size_edge_cases() {
        // Just under KB threshold
        assert_eq!(format_size(999), "999 bytes");
        // Just at KB threshold
        assert_eq!(format_size(1_000), "1.00 KB");
        // Just under MB threshold
        assert_eq!(format_size(999_999), "1000.00 KB");
        // Just at MB threshold
        assert_eq!(format_size(1_000_000), "1.00 MB");
    }

    #[test]
    fn test_format_size_precision() {
        assert_eq!(format_size(1_234_567), "1.23 MB");
        assert_eq!(format_size(1_235_000), "1.24 MB"); // Rounding
        assert_eq!(format_size(1_234_567_890), "1.23 GB");
    }

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate("hello", 5), "hello");
        assert_eq!(truncate("hello world", 11), "hello world");
    }

    #[test]
    fn test_truncate_long_string() {
        assert_eq!(truncate("hello world", 8), "hello...");
        assert_eq!(truncate("this is a very long string", 10), "this is...");
    }

    #[test]
    fn test_truncate_replaces_newlines() {
        assert_eq!(truncate("hello\nworld", 20), "hello world");
        assert_eq!(truncate("line1\nline2\nline3", 20), "line1 line2 line3");
    }

    #[test]
    fn test_truncate_replaces_tabs() {
        assert_eq!(truncate("hello\tworld", 20), "hello world");
        assert_eq!(truncate("col1\tcol2\tcol3", 20), "col1 col2 col3");
    }

    #[test]
    fn test_truncate_mixed_whitespace() {
        assert_eq!(truncate("hello\n\tworld", 20), "hello  world");
        assert_eq!(truncate("a\nb\tc\nd", 10), "a b c d");
    }

    #[test]
    fn test_truncate_with_newlines_then_truncate() {
        let input = "line one\nline two\nline three";
        let result = truncate(input, 15);
        assert_eq!(result, "line one lin...");
    }

    #[test]
    fn test_truncate_empty_string() {
        assert_eq!(truncate("", 10), "");
        assert_eq!(truncate("", 0), "");
    }

    #[test]
    fn test_truncate_unicode() {
        assert_eq!(truncate("héllo", 10), "héllo");
        assert_eq!(truncate("日本語テスト", 20), "日本語テスト");
    }

    #[test]
    fn test_truncate_only_whitespace() {
        assert_eq!(truncate("\n\n\n", 10), "   ");
        assert_eq!(truncate("\t\t\t", 10), "   ");
    }

    #[test]
    fn test_truncate_minimum_length() {
        assert_eq!(truncate("hello", 4), "h...");
    }

    #[test]
    fn test_truncate_realistic_document() {
        let doc = "This is a document with multiple paragraphs.\n\nIt has some content that spans several lines.\n\nAnd more content here.";
        let result = truncate(doc, 60);

        assert_eq!(result.len(), 60);
        assert!(result.ends_with("..."));
        assert!(!result.contains('\n'));
    }

    #[test]
    fn test_format_size_realistic_file_sizes() {
        // Common file sizes
        assert_eq!(format_size(4_096), "4.10 KB"); // 4KB block
        assert_eq!(format_size(65_536), "65.54 KB"); // 64KB
        assert_eq!(format_size(1_048_576), "1.05 MB"); // 1 MiB
        assert_eq!(format_size(104_857_600), "104.86 MB"); // 100 MiB
        assert_eq!(format_size(1_073_741_824), "1.07 GB"); // 1 GiB
    }
    #[test]
    fn test_encode_batch_size_is_reasonable() {
        const { assert!(ENCODE_BATCH_SIZE > 0) };
        const { assert!(ENCODE_BATCH_SIZE <= 128) };
        assert_eq!(ENCODE_BATCH_SIZE, 32);
    }

    /// The JSONL shape `--from-chunks` accepts. Parsing is asserted separately
    /// from indexing so a malformed line can be checked without loading a model.
    fn parse_line(line: &str) -> Result<(String, HashMap<String, String>)> {
        let value: serde_json::Value = serde_json::from_str(line)?;
        let text = value
            .get("text")
            .and_then(|t| t.as_str())
            .ok_or_else(|| anyhow!("no \"text\" field"))?;
        let mut meta = HashMap::new();
        if let Some(obj) = value.get("metadata").and_then(|m| m.as_object()) {
            for (k, v) in obj {
                if let Some(sv) = v.as_str() {
                    meta.insert(k.clone(), sv.to_string());
                }
            }
        }
        Ok((text.to_string(), meta))
    }

    #[test]
    fn a_chunk_line_needs_only_text() {
        let (text, meta) = parse_line(r#"{"text": "hello"}"#).unwrap();
        assert_eq!(text, "hello");
        assert!(meta.is_empty(), "metadata is optional");
    }

    #[test]
    fn string_metadata_is_carried_through() {
        let (_, meta) =
            parse_line(r#"{"text": "x", "metadata": {"source": "a.txt", "page": "3"}}"#).unwrap();
        assert_eq!(meta.get("source").map(String::as_str), Some("a.txt"));
        assert_eq!(meta.get("page").map(String::as_str), Some("3"));
    }

    /// Non-string metadata values are dropped rather than stringified, because the
    /// index stores strings and guessing a format for a number or an object would
    /// be inventing data.
    #[test]
    fn non_string_metadata_is_dropped_not_coerced() {
        let (_, meta) =
            parse_line(r#"{"text": "x", "metadata": {"n": 3, "ok": "yes", "sub": {"a": 1}}}"#)
                .unwrap();
        assert_eq!(meta.len(), 1);
        assert_eq!(meta.get("ok").map(String::as_str), Some("yes"));
    }

    /// A line with no text would otherwise become an index that is quietly short.
    #[test]
    fn a_line_without_text_is_an_error() {
        let err = parse_line(r#"{"body": "oops"}"#).unwrap_err().to_string();
        assert!(err.contains("text"), "{err}");
    }

    #[test]
    fn malformed_json_is_an_error() {
        assert!(parse_line("{not json").is_err());
        assert!(parse_line("").is_err());
    }
}
