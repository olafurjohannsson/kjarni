use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "kjarni")]
#[command(about = "Kjarni: The SQLite of AI", long_about = None)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,
}

#[derive(Subcommand, Debug, PartialEq)]
pub enum Commands {
    /// Manage models (list, download, info)
    Model {
        #[command(subcommand)]
        action: ModelCommands,
    },

    /// Generate text from a prompt
    Generate {
        /// The prompt (or file path, or stdin if not provided)
        prompt: Option<String>,

        #[arg(short, long, default_value = "llama-3.2-1b")]
        model: String,

        #[arg(long)]
        model_path: Option<String>,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value_t = 100)]
        max_tokens: usize,

        /// Sampling temperature (0.0 = greedy, higher = more random)
        #[arg(short, long, default_value_t = 0.7)]
        temperature: f32,

        /// Top-K sampling (limits to K most likely tokens)
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-P (nucleus) sampling threshold
        #[arg(long)]
        top_p: Option<f32>,

        /// Min-P sampling threshold  
        #[arg(long)]
        min_p: Option<f32>,

        /// Repetition penalty (1.0 = no penalty)
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,

        /// Use greedy decoding (ignores temperature)
        #[arg(long)]
        greedy: bool,

        /// Use GPU
        #[arg(long)]
        gpu: bool,

        /// Disable streaming output
        #[arg(long)]
        no_stream: bool,

        /// Suppress status messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Summarize text
    Summarize {
        /// Input text (or read from stdin)
        #[arg(short, long)]
        input: Option<String>,

        /// Model to use
        #[arg(short, long, default_value = "distilbart-cnn")]
        model: String,

        /// Path to local model
        #[arg(long)]
        model_path: Option<String>,

        /// Minimum summary length
        #[arg(long)]
        min_length: Option<usize>,

        /// Maximum summary length
        #[arg(long)]
        max_length: Option<usize>,

        /// Number of beams for beam search
        #[arg(long)]
        num_beams: Option<usize>,

        /// Length penalty for beam search (< 1 shorter, > 1 longer)
        #[arg(long)]
        length_penalty: Option<f32>,

        /// Block repeated n-grams of this size
        #[arg(long)]
        no_repeat_ngram: Option<usize>,

        /// Use greedy decoding (deterministic, fastest)
        #[arg(long)]
        greedy: bool,

        /// Disable streaming output
        #[arg(long)]
        no_stream: bool,

        /// Use GPU
        #[arg(long)]
        gpu: bool,

        /// Suppress progress messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Translate text between languages
    Translate {
        /// Input text (or read from stdin)
        #[arg(short, long)]
        input: Option<String>,

        /// Model to use
        #[arg(short, long, default_value = "flan-t5-base")]
        model: String,

        /// Path to local model
        #[arg(long)]
        model_path: Option<String>,

        /// Source language (e.g., en, de, fr)
        #[arg(long)]
        src: Option<String>,

        /// Target language (e.g., en, de, fr)
        #[arg(long)]
        dst: Option<String>,

        /// Maximum output length
        #[arg(long)]
        max_length: Option<usize>,

        /// Number of beams for beam search
        #[arg(long)]
        num_beams: Option<usize>,

        /// Length penalty for beam search (< 1 shorter, > 1 longer)
        #[arg(long)]
        length_penalty: Option<f32>,

        /// Block repeated n-grams of this size
        #[arg(long)]
        no_repeat_ngram: Option<usize>,

        /// Use greedy decoding (deterministic, fastest)
        #[arg(long)]
        greedy: bool,

        /// Disable streaming output
        #[arg(long)]
        no_stream: bool,

        /// Use GPU
        #[arg(long)]
        gpu: bool,

        /// Suppress progress messages
        #[arg(short, long)]
        quiet: bool,
    },

    Embed {
        /// Input text, file path, or stdin if not provided
        input: Option<String>,

        #[arg(short, long, default_value = "minilm-l6-v2")]
        model: String,

        #[arg(long)]
        model_path: Option<String>,

        #[arg(long, default_value = "raw")]
        format: String,

        #[arg(long)]
        normalize: bool,

        #[arg(long, default_value = "cls")]
        pooling: String,

        #[arg(long)]
        gpu: bool,

        /// Suppress status messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Transcribe audio to text
    Transcribe {
        /// Path to audio file (wav, mp3, flac, ogg)
        file: String,

        /// Model to use (whisper-small, whisper-large-v3)
        #[arg(short, long, default_value = "whisper-small")]
        model: String,

        /// Path to local model directory (not yet implemented)
        #[arg(long)]
        model_path: Option<String>,

        /// Language code (e.g., en, fr, de). Omit for auto-detect.
        #[arg(short, long)]
        language: Option<String>,

        /// Translate to English instead of transcribing
        #[arg(long)]
        translate: bool,

        /// Include timestamps in output
        #[arg(short, long)]
        timestamps: bool,

        /// Maximum tokens per 30-second chunk
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Disable streaming (wait for full result)
        #[arg(long)]
        no_stream: bool,

        /// Use GPU acceleration
        #[arg(long)]
        gpu: bool,

        /// Suppress progress output
        #[arg(short, long)]
        quiet: bool,
    },

    /// Classify text using a classification model
    Classify {
        /// Input text(s) to classify. Use - for stdin.
        input: Vec<String>,

        /// Model name from registry
        #[arg(short, long, default_value = "distilbert-sentiment")]
        model: String,

        /// Load model from local path instead of registry
        #[arg(long, value_name = "PATH")]
        model_path: Option<String>,

        /// Custom labels (comma-separated, order must match model output)
        /// Example: --labels "negative,positive" or --labels "neikvætt,jákvætt"
        #[arg(long, value_name = "LABELS")]
        labels: Option<String>,

        /// Return top K predictions
        #[arg(long, default_value = "5")]
        top_k: usize,

        /// Minimum confidence threshold (0.0-1.0)
        #[arg(long)]
        threshold: Option<f32>,

        /// Maximum sequence length (truncates longer inputs)
        #[arg(long)]
        max_length: Option<usize>,

        /// Batch size for inference
        #[arg(long)]
        batch_size: Option<usize>,

        /// Use multi-label classification (sigmoid instead of softmax)
        #[arg(long)]
        multi_label: bool,

        /// Output format: json, jsonl, text
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Run on GPU
        #[arg(long)]
        gpu: bool,

        /// Model precision: f32, f16, bf16
        #[arg(long)]
        dtype: Option<String>,

        /// Suppress progress output
        #[arg(short, long)]
        quiet: bool,
    },

    /// Rerank documents by relevance to a query
    Rerank {
        /// The query to rank against
        query: String,

        /// Documents to rerank (or read from stdin, one per line)
        documents: Vec<String>,

        #[arg(short, long, default_value = "minilm-l6-v2-cross-encoder")]
        model: String,

        #[arg(long)]
        model_path: Option<String>,

        /// Return only top K results
        #[arg(short = 'k', long)]
        top_k: Option<usize>,

        /// Output format: json, jsonl, text, docs
        #[arg(short, long, default_value = "text")]
        format: String,

        #[arg(long)]
        gpu: bool,

        #[arg(short, long)]
        quiet: bool,
    },

    /// Interactive chat mode
    Chat {
        #[arg(short, long, default_value = "llama-3.2-8b-instruct")]
        model: String,

        #[arg(long)]
        model_path: Option<String>,

        /// System prompt to set assistant behavior
        #[arg(short, long)]
        system: Option<String>,

        /// Sampling temperature
        #[arg(short, long, default_value_t = 0.7)]
        temperature: f32,

        /// Max tokens per response
        #[arg(short = 'n', long, default_value_t = 512)]
        max_tokens: usize,

        #[arg(long)]
        gpu: bool,

        #[arg(short, long)]
        quiet: bool,
    },

    /// Create or manage search indexes
    Index {
        #[command(subcommand)]
        action: IndexCommands,
    },

    /// Search an index
    Search {
        /// Path to the index file
        index_path: String,

        /// Search query
        query: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value_t = 10)]
        top_k: usize,

        /// Search mode: hybrid, semantic, keyword
        #[arg(long, default_value = "hybrid")]
        mode: String,

        /// Encoder model for semantic search
        #[arg(short, long, default_value = "minilm-l6-v2")]
        model: String,

        /// Reranking model (optional)
        /// Use a cross-encoder model to rerank initial results
        /// Example: --rerank-model "ms-marco-minilm"
        #[arg(long, default_value = None)]
        rerank_model: Option<String>,

        /// Output format: json, jsonl, text
        #[arg(short, long, default_value = "text")]
        format: String,

        #[arg(long)]
        gpu: bool,

        #[arg(short, long)]
        quiet: bool,
    },

    /// Compute similarity between two texts
    Similarity {
        /// First text (or file path)
        text1: String,

        /// Second text (or file path)
        text2: String,

        /// Encoder model
        #[arg(short, long, default_value = "minilm-l6-v2")]
        model: String,

        #[arg(long)]
        gpu: bool,

        #[arg(short, long)]
        quiet: bool,
    },
}

#[derive(Subcommand, Debug, PartialEq)]
pub enum ModelCommands {
    /// List all available models
    List {
        /// Filter by architecture (e.g., llama, bert, t5)
        #[arg(short, long)]
        arch: Option<String>,

        /// Filter by task (e.g., chat, embedding, classification, summarization)
        #[arg(short, long)]
        task: Option<String>,

        /// Show only downloaded models
        #[arg(short, long)]
        downloaded: bool,
    },

    /// Download a model
    Download {
        name: String,

        #[arg(long)]
        gguf: bool,

        #[arg(short, long)]
        quiet: bool,
    },

    /// Remove a downloaded model
    Remove { name: String },

    /// Show detailed info about a model
    Info { name: String },

    /// Search for models by name or description
    Search { query: String },
}

#[derive(Subcommand, Debug, PartialEq)]
pub enum IndexCommands {
    /// Create a new index from documents
    Create {
        /// Output index file path
        output: String,

        /// Input files/directories (uses built-in chunking)
        #[arg(conflicts_with = "from_chunks")]
        inputs: Vec<String>,

        /// Pre-chunked JSONL file (bypass chunking)
        #[arg(long, conflicts_with = "inputs")]
        from_chunks: Option<String>,

        /// Chunk size for splitting documents
        #[arg(long, default_value_t = 1000)]
        chunk_size: usize,

        /// Chunk overlap
        #[arg(long, default_value_t = 200)]
        chunk_overlap: usize,

        /// Encoder model for embeddings
        #[arg(short, long, default_value = "minilm-l6-v2")]
        model: String,

        #[arg(long)]
        gpu: bool,

        #[arg(short, long)]
        quiet: bool,
    },

    /// Add documents to an existing index
    Add {
        /// Index file path
        index_path: String,

        /// Input files or directories to add
        inputs: Vec<String>,

        /// Chunk size for splitting documents
        #[arg(long, default_value_t = 1000)]
        chunk_size: usize,

        #[arg(long, default_value_t = 200)]
        chunk_overlap: usize,

        #[arg(short, long, default_value = "minilm-l6-v2")]
        model: String,

        #[arg(long)]
        gpu: bool,

        #[arg(short, long)]
        quiet: bool,
    },

    /// Show index info
    Info {
        /// Index file path
        index_path: String,
    },
}

/// Convert verbosity count to log level string
pub fn verbosity_to_log_level(verbose: u8) -> &'static str {
    match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    fn parse_args(args: &[&str]) -> Result<Cli, clap::Error> {
        let mut full_args = vec!["kjarni"];
        full_args.extend(args);
        Cli::try_parse_from(full_args)
    }

    #[test]
    fn test_verbosity_to_log_level_zero() {
        assert_eq!(verbosity_to_log_level(0), "warn");
    }

    #[test]
    fn test_verbosity_to_log_level_one() {
        assert_eq!(verbosity_to_log_level(1), "info");
    }

    #[test]
    fn test_verbosity_to_log_level_two() {
        assert_eq!(verbosity_to_log_level(2), "debug");
    }

    #[test]
    fn test_verbosity_to_log_level_three() {
        assert_eq!(verbosity_to_log_level(3), "trace");
    }

    #[test]
    fn test_verbosity_to_log_level_high() {
        assert_eq!(verbosity_to_log_level(10), "trace");
        assert_eq!(verbosity_to_log_level(255), "trace");
    }

    #[test]
    fn test_generate_minimal() {
        let cli = parse_args(&["generate"]).unwrap();

        match cli.command {
            Commands::Generate {
                prompt,
                model,
                max_tokens,
                temperature,
                greedy,
                gpu,
                quiet,
                ..
            } => {
                assert!(prompt.is_none());
                assert_eq!(model, "llama-3.2-1b");
                assert_eq!(max_tokens, 100);
                assert!((temperature - 0.7).abs() < 0.001);
                assert!(!greedy);
                assert!(!gpu);
                assert!(!quiet);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_with_prompt() {
        let cli = parse_args(&["generate", "Hello world"]).unwrap();

        match cli.command {
            Commands::Generate { prompt, .. } => {
                assert_eq!(prompt, Some("Hello world".to_string()));
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_with_model() {
        let cli = parse_args(&["generate", "-m", "phi3.5-mini"]).unwrap();

        match cli.command {
            Commands::Generate { model, .. } => {
                assert_eq!(model, "phi3.5-mini");
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_with_max_tokens() {
        let cli = parse_args(&["generate", "-n", "500"]).unwrap();

        match cli.command {
            Commands::Generate { max_tokens, .. } => {
                assert_eq!(max_tokens, 500);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_with_temperature() {
        let cli = parse_args(&["generate", "-t", "1.5"]).unwrap();

        match cli.command {
            Commands::Generate { temperature, .. } => {
                assert!((temperature - 1.5).abs() < 0.001);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_with_sampling_params() {
        let cli = parse_args(&[
            "generate", "--top-k", "50", "--top-p", "0.9", "--min-p", "0.05",
        ])
        .unwrap();

        match cli.command {
            Commands::Generate {
                top_k,
                top_p,
                min_p,
                ..
            } => {
                assert_eq!(top_k, Some(50));
                assert_eq!(top_p, Some(0.9));
                assert_eq!(min_p, Some(0.05));
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_with_greedy() {
        let cli = parse_args(&["generate", "--greedy"]).unwrap();

        match cli.command {
            Commands::Generate { greedy, .. } => {
                assert!(greedy);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_with_gpu() {
        let cli = parse_args(&["generate", "--gpu"]).unwrap();

        match cli.command {
            Commands::Generate { gpu, .. } => {
                assert!(gpu);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_with_no_stream() {
        let cli = parse_args(&["generate", "--no-stream"]).unwrap();

        match cli.command {
            Commands::Generate { no_stream, .. } => {
                assert!(no_stream);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_with_quiet() {
        let cli = parse_args(&["generate", "-q"]).unwrap();

        match cli.command {
            Commands::Generate { quiet, .. } => {
                assert!(quiet);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_generate_all_options() {
        let cli = parse_args(&[
            "generate",
            "test prompt",
            "-m",
            "llama3.2-3b-instruct",
            "-n",
            "256",
            "-t",
            "0.8",
            "--top-k",
            "40",
            "--top-p",
            "0.95",
            "--repetition-penalty",
            "1.2",
            "--greedy",
            "--gpu",
            "--no-stream",
            "-q",
        ])
        .unwrap();

        match cli.command {
            Commands::Generate {
                prompt,
                model,
                max_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                greedy,
                gpu,
                no_stream,
                quiet,
                ..
            } => {
                assert_eq!(prompt, Some("test prompt".to_string()));
                assert_eq!(model, "llama3.2-3b-instruct");
                assert_eq!(max_tokens, 256);
                assert!((temperature - 0.8).abs() < 0.001);
                assert_eq!(top_k, Some(40));
                assert_eq!(top_p, Some(0.95));
                assert!((repetition_penalty - 1.2).abs() < 0.001);
                assert!(greedy);
                assert!(gpu);
                assert!(no_stream);
                assert!(quiet);
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_chat_defaults() {
        let cli = parse_args(&["chat"]).unwrap();

        match cli.command {
            Commands::Chat {
                model,
                system,
                temperature,
                max_tokens,
                gpu,
                quiet,
                ..
            } => {
                assert_eq!(model, "llama-3.2-8b-instruct");
                assert!(system.is_none());
                assert!((temperature - 0.7).abs() < 0.001);
                assert_eq!(max_tokens, 512);
                assert!(!gpu);
                assert!(!quiet);
            }
            _ => panic!("Expected Chat command"),
        }
    }

    #[test]
    fn test_chat_with_system_prompt() {
        let cli = parse_args(&["chat", "-s", "You are a helpful assistant"]).unwrap();

        match cli.command {
            Commands::Chat { system, .. } => {
                assert_eq!(system, Some("You are a helpful assistant".to_string()));
            }
            _ => panic!("Expected Chat command"),
        }
    }

    #[test]
    fn test_chat_with_model() {
        let cli = parse_args(&["chat", "-m", "phi3.5-mini"]).unwrap();

        match cli.command {
            Commands::Chat { model, .. } => {
                assert_eq!(model, "phi3.5-mini");
            }
            _ => panic!("Expected Chat command"),
        }
    }

    #[test]
    fn test_classify_defaults() {
        let cli = parse_args(&["classify"]).unwrap();

        match cli.command {
            Commands::Classify {
                input,
                model,
                top_k,
                format,
                multi_label,
                gpu,
                quiet,
                ..
            } => {
                assert!(input.is_empty());
                assert_eq!(model, "distilbert-sentiment");
                assert_eq!(top_k, 5);
                assert_eq!(format, "text");
                assert!(!multi_label);
                assert!(!gpu);
                assert!(!quiet);
            }
            _ => panic!("Expected Classify command"),
        }
    }

    #[test]
    fn test_classify_with_input() {
        let cli = parse_args(&["classify", "This is great!"]).unwrap();

        match cli.command {
            Commands::Classify { input, .. } => {
                assert_eq!(input, vec!["This is great!".to_string()]);
            }
            _ => panic!("Expected Classify command"),
        }
    }

    #[test]
    fn test_classify_with_multiple_inputs() {
        let cli = parse_args(&["classify", "text one", "text two", "text three"]).unwrap();

        match cli.command {
            Commands::Classify { input, .. } => {
                assert_eq!(input.len(), 3);
                assert_eq!(input[0], "text one");
                assert_eq!(input[1], "text two");
                assert_eq!(input[2], "text three");
            }
            _ => panic!("Expected Classify command"),
        }
    }

    #[test]
    fn test_classify_with_labels() {
        let cli = parse_args(&["classify", "--labels", "bad,good"]).unwrap();

        match cli.command {
            Commands::Classify { labels, .. } => {
                assert_eq!(labels, Some("bad,good".to_string()));
            }
            _ => panic!("Expected Classify command"),
        }
    }

    #[test]
    fn test_classify_with_multi_label() {
        let cli = parse_args(&["classify", "--multi-label"]).unwrap();

        match cli.command {
            Commands::Classify { multi_label, .. } => {
                assert!(multi_label);
            }
            _ => panic!("Expected Classify command"),
        }
    }

    #[test]
    fn test_search_minimal() {
        let cli = parse_args(&["search", "./index", "my query"]).unwrap();

        match cli.command {
            Commands::Search {
                index_path,
                query,
                top_k,
                mode,
                model,
                format,
                ..
            } => {
                assert_eq!(index_path, "./index");
                assert_eq!(query, "my query");
                assert_eq!(top_k, 10);
                assert_eq!(mode, "hybrid");
                assert_eq!(model, "minilm-l6-v2");
                assert_eq!(format, "text");
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_search_with_top_k() {
        let cli = parse_args(&["search", "./index", "query", "-k", "20"]).unwrap();

        match cli.command {
            Commands::Search { top_k, .. } => {
                assert_eq!(top_k, 20);
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_search_with_mode() {
        let cli = parse_args(&["search", "./index", "query", "--mode", "semantic"]).unwrap();

        match cli.command {
            Commands::Search { mode, .. } => {
                assert_eq!(mode, "semantic");
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_search_with_rerank_model() {
        let cli = parse_args(&[
            "search",
            "./index",
            "query",
            "--rerank-model",
            "ms-marco-minilm",
        ])
        .unwrap();

        match cli.command {
            Commands::Search { rerank_model, .. } => {
                assert_eq!(rerank_model, Some("ms-marco-minilm".to_string()));
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_similarity_minimal() {
        let cli = parse_args(&["similarity", "text one", "text two"]).unwrap();

        match cli.command {
            Commands::Similarity {
                text1,
                text2,
                model,
                gpu,
                quiet,
            } => {
                assert_eq!(text1, "text one");
                assert_eq!(text2, "text two");
                assert_eq!(model, "minilm-l6-v2");
                assert!(!gpu);
                assert!(!quiet);
            }
            _ => panic!("Expected Similarity command"),
        }
    }

    #[test]
    fn test_model_list_defaults() {
        let cli = parse_args(&["model", "list"]).unwrap();

        match cli.command {
            Commands::Model {
                action:
                    ModelCommands::List {
                        arch,
                        task,
                        downloaded,
                    },
            } => {
                assert!(arch.is_none());
                assert!(task.is_none());
                assert!(!downloaded);
            }
            _ => panic!("Expected Model List command"),
        }
    }

    #[test]
    fn test_model_list_with_filters() {
        let cli = parse_args(&[
            "model",
            "list",
            "--arch",
            "bert",
            "--task",
            "embedding",
            "--downloaded",
        ])
        .unwrap();

        match cli.command {
            Commands::Model {
                action:
                    ModelCommands::List {
                        arch,
                        task,
                        downloaded,
                    },
            } => {
                assert_eq!(arch, Some("bert".to_string()));
                assert_eq!(task, Some("embedding".to_string()));
                assert!(downloaded);
            }
            _ => panic!("Expected Model List command"),
        }
    }

    #[test]
    fn test_model_download() {
        let cli = parse_args(&["model", "download", "minilm-l6-v2"]).unwrap();

        match cli.command {
            Commands::Model {
                action: ModelCommands::Download { name, gguf, quiet },
            } => {
                assert_eq!(name, "minilm-l6-v2");
                assert!(!gguf);
                assert!(!quiet);
            }
            _ => panic!("Expected Model Download command"),
        }
    }

    #[test]
    fn test_model_download_gguf() {
        let cli = parse_args(&["model", "download", "llama3.2-1b", "--gguf"]).unwrap();

        match cli.command {
            Commands::Model {
                action: ModelCommands::Download { name, gguf, .. },
            } => {
                assert_eq!(name, "llama3.2-1b");
                assert!(gguf);
            }
            _ => panic!("Expected Model Download command"),
        }
    }

    #[test]
    fn test_model_info() {
        let cli = parse_args(&["model", "info", "phi3.5-mini"]).unwrap();

        match cli.command {
            Commands::Model {
                action: ModelCommands::Info { name },
            } => {
                assert_eq!(name, "phi3.5-mini");
            }
            _ => panic!("Expected Model Info command"),
        }
    }

    #[test]
    fn test_model_remove() {
        let cli = parse_args(&["model", "remove", "old-model"]).unwrap();

        match cli.command {
            Commands::Model {
                action: ModelCommands::Remove { name },
            } => {
                assert_eq!(name, "old-model");
            }
            _ => panic!("Expected Model Remove command"),
        }
    }

    #[test]
    fn test_model_search() {
        let cli = parse_args(&["model", "search", "llama"]).unwrap();

        match cli.command {
            Commands::Model {
                action: ModelCommands::Search { query },
            } => {
                assert_eq!(query, "llama");
            }
            _ => panic!("Expected Model Search command"),
        }
    }

    #[test]
    fn test_index_create_minimal() {
        let cli = parse_args(&["index", "create", "output.idx"]).unwrap();

        match cli.command {
            Commands::Index {
                action:
                    IndexCommands::Create {
                        output,
                        inputs,
                        chunk_size,
                        chunk_overlap,
                        model,
                        ..
                    },
            } => {
                assert_eq!(output, "output.idx");
                assert!(inputs.is_empty());
                assert_eq!(chunk_size, 1000);
                assert_eq!(chunk_overlap, 200);
                assert_eq!(model, "minilm-l6-v2");
            }
            _ => panic!("Expected Index Create command"),
        }
    }

    #[test]
    fn test_index_create_with_inputs() {
        let cli = parse_args(&[
            "index",
            "create",
            "out.idx",
            "file1.txt",
            "file2.txt",
            "dir/",
        ])
        .unwrap();

        match cli.command {
            Commands::Index {
                action: IndexCommands::Create { inputs, .. },
            } => {
                assert_eq!(inputs.len(), 3);
                assert_eq!(inputs[0], "file1.txt");
                assert_eq!(inputs[1], "file2.txt");
                assert_eq!(inputs[2], "dir/");
            }
            _ => panic!("Expected Index Create command"),
        }
    }

    #[test]
    fn test_index_create_with_options() {
        let cli = parse_args(&[
            "index",
            "create",
            "out.idx",
            "--chunk-size",
            "500",
            "--chunk-overlap",
            "100",
            "-m",
            "nomic-embed-text",
            "--gpu",
            "-q",
        ])
        .unwrap();

        match cli.command {
            Commands::Index {
                action:
                    IndexCommands::Create {
                        chunk_size,
                        chunk_overlap,
                        model,
                        gpu,
                        quiet,
                        ..
                    },
            } => {
                assert_eq!(chunk_size, 500);
                assert_eq!(chunk_overlap, 100);
                assert_eq!(model, "nomic-embed-text");
                assert!(gpu);
                assert!(quiet);
            }
            _ => panic!("Expected Index Create command"),
        }
    }

    #[test]
    fn test_index_add() {
        let cli = parse_args(&["index", "add", "existing.idx", "newfile.txt"]).unwrap();

        match cli.command {
            Commands::Index {
                action:
                    IndexCommands::Add {
                        index_path, inputs, ..
                    },
            } => {
                assert_eq!(index_path, "existing.idx");
                assert_eq!(inputs, vec!["newfile.txt".to_string()]);
            }
            _ => panic!("Expected Index Add command"),
        }
    }

    #[test]
    fn test_index_info() {
        let cli = parse_args(&["index", "info", "my.idx"]).unwrap();

        match cli.command {
            Commands::Index {
                action: IndexCommands::Info { index_path },
            } => {
                assert_eq!(index_path, "my.idx");
            }
            _ => panic!("Expected Index Info command"),
        }
    }

    #[test]
    fn test_verbose_zero() {
        let cli = parse_args(&["generate"]).unwrap();
        assert_eq!(cli.verbose, 0);
    }

    #[test]
    fn test_verbose_one() {
        let cli = parse_args(&["-v", "generate"]).unwrap();
        assert_eq!(cli.verbose, 1);
    }

    #[test]
    fn test_verbose_two() {
        let cli = parse_args(&["-vv", "generate"]).unwrap();
        assert_eq!(cli.verbose, 2);
    }

    #[test]
    fn test_verbose_three() {
        let cli = parse_args(&["-vvv", "generate"]).unwrap();
        assert_eq!(cli.verbose, 3);
    }

    #[test]
    fn test_verbose_long_form() {
        let cli = parse_args(&["--verbose", "--verbose", "generate"]).unwrap();
        assert_eq!(cli.verbose, 2);
    }

    #[test]
    fn test_verbose_after_command() {
        // Global flag can come after command
        let cli = parse_args(&["generate", "-v"]).unwrap();
        assert_eq!(cli.verbose, 1);
    }

    #[test]
    fn test_missing_command() {
        let result = parse_args(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_command() {
        let result = parse_args(&["unknown"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_required_arg() {
        // transcribe requires a file
        let result = parse_args(&["transcribe"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_number() {
        let result = parse_args(&["generate", "-n", "not_a_number"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_float() {
        let result = parse_args(&["generate", "-t", "not_a_float"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rerank_minimal() {
        let cli = parse_args(&["rerank", "my query"]).unwrap();

        match cli.command {
            Commands::Rerank {
                query,
                documents,
                model,
                top_k,
                format,
                ..
            } => {
                assert_eq!(query, "my query");
                assert!(documents.is_empty());
                assert_eq!(model, "minilm-l6-v2-cross-encoder");
                assert!(top_k.is_none());
                assert_eq!(format, "text");
            }
            _ => panic!("Expected Rerank command"),
        }
    }

    #[test]
    fn test_rerank_with_documents() {
        let cli = parse_args(&["rerank", "query", "doc1", "doc2", "doc3"]).unwrap();

        match cli.command {
            Commands::Rerank {
                query, documents, ..
            } => {
                assert_eq!(query, "query");
                assert_eq!(documents.len(), 3);
            }
            _ => panic!("Expected Rerank command"),
        }
    }

    #[test]
    fn test_rerank_with_top_k() {
        let cli = parse_args(&["rerank", "query", "-k", "5"]).unwrap();

        match cli.command {
            Commands::Rerank { top_k, .. } => {
                assert_eq!(top_k, Some(5));
            }
            _ => panic!("Expected Rerank command"),
        }
    }

    #[test]
    fn test_summarize_defaults() {
        let cli = parse_args(&["summarize"]).unwrap();

        match cli.command {
            Commands::Summarize {
                input,
                model,
                min_length,
                max_length,
                num_beams,
                ..
            } => {
                assert!(input.is_none());
                assert_eq!(model, "distilbart-cnn");
                assert!(min_length.is_none());
                assert!(max_length.is_none());
                assert!(num_beams.is_none());
            }
            _ => panic!("Expected Summarize command"),
        }
    }

    #[test]
    fn test_summarize_with_options() {
        let cli = parse_args(&[
            "summarize",
            "--input",
            "input.txt",
            "--min-length",
            "50",
            "--max-length",
            "200",
            "--num-beams",
            "4",
            "--length-penalty",
            "1.5",
        ])
        .unwrap();

        match cli.command {
            Commands::Summarize {
                input,
                min_length,
                max_length,
                num_beams,
                length_penalty,
                ..
            } => {
                assert_eq!(input, Some("input.txt".to_string()));
                assert_eq!(min_length, Some(50));
                assert_eq!(max_length, Some(200));
                assert_eq!(num_beams, Some(4));
                assert_eq!(length_penalty, Some(1.5));
            }
            _ => panic!("Expected Summarize command"),
        }
    }

    #[test]
    fn test_translate_defaults() {
        let cli = parse_args(&["translate"]).unwrap();

        match cli.command {
            Commands::Translate {
                input,
                model,
                src,
                dst,
                ..
            } => {
                assert!(input.is_none());
                assert_eq!(model, "flan-t5-base");
                assert!(src.is_none());
                assert!(dst.is_none());
            }
            _ => panic!("Expected Translate command"),
        }
    }

    #[test]
    fn test_translate_with_languages() {
        let cli = parse_args(&["translate", "--src", "en", "--dst", "is"]).unwrap();

        match cli.command {
            Commands::Translate { src, dst, .. } => {
                assert_eq!(src, Some("en".to_string()));
                assert_eq!(dst, Some("is".to_string()));
            }
            _ => panic!("Expected Translate command"),
        }
    }
    #[test]
    fn test_transcribe_minimal() {
        let cli = parse_args(&["transcribe", "audio.wav"]).unwrap();

        match cli.command {
            Commands::Transcribe {
                file,
                model,
                language,
                ..
            } => {
                assert_eq!(file, "audio.wav");
                assert_eq!(model, "whisper-small");
                assert!(language.is_none());
            }
            _ => panic!("Expected Transcribe command"),
        }
    }

    #[test]
    fn test_classify_with_model() {
        let cli = parse_args(&["classify", "i hate mondays", "--model", "toxic-bert"]).unwrap();
        match cli.command {
            Commands::Classify { input, model, .. } => {
                assert_eq!(input, vec!["i hate mondays".to_string()]);
                assert_eq!(model, "toxic-bert");
            }
            _ => panic!("Expected Classify command"),
        }
    }

    #[test]
    fn test_transcribe_with_options() {
        let cli = parse_args(&[
            "transcribe",
            "audio.mp3",
            "-m",
            "whisper-large-v3",
            "--language",
            "is",
        ])
        .unwrap();

        match cli.command {
            Commands::Transcribe {
                file,
                model,
                language,
                ..
            } => {
                assert_eq!(file, "audio.mp3");
                assert_eq!(model, "whisper-large-v3");
                assert_eq!(language, Some("is".to_string()));
            }
            _ => panic!("Expected Transcribe command"),
        }
    }
}

mod send_sync_tests {
    use kjarni::{
        Classifier, Embedder, Indexer, Reranker, Searcher, chat::Chat, generator::Generator,
    };
    // Compile time verificatio
    const _: () = {
        const fn assert_send<T: Send>() {}
        const fn assert_sync<T: Sync>() {}
        assert_send::<Embedder>();
        assert_sync::<Embedder>();

        assert_send::<Indexer>();
        assert_sync::<Indexer>();

        assert_send::<Searcher>();
        assert_sync::<Searcher>();

        assert_send::<Reranker>();
        assert_sync::<Reranker>();

        assert_send::<Generator>();
        assert_sync::<Generator>();

        assert_send::<Chat>();
        assert_sync::<Chat>();

        assert_send::<Classifier>();
        assert_sync::<Classifier>();
    };
}
