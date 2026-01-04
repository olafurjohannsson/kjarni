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

#[derive(Subcommand)]
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

    /// Summarize text using an encoder-decoder model
    Summarize {
        /// Input text, file path, or stdin if not provided
        input: Option<String>,

        #[arg(short, long, default_value = "distilbart-cnn")]
        model: String,

        #[arg(long)]
        model_path: Option<String>,

        /// Minimum length of summary
        #[arg(long)]
        min_length: Option<usize>,

        /// Maximum length of summary
        #[arg(long)]
        max_length: Option<usize>,

        /// Number of beams for beam search
        #[arg(long)]
        num_beams: Option<usize>,

        /// Length penalty (>1.0 favors longer, <1.0 favors shorter)
        #[arg(long)]
        length_penalty: Option<f32>,

        /// N-gram size to prevent repetition
        #[arg(long)]
        no_repeat_ngram: Option<usize>,

        /// Use GPU
        #[arg(long)]
        gpu: bool,

        /// Suppress status messages
        #[arg(short, long)]
        quiet: bool,
    },

    /// Translate text between languages
    Translate {
        /// Input text, file path, or stdin if not provided
        input: Option<String>,

        #[arg(short, long, default_value = "marian-en-is")]
        model: String,

        #[arg(long)]
        model_path: Option<String>,

        #[arg(long)]
        src: Option<String>,

        #[arg(long)]
        dst: Option<String>,
    },

    /// Transcribe audio to text
    Transcribe {
        /// Path to audio file
        file: String,

        #[arg(short, long, default_value = "whisper-tiny")]
        model: String,

        #[arg(long)]
        model_path: Option<String>,

        #[arg(long)]
        language: Option<String>,
    },

    /// Encode text to embeddings
    Encode {
        /// Input text, file path, or stdin if not provided
        input: Option<String>,

        #[arg(short, long, default_value = "minilm-l6-v2")]
        model: String,

        #[arg(long)]
        model_path: Option<String>,

        /// Output format: json, jsonl, raw, numpy
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Normalize embeddings (L2)
        #[arg(long, default_value_t = true)]
        normalize: bool,

        /// Pooling strategy: mean, cls, max
        #[arg(long, default_value = "mean")]
        pooling: String,

        /// Use GPU
        #[arg(long)]
        gpu: bool,
    },

    /// Classify text using a sequence classification model
    Classify {
        /// Text to classify (or file path, or stdin)
        #[arg(trailing_var_arg = true)]
        input: Vec<String>,

        /// Classification model
        #[arg(short, long, default_value = "distilbert-sentiment")]
        model: String,

        /// Return top-k predictions
        #[arg(short = 'k', long, default_value_t = 1)]
        top_k: usize,

        /// Output format: text, json, jsonl
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Custom labels (comma-separated, for zero-shot)
        #[arg(long)]
        labels: Option<String>,

        #[arg(long)]
        gpu: bool,

        #[arg(short, long)]
        quiet: bool,
    },

    /// Rerank documents by relevance to a query
    Rerank {
        /// The query to rank against
        query: String,

        /// Documents to rerank (or read from stdin, one per line)
        #[arg(trailing_var_arg = true)]
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

    /// Interactive REPL for experimentation
    Repl {
        #[arg(short, long, default_value = "llama-3.2-1b")]
        model: String,

        #[arg(long)]
        model_path: Option<String>,

        #[arg(long, default_value = "chat")]
        mode: String,

        #[arg(long)]
        gpu: bool,
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

#[derive(Subcommand)]
pub enum ModelCommands {
    /// List all available models
    List {
        #[arg(short, long)]
        arch: Option<String>,
    },

    /// Download a model
    Download {
        name: String,
        gguf: bool,
    },

    Remove {
        name: String,
    },

    /// Show detailed info about a model
    Info {
        name: String,
    },

    /// Search for models by name or description
    Search {
        query: String,
    },
}

#[derive(Subcommand)]
pub enum IndexCommands {
    /// Create a new index from documents
    Create {
        /// Output index file path
        output: String,

        /// Input files or directories to index
        #[arg(trailing_var_arg = true)]
        inputs: Vec<String>,

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
        #[arg(trailing_var_arg = true)]
        inputs: Vec<String>,

        /// Chunk size for splitting documents
        #[arg(long, default_value_t = 1000)]
        chunk_size: usize,

        /// Chunk overlap
        #[arg(long, default_value_t = 200)]
        chunk_overlap: usize,

        /// Encoder model (must match index)
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
