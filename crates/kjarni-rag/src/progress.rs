//! Progress reporting for indexing operations

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Progress stage during indexing
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressStage {
    /// Discovering files to index
    Scanning = 0,
    /// Loading and chunking documents
    Loading = 1,
    /// Generating embeddings
    Embedding = 2,
    /// Writing to index
    Writing = 3,
    /// Committing/finalizing index
    Committing = 4,
    /// Search in progress
    Searching = 5,
    /// Reranking results
    Reranking = 6,
}

/// Progress update
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Progress {
    /// Current stage
    pub stage: ProgressStage,
    /// Current item number
    pub current: usize,
    /// Total items
    pub total: Option<usize>,
    /// Optional message (for FFI, this is index into a message buffer)
    pub message_len: usize,
}

impl Progress {
    pub fn new(stage: ProgressStage, current: usize, total: Option<usize>) -> Self {
        Self {
            stage,
            current,
            total,
            message_len: 0,
        }
    }
    
    pub fn scanning(current: usize) -> Self {
        Self::new(ProgressStage::Scanning, current, None)
    }
    
    pub fn loading(current: usize, total: Option<usize>) -> Self {
        Self::new(ProgressStage::Loading, current, total)
    }
    
    pub fn embedding(current: usize, total: Option<usize>) -> Self {
        Self::new(ProgressStage::Embedding, current, total)
    }
    
    pub fn writing(current: usize, total: Option<usize>) -> Self {
        Self::new(ProgressStage::Writing, current, total)
    }
    
    pub fn committing() -> Self {
        Self::new(ProgressStage::Committing, 0, None)
    }
}

/// Cancellation token for long-running operations
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    cancelled: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// Cancel the operation
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
    
    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
    
    /// Reset the token
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }
}

/// Progress callback type (for Rust usage)
pub type ProgressCallback = Box<dyn Fn(&Progress, Option<&str>) + Send + Sync>;

/// Progress reporter that can output to stderr or callback
pub struct ProgressReporter {
    callback: Option<ProgressCallback>,
    quiet: bool,
    last_stage: Option<ProgressStage>,
}

impl ProgressReporter {
    pub fn new(quiet: bool) -> Self {
        Self {
            callback: None,
            quiet,
            last_stage: None,
        }
    }
    
    pub fn with_callback<F>(callback: F) -> Self 
    where F: Fn(&Progress, Option<&str>) + Send + Sync + 'static 
    {
        Self {
            callback: Some(Box::new(callback)),
            quiet: true, // Don't output to stderr if callback provided
            last_stage: None,
        }
    }
    
    pub fn report(&mut self, progress: &Progress, message: Option<&str>) {
        // Call callback if provided
        if let Some(ref cb) = self.callback {
            cb(progress, message);
        }
        
        // Output to stderr if not quiet
        if !self.quiet {
            self.report_stderr(progress, message);
        }
    }
    
    fn report_stderr(&mut self, progress: &Progress, message: Option<&str>) {
        // Print stage header if changed
        if self.last_stage != Some(progress.stage) {
            self.last_stage = Some(progress.stage);
            let stage_name = match progress.stage {
                ProgressStage::Scanning => "Scanning files...",
                ProgressStage::Loading => "Loading documents...",
                ProgressStage::Embedding => "Generating embeddings...",
                ProgressStage::Writing => "Writing to index...",
                ProgressStage::Committing => "Committing index...",
                ProgressStage::Searching => "Searching...",
                ProgressStage::Reranking => "Reranking...",
            };
            eprintln!("{}", stage_name);
        }
        
        // Print progress
        if progress.total.unwrap_or(0) > 0 {
            eprint!("\r  [{}/{}]", progress.current, progress.total.unwrap_or(0));
            if let Some(msg) = message {
                eprint!(" {}", truncate_path(msg, 50));
            }
            eprint!("          "); // Clear trailing chars
        } else if progress.current > 0 {
            eprint!("\r  {} items processed", progress.current);
        }
        
        // Newline on completion
        if progress.stage == ProgressStage::Committing {
            eprintln!();
        }
    }
}

fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        return path.to_string();
    }
    
    // Try to show filename
    if let Some(pos) = path.rfind('/').or_else(|| path.rfind('\\')) {
        let filename = &path[pos + 1..];
        if filename.len() < max_len - 3 {
            return format!("...{}", &path[path.len() - max_len + 3..]);
        }
    }
    
    format!("{}...", &path[..max_len - 3])
}