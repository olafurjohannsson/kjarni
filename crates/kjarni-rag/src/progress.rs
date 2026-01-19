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
pub type ProgressCallback = Box<dyn FnMut(&Progress, Option<&str>) + Send + Sync>;

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
    where F: FnMut(&Progress, Option<&str>) + Send + Sync + 'static 
    {
        Self {
            callback: Some(Box::new(callback)),
            quiet: true, // Don't output to stderr if callback provided
            last_stage: None,
        }
    }
    
    pub fn report(&mut self, progress: &Progress, message: Option<&str>) {
        // Call callback if provided
        if let Some(ref mut cb) = self.callback {
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


#[cfg(test)]
mod progress_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use crate::progress::{CancelToken, Progress, ProgressReporter, ProgressStage};

    // ============================================================================
    // PROGRESS STRUCT TESTS
    // ============================================================================

    #[test]
    fn test_progress_new() {
        let progress = Progress::new(ProgressStage::Scanning, 5, Some(10));

        assert_eq!(progress.stage, ProgressStage::Scanning);
        assert_eq!(progress.current, 5);
        assert_eq!(progress.total, Some(10));
        assert_eq!(progress.message_len, 0);
    }

    #[test]
    fn test_progress_new_without_total() {
        let progress = Progress::new(ProgressStage::Loading, 42, None);

        assert_eq!(progress.stage, ProgressStage::Loading);
        assert_eq!(progress.current, 42);
        assert_eq!(progress.total, None);
    }

    #[test]
    fn test_progress_scanning() {
        let progress = Progress::scanning(15);

        assert_eq!(progress.stage, ProgressStage::Scanning);
        assert_eq!(progress.current, 15);
        assert_eq!(progress.total, None); // Scanning doesn't know total
    }

    #[test]
    fn test_progress_loading() {
        let progress = Progress::loading(50, Some(100));

        assert_eq!(progress.stage, ProgressStage::Loading);
        assert_eq!(progress.current, 50);
        assert_eq!(progress.total, Some(100));
    }

    #[test]
    fn test_progress_loading_without_total() {
        let progress = Progress::loading(25, None);

        assert_eq!(progress.stage, ProgressStage::Loading);
        assert_eq!(progress.current, 25);
        assert_eq!(progress.total, None);
    }

    #[test]
    fn test_progress_embedding() {
        let progress = Progress::embedding(75, Some(200));

        assert_eq!(progress.stage, ProgressStage::Embedding);
        assert_eq!(progress.current, 75);
        assert_eq!(progress.total, Some(200));
    }

    #[test]
    fn test_progress_writing() {
        let progress = Progress::writing(30, Some(50));

        assert_eq!(progress.stage, ProgressStage::Writing);
        assert_eq!(progress.current, 30);
        assert_eq!(progress.total, Some(50));
    }

    #[test]
    fn test_progress_committing() {
        let progress = Progress::committing();

        assert_eq!(progress.stage, ProgressStage::Committing);
        assert_eq!(progress.current, 0);
        assert_eq!(progress.total, None);
    }

    // ============================================================================
    // PROGRESS STAGE TESTS
    // ============================================================================

    #[test]
    fn test_progress_stage_variants() {
        // Verify all stages exist and have correct repr(C) values
        assert_eq!(ProgressStage::Scanning as u8, 0);
        assert_eq!(ProgressStage::Loading as u8, 1);
        assert_eq!(ProgressStage::Embedding as u8, 2);
        assert_eq!(ProgressStage::Writing as u8, 3);
        assert_eq!(ProgressStage::Committing as u8, 4);
        assert_eq!(ProgressStage::Searching as u8, 5);
        assert_eq!(ProgressStage::Reranking as u8, 6);
    }

    #[test]
    fn test_progress_stage_equality() {
        assert_eq!(ProgressStage::Scanning, ProgressStage::Scanning);
        assert_ne!(ProgressStage::Scanning, ProgressStage::Loading);
    }

    #[test]
    fn test_progress_stage_clone() {
        let stage = ProgressStage::Embedding;
        let cloned = stage.clone();
        assert_eq!(stage, cloned);
    }

    #[test]
    fn test_progress_stage_copy() {
        let stage = ProgressStage::Writing;
        let copied: ProgressStage = stage; // Copy, not move
        assert_eq!(stage, copied); // Original still usable
    }

    #[test]
    fn test_progress_stage_debug() {
        let debug_str = format!("{:?}", ProgressStage::Reranking);
        assert_eq!(debug_str, "Reranking");
    }

    // ============================================================================
    // PROGRESS CLONE TESTS
    // ============================================================================

    #[test]
    fn test_progress_clone() {
        let original = Progress::embedding(50, Some(100));
        let cloned = original.clone();

        assert_eq!(original.stage, cloned.stage);
        assert_eq!(original.current, cloned.current);
        assert_eq!(original.total, cloned.total);
    }

    #[test]
    fn test_progress_debug() {
        let progress = Progress::scanning(10);
        let debug_str = format!("{:?}", progress);

        assert!(debug_str.contains("Scanning"));
        assert!(debug_str.contains("10"));
    }

    // ============================================================================
    // CANCEL TOKEN TESTS
    // ============================================================================

    #[test]
    fn test_cancel_token_new() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancel_token_default() {
        let token = CancelToken::default();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancel_token_cancel() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancel_token_reset() {
        let token = CancelToken::new();

        token.cancel();
        assert!(token.is_cancelled());

        token.reset();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancel_token_multiple_cancels() {
        let token = CancelToken::new();

        // Multiple cancels should be idempotent
        token.cancel();
        token.cancel();
        token.cancel();

        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancel_token_clone() {
        let token1 = CancelToken::new();
        let token2 = token1.clone();

        // Both point to same underlying atomic
        token1.cancel();

        assert!(token1.is_cancelled());
        assert!(token2.is_cancelled()); // Clone sees the cancellation
    }

    #[test]
    fn test_cancel_token_thread_safety() {
        let token = CancelToken::new();
        let token_clone = token.clone();

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            token_clone.cancel();
        });

        // Poll until cancelled
        let mut iterations = 0;
        while !token.is_cancelled() && iterations < 100 {
            thread::sleep(Duration::from_millis(5));
            iterations += 1;
        }

        handle.join().unwrap();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancel_token_multiple_threads() {
        let token = CancelToken::new();
        let check_count = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];

        // Spawn 10 threads that check cancellation
        for _ in 0..10 {
            let t = token.clone();
            let count = check_count.clone();

            handles.push(thread::spawn(move || {
                while !t.is_cancelled() {
                    count.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(1));
                }
            }));
        }

        // Let threads run briefly
        thread::sleep(Duration::from_millis(20));

        // Cancel
        token.cancel();

        // Wait for all threads
        for h in handles {
            h.join().unwrap();
        }

        // All threads should have stopped
        assert!(token.is_cancelled());
        assert!(check_count.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_cancel_token_reset_thread_safety() {
        let token = CancelToken::new();
        let token_clone = token.clone();

        token.cancel();
        assert!(token.is_cancelled());

        let handle = thread::spawn(move || {
            token_clone.reset();
        });

        handle.join().unwrap();
        assert!(!token.is_cancelled());
    }

    // ============================================================================
    // PROGRESS REPORTER TESTS
    // ============================================================================

    #[test]
    fn test_progress_reporter_new_quiet() {
        let reporter = ProgressReporter::new(true);
        // Should not panic when reporting
        let mut reporter = reporter;
        let progress = Progress::scanning(5);
        reporter.report(&progress, None);
    }

    #[test]
    fn test_progress_reporter_new_verbose() {
        // This will output to stderr - mainly testing it doesn't panic
        let mut reporter = ProgressReporter::new(false);
        let progress = Progress::scanning(5);
        reporter.report(&progress, Some("test.txt"));
    }

    #[test]
    fn test_progress_reporter_with_callback() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let mut reporter = ProgressReporter::with_callback(move |progress, message| {
            call_count_clone.fetch_add(1, Ordering::Relaxed);
            assert_eq!(progress.stage, ProgressStage::Embedding);
            assert_eq!(progress.current, 50);
            assert_eq!(message, Some("processing..."));
        });

        let progress = Progress::embedding(50, Some(100));
        reporter.report(&progress, Some("processing..."));

        assert_eq!(call_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_progress_reporter_callback_multiple_calls() {
        let stages_seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let stages_clone = stages_seen.clone();

        let mut reporter = ProgressReporter::with_callback(move |progress, _| {
            stages_clone.lock().unwrap().push(progress.stage);
        });

        reporter.report(&Progress::scanning(1), None);
        reporter.report(&Progress::loading(1, Some(10)), None);
        reporter.report(&Progress::embedding(1, Some(10)), None);
        reporter.report(&Progress::writing(1, Some(10)), None);
        reporter.report(&Progress::committing(), None);

        let stages = stages_seen.lock().unwrap();
        assert_eq!(stages.len(), 5);
        assert_eq!(stages[0], ProgressStage::Scanning);
        assert_eq!(stages[1], ProgressStage::Loading);
        assert_eq!(stages[2], ProgressStage::Embedding);
        assert_eq!(stages[3], ProgressStage::Writing);
        assert_eq!(stages[4], ProgressStage::Committing);
    }

    #[test]
    fn test_progress_reporter_callback_with_message() {
        let messages = Arc::new(std::sync::Mutex::new(Vec::new()));
        let messages_clone = messages.clone();

        let mut reporter = ProgressReporter::with_callback(move |_, message| {
            messages_clone
                .lock()
                .unwrap()
                .push(message.map(String::from));
        });

        reporter.report(&Progress::loading(1, None), Some("file1.txt"));
        reporter.report(&Progress::loading(2, None), None);
        reporter.report(&Progress::loading(3, None), Some("file3.txt"));

        let msgs = messages.lock().unwrap();
        assert_eq!(msgs[0], Some("file1.txt".to_string()));
        assert_eq!(msgs[1], None);
        assert_eq!(msgs[2], Some("file3.txt".to_string()));
    }

    #[test]
    fn test_progress_reporter_stage_change_detection() {
        // Test that stage changes are tracked
        let stage_changes = Arc::new(AtomicUsize::new(0));
        let changes_clone = stage_changes.clone();

        let mut last_stage: Option<ProgressStage> = None;

        let mut reporter = ProgressReporter::with_callback(move |progress, _| {
            if last_stage != Some(progress.stage) {
                changes_clone.fetch_add(1, Ordering::Relaxed);
                last_stage = Some(progress.stage);
            }
        });

        // Same stage multiple times
        reporter.report(&Progress::loading(1, None), None);
        reporter.report(&Progress::loading(2, None), None);
        reporter.report(&Progress::loading(3, None), None);

        // Different stage
        reporter.report(&Progress::embedding(1, None), None);

        assert_eq!(stage_changes.load(Ordering::Relaxed), 2); // Loading once, Embedding once
    }

    // ============================================================================
    // EDGE CASES
    // ============================================================================

    #[test]
    fn test_progress_zero_values() {
        let progress = Progress::new(ProgressStage::Scanning, 0, Some(0));

        assert_eq!(progress.current, 0);
        assert_eq!(progress.total, Some(0));
    }

    #[test]
    fn test_progress_large_values() {
        let progress = Progress::new(ProgressStage::Embedding, usize::MAX, Some(usize::MAX));

        assert_eq!(progress.current, usize::MAX);
        assert_eq!(progress.total, Some(usize::MAX));
    }

    #[test]
    fn test_progress_current_exceeds_total() {
        // This is valid - might happen if total was estimated
        let progress = Progress::new(ProgressStage::Loading, 150, Some(100));

        assert_eq!(progress.current, 150);
        assert_eq!(progress.total, Some(100));
    }

    // ============================================================================
    // INTEGRATION TESTS
    // ============================================================================

    #[test]
    fn test_full_indexing_workflow() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let events_clone = events.clone();

        let mut reporter = ProgressReporter::with_callback(move |progress, message| {
            events_clone.lock().unwrap().push((
                progress.stage,
                progress.current,
                progress.total,
                message.map(String::from),
            ));
        });

        // Simulate indexing workflow
        reporter.report(&Progress::scanning(0), Some("Starting scan"));
        reporter.report(&Progress::scanning(10), None);
        reporter.report(&Progress::scanning(25), None);

        reporter.report(&Progress::loading(0, Some(25)), Some("Loading documents"));
        reporter.report(&Progress::loading(10, Some(25)), None);
        reporter.report(&Progress::loading(25, Some(25)), None);

        reporter.report(&Progress::embedding(0, Some(100)), Some("Generating embeddings"));
        reporter.report(&Progress::embedding(50, Some(100)), None);
        reporter.report(&Progress::embedding(100, Some(100)), None);

        reporter.report(&Progress::writing(0, Some(100)), Some("Writing index"));
        reporter.report(&Progress::writing(100, Some(100)), None);

        reporter.report(&Progress::committing(), Some("Finalizing"));

        let events = events.lock().unwrap();
        assert_eq!(events.len(), 12);

        // Verify workflow order
        assert_eq!(events[0].0, ProgressStage::Scanning);
        assert_eq!(events[3].0, ProgressStage::Loading);
        assert_eq!(events[6].0, ProgressStage::Embedding);
        assert_eq!(events[9].0, ProgressStage::Writing);
        assert_eq!(events[11].0, ProgressStage::Committing);
    }

    #[test]
    fn test_cancellable_operation() {
        let token = CancelToken::new();
        let token_clone = token.clone();
        let processed = Arc::new(AtomicUsize::new(0));
        let processed_clone = processed.clone();

        // Simulate a cancellable operation
        let handle = thread::spawn(move || {
            for i in 0..1000 {
                if token_clone.is_cancelled() {
                    return i;
                }
                processed_clone.fetch_add(1, Ordering::Relaxed);
                thread::sleep(Duration::from_micros(100));
            }
            1000
        });

        // Cancel after a short delay
        thread::sleep(Duration::from_millis(10));
        token.cancel();

        let stopped_at = handle.join().unwrap();

        // Should have stopped before completing all 1000
        assert!(stopped_at < 1000);
        assert!(processed.load(Ordering::Relaxed) < 1000);
    }
}