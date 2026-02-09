//! Cooperative cancellation

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A token that can be checked to determine if cancellation was requested.
#[derive(Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates a new cancellation token pair.
    ///
    /// Returns:
    /// - `CancellationToken` - Pass this to the generation function
    /// - `CancellationHandle` - Keep this to trigger cancellation
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (token, handle) = CancellationToken::new();
    ///
    /// // Pass token to generator
    /// let result = generator.generate_cancellable("prompt", &config, token).await;
    ///
    /// // Or cancel from another task
    /// handle.cancel();
    /// ```
    pub fn new() -> (Self, CancellationHandle) {
        let cancelled = Arc::new(AtomicBool::new(false));
        let token = CancellationToken {
            cancelled: cancelled.clone(),
        };
        let handle = CancellationHandle { cancelled };
        (token, handle)
    }

    /// Creates a token that is never cancelled.
    pub fn never() -> Self {
        CancellationToken {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a token that is already cancelled.
    pub fn already_cancelled() -> Self {
        CancellationToken {
            cancelled: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Checks if cancellation has been requested.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Returns an error if cancellation was requested.
    #[inline]
    pub fn check(&self) -> Result<(), CancellationError> {
        if self.is_cancelled() {
            Err(CancellationError)
        } else {
            Ok(())
        }
    }
}

impl Default for CancellationToken {
    /// Default token is never cancelled.
    fn default() -> Self {
        Self::never()
    }
}

impl std::fmt::Debug for CancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

/// A handle that can trigger cancellation.
#[derive(Clone)]
pub struct CancellationHandle {
    cancelled: Arc<AtomicBool>,
}

impl CancellationHandle {
    /// Signals cancellation to all associated tokens.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Checks if cancellation has already been triggered.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for CancellationHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancellationHandle")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

/// Error returned when an operation was cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancellationError;

impl std::fmt::Display for CancellationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "operation was cancelled")
    }
}

impl std::error::Error for CancellationError {}


/// Extension trait for creating timeout-based cancellation.
impl CancellationHandle {
    /// Creates a handle that auto-cancels after a timeout.
    pub fn cancel_after(self, timeout: std::time::Duration) {
        let handle = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            handle.cancel();
        });
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_send_sync() {
        let (token, handle) = CancellationToken::new();

        // Move token to another thread
        let t1 = thread::spawn(move || {
            assert!(!token.is_cancelled());
            // Wait for cancel
            while !token.is_cancelled() {
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(token.is_cancelled());
        });

        // Cancel from main thread
        std::thread::sleep(Duration::from_millis(50));
        handle.cancel();

        t1.join().unwrap();
    }

    #[tokio::test]
    async fn test_cancel_after() {
        let (token, handle) = CancellationToken::new();

        // Cancel after 50ms
        handle.cancel_after(Duration::from_millis(50));

        assert!(!token.is_cancelled());

        // Wait 100ms
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(token.is_cancelled());
    }

    #[test]
    fn test_new_token_not_cancelled() {
        let (token, _handle) = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancel_sets_flag() {
        let (token, handle) = CancellationToken::new();
        assert!(!token.is_cancelled());

        handle.cancel();

        assert!(token.is_cancelled());
    }

    #[test]
    fn test_multiple_cancels_idempotent() {
        let (token, handle) = CancellationToken::new();

        handle.cancel();
        handle.cancel();
        handle.cancel();

        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cloned_tokens_share_state() {
        let (token1, handle) = CancellationToken::new();
        let token2 = token1.clone();

        assert!(!token1.is_cancelled());
        assert!(!token2.is_cancelled());

        handle.cancel();

        assert!(token1.is_cancelled());
        assert!(token2.is_cancelled());
    }

    #[test]
    fn test_never_token() {
        let token = CancellationToken::never();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_already_cancelled() {
        let token = CancellationToken::already_cancelled();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_check_returns_error_when_cancelled() {
        let (token, handle) = CancellationToken::new();

        assert!(token.check().is_ok());

        handle.cancel();

        assert!(token.check().is_err());
    }
}
