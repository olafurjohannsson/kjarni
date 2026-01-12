// kjarni-transformers/src/common/cancellation.rs

//! Cooperative cancellation for long-running generation tasks.
//!
//! This module provides a lightweight cancellation mechanism that allows
//! callers to signal that generation should stop early. It's designed for:
//!
//! - User-initiated cancellation (Ctrl+C, cancel button)
//! - Timeout-based cancellation
//! - Resource cleanup triggers
//!
//! # Design
//!
//! The cancellation system uses a simple atomic boolean shared between
//! the caller (who can trigger cancellation) and the generator (who checks it).
//!
//! ```text
//! ┌──────────────┐                    ┌──────────────────┐
//! │   Caller     │                    │    Generator     │
//! │              │    Arc<AtomicBool> │                  │
//! │  .cancel() ──┼───────────────────►│ .is_cancelled()  │
//! │              │                    │      ▼           │
//! │              │                    │  if true: break  │
//! └──────────────┘                    └──────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::common::CancellationToken;
//!
//! let (token, handle) = CancellationToken::new();
//!
//! // Spawn generation with the token
//! let generation_task = tokio::spawn(async move {
//!     generator.generate_with_cancellation("prompt", &config, token).await
//! });
//!
//! // Cancel after 5 seconds
//! tokio::time::sleep(Duration::from_secs(5)).await;
//! handle.cancel();
//!
//! let result = generation_task.await?;
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A token that can be checked to determine if cancellation was requested.
///
/// This is the "receiver" side - passed into the generation function.
/// It can only check cancellation status, not trigger it.
///
/// # Thread Safety
///
/// `CancellationToken` is `Send + Sync` and can be safely shared across threads.
/// The check is a single atomic load with relaxed ordering (very fast).
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
    ///
    /// Useful when cancellation support is optional and you want to
    /// avoid `Option<CancellationToken>` in APIs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // These are equivalent:
    /// generator.generate_cancellable("prompt", &config, CancellationToken::never()).await;
    /// generator.generate("prompt", &config).await;
    /// ```
    pub fn never() -> Self {
        CancellationToken {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a token that is already cancelled.
    ///
    /// Useful for testing cancellation handling.
    pub fn already_cancelled() -> Self {
        CancellationToken {
            cancelled: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Checks if cancellation has been requested.
    ///
    /// This is a very fast operation (single atomic load) and can be
    /// called frequently in hot loops without performance impact.
    ///
    /// # Returns
    ///
    /// `true` if `cancel()` was called on the associated `CancellationHandle`.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Returns an error if cancellation was requested.
    ///
    /// Convenience method for use with the `?` operator.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for step in 0..max_steps {
    ///     token.check()?;  // Returns Err if cancelled
    ///     // ... do work ...
    /// }
    /// ```
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
///
/// This is the "sender" side - kept by the caller to signal cancellation.
/// When `cancel()` is called, all associated `CancellationToken`s will
/// return `true` from `is_cancelled()`.
///
/// # Drop Behavior
///
/// Dropping the handle does NOT cancel the token. This allows the handle
/// to go out of scope without affecting ongoing generation.
#[derive(Clone)]
pub struct CancellationHandle {
    cancelled: Arc<AtomicBool>,
}

impl CancellationHandle {
    /// Signals cancellation to all associated tokens.
    ///
    /// This is idempotent - calling it multiple times has no additional effect.
    ///
    /// # Thread Safety
    ///
    /// Can be safely called from any thread at any time.
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

// Allow converting CancellationError to anyhow::Error
// impl From<CancellationError> for anyhow::Error {
//     fn from(_: CancellationError) -> Self {
//         anyhow::anyhow!("Generation cancelled")
//     }
// }

/// Extension trait for creating timeout-based cancellation.
impl CancellationHandle {
    /// Creates a handle that auto-cancels after a timeout.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (token, handle) = CancellationToken::new();
    /// handle.cancel_after(Duration::from_secs(30));
    ///
    /// // Generation will be cancelled after 30 seconds
    /// generator.generate_cancellable("prompt", &config, token).await;
    /// ```
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
        // No way to cancel it - that's the point
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
