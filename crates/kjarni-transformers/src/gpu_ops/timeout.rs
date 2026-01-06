// kjarni-transformers/src/gpu_ops/timeout.rs

//! GPU operation timeout handling.
//!
//! WebGPU operations can potentially hang indefinitely if the GPU becomes
//! unresponsive. This module provides utilities to add timeout protection
//! to GPU operations.
//!
//! # Challenge
//!
//! WebGPU's `device.poll()` is blocking and doesn't have a built-in timeout.
//! We work around this by:
//!
//! 1. Using non-blocking poll in a loop
//! 2. Checking elapsed time between iterations
//! 3. Returning an error if timeout exceeded
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::gpu_ops::timeout::GpuTimeout;
//!
//! let timeout = GpuTimeout::new(Duration::from_secs(30));
//!
//! let result = timeout.poll_until_complete(&device, || {
//!     // Check if operation is done
//!     rx.try_recv().is_ok()
//! }).await?;
//! ```

use anyhow::Result;
use std::time::{Duration, Instant};
use wgpu::Device;

/// Default timeout for GPU operations (30 seconds).
pub const DEFAULT_GPU_TIMEOUT: Duration = Duration::from_secs(30);

/// Timeout configuration for GPU operations.
#[derive(Debug, Clone, Copy)]
pub struct GpuTimeoutConfig {
    /// Maximum time to wait for GPU operations.
    pub timeout: Duration,

    /// How often to poll for completion.
    /// Lower values = more responsive cancellation, higher CPU usage.
    pub poll_interval: Duration,
}

impl Default for GpuTimeoutConfig {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_GPU_TIMEOUT,
            poll_interval: Duration::from_millis(1),
        }
    }
}

impl GpuTimeoutConfig {
    /// Creates a new timeout config with the specified duration.
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            ..Default::default()
        }
    }

    /// Creates a config with no timeout (waits indefinitely).
    pub fn no_timeout() -> Self {
        Self {
            timeout: Duration::MAX,
            ..Default::default()
        }
    }
}

/// Error returned when a GPU operation times out.
#[derive(Debug, Clone)]
pub struct GpuTimeoutError {
    /// The operation that timed out.
    pub operation: String,
    /// How long we waited before timing out.
    pub elapsed: Duration,
    /// The configured timeout.
    pub timeout: Duration,
}

impl std::fmt::Display for GpuTimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU operation '{}' timed out after {:.2}s (limit: {:.2}s)",
            self.operation,
            self.elapsed.as_secs_f64(),
            self.timeout.as_secs_f64()
        )
    }
}

impl std::error::Error for GpuTimeoutError {}

// impl From<GpuTimeoutError> for anyhow::Error {
//     fn from(e: GpuTimeoutError) -> Self {
//         anyhow::anyhow!("{}", e)
//     }
// }

/// Polls the GPU device until a condition is met or timeout expires.
///
/// # Arguments
///
/// * `device` - The WGPU device to poll
/// * `config` - Timeout configuration
/// * `operation_name` - Name for error messages
/// * `is_complete` - Closure that returns true when operation is done
///
/// # Returns
///
/// `Ok(())` if operation completed, `Err(GpuTimeoutError)` if timed out.
///
/// # Example
///
/// ```ignore
/// poll_with_timeout(
///     &device,
///     GpuTimeoutConfig::new(Duration::from_secs(10)),
///     "buffer_map",
///     || rx.try_recv().is_ok()
/// )?;
/// ```
pub fn poll_with_timeout<F>(
    device: &Device,
    config: GpuTimeoutConfig,
    operation_name: &str,
    mut is_complete: F,
) -> Result<(), GpuTimeoutError>
where
    F: FnMut() -> bool,
{
    let start = Instant::now();

    loop {
        // Poll GPU (non-blocking)
        let _ = device.poll(wgpu::PollType::Poll);

        // Check if operation completed
        if is_complete() {
            return Ok(());
        }

        // Check timeout
        let elapsed = start.elapsed();
        if elapsed >= config.timeout {
            return Err(GpuTimeoutError {
                operation: operation_name.to_string(),
                elapsed,
                timeout: config.timeout,
            });
        }

        // Brief sleep to avoid spinning
        std::thread::sleep(config.poll_interval);
    }
}

/// Async version of poll_with_timeout that yields to the runtime.
///
/// Preferred for async contexts as it doesn't block the thread.
///
/// # Example
///
/// ```ignore
/// poll_with_timeout_async(
///     &device,
///     GpuTimeoutConfig::new(Duration::from_secs(10)),
///     "buffer_map",
///     || rx.try_recv().is_ok()
/// ).await?;
/// ```
pub async fn poll_with_timeout_async<F>(
    device: &Device,
    config: GpuTimeoutConfig,
    operation_name: &str,
    mut is_complete: F,
) -> Result<(), GpuTimeoutError>
where
    F: FnMut() -> bool,
{
    let start = Instant::now();

    loop {
        // Poll GPU
        let _ = device.poll(wgpu::PollType::Poll);

        // Check if operation completed
        if is_complete() {
            return Ok(());
        }

        // Check timeout
        let elapsed = start.elapsed();
        if elapsed >= config.timeout {
            return Err(GpuTimeoutError {
                operation: operation_name.to_string(),
                elapsed,
                timeout: config.timeout,
            });
        }

        // Yield to runtime instead of blocking
        // This allows other async tasks to make progress
        tokio::time::sleep(config.poll_interval).await;
    }
}

/// Extension trait for Device to add timeout-aware methods.
pub trait DeviceTimeoutExt {
    /// Polls until work is submitted, with timeout.
    fn poll_with_timeout(&self, config: GpuTimeoutConfig) -> Result<(), GpuTimeoutError>;
}

impl DeviceTimeoutExt for Device {
    fn poll_with_timeout(&self, config: GpuTimeoutConfig) -> Result<(), GpuTimeoutError> {
        let start = Instant::now();

        loop {
            // Do a blocking poll with a small timeout
            let maintained = self.poll(wgpu::PollType::Poll);

            // If no more work to do, we're done
            if !maintained.is_ok() {
                return Ok(());
            }

            let elapsed = start.elapsed();
            if elapsed >= config.timeout {
                return Err(GpuTimeoutError {
                    operation: "device_poll".to_string(),
                    elapsed,
                    timeout: config.timeout,
                });
            }

            std::thread::sleep(config.poll_interval);
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_timeout_config_default() {
//         let config = GpuTimeoutConfig::default();
//         assert_eq!(config.timeout, Duration::from_secs(30));
//     }

//     #[test]
//     fn test_immediate_completion() {
//         // Create a dummy device for testing
//         // In real tests, you'd use a real WGPU device
//         let completed = std::sync::atomic::AtomicBool::new(true);

//         let result = poll_with_timeout(
//             // Can't easily test without a real device, so this is more of a compile check
//             // &device,
//             // GpuTimeoutConfig::new(Duration::from_millis(100)),
//             // "test_op",
//             // || completed.load(std::sync::atomic::Ordering::Relaxed),
//             &create_test_device(),
//             GpuTimeoutConfig::new(Duration::from_millis(100)),
//             "test_op",
//             || true, // Immediately complete
//         );

//         assert!(result.is_ok());
//     }

//     // Helper to create a test device (you'd implement this based on your test setup)
//     fn create_test_device() -> Device {
//         // This would be your actual test device creation
//         // For unit tests, you might want to mock this
//         todo!("Create test device")

//     }
// }