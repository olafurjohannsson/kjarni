//! GPU timeout

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

/// Polls the GPU device until a condition is met or timeout expires
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

/// Async version of poll_with_timeout that yields to the runtime
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
        tokio::time::sleep(config.poll_interval).await;
    }
}

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
