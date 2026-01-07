//! Device selection and download policies.

use kjarni_transformers::traits::Device;

/// Execution device for models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KjarniDevice {
    /// Run on CPU (default, always available).
    #[default]
    Cpu,

    /// Run on GPU via WebGPU.
    Gpu,

    /// Automatically select best available device.
    Auto,
}

impl KjarniDevice {
    /// Resolve Auto to a concrete device.
    pub fn resolve(self) -> Self {
        match self {
            Self::Auto => {
                // TODO: Implement GPU availability check
                // For now, default to CPU as it's always available
                Self::Cpu
            }
            other => other,
        }
    }

    /// Convert to the low-level Device type.
    pub fn to_device(self) -> Device {
        match self.resolve() {
            Self::Cpu => Device::Cpu,
            Self::Gpu | Self::Auto => Device::Wgpu,
        }
    }

    /// Check if this is CPU.
    pub fn is_cpu(&self) -> bool {
        matches!(self.resolve(), Self::Cpu)
    }

    /// Check if this is GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(self.resolve(), Self::Gpu)
    }
}

impl From<KjarniDevice> for Device {
    fn from(d: KjarniDevice) -> Self {
        d.to_device()
    }
}

/// Policy for downloading models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DownloadPolicy {
    /// Download model if not present locally (default).
    #[default]
    IfMissing,

    /// Never download, fail if model not present.
    Never,

    /// Always check for updates and download if newer.
    Eager,
}