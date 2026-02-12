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

    pub fn is_cpu(&self) -> bool {
        matches!(self.resolve(), Self::Cpu)
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kjarni_device_default_is_cpu() {
        let device = KjarniDevice::default();
        assert_eq!(device, KjarniDevice::Cpu);
    }

    #[test]
    fn test_resolve_cpu_returns_cpu() {
        let device = KjarniDevice::Cpu;
        assert_eq!(device.resolve(), KjarniDevice::Cpu);
    }

    #[test]
    fn test_resolve_gpu_returns_gpu() {
        let device = KjarniDevice::Gpu;
        assert_eq!(device.resolve(), KjarniDevice::Gpu);
    }

    #[test]
    fn test_resolve_auto_returns_cpu() {
        // Current implementation defaults Auto to Cpu
        let device = KjarniDevice::Auto;
        assert_eq!(device.resolve(), KjarniDevice::Cpu);
    }

    #[test]
    fn test_resolve_is_idempotent() {
        // Resolving twice should give the same result
        let cpu = KjarniDevice::Cpu;
        assert_eq!(cpu.resolve().resolve(), KjarniDevice::Cpu);

        let gpu = KjarniDevice::Gpu;
        assert_eq!(gpu.resolve().resolve(), KjarniDevice::Gpu);

        let auto = KjarniDevice::Auto;
        assert_eq!(auto.resolve().resolve(), KjarniDevice::Cpu);
    }

    #[test]
    fn test_to_device_cpu() {
        let device = KjarniDevice::Cpu;
        assert_eq!(device.to_device(), Device::Cpu);
    }

    #[test]
    fn test_to_device_gpu() {
        let device = KjarniDevice::Gpu;
        assert_eq!(device.to_device(), Device::Wgpu);
    }

    #[test]
    fn test_to_device_auto() {
        let device = KjarniDevice::Auto;
        assert_eq!(device.to_device(), Device::Cpu);
    }

    #[test]
    fn test_is_cpu_for_cpu() {
        assert!(KjarniDevice::Cpu.is_cpu());
    }

    #[test]
    fn test_is_cpu_for_gpu() {
        assert!(!KjarniDevice::Gpu.is_cpu());
    }

    #[test]
    fn test_is_cpu_for_auto() {
        // Auto resolves to Cpu currently
        assert!(KjarniDevice::Auto.is_cpu());
    }

    #[test]
    fn test_is_gpu_for_cpu() {
        assert!(!KjarniDevice::Cpu.is_gpu());
    }

    #[test]
    fn test_is_gpu_for_gpu() {
        assert!(KjarniDevice::Gpu.is_gpu());
    }

    #[test]
    fn test_is_gpu_for_auto() {
        assert!(!KjarniDevice::Auto.is_gpu());
    }

    #[test]
    fn test_is_cpu_and_is_gpu_mutually_exclusive() {
        let devices = [KjarniDevice::Cpu, KjarniDevice::Gpu, KjarniDevice::Auto];

        for device in devices {
            let resolved = device.resolve();
            if resolved == KjarniDevice::Cpu {
                assert!(device.is_cpu());
                assert!(!device.is_gpu());
            } else if resolved == KjarniDevice::Gpu {
                assert!(!device.is_cpu());
                assert!(device.is_gpu());
            }
        }
    }
    #[test]
    fn test_from_kjarni_device_cpu() {
        let device: Device = KjarniDevice::Cpu.into();
        assert_eq!(device, Device::Cpu);
    }

    #[test]
    fn test_from_kjarni_device_gpu() {
        let device: Device = KjarniDevice::Gpu.into();
        assert_eq!(device, Device::Wgpu);
    }

    #[test]
    fn test_from_kjarni_device_auto() {
        let device: Device = KjarniDevice::Auto.into();
        assert_eq!(device, Device::Cpu);
    }

    #[test]
    fn test_from_explicit_conversion() {
        assert_eq!(Device::from(KjarniDevice::Cpu), Device::Cpu);
        assert_eq!(Device::from(KjarniDevice::Gpu), Device::Wgpu);
        assert_eq!(Device::from(KjarniDevice::Auto), Device::Cpu);
    }
    #[test]
    fn test_kjarni_device_debug_cpu() {
        let debug_str = format!("{:?}", KjarniDevice::Cpu);
        assert_eq!(debug_str, "Cpu");
    }

    #[test]
    fn test_kjarni_device_debug_gpu() {
        let debug_str = format!("{:?}", KjarniDevice::Gpu);
        assert_eq!(debug_str, "Gpu");
    }

    #[test]
    fn test_kjarni_device_debug_auto() {
        let debug_str = format!("{:?}", KjarniDevice::Auto);
        assert_eq!(debug_str, "Auto");
    }
    #[test]
    fn test_kjarni_device_clone() {
        let cpu = KjarniDevice::Cpu;
        let cpu_clone = cpu.clone();
        assert_eq!(cpu, cpu_clone);

        let gpu = KjarniDevice::Gpu;
        let gpu_clone = gpu.clone();
        assert_eq!(gpu, gpu_clone);

        let auto = KjarniDevice::Auto;
        let auto_clone = auto.clone();
        assert_eq!(auto, auto_clone);
    }

    #[test]
    fn test_kjarni_device_copy() {
        let device = KjarniDevice::Gpu;
        let copied = device; // Copy, not move
        assert_eq!(device, copied);
        assert_eq!(device, KjarniDevice::Gpu);
    }
    #[test]
    fn test_kjarni_device_equality() {
        assert_eq!(KjarniDevice::Cpu, KjarniDevice::Cpu);
        assert_eq!(KjarniDevice::Gpu, KjarniDevice::Gpu);
        assert_eq!(KjarniDevice::Auto, KjarniDevice::Auto);
    }

    #[test]
    fn test_kjarni_device_inequality() {
        assert_ne!(KjarniDevice::Cpu, KjarniDevice::Gpu);
        assert_ne!(KjarniDevice::Cpu, KjarniDevice::Auto);
        assert_ne!(KjarniDevice::Gpu, KjarniDevice::Auto);
    }
    #[test]
    fn test_download_policy_default_is_if_missing() {
        let policy = DownloadPolicy::default();
        assert_eq!(policy, DownloadPolicy::IfMissing);
    }
    #[test]
    fn test_download_policy_debug_if_missing() {
        let debug_str = format!("{:?}", DownloadPolicy::IfMissing);
        assert_eq!(debug_str, "IfMissing");
    }

    #[test]
    fn test_download_policy_debug_never() {
        let debug_str = format!("{:?}", DownloadPolicy::Never);
        assert_eq!(debug_str, "Never");
    }

    #[test]
    fn test_download_policy_debug_eager() {
        let debug_str = format!("{:?}", DownloadPolicy::Eager);
        assert_eq!(debug_str, "Eager");
    }
    #[test]
    fn test_download_policy_clone() {
        let if_missing = DownloadPolicy::IfMissing;
        assert_eq!(if_missing, if_missing.clone());

        let never = DownloadPolicy::Never;
        assert_eq!(never, never.clone());

        let eager = DownloadPolicy::Eager;
        assert_eq!(eager, eager.clone());
    }
    #[test]
    fn test_download_policy_copy() {
        let policy = DownloadPolicy::Never;
        let copied = policy;
        assert_eq!(policy, copied);
        assert_eq!(policy, DownloadPolicy::Never);
    }
    #[test]
    fn test_download_policy_equality() {
        assert_eq!(DownloadPolicy::IfMissing, DownloadPolicy::IfMissing);
        assert_eq!(DownloadPolicy::Never, DownloadPolicy::Never);
        assert_eq!(DownloadPolicy::Eager, DownloadPolicy::Eager);
    }

    #[test]
    fn test_download_policy_inequality() {
        assert_ne!(DownloadPolicy::IfMissing, DownloadPolicy::Never);
        assert_ne!(DownloadPolicy::IfMissing, DownloadPolicy::Eager);
        assert_ne!(DownloadPolicy::Never, DownloadPolicy::Eager);
    }
    #[test]
    fn test_download_policy_all_variants() {
        let variants = [
            DownloadPolicy::IfMissing,
            DownloadPolicy::Never,
            DownloadPolicy::Eager,
        ];

        for (i, v1) in variants.iter().enumerate() {
            for (j, v2) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(v1, v2);
                } else {
                    assert_ne!(v1, v2);
                }
            }
        }
    }
    #[test]
    fn test_kjarni_device_all_variants() {
        let variants = [
            KjarniDevice::Cpu,
            KjarniDevice::Gpu,
            KjarniDevice::Auto,
        ];
        for (i, v1) in variants.iter().enumerate() {
            for (j, v2) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(v1, v2);
                } else {
                    assert_ne!(v1, v2);
                }
            }
        }
    }
    #[test]
    fn test_device_workflow() {
        let user_wants_gpu = true;
        
        let device = if user_wants_gpu {
            KjarniDevice::Gpu
        } else {
            KjarniDevice::Cpu
        };
        
        let resolved = device.resolve();
        let low_level: Device = resolved.into();
        
        assert_eq!(resolved, KjarniDevice::Gpu);
        assert_eq!(low_level, Device::Wgpu);
    }

    #[test]
    fn test_auto_device_workflow() {
        let device = KjarniDevice::Auto;
        let resolved = device.resolve();
        assert!(resolved == KjarniDevice::Cpu || resolved == KjarniDevice::Gpu);
        let low_level: Device = device.into();
        assert!(low_level == Device::Cpu || low_level == Device::Wgpu);
    }

    #[test]
    fn test_device_consistency() {
        for device in [KjarniDevice::Cpu, KjarniDevice::Gpu, KjarniDevice::Auto] {
            let via_method = device.to_device();
            let via_from: Device = device.into();
            assert_eq!(via_method, via_from);
        }
    }
    #[test]
    fn test_multiple_resolves() {
        let auto = KjarniDevice::Auto;
        let r1 = auto.resolve();
        let r2 = r1.resolve();
        let r3 = r2.resolve();
        
        assert_eq!(r1, r2);
        assert_eq!(r2, r3);
    }

    #[test]
    fn test_is_methods_after_resolve() {
        let auto = KjarniDevice::Auto;
        let resolved = auto.resolve();
        
        assert_eq!(auto.is_cpu(), resolved.is_cpu());
        assert_eq!(auto.is_gpu(), resolved.is_gpu());
    }
    #[test]
    fn test_kjarni_device_size() {
        assert!(std::mem::size_of::<KjarniDevice>() <= 1);
    }

    #[test]
    fn test_download_policy_size() {
        assert!(std::mem::size_of::<DownloadPolicy>() <= 1);
    }
}