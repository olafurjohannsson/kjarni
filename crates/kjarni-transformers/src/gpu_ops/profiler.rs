use crate::WgpuContext;
use std::sync::Mutex;

/// Conditionally profile a compute pass.
///
/// In release builds without `profiling` feature, this compiles to
/// a plain compute pass with zero overhead.
#[cfg(feature = "profiling")]
#[macro_export]
macro_rules! gpu_profile {
    ($ctx:expr, $encoder:expr, $label:expr, $body:expr) => {{
        $ctx.profiler.profile(
            $encoder.raw(),
            $label,
            |pass: &mut wgpu::ComputePass<'_>| $body(pass),
        )
    }};
}

#[cfg(not(feature = "profiling"))]
#[macro_export]
macro_rules! gpu_profile {
    ($ctx:expr, $encoder:expr, $label:expr, $body:expr) => {{
        let _ = $label;
        let pass: &mut wgpu::ComputePass<'static> = $encoder.pass();
        $body(pass)
    }};
}

pub struct GpuProfiler {
    /// `None` when the adapter has no timestamp queries, which is the normal
    /// case in a browser: WebGPU keeps them behind a developer flag.
    query_set: Option<wgpu::QuerySet>,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    labels: Mutex<Vec<String>>,
    max_queries: u32,
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device, max_queries: u32, timestamps: bool) -> Self {
        let query_set = timestamps.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Profiler Query Set"),
                ty: wgpu::QueryType::Timestamp,
                count: max_queries,
            })
        });

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Profiler Resolve Buffer"),
            size: (max_queries as u64) * 8, // u64 per query
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let destination_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Profiler Readback Buffer"),
            size: (max_queries as u64) * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buffer,
            destination_buffer,
            labels: Mutex::new(Vec::new()),
            max_queries,
        }
    }

    /// Wraps a compute pass with timestamps
    pub fn profile<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        label: &str,
        callback: impl FnOnce(&mut wgpu::ComputePass<'a>),
    ) {
        let mut labels = self.labels.lock().unwrap();
        let index = labels.len() as u32 * 2;

        let Some(query_set) = self.query_set.as_ref() else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            callback(&mut pass);
            return;
        };

        if index + 2 > self.max_queries {
            // Buffer full, just run without profiling
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            callback(&mut pass);
            return;
        }

        labels.push(label.to_string());

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: Some(index + 1),
            }),
        });

        callback(&mut pass);
    }

    /// Prepares buffers for reading (Must call before resolve)
    pub fn process_results(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = { self.labels.lock().unwrap().len() as u32 * 2 };
        let Some(query_set) = self.query_set.as_ref() else {
            return;
        };
        if count == 0 {
            return;
        }

        encoder.resolve_query_set(query_set, 0..count, &self.resolve_buffer, 0);

        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            (count as u64) * 8,
        );
    }

    /// Reads the timings back, totalled per label, longest first, as
    /// (label, milliseconds, dispatches), and starts counting afresh. Call after
    /// `process_results` has been submitted.
    pub async fn totals(&self, context: &WgpuContext) -> Vec<(String, f64, usize)> {
        let labels = std::mem::take(&mut *self.labels.lock().unwrap());
        if labels.is_empty() || self.query_set.is_none() {
            return Vec::new();
        }
        let slice = self
            .destination_buffer
            .slice(0..((labels.len() as u64) * 16));
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        crate::gpu_ops::context::drain(&context.device);
        if !matches!(rx.receive().await, Some(Ok(()))) {
            return Vec::new();
        }
        let period = context.queue.get_timestamp_period() as f64;
        let mut totals: Vec<(String, f64, usize)> = Vec::new();
        {
            let data = slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);
            for (i, label) in labels.iter().enumerate() {
                let (start, end) = (timestamps[i * 2], timestamps[i * 2 + 1]);
                let ms = end.saturating_sub(start) as f64 * period / 1e6;
                match totals.iter_mut().find(|t| &t.0 == label) {
                    Some(t) => {
                        t.1 += ms;
                        t.2 += 1;
                    }
                    None => totals.push((label.clone(), ms, 1)),
                }
            }
        }
        self.destination_buffer.unmap();
        totals.sort_by(|a, b| b.1.total_cmp(&a.1));
        totals
    }

    /// Reads data back from GPU (Async/Blocking)
    pub async fn print_stats(&self, context: &WgpuContext) {
        let labels = { self.labels.lock().unwrap().clone() };

        if labels.is_empty() {
            return;
        }

        let slice = self
            .destination_buffer
            .slice(0..((labels.len() as u64) * 16));

        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());

        crate::gpu_ops::context::drain(&context.device);
        rx.receive().await.unwrap().expect("RX ERROR"); // TODO: do better

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let period = context.queue.get_timestamp_period();

        log::info!("\n=== GPU KERNEL PROFILER ===");
        let mut total = 0.0;
        for (i, label) in labels.iter().enumerate() {
            let start = timestamps[i * 2];
            let end = timestamps[i * 2 + 1];

            if end < start {
                continue;
            }

            let duration_ns = (end - start) as f32 * period;
            let duration_ms = duration_ns / 1_000_000.0;
            total += duration_ms;

            log::info!("{:<40} : {:.4} ms", label, duration_ms);
        }
        log::info!("TOTAL GPU TIME: {:.4} ms", total);

        drop(data);
        self.destination_buffer.unmap();

        // Reset for next frame
        self.labels.lock().unwrap().clear();
    }
}

#[cfg(all(test, feature = "profiling"))]
mod tests {
    use crate::WgpuContext;
    use crate::gpu::{ComputeKernel, GpuTensor};
    use ndarray::Array1;

    /// Dispatches through `ComputeKernel` are timed under their kernel's label and
    /// totalled per label.
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn kernels_are_totalled_per_label() {
        let ctx = WgpuContext::new().await.unwrap();
        let kernel = ComputeKernel::new(
            &ctx.device,
            "Profiled fill",
            wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var<uniform> n: vec4<u32>;
                     @group(0) @binding(1) var<storage, read_write> out: array<f32>;
                     @compute @workgroup_size(64)
                     fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                         if (id.x < n.x) { out[id.x] = f32(id.x); }
                     }"
                    .into(),
                ),
            },
            &[false],
        );
        let out = GpuTensor::from_ndarray(&ctx, &Array1::<f32>::zeros(1024)).unwrap();
        let uniforms = ctx.uniform_buffer(&[1024u32, 0, 0, 0]);
        ctx.profiler.totals(&ctx).await;
        let mut encoder = crate::gpu::GpuEncoder::from_encoder(
            ctx.device.create_command_encoder(&Default::default()),
        );
        kernel.encode(&ctx, &mut encoder, &uniforms, &[&out], 1024, 64);
        kernel.encode(&ctx, &mut encoder, &uniforms, &[&out], 1024, 64);
        ctx.profiler.process_results(&mut encoder);
        ctx.queue.submit(Some(encoder.finish()));

        let totals = ctx.profiler.totals(&ctx).await;
        if ctx.profiler.query_set.is_none() {
            assert!(totals.is_empty());
            return;
        }
        assert_eq!(totals.len(), 1, "{totals:?}");
        assert_eq!((totals[0].0.as_str(), totals[0].2), ("Profiled fill", 2));
        assert!(totals[0].1 >= 0.0);
    }
}
