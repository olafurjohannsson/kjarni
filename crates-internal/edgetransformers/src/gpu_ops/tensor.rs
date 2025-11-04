use crate::WgpuContext; // Assuming WgpuContext is accessible from the crate root
use anyhow::{Result, anyhow};
use ndarray::{Array, Array2, Array3, Dimension};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};
use std::fmt;

/// Defines the data type of the elements in a GpuTensor.
/// This is crucial for handling different weight formats from files like GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    // Add other types like F16, Q8_0, etc., as you implement them
}

/// A GPU-backed tensor that bundles a wgpu::Buffer with its shape and data type.
/// It holds a reference-counted pointer to the buffer and context, making it cheap to clone.
#[derive(Clone)]
pub struct GpuTensor {
    buffer: Arc<Buffer>,
    shape: Vec<usize>,
    dtype: DType,
    context: Arc<WgpuContext>,
}

impl fmt::Debug for GpuTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("buffer_size", &self.buffer.size()) 
            .finish_non_exhaustive() // Indicates that fields like `context` are omitted
    }
}

impl GpuTensor {
    /// Creates a new GpuTensor from an existing buffer and its metadata.
    /// This is the primary constructor.
    pub fn new(
        buffer: Arc<Buffer>,
        shape: Vec<usize>,
        dtype: DType,
        context: Arc<WgpuContext>,
    ) -> Self {
        let expected_size = shape.iter().product::<usize>() * std::mem::size_of::<f32>(); // Adjust this for other dtypes later
        assert_eq!(
            buffer.size() as usize,
            expected_size,
            "Buffer size does not match shape dimensions"
        );
        Self {
            buffer,
            shape,
            dtype,
            context,
        }
    }

    /// Creates a new, uninitialized tensor, for example, to serve as an output buffer for a kernel.
    pub fn uninitialized(
        context: &Arc<WgpuContext>,
        shape: Vec<usize>,
        dtype: DType,
        label: &str,
    ) -> Self {
        let size = (shape.iter().product::<usize>() * std::mem::size_of::<f32>()) as u64; // Adjust for dtype
        let buffer = context.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer: Arc::new(buffer),
            shape,
            dtype,
            context: context.clone(),
        }
    }

    /// Creates a GpuTensor by copying data from a CPU-based ndarray::Array.
    /// This is the primary way to get data from the CPU onto the GPU.
    pub fn from_ndarray<A, D>(context: &Arc<WgpuContext>, arr: &Array<A, D>) -> Result<Self>
    where
        A: bytemuck::Pod,
        D: Dimension,
    {
        let shape = arr.shape().to_vec();
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tensor from ndarray"),
                contents: bytemuck::cast_slice(
                    arr.as_standard_layout().as_slice().ok_or_else(|| {
                        anyhow!("Failed to get slice from ndarray for GPU transfer")
                    })?,
                ),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape,
            dtype: DType::F32, // Assuming F32 for now
            context: context.clone(),
        })
    }

    pub async fn to_ndarray_2d<A>(&self) -> Result<Array2<A>>
    where
        A: bytemuck::Pod + Copy,
    {
        anyhow::ensure!(self.rank() == 2, "Tensor rank is not 2");
        let raw_data = self.read_raw_data().await?;
        let data_slice: &[A] = bytemuck::cast_slice(&raw_data);
        Ok(Array2::from_shape_vec(
            (self.shape[0], self.shape[1]),
            data_slice.to_vec(),
        )?)
    }

    /// Asynchronously reads the GpuTensor's data back to the CPU as an Array3.
    pub async fn to_ndarray_3d<A>(&self) -> Result<Array3<A>>
    where
        A: bytemuck::Pod + Copy,
    {
        anyhow::ensure!(self.rank() == 3, "Tensor rank is not 3");
        let raw_data = self.read_raw_data().await?;
        let data_slice: &[A] = bytemuck::cast_slice(&raw_data);
        Ok(Array3::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2]),
            data_slice.to_vec(),
        )?)
    }

    async fn read_raw_data(&self) -> Result<Vec<u8>> {
        let buffer_slice = self.buffer.slice(..);

        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.context.device.poll(wgpu::PollType::wait_indefinitely());
        rx.receive()
            .await
            .ok_or(anyhow!("GPU readback channel closed"))??;

        let data = buffer_slice.get_mapped_range().to_vec();
        self.buffer.unmap();

        Ok(data)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}
