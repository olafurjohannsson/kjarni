use crate::WgpuContext; // Assuming WgpuContext is accessible from the crate root
use crate::gpu_ops::primitives::layout::permute::GpuPermute;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use anyhow::{Result, anyhow};
use ndarray::{Array, Array2, Array3, Array4, Dimension};
use std::fmt;
use std::sync::Arc;
use wgpu::CommandEncoder;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

/// Defines the data type of the elements in a GpuTensor.
/// This is crucial for handling different weight formats from files like GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    U32,
    I8,
    Q8_0,
    Q8_1,
    Q4_0,
    Q4_1,
    Q2_0,
    Q2_1,
    // Add other types like F16, Q8_0, etc., as you implement them
}
impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::U32 => std::mem::size_of::<u32>(),
            DType::I8 => std::mem::size_of::<i8>(),
            // TODO: Add sizes for other dtypes as they are implemented.
            // For now, panic if the size is not defined.
            _ => unimplemented!("Size for dtype {:?} is not defined", self),
        }
    }
}

pub trait GpuDType: bytemuck::Pod {
    const DTYPE: DType;
}

impl GpuDType for f32 {
    const DTYPE: DType = DType::F32;
}
impl GpuDType for u32 {
    const DTYPE: DType = DType::U32;
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
        let expected_size = shape.iter().product::<usize>() * dtype.size_of();
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

    /// Creates a new GpuTensor initialized with zeros.
    ///
    /// # Arguments
    /// * `context` - The WGPU context.
    /// * `shape` - The desired shape of the tensor.
    /// * `dtype` - The data type of the tensor elements.
    /// * `label` - A descriptive label for debugging.
    ///
    /// # Returns
    /// A new `GpuTensor` filled with zeros.
    pub fn zeros(
        context: &Arc<WgpuContext>,
        shape: Vec<usize>,
        dtype: DType,
        label: &str,
    ) -> Result<Self> {
        let num_elements = shape.iter().product::<usize>();
        let size_in_bytes = num_elements * dtype.size_of();

        // Create a zero-filled vector on the CPU.
        let zeros_data = vec![0u8; size_in_bytes];

        // Create a GPU buffer and immediately initialize it with the zero data.
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &zeros_data,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape,
            dtype,
            context: context.clone(),
        })
    }
    pub fn dims2(&self) -> (usize, usize) {
        assert_eq!(self.rank(), 2, "Tensor is not rank 2");
        (self.shape[0], self.shape[1])
    }
    /// Returns the dimensions of the tensor as a 3-tuple.
    /// Panics if the tensor is not rank 3.
    pub fn dims3(&self) -> (usize, usize, usize) {
        assert_eq!(self.rank(), 3, "Tensor is not rank 3");
        (self.shape[0], self.shape[1], self.shape[2])
    }

    /// Returns the dimensions of the tensor as a 4-tuple.
    /// Panics if the tensor is not rank 4.
    pub fn dims4(&self) -> (usize, usize, usize, usize) {
        assert_eq!(self.rank(), 4, "Tensor is not rank 4");
        (self.shape[0], self.shape[1], self.shape[2], self.shape[3])
    }
    // View a 4D tensor as 3D by merging the first two dimensions
    /// [B, H, S, D] -> [B*H, S, D]
    pub fn view_as_3d(&self, dim0: usize, dim1: usize, dim2: usize) -> Result<GpuTensor> {
        assert_eq!(self.rank(), 4, "Can only view 4D tensors as 3D");

        let (b, h, s, d) = self.dims4();
        assert_eq!(dim0, b * h, "First dimension must equal batch * heads");
        assert_eq!(dim1, s, "Second dimension must match");
        assert_eq!(dim2, d, "Third dimension must match");

        // Create a new tensor with 3D shape but same buffer
        Ok(GpuTensor {
            context: self.context.clone(),
            buffer: self.buffer.clone(),
            shape: vec![dim0, dim1, dim2],
            dtype: self.dtype,
            // label: format!("{}_view3d", self.label),
        })
    }

    /// View a [B*H, S, D] tensor as [B, H, S, D]
    pub fn view_as_4d(&self, b: usize, h: usize, s: usize, d: usize) -> Result<GpuTensor> {
        assert_eq!(self.rank(), 3, "Can only view 3D tensors as 4D");
        let (bh, s0, d0) = self.dims3();
        assert_eq!(bh, b * h, "First dimension mismatch (B*H != bh)");
        assert_eq!(s, s0, "Second dimension mismatch");
        assert_eq!(d, d0, "Third dimension mismatch");

        Ok(GpuTensor {
            context: self.context.clone(),
            buffer: self.buffer.clone(),
            shape: vec![b, h, s, d],
            dtype: self.dtype,
        })
    }
    /// Creates a new view of the tensor with a different shape, without copying data.
    ///
    /// This is a metadata-only operation. The new `GpuTensor` shares the same underlying
    /// GPU buffer as the original.
    ///
    /// Panics if the total number of elements in the new shape does not match the original.
    pub fn view(&self, shape: Vec<usize>) -> Self {
        let new_num_elements: usize = shape.iter().product();
        assert_eq!(
            self.num_elements(),
            new_num_elements,
            "Cannot view tensor of shape {:?} as {:?}; number of elements mismatch.",
            self.shape(),
            &shape
        );

        Self {
            buffer: self.buffer.clone(), // Clone the Arc, not the buffer data
            shape,
            dtype: self.dtype,
            context: self.context.clone(),
        }
    }

    /// Creates a new, uninitialized tensor, for example, to serve as an output buffer for a kernel.
    pub fn uninitialized(
        context: &Arc<WgpuContext>,
        shape: Vec<usize>,
        dtype: DType,
        label: &str,
    ) -> Self {
        let size = (shape.iter().product::<usize>() * dtype.size_of()) as u64;
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

    /// Creates a new GpuTensor by copying a slice from this tensor using a dedicated kernel.
    ///
    /// # Arguments
    /// * `encoder` - The command encoder to record the copy command.
    /// * `slice_kernel` - A reference to the pre-initialized GpuSlice kernel.
    /// * `offset` - The starting indices for the slice in each dimension (e.g., [0, 0, 5, 0]).
    /// * `shape` - The desired shape of the new sliced tensor (e.g., [1, 4, 1, 64]).
    ///
    /// # Returns
    /// A new, tightly-sized `GpuTensor` containing the copied data.
    pub fn slice(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slice_kernel: &GpuSlice,
        offset: &[usize],
        shape: &[usize],
    ) -> Result<GpuTensor> {
        assert_eq!(
            self.rank(),
            offset.len(),
            "Offset dimensions must match tensor rank."
        );
        assert_eq!(
            self.rank(),
            shape.len(),
            "Shape dimensions must match tensor rank."
        );
        for i in 0..self.rank() {
            assert!(
                offset[i] + shape[i] <= self.shape()[i],
                "Slice exceeds source tensor dimensions."
            );
        }

        // 1. Create the destination tensor with the target shape.
        let dst_tensor =
            GpuTensor::uninitialized(&self.context, shape.to_vec(), self.dtype, "Sliced Tensor");

        // 2. Encode the kernel call to perform the copy.
        slice_kernel.encode(encoder, self, &dst_tensor, offset);

        // 3. Return the new tensor.
        Ok(dst_tensor)
    }

    /// Permutes the dimensions of the tensor according to the specified axes.
    ///
    /// This operation performs a GPU copy and requires a command encoder.
    ///
    /// # Arguments
    /// * `encoder` - A command encoder to record the GPU operation.
    /// * `permute_kernel` - A reference to the pre-compiled GpuPermute kernel.
    /// * `axes` - A slice defining the permutation. e.g., `&[0, 2, 1, 3]` to swap axes 1 and 2.
    ///
    /// # Returns
    /// A new `GpuTensor` containing the permuted data.
    pub fn permute(
        &self,
        encoder: &mut CommandEncoder,
        permute_kernel: &GpuPermute,
        axes: &[usize],
    ) -> Self {
        assert_eq!(
            self.rank(),
            axes.len(),
            "Permutation axes must match tensor rank"
        );

        // 1. Calculate the shape of the output tensor
        let mut output_shape = self.shape().to_vec();
        for i in 0..self.rank() {
            output_shape[i] = self.shape()[axes[i]];
        }

        // 2. Create a new, uninitialized tensor to hold the permuted output
        let output = GpuTensor::uninitialized(
            &self.context,
            output_shape,
            self.dtype,
            "Permute Output", // A descriptive label for debugging
        );

        // 3. Encode the kernel call to perform the data permutation
        permute_kernel.encode(encoder, self, &output, axes);

        // 4. Return the new tensor
        output
    }
    /// Creates a GpuTensor by copying data from a CPU-based ndarray::Array.
    /// This is the primary way to get data from the CPU onto the GPU.
    pub fn from_ndarray<A, D>(context: &Arc<WgpuContext>, arr: &Array<A, D>) -> Result<Self>
    where
        A: GpuDType,
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
            dtype: A::DTYPE,
            context: context.clone(),
        })
    }

    pub async fn to_ndarray_2d<A>(&self) -> Result<Array2<A>>
    where
        A: GpuDType + Copy,
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
        A: GpuDType + Copy,
    {
        anyhow::ensure!(self.rank() == 3, "Tensor rank is not 3");
        let raw_data = self.read_raw_data().await?;
        let data_slice: &[A] = bytemuck::cast_slice(&raw_data);
        Ok(Array3::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2]),
            data_slice.to_vec(),
        )?)
    }

    /// Asynchronously reads the GpuTensor's data back to the CPU as an Array3.
    pub async fn to_ndarray_4d<A>(&self) -> Result<Array4<A>>
    where
        A: GpuDType + Copy,
    {
        anyhow::ensure!(self.rank() == 4, "Tensor rank is not 4");
        let raw_data = self.read_raw_data().await?;
        let data_slice: &[A] = bytemuck::cast_slice(&raw_data);
        Ok(Array4::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2], self.shape[3]),
            data_slice.to_vec(),
        )?)
    }

    // pub async fn read_raw_data(&self) -> Result<Vec<u8>> {
    //     let buffer_slice = self.buffer.slice(..);

    //     let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    //     buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
    //         let _ = tx.send(result);
    //     });

    //     self.context
    //         .device
    //         .poll(wgpu::PollType::wait_indefinitely());
    //     rx.receive()
    //         .await
    //         .ok_or(anyhow!("GPU readback channel closed"))??;

    //     let data = buffer_slice.get_mapped_range().to_vec();
    //     self.buffer.unmap();

    //     Ok(data)
    // }
    pub async fn read_raw_data(&self) -> Result<Vec<u8>> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let buffer_size = self.buffer.size();

        // 1. Create a temporary "staging" buffer that is readable by the CPU.
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Readback Buffer"),
            size: buffer_size,
            // This is the correct usage combination for a readback buffer.
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 2. Encode a command to copy from the GPU-only compute buffer to the staging buffer.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &staging_buffer, 0, buffer_size);

        // 3. Submit the copy command to the GPU.
        queue.submit(Some(encoder.finish()));

        // 4. Map the staging buffer and wait for the result.
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // This is a critical step to ensure the submission is processed.
        device.poll(wgpu::PollType::wait_indefinitely());

        // Wait for the map_async callback to complete
        rx.receive()
            .await
            .ok_or(anyhow!("GPU readback channel closed"))??;

        // 5. Get the data from the mapped staging buffer.
        let data = buffer_slice.get_mapped_range().to_vec();

        // 6. Unmap the staging buffer so it can be reused or dropped.
        staging_buffer.unmap();

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
    pub fn dtype_as_str(&self) -> &str {
        match self.dtype {
            DType::F32 => "f32",
            DType::U32 => "u32",
            DType::I8 => "i8",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::Q8_0 => "q8_0",
            DType::Q8_1 => "q8_1",
            DType::Q4_0 => "q4_0",
            DType::Q4_1 => "q4_1",
            DType::Q2_0 => "q2_0",
            DType::Q2_1 => "q2_1",
            _ => "unknown",
        }
    }
}
