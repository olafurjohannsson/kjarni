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
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(0);

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
// #[derive(Clone)]
pub struct GpuTensor {
    buffer: Arc<Buffer>,
    shape: Vec<usize>,
    dtype: DType,
    context: Arc<WgpuContext>,
    id: u64,
}
impl Clone for GpuTensor {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer), // ✅ Cheap Arc clone
            shape: self.shape.clone(),
            dtype: self.dtype,
            context: self.context.clone(),
            id: self.id, // ✅ CRITICAL: Keep same ID (same buffer!)
        }
    }
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
    /// Internal constructor - generates new ID (new allocation)
    fn new_allocation(
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
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed), // ✅ New ID
        }
    }

    /// Internal constructor - keeps same ID (view operation)
    fn new_view(
        buffer: Arc<Buffer>,
        shape: Vec<usize>,
        dtype: DType,
        context: Arc<WgpuContext>,
        id: u64, // Inherit parent's ID
    ) -> Self {
        Self {
            buffer,
            shape,
            dtype,
            context,
            id, // ✅ Keep same ID
        }
    }

    /// Get the unique buffer ID
    pub fn buffer_id(&self) -> u64 {
        self.id
    }

    /// Creates a view with different axis permutation (metadata only)
    pub fn permute_axes(&self, axes: &[usize]) -> GpuTensor {
        assert_eq!(axes.len(), self.rank(), "Permutation axes must match tensor rank");

        let mut new_shape = vec![0; self.rank()];
        for (i, &axis) in axes.iter().enumerate() {
            new_shape[i] = self.shape[axis];
        }

        Self::new_view(
            self.buffer.clone(),
            new_shape,
            self.dtype,
            self.context.clone(),
            self.id, // ✅ Keep same ID (it's a view)
        )
    }

    /// Creates a new GpuTensor initialized with zeros
    pub fn zeros(
        context: &Arc<WgpuContext>,
        shape: Vec<usize>,
        dtype: DType,
        label: &str,
    ) -> Result<Self> {
        let num_elements = shape.iter().product::<usize>();
        let size_in_bytes = num_elements * dtype.size_of();
        let zeros_data = vec![0u8; size_in_bytes];

        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &zeros_data,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

        Ok(Self::new_allocation(
            Arc::new(buffer),
            shape,
            dtype,
            context.clone(),
        )) // ✅ New allocation = new ID
    }

    /// Creates a new view with different shape (no copy)
    pub fn view(&self, shape: Vec<usize>) -> Self {
        let new_num_elements: usize = shape.iter().product();
        assert_eq!(
            self.num_elements(),
            new_num_elements,
            "Cannot view tensor of shape {:?} as {:?}; number of elements mismatch.",
            self.shape(),
            &shape
        );

        Self::new_view(
            Arc::clone(&self.buffer),
            shape,
            self.dtype,
            self.context.clone(),
            self.id, // ✅ Keep same ID (it's a view)
        )
    }

    /// View 4D as 3D
    pub fn view_as_3d(&self, dim0: usize, dim1: usize, dim2: usize) -> Result<GpuTensor> {
        assert_eq!(self.rank(), 4, "Can only view 4D tensors as 3D");
        let (b, h, s, d) = self.dims4();
        assert_eq!(dim0, b * h, "First dimension must equal batch * heads");
        assert_eq!(dim1, s, "Second dimension must match");
        assert_eq!(dim2, d, "Third dimension must match");

        Ok(Self::new_view(
            self.buffer.clone(),
            vec![dim0, dim1, dim2],
            self.dtype,
            self.context.clone(),
            self.id, // ✅ Keep same ID (it's a view)
        ))
    }

    /// View 3D as 4D
    pub fn view_as_4d(&self, b: usize, h: usize, s: usize, d: usize) -> Result<GpuTensor> {
        assert_eq!(self.rank(), 3, "Can only view 3D tensors as 4D");
        let (bh, s0, d0) = self.dims3();
        assert_eq!(bh, b * h, "First dimension mismatch (B*H != bh)");
        assert_eq!(s, s0, "Second dimension mismatch");
        assert_eq!(d, d0, "Third dimension mismatch");

        Ok(Self::new_view(
            self.buffer.clone(),
            vec![b, h, s, d],
            self.dtype,
            self.context.clone(),
            self.id, // ✅ Keep same ID (it's a view)
        ))
    }

    /// Creates uninitialized tensor
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

        Self::new_allocation(
            Arc::new(buffer),
            shape,
            dtype,
            context.clone(),
        ) // ✅ New allocation = new ID
    }

    /// Creates from ndarray
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

        Ok(Self::new_allocation(
            Arc::new(buffer),
            shape,
            A::DTYPE,
            context.clone(),
        )) // ✅ New allocation = new ID
    }

    /// Permute with actual data copy (creates new buffer)
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

        let mut output_shape = self.shape().to_vec();
        for i in 0..self.rank() {
            output_shape[i] = self.shape()[axes[i]];
        }

        // Creates NEW buffer, so new ID
        let output = GpuTensor::uninitialized(
            &self.context,
            output_shape,
            self.dtype,
            "Permute Output",
        ); // ✅ New buffer = new ID

        permute_kernel.encode(encoder, self, &output, axes);
        output
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    async fn setup_context() -> Arc<WgpuContext> {
        Arc::new(WgpuContext::new().await.unwrap())
    }

    #[tokio::test]
    async fn test_tensor_creation() {
        let ctx = setup_context().await;
        let shape = vec![2, 3, 4];
        let tensor = GpuTensor::zeros(&ctx, shape.clone(), DType::F32, "test").unwrap();

        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert_eq!(tensor.rank(), 3);
        assert_eq!(tensor.num_elements(), 24);
    }

    #[tokio::test]
    async fn test_from_ndarray() {
        let ctx = setup_context().await;
        let arr = Array3::<f32>::ones((2, 3, 4));
        let tensor = GpuTensor::from_ndarray(&ctx, &arr).unwrap();

        assert_eq!(tensor.shape(), &[2, 3, 4]);

        let result = tensor.to_ndarray_3d::<f32>().await.unwrap();
        assert_eq!(result, arr);
    }

    #[tokio::test]
    async fn test_view_operations() {
        let ctx = setup_context().await;
        let tensor = GpuTensor::zeros(&ctx, vec![2, 3, 4], DType::F32, "test").unwrap();

        // Test view
        let viewed = tensor.view(vec![6, 4]);
        assert_eq!(viewed.shape(), &[6, 4]);
        assert_eq!(viewed.num_elements(), 24);
    }

    #[tokio::test]
    async fn test_view_4d_to_3d() {
        let ctx = setup_context().await;
        let tensor = GpuTensor::zeros(&ctx, vec![2, 4, 8, 64], DType::F32, "test").unwrap();

        let viewed = tensor.view_as_3d(8, 8, 64).unwrap();
        assert_eq!(viewed.shape(), &[8, 8, 64]);
    }

    #[tokio::test]
    async fn test_view_3d_to_4d() {
        let ctx = setup_context().await;
        let tensor = GpuTensor::zeros(&ctx, vec![8, 10, 64], DType::F32, "test").unwrap();

        let viewed = tensor.view_as_4d(2, 4, 10, 64).unwrap();
        assert_eq!(viewed.shape(), &[2, 4, 10, 64]);
    }

    #[tokio::test]
    async fn test_dims_helpers() {
        let ctx = setup_context().await;

        let t2d = GpuTensor::zeros(&ctx, vec![5, 10], DType::F32, "test").unwrap();
        assert_eq!(t2d.dims2(), (5, 10));

        let t3d = GpuTensor::zeros(&ctx, vec![2, 3, 4], DType::F32, "test").unwrap();
        assert_eq!(t3d.dims3(), (2, 3, 4));

        let t4d = GpuTensor::zeros(&ctx, vec![1, 2, 3, 4], DType::F32, "test").unwrap();
        assert_eq!(t4d.dims4(), (1, 2, 3, 4));
    }

    #[tokio::test]
    async fn test_roundtrip_cpu_gpu() {
        let ctx = setup_context().await;
        let original =
            Array3::<f32>::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as f32);

        let gpu_tensor = GpuTensor::from_ndarray(&ctx, &original).unwrap();
        let result = gpu_tensor.to_ndarray_3d::<f32>().await.unwrap();

        assert_eq!(result, original);
    }

    #[tokio::test]
    #[should_panic(expected = "Cannot view tensor")]
    async fn test_invalid_view() {
        let ctx = setup_context().await;
        let tensor = GpuTensor::zeros(&ctx, vec![2, 3, 4], DType::F32, "test").unwrap();

        // Wrong number of elements
        tensor.view(vec![2, 3, 5]);
    }

    #[tokio::test]
    async fn test_dtype_size() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::U32.size_of(), 4);
        assert_eq!(DType::I8.size_of(), 1);
    }
}
