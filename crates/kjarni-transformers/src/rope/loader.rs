use crate::{WgpuContext, gpu_ops::blocks::rope::GpuRoPE, rope::RoPE, traits::ModelMetadata};
use anyhow::Result;
use std::sync::Arc;

pub struct LoadedRoPE {
    pub cpu: Arc<RoPE>,
    pub gpu: Option<Arc<GpuRoPE>>,
}

impl LoadedRoPE {
    pub fn new(
        ctx: Option<&Arc<WgpuContext>>,
        meta: &ModelMetadata,
        load_gpu: bool,
    ) -> Result<Self> {
        let cpu_rope = Arc::new(RoPE::new_with_scaling(
            meta.head_dim,
            meta.max_seq_len,
            meta.rope_theta.unwrap_or(10000.0),
            meta.rope_scaling.as_ref(),
        ));

        let gpu = if load_gpu {
            let c = ctx.ok_or_else(|| anyhow::anyhow!("GPU RoPE requires Context"))?;
            Some(Arc::new(GpuRoPE::new(
                c,
                &cpu_rope.cos_cache,
                &cpu_rope.sin_cache,
            )?))
        } else {
            None
        };

        Ok(Self { cpu: cpu_rope, gpu })
    }
}
