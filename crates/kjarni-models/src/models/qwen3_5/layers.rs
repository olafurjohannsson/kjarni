//! The two token mixers of a Qwen 3.5 layer, written against transformers'
//! `Qwen3_5Attention` and `Qwen3_5GatedDeltaNet`.

use anyhow::Result;
use kjarni_transformers::{
    cpu::decoder::streaming_attention::{decode_attention, streaming_causal_attention},
    linear_layer::LinearLayer,
    normalization::RMSNorm,
    rope::RoPE,
};
use ndarray::{Array1, Array2, Array3, Array4, ArrayViewMut3, Axis, s};
use rayon::prelude::*;

use super::cache::LinearState;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `torch.nn.functional.softplus` with its default threshold of 20, above which
/// it returns `x` unchanged.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Qwen 3.5's RMSNorm scales by `1 + weight`, so the checkpoint stores the
/// offset from one. Adding the one at load time lets the plain norm run it.
pub fn offset_norm(weight: Array1<f32>, eps: f32) -> RMSNorm {
    RMSNorm::new(weight + 1.0, eps)
}

/// Attention with per-head RMSNorm on q and k, rotary embedding on the leading
/// `rotary_dim` dimensions of each head, and a sigmoid gate on the output that
/// `q_proj` computes alongside the query.
pub struct GatedAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
}

impl GatedAttention {
    /// Rotates the first `rotary_dim` values of each `head_dim` row of `x`, a
    /// `[tokens * heads, head_dim]` matrix whose rows run head-fastest.
    fn rotate(&self, x: &mut Array2<f32>, heads: usize, offset: usize, rope: &RoPE) {
        let half = self.rotary_dim / 2;
        for (row_idx, mut row) in x.axis_iter_mut(Axis(0)).enumerate() {
            let pos = offset + row_idx / heads;
            let cos = rope.cos_cache.row(pos);
            let sin = rope.sin_cache.row(pos);
            for i in 0..half {
                let (x0, x1) = (row[i], row[i + half]);
                row[i] = x0 * cos[i] - x1 * sin[i];
                row[i + half] = x0 * sin[i] + x1 * cos[i];
            }
        }
    }

    pub fn forward(
        &self,
        hidden: &Array3<f32>,
        mut k_cache: ArrayViewMut3<f32>,
        mut v_cache: ArrayViewMut3<f32>,
        offset: usize,
        rope: &RoPE,
    ) -> Result<Array3<f32>> {
        let (batch, seq, width) = hidden.dim();
        let (h, kvh, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let rows = batch * seq;
        let x = hidden.view().into_shape_with_order((rows, width))?;

        // q_proj gives, per head, the query followed by its gate.
        let qg = self.q_proj.matmul(&x);
        let qg = qg.into_shape_with_order((rows, h, 2 * hd))?;
        let mut q = qg
            .slice(s![.., .., ..hd])
            .to_owned()
            .into_shape_with_order((rows * h, hd))?;
        let gate = qg
            .slice(s![.., .., hd..])
            .to_owned()
            .into_shape_with_order((rows, h * hd))?;
        q = self.q_norm.forward_2d(&q);

        let k = self
            .k_proj
            .matmul(&x)
            .into_shape_with_order((rows * kvh, hd))?;
        let mut k = self.k_norm.forward_2d(&k);
        let v = self.v_proj.matmul(&x);

        self.rotate(&mut q, h, offset, rope);
        self.rotate(&mut k, kvh, offset, rope);

        let total = k_cache.shape()[1];
        let start = total - seq;
        k_cache
            .slice_mut(s![.., start.., ..])
            .assign(&k.into_shape_with_order((batch, seq, kvh * hd))?);
        v_cache
            .slice_mut(s![.., start.., ..])
            .assign(&v.into_shape_with_order((batch, seq, kvh * hd))?);

        let q_heads = q
            .into_shape_with_order((batch, seq, h, hd))?
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .to_owned();
        let scale = (hd as f32).powf(-0.5);
        let context: Array4<f32> = if seq == 1 {
            decode_attention(&q_heads, &k_cache.view(), &v_cache.view(), kvh, scale)
        } else {
            streaming_causal_attention(
                &q_heads,
                &k_cache.view(),
                &v_cache.view(),
                kvh,
                scale,
                start,
            )
        };
        let mut context = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((rows, h * hd))?
            .to_owned();
        context.zip_mut_with(&gate, |c, &g| *c *= sigmoid(g));

        let out = self.o_proj.matmul(&context.view());
        Ok(out.into_shape_with_order((batch, seq, width))?)
    }
}

/// Gated DeltaNet: a causal depthwise convolution over q, k and v, then a
/// delta-rule update of one `[key_dim, value_dim]` state per head, read out by
/// the query and normalised with a SiLU gate.
pub struct GatedDeltaNet {
    pub in_proj_qkv: LinearLayer,
    pub in_proj_z: LinearLayer,
    pub in_proj_b: LinearLayer,
    pub in_proj_a: LinearLayer,
    pub out_proj: LinearLayer,
    /// `[channels, kernel]`.
    pub conv_weight: Array2<f32>,
    /// `-exp(A_log)`, one per value head.
    pub neg_a: Array1<f32>,
    pub dt_bias: Array1<f32>,
    /// The gated norm scales by its weight as stored, without the offset.
    pub norm_weight: Array1<f32>,
    pub eps: f32,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// Value heads are in llama.cpp's tiled order: head `h` reads key head
    /// `h % num_k_heads`. Otherwise grouped, as Hugging Face stores them: key
    /// head `h / (num_v_heads / num_k_heads)`.
    pub tiled_value_heads: bool,
}

impl GatedDeltaNet {
    /// Runs `hidden` on from `state` and leaves the state at its end. With
    /// `capture`, also returns the state as it was after that many of the
    /// tokens, so a later prompt sharing them can start there.
    pub fn forward(
        &self,
        hidden: &Array3<f32>,
        state: &mut LinearState,
        capture: Option<usize>,
    ) -> Result<(Array3<f32>, Option<LinearState>)> {
        let (batch, seq, width) = hidden.dim();
        let rows = batch * seq;
        let x = hidden.view().into_shape_with_order((rows, width))?;
        let (nk, nv, dk, dv) = (
            self.num_k_heads,
            self.num_v_heads,
            self.key_dim,
            self.value_dim,
        );
        let key_width = nk * dk;

        let qkv = self.in_proj_qkv.matmul(&x);
        let z = self.in_proj_z.matmul(&x);
        let b = self.in_proj_b.matmul(&x);
        let a = self.in_proj_a.matmul(&x);

        let channels = qkv.ncols();
        let capture = capture.filter(|&c| c > 0 && c < seq);
        let (qkv, captured_conv) = self.convolve(
            &qkv.into_shape_with_order((batch, seq, channels))?,
            state,
            capture,
        );
        let mut captured_recurrent = capture.map(|_| Array4::<f32>::zeros(state.recurrent.dim()));

        let rep = nv / nk;
        let scale = (dk as f32).powf(-0.5);
        let mut core = Array4::<f32>::zeros((batch, seq, nv, dv));
        // Where heads write the captured state; a scratch array when not capturing.
        let mut scratch = Array4::<f32>::zeros((batch, nv, 0, 0));
        for bi in 0..batch {
            let mut rec = state.recurrent.slice_mut(s![bi, .., .., ..]);
            let mut out = core.slice_mut(s![bi, .., .., ..]);
            let mut kept = match captured_recurrent.as_mut() {
                Some(c) => c.slice_mut(s![bi, .., .., ..]),
                None => scratch.slice_mut(s![bi, .., .., ..]),
            };
            // Heads are independent, so each runs its whole sequence on one thread.
            rec.axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(out.axis_iter_mut(Axis(1)).into_par_iter())
                .zip(kept.axis_iter_mut(Axis(0)).into_par_iter())
                .enumerate()
                .for_each(|(head, ((mut st, mut out_h), mut kept_h))| {
                    let kh = if self.tiled_value_heads {
                        head % nk
                    } else {
                        head / rep
                    };
                    let st = st.as_slice_mut().expect("recurrent state is contiguous");
                    let mut q = vec![0f32; dk];
                    let mut k = vec![0f32; dk];
                    let mut delta = vec![0f32; dv];
                    for t in 0..seq {
                        let row = qkv.slice(s![bi, t, ..]);
                        for i in 0..dk {
                            q[i] = row[kh * dk + i];
                            k[i] = row[key_width + kh * dk + i];
                        }
                        l2_normalize(&mut q);
                        l2_normalize(&mut k);
                        q.iter_mut().for_each(|v| *v *= scale);
                        let v = row.slice(s![
                            2 * key_width + head * dv..2 * key_width + (head + 1) * dv
                        ]);

                        let r = bi * seq + t;
                        let beta = sigmoid(b[[r, head]]);
                        let g =
                            (self.neg_a[head] * softplus(a[[r, head]] + self.dt_bias[head])).exp();

                        // S <- g S;  delta = beta (v - Sᵀk);  S <- S + k deltaᵀ;  o = Sᵀq
                        st.iter_mut().for_each(|s| *s *= g);
                        delta.iter_mut().for_each(|d| *d = 0.0);
                        for i in 0..dk {
                            let ki = k[i];
                            let srow = &st[i * dv..(i + 1) * dv];
                            for j in 0..dv {
                                delta[j] += srow[j] * ki;
                            }
                        }
                        for j in 0..dv {
                            delta[j] = (v[j] - delta[j]) * beta;
                        }
                        let mut o = out_h.slice_mut(s![t, ..]);
                        o.fill(0.0);
                        for i in 0..dk {
                            let (ki, qi) = (k[i], q[i]);
                            let srow = &mut st[i * dv..(i + 1) * dv];
                            for j in 0..dv {
                                srow[j] += ki * delta[j];
                                o[j] += srow[j] * qi;
                            }
                        }
                        if capture == Some(t + 1) {
                            kept_h
                                .as_slice_mut()
                                .expect("captured state is contiguous")
                                .copy_from_slice(st);
                        }
                    }
                });
        }

        // RMSNorm per head on the output, then the SiLU gate from z.
        let mut core = core.into_shape_with_order((rows * nv, dv))?;
        let z = z.into_shape_with_order((rows * nv, dv))?;
        core.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(z.axis_iter(Axis(0)).into_par_iter())
            .for_each(|(mut row, zrow)| {
                let mean = row.iter().map(|v| v * v).sum::<f32>() / dv as f32;
                let inv = 1.0 / (mean + self.eps).sqrt();
                for j in 0..dv {
                    row[j] = row[j] * inv * self.norm_weight[j] * silu(zrow[j]);
                }
            });
        let core = core.into_shape_with_order((rows, nv * dv))?;
        let out = self.out_proj.matmul(&core.view());
        let captured = match (captured_conv, captured_recurrent) {
            (Some(conv), Some(recurrent)) => Some(LinearState { conv, recurrent }),
            _ => None,
        };
        Ok((out.into_shape_with_order((batch, seq, width))?, captured))
    }

    /// The causal depthwise convolution with SiLU, continuing from the inputs
    /// left in `state.conv` and leaving the newest ones there. With `capture`,
    /// also returns the window as it stood after that many inputs.
    fn convolve(
        &self,
        x: &Array3<f32>,
        state: &mut LinearState,
        capture: Option<usize>,
    ) -> (Array3<f32>, Option<Array3<f32>>) {
        let (batch, seq, channels) = x.dim();
        let kernel = self.conv_weight.ncols();
        let window = kernel - 1;
        let mut out = Array3::<f32>::zeros((batch, seq, channels));
        let mut kept =
            Array3::<f32>::zeros((batch, channels, if capture.is_some() { window } else { 0 }));
        for bi in 0..batch {
            let mut conv = state.conv.slice_mut(s![bi, .., ..]);
            let mut kept_b = kept.slice_mut(s![bi, .., ..]);
            out.slice_mut(s![bi, .., ..])
                .axis_iter_mut(Axis(1))
                .into_par_iter()
                .zip(conv.axis_iter_mut(Axis(0)).into_par_iter())
                .zip(kept_b.axis_iter_mut(Axis(0)).into_par_iter())
                .enumerate()
                .for_each(|(c, ((mut out_c, mut held), mut kept_c))| {
                    let w = self.conv_weight.row(c);
                    // The held inputs, then this call's, oldest first.
                    let mut line: Vec<f32> = held.iter().copied().collect();
                    line.extend((0..seq).map(|t| x[[bi, t, c]]));
                    for t in 0..seq {
                        let acc: f32 = (0..kernel).map(|j| w[j] * line[t + j]).sum();
                        out_c[t] = silu(acc);
                    }
                    for j in 0..window {
                        held[j] = line[seq + j];
                    }
                    if let Some(n) = capture {
                        for j in 0..window {
                            kept_c[j] = line[n + j];
                        }
                    }
                });
        }
        (out, capture.map(|_| kept))
    }
}

/// `x / sqrt(sum(x²) + 1e-6)`, as the FLA kernels and transformers compute it.
fn l2_normalize(x: &mut [f32]) {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
    x.iter_mut().for_each(|v| *v *= inv);
}
