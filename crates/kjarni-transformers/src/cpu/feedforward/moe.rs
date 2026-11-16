//! Mixture-of-experts feed-forward, as Qwen 3.5's MoE models use it.
//!
//! A router scores every expert for each token; the token goes through its
//! `top_k` best, and their outputs are summed, weighted by the router's softmax
//! renormalized over those `top_k`. A shared expert, scaled by a sigmoid gate
//! of its own, runs on every token and is added on top.

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Axis};
use rayon::prelude::*;

use crate::activations::apply_activation_2d;
use crate::cpu::feedforward::SwiGluFeedForward;
use crate::linear_layer::{LinearLayer, RowInput};

/// Output rows each thread takes at a time on the one-token path.
const ROWS_PER_JOB: usize = 64;

/// An expert every token goes through, scaled by `sigmoid(gate · x)`.
pub struct SharedExpert {
    pub mlp: SwiGluFeedForward,
    /// One weight per hidden unit; `None` adds the expert unscaled.
    pub gate: Option<Array1<f32>>,
}

pub struct MoeFeedForward {
    /// `[num_experts, hidden]`.
    pub router: LinearLayer,
    pub experts: Vec<SwiGluFeedForward>,
    /// How many experts each token goes through.
    pub top_k: usize,
    /// Renormalize the chosen experts' weights to sum to one, as Qwen does.
    pub normalize_top_k: bool,
    pub shared: Option<SharedExpert>,
}

impl MoeFeedForward {
    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, dim) = hidden.dim();
        let x = hidden.view().into_shape_with_order((batch * seq, dim))?;
        let tokens = x.nrows();
        if tokens == 1 {
            let row = x.row(0).to_vec();
            let out = self.forward_one(&row);
            return Ok(Array3::from_shape_vec((batch, seq, dim), out)?);
        }

        let scores = self.router.matmul(&x);
        let routes: Vec<Vec<(usize, f32)>> = scores
            .outer_iter()
            .map(|row| self.route(row.as_slice().expect("router output is contiguous")))
            .collect();

        // Each expert takes its tokens as one batch.
        let mut chosen: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.experts.len()];
        for (token, route) in routes.iter().enumerate() {
            for &(expert, weight) in route {
                chosen[expert].push((token, weight));
            }
        }
        let outputs: Vec<(usize, Array2<f32>)> = chosen
            .par_iter()
            .enumerate()
            .filter(|(_, picks)| !picks.is_empty())
            .map(|(expert, picks)| {
                let rows: Vec<usize> = picks.iter().map(|&(t, _)| t).collect();
                let input = x.select(Axis(0), &rows);
                self.experts[expert]
                    .forward_2d(&input)
                    .map(|out| (expert, out))
            })
            .collect::<Result<_>>()?;

        let mut out = match &self.shared {
            Some(shared) => {
                let mut y = shared.mlp.forward_2d(&x.to_owned())?;
                if let Some(gate) = &shared.gate {
                    for (mut row, input) in y.outer_iter_mut().zip(x.outer_iter()) {
                        let g = 1.0 / (1.0 + (-input.dot(gate)).exp());
                        row *= g;
                    }
                }
                y
            }
            None => Array2::zeros((tokens, dim)),
        };
        for (expert, y) in outputs {
            for (&(token, weight), row) in chosen[expert].iter().zip(y.outer_iter()) {
                out.row_mut(token).scaled_add(weight, &row);
            }
        }
        Ok(out.into_shape_with_order((batch, seq, dim))?)
    }

    /// One token. Its experts are small, eight 512-row matrices a layer in
    /// Qwen3.5-35B-A3B, so rather than spread each over the threads in turn,
    /// every expert's gate and up rows go out in one dispatch and every down
    /// row in a second.
    fn forward_one(&self, x: &[f32]) -> Vec<f32> {
        let scores = self
            .router
            .matmul(&ndarray::ArrayView2::from_shape((1, x.len()), x).expect("one row"));
        let mut members: Vec<(&SwiGluFeedForward, f32)> = self
            .route(scores.as_slice().expect("router output is contiguous"))
            .into_iter()
            .map(|(e, w)| (&self.experts[e], w))
            .collect();
        if let Some(shared) = &self.shared {
            let scale = shared.gate.as_ref().map_or(1.0, |g| {
                let s: f32 = g.iter().zip(x).map(|(g, x)| g * x).sum();
                1.0 / (1.0 + (-s).exp())
            });
            members.push((&shared.mlp, scale));
        }

        let input = RowInput::new(x);
        let mut gate: Vec<Vec<f32>> = members
            .iter()
            .map(|(m, _)| vec![0.0; m.gate.out_features()])
            .collect();
        let mut up: Vec<Vec<f32>> = members
            .iter()
            .map(|(m, _)| vec![0.0; m.up.out_features()])
            .collect();
        let mut jobs: Vec<(&LinearLayer, usize, &mut [f32])> = Vec::new();
        for ((m, _), (g, u)) in members.iter().zip(gate.iter_mut().zip(up.iter_mut())) {
            for (layer, out) in [(&m.gate, g), (&m.up, u)] {
                for (i, chunk) in out.chunks_mut(ROWS_PER_JOB).enumerate() {
                    jobs.push((layer, i * ROWS_PER_JOB, chunk));
                }
            }
        }
        jobs.into_par_iter()
            .for_each(|(layer, first, out)| layer.rows_into(&input, first, out));

        // silu(gate) * up, per member.
        let hidden: Vec<Vec<f32>> = members
            .iter()
            .zip(gate.into_iter().zip(up))
            .map(|((m, _), (g, u))| {
                let mut g = Array2::from_shape_vec((1, g.len()), g).expect("one row");
                apply_activation_2d(&mut g, m.activation);
                let mut h = g.into_raw_vec_and_offset().0;
                h.iter_mut().zip(&u).for_each(|(h, u)| *h *= u);
                h
            })
            .collect();

        let inputs: Vec<RowInput> = hidden.iter().map(|h| RowInput::new(h)).collect();
        let mut down: Vec<Vec<f32>> = members
            .iter()
            .map(|(m, _)| vec![0.0; m.down.out_features()])
            .collect();
        let mut jobs: Vec<(&LinearLayer, &RowInput, usize, &mut [f32])> = Vec::new();
        for (((m, _), input), out) in members.iter().zip(&inputs).zip(down.iter_mut()) {
            for (i, chunk) in out.chunks_mut(ROWS_PER_JOB).enumerate() {
                jobs.push((&m.down, input, i * ROWS_PER_JOB, chunk));
            }
        }
        jobs.into_par_iter()
            .for_each(|(layer, input, first, out)| layer.rows_into(input, first, out));

        let mut out = vec![0.0; x.len()];
        for ((_, weight), y) in members.iter().zip(&down) {
            out.iter_mut().zip(y).for_each(|(o, y)| *o += weight * y);
        }
        out
    }

    /// The `top_k` experts for one token's router scores, best first, with
    /// their weights.
    pub fn route(&self, scores: &[f32]) -> Vec<(usize, f32)> {
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
        let total: f32 = exp.iter().sum();
        let mut order: Vec<usize> = (0..scores.len()).collect();
        // Highest first; ties go to the lower index.
        order.sort_by(|&a, &b| exp[b].total_cmp(&exp[a]).then(a.cmp(&b)));
        order.truncate(self.top_k);
        let picked: f32 = order.iter().map(|&e| exp[e]).sum();
        let scale = if self.normalize_top_k { picked } else { total };
        order.into_iter().map(|e| (e, exp[e] / scale)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;

    fn expert(scale: f32) -> SwiGluFeedForward {
        let eye =
            |n: usize| Array2::from_shape_fn((n, n), |(i, j)| if i == j { scale } else { 0.0 });
        SwiGluFeedForward::new(
            LinearLayer::new_f32(eye(2), None),
            LinearLayer::new_f32(eye(2), None),
            LinearLayer::new_f32(eye(2), None),
            Activation::SilU,
        )
    }

    fn moe(router: [[f32; 2]; 3], top_k: usize, shared: Option<SharedExpert>) -> MoeFeedForward {
        MoeFeedForward {
            router: LinearLayer::new_f32(
                Array2::from_shape_fn((3, 2), |(i, j)| router[i][j]),
                None,
            ),
            experts: vec![expert(1.0), expert(2.0), expert(3.0)],
            top_k,
            normalize_top_k: true,
            shared,
        }
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn routing_takes_the_best_experts_and_renormalizes_their_weights() {
        let m = moe([[0.0; 2]; 3], 2, None);
        // softmax of [1, 3, 2] is [0.090, 0.665, 0.245]; the top two renormalized
        // are 0.665 / 0.910 and 0.245 / 0.910.
        let route = m.route(&[1.0, 3.0, 2.0]);
        assert_eq!(route.iter().map(|r| r.0).collect::<Vec<_>>(), [1, 2]);
        assert!((route[0].1 - 0.731_058_6).abs() < 1e-6, "{route:?}");
        assert!((route[1].1 - 0.268_941_4).abs() < 1e-6, "{route:?}");
        // Without renormalizing, the softmax over all experts.
        let all = MoeFeedForward {
            normalize_top_k: false,
            ..m
        };
        assert!((all.route(&[1.0, 3.0, 2.0])[0].1 - 0.665_240_9).abs() < 1e-6);
        // Ties go to the lower index.
        let tie = moe([[0.0; 2]; 3], 1, None);
        assert_eq!(tie.route(&[0.5, 0.5, 0.5])[0].0, 0);
    }

    /// The one-token path and the batched one agree.
    #[test]
    fn one_token_matches_the_same_token_in_a_batch() {
        let shared = SharedExpert {
            mlp: expert(0.5),
            gate: Some(Array1::from(vec![0.3, -0.2])),
        };
        let m = moe([[1.0, -2.0], [0.5, 0.5], [-1.0, 2.0]], 2, Some(shared));
        let x = Array3::from_shape_vec((1, 2, 2), vec![0.7, -0.4, 0.1, 0.9]).unwrap();
        let both = m.forward(&x).unwrap();
        for t in 0..2 {
            let one = m
                .forward(&x.slice(ndarray::s![.., t..t + 1, ..]).to_owned())
                .unwrap();
            for d in 0..2 {
                let (a, b) = (one[[0, 0, d]], both[[0, t, d]]);
                assert!(
                    (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                    "{one:?} vs {both:?}"
                );
            }
        }
    }

    #[test]
    fn each_token_gets_its_own_experts_plus_the_gated_shared_one() {
        // Every matrix of expert e is s = e + 1 times the identity, so on each
        // unit it computes s * silu(s * x) * s * x. Token 0 points along unit 0,
        // which the router sends to expert 0; token 1 along unit 1, to expert 2.
        let router = [[10.0, 0.0], [0.0, 0.0], [0.0, 10.0]];
        let shared = SharedExpert {
            mlp: expert(1.0),
            gate: Some(Array1::from(vec![0.0, 0.0])),
        };
        let m = moe(router, 1, Some(shared));
        let x = Array3::from_shape_vec((1, 2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let y = m.forward(&x).unwrap();
        let through = |s: f32, v: f32| s * silu(s * v) * s * v;
        // Token 0: expert 0 with weight 1, plus the shared expert at sigmoid(0) = 0.5.
        let t0 = through(1.0, 1.0) + 0.5 * through(1.0, 1.0);
        // Token 1: expert 2.
        let t1 = through(3.0, 1.0) + 0.5 * through(1.0, 1.0);
        assert!((y[[0, 0, 0]] - t0).abs() < 1e-5, "{y:?}");
        assert!(y[[0, 0, 1]].abs() < 1e-6);
        assert!((y[[0, 1, 1]] - t1).abs() < 1e-5, "{y:?}");
        assert!(y[[0, 1, 0]].abs() < 1e-6);
    }
}
