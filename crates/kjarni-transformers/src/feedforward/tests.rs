//! SwiGLU Feed-Forward Network
//!
//! Used in LLaMA and other modern architectures.
//! SwiGLU: SiLU(gate(x)) ⊙ up(x), then down()
//!
//! Unlike standard FFN with 2 projections, SwiGLU has 3:
//! - gate_proj: [hidden_size, intermediate_size]
//! - up_proj:   [hidden_size, intermediate_size]  
//! - down_proj: [intermediate_size, hidden_size]

use anyhow::Result;
use ndarray::{Array2, Array3};

use crate::feedforward::swiglu::SwiGluFeedForward;


#[test]
fn test_swiglu_ffn_shapes() -> Result<()> {
    let hidden_size = 64;
    let intermediate_size = 256;
    let batch_size = 2;
    let seq_len = 10;

    // LinearLayer convention: [out_features, in_features]
    // gate/up: hidden → intermediate, so [intermediate, hidden]
    // down: intermediate → hidden, so [hidden, intermediate]
    let gate_weight: Array2<f32> = Array2::zeros((intermediate_size, hidden_size));  // [256, 64]
    let up_weight: Array2<f32> = Array2::zeros((intermediate_size, hidden_size));    // [256, 64]
    let down_weight: Array2<f32> = Array2::zeros((hidden_size, intermediate_size));  // [64, 256]

    let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

    let input = Array3::zeros((batch_size, seq_len, hidden_size));
    let output = ffn.forward(&input)?;

    assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
    Ok(())
}

#[test]
fn test_swiglu_ffn_basic() -> Result<()> {
    // Simple test with small dimensions
    let hidden_size = 4;
    let intermediate_size = 8;

    // Identity-like weights for testing
    let mut gate_weight = Array2::zeros((intermediate_size, hidden_size)); // (8, 4)
    let mut up_weight = Array2::zeros((intermediate_size, hidden_size)); // (8, 4)  
    let mut down_weight = Array2::zeros((hidden_size, intermediate_size)); // (4, 8)

    // Set some non-zero values
    for i in 0..hidden_size.min(intermediate_size) {
        gate_weight[[i, i]] = 1.0;
        up_weight[[i, i]] = 1.0;
        down_weight[[i, i]] = 1.0;
    }

    let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

    let input = Array3::ones((1, 1, hidden_size));
    let output = ffn.forward(&input)?;

    // Output should be finite and reasonable
    assert!(output.iter().all(|&x| x.is_finite()));
    assert_eq!(output.shape(), &[1, 1, hidden_size]);
    Ok(())
}


#[test]
fn test_swiglu_vs_standard_ffn() -> Result<()> {
    let hidden_size = 2;
    let intermediate_size = 4;

    // [out_features, in_features]
    // gate: [intermediate, hidden] = [4, 2]
    let gate_weight = Array2::from_shape_vec(
        (4, 2),
        vec![
            1.0, 0.0,  // output 0 weights
            0.0, 1.0,  // output 1 weights
            0.0, 0.0,  // output 2 weights
            0.0, 0.0,  // output 3 weights
        ],
    ).unwrap();

    // up: [intermediate, hidden] = [4, 2]
    let up_weight = Array2::from_shape_vec(
        (4, 2),
        vec![
            0.0, 0.0,  // output 0 weights
            0.0, 0.0,  // output 1 weights
            1.0, 0.0,  // output 2 weights
            0.0, 1.0,  // output 3 weights
        ],
    ).unwrap();

    // down: [hidden, intermediate] = [2, 4]
    let down_weight = Array2::from_shape_vec(
        (2, 4),
        vec![
            1.0, 0.0, 1.0, 0.0,  // output 0 weights
            0.0, 1.0, 0.0, 1.0,  // output 1 weights
        ],
    ).unwrap();

    let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

    let input = Array3::from_shape_vec((1, 1, 2), vec![1.0, 2.0]).unwrap();
    let output = ffn.forward(&input)?;

    // With input [1, 2]:
    // gate_out = [1*1 + 2*0, 1*0 + 2*1, 0, 0] = [1, 2, 0, 0]
    // up_out   = [0, 0, 1*1 + 2*0, 1*0 + 2*1] = [0, 0, 1, 2]
    // silu(gate) * up = [silu(1)*0, silu(2)*0, silu(0)*1, silu(0)*2]
    //                 = [0, 0, 0, 0]
    // down_out = [0, 0]

    assert_eq!(output.shape(), &[1, 1, 2]);
    assert!(output.iter().all(|&x| x.abs() < 1e-6), "Expected zeros, got {:?}", output);
    Ok(())
}
#[test]
fn test_swiglu_pytorch_parity() -> Result<()> {
    // Test against known PyTorch output
    // PyTorch code:
    // ```python
    // import torch
    // import torch.nn.functional as F
    //
    // x = torch.tensor([[[1.0, 2.0]]])  # [1, 1, 2]
    // gate_weight = torch.tensor([[1.0, 0.5], [0.5, 1.0]])  # [2, 2]
    // up_weight = torch.tensor([[0.5, 1.0], [1.0, 0.5]])    # [2, 2]
    // down_weight = torch.tensor([[1.0, 0.5], [0.5, 1.0]])  # [2, 2]
    //
    // gate_out = x @ gate_weight  # [1, 1, 2]
    // up_out = x @ up_weight      # [1, 1, 2]
    // activated = F.silu(gate_out) * up_out
    // output = activated @ down_weight
    // print(output)
    // ```
    // Output: tensor([[[6.7143, 6.8227]]])

    let gate_weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.5, 1.0]).unwrap();

    let up_weight = Array2::from_shape_vec((2, 2), vec![0.5, 1.0, 1.0, 0.5]).unwrap();

    let down_weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.5, 1.0]).unwrap();

    let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

    let input = Array3::from_shape_vec((1, 1, 2), vec![1.0, 2.0]).unwrap();
    let output = ffn.forward(&input)?;

    assert!(
        (output[[0, 0, 0]] - 6.7142).abs() < 1e-3,
        "Expected ~6.7142, got {}",
        output[[0, 0, 0]]
    );
    assert!(
        (output[[0, 0, 1]] - 6.8224).abs() < 1e-3,
        "Expected ~6.8224, got {}",
        output[[0, 0, 1]]
    );
    Ok(())
}

#[test]
fn test_swiglu_nonlinearity() -> std::result::Result<(), anyhow::Error> {
    // Test that SwiGLU is actually non-linear
    let hidden_size = 4;
    let intermediate_size = 4;

    let gate_weight: Array2<f32> = Array2::eye(hidden_size);
    let up_weight: Array2<f32> = Array2::eye(hidden_size);
    let down_weight: Array2<f32> = Array2::eye(hidden_size);

    let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

    // Input 1: all ones
    let input1 = Array3::ones((1, 1, hidden_size));
    let output1 = ffn.forward(&input1)?;

    // Input 2: all twos (2x input1)
    let input2 = Array3::from_elem((1, 1, hidden_size), 2.0);
    let output2 = ffn.forward(&input2)?;

    // Due to SiLU non-linearity, output2 should NOT be 2*output1
    let ratio = output2[[0, 0, 0]] / output1[[0, 0, 0]];
    assert!((ratio - 2.0).abs() > 0.1, "SwiGLU should be non-linear");

    Ok(())
}
