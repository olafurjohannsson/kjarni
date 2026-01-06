use anyhow::{anyhow, Result};
use ndarray::{s, Array2, ArrayView1};
use std::path::Path;

use kjarni_transformers::cpu::kernels::dequantize::{dequantize_q4_k_block, dequantize_q6_k_block, dequantize_q8_0_block};
use kjarni_transformers::cpu::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0, QK_K};
use kjarni_transformers::tensor::{DType, TensorView};
use kjarni_transformers::weights::gguf_loader::{GgufHfMapper, GgufLoader};
use kjarni_transformers::weights::safetensors_loader::SafeTensorsLoader;
// --- ASSUMPTIONS: Adjust these `use` statements to match your library's structure ---
use kjarni_transformers::weights::WeightLoader;

// --- CONFIGURATION ---
// Set the model you want to inspect here
const MODEL_NAME: &str = "Llama-3.2-3B-Instruct"; // or "Llama-3.2-1B-Instruct"

// Set the correct head counts for the chosen model
const HEAD_COUNT: usize = 48;      // 32 for 1B, 48 for 3B
const KV_HEAD_COUNT: usize = 8;    // 8 for both 1B and 3B

/// A definitive diagnostic tool to analyze GGUF tensor layouts by checking all rows.
fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           GGUF Tensor Layout Inspector (Full Analysis)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\nModel: {}", MODEL_NAME);

    // --- LOAD BOTH MODELS ---
    println!("\n[1] Loading models...");
    let gguf_loader = GgufLoader::new(&Path::new(&format!(
        "/home/olafurj/.cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    )))?;
    println!("  ✓ GGUF loaded");

    let st_loader = SafeTensorsLoader::new(&Path::new(&format!(
        "/home/olafurj/.cache/kjarni/meta-llama_{}/",
        MODEL_NAME
    )))?;
    println!("  ✓ SafeTensors loaded");

    // --- ANALYSIS LOOP ---
    println!("\n[2] Analyzing all GGUF tensors (this will be slow)...");
    let mut report = Vec::new();
    let mut tensor_names: Vec<_> = gguf_loader.tensor_names().into_iter().map(String::from).collect();
    tensor_names.sort();

    for gguf_name in tensor_names {
        let raw_gguf_view = gguf_loader.get_raw_from_gguf_name(&gguf_name)?;
        if raw_gguf_view.shape.len() != 2 || !is_quantized(raw_gguf_view.dtype) {
            continue;
        }

        let hf_name = match gguf_loader.gguf_to_hf_name(&gguf_name) {
            Some(name) => name,
            None => continue,
        };

        print!(" - Analyzing '{: <40}'... ", gguf_name);

        let ground_truth = match to_f32_array2(&st_loader.get_raw(&hf_name)?) {
            Ok(t) => t,
            Err(_) => {
                println!("SKIPPED (unsupported dtype).");
                continue;
            }
        };

        let gguf_raw_f32 = match dequantize_tensor(&raw_gguf_view) {
            Ok(t) => t,
            Err(_) => {
                println!("SKIPPED (dequant failed).");
                continue;
            }
        };

        let is_qk = is_qk_tensor(&gguf_name);
        let n_head = if is_qk {
            if gguf_name.contains("attn_q") { HEAD_COUNT } else { KV_HEAD_COUNT }
        } else { 0 };

        let (layout, confidence) = analyze_tensor_layout_full(&ground_truth, &gguf_raw_f32, is_qk, n_head);

        let status = match layout {
            LayoutType::DirectMatch => "✓ direct match",
            LayoutType::Interleaved => "✓ interleaved",
            LayoutType::Permuted => "✓ permuted (Q/K)",
            LayoutType::Unknown => "✗ MISMATCH",
        };
        println!("{}", status);

        report.push((gguf_name, raw_gguf_view.dtype, layout, confidence));
    }

    print_report(&report);
    Ok(())
}

// =============================================================================
// CORE ANALYSIS LOGIC
// =============================================================================

#[derive(Debug, PartialEq, Clone, Copy)]
enum LayoutType {
    DirectMatch,
    Interleaved,
    Permuted,
    Unknown,
}

/// A robust function to detect the physical layout of a GGUF tensor.
/// It checks every row for direct match, super-block interleaving, and Q/K permutation.
fn analyze_tensor_layout_full(
    truth: &Array2<f32>,
    gguf_raw: &Array2<f32>,
    is_qk: bool,
    n_head: usize,
) -> (LayoutType, f32) {
    if truth.shape() != gguf_raw.shape() { return (LayoutType::Unknown, 0.0); }
    let (rows, _) = truth.dim();
    if rows == 0 { return (LayoutType::DirectMatch, 1.0); }

    let head_dim = if is_qk && n_head > 0 && rows % n_head == 0 { rows / n_head } else { 0 };
    let half_head = if head_dim > 0 { head_dim / 2 } else { 0 };

    let mut direct_matches = 0;
    let mut interleaved_matches = 0;
    let mut permuted_matches = 0;

    for gguf_row_idx in 0..rows {
        let gguf_row = gguf_raw.row(gguf_row_idx);

        // Test 1: Direct Match
        if rows_are_similar(&truth.row(gguf_row_idx), &gguf_row) {
            direct_matches += 1;
            continue;
        }

        // Test 2: Super-block Interleaving
        if rows >= 64 {
            let super_tile = gguf_row_idx / 64;
            let within_tile_phys = gguf_row_idx % 64;
            let logical_row_interleaved = if within_tile_phys % 2 == 0 {
                super_tile * 64 + within_tile_phys / 2
            } else {
                super_tile * 64 + 32 + (within_tile_phys / 2)
            };
            if logical_row_interleaved < rows && rows_are_similar(&truth.row(logical_row_interleaved), &gguf_row) {
                interleaved_matches += 1;
                continue;
            }
        }

        // Test 3: Q/K Permutation
        if is_qk && head_dim > 0 {
            let head_idx = gguf_row_idx / head_dim;
            let within_head_phys = gguf_row_idx % head_dim;
            let logical_row_permuted = if within_head_phys < half_head {
                head_idx * head_dim + (within_head_phys * 2)
            } else {
                head_idx * head_dim + ((within_head_phys - half_head) * 2 + 1)
            };
            if logical_row_permuted < rows && rows_are_similar(&truth.row(logical_row_permuted), &gguf_row) {
                permuted_matches += 1;
                continue;
            }
        }
    }

    let direct_conf = direct_matches as f32 / rows as f32;
    let interleaved_conf = interleaved_matches as f32 / rows as f32;
    let permuted_conf = permuted_matches as f32 / rows as f32;

    if direct_conf > 0.99 { (LayoutType::DirectMatch, direct_conf) } else if interleaved_conf > 0.99 { (LayoutType::Interleaved, interleaved_conf) } else if permuted_conf > 0.99 { (LayoutType::Permuted, permuted_conf) } else { (LayoutType::Unknown, 0.0) }
}

/// Computes similarity between two vectors, accounting for quantization noise.
fn rows_are_similar(row1: &ArrayView1<f32>, row2: &ArrayView1<f32>) -> bool {
    if row1.len() != row2.len() { return false; }
    let match_count = row1.iter().zip(row2.iter())
        .filter(|&(&t, &g)| (t - g).abs() < 0.02) // Tolerance for Q4
        .count();
    match_count > row1.len() * 95 / 100 // Require a 95% match
}

// =============================================================================
// DATA LOADING & CONVERSION HELPERS
// =============================================================================

fn is_quantized(dtype: DType) -> bool { matches!(dtype, DType::Q4_K | DType::Q5_K | DType::Q6_K | DType::Q8_0) }
fn is_qk_tensor(name: &str) -> bool { name.contains("attn_q") || name.contains("attn_k") }

fn dequantize_tensor(view: &TensorView) -> Result<Array2<f32>> {
    match view.dtype {
        DType::Q4_K => dequantize_raw::<BlockQ4_K>(view, QK_K, dequantize_q4_k_block),
        DType::Q6_K => dequantize_raw::<BlockQ6_K>(view, QK_K, dequantize_q6_k_block),
        DType::Q8_0 => dequantize_raw::<BlockQ8_0>(view, 32, dequantize_q8_0_block),
        _ => Err(anyhow!("Unsupported quantization type for inspector: {:?}", view.dtype)),
    }
}

fn dequantize_raw<B: bytemuck::Pod>(
    view: &TensorView,
    block_size: usize,
    dequant_fn: impl Fn(&B, &mut [f32]),
) -> Result<Array2<f32>> {
    let shape = &view.shape;
    let (rows, cols) = (shape[0], shape[1]);
    let blocks: &[B] = bytemuck::try_cast_slice(&view.bytes).map_err(|opt| anyhow!("Failed to cast bytes for '{}': {}", view.name, opt))?;
    let blocks_per_row = cols / block_size;
    if blocks.len() != rows * blocks_per_row { return Err(anyhow!("Block count mismatch for '{}'", view.name)); }
    let mut out = Array2::<f32>::zeros((rows, cols));
    let mut buf = vec![0.0f32; block_size];
    for (block_idx, block) in blocks.iter().enumerate() {
        dequant_fn(block, &mut buf);
        let row = block_idx / blocks_per_row;
        let col_start = (block_idx % blocks_per_row) * block_size;
        out.slice_mut(s![row, col_start..col_start + block_size]).assign(&ArrayView1::from(&buf));
    }
    Ok(out)
}

fn to_f32_array2(view: &TensorView) -> Result<Array2<f32>> {
    let shape = &view.shape;
    if shape.len() != 2 { return Err(anyhow!("Expected 2D tensor, got {:?}D", shape.len())); }
    let (rows, cols) = (shape[0], shape[1]);
    match view.dtype {
        DType::BF16 => {
            let data: &[half::bf16] = bytemuck::cast_slice(&view.bytes);
            let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
            Ok(Array2::from_shape_vec((rows, cols), f32_data)?)
        }
        DType::F16 => {
            let data: &[half::f16] = bytemuck::cast_slice(&view.bytes);
            let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
            Ok(Array2::from_shape_vec((rows, cols), f32_data)?)
        }
        DType::F32 => {
            let data: &[f32] = bytemuck::cast_slice(&view.bytes);
            Ok(Array2::from_shape_vec((rows, cols), data.to_vec())?)
        }
        _ => Err(anyhow!("Unsupported source dtype for to_f32_array2: {:?}", view.dtype)),
    }
}

// =============================================================================
// REPORTING
// =============================================================================

fn print_report(results: &[(String, DType, LayoutType, f32)]) {
    println!("\n--- [3] Analysis Report for {} ---\n", MODEL_NAME);
    println!("{:<40} | {:<10} | {:<15} | {:<10}", "Tensor Name", "DType", "Detected Layout", "Confidence");
    println!("{:-<40}-+-{:-<12}-+-{:-<17}-+-{:-<12}", "", "", "", "");
    for (name, dtype, layout, confidence) in results {
        println!("{:<40} | {:<10} | {:<15} | {:>8.1}%", truncate(name, 40), format!("{:?}", dtype), format!("{:?}", layout), confidence * 100.0);
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len { s.to_string() } else { format!("{}...", &s[..max_len - 3]) }
}