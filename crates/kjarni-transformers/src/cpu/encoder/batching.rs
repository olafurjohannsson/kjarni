//! Length-bucketed batching for encoder heads that score whole sequences.
//!
//! A batch is padded to its longest member, and a padded position costs as much
//! as a real one, so sixty-four texts go faster as a few groups of similar length
//! than as one block padded to the longest. Smaller groups also keep each
//! layer's attention scores, which grow with rows x length², small enough to
//! stay in cache: a single 64 x 140 block of MiniLM pairs ran at half the speed
//! of the same work in groups.
//!
//! [`SentenceEncoderModel::encode_batch`](super::SentenceEncoderModel::encode_batch)
//! already did this for embeddings. Cross-encoders and sequence classifiers each
//! padded the whole batch, and were two to three times slower than PyTorch on a
//! batch of 64 as a result. This is the same grouping, for any head that turns a
//! padded batch into one row of outputs per input.

use anyhow::{Result, anyhow};
use ndarray::Array2;
use tokenizers::Encoding;

/// Positions, padding included, per forward pass: enough rows for the matmuls to
/// use every core, few enough that the attention scores stay cache-sized.
pub const TOKENS_PER_BATCH: usize = 8192;

/// Cap for short inputs, where the token budget alone would allow thousands.
pub const MAX_ROWS_PER_BATCH: usize = 256;

/// Inputs for one forward pass, padded to the longest member of the group.
#[derive(Debug)]
pub struct PaddedBatch {
    pub input_ids: Array2<u32>,
    pub attention_mask: Array2<f32>,
    pub token_type_ids: Array2<u32>,
}

/// Number of positions an encoding needs. Tokenizers configured to pad leave
/// every encoding at the batch's longest length, so the real length comes from
/// the attention mask. Right padding is assumed, as Hugging Face tokenizers do by
/// default; a left-padded encoding keeps its full length, which is correct, only
/// less tightly packed.
fn needed_len(e: &Encoding) -> usize {
    let mask = e.get_attention_mask();
    let real = mask.iter().filter(|&&m| m != 0).count();
    if mask[..real].iter().all(|&m| m != 0) {
        real
    } else {
        mask.len()
    }
}

/// Pads `group` to its longest member.
pub fn pad(group: &[&Encoding]) -> PaddedBatch {
    let width = group.iter().map(|e| needed_len(e)).max().unwrap_or(0);
    let rows = group.len();
    let mut input_ids = Array2::<u32>::zeros((rows, width));
    let mut attention_mask = Array2::<f32>::zeros((rows, width));
    let mut token_type_ids = Array2::<u32>::zeros((rows, width));
    for (i, e) in group.iter().enumerate() {
        let n = e.len().min(width);
        for j in 0..n {
            input_ids[[i, j]] = e.get_ids()[j];
            attention_mask[[i, j]] = e.get_attention_mask()[j] as f32;
            token_type_ids[[i, j]] = e.get_type_ids()[j];
        }
    }
    PaddedBatch {
        input_ids,
        attention_mask,
        token_type_ids,
    }
}

/// Groups `encodings` by length into batches of at most `token_budget` padded
/// positions and `max_rows` rows, runs `forward` on each, and returns its output
/// rows in the order the encodings were given.
///
/// `forward` must return one row per input row of the batch it is given.
pub fn map_bucketed<F>(
    encodings: &[Encoding],
    token_budget: usize,
    max_rows: usize,
    mut forward: F,
) -> Result<Vec<Vec<f32>>>
where
    F: FnMut(&PaddedBatch) -> Result<Array2<f32>>,
{
    let lengths: Vec<usize> = encodings.iter().map(needed_len).collect();
    let mut order: Vec<usize> = (0..encodings.len()).collect();
    order.sort_by_key(|&i| lengths[i]);

    let mut out: Vec<Vec<f32>> = vec![Vec::new(); encodings.len()];
    let mut start = 0;
    while start < order.len() {
        // Sorted ascending, so the last row added sets the group's width.
        let mut end = start + 1;
        while end < order.len()
            && end - start < max_rows.max(1)
            && (end + 1 - start) * lengths[order[end]] <= token_budget
        {
            end += 1;
        }
        let group: Vec<&Encoding> = order[start..end].iter().map(|&i| &encodings[i]).collect();
        let rows = forward(&pad(&group))?;
        if rows.nrows() != group.len() {
            return Err(anyhow!(
                "forward returned {} rows for a batch of {}",
                rows.nrows(),
                group.len()
            ));
        }
        for (&i, row) in order[start..end].iter().zip(rows.outer_iter()) {
            out[i] = row.to_vec();
        }
        start = end;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Encoding;

    /// An encoding of `real` tokens right-padded to `width`, ids counting from `first`.
    fn enc(first: u32, real: usize, width: usize) -> Encoding {
        let ids: Vec<u32> = (0..width as u32)
            .map(|j| if (j as usize) < real { first + j } else { 0 })
            .collect();
        let mask: Vec<u32> = (0..width).map(|j| u32::from(j < real)).collect();
        Encoding::new(
            ids,
            vec![0; width],
            vec![String::new(); width],
            vec![None; width],
            vec![(0, 0); width],
            vec![0; width],
            mask,
            vec![],
            Default::default(),
        )
    }

    /// Row output: the first id of each row, so the test can see where it went.
    fn first_ids(b: &PaddedBatch) -> Result<Array2<f32>> {
        Ok(b.input_ids
            .column(0)
            .mapv(|x| x as f32)
            .insert_axis(ndarray::Axis(1)))
    }

    #[test]
    fn rows_come_back_in_input_order() {
        let encs = vec![
            enc(100, 9, 10),
            enc(200, 2, 10),
            enc(300, 5, 10),
            enc(400, 10, 10),
        ];
        let out = map_bucketed(&encs, 12, 256, first_ids).unwrap();
        let firsts: Vec<f32> = out.iter().map(|r| r[0]).collect();
        assert_eq!(firsts, vec![100.0, 200.0, 300.0, 400.0]);
    }

    #[test]
    fn groups_are_padded_to_their_own_longest_and_respect_the_budget() {
        let encs = vec![enc(1, 3, 20), enc(2, 20, 20), enc(3, 3, 20), enc(4, 4, 20)];
        let mut seen = Vec::new();
        map_bucketed(&encs, 16, 256, |b| {
            let (rows, width) = b.input_ids.dim();
            assert!(
                rows * width <= 16 || rows == 1,
                "{rows} x {width} over budget"
            );
            seen.push((rows, width));
            first_ids(b)
        })
        .unwrap();
        // The three short ones share a pass at width 4; the long one goes alone.
        assert_eq!(seen, vec![(3, 4), (1, 20)]);
    }

    #[test]
    fn max_rows_caps_a_group() {
        let encs: Vec<Encoding> = (0..10).map(|i| enc(i, 2, 2)).collect();
        let mut sizes = Vec::new();
        map_bucketed(&encs, 1 << 20, 4, |b| {
            sizes.push(b.input_ids.nrows());
            first_ids(b)
        })
        .unwrap();
        assert_eq!(sizes, vec![4, 4, 2]);
    }

    #[test]
    fn an_input_longer_than_the_budget_still_runs() {
        let encs = vec![enc(7, 50, 50)];
        let out = map_bucketed(&encs, 8, 256, first_ids).unwrap();
        assert_eq!(out, vec![vec![7.0]]);
    }

    #[test]
    fn padding_mask_and_type_ids_are_copied() {
        let b = pad(&[&enc(10, 2, 5), &enc(20, 3, 5)]);
        assert_eq!(b.input_ids.dim(), (2, 3));
        assert_eq!(b.attention_mask.row(0).to_vec(), vec![1.0, 1.0, 0.0]);
        assert_eq!(b.input_ids.row(1).to_vec(), vec![20, 21, 22]);
    }

    #[test]
    fn a_wrong_row_count_is_an_error() {
        let encs = vec![enc(1, 2, 2), enc(2, 2, 2)];
        let err = map_bucketed(&encs, 1 << 20, 256, |_| Ok(Array2::zeros((1, 1)))).unwrap_err();
        assert!(err.to_string().contains("rows"));
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert!(map_bucketed(&[], 8192, 256, first_ids).unwrap().is_empty());
    }
}
