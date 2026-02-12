//! Shared utility functions

/// Compute cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Embeddings must have the same length");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Compute pairwise cosine similarity matrix for a batch of embeddings
pub fn similarity_matrix(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        matrix[i][i] = 1.0; // Self-similarity is always 1.0
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim; // Symmetric
        }
    }

    matrix
}

/// Find the most similar embeddings to a query embedding
pub fn find_most_similar(
    query: &[f32],
    candidates: &[Vec<f32>],
    top_k: usize,
) -> Vec<(usize, f32)> {
    let mut similarities: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(idx, emb)| (idx, cosine_similarity(query, emb)))
        .collect();

    // Sort by similarity (descending)
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top K
    similarities.truncate(top_k);

    similarities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_matrix_symmetry() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];

        let matrix = similarity_matrix(&embeddings);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-6);
            }
        }

        // Check diagonal is 1.0
        for i in 0..3 {
            assert!((matrix[i][i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_find_most_similar() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            vec![0.9, 0.1, 0.0], // Most similar
            vec![0.0, 1.0, 0.0], // Least similar
            vec![0.7, 0.3, 0.0], // Second most similar
        ];

        let top_2 = find_most_similar(&query, &candidates, 2);

        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].0, 0); // Index 0 is most similar
        assert_eq!(top_2[1].0, 2); // Index 2 is second
        assert!(top_2[0].1 > top_2[1].1); // Scores are descending
    }
}
