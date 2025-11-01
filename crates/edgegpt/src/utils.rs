//! Shared utility functions

/// Compute cosine similarity between two embeddings
///
/// # Arguments
/// * `a` - First embedding vector
/// * `b` - Second embedding vector
///
/// # Returns
/// Cosine similarity score between -1.0 and 1.0
///
/// # Example
/// ```
/// use edgegpt::utils::cosine_similarity;
///
/// let emb1 = vec![0.5, 0.5, 0.5];
/// let emb2 = vec![0.6, 0.4, 0.5];
/// let sim = cosine_similarity(&emb1, &emb2);
/// println!("Similarity: {:.4}", sim);
/// ```
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
///
/// # Arguments
/// * `embeddings` - Slice of embedding vectors
///
/// # Returns
/// 2D similarity matrix where matrix[i][j] is the similarity between embeddings[i] and embeddings[j]
///
/// # Example
/// ```
/// use edgegpt::utils::similarity_matrix;
///
/// let embeddings = vec![
///     vec![0.5, 0.5, 0.5],
///     vec![0.6, 0.4, 0.5],
///     vec![0.1, 0.1, 0.9],
/// ];
///
/// let matrix = similarity_matrix(&embeddings);
/// println!("Similarity between 0 and 1: {:.4}", matrix[0][1]);
/// ```
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
///
/// # Arguments
/// * `query` - Query embedding
/// * `candidates` - Slice of candidate embeddings
/// * `top_k` - Number of top results to return
///
/// # Returns
/// Vector of (index, similarity_score) tuples, sorted by similarity (highest first)
///
/// # Example
/// ```
/// use edgegpt::utils::find_most_similar;
///
/// let query = vec![0.5, 0.5, 0.5];
/// let candidates = vec![
///     vec![0.6, 0.4, 0.5],
///     vec![0.1, 0.1, 0.9],
///     vec![0.5, 0.5, 0.6],
/// ];
///
/// let top_2 = find_most_similar(&query, &candidates, 2);
/// for (idx, score) in top_2 {
///     println!("Candidate {} has similarity {:.4}", idx, score);
/// }
/// ```
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
