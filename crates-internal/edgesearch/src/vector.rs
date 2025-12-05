use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorStore {
    pub embeddings: Vec<Vec<f32>>,
    pub dimension: usize,
}

impl VectorStore {
    pub fn new(embeddings: Vec<Vec<f32>>) -> Result<Self> {
        if embeddings.is_empty() {
            return Ok(Self {
                embeddings: vec![],
                dimension: 0,
            });
        }

        let dimension = embeddings[0].len();

        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != dimension {
                return Err(anyhow::anyhow!(
                    "Embedding {} has dimension {} but expected {}",
                    i,
                    emb.len(),
                    dimension
                ));
            }
        }

        Ok(Self {
            embeddings,
            dimension,
        })
    }

    /// Create empty store with known dimension
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            embeddings: vec![],
            dimension,
        }
    }

    /// Number of embeddings
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Add a single embedding
    pub fn add(&mut self, embedding: Vec<f32>) -> Result<usize> {
        if self.dimension == 0 {
            self.dimension = embedding.len();
        } else if embedding.len() != self.dimension {
            return Err(anyhow::anyhow!(
                "Embedding has dimension {} but store expects {}",
                embedding.len(),
                self.dimension
            ));
        }

        let idx = self.embeddings.len();
        self.embeddings.push(embedding);
        Ok(idx)
    }

    /// Add multiple embeddings
    pub fn add_batch(&mut self, embeddings: Vec<Vec<f32>>) -> Result<Vec<usize>> {
        let start_idx = self.embeddings.len();
        for (i, emb) in embeddings.into_iter().enumerate() {
            self.add(emb)
                .map_err(|e| anyhow::anyhow!("Embedding {}: {}", i, e))?;
        }
        Ok((start_idx..self.embeddings.len()).collect())
    }

    /// Get embedding by index
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        self.embeddings.get(index).map(|v| v.as_slice())
    }

    /// Search with minimum similarity threshold
    pub fn search_with_threshold(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_similarity: f32,
    ) -> Vec<(usize, f32)> {
        self.search(query_embedding, limit)
            .into_iter()
            .filter(|(_, score)| *score >= min_similarity)
            .collect()
    }

    /// Dot product similarity (for normalized vectors)
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Euclidean distance (L2)
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize all embeddings to unit vectors (for faster dot product search)
    pub fn normalize(&mut self) {
        for emb in &mut self.embeddings {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-9 {
                for x in emb.iter_mut() {
                    *x /= norm;
                }
            }
        }
    }
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..a.len() {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let denominator = (norm_a.sqrt() * norm_b.sqrt()).max(1e-9);
        dot_product / denominator
    }

    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Vec<(usize, f32)> {
        if self.embeddings.is_empty() || query_embedding.len() != self.dimension {
            return vec![];
        }

        let mut similarities: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(idx, emb)| (idx, Self::cosine_similarity(query_embedding, emb)))
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(limit);
        similarities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_store_new_empty() {
        let store = VectorStore::new(vec![]).unwrap();
        assert_eq!(store.dimension, 0);
        assert!(store.embeddings.is_empty());
    }

    #[test]
    fn test_vector_store_new_valid() {
        let embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let store = VectorStore::new(embeddings).unwrap();

        assert_eq!(store.dimension, 3);
        assert_eq!(store.embeddings.len(), 2);
    }

    #[test]
    fn test_vector_store_new_dimension_mismatch() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0], // Wrong dimension
        ];
        let result = VectorStore::new(embeddings);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dimension"));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let sim = VectorStore::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        let sim = VectorStore::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];

        let sim = VectorStore::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        let sim = VectorStore::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];

        let sim = VectorStore::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6); // Should handle gracefully
    }

    #[test]
    fn test_search_empty_store() {
        let store = VectorStore::new(vec![]).unwrap();
        let query = vec![1.0, 2.0, 3.0];

        let results = store.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let store = VectorStore::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        let query = vec![1.0, 2.0]; // Wrong dimension

        let results = store.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_returns_sorted() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0], // idx 0
            vec![0.9, 0.1, 0.0], // idx 1 - most similar to query
            vec![0.0, 1.0, 0.0], // idx 2 - orthogonal
        ];
        let store = VectorStore::new(embeddings).unwrap();
        let query = vec![1.0, 0.0, 0.0];

        let results = store.search(&query, 10);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0); // Exact match first
        assert_eq!(results[1].0, 1); // Close second
        assert_eq!(results[2].0, 2); // Orthogonal last

        // Verify scores are descending
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[test]
    fn test_search_limit() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.8, 0.2],
            vec![0.7, 0.3],
            vec![0.6, 0.4],
        ];
        let store = VectorStore::new(embeddings).unwrap();
        let query = vec![1.0, 0.0];

        let results = store.search(&query, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_limit_exceeds_size() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.9, 0.1]];
        let store = VectorStore::new(embeddings).unwrap();
        let query = vec![1.0, 0.0];

        let results = store.search(&query, 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_serde_roundtrip() {
        let store = VectorStore::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();

        let json = serde_json::to_string(&store).unwrap();
        let restored: VectorStore = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.dimension, 3);
        assert_eq!(restored.embeddings.len(), 2);
        assert_eq!(restored.embeddings[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_default() {
        let store = VectorStore::default();
        assert!(store.embeddings.is_empty());
        assert_eq!(store.dimension, 0);
    }
    #[test]
    fn test_with_dimension() {
        let store = VectorStore::with_dimension(384);
        assert_eq!(store.dimension, 384);
        assert!(store.is_empty());
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut store = VectorStore::with_dimension(3);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.add(vec![1.0, 2.0, 3.0]).unwrap();
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_add_single() {
        let mut store = VectorStore::with_dimension(3);

        let idx = store.add(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(idx, 0);

        let idx = store.add(vec![4.0, 5.0, 6.0]).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let mut store = VectorStore::with_dimension(3);
        store.add(vec![1.0, 2.0, 3.0]).unwrap();

        let result = store.add(vec![1.0, 2.0]); // Wrong dim
        assert!(result.is_err());
    }

    #[test]
    fn test_add_sets_dimension() {
        let mut store = VectorStore::default();
        assert_eq!(store.dimension, 0);

        store.add(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(store.dimension, 3);
    }

    #[test]
    fn test_get() {
        let store = VectorStore::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        assert_eq!(store.get(0), Some([1.0, 2.0].as_slice()));
        assert_eq!(store.get(1), Some([3.0, 4.0].as_slice()));
        assert_eq!(store.get(99), None);
    }

    #[test]
    fn test_search_with_threshold() {
        let store = VectorStore::new(vec![
            vec![1.0, 0.0], // sim = 1.0
            vec![0.7, 0.7], // sim ≈ 0.7
            vec![0.0, 1.0], // sim = 0.0
        ])
        .unwrap();

        let results = store.search_with_threshold(&[1.0, 0.0], 10, 0.5);

        assert_eq!(results.len(), 2); // Only first two pass threshold
        assert!(results.iter().all(|(_, score)| *score >= 0.5));
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let dot = VectorStore::dot_product(&a, &b);
        assert!((dot - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let dist = VectorStore::euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6); // 3-4-5 triangle
    }

    #[test]
    fn test_normalize() {
        let mut store = VectorStore::new(vec![
            vec![3.0, 4.0], // norm = 5
        ])
        .unwrap();

        store.normalize();

        let emb = store.get(0).unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((emb[0] - 0.6).abs() < 1e-6);
        assert!((emb[1] - 0.8).abs() < 1e-6);
    }
}
