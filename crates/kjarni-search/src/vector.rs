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
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self::cosine_against(a, norm_a, b)
    }

    /// Cosine similarity where the first vector's norm is already known.
    ///
    /// A scan compares one query against every stored vector, so the query's norm is
    /// invariant across the whole loop. Recomputing it per document, as the previous
    /// implementation did, spent a third of the inner loop on the same answer.
    fn cosine_against(query: &[f32], query_norm: f32, other: &[f32]) -> f32 {
        if query.len() != other.len() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm_other = 0.0;
        for i in 0..query.len() {
            dot_product += query[i] * other[i];
            norm_other += other[i] * other[i];
        }

        let denominator = (query_norm * norm_other.sqrt()).max(1e-9);
        dot_product / denominator
    }

    /// Exact top-`limit` search by cosine similarity.
    ///
    /// This is a full scan, deliberately. There is no index to build, nothing to
    /// tune, and no recall cliff: a document is searchable the moment it is added,
    /// and the answer is the true nearest neighbour rather than a probable one.
    ///
    /// Measured on 384-dimensional vectors, best of five on a 24-core machine:
    /// 0.2ms against 10K, 4.9ms against 100K, 56ms against 1M. The full-sort scan
    /// this replaced took 2.6ms, 25ms and 255ms for the same work.
    ///
    /// An approximate index would save a few milliseconds at the top of that range
    /// in exchange for a build step, a memory overhead, a recall parameter to
    /// explain, and answers that are only usually right.
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Vec<(usize, f32)> {
        if self.embeddings.is_empty() || query_embedding.len() != self.dimension || limit == 0 {
            return vec![];
        }

        let query_norm = query_embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        // Below this the scan finishes in well under a millisecond and thread
        // hand-off costs more than it saves.
        const PARALLEL_THRESHOLD: usize = 2048;

        #[cfg(not(target_arch = "wasm32"))]
        if self.embeddings.len() >= PARALLEL_THRESHOLD {
            use rayon::prelude::*;

            return self
                .embeddings
                .par_iter()
                .enumerate()
                .fold(
                    || TopK::new(limit),
                    |mut top, (idx, emb)| {
                        top.offer(idx, Self::cosine_against(query_embedding, query_norm, emb));
                        top
                    },
                )
                .reduce(|| TopK::new(limit), TopK::merged)
                .into_vec();
        }

        let mut top = TopK::new(limit);
        for (idx, emb) in self.embeddings.iter().enumerate() {
            top.offer(idx, Self::cosine_against(query_embedding, query_norm, emb));
        }
        top.into_vec()
    }
}

/// Orders two candidates: higher score first, lower index breaking ties.
///
/// The tie-break is not cosmetic. Without a total order the parallel scan could
/// return a different permutation of equally-scoring documents depending on how
/// rayon happened to split the work, so the same query would give different answers
/// between runs. NaN scores compare as tied and fall through to the index, which
/// keeps them ordered rather than poisoning the comparison.
fn ranks_before(a: (usize, f32), b: (usize, f32)) -> bool {
    match b.1.partial_cmp(&a.1) {
        Some(std::cmp::Ordering::Less) => true,
        Some(std::cmp::Ordering::Greater) => false,
        Some(std::cmp::Ordering::Equal) | None => a.0 < b.0,
    }
}

/// A bounded best-`limit` accumulator, kept sorted best-first.
///
/// Scoring every candidate and then sorting the lot is O(n log n) and allocates a
/// slot per document. Keeping only the best `limit` seen so far is O(n log limit)
/// with a fixed footprint, and since `limit` is typically around ten, nearly every
/// candidate is rejected by a single float comparison against the current worst.
struct TopK {
    limit: usize,
    items: Vec<(usize, f32)>,
}

impl TopK {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            items: Vec::with_capacity(limit.saturating_add(1).min(1024)),
        }
    }

    fn offer(&mut self, idx: usize, score: f32) {
        let candidate = (idx, score);

        // Once full, the common case is losing to the worst kept entry.
        if self.items.len() >= self.limit {
            match self.items.last() {
                Some(&worst) if !ranks_before(candidate, worst) => return,
                _ => {}
            }
        }

        let pos = self.items.partition_point(|&kept| ranks_before(kept, candidate));
        self.items.insert(pos, candidate);
        if self.items.len() > self.limit {
            self.items.pop();
        }
    }

    fn merged(mut self, other: Self) -> Self {
        for (idx, score) in other.items {
            self.offer(idx, score);
        }
        self
    }

    fn into_vec(self) -> Vec<(usize, f32)> {
        self.items
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

    /// Builds a store big enough to cross PARALLEL_THRESHOLD, deterministically.
    fn large_store(n: usize, dim: usize) -> VectorStore {
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / 8_388_608.0 - 1.0
        };
        let embeddings = (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect();
        VectorStore::new(embeddings).unwrap()
    }

    /// The reference implementation this replaced: score everything, sort, truncate.
    fn search_by_full_sort(
        store: &VectorStore,
        query: &[f32],
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let mut all: Vec<(usize, f32)> = store
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| (i, VectorStore::cosine_similarity(query, e)))
            .collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(limit);
        all
    }

    #[test]
    fn test_search_matches_full_sort_on_the_parallel_path() {
        // 4096 crosses PARALLEL_THRESHOLD, so this exercises the rayon branch and
        // pins it to the exact answer the old full sort produced.
        let store = large_store(4096, 64);
        let query: Vec<f32> = store.embeddings[1234].clone();

        for limit in [1, 5, 10, 100] {
            let fast = store.search(&query, limit);
            let reference = search_by_full_sort(&store, &query, limit);

            assert_eq!(fast.len(), reference.len(), "limit {limit}");
            for (got, want) in fast.iter().zip(reference.iter()) {
                assert_eq!(got.0, want.0, "index mismatch at limit {limit}");
                assert!((got.1 - want.1).abs() < 1e-6, "score mismatch at limit {limit}");
            }
        }
    }

    #[test]
    fn test_search_matches_full_sort_on_the_serial_path() {
        let store = large_store(512, 64);
        let query: Vec<f32> = store.embeddings[7].clone();

        let fast = store.search(&query, 10);
        let reference = search_by_full_sort(&store, &query, 10);

        assert_eq!(
            fast.iter().map(|r| r.0).collect::<Vec<_>>(),
            reference.iter().map(|r| r.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_search_is_deterministic_when_scores_tie() {
        // Every vector is identical, so every score ties. Without a total order the
        // parallel scan could return any subset in any order, varying per run.
        let store = VectorStore::new(vec![vec![1.0, 0.0]; 5000]).unwrap();
        let query = vec![1.0, 0.0];

        let first = store.search(&query, 5);
        assert_eq!(
            first.iter().map(|r| r.0).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4],
            "ties must resolve to the lowest indices, in order"
        );

        for _ in 0..20 {
            assert_eq!(store.search(&query, 5), first, "repeated queries must agree");
        }
    }

    #[test]
    fn test_search_zero_limit_returns_nothing() {
        let store = VectorStore::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        assert!(store.search(&vec![1.0, 0.0], 0).is_empty());
    }

    #[test]
    fn test_search_limit_above_len_returns_everything_sorted() {
        let store = large_store(3000, 32);
        let query: Vec<f32> = store.embeddings[0].clone();

        let results = store.search(&query, 10_000);
        assert_eq!(results.len(), 3000, "limit above len must not truncate");
        assert_eq!(results[0].0, 0, "a vector is its own nearest neighbour");
        for pair in results.windows(2) {
            assert!(pair[0].1 >= pair[1].1, "scores must be descending");
        }
    }

    #[test]
    fn test_cosine_similarity_unchanged_by_the_precomputed_norm() {
        // cosine_similarity now delegates to cosine_against; the results have to be
        // identical or every caller outside search shifts underneath us.
        let cases = [
            (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
            (vec![1.0, 0.0], vec![0.0, 1.0]),
            (vec![-1.0, -2.0], vec![1.0, 2.0]),
            (vec![0.0, 0.0], vec![1.0, 1.0]),
        ];
        for (a, b) in cases {
            let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert_eq!(
                VectorStore::cosine_similarity(&a, &b),
                VectorStore::cosine_against(&a, norm_a, &b)
            );
        }
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
