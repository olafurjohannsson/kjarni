use std::collections::HashMap;

pub fn hybrid_search(
    keyword_results: Vec<(usize, f32)>,
    semantic_results: Vec<(usize, f32)>,
    limit: usize,
) -> Vec<(usize, f32)> {
    let mut combined_scores: HashMap<usize, f32> = HashMap::new();
    let k = 60.0;

    for (rank, (idx, _score)) in keyword_results.iter().enumerate() {
        let score = 1.0 / (k + (rank + 1) as f32);
        combined_scores
            .entry(*idx)
            .and_modify(|s| *s += score)
            .or_insert(score);
    }

    for (rank, (idx, _score)) in semantic_results.iter().enumerate() {
        let score = 1.0 / (k + (rank + 1) as f32);
        combined_scores
            .entry(*idx)
            .and_modify(|s| *s += score)
            .or_insert(score);
    }

    let mut final_results: Vec<(usize, f32)> = combined_scores.into_iter().collect();
    final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    final_results.truncate(limit);
    final_results
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_search_basic() {
        let keywords = vec![(0, 1.0), (1, 0.5)];
        let semantic = vec![(1, 0.9), (2, 0.4)];

        let results = hybrid_search(keywords, semantic, 10);

        // Doc 1 appears in both, should rank highest
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_hybrid_search_empty() {
        let results = hybrid_search(vec![], vec![], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hybrid_search_limit() {
        let keywords = vec![(0, 1.0), (1, 0.9), (2, 0.8)];
        let semantic = vec![(3, 0.9), (4, 0.8), (5, 0.7)];

        let results = hybrid_search(keywords, semantic, 2);
        assert_eq!(results.len(), 2);
    }
}