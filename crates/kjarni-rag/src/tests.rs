


#[cfg(test)]
mod rag_integration_tests {
    use crate::{IndexReader, IndexWriter, config::IndexConfig};
    use std::collections::HashMap;
    use tempfile::tempdir;

    /// Helper to create a normalized vector
    fn mock_embedding(val: f32, dim: usize) -> Vec<f32> {
        let mut vec = vec![0.0; dim];
        vec[0] = val; // Put value in first dimension
        // Normalize roughly just for sanity if needed, 
        // but cosine sim handles non-normalized inputs in your implementation usually.
        vec
    }

    #[test]
    fn test_rag_full_lifecycle() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let index_path = dir.path().join("my_index");
        let dim = 4;

        // =========================================================
        // 1. WRITE INDEX
        // =========================================================
        {
            let config = IndexConfig {
                dimension: dim,
                max_docs_per_segment: 2, // Force segmentation (flush after 2 docs)
                ..Default::default()
            };

            let mut writer = IndexWriter::open(&index_path, config)?;

            // Doc 1: "Apple" (Vector A)
            let mut meta1 = HashMap::new();
            meta1.insert("category".to_string(), "fruit".to_string());
            writer.add(
                "Apple is a fruit", 
                &[1.0, 0.0, 0.0, 0.0], 
                Some(&meta1)
            )?;

            // Doc 2: "Car" (Vector B)
            writer.add(
                "Car is a vehicle", 
                &[0.0, 1.0, 0.0, 0.0], 
                None
            )?;

            // Doc 3: "Banana" (Vector A - similar to Apple)
            // This forces a flush because max_docs_per_segment = 2
            writer.add(
                "Banana is yellow", 
                &[0.9, 0.1, 0.0, 0.0], 
                None
            )?;

            writer.commit()?;
        } // Writer drops here

        // =========================================================
        // 2. READ INDEX
        // =========================================================
        let reader = IndexReader::open(&index_path)?;
        
        // Verify structure
        assert_eq!(reader.len(), 3);
        assert_eq!(reader.dimension(), dim);
        // We added 3 docs with max_2 per segment, so we expect 2 segments (2 + 1)
        assert_eq!(reader.segment_count(), 2);

        // =========================================================
        // 3. SEARCH (Semantic)
        // =========================================================
        // Search for "Apple" vector
        let query_vec = [1.0, 0.0, 0.0, 0.0]; 
        let results = reader.search_semantic(&query_vec, 10);

        assert_eq!(results.len(), 3);
        // Top result should be Doc 0 (Apple)
        assert_eq!(results[0].text, "Apple is a fruit");
        assert!(results[0].score > 0.99); 
        
        // Second result should be Banana (0.9 similarity)
        assert_eq!(results[1].text, "Banana is yellow");
        
        // Last result should be Car (0.0 similarity)
        assert_eq!(results[2].text, "Car is a vehicle");

        // =========================================================
        // 4. SEARCH (Keyword)
        // =========================================================
        let results = reader.search_keywords("yellow", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Banana is yellow");

        // =========================================================
        // 5. SEARCH (Hybrid)
        // =========================================================
        // Query: "vehicle" (Keyword matches Car) + Vector A (Matches Apple)
        // Hybrid should rank both reasonably high.
        let results = reader.search_hybrid("vehicle", &[1.0, 0.0, 0.0, 0.0], 10);
        
        // We expect Car (keyword match) and Apple (vector match) to be top 2
        let top_texts: Vec<&str> = results.iter().take(2).map(|r| r.text.as_str()).collect();
        assert!(top_texts.contains(&"Car is a vehicle"));
        assert!(top_texts.contains(&"Apple is a fruit"));

        // =========================================================
        // 6. METADATA CHECK
        // =========================================================
        // Apple was Doc 0 (global ID might differ depending on segment order, but text matches)
        let apple_doc = results.iter().find(|r| r.text.contains("Apple")).unwrap();
        assert_eq!(apple_doc.metadata.get("category"), Some(&"fruit".to_string()));

        Ok(())
    }

    #[test]
    fn test_appending_to_existing_index() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let index_path = dir.path().join("append_index");
        let config = IndexConfig::with_dimension(2);

        // Batch 1
        {
            let mut writer = IndexWriter::open(&index_path, config)?;
            writer.add("Doc 1", &[1.0, 0.0], None)?;
            writer.commit()?;
        }

        // Batch 2 (Open Existing)
        {
            let mut writer = IndexWriter::open_existing(&index_path)?;
            assert_eq!(writer.len(), 1); // Should know about existing docs
            writer.add("Doc 2", &[0.0, 1.0], None)?;
            writer.commit()?;
        }

        // Verify
        let reader = IndexReader::open(&index_path)?;
        assert_eq!(reader.len(), 2);
        
        // Ensure we can search both
        let r1 = reader.search_keywords("Doc", 10);
        assert_eq!(r1.len(), 2);

        Ok(())
    }
}