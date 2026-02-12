


#[cfg(test)]
mod rag_integration_tests {
    use crate::{IndexReader, IndexWriter, config::IndexConfig};
    use std::collections::HashMap;
    use tempfile::tempdir;


    #[test]
    fn test_rag_full_lifecycle() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let index_path = dir.path().join("my_index");
        let dim = 4;

        {
            let config = IndexConfig {
                dimension: dim,
                max_docs_per_segment: 2, 
                ..Default::default()
            };

            let mut writer = IndexWriter::open(&index_path, config)?;

            let mut meta1 = HashMap::new();
            meta1.insert("category".to_string(), "fruit".to_string());
            writer.add(
                "Apple is a fruit", 
                &[1.0, 0.0, 0.0, 0.0], 
                Some(&meta1)
            )?;

            writer.add(
                "Car is a vehicle", 
                &[0.0, 1.0, 0.0, 0.0], 
                None
            )?;

            writer.add(
                "Banana is yellow", 
                &[0.9, 0.1, 0.0, 0.0], 
                None
            )?;

            writer.commit()?;
        } 
        let reader = IndexReader::open(&index_path)?;
        
        assert_eq!(reader.len(), 3);
        assert_eq!(reader.dimension(), dim);
        assert_eq!(reader.segment_count(), 2);

        let query_vec = [1.0, 0.0, 0.0, 0.0]; 
        let results = reader.search_semantic(&query_vec, 10);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].text, "Apple is a fruit");
        assert!(results[0].score > 0.99); 
        
        assert_eq!(results[1].text, "Banana is yellow");
        
        assert_eq!(results[2].text, "Car is a vehicle");

        let results = reader.search_keywords("yellow", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "Banana is yellow");

        let results = reader.search_hybrid("vehicle", &[1.0, 0.0, 0.0, 0.0], 10);
        
        let top_texts: Vec<&str> = results.iter().take(2).map(|r| r.text.as_str()).collect();
        assert!(top_texts.contains(&"Car is a vehicle"));
        assert!(top_texts.contains(&"Apple is a fruit"));

        let apple_doc = results.iter().find(|r| r.text.contains("Apple")).unwrap();
        assert_eq!(apple_doc.metadata.get("category"), Some(&"fruit".to_string()));

        Ok(())
    }

    #[test]
    fn test_appending_to_existing_index() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let index_path = dir.path().join("append_index");
        let config = IndexConfig::with_dimension(2);

        {
            let mut writer = IndexWriter::open(&index_path, config)?;
            writer.add("Doc 1", &[1.0, 0.0], None)?;
            writer.commit()?;
        }

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