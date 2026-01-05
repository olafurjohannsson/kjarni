use crate::{IndexWriter, IndexReader, Document};
use tempfile::tempdir;

#[test]
fn test_rag_lifecycle() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("index");

    // 1. Write Index
    let mut writer = IndexWriter::new(&path)?;
    writer.add_document(Document {
        id: "doc1".to_string(),
        content: "Hello world".to_string(),
        metadata: None,
        embedding: vec![0.1, 0.2, 0.3], // Mock embedding
    })?;
    writer.commit()?;

    // 2. Read Index
    let reader = IndexReader::open(&path)?;
    assert_eq!(reader.num_docs(), 1);

    // 3. Search (Mock Vector Search)
    let query_vec = vec![0.1, 0.2, 0.3];
    let results = reader.search(&query_vec, 1)?;
    
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "doc1");

    Ok(())
}