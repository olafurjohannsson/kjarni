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
            writer.add("Apple is a fruit", &[1.0, 0.0, 0.0, 0.0], Some(&meta1))?;

            writer.add("Car is a vehicle", &[0.0, 1.0, 0.0, 0.0], None)?;

            writer.add("Banana is yellow", &[0.9, 0.1, 0.0, 0.0], None)?;

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
        assert_eq!(
            apple_doc.metadata.get("category"),
            Some(&"fruit".to_string())
        );

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

    fn config(dimension: usize, max_docs_per_segment: usize) -> IndexConfig {
        IndexConfig {
            dimension,
            max_docs_per_segment,
            ..Default::default()
        }
    }

    #[test]
    fn a_filter_finds_a_match_ranked_below_any_multiple_of_the_limit() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let mut writer = IndexWriter::open(&path, config(4, 10_000))?;
        for i in 0..200 {
            let lang = if i == 150 { "de" } else { "en" };
            let meta = HashMap::from([("lang".to_string(), lang.to_string())]);
            let v = [1.0 - i as f32 / 1000.0, i as f32 / 1000.0, 0.0, 0.0];
            writer.add(&format!("apple document {i}"), &v, Some(&meta))?;
        }
        writer.commit()?;

        let reader = IndexReader::open(&path)?;
        let filter = crate::MetadataFilter::default().must("lang", "de");
        let query = [1.0, 0.0, 0.0, 0.0];
        for results in [
            reader.search_semantic_filtered(&query, 10, &filter),
            reader.search_keywords_filtered("apple", 10, &filter),
            reader.search_hybrid_filtered("apple", &query, 10, &filter),
        ] {
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].text, "apple document 150");
        }
        Ok(())
    }

    #[test]
    fn identical_text_scores_the_same_in_every_segment() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let mut writer = IndexWriter::open(&path, config(2, 10_000))?;
        writer.add("zebra migration routes", &[1.0, 0.0], None)?;
        for i in 1..5000 {
            writer.add(&format!("filler text number n{i}"), &[1.0, 0.0], None)?;
        }
        writer.commit()?;
        let mut writer = IndexWriter::open_existing(&path)?;
        writer.add("zebra migration routes", &[1.0, 0.0], None)?;
        writer.commit()?;

        let results = IndexReader::open(&path)?.search_keywords("zebra", 5);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].score, results[1].score);
        assert_eq!((results[0].document_id, results[1].document_id), (0, 5000));
        Ok(())
    }

    #[test]
    fn open_refuses_a_path_that_already_holds_an_index() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let mut writer = IndexWriter::open(&path, config(3, 100))?;
        writer.add("old doc", &[1.0, 0.0, 0.0], None)?;
        writer.commit()?;

        assert!(IndexWriter::open(&path, config(4, 100)).is_err());

        let reader = IndexReader::open(&path)?;
        assert_eq!((reader.len(), reader.dimension()), (1, 3));
        Ok(())
    }

    #[test]
    fn nothing_is_visible_until_commit_and_dropping_discards_it() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let mut writer = IndexWriter::open(&path, config(2, 2))?;
        for i in 0..6 {
            writer.add(&format!("doc {i}"), &[1.0, 0.0], None)?;
        }
        assert_eq!(IndexReader::open(&path)?.len(), 0);
        drop(writer);
        assert_eq!(IndexReader::open(&path)?.len(), 0);

        let mut writer = IndexWriter::open_existing(&path)?;
        for i in 0..5 {
            writer.add(&format!("doc {i}"), &[1.0, 0.0], None)?;
        }
        writer.commit()?;
        let reader = IndexReader::open(&path)?;
        assert_eq!((reader.len(), reader.segment_count()), (5, 3));
        // Three live segments and nothing left over from the abandoned writer.
        assert_eq!(std::fs::read_dir(path.join("segments"))?.count(), 3);
        Ok(())
    }

    #[test]
    fn a_damaged_segment_fails_the_open() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let mut writer = IndexWriter::open(&path, config(2, 2))?;
        for i in 0..6 {
            writer.add(&format!("doc {i}"), &[1.0, 0.0], None)?;
        }
        writer.commit()?;
        std::fs::write(path.join("segments/seg_000001/bm25.bin"), b"x")?;

        let error = IndexReader::open(&path).err().expect("open must fail");
        assert!(error.to_string().contains("seg_000001"), "{error}");
        Ok(())
    }

    #[test]
    fn a_newer_index_version_is_refused() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let newer = IndexConfig {
            version: crate::config::INDEX_VERSION + 1,
            ..config(2, 2)
        };
        IndexWriter::open(&path, newer)?.commit()?;

        let error = IndexReader::open(&path).err().expect("open must fail");
        assert!(error.to_string().contains("newer"), "{error}");
        Ok(())
    }

    #[test]
    fn a_nan_vector_scores_zero_instead_of_ranking_first() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let mut writer = IndexWriter::open(&path, config(2, 100))?;
        writer.add("broken", &[f32::NAN, 1.0], None)?;
        writer.add("fine", &[1.0, 0.0], None)?;
        writer.commit()?;

        let results = IndexReader::open(&path)?.search_semantic(&[1.0, 0.0], 2);
        assert_eq!(results[0].text, "fine");
        assert_eq!(results[1].score, 0.0);
        Ok(())
    }

    #[test]
    fn index_build_leaves_the_target_alone_until_installed() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let mut writer = IndexWriter::open(&path, config(2, 100))?;
        writer.add("old", &[1.0, 0.0], None)?;
        writer.commit()?;

        // Abandoned: the old index survives and no build folder is left.
        {
            let build = crate::IndexBuild::new(&path)?;
            let mut writer = IndexWriter::open(build.path(), config(2, 100))?;
            writer.add("abandoned", &[1.0, 0.0], None)?;
            writer.commit()?;
        }
        assert_eq!(IndexReader::open(&path)?.get_document(0)?, "old");
        assert_eq!(std::fs::read_dir(dir.path())?.count(), 1);

        let build = crate::IndexBuild::new(&path)?;
        let mut writer = IndexWriter::open(build.path(), config(2, 100))?;
        writer.add("new", &[1.0, 0.0], None)?;
        writer.commit()?;
        build.install()?;
        assert_eq!(IndexReader::open(&path)?.get_document(0)?, "new");
        assert_eq!(std::fs::read_dir(dir.path())?.count(), 1);
        Ok(())
    }

    #[test]
    fn a_directory_walk_skips_an_index_inside_it() -> anyhow::Result<()> {
        let dir = tempdir()?;
        IndexWriter::open(dir.path().join("idx"), config(2, 2))?.commit()?;
        std::fs::write(dir.path().join("notes.json"), "{}")?;

        let files = crate::DocumentLoader::with_defaults().files_in(dir.path());
        assert_eq!(files, [dir.path().join("notes.json")]);
        Ok(())
    }

    #[test]
    fn source_code_is_stored_exactly_as_written() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let source = "if (a < b) { keep(arr[0]); }\nlet v = vec![1, 2];\nreturn rest;";
        let file = dir.path().join("code.rs");
        std::fs::write(&file, source)?;

        let chunks = crate::DocumentLoader::with_defaults().load_file(&file)?;
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, source);
        Ok(())
    }

    #[test]
    fn markdown_files_are_still_cleaned() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let file = dir.path().join("note.md");
        std::fs::write(&file, "See [the docs](http://x.y) and <b>this</b>.")?;

        let chunks = crate::DocumentLoader::with_defaults().load_file(&file)?;
        assert_eq!(chunks[0].text, "See the docs and this.");
        Ok(())
    }

    #[test]
    fn source_globs_match_relative_absolute_and_backslash_paths() {
        let filter = crate::MetadataFilter::default().source("docs/**");
        for source in [
            "docs/a.md",
            "./docs/a.md",
            "/home/u/docs/a.md",
            r"C:\u\docs\a.md",
        ] {
            let meta = HashMap::from([("source".to_string(), source.to_string())]);
            assert!(filter.matches(&meta), "{source}");
        }
        let other = HashMap::from([("source".to_string(), "src/a.md".to_string())]);
        assert!(!filter.matches(&other));
    }

    fn with_source(source: &str) -> HashMap<String, String> {
        HashMap::from([(crate::SOURCE_PATH_KEY.to_string(), source.to_string())])
    }

    #[test]
    fn a_second_writer_is_refused_until_the_first_is_gone() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let first = IndexWriter::open(&path, config(2, 10))?;

        let error = IndexWriter::open_existing(&path)
            .err()
            .expect("second writer must be refused");
        assert!(error.to_string().contains("another writer"), "{error}");

        first.commit()?;
        IndexWriter::open_existing(&path)?.commit()?;
        Ok(())
    }

    #[test]
    fn replacing_a_file_swaps_its_chunks_and_keeps_the_others() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let mut writer = IndexWriter::open(&path, config(2, 2))?;
        writer.add("old alpha", &[1.0, 0.0], Some(&with_source("/a")))?;
        writer.add("keep beta", &[1.0, 0.0], Some(&with_source("/b")))?;
        writer.add("old gamma", &[1.0, 0.0], Some(&with_source("/a")))?;
        writer.commit()?;

        let mut writer = IndexWriter::open_existing(&path)?;
        writer.add("new delta", &[1.0, 0.0], Some(&with_source("/a")))?;
        let removed = writer.remove_where(|m| m.get(crate::SOURCE_PATH_KEY).unwrap() == "/a")?;
        assert_eq!(removed, 2);
        assert_eq!(writer.len(), 2);

        // Nothing changes for readers until the commit.
        assert_eq!(IndexReader::open(&path)?.len(), 3);
        writer.commit()?;

        let reader = IndexReader::open(&path)?;
        assert_eq!(reader.len(), 2);
        assert!(reader.search_keywords("old", 5).is_empty());
        assert_eq!(reader.search_keywords("beta", 5).len(), 1);
        assert_eq!(reader.search_keywords("delta", 5).len(), 1);
        let meta = reader.search_keywords("beta", 5).remove(0).metadata;
        assert_eq!(meta.get(crate::SOURCE_PATH_KEY).unwrap(), "/b");
        Ok(())
    }

    #[test]
    fn an_abandoned_removal_changes_nothing() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let mut writer = IndexWriter::open(&path, config(2, 10))?;
        writer.add("alpha", &[1.0, 0.0], Some(&with_source("/a")))?;
        writer.commit()?;

        let mut writer = IndexWriter::open_existing(&path)?;
        assert_eq!(writer.remove_where(|_| true)?, 1);
        drop(writer);

        let reader = IndexReader::open(&path)?;
        assert_eq!(reader.get_document(0)?, "alpha");
        assert_eq!(std::fs::read_dir(path.join("segments"))?.count(), 1);
        Ok(())
    }

    #[test]
    fn unlisted_segments_from_a_crash_are_invisible_and_cleaned_up() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let mut writer = IndexWriter::open(&path, config(2, 10))?;
        writer.add("alpha", &[1.0, 0.0], None)?;
        writer.commit()?;

        let crashed = path.join("segments/seg_000099");
        std::fs::create_dir_all(&crashed)?;
        std::fs::write(crashed.join("junk"), "partial")?;
        assert_eq!(IndexReader::open(&path)?.len(), 1);

        IndexWriter::open_existing(&path)?.commit()?;
        assert!(!crashed.exists());
        Ok(())
    }

    #[test]
    fn small_segments_merge_without_losing_documents_or_order() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        IndexWriter::open(&path, config(2, 100))?.commit()?;
        for i in 0..8 {
            let mut writer = IndexWriter::open_existing(&path)?;
            let meta = with_source(&format!("/{i}"));
            writer.add(&format!("doc {i}"), &[1.0, i as f32], Some(&meta))?;
            writer.commit()?;
        }

        let reader = IndexReader::open(&path)?;
        assert_eq!((reader.len(), reader.segment_count()), (8, 1));
        for i in 0..8 {
            assert_eq!(reader.get_document(i)?, format!("doc {i}"));
            assert_eq!(
                reader.get_metadata(i)?.get(crate::SOURCE_PATH_KEY).unwrap(),
                &format!("/{i}")
            );
        }
        let best = reader.search_semantic(&[1.0, 7.0], 1).remove(0);
        assert_eq!(best.text, "doc 7");
        assert!((best.score - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn a_reader_notices_a_new_commit() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        IndexWriter::open(&path, config(2, 10))?.commit()?;
        let reader = IndexReader::open(&path)?;
        assert!(reader.is_current());

        IndexWriter::open_existing(&path)?.commit()?;
        assert!(!reader.is_current());
        Ok(())
    }

    #[test]
    fn an_older_index_version_is_refused() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("index");
        let older = IndexConfig {
            version: crate::config::INDEX_VERSION - 1,
            ..config(2, 2)
        };
        IndexWriter::open(&path, older)?.commit()?;

        let error = IndexReader::open(&path).err().expect("open must fail");
        assert!(error.to_string().contains("rebuild"), "{error}");
        assert!(IndexWriter::open_existing(&path).is_err());
        Ok(())
    }

    #[test]
    fn loaded_chunks_record_an_absolute_path_and_folders_list_in_order() -> anyhow::Result<()> {
        let dir = tempdir()?;
        for name in ["c.txt", "a.txt", "b.txt"] {
            std::fs::write(dir.path().join(name), "some text")?;
        }
        let loader = crate::DocumentLoader::with_defaults();

        let files = loader.files_in(dir.path());
        let names: Vec<_> = files.iter().map(|f| f.file_name().unwrap()).collect();
        assert_eq!(names, ["a.txt", "b.txt", "c.txt"]);

        let chunks = loader.load_file(&files[0])?;
        let identity = chunks[0].metadata.to_hashmap();
        assert_eq!(
            identity.get(crate::SOURCE_PATH_KEY).unwrap(),
            &crate::source_path(&files[0])
        );
        assert!(std::path::Path::new(identity.get(crate::SOURCE_PATH_KEY).unwrap()).is_absolute());
        Ok(())
    }
}
