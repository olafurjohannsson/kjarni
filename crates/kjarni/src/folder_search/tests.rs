use super::*;

const RERANKER: &str = "minilm-l6-v2-cross-encoder";

fn write(dir: &Path, name: &str, text: &str) {
    let path = dir.join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn names(found: &FolderResults) -> Vec<String> {
    found
        .results
        .iter()
        .map(|h| h.source.file_name().unwrap().to_string_lossy().into_owned())
        .collect()
}

/// A second search after files change updates the index rather than
/// rebuilding it, and finds what was written since.
#[tokio::test(flavor = "multi_thread")]
async fn follows_edits_to_the_folder() {
    let cache = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let notes = dir.path().canonicalize().unwrap();
    write(&notes, "boat.md", "The ferry leaves the harbour at nine.");
    write(&notes, "bread.md", "Knead the dough and let it rise.");
    let search = FolderSearch::new().cache_dir(cache.path());

    let first = search
        .search(&notes, "when does the ferry go", 1)
        .await
        .unwrap();
    assert_eq!(first.update, Some(IndexUpdate::Built { files: 2 }));
    assert_eq!(names(&first), ["boat.md"]);
    assert_eq!(first.indexed_folder, notes);

    let unchanged = search.search(&notes, "baking", 1).await.unwrap();
    assert_eq!(unchanged.update, None);

    write(&notes, "bread.md", "Rye bread needs a sourdough starter.");
    write(&notes, "bike.md", "Pump the bicycle tyres every week.");
    let updated = search.search(&notes, "sourdough starter", 1).await.unwrap();
    assert_eq!(
        updated.update,
        Some(IndexUpdate::Updated {
            added: 1,
            changed: 1,
            removed: 0
        })
    );
    assert_eq!(names(&updated), ["bread.md"]);
    assert!(updated.results[0].text.contains("Rye bread"));

    std::fs::remove_file(notes.join("boat.md")).unwrap();
    let removed = search.search(&notes, "ferry harbour", 3).await.unwrap();
    assert_eq!(
        removed.update,
        Some(IndexUpdate::Updated {
            added: 0,
            changed: 0,
            removed: 1
        })
    );
    assert!(!names(&removed).contains(&"boat.md".to_string()));
}

/// A folder inside an indexed one is searched through the parent's index,
/// and only its own files come back.
#[tokio::test(flavor = "multi_thread")]
async fn a_subfolder_uses_its_parents_index() {
    let cache = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    write(
        &root,
        "travel/boat.md",
        "The ferry leaves the harbour at nine.",
    );
    write(
        &root,
        "travel/train.md",
        "The night train to the coast is slow.",
    );
    write(
        &root,
        "kitchen/bread.md",
        "Knead the dough and let it rise.",
    );
    let search = FolderSearch::new().cache_dir(cache.path());

    search.search(&root, "anything", 1).await.unwrap();
    let travel = root.join("travel");
    let found = search.search(&travel, "baking bread", 5).await.unwrap();
    assert_eq!(found.indexed_folder, root);
    assert_eq!(found.update, None);
    let mut got = names(&found);
    got.sort();
    assert_eq!(got, ["boat.md", "train.md"]);
    assert!(!search.index_path(&travel).exists());
}

/// With a reranker, the hits carry the cross-encoder's scores for the query
/// and passage, best first.
#[tokio::test(flavor = "multi_thread")]
async fn reranks_with_the_cross_encoder() {
    let cache = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let notes = dir.path().canonicalize().unwrap();
    write(&notes, "boat.md", "The ferry leaves the harbour at nine.");
    write(&notes, "bread.md", "Knead the dough and let it rise.");
    write(&notes, "bike.md", "Pump the bicycle tyres every week.");
    let search = FolderSearch::new()
        .cache_dir(cache.path())
        .reranker(RERANKER);
    let query = "what time does the boat sail";

    let found = search.search(&notes, query, 3).await.unwrap();
    assert_eq!(names(&found)[0], "boat.md");

    let reranker = crate::reranker::Reranker::builder(RERANKER)
        .quiet(true)
        .build()
        .await
        .unwrap();
    for hit in &found.results {
        let direct = reranker.rerank(query, &[hit.text.as_str()]).await.unwrap();
        // Scored alone, a pair has no padding beside it; the scores agree to rounding.
        let diff = (hit.score - direct[0].score).abs();
        assert!(
            diff <= 1e-5 * direct[0].score.abs().max(1.0),
            "{}",
            hit.source.display()
        );
    }
    assert!(found.results.windows(2).all(|w| w[0].score >= w[1].score));
}

/// A folder over the limit is refused before any model loads, naming its
/// largest subfolders with their document counts.
#[tokio::test]
async fn a_folder_with_too_many_documents_names_smaller_ones() {
    let cache = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    for i in 0..3 {
        write(&root, &format!("notes/{i}.md"), "a note");
    }
    write(&root, "code/main.rs", "fn main() {}");
    write(&root, "photos/cat.jpg", "not text");
    write(&root, "top.txt", "at the top");
    let search = FolderSearch::new().cache_dir(cache.path()).max_documents(4);

    let err = search.search(&root, "x", 1).await.unwrap_err().to_string();
    let expected = format!(
        "{} holds 5 documents, more than the 4 a search indexes at once. \
         Search a folder inside it instead; the largest are {} (3), {} (1).",
        root.display(),
        root.join("notes").display(),
        root.join("code").display()
    );
    assert_eq!(err, expected);
    assert!(!search.index_path(&root).exists());
}

#[test]
fn scan_counts_what_the_indexer_would_read() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    write(d, "a.md", "one");
    let before = scan(d, DEFAULT_MODEL);
    assert_eq!(before.documents, 1);
    assert!(before.documents_by_subfolder.is_empty());

    // Excluded and hidden folders count for nothing.
    write(d, "target/debug/x.rs", "generated");
    write(d, ".git/HEAD", "ref");
    let hidden = scan(d, DEFAULT_MODEL);
    assert_eq!(hidden.stamp, before.stamp);
    assert_eq!(hidden.documents, 1);

    // A photo changes the stamp but is no document; a note in a subfolder is both.
    write(d, "pics/cat.jpg", "jpeg");
    let photo = scan(d, DEFAULT_MODEL);
    assert_ne!(photo.stamp, before.stamp);
    assert_eq!(photo.documents, 1);
    write(d, "pics/notes/b.md", "two");
    let note = scan(d, DEFAULT_MODEL);
    assert_eq!(note.documents, 2);
    assert_eq!(note.documents_by_subfolder.get("pics"), Some(&1));

    assert_ne!(scan(d, "other-model").stamp, note.stamp);
}

#[test]
fn helpers() {
    assert!(wildcard("*.min.js", "app.min.js") && !wildcard("*.min.js", "app.js"));
    assert!(wildcard("target", "target") && !wildcard("target", "targets"));
    assert_eq!(folder_key(Path::new("/a")), folder_key(Path::new("/a")));
    assert_ne!(folder_key(Path::new("/a")), folder_key(Path::new("/b")));
    assert_eq!(within(Path::new("/home/me/notes")), "/home/me/notes/**");
    assert!(within_is_literal(Path::new(r"\\?\C:\Users\me\notes")));
    assert!(!within_is_literal(Path::new("/home/me/[draft]")));
    assert_eq!(
        IndexUpdate::Updated {
            added: 1,
            changed: 2,
            removed: 3
        }
        .describe(Path::new("/n")),
        "Updated the index of /n: 1 new, 2 changed, 3 removed"
    );
}
