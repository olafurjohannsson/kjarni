//! Indexing a folder and searching it through the C interface, as the C#,
//! Go, C++ and Python bindings do.

use crate::callback::*;
use crate::indexer::*;
use crate::searcher::*;
use crate::{KjarniDevice, KjarniErrorCode};
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// The texts of `results`, in order, and frees them.
unsafe fn texts(results: KjarniSearchResults) -> Vec<String> {
    let out = if results.len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(results.results, results.len) }
            .iter()
            .map(|r| {
                unsafe { CStr::from_ptr(r.text) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    };
    unsafe { kjarni_search_results_free(results) };
    out
}

#[test]
fn null_arguments_are_refused_before_any_model_is_needed() {
    let path = c("/nowhere");
    let inputs = [path.as_ptr()];
    let mut stats: KjarniIndexStats = unsafe { std::mem::zeroed() };
    let mut added = 0usize;
    let mut results = KjarniSearchResults::empty();
    unsafe {
        assert_eq!(
            kjarni_indexer_new(ptr::null(), ptr::null_mut()),
            KjarniErrorCode::NullPointer
        );
        assert_eq!(
            kjarni_indexer_create(
                ptr::null_mut(),
                path.as_ptr(),
                inputs.as_ptr(),
                1,
                0,
                &mut stats
            ),
            KjarniErrorCode::NullPointer
        );
        assert_eq!(
            kjarni_indexer_add(
                ptr::null_mut(),
                path.as_ptr(),
                inputs.as_ptr(),
                1,
                &mut added
            ),
            KjarniErrorCode::NullPointer
        );
        assert_eq!(
            kjarni_searcher_new(ptr::null(), ptr::null_mut()),
            KjarniErrorCode::NullPointer
        );
        assert_eq!(
            kjarni_searcher_search(ptr::null_mut(), path.as_ptr(), path.as_ptr(), &mut results),
            KjarniErrorCode::NullPointer
        );
        assert_eq!(
            kjarni_search_keywords(ptr::null(), path.as_ptr(), 3, &mut results),
            KjarniErrorCode::NullPointer
        );
        assert_eq!(
            kjarni_index_delete(ptr::null()),
            KjarniErrorCode::NullPointer
        );
        assert_eq!(kjarni_indexer_dimension(ptr::null()), 0);
        assert!(!kjarni_searcher_has_reranker(ptr::null()));
        kjarni_indexer_free(ptr::null_mut());
        kjarni_searcher_free(ptr::null_mut());
    }
}

#[test]
fn defaults_are_what_the_bindings_document() {
    let config = kjarni_indexer_config_default();
    assert_eq!(config.device as i32, KjarniDevice::Cpu as i32);
    assert!(config.model_name.is_null());
    let options = kjarni_search_options_default();
    assert!(options.source_pattern.is_null());
    assert!(options.filter_key.is_null());
}

extern "C" fn count_progress(progress: KjarniProgress, user_data: *mut c_void) {
    let events = unsafe { &*(user_data as *const std::sync::Mutex<Vec<(usize, usize)>>) };
    events
        .lock()
        .unwrap()
        .push((progress.current, progress.total));
}

#[test]
#[ignore = "downloads minilm-l6-v2"]
fn index_a_folder_add_to_it_search_it_and_delete_it() {
    let docs = tempfile::tempdir().unwrap();
    std::fs::write(
        docs.path().join("refunds.txt"),
        "Refunds are processed within thirty days of the return.",
    )
    .unwrap();
    std::fs::write(
        docs.path().join("shipping.txt"),
        "Shipping is free on orders over fifty euros.",
    )
    .unwrap();
    std::fs::write(
        docs.path().join("hours.txt"),
        "Our office is closed on public holidays.",
    )
    .unwrap();
    let home = tempfile::tempdir().unwrap();
    let index = c(home.path().join("index").to_str().unwrap());
    let folder = c(docs.path().to_str().unwrap());
    let inputs = [folder.as_ptr()];
    let model = c("minilm-l6-v2");

    unsafe {
        let mut config = kjarni_indexer_config_default();
        config.model_name = model.as_ptr();
        config.quiet = 1;
        let mut indexer = ptr::null_mut();
        assert_eq!(
            kjarni_indexer_new(&config, &mut indexer),
            KjarniErrorCode::Ok
        );
        assert_eq!(kjarni_indexer_dimension(indexer), 384);
        let mut name = [0 as c_char; 64];
        let n = kjarni_indexer_model_name(indexer, name.as_mut_ptr(), name.len());
        assert!(n > 0);
        assert!(
            CStr::from_ptr(name.as_ptr())
                .to_str()
                .unwrap()
                .contains("minilm")
        );

        // A token cancelled up front stops the build.
        let token = kjarni_cancel_token_new();
        kjarni_cancel_token_cancel(token);
        let mut stats: KjarniIndexStats = std::mem::zeroed();
        let code = kjarni_indexer_create_with_callback(
            indexer,
            index.as_ptr(),
            inputs.as_ptr(),
            1,
            1,
            None,
            ptr::null_mut(),
            token,
            &mut stats,
        );
        assert_eq!(code, KjarniErrorCode::Cancelled);
        kjarni_cancel_token_free(token);

        let events: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
        let code = kjarni_indexer_create_with_callback(
            indexer,
            index.as_ptr(),
            inputs.as_ptr(),
            1,
            1,
            Some(count_progress),
            &events as *const std::sync::Mutex<Vec<(usize, usize)>> as *mut c_void,
            ptr::null(),
            &mut stats,
        );
        assert_eq!(code, KjarniErrorCode::Ok);
        assert_eq!(stats.files_processed, 3);
        assert_eq!(stats.dimension, 384);
        let recorded = events.lock().unwrap().clone();
        assert!(!recorded.is_empty(), "progress was reported");
        for (current, total) in &recorded {
            if *current > 0 {
                assert!(*total >= *current, "progress reported {current}/{total}");
            }
        }

        let mut info = std::mem::zeroed::<KjarniIndexInfo>();
        assert_eq!(
            kjarni_index_info(index.as_ptr(), &mut info),
            KjarniErrorCode::Ok
        );
        let before = info.document_count;
        assert_eq!(info.dimension, 384);
        assert!(
            CStr::from_ptr(info.embedding_model)
                .to_str()
                .unwrap()
                .contains("minilm")
        );
        kjarni_index_info_free(info);

        let more = tempfile::tempdir().unwrap();
        std::fs::write(
            more.path().join("returns.txt"),
            "Returns need the original receipt.",
        )
        .unwrap();
        let more_path = c(more.path().to_str().unwrap());
        let mut added = 0usize;
        assert_eq!(
            kjarni_indexer_add(
                indexer,
                index.as_ptr(),
                [more_path.as_ptr()].as_ptr(),
                1,
                &mut added
            ),
            KjarniErrorCode::Ok
        );
        assert_eq!(added, 1);
        let mut info = std::mem::zeroed::<KjarniIndexInfo>();
        assert_eq!(
            kjarni_index_info(index.as_ptr(), &mut info),
            KjarniErrorCode::Ok
        );
        assert_eq!(info.document_count, before + 1);
        kjarni_index_info_free(info);
        kjarni_indexer_free(indexer);

        let mut sconfig = kjarni_searcher_config_default();
        sconfig.model_name = model.as_ptr();
        sconfig.quiet = 1;
        let mut searcher = ptr::null_mut();
        assert_eq!(
            kjarni_searcher_new(&sconfig, &mut searcher),
            KjarniErrorCode::Ok
        );
        assert!(!kjarni_searcher_has_reranker(searcher));

        let query = c("How long until I get my money back?");
        let mut results = KjarniSearchResults::empty();
        assert_eq!(
            kjarni_searcher_search(searcher, index.as_ptr(), query.as_ptr(), &mut results),
            KjarniErrorCode::Ok
        );
        let semantic = texts(results);
        assert!(semantic[0].starts_with("Refunds"), "{semantic:?}");

        let mut options = kjarni_search_options_default();
        options.mode = KjarniSearchMode::Hybrid as i32;
        options.top_k = 1;
        let mut results = KjarniSearchResults::empty();
        assert_eq!(
            kjarni_searcher_search_with_options(
                searcher,
                index.as_ptr(),
                query.as_ptr(),
                &options,
                &mut results
            ),
            KjarniErrorCode::Ok
        );
        assert_eq!(texts(results).len(), 1);
        kjarni_searcher_free(searcher);

        // Keyword search needs no model.
        let word = c("receipt");
        let mut results = KjarniSearchResults::empty();
        assert_eq!(
            kjarni_search_keywords(index.as_ptr(), word.as_ptr(), 3, &mut results),
            KjarniErrorCode::Ok
        );
        assert_eq!(texts(results), ["Returns need the original receipt."]);

        assert_eq!(kjarni_index_delete(index.as_ptr()), KjarniErrorCode::Ok);
        let mut info = std::mem::zeroed::<KjarniIndexInfo>();
        assert_ne!(
            kjarni_index_info(index.as_ptr(), &mut info),
            KjarniErrorCode::Ok
        );
    }
}
