use kjarni_transformers::models::ModelType;

use super::validation::validate;
use super::*;

#[test]
fn the_default_model_is_a_token_classifier() {
    validate(ModelType::resolve(DEFAULT_MODEL).unwrap()).unwrap();
}

#[test]
fn a_sentiment_model_is_refused_with_a_reason() {
    let err = validate(ModelType::resolve("distilbert-sentiment").unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("token classification"), "{err}");
}

#[tokio::test]
async fn a_missing_folder_is_reported_as_missing() {
    let err = EntityRecognizer::from_path("/no/such/model/folder")
        .quiet(true)
        .build()
        .await
        .err()
        .unwrap();
    assert!(matches!(err, NerError::ModelPathNotFound(_)), "{err}");
}

/// Downloads distilbert-ner (261 MB) the first time. Entities are from
/// transformers' pipeline("ner", aggregation_strategy="first") on the CPU.
#[tokio::test]
#[ignore = "downloads distilbert-ner"]
async fn distilbert_ner_matches_transformers() {
    let ner = EntityRecognizer::builder(DEFAULT_MODEL)
        .quiet(true)
        .build()
        .await
        .unwrap();
    let got = ner
        .recognize("Sarah Connor met Apple's CEO in San Francisco on Tuesday.")
        .await
        .unwrap();
    let want = [
        ("PER", "Sarah Connor", 0, 12, 0.99879),
        ("ORG", "Apple", 17, 22, 0.99710),
        ("LOC", "San Francisco", 32, 45, 0.99761),
    ];
    assert_eq!(got.len(), want.len(), "{got:?}");
    for (g, (label, text, start, end, score)) in got.iter().zip(want) {
        assert_eq!(
            (g.label.as_str(), g.text.as_str(), g.start, g.end),
            (label, text, start, end)
        );
        assert!(
            (g.score - score).abs() < 1e-4,
            "{text}: {} vs {score}",
            g.score
        );
    }
}
