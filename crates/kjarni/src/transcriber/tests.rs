//! Transcriber integration tests.

#[cfg(test)]
mod tests {
    use super::super::*;

    /// A phrase that must appear in the transcription of the fixture.
    ///
    /// Asserting on content rather than `!is_empty()` is the point: a decode
    /// that silently produces fluent nonsense — the failure mode a broken
    /// kernel or a mis-loaded weight actually produces — passes an emptiness
    /// check and fails this one.
    const EXPECTED_PHRASE: &str = "ask not what your country can do for you";

    /// Absolute path to the spoken-word fixture used by the `#[ignore]`d
    /// integration tests.
    ///
    /// Resolved from CARGO_MANIFEST_DIR rather than a relative path, which would
    /// depend on the directory cargo happens to run the test binary from.
    fn speech_fixture() -> std::path::PathBuf {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/speech.wav");

        assert!(
            path.exists(),
            "missing speech fixture at {}\n\
             These tests transcribe real speech, so they need a short \
             spoken-word WAV (16 kHz mono is ideal).\n\
             Add one at that path, then re-run with `cargo test -- --ignored`.",
            path.display()
        );

        path
    }

    #[test]
    fn test_task_display() {
        assert_eq!(Task::Transcribe.to_string(), "transcribe");
        assert_eq!(Task::Translate.to_string(), "translate");
    }

    #[test]
    fn test_preset_defaults() {
        let preset = TranscriberPreset::Default;
        assert_eq!(preset.task(), Task::Transcribe);
        assert_eq!(preset.language(), None);
        assert!(!preset.timestamps());
    }

    #[test]
    fn test_preset_english() {
        let preset = TranscriberPreset::English;
        assert_eq!(preset.language(), Some("en"));
        assert_eq!(preset.task(), Task::Transcribe);
    }

    #[test]
    fn test_preset_translate() {
        let preset = TranscriberPreset::Translate;
        assert_eq!(preset.task(), Task::Translate);
    }

    #[test]
    fn test_error_display() {
        let err = TranscriberError::InvalidConfig("bad value".into());
        assert!(err.to_string().contains("bad value"));

        let err = TranscriberError::UnsupportedFormat("xyz".into());
        assert!(err.to_string().contains("xyz"));
    }

    #[test]
    fn test_progress_constructors() {
        let p = TranscriptionProgress::encoding(2, 10);
        assert_eq!(p.stage, TranscriptionStage::Encoding);
        assert_eq!(p.current, 2);
        assert_eq!(p.total, Some(10));

        let p = TranscriptionProgress::loading_audio();
        assert_eq!(p.stage, TranscriptionStage::LoadingAudio);
    }

    #[tokio::test]
    #[ignore = "requires model weights"]
    async fn test_transcribe_file() {
        let transcriber = Transcriber::builder("whisper-small")
            .cpu()
            .language("en")
            .quiet()
            .build()
            .await
            .expect("Failed to build transcriber");

        assert_eq!(transcriber.device(), kjarni_transformers::Device::Cpu);

        let result = transcriber
            .transcribe_file(speech_fixture())
            .expect("Transcription failed");

        assert!(
            result.text.to_lowercase().contains(EXPECTED_PHRASE),
            "transcription did not contain the expected phrase.\n  expected: {EXPECTED_PHRASE}\n  got: {}",
            result.text
        );
        assert_eq!(result.language, "en");
        assert!(result.duration_secs > 0.0);

        println!("Transcription: {}", result.text);
    }

    #[tokio::test]
    #[ignore = "requires model weights"]
    async fn test_transcribe_with_timestamps() {
        let transcriber = Transcriber::builder("whisper-small")
            .cpu()
            .language("en")
            .timestamps(true)
            .quiet()
            .build()
            .await
            .expect("Failed to build transcriber");

        let result = transcriber
            .transcribe_file(speech_fixture())
            .expect("Transcription failed");

        assert!(!result.segments.is_empty(), "Should have timed segments");

        let joined = result
            .segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        assert!(
            joined.contains(EXPECTED_PHRASE),
            "segments did not reconstruct the expected phrase.\n  expected: {EXPECTED_PHRASE}\n  got: {joined}"
        );

        for seg in &result.segments {
            assert!(seg.end >= seg.start, "Segment end should be >= start");
            assert!(!seg.text.is_empty(), "Segment text should not be empty");
            println!("[{:.2} -> {:.2}] {}", seg.start, seg.end, seg.text);
        }
    }

    #[tokio::test]
    #[ignore = "requires model weights"]
    async fn test_stream_file() {
        use futures::StreamExt;

        let transcriber = Transcriber::builder("whisper-small")
            .cpu()
            .language("en")
            .quiet()
            .build()
            .await
            .expect("Failed to build transcriber");

        let stream = transcriber
            .stream_file(speech_fixture())
            .await
            .expect("Stream failed");

        futures::pin_mut!(stream);

        let mut token_count = 0;
        let mut text = String::new();

        while let Some(result) = stream.next().await {
            let token = result.expect("Token error");
            if !token.is_special {
                text.push_str(&token.text);
            }
            token_count += 1;
        }

        assert!(token_count > 0, "Should have received tokens");
        assert!(
            text.to_lowercase().contains(EXPECTED_PHRASE),
            "streamed text did not contain the expected phrase.\n  expected: {EXPECTED_PHRASE}\n  got: {text}"
        );
        println!("Streamed {} tokens: {}", token_count, text);
    }

    #[tokio::test]
    #[ignore = "requires model weights"]
    async fn test_progress_callback() {
        use std::sync::{Arc, Mutex};

        let stages_seen = Arc::new(Mutex::new(Vec::new()));
        let stages_clone = stages_seen.clone();

        let transcriber = Transcriber::builder("whisper-small")
            .cpu()
            .language("en")
            .quiet()
            .on_progress(move |progress, _msg| {
                stages_clone.lock().unwrap().push(progress.stage);
            })
            .build()
            .await
            .expect("Failed to build transcriber");

        let _ = transcriber
            .transcribe_file(speech_fixture())
            .expect("Transcription failed");

        let stages = stages_seen.lock().unwrap();
        assert!(!stages.is_empty(), "Should have received progress updates");
        assert!(
            stages.contains(&TranscriptionStage::Encoding),
            "Should see Encoding stage"
        );
        assert!(
            stages.contains(&TranscriptionStage::Decoding),
            "Should see Decoding stage"
        );
    }

    /// whisper-large-v3 ships half-precision weights only. The expected text is
    /// transformers' greedy output for this clip on the same weights, as f32.
    #[tokio::test]
    #[ignore = "needs a GPU and whisper-large-v3 (3 GB) in the local model cache"]
    async fn large_v3_matches_transformers_on_both_devices() {
        const EXPECTED: &str = " And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
        let path = speech_fixture();
        let on_cpu = Transcriber::builder("whisper-large-v3")
            .language("en")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap()
            .transcribe_file(&path)
            .unwrap();
        assert_eq!(on_cpu.text, EXPECTED);

        // Streaming decodes with timestamps off, which needs `<|notimestamps|>`.
        let mut stream = Transcriber::builder("whisper-large-v3")
            .language("en")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap()
            .stream_file(&path)
            .await
            .unwrap();
        let mut streamed = String::new();
        while let Some(token) = futures::StreamExt::next(&mut stream).await {
            streamed.push_str(&token.unwrap().text);
        }
        assert_eq!(streamed, EXPECTED);

        let on_gpu = Transcriber::builder("whisper-large-v3")
            .language("en")
            .gpu()
            .quiet()
            .build()
            .await
            .unwrap()
            .transcribe_file(&path)
            .unwrap();
        assert_eq!(on_gpu.text, EXPECTED);
    }

    #[tokio::test]
    async fn a_panic_feeding_the_stream_reaches_the_reader_as_an_error() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<TranscriberResult<u32>>(1);
        let job = tokio::task::spawn_blocking(|| panic!("boom"));

        super::super::model::report_panic(job, tx).await;

        let error = rx.recv().await.expect("an error, not a closed channel");
        assert!(matches!(
            error,
            Err(TranscriberError::TranscriptionFailed(_))
        ));
    }

    /// The whole path on each device: mel, encoder, decoder, timestamps and text. They
    /// differ only in the encoder and decoder kernels, so a transcript that changes
    /// means one of them does.
    #[tokio::test]
    #[ignore = "needs a GPU and whisper-small in the local model cache"]
    async fn gpu_transcript_matches_cpu_word_for_word() {
        let path = speech_fixture();
        let on_cpu = Transcriber::builder("whisper-small")
            .language("en")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap()
            .transcribe_file(&path)
            .unwrap();
        let on_gpu = Transcriber::builder("whisper-small")
            .language("en")
            .gpu()
            .quiet()
            .build()
            .await
            .unwrap()
            .transcribe_file(&path)
            .unwrap();
        assert!(!on_cpu.text.trim().is_empty());
        assert_eq!(on_cpu.text, on_gpu.text);
    }
}
