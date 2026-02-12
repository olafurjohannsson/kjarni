//! Transcriber integration tests.

#[cfg(test)]
mod tests {
    use super::super::*;

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
            .transcribe_file("../../crates/kjarni-models/examples/hideyowife.wav")
            .expect("Transcription failed");

        assert!(!result.text.is_empty(), "Transcription should not be empty");
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
            .transcribe_file("../../crates/kjarni-models/examples/hideyowife.wav")
            .expect("Transcription failed");

        assert!(!result.segments.is_empty(), "Should have timed segments");

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
            .stream_file("../../crates/kjarni-models/examples/hideyowife.wav")
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
        assert!(!text.is_empty(), "Should have received text");
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
            .transcribe_file("../../crates/kjarni-models/examples/hideyowife.wav")
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
}