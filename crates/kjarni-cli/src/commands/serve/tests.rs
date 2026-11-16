use std::sync::Arc;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::audio::{Format, audio_format, srt, vtt};
use super::generation::{InMessage, Sampling, messages, offered_tools};
use super::models::{Defaults, Models, resolve};
use super::stop::{StopFilter, Trim};
use super::{AppState, RouterOptions, router};

fn app_with(api_key: Option<&str>) -> Router {
    let state = Arc::new(AppState {
        models: Models::new(Defaults::default(), false, false),
    });
    router(
        state,
        RouterOptions {
            api_key: api_key.map(str::to_string),
            cors: false,
            log: false,
        },
    )
}

fn app() -> Router {
    app_with(None)
}

async fn send(app: Router, request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, body.to_vec())
}

async fn post(app: Router, path: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::post(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let (status, body) = send(app, request).await;
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

async fn get(app: Router, path: &str) -> (StatusCode, Value) {
    let (status, body) = send(app, Request::get(path).body(Body::empty()).unwrap()).await;
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

/// Feeds `pieces` through a filter and returns what came out and whether it
/// stopped.
fn filter(stops: &[&str], pieces: &[&str]) -> (String, bool) {
    let mut f = StopFilter::new(stops.iter().map(|s| s.to_string()).collect());
    let mut out = String::new();
    for piece in pieces {
        let (text, stopped) = f.push(piece);
        out.push_str(&text);
        if stopped {
            return (out, true);
        }
    }
    out.push_str(&f.finish());
    (out, false)
}

mod stop_filter {
    use super::*;

    #[test]
    fn passes_text_through_without_stops() {
        assert_eq!(
            filter(&[], &["Hello", " world"]),
            ("Hello world".into(), false)
        );
    }

    #[test]
    fn cuts_at_a_stop_inside_one_piece() {
        assert_eq!(filter(&["END"], &["one END two"]), ("one ".into(), true));
    }

    #[test]
    fn cuts_at_a_stop_split_across_pieces() {
        assert_eq!(
            filter(&["</s>"], &["done<", "/", "s>after"]),
            ("done".into(), true)
        );
    }

    #[test]
    fn holds_back_only_while_a_stop_could_still_start() {
        let mut f = StopFilter::new(vec!["STOP".into()]);
        assert_eq!(f.push("abc ST"), ("abc ".into(), false));
        // "STx" can no longer become "STOP", so it is released.
        assert_eq!(f.push("x"), ("STx".into(), false));
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn releases_a_partial_stop_when_the_text_ends() {
        assert_eq!(filter(&["STOP"], &["abc ST"]), ("abc ST".into(), false));
    }

    #[test]
    fn the_earliest_stop_wins() {
        assert_eq!(filter(&["b", "a"], &["xxab"]), ("xx".into(), true));
    }

    #[test]
    fn splits_only_on_character_boundaries() {
        assert_eq!(filter(&["ég"], &["hæ é", "g"]), ("hæ ".into(), true));
        assert_eq!(filter(&["ég"], &["hæ é", "x"]), ("hæ éx".into(), false));
    }

    #[test]
    fn ignores_empty_stops() {
        assert_eq!(filter(&[""], &["abc"]), ("abc".into(), false));
    }
}

mod trim {
    use super::*;

    /// Every way of cutting `text` in two must stream to the same result as
    /// trimming it whole.
    #[test]
    fn streaming_matches_trimming_the_whole_text() {
        for text in ["  Hello  world \n", "\n\nHi", "x", "   ", "a \n b\t", ""] {
            let chars: Vec<(usize, char)> = text.char_indices().collect();
            for &(cut, _) in chars.iter().chain([&(text.len(), ' ')]) {
                let mut trim = Trim::default();
                let mut out = trim.push(&text[..cut]);
                out.push_str(&trim.push(&text[cut..]));
                assert_eq!(out, text.trim(), "{text:?} cut at {cut}");
            }
        }
    }

    #[test]
    fn keeps_whitespace_between_pieces() {
        let mut trim = Trim::default();
        let pieces = [" The", " cat", " ", "\n", "sat", "\n\n"];
        let out: String = pieces.iter().map(|p| trim.push(p)).collect();
        assert_eq!(out, "The cat \nsat");
    }
}

mod sampling {
    use super::*;

    fn overrides(body: Value) -> Result<kjarni::generation::GenerationOverrides, String> {
        let sampling: Sampling = serde_json::from_value(body).unwrap();
        sampling.overrides().map_err(|e| e.message)
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let o = overrides(json!({ "temperature": 0 })).unwrap();
        assert_eq!(o.do_sample, Some(false));
        assert_eq!(o.temperature, None);
    }

    #[test]
    fn a_positive_temperature_samples() {
        let o = overrides(json!({ "temperature": 0.7, "top_p": 0.9 })).unwrap();
        assert_eq!(o.do_sample, Some(true));
        assert_eq!(o.temperature, Some(0.7));
        assert_eq!(o.top_p, Some(0.9));
    }

    #[test]
    fn no_temperature_keeps_the_model_default() {
        let o = overrides(json!({})).unwrap();
        assert_eq!(
            (o.do_sample, o.temperature, o.max_new_tokens),
            (None, None, None)
        );
    }

    #[test]
    fn max_completion_tokens_wins_over_max_tokens() {
        let o = overrides(json!({ "max_tokens": 10, "max_completion_tokens": 20 })).unwrap();
        assert_eq!(o.max_new_tokens, Some(20));
        let o = overrides(json!({ "max_tokens": 10 })).unwrap();
        assert_eq!(o.max_new_tokens, Some(10));
    }

    #[test]
    fn out_of_range_values_are_refused() {
        assert!(overrides(json!({ "temperature": 2.5 })).is_err());
        assert!(overrides(json!({ "temperature": -1 })).is_err());
        assert!(overrides(json!({ "top_p": 0 })).is_err());
        assert!(overrides(json!({ "max_tokens": 0 })).is_err());
    }

    #[test]
    fn unsupported_requests_are_refused_not_ignored() {
        assert!(overrides(json!({ "n": 2 })).is_err());
        assert!(overrides(json!({ "logprobs": true })).is_err());
        assert!(overrides(json!({ "logprobs": 5 })).is_err());
        assert!(overrides(json!({ "top_logprobs": 3 })).is_err());
        // What clients send by default.
        assert!(overrides(json!({ "n": 1, "logprobs": false })).is_ok());
        assert!(overrides(json!({ "logprobs": null })).is_ok());
    }
}

mod message_conversion {
    use super::*;
    use kjarni::chat::Role;

    fn convert(body: Value) -> Result<Vec<(Role, String)>, String> {
        let input: Vec<InMessage> = serde_json::from_value(body).unwrap();
        messages(input, &|c| c.to_qwen_xml())
            .map(|m| m.into_iter().map(|m| (m.role, m.content)).collect())
            .map_err(|e| e.message)
    }

    #[test]
    fn maps_roles() {
        let got = convert(json!([
            { "role": "developer", "content": "Be brief." },
            { "role": "user", "content": "Hi" },
            { "role": "assistant", "content": "Hello" },
        ]))
        .unwrap();
        let roles: Vec<String> = got.iter().map(|(r, _)| r.to_string()).collect();
        assert_eq!(roles, ["system", "user", "assistant"]);
        assert_eq!(got[0].1, "Be brief.");
    }

    #[test]
    fn joins_text_parts() {
        let got = convert(json!([{ "role": "user", "content": [
            { "type": "text", "text": "Hello " },
            { "type": "text", "text": "there" },
        ]}]))
        .unwrap();
        assert_eq!(got[0].1, "Hello there");
    }

    #[test]
    fn refuses_what_it_cannot_read() {
        let image = json!([{ "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": "data:," } },
        ]}]);
        assert!(convert(image).unwrap_err().contains("image_url"));
        let legacy = json!([{ "role": "function", "content": "42", "name": "f" }]);
        assert!(convert(legacy).unwrap_err().contains("function"));
        assert!(convert(json!([{ "role": "robot", "content": "x" }])).is_err());
        assert!(convert(json!([])).is_err());
    }
}

mod tool_messages {
    use super::*;
    use kjarni::chat::Role;

    #[test]
    fn an_assistant_call_and_its_result_become_the_model_s_own_format() {
        let input: Vec<InMessage> = serde_json::from_value(json!([
            { "role": "user", "content": "Weather in Reykjavik?" },
            { "role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": { "name": "weather", "arguments": "{\"city\": \"Reykjavik\"}" },
            }]},
            { "role": "tool", "tool_call_id": "call_1", "content": "4°C, windy" },
        ]))
        .unwrap();
        let got = messages(input, &|c| c.to_qwen_xml()).unwrap();
        assert_eq!(got[1].role, Role::Assistant);
        assert_eq!(
            got[1].content,
            "<tool_call>\n<function=weather>\n<parameter=city>\nReykjavik\n</parameter>\n</function>\n</tool_call>"
        );
        assert_eq!(
            (got[2].role, got[2].content.as_str()),
            (Role::Tool, "4°C, windy")
        );
    }

    #[test]
    fn text_before_calls_is_kept_and_calls_are_separated() {
        let input: Vec<InMessage> = serde_json::from_value(json!([
            { "role": "assistant", "content": "Checking.", "tool_calls": [
                { "function": { "name": "a", "arguments": {} } },
                { "function": { "name": "b", "arguments": {} } },
            ]},
        ]))
        .unwrap();
        let got = messages(input, &|c| format!("[{}]", c.name)).unwrap();
        assert_eq!(got[0].content, "Checking.\n\n[a]\n[b]");
    }

    #[test]
    fn tool_choice_auto_and_none() {
        let tools = vec![json!({ "type": "function", "function": {
            "name": "weather", "description": "Current weather.",
            "parameters": { "type": "object", "properties": { "city": { "type": "string" } } },
        }})];
        let auto = offered_tools(Some(&tools), None).unwrap();
        assert_eq!(auto[0].name, "weather");
        assert_eq!(auto[0].description, "Current weather.");
        assert!(
            offered_tools(Some(&tools), Some(&json!("none")))
                .unwrap()
                .is_empty()
        );
        let forced = json!({ "type": "function", "function": { "name": "weather" } });
        assert_eq!(
            offered_tools(Some(&tools), Some(&forced))
                .unwrap_err()
                .param,
            Some("tool_choice")
        );
        let nameless = vec![json!({ "type": "function", "function": {} })];
        assert_eq!(
            offered_tools(Some(&nameless), None).unwrap_err().param,
            Some("tools")
        );
    }
}

mod model_names {
    use super::*;

    #[test]
    fn no_name_means_the_default() {
        assert_eq!(resolve(None, "minilm-l6-v2").unwrap(), "minilm-l6-v2");
        assert_eq!(resolve(Some(""), "minilm-l6-v2").unwrap(), "minilm-l6-v2");
        assert_eq!(
            resolve(Some("default"), "minilm-l6-v2").unwrap(),
            "minilm-l6-v2"
        );
    }

    #[test]
    fn a_known_name_is_used() {
        assert_eq!(
            resolve(Some("minilm-zeroshot"), "x").unwrap(),
            "minilm-zeroshot"
        );
    }

    #[test]
    fn an_unknown_name_is_not_found_rather_than_the_default() {
        let e = resolve(Some("gpt-4o-mini"), "minilm-l6-v2").unwrap_err();
        assert_eq!(e.status, StatusCode::NOT_FOUND);
        assert_eq!(e.code, Some("model_not_found"));
    }

    #[test]
    fn every_default_is_a_real_model() {
        let d = Defaults::default();
        for name in [
            &d.chat,
            &d.embed,
            &d.classify,
            &d.zero_shot,
            &d.rerank,
            &d.transcribe,
            &d.summarize,
            &d.ner,
        ] {
            assert!(
                kjarni::ModelType::from_cli_name(name).is_some(),
                "default model '{name}' is not in the registry"
            );
        }
    }
}

mod audio_files {
    use super::*;
    use kjarni::transcriber::{TranscriptionResult, TranscriptionSegment};

    #[test]
    fn the_format_comes_from_the_name_or_the_bytes() {
        assert_eq!(audio_format(Some("a.MP3"), b"").unwrap(), "mp3");
        assert_eq!(audio_format(Some("blob"), b"RIFF....WAVE").unwrap(), "wav");
        assert_eq!(audio_format(None, b"fLaC....").unwrap(), "flac");
        assert_eq!(audio_format(None, b"OggS....").unwrap(), "ogg");
        assert_eq!(audio_format(None, b"ID3....").unwrap(), "mp3");
        assert_eq!(audio_format(None, &[0xFF, 0xFB, 0x90]).unwrap(), "mp3");
    }

    #[test]
    fn unsupported_formats_are_named_in_the_error() {
        let e = audio_format(Some("voice.m4a"), b"....ftypM4A").unwrap_err();
        assert!(e.message.contains("'m4a'"), "{}", e.message);
    }

    #[test]
    fn response_formats() {
        assert_eq!(Format::parse("srt").unwrap(), Format::Srt);
        assert!(Format::parse("xml").is_err());
    }

    fn result() -> TranscriptionResult {
        TranscriptionResult {
            text: " Hello there. General Kenobi.".into(),
            segments: vec![
                TranscriptionSegment {
                    start: 0.0,
                    end: 1.5,
                    text: " Hello there.".into(),
                },
                TranscriptionSegment {
                    start: 61.25,
                    end: 3725.004,
                    text: " General Kenobi.".into(),
                },
            ],
            language: "en".into(),
            duration_secs: 3726.0,
        }
    }

    #[test]
    fn srt_numbers_cues_and_uses_commas() {
        assert_eq!(
            srt(&result()),
            "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n\
             2\n00:01:01,250 --> 01:02:05,004\nGeneral Kenobi.\n\n"
        );
    }

    #[test]
    fn vtt_has_a_header_and_uses_dots() {
        assert_eq!(
            vtt(&result()),
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello there.\n\n\
             00:01:01.250 --> 01:02:05.004\nGeneral Kenobi.\n\n"
        );
    }
}

mod http {
    use super::*;

    #[tokio::test]
    async fn health_answers() {
        let (status, body) = get(app(), "/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "status": "ok" }));
    }

    #[tokio::test]
    async fn models_lists_the_registry_with_endpoints() {
        let (status, body) = get(app(), "/v1/models").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "list");
        let entry = |id: &str| {
            body["data"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == id)
                .cloned()
                .unwrap_or_else(|| panic!("{id} missing"))
        };
        assert_eq!(entry("minilm-zeroshot")["endpoint"], "classify");
        assert_eq!(entry("minilm-l6-v2")["endpoint"], "embeddings");
        assert_eq!(entry("qwen2.5-0.5b-instruct")["endpoint"], "chat");
        assert_eq!(entry("whisper-small")["endpoint"], "audio");
        assert_eq!(entry("minilm-l6-v2")["object"], "model");
        assert_eq!(entry("minilm-l6-v2")["loaded"], false);
    }

    #[tokio::test]
    async fn one_model_by_id() {
        let (status, body) = get(app(), "/v1/models/minilm-l6-v2").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["id"], "minilm-l6-v2");
        let (status, body) = get(app(), "/v1/models/nope").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");
    }

    #[tokio::test]
    async fn an_unknown_model_is_an_openai_style_404() {
        let (status, body) = post(
            app(),
            "/v1/embeddings",
            json!({ "model": "text-embedding-3-small", "input": "hi" }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "model");
        assert_eq!(body["error"]["code"], "model_not_found");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("text-embedding-3-small")
        );
    }

    #[tokio::test]
    async fn a_malformed_body_is_an_openai_style_400() {
        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from("{not json"))
            .unwrap();
        let (status, body) = send(app(), request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");

        let (status, body) = post(app(), "/v1/chat/completions", json!({ "model": "x" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("messages")
        );
    }

    /// These are refused before any model loads, so they need none.
    #[tokio::test]
    async fn unsupported_features_are_refused_up_front() {
        let user = json!([{ "role": "user", "content": "hi" }]);
        let cases = [
            (
                "/v1/chat/completions",
                json!({ "messages": user, "tools": [{ "type": "function", "function": { "name": "f" } }], "tool_choice": "required" }),
                "tool_choice",
            ),
            (
                "/v1/chat/completions",
                json!({ "messages": user, "tools": [{ "type": "function", "function": {} }] }),
                "tools",
            ),
            (
                "/v1/chat/completions",
                json!({ "messages": user, "response_format": { "type": "json_object" } }),
                "response_format",
            ),
            (
                "/v1/chat/completions",
                json!({ "messages": user, "n": 3 }),
                "n",
            ),
            (
                "/v1/completions",
                json!({ "prompt": "a", "echo": true }),
                "echo",
            ),
            ("/v1/completions", json!({ "prompt": [1, 2, 3] }), "prompt"),
            ("/v1/completions", json!({ "prompt": ["a", "b"] }), "prompt"),
            (
                "/v1/embeddings",
                json!({ "input": "a", "encoding_format": "int8" }),
                "encoding_format",
            ),
            ("/v1/embeddings", json!({ "input": [[1, 2]] }), "input"),
            (
                "/v1/classify",
                json!({ "input": "a", "multi_label": true }),
                "labels",
            ),
            (
                "/v1/classify",
                json!({ "input": "a", "labels": [] }),
                "labels",
            ),
            (
                "/v1/rerank",
                json!({ "query": "q", "documents": [] }),
                "documents",
            ),
            (
                "/v1/translate",
                json!({ "input": "Hallo", "from": "de" }),
                "from",
            ),
            ("/v1/ner", json!({ "input": [] }), "input"),
        ];
        for (path, body, param) in cases {
            let (status, got) = post(app(), path, body.clone()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path} {body}: {got}");
            assert_eq!(got["error"]["param"], param, "{path} {body}: {got}");
        }
    }

    #[tokio::test]
    async fn tools_with_tool_choice_none_are_not_refused() {
        // tool_choice "none" means the tools won't be used, so the request
        // gets as far as resolving the model.
        let (status, body) = post(
            app(),
            "/v1/chat/completions",
            json!({
                "model": "no-such-model",
                "messages": [{ "role": "user", "content": "hi" }],
                "tools": [{ "type": "function", "function": { "name": "f" } }],
                "tool_choice": "none",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    }

    #[tokio::test]
    async fn an_unknown_endpoint_is_a_404() {
        let (status, body) = post(app(), "/v1/images/generations", json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn an_api_key_guards_v1_but_not_health() {
        let guarded = || app_with(Some("secret"));
        let (status, body) = get(guarded(), "/v1/models").await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"]["code"], "invalid_api_key");

        let wrong = Request::get("/v1/models")
            .header(header::AUTHORIZATION, "Bearer secreT")
            .body(Body::empty())
            .unwrap();
        assert_eq!(send(guarded(), wrong).await.0, StatusCode::UNAUTHORIZED);

        let right = Request::get("/v1/models")
            .header(header::AUTHORIZATION, "Bearer secret")
            .body(Body::empty())
            .unwrap();
        assert_eq!(send(guarded(), right).await.0, StatusCode::OK);

        assert_eq!(get(guarded(), "/health").await.0, StatusCode::OK);
    }

    #[tokio::test]
    async fn audio_needs_a_file() {
        let boundary = "XyZ";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-small\r\n--{boundary}--\r\n"
        );
        let request = Request::post("/v1/audio/transcriptions")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        let (status, body) = send(app(), request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["param"], "file");
    }
}

/// These load real models, downloading them the first time, and check that
/// the server returns what Kjarni's library does, and where a reference
/// exists, what transformers does.
mod with_models {
    use super::*;
    use kjarni::chat::Chat;
    use kjarni::generation::GenerationOverrides;

    fn close(got: &Value, want: f64, tolerance: f64) -> bool {
        (got.as_f64().unwrap() - want).abs() < tolerance
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads minilm-l6-v2"]
    async fn embeddings_match_transformers() {
        let (status, body) = post(
            app(),
            "/v1/embeddings",
            json!({ "input": ["hello", "hello"] }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["model"], "minilm-l6-v2");
        // sentence-transformers' unnormalized vector for "hello", over its norm.
        let raw = [-0.378313, 0.331226, 0.314387, 0.517039, -0.498711];
        let norm = 6.026801;
        let got = body["data"][1]["embedding"].as_array().unwrap();
        assert_eq!(got.len(), 384);
        for (g, r) in got.iter().zip(raw) {
            assert!(close(g, r / norm, 1e-3), "{g} vs {}", r / norm);
        }
        assert_eq!(body["data"][1]["index"], 1);
        // [CLS] hello [SEP], twice.
        assert_eq!(body["usage"]["prompt_tokens"], 6);

        let (_, b64) = post(
            app(),
            "/v1/embeddings",
            json!({ "input": "hello", "encoding_format": "base64" }),
        )
        .await;
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64["data"][0]["embedding"].as_str().unwrap())
            .unwrap();
        let first = f32::from_le_bytes(bytes[..4].try_into().unwrap());
        assert!((first as f64 - got[0].as_f64().unwrap()).abs() < 1e-7);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads minilm-zeroshot"]
    async fn zero_shot_matches_transformers() {
        let (status, body) = post(
            app(),
            "/v1/classify",
            json!({
                "input": "The shoes run a full size small, I had to return them.",
                "labels": ["sizing", "shipping", "price", "quality"],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["model"], "minilm-zeroshot");
        let scores = &body["data"][0]["scores"];
        let want = [
            ("sizing", 0.644393),
            ("quality", 0.153637),
            ("price", 0.133332),
            ("shipping", 0.068638),
        ];
        for (got, (label, score)) in scores.as_array().unwrap().iter().zip(want) {
            assert_eq!(got["label"], label);
            assert!(
                close(&got["score"], score, 1e-4),
                "{label}: {}",
                got["score"]
            );
        }
        assert_eq!(body["data"][0]["label"], "sizing");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads distilbert-ner"]
    async fn ner_matches_transformers() {
        let (status, body) = post(
            app(),
            "/v1/ner",
            json!({ "input": "Sarah Connor met Apple's CEO in San Francisco on Tuesday." }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let entities = body["data"][0]["entities"].as_array().unwrap();
        let want = [
            ("PER", "Sarah Connor", 0, 12, 0.99879),
            ("ORG", "Apple", 17, 22, 0.99710),
            ("LOC", "San Francisco", 32, 45, 0.99761),
        ];
        assert_eq!(entities.len(), want.len());
        for (e, (label, text, start, end, score)) in entities.iter().zip(want) {
            assert_eq!(
                (e["label"].as_str(), e["text"].as_str()),
                (Some(label), Some(text))
            );
            assert_eq!(
                (e["start"].as_u64(), e["end"].as_u64()),
                (Some(start), Some(end))
            );
            assert!(close(&e["score"], score, 1e-4));
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads minilm-l6-v2-cross-encoder"]
    async fn rerank_matches_the_library() {
        let documents = [
            "Paris is the capital of France.",
            "Bananas are yellow.",
            "France's capital city is Paris, on the Seine.",
        ];
        let (status, body) = post(
            app(),
            "/v1/rerank",
            json!({ "query": "What is the capital of France?", "documents": documents, "top_n": 2 }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let results = body["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);

        let reranker = kjarni::reranker::RerankerBuilder::new("minilm-l6-v2-cross-encoder")
            .quiet(true)
            .build()
            .await
            .unwrap();
        let want = reranker
            .rerank("What is the capital of France?", &documents)
            .await
            .unwrap();
        for (got, want) in results.iter().zip(&want) {
            assert_eq!(got["index"], want.index);
            assert!(close(&got["logit"], want.score as f64, 1e-5));
            assert!(close(
                &got["relevance_score"],
                want.probability() as f64,
                1e-6
            ));
            assert_eq!(got["document"]["text"], want.document);
        }
    }

    const CHAT_MODEL: &str = "qwen2.5-0.5b-instruct";

    fn chat_body(stream: bool) -> Value {
        json!({
            "model": CHAT_MODEL,
            "messages": [
                { "role": "system", "content": "Answer in one short sentence." },
                { "role": "user", "content": "What is the capital of France?" },
            ],
            "temperature": 0,
            "max_tokens": 24,
            "stream": stream,
            "stream_options": { "include_usage": true },
        })
    }

    /// The data lines of a server-sent event stream.
    fn sse_data(body: &[u8]) -> Vec<String> {
        String::from_utf8(body.to_vec())
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("data: ").map(str::to_string))
            .collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads qwen2.5-0.5b-instruct"]
    async fn chat_matches_the_library_streamed_or_not() {
        let app = app();
        let (status, whole) = post(app.clone(), "/v1/chat/completions", chat_body(false)).await;
        assert_eq!(status, StatusCode::OK, "{whole}");
        assert_eq!(whole["object"], "chat.completion");
        assert_eq!(whole["choices"][0]["message"]["role"], "assistant");
        let text = whole["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();

        // The library, given the same conversation and settings.
        let chat = Chat::builder(CHAT_MODEL).quiet().build().await.unwrap();
        let prompt = chat
            .render_messages(&[
                kjarni::chat::Message::system("Answer in one short sentence."),
                kjarni::chat::Message::user("What is the capital of France?"),
            ])
            .unwrap();
        let overrides = GenerationOverrides {
            do_sample: Some(false),
            max_new_tokens: Some(24),
            ..Default::default()
        };
        let want = chat.generate(&prompt, &overrides).await.unwrap();
        assert_eq!(text, want);
        assert!(text.contains("Paris"), "{text}");
        let usage = &whole["usage"];
        // transformers' apply_chat_template gives 26 tokens for this conversation.
        assert_eq!(usage["prompt_tokens"], 26);
        assert_eq!(chat.generator().count_tokens(&prompt).unwrap(), 26);
        assert_eq!(
            usage["total_tokens"].as_u64().unwrap(),
            usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap()
        );

        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(chat_body(true).to_string()))
            .unwrap();
        let (status, body) = send(app, request).await;
        assert_eq!(status, StatusCode::OK);
        let events = sse_data(&body);
        assert_eq!(events.last().unwrap(), "[DONE]");
        let chunks: Vec<Value> = events[..events.len() - 1]
            .iter()
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let streamed: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(streamed, text);
        let finish = &chunks[chunks.len() - 2];
        assert_eq!(
            finish["choices"][0]["finish_reason"],
            whole["choices"][0]["finish_reason"]
        );
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"], json!([]));
        assert_eq!(last["usage"], whole["usage"]);
    }

    fn weather_request(messages: Value, stream: bool) -> Value {
        json!({
            "model": "qwen3.5-0.8b",
            "messages": messages,
            "tools": [{ "type": "function", "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city.",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string", "description": "The city name." } },
                    "required": ["city"],
                },
            }}],
            "temperature": 0,
            "max_tokens": 80,
            "stream": stream,
        })
    }

    /// A whole tool round trip through the API, as an OpenAI client makes it:
    /// the model calls the tool, the client runs it and sends the result, and
    /// the model answers from it. Streamed and not, the call is the same.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads qwen3.5-0.8b"]
    async fn a_tool_round_trip() {
        let app = app();
        let ask =
            json!([{ "role": "user", "content": "What is the weather in Reykjavik right now?" }]);
        let (status, first) = post(
            app.clone(),
            "/v1/chat/completions",
            weather_request(ask.clone(), false),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{first}");
        let choice = &first["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "{first}");
        let call = &choice["message"]["tool_calls"][0];
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        let args: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({ "city": "Reykjavik" }));

        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(weather_request(ask.clone(), true).to_string()))
            .unwrap();
        let (status, body) = send(app.clone(), request).await;
        assert_eq!(status, StatusCode::OK);
        let chunks: Vec<Value> = sse_data(&body)
            .iter()
            .filter(|e| *e != "[DONE]")
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        let streamed_call = chunks
            .iter()
            .find_map(|c| c["choices"][0]["delta"]["tool_calls"].get(0).cloned())
            .expect("a tool_calls delta");
        assert_eq!(streamed_call["index"], 0);
        assert_eq!(streamed_call["function"], call["function"]);
        // No markup leaks into the streamed text.
        let text: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert!(!text.contains("<tool_call>"), "{text}");
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );

        let mut history = ask.as_array().unwrap().clone();
        history.push(choice["message"].clone());
        history.push(json!({ "role": "tool", "tool_call_id": call["id"], "content": "{\"temperature_c\": -3, \"sky\": \"snowing\"}" }));
        let (status, answer) = post(
            app,
            "/v1/chat/completions",
            weather_request(json!(history), false),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{answer}");
        assert_eq!(answer["choices"][0]["finish_reason"], "stop", "{answer}");
        let text = answer["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(text.contains("-3") || text.contains("snow"), "{text}");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads qwen2.5-0.5b-instruct"]
    async fn length_limits_and_stops_set_the_finish_reason() {
        let app = app();
        let ask = |extra: Value| {
            let mut body = json!({
                "model": CHAT_MODEL,
                "messages": [{ "role": "user", "content": "Count from 1 to 20, separated by spaces." }],
                "temperature": 0,
            });
            for (k, v) in extra.as_object().unwrap() {
                body[k] = v.clone();
            }
            body
        };
        let (_, short) = post(
            app.clone(),
            "/v1/chat/completions",
            ask(json!({ "max_tokens": 3 })),
        )
        .await;
        assert_eq!(short["choices"][0]["finish_reason"], "length", "{short}");
        assert_eq!(short["usage"]["completion_tokens"], 3);

        let (_, full) = post(
            app.clone(),
            "/v1/chat/completions",
            ask(json!({ "max_tokens": 80 })),
        )
        .await;
        let full_text = full["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(full_text.contains(" 5 "), "{full_text}");

        let (_, stopped) = post(
            app,
            "/v1/chat/completions",
            ask(json!({ "max_tokens": 80, "stop": [" 5"] })),
        )
        .await;
        let text = stopped["choices"][0]["message"]["content"]
            .as_str()
            .unwrap();
        assert_eq!(stopped["choices"][0]["finish_reason"], "stop");
        assert_eq!(text, &full_text[..full_text.find(" 5").unwrap()]);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads qwen2.5-0.5b-instruct"]
    async fn completions_continue_raw_text() {
        let (status, body) = post(
            app(),
            "/v1/completions",
            json!({ "model": CHAT_MODEL, "prompt": "The capital of France is", "max_tokens": 5, "temperature": 0 }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["object"], "text_completion");

        let chat = Chat::builder(CHAT_MODEL).quiet().build().await.unwrap();
        let overrides = GenerationOverrides {
            do_sample: Some(false),
            max_new_tokens: Some(5),
            ..Default::default()
        };
        let want = chat
            .generator()
            .generate_with_config("The capital of France is", &overrides)
            .await
            .unwrap();
        assert_eq!(body["choices"][0]["text"], want);
        assert_eq!(body["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "downloads whisper-small"]
    async fn transcription_matches_the_library() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../kjarni/tests/fixtures/speech.wav");
        let audio = std::fs::read(&path).unwrap();
        let app = app();
        let upload = |format: &str| {
            let boundary = "kjarniboundary";
            let mut body = Vec::new();
            for (name, value) in [("model", "whisper-small"), ("response_format", format)] {
                body.extend(
                    format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                        .as_bytes(),
                );
            }
            body.extend(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\n\
                     Content-Type: audio/wav\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend(&audio);
            body.extend(format!("\r\n--{boundary}--\r\n").as_bytes());
            Request::post("/v1/audio/transcriptions")
                .header(
                    header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap()
        };

        let (status, body) = send(app.clone(), upload("json")).await;
        assert_eq!(status, StatusCode::OK);
        let body: Value = serde_json::from_slice(&body).unwrap();

        let transcriber = kjarni::transcriber::Transcriber::builder("whisper-small")
            .quiet()
            .build()
            .await
            .unwrap();
        let want = transcriber.transcribe_file(&path).unwrap();
        assert_eq!(body["text"], want.text.trim());

        let (status, srt) = send(app, upload("srt")).await;
        assert_eq!(status, StatusCode::OK);
        let srt = String::from_utf8(srt).unwrap();
        assert!(srt.starts_with("1\n00:00:"), "{srt}");
    }
}
