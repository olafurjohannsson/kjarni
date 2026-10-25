//! Speech-to-text (Whisper) over the C ABI.
//!
//! A handle owns a loaded model. Audio comes in as a file path, or as raw mono
//! PCM samples for a host that captures audio itself, such as a microphone in a
//! game or a desktop app. The language is detected from the audio unless one is
//! given.

use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime, opt_str};
use kjarni::transcriber::{Transcriber, TranscriberError, TranscriptionResult};
use std::ffi::{CStr, CString, c_char};
use std::ptr;

/// Configuration for a Transcriber.
#[repr(C)]
pub struct KjarniTranscriberConfig {
    pub device: KjarniDevice,
    /// Model name, e.g. "whisper-small" (the default) or "whisper-large-v3".
    pub model_name: *const c_char,
    /// Spoken language as an ISO code such as "en" or "is". Null detects it from
    /// the audio.
    pub language: *const c_char,
    /// Non-zero translates the speech into English instead of transcribing it.
    pub translate: i32,
    /// Non-zero suppresses progress output.
    pub quiet: i32,
}

/// Get default transcriber configuration.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_transcriber_config_default() -> KjarniTranscriberConfig {
    KjarniTranscriberConfig {
        device: KjarniDevice::Cpu,
        model_name: ptr::null(),
        language: ptr::null(),
        translate: 0,
        quiet: 0,
    }
}

/// Opaque handle to a Transcriber.
pub struct KjarniTranscriber {
    inner: Transcriber,
}

/// One timed piece of a transcript. `text` is owned by the enclosing
/// [`KjarniTranscription`].
#[repr(C)]
pub struct KjarniTranscriptionSegment {
    /// Start, in seconds from the beginning of the audio.
    pub start: f32,
    /// End, in seconds from the beginning of the audio.
    pub end: f32,
    pub text: *mut c_char,
}

/// A transcript. Free with [`kjarni_transcription_free`].
#[repr(C)]
pub struct KjarniTranscription {
    /// The whole transcript.
    pub text: *mut c_char,
    /// The language transcribed, as an ISO code: the one given, or the one
    /// detected.
    pub language: *mut c_char,
    pub duration_secs: f32,
    pub segments: *mut KjarniTranscriptionSegment,
    pub len: usize,
}

impl KjarniTranscription {
    fn empty() -> Self {
        Self {
            text: ptr::null_mut(),
            language: ptr::null_mut(),
            duration_secs: 0.0,
            segments: ptr::null_mut(),
            len: 0,
        }
    }

    /// Text with an interior NUL cannot cross as a C string; such characters are
    /// dropped rather than truncating everything after them.
    fn c_string(s: &str) -> *mut c_char {
        CString::new(s.replace('\0', ""))
            .expect("NULs removed")
            .into_raw()
    }

    fn from_result(result: TranscriptionResult) -> Self {
        let mut segments: Vec<KjarniTranscriptionSegment> = result
            .segments
            .iter()
            .map(|s| KjarniTranscriptionSegment {
                start: s.start,
                end: s.end,
                text: Self::c_string(&s.text),
            })
            .collect();
        segments.shrink_to_fit();
        let len = segments.len();
        let segments_ptr = if len == 0 {
            ptr::null_mut()
        } else {
            Box::into_raw(segments.into_boxed_slice()) as *mut KjarniTranscriptionSegment
        };
        Self {
            text: Self::c_string(&result.text),
            language: Self::c_string(&result.language),
            duration_secs: result.duration_secs,
            segments: segments_ptr,
            len,
        }
    }
}

/// Free a transcript and every string in it.
///
/// # Safety
///
/// - `transcription` must be a value written by a transcribe call, passed here
///   exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_transcription_free(transcription: KjarniTranscription) {
    crate::panic::guard("kjarni_transcription_free", (), || unsafe {
        if !transcription.text.is_null() {
            let _ = CString::from_raw(transcription.text);
        }
        if !transcription.language.is_null() {
            let _ = CString::from_raw(transcription.language);
        }
        if !transcription.segments.is_null() && transcription.len > 0 {
            let slice = std::slice::from_raw_parts_mut(transcription.segments, transcription.len);
            for segment in slice.iter_mut() {
                if !segment.text.is_null() {
                    let _ = CString::from_raw(segment.text);
                }
            }
            let _ = Box::from_raw(slice as *mut [KjarniTranscriptionSegment]);
        }
    })
}

fn error_code(e: &TranscriberError) -> KjarniErrorCode {
    match e {
        TranscriberError::ModelLoadFailed(_) => KjarniErrorCode::LoadFailed,
        TranscriberError::InvalidAudioPath(_)
        | TranscriberError::UnsupportedFormat(_)
        | TranscriberError::InvalidConfig(_) => KjarniErrorCode::InvalidConfig,
        TranscriberError::AudioLoadFailed(_) | TranscriberError::TranscriptionFailed(_) => {
            KjarniErrorCode::InferenceFailed
        }
        TranscriberError::GpuUnavailable => KjarniErrorCode::GpuUnavailable,
        TranscriberError::Cancelled => KjarniErrorCode::Cancelled,
    }
}

/// Create a new Transcriber. Downloads the model on first use.
///
/// # Safety
///
/// - `config` must be null (defaults) or point to a valid config whose strings
///   are null or nul-terminated.
/// - `out` must be a valid, writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_transcriber_new(
    config: *const KjarniTranscriberConfig,
    out: *mut *mut KjarniTranscriber,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_transcriber_new",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                let default_config = kjarni_transcriber_config_default();
                let config = if config.is_null() {
                    &default_config
                } else {
                    &*config
                };
                let model = match opt_str(config.model_name) {
                    Ok(m) => m.unwrap_or("whisper-small"),
                    Err(e) => return e,
                };
                let language = match opt_str(config.language) {
                    Ok(l) => l,
                    Err(e) => return e,
                };

                let mut builder = Transcriber::builder(model).timestamps(true);
                builder = match config.device {
                    KjarniDevice::Gpu => builder.gpu(),
                    KjarniDevice::Cpu => builder.cpu(),
                };
                if let Some(lang) = language {
                    builder = builder.language(lang);
                }
                if config.translate != 0 {
                    builder = builder.translate();
                }
                if config.quiet != 0 {
                    builder = builder.quiet();
                }

                match get_runtime().block_on(builder.build()) {
                    Ok(inner) => {
                        *out = Box::into_raw(Box::new(KjarniTranscriber { inner }));
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        error_code(&e)
                    }
                }
            }
        },
    )
}

/// Free a Transcriber.
///
/// # Safety
///
/// - `transcriber` must be null, or a handle from `kjarni_transcriber_new` not
///   already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_transcriber_free(transcriber: *mut KjarniTranscriber) {
    crate::panic::guard("kjarni_transcriber_free", (), || unsafe {
        if !transcriber.is_null() {
            let _ = Box::from_raw(transcriber);
        }
    })
}

/// Transcribe an audio file (wav, mp3, flac or ogg).
///
/// # Safety
///
/// - `transcriber` must be a live handle; `path` a nul-terminated string.
/// - `out` must be a valid, writable pointer. It is written on success and on
///   failure; free it with `kjarni_transcription_free` either way.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_transcriber_transcribe_file(
    transcriber: *const KjarniTranscriber,
    path: *const c_char,
    out: *mut KjarniTranscription,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_transcriber_transcribe_file",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if transcriber.is_null() || path.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                *out = KjarniTranscription::empty();
                let path = match CStr::from_ptr(path).to_str() {
                    Ok(p) => p,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };
                match (*transcriber).inner.transcribe_file(path) {
                    Ok(result) => {
                        *out = KjarniTranscription::from_result(result);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        error_code(&e)
                    }
                }
            }
        },
    )
}

/// Transcribe raw mono PCM samples in -1..1 at any sample rate; they are
/// resampled to 16 kHz.
///
/// # Safety
///
/// - `transcriber` must be a live handle.
/// - `samples` must point to `len` readable floats (it may be null when `len` is 0).
/// - `out` must be a valid, writable pointer; free it with
///   `kjarni_transcription_free` either way.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_transcriber_transcribe_pcm(
    transcriber: *const KjarniTranscriber,
    samples: *const f32,
    len: usize,
    sample_rate: u32,
    out: *mut KjarniTranscription,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_transcriber_transcribe_pcm",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if transcriber.is_null() || out.is_null() || (samples.is_null() && len > 0) {
                    return KjarniErrorCode::NullPointer;
                }
                *out = KjarniTranscription::empty();
                if sample_rate == 0 {
                    set_last_error("sample_rate must be greater than zero".to_string());
                    return KjarniErrorCode::InvalidConfig;
                }
                let samples: &[f32] = if len == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(samples, len)
                };
                match (*transcriber).inner.transcribe_audio(samples, sample_rate) {
                    Ok(result) => {
                        *out = KjarniTranscription::from_result(result);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        error_code(&e)
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_detect_the_language() {
        let c = kjarni_transcriber_config_default();
        assert!(c.model_name.is_null() && c.language.is_null());
        assert_eq!((c.translate, c.quiet), (0, 0));
    }

    #[test]
    fn null_arguments_are_errors_not_crashes() {
        unsafe {
            assert_eq!(
                kjarni_transcriber_new(ptr::null(), ptr::null_mut()),
                KjarniErrorCode::NullPointer
            );
            let mut out = KjarniTranscription::empty();
            assert_eq!(
                kjarni_transcriber_transcribe_file(ptr::null(), ptr::null(), &mut out),
                KjarniErrorCode::NullPointer
            );
            assert_eq!(
                kjarni_transcriber_transcribe_pcm(ptr::null(), ptr::null(), 0, 16000, &mut out),
                KjarniErrorCode::NullPointer
            );
            kjarni_transcriber_free(ptr::null_mut());
            kjarni_transcription_free(KjarniTranscription::empty());
        }
    }

    /// Every string in a result is freed exactly once, including segments.
    #[test]
    fn a_transcription_round_trips_through_its_free() {
        let result = TranscriptionResult {
            text: "Hello there.".into(),
            segments: vec![
                kjarni::transcriber::TranscriptionSegment {
                    start: 0.0,
                    end: 1.5,
                    text: "Hello".into(),
                },
                kjarni::transcriber::TranscriptionSegment {
                    start: 1.5,
                    end: 2.0,
                    text: "the\0re.".into(),
                },
            ],
            language: "en".into(),
            duration_secs: 2.0,
        };
        let c = KjarniTranscription::from_result(result);
        unsafe {
            assert_eq!(CStr::from_ptr(c.text).to_str().unwrap(), "Hello there.");
            assert_eq!(CStr::from_ptr(c.language).to_str().unwrap(), "en");
            assert_eq!(c.len, 2);
            let second = &*c.segments.add(1);
            assert_eq!(second.start, 1.5);
            assert_eq!(CStr::from_ptr(second.text).to_str().unwrap(), "there.");
            kjarni_transcription_free(c);
        }
    }

    /// The real model, when it is cached: a second of silence transcribes without
    /// error and reports its duration.
    #[test]
    fn transcribes_pcm_when_the_model_is_present() {
        if !kjarni::registry::is_model_downloaded("whisper-small").unwrap_or(false) {
            eprintln!("skipping: whisper-small is not downloaded");
            return;
        }
        unsafe {
            let mut config = kjarni_transcriber_config_default();
            config.quiet = 1;
            config.language = c"en".as_ptr();
            let mut t = ptr::null_mut();
            assert_eq!(kjarni_transcriber_new(&config, &mut t), KjarniErrorCode::Ok);
            let silence = vec![0.0f32; 16000];
            let mut out = KjarniTranscription::empty();
            let code =
                kjarni_transcriber_transcribe_pcm(t, silence.as_ptr(), silence.len(), 16000, &mut out);
            assert_eq!(code, KjarniErrorCode::Ok);
            assert!((out.duration_secs - 1.0).abs() < 0.01);
            assert_eq!(CStr::from_ptr(out.language).to_str().unwrap(), "en");
            kjarni_transcription_free(out);
            kjarni_transcriber_free(t);
        }
    }
}
