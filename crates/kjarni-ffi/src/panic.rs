//! The panic barrier for the C ABI.
//!
//! Every `extern "C"` entry point in this crate runs its body inside [`guard`].
//!
//! A Rust panic must not cross an `extern "C"` frame. Since Rust 1.71 the compiler
//! inserts an abort shim to enforce that, so an unguarded panic in the engine does
//! not surface to the caller as an error — it terminates the process hosting the
//! library. For a `.so` loaded into a .NET service or a Go binary that is the worst
//! possible failure mode: no error code, no message, no stack, and nothing the
//! caller could have done differently.
//!
//! [`guard`] converts that into [`KjarniErrorCode::Panic`] plus a message on
//! [`crate::kjarni_last_error_message`], which is what every other failure in this
//! crate already looks like from C.
//!
//! This is a backstop, not a licence to panic. A panic reaching here is a bug in
//! Kjarni; the barrier only decides whether the caller gets to hear about it.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::error::set_last_error;

// The barrier only exists if unwinding does. Under `panic = "abort"` every
// `catch_unwind` below is dead code and the first panic in the engine takes the
// host process with it -- silently, which is how the workspace release profile
// carried that setting for as long as it did. Refuse to build instead.
#[cfg(panic = "abort")]
compile_error!(
    "kjarni-ffi must be built with panic=unwind. It is a cdylib loaded into a host \
     process, and `panic = \"abort\"` turns every recoverable engine panic into a \
     hard termination of that process. Remove `panic = \"abort\"` from the profile \
     used to build this crate."
);

/// Recover the message from a panic payload.
///
/// `panic!("...")` with no formatting yields a `&'static str`; with formatting it
/// yields a `String`. Anything else (a `panic_any` from a dependency) has no
/// printable form, so it is named rather than rendered.
fn payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Run `body`, returning `fallback` if it panics.
///
/// `func` names the entry point and appears in the recorded error message; the
/// caller sees only an error code, so the name is the only way to locate the frame
/// the panic came from.
///
/// # Unwind safety
///
/// The closure is wrapped in [`AssertUnwindSafe`] because every entry point
/// captures raw pointers from C, which are never `UnwindSafe`. The assertion holds
/// for the reason `UnwindSafe` exists: nothing observes a half-updated value across
/// the boundary. Out-parameters are the caller's memory and are documented as
/// undefined on a non-`Ok` return, and a handle whose interior was left inconsistent
/// by a panic is only reachable again through another entry point, which is itself
/// guarded. The one thing this does *not* promise is that an operation interrupted
/// mid-flight had no side effects — a panic partway through `kjarni_indexer_create`
/// can leave a partial index on disk, the same as a process kill would.
pub(crate) fn guard<R>(func: &'static str, fallback: R, body: impl FnOnce() -> R) -> R {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            set_last_error(format!("panic in {func}: {}", payload_message(&*payload)));
            fallback
        }
    }
}

#[cfg(test)]
mod panic_guard_tests {
    use super::*;
    use crate::{KjarniErrorCode, kjarni_clear_error, kjarni_last_error_message};
    use std::ffi::CStr;
    use std::sync::Mutex;

    /// The default hook prints the panic and its backtrace to stderr. That is wanted
    /// in production — the barrier hides the panic from the caller, not from the
    /// operator — but it makes the test output look like a failure, so the tests
    /// that panic on purpose silence it for their duration.
    ///
    /// The hook is process-global while these tests run in parallel, so the swap is
    /// serialised. Without the lock two tests can interleave their take/set pairs and
    /// leave the silent hook installed permanently, which would swallow the output of
    /// every genuinely failing test in the binary.
    fn without_panic_output<T>(f: impl FnOnce() -> T) -> T {
        static HOOK: Mutex<()> = Mutex::new(());
        // A panicking test would poison the lock for every test after it, and the
        // panics here are the point, so the guard is taken either way.
        let _lock = HOOK.lock().unwrap_or_else(|e| e.into_inner());

        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = f();
        std::panic::set_hook(previous);
        result
    }

    fn last_error() -> Option<String> {
        let ptr = kjarni_last_error_message();
        if ptr.is_null() {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    #[test]
    fn returns_the_body_value_when_nothing_panics() {
        assert_eq!(guard("t", 0usize, || 42usize), 42);
    }

    #[test]
    fn does_not_touch_the_error_state_on_success() {
        kjarni_clear_error();
        guard("t", KjarniErrorCode::Panic, || KjarniErrorCode::Ok);
        assert_eq!(last_error(), None);
    }

    #[test]
    fn returns_the_fallback_when_the_body_panics() {
        let code = without_panic_output(|| {
            guard("t", KjarniErrorCode::Panic, || -> KjarniErrorCode {
                panic!("boom")
            })
        });
        assert_eq!(code, KjarniErrorCode::Panic);
    }

    #[test]
    fn records_a_static_str_payload() {
        kjarni_clear_error();
        without_panic_output(|| guard("some_entry_point", (), || panic!("boom")));

        let msg = last_error().expect("a panic must leave a message");
        assert!(msg.contains("some_entry_point"), "{msg}");
        assert!(msg.contains("boom"), "{msg}");
    }

    #[test]
    fn records_a_formatted_string_payload() {
        kjarni_clear_error();
        let n = 7;
        without_panic_output(|| guard("t", (), || panic!("index {n} out of range")));

        assert!(
            last_error().is_some_and(|m| m.contains("index 7 out of range")),
            "formatted payloads arrive as String, not &'static str"
        );
    }

    #[test]
    fn records_a_payload_that_is_not_a_string() {
        kjarni_clear_error();
        without_panic_output(|| guard("t", (), || std::panic::panic_any(9u32)));

        assert!(
            last_error().is_some_and(|m| m.contains("non-string panic payload")),
            "an unprintable payload must still leave a message"
        );
    }

    /// The message is the only diagnostic a C caller gets, so it has to survive the
    /// trip through `CString` — which fails on an interior nul and would otherwise
    /// leave `kjarni_last_error_message` returning the *previous* error.
    #[test]
    fn a_message_is_recorded_even_when_the_payload_contains_a_nul() {
        kjarni_clear_error();
        without_panic_output(|| guard("t", (), || panic!("before\0after")));

        assert!(
            last_error().is_some(),
            "a nul in the payload lost the message"
        );
    }

    /// The unwrap that started all this: the real shape of the bug the barrier is
    /// here to contain.
    #[test]
    fn contains_an_unwrap_on_none() {
        let dim = without_panic_output(|| {
            guard("kjarni_embedder_dim", 0usize, || {
                let empty: Option<usize> = None;
                #[allow(
                    clippy::unnecessary_literal_unwrap,
                    reason = "the unwrap is the bug under test"
                )]
                empty.unwrap()
            })
        });
        assert_eq!(dim, 0);
    }

    /// The unit tests above call `guard` directly. This one goes through a real
    /// `extern "C"` frame, shaped exactly like the 66 generated ones, because that
    /// is where the abort shim lives: a panic escaping an `extern "C"` function
    /// aborts rather than propagating, so the guard has to sit *inside* the frame.
    /// If a future refactor moves it outside, this test dies with the process.
    #[unsafe(no_mangle)]
    extern "C" fn kjarni_test_entry_point_that_panics(out: *mut usize) -> KjarniErrorCode {
        guard(
            "kjarni_test_entry_point_that_panics",
            KjarniErrorCode::Panic,
            || -> KjarniErrorCode {
                if out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                panic!("engine fell over");
            },
        )
    }

    #[test]
    fn a_panic_inside_an_extern_c_frame_returns_instead_of_aborting() {
        kjarni_clear_error();
        let mut out: usize = 0;

        let code = without_panic_output(|| kjarni_test_entry_point_that_panics(&mut out));

        assert_eq!(code, KjarniErrorCode::Panic);
        assert!(
            last_error().is_some_and(|m| m.contains("engine fell over")),
            "the caller must be able to find out what happened"
        );
    }

    /// A guarded entry point still has to behave normally on the paths that do not
    /// panic -- the barrier must not swallow ordinary error codes.
    #[test]
    fn ordinary_error_codes_pass_through_the_guard_unchanged() {
        let code = kjarni_test_entry_point_that_panics(std::ptr::null_mut());
        assert_eq!(code, KjarniErrorCode::NullPointer);
    }

    #[test]
    fn guards_nested_inside_one_another_each_catch_their_own() {
        let outer = without_panic_output(|| {
            guard("outer", -1i32, || {
                let inner = guard("inner", -2i32, || -> i32 { panic!("inner boom") });
                assert_eq!(inner, -2, "the inner guard must absorb its own panic");
                inner + 10
            })
        });
        assert_eq!(outer, 8, "the outer guard must not have been triggered");
    }
}
