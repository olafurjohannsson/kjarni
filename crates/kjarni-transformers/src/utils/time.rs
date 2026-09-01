//! A monotonic clock that also exists in a browser.
//!
//! `std::time::Instant::now()` panics on wasm32-unknown-unknown with "time not
//! implemented on this platform": the target has no clock, and the panic only
//! fires when the timing code actually runs, so it survives compilation and
//! reaches the browser. `web_time` exposes the same API on top of
//! `performance.now()`, which leaves timing code unchanged and still measuring.
//!
//! `Duration` is arithmetic rather than a clock, so it comes from `std` on every
//! target.
pub use std::time::Duration;

#[cfg(target_arch = "wasm32")]
pub use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;
