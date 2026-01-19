# Install
rustup +nightly component add miri

# Run tests under Miri
cargo +nightly miri test

# With leak detection
MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-ignore-leaks" cargo +nightly miri test