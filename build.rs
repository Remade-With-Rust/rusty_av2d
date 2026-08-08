#![deny(clippy::all)]

// The decoder is pure Rust: there is no assembly to assemble and no C to compile,
// so this build script intentionally does nothing. It is kept (rather than removed
// from Cargo.toml) as the hook for any future codegen.
fn main() {}
