//! The diagnostic environment switches (`MMVP`, `MLEAF`, `ATDBG`, ...), read once.
//!
//! Probes sit on per-block and per-sample paths, and each `std::env::var` is an OS environment
//! lookup under a lock. The decoder never changes its environment, so a snapshot taken at the
//! first read answers every later one from memory.

use std::collections::HashMap;
use std::sync::OnceLock;

fn snapshot() -> &'static HashMap<String, String> {
    static ENV: OnceLock<HashMap<String, String>> = OnceLock::new();
    // Like `std::env::var`, a name or value that is not valid Unicode reads as unset.
    ENV.get_or_init(|| std::env::vars_os().filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?))).collect())
}

/// Whether `name` is set: `std::env::var(name).is_ok()`.
#[inline]
pub(crate) fn is_set(name: &str) -> bool {
    snapshot().contains_key(name)
}

/// The value of `name`, if set: `std::env::var(name).ok()` without the allocation.
#[inline]
pub(crate) fn get(name: &str) -> Option<&'static str> {
    snapshot().get(name).map(String::as_str)
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_the_environment() {
        // PATH is set on every host the tests run on; this name is not.
        assert_eq!(super::is_set("PATH"), std::env::var("PATH").is_ok());
        assert_eq!(super::get("PATH"), std::env::var("PATH").ok().as_deref());
        assert!(!super::is_set("RUSTY_AV2D_SURELY_UNSET_7f3a"));
    }
}
