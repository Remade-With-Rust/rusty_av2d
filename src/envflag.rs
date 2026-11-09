//! The diagnostic environment switches (`MMVP`, `MLEAF`, `ATDBG`, ...), read once.
//!
//! Probes sit on per-block and per-sample paths. Each `std::env::var` is an OS environment
//! lookup under a lock, so the environment is snapshotted at the first read. Each call site then
//! caches its own answer: [`env_is_set!`](crate::env_is_set) and [`env_get!`](crate::env_get)
//! cost one atomic load after their first call, with no hashing on the hot path.

use std::sync::OnceLock;

/// An environment variable name as the OS compares it: Windows names are case-insensitive
/// (`std::env::var("PATH")` finds `Path`), elsewhere they are exact.
fn key(name: &str) -> std::borrow::Cow<'_, str> {
    if cfg!(windows) {
        std::borrow::Cow::Owned(name.to_uppercase())
    } else {
        std::borrow::Cow::Borrowed(name)
    }
}

/// The environment, sorted by (OS-compared) name. A sorted `Vec` rather than a `HashMap`: lookups
/// happen once per call site, and a randomly seeded hash made the decoder's instruction count vary
/// run to run.
fn snapshot() -> &'static [(String, String)] {
    static ENV: OnceLock<Vec<(String, String)>> = OnceLock::new();
    ENV.get_or_init(|| {
        // Like `std::env::var`, a name or value that is not valid Unicode reads as unset.
        let mut v: Vec<(String, String)> = std::env::vars_os()
            .filter_map(|(k, v)| Some((key(&k.into_string().ok()?).into_owned(), v.into_string().ok()?)))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    })
}

fn find(name: &str) -> Option<&'static str> {
    let env = snapshot();
    let name = key(name);
    env.binary_search_by(|(k, _)| k.as_str().cmp(&name)).ok().map(|i| env[i].1.as_str())
}

/// Whether `name` is set: `std::env::var(name).is_ok()`. Use [`env_is_set!`](crate::env_is_set)
/// on any path that runs more than once.
#[inline(never)]
pub(crate) fn is_set(name: &str) -> bool {
    find(name).is_some()
}

/// The value of `name`, if set: `std::env::var(name).ok()` without the allocation. Use
/// [`env_get!`](crate::env_get) on any path that runs more than once.
#[inline(never)]
pub(crate) fn get(name: &str) -> Option<&'static str> {
    find(name)
}

/// `envflag::is_set(name)`, cached at the call site.
#[macro_export]
#[doc(hidden)]
macro_rules! env_is_set {
    ($name:literal) => {{
        static CACHED: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
        *CACHED.get_or_init(|| $crate::envflag::is_set($name))
    }};
}

/// `envflag::get(name)`, cached at the call site.
#[macro_export]
#[doc(hidden)]
macro_rules! env_get {
    ($name:literal) => {{
        static CACHED: ::std::sync::OnceLock<::core::option::Option<&'static str>> = ::std::sync::OnceLock::new();
        *CACHED.get_or_init(|| $crate::envflag::get($name))
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_the_environment() {
        // PATH is set on every host the tests run on; this name is not.
        assert_eq!(super::is_set("PATH"), std::env::var("PATH").is_ok());
        assert_eq!(super::get("PATH"), std::env::var("PATH").ok().as_deref());
        assert!(!super::is_set("RUSTY_AV2D_SURELY_UNSET_7f3a"));
        assert_eq!(crate::env_is_set!("PATH"), std::env::var("PATH").is_ok());
        assert_eq!(crate::env_get!("PATH"), std::env::var("PATH").ok().as_deref());
        assert!(!crate::env_is_set!("RUSTY_AV2D_SURELY_UNSET_7f3a"));
        // Windows compares names case-insensitively; std does, so must we.
        assert_eq!(super::get("path"), std::env::var("path").ok().as_deref());
    }
}
