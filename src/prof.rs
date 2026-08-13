//! Stage profiler (docs/plan.md Phase 0) — cumulative per-stage wall time.
//!
//! Compiled only under the `profiling` cargo feature; with the feature off every
//! macro expands to nothing and the decoder carries zero cost. With it on, each
//! [`prof_scope!`] accumulates elapsed nanoseconds into a per-stage atomic, and
//! [`report`] prints the table. The CLI calls `report` at exit; library users can
//! call it whenever.
//!
//!     cargo build --release --features profiling
//!     target/release/dav1d -i clip.ivf -o out.yuv --threads 1   # table on stderr
//!
//! The counters are process-global atomics (not thread-locals) so the report is
//! complete even though decode runs on a worker thread.

#[cfg(feature = "profiling")]
pub mod imp {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const N_STAGES: usize = 12;
    pub static NAMES: [&str; N_STAGES] = [
        "sb_decode",    // the whole per-SB parse+recon loop
        "coef",         // coefficient decode (entropy)
        "intra_pred",   // intra prediction kernels
        "inter_mc",     // motion compensation (translate + warp)
        "itx",          // inverse transforms + residual add
        "recon_pad",    // padded-buffer mirror (write_recon_pad)
        "deblock",      // in-loop deblocking
        "cdef_ccso",    // CDEF + CCSO
        "lr",           // loop restoration (NS/PC-Wiener, incl. chroma)
        "gdf",          // guided filter
        "grain",        // film-grain synthesis
        "emit",         // plane hand-off / output copy
    ];
    pub static ACC: [AtomicU64; N_STAGES] =
        [const { AtomicU64::new(0) }; N_STAGES];
    pub static HITS: [AtomicU64; N_STAGES] =
        [const { AtomicU64::new(0) }; N_STAGES];

    pub struct Scope {
        idx: usize,
        t0: std::time::Instant,
    }
    impl Scope {
        #[inline]
        pub fn new(idx: usize) -> Scope {
            Scope { idx, t0: std::time::Instant::now() }
        }
    }
    impl Drop for Scope {
        #[inline]
        fn drop(&mut self) {
            ACC[self.idx].fetch_add(self.t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            HITS[self.idx].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn report() {
        let total: u64 = ACC.iter().map(|a| a.load(Ordering::Relaxed)).sum();
        // `sb_decode` CONTAINS coef/intra/inter/itx/recon_pad, so it is excluded
        // from the "total" denominator to avoid double counting; it is shown as
        // its own line for the outer/inner split.
        let outer: u64 = ACC[0].load(Ordering::Relaxed)
            + (6..N_STAGES).map(|i| ACC[i].load(Ordering::Relaxed)).sum::<u64>();
        eprintln!("== rusty_av2d stage profile (wall ns, cumulative) ==");
        eprintln!("{:<12} {:>12} {:>10} {:>7}", "stage", "ms", "calls", "%outer");
        for i in 0..N_STAGES {
            let ns = ACC[i].load(Ordering::Relaxed);
            let n = HITS[i].load(Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            eprintln!(
                "{:<12} {:>12.2} {:>10} {:>6.1}%",
                NAMES[i],
                ns as f64 / 1e6,
                n,
                100.0 * ns as f64 / outer.max(1) as f64
            );
        }
        let _ = total;
        eprintln!("(sb_decode contains coef/intra/inter/itx/recon_pad; %outer = share of sb_decode+filters+emit)");
    }
}

/// Time a lexical scope into stage `$idx` (see `imp::NAMES` for indices).
#[macro_export]
#[cfg(feature = "profiling")]
macro_rules! prof_scope {
    ($idx:expr) => {
        let _prof_guard = $crate::prof::imp::Scope::new($idx);
    };
}
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! prof_scope {
    ($idx:expr) => {};
}

/// Print the cumulative table (no-op without the feature).
pub fn report() {
    #[cfg(feature = "profiling")]
    imp::report();
}
