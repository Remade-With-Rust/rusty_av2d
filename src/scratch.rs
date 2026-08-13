//! Thread-local scratch buffers (docs/plan.md Phase 1 item 7).
//!
//! The decode path used to allocate `vec![0i32; n]` per transform unit / block
//! (`cf`, `coeff`, `residual`, `pred`, …). With `rusty_alloc` installed the
//! marginal cost is small but real, and `vec![0; n]` additionally pays a memset
//! the consumer often overwrites. This pool keeps the call sites' semantics —
//! [`zeroed`] hands out a buffer of exactly `n` zeros — while reusing capacity
//! across calls.
//!
//! Usage: `let mut cf = scratch::zeroed(n); ... cf[i] ...` — the guard derefs to
//! `[i32]` and returns the buffer to the pool on drop. Nesting is fine (the pool
//! is a stack); a leaked guard merely shrinks the pool by one buffer.

use std::cell::RefCell;

thread_local! {
    static POOL: RefCell<Vec<Vec<i32>>> = const { RefCell::new(Vec::new()) };
}

pub struct Scratch {
    buf: Vec<i32>,
}

impl std::ops::Deref for Scratch {
    type Target = [i32];
    #[inline]
    fn deref(&self) -> &[i32] {
        &self.buf
    }
}
impl std::ops::DerefMut for Scratch {
    #[inline]
    fn deref_mut(&mut self) -> &mut [i32] {
        &mut self.buf
    }
}
impl Drop for Scratch {
    #[inline]
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.buf);
        POOL.with(|p| {
            let mut pool = p.borrow_mut();
            if pool.len() < 32 {
                pool.push(buf);
            }
        });
    }
}

/// A zeroed scratch buffer of exactly `n` elements (the `vec![0i32; n]` shape,
/// minus the allocation).
#[inline]
pub fn zeroed(n: usize) -> Scratch {
    let mut buf = POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
    buf.clear();
    buf.resize(n, 0);
    Scratch { buf }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_is_zeroed_and_reuses() {
        {
            let mut a = zeroed(64);
            a[0] = 77;
            a[63] = -1;
        } // returned to pool dirty
        let b = zeroed(64);
        assert!(b.iter().all(|&v| v == 0), "reused buffer must be re-zeroed");
        assert_eq!(b.len(), 64);
        let c = zeroed(16);
        assert_eq!(c.len(), 16);
    }
}
