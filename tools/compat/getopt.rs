//! Pure-Rust `getopt_long`, replacing the BSD `getopt.c` that used to be
//! compiled for `*-windows-msvc` (it went with the inherited dav1d C tree —
//! this crate links no C anywhere, including its tools).
//!
//! Implements exactly the subset the CLI parser uses, with BSD semantics:
//! short options from an optstring (`"i:o:q"`, `:` = required argument,
//! bundling and attached arguments supported), long options from an
//! `option` table (`--name value` and `--name=value`), `--` terminates,
//! unknown options print a diagnostic and return `'?'`. `optarg`/`optind`
//! are module statics the parser imports; they are process-global like their
//! C namesakes, which is fine for a CLI that parses once on the main thread.

use libc::{c_char, c_int};
use std::ffi::CStr;

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct option {
    pub name: *const c_char,
    pub has_arg: c_int,
    pub flag: *mut c_int,
    pub val: c_int,
}

#[allow(non_upper_case_globals)]
pub static mut optarg: *mut c_char = std::ptr::null_mut();
#[allow(non_upper_case_globals)]
pub static mut optind: c_int = 1;
// Position inside a bundled short-option argv element ("-abc"); 0 = at start.
static mut OPTPOS: usize = 0;

unsafe fn argv_at(nargv: *const *mut c_char, i: c_int) -> *mut c_char {
    *nargv.offset(i as isize)
}

/// BSD-style `getopt_long`. Returns the option's `val` (long) or character
/// (short), `'?'` for unknown/missing-argument, and `-1` when done.
#[allow(static_mut_refs)]
pub unsafe fn getopt_long(
    nargc: c_int,
    nargv: *const *mut c_char,
    options: *const c_char,
    long_options: *const option,
    idx: *mut c_int,
) -> c_int {
    optarg = std::ptr::null_mut();
    if optind >= nargc {
        return -1;
    }
    let arg_p = argv_at(nargv, optind);
    if arg_p.is_null() {
        return -1;
    }
    let arg = CStr::from_ptr(arg_p).to_bytes();
    if OPTPOS == 0 {
        if arg.len() < 2 || arg[0] != b'-' {
            return -1; // first non-option stops parsing (no permutation)
        }
        if arg == b"--" {
            optind += 1;
            return -1;
        }
    }

    // ---- long option: --name / --name=value / --name value ----
    if OPTPOS == 0 && arg.starts_with(b"--") {
        let body = &arg[2..];
        let (name, attached) = match body.iter().position(|&b| b == b'=') {
            Some(eq) => (&body[..eq], Some(&body[eq + 1..])),
            None => (body, None),
        };
        let mut oi = 0usize;
        loop {
            let o = &*long_options.add(oi);
            if o.name.is_null() {
                eprintln!(
                    "unknown option -- {}",
                    String::from_utf8_lossy(name)
                );
                optind += 1;
                return b'?' as c_int;
            }
            if CStr::from_ptr(o.name).to_bytes() == name {
                optind += 1;
                if o.has_arg != 0 {
                    if let Some(v) = attached {
                        // point into the original argv storage (past the '=')
                        let off = (arg.len() - v.len()) as isize;
                        optarg = arg_p.offset(off);
                    } else {
                        if optind >= nargc {
                            eprintln!(
                                "option requires an argument -- {}",
                                String::from_utf8_lossy(name)
                            );
                            return b'?' as c_int;
                        }
                        optarg = argv_at(nargv, optind);
                        optind += 1;
                    }
                } else if attached.is_some() {
                    eprintln!(
                        "option doesn't take an argument -- {}",
                        String::from_utf8_lossy(name)
                    );
                    return b'?' as c_int;
                }
                if !idx.is_null() {
                    *idx = oi as c_int;
                }
                if !o.flag.is_null() {
                    *o.flag = o.val;
                    return 0;
                }
                return o.val;
            }
            oi += 1;
        }
    }

    // ---- short option(s): -a, -abc, -ivalue, -i value ----
    if OPTPOS == 0 {
        OPTPOS = 1; // skip the '-'
    }
    let ch = arg[OPTPOS];
    OPTPOS += 1;
    let optstr = CStr::from_ptr(options).to_bytes();
    let pos = optstr.iter().position(|&b| b == ch);
    let takes_arg = pos.is_some_and(|p| optstr.get(p + 1) == Some(&b':'));
    let at_end = OPTPOS >= arg.len();
    match pos {
        None => {
            eprintln!("unknown option -- {}", ch as char);
            if at_end {
                OPTPOS = 0;
                optind += 1;
            }
            b'?' as c_int
        }
        Some(_) if takes_arg => {
            if !at_end {
                optarg = arg_p.add(OPTPOS); // attached: -ivalue
            } else {
                optind += 1;
                if optind >= nargc {
                    eprintln!("option requires an argument -- {}", ch as char);
                    OPTPOS = 0;
                    return b'?' as c_int;
                }
                optarg = argv_at(nargv, optind);
            }
            OPTPOS = 0;
            optind += 1;
            ch as c_int
        }
        Some(_) => {
            if at_end {
                OPTPOS = 0;
                optind += 1;
            }
            ch as c_int
        }
    }
}
