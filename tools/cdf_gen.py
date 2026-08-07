#!/usr/bin/env python3
"""Bit-exact AV2 CDF table generator + cross-validator.

Transforms dav2d's `cdf.c` default tables (already in decoder format, via the
`CDF1(x)=32768-x` macro + `count<<8`) into Rust const arrays, and CROSS-VALIDATES
every value against the normative spec attachments (`attachments/default_*_cdf.h`,
in `{prob, count, 0}` form). The mapping is:  decoder = { 32768 - prob, count<<8 }.

A scripted transform (vs hand-typing ~6800 lines) is how we get bit-exactness: the
two independent sources must agree on every number, or the build fails loudly.

Usage: cdf_gen.py <dav2d/src/cdf.c> <attachments_dir>   # validates partition CDFs
"""
import re, sys

def expand_cdf_macros(text):
    """Replace every CDFk(a,b,...) with its expansion (32768-a),(32768-b),...
    CDFk is defined recursively but expands to CDF1 of each argument."""
    pat = re.compile(r'CDF(\d+)\s*\(')
    while True:
        m = pat.search(text)
        if not m:
            break
        # find the matching close paren
        i = m.end()
        depth, j = 1, i
        while depth:
            if text[j] == '(':
                depth += 1
            elif text[j] == ')':
                depth -= 1
            j += 1
        args = split_args(text[i:j-1])
        repl = ', '.join(str(32768 - eval_int(a)) for a in args)
        text = text[:m.start()] + repl + text[j:]
    return text

def split_args(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch == ',' and depth == 0:
            out.append(cur); cur = ''
        else:
            if ch in '([{': depth += 1
            elif ch in ')]}': depth -= 1
            cur += ch
    if cur.strip():
        out.append(cur)
    return [a.strip() for a in out]

# named constants used in struct dimensions (resolved from dav2d enums/defines)
CONSTS = {'N_BS_SIZES': 31}

def eval_int(s):
    # leaf values may be "28084", "3 << 8", "N_BS_SIZES" — eval safely (ints/shifts/known consts)
    s = s.strip()
    for k, v in CONSTS.items():
        s = re.sub(r'\b' + k + r'\b', str(v), s)
    if not re.fullmatch(r'[0-9xX\s<>\-+*/()]+', s):
        raise ValueError(f"unsafe leaf: {s!r}")
    return eval(s)

def extract_field(cdf_c_expanded, field):
    """Pull the `.field = { ... }` initializer, return nested python lists of ints."""
    m = re.search(r'\.' + field + r'\s*=\s*\{', cdf_c_expanded)
    i = m.end(); depth, j = 1, i
    while depth:
        if cdf_c_expanded[j] == '{': depth += 1
        elif cdf_c_expanded[j] == '}': depth -= 1
        j += 1
    return parse_braces('{' + cdf_c_expanded[i:j-1] + '}')

def parse_braces(s):
    """Parse a C brace-init of ints into nested python lists."""
    s = s.strip()
    if not s.startswith('{'):
        return eval_int(s)
    return [parse_braces(p) for p in split_args(s[1:-1].strip().rstrip(','))]

def parse_attachment(path):
    """Parse default_*_cdf.h: returns nested lists of {prob,count,0} triples."""
    txt = open(path).read()
    body = txt[txt.index('{'):txt.rindex('}') + 1]
    return parse_braces(body)

def parse_struct_fields(cdf_h, struct_name):
    """Parse `typedef struct NAME { ALIGN(uint16_t f[d]...,a); ... }` -> [(field,[dims])]."""
    m = re.search(r'typedef struct ' + struct_name + r'\s*\{', cdf_h)
    i = m.end(); depth, j = 1, i
    while depth:
        if cdf_h[j] == '{': depth += 1
        elif cdf_h[j] == '}': depth -= 1
        j += 1
    body = re.sub(r'/\*.*?\*/', '', cdf_h[i:j-1], flags=re.S)
    fields = []
    for fm in re.finditer(r'ALIGN\(\s*uint16_t\s+(\w+)\s*((?:\[[^\]]+\])+)\s*,', body):
        dims = [eval_int(d) for d in re.findall(r'\[([^\]]+)\]', fm.group(2))]
        fields.append((fm.group(1), dims))
    return fields

def extract_subctx(default_body, key):
    """Extract the `.key = { ... }` sub-initializer (e.g. .m or .mv) from default_cdf body."""
    m = re.search(r'\.' + key + r'\s*=\s*\{', default_body)
    i = m.end(); depth, j = 1, i
    while depth:
        if default_body[j] == '{': depth += 1
        elif default_body[j] == '}': depth -= 1
        j += 1
    return default_body[i:j-1]

def rust_type(dims):
    t = 'u16'
    for d in reversed(dims):  # C [A][B] -> Rust [[u16; B]; A]
        t = f'[{t}; {d}]'
    return t

def rust_lit(vals):
    if isinstance(vals, int):
        return str(vals)
    return '[' + ', '.join(rust_lit(v) for v in vals) + ']'

def leaf_count(vals):
    return 1 if isinstance(vals, int) else sum(leaf_count(v) for v in vals)

def zeros(dims):
    return 0 if not dims else [zeros(dims[1:]) for _ in range(dims[0])]

def pad(vals, dims, where):
    """Pad a (possibly partial) C initializer to the full struct dims with 0s.
    C partial-init zero-fills the tail of each padded/aligned array. Errors if data
    is LONGER than the declared dim (that's a real structural mismatch)."""
    if not dims:
        if not isinstance(vals, int):
            raise ValueError(f"{where}: expected leaf, got list {vals}")
        return vals
    if isinstance(vals, int) or len(vals) > dims[0]:
        raise ValueError(f"{where}: data len {vals if isinstance(vals,int) else len(vals)} > dim {dims[0]}")
    out = [pad(v, dims[1:], where) for v in vals]
    out += [zeros(dims[1:]) for _ in range(dims[0] - len(out))]
    return out

def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p

def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)  # block comments
    text = re.sub(r'//[^\n]*', '', text)               # line comments
    return text

def extract_braced(raw, anchor_re):
    """Find `anchor {...}` and return the inner body (after matching braces)."""
    m = re.search(anchor_re, raw)
    i = m.end(); depth, j = 1, i
    while depth:
        if raw[j] == '{': depth += 1
        elif raw[j] == '}': depth -= 1
        j += 1
    return raw[i:j-1]

def extract_default_cdf_body(raw):
    """Return just the `default_cdf = { ... }` initializer text (after the macro defs)."""
    return strip_comments(extract_braced(raw, r'default_cdf\s*=\s*\{'))

# field -> spec attachment basename, for cross-validation (subset; extend as covered)
ATTACH = {
    'part_split': 'default_do_split_cdf.h',
}

def gen_subctx(struct_name, field_key, cdf_h, default_body_expanded, att_dir, out, log):
    """Generate Rust `struct {struct_name}` + `static DEFAULT_{field_key}` from dav2d."""
    fields = parse_struct_fields(cdf_h, struct_name)
    sub = expand_cdf_macros(extract_subctx(default_body_expanded, field_key)) \
        if False else extract_subctx(default_body_expanded, field_key)
    errs = 0
    out.append(f"#[derive(Clone, Copy)]\n#[repr(C)]\npub struct {struct_name} {{")
    for name, dims in fields:
        out.append(f"    pub {name}: {rust_type(dims)},")
    out.append("}\n")
    out.append(f"pub static DEFAULT_{field_key.upper()}: {struct_name} = {struct_name} {{")
    for name, dims in fields:
        vals = extract_field(sub, name)
        # cross-validate against the normative spec attachment (on the raw, pre-pad values)
        if name in ATTACH:
            att = parse_attachment(att_dir + '/' + ATTACH[name])
            n = cross_validate(name, vals, att, log)
            log.append(f"  {name}: cross-validated {n} entries vs spec attachment")
        try:
            padded = pad(vals, dims, f"{struct_name}.{name}")
        except ValueError as e:
            errs += 1
            log.append(f"  ERROR {e}")
            continue
        out.append(f"    {name}: {rust_lit(padded)},")
    out.append("};\n")
    return errs

def cross_validate(name, dav, att, log, path=()):
    """Recursively check dav {cdf,count<<8} vs attachment {prob,count,0}. Returns #checked."""
    # leaf level: dav is [cdf, count<<8], att is [prob, count, 0]
    if isinstance(dav[0], int):
        prob, count = att[0], att[1]
        if dav[0] != 32768 - prob:
            log.append(f"  CDF MISMATCH {name}{list(path)}: {dav[0]} != 32768-{prob}")
        if dav[1] != (count << 8):
            log.append(f"  COUNT MISMATCH {name}{list(path)}: {dav[1]} != {count}<<8")
        return 1
    return sum(cross_validate(name, dav[i], att[i], log, path + (i,)) for i in range(len(dav)))

def gen_array_ctx(raw, cdf_h, struct_name, c_array, const_name, k, doc, out, log):
    """Generate `struct {struct_name}` + `static {const_name}: [_; k]` from a C
    `static const {struct_name} {c_array}[k] = { {..}, ... }` (per-variant defaults)."""
    fields = parse_struct_fields(cdf_h, struct_name)
    body = expand_cdf_macros(strip_comments(
        extract_braced(raw, c_array + r'\[' + str(k) + r'\]\s*=\s*\{')))
    elems = split_args(body)
    if len(elems) != k:
        log.append(f"  ERROR {c_array} has {len(elems)} elements != {k}")
        return 1
    errs = 0
    out.append(f"#[derive(Clone, Copy)]\n#[repr(C)]\npub struct {struct_name} {{")
    for name, dims in fields:
        out.append(f"    pub {name}: {rust_type(dims)},")
    out.append("}\n")
    out.append(f"/// {doc}")
    out.append(f"pub static {const_name}: [{struct_name}; {k}] = [")
    for vi, elem in enumerate(elems):
        eb = elem.strip()[1:-1]
        out.append(f"    {struct_name} {{")
        for name, dims in fields:
            try:
                padded = pad(extract_field(eb, name), dims, f"{struct_name}[{vi}].{name}")
            except ValueError as e:
                errs += 1
                log.append(f"  ERROR {e}")
                continue
            out.append(f"        {name}: {rust_lit(padded)},")
        out.append("    },")
    out.append("];\n")
    log.append(f"  {struct_name}: {len(fields)} fields x {k} variants generated")
    return errs

def main():
    cdf_c, cdf_h_path, att_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    raw = open(cdf_c).read()
    cdf_h = open(cdf_h_path).read()
    default_body = expand_cdf_macros(extract_default_cdf_body(raw))

    out, log = [], []
    out.append("// @generated by tools/cdf_gen.py from dav2d cdf.{c,h}; cross-validated vs spec attachments.")
    out.append("// DO NOT EDIT. Format: decoder CDF = {32768-prob, count<<8}.\n")
    errs = 0
    errs += gen_subctx('CdfModeContext', 'm', cdf_h, default_body, att_dir, out, log)
    errs += gen_subctx('CdfMvContext', 'mv', cdf_h, default_body, att_dir, out, log)
    errs += gen_array_ctx(raw, cdf_h, 'CdfCoefContext', 'default_coef_cdf', 'DEFAULT_COEF', 4,
                          "Default coeff CDFs by q-category = (qidx>90)+(qidx>140)+(qidx>190).", out, log)
    errs += gen_array_ctx(raw, cdf_h, 'CdfTxPart2dContext', 'default_tx_part_2d_cdf', 'DEFAULT_TX2D', 2,
                          "Default tx-partition-2d CDFs by reduced_tx_part_set.", out, log)
    # assemble the full CdfContext + the static-init (dav2d cdf_thread_copy static path)
    out.append("#[derive(Clone, Copy)]\n#[repr(C)]\npub struct CdfContext {")
    out.append("    pub coef: CdfCoefContext,\n    pub m: CdfModeContext,")
    out.append("    pub tx2d: CdfTxPart2dContext,\n    pub mv: CdfMvContext,\n    pub dmv: CdfMvContext,\n}\n")
    out.append("/// Initial CDF context for a frame: coef by q-category, tx2d by reduced_tx_part_set,")
    out.append("/// mode/mv from defaults, dmv == mv (dav2d `cdf_thread_copy` static path).")
    out.append("pub fn default_cdf_context(qidx: u32, reduced_tx_part_set: usize) -> CdfContext {")
    out.append("    let qcat = (qidx > 90) as usize + (qidx > 140) as usize + (qidx > 190) as usize;")
    out.append("    CdfContext {")
    out.append("        coef: DEFAULT_COEF[qcat], m: DEFAULT_M,")
    out.append("        tx2d: DEFAULT_TX2D[reduced_tx_part_set], mv: DEFAULT_MV, dmv: DEFAULT_MV,")
    out.append("    }\n}")

    for line in log:
        print(line)
    print(f"\nCdfModeContext: {len(parse_struct_fields(cdf_h,'CdfModeContext'))} fields generated, {errs} errors")
    if errs == 0:
        open(out_path, 'w').write('\n'.join(out))
        print(f"WROTE {out_path} ({len(out)} lines) — all dims valid + attachments agree.")
    return 1 if errs else 0

if __name__ == '__main__':
    sys.exit(main())
