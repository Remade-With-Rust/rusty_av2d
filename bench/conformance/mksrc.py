#!/usr/bin/env python3
"""Generate a raw YUV source for the AV2 conformance corpus.

Produces `nframes` of moving/textured content so avmenc has real signal to compress
(gradients + a moving box + a little structured noise → exercises intra + inter tools).

Usage: mksrc.py <out.yuv> <w> <h> <nframes> [seed] [format] [bitdepth]
  format: 420 (default) | 422 | 444; bitdepth: 8 (default) | 10 (16-bit LE samples)
"""
import sys, struct


def main():
    out, w, h, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    fmt = sys.argv[6] if len(sys.argv) > 6 else "420"
    bd = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    ssx = 1 if fmt in ("420", "422") else 0
    ssy = 1 if fmt == "420" else 0
    cw, ch = (w + ssx) >> ssx, (h + ssy) >> ssy
    sh = bd - 8  # value upshift for >8-bit
    rng = seed
    def rnd():
        nonlocal rng
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        return rng
    def emit(f, plane):
        if bd == 8:
            f.write(bytes(plane))
        else:
            f.write(struct.pack(f"<{len(plane)}H", *[v << sh for v in plane]))
    with open(out, "wb") as f:
        for t in range(n):
            # luma: diagonal gradient that drifts + a moving bright box (motion for inter)
            box_x, box_y = (t * 7) % max(1, w - 32), (t * 5) % max(1, h - 32)
            y = [0] * (w * h)
            for j in range(h):
                base = (j * 3 + t * 2) & 0xFF
                row = j * w
                for i in range(w):
                    v = (base + i * 2 + ((rnd() >> 8) & 7)) & 0xFF
                    if box_x <= i < box_x + 32 and box_y <= j < box_y + 32:
                        v = 235
                    y[row + i] = v
            emit(f, y)
            # chroma: smooth planes that also drift (gives CfL/chroma something non-flat)
            for cbase in (0x60, 0xA0):
                c = [0] * (cw * ch)
                for j in range(ch):
                    for i in range(cw):
                        c[j * cw + i] = (cbase + i + j + t * 3) & 0xFF
                emit(f, c)
    px = (w * h + 2 * cw * ch) * (1 if bd == 8 else 2)
    print(f"wrote {out}: {n}f {w}x{h} {fmt} {bd}-bit = {n * px} bytes")


if __name__ == "__main__":
    main()
