//! AV2 reconstruction frame buffer + intra-edge gathering.
//!
//! A thread-local per-plane pixel buffer (`i32` samples, 8-bit range 0..=255) that the recon
//! pass writes into as blocks decode. Intra prediction reads already-reconstructed neighbour
//! pixels (top row / left column / top-left corner) out of this buffer. This is ADDITIVE to the
//! (already bit-exact) entropy parse — it only consumes the parsed coefficients + mode.

use std::cell::RefCell;

/// One reconstructed plane: `stride`-major `u16` samples (docs/plan.md Phase 1
/// item 1 — samples were `i32`, which was 2–4× the memory traffic the data
/// needs and halved every vector's effective width; `u16` covers 8- and
/// 10/12-bit content in one layout, and block-local intermediates stay `i32`).
#[derive(Default, Clone)]
pub struct Plane {
    pub px: Vec<u16>,
    pub w: usize,
    pub h: usize,
    pub stride: usize,
}

impl Plane {
    pub(crate) fn alloc(w: usize, h: usize) -> Self {
        let stride = w;
        Plane { px: vec![0u16; stride * h.max(1)], w, h, stride }
    }
    /// Allocate a plane whose stride/height are rounded UP to a superblock multiple (`sb`), so a
    /// block spilling past the visible `w`/`h` edge still fits. `w`/`h` are set to the PADDED dims
    /// (not the visible ones) so an intra-edge gather over this plane clamps to the padded extent
    #[inline]
    #[track_caller]
    pub fn at(&self, x: usize, y: usize) -> i32 {
        let i = y * self.stride + x;
        if std::env::var("ATDBG").is_ok() && i >= self.px.len() {
            crate::dlog!("[ATDBG] Plane::at OOB x={x} y={y} w={} h={} len={} caller={}", self.w, self.h, self.px.len(), std::panic::Location::caller());
        }
        // HARDENING (root fix): every Plane::at caller in the recon/filter paths derives (x,y)
        // from block geometry; a corrupt stream can push any of them outside the plane. Reading
        // 0 outside is the same discipline as the C reference's edge emulation, and it removes
        // the whole class at the source rather than caller by caller.
        self.px.get(i).map(|&v| v as i32).unwrap_or(0)
    }

    /// Checked pixel WRITE — the write-side twin of `at()`. Out-of-plane writes are dropped
    /// (a corrupt stream's geometry can address outside the plane; a valid one never does).
    /// Using this instead of `px[i] = v` makes a recon path total: no bounds panic is
    /// reachable through it, which is the structural (not sampled) panic-freedom argument.
    #[inline]
    pub fn set_at(&mut self, x: usize, y: usize, v: i32) {
        let i = y * self.stride + x;
        if let Some(p) = self.px.get_mut(i) {
            *p = v as u16;
        }
    }

    /// Checked row-span write (the `copy_from_slice` twin): clips the span to the plane.
    #[inline]
    pub fn set_row(&mut self, x: usize, y: usize, src: &[i32]) {
        let base = y * self.stride + x;
        let n = src.len().min(self.px.len().saturating_sub(base));
        for (d, &v) in self.px[base..base + n].iter_mut().zip(src) {
            *d = v as u16;
        }
    }
}

/// The current decoded frame: Y + U + V planes (4:2:0), plus the frame Y-AC q-index.
#[derive(Default)]
pub struct Frame {
    pub pl: [Plane; 3],
    pub yac: u8,
    pub bitdepth_max: i32,
    /// seq_hdr.av2.intra_edge_filter — enables directional edge filtering in z1/z2/z3.
    pub edge_filter: bool,
    /// seq `enable_ibp` — Intra Boundary Prediction gradient (DC + z1/z3, tx != 4x4).
    pub ibp: bool,
    /// SB-relative (16×16 mi) decode-order grid (avm `is_mi_coded`) — drives top-right /
    /// bottom-left intra-edge availability. Reset per SB.
    pub mi_coded: Vec<bool>,
    /// The SB (`(bx4>>4, by4>>4)`) that `mi_coded` currently describes; sentinel = fresh.
    pub mi_coded_sb: (usize, usize),
    /// Frame-wide per-mi joint intra mode (avm `joint_y_mode_delta_angle`): 0..4 = non-directional
    /// (DC/SMOOTH/SMOOTH_V/SMOOTH_H/PAETH), 5..60 = directional (`5 + midx`). Drives both the
    /// neighbour-based mode-list reordering and the edge-filter smooth `type`. 0 = DC (default).
    pub joint: Vec<u8>,
    pub iw4: usize,
    pub ih4: usize,
    /// Deblock edge grids (per-4×4 cell, `iw4*ih4`). `db_left`/`db_top` mark a block-left
    /// (vertical) / block-top (horizontal) edge at that cell; `db_lw`/`db_lh` are the tx
    /// width/height levels `min(log2(w4), 3)` / `min(log2(h4), 3)` of the covering block.
    /// The edge strength index is the `min` of the two adjacent cells' levels (dav2d lf_mask).
    pub db_lw: Vec<u8>,
    pub db_lh: Vec<u8>,
    pub db_left: Vec<bool>,
    pub db_top: Vec<bool>,
    /// Chroma deblock grids (per chroma-4×4 cell, `ciw4*cih4`). Same semantics as the luma
    /// grids but levels are capped at 2 (`max_width_uv` has 3 entries) and the chroma tx =
    /// luma tx subsampled 4:2:0. `ciw4`/`cih4` are the chroma plane dims in 4px units.
    pub cdb_lw: Vec<u8>,
    pub cdb_lh: Vec<u8>,
    pub cdb_left: Vec<bool>,
    pub cdb_top: Vec<bool>,
    /// Sub-PU WEAK-edge markers (dav2d lf_mask masks[..][4]): a cell whose V (left) / H (top)
    /// deblock edge was ADDED by mask_subpu_edges (not a real block/tx edge) filters with
    /// thresholds >> 3 (dav deblock_tmpl setup_thr `>> 3*subpu`). Luma + chroma grids.
    pub db_spv: Vec<bool>,
    pub db_sph: Vec<bool>,
    pub cdb_spv: Vec<bool>,
    pub cdb_sph: Vec<bool>,
    pub ciw4: usize,
    pub cih4: usize,
    /// CHROMA decode-order grid (avm `is_mi_coded[CHROMA tree]`) — SB-relative 16×16 in LUMA mi
    /// units, marked as chroma blocks decode (their luma-equivalent region). Chroma intra edge
    /// availability (`has_top_right`/`has_bottom_left`) reads THIS, not the luma grid: the chroma
    /// tree has its own decode order (SDP), so the luma grid gives the wrong availability.
    pub mi_coded_c: Vec<bool>,
    /// Frame-wide per-chroma-4×4-cell SMOOTH flag (dav2d `b->is_sm[1]`): 1 iff the covering chroma
    /// block used a SMOOTH mode (uv_mode 9/10/11). Drives the intra edge-filter strength/type for
    /// the NEIGHBOUR's edge. `ciw4*cih4`.
    pub sm_c: Vec<u8>,
    /// CDEF per-64px-SB index (dav2d `cdef_idx`), `sbw*sbh`, `-1` = no CDEF. Selects a
    /// (y,uv) strength pair from `CdefCfg`. Set at each SB's first leaf during recon.
    pub cdef_idx: Vec<i8>,
    pub cdef_sbw: usize,
    /// CDEF noskip mask (dav2d `noskip_mask`): per 8px-stripe (`y4>>1`) × 4px-col, true when a
    /// non-skip luma block covers it. An 8×8 CDEF block filters only if its 2 cols are marked.
    pub noskip: Vec<bool>,
    pub noskip_stripes: usize,
    /// CCSO per-256px-luma-block enable, flat `plane*ccso_n256 + block`. Decoded once per 256px
    /// SB per plane at the SB's first leaf. A sample is CCSO-filtered iff its block's flag is on.
    pub ccso_blk: Vec<bool>,
    pub ccso_col256: usize,
    pub ccso_n256: usize,
    /// log2 of the (tile-adaptive) CCSO unit in PIXELS (8 = 256px default).
    pub ccso_px_shift: u32,
    /// The (tile-adaptive) GDF block size in PIXELS (128 default).
    pub gdf_bs_px: usize,
    /// GDF per-block on/off flags (avm `gdf_block_flags`), row-major `gdf_block_num_w`. Decoded
    /// once per GDF block (default 128px) at the SB's first leaf when GDF mode==2.
    pub gdf_blk: Vec<bool>,
    pub gdf_blk_w: usize,
    /// LR noskip mask (dav lr_noskip_mask): per 4px luma cell, true when a luma TX unit
    /// with coded coefficients (eob != -1) covers it. Feeds PC-Wiener classification.
    pub lr_noskip: Vec<bool>,
    /// Per-64px-SB effective qindex under delta-q (dav2d `lf_mask->qidx`), `cdef_sbw*sbh`.
    /// Equal to the frame yac everywhere when `delta.q.present` is off. Read by the deblock
    /// filter-level lut (per-SB, not per-frame).
    pub sb_qidx: Vec<u16>,
}

impl Frame {
    /// Clear the SB decode-order grids (luma + chroma) when crossing into a new 64×64 SB.
    /// Called at the SB's first LUMA leaf; the chroma grid is then populated during this SB's
    /// chroma decode (which runs after luma, before the next SB's luma → correct reset timing).
    pub fn ensure_sb(&mut self, bx4: usize, by4: usize) {
        let sh = crate::av2_recon::sb_step4().trailing_zeros();
        let sb = (bx4 >> sh, by4 >> sh);
        if sb != self.mi_coded_sb {
            for v in self.mi_coded.iter_mut() {
                if !crate::av2_recon::work_tick("frm:140") { break; }
                *v = false;
            }
            for v in self.mi_coded_c.iter_mut() {
                if !crate::av2_recon::work_tick("frm:143") { break; }
                *v = false;
            }
            self.mi_coded_sb = sb;
        }
    }
    /// Is the SB-relative mi at `(row, col)` (0..16) decoded? Out-of-grid → false.
    pub fn mi_coded_at(&self, row: i32, col: i32) -> bool {
        // The window is exactly one SB: probes past the SB edge are unavailable.
        let st = crate::av2_recon::sb_step4() as i32;
        if !(0..st).contains(&row) || !(0..st).contains(&col) {
            return false;
        }
        self.mi_coded[(row * 32 + col) as usize]
    }
    /// Chroma-tree decode-order grid lookup (SB-relative, luma mi units). Out-of-grid → false.
    pub fn mi_coded_c_at(&self, row: i32, col: i32) -> bool {
        let st = crate::av2_recon::sb_step4() as i32;
        if !(0..st).contains(&row) || !(0..st).contains(&col) {
            return false;
        }
        self.mi_coded_c[(row * 32 + col) as usize]
    }
    /// Mark a chroma block's luma-equivalent region (SB-relative) into the chroma decode grid.
    pub fn mark_coded_c(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize) {
        let st = crate::av2_recon::sb_step4();
        let (sr, sc) = (by4 & (st - 1), bx4 & (st - 1));
        for r in sr..(sr + bh4).min(st) {
            if !crate::av2_recon::work_tick("frm:170") { break; }
            for c in sc..(sc + bw4).min(st) {
                if !crate::av2_recon::work_tick("frm:171") { break; }
                self.mi_coded_c[r * 32 + c] = true;
            }
        }
    }
    /// Chroma analogue of [`mark_coded_avail`]: mark the chroma decode-order availability
    /// grid for a leaf that extends past the frame edge, before the off-frame bounds check.
    pub fn mark_coded_c_avail(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize) {
        self.mark_coded_c(bx4, by4, bw4, bh4);
    }
    /// Mark a block's mi region (SB-relative) decoded + record its joint mode frame-wide.
    /// Mark ONLY the SB-relative decode-order availability grid (`mi_coded`) for a leaf.
    /// dav2d sets `is_coded` for the full block extent of EVERY decoded leaf — including
    /// blocks that extend past the frame's right/bottom edge (only pixel writes are clipped).
    /// This must run before `recon_intra_luma`'s off-frame bounds check, or an interior
    /// block's top-right / bottom-left availability is understated (its edge-of-frame
    /// neighbour never marks the grid), corrupting its directional prediction edge.
    pub fn mark_coded_avail(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize) {
        let st = crate::av2_recon::sb_step4();
        let (sr, sc) = (by4 & (st - 1), bx4 & (st - 1));
        for r in sr..(sr + bh4).min(st) {
            if !crate::av2_recon::work_tick("frm:191") { break; }
            for c in sc..(sc + bw4).min(st) {
                if !crate::av2_recon::work_tick("frm:192") { break; }
                self.mi_coded[r * 32 + c] = true;
            }
        }
    }
    pub fn mark_coded(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize, joint: u8) {
        let st = crate::av2_recon::sb_step4();
        let (sr, sc) = (by4 & (st - 1), bx4 & (st - 1));
        for r in sr..(sr + bh4).min(st) {
            if !crate::av2_recon::work_tick("frm:200") { break; }
            for c in sc..(sc + bw4).min(st) {
                if !crate::av2_recon::work_tick("frm:201") { break; }
                self.mi_coded[r * 32 + c] = true;
            }
        }
        for r in by4..(by4 + bh4).min(self.ih4) {
            if !crate::av2_recon::work_tick("frm:205") { break; }
            for c in bx4..(bx4 + bw4).min(self.iw4) {
                if !crate::av2_recon::work_tick("frm:206") { break; }
                self.joint[r * self.iw4 + c] = joint;
            }
        }
        self.mark_db(bx4, by4, bw4, bh4);
    }
    /// Record a coded luma block's deblock edges + tx levels (dav2d lf_mask, block=single-TX).
    /// The block's left column / top row are block edges; interior cells are not.
    /// mark_db with EXPLICIT tx levels — for edge-clamped units whose visible cell span is
    /// not the TX size (dav derives the level from t_dim, not the clamped extent).
    pub fn mark_db_lvl(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize, lw: u8, lh: u8) {
        if self.db_lw.is_empty() {
            return;
        }
        let (iw4, ih4) = (self.iw4, self.ih4);
        for r in by4..(by4 + bh4).min(ih4) {
            if !crate::av2_recon::work_tick("frm:221") { break; }
            for c in bx4..(bx4 + bw4).min(iw4) {
                if !crate::av2_recon::work_tick("frm:222") { break; }
                let cell = r * iw4 + c;
                self.db_lw[cell] = lw;
                self.db_lh[cell] = lh;
                self.db_left[cell] = c == bx4;
                self.db_top[cell] = r == by4;
            }
        }
    }
    pub fn mark_db(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize) {
        if self.db_lw.is_empty() {
            return;
        }
        let lw = (bw4.trailing_zeros()).min(3) as u8; // min(log2(bw4), 3)
        let lh = (bh4.trailing_zeros()).min(3) as u8;
        let (iw4, ih4) = (self.iw4, self.ih4);
        for r in by4..(by4 + bh4).min(ih4) {
            if !crate::av2_recon::work_tick("frm:238") { break; }
            for c in bx4..(bx4 + bw4).min(iw4) {
                if !crate::av2_recon::work_tick("frm:239") { break; }
                let cell = r * iw4 + c;
                self.db_lw[cell] = lw;
                self.db_lh[cell] = lh;
                self.db_left[cell] = c == bx4;
                self.db_top[cell] = r == by4;
            }
        }
    }
    /// Sub-PU deblock layer, LUMA (dav2d lf_mask.c create_db_mask, `subpu_l2 != 3` arm):
    /// cap the block's cell tx-levels at `subpu_l2` (the capped level propagates to outer
    /// edges via the min-of-adjacent-cells rule at apply time, mirroring dav's a/l arrays),
    /// then ADD inner edges on a `(1<<subpu_l2)`-cell cadence (dav mask_subpu_edges). Edges
    /// that are NEW (not already block/tx edges) are weak (thresholds >>3) unless `fm2`
    /// (TIP-as-output frames: ds_sub_pu_mask==0 → normal strength) or the block-relative
    /// offset is a multiple of 16 cells (`c & 15` gate).
    pub fn mark_db_subpu(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize, subpu_l2: usize, fm2: bool) {
        if self.db_lw.is_empty() || subpu_l2 >= 3 {
            return;
        }
        let (iw4, ih4) = (self.iw4, self.ih4);
        let sl = subpu_l2 as u8;
        for r in by4..(by4 + bh4).min(ih4) {
            if !crate::av2_recon::work_tick("frm:261") { break; }
            for c in bx4..(bx4 + bw4).min(iw4) {
                if !crate::av2_recon::work_tick("frm:262") { break; }
                let cell = r * iw4 + c;
                self.db_lw[cell] = self.db_lw[cell].min(sl);
                self.db_lh[cell] = self.db_lh[cell].min(sl);
            }
        }
        let sz = 1usize << subpu_l2;
        let mut x = sz;
        while x < bw4 {
            let c = bx4 + x;
            if c < iw4 {
                for r in by4..(by4 + bh4).min(ih4) {
                    if !crate::av2_recon::work_tick("frm:273") { break; }
                    let cell = r * iw4 + c;
                    if !self.db_left[cell] {
                        self.db_left[cell] = true;
                        if !fm2 && (x & 15) != 0 {
                            self.db_spv[cell] = true;
                        }
                    }
                }
            }
            x += sz;
        }
        let mut y = sz;
        while y < bh4 {
            let r = by4 + y;
            if r < ih4 {
                for c in bx4..(bx4 + bw4).min(iw4) {
                    if !crate::av2_recon::work_tick("frm:289") { break; }
                    let cell = r * iw4 + c;
                    if !self.db_top[cell] {
                        self.db_top[cell] = true;
                        if !fm2 && (y & 15) != 0 {
                            self.db_sph[cell] = true;
                        }
                    }
                }
            }
            y += sz;
        }
    }
    /// Sub-PU deblock layer, CHROMA. Cell coords/dims in CHROMA 4px cells; cadence
    /// `(1<<subpu_l2)>>ss` chroma cells; level caps `clamp(subpu_l2 - ss, 0, 2)` per axis
    /// (dav create_db_mask hlim/vlim + mask_subpu_edges h/v_subpu_l2).
    pub fn mark_db_subpu_chroma(&mut self, cbx4: usize, cby4: usize, cbw4: usize, cbh4: usize, subpu_l2: usize, ss_hor: usize, ss_ver: usize, fm2: bool) {
        if self.cdb_lw.is_empty() || subpu_l2 >= 3 {
            return;
        }
        let (iw4, ih4) = (self.ciw4, self.cih4);
        let lw_cap = (subpu_l2 as i32 - ss_hor as i32).clamp(0, 2) as u8;
        let lh_cap = (subpu_l2 as i32 - ss_ver as i32).clamp(0, 2) as u8;
        for r in cby4..(cby4 + cbh4).min(ih4) {
            if !crate::av2_recon::work_tick("frm:312") { break; }
            for c in cbx4..(cbx4 + cbw4).min(iw4) {
                if !crate::av2_recon::work_tick("frm:313") { break; }
                let cell = r * iw4 + c;
                self.cdb_lw[cell] = self.cdb_lw[cell].min(lw_cap);
                self.cdb_lh[cell] = self.cdb_lh[cell].min(lh_cap);
            }
        }
        let hsz = (1usize << subpu_l2) >> ss_hor;
        let vsz = (1usize << subpu_l2) >> ss_ver;
        if hsz > 0 {
            let mut x = hsz;
            while x < cbw4 {
                let c = cbx4 + x;
                if c < iw4 {
                    for r in cby4..(cby4 + cbh4).min(ih4) {
                        if !crate::av2_recon::work_tick("frm:326") { break; }
                        let cell = r * iw4 + c;
                        if !self.cdb_left[cell] {
                            self.cdb_left[cell] = true;
                            if !fm2 && (x & 15) != 0 {
                                self.cdb_spv[cell] = true;
                            }
                        }
                    }
                }
                x += hsz;
            }
        }
        if vsz > 0 {
            let mut y = vsz;
            while y < cbh4 {
                let r = cby4 + y;
                if r < ih4 {
                    for c in cbx4..(cbx4 + cbw4).min(iw4) {
                        if !crate::av2_recon::work_tick("frm:344") { break; }
                        let cell = r * iw4 + c;
                        if !self.cdb_top[cell] {
                            self.cdb_top[cell] = true;
                            if !fm2 && (y & 15) != 0 {
                                self.cdb_sph[cell] = true;
                            }
                        }
                    }
                }
                y += vsz;
            }
        }
    }
    /// Store a 64px SB's decoded CDEF index (called at the SB's first leaf during recon).
    pub fn set_cdef_idx(&mut self, bx4: usize, by4: usize, v: i8) {
        if self.cdef_idx.is_empty() {
            return;
        }
        let sb = (by4 >> 4) * self.cdef_sbw + (bx4 >> 4);
        if sb < self.cdef_idx.len() {
            self.cdef_idx[sb] = v;
        }
    }
    /// Store a 64px SB's effective delta-q qindex (called at the SB's first has_luma leaf).
    pub fn set_sb_qidx(&mut self, bx4: usize, by4: usize, q: u16) {
        if self.sb_qidx.is_empty() {
            return;
        }
        // The grid is per-64px cell; a 128px superblock's delta-q fills its 2x2 cells.
        // Per-axis bounds: an edge SB's out-of-frame 64-col/row must NOT wrap into the
        // next grid row (SB(96,0) at 432px wrote its col-7 cell into (0,1) = a ±1 deblock
        // threshold error on every H edge of SB(0,0)'s second 64-row).
        let n64 = crate::av2_recon::sb_step4() / 16;
        let rows = self.sb_qidx.len() / self.cdef_sbw.max(1);
        for dy in 0..n64 {
            if !crate::av2_recon::work_tick("frm:379") { break; }
            for dx in 0..n64 {
                if !crate::av2_recon::work_tick("frm:380") { break; }
                let (cx, cy) = ((bx4 >> 4) + dx, (by4 >> 4) + dy);
                if cx < self.cdef_sbw && cy < rows {
                    self.sb_qidx[cy * self.cdef_sbw + cx] = q;
                }
            }
        }
    }
    /// The effective qindex at luma pixel `(lx, ly)` (its covering 64px SB); `frame_yac` fallback
    /// (0 = unset — the grid is only populated when the frame codes delta-q).
    pub fn sb_qidx_at(&self, lx: usize, ly: usize, frame_yac: u32) -> u32 {
        if self.sb_qidx.is_empty() {
            return frame_yac;
        }
        let sb = (ly >> 6) * self.cdef_sbw + (lx >> 6);
        match self.sb_qidx.get(sb) {
            Some(&q) if q != 0 => q as u32,
            _ => frame_yac,
        }
    }
    /// Store a GDF block's decoded on/off flag (128px block, at `(bx4|by4)&31==0`).
    /// dav recon_tmpl.c:2694: mark a luma TX unit with coded coefficients into the LR
    /// noskip mask (per 4px cell).
    pub fn mark_lr_noskip(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize) {
        if self.lr_noskip.is_empty() {
            return;
        }
        for r in by4..(by4 + bh4).min(self.ih4) {
            if !crate::av2_recon::work_tick("frm:407") { break; }
            for c in bx4..(bx4 + bw4).min(self.iw4) {
                if !crate::av2_recon::work_tick("frm:408") { break; }
                self.lr_noskip[r * self.iw4 + c] = true;
            }
        }
    }
    pub fn set_gdf_blk(&mut self, bx4: usize, by4: usize, on: bool) {
        if self.gdf_blk.is_empty() {
            return;
        }
        let sh = (self.gdf_bs_px as u32 / 4).trailing_zeros();
        let idx = (by4 >> sh) * self.gdf_blk_w + (bx4 >> sh);
        if idx < self.gdf_blk.len() {
            self.gdf_blk[idx] = on;
        }
    }
    /// Store a 256px SB's decoded CCSO enable flag for `plane` (called at the SB's first leaf).
    pub fn set_ccso_blk(&mut self, bx4: usize, by4: usize, plane: usize, on: bool) {
        if self.ccso_blk.is_empty() {
            return;
        }
        let sh = self.ccso_px_shift - 2;
        let block = (by4 >> sh) * self.ccso_col256 + (bx4 >> sh);
        let i = plane * self.ccso_n256 + block;
        if i < self.ccso_blk.len() {
            self.ccso_blk[i] = on;
        }
    }
    /// Is CCSO enabled for `plane` at luma pixel `(lx, ly)` (its covering 256px block)?
    pub fn ccso_blk_on(&self, plane: usize, lx: usize, ly: usize) -> bool {
        if self.ccso_blk.is_empty() {
            return false;
        }
        let block = (ly >> self.ccso_px_shift) * self.ccso_col256 + (lx >> self.ccso_px_shift);
        let i = plane * self.ccso_n256 + block;
        i < self.ccso_blk.len() && self.ccso_blk[i]
    }
    /// Mark a non-skip luma block into the CDEF noskip mask (dav2d decode.c:3517: per 8px-stripe
    /// `(by4+y)>>1`, columns `[bx4, bx4+bw4)`, one stripe entry per 2 4px-rows).
    pub fn mark_noskip(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize) {
        if self.noskip.is_empty() {
            return;
        }
        let (iw4, ns) = (self.iw4, self.noskip_stripes);
        let mut y = 0;
        while y < bh4 {
            let stripe = (by4 >> 1) + (y >> 1);
            if stripe < ns {
                for c in bx4..(bx4 + bw4).min(iw4) {
                    if !crate::av2_recon::work_tick("frm:455") { break; }
                    self.noskip[stripe * iw4 + c] = true;
                }
            }
            y += 2;
        }
    }
    /// Record a chroma leaf's deblock edges (chroma-4px coords). Chroma tx = the leaf's
    /// chroma tx (block = single TX); levels cap at 2 (`max_width_uv`).
    pub fn mark_db_chroma(&mut self, cbx4: usize, cby4: usize, cbw4: usize, cbh4: usize) {
        if self.cdb_lw.is_empty() {
            return;
        }
        if std::env::var("DBLK444").is_ok()
            && (cbx4 * 4 <= 208 && (cbx4 + cbw4) * 4 >= 192 && cby4 * 4 <= 124 && (cby4 + cbh4) * 4 >= 112)
        {
            crate::dlog!("[MMARK] c=({},{}) w={} h={}", cbx4 * 4, cby4 * 4, cbw4 * 4, cbh4 * 4);
        }
        let lw = (cbw4.trailing_zeros()).min(2) as u8; // min(log2(cbw4), 2)
        let lh = (cbh4.trailing_zeros()).min(2) as u8;
        let (ciw4, cih4) = (self.ciw4, self.cih4);
        for r in cby4..(cby4 + cbh4).min(cih4) {
            if !crate::av2_recon::work_tick("frm:476") { break; }
            for c in cbx4..(cbx4 + cbw4).min(ciw4) {
                if !crate::av2_recon::work_tick("frm:477") { break; }
                let cell = r * ciw4 + c;
                self.cdb_lw[cell] = lw;
                self.cdb_lh[cell] = lh;
                self.cdb_left[cell] = c == cbx4;
                self.cdb_top[cell] = r == cby4;
            }
        }
    }
    /// Frame-wide joint mode of the mi at `(bx4, by4)`; out-of-frame → 0 (DC, non-directional).
    pub fn joint_at(&self, bx4: i32, by4: i32) -> u8 {
        if bx4 < 0 || by4 < 0 || bx4 >= self.iw4 as i32 || by4 >= self.ih4 as i32 {
            return 0;
        }
        self.joint[(by4 * self.iw4 as i32 + bx4) as usize]
    }
    /// Neighbour used a SMOOTH mode (joint 1/2/3) — the edge-filter `type`.
    pub fn smooth_at(&self, bx4: i32, by4: i32) -> bool {
        (1..=3).contains(&self.joint_at(bx4, by4))
    }
}

thread_local! {
    pub static FRAME: RefCell<Frame> = RefCell::new(Frame::default());
    /// Process-global pixel max, `(1 << bitdepth) - 1` (255 for 8-bit, 1023 for 10-bit).
    /// Set by `reset_frame`; read by kernels that don't take a bd param (MC puts, blends).
    pub static BDMAX: std::cell::Cell<i32> = const { std::cell::Cell::new(255) };
    /// Process-global chroma subsampling `(ss_hor, ss_ver)` from the seq-header layout:
    /// 4:2:0 = (1,1), 4:2:2 = (1,0), 4:4:4 = (0,0). Set at the seq parse (rav2e ss.rs pattern);
    /// the 4:2:0 default keeps every existing stream byte-identical.
    pub static SS: std::cell::Cell<(u32, u32)> = const { std::cell::Cell::new((1, 1)) };
    /// Debug isolation: dav2d pre-filter luma reference plane. When present, `recon_intra_luma`
    /// gathers intra edges from THIS (always-correct neighbours) instead of its own output, so
    /// each block is scored independently of neighbour error-propagation.
    pub static REF_LUMA: RefCell<Option<Plane>> = const { RefCell::new(None) };
    /// Per-block isolation scoreboard: (#blocks bit-exact vs REF, #blocks total).
    pub static REF_SCORE: std::cell::Cell<(u32, u32)> = const { std::cell::Cell::new((0, 0)) };
    /// dav2d pre-filter chroma reference planes [U, V] for the chroma-recon isolation harness.
    pub static REF_CHROMA: RefCell<[Option<Plane>; 2]> = const { RefCell::new([None, None]) };
    /// Chroma per-block scoreboard: (#U-exact, #V-exact, #blocks total).
    pub static REF_CHROMA_SCORE: std::cell::Cell<(u32, u32, u32)> = const { std::cell::Cell::new((0, 0, 0)) };
    /// SB-aligned PADDED reconstruction buffer [Y, U, V] (empty until `reset_frame`). Every recon
    /// write mirrors its FULL block extent here (the visible-frame `FRAME` write stays clipped for
    /// the bit-exact, stride==w filter chain). A later block's intra gather reads THIS instead of
    /// `f.pl` (when no dav reference is loaded), so an edge block's off-frame top-right/left samples
    /// are the REAL recon of the spilling neighbour — matching dav's aligned picture buffer.

    /// Stage D: frame-1 reconstructed+FILTERED output planes [Y, U, V] — the inter-prediction
    /// reference (dav2d `f->refp[0]`). Loaded from the normal (filtered) dav2d frame-1 output.
    pub static REF_FRAME1: RefCell<[Option<Plane>; 3]> = const { RefCell::new([None, None, None]) };
    /// Mine's OWN just-decoded + fully-filtered frame-1 planes, stashed to serve as the inter
    /// frame's reference — a true standalone decode with no external reference file.
    pub static STASH_FRAME1: RefCell<[Option<Plane>; 3]> = const { RefCell::new([None, None, None]) };
    /// The 8-slot reference-PICTURE buffer (dav2d `c->refs[i].p`): each decoded+filtered frame's
    /// [Y,U,V] planes, written into every slot selected by refresh_frame_flags. Parallel to the
    /// REF_SLOTS metadata. The compound/multi-ref MC indexes this by `refidx[ref]`.
    pub static REF_PICS: RefCell<[Option<[Plane; 3]>; 8]> =
        const { RefCell::new([None, None, None, None, None, None, None, None]) };
    /// Stage D inter-recon scoreboard: (#luma blocks bit-exact vs REF, #luma blocks total).
    pub static INTER_SCORE: std::cell::Cell<(u32, u32)> = const { std::cell::Cell::new((0, 0)) };
    /// Stage D harness: dav2d's frame-2 luma inter PREDICTION plane (pre-residual), for scoring
    /// my per-block MC output. Loaded from the `dav2d_f2pred_capture` dump.
    pub static REF_F2PRED: RefCell<Option<Plane>> = const { RefCell::new(None) };
    /// Stage D harness: dav2d's frame-2 chroma [U,V] inter PREDICTION planes (pre-residual).
    pub static REF_F2PREDC: RefCell<[Option<Plane>; 2]> = const { RefCell::new([None, None]) };
    /// Stage D chroma inter-recon scoreboard: (#U-exact, #V-exact, #blocks total).
    pub static INTER_SCORE_C: std::cell::Cell<(u32, u32, u32)> = const { std::cell::Cell::new((0, 0, 0)) };
    /// Stage D WARP-MC luma scoreboard: (#warp luma blocks bit-exact, #warp luma blocks total).
    pub static INTER_SCORE_W: std::cell::Cell<(u32, u32)> = const { std::cell::Cell::new((0, 0)) };
    /// Stage D WARP-MC chroma scoreboard: (#U-exact, #V-exact, #blocks total).
    pub static INTER_SCORE_WC: std::cell::Cell<(u32, u32, u32)> = const { std::cell::Cell::new((0, 0, 0)) };
    /// Stage E harness: dav2d's frame-2 luma pre-filter RECON plane (prediction + residual), from
    /// the `dav2d_f2recon_capture` dump. For scoring my per-block MC+residual recon.
    pub static REF_F2RECON: RefCell<Option<Plane>> = const { RefCell::new(None) };
    /// dav2d pre-filter frame-2 CHROMA recon (prediction + residual), U/V planes (216×120),
    /// from `dav2d_f2reconc_capture` → dav_f2reconc.yuv. Oracle for the chroma Stage-E recon.
    pub static REF_F2RECONC: RefCell<[Option<Plane>; 2]> = const { RefCell::new([None, None]) };
    /// (u_ok, v_ok, total) scoreboard for the frame-2 chroma RECON.
    pub static INTER_SCORE_RC: std::cell::Cell<(u32, u32, u32)> = const { std::cell::Cell::new((0, 0, 0)) };
    /// Stage E luma recon (pred+residual) scoreboard: (#luma blocks bit-exact, #total).
    pub static INTER_SCORE_R: std::cell::Cell<(u32, u32)> = const { std::cell::Cell::new((0, 0)) };
    /// Stage E: the frame-2 base Y-AC q-index, for the inter residual dequant in decode_leaf (which
    /// has no `f`). Set from `yac` at the frame-2 SB-loop setup.
    pub static F2_YAC: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Per-plane DC quantizer step [luma, u, v] = `dq_lookup(clip(yac + Xdc_delta))` (dav2d
    /// decode.c:134 `dq[.][.][0]`). The DC coefficient (index 0) dequantizes with THIS, not the AC
    /// step `dq_lookup(yac)` — AV2 signals a separate DC delta (`ydc_delta` etc.). Set at frame-hdr
    /// parse; 0 = "unset, fall back to the AC step". Fixes the DC-off-by-1 on frames with a DC delta.
    pub static F2_DCQ: std::cell::Cell<[u32; 3]> = const { std::cell::Cell::new([0, 0, 0]) };
    /// Per-plane AC quantizers `dq_lookup(clip(yac + {0, uac_delta, vac_delta}))` — the chroma
    /// AC dequant must honor the frame's uac/vac deltas (they are nonzero on some streams).
    pub static F2_ACQ: std::cell::Cell<[u32; 3]> = const { std::cell::Cell::new([0, 0, 0]) };
    /// The frame's per-plane q deltas [ydc, udc, vdc, uac, vac] (obu quant section) — kept so the
    /// per-SB delta-q read can recompute F2_DCQ/F2_ACQ from the NEW qindex (dav init_quant_tables).
    pub static F2_QDELTAS: std::cell::Cell<[i32; 5]> = const { std::cell::Cell::new([0; 5]) };
    /// Decode-order frame counter (incremented at each frame's LAST_QIDX reset) - probe gating.
    pub static DECODE_FRAME_N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Seq-header `inter_ddt`: when set, inter blocks remap (flip)adst→(f)ddt in the itx
    /// via `txtp += txtp & tx_ddt_mask[tx]` (dav2d recon_tmpl.c:2713).
    pub static INTER_DDT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Per-4px interpolation filter of each decoded frame-2 inter block, indexed
    /// `(by4&63)*128 + (bx4&127)` (mirrors the refmvs grid). Read by the sub-8×8 shared-chroma
    /// per-sub-block MC (dav uses each sub-block's own `b2->filter`).
    /// Per-4px interpolation-filter grid. WAS a fixed 128x64 array indexed with WRAPPING masks
    /// (`& 127` / `& 63`) — correct only while the frame fit in 512x256 4px units, so a >512px
    /// frame aliased its own filter values (the >512px correctness gap). Now a frame-sized Vec
    /// with an explicit stride and no masking.
    pub static FILTER_GRID: RefCell<Vec<u8>> = RefCell::new(vec![0u8; 128 * 64]);
    pub static FILTER_GRID_STRIDE: std::cell::Cell<usize> = const { std::cell::Cell::new(128) };
    /// True only during the scored frame-1 SB-loop pass; the later "full recursion" re-runs the
    /// recon (a scaffold) and must NOT overwrite the frame buffer or double-count the scoreboard.
    pub static RECON_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Temporary per-block debug gate.
    pub static DBG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark the scored recon pass over (called after the frame dump) — later scaffold passes no-op.
pub fn end_recon_pass() {
    RECON_ACTIVE.with(|a| a.set(false));
}

/// Load the dav2d pre-filter luma reference (`w`×`h`, 8-bit raw) for per-block isolation scoring.
pub fn load_ref_luma(path: &str, w: usize, h: usize) {
    if let Ok(bytes) = std::fs::read(path) {
        if bytes.len() >= w * h {
            let mut p = Plane::alloc(w, h);
            for i in 0..w * h {
                if !crate::av2_recon::work_tick("frm:598") { break; }
                p.px[i] = bytes[i] as u16;
            }
            REF_LUMA.with(|r| *r.borrow_mut() = Some(p));
            REF_SCORE.with(|s| s.set((0, 0)));
            RECON_ACTIVE.with(|a| a.set(true));
        }
    }
}

/// Load the dav2d pre-filter chroma references (U, V) from a whole-frame I420 dump `path`
/// (frame-1 chroma at byte `luma_sz` for U, `luma_sz+cw*ch` for V), `cw`×`ch` each.
pub fn load_ref_chroma(path: &str, cw: usize, ch: usize, luma_sz: usize) {
    if let Ok(bytes) = std::fs::read(path) {
        let csz = cw * ch;
        if bytes.len() >= luma_sz + 2 * csz {
            let mut pu = Plane::alloc(cw, ch);
            let mut pv = Plane::alloc(cw, ch);
            for i in 0..csz {
                if !crate::av2_recon::work_tick("frm:616") { break; }
                pu.px[i] = bytes[luma_sz + i] as u16;
                pv.px[i] = bytes[luma_sz + csz + i] as u16;
            }
            REF_CHROMA.with(|r| *r.borrow_mut() = [Some(pu), Some(pv)]);
            REF_CHROMA_SCORE.with(|s| s.set((0, 0, 0)));
        }
    }
}

/// Load the frame-1 reconstructed+FILTERED reference (Y,U,V) from a whole-frame I420 dump
/// `path` (the normal dav2d output; frame 1 = first `luma_sz + 2*csz` bytes). This is the
/// Stage-D inter-prediction reference. `w`×`h` luma, `cw`×`ch` chroma.
pub fn load_ref_frame1(path: &str, w: usize, h: usize) {
    if let Ok(bytes) = std::fs::read(path) {
        let (cw, ch) = ((w + 1) >> 1, (h + 1) >> 1);
        let (luma_sz, csz) = (w * h, cw * ch);
        if bytes.len() >= luma_sz + 2 * csz {
            let mut py = Plane::alloc(w, h);
            let mut pu = Plane::alloc(cw, ch);
            let mut pv = Plane::alloc(cw, ch);
            for i in 0..luma_sz {
                if !crate::av2_recon::work_tick("frm:637") { break; }
                py.px[i] = bytes[i] as u16;
            }
            for i in 0..csz {
                if !crate::av2_recon::work_tick("frm:640") { break; }
                pu.px[i] = bytes[luma_sz + i] as u16;
                pv.px[i] = bytes[luma_sz + csz + i] as u16;
            }
            REF_FRAME1.with(|r| *r.borrow_mut() = [Some(py), Some(pu), Some(pv)]);
            INTER_SCORE.with(|s| s.set((0, 0)));
        }
    }
}

/// Stash mine's just-decoded, fully-filtered frame-1 FRAME planes so the inter frame can
/// reference them — a true standalone decode with no external reference file.
/// Clear every cross-frame plane/config stash (new-sequence reset).
pub fn reset_stream_state() {
    STASH_FRAME1.with(|s| *s.borrow_mut() = [None, None, None]);
    REF_PICS.with(|r| *r.borrow_mut() = std::array::from_fn(|_| None));
    CCSO_SLOT_MAP.with(|m| *m.borrow_mut() = std::array::from_fn(|_| None));
    CCSO_SLOT_CFG.with(|c| *c.borrow_mut() = std::array::from_fn(|_| None));
    REF_LUMA.with(|r| *r.borrow_mut() = None);
    REF_CHROMA.with(|r| *r.borrow_mut() = [None, None]);
}

pub fn stash_decoded_frame1() {
    FRAME.with(|fr| {
        let f = fr.borrow();
        let planes: [Option<Plane>; 3] =
            std::array::from_fn(|i| if f.pl[i].w == 0 { None } else { Some(f.pl[i].clone()) });
        STASH_FRAME1.with(|s| *s.borrow_mut() = planes);
    });
    // MREFDUMP=<prefix>: write each decoded frame's final recon (decode order) as
    // <prefix>_<n>.yuv — the mirror of avm's PREFDUMP for per-coded-frame recon diffs.
    if let Ok(prefix) = std::env::var("MREFDUMP") {
        use std::io::Write;
        thread_local! { static DUMP_N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
        let n = DUMP_N.with(|c| { let v = c.get(); c.set(v + 1); v });
        FRAME.with(|fr| {
            let f = fr.borrow();
            if f.pl[0].w == 0 { return; }
            let hbd = f.bitdepth_max > 255;
            if let Ok(mut fp) = std::fs::File::create(format!("{prefix}_{n}.yuv")) {
                for pl in 0..3 {
                    let p = &f.pl[pl];
                    if p.w == 0 { continue; }
                    for y in 0..p.h {
                        if !crate::av2_recon::work_tick("frm:683") { break; }
                        for x in 0..p.w {
                            if !crate::av2_recon::work_tick("frm:684") { break; }
                            let v = p.px[y * p.stride + x];
                            if hbd {
                                let _ = fp.write_all(&(v as u16).to_le_bytes());
                            } else {
                                let _ = fp.write_all(&[(v as u8)]);
                            }
                        }
                    }
                }
            }
        });
    }
}

/// Commit the just-decoded + fully-filtered FRAME planes into every reference-picture slot
/// selected by `refresh_frame_flags` (dav2d ref-buffer update). Parallel to REF_SLOTS metadata.
pub fn update_ref_pics(refresh: u32) {
    FRAME.with(|fr| {
        let f = fr.borrow();
        if f.pl[0].w == 0 {
            return;
        }
        let planes: [Plane; 3] = std::array::from_fn(|i| f.pl[i].clone());
        REF_PICS.with(|r| {
            let mut slots = r.borrow_mut();
            for i in 0..8 {
                if refresh & (1 << i) != 0 {
                    slots[i] = Some(planes.clone());
                }
            }
        });
    });
}

thread_local! {
    /// The ref-picture slot currently loaded into REF_FRAME1 (dav2d `b->ref.ref[0]` resolved to a
    /// slot). `ensure_ref1_slot` reloads only when the block's ref changes, so a mostly-primary
    /// frame reloads rarely. Sentinel `usize::MAX` = "unknown, force reload".
    pub static CUR_REF1_SLOT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Point REF_FRAME1 (the inter frame's primary reference) at reference-picture slot `slot`
/// (= `refidx[0]`). Returns true if that slot holds a decoded picture.
pub fn load_ref_frame1_from_slot(slot: usize) -> bool {
    REF_PICS.with(|r| {
        let pics = r.borrow();
        if let Some(p) = pics.get(slot).and_then(|s| s.as_ref()) {
            REF_FRAME1.with(|rf| *rf.borrow_mut() = [Some(p[0].clone()), Some(p[1].clone()), Some(p[2].clone())]);
            INTER_SCORE.with(|sc| sc.set((0, 0)));
            CUR_REF1_SLOT.with(|c| c.set(slot));
            true
        } else {
            false
        }
    })
}

/// Ensure REF_FRAME1 holds ref-picture slot `slot` (the block's `refidx[ref0]`) before its MC.
/// A no-op when that slot is already loaded, so single-reference frames never re-clone; a B-frame
/// block picking a non-primary ref triggers one reload. Falls back silently if the slot is empty.
pub fn ensure_ref1_slot(slot: usize) {
    if CUR_REF1_SLOT.with(|c| c.get()) == slot {
        return;
    }
    load_ref_frame1_from_slot(slot);
}

/// Point REF_FRAME1 (the inter frame's reference) at mine's stashed decoded frame 1, if present.
/// Returns true on success so the caller can skip the external-reference-file fallback.
pub fn load_ref_frame1_from_stash() -> bool {
    STASH_FRAME1.with(|s| {
        let st = s.borrow();
        if st[0].is_some() {
            REF_FRAME1.with(|r| *r.borrow_mut() = st.clone());
            INTER_SCORE.with(|sc| sc.set((0, 0)));
            true
        } else {
            false
        }
    })
}

/// Allocate/clear the frame buffer for a `w`×`h` (luma) 4:2:0 frame.
/// `bdmax` = (1 << bitdepth) − 1 (255 for 8-bit, 1023 for 10-bit).
pub fn reset_frame(w: usize, h: usize, yac: u8, edge_filter: bool, ibp: bool, bdmax: i32) {
    FRAME_NO.with(|c| c.set(c.get() + 1));
    // Size the per-4px filter grid to THIS frame (no wrapping masks — see FILTER_GRID).
    {
        let (giw4, gih4) = ((w + 3) / 4, (h + 3) / 4);
        let stride = giw4.max(128);
        FILTER_GRID_STRIDE.with(|c| c.set(stride));
        FILTER_GRID.with(|g| {
            let mut v = g.borrow_mut();
            v.clear();
            v.resize(stride * gih4.max(64), 0);
        });
    }
    let (ss_hor, ss_ver) = SS.with(|c| c.get());
    let (cw, ch) = ((w + ss_hor as usize) >> ss_hor, (h + ss_ver as usize) >> ss_ver);
    // SB-aligned padded recon mirror (luma SB=64px, chroma SB = 64>>ss_hor px) for the
    // edge-block gather.

    FRAME.with(|f| {
        let mut f = f.borrow_mut();
        f.pl[0] = Plane::alloc(w, h);
        f.pl[1] = Plane::alloc(cw, ch);
        f.pl[2] = Plane::alloc(cw, ch);
        f.yac = yac;
        f.bitdepth_max = bdmax;
        BDMAX.with(|b| b.set(bdmax));
        f.edge_filter = edge_filter;
        f.ibp = ibp;
        f.iw4 = w >> 2;
        f.ih4 = h >> 2;
        f.mi_coded = vec![false; 1024];
        f.mi_coded_c = vec![false; 1024];
        f.mi_coded_sb = (usize::MAX, usize::MAX);
        let n = (w >> 2) * (h >> 2);
        f.joint = vec![0u8; n];
        f.db_lw = vec![0u8; n];
        f.db_lh = vec![0u8; n];
        f.db_left = vec![false; n];
        f.db_top = vec![false; n];
        f.ciw4 = f.pl[1].w >> 2;
        f.cih4 = f.pl[1].h >> 2;
        let cn = f.ciw4 * f.cih4;
        f.cdb_lw = vec![0u8; cn];
        f.cdb_lh = vec![0u8; cn];
        f.cdb_left = vec![false; cn];
        f.cdb_top = vec![false; cn];
        f.db_spv = vec![false; n];
        f.db_sph = vec![false; n];
        f.cdb_spv = vec![false; cn];
        f.cdb_sph = vec![false; cn];
        f.sm_c = vec![0u8; cn];
        // CDEF: per-64px-SB index grid + per-8px-stripe noskip mask.
        f.cdef_sbw = f.iw4.div_ceil(16);
        let sbh = f.ih4.div_ceil(16);
        f.cdef_idx = vec![-1i8; f.cdef_sbw * sbh];
        // 0 = "unset" (frame-alloc can precede the yac parse) → sb_qidx_at falls back to yac.
        f.sb_qidx = vec![0u16; f.cdef_sbw * sbh];
        f.noskip_stripes = f.ih4.div_ceil(2);
        f.noskip = vec![false; f.noskip_stripes * f.iw4];
        f.lr_noskip = vec![false; f.iw4 * f.ih4];
        // CCSO: per-unit enable, 3 planes. Unit is TILE-ADAPTIVE (256px default;
        // avm get_ccso_unit_size_log2_adaptive_tile) — grid granularity == the unit.
        let (ccso_u4, gdf_bs4) = crate::av2_recon::FILTER_UNITS.with(|c| c.get());
        f.ccso_px_shift = (ccso_u4 as u32 * 4).trailing_zeros();
        f.ccso_col256 = f.iw4.div_ceil(ccso_u4);
        let crow256 = f.ih4.div_ceil(ccso_u4);
        f.ccso_n256 = f.ccso_col256 * crow256;
        // Save the OUTGOING (previous frame's) ccso map before clearing, so an INTER frame with
        // sb_reuse can inherit its per-SB flags (dav2d `prev_ccsomap`). Only when the grid size
        // matches (same resolution).
        if f.ccso_blk.len() == 3 * f.ccso_n256 {
            let prev = f.ccso_blk.clone();
            PREV_CCSO_MAP.with(|m| *m.borrow_mut() = prev);
        }
        f.ccso_blk = vec![false; 3 * f.ccso_n256];
        // GDF: per-block flags (128px default; TILE-ADAPTIVE to 64px — avm init_gdf).
        f.gdf_bs_px = gdf_bs4 * 4;
        let gbs = f.gdf_bs_px;
        f.gdf_blk_w = 1 + (w - 1) / gbs;
        let gbh = 1 + (h - 1) / gbs;
        f.gdf_blk = vec![false; f.gdf_blk_w * gbh];
    });
    BTYPE.with(|b| *b.borrow_mut() = vec![0u8; w * h]);
    BTYPE_WH.with(|c| c.set((w, h)));
}

/// Frame-header deblock params (dav2d `frame_hdr->deblock`), stashed during parse for the
/// filter pass. `dq_*` are the per-direction/plane q-index deltas; `level_*` are the on flags.
#[derive(Clone, Copy, Default)]
pub struct DeblockCfg {
    pub level_y0: bool,
    pub level_y1: bool,
    pub level_u: bool,
    pub level_v: bool,
    pub dq_y0: i32,
    pub dq_y1: i32,
    pub dq_u: i32,
    pub dq_v: i32,
    /// Chroma AC q-index deltas (frame_hdr->quant.{uac,vac}_delta) — feed the chroma thresholds.
    pub uac_delta: i32,
    pub vac_delta: i32,
    /// frame `deblock.sub_pu` (dav2d obu.c: lf_sub_pu bit, INTER frames when seq db_sub_pu):
    /// sub-PU-refined blocks (TIP / compound-opfl / refinemv-avg) add weak inner deblock edges
    /// on a (1<<subpu_l2)-cell cadence with thresholds >>3 (dav lf_mask.c subpu_flt_lvl +
    /// mask_subpu_edges + deblock_tmpl setup_thr `>> 3*subpu`).
    pub sub_pu: bool,
}

thread_local! {
    pub static DEBLOCK_CFG: std::cell::Cell<DeblockCfg> = const { std::cell::Cell::new(DeblockCfg {
        level_y0: false, level_y1: false, level_u: false, level_v: false,
        dq_y0: 0, dq_y1: 0, dq_u: 0, dq_v: 0, uac_delta: 0, vac_delta: 0, sub_pu: false,
    }) };
    pub static CDEF_CFG: std::cell::Cell<CdefCfg> = const { std::cell::Cell::new(CdefCfg {
        enabled: false, damping: 3, n_strengths: 1, on_skiptx: false,
        y_strength: [0; 8], uv_strength: [0; 8],
    }) };
}

/// Frame-header CCSO params for one plane (avm `CcsoInfo`). `filter_offset` is the per-class
/// signed offset LUT, indexed `(band<<4)|(cls0<<2)|cls1`. `bo_only` = band-offset only (no
/// edge classes); `ext_filter` selects the neighbour sample pair; `edge_clf` suppresses the
/// `+2` class. `quant_step` = `CCSO_QUANT_SZ[scale][quant]`.
#[derive(Clone, Default)]
pub struct CcsoPlaneCfg {
    pub enabled: bool,
    pub bo_only: bool,
    pub quant_step: i32,
    pub ext_filter: usize,
    pub edge_clf: bool,
    pub max_band_log2: u32,
    pub filter_offset: Vec<i8>,
    /// INTER frame: per-SB ccso on/off flags are INHERITED from the ref frame's ccso map
    /// (dav2d decode.c:1950 `prev_ccsomap`), not decoded. Set from the frame-2 header bit.
    pub sb_reuse: bool,
    /// The resolved REF SLOT (refidx[ccso refidx]) this plane's reuse/sb_reuse points at.
    pub reuse_slot: u8,
}

/// Frame CCSO config (3 planes). Classification reads the PRE-CDEF (post-deblock) luma;
/// the offset is added to the POST-CDEF plane (avm `ccso_frame` / decodeframe.c: ext_rec_y
/// is snapshotted before `av2_cdef_frame`).
#[derive(Clone, Default)]
pub struct CcsoCfg {
    pub enabled: bool,
    pub p: Vec<CcsoPlaneCfg>,
}

thread_local! {
    pub static CCSO_CFG: RefCell<CcsoCfg> = RefCell::new(CcsoCfg::default());
    /// Previous frame's per-SB CCSO map (`3*ccso_n256`), saved at each `reset_frame`. An INTER
    /// frame with `sb_reuse` inherits its per-SB flags from this (dav2d `prev_ccsomap`).
    pub static PREV_CCSO_MAP: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
    /// Per-ref-slot CCSO per-SB flag maps (dav2d `c->refs[slot].ccsomap`): saved at
    /// update_ref_slots for every refreshed slot; an sb_reuse plane inherits from
    /// CCSO_SLOT_MAP[refidx[ccso.p[p].refidx]].
    pub static CCSO_SLOT_MAP: RefCell<[Option<Vec<bool>>; 8]> = RefCell::new(std::array::from_fn(|_| None));
    /// Per-ref-slot CCSO frame configs (dav2d refhdr->ccso): `reuse` copies a plane's config
    /// (bo_only..filter_off) from the slot's saved cfg.
    pub static CCSO_SLOT_CFG: RefCell<[Option<CcsoCfg>; 8]> = RefCell::new(std::array::from_fn(|_| None));
    /// Debug: per-luma-pixel block-type map (0=unwritten, 1=inter, 2=intra, 3=intrabc, 4=bawp),
    /// for categorizing the frame-2 recon misses. Sized `w*h`, cleared at `reset_frame`.
    pub static BTYPE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// Luma frame (w, h) for `mark_btype` — read without touching FRAME (which may be borrowed).
    pub static BTYPE_WH: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    /// Debug: which frame we're reconstructing (incremented at reset_frame). For frame-tagged probes.
    pub static FRAME_NO: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// The current BAWP block's derived LUMA alpha — the chroma BAWP morph reuses it (dav recon
    /// 2852: chroma `alpha = have_left||have_above ? bawp[0].alpha : 256`). Set by the luma morph.
    pub static BAWP_ALPHA: std::cell::Cell<i32> = const { std::cell::Cell::new(256) };
    pub static GDF_CFG: std::cell::Cell<GdfCfg> = const { std::cell::Cell::new(GdfCfg {
        enabled: false, mode: 0, qp_idx: 0, scale_idx: 0, block_size: 128,
    }) };
}

/// Debug: after a block's luma is written to FRAME, compare it to dav's pre-filter recon
/// (REF_F2RECON) and print the first mismatch — in DECODE order — so the true cascade ROOT
/// (earliest wrong block, any type) is visible. `tag` labels the block type.
pub fn dbg_block_miss(bx4: usize, by4: usize, bw4: usize, bh4: usize, tag: &str) {
    if std::env::var("RECONDBG").is_err() {
        return;
    }
    REF_F2RECON.with(|r| {
        if let Some(rp) = r.borrow().as_ref() {
            FRAME.with(|fr| {
                let f = fr.borrow();
                if f.pl[0].w == 0 { return; }
                let (w, h) = (bw4 * 4, bh4 * 4);
                for yy in 0..h {
                    if !crate::av2_recon::work_tick("frm:952") { break; }
                    for xx in 0..w {
                        if !crate::av2_recon::work_tick("frm:953") { break; }
                        let (px, py) = (bx4 * 4 + xx, by4 * 4 + yy);
                        if px < rp.w && py < rp.h {
                            let m = f.pl[0].px[py * f.pl[0].stride + px] as i32;
                            if m != rp.at(px, py) {
                                crate::dlog!("BLKMISS[{tag}] fn={} ({bx4},{by4}) w={w} h={h} at({xx},{yy}) mine={m} dav={}", DECODE_FRAME_N.with(|c| c.get()), rp.at(px, py));
                                return;
                            }
                        }
                    }
                }
            });
        }
    });
}

/// Decode-order CHROMA block-miss: compare a just-reconstructed chroma block (pixel origin
/// `px0,py0`, size `w×h`) to dav's frame-2 chroma recon (`REF_F2RECONC`) and print the FIRST
/// mismatching pixel — the causal root, since it fires in decode order before any cascade.
pub fn dbg_block_miss_c(px0: usize, py0: usize, w: usize, h: usize, tag: &str) {
    if std::env::var("CRECONDBG").is_err() {
        return;
    }
    REF_F2RECONC.with(|r| {
        let rb = r.borrow();
        FRAME.with(|fr| {
            let f = fr.borrow();
            if f.pl[1].w == 0 { return; }
            for (pl, rpo) in rb.iter().enumerate() {
                let Some(rp) = rpo.as_ref() else { continue };
                for yy in 0..h {
                    if !crate::av2_recon::work_tick("frm:983") { break; }
                    for xx in 0..w {
                        if !crate::av2_recon::work_tick("frm:984") { break; }
                        let (px, py) = (px0 + xx, py0 + yy);
                        if px < rp.w && py < rp.h {
                            let m = f.pl[pl + 1].px[py * f.pl[pl + 1].stride + px] as i32;
                            if m != 0 && m != rp.at(px, py) {
                                crate::dlog!("CBLKMISS[{tag}] pl={pl} ({px0},{py0}) w={w} h={h} at({xx},{yy}) mine={m} dav={}", rp.at(px, py));
                                return;
                            }
                        }
                    }
                }
            }
        });
    });
}


/// Tag the luma block-type map over a `w4`×`h4` block at `(bx4, by4)` (4px units) with `t`.
/// Reads dims from BTYPE_WH (NOT FRAME) so it is safe to call while FRAME is borrowed.
pub fn mark_btype(bx4: usize, by4: usize, bw4: usize, bh4: usize, t: u8) {
    let (fw, fh) = BTYPE_WH.with(|c| c.get());
    if fw == 0 { return; }
    BTYPE.with(|b| {
        let mut m = b.borrow_mut();
        if m.len() != fw * fh { return; }
        for y in by4 * 4..((by4 + bh4) * 4).min(fh) {
            if !crate::av2_recon::work_tick("frm:1031") { break; }
            for x in bx4 * 4..((bx4 + bw4) * 4).min(fw) {
                if !crate::av2_recon::work_tick("frm:1032") { break; }
                m[y * fw + x] = t;
            }
        }
    });
}

/// Frame-header GDF params (avm `GdfInfo`). `mode` 1 = all blocks on, 2 = per-block flags;
/// `qp_idx`/`scale_idx` = the parsed picture qp-offset / scale; `block_size` = GDF block px.
#[derive(Clone, Copy)]
pub struct GdfCfg {
    pub enabled: bool,
    pub mode: i32,
    pub qp_idx: i32,
    pub scale_idx: i32,
    pub block_size: i32,
}

/// Frame-header CDEF params (dav2d `frame_hdr->cdef`), stashed during parse. `y_strength`/
/// `uv_strength` are the per-index packed levels (`>>2` = primary, `&3` = secondary); the
/// per-SB `cdef_idx` selects one. `on_skiptx` filters all blocks (ignores the skip mask).
#[derive(Clone, Copy)]
pub struct CdefCfg {
    pub enabled: bool,
    pub damping: i32,
    pub n_strengths: usize,
    pub on_skiptx: bool,
    pub y_strength: [i32; 8],
    pub uv_strength: [i32; 8],
}

/// Deblock one plane of `none` (pre-filter) with the given grids + thresholds, score vs
/// `deblk` (post-deblock oracle) over the byte window `[off, off+w*h)`, print stats.
#[allow(clippy::too_many_arguments)]
fn verify_plane(
    tag: &str,
    none: &[u8],
    deblk: &[u8],
    off: usize,
    iw4: usize,
    ih4: usize,
    w: usize,
    h: usize,
    grids: (&[u8], &[u8], &[bool], &[bool]),
    thr: (i32, i32, i32, i32),
    bdmax: i32,
    max_width: &[i32],
    band_cells: usize,
    band_neg_cap: i32,
) {
    use crate::av2_deblock::deblock_plane;
    let n = w * h;
    if off + n > none.len() || off + n > deblk.len() {
        crate::dlog!("[C1 {tag}] oracle too short");
        return;
    }
    let mut buf: Vec<i32> = (0..n).map(|i| none[off + i] as i32).collect();
    let (lw, lh, left, top) = grids;
    let (q_v, s_v, q_h, s_h) = thr;
    deblock_plane(&mut buf, iw4, ih4, w, lw, lh, left, top, q_v, s_v, q_h, s_h, None, &[], &[], bdmax, max_width, band_cells, band_neg_cap, true, true, &[]);
    if tag == "DEBLOCK-Y" {
        let bytes: Vec<u8> = buf.iter().map(|&v| v.clamp(0, 255) as u8).collect();
        let _ = std::fs::write(&crate::av2_recon::cap_path("mine_f2deblk_y.bin"), &bytes);
    }
    let mut mism = 0u32;
    let mut dav_changed = 0u32;
    let mut first: Vec<(usize, usize, i32, i32)> = Vec::new();
    for i in 0..n {
        if !crate::av2_recon::work_tick("frm:1099") { break; }
        let mine = buf[i].clamp(0, 255) as u8;
        let want = deblk[off + i];
        if want != none[off + i] {
            dav_changed += 1;
        }
        if mine != want {
            mism += 1;
            if first.len() < 12 {
                first.push((i % w, i / w, mine as i32, want as i32));
            }
        }
    }
    crate::dlog!(
        "[C1 {tag}] {}/{} px exact ({:.3}%); {mism} mismatch, dav changed {dav_changed}",
        n as u32 - mism, n, 100.0 * (n as u32 - mism) as f64 / n as f64
    );
    for (x, y, m, wv) in &first {
        crate::dlog!("    ({x},{y}) mine={m} want={wv}");
    }
}

/// Isolation harness for C1: deblock the dav2d pre-filter frame (`none_path`) with our
/// edge grids + derived thresholds, and score against dav2d's post-deblock frame
/// (`deblk_path`). Reads block geometry from the current FRAME grids (populated during
/// recon) and the qindex + deltas from DEBLOCK_CFG. Verifies luma + both chroma planes.
pub fn run_deblock_verify(none_path: &str, deblk_path: &str) {
    use crate::av2_deblock::{deblock_quant_thr, deblock_side_thr, MAX_WIDTH_UV_TBL, MAX_WIDTH_Y_TBL};
    let (iw4, ih4, w, h, cw, ch, ciw4, cih4, yac, bdmax,
         db_lw, db_lh, db_left, db_top, cdb_lw, cdb_lh, cdb_left, cdb_top) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.iw4, f.ih4, f.pl[0].w, f.pl[0].h, f.pl[1].w, f.pl[1].h, f.ciw4, f.cih4,
         f.yac as i32, f.bitdepth_max,
         f.db_lw.clone(), f.db_lh.clone(), f.db_left.clone(), f.db_top.clone(),
         f.cdb_lw.clone(), f.cdb_lh.clone(), f.cdb_left.clone(), f.cdb_top.clone())
    });
    if w == 0 || db_lw.is_empty() {
        return;
    }
    let cfg = DEBLOCK_CFG.with(|c| c.get());
    if !cfg.level_y0 && !cfg.level_y1 {
        crate::dlog!("[C1] deblock disabled (level_y0={} level_y1={})", cfg.level_y0, cfg.level_y1);
        return;
    }
    let none = match std::fs::read(none_path) {
        Ok(b) => b,
        _ => { crate::dlog!("[C1] could not read {none_path}"); return; }
    };
    let deblk = match std::fs::read(deblk_path) {
        Ok(b) => b,
        _ => { crate::dlog!("[C1] could not read {deblk_path}"); return; }
    };

    // Luma: dir 0 = cols (vertical edges) uses dq_y0, dir 1 = rows uses dq_y1.
    let q_thr_v = deblock_quant_thr(yac + 8 * cfg.dq_y0, 0);
    let side_thr_v = deblock_side_thr(yac + 8 * cfg.dq_y0, 0);
    let q_thr_h = deblock_quant_thr(yac + 8 * cfg.dq_y1, 0);
    let side_thr_h = deblock_side_thr(yac + 8 * cfg.dq_y1, 0);
    crate::dlog!(
        "[C1] qidx={yac} dq_y0={} dq_y1={} uac={} vac={} → yV(q={q_thr_v},s={side_thr_v}) yH(q={q_thr_h},s={side_thr_h})",
        cfg.dq_y0, cfg.dq_y1, cfg.uac_delta, cfg.vac_delta
    );
    verify_plane(
        "DEBLOCK-Y", &none, &deblk, 0, iw4, ih4, w, h,
        (&db_lw, &db_lh, &db_left, &db_top),
        (q_thr_v, side_thr_v, q_thr_h, side_thr_h),
        bdmax, &MAX_WIDTH_Y_TBL, 16, 6,
    );

    // Chroma: U/V use one threshold per plane (both directions), from the ac q-index deltas.
    let luma_sz = w * h;
    let chroma_sz = cw * ch;
    if cfg.level_u {
        let uac = yac + cfg.uac_delta + 8 * cfg.dq_u;
        let (q, s) = (deblock_quant_thr(uac, 0), deblock_side_thr(uac, 0));
        verify_plane(
            "DEBLOCK-U", &none, &deblk, luma_sz, ciw4, cih4, cw, ch,
            (&cdb_lw, &cdb_lh, &cdb_left, &cdb_top),
            (q, s, q, s), bdmax, &MAX_WIDTH_UV_TBL, 8, 2,
        );
    }
    if cfg.level_v {
        let vac = yac + cfg.vac_delta + 8 * cfg.dq_v;
        let (q, s) = (deblock_quant_thr(vac, 0), deblock_side_thr(vac, 0));
        verify_plane(
            "DEBLOCK-V", &none, &deblk, luma_sz + chroma_sz, ciw4, cih4, cw, ch,
            (&cdb_lw, &cdb_lh, &cdb_left, &cdb_top),
            (q, s, q, s), bdmax, &MAX_WIDTH_UV_TBL, 8, 2,
        );
    }
}

/// Debug: dump mine's per-4px luma edge-width map in dav's `dav_f2wmap.bin` format
/// (`i8[2][60*108]`: dir 0 = vertical edges, dir 1 = horizontal edges; value = intended
/// max_width_pos). Compare against dav's dump to localize exactly which edges/widths differ.
pub fn dump_frame2_luma_wmap(path: &str) {
    use crate::av2_deblock::MAX_WIDTH_Y_TBL as MW;
    let (iw4, ih4, db_lw, db_lh, db_left, db_top) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.iw4, f.ih4, f.db_lw.clone(), f.db_lh.clone(), f.db_left.clone(), f.db_top.clone())
    });
    if db_lw.is_empty() { return; }
    let mut wmap = vec![0i8; 2 * 60 * 108];
    // dir 0: vertical edges (x4>=1, db_left). level = min(this cell, left cell).
    for y4 in 0..ih4.min(60) {
        if !crate::av2_recon::work_tick("frm:1203") { break; }
        for x4 in 1..iw4.min(108) {
            if !crate::av2_recon::work_tick("frm:1204") { break; }
            let cell = y4 * iw4 + x4;
            if db_left[cell] {
                let level = db_lw[cell].min(db_lw[cell - 1]) as usize;
                wmap[y4 * 108 + x4] = MW[level] as i8;
            }
        }
    }
    // dir 1: horizontal edges (y4>=1, db_top). level = min(this cell, above cell).
    for y4 in 1..ih4.min(60) {
        if !crate::av2_recon::work_tick("frm:1213") { break; }
        for x4 in 0..iw4.min(108) {
            if !crate::av2_recon::work_tick("frm:1214") { break; }
            let cell = y4 * iw4 + x4;
            if db_top[cell] {
                let level = db_lh[cell].min(db_lh[cell - iw4]) as usize;
                wmap[6480 + y4 * 108 + x4] = MW[level] as i8;
            }
        }
    }
    let bytes: Vec<u8> = wmap.iter().map(|&v| v as u8).collect();
    let _ = std::fs::write(path, &bytes);
}

/// Score a filtered i32 plane `out` (window `[off, off+n)`) vs the `oracle` bytes and print
/// stats; `input` is the pre-filter frame (for the "dav changed" Δ count).
fn score_filtered(tag: &str, out: &[i32], oracle: &[u8], input: &[u8], off: usize, w: usize, h: usize) {
    let n = w * h;
    let mut mism = 0u32;
    let mut dav_changed = 0u32;
    let mut first: Vec<(usize, usize, i32, i32)> = Vec::new();
    for i in 0..n {
        if !crate::av2_recon::work_tick("frm:1233") { break; }
        let mine = out[i].clamp(0, 255) as u8;
        let want = oracle[off + i];
        if want != input[off + i] {
            dav_changed += 1;
        }
        if mine != want {
            mism += 1;
            if first.len() < 12 {
                first.push((i % w, i / w, mine as i32, want as i32));
            }
        }
    }
    crate::dlog!(
        "[{tag}] {}/{} px exact ({:.3}%); {mism} mismatch, dav changed {dav_changed}",
        n as u32 - mism, n, 100.0 * (n as u32 - mism) as f64 / n as f64
    );
    for (x, y, m, wv) in &first {
        crate::dlog!("    ({x},{y}) mine={m} want={wv}");
    }
}

/// Isolation harness for C2: CDEF the dav2d post-deblock frame (`deblk_path`, the CDEF
/// input) with our per-SB indices + noskip mask + frame strengths, score vs dav2d's
/// post-CDEF frame (`cdef_path`). Luma plane (chroma added once luma is bit-exact).
pub fn run_cdef_verify(deblk_path: &str, cdef_path: &str) {
    use crate::av2_filter::{adjust_strength, cdef_block, cdef_find_dir};
    let (iw4, ih4, w, h, cw, ch, bdmax, cdef_idx, cdef_sbw, noskip) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.iw4, f.ih4, f.pl[0].w, f.pl[0].h, f.pl[1].w, f.pl[1].h,
         f.bitdepth_max, f.cdef_idx.clone(), f.cdef_sbw, f.noskip.clone())
    });
    if w == 0 || cdef_idx.is_empty() {
        return;
    }
    let cfg = CDEF_CFG.with(|c| c.get());
    if !cfg.enabled {
        crate::dlog!("[C2] cdef disabled");
        return;
    }
    let bd_min8 = 0i32; // 8-bit
    let damping = cfg.damping + bd_min8;
    crate::dlog!(
        "[C2] damping={damping} n_strengths={} on_skiptx={} y_str={:?} uv_str={:?}",
        cfg.n_strengths, cfg.on_skiptx, &cfg.y_strength[..cfg.n_strengths], &cfg.uv_strength[..cfg.n_strengths]
    );
    let deblk = match std::fs::read(deblk_path) {
        Ok(b) => b,
        _ => { crate::dlog!("[C2] could not read {deblk_path}"); return; }
    };
    let cdef = match std::fs::read(cdef_path) {
        Ok(b) => b,
        _ => { crate::dlog!("[C2] could not read {cdef_path}"); return; }
    };
    // Guard: the dump files are 420-sized golden captures; skip when the current frame's
    // plane geometry doesn't match (e.g. a 4:2:2 decode with stale 420 files).
    let (gcw, gch) = FRAME.with(|fr| { let f = fr.borrow(); (f.pl[1].w, f.pl[1].h) });
    if deblk.len() < w * h + 2 * gcw * gch || cdef.len() < w * h + 2 * gcw * gch {
        crate::dlog!("[C2] dump files don't match current plane geometry — skipping harness");
        return;
    }
    let stride = w;
    let input: Vec<i32> = (0..w * h).map(|i| deblk[i] as i32).collect();
    let mut output = input.clone();
    // Chroma (4:2:0): each 8×8 luma block → a 4×4 chroma block at (bx4*2, by4*2) chroma-px.
    let luma_sz = w * h;
    let chroma_sz = cw * ch;
    let cin_u: Vec<i32> = (0..chroma_sz).map(|i| deblk[luma_sz + i] as i32).collect();
    let cin_v: Vec<i32> = (0..chroma_sz).map(|i| deblk[luma_sz + chroma_sz + i] as i32).collect();
    let mut cout_u = cin_u.clone();
    let mut cout_v = cin_v.clone();
    for by4 in (0..ih4).step_by(2) {
        if !crate::av2_recon::work_tick("frm:1304") { break; }
        for bx4 in (0..iw4).step_by(2) {
            if !crate::av2_recon::work_tick("frm:1305") { break; }
            let sb = (by4 >> 4) * cdef_sbw + (bx4 >> 4);
            let ci = cdef_idx[sb];
            if ci < 0 {
                continue;
            }
            let ci = ci as usize;
            let (y_lvl, uv_lvl) = (cfg.y_strength[ci], cfg.uv_strength[ci]);
            if y_lvl == 0 && uv_lvl == 0 {
                continue;
            }
            if !cfg.on_skiptx {
                let stripe = by4 >> 1;
                let m = noskip[stripe * iw4 + bx4] || (bx4 + 1 < iw4 && noskip[stripe * iw4 + bx4 + 1]);
                if !m {
                    continue;
                }
            }
            let y_pri = (y_lvl >> 2) << bd_min8;
            let mut y_sec = y_lvl & 3;
            y_sec += (y_sec == 3) as i32;
            y_sec <<= bd_min8;
            let uv_pri = (uv_lvl >> 2) << bd_min8;
            let mut uv_sec = uv_lvl & 3;
            uv_sec += (uv_sec == 3) as i32;
            uv_sec <<= bd_min8;
            let (have_top, have_bottom) = (by4 > 0, by4 + 2 < ih4);
            let (have_left, have_right) = (bx4 > 0, bx4 + 2 < iw4);
            let in_off = (by4 * 4) * stride + bx4 * 4;
            let (dir, var) = if y_pri > 0 || uv_pri > 0 {
                cdef_find_dir(&input, in_off, stride, bdmax)
            } else {
                (0, 0)
            };
            // Luma (8×8).
            if y_pri > 0 {
                let adj = adjust_strength(y_pri, var);
                if adj > 0 || y_sec > 0 {
                    cdef_block(&mut output, in_off, stride, &input, in_off, stride, 8, 8, adj, y_sec, dir, damping, have_top, have_bottom, have_left, have_right, bdmax);
                }
            } else if y_sec > 0 {
                cdef_block(&mut output, in_off, stride, &input, in_off, stride, 8, 8, 0, y_sec, 0, damping, have_top, have_bottom, have_left, have_right, bdmax);
            }
            // Chroma (4×4) — dir passthrough (I420 uv_dir = identity); damping-1.
            if uv_lvl != 0 {
                let uvdir = if uv_pri > 0 { dir } else { 0 };
                let cin_off = (by4 * 2) * cw + bx4 * 2;
                cdef_block(&mut cout_u, cin_off, cw, &cin_u, cin_off, cw, 4, 4, uv_pri, uv_sec, uvdir, damping - 1, have_top, have_bottom, have_left, have_right, bdmax);
                cdef_block(&mut cout_v, cin_off, cw, &cin_v, cin_off, cw, 4, 4, uv_pri, uv_sec, uvdir, damping - 1, have_top, have_bottom, have_left, have_right, bdmax);
            }
        }
    }
    score_filtered("C2 CDEF-Y", &output, &cdef, &deblk, 0, w, h);
    score_filtered("C2 CDEF-U", &cout_u, &cdef, &deblk, luma_sz, cw, ch);
    score_filtered("C2 CDEF-V", &cout_v, &cdef, &deblk, luma_sz + chroma_sz, cw, ch);
}

/// Isolation harness for C3: apply CCSO to the dav2d post-CDEF frame (`cdef_path`), classifying
/// from the PRE-CDEF (post-deblock) luma (`deblk_path`, avm `ext_rec_y`); score vs dav2d's
/// post-CCSO frame (`ccso_path`). Verifies all three planes.
pub fn run_ccso_verify(deblk_path: &str, cdef_path: &str, ccso_path: &str) {
    use crate::av2_filter::{ccso_score, CCSO_POS};
    let ccso_sh = FRAME.with(|fr| fr.borrow().ccso_px_shift);
    let (w, h, cw, ch, ccso_blk, ccso_col256, ccso_n256) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.pl[0].w, f.pl[0].h, f.pl[1].w, f.pl[1].h, f.ccso_blk.clone(), f.ccso_col256, f.ccso_n256)
    });
    if w == 0 {
        return;
    }
    let cfg = CCSO_CFG.with(|c| c.borrow().clone());
    if !cfg.enabled {
        crate::dlog!("[C3] ccso disabled");
        return;
    }
    let deblk = match std::fs::read(deblk_path) {
        Ok(b) => b,
        _ => { crate::dlog!("[C3] could not read {deblk_path}"); return; }
    };
    let cdef = match std::fs::read(cdef_path) {
        Ok(b) => b,
        _ => { crate::dlog!("[C3] could not read {cdef_path}"); return; }
    };
    let ccso = match std::fs::read(ccso_path) {
        Ok(b) => b,
        _ => { crate::dlog!("[C3] could not read {ccso_path}"); return; }
    };
    let need = w * h + 2 * cw * ch;
    if deblk.len() < need || cdef.len() < need || ccso.len() < need {
        crate::dlog!("[C3] dump files don't match current plane geometry — skipping harness");
        return;
    }
    let luma_sz = w * h;
    let chroma_sz = cw * ch;
    // Classification luma = post-deblock (pre-CDEF) luma, clamp-to-edge padded (via coord clamp).
    let dluma: Vec<i32> = (0..luma_sz).map(|i| deblk[i] as i32).collect();
    let (hi, wi) = (h as i32, w as i32);
    let cl = |v: i32, m: i32| v.clamp(0, m - 1);
    for pl in 0..3 {
        let pc = &cfg.p[pl];
        let (pw, ph, ss_hor, ss_ver, off) = match pl {
            0 => (w, h, 0u32, 0u32, 0usize),
            1 => (cw, ch, 1, 1, luma_sz),
            _ => (cw, ch, 1, 1, luma_sz + chroma_sz),
        };
        let mut out: Vec<i32> = (0..pw * ph).map(|i| cdef[off + i] as i32).collect();
        crate::dlog!(
            "[C3 cfg] pl={pl} en={} bo_only={} ext={} edge_clf={} q={} mbl2={}",
            pc.enabled, pc.bo_only, pc.ext_filter, pc.edge_clf, pc.quant_step, pc.max_band_log2
        );
        if pc.enabled {
            let single_band = pc.max_band_log2 == 0;
            let shift = 8u32.saturating_sub(pc.max_band_log2);
            let (dy, dx) = (CCSO_POS[pc.ext_filter][0] as i32, CCSO_POS[pc.ext_filter][1] as i32);
            let q = pc.quant_step;
            for py in 0..ph {
                if !crate::av2_recon::work_tick("frm:1420") { break; }
                for px in 0..pw {
                    if !crate::av2_recon::work_tick("frm:1421") { break; }
                    let lx = (px << ss_hor) as i32;
                    let ly = (py << ss_ver) as i32;
                    let block = ((ly >> ccso_sh) as usize) * ccso_col256 + ((lx >> ccso_sh) as usize);
                    if !ccso_blk[pl * ccso_n256 + block] {
                        continue;
                    }
                    let center = dluma[(ly * wi + lx) as usize];
                    let band = if single_band { 0 } else { center >> shift };
                    let (cls0, cls1) = if pc.bo_only {
                        (0u32, 0u32)
                    } else {
                        let n0 = dluma[(cl(ly + dy, hi) * wi + cl(lx + dx, wi)) as usize];
                        let n1 = dluma[(cl(ly - dy, hi) * wi + cl(lx - dx, wi)) as usize];
                        (ccso_score(n0 - center, q, pc.edge_clf), ccso_score(n1 - center, q, pc.edge_clf))
                    };
                    let lut = ((band as usize) << 4) | ((cls0 as usize) << 2) | (cls1 as usize);
                    let offset = pc.filter_offset.get(lut).copied().unwrap_or(0) as i32;
                    let idx = py * pw + px;
                    out[idx] = (out[idx] + offset).clamp(0, 255);
                }
            }
        }
        let tag = ["C3 CCSO-Y", "C3 CCSO-U", "C3 CCSO-V"][pl];
        score_filtered(tag, &out, &ccso, &cdef, off, pw, ph);
    }
}

/// Isolation harness for C4: apply GDF (luma-only) to the dav2d post-CCSO luma (`ccso_path`),
/// score vs dav2d's post-GDF luma (`all_path`, loop-restoration off → all == post-GDF).
pub fn run_gdf_verify(ccso_path: &str, all_path: &str, deblk_path: &str) {
    let (w, h, yac, gdf_blk, _gbw) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.pl[0].w, f.pl[0].h, f.yac as i32, f.gdf_blk.clone(), f.gdf_blk_w)
    });
    if w == 0 {
        return;
    }
    let cfg = GDF_CFG.with(|c| c.get());
    if !cfg.enabled {
        crate::dlog!("[C4] gdf disabled");
        return;
    }
    crate::dlog!(
        "[C4] mode={} qp_idx={} scale_idx={} block_size={} blocks_on={}/{}",
        cfg.mode, cfg.qp_idx, cfg.scale_idx,
        FRAME.with(|fr| fr.borrow().gdf_bs_px as i32),
        gdf_blk.iter().filter(|&&b| b).count(), gdf_blk.len()
    );
    let ccso = match std::fs::read(ccso_path) {
        Ok(b) => b,
        _ => { crate::dlog!("[C4] could not read {ccso_path}"); return; }
    };
    let all = match std::fs::read(all_path) {
        Ok(b) => b,
        _ => { crate::dlog!("[C4] could not read {all_path}"); return; }
    };
    let deblk = match std::fs::read(deblk_path) {
        Ok(b) => b,
        _ => { crate::dlog!("[C4] could not read {deblk_path}"); return; }
    };
    let luma: Vec<i32> = (0..w * h).map(|i| ccso[i] as i32).collect();
    let dblk: Vec<i32> = (0..w * h).map(|i| deblk[i] as i32).collect();
    let mut out = luma.clone();
    crate::av2_gdf::gdf_filter_frame(
        &luma, &dblk, &mut out, w, h, 8, yac, cfg.mode, cfg.qp_idx, cfg.scale_idx, FRAME.with(|fr| fr.borrow().gdf_bs_px as i32), &gdf_blk, 0,
    );
    score_filtered("C4 GDF-Y", &out, &all, &ccso, 0, w, h);
}

/// Chain all 4 in-loop filters (deblock → CDEF → CCSO → GDF) on the assembled FRAME buffer and
/// write the filtered result back into `FRAME.pl[]` — the real frame-2 output path. Mirrors the
/// `run_*_verify` kernels but reads/writes FRAME (no dav2d intermediate files). The cross-filter
/// data-flow matches dav: CDEF reads post-deblock; CCSO classifies from POST-DEBLOCK luma but adds
/// to POST-CDEF; GDF filters POST-CCSO luma using the POST-DEBLOCK luma as its guided reference.
/// Debug: compare mine's in-chain intermediate (w-packed pl0/pl1/pl2) against a dav same-run
/// I420 oracle at `path` (luma w*h, U/V cw*ch). No-op if the oracle file is absent.
fn stage_cmp(tag: &str, pl0: &[i32], pl1: &[i32], pl2: &[i32], w: usize, h: usize, cw: usize, ch: usize, path: &str) {
    let o = match std::fs::read(path) { Ok(b) => b, _ => return };
    let luma = w * h;
    let chroma = cw * ch;
    if o.len() < luma + 2 * chroma { return; }
    for (pl, (buf, pw, ph, off)) in [(pl0, w, h, 0usize), (pl1, cw, ch, luma), (pl2, cw, ch, luma + chroma)].into_iter().enumerate() {
        let mut ok = 0usize;
        let mut first = None;
        let n = pw * ph;
        for i in 0..n {
            if !crate::av2_recon::work_tick("frm:1507") { break; }
            let m = buf[i].clamp(0, 255) as u8;
            if m == o[off + i] { ok += 1; } else if first.is_none() { first = Some((i % pw, i / pw, m as i32, o[off + i] as i32)); }
        }
        crate::dlog!("[{tag} pl={pl}] {ok}/{n} ({:.3}%) first-miss={first:?}", 100.0 * ok as f64 / n as f64);
    }
}

pub fn filter_frame_chain(gdf_ref_dst: usize) {
    use crate::av2_deblock::{deblock_plane, deblock_quant_thr, deblock_side_thr, MAX_WIDTH_UV_TBL, MAX_WIDTH_Y_TBL};
    use crate::av2_filter::{adjust_strength, cdef_block, cdef_find_dir, ccso_score, CCSO_POS};
    let (iw4, ih4, w, h, cw, ch, ciw4, cih4, yac, bdmax) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.iw4, f.ih4, f.pl[0].w, f.pl[0].h, f.pl[1].w, f.pl[1].h, f.ciw4, f.cih4, f.yac as i32, f.bitdepth_max)
    });
    if w == 0 {
        return;
    }
    if let Ok(prefix) = std::env::var("MPREF") {
        thread_local! { static MPREF_N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
        let n = MPREF_N.with(|c| { let v = c.get(); c.set(v + 1); v });
        let mut out: Vec<u8> = Vec::new();
        FRAME.with(|fr| {
            let f = fr.borrow();
            for pl in 0..3 {
                let p = &f.pl[pl];
                for y in 0..p.h {
                    if !crate::av2_recon::work_tick("frm:1533") { break; }
                    for x in 0..p.w {
                        if !crate::av2_recon::work_tick("frm:1534") { break; }
                        out.push(p.px[y * p.stride + x].clamp(0, 255) as u8);
                    }
                }
            }
        });
        let _ = std::fs::write(format!("{prefix}_{n}.yuv"), &out);
    }
    // The in-loop filter chain works in i32 (intermediates go negative); the planes
    // are u16, so widen once here and narrow once at the write-back below.
    let widen = |v: &Vec<u16>| -> Vec<i32> { v.iter().map(|&x| x as i32).collect() };
    let (mut pl0, mut pl1, mut pl2) = FRAME.with(|fr| {
        let f = fr.borrow();
        (widen(&f.pl[0].px), widen(&f.pl[1].px), widen(&f.pl[2].px))
    });
    if let Ok(prefix) = std::env::var("PREFDUMP") {
        thread_local! { static PREF_N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
        let n = PREF_N.with(|c| { let v = c.get(); c.set(v + 1); v });
        let mut out: Vec<u8> = Vec::new();
        FRAME.with(|fr| {
            let f = fr.borrow();
            let hbd = f.bitdepth_max > 255;
            for pl in 0..3 {
                let p = &f.pl[pl];
                for y in 0..p.h {
                    if !crate::av2_recon::work_tick("frm:1555") { break; }
                    for x in 0..p.w {
                        if !crate::av2_recon::work_tick("frm:1556") { break; }
                        let v = p.px[y * p.stride + x];
                        if hbd {
                            out.extend_from_slice(&(v as u16).to_le_bytes());
                        } else {
                            out.push(v as u8);
                        }
                    }
                }
            }
        });
        let _ = std::fs::write(format!("{prefix}_{n}.yuv"), &out);
    }
    if std::env::var("MDBGRID").is_ok() {
        let dc = DEBLOCK_CFG.with(|c| c.get());
        crate::dlog!("[MDBCFG] y0={} y1={} u={} v={} sub_pu={} dqy0={} dqy1={} dqu={} dqv={}",
            dc.level_y0 as u8, dc.level_y1 as u8, dc.level_u as u8, dc.level_v as u8,
            dc.sub_pu as u8, dc.dq_y0, dc.dq_y1, dc.dq_u, dc.dq_v);
        FRAME.with(|fr| {
            let f = fr.borrow();
            if !f.db_lw.is_empty() {
                for y4 in 15..18usize {
                    if !crate::av2_recon::work_tick("frm:1577") { break; }
                    let mut row = String::new();
                    for x4 in 12..16usize {
                        if !crate::av2_recon::work_tick("frm:1579") { break; }
                        let cell = y4 * f.iw4 + x4;
                        row.push_str(&format!("Y({x4},{y4}) T{} lh{} | ", f.db_top[cell] as u8, f.db_lh[cell]));
                    }
                    crate::dlog!("[MDBGRIDY] {row}");
                }
            }
            if !f.cdb_lw.is_empty() {
                for cy in 21..26usize {
                    if !crate::av2_recon::work_tick("frm:1587") { break; }
                    let mut row = String::new();
                    for cx in 49..55usize {
                        if !crate::av2_recon::work_tick("frm:1589") { break; }
                        let cell = cy * f.ciw4 + cx;
                        row.push_str(&format!("({},{}) L{}{} T{}{} lw{} lh{} | ", cx, cy,
                            f.cdb_left[cell] as u8, f.cdb_spv[cell] as u8,
                            f.cdb_top[cell] as u8, f.cdb_sph[cell] as u8,
                            f.cdb_lw[cell], f.cdb_lh[cell]));
                    }
                    crate::dlog!("[MDBGRID] {row}");
                }
            }
        });
    }
    // ---- 1. DEBLOCK (all planes, in place) ----
    if let Ok(path) = std::env::var("MDUMP_RECON") {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for v in pl0.iter() { f.write_all(&(*v as u16).to_le_bytes()).unwrap(); }
    }
    if std::env::var("MDBW").is_ok() {
        crate::av2_deblock::QMAP.with(|w| *w.borrow_mut() = vec![-1i16; 6 * 6480]);
    }
    let mut dcfg = DEBLOCK_CFG.with(|c| c.get());
    if std::env::var("NODEBLOCK").is_ok() {
        dcfg.level_y0 = false;
        dcfg.level_y1 = false;
        dcfg.level_u = false;
        dcfg.level_v = false;
    }
    let (db_lw, db_lh, db_left, db_top, cdb_lw, cdb_lh, cdb_left, cdb_top) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.db_lw.clone(), f.db_lh.clone(), f.db_left.clone(), f.db_top.clone(), f.cdb_lw.clone(), f.cdb_lh.clone(), f.cdb_left.clone(), f.cdb_top.clone())
    });
    let (db_spv, db_sph, cdb_spv, cdb_sph) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.db_spv.clone(), f.db_sph.clone(), f.cdb_spv.clone(), f.cdb_sph.clone())
    });
    // Tile-column boundary V edges (4px units) — the deblock kernel caps max_width_neg
    // there (dav2d db_apply `tile_end == x64*16` → deblock_sb edge=1: luma ≤6, chroma ≤2),
    // because the left tile's side was decoded independently.
    let ti = crate::av2_recon::TILE_INFO.with(|c| c.get());
    let mut tile_v_y: Vec<usize> = Vec::new();
    for c in 1..ti.cols as usize {
        if !crate::av2_recon::work_tick("frm:1625") { break; }
        let x4 = ti.col_start4[c] as usize;
        if x4 > 0 && x4 < iw4 {
            tile_v_y.push(x4);
        }
    }
    let ss_hor = (ciw4 < iw4) as usize;
    let tile_v_c: Vec<usize> = tile_v_y.iter().map(|&x| x >> ss_hor).collect();
    let hbd = ((bdmax + 1).trailing_zeros() as i32 - 8) / 2;
    // Per-SB deblock thresholds under delta-q (dav db_apply: init_deblock_thr_lut per
    // lflvl->qidx[SB]). The grid holds each SB's effective qindex; thresholds per plane
    // derive from it exactly like the frame-level scalars derive from yac.
    let (sbq_grid, sbq_w) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.sb_qidx.clone(), f.cdef_sbw)
    });
    let sb_q = |extra: i32| -> Vec<(i32, i32, i32, i32)> {
        sbq_grid.iter().map(|&q0| {
            let q = if q0 != 0 { q0 as i32 } else { yac };
            (0, 0, 0, 0, q).4
        }).map(|q| (q + extra, q + extra, q + extra, q + extra)).collect()
    };
    let _ = &sb_q;
    let (ss_h_c, ss_v_c) = (SS.with(|c| c.get()).0 as u32, SS.with(|c| c.get()).1 as u32);
    if dcfg.level_y0 || dcfg.level_y1 {
        let (qv, sv) = (deblock_quant_thr(yac + 8 * dcfg.dq_y0, hbd), deblock_side_thr(yac + 8 * dcfg.dq_y0, hbd));
        let (qh, sh) = (deblock_quant_thr(yac + 8 * dcfg.dq_y1, hbd), deblock_side_thr(yac + 8 * dcfg.dq_y1, hbd));
        let ythr: Vec<(i32, i32, i32, i32)> = sbq_grid.iter().map(|&q0| {
            let q = if q0 != 0 { q0 as i32 } else { yac };
            (deblock_quant_thr(q + 8 * dcfg.dq_y0, hbd), deblock_side_thr(q + 8 * dcfg.dq_y0, hbd),
             deblock_quant_thr(q + 8 * dcfg.dq_y1, hbd), deblock_side_thr(q + 8 * dcfg.dq_y1, hbd))
        }).collect();
        if std::env::var("DQGRID").is_ok() {
            crate::dlog!("[DQGRID] w={sbq_w} len={} grid={:?}", sbq_grid.len(), &sbq_grid[..sbq_grid.len().min(28)]);
            crate::dlog!("[DQGRID] ythr[0..8]={:?}", &ythr[..ythr.len().min(8)]);
        }
        let sbt = if ythr.is_empty() { None } else { Some((ythr.as_slice(), sbq_w, 0u32, 0u32)) };
        deblock_plane(&mut pl0, iw4, ih4, w, &db_lw, &db_lh, &db_left, &db_top, qv, sv, qh, sh, sbt, &db_spv, &db_sph, bdmax, &MAX_WIDTH_Y_TBL, 16, 6, dcfg.level_y0, dcfg.level_y1, &tile_v_y);
    }
    if dcfg.level_u {
        let uac = yac + dcfg.uac_delta + 8 * dcfg.dq_u;
        let (q, s) = (deblock_quant_thr(uac, hbd), deblock_side_thr(uac, hbd));
        let uthr: Vec<(i32, i32, i32, i32)> = sbq_grid.iter().map(|&q0| {
            let qq = (if q0 != 0 { q0 as i32 } else { yac }) + dcfg.uac_delta + 8 * dcfg.dq_u;
            let (a, b) = (deblock_quant_thr(qq, hbd), deblock_side_thr(qq, hbd));
            (a, b, a, b)
        }).collect();
        let sbt = if uthr.is_empty() { None } else { Some((uthr.as_slice(), sbq_w, ss_h_c, ss_v_c)) };
        crate::av2_deblock::DBG_TAG.with(|c| c.set(std::env::var("DBLK444").is_ok()));
        deblock_plane(&mut pl1, ciw4, cih4, cw, &cdb_lw, &cdb_lh, &cdb_left, &cdb_top, q, s, q, s, sbt, &cdb_spv, &cdb_sph, bdmax, &MAX_WIDTH_UV_TBL, 16 >> SS.with(|c| c.get()).1, 2, true, true, &tile_v_c);
        crate::av2_deblock::DBG_TAG.with(|c| c.set(false));
    }
    if dcfg.level_v {
        let vac = yac + dcfg.vac_delta + 8 * dcfg.dq_v;
        let (q, s) = (deblock_quant_thr(vac, hbd), deblock_side_thr(vac, hbd));
        let vthr: Vec<(i32, i32, i32, i32)> = sbq_grid.iter().map(|&q0| {
            let qq = (if q0 != 0 { q0 as i32 } else { yac }) + dcfg.vac_delta + 8 * dcfg.dq_v;
            let (a, b) = (deblock_quant_thr(qq, hbd), deblock_side_thr(qq, hbd));
            (a, b, a, b)
        }).collect();
        let sbt = if vthr.is_empty() { None } else { Some((vthr.as_slice(), sbq_w, ss_h_c, ss_v_c)) };
        deblock_plane(&mut pl2, ciw4, cih4, cw, &cdb_lw, &cdb_lh, &cdb_left, &cdb_top, q, s, q, s, sbt, &cdb_spv, &cdb_sph, bdmax, &MAX_WIDTH_UV_TBL, 16 >> SS.with(|c| c.get()).1, 2, true, true, &tile_v_c);
    }
    stage_cmp("POST-DEBLK", &pl0, &pl1, &pl2, w, h, cw, ch, &crate::av2_recon::cap_path("dav_f2postdeblk.yuv"));
    if let Ok(path) = std::env::var("MDUMP_PD") {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for v in pl0.iter() { f.write_all(&(*v as u16).to_le_bytes()).unwrap(); }
    }
    // Debug: replace mine's post-deblock with dav's (same-run) so CDEF/CCSO/GDF are tested on
    // perfect input, removing any deblock cascade. Isolates cdef/ccso/gdf param bugs.
    if std::env::var("FORCE_PD").is_ok() {
        if let Ok(o) = std::fs::read(&crate::av2_recon::cap_path("dav_f2postdeblk.yuv")) {
            for i in 0..(w * h) { pl0[i] = o[i] as i32; }
            for i in 0..(cw * ch) { pl1[i] = o[w * h + i] as i32; pl2[i] = o[w * h + cw * ch + i] as i32; }
        }
    }
    // Post-deblock luma: CCSO classification source + GDF guided reference.
    let deblk_luma = pl0.clone();
    // Post-deblock chroma: LR stripe-boundary rows for the chroma NS-Wiener apply
    // (avm save_deblock_boundary_lines runs per plane; only the +-2 rows at stripe
    // edges are read from these).
    let deblk_u = pl1.clone();
    let deblk_v = pl2.clone();

    // env OH1PD: dump the POST-DEBLOCK planes for the oh=1 frame (filter-stage A/B vs dav PD1).
    if std::env::var("MDBW").is_ok() {
        let oh = crate::av2_recon::CUR_FRAME_REF.with(|c| c.get().0);
        dump_frame2_luma_wmap(&(crate::av2_recon::cap_path(&format!("mine_oh{oh}_wmap.bin"))));
        crate::av2_deblock::QMAP.with(|w| {
            let w = w.borrow();
            // Skip the dump for frames whose deblock never ran (fm2 TIP frames share oh=0
            // with the hidden key and would clobber its map with -1s).
            if w.len() == 6 * 6480 && w.iter().any(|&v| v != -1) {
                let bytes: Vec<u8> = w.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect();
                let _ = std::fs::write((crate::av2_recon::cap_path(&format!("mine_oh{oh}_qmap.bin"))), &bytes);
            }
        });
    }
    if std::env::var("OH1PD").is_ok() {
        let mut out = Vec::new();
        for y in 0..h {
            if !crate::av2_recon::work_tick("frm:1716") { break; }
            for x in 0..w {
                if !crate::av2_recon::work_tick("frm:1717") { break; }
                out.push(pl0[y * w + x] as u8);
            }
        }
        for p in [&pl1, &pl2] {
            if !crate::av2_recon::work_tick("frm:1721") { break; }
            for y in 0..ch {
                if !crate::av2_recon::work_tick("frm:1722") { break; }
                for x in 0..cw {
                    if !crate::av2_recon::work_tick("frm:1723") { break; }
                    out.push(p[y * cw + x] as u8);
                }
            }
        }
        let oh = crate::av2_recon::CUR_FRAME_REF.with(|c| c.get().0);
        let path = (crate::av2_recon::cap_path(&format!("mine_oh{oh}_postdeblock.yuv")));
        let _ = std::fs::write(path, &out);
    }
    // ---- 2. CDEF (post-deblock → post-CDEF) ----
    let ccfg = CDEF_CFG.with(|c| c.get());
    let (cdef_idx, cdef_sbw, noskip) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.cdef_idx.clone(), f.cdef_sbw, f.noskip.clone())
    });
    if ccfg.enabled && !cdef_idx.is_empty() && std::env::var("NOCDEF").is_err() {
        // Levels and damping scale with bitdepth (dav2d cdef_apply_tmpl.c:106-111, 280-288).
        let bd_min8 = (bdmax + 1).trailing_zeros() as i32 - 8;
        let damping = ccfg.damping + bd_min8;
        let (yin, uin, vin) = (pl0.clone(), pl1.clone(), pl2.clone());
        for by4 in (0..ih4).step_by(2) {
            if !crate::av2_recon::work_tick("frm:1743") { break; }
            for bx4 in (0..iw4).step_by(2) {
                if !crate::av2_recon::work_tick("frm:1744") { break; }
                let sb = (by4 >> 4) * cdef_sbw + (bx4 >> 4);
                let ci = cdef_idx[sb];
                if ci < 0 {
                    continue;
                }
                let (y_lvl, uv_lvl) = (ccfg.y_strength[ci as usize], ccfg.uv_strength[ci as usize]);
                if y_lvl == 0 && uv_lvl == 0 {
                    continue;
                }
                if !ccfg.on_skiptx {
                    let stripe = by4 >> 1;
                    let m = noskip[stripe * iw4 + bx4] || (bx4 + 1 < iw4 && noskip[stripe * iw4 + bx4 + 1]);
                    if !m {
                        continue;
                    }
                }
                let y_pri = (y_lvl >> 2) << bd_min8;
                let mut y_sec = y_lvl & 3;
                y_sec += (y_sec == 3) as i32;
                y_sec <<= bd_min8;
                let uv_pri = (uv_lvl >> 2) << bd_min8;
                let mut uv_sec = uv_lvl & 3;
                uv_sec += (uv_sec == 3) as i32;
                uv_sec <<= bd_min8;
                let (ht, hb) = (by4 > 0, by4 + 2 < ih4);
                let (hl, hr) = (bx4 > 0, bx4 + 2 < iw4);
                let in_off = (by4 * 4) * w + bx4 * 4;
                let (dir, var) = if y_pri > 0 || uv_pri > 0 { cdef_find_dir(&yin, in_off, w, bdmax) } else { (0, 0) };
                if y_pri > 0 {
                    let adj = adjust_strength(y_pri, var);
                    if adj > 0 || y_sec > 0 {
                        cdef_block(&mut pl0, in_off, w, &yin, in_off, w, 8, 8, adj, y_sec, dir, damping, ht, hb, hl, hr, bdmax);
                    }
                } else if y_sec > 0 {
                    cdef_block(&mut pl0, in_off, w, &yin, in_off, w, 8, 8, 0, y_sec, 0, damping, ht, hb, hl, hr, bdmax);
                }
                if uv_lvl != 0 {
                    let uvdir = if uv_pri > 0 { dir } else { 0 };
                    // Chroma cell = 8x8 luma px subsampled per axis; damping-1 only when
                    // the plane is subsampled (dav2d 4:2:0 damping-1; avm: full-res chroma
                    // keeps the luma damping).
                    let (sshf, ssvf) = { let ss = SS.with(|c| c.get()); (ss.0 as usize, ss.1 as usize) };
                    let (ccw, cch) = (8 >> sshf, 8 >> ssvf);
                    // Chroma damping is ALWAYS damping-1 (avm cdef_block.c:318, any format);
                    // mixed subsampling remaps the direction (conv422/conv440, cdef_block.c:345).
                    let cdamp = damping - 1;
                    let uvdir = if sshf != ssvf {
                        const CONV422: [usize; 8] = [7, 0, 2, 4, 5, 6, 6, 6];
                        const CONV440: [usize; 8] = [1, 2, 2, 2, 3, 4, 6, 0];
                        if sshf == 1 { CONV422[uvdir] } else { CONV440[uvdir] }
                    } else {
                        uvdir
                    };
                    let cin_off = ((by4 * 4) >> ssvf) * cw + ((bx4 * 4) >> sshf);
                    cdef_block(&mut pl1, cin_off, cw, &uin, cin_off, cw, ccw, cch, uv_pri, uv_sec, uvdir, cdamp, ht, hb, hl, hr, bdmax);
                    cdef_block(&mut pl2, cin_off, cw, &vin, cin_off, cw, ccw, cch, uv_pri, uv_sec, uvdir, cdamp, ht, hb, hl, hr, bdmax);
                }
            }
        }
    }

    if std::env::var("CFGDBG").is_ok() {
        crate::dlog!("[MINECFG] cdef damping={} n_str={} on_skiptx={} y_str={:?} uv_str={:?}", ccfg.damping, ccfg.n_strengths, ccfg.on_skiptx, &ccfg.y_strength[..ccfg.n_strengths], &ccfg.uv_strength[..ccfg.n_strengths]);
        FRAME.with(|fr| {
            let f = fr.borrow();
            let sbw = f.cdef_sbw;
            let sbh = f.ih4.div_ceil(16);
            for r in 0..sbh {
                if !crate::av2_recon::work_tick("frm:1812") { break; }
                let row: Vec<i8> = (0..sbw).map(|c| f.cdef_idx.get(r * sbw + c).copied().unwrap_or(-9)).collect();
                crate::dlog!("[MINECDEFIDX] row{r}: {row:?}");
            }
        });
    }
    // ---- 3. CCSO (classify from post-deblock luma, add to post-CDEF) ----
    let sccfg = CCSO_CFG.with(|c| c.borrow().clone());
    let ccso_sh = FRAME.with(|fr| fr.borrow().ccso_px_shift);
    let (mut ccso_blk, ccso_col256, ccso_n256) = FRAME.with(|fr| {
        let f = fr.borrow();
        (f.ccso_blk.clone(), f.ccso_col256, f.ccso_n256)
    });
    // INTER sb_reuse: the per-SB ccso flags aren't decoded — inherit them from the REF SLOT's
    // saved map (dav2d decode.c:1894 `prev_ccsomap[p] = refs[refidx[ccso refidx]].ccsomap`).
    CCSO_SLOT_MAP.with(|m| {
        let slots = m.borrow();
        for pl in 0..3 {
            let pc = match sccfg.p.get(pl) {
                Some(pc) if pc.sb_reuse => pc,
                _ => continue,
            };
            if let Some(prev) = slots[pc.reuse_slot as usize].as_ref() {
                if prev.len() == 3 * ccso_n256 {
                    for b in 0..ccso_n256 {
                        if !crate::av2_recon::work_tick("frm:1836") { break; }
                        ccso_blk[pl * ccso_n256 + b] = prev[pl * ccso_n256 + b];
                    }
                }
            }
        }
    });
    // Write the MERGED map back so this frame's slot save carries the complete flags (a later
    // frame's sb_reuse chain — e.g. key → reshow → P — reads them from the slot).
    FRAME.with(|fr| fr.borrow_mut().ccso_blk = ccso_blk.clone());
    if std::env::var("CFGDBG").is_ok() {
        for pl in 0..3 {
            let p = &sccfg.p[pl];
            let on_sbs = (0..ccso_n256).filter(|&b| ccso_blk.get(pl * ccso_n256 + b).copied().unwrap_or(false)).count();
            // band offsets (bo_only: at index band<<4)
            let bands: Vec<i8> = (0..(1usize << p.max_band_log2)).map(|b| p.filter_offset.get(b << 4).copied().unwrap_or(0)).collect();
            crate::dlog!("[MINECFG] ccso pl={pl} en={} bo_only={} mbl2={} ON_SBS={on_sbs}/{ccso_n256} band_off={bands:?}", p.enabled, p.bo_only, p.max_band_log2);
        }
    }
    if sccfg.enabled && std::env::var("NOCCSO").is_err() {
        let (hi, wi) = (h as i32, w as i32);
        let cl = |v: i32, m: i32| v.clamp(0, m - 1);
        for pl in 0..3usize {
            if !crate::av2_recon::work_tick("frm:1858") { break; }
            let pc = &sccfg.p[pl];
            if !pc.enabled {
                continue;
            }
            let (pw, ph, ss_hor, ss_ver) = if pl == 0 {
                (w, h, 0u32, 0u32)
            } else {
                let ss = SS.with(|c| c.get());
                (cw, ch, ss.0, ss.1)
            };
            let out = match pl {
                0 => &mut pl0,
                1 => &mut pl1,
                _ => &mut pl2,
            };
            let single_band = pc.max_band_log2 == 0;
            // Band shift is bitdepth-relative (dav2d ccso_tmpl.c:109: bitdepth - max_band_log2).
            let bd = (bdmax + 1).trailing_zeros();
            let shift = bd.saturating_sub(pc.max_band_log2);
            let (dy, dx) = (CCSO_POS[pc.ext_filter][0] as i32, CCSO_POS[pc.ext_filter][1] as i32);
            let q = pc.quant_step;
            for py in 0..ph {
                if !crate::av2_recon::work_tick("frm:1880") { break; }
                for px in 0..pw {
                    if !crate::av2_recon::work_tick("frm:1881") { break; }
                    let (lx, ly) = ((px << ss_hor) as i32, (py << ss_ver) as i32);
                    let block = ((ly >> ccso_sh) as usize) * ccso_col256 + ((lx >> ccso_sh) as usize);
                    if !ccso_blk[pl * ccso_n256 + block] {
                        continue;
                    }
                    let center = deblk_luma[(ly * wi + lx) as usize];
                    let band = if single_band { 0 } else { center >> shift };
                    let (cls0, cls1) = if pc.bo_only {
                        (0u32, 0u32)
                    } else {
                        let n0 = deblk_luma[(cl(ly + dy, hi) * wi + cl(lx + dx, wi)) as usize];
                        let n1 = deblk_luma[(cl(ly - dy, hi) * wi + cl(lx - dx, wi)) as usize];
                        (ccso_score(n0 - center, q, pc.edge_clf), ccso_score(n1 - center, q, pc.edge_clf))
                    };
                    let lut = ((band as usize) << 4) | ((cls0 as usize) << 2) | (cls1 as usize);
                    let offset = pc.filter_offset.get(lut).copied().unwrap_or(0) as i32;
                    let idx = py * pw + px;
                    if std::env::var("CCSODBG").is_ok() && pl == 0 && py == 0 && (px == 338 || px == 167) {
                        crate::dlog!("[MCCSOA] px={px} src={center} band={band} cls={cls0},{cls1} off={offset} dst_in={} thr={q} shift={shift} bo={} eclf={} ext={}", out[idx], pc.bo_only as u8, pc.edge_clf as u8, pc.ext_filter);
                    }
                    out[idx] = (out[idx] + offset).clamp(0, bdmax);
                }
            }
        }
    }

    stage_cmp("POST-CDEF", &pl0, &pl1, &pl2, w, h, cw, ch, &crate::av2_recon::cap_path("dav_f2postcdef.yuv"));
    if let Ok(path) = std::env::var("MDUMP_CDEF") {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for v in pl0.iter() { f.write_all(&(*v as u16).to_le_bytes()).unwrap(); }
    }
    // ---- 3b. LOOP RESTORATION (NS/PC-Wiener), luma. The wiener reads the pre-wiener
    // (post-CCSO) snapshot with post-deblock stripe-boundary rows; GDF's guided input
    // stays PRE-wiener (dav lr_stripe: gdf_prep before wiener, gdf_add after). ----
    let mut lr_src: Option<Vec<i32>> = None;
    if crate::av2_lr::LR_CFG.with(|c| c.borrow().enabled()) && std::env::var("NOLR").is_err() {
        let (lr_noskip, f_iw4) = FRAME.with(|fr| {
            let f = fr.borrow();
            (f.lr_noskip.clone(), f.iw4)
        });
        let src = pl0.clone();
        if let Ok(path) = std::env::var("MLRPRE") {
            use std::io::Write;
            let mut f = std::fs::File::create(path).unwrap();
            for v in src.iter() { f.write_all(&(*v as u16).to_le_bytes()).unwrap(); }
        }
        crate::av2_lr::lr_filter_luma(&mut pl0, &src, &deblk_luma, w, h, yac as u32, bdmax, &lr_noskip, f_iw4);
        // Chroma NS-Wiener (cross-component): reads the PRE-LR luma (`src`) through the
        // CfL-style downsample, never the freshly filtered pl0 -- avm builds its ds-luma
        // copy before any LR is applied (wienerns_copy_luma_with_virtual_lines).
        {
            let (ssh, ssv) = { let s = crate::av2_frame::SS.with(|c| c.get()); (s.0 as usize, s.1 as usize) };
            let ds_type = crate::av2_recon::HDR_TOOL_CFG.with(|c| c.get().cfl_ds_filter);
            let srcu = pl1.clone();
            crate::av2_lr::lr_filter_chroma(&mut pl1, &srcu, &deblk_u, 1, cw, ch, &src, &deblk_luma, w, h, ssh, ssv, ds_type, bdmax);
            let srcv = pl2.clone();
            crate::av2_lr::lr_filter_chroma(&mut pl2, &srcv, &deblk_v, 2, cw, ch, &src, &deblk_luma, w, h, ssh, ssv, ds_type, bdmax);
        }
        lr_src = Some(src);
    }
    // ---- 4. GDF (luma only: PRE-wiener guided input, post-deblock reference) ----
    let gcfg = if std::env::var("NOGDF").is_ok() { GdfCfg { enabled: false, ..GDF_CFG.with(|c| c.get()) } } else { GDF_CFG.with(|c| c.get()) };
    let gdf_blk = FRAME.with(|fr| fr.borrow().gdf_blk.clone());
    if gcfg.enabled {
        let mut out = pl0.clone();
        let guided: &Vec<i32> = lr_src.as_ref().unwrap_or(&pl0);
        // gdf_get_ref_dst_idx (avm): 0 for the INTRA keyframe → intra GDF tables; 1 for the
        // frame-2 INTER single-ref dist-1 → inter GDF tables. Passed in by the caller per frame.
        crate::av2_gdf::gdf_filter_frame(guided, &deblk_luma, &mut out, w, h, (bdmax + 1).trailing_zeros() as i32, yac, gcfg.mode, gcfg.qp_idx, gcfg.scale_idx, FRAME.with(|fr| fr.borrow().gdf_bs_px as i32), &gdf_blk, gdf_ref_dst);
        pl0 = out;
    }

    // MFIN: per-decode-frame FINAL dump (mirrors dav DFINDUMP — covers hidden frames).
    if std::env::var("MFIN").is_ok() {
        thread_local! { static FIN_N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
        let n = FIN_N.with(|c| { let v = c.get(); c.set(v + 1); v });
        let mut out: Vec<u8> = Vec::with_capacity(w * h * 3 / 2);
        for v in &pl0 { out.push((*v).clamp(0, 255) as u8); }
        for p in [&pl1, &pl2] {
            for v in p.iter() { out.push((*v).clamp(0, 255) as u8); }
                if !crate::av2_recon::work_tick("frm:1947") { break; }
        }
        let _ = std::fs::write((crate::av2_recon::cap_path(&format!("mine_fin_{n}.yuv"))), &out);
    }
    FRAME.with(|fr| {
        let mut f = fr.borrow_mut();
        // Narrow back to the u16 planes (values are post-filter pixels in [0, bdmax]).
        let narrow = |dst: &mut Vec<u16>, src: &Vec<i32>| {
            dst.clear();
            dst.extend(src.iter().map(|&v| v as u16));
        };
        narrow(&mut f.pl[0].px, &pl0);
        narrow(&mut f.pl[1].px, &pl1);
        narrow(&mut f.pl[2].px, &pl2);
    });
}

/// Gather the intra prediction edges for a `w`×`h` block at plane-pixel `(x, y)` on plane `pl`.
/// Returns `(top[.. w + h + 1], left[.. w + h + 1], corner)` with dav2d/AV1 edge-availability
/// semantics: unavailable top → replicate the left-top (or 1<<(bd-1) base); unavailable left →
/// replicate the top; missing top-right / bottom-left → replicate the last available sample.
/// `have_top`/`have_left` are the block-edge availabilities (frame/tile boundary).
pub fn gather_edges(
    p: &Plane,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    have_top: bool,
    have_left: bool,
    have_top_right: usize, // # extra top-right pixels available (0..=w)
    have_bottom_left: usize, // # extra bottom-left pixels available (0..=h)
    base: i32,
) -> (Vec<i32>, Vec<i32>, i32) {
    let sz = w + h + 1;
    let mut top = vec![base; sz + 16];
    let mut left = vec![base; sz + 16];
    // corner (top-left)
    let corner = if have_top && have_left {
        p.at(x - 1, y - 1)
    } else if have_top {
        p.at(x, y - 1)
    } else if have_left {
        p.at(x - 1, y)
    } else {
        base
    };
    // top row: [x .. x+w) then top-right [x+w .. x+w+have_top_right). Only `h` top-right samples
    // are ever consumed (z1 max_base_x = w+h-1), so cap the real fill at `h` (also keeps the
    // w+h+16 buffer in-bounds for wide blocks).
    if have_top {
        // Clamp x to the visible frame edge: a block spilling past the right edge reads its
        // off-frame top samples as the replicated edge column (dav's padded recon; garbage-free).
        let xmax = p.w - 1;
        for i in 0..w {
            if !crate::av2_recon::work_tick("frm:1996") { break; }
            top[i] = p.at((x + i).min(xmax), y - 1);
        }
        let tr = have_top_right.min(h);
        for i in 0..tr {
            if !crate::av2_recon::work_tick("frm:2000") { break; }
            top[w + i] = p.at((x + w + i).min(xmax), y - 1);
        }
        let last = if w + tr > 0 { top[w + tr - 1] } else { corner };
        for i in (w + tr)..top.len() {
            if !crate::av2_recon::work_tick("frm:2004") { break; }
            top[i] = last;
        }
    } else {
        // no top: the left-top pixel, or base-1 with NO neighbours at all (dav
        // ipred_prepare_tmpl.c pixel_set(top, have_left ? *left : base-1) — 127 at 8-bit;
        // the allintra frame-corner SMOOTH block reads these, DC-only corners hid it).
        let v = if have_left { p.at(x - 1, y) } else { base - 1 };
        for t in top.iter_mut() {
            if !crate::av2_recon::work_tick("frm:2012") { break; }
            *t = v;
        }
    }
    // left col: [y .. y+h) then bottom-left [y+h .. y+h+have_bottom_left)
    if have_left {
        // Clamp y to the visible frame edge (bottom-spilling block reads replicated edge row).
        let ymax = p.h - 1;
        for i in 0..h {
            if !crate::av2_recon::work_tick("frm:2020") { break; }
            left[i] = p.at(x - 1, (y + i).min(ymax));
        }
        // Only `w` bottom-left samples are ever consumed (z3 max_base_y = w+h-1); cap the fill.
        let bl = have_bottom_left.min(w);
        for i in 0..bl {
            if !crate::av2_recon::work_tick("frm:2025") { break; }
            left[h + i] = p.at(x - 1, (y + h + i).min(ymax));
        }
        let last = if h + bl > 0 { left[h + bl - 1] } else { corner };
        for i in (h + bl)..left.len() {
            if !crate::av2_recon::work_tick("frm:2029") { break; }
            left[i] = last;
        }
    } else {
        // no left: the top pixel, or base+1 with NO neighbours (dav: 129 at 8-bit).
        let v = if have_top { p.at(x, y - 1) } else { base + 1 };
        for l in left.iter_mut() {
            if !crate::av2_recon::work_tick("frm:2035") { break; }
            *l = v;
        }
    }
    (top, left, corner)
}
