"""Generative STUDY layers inspired by Shipibo-Konibo kené and Huni Kuin kenê kuin (KenePass).

    Generative study inspired by Shipibo-Konibo kené / Huni Kuin kenê kuin. For public shows, credit and
    collaborate with the originating communities: see research/kene_reference.

Nothing here copies an artwork and nothing here is kené: these are procedural studies of the visual GRAMMAR that
research/kene_reference/README.md documents, written after looking at the open-licensed study photographs there.
(The engine's older `l_kene` layer is the unrelated "Folding Blocks" pattern; it is not touched.)

Layer 1, `l_kene_shipibo` (README section 1):
  * a BOUNDED field (panel, band, medallion) with a frame; lines that reach the frame join it
  * quadrant / cross topology: seeds are laid out symmetrically round a central nexus (quincunx) and the field is
    partitioned with stepped (L1 / L-infinity) metrics, so every primary line is a stepped rectilinear path
  * three-tier line hierarchy: bold primary lines (the region outlines), a thinner secondary line running parallel
    just inside every primary line, and fine tertiary fill (short interlocking knob-ended paths) in every cell left
  * eye nodes (small nested squares), crosses and stepped spirals as motifs inside the larger regions
  * primary lines are the edges of a partition, so they can never dead-end (tested)
  Everything is generated on the CPU from a seed (`shipibo_design`) into a small RGBA8 tile texture:
    R = region label, G = fill-line bits (N, E, S, W, +16 = interior tile), B = fill colour index, A = pen order
  and the fragment shader draws it with signed distances + analytic pixel footprints (crisp, no shimmer).
Layer 2, `l_kene_hunikuin` (README section 2): diagonal, interlocking, figure-ground reversible stepped geometry on a
  bead lattice; motif families are named only as the README names them (`HK_MOTIFS`); bead colour conventions from
  README 2.3. Tiles are generated on the CPU (`hk_tile`) and rendered as glass beads, woven stitches or flat paint.
Yawanawá-inspired: two bead families (HK_EXTRA), added 2026-11-26 from the grammar of privately supplied study
  photographs (research/kene_reference/user_refs, never published, never copied): ombré diamond frames round a key or
  a four-squares eye, and a lattice of nested chevrons. The names are plain descriptions, not traditional names.

    KenePass(ctx, out_size).render(values, clock, t, dt, bands, palette_rgb) -> RGBA8 texture
    rgb = light with the layer amounts applied, a = how much it covers what is underneath.

    python kene_real.py --gallery      -> test_frames/kene_real/*.png + contact_sheet.png + reference_pairs.png
    python kene_real.py --bench
"""
import heapq
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import moderngl
except ImportError:                                     # the CPU half (designs, colours, tests) works without GL
    moderngl = None

HERE = Path(__file__).resolve().parent
TAU = 2.0 * math.pi
CREDIT = ("Generative study inspired by Shipibo-Konibo kené / Huni Kuin kenê kuin. For public shows, credit and "
          "collaborate with the originating communities — see research/kene_reference.")
FRAME = 255                                             # label of everything outside the bounded field
F = 3                                                   # fine tiles per macro cell
BREATH_S = 7.5

STYLES = ["Light on black", "Mud-dyed cloth", "Embroidery", "Ceramic", "Beadwork",
          "Painted kené (study)", "Cloth kené (study)", "Embroidered kené (study)"]     # appended 2026-11-26 (spec A, B, C)
COLOR_SETS = ["Follow show palette", "Cotton white", "Magenta / gold / red", "Gold / white / green",
              "Turquoise / white / magenta", "Red / violet / lime"]
LAYOUTS = ["Single panel", "Panels", "Band", "Medallion", "Follow the show"]
PEN = ["Off (Drawn fader)", "Draw in: 4 bars", "Draw in: 8 bars", "Draw in: 16 bars", "Draw in: 32 bars"]
PEN_BARS = [0, 4, 8, 16, 32]
PEN_NEW = ["Same design", "New design each time"]
N_SEEDS = 32
FORMS = ["Outlined regions", "Regions + arms and terminals"]
# structures the three reference studies bring with them (style_uniforms()["form"]): 2 medallion lattice with bulbs,
# 3 packed cells with twigs on the channels, 4 portrait panel with a central spine
F_LATTICE, F_CELLS, F_SPINE = 2, 3, 4
ARROW_OPEN = 4
ARROW, TBAR, CAP = 1, 2, 3                              # terminal codes of an arm end (tile B channel, low bits)
# Huni Kuin motif families: names exactly as in README 2.2 (nawan kenê and kashi are left out: the README gives no
# visual structure for them that could be generated honestly)
HK_MOTIFS = ["txede bedu", "dunu kenê", "inu tae", "inu tae txede bedu", "kapë hina"]
# appended 2026-11-26 (descriptive names of my own, NOT traditional names): a woven-band fret study, and two
# Yawanawá-inspired beadwork studies (ombré diamond frames round a key / four-squares eye; chevron X lattice)
HK_EXTRA = ["Woven band (Huni Kuin-inspired study)", "Bead tiles (Huni Kuin-inspired study)",
            "Bead straps (Yawanawá-inspired study)", "Key-fret beadwork (study)"]            # spec D, E, F, G
HK_FAMILIES = HK_MOTIFS + HK_EXTRA
NF = len(HK_FAMILIES)
HK_MOTIF_LABELS = [n + " (study)" for n in HK_MOTIFS] + HK_EXTRA   # the panel says what the output is: a study
HK_MEDIA = ["Glass beads", "Woven", "Painted", "Glass beads, staggered rows"]
HK_COLOR_NAMES = ["Black & white", "Red / yellow / white", "Blue / turquoise / orange", "Green / yellow / red",
                  "All eight bead colours", "Follow show palette",
                  "Rainbow ombré", "Blue ombré", "Green ombré", "Black / white / orange"]      # appended 2026-11-26
HK_CYCLE = ["Hold", "Morph every 8 bars", "Morph every 16 bars", "Morph every 32 bars"]
HK_CYCLE_BARS = [0, 8, 16, 32]
HK_T = 48                                               # bead tile period
OPTIONS = {"kene_style": STYLES, "kene_colors": COLOR_SETS, "kene_layout": LAYOUTS, "kene_pen": PEN,
           "kene_pen_new": PEN_NEW, "kene_seed": [str(i + 1) for i in range(N_SEEDS)],
           "kene_form": FORMS, "hk_motif": HK_MOTIF_LABELS, "hk_medium": HK_MEDIA, "hk_colors": HK_COLOR_NAMES, "hk_cycle": HK_CYCLE}

# README 1.6: embroidery threads (magenta, turquoise, lime, golden yellow, violet, red, white on black cloth)
_MAG, _GOLD, _RED, _TURQ = (1.0, 0.10, 0.62), (1.0, 0.78, 0.08), (1.0, 0.10, 0.12), (0.05, 0.85, 0.80)
_WHITE, _GREEN, _LIME, _VIOLET, _ORANGE = (1.0, 0.96, 0.88), (0.10, 0.85, 0.30), (0.62, 1.0, 0.12), (0.62, 0.25, 1.0), (1.0, 0.45, 0.05)
# (primary, secondary, fill, fill in alternate regions)
THREADS = {2: (_MAG, _GOLD, _RED, _TURQ), 3: (_ORANGE, _WHITE, _GREEN, _RED),
           4: (_TURQ, _WHITE, _MAG, _LIME), 5: (_RED, _VIOLET, _LIME, _GOLD)}
COTTON = ((1.0, 0.93, 0.80), (0.95, 0.85, 0.68), (0.80, 0.70, 0.55), (0.85, 0.72, 0.52))
ACCENTS = ((1.0, 0.16, 0.10), (1.0, 0.62, 0.08), (1.0, 0.86, 0.15))       # small filled blocks: red, ochre, yellow
PAINT_GROUND, PAINT_INK = (0.93, 0.91, 0.87), (0.16, 0.14, 0.13)           # spec A: cream #E9E4DB.., ink #2A2522
PAINT_ACC = ((0.68, 0.24, 0.14), (0.80, 0.65, 0.16), (0.80, 0.65, 0.16))  # rust #A8361F.., ochre #C9A227..
CLOTH_FIELD, CLOTH_INK, CLOTH_BORDER = (0.80, 0.76, 0.70), (0.17, 0.14, 0.13), (0.23, 0.21, 0.19)   # spec B
MAG_GROUND = (0.72, 0.15, 0.36)                                            # spec C: raspberry #C2285A
CLOTH_GROUND, MUD_INK = (0.80, 0.72, 0.56), (0.10, 0.06, 0.035)          # README 1.6: black / chocolate on cream
SLIP_GROUND, OXIDE_RED, CERAMIC_BLACK = (0.84, 0.78, 0.64), (0.66, 0.17, 0.07), (0.07, 0.05, 0.04)
# README 2.3 bead colours: black, white, crimson red, golden yellow, orange, cobalt blue, turquoise, emerald green
_B_WHITE, _B_RED, _B_YEL, _B_ORA = (1.0, 0.97, 0.92), (0.95, 0.06, 0.12), (1.0, 0.80, 0.05), (1.0, 0.42, 0.04)
_B_COB, _B_TURQ, _B_EMER = (0.08, 0.22, 1.0), (0.05, 0.85, 0.82), (0.04, 0.80, 0.30)
HK_COLORWAYS = [
    [_B_WHITE] * 6,
    [_B_RED, _B_YEL, _B_WHITE, _B_RED, _B_YEL, _B_WHITE],
    [_B_COB, _B_TURQ, _B_ORA, _B_COB, _B_WHITE, _B_TURQ],
    [_B_EMER, _B_YEL, _B_RED, _B_EMER, _B_YEL, _B_RED],
    [_B_RED, _B_YEL, _B_TURQ, _B_ORA, _B_COB, _B_EMER],
    None,                                               # follow the show palette
    # ombré ramps (spec E / F / G), six steps + two key colours (entries 7, 8: the fret pair / the eye)
    [(0.85, 0.14, 0.10), (0.96, 0.48, 0.13), (0.97, 0.82, 0.12), (0.60, 0.80, 0.20), (0.25, 0.64, 0.30), (0.06, 0.42, 0.39),
     (0.78, 0.06, 0.18), (0.06, 0.55, 0.50)],
    [(0.95, 0.95, 0.95), (0.31, 0.76, 0.91), (0.50, 0.56, 0.89), (0.16, 0.29, 0.79), (0.11, 0.16, 0.48), (0.06, 0.09, 0.30),
     (0.84, 0.15, 0.12), (0.55, 0.78, 0.25)],
    [(0.95, 1.0, 0.92), (0.71, 0.90, 0.55), (0.25, 0.64, 0.30), (0.11, 0.42, 0.20), (0.07, 0.30, 0.15), (0.04, 0.20, 0.10),
     (0.95, 1.0, 0.92), (0.11, 0.42, 0.20)],
    [_B_WHITE, (1.0, 0.50, 0.03), _B_WHITE, _B_WHITE, (1.0, 0.50, 0.03), _B_WHITE, (1.0, 0.50, 0.03), _B_WHITE],
]
HK_COLORWAYS = [c if c is None else (list(c) + list(c[:2]))[:8] for c in HK_COLORWAYS]      # every colourway: 8 entries
WHITE_ROLE = 9                                          # bead role 9 = white whatever the colourway (a key / an eye)


def _num(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _val(values, name, default, lo, hi):
    return min(max(_num(values.get(name, default), default), lo), hi)


def _idx(values, name, n, default=0):
    return int(round(_val(values, name, default, 0, n - 1)))


# =====================================================================================================================
# Shipibo-Konibo-inspired design generator (CPU)
# =====================================================================================================================
_OPS = {
    "c2": [(1, 0, 0, 1), (-1, 0, 0, -1)],
    "d2": [(1, 0, 0, 1), (-1, 0, 0, 1), (1, 0, 0, -1), (-1, 0, 0, -1)],
    "c4": [(1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0)],
}
_OPS["d4"] = _OPS["c4"] + [(-a, -b, c, d) for a, b, c, d in _OPS["c4"]]
# stepped square spiral / hook: a track one tile wide with two tiles between the windings, in a 7 x 7 block
_SPIRAL = ([(x, 0) for x in range(7)] + [(6, y) for y in range(1, 7)] + [(x, 6) for x in range(6)] +
           [(0, y) for y in range(3, 6)] + [(x, 3) for x in range(1, 4)])
_SPIRAL = [(x - 3, y - 3) for x, y in _SPIRAL]


def _metric(dx, dy, kind):
    if kind == "cross":                                 # concave star: arms along the axes (the cross that parts the quadrants)
        return 1.5 * np.minimum(dx, dy) + 0.45 * np.maximum(dx, dy)
    return (1.0 - kind) * 0.75 * (dx + dy) + kind * np.maximum(dx, dy)      # 0 stepped diamond .. 1 square


def _depth(lab, wrap_x):
    """Chebyshev distance (in cells) from each cell to the nearest cell of another label / the frame."""
    h, w = lab.shape
    depth = np.zeros((h, w), int)
    cur = np.ones((h, w), bool)
    for k in range(1, max(h, w)):
        p = np.pad(lab, 1, mode="wrap") if wrap_x else np.pad(lab, 1, constant_values=FRAME)
        if wrap_x:
            p[0, :], p[-1, :] = FRAME, FRAME
        pc = np.pad(cur, 1, mode="wrap" if wrap_x else "constant")
        if wrap_x:
            pc[0, :], pc[-1, :] = False, False
        nxt = cur.copy()
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                nxt &= (p[dy:dy + h, dx:dx + w] == lab) & pc[dy:dy + h, dx:dx + w]
        if not nxt.any():
            break
        depth[nxt] = k
        cur = nxt
    return depth


def shipibo_design(seed, gx, gy, wrap_x=False, form=1):
    """(3*gy, 3*gx, 4) uint8 tile map of one bounded design. gx, gy: macro cells (even numbers). Row 0 = bottom."""
    if form == F_SPINE:
        return _spine_panel(seed, gx, gy)
    gx, gy = max(4, int(gx) // 2 * 2), max(4, int(gy) // 2 * 2)
    rng = np.random.default_rng([int(seed) % 100003, gx, gy, int(bool(wrap_x))])
    if wrap_x:
        kind = "d2"
    elif gx == gy:
        kind = str(rng.choice(["c4", "d4", "d2", "c2"], p=[0.25, 0.30, 0.35, 0.10]))      # mirror symmetry dominates
    else:
        kind = str(rng.choice(["d2", "c2"], p=[0.85, 0.15]))
    if form in (F_LATTICE, F_CELLS):
        kind = "d2"
    ops = _OPS[kind]
    X, Y = np.meshgrid(np.arange(gx) - (gx - 1) / 2.0, np.arange(gy) - (gy - 1) / 2.0)
    hx, hy, small = gx / 2.0, gy / 2.0, min(gx, gy)

    # ---- seeds: the central nexus + a handful in the fundamental domain, repeated by the symmetry (quincunx and up)
    seeds = [(0.0, 0.0, ("cross", 0.0, 0.5, 1.0)[int(rng.choice(4, p=[0.4, 0.25, 0.2, 0.15]))], rng.uniform(1.2, 0.16 * small + 1.2))]
    want = max(3, int(round(gx * gy / 60.0))) + int(rng.integers(0, 2))
    for _ in range(200):
        if len(seeds) > want:
            break
        sx, sy = rng.uniform(0.5, hx - 0.5), rng.uniform(0.5, hy - 0.5)
        if all(abs(sx - s[0]) + abs(sy - s[1]) > 0.42 * small for s in seeds):
            seeds.append((sx, sy, (0.0, 0.5, 1.0)[int(rng.choice(3, p=[0.6, 0.3, 0.1]))], rng.uniform(0.0, 1.0)))
    if form == 1 and not wrap_x and min(gx, gy) >= 16:   # medallions on a rhombic lattice round the central one
        seeds.insert(1, (hx / 2.0, hy / 2.0, 0.0 if rng.random() < 0.6 else 0.5, rng.uniform(1.5, 3.0)))
    if form == F_LATTICE:                               # medallions on a rhombic lattice: centre, (W/2, H/2), (W, 0), (0, H);
        w0 = rng.uniform(0.0, 1.0)                      # octagon metric = orthogonal channels with stepped diagonals
        seeds = [(0.0, 0.0, 0.5, w0), (hx / 2.0, hy / 2.0, 0.5, w0 + rng.uniform(-0.6, 0.6)),
                 (hx - 0.5, 0.0, 0.5, w0 + rng.uniform(-1.0, 0.5)), (0.0, hy - 0.5, 0.5, w0 + rng.uniform(-1.0, 0.5))]
        for _ in range(int(rng.integers(0, 3))):
            seeds.append((rng.uniform(1.0, hx - 1.0), rng.uniform(1.0, hy - 1.0), 0.5, rng.uniform(-2.5, -1.0)))
    if form == F_CELLS:                                 # many small chamfered cells packed edge to edge
        seeds = [(0.0, 0.0, 0.5, 0.0)]
        for _ in range(4000):
            if len(seeds) > gx * gy / 14.0:
                break
            sx, sy = rng.uniform(0.3, hx - 0.3), rng.uniform(0.3, hy - 0.3)
            if all(abs(sx - q[0]) + abs(sy - q[1]) > 2.6 for q in seeds):
                seeds.append((sx, sy, 0.5, rng.uniform(0.0, 0.5)))
    best = np.full((gy, gx), 1e9)
    macro = np.zeros((gy, gx), np.uint8)
    for lab, (sx, sy, kd, wt) in enumerate(seeds, start=1):
        d = np.full((gy, gx), 1e9)
        for a, b, c, e in ops:
            px, py = a * sx + b * sy, c * sx + e * sy
            dx = np.abs(X - px)
            if wrap_x:
                dx = np.minimum(dx, gx - dx)
            d = np.minimum(d, _metric(dx, np.abs(Y - py), kd) - wt)
        take = d < best - 1e-9
        macro[take], best[take] = lab, d[take]
    lab = np.repeat(np.repeat(macro, F, 0), F, 1)
    ny, nx = lab.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    nlab = [200]

    def paint(tiles, only=None):
        """Give the tiles (centred fine coords) and their symmetric images a label of their own."""
        nlab[0] += 1
        for a, b, c, e in ops:
            for x, y in tiles:
                i, j = a * x + b * y + cx, c * x + e * y + cy
                i, j = int(round(i)), int(round(j))
                if wrap_x:
                    i %= nx
                if 0 <= i < nx and 0 <= j < ny and (only is None or lab[j, i] == only or lab[j, i] > 200):
                    lab[j, i] = nlab[0]

    # ---- the central eye: nested squares on the nexus
    c0 = lab[ny // 2, nx // 2]
    roomy = (lab[ny // 2 - 5:ny // 2 + 5, nx // 2 - 5:nx // 2 + 5] == c0).all()
    cross_centre = form in (1, F_LATTICE) and roomy                            # a stepped cross of filled blocks
    if form == F_CELLS:
        cross_centre = True                                                    # (no eye stamp either)
    if not cross_centre:
        if (lab[ny // 2 - 3:ny // 2 + 3, nx // 2 - 3:nx // 2 + 3] == c0).all():
            paint([(x + 0.5, y + 0.5) for x in range(-2, 2) for y in range(-2, 2)])
        paint([(x + 0.5, y + 0.5) for x in range(-1, 1) for y in range(-1, 1)])

    # ---- motifs inside the larger regions: eye node, cross, stepped spiral (one per region, mirrored by the symmetry)
    depth = _depth(macro, wrap_x)
    depth[(np.abs(X) < 2.5) & (np.abs(Y) < 2.5)] = 0
    for region in range(1, len(seeds) + 1):
        if form >= F_LATTICE:
            break                                       # the studies carry their own motifs (below)
        m = (macro == region) & (X > 0) & (Y > 0) & (depth >= 1)
        if kind in ("c2",):
            m = (macro == region) & (Y > 0) & (depth >= 1)
        if not m.any():
            continue
        js, is_ = np.nonzero(m & (depth == depth[m].max()))
        k = int(rng.integers(0, len(js)))
        dep, mx, my = depth[js[k], is_[k]], X[js[k], is_[k]] * F, Y[js[k], is_[k]] * F
        pick = rng.random()
        if pick < 0.28 and not form:                                          # stepped spiral (outline designs only) (fits a 3 x 3 macro block)
            rot, flip = int(rng.integers(0, 4)), rng.random() < 0.5
            pts = []
            for x, y in _SPIRAL:
                x = -x if flip else x
                for _r in range(rot):
                    x, y = -y, x
                pts.append((mx + x, my + y))
            paint(pts, only=region)
        elif pick < 0.6 and dep >= 2:                                          # cross of five cells with an eye
            cells = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
            paint([(mx + F * a + x, my + F * b + y) for a, b in cells for x in (-1, 0, 1) for y in (-1, 0, 1)], only=region)
            paint([(mx, my)])
        else:                                                                  # eye node: a square with a dot
            paint([(mx + x, my + y) for x in (-1, 0, 1) for y in (-1, 0, 1)], only=region)

    # ---- ring tiles (they carry the primary + secondary lines) and interior tiles (fine fill)
    p = np.pad(lab, 1, mode="wrap") if wrap_x else np.pad(lab, 1, constant_values=FRAME)
    if wrap_x:
        p[0, :], p[-1, :] = FRAME, FRAME
    interior = np.ones((ny, nx), bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            interior &= p[dy:dy + ny, dx:dx + nx] == lab

    # ---- pen order: distance along the line network from the centre (Dijkstra; ring tiles are the roads)
    ring = ~interior
    dist = np.full((ny, nx), np.inf)
    jj, ii = np.nonzero(ring)
    r2 = (ii - cx) ** 2 + (jj - cy) ** 2
    heap = [(0.0, int(j), int(i)) for j, i in zip(jj[r2 <= r2.min() + 1e-6], ii[r2 <= r2.min() + 1e-6])]
    for _d, j, i in heap:
        dist[j, i] = 0.0
    heapq.heapify(heap)
    nb8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        d, j, i = heapq.heappop(heap)
        if d > dist[j, i]:
            continue
        for dj, di in nb8:
            j2, i2 = j + dj, i + di
            if wrap_x:
                i2 %= nx
            if not (0 <= j2 < ny and 0 <= i2 < nx):
                continue
            both = ring[j, i] and ring[j2, i2]
            if not both and dj and di:
                continue                                                       # off the roads: 4 neighbours only
            nd = d + ((1.4 if dj and di else 1.0) if both else 2.5)
            if nd < dist[j2, i2]:
                dist[j2, i2] = nd
                heapq.heappush(heap, (nd, j2, i2))
    order = np.round(dist / max(float(dist.max()), 1.0) * 255.0).astype(np.uint8)

    bits = np.where(interior, 16, 0).astype(np.uint8)
    code = np.zeros((ny, nx), np.uint8)
    seen = ~interior
    # ---- arms: free primary paths inside the regions, stepped, with terminals (arrowhead, T-bar, small end-cap) and
    #      a small filled accent block at an end; each one is repeated by the symmetry (bits +32 = arm, +64 = block)
    rb = np.zeros((ny, nx), np.uint8)                   # ring tiles' B: bits 0-2 line colour, 3 gem, 4-5 its corner, 6 gem colour
    if form == F_CELLS:
        # twigs ride the channels: short runs of the channel line take a twig colour (1..5); eye dots in some cells
        for dj, di in ((1, 0), (0, 1)):
            a, b = lab[:ny - dj, :nx - di], lab[dj:, di:]
            for j, i in zip(*(v.tolist() for v in np.nonzero(a != b))):
                h = (int(min(a[j, i], b[j, i])) * 7919 + int(max(a[j, i], b[j, i])) * 104729 + (j // 3) * 31 + (i // 3) * 17 + dj) % 11
                if h < 5:
                    rb[j, i] = rb[j + dj, i + di] = 1 + h
        dep = _depth(lab, False)
        for region in np.unique(lab):
            m = (lab == region) & interior
            if m.any() and rng.random() < 0.45:
                js, is_ = np.nonzero(m & (dep == dep[m].max()))
                bits[js[0], is_[0]] |= 64
                code[js[0], is_[0]] = 1
                seen[js[0], is_[0]] = True
    if form == F_LATTICE:
        # red channel segments: the horizontal runs of the channels between the outer medallions
        for dj, di in ((1, 0),):
            a, b = lab[:ny - dj, :nx - di], lab[dj:, di:]
            pair = (a != b) & (np.minimum(a, b) >= 2) & (np.maximum(a, b) <= 4)
            for j, i in zip(*(v.tolist() for v in np.nonzero(pair))):
                rb[j, i] = rb[j + dj, i + di] = 1
        # junction gems: a small square where three channels meet
        pl = np.pad(lab, 1, constant_values=FRAME)
        for j in range(1, ny):
            for i in range(1, nx):
                quad = {int(lab[j - 1, i - 1]), int(lab[j - 1, i]), int(lab[j, i - 1]), int(lab[j, i])}
                if len(quad) >= 3:
                    colr = 64 if (len(quad) + j + i) % 2 else 0
                    for (tj, ti, cn) in ((j, i, 2), (j, i - 1, 3), (j - 1, i, 1), (j - 1, i - 1, 0)):
                        rb[tj, ti] = (rb[tj, ti] & 7) | 8 | (cn << 4) | colr
    if form in (1, F_LATTICE):
        blocked = np.zeros((ny, nx), bool)
        if form == F_LATTICE:                           # bowtie / double axe at the centres of the secondary medallions
            bx, by = int(round(hx / 2.0 * F)), int(round(hy / 2.0 * F))
            shape = [(x, y) for x in range(-3, 4) for y in (-1, 0, 1) if abs(x) >= 2 or y == 0]
            cells = [(int(ny // 2 + sy * by + y), int(nx // 2 + sx * bx + x)) for sx in (-1, 1) for sy in (-1, 1) for x, y in shape]
            if all(0 <= tj < ny and 0 <= ti < nx and interior[tj, ti] for tj, ti in cells):
                for tj, ti in cells:
                    bits[tj, ti] |= 64
                    code[tj, ti] = 3
                    seen[tj, ti] = True
                    blocked[max(tj - 1, 0):tj + 2, max(ti - 1, 0):ti + 2] = True
        if cross_centre:                                # stepped Greek cross of filled squares round an open centre
            for y in range(-3, 3):
                for x in range(-3, 3):
                    r1 = abs(x + 0.5) + abs(y + 0.5)
                    if 1.5 < r1 <= 3.0 and interior[ny // 2 + y, nx // 2 + x]:
                        bits[ny // 2 + y, nx // 2 + x] |= 64
                        code[ny // 2 + y, nx // 2 + x] = 3                      # 3 = the ink itself (black squares)
                        seen[ny // 2 + y, nx // 2 + x] = True
            blocked[ny // 2 - 4:ny // 2 + 4, nx // 2 - 4:nx // 2 + 4] = True

        def ok(j, i, L):
            return 0 <= j < ny and 0 <= i < nx and interior[j, i] and not blocked[j, i] and lab[j, i] == L

        jj, ii = np.nonzero(interior)
        for _try in range(int(len(jj) / 6) + 4):
            k = int(rng.integers(0, len(jj)))
            j, i = int(jj[k]), int(ii[k])
            L = lab[j, i]
            sj, si = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
            horiz, straight = rng.random() < 0.5, rng.random() < 0.3
            want, path, run = int(rng.integers(4, 10)), [(j, i)], int(rng.integers(1, 4))
            while len(path) < want:
                j2, i2 = (j, i + si) if horiz else (j + sj, i)
                if not ok(j2, i2, L) or (j2, i2) in path:
                    break
                path.append((j2, i2))
                j, i, run = j2, i2, run - 1
                if run <= 0 and not straight:
                    horiz, run = not horiz, int(rng.integers(1, 4))
            ends = (int(rng.choice((ARROW, TBAR, CAP), p=(0.45, 0.35, 0.2))), int(rng.choice((0, ARROW, TBAR, CAP), p=(0.2, 0.2, 0.25, 0.35))))
            if form == F_LATTICE and rng.random() < 0.5:     # a bulb: the line curls into a C and ends in an arrowhead
                n = int(rng.integers(3, 6))                  # pointing back into the loop
                j, i = path[0]
                a1, b1 = ((0, si), (sj, 0)) if rng.random() < 0.5 else ((sj, 0), (0, si))
                moves = [a1] * n + [b1] * n + [(-a1[0], -a1[1])] * n + [(-b1[0], -b1[1])] * (n - 2)
                path = [(j, i)]
                for dj, di in moves:
                    j, i = j + dj, i + di
                    if not ok(j, i, L):
                        path = []
                        break
                    path.append((j, i))
                ends = (0, ARROW)
            if len(path) < 3:
                continue
            acc = int(rng.integers(0, 3))
            ej, ei = path[-1]
            pj, pi = path[-2]
            blk = None
            if rng.random() < 0.8:                      # an L / step of two or three filled squares just past one end
                dj, di = ej - pj, ei - pi
                sd = int(rng.choice((-1, 1)))
                blk = [(ej + dj, ei + di), (ej + dj + (sd if di else 0), ei + di + (sd if dj else 0))]
                if rng.random() < 0.5:
                    blk.append((ej + 2 * dj + (sd if di else 0), ei + 2 * di + (sd if dj else 0)))
            twig = int(rng.integers(0, 4))
            images, good = [], True
            for a, b, c, e in ops:
                img = []
                for (tj, ti) in path + (blk or []):
                    x, y = ti - cx, tj - cy
                    i3, j3 = int(round(a * x + b * y + cx)), int(round(c * x + e * y + cy))
                    if wrap_x:
                        i3 %= nx
                    img.append((j3, i3))
                if len(set(img)) != len(img) or not all(
                        ok(j3, i3, lab[img[0]] if 0 <= img[0][0] < ny and 0 <= img[0][1] < nx else -1) for j3, i3 in img):
                    good = False
                    break
                images.append(img)
            if not good:
                continue
            sets = [frozenset(im) for im in images]
            uniq = list(dict.fromkeys(sets))
            near = lambda A, B: any(abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1 for a in A for b in B)
            if any(A != B and near(A, B) for n, A in enumerate(uniq) for B in uniq[n + 1:]):
                continue                                                       # two images would touch: skip this arm
            parts = {(frozenset(im), frozenset(im[:len(path)])) for im in images}
            if len(parts) != len(uniq):
                continue                                                       # same tiles, different roles (it sits on an axis)
            for img in images:
                pth = img[:len(path)]
                for (j0, i0), (j1, i1) in zip(pth, pth[1:]):
                    dj, di = j1 - j0, i1 - i0
                    if wrap_x:
                        di = (di + nx // 2) % nx - nx // 2
                    mb, tb = {(1, 0): (1, 4), (0, 1): (2, 8), (-1, 0): (4, 1), (0, -1): (8, 2)}[(dj, di)]
                    bits[j0, i0] |= mb | 32
                    bits[j1, i1] |= tb | 32
                code[pth[0]], code[pth[-1]] = ends[0], ends[1]
                for t in pth:
                    code[t] |= twig << 3                                       # which accent colour a twig takes (embroidery)
                for t in img[len(path):]:
                    bits[t] |= 64
                    code[t] = acc
                for (j3, i3) in img:
                    seen[j3, i3] = True
                    blocked[max(j3 - 1, 0):j3 + 2, max(i3 - 1, 0):i3 + 2] = True

    # ---- fine fill: cover every interior tile left with short knob-ended paths (or, for some seeds, long meanders)
    lo, hi = (3, 7) if rng.random() < 0.6 else (8, 26)
    if form >= F_LATTICE:
        lo, hi = 14, 60                                 # one uniform wandering labyrinth line, not bones
    bits |= meander_fill(~seen, rng, lo, hi, lab, order, wrap_x)
    out = np.zeros((ny, nx, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 3] = lab, bits, order
    out[..., 2] = np.where((lab % 2 == 0) | (lab > 200), 128, 0) | code        # bit 7: alternate fill colour; low bits: terminal / block
    out[..., 2] = np.where(interior, out[..., 2], rb)                           # ring tiles: line colour / junction gem
    return out


def line_dead_ends(ve, he):
    """Number of lattice corners where exactly one line edge ends. ve: (h+2, w+1) vertical edges, he: (h+1, w+2)
    horizontal edges, 0 / 1, as laid out by an edge map padded with the frame."""
    ve, he = np.asarray(ve, int), np.asarray(he, int)
    deg = ve[:-1, :] + ve[1:, :] + he[:, :-1] + he[:, 1:]
    return int((deg == 1).sum())


def _spine_panel(seed, gx, gy):
    """Portrait panel study (GRAMMAR_SPEC B): a vertical spine down the centre carrying a chain lozenge - diamond -
    lozenge, arrowheads (outlines) at top and bottom pointing in, stepped side hooks ending in small open squares,
    loose dumbbells near the edges, a coarse hand-made meander everywhere. Strictly mirrored left / right and
    top / bottom. gx, gy: macro cells (odd, so the spine has a column of its own)."""
    gx, gy = max(5, int(gx) | 1), max(9, int(gy) | 1)
    rng = np.random.default_rng([int(seed) % 100003, gx, gy, 4])
    nx, ny = gx * F, gy * F
    cx, cy = nx // 2, ny // 2
    lab = np.ones((ny, nx), np.uint8)
    yy, xx = np.mgrid[0:ny, 0:nx]
    x, y = xx - cx, yy - cy
    half = int(rng.integers(4, 7))                                              # the central lozenge (elongated hexagon)
    lab[(np.abs(x) <= np.minimum(2, half + 1 - np.abs(y))) & (np.abs(y) <= half + 1)] = 2
    top = ny // 2 - 8
    joints = sorted({int(v) for v in np.linspace(half + 5, top - 5, 3 if ny > 60 else 2)})
    for n, yj in enumerate(joints):                                             # small diamonds at the spine's joints
        for sgn in (-1, 1):
            lab[(np.abs(x) + np.abs(y - sgn * yj)) <= 1] = 3 + n
    p = np.pad(lab, 1, constant_values=FRAME)
    interior = np.ones((ny, nx), bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            interior &= p[dy:dy + ny, dx:dx + nx] == lab
    bits = np.zeros((ny, nx), np.uint8)
    code = np.zeros((ny, nx), np.uint8)
    used = lab != 1
    used |= ~interior & (lab == 1) & False

    def put(path, e0, e1):
        """Lay an arm (list of (j, i)) and its mirror images; returns False (nothing drawn) if it does not fit."""
        imgs = [[(cy + sy * (j - cy), cx + sx * (i - cx)) for j, i in path] for sx in (1, -1) for sy in (1, -1)]
        flat = {t for im in imgs for t in im}
        if any(not (1 <= j < ny - 1 and 1 <= i < nx - 1) or used[j, i] or lab[j, i] != 1 for j, i in flat):
            return False
        for im in imgs:
            for (j0, i0), (j1, i1) in zip(im, im[1:]):
                mb, tb = {(1, 0): (1, 4), (0, 1): (2, 8), (-1, 0): (4, 1), (0, -1): (8, 2)}[(j1 - j0, i1 - i0)]
                bits[j0, i0] |= mb | 32
                bits[j1, i1] |= tb | 32
            code[im[0]], code[im[-1]] = max(code[im[0]], e0), max(code[im[-1]], e1)
        for j, i in flat:
            used[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2] = True
        return True

    # the spine: lozenge tip -> diamond -> ... -> an arrowhead pointing back to the centre
    stops = [half + 2] + [v for yj in joints for v in (yj - 2, yj + 2)] + [top]
    for a, b in zip(stops[0::2], stops[1::2]):
        if b > a:
            put([(cy + v, cx) for v in range(a, b + 1)], 0, 0)
    put([(cy + v, cx) for v in range(ny // 2 - 2, top + 2, -1)], 0, ARROW_OPEN)       # a stub from the frame, arrow pointing in
    # stepped side hooks from each joint (and the lozenge), ending in a small open square
    for yj in [0] + joints:
        for _try in range(6):
            j, i = cy + yj, cx + (3 if yj == 0 else 2)
            path, up = [(j, i)], int(rng.choice((-1, 1)))
            for _k in range(int(rng.integers(2, 5))):
                for _r in range(int(rng.integers(2, 4))):
                    i += 1
                    path.append((j, i))
                for _r in range(int(rng.integers(1, 3))):
                    j += up
                    path.append((j, i))
            if put(path, 0, CAP):
                break
    # dumbbells floating near the side edges
    for _try in range(40):
        j, i = int(rng.integers(3, cy - 1)), int(rng.integers(cx + 5, nx - 5))
        put([(j, i), (j, i + 1), (j, i + 2)] if rng.random() < 0.6 else [(j, i), (j + 1, i), (j + 2, i)], TBAR, TBAR)
    arm = (bits & 32) > 0
    fill = (lab == 1) & ~arm                                                    # edge to edge: ring tiles are filled too
    fill |= interior & ~arm
    ring = ~interior
    order = np.round((np.abs(y) + 0.6 * np.abs(x)) / float(ny // 2 + 0.6 * (nx // 2)) * 255.0).astype(np.uint8)
    lo, hi = (5, 14) if rng.random() < 0.5 else (8, 30)
    bits |= meander_fill(fill, rng, lo, hi, lab, order)
    bits[fill | arm] |= 16
    out = np.zeros((ny, nx, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = lab, bits, code, order
    return out


def design_stats(d):
    lab, bits = d[..., 0].astype(int), d[..., 1].astype(int)
    interior = (bits & 16) > 0
    arm, block = (bits & 32) > 0, (bits & 64) > 0
    deg = sum(((bits & m) > 0).astype(int) for m in (1, 2, 4, 8))
    ny, nx = lab.shape
    recip, cross = True, 0
    for dj, di, mb, tb in ((1, 0, 1, 4), (0, 1, 2, 8)):
        a = bits[:ny - dj, :nx - di] & mb > 0
        b = bits[dj:, di:] & tb > 0
        recip &= bool(np.array_equal(a, b))
        cross += int((a & (lab[:ny - dj, :nx - di] != lab[dj:, di:])).sum())
    return dict(ring=float((~interior).mean()), interior=float(interior.mean()),
                dots=float(((bits & 15) == 0)[interior & ~block].mean()) if interior.any() else 1.0,   # tiles no line reached
                arms=float(arm.mean()), blocks=int(block.sum()),
                terminals=sorted(set((d[..., 2].astype(int) & 7)[arm & (deg == 1)].tolist())),
                block_frac=float(block.sum() / max(int(interior.sum()), 1)),
                reciprocal=recip and not (bits[~interior] & 15).any(), cross_label=cross,
                labels=len(np.unique(lab)))


def grid_dims(layout, scale, aspect=1.6, dense=False, form=1):
    """(gx, gy, variants, wrap_x) macro cells for a layout class: 0 single, 1 panels, 2 band, 3 medallion, 5 polar."""
    s = min(max(scale, 0.0), 1.0)
    k = float(dense) if dense else 1.0                   # painted / embroidered studies: a finer lattice (True = 1)
    k = 1.7 if dense is True else k
    if form == F_SPINE and layout == 0:                  # portrait 1 : 2
        gy = 2 * int(6 + 4 * s) + 1
        return (gy // 2) | 1, gy, 1, False
    if layout == 0:
        gy = 2 * int((4 + 5 * s) * k + 0.5)
        return 2 * int(gy * aspect * 0.86 / 0.9 / 2), gy, 1, False            # leaves room for a border band
    if layout == 2:
        gy = 2 * int(2.5 + 3 * s + 0.5)
        return 2 * int(1.4 * gy), gy, 1, True
    if layout == 3:
        g = 2 * int((4 + 5 * s) * k + 0.5)
        return g, g, 1, False
    g = 2 * int(3 + 3 * s + 0.5)
    return g, g, 4, False


# =====================================================================================================================
# Huni Kuin-inspired bead tiles (CPU)
# =====================================================================================================================
# ---- the three primitives (GRAMMAR_SPEC H2 - H4) ---------------------------------------------------------------
def fret(i, j, period=12, width=2, diagonal=False, hand=1):
    """FRET primitive: 0 / 1 on the integer lattice. Each period x period cell holds a square spiral (concentric
    square bands `width` cells wide, broken along the cell's axis so that one half slips by a ring pitch: the rings
    join up into a double spiral); neighbouring cells hold the COMPLEMENT, so figure and ground are the same fret, interlocked:
    fret(i + period, j) == 1 - fret(i, j). diagonal=True builds it on the 45 degree staircase lattice
    (u = (i + j) // 2, v = (i - j) // 2): there the complement sits at (i + period, j + period) and the pattern repeats
    every 2 * period along i or j. hand = -1 mirrors the spiral.
    Turns = period / (4 * width)."""
    i, j = np.asarray(i, int), np.asarray(j, int)
    u, v = ((i + j) >> 1, (i - j) >> 1) if diagonal else (i, j)      # diagonal: 1 : 1 staircases, strokes stay joined
    P, w = int(period), int(width)
    cu, cv = np.floor_divide(u, P), np.floor_divide(v, P)
    x, y = (u - cu * P) - (P - 1) / 2.0, (v - cv * P) - (P - 1) / 2.0
    y = np.where(x * hand > 0, y + 2 * w, y)            # the break: one half slips by one ring pitch, so ring k meets ring k + 2
    ring = np.floor((np.maximum(np.abs(x), np.abs(y)) + 0.5) / w).astype(int)
    return ((ring + cu + cv) % 2).astype(np.uint8)


def ombre(d, n=6, start=0):
    """OMBRÉ primitive: ring / chevron index d (cells from the frame) -> ramp role 1..n, exactly one cell per step,
    held at the ends."""
    return (1 + np.clip(np.asarray(d, int) - int(start), 0, int(n) - 1)).astype(np.uint8)


def meander_fill(mask, rng, lo=3, hi=7, lab=None, order=None, wrap_x=False):
    """MEANDER primitive: cover every tile of `mask` with lattice paths (bits 1 N, 2 E, 4 S, 8 W towards the next
    tile of the same path). Paths never branch or cross (degree <= 2), never leave the mask nor cross from one
    label to another; a tile no path could reach stays 0 (drawn as a dot). lo..hi = path length in tiles: short
    = knob-ended 'bones', long = a wandering labyrinth line. `order` = which tiles start first."""
    mask = np.asarray(mask, bool)
    ny, nx = mask.shape
    lab = np.zeros((ny, nx), int) if lab is None else lab
    bits = np.zeros((ny, nx), np.uint8)
    seen = ~mask
    dirs = [(1, 0, 1, 4), (0, 1, 2, 8), (-1, 0, 4, 1), (0, -1, 8, 2)]          # dj, di, my bit, their bit  (N E S W)
    jj, ii = np.nonzero(mask)
    key = np.zeros(len(jj)) if order is None else np.asarray(order)[mask]
    for _o, j, i in sorted(zip(key.tolist(), jj.tolist(), ii.tolist())):
        if seen[j, i]:
            continue
        seen[j, i] = True
        last, length = None, int(rng.integers(lo, hi + 1))
        for _step in range(length - 1):
            cand = []
            for k, (dj, di, mb, tb) in enumerate(dirs):
                j2, i2 = j + dj, (i + di) % nx if wrap_x else i + di
                if 0 <= j2 < ny and 0 <= i2 < nx and not seen[j2, i2] and lab[j2, i2] == lab[j, i]:
                    cand.append((k, j2, i2, mb, tb))
            if not cand:
                break
            turn = [c for c in cand if c[0] != last] or cand                   # the line turns: brackets, hooks, steps
            pool = turn if rng.random() < 0.7 else cand
            k, j2, i2, mb, tb = pool[int(rng.integers(0, len(pool)))]
            bits[j, i] |= mb
            bits[j2, i2] |= tb
            seen[j2, i2] = True
            j, i, last = j2, i2, k
    return bits


def fret_along(i, j, period=12, width=2, diagonal=False, hand=1):
    """0..1 position along the fret's spiral (ring + angle), for lighting beads one after another along it."""
    i, j = np.asarray(i, int), np.asarray(j, int)
    u, v = ((i + j) >> 1, (i - j) >> 1) if diagonal else (i, j)
    P, w = int(period), int(width)
    x, y = (u % P) - (P - 1) / 2.0, (v % P) - (P - 1) / 2.0
    y2 = np.where(x * hand > 0, y + 2 * w, y)
    ring = np.floor((np.maximum(np.abs(x), np.abs(y2)) + 0.5) / w)
    return ((ring + np.arctan2(y, x) / TAU) / max(P / (2.0 * w), 1.0)) % 1.0


def _role(k, cycle=6):
    """Alternating dark / light bands (README 2.3): odd bands are ground (0), even ones step through the colours."""
    k = np.asarray(k, int)
    return np.where(k % 2 == 1, 0, 1 + (k // 2) % cycle).astype(np.uint8)


def hk_tile(family, seed=0, size=HK_T):
    """(size, size, 2) uint8: [..., 0] colour role (0 = ground, 1..6 = colourway entry), [..., 1] position along the
    motif's path (for lighting the beads in sequence). Periodic with HK_T. Geometry: my reading of the one-line
    structure notes in README 2.2 on a twill / bead lattice; a study of the family, not a traditional pattern."""
    j, i = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cen = lambda v, p: np.abs((v + p // 2) % p - p // 2)                    # distance to the nearest multiple of p
    P = 12
    a, b = cen(i, P), cen(j, P)
    ang = (np.arctan2(((j + 6) % P) - 6 + 1e-3, ((i + 6) % P) - 6 + 1e-3) / TAU) % 1.0
    if family == 0:                                     # txede bedu: the foundational diamond (losango), nested
        d = np.minimum(a + b, (P // 2 - a) + (P // 2 - b))
        on2 = (P // 2 - a) + (P // 2 - b) < a + b
        role = _role(d // 2 if seed % 2 == 0 else d, 3)                      # bands two beads wide (odd seeds: one)
        role = np.where((d == 0), 3, role)                                  # the eye
        along = np.where(on2, (ang + 0.5) % 1.0, ang)
    elif family == 1:                                   # dunu kenê: diamond chain down a spine + stepped meander echoes
        a2, b2 = cen(i, 24), cen(j, 8)
        s = a2 + b2
        role = _role(s // 2 + 1, 3)
        role = np.where(s <= 1, 2, role)                                    # spine diamonds have a core
        along = (j / float(HK_T)) % 1.0
    elif family in (2, 3):                              # inu tae: angular stepped / branched motif
        pad = np.maximum(a, b) <= 1
        toes = (np.abs(a - b) <= 0) & (np.maximum(a, b) >= 2) & (np.maximum(a, b) <= 4)
        claws = ((a == 5) & (b == 4)) | ((a == 4) & (b == 5))
        step = ((a == 3) & (b <= 1)) | ((b == 3) & (a <= 1))                 # stepped heel between the branches
        role = np.where(pad, 1, np.where(toes | claws, 2, np.where(step, 3, 0)))
        if family == 3:                                 # inu tae txede bedu: the paw nested with the diamond eye
            d = a + b
            role = np.where(np.maximum(a, b) <= 0, 3, np.where((d <= 2), 1, 0))
            role = np.where((np.abs(a - b) <= 0) & (a >= 2) & (a <= 3), 2, role)
            role = np.where(d == 5, 3, role)                    # the diamond round it
            role = np.where((d >= 8) & (d % 2 == 0), 1, role)
            role = np.where(((a == 0) & (b == 4)) | ((b == 0) & (a == 4)), 2, role)
        else:
            d2 = (P // 2 - a) + (P // 2 - b)
            role = np.where((d2 <= 2) & (d2 % 2 == 0), 3, role)              # small diamonds between the paws
            role = np.where(((a == 6) & (b <= 2)) | ((b == 6) & (a <= 2)), 1, role)
        along = ang
    elif family == 4:                                   # kapë hina: stepped zigzag, serrated scales
        a3 = cen(i, 8)
        s = j + a3 + (i % 2) * (1 if seed % 2 == 0 else 0)
        role = _role(s // 2, 6)
        along = (i / float(HK_T)) % 1.0
    elif family == 5:                                   # D woven bands, stacked: three band types of 16 rows each
        band = ((j + 8) // 16 + seed) % 3               # (14 pattern rows + a black row + a dotted orange selvedge)
        y = (j + 8) % 16 - 7.5
        ay = np.abs(y)
        key = fret(i, j, 8, 1, diagonal=True)                               # 45 degree staircase key hooks, 2 turns
        r0 = np.where(key == 1, np.where(ay <= 1.5, 2, 1), 0)               # white on black, orange along the mid-line
        dd = (cen(i, 16) + np.floor(ay)).astype(int)                        # concentric diamonds, tips touching:
        r1 = np.where((dd // 2) % 2 == 0, 2, 0)                              # nested outlines two floats wide, orange / black
        r1 = np.where((dd // 2 == 4) | (dd // 2 == 6), 1, r1)                # stepped chevrons between them in white
        e = (cen(i, 6) + np.floor(ay)).astype(int)                          # chain of small orange diamonds, tip to tip,
        r2 = np.where(e <= 2, np.where(e == 0, 0, 2), np.where((e == 4) | (e == 5) | (e == 8), 1, 0))   # white zigzag echoes
        role = np.where(band == 0, r0, np.where(band == 1, r1, r2))
        role = np.where(ay > 7.0, np.where(i % 2 == 0, 2, 0), np.where(ay > 6.0, 0, role))
        along = np.where(band == 0, fret_along(i, j, 8, 1, True), (i / 24.0) % 1.0)
    elif family == 6:                                   # E bead tiles: a grid of four different 22 x 22 tiles, gaps between
        ti, tj = (i // 24) % 2, (j // 24) % 2
        x, y = (i % 24) - 11.5, (j % 24) - 11.5
        ax, ay = np.abs(x), np.abs(y)
        d = (ax + ay).astype(int)                                           # 1, 2, ... from the tile centre outward
        frame = ombre(17 - d, 6)                                            # six ombré rings, one bead each, corners in
        xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
        key = (fret(xi + 6, yi + 6, 12, 2, hand=1 if seed % 2 == 0 else -1) == 1) & (np.maximum(ax, ay) < 6)   # big S-fret,
        t0 = np.where(d <= 10, np.where(key, WHITE_ROLE, 0), np.where(d == 11, 0, frame))                    # 2 beads wide
        eyes = (np.abs(ax - 3.0) <= 1.0) & (np.abs(ay - 3.0) <= 1.0)        # four eyes: 2 x 2 squares, outlined
        ring = (np.abs(ax - 3.0) <= 2.0) & (np.abs(ay - 3.0) <= 2.0) & ~eyes
        t1 = np.where(d <= 10, np.where(eyes, 7, np.where(ring, 8, 0)), np.where(d == 11, 0, frame))
        ov = np.sqrt(((ax - 4.5) / 3.0) ** 2 + (ay / 6.5) ** 2)             # eye pair: two upright ovals on an ombré ground
        t2 = np.where(ov <= 1.0, np.where((np.abs(ax - 4.5) < 1.0) & (ay < 3.0), 0, WHITE_ROLE),
                      np.where(ov <= 1.25, 0, ombre(np.floor((ov - 1.25) * 5.0).astype(int), 6)))
        rib = (fret(xi, yi, 8, 1, diagonal=True) == 1) & (np.abs(x - y) <= 9.5)                 # diagonal key ribbon
        t3 = np.where(rib, WHITE_ROLE, np.where(np.abs(x - y) <= 10.5, 0, ombre(((xi + yi + 24) // 3) % 6, 6)))
        t3 = np.where(np.abs(x - y) <= 9.5, np.where(rib, WHITE_ROLE, ombre(((xi + yi + 24) // 4) % 6, 6)), t3)
        role = np.where(ti == 0, np.where(tj == 0, t0, t2), np.where(tj == 0, t1, t3))
        role = np.where((ax > 11.0) | (ay > 11.0), 0, role)                 # the gap between tiles
        along = np.where((ti == 0) & (tj == 0), fret_along(xi + 6, yi + 6, 12, 2), ang)
    elif family == 7:                                   # F three vertical straps, 13 beads wide, dark ground between
        strap = ((i // 16) + seed) % 3
        x, y = (i % 16) - 7.0, (j % 16) - 7.5
        a, b = np.abs(x), np.abs(y)
        # strap 1: an X of ombré chevron arms; nested-ring diamonds with a white cross between the X's
        e = np.floor(np.abs(a - b + 0.5)).astype(int)
        s1 = np.where(e <= 3, ombre(e, 4), np.where(e == 4, 8, 0))
        dd = np.floor(a + (8.0 - b)).astype(int)
        s1 = np.where((b - a > 4.5), np.where(dd <= 1, WHITE_ROLE, np.where(dd % 2 == 0, 5, 6)), s1)
        s1 = np.where((a - b > 4.5), np.where(e % 2 == 0, 8, 0), s1)
        # strap 2: nested chevrons in greens pointing at a dark X, a white diamond outline at its centre
        v = np.floor(b - a + 8.0).astype(int)
        s2 = np.choose(v % 4, [WHITE_ROLE, 4, 5, 6])
        s2 = np.where(np.abs(a - b + 0.5) < 1.0, 6, s2)
        s2 = np.where(np.floor(a + b).astype(int) == 3, WHITE_ROLE, np.where(a + b < 3.0, 0, s2))
        # strap 3: a chain of rhombi each holding four small diamonds; rainbow rays fan out to the edges
        q = a + b * 0.8
        four = (np.abs(a - 2.0) + np.abs(b - 2.5)) <= 1.0
        s3 = np.where(q <= 5.0, np.where(four, 7, 8), np.where(q <= 5.8, 0, ombre(np.floor(q - 5.8).astype(int), 6)))
        s3 = np.where(q > 11.8, 0, s3)
        role = np.where(strap == 0, s1, np.where(strap == 1, s2, s3))
        role = np.where(a > 6.0, 0, role)                                   # 13 beads, then dark ground
        along = (j / 16.0) % 1.0
    else:                                               # G key-fret panels: red and green T-frets, 2-bead lines with
        A, B = cen(i, 48), cen(j, 48)                   # 1-bead dark separators, in diamonds; blue chevron bands between
        d = np.minimum(A + B, 48 - (A + B))
        ci, cj = i // 3, j // 3
        f = fret(ci, cj, 8, 1, hand=1 if seed % 2 == 0 else -1).astype(int)
        sep = ((i % 3 == 2) & (f != fret(ci + 1, cj, 8, 1, hand=1 if seed % 2 == 0 else -1))) | \
              ((j % 3 == 2) & (f != fret(ci, cj + 1, 8, 1, hand=1 if seed % 2 == 0 else -1)))
        panel = np.where(sep, 0, np.where(f == 1, 7, 8))
        chev = ombre(d - 19, 5)                                             # white peak line next to the panel, then
        role = np.where(d <= 17, panel, np.where(d == 18, 7, chev))         # sky, periwinkle, royal, navy
        along = fret_along(ci, cj, 8, 1)
    out = np.zeros((size, size, 2), np.uint8)
    out[..., 0], out[..., 1] = role, np.round(along * 255.0).astype(np.uint8)
    return out


def hk_atlas(seed=0):
    return np.concatenate([hk_tile(k, seed) for k in range(NF)], axis=1)


def hk_atlas_all():
    """Both tile variants side by side (variant 0 families, then variant 1): shader family index = family + NF * variant."""
    return np.concatenate([hk_atlas(0), hk_atlas(1)], axis=1)


def _bright(c):
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return c / max(float(c.max()), 1e-3)


def hk_colours(values, palette):
    k = _idx(values, "hk_colors", len(HK_COLORWAYS), 1)
    cols = HK_COLORWAYS[k]
    if cols is None:
        pal = palette if palette is not None and len(palette) else [_B_RED, _B_YEL, _B_TURQ, _B_COB, _B_EMER]
        cols = [tuple(_bright(pal[n % len(pal)])) for n in range(8)]
    return [tuple(float(x) for x in c) for c in cols]


# =====================================================================================================================
# colours per medium, pen cycle (CPU)
# =====================================================================================================================
def style_uniforms(values, palette):
    style = _idx(values, "kene_style", len(STYLES))
    cset = _idx(values, "kene_colors", len(COLOR_SETS), 2)
    wgt = _val(values, "kene_weight", 0.5, 0.0, 1.0)
    w = [0.17 + 0.12 * wgt, 0.07 + 0.05 * wgt, 0.05 + 0.04 * wgt]
    if cset == 0:
        pal = palette if palette is not None and len(palette) else list(THREADS[2])
        pal = [_bright(pal[n % len(pal)]) for n in range(4)]
        cols = [tuple(pal[0]), tuple(_bright(pal[2] * 0.75 + 0.25)), tuple(pal[1]), tuple(pal[3])]
    elif cset == 1:
        cols = list(COTTON)
    else:
        cols = list(THREADS[cset])
    out = dict(ground=(0.0, 0.0, 0.0), ground_on=0.0, stitch=0.0, bead=0.0, tier=(1.0, 0.9, 0.72), border=0.0,
               acc=ACCENTS, arm=None, echo2=None, dense=False, arm_w=None, acc_on=1.0, twig=0.0, frame=None, wob=0.0,
               form=None, hair=0.0, knob=1.0, cborder=0.0)
    if cset == 0:
        out["acc"] = tuple(tuple(_bright(pal[n])) for n in (3, 1, 2))
    if style == 1:                                      # mud-dyed cotton: chocolate / black on cream, one ink
        cols = [MUD_INK, MUD_INK, MUD_INK, tuple(1.4 * x for x in MUD_INK)]
        out.update(ground=CLOTH_GROUND, ground_on=1.0, tier=(1.0, 1.0, 1.0))
        w = [w[0], w[1] * 0.9, w[2] * 1.5]
    elif style == 2:
        out.update(stitch=1.0)
    elif style == 3:                                    # ceramic: bold black + fine red-oxide and black lines on cream slip
        cols = [CERAMIC_BLACK, OXIDE_RED, tuple(2.2 * x for x in CERAMIC_BLACK), OXIDE_RED]
        out.update(ground=SLIP_GROUND, ground_on=1.0, tier=(1.0, 1.0, 1.0))
        w = [w[0] * 1.25, w[1] * 0.75, w[2] * 0.8]
    elif style == 4:
        out.update(bead=3.0, tier=(1.0, 1.0, 0.9))
    elif style == 5:                                    # A painted: bold ink lines on cream, a dense fine labyrinth
        cols = [PAINT_INK] * 4                                              # fill, small red / ochre / black filled blocks
        out.update(ground=PAINT_GROUND, ground_on=1.0, tier=(1.0, 1.0, 1.0), acc=PAINT_ACC, dense=2.2, arm_w=0.34, wob=0.3,
                   form=F_LATTICE, hair=1.0, knob=0.0)
        w = [0.52 + 0.14 * wgt, 0.0, 0.115 + 0.03 * wgt]                    # line ~1.2 tiles; fill: spacing 1 tile, ~3.7 x its width
    elif style == 6:                                    # B cloth: hand-drawn ink on a cream panel in a wide dark border,
        cols = [CLOTH_INK] * 4                                              # coarser fill, no colour
        out.update(ground=CLOTH_FIELD, ground_on=1.0, tier=(1.0, 1.0, 1.0), acc_on=0.0, frame=CLOTH_BORDER, wob=0.9,
                   form=F_SPINE, knob=0.0)
        w = [0.28 + 0.08 * wgt, 0.0, 0.11 + 0.03 * wgt]
    elif style == 7:                                    # C embroidered: magenta cloth, navy channels, yellow + orange
        cols = [(0.10, 0.14, 0.42), (0.95, 0.54, 0.11), (0.24, 0.20, 0.72), (0.27, 0.19, 0.68)]   # double outline, violet fill
        out.update(ground=MAG_GROUND, ground_on=1.0, tier=(1.0, 1.0, 1.0), stitch=1.0, border=1.0, echo2=(0.96, 0.82, 0.13),
                   arm=(0.10, 0.63, 0.71), acc=((0.94, 0.42, 0.77), (0.22, 0.71, 0.29), (0.78, 0.94, 0.78)), dense=1.25,
                   twig=1.0, arm_w=0.24, form=F_CELLS, knob=0.0, cborder=1.0)
        w = [0.24, 0.20, 0.20]                           # orange outer + yellow inner outline; fill line = half its spacing
    if style == 2:
        out["border"] = 1.0
    out.update(cols=[tuple(float(x) for x in c) for c in cols], w=tuple(w))
    out["arm"] = tuple(out["arm"] or out["cols"][0])
    out["echo2"] = tuple(out["echo2"] or out["cols"][1])
    return out


def pen_state(values, beat):
    """{progress 0..1, seed, fade}: how much of the design is drawn. With `kene_pen` on, the design is drawn over the
    first 72 % of every cycle, held, faded out over the last 6 %, and (kene_pen_new) replaced while nothing shows."""
    seed = _idx(values, "kene_seed", N_SEEDS)
    draw = _val(values, "kene_draw", 1.0, 0.0, 1.0)
    bars = PEN_BARS[_idx(values, "kene_pen", len(PEN))]
    if not bars:
        return dict(progress=draw, seed=seed, next_seed=seed, fade=1.0)
    cyc = max(_num(beat), 0.0) / (4.0 * bars)
    n, fr = int(cyc), cyc - int(cyc)
    if _idx(values, "kene_pen_new", 2, 1):
        seed = (seed + n) % N_SEEDS
    x = min(max((fr - 0.94) / 0.06, 0.0), 1.0)
    y = min(max(fr / 0.06, 0.0), 1.0)                  # ... and fades back in: a lit cloth / slip ground never cuts on
    nxt = (seed + 1) % N_SEEDS if _idx(values, "kene_pen_new", 2, 1) else seed
    return dict(progress=min(fr / 0.72, 1.0) * draw, seed=seed, next_seed=nxt,
                fade=min(1.0 - x * x * (3.0 - 2.0 * x), y * y * (3.0 - 2.0 * y)))


# =====================================================================================================================
# GLSL
# =====================================================================================================================
VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos,0,1); }
"""

FS = """
#version 330
uniform sampler2D uD;          // design tiles: R label, G fill bits, B fill colour, A pen order; variants stacked in y
uniform sampler2D uH;          // bead tiles: R colour role, G position along the motif; families side by side
uniform vec2 uRes;
uniform vec3 uGrid;            // tiles x, tiles y (one variant), variants
uniform int uLay;              // 0 single panel, 1 panels, 2 band, 3 medallion, 5 polar (mandala / tunnel)
uniform float uWrap, uAmtS, uAmtH, uFolds, uRot;
uniform vec2 uScroll;          // panels: cells (wrapped at 8); band: x = fraction of the band period
uniform float uPolar;          // polar ring scroll (wrapped at 8: the variant AND the pen rank of a ring repeat there)
uniform vec3 uCol[4];          // primary, secondary, fill, fill in alternate regions
uniform vec3 uGround, uTier, uW;
uniform vec3 uArm, uEcho2, uAcc[3];   // free arms, the outer half of a two-tone echo line, accent blocks
uniform float uBorder, uPH;           // diamond-chain border band round a single panel; the panel's height
uniform float uArmW, uAccOn, uTwig, uWob, uFrameOn, uHOmb, uHair, uKnob, uCBorder, uHEdge;
uniform vec3 uFrame;
uniform float uGroundOn, uStitch, uBead, uRound, uCurve, uFill, uDraw, uFlow, uFlowPh, uBreath, uGlow;
uniform float uChase, uChasePh, uChaseW, uBass, uHigh;
uniform float uHB, uHNB, uHMed, uHSeq, uHSeqPh, uHGlow, uHMorph, uHT;
uniform vec2 uHFam, uHScroll;  // family now / next; bead scroll (wrapped at the tile period)
uniform float uHPolar;
uniform vec3 uHC[8];
in vec2 v_uv; out vec4 f;
const float TAU = 6.28318530718, PI = 3.14159265359;
float max3(vec3 c){ return max(c.r, max(c.g, c.b)); }
mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }
float tri(float x){ return abs(fract(x)-0.5)*2.0; }

// antialiased stroke: d = distance to the centre line, w = half width, fw = one pixel (all in the same units).
// A stroke thinner than a pixel keeps a pixel of width and dims instead: no shimmer.
float stroke(float d, float w, float fw){
  float ww = max(w, 0.55*fw);
  return (1.0 - smoothstep(ww-0.7*fw, ww+0.7*fw, d)) * min(1.0, w/ww);
}
float seg(vec2 p, vec2 e){ return length(p - e*clamp(dot(p,e)/dot(e,e), 0.0, 1.0)); }

vec4 tileAt(ivec2 ij, float v){
  int nx = int(uGrid.x), ny = int(uGrid.y);
  if (uWrap > 0.5) ij.x = int(mod(float(ij.x), uGrid.x));
  if (ij.x < 0 || ij.y < 0 || ij.x >= nx || ij.y >= ny) return vec4(1.0, 0.0, 0.0, -1.0);
  return texelFetch(uD, ivec2(ij.x, ij.y + int(v)*ny), 0);
}

// g: position in fine tiles; fw: one pixel in tiles; v: design variant; prog: pen progress; cloth: lit ground mask
vec4 shipibo(vec2 g, float fw, float v, float prog, float cloth){
  float disc = 1.0, sfw = fw;
  if (uWob > 0.0) g += uWob*0.07*vec2(sin(g.y*1.05 + 0.6*sin(g.x*0.37)), sin(g.x*0.95 + 1.7 + 0.6*sin(g.y*0.31)));   // hand-drawn
  if (uBead > 0.0) {                                   // beadwork: one decision per bead, drawn as a round bead
    vec2 gb = g*uBead, bc = fract(gb)-0.5; float bfw = fw*uBead;
    disc = (1.0 - smoothstep(0.40-bfw, 0.40+bfw, length(bc))) * (0.72 + 0.5*(1.0-smoothstep(0.0, 0.30, length(bc-vec2(-0.10, 0.12)))));
    disc = mix(disc, 0.80, smoothstep(0.30, 0.55, bfw));   // beads under ~3 px: plain lines, no moire
    g = (floor(gb)+0.5)/uBead; fw = 0.01;
  }
  ivec2 ij = ivec2(floor(g)); vec2 c = fract(g)-0.5;
  vec4 t0 = tileAt(ij, v);
  float dO = 9.0, A[9], a0 = t0.a, amax = 0.0;
  for (int k = 0; k < 9; k++) {
    ivec2 o = ivec2(k%3-1, k/3-1);
    vec4 tn = k == 4 ? t0 : tileAt(ij+o, v);
    A[k] = tn.a; amax = max(amax, tn.a);
    if (abs(tn.r-t0.r) > 0.002) {
      vec2 e = max(abs(c-vec2(o))-0.5, 0.0);
      dO = min(dO, mix(max(e.x, e.y), length(e), uRound));
    }
  }
  if (a0 < 0.0) a0 = amax;                              // outside the field: the frame line takes its neighbours' order
  for (int k = 0; k < 9; k++) if (A[k] < 0.0) A[k] = a0;
  int sx = c.x >= 0.0 ? 1 : -1, sy = c.y >= 0.0 ? 1 : -1;
  float ord = mix(mix(a0, A[4+sx], abs(c.x)), mix(A[4+3*sy], A[4+sx+3*sy], abs(c.x)), abs(c.y));

  float br = 1.0 + 0.16*uBreath;
  float aP = stroke(dO, uW.x*br, fw);
  if (uHair > 0.5) aP = max(stroke(dO, uW.x*br - 0.30, fw), stroke(abs(dO - uW.x*br + 0.07), 0.07, fw));   // bold line + parallel hairline
  float aS = stroke(abs(dO-0.5), uW.y*br, fw);
  int b = int(t0.g*255.0+0.5), cb = int(t0.b*255.0+0.5), code = cb & 7; float dT = 9.0, dFill = 9.0, dTerm = 9.0, aArm = 0.0, aBlk = 0.0;
  if ((b & 16) != 0) {
    int deg = 0; vec2 cn = vec2(0.0);
    if ((b & 1) != 0) { dT = min(dT, seg(c, vec2(0.0, 0.5)));  deg++; cn.y = 0.5; }
    if ((b & 2) != 0) { dT = min(dT, seg(c, vec2(0.5, 0.0)));  deg++; cn.x = 0.5; }
    if ((b & 4) != 0) { dT = min(dT, seg(c, vec2(0.0, -0.5))); deg++; cn.y = -0.5; }
    if ((b & 8) != 0) { dT = min(dT, seg(c, vec2(-0.5, 0.0))); deg++; cn.x = -0.5; }
    if (deg == 2 && cn.x != 0.0 && cn.y != 0.0) dT = mix(dT, abs(length(c-cn)-0.5), uRound);   // curving elbow
    if ((b & 32) != 0) {                               // an arm: a free primary line, with a terminal where it ends
      if (deg == 1) {
        vec2 dv = cn*2.0, pv = vec2(-dv.y, dv.x); float fa = dot(c, -dv), fb = abs(dot(c, pv));
        if (code == 1) dFill = max(fb - (0.42-fa)*0.80, -fa-0.16);                              // arrowhead
        else if (code == 4) dTerm = abs(max(fb - (0.36-fa)*0.80, -fa-0.16));                    // arrowhead, outline only
        else if (code == 2) dTerm = seg(c + pv*0.36, pv*0.72);                                  // T-bar
        else if (code == 3) dTerm = abs(max(abs(c.x), abs(c.y)) - 0.30);                        // small end-cap
      }
      // terminals are drawn with a stroke that stays inside the tile (nothing is ever cut at a tile edge: no shimmer)
      aArm = max(max(stroke(dT, uArmW*br, fw), stroke(dTerm, min(uArmW, 0.12), fw)), 1.0 - smoothstep(-fw, fw, dFill));
      dT = 9.0;
    } else if ((b & 64) != 0) {                        // a small filled accent block
      aBlk = 1.0 - smoothstep(-fw, fw, max(abs(c.x), abs(c.y)) - (code == 3 ? 0.5 : 0.36));
      if (uTwig > 0.5) aBlk = 1.0 - smoothstep(-fw, fw, length(c) - 0.34);                       // an eye dot
      dT = 9.0;
    } else {
      if (deg == 0) dT = length(c) - 1.6*uW.z;                                                // a dot: the seed in an eye
      if (deg == 1 && uKnob > 0.5) dT = min(dT, length(c) - 1.0*uW.z);                        // knob end
    }
  }
  float aT = stroke(dT, uW.z, fw) * uFill;

  // the pen: order 0 (centre) .. 1 along the line network; primary first, its echo next, the fill last
  float e = 0.035, o = ord*0.55;
  float rP = smoothstep(o, o+e, prog), rS = smoothstep(o+0.12, o+0.12+e, prog), rT = smoothstep(o+0.30, o+0.30+e, prog);
  float head = max(aP, aArm)*rP*(1.0-smoothstep(o+e, o+4.0*e, prog)) + aS*rS*(1.0-smoothstep(o+0.12+e, o+0.12+4.0*e, prog))
             + aT*rT*(1.0-smoothstep(o+0.30+e, o+0.30+3.0*e, prog));
  head *= step(prog, 0.999);
  // light travelling the primary paths outward from the centre, one pulse train per bar
  float x = fract(ord*3.0 - uFlowPh), pulse = x*x*x*x*(1.0-smoothstep(0.94, 1.0, x));
  float cw = abs(fract(ord - uChasePh + 0.5) - 0.5)*2.0;
  float lit = mix(1.0, 0.10 + 0.90*(1.0 - smoothstep(0.5*uChaseW, uChaseW, cw)), uChase);
  float st = 1.0 - uStitch*0.30*(0.5+0.5*sin((g.x+g.y)*TAU*4.0))*(1.0-smoothstep(0.06, 0.16, sfw));   // chain-stitch texture

  vec3 col = uGround*uGroundOn*cloth;
  int tw = (cb >> 3) & 7;
  vec3 cT = cb >= 128 ? uCol[3] : uCol[2], cS = dO < 0.5 ? uCol[1] : uEcho2, cB = code == 3 ? uCol[0] : uAcc[min(code, 2)];
  vec3 cA = (uTwig > 0.5 && tw > 0) ? uAcc[(tw-1) % 3] : uArm;
  float kP, kS, kT, kA = aArm*rP, kB = aBlk*rT*uAccOn;
  vec3 cP = uCol[0]; float aGem = 0.0; vec3 cG = uCol[0];
  if ((b & 16) == 0 && cb > 0) {                       // ring tile: a coloured run of the line / a junction gem
    int pc = cb & 7;
    if (pc > 0) cP = pc == 4 ? uArm : (pc == 5 ? uEcho2 : uAcc[(pc-1) % 3]);
    if ((cb & 8) != 0) {
      int k = (cb >> 4) & 3;
      vec2 cn2 = vec2((k == 0 || k == 3) ? 0.5 : -0.5, k <= 1 ? 0.5 : -0.5);
      aGem = (1.0 - smoothstep(-fw, fw, max(abs(c.x-cn2.x), abs(c.y-cn2.y)) - 0.46))*rP;
      cG = (cb & 64) != 0 ? uAcc[0] : uCol[0];
    }
  }
  if (uGroundOn > 0.5) {                               // ink on cloth / slip: the light travels as a soft lift of the ground
    kP = aP*rP; kS = aS*rS; kT = aT*rT;
    col *= 0.90 + 0.10*uBreath + 0.10*uFlow*pulse*lit;
    col = mix(col, cT, kT); col = mix(col, cS, kS); col = mix(col, cP, kP); col = mix(col, cG, aGem);
    col = mix(col, cA, kA); col = mix(col, cB, kB);
    col = mix(col, uGround*1.15, 0.5*head);
  } else {
    float bP = uTier.x*(0.70 + 0.30*uFlow*pulse + 0.10*uBass)*lit*st;
    float bS = uTier.y*(0.78 + 0.22*uFlow*pulse)*lit*st;
    float bT = uTier.z*(0.80 + 0.20*uHigh)*mix(1.0, lit, 0.7)*st;
    kP = aP*rP; kS = aS*rS; kT = aT*rT;
    col = mix(col, cT*bT, kT); col = mix(col, cS*bS, kS); col = mix(col, cP*bP, kP); col = mix(col, cG*bP, aGem);
    col = mix(col, cA*bP, kA); col = mix(col, cB*(0.80 + 0.20*uBass)*lit, kB);
    float halo = 1.0 - smoothstep(uW.x, 1.0, dO);
    col += uCol[0]*uGlow*0.30*halo*halo*rP*lit*(1.0-kP)*(0.6+0.4*uFlow*pulse);
    col += mix(uCol[0], vec3(1.0), 0.6)*head*0.55;
  }
  float al = max(uGroundOn*cloth, max(max(kP, max(kA, aGem)), max(kS, max(kT, kB))));
  return vec4(col*disc, uBead > 0.0 ? al*min(1.0, disc*1.4) : al);
}

vec4 ringLines(float d, float fw, float prog){          // decorative border lines round a field (d in tiles, outward)
  float a1 = stroke(abs(d-1.7), uW.y, fw), a2 = stroke(abs(d-2.5), uW.x*0.6, fw);
  vec3 c1 = uGroundOn > 0.5 ? uCol[1] : uCol[1]*uTier.y*0.8, c2 = uGroundOn > 0.5 ? uCol[0] : uCol[3]*uTier.x*0.7;
  float r = smoothstep(0.0, 0.08, prog);
  return vec4(mix(c1*a1, c2, a2)*r, max(a1, a2)*r);
}

vec4 shipiboLayout(vec2 P, float px){
  vec2 N = uGrid.xy;
  if (uLay == 0 || uLay == 3) {                        // one bounded field: rectangle or medallion
    float ts = (uLay == 0 ? uPH : 0.86)/N.y;
    vec2 q = P;
    if (uCurve > 0.0) {                                // curving lines: a gentle swirl that leaves the frame straight
      vec2 u = q/(0.5*N*ts); float env = (1.0-min(u.x*u.x, 1.0))*(1.0-min(u.y*u.y, 1.0));
      if (uLay == 3) env = 1.0;
      q = rot(uCurve*0.16*env*sin(PI*min(length(u), 1.0)))*q;
      q += uCurve*0.75*ts*env*vec2(sin(3.0*PI*u.y)+0.5*sin(5.0*PI*u.x), -sin(3.0*PI*u.x)-0.5*sin(5.0*PI*u.y));
    }
    vec2 g = q/ts + 0.5*N; float fw = px/ts*(1.0+0.6*uCurve);
    if (uLay == 0) {
      vec2 o = abs(P) - 0.5*N*ts, oc = o - (uBorder > 0.5 ? 7.4 : (uFrameOn > 0.5 ? 0.06/ts : 1.3))*ts;
      float cloth = 1.0 - smoothstep(-px, px, max(oc.x, oc.y));
      vec4 s = shipibo(g, fw, 0.0, uDraw, cloth);
      float d = max(o.x, o.y)/ts;
      if (uFrameOn > 0.5) {                            // a cream panel inside a wide dark cloth border
        float outp = smoothstep(-fw, fw, d - 0.9);
        s.rgb = mix(s.rgb, uFrame*cloth, outp); s.a = max(s.a, outp*cloth);
      }
      if (uBorder > 0.5 && d > 1.15) {                 // border band: a chain of diamonds in a lattice, rails, zig-zag edge
        float sc = (o.x > o.y ? P.y : P.x)/ts, u = fract(sc/3.0)-0.5, v = (d-3.5)/3.0, dd = abs(u)+abs(v), fb = fw/3.0;
        float inb = step(abs(v), 0.5), r = smoothstep(0.0, 0.08, uDraw);
        float dia = (1.0 - smoothstep(0.27-fb, 0.27+fb, dd))*inb, lat = stroke(abs(dd-0.42), 0.045, fb)*inb;
        float rails = stroke(abs(abs(d-3.5)-1.75), uW.y*1.4, fw), zig = stroke(abs(d - 6.3 - 0.8*(tri(sc/1.5)-0.5)), uW.y*1.3, fw);
        float lum = uGroundOn > 0.5 ? 1.0 : 0.85;
        vec3 bc = s.rgb;
        if (uCBorder > 0.5) {                          // zig-zag, purple band, white field of the chain, navy + crimson stripes
          float zin = stroke(abs(d - 1.15 - 0.35*(tri(sc/1.2)-0.5)), 0.10, fw);
          bc = mix(bc, vec3(0.42, 0.14, 0.52), step(1.5, d)*step(d, 1.95)*r);
          bc = mix(bc, vec3(0.93, 0.92, 0.86), inb*r);
          bc = mix(bc, vec3(0.94, 0.42, 0.77), stroke(abs(dd-0.5), 0.05, fb)*inb*r);
          bc = mix(bc, uCol[0], step(5.05, d)*step(d, 5.5)*r); bc = mix(bc, vec3(0.80, 0.08, 0.22), step(5.5, d)*step(d, 5.95)*r);
          bc = mix(bc, vec3(0.95, 0.93, 0.80), zin*r);
          lat = stroke(abs(dd-0.40), 0.06, fb)*inb; rails = 0.0;
        }
        bc = mix(bc, (uCBorder > 0.5 ? vec3(0.97, 0.66, 0.10) : uAcc[1])*lum, dia*r);
        bc = mix(bc, (uCBorder > 0.5 ? vec3(0.45, 0.80, 0.20) : (uGroundOn > 0.5 ? vec3(0.95, 0.95, 0.88) : uCol[2]))*lum, lat*r);
        bc = mix(bc, uCol[0]*(uGroundOn > 0.5 ? 1.0 : 0.8), rails*r); bc = mix(bc, uEcho2*lum, zig*r);
        s = vec4(bc, max(s.a, max(max(dia, lat), max(rails, zig))*r));
      }
      return s;
    }
    float R = 0.5*N.y*ts, r = length(P), dr = (r-R)/ts;
    float inside = 1.0 - smoothstep(-fw, fw, dr + 0.45);
    vec4 s = shipibo(g, fw, 0.0, uDraw, 1.0) * inside;
    float cloth = 1.0 - smoothstep(-fw, fw, dr - 4.6);
    vec4 b = vec4(0.0);
    float a0 = stroke(abs(dr), uW.x, fw) * smoothstep(0.0, 0.06, uDraw);
    vec4 rl = ringLines(dr, fw, uDraw);
    float zz = stroke(abs(dr - 3.6 - 0.45*(tri(atan(P.y, P.x)/TAU*floor(N.y*2.2))-0.5)), uW.y, fw) * smoothstep(0.02, 0.12, uDraw);
    vec3 cP = uGroundOn > 0.5 ? uCol[0] : uCol[0]*uTier.x*0.8;
    b.rgb = uGround*uGroundOn*cloth*(1.0-inside);
    b.rgb = mix(b.rgb, rl.rgb/max(rl.a, 1e-3), rl.a); b.rgb = mix(b.rgb, uGroundOn > 0.5 ? uCol[1] : uCol[2]*uTier.y*0.8, zz);
    b.rgb = mix(b.rgb, cP, a0);
    b.a = max(max(uGroundOn*cloth*(1.0-inside), rl.a), max(zz, a0));
    return vec4(s.rgb + b.rgb*(1.0-s.a), max(s.a, b.a));
  }
  if (uLay == 2) {                                     // band: a frieze that repeats along x, bordered above and below
    float ts = 0.5/N.y;
    vec2 g = vec2(P.x/ts + uScroll.x*N.x, P.y/ts + 0.5*N.y); float fw = px/ts;
    float dy = (abs(P.y) - 0.25)/ts;
    float cloth = 1.0 - smoothstep(-fw, fw, dy - 6.2);
    vec4 s = shipibo(g, fw, 0.0, uDraw, cloth);
    vec4 rl = ringLines(dy + 0.9, fw, uDraw);
    float zz = stroke(abs(dy - 4.4 - 0.9*(tri(g.x/3.0)-0.5)), uW.y, fw) * smoothstep(0.02, 0.12, uDraw);
    vec3 cz = uGroundOn > 0.5 ? uCol[1] : uCol[2]*uTier.y*0.8;
    s.rgb = mix(s.rgb, rl.rgb/max(rl.a, 1e-3), rl.a*step(1.6, dy)); s.rgb = mix(s.rgb, cz, zz);
    s.a = max(s.a, max(rl.a*step(1.6, dy), zz));
    return s;
  }
  // panels (flat) or polar panels (mandala / tunnel): every cell is a bounded field of its own
  vec2 pc; float sc;
  if (uLay == 1) { pc = P/0.5 + uScroll; sc = px/0.5; }
  else {
    float r = max(length(P), 1e-4), a = atan(P.y, P.x) + uRot;
    pc = vec2(a/TAU*uFolds, log(r)*uFolds/TAU + uPolar); sc = px*uFolds/(TAU*r);
  }
  vec2 id = floor(pc), loc = fract(pc)-0.5;
  if (uLay == 5) id.x = mod(id.x, uFolds);
  float ts = 0.84/N.y, fw = sc/ts;
  float v = mod(id.x + 2.0*mod(id.y, 2.0), uGrid.z);
  float rank = mod(id.x*3.0 + id.y*5.0, 8.0)/8.0;       // panel by panel: each cell is drawn in its turn
  float prog = clamp(uDraw*1.7 - 0.7*rank, 0.0, 1.0);
  vec2 o = abs(loc) - 0.5*N*ts - 1.2*ts;
  float cloth = 1.0 - smoothstep(-sc, sc, max(o.x, o.y));
  vec4 s = shipibo(loc/ts + 0.5*N, fw, v, prog, cloth);
  return s * (1.0 - smoothstep(0.30, 0.60, fw));         // too small to draw (centre of the tunnel): fade to black
}

// sine hash in the form used in Inigo Quilez's MIT-licensed shaders, Copyright (c) 2013 Inigo Quilez, MIT License (THIRD_PARTY_NOTICES.md)
float hash21(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7)))*43758.5453); }

vec3 beadCol(vec2 b, out float lit){                    // colour of bead b (integer lattice position) + its sequence light
  vec2 m = mod(b, uHT);
  vec4 tA = texelFetch(uH, ivec2(int(m.x + uHFam.x*uHT), int(m.y)), 0);
  vec4 t = tA;
  float dip = 1.0;
  if (uHMorph > 0.0) {                                  // motif morph: beads change over one by one along the diagonals
    float thr = 0.05 + 0.65*mod(m.x+m.y, uHT)/uHT + 0.25*hash21(m);
    if (uHMorph > thr) t = texelFetch(uH, ivec2(int(m.x + uHFam.y*uHT), int(m.y)), 0);
    dip = mix(1.9, 1.0, smoothstep(0.0, 0.07, abs(uHMorph-thr)));
  }
  float role = floor(t.r*255.0+0.5);
  float x = fract(t.g*2.0 + role*0.17 - uHSeqPh);
  lit = (0.62 + 0.38*(1.0-uHSeq) + uHSeq*(0.75*x*x*x*x + 0.25*uHigh*step(0.93, hash21(m+floor(uHSeqPh*4.0))))) * dip;
  if (role < 0.5) { lit = 0.0; return vec3(0.0); }
  if (role > 8.5) return vec3(1.0, 0.97, 0.92);         // white: the key / eye inside an ombré frame
  if (role > 6.5) return uHC[int(role-1.0)];            // the two key colours of the colourway
  if (uHOmb > 0.0) {                                    // ombré cycling: the ramp slides ring by ring, cross-faded
    float k = mod(role-1.0+floor(uHOmb), 6.0), fq = fract(uHOmb);
    return mix(uHC[int(k)], uHC[int(mod(k+1.0, 6.0))], fq*fq*(3.0-2.0*fq));
  }
  float cw = abs(fract(t.g + role*0.17 - uChasePh + 0.5) - 0.5)*2.0;
  lit *= mix(1.0, 0.10 + 0.90*(1.0 - smoothstep(0.5*uChaseW, uChaseW, cw)), uChase);
  return uHC[int(mod(role-1.0, 6.0))];
}

vec4 hunikuin(vec2 P, float px){
  vec2 L; float fw, mask = 1.0;
  if (uLay == 3 || uLay == 5) {                         // rosette / tunnel: a conformal bead lattice (round beads everywhere)
    float r = max(length(P), 1e-4);
    L = vec2((atan(P.y, P.x) + uRot)/TAU*uHNB, log(r)*uHNB/TAU + uHPolar); fw = px*uHNB/(TAU*r);
    L.x = mod(L.x, uHNB);
    if (uLay == 3) mask = 1.0 - smoothstep(0.455, 0.46, r);
  } else {
    L = P*uHB + uHScroll; fw = px*uHB;
    if (uHMed > 0.5 && uHMed < 1.5) L.x = P.x*uHB*1.4 + uHScroll.x;   // woven floats are taller than wide (1 : 1.4)
    if (uHMed > 2.5) L.x += 0.5*mod(floor(L.y), 2.0);   // staggered rows, as beads sit in the stitch
    if (uLay == 2) { L.y = P.y*uHB; mask = step(abs(floor(L.y)+0.5), floor(uHB*0.22)+0.6); }
  }
  vec2 b = floor(L), c = fract(L)-0.5;
  float lit; vec3 col = beadCol(b, lit);
  if (uLay == 2) {                                      // band edges: two solid rows, like the selvedge of a headband
    float row = abs(floor(L.y)+0.5), edge = floor(uHB*0.22)+0.6;
    if (row > edge-2.0 && uHEdge > 0.5) { col = uHC[row > edge-1.0 ? 0 : 1]; lit = 0.8; }
  }
  float fade = 1.0 - smoothstep(0.30, 0.60, fw), a, shade = 1.0;
  if (uHMed < 0.5 || uHMed > 2.5) {                     // glass beads: round, a gap between them, a small highlight
    float r = length(c);
    a = 1.0 - smoothstep(0.40-fw, 0.40+fw, r);
    shade = 0.70 + 0.30*(1.0-smoothstep(0.10, 0.42, r)) + 0.45*(1.0-smoothstep(0.0, 0.20, length(c-vec2(-0.11, 0.13))));
    float halo = uHGlow*0.22*(1.0-smoothstep(0.40, 0.47, r))*(1.0-a);   // the gaps between beads stay black
    a = mix(a, 0.52, smoothstep(0.25, 0.5, fw));          // beads smaller than ~3 px: their average, no moire
    vec3 o = col*lit*(a*shade + halo);
    return vec4(o, max(a, halo)*step(0.001, lit)) * mask * fade;
  }
  if (uHMed < 1.5) {                                    // woven: a clean grid of floats with a dark gap between them
    vec2 q = abs(c); float fx = fw*1.4;
    a = (1.0 - smoothstep(0.44-fx, 0.44+fx, q.x))*(1.0 - smoothstep(0.46-fw, 0.46+fw, q.y));
    shade = 0.86 + 0.14*cos(c.x*TAU)*(1.0-smoothstep(0.08, 0.2, fw));
    a = mix(a, 0.8, smoothstep(0.25, 0.5, fw));
    return vec4(col*lit*a*shade, a*step(0.001, lit)) * mask * fade;
  }
  // painted: flat stepped shapes; the edge to the neighbouring cell is antialiased
  float lx, ly; vec3 o = col*lit;
  vec3 cx = beadCol(b + vec2(sign(c.x), 0.0), lx)*lx, cy = beadCol(b + vec2(0.0, sign(c.y)), ly)*ly;
  o = mix(o, cx, 0.5*smoothstep(0.5-fw, 0.5, abs(c.x)));
  o = mix(o, cy, 0.5*smoothstep(0.5-fw, 0.5, abs(c.y)));
  o *= 0.86;
  return vec4(o, smoothstep(0.0, 0.2, max3(o))) * mask * fade;
}

void main(){
  vec2 P = (v_uv-0.5)*vec2(uRes.x/uRes.y, 1.0); float px = 1.0/uRes.y;
  vec4 S = vec4(0.0), H = vec4(0.0);
  if (uAmtH > 0.0) H = hunikuin(P, px)*uAmtH;
  if (uAmtS > 0.0) S = shipiboLayout(P, px)*uAmtS;
  vec3 col = H.rgb*(1.0-S.a) + S.rgb;
  col = max(col, 0.0); col /= max(1.0, max3(col));       // hue-preserving: never white-out
  f = vec4(col, clamp(S.a + H.a*(1.0-S.a), 0.0, 1.0));
}
"""


SWAP_S, FADE_S = 0.5, 0.15                             # at most one design swap per SWAP_S (wall), faded out and in
PAN_WRAP = POLAR_WRAP = 8.0                             # panels: variant (mod 4 / mod 2) and pen rank (mod 8) repeat at 8 cells


class DesignBuilder:
    """Builds tile maps on a daemon thread. request(key) queues (newest first), get(key) -> array | None, never blocks.
    key = (seed, gx, gy, variants, wrap_x, form). The cache keeps the 48 most recent maps (about 20-60 kB each)."""
    KEEP = 48

    def __init__(self):
        import threading
        self._lock, self._wake = threading.Lock(), threading.Event()
        self._cache, self._todo, self._stop, self._busy = {}, [], False, False
        self.built, self.build_threads = 0, set()
        self._thread = threading.Thread(target=self._run, name="kene-designs", daemon=True)
        self._thread.start()

    @staticmethod
    def make(key):
        seed, gx, gy, nv, wrap, form = key
        return np.concatenate([shipibo_design(seed * 7 + v, gx, gy, wrap, form) for v in range(nv)], axis=0)

    def _store(self, key, data):
        import threading
        with self._lock:
            self._cache.pop(key, None)
            self._cache[key] = data
            while len(self._cache) > self.KEEP:
                self._cache.pop(next(iter(self._cache)))
            self.built += 1
            self.build_threads.add(threading.get_ident())

    def get(self, key):
        with self._lock:
            return self._cache.get(key)

    def request(self, key):
        with self._lock:
            if key in self._cache or (self._todo and self._todo[-1] == key):
                return
            if key in self._todo:
                self._todo.remove(key)
            self._todo.append(key)                       # the newest request is built first
            del self._todo[:-4]                          # a knob sweep leaves at most four behind
        self._wake.set()

    def build_now(self, key):
        if self.get(key) is None:
            self._store(key, self.make(key))

    def pending(self):
        with self._lock:
            return len(self._todo) + (1 if self._busy else 0)

    def _run(self):
        while not self._stop:
            self._wake.wait(0.5)
            while not self._stop:
                with self._lock:
                    if not self._todo:
                        self._wake.clear()
                        break
                    key = self._todo.pop()
                    self._busy = True
                try:
                    if self.get(key) is None:
                        self._store(key, self.make(key))
                except Exception as ex:                  # a failed build must never take the thread (or the show) down
                    print(f"[kene] design build failed: {ex}")
                finally:
                    self._busy = False

    def close(self):
        self._stop = True
        self._wake.set()


def _blend_query():
    try:
        from beings import _blend_query as q
        return q()
    except Exception:
        return None


class KenePass:
    """render(values, clock, t, dt, bands, palette_rgb) -> RGBA8 texture (see the module docstring)."""

    IN_UNITS = (20, 21)                                  # private texture units for the two tile maps

    def __init__(self, ctx, out_size=(1280, 800)):
        self.ctx = ctx
        self.out_size = (max(16, int(out_size[0])), max(16, int(out_size[1])))
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self._vao = ctx.vertex_array(self._prog, [(self._quad, "2f", "in_pos")])
        self._tex = ctx.texture(self.out_size, 4)
        self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._tex.repeat_x = self._tex.repeat_y = False
        self._fbo = ctx.framebuffer(color_attachments=[self._tex])
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._blank = True
        self.draws = 0
        self._dtex, self._dkey = None, None
        self.builder = DesignBuilder()                   # designs are built on its thread, never on the render thread
        self.wall_dt, self._wclock = None, 0.0
        self.sync = False                                # True (stand-alone stills, tests): build in place, no fades
        self._mix, self._swap_t, self._wall, self.swaps = 0.0, -1e9, None, 0
        self._htex = None
        self._grid = (1, 1, 1)
        self.pan = [0.0, 0.0]                            # panels scroll (cells, wrapped at 8)
        self.band_scroll = 0.0                           # fraction of the band period
        self.polar = 0.0                                 # ring scroll (wrapped at POLAR_WRAP)
        self.hk_scroll = [0.0, 0.0]                      # beads (wrapped at HK_T)
        self.hk_polar = 0.0
        self.rot = 0.0
        self._hk_now = self._hk_next = self._hk_want = None
        self._hk_morph = 0.0
        self._last_t = None
        self._blend_on = _blend_query()

    # ---------------------------------------------------------------- tile textures
    def _key(self, seed, layout, scale, form=1, dense=False):
        if form == F_SPINE and layout != 0:
            form = 1
        gx, gy, nv, wrap = grid_dims(layout, scale, self.out_size[0] / self.out_size[1], dense, form)
        return (int(seed), gx, gy, nv, wrap, int(form))

    def _upload(self, key):
        data = self.builder.get(key)
        size = (data.shape[1], data.shape[0])
        if self._dtex is None or self._dtex.size != size:
            if self._dtex is not None:
                self._dtex.release()
            self._dtex = self.ctx.texture(size, 4, alignment=1)
            self._dtex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._dtex.write(np.ascontiguousarray(data).tobytes())
        self._dkey, self._grid = key, (key[1] * F, key[2] * F, key[3])

    def _design(self, key, wdt, also=()):
        """Show design `key`. Designs are built on the worker thread (never here, unless self.sync); the one on screen
        stays until the new one is ready, then the layer fades out, swaps and fades in (FADE_S each way), at most one
        swap per SWAP_S of WALL time: a sweeping seed knob can never flicker the field, and freeze / speed 0 do not
        stall it. `also` = keys to build ahead (the pen's next design)."""
        now = self._wclock                               # wall time this pass has run (not show time)
        if self.sync:
            self.builder.build_now(key)
            if key != self._dkey:
                self._upload(key)
            self._mix = 1.0
            return
        for k in tuple(also) + (key,):                   # the wanted key is requested last = built first
            self.builder.request(k)
        ready = self.builder.get(key) is not None
        if self._dkey is None:                           # nothing on screen yet: wait for the first design, then fade in
            self._mix = 0.0
            if ready:
                self._upload(key)
                self._swap_t = now
            return
        if key != self._dkey and ready and now - self._swap_t >= SWAP_S:
            self._mix = max(0.0, self._mix - wdt / FADE_S)
            if self._mix <= 0.0:
                self._upload(key)
                self._swap_t = now
                self.swaps += 1
        else:
            self._mix = min(1.0, self._mix + wdt / FADE_S)

    def _beads(self):
        if self._htex is None:
            self._htex = self.ctx.texture((HK_T * NF * 2, HK_T), 2, alignment=1)
            self._htex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._htex.write(np.ascontiguousarray(hk_atlas_all()).tobytes())

    def release(self):
        self.builder.close()
        for o in (self._fbo, self._tex, self._dtex, self._htex, self._vao, self._prog, self._quad):
            try:
                if o is not None:
                    o.release()
            except Exception:
                pass
        self._dtex = self._htex = None

    close = release

    def _set(self, name, value):
        if name in self._prog:
            self._prog[name].value = value

    # ---------------------------------------------------------------- motion (float64, wrapped where it repeats)
    def _advance(self, dt, wdt, values, layout, clock):
        """dt = the engine's show dt (speed and freeze are already in it: never multiply by `speed` again);
        wdt = wall-clock dt, for things that must keep their own time (the bead morph, design fades)."""
        flow, trav = _val(values, "flow", 0.0, -1.0, 1.0), _val(values, "travel", 0.0, -1.0, 1.0)
        ang = _val(values, "flow_dir", 0.0, 0.0, 1.0) * TAU
        v = dt * (0.10 * flow + 0.30 * trav)             # frame heights per second
        self.pan[0] = (self.pan[0] + v * math.cos(ang) / 0.5) % PAN_WRAP
        self.pan[1] = (self.pan[1] + v * math.sin(ang) / 0.5) % PAN_WRAP
        gx, gy = self._grid[0], self._grid[1]
        self.band_scroll = (self.band_scroll + v / (0.5 / max(gy, 1) * max(gx, 1))) % 1.0
        folds = max(2.0, round(_val(values, "folds", 8.0, 2.0, 24.0)))
        self.polar = (self.polar - dt * (0.45 * trav + 0.05 * flow) * folds / TAU * 2.0) % POLAR_WRAP
        hb = self._hb(values)
        self.hk_scroll[0] = (self.hk_scroll[0] + v * math.cos(ang) * hb) % HK_T
        self.hk_scroll[1] = (self.hk_scroll[1] + v * math.sin(ang) * hb) % HK_T
        self.hk_polar = (self.hk_polar - dt * (0.45 * trav + 0.05 * flow) * self._hnb(values) / TAU * 2.0) % HK_T
        if "rot_phase" in values:
            self.rot = _num(values["rot_phase"]) % TAU
        else:
            self.rot = (self.rot + dt * (0.3 * _val(values, "rotate", 0.0, -1.0, 1.0) + 1.6 * _val(values, "spin", 0.0, -1.0, 1.0))) % TAU
        # Huni Kuin motif: a change (fader, scene, auto cycle, tile variant) morphs bead by bead over two beats of
        # WALL time. A change that arrives during a morph waits for it to finish (newest wins): no pops on a sweep.
        fam = _idx(values, "hk_motif", NF)
        bars = HK_CYCLE_BARS[_idx(values, "hk_cycle", len(HK_CYCLE))]
        beat = _num((clock or {}).get("beat", 0.0))
        if bars:
            fam = (fam + int(max(beat, 0.0) / (4.0 * bars))) % NF
        fam += NF * (_idx(values, "kene_seed", N_SEEDS) % 2)      # the tile variant follows the Design knob only
        self._hk_want = fam
        if self._hk_now is None or self.sync:
            self._hk_now = self._hk_next = fam
            self._hk_morph = 0.0
        if self._hk_morph <= 0.0 and self._hk_want != self._hk_now:
            self._hk_next, self._hk_morph = self._hk_want, 1e-6
        if self._hk_morph > 0.0:
            bpm = _num((clock or {}).get("bpm", 124.0), 124.0)
            self._hk_morph += wdt * min(max(bpm, 60.0), 200.0) / 60.0 / 2.0
            if self._hk_morph >= 1.0:
                self._hk_now, self._hk_morph = self._hk_next, 0.0

    @staticmethod
    def _hb(values):
        return float(2 * int(10 + 26 * _val(values, "hk_size", 0.5, 0.0, 1.0)))          # beads per frame height: 20 .. 72

    def _hnb(self, values):
        return float(HK_T * max(2, int(round(self._hb(values) * 2.2 / HK_T))))            # beads round the rosette

    # ---------------------------------------------------------------- draw
    def render(self, values, clock=None, t=0.0, dt=None, bands=(0.0, 0.0, 0.0), palette_rgb=None):
        ctx = self.ctx
        amt_s = _val(values, "l_kene_shipibo", 0.0, 0.0, 1.0)
        amt_h = _val(values, "l_kene_hunikuin", 0.0, 0.0, 1.0)
        t = _num(t)
        if dt is None:
            dt = 0.0 if self._last_t is None else t - self._last_t
        dt = min(max(_num(dt), 0.0), 0.25)
        if amt_s <= 0.001 and amt_h <= 0.001:            # off: nothing moves, nothing is drawn
            if not self._blank:
                self._fbo.clear(0.0, 0.0, 0.0, 0.0)
                self._blank = True
            self._last_t = self._wall = None
            return self._tex
        bands = [min(max(_num(b), 0.0), 1.0) for b in (list(bands if bands is not None else ()) + [0, 0, 0])[:3]]
        pal = None
        if palette_rgb is not None:
            try:
                p = np.asarray(palette_rgb, dtype=float).reshape(-1, 3)
                pal = np.clip(p, 0.0, 1.0) if len(p) and np.isfinite(p).all() else None
            except (TypeError, ValueError):
                pal = None
        clock = clock or {}
        beat = _num(clock.get("beat", t * 124.0 / 60.0))
        lay = _idx(values, "kene_layout", len(LAYOUTS))
        if lay == 4:                                     # follow the show: mandala / tunnel when symmetry is up, else panels
            lay = 5 if _val(values, "symmetry", 1.0, 0.0, 1.0) >= 0.5 else 1
        pen = pen_state(values, beat)
        now = time.perf_counter()
        wdt = 0.0 if self._wall is None else min(max(now - self._wall, 0.0), 0.1)
        if self.wall_dt is not None:                     # tests: a wall clock they can step
            wdt = self.wall_dt
        self._wall = now
        self._wclock += wdt
        if amt_s > 0.001:                                # the bead layer alone never builds a design
            sc = _val(values, "kene_scale", 0.5, 0.0, 1.0)
            stq = style_uniforms(values, None)
            form, dense = stq["form"] or _idx(values, "kene_form", len(FORMS), 1), stq["dense"]
            self._design(self._key(pen["seed"], lay, sc, form, dense), wdt,
                         also=(self._key(pen["next_seed"], lay, sc, form, dense),))
        self._beads()
        self._advance(dt, wdt, values, lay, clock)
        self._last_t = t
        if self._dtex is None:                           # first design still being built: beads only for now
            amt_s = 0.0
            if amt_h <= 0.001:
                return self._tex
        st = style_uniforms(values, pal)
        hk = hk_colours(values, pal)
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        blend_on = self._blend_on() if self._blend_on is not None else False
        try:
            self._fbo.use()
            ctx.viewport = (0, 0, *self.out_size)
            ctx.disable(moderngl.BLEND)
            if self._dtex is not None:
                self._dtex.use(self.IN_UNITS[0])
            self._htex.use(self.IN_UNITS[1])
            s = self._set
            s("uD", self.IN_UNITS[0]); s("uH", self.IN_UNITS[1])
            s("uRes", tuple(float(x) for x in self.out_size))
            s("uGrid", tuple(float(x) for x in self._grid))
            s("uLay", int(lay)); s("uWrap", 1.0 if lay == 2 else 0.0)
            s("uAmtS", amt_s * pen["fade"] * self._mix); s("uAmtH", amt_h)
            s("uFolds", float(max(2.0, round(_val(values, "folds", 8.0, 2.0, 24.0)))))
            s("uRot", float(self.rot))
            s("uScroll", (float(self.pan[0]), float(self.pan[1])) if lay != 2 else (float(self.band_scroll), 0.0))
            s("uPolar", float(self.polar))
            if "uCol" in self._prog:
                self._prog["uCol"].write(np.asarray(st["cols"], dtype="f4").tobytes())
            s("uGround", st["ground"]); s("uGroundOn", st["ground_on"]); s("uTier", st["tier"]); s("uW", st["w"])
            s("uStitch", st["stitch"]); s("uBead", st["bead"])
            s("uArm", st["arm"]); s("uEcho2", st["echo2"]); s("uBorder", st["border"] if lay == 0 else 0.0)
            s("uPH", 0.74 if st["border"] else (0.82 if st["frame"] else 0.90))
            s("uArmW", float(st["arm_w"] or st["w"][0])); s("uAccOn", st["acc_on"]); s("uTwig", st["twig"])
            s("uWob", st["wob"] if lay in (0, 3) else 0.0)
            s("uFrameOn", 1.0 if (st["frame"] and lay == 0) else 0.0); s("uFrame", tuple(st["frame"] or (0.0, 0.0, 0.0)))
            s("uHOmb", (max(beat, 0.0) / 8.0 * _val(values, "hk_ombre", 0.0, 0.0, 1.0)) % 6.0)
            s("uHair", st["hair"]); s("uKnob", st["knob"]); s("uCBorder", st["cborder"])
            s("uHEdge", 0.0 if int(self._hk_now or 0) % NF == 5 else 1.0)
            if "uAcc" in self._prog:
                self._prog["uAcc"].write(np.asarray(st["acc"], dtype="f4").tobytes())
            curve = _val(values, "kene_curve", 0.25, 0.0, 1.0)
            s("uRound", min(1.0, curve * 1.6)); s("uCurve", max(0.0, curve - 0.3) / 0.7)
            s("uFill", _val(values, "kene_fill", 0.85, 0.0, 1.0))
            s("uDraw", float(pen["progress"]))
            s("uFlow", _val(values, "kene_flow", 0.35, 0.0, 1.0)); s("uFlowPh", (beat / 4.0) % 1.0)
            s("uBreath", _val(values, "kene_breathe", 0.3, 0.0, 1.0) * math.sin(TAU * ((t / BREATH_S) % 1.0)))
            s("uGlow", _val(values, "kene_glow", 0.4, 0.0, 1.0))
            chase = values.get("chase_phase")
            if chase is None:
                chase = _num(clock.get("bar_phase", 0.0)) * _val(values, "chase_rate", 1.0, -4.0, 4.0)
            s("uChase", _val(values, "kene_chase", 0.0, 0.0, 1.0) * _val(values, "chase", 0.5, 0.0, 1.0))
            s("uChasePh", _num(chase) % 1.0); s("uChaseW", _val(values, "chase_width", 0.25, 0.05, 0.95))
            s("uBass", bands[0]); s("uHigh", bands[2])
            s("uHB", self._hb(values)); s("uHNB", self._hnb(values)); s("uHT", float(HK_T))
            s("uHMed", float(_idx(values, "hk_medium", len(HK_MEDIA))))
            s("uHSeq", _val(values, "hk_seq", 0.5, 0.0, 1.0))
            step = beat * 2.0                            # the lit run steps along the motif on every half beat
            s("uHSeqPh", ((math.floor(step) + min(1.0, (step % 1.0) * 3.0)) / 16.0) % 1.0)
            s("uHGlow", _val(values, "hk_glow", 0.4, 0.0, 1.0))
            s("uHMorph", float(min(self._hk_morph, 1.0)))
            s("uHFam", (float(self._hk_now), float(self._hk_next)))
            s("uHScroll", (float(self.hk_scroll[0]), float(self.hk_scroll[1]))); s("uHPolar", float(self.hk_polar))
            if "uHC" in self._prog:
                self._prog["uHC"].write(np.asarray(hk, dtype="f4").tobytes())
            self._vao.render(moderngl.TRIANGLE_STRIP)
            self.draws += 1
            self._blank = False
            return self._tex
        finally:
            if blend_on:
                ctx.enable(moderngl.BLEND)
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:
                ctx.screen.use()


# =====================================================================================================================
# stand-alone helpers: gallery, reference pairs, bench
# =====================================================================================================================
GALLERY_PALETTE = [(1.0, 0.25, 0.7), (0.15, 0.9, 0.85), (1.0, 0.8, 0.15), (0.55, 0.25, 1.0), (0.2, 1.0, 0.35)]


def hidden_context(size=(64, 64)):
    from math_layers import hidden_context as hc
    return hc(size)


def default_values(**over):
    import params
    v = params.defaults()
    v.update(flow=0.0, travel=0.0, rotate=0.0)
    v.update(over)
    return v


def make_clock(t, bpm=124.0, beat=None):
    b = t * bpm / 60.0 if beat is None else beat
    return dict(bpm=bpm, beat=b, beat_phase=b % 1.0, bar_phase=(b / 4) % 1.0, phrase_phase=(b / 32) % 1.0)


def render_once(kp, values, t=4.0, dt=1 / 60, bands=(0.5, 0.3, 0.2), palette=None, beat=None):
    kp.sync = True                                      # stills: build the design in place, no fades
    tex = kp.render(values, make_clock(t, beat=beat), t, dt, bands, palette or GALLERY_PALETTE)
    return np.frombuffer(tex.read(), dtype=np.uint8).reshape(tex.height, tex.width, 4)[::-1].copy()


def frame_stats(img):
    rgb = img[..., :3].astype(float)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    lit = mx > 60
    return dict(black=float((mx <= 4).mean()), blown=float((mn >= 250).mean()), mean=float(rgb.mean()),
                sat=float(((mx - mn) / np.maximum(mx, 1))[lit].mean()) if lit.any() else 0.0)


S, H = "l_kene_shipibo", "l_kene_hunikuin"
SHOTS = [
    ("shipibo_light_panel", {S: 1, "kene_seed": 0}),
    ("shipibo_light_seed7", {S: 1, "kene_seed": 7, "kene_colors": 4}),
    ("shipibo_light_curving", {S: 1, "kene_seed": 2, "kene_curve": 0.9, "kene_colors": 3}),
    ("shipibo_cotton_white", {S: 1, "kene_seed": 4, "kene_colors": 1, "kene_curve": 0.0}),
    ("shipibo_palette_follow", {S: 1, "kene_seed": 5, "kene_colors": 0}),
    ("shipibo_mud_cloth", {S: 1, "kene_style": 1, "kene_seed": 1, "kene_curve": 0.5}),
    ("shipibo_embroidery_medallion", {S: 1, "kene_style": 2, "kene_layout": 3, "kene_seed": 3, "kene_curve": 0.5}),
    ("shipibo_embroidery_panels", {S: 1, "kene_style": 2, "kene_layout": 1, "kene_colors": 4, "kene_seed": 6}),
    ("shipibo_ceramic_band", {S: 1, "kene_style": 3, "kene_layout": 2, "kene_seed": 2, "kene_curve": 0.0}),
    ("shipibo_mud_band", {S: 1, "kene_style": 1, "kene_layout": 2, "kene_seed": 9, "kene_curve": 0.0}),
    ("shipibo_beadwork", {S: 1, "kene_style": 4, "kene_seed": 8, "kene_colors": 5, "kene_scale": 0.2}),
    ("shipibo_mandala", {S: 1, "kene_layout": 4, "symmetry": 1.0, "kene_colors": 2}),
    ("shipibo_pen_30", {S: 1, "kene_seed": 0, "kene_draw": 0.3}),
    ("shipibo_pen_60", {S: 1, "kene_seed": 0, "kene_draw": 0.6}),
    ("shipibo_chase", {S: 1, "kene_seed": 0, "kene_chase": 1.0, "chase": 1.0}),
    ("hunikuin_txede_bedu_beads", {H: 1, "hk_motif": 0}),
    ("hunikuin_dunu_kene_beads", {H: 1, "hk_motif": 1, "hk_colors": 2}),
    ("hunikuin_inu_tae_beads", {H: 1, "hk_motif": 2, "hk_colors": 3}),
    ("hunikuin_inu_tae_txede_bedu", {H: 1, "hk_motif": 3, "hk_colors": 4}),
    ("hunikuin_kape_hina_beads", {H: 1, "hk_motif": 4, "hk_colors": 4}),
    ("hunikuin_woven_bw", {H: 1, "hk_motif": 1, "hk_medium": 1, "hk_colors": 0, "hk_size": 0.3}),
    ("hunikuin_painted", {H: 1, "hk_motif": 0, "hk_medium": 2, "hk_colors": 0, "hk_size": 0.2}),
    ("hunikuin_band", {H: 1, "hk_motif": 0, "kene_layout": 2, "hk_colors": 4}),
    ("hunikuin_rosette", {H: 1, "hk_motif": 4, "kene_layout": 3, "hk_colors": 1}),
    ("both_layers", {S: 0.9, H: 0.5, "kene_layout": 3, "hk_colors": 2, "hk_size": 0.8}),
    ("study_A_painted", {S: 1, "kene_style": 5, "kene_seed": 3, "kene_curve": 0.3, "kene_scale": 0.6}),
    ("study_A_painted_b", {S: 1, "kene_style": 5, "kene_seed": 12, "kene_curve": 0.3, "kene_scale": 0.6}),
    ("study_A_inverted_light", {S: 1, "kene_style": 0, "kene_colors": 1, "kene_seed": 3, "kene_curve": 0.5}),
    ("study_B_cloth", {S: 1, "kene_style": 6, "kene_seed": 14, "kene_curve": 0.3, "kene_scale": 0.5}),
    ("study_C_embroidered", {S: 1, "kene_style": 7, "kene_seed": 6, "kene_curve": 0.3, "kene_scale": 0.5}),
    ("study_D_woven_band_key", {H: 1, "hk_motif": 5, "hk_medium": 1, "hk_colors": 9, "kene_layout": 2, "hk_size": 0.33}),
    ("study_D_woven_bands_stacked", {H: 1, "hk_motif": 5, "hk_medium": 1, "hk_colors": 9, "hk_size": 0.45}),
    ("study_E_bead_tiles", {H: 1, "hk_motif": 6, "hk_medium": 0, "hk_colors": 6, "hk_size": 0.6}),
    ("study_E_bead_tiles_green", {H: 1, "hk_motif": 6, "hk_medium": 0, "hk_colors": 8, "hk_size": 0.35, "kene_seed": 1}),
    ("study_F_bead_straps", {H: 1, "hk_motif": 7, "hk_medium": 0, "hk_colors": 6, "hk_size": 0.5}),
    ("study_F_bead_straps_green", {H: 1, "hk_motif": 7, "hk_medium": 3, "hk_colors": 8, "hk_size": 0.3, "kene_seed": 1}),
    ("study_G_key_fret", {H: 1, "hk_motif": 8, "hk_medium": 0, "hk_colors": 7, "hk_size": 0.8}),
]
# PRIVATE study references (research/kene_reference/user_refs: copyrighted, never published) next to the studies
USER_PAIRS = [
    ("shipiboKeneSaraFlores.jpeg", (0.0, 0.0, 1.0, 1.0), "study_A_painted"),
    ("ShipiboKene.png", (0.05, 0.05, 0.95, 0.95), "study_B_cloth"),
    ("YawanawaRug.png", (0.0, 0.25, 1.0, 0.85), "study_C_embroidered"),
    ("domiruthperutravel-itinerary-Huni-Kuin-1.jpg", (0.0, 0.0, 1.0, 1.0), "study_D_woven_bands_stacked"),
    ("HuniKuinBracelets.jpg", (0.05, 0.2, 0.95, 0.8), "study_E_bead_tiles"),
    ("YawanawaBreadwork.jpg", (0.0, 0.0, 1.0, 1.0), "study_F_bead_straps"),
    ("CatsClawBreastplatebust2.webp", (0.1, 0.25, 0.9, 0.65), "study_G_key_fret"),
]
# reference photo (research/kene_reference/images), crop box as fractions (l, t, r, b), the generated shot it sits next to
PAIRS = [
    ("wikimedia_shipibo_art_1.jpg", (0.10, 0.14, 0.88, 0.86), "shipibo_mud_cloth"),
    ("wikimedia_shipibo_art_4.jpg", (0.25, 0.08, 0.75, 0.92), "shipibo_embroidery_medallion"),
    ("wikimedia_olinda_silvano_textile.jpg", (0.52, 0.28, 1.0, 1.0), "shipibo_embroidery_panels"),
    ("artic_85528.jpg", (0.28, 0.05, 0.72, 0.60), "shipibo_mud_band"),
    ("wikimedia_masato_chomo_liverpool.jpg", (0.04, 0.12, 0.96, 0.70), "shipibo_ceramic_band"),
    ("wikimedia_vestimenta_mujer.jpg", (0.54, 0.19, 0.88, 0.31), "shipibo_beadwork"),
    ("wikimedia_shipibo_art_2.jpg", (0.0, 0.0, 1.0, 1.0), "shipibo_light_curving"),
    ("wikimedia_hunikuin_katxanawa.jpg", (0.30, 0.16, 0.70, 0.30), "hunikuin_woven_bw"),
    ("wikimedia_mahku_venice.jpg", (0.38, 0.288, 0.72, 0.342), "hunikuin_band"),
    ("wikimedia_mahku_venice.jpg", (0.32, 0.36, 0.62, 0.50), "hunikuin_txede_bedu_beads"),
    ("smvk_vkm_14630.jpg", (0.05, 0.10, 0.95, 0.90), "hunikuin_painted"),
    ("wikimedia_amnh_cashinahua_belt.jpg", (0.35, 0.74, 1.0, 1.0), "hunikuin_dunu_kene_beads"),
]


def gallery(out_dir=None, out_size=(1280, 800), only=None):
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir or HERE / "test_frames" / "kene_real")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx, _win = hidden_context()
    shots = [s for s in SHOTS if not only or s[0] in only]
    tw, th, cols = 320, 200, 5
    sheet = Image.new("RGB", (tw * cols, (th + 16) * math.ceil(len(shots) / cols)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for i, (tag, over) in enumerate(shots):
        kp = KenePass(ctx, out_size)
        try:
            img = render_once(kp, default_values(**over), t=5.6, beat=11.4)
        finally:
            kp.release()
        st = frame_stats(img)
        print(f"[kene] {tag:32s} black={st['black']:5.1%} blown={st['blown']:5.2%} mean={st['mean']:5.1f} sat={st['sat']:4.0%}")
        im = Image.fromarray(img[..., :3])
        im.save(out_dir / f"{tag}.png")
        x, y = (i % cols) * tw, (i // cols) * (th + 16)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        dr.text((x + 4, y + th + 2), tag, fill=(200, 200, 200))
    if not only:
        sheet.save(out_dir / "contact_sheet.png")
        pairs_sheet(out_dir)
        user = HERE / "research" / "kene_reference" / "user_refs"
        if user.is_dir():
            pairs_sheet(out_dir, USER_PAIRS, user, "user_ref_pairs.png",
                        "INTERNAL ONLY, NEVER PUBLISH. LEFT: private study reference (copyrighted)   RIGHT: generative study (not a copy)")
    print(f"[kene] {len(shots)} frames -> {out_dir}")
    return out_dir


def pairs_sheet(out_dir, pairs=None, ref_dir=None, name="reference_pairs.png",
                note="LEFT: study photograph (research/kene_reference/images, licences in PROVENANCE.csv)   "
                     "RIGHT: generative study (not kené; not a copy)"):
    """INTERNAL study sheets (never published): a study photo (left) next to the generated study (right)."""
    from PIL import Image, ImageDraw
    PAIRS = pairs or globals()["PAIRS"]
    ref_dir = Path(ref_dir or HERE / "research" / "kene_reference" / "images")
    cw, ch, pad = 560, 350, 22
    sheet = Image.new("RGB", (2 * (2 * cw + 12) + 12, 30 + (ch + pad) * math.ceil(len(PAIRS) / 2)), (14, 14, 14))
    dr = ImageDraw.Draw(sheet)
    dr.text((8, 8), note, fill=(230, 230, 230))
    for n, (ref, box, tag) in enumerate(PAIRS):
        x0, y0 = 12 + (n % 2) * (2 * cw + 12), 30 + (n // 2) * (ch + pad)
        try:
            im = Image.open(ref_dir / ref).convert("RGB")
            w, h = im.size
            im = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
            im.thumbnail((cw - 4, ch), Image.LANCZOS)
            sheet.paste(im, (x0 + (cw - 4 - im.width) // 2, y0 + (ch - im.height) // 2))
        except OSError as ex:
            dr.text((x0 + 8, y0 + 8), f"missing: {ref} ({ex})", fill=(255, 120, 120))
        try:
            gen = Image.open(Path(out_dir) / f"{tag}.png").convert("RGB").resize((cw - 4, ch), Image.LANCZOS)
            sheet.paste(gen, (x0 + cw, y0))
        except OSError:
            pass
        dr.text((x0 + 2, y0 + ch + 3), f"{ref}   |   {tag}", fill=(190, 190, 190))
    sheet.save(Path(out_dir) / name)


def bench(out_size=(1280, 800), frames=120):
    ctx, _win = hidden_context()
    rows = []
    cases = [("shipibo default", {S: 1}), ("shipibo band", {S: 1, "kene_layout": 2}),
             ("shipibo mandala", {S: 1, "kene_layout": 4, "symmetry": 1.0}), ("shipibo beadwork", {S: 1, "kene_style": 4}),
             ("hunikuin beads", {H: 1}), ("hunikuin painted", {H: 1, "hk_medium": 2}), ("both", {S: 1, H: 1})]
    for tag, over in cases:
        kp = KenePass(ctx, out_size)
        kp.sync = True
        try:
            vals = default_values(**over)
            for i in range(3):
                kp.render(vals, make_clock(5.0), 5.0, 1 / 60, (0.5, 0.3, 0.2), GALLERY_PALETTE)
            ctx.finish()
            t0 = time.perf_counter()
            for i in range(frames):
                kp.render(vals, make_clock(5.0 + i / 60), 5.0 + i / 60, 1 / 60, (0.5, 0.3, 0.2), GALLERY_PALETTE)
            ctx.finish()
            ms = (time.perf_counter() - t0) / frames * 1000.0
        finally:
            kp.release()
        rows.append((tag, ms))
        print(f"[kene] {tag:18s} {ms:6.2f} ms/frame")
    return rows


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        gallery(only=[a for a in sys.argv[1:] if not a.startswith("--")] or None)
