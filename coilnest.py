"""COIL NEST: a frame-filling nest of 6-14 thick serpent tubes (the "Serpents" grammar: no room, no sky, the bodies
ARE the architecture, pitch-black crevices between them), skinned along the body and gated by a beat-synced programme.

Inspired by the visual grammar described in research/critique/video_7pGV8fTCOGQ.md (TAS Visuals, "Serpents", 2016).
Nothing of that piece is copied: geometry, patterns, colours and timing are our own. Some ideas started from the
reference shaders in research/critique/video_code/ (reference shader drafted with Gemini (Google) for this project);
the method here is different (analytic 2.5D tube impostors, no raymarch).

    CoilNestPass(ctx, out_size).render(values, clock, t, dt, bands, palette_rgb, layer=None, skins=None) -> RGBA8 tex
    rgb = light with l_coilnest applied, a = how much of what is underneath it hides (coil_floor 1 = the whole frame).

HOW (cost is the user's first priority; budget 3 ms at 1280x800, default 10 bodies, GTX 1650 Ti)
  * CPU (`Nest`, float64): every body is a smooth curve (closed coil, open body, tapering tail or a stub rising to a
    medallion cap / head) sampled at 24 knots, projected with a breathing dolly camera, written as a 24 x 28 RGBA32F
    knot texture (x, y, screen radius, height | tangent x, y, u along the body, world radius): about 11 KB per frame.
  * pass 1, quarter resolution: every coarse texel scans all bodies' knot polylines and keeps the 4 best candidates
    (covering bodies first, highest first) with the knot parameter of the closest point.
  * pass 2, full resolution: per pixel 2 Gauss-Newton steps on the cubic Hermite curve of each candidate from that
    parameter (exact, smooth closest point), the highest covering body wins (exact over / under weave), the others
    give contact shadow. Cylindrical normal, u along the body, v round the girth (asin: the kené grid foreshortens to
    the silhouette), limb darkening, crevice AO, depth darkening, rim sweep. No marching, no loops over all bodies.
  * the skin is FOUR semantic channels (primary lines, fine meander fill, cloth ground, jewel accents), from the
    procedural kené here, or from a 4-channel layer texture (kené study contract), and the cloth ground can be an
    image (SKINS slot "coilnest:bodies" of materials.py) or any engine layer's colour. The `Programme` gates the four
    channels: linework only -> white engraving -> floating ground patches -> full embroidered cloth -> rim sweep,
    on the bar, smoothed (2-pole low-pass + slew cap: no full change faster than 0.45 s, no flicker from a wiggled
    fader, WCAG 2.3.1-safe) and exposed as params (coil_stage / coil_prog / coil_bars / coil_prog_amt) so timeline
    and clips can drive it.
  * off (l_coilnest 0): nothing is computed, uploaded or drawn.

Long shows: every phase is integrated in float64 and wrapped where the picture repeats exactly (writhe and dolly at
TAU with integer harmonics, skin scroll at a whole number of pattern cells: the cells of a closed coil, 64 otherwise).
Hash: "Hash without Sine" by David Hoskins (MIT, see THIRD_PARTY_NOTICES.md).
"""
import math
import sys

import numpy as np

try:
    import moderngl
except Exception:                     # CPU-only tests
    moderngl = None

TAU = 2.0 * math.pi
NMAX = 14
M = 24                                # knots per body
OPEN_WRAP = 64                        # pattern cells after which an open body's skin repeats exactly
COARSE = 4                            # pass 1 runs at 1/4 of the pass 2 resolution
LAYER_COLOUR, LAYER_KENE4 = 1, 2      # layer skin kinds

PARAM_NAMES = ("l_coilnest", "coil_count", "coil_thick", "coil_writhe", "coil_slither", "coil_dolly", "coil_heads",
               "coil_caps", "coil_skin", "coil_layer", "coil_prog", "coil_bars", "coil_stage", "coil_prog_amt",
               "coil_g_line", "coil_g_fill", "coil_g_ground", "coil_g_accent", "coil_rim", "coil_palette",
               "coil_glow", "coil_floor", "coilnest_res")
STAGES = ["Linework only", "White engraving", "Floating ground patches", "Full embroidered cloth", "Rim sweep"]
OPTIONS = {
    "coil_heads": ["No heads (endless coils)", "Heads"],
    "coil_caps": ["Plain ends", "Medallion caps"],
    "coil_skin": ["Procedural kené", "SKINS slot (image / AI)", "Kené study channels (4-channel layer)",
                  "Engine layer (colour)"],
    "coil_layer": ["Kené study", "Bead study", "Folding Blocks", "Beadwork", "Sacred lines", "Beings", "Math stack",
                   "Math fractals", "Text", "Crowd camera", "Video loops", "Filtered output"],
    "coil_prog": ["Manual stage (timeline / clips)", "Gating programme on the bar"],
    "coil_bars": ["2 bars a stage", "4 bars a stage", "8 bars a stage", "16 bars a stage"],
    "coil_stage": list(STAGES),
}
# coil_layer order -> materials.LayerHub source ids ("lines4" = the 4-channel line layer is coil_skin 2)
HUB_IDS = ["kene", "beadstudy", "blocks", "beads", "lines", "beings", "stack", "fractal", "text", "camera", "video",
           "output"]
BARS = (2, 4, 8, 16)

# ------------------------------------------------------------------ gating programme
G_LINE, G_FILL, G_GROUND, G_ACCENT, G_RIM, G_MONO, G_PATCH, G_REVEAL = range(8)
#                 line  fill  ground accent rim  mono  patch reveal
GAINS = np.array([[1.00, 0.12, 0.00, 0.45, 0.00, 0.00, 0.00, 1.0],     # linework only (neon outlines on velvet black)
                  [0.75, 1.00, 0.00, 0.20, 0.00, 0.85, 0.00, 0.0],     # white engraving (cells light one by one)
                  [0.28, 0.10, 1.00, 0.60, 0.00, 0.00, 1.00, 1.0],     # floating patches of cloth ground
                  [1.00, 0.85, 1.00, 1.00, 0.15, 0.00, 0.00, 1.0],     # full embroidered cloth
                  [0.10, 0.04, 0.00, 0.45, 1.00, 0.35, 0.00, 1.0]])    # rim-light sweep in the dark
FULL = GAINS[3]
XFADE = 0.3                           # the last 30 % of a stage cross-fades into the next (ends on the downbeat)
LUM_W = np.array([0.30, 0.25, 0.45, 0.08, 0.12])   # rough share of each channel in the picture's light (tests)
TAU_LP = 0.12                         # two one-pole low-passes in a row (a fader wiggled at 5+ Hz is flattened)
SLEW = 1.0 / 0.45                     # ... then a rate cap: a full 0 -> 1 change takes at least 0.45 s
MAX_LUM_RATE = SLEW * 1.5 * float(LUM_W.sum())     # gains can sit above 1 (manual x1.5 multipliers)
BASS_LIFT = 0.15                      # bass brightens the primary lines by at most 15 %


def _num(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _val(values, name, default, lo, hi):
    return min(max(_num((values or {}).get(name, default), default), lo), hi)


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class Programme:
    """Stage position -> 8 gains (see GAINS). Auto: (bars / bars-per-stage) mod 5 from the beat counter (the same
    beat always gives the same stage: deterministic, survives restarts). Manual: coil_stage. Smoothed + slew capped."""

    def __init__(self):
        self.g = FULL.copy()
        self._lp = FULL.copy()
        self.x = 3.0
        self.lift = 0.0
        self._b1 = self._b2 = 0.0

    @staticmethod
    def target(values, clock):
        auto = _val(values, "coil_prog", 1.0, 0.0, 1.0) >= 0.5
        if auto:
            beat = _num((clock or {}).get("beat", 0.0))
            bars = BARS[int(round(_val(values, "coil_bars", 1.0, 0.0, 3.0)))]
            x = (beat / 4.0 / bars) % 5.0
        else:
            x = float(int(round(_val(values, "coil_stage", 3.0, 0.0, 4.0))))
        i, f = int(x) % 5, x - int(x)
        a = GAINS[i].copy()
        b = GAINS[(i + 1) % 5].copy()
        if i == 1:                                    # white engraving: the cells light one by one, first half
            a[G_REVEAL] = _smooth(f / 0.5) if auto else 1.0
        w = _smooth((f - (1.0 - XFADE)) / XFADE) if auto else 0.0
        g = a + (b - a) * w
        amt = _val(values, "coil_prog_amt", 1.0, 0.0, 1.0)
        g = FULL + (g - FULL) * amt
        for k, n in ((G_LINE, "coil_g_line"), (G_FILL, "coil_g_fill"), (G_GROUND, "coil_g_ground"),
                     (G_ACCENT, "coil_g_accent")):
            g[k] *= _val(values, n, 1.0, 0.0, 1.5)
        g[G_RIM] = g[G_RIM] * _val(values, "coil_rim", 0.6, 0.0, 1.0) / 0.6 if g[G_RIM] > 0 else 0.0
        g[G_RIM] = min(g[G_RIM], 1.5)
        return x, g

    def step(self, dt, values, clock, bass=0.0, snap=False):
        dt = min(max(_num(dt), 0.0), 1.0)
        self.x, tgt = self.target(values, clock)
        if snap:
            self.g[:] = self._lp[:] = tgt
        elif dt > 0:
            k = 1.0 - math.exp(-dt / TAU_LP)
            self._lp += (tgt - self._lp) * k
            want = self.g + (self._lp - self.g) * k
            cap = SLEW * dt
            self.g += np.clip(want - self.g, -cap, cap)
        bass = min(max(_num(bass), 0.0), 1.0)
        if dt > 0:                                    # bass lift: attack 80 ms / release 350 ms, then a second pole
            k1 = 1.0 - math.exp(-dt / (0.08 if bass > self._b1 else 0.35))
            self._b1 += (bass - self._b1) * k1
            self._b2 += (self._b1 - self._b2) * (1.0 - math.exp(-dt / 0.12))
        self.lift = BASS_LIFT * min(max(self._b2, 0.0), 1.0)
        return self.g


# ------------------------------------------------------------------ geometry
# kind: 0 open (both ends off frame), 1 closed coil, 2 tail (one end off frame, tapers to a needle), 3 stub (rises
# from the depth to a medallion cap / head). Content units: x +-0.8, y +-0.5 (aspect 1.6). z = height (camera side +).
# (kind, z, z swing, radius, a, b, (harmonic, amplitude, phase) x3, family, slither direction)
# open / tail / stub: a = start (x, y), b = end (x, y); coil: a = centre, b = radii
LAYOUT = [
    (0, 0.00, 0.03, 0.190, (-1.15, -0.06), (1.15, 0.02), ((1.5, 0.15, 0.3), (2.5, 0.05, 1.7), (3.5, 0.025, 0.4)), 1, 1),
    (0, 0.04, 0.03, 0.180, (-1.15, 0.40), (1.15, 0.31), ((1.5, 0.11, 2.1), (2.5, 0.04, 0.2), (3.5, 0.02, 2.9)), 3, -1),
    (0, 0.02, 0.03, 0.180, (-1.15, -0.42), (1.15, -0.37), ((1.5, 0.10, 4.0), (2.5, 0.05, 2.6), (3.5, 0.02, 1.1)), 2, -1),
    (1, 0.30, 0.10, 0.118, (-0.16, 0.02), (0.33, 0.29), ((2.0, 0.035, 0.6), (3.0, 0.010, 2.2), (4.0, 0.003, 1.0)), 0, 1),
    (2, 0.46, 0.04, 0.092, (-0.60, 0.78), (-0.52, -0.33), ((1.0, 0.08, 0.2), (2.0, 0.04, 1.5), (3.0, 0.02, 0.4)), 4, 1),
    (0, 0.58, 0.04, 0.165, (-1.10, -0.66), (1.10, -0.60), ((1.5, 0.06, 0.9), (2.5, 0.03, 3.1), (3.5, 0.015, 2.0)), 5, 1),
    (0, 0.30, 0.10, 0.130, (0.20, 0.80), (1.00, -0.75), ((1.5, 0.10, 1.2), (2.5, 0.04, 0.5), (3.5, 0.02, 2.5)), 6, -1),
    (0, 0.55, 0.04, 0.150, (-1.10, 0.64), (1.10, 0.57), ((1.5, 0.06, 2.4), (2.5, 0.03, 0.7), (3.5, 0.015, 1.4)), 4, -1),
    (2, 0.50, 0.04, 0.082, (1.05, 0.12), (0.20, 0.22), ((1.0, 0.06, 1.0), (2.0, 0.03, 2.0), (3.0, 0.02, 0.3)), 2, 1),
    (3, 0.00, 0.48, 0.105, (-1.10, -0.02), (-0.40, -0.26), ((1.0, 0.03, 0.4), (2.0, 0.01, 1.0), (3.0, 0.0, 0.0)), 6, 1),
    (1, 0.20, 0.07, 0.090, (0.48, -0.16), (0.23, 0.21), ((2.0, 0.025, 1.9), (3.0, 0.008, 0.3), (4.0, 0.002, 2.2)), 1, -1),
    (0, 0.62, 0.04, 0.135, (-0.96, 0.85), (-0.84, -0.85), ((1.5, 0.07, 0.7), (2.5, 0.03, 1.9), (3.5, 0.015, 0.2)), 3, 1),
    (2, 0.64, 0.05, 0.080, (1.10, -0.52), (0.58, -0.17), ((1.0, 0.045, 2.6), (2.0, 0.012, 0.9), (3.0, 0.0, 1.6)), 0, -1),
    (3, 0.00, 0.48, 0.095, (1.10, 0.52), (0.40, 0.36), ((1.0, 0.03, 1.1), (2.0, 0.01, 2.0), (3.0, 0.0, 0.0)), 5, -1),
]
CAM_H = 2.0                           # camera height above the floor (z 0)
Z_REF = 0.3                           # height that is drawn 1:1 when the dolly is at rest


class Nest:
    """CPU side: phases (float64, wrapped) + the knot texture for the pass. step(dt, values, clock) -> self."""

    def __init__(self, aspect=1.6):
        self.aspect = aspect
        self.ph_writhe = 0.0          # wrapped at TAU; integer harmonics only -> exact repeat
        self.ph_dolly = 0.0
        self.scroll = np.zeros(NMAX)  # skin phase per body in cells, wrapped at self.wrap
        self.n = 10
        self.A = np.zeros((NMAX, M, 4), dtype="f8")    # x, y, screen radius, height
        self.B = np.zeros((NMAX, M, 4), dtype="f8")    # tangent x, y (per knot index), u (world arc), world radius
        self.kind = np.array([e[0] for e in LAYOUT])
        self.closed = self.kind == 1
        self.last = np.full(NMAX, M - 1)
        self.fam = np.array([e[7] for e in LAYOUT])
        self.dirn = np.array([float(e[8]) for e in LAYOUT])
        self.spd = 0.8 + 0.4 * ((np.arange(NMAX) * 0.618034) % 1.0)
        self.length = np.ones(NMAX)
        self.cells = np.ones(NMAX)
        self.wrap = np.full(NMAX, OPEN_WRAP)
        self.scale_min = 1.0
        self.thick = 1.0
        self._rest = None
        self.heads = False
        self._dolly = 0.6

    @property
    def knots(self):
        return self.A, self.B

    def texture_data(self):
        out = np.empty((2 * NMAX, M, 4), dtype="f4")
        out[0::2] = self.A
        out[1::2] = self.B
        return out

    def _world(self, i, s, w):
        """Body i at parameters s (array) with writhe amount w -> x, y, z, radius (world)."""
        kind, z0, zs, R, a, b, harm, _fam, _d = LAYOUT[i]
        ph = self.ph_writhe
        R = R * self.thick * (1.0 + 0.04 * w * math.sin(2.0 * ph + i))
        if kind == 1:
            disp = np.zeros_like(s)
            for k, (hk, amp, phs) in enumerate(harm):
                disp += amp * np.sin(TAU * hk * s + phs + w * 1.1 * math.sin((k + 1) * ph + i * 0.7))
            ang = TAU * s
            x = a[0] + b[0] * np.cos(ang) * (1.0 + disp)
            y = a[1] + b[1] * np.sin(ang) * (1.0 + disp)
            z = z0 + zs * np.sin(TAU * 2.0 * s + i + w * 0.6 * math.sin(ph + i))
            r = np.full_like(s, R)
        else:
            ax, ay = a
            bx, by = b
            dx, dy = bx - ax, by - ay
            ln = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / ln, dx / ln
            lat = np.zeros_like(s)
            for k, (hk, amp, phs) in enumerate(harm):
                lat += amp * np.sin(math.pi * hk * s + phs + w * 0.9 * math.sin((k + 1) * ph + i * 1.3))
            if kind == 2:                             # the tail tip sways most
                lat += w * 0.07 * s * s * math.sin(ph + i)
            x = ax + dx * s + nx * lat
            y = ay + dy * s + ny * lat
            if kind == 3:                             # a stub rises from under the others to a cap facing the camera
                rise = np.clip(s, 0.0, 1.0)           # gently over its whole length: the projected path never doubles back
                z = z0 + zs * rise ** 1.3
                r = np.full_like(s, R) * (1.0 + 0.25 * rise)
                if self.heads:                        # a broader, blunt ovoid head at the top end
                    r = r * (1.0 + 0.35 * np.clip((s - 0.78) / 0.22, 0.0, 1.0) ** 1.5)
            else:
                z = z0 + zs * np.sin(math.pi * 2.0 * s + i + w * 0.6 * math.sin(ph + i))
                r = np.full_like(s, R)
            if kind == 2:                             # needle tail over the last ~35 % (smooth concave taper)
                tp = np.clip((1.0 - s) / 0.35, 0.0, 1.0)
                r = R * (0.025 + 0.975 * tp ** 1.35)
        return x, y, z, r

    def _project(self, x, y, z):
        d = _num(self._dolly, 0.6)
        hc = CAM_H * (1.0 + 0.15 * d * math.sin(self.ph_dolly))
        sc = (CAM_H - Z_REF) / np.maximum(hc - z, 0.2)
        cx = 0.02 * d * math.sin(self.ph_dolly * 2.0 + 0.5)
        cy = 0.015 * d * math.cos(self.ph_dolly * 3.0)
        return (x - cx) * sc, (y - cy) * sc, sc

    def step(self, dt, values, clock=None, geometry=True):
        """Integrate the phases (always) and rebuild the knots (geometry=True)."""
        dt = min(max(_num(dt), 0.0), 1.0)
        self.n = int(round(_val(values, "coil_count", 10.0, 6.0, 14.0)))
        w = _val(values, "coil_writhe", 0.45, 0.0, 1.0)
        self._dolly = _val(values, "coil_dolly", 0.6, 0.0, 1.0)
        self.thick = 0.75 + 0.55 * _val(values, "coil_thick", 0.5, 0.0, 1.0)
        self.heads = _val(values, "coil_heads", 0.0, 0.0, 1.0) >= 0.5
        slither = _val(values, "coil_slither", 0.4, 0.0, 1.0)
        self.ph_writhe = (self.ph_writhe + dt * TAU / 20.0) % TAU        # 20 s base cycle (0.05-0.15 Hz harmonics)
        self.ph_dolly = (self.ph_dolly + dt * TAU / 24.0) % TAU          # 24 s breathing dolly
        if self._rest is None or self._rest[0] != self.thick:
            rest = self._rest_lengths()
            cl = np.array([1.45 * e[3] for e in LAYOUT]) * self.thick
            wrap = np.where(self.closed, np.maximum(2.0, 2.0 * np.round(rest / cl / 2.0)), OPEN_WRAP).astype(int)
            self._rest = (self.thick, rest, cl, wrap)
            self.scroll %= wrap
        _t, rest, cl, wrap = self._rest
        self.wrap = wrap
        self.scroll = (self.scroll + dt * slither * 0.45 * self.dirn * self.spd) % wrap
        if not geometry:
            return self
        smin = 10.0
        for i in range(NMAX):
            closed = bool(self.closed[i])
            s = np.arange(M) / (M if closed else (M - 1))
            ds = 1.0 / (M if closed else (M - 1))
            x, y, z, r = self._world(i, s, w)
            X, Y, sc = self._project(x, y, z)
            e = 1e-3
            x1, y1, z1, _ = self._world(i, s + e, w)
            x0, y0, z0, _ = self._world(i, s - e, w)
            X1, Y1, _ = self._project(x1, y1, z1)
            X0, Y0, _ = self._project(x0, y0, z0)
            tx, ty = (X1 - X0) / (2 * e) * ds, (Y1 - Y0) / (2 * e) * ds
            sf = np.arange(4 * M + 1) / (4 * M)            # u = world arc length, measured on a 4x finer sampling
            xf, yf, zf, _ = self._world(i, sf, w)
            seg = np.sqrt(np.diff(xf) ** 2 + np.diff(yf) ** 2 + np.diff(zf) ** 2)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            L = float(cum[-1])
            self.length[i] = L
            self.cells[i] = wrap[i] / L if closed else 1.0 / cl[i]     # a closed coil: a whole, even number of cells
            self.A[i, :, 0], self.A[i, :, 1], self.A[i, :, 2], self.A[i, :, 3] = X, Y, np.maximum(r * sc, 1e-3), z
            self.B[i, :, 0], self.B[i, :, 1], self.B[i, :, 2], self.B[i, :, 3] = tx, ty, np.interp(s, sf, cum), r
            smin = min(smin, float(sc.min()))
        self.scale_min = smin
        return self

    def _rest_lengths(self):
        keep = self.ph_writhe
        self.ph_writhe = 0.0
        out = np.zeros(NMAX)
        for i in range(NMAX):
            sf = np.linspace(0.0, 1.0, 4 * M + 1)
            xf, yf, zf, _ = self._world(i, sf, 0.0)
            out[i] = np.sum(np.sqrt(np.diff(xf) ** 2 + np.diff(yf) ** 2 + np.diff(zf) ** 2))
        self.ph_writhe = keep
        return out


def hermite(nest, i, s):
    """numpy mirror of the shader's curve: body i at knot parameters s -> P (n,2), D (n,2), r, z, U, R."""
    A, B = nest.A[i], nest.B[i]
    s = np.asarray(s, dtype="f8")
    last = int(nest.last[i])
    if nest.closed[i]:
        s = np.where(s >= last + 1.0, last + 1.0, np.mod(s, last + 1.0))
    else:
        s = np.clip(s, 0.0, float(last))
    k = np.minimum(np.floor(s).astype(int), last if nest.closed[i] else last - 1)
    t = s - k
    k1 = np.where(k >= last, 0, k + 1)
    t2, t3 = t * t, t * t * t
    h00, h10, h01, h11 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2
    d00, d10, d01, d11 = 6 * t2 - 6 * t, 3 * t2 - 4 * t + 1, -6 * t2 + 6 * t, 3 * t2 - 2 * t
    P0, P1, T0, T1 = A[k, :2], A[k1, :2], B[k, :2], B[k1, :2]
    P = h00[:, None] * P0 + h10[:, None] * T0 + h01[:, None] * P1 + h11[:, None] * T1
    D = d00[:, None] * P0 + d10[:, None] * T0 + d01[:, None] * P1 + d11[:, None] * T1
    r = A[k, 2] + (A[k1, 2] - A[k, 2]) * t
    z = A[k, 3] + (A[k1, 3] - A[k, 3]) * t
    U1 = np.where(k1 == 0, nest.length[i], B[k1, 2]) if nest.closed[i] else B[k1, 2]
    U = B[k, 2] + (U1 - B[k, 2]) * t
    R = B[k, 3] + (B[k1, 3] - B[k, 3]) * t
    return P, D, r, z, U, R


# ------------------------------------------------------------------ GLSL
VS = """
#version 330
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

COMMON = """
#version 330
uniform sampler2D uKnots;            // RGBA32F 24 x 28: row 2i = x, y, screen r, height; row 2i+1 = tangent, u, R
uniform vec4 uTube[14];              // closed, last knot, family, kind
uniform vec4 uTube2[14];             // cells per u, scroll (cells), wrap (cells), total length
uniform int uN;
uniform vec2 uFine;                  // pass 2 target size (px)
uniform float uAsp;
vec4 kA(int i, int k){ return texelFetch(uKnots, ivec2(k, 2*i), 0); }
vec4 kB(int i, int k){ return texelFetch(uKnots, ivec2(k, 2*i+1), 0); }
"""

COARSE_FS = COMMON + """
out vec4 o;
void main(){
  vec2 p = (gl_FragCoord.xy*%(C)d.0/uFine - 0.5)*vec2(uAsp, 1.0);
  float marg = 3.2/uFine.y;                           // half diagonal of a %(C)dx%(C)d block of pass-2 pixels
  vec4 ck = vec4(-1e9), cv = vec4(-1.0);
  for (int i = 0; i < 14; i++) {
    if (i >= uN) break;
    int last = int(uTube[i].y);
    bool closed = uTube[i].x > 0.5;
    int nseg = closed ? last + 1 : last;
    float best = 1e9, bs = 0.0, bz = 0.0, br = 0.0;
    vec4 a = kA(i, 0);
    for (int k = 0; k < 24; k++) {
      if (k >= nseg) break;
      vec4 b = kA(i, k == last ? 0 : k + 1);
      vec2 ab = b.xy - a.xy;
      float h = clamp(dot(p - a.xy, ab)/max(dot(ab, ab), 1e-9), 0.0, 1.0);
      float rr = mix(a.z, b.z, h);
      float d = length(p - a.xy - ab*h) - rr;
      if (d < best) { best = d; bs = float(k) + h; bz = mix(a.w, b.w, h); br = rr; }
      a = b;
    }
    if (best < br*0.8 + marg + 0.03) {               // covering, or close enough to shadow what it touches
      float kk = bz + br + (best < marg + 0.03 ? 10.0 : 0.0);
      float e = float(i)*64.0 + bs;
      if (kk > ck.w) {
        if (kk > ck.x)      { ck = vec4(kk, ck.xyz); cv = vec4(e, cv.xyz); }
        else if (kk > ck.y) { ck = vec4(ck.x, kk, ck.yz); cv = vec4(cv.x, e, cv.yz); }
        else if (kk > ck.z) { ck = vec4(ck.xy, kk, ck.z); cv = vec4(cv.xy, e, cv.z); }
        else                { ck.w = kk; cv.w = e; }
      }
    }
  }
  o = cv;
}
""" % dict(C=COARSE)

KENE_GLSL = """
const float TAU = 6.28318530718;
// Hash without Sine, David Hoskins (MIT; THIRD_PARTY_NOTICES.md)
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y)*p3.z); }
float aaL(float d, float w, float fw){ return 1.0 - smoothstep(w - fw, w + fw, d); }
// fine orthogonal meander: square Truchet (quarter squares round opposite corners), 6 per cell
float meander(vec2 q, float fw, float wrap){
  vec2 sq = q*8.0, sc = floor(sq), sf = fract(sq) - 0.5;
  vec2 key = vec2(mod(sc.x, wrap*8.0), sc.y);
  if (hash12(key + 7.0) > 0.5) sf.x = -sf.x;
  if (hash12(key + 13.0) > 0.6) sf = sf.yx;
  float d = min(abs(max(abs(sf.x + 0.5), abs(sf.y + 0.5)) - 0.5), abs(max(abs(sf.x - 0.5), abs(sf.y - 0.5)) - 0.5));
  float f8 = fw*8.0;
  return aaL(d, 0.12, max(f8, 0.01))*(1.0 - smoothstep(0.25, 0.6, f8));
}
// parallel echo lines inside a cell, following its outline (dl = distance to the outline), broken in short steps
float echo(float dl, float fw, float hc, vec2 q, float wrap){
  float sp = 0.052, k = dl/sp, b = floor(k);
  float d = abs(fract(k) - 0.5)*sp;
  float brk = step(0.16, hash12(vec2(b, mod(floor(q.x*5.0 + hc*7.0), wrap*5.0) + 640.0*floor(q.y*5.0))));
  return aaL(d, 0.0085, max(fw, 0.004))*brk*(1.0 - smoothstep(0.012, 0.03, fw));
}
// The four semantic channels of a kené-type skin in cell space q (1 cell = 1 unit, 4 cells round the girth):
// x = primary lines, y = fine fill, z = cloth ground, w = jewel accents. halo = soft glow of the primary lines,
// hc = this cell's hash (cell-by-cell reveal), ac = which accent colour.
vec4 kene(vec2 q, vec2 fq, int type, float wrap, out float halo, out float hc, out float ac){
  float fw = max(fq.x, fq.y);
  float P = 0.0, F = 0.0, A = 0.0, region = 0.0, dl = 1.0;
  if (type == 0) {                                    // stepped-diamond medallions with an echo line and a cross
    vec2 r = vec2(q.x + q.y, q.x - q.y)*0.5, id = floor(r), f = fract(r) - 0.5, a = abs(f);
    float fr = fw*0.71, cheb = max(a.x, a.y), manh = a.x + a.y, m = max(cheb, manh*0.72);
    dl = abs(m - 0.455);
    P = max(aaL(dl, 0.034, fr), aaL(abs(m - 0.37), 0.012, fr));
    P = max(P, aaL(abs(cheb - 0.20), 0.022, fr));      // nested diamonds (in q these are diamonds, in r squares)
    dl = min(dl, abs(cheb - 0.20));
    P = max(P, aaL(cheb - 0.075, 0.0, fr));
    region = step(m, 0.345)*step(0.235, cheb);
    hc = hash12(vec2(mod(id.x + id.y, wrap), id.x - id.y));
    vec2 vid = floor(r + 0.5);
    float dv = abs(a.x - 0.5) + abs(a.y - 0.5);
    float hv = hash12(vec2(mod(vid.x + vid.y, wrap), vid.x - vid.y) + 3.1);
    A = aaL(dv, 0.085, fr)*step(0.35, hv);
    ac = hv;
  } else if (type == 1) {                             // elongated hex cells, double rail, cross-stars
    const vec2 s = vec2(1.0, 1.7320508);
    vec2 qq = vec2(q.x, q.y*0.8);
    vec2 a1 = mod(qq, s) - s*0.5, b1 = mod(qq - s*0.5, s) - s*0.5;
    vec2 gv = dot(a1, a1) < dot(b1, b1) ? a1 : b1, cid = qq - gv, ag = abs(gv);
    float hd = max(dot(ag, normalize(s)), ag.x);
    dl = abs(0.485 - hd);
    P = max(aaL(dl, 0.03, fw), aaL(abs(0.405 - hd), 0.011, fw));
    hc = hash12(vec2(mod(floor(cid.x*2.0 + 0.5), wrap*2.0), floor(cid.y*2.0 + 0.5)));
    float star = step(0.5, hc), manh = ag.x + ag.y, chev = max(ag.x, ag.y);
    P = max(P, star*aaL(abs(mix(manh, chev, 0.35) - 0.22), 0.022, fw));
    P = max(P, star*aaL(abs(chev - 0.07), 0.018, fw));
    region = step(hd, 0.37)*(star > 0.5 ? step(0.30, mix(manh, chev, 0.35)) : 1.0);
    A = (1.0 - star)*aaL(manh, 0.075, fw)*step(0.3, fract(hc*7.31));
    ac = fract(hc*3.7);
  } else {                                            // orthogonal double channels, stepped crosses, centre diamond
    vec2 id = floor(q), f = fract(q) - 0.5, a = abs(f);
    float de = min(0.5 - a.x, 0.5 - a.y), manh = a.x + a.y;
    dl = de;
    P = max(aaL(de, 0.032, fw), aaL(abs(de - 0.075), 0.011, fw));
    float c1 = min(max(a.x - 0.30, a.y - 0.10), max(a.x - 0.10, a.y - 0.30));          // greek cross
    float c2 = min(max(a.x - 0.20, a.y - 0.20), c1);                                    // stepped by a square
    P = max(P, aaL(abs(c2), 0.026, fw));
    P = max(P, aaL(abs(manh - 0.11), 0.018, fw));
    region = step(0.05, c2)*step(0.11, de);
    hc = hash12(vec2(mod(id.x, wrap), id.y));
    float dv = abs(a.x - 0.5) + abs(a.y - 0.5);
    vec2 vid = floor(q + 0.5);
    float hv = hash12(vec2(mod(vid.x, wrap), vid.y) + 5.7);
    A = aaL(dv, 0.09, fw)*step(0.3, hv);
    ac = hv;
  }
  F = (type == 2 ? meander(q, fw, wrap) : echo(dl, fw, hc, q, wrap))*region*(1.0 - P);
  halo = exp(-dl*16.0);
  return vec4(P, F, 1.0, A*(1.0 - P));
}
"""

FINE_FS = COMMON + KENE_GLSL + """
uniform sampler2D uCand;
uniform vec4 uG0, uG1;               // gains: line, fill, ground, accent | rim, mono, patch, reveal
uniform float uLift, uAmt, uFloor, uGlow, uPalMix, uSweep, uHeads, uCaps, uDebug, uBake, uBakePh;
uniform vec2 uPatch;                 // patch-noise drift (wrapped at 64)
uniform vec3 uPal[5];
uniform sampler2D texLay;            // layer skin, a plain texture: colour (mode 1) or 4 channels (mode 2)
uniform sampler2DArray texSkinL;     // ... or a layer of materials.LayerHub's array (unit 23), layer uLayIdx
uniform float uLayMode, uLayCells, uLaySrc, uLayIdx;
out vec4 o;
const vec3 FAM[35] = vec3[35](
    vec3(.012,.010,.024), vec3(.82,.15,.70), vec3(.86,.92,.88), vec3(.30,.78,.74), vec3(.95,.95,.95),
    vec3(.18,.15,.74), vec3(.78,.33,.16), vec3(.94,.88,.80), vec3(.95,.78,.24), vec3(.90,.06,.10),
    vec3(.52,.02,.13), vec3(.92,.80,.26), vec3(.05,.00,.02), vec3(.82,.18,.72), vec3(.14,.32,.95),
    vec3(.11,.04,.72), vec3(.92,.86,.56), vec3(.76,.82,.96), vec3(.96,.96,.92), vec3(.95,.78,.24),
    vec3(.13,.50,.38), vec3(.02,.04,.03), vec3(.36,.80,.58), vec3(.90,.04,.06), vec3(.14,.34,1.0),
    vec3(.03,.065,.06), vec3(.88,.48,.26), vec3(.90,.89,.88), vec3(.38,.84,.80), vec3(.95,.95,.95),
    vec3(.05,.05,.34), vec3(.88,.88,.92), vec3(.62,.62,.84), vec3(.12,.92,.10), vec3(.90,.22,.78));
vec3 famCol(int f, int k){ return FAM[f*5 + k]; }   // k: 0 ground, 1 primary, 2 fill, 3 accent a, 4 accent b
int famType(int f){ return f == 0 || f == 3 ? 1 : (f == 2 || f == 6 ? 0 : 2); }
float vnoise(vec2 x){
  vec2 i = floor(x), f = fract(x); f = f*f*(3.0 - 2.0*f);
  float a = hash12(mod(i, 64.0)), b = hash12(mod(i + vec2(1, 0), 64.0));
  float c = hash12(mod(i + vec2(0, 1), 64.0)), d = hash12(mod(i + vec2(1, 1), 64.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
void ev(int i, float s, out vec2 P, out vec2 D, out float r, out float z, out float U, out float R){
  int last = int(uTube[i].y);
  bool closed = uTube[i].x > 0.5;
  s = closed ? mod(s, float(last + 1)) : clamp(s, 0.0, float(last));
  int k = min(int(floor(s)), closed ? last : last - 1);
  float t = s - float(k);
  int k1 = k >= last ? 0 : k + 1;
  vec4 a0 = kA(i, k), a1 = kA(i, k1), b0 = kB(i, k), b1 = kB(i, k1);
  float t2 = t*t, t3 = t2*t;
  P = (2.0*t3 - 3.0*t2 + 1.0)*a0.xy + (t3 - 2.0*t2 + t)*b0.xy + (3.0*t2 - 2.0*t3)*a1.xy + (t3 - t2)*b1.xy;
  D = (6.0*t2 - 6.0*t)*a0.xy + (3.0*t2 - 4.0*t + 1.0)*b0.xy + (6.0*t - 6.0*t2)*a1.xy + (3.0*t2 - 2.0*t)*b1.xy;
  r = mix(a0.z, a1.z, t); z = mix(a0.w, a1.w, t);
  U = mix(b0.z, (closed && k1 == 0) ? uTube2[i].w : b1.z, t);
  R = mix(b0.w, b1.w, t);
}
float closest(int i, vec2 p, float s){
  // 2 Newton steps on f(s) = |P(s) - p|^2 / 2 with the curve's second derivative (converges on tight bends, where
  // Gauss-Newton crawls); the step is capped at one knot and the curvature term never makes f'' small or negative.
  int last = int(uTube[i].y);
  bool closed = uTube[i].x > 0.5;
  for (int n = 0; n < 2; n++) {
    s = closed ? mod(s, float(last + 1)) : clamp(s, 0.0, float(last));
    int k = min(int(floor(s)), closed ? last : last - 1);
    float t = s - float(k);
    int k1 = k >= last ? 0 : k + 1;
    vec4 a0 = kA(i, k), a1 = kA(i, k1), b0 = kB(i, k), b1 = kB(i, k1);
    float t2 = t*t, t3 = t2*t;
    vec2 P = (2.0*t3 - 3.0*t2 + 1.0)*a0.xy + (t3 - 2.0*t2 + t)*b0.xy + (3.0*t2 - 2.0*t3)*a1.xy + (t3 - t2)*b1.xy;
    vec2 D = (6.0*t2 - 6.0*t)*a0.xy + (3.0*t2 - 4.0*t + 1.0)*b0.xy + (6.0*t - 6.0*t2)*a1.xy + (3.0*t2 - 2.0*t)*b1.xy;
    vec2 D2 = (12.0*t - 6.0)*a0.xy + (6.0*t - 4.0)*b0.xy + (6.0 - 12.0*t)*a1.xy + (6.0*t - 2.0)*b1.xy;
    vec2 Q = P - p;
    float dd = dot(D, D);
    s -= clamp(dot(Q, D)/max(dd + dot(Q, D2), 0.3*dd + 1e-8), -1.0, 1.0);
  }
  return closed ? mod(s, float(last + 1)) : clamp(s, 0.0, float(last));
}
vec3 palF(float x){ x = fract(x)*4.0; int i = int(x); float f = x - float(i);
  return mix(uPal[i], uPal[min(i + 1, 4)], f); }
//@SKIN
// cells per layer tile along u: uLayCells, snapped so a closed coil holds whole tiles (`per` tiles = one period)
float lcu(float wrap, float per){ return wrap >= 63.5 ? uLayCells : wrap/max(per, per*floor(wrap/(uLayCells*per) + 0.5)); }
vec4 layTap(vec2 uv){
  if (uLaySrc > 0.5) { uv = 1.0 - abs(mod(uv, 2.0) - 1.0); return texture(texSkinL, vec3(uv, uLayIdx)); }
  return texture(texLay, uv);
}
void main(){
  vec2 fc = gl_FragCoord.xy;
  vec2 p = (fc/uFine - 0.5)*vec2(uAsp, 1.0);
  float px = 1.0/uFine.y;
  float cov = 1.0, h = 1.0, ao = 1.0, depthK = 1.0, wrap = 8.0, qx = 0.0, bs = 0.0;
  vec3 n = vec3(0.0, 0.0, 1.0);
  vec2 q, gx, gy, fq;
  int bi = 0, fam = 0, kind = 0;
  if (uBake > 0.5) {                                  // flat skin strip for the scene's serpents: 7 family bands,
    float band = uFine.y/7.0;                         // 8 cells along (tileable), v = the front half of the girth
    fam = min(int(fc.y/band), 6);
    float y = fract(fc.y/band);
    float dn = clamp(y*2.0 - 1.0, -1.0, 1.0);
    qx = fc.x/uFine.x*8.0 + uBakePh;
    q = vec2(qx, (0.5 + asin(dn)/TAU)*4.0);
    gx = vec2(8.0/uFine.x, 0.0); gy = vec2(0.0, 2.0/band);
    fq = clamp(abs(gx) + abs(gy), vec2(1e-4), vec2(0.5));
    p = vec2(qx*0.1, y);
  } else {
    vec4 cand = texelFetch(uCand, ivec2(fc)/%(C)d, 0);
    if (cand.x < 0.0) { o = vec4(0.0, 0.0, 0.0, uFloor*uAmt); return; }
    float zc[4]; float dd[4]; float rr[4]; float RR[4];
    for (int c = 0; c < 4; c++) { zc[c] = -1e9; dd[c] = 1e9; rr[c] = 1.0; RR[c] = 1.0; }
    float bz = -1e9, bd = 0.0, br = 1.0, bU = 0.0, bR = 1.0, bzc = 0.0, bh = 0.0;
    bi = -1; int bc = -1;
    vec2 bP = vec2(0), bD = vec2(1, 0);
    for (int c = 0; c < 4; c++) {
      float e = cand[c];
      if (e < 0.0) break;
      int i = int(floor(e/64.0 + 1e-4));
      float s = closest(i, p, e - float(i)*64.0);
      vec2 P, D; float r, z, U, R;
      ev(i, s, P, D, r, z, U, R);
      float d = length(p - P);
      zc[c] = z; dd[c] = d; rr[c] = r; RR[c] = R;
        if (d < r) {
        float h = sqrt(max(1.0 - d*d/(r*r), 0.0));
        float zt = z + R*h;
        if (zt > bz) { bz = zt; bi = i; bc = c; bs = s; bd = d; br = r; bU = U; bR = R; bzc = z; bh = h; bP = P; bD = D; }
      }
    }
    if (bi < 0) { o = vec4(0.0, 0.0, 0.0, uFloor*uAmt); return; }
    cov = clamp((br - bd)/px, 0.0, 1.0);
    vec2 T = normalize(bD + vec2(1e-7, 0.0)), N2 = vec2(-T.y, T.x);
    vec2 rel = p - bP;
    float dn = clamp(dot(rel, N2)/br, -1.0, 1.0);
    n = normalize(vec3(rel/br, bh + 1e-4));
    h = bh;
    // contact shadow from the bodies above / beside this one (the over-under weave), crushed to black
    float occ = 0.0;
    for (int c = 0; c < 4; c++) {
      if (c == bc || zc[c] < -1e8) continue;
      float top = zc[c] + RR[c];
      if (top < bz - 0.01) continue;
      float e = (dd[c] - rr[c])/(rr[c]*0.6 + 1e-4);
      occ += (1.0 - smoothstep(0.0, 1.0, e))*clamp((top - bz)/(RR[c] + 1e-4)*1.6 + 0.35, 0.0, 1.0);
    }
    ao = clamp(1.0 - 0.92*occ, 0.0, 1.0);
    ao *= ao;
    depthK = mix(0.5, 1.0, smoothstep(-0.05, 0.68, bz));
    // body uv: u along (cells, scrolled), v round the girth (front half = 0.25 .. 0.75)
    wrap = uTube2[bi].z;
    qx = bU*uTube2[bi].x + uTube2[bi].y;
    float v = 0.5 + asin(dn)/TAU;
    q = vec2(qx, v*4.0);
    gx = dFdx(q); gy = dFdy(q);
    gx.x -= wrap*floor(gx.x/wrap + 0.5); gy.x -= wrap*floor(gy.x/wrap + 0.5);
    fq = clamp(abs(gx) + abs(gy), vec2(1e-4), vec2(0.5));
    if (uDebug > 0.5) {
      float a = TAU*qx/wrap;
      o = vec4(0.5 + 0.5*cos(a), 0.5 + 0.5*sin(a), (float(bi) + 1.0)/16.0, cov > 0.99 ? 1.0 : 0.0);
      return;
    }
    fam = int(uTube[bi].z + 0.5); kind = int(uTube[bi].w + 0.5);
  }
  float halo, hc, ac;
  vec4 ch = kene(q, fq, famType(fam), wrap, halo, hc, ac);
  // ---- end caps: a stub's top end is a medallion (or a head with jewel eyes)
  float last = uTube[bi].y;
  float capW = 0.0;
  if (kind == 3) {
    if (bs > last - 3.0) {
      vec2 Pe, De; float re, ze, Ue, Re;
      ev(bi, last, Pe, De, re, ze, Ue, Re);
      vec2 cr = (p - Pe)/re;
      float rho = length(cr), al = atan(cr.y, cr.x + 1e-6);
      capW = 1.0 - smoothstep(0.97, 1.03, rho);
      vec4 cap;
      if (uHeads > 0.5) {                              // head: the skin runs on, two jewel eyes look out
        vec2 Te = normalize(De + vec2(1e-7, 0.0)), Ne = vec2(-Te.y, Te.x);
        float ea = dot(cr, Te), eb = abs(dot(cr, Ne));
        float eye = aaL(abs(ea - 0.22)*1.4 + abs(eb - 0.40), 0.20, 0.03);
        float brow = aaL(abs(length(vec2(ea - 0.22, eb - 0.40)) - 0.27), 0.035, 0.03);      // a ring round each eye
        float snout = aaL(abs(rho - 0.86), 0.05, 0.03)*step(0.0, ea);                       // the lip line
        cap = vec4(max(ch.x*0.6, max(brow, snout)), ch.y*(1.0 - eye), 1.0, eye);
        ac = eye > 0.5 ? 0.9 : ac;
      } else if (uCaps > 0.5) {                        // medallion: rosette of petals in concentric rings
        float pr = rho*(1.0 + 0.16*cos(6.0*al)), fwc = fwidth(rho) + 1e-4;
        float pl = max(max(aaL(abs(pr - 0.74), 0.045, fwc), aaL(abs(rho - 0.93), 0.03, fwc)),
                       max(aaL(abs(pr - 0.46), 0.03, fwc), aaL(abs(rho - 0.22), 0.025, fwc)));
        float fl = meander(vec2(al*6.0/TAU*4.0, rho*5.0), fwc*6.0, 64.0)*step(0.5, pr)*step(pr, 0.69);
        cap = vec4(pl, fl*(1.0 - pl), 1.0, aaL(rho, 0.1, fwc));
        halo = exp(-abs(pr - 0.74)*16.0);
        ac = 0.05;
      } else {
        cap = ch;
      }
      if (capW > 0.0) ch = mix(ch, cap, capW);
    }
  }
  float gL = uG0.x*(1.0 + uLift), gF = uG0.y, gG = uG0.z, gA = uG0.w;
  float gR = uG1.x, mono = uG1.y, patch = uG1.z, rev = uG1.w;
  vec3 gcol = famCol(fam, 0), pcol = famCol(fam, 1), fcol = famCol(fam, 2);
  vec3 acol = ac > 0.6 ? famCol(fam, 4) : famCol(fam, 3);
  if (capW > 0.5 && uCaps > 0.5 && uHeads < 0.5) pcol = vec3(.90, .66, .18);
  if (uPalMix > 0.0) {
    gcol = mix(gcol, palF(float(fam)*0.17)*0.55, uPalMix);
    pcol = mix(pcol, palF(float(fam)*0.17 + 0.35), uPalMix);
  }
  // ---- layer skins (layers as skins): 4 channels replace the procedural masks, or a colour becomes the cloth
  if (uLayMode > 1.5) {
    vec4 lc = layTap(q/vec2(lcu(wrap, 1.0), uLayCells));
    ch = vec4(lc.r, lc.g*(1.0 - lc.r), lc.b, lc.a*(1.0 - lc.r));
    halo = lc.r*0.6;
  } else if (uLayMode > 0.5) {
    vec2 lu = q/vec2(lcu(wrap, 2.0), uLayCells); lu = 1.0 - abs(mod(lu, 2.0) - 1.0);   // mirrored: seamless for any image
    vec3 lcol = layTap(lu).rgb;
    float l = dot(lcol, vec3(.299, .587, .114));
    gcol = max(lcol - 0.04, 0.0)*1.15*smoothstep(0.02, 0.2, l);
  }
//@SKIN_MIX
  pcol = mix(pcol, vec3(0.96), mono*0.75);
  fcol = mix(fcol, vec3(1.0), mono);
  // spatial reveal masks: floating ground patches (screen space), cells lighting one by one (fill)
  float pm = mix(1.0, smoothstep(0.52, 0.66, vnoise(p*2.6 + uPatch)), patch);
  float rm = smoothstep(hc - 0.06, hc, rev*1.06);
  // light: matte velvet. Emissive lines wrap round the tube (limb^0.7), the cloth is lit and darker at the limb.
  vec3 Ld = normalize(vec3(-0.35, 0.5, 0.8));
  float diff = clamp(dot(n, Ld)*0.55 + 0.45, 0.0, 1.0);
  float limbE = pow(h, 0.7), limbG = pow(h, 1.4);
  float weave = 0.86 + 0.14*sin(q.x*TAU*12.0)*sin(q.y*TAU*12.0)*(1.0 - smoothstep(0.03, 0.08, fq.x));
  vec3 col = gcol*gG*pm*ch.z*weave*diff*limbG*ao*1.25;
  float eK = limbE*(0.38 + 0.62*ao);
  col = col*(1.0 - 0.88*ch.x*min(gL, 1.0)) + pcol*ch.x*gL*eK*0.95;
  col += pcol*halo*uGlow*0.18*gL*eK*(1.0 - ch.x);
  float fm = ch.y*rm;
  col = col*(1.0 - 0.8*fm*min(gF, 1.0)) + fcol*fm*gF*eK*0.85;
  col = col*(1.0 - ch.w*min(gA, 1.0)) + acol*ch.w*gA*eK*1.05;
  // rim-light sweep: a grazing silver streak along the raised contours, travelling across the frame
  float sw = exp(-pow((dot(p, vec2(0.8, 0.6)) - uSweep)/0.22, 2.0));
  float rim = gR*sw*pow(1.0 - h, 2.0)*h*4.2*(0.3 + 0.7*max(ch.x, fm));
  col += vec3(0.82, 0.88, 1.0)*rim*(0.5 + 0.5*ao);
  col *= depthK;
  col = col/(1.0 + 0.35*col);
  if (any(isnan(col)) || any(isinf(col))) col = vec3(0.0);
  col = clamp(col, 0.0, 1.0);
  o = vec4(col*cov*uAmt, mix(cov, 1.0, uFloor)*uAmt);
}
""".replace("%(C)d", str(COARSE))

BAKE_FS = KENE_GLSL.replace("const float TAU", "#version 330\nconst float TAU") + """
uniform vec2 uSize;
out vec4 o;
void main(){
  vec2 q = gl_FragCoord.xy/uSize*4.0;                 // 4 x 4 cells per tile, tileable
  float halo, hc, ac;
  vec4 ch = kene(q, vec2(4.0/uSize.x), 0, 4.0, halo, hc, ac);
  o = vec4(ch.x, ch.y, 1.0 - ch.x, ch.w);
}
"""


def _blend_query():
    try:
        import ctypes
        lib = ctypes.windll.opengl32 if sys.platform == "win32" else ctypes.CDLL("libGL.so.1")
        lib.glIsEnabled.restype = ctypes.c_ubyte
        lib.glIsEnabled.argtypes = [ctypes.c_uint]
        return lambda: bool(lib.glIsEnabled(0x0BE2))
    except Exception:
        return None


def fine_source(skin_glsl=None):
    """Pass 2 source. With skin_glsl (materials.glsl(1, "palF")) the cloth ground can be a SKINS slot image."""
    if not skin_glsl:
        return FINE_FS.replace("//@SKIN\n", "").replace("//@SKIN_MIX\n", "")
    mix = ("  if (skA[0].w > 0.0005) {                           // SKINS slot coilnest:bodies: the image is the cloth\n"
           "    vec3 S = skTap(0, vec2(q.y/4.0, q.x/4.0), vec2(gx.y*0.25, gx.x*0.25), vec2(gy.y*0.25, gy.x*0.25));\n"
           "    S = max(S - 0.04, 0.0)*1.15*smoothstep(0.02, 0.2, dot(S, vec3(.299, .587, .114)));\n"
           "    gcol = mix(gcol, S*skE[0].rgb*skD[0].w, skA[0].w);\n"
           "  }\n")
    return FINE_FS.replace("//@SKIN\n", skin_glsl).replace("//@SKIN_MIX\n", mix)


class CoilNestPass:
    """render(values, clock, t, dt, bands, palette_rgb, layer=None, skins=None) -> RGBA8 texture (see the docstring).

    layer = (texture, LAYER_COLOUR | LAYER_KENE4), or ("hub", materials.LayerHub, array layer, kind), or None:
            a layer used as the bodies' skin (coil_skin 2 / 3).
    skins = materials.SkinEngine or None: the SKINS slot "coilnest:bodies" (coil_skin 1)."""

    LAY_UNIT, HUB_UNIT, KNOT_UNIT, CAND_UNIT = 19, 23, 27, 28   # private units; 23 = materials.UNIT_L (LayerHub)

    def __init__(self, ctx, out_size=(1280, 800)):
        self.ctx = ctx
        self.out_size = (max(16, int(out_size[0])), max(16, int(out_size[1])))
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._pc = ctx.program(vertex_shader=VS, fragment_shader=COARSE_FS)
        self._pf = ctx.program(vertex_shader=VS, fragment_shader=fine_source())
        self._vc = ctx.vertex_array(self._pc, [(self._quad, "2f", "in_pos")])
        self._vf = ctx.vertex_array(self._pf, [(self._quad, "2f", "in_pos")])
        self._ps = None                                   # skinned program (compiled on first use; False = failed)
        self._knots = ctx.texture((M, 2 * NMAX), 4, dtype="f4")
        self._knots.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._dummy = ctx.texture((1, 1), 4, bytes(4))
        self._targets = {}
        self.nest = Nest(self.out_size[0] / self.out_size[1])
        self.prog = Programme()
        self.draws = 0
        self.debug = 0
        self.res = 1.0
        self._blank = True
        self._last_t = None
        self._pal = np.array([[0.9, 0.2, 0.6], [0.95, 0.7, 0.2], [0.2, 0.5, 0.95], [0.1, 0.8, 0.5], [0.9, 0.9, 0.9]], "f4")
        self._blend_on = _blend_query()
        self._target(1.0)
        self._adv_t, self._adv = None, None
        self.bake_ph = 0.0
        self._bake = None                                 # (texture, fbo) of the serpents' skin strip, made on first use
        self.bakes = 0

    def reset(self, t=0.0):
        """Stills / tests: the nest as it is at show time t after continuous play (phases from t), gains snap."""
        t = _num(t)
        self.nest = Nest(self.out_size[0] / self.out_size[1])
        self.nest.ph_writhe = (t * TAU / 20.0) % TAU
        self.nest.ph_dolly = (t * TAU / 24.0) % TAU
        self.prog = Programme()
        self._blank = True
        self._last_t = None
        self._adv_t, self._adv = None, None
        self.bake_ph = 0.0

    def _advance(self, values, clock, t, dt, bands):
        """CPU logic once per frame (render and bake may both run): nest phases + knots, gating gains."""
        if self._adv_t == t and self._adv is not None:
            return self._adv
        snap = self._blank
        nest = self.nest.step(dt, values, clock)
        g = self.prog.step(dt, values, clock, bands[0], snap=snap)
        self.bake_ph = (self.bake_ph + dt * _val(values, "coil_slither", 0.4, 0.0, 1.0) * 0.45) % 8.0
        self._adv_t, self._adv = t, (nest, g)
        return self._adv

    def _target(self, res):
        res = round(float(res), 2)
        if res not in self._targets:
            w, h = max(16, int(self.out_size[0] * res)), max(16, int(self.out_size[1] * res))
            tex = self.ctx.texture((w, h), 4)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            fbo.clear(0.0, 0.0, 0.0, 0.0)
            cw, chh = -(-w // COARSE), -(-h // COARSE)
            ct = self.ctx.texture((cw, chh), 4, dtype="f4")
            ct.filter = (moderngl.NEAREST, moderngl.NEAREST)
            cf = self.ctx.framebuffer(color_attachments=[ct])
            self._targets[res] = (tex, fbo, ct, cf)
        return self._targets[res]

    def release(self):
        objs = [o for t in self._targets.values() for o in t[::-1]]
        objs += [self._vc, self._vf, self._pc, self._pf, self._quad, self._knots, self._dummy]
        if self._bake:
            objs += [self._bake[1], self._bake[0]]
            self._bake = None
        if self._ps:
            objs += [self._ps[1], self._ps[0]]
        self._targets = {}
        for o in objs:
            try:
                o.release()
            except Exception:
                pass

    close = release

    def _uniforms(self, values, clock, t, g, nest, amt, w, h, lay_mode, lay_src, lay_idx, bake=0.0):
        beat = _num(clock.get("beat", t * 2.0))
        return {"uKnots": self.KNOT_UNIT, "uCand": self.CAND_UNIT, "uN": nest.n, "uFine": (float(w), float(h)),
                "uAsp": w / h, "uG0": tuple(float(x) for x in g[:4]), "uG1": tuple(float(x) for x in g[4:8]),
                "uLift": float(self.prog.lift), "uAmt": amt, "uFloor": _val(values, "coil_floor", 1.0, 0.0, 1.0),
                "uGlow": _val(values, "coil_glow", 0.4, 0.0, 1.0), "uPalMix": _val(values, "coil_palette", 0.0, 0.0, 1.0),
                "uSweep": -1.3 + 2.6 * ((beat / 8.0) % 1.0), "uHeads": 1.0 if nest.heads else 0.0,
                "uCaps": 1.0 if _val(values, "coil_caps", 1.0, 0.0, 1.0) >= 0.5 else 0.0,
                "uDebug": float(self.debug), "uPatch": ((beat * 0.05) % 64.0, (beat * 0.031) % 64.0),
                "texLay": self.LAY_UNIT, "texSkinL": self.HUB_UNIT, "uLayMode": lay_mode, "uLaySrc": lay_src,
                "uLayIdx": lay_idx, "uLayCells": 4.0, "uBake": bake, "uBakePh": float(self.bake_ph)}

    def _bind_layer(self, values, layer):
        lay_mode, lay_src, lay_idx = 0.0, 0.0, 0.0
        skin_mode = int(round(_val(values, "coil_skin", 0.0, 0.0, 3.0)))
        if layer is not None and skin_mode in (2, 3):
            try:
                if layer[0] == "hub":
                    _h, hub, idx, kind = layer
                    if int(idx) >= 0:
                        hub.use()
                        lay_src, lay_idx = 1.0, float(int(idx))
                        lay_mode = float(LAYER_KENE4 if kind == LAYER_KENE4 else LAYER_COLOUR)
                else:
                    ltex, kind = layer
                    ltex.use(self.LAY_UNIT)
                    lay_mode = float(LAYER_KENE4 if kind == LAYER_KENE4 else LAYER_COLOUR)
            except Exception:
                lay_mode = 0.0
        if lay_mode == 0.0 or lay_src > 0.5:
            self._dummy.use(self.LAY_UNIT)
        return skin_mode, lay_mode, lay_src, lay_idx

    def bake(self, values, clock=None, t=0.0, dt=1 / 60, bands=(0.0, 0.0, 0.0), palette_rgb=None, layer=None,
             size=(512, 448)):
        """The bodies' skin, flat, for the scene's serpents (serpent_skin): 7 bands (one per colour family), 8 cells
        along (tileable), the same gating programme and layer skin. RGBA8, mipmapped. Works with l_coilnest at 0."""
        ctx = self.ctx
        clock = clock if isinstance(clock, dict) else {}
        t, dt = _num(t), min(max(_num(dt), 0.0), 0.25)
        bands = [min(max(_num(b), 0.0), 1.0) for b in (list(bands if bands is not None else ()) + [0, 0, 0])[:3]]
        if palette_rgb is not None:
            try:
                pal = np.asarray(palette_rgb, dtype="f4").reshape(-1, 3)
                if len(pal) and np.isfinite(pal).all():
                    self._pal = np.clip(pal[np.arange(5) % len(pal)], 0.0, 1.0)
            except (TypeError, ValueError):
                pass
        if self._bake is None:
            tex = ctx.texture(size, 4)
            tex.repeat_x, tex.repeat_y = True, False
            self._bake = (tex, ctx.framebuffer(color_attachments=[tex]))
        tex, fbo = self._bake
        nest, g = self._advance(values, clock, t, dt, bands)
        self._blank = False
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        blend_on = self._blend_on() if self._blend_on is not None else False
        try:
            ctx.disable(moderngl.BLEND)
            _sm, lay_mode, lay_src, lay_idx = self._bind_layer(values, layer)
            vals = self._uniforms(values, clock, t, g, nest, 1.0, size[0], size[1], lay_mode, lay_src, lay_idx, bake=1.0)
            vals["uFloor"], vals["uDebug"] = 1.0, 0.0
            self._knots.use(self.KNOT_UNIT)
            self._dummy.use(self.CAND_UNIT)
            for name, val in vals.items():
                self._set(self._pf, name, val)
            self._pf["uTube"].write(np.zeros((NMAX, 4), "f4").tobytes())
            if "uPal" in self._pf:
                self._pf["uPal"].write(self._pal.astype("f4").tobytes())
            fbo.use()
            ctx.viewport = (0, 0, *size)
            self._vf.render(moderngl.TRIANGLE_STRIP)
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self.bakes += 1
            return tex
        finally:
            if blend_on:
                ctx.enable(moderngl.BLEND)
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:
                ctx.screen.use()

    def _skin_prog(self):
        if self._ps is None:
            try:
                import materials
                prog = self.ctx.program(vertex_shader=VS, fragment_shader=fine_source(materials.glsl(1, "palF")))
                self._ps = (prog, self.ctx.vertex_array(prog, [(self._quad, "2f", "in_pos")]))
            except Exception as ex:
                self._ps = False
                print(f"[coilnest] SKINS slot disabled (shader did not compile): {str(ex)[:300]}")
        return self._ps or None

    @staticmethod
    def _set(prog, name, value):
        if name in prog:
            prog[name].value = value

    def render(self, values, clock=None, t=0.0, dt=None, bands=(0.0, 0.0, 0.0), palette_rgb=None, layer=None,
               skins=None):
        ctx = self.ctx
        amt = _val(values, "l_coilnest", 0.0, 0.0, 1.0)
        t = _num(t)
        if dt is None:
            dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        dt = min(max(_num(dt), 0.0), 0.25)
        bands = [min(max(_num(b), 0.0), 1.0) for b in (list(bands if bands is not None else ()) + [0, 0, 0])[:3]]
        clock = clock if isinstance(clock, dict) else {}
        if palette_rgb is not None:
            try:
                pal = np.asarray(palette_rgb, dtype="f4").reshape(-1, 3)
                if len(pal) and np.isfinite(pal).all():
                    self._pal = np.clip(pal[np.arange(5) % len(pal)], 0.0, 1.0)
            except (TypeError, ValueError):
                pass
        if amt <= 0.001:                                  # off: nothing moves, nothing is drawn
            if not self._blank:
                for tt in self._targets.values():
                    tt[1].clear(0.0, 0.0, 0.0, 0.0)
                self._blank = True
            return self._targets[1.0][0]
        nest, g = self._advance(values, clock, t, dt, bands)
        res = round(_val(values, "coilnest_res", 1.0, 0.35, 1.0), 2)
        self.res = res
        tex, fbo, ct, cf = self._target(res)
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        blend_on = self._blend_on() if self._blend_on is not None else False
        try:
            ctx.disable(moderngl.BLEND)
            self._knots.write(nest.texture_data().tobytes())
            self._knots.use(self.KNOT_UNIT)
            tube = np.zeros((NMAX, 4), "f4")
            tube2 = np.zeros((NMAX, 4), "f4")
            tube[:, 0] = nest.closed
            tube[:, 1] = nest.last
            tube[:, 2] = nest.fam
            tube[:, 3] = nest.kind
            tube2[:, 0] = nest.cells
            tube2[:, 1] = nest.scroll
            tube2[:, 2] = nest.wrap
            tube2[:, 3] = nest.length
            w, h = tex.size
            # pass 1: candidates at 1/4 resolution
            pc = self._pc
            for name, val in (("uKnots", self.KNOT_UNIT), ("uN", nest.n), ("uFine", (float(w), float(h))),
                              ("uAsp", w / h)):
                self._set(pc, name, val)
            pc["uTube"].write(tube.tobytes())
            if "uTube2" in pc:
                pc["uTube2"].write(tube2.tobytes())
            cf.use()
            ctx.viewport = (0, 0, *ct.size)
            self._vc.render(moderngl.TRIANGLE_STRIP)
            # pass 2
            skin_mode, lay_mode, lay_src, lay_idx = self._bind_layer(values, layer)
            prog, vao = self._pf, self._vf
            if skin_mode == 1 and skins is not None and not self.debug:
                try:
                    if skins.active("coilnest"):
                        sp = self._skin_prog()
                        if sp:
                            prog, vao = sp
                            skins.bind(prog, "coilnest")
                except Exception as ex:
                    print(f"[coilnest] skin bind failed: {ex}")
                    prog, vao = self._pf, self._vf
            ct.use(self.CAND_UNIT)
            vals = self._uniforms(values, clock, t, g, nest, amt, w, h, lay_mode, lay_src, lay_idx)
            for name, val in vals.items():
                self._set(prog, name, val)
            prog["uTube"].write(tube.tobytes())
            prog["uTube2"].write(tube2.tobytes())
            if "uPal" in prog:
                prog["uPal"].write(self._pal.astype("f4").tobytes())
            fbo.use()
            ctx.viewport = (0, 0, w, h)
            vao.render(moderngl.TRIANGLE_STRIP)
            self.draws += 1
            self._blank = False
            return tex
        finally:
            if blend_on:
                ctx.enable(moderngl.BLEND)
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:
                ctx.screen.use()


def bake_channels(ctx, size=256):
    """A tileable 4-channel kené tile (R primary, G fill, B ground, A accents) = the layer-skin contract, made from
    our own procedural kené. Tests and the gallery use it until the kené study layer exports its channels."""
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
    prog = ctx.program(vertex_shader=VS, fragment_shader=BAKE_FS)
    vao = ctx.vertex_array(prog, [(quad, "2f", "in_pos")])
    tex = ctx.texture((size, size), 4)
    fbo = ctx.framebuffer(color_attachments=[tex])
    prev = ctx.fbo
    fbo.use()
    ctx.viewport = (0, 0, size, size)
    prog["uSize"].value = (float(size), float(size))
    vao.render(moderngl.TRIANGLE_STRIP)
    prev.use()
    for o in (fbo, vao, prog, quad):
        o.release()
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = True
    return tex


# ------------------------------------------------------------------ helpers (tests, gallery, bench)


def hidden_context(size=(64, 64)):
    from math_layers import hidden_context as hc
    return hc(size)


def default_values(**over):
    import params
    v = params.defaults()
    v["l_coilnest"] = 1.0
    v.update(over)
    return v


def make_clock(t, bpm=124.0):
    return make_clock_beats(t * bpm / 60.0, bpm)


def make_clock_beats(b, bpm=124.0):
    return dict(bpm=bpm, beat=b, beat_phase=b % 1.0, bar_phase=(b / 4) % 1.0, phrase_phase=(b / 32) % 1.0)


def read_texture(tex):
    return np.frombuffer(tex.read(), dtype=np.uint8).reshape(tex.height, tex.width, 4)[::-1]


def render_frames(cp, values, t0=4.0, frames=1, dt=1 / 60, bands=(0.3, 0.2, 0.1), palette=None, layer=None, skins=None):
    cp.reset(t0)                                          # a still of these values (gains snap), not a fade
    tex = None
    for i in range(frames):
        t = t0 + i * dt
        tex = cp.render(values, make_clock(t), t, dt, bands, palette, layer=layer, skins=skins)
    return read_texture(tex)


def measure(cp, values, frames=40, warm=10):
    """GPU time of one frame of the pass (both passes), median of `frames` timer queries, in ms."""
    ctx = cp.ctx
    q = ctx.query(time=True)
    out = []
    for k in range(warm + frames):
        t = 4.0 + k / 60
        with q:
            cp.render(values, make_clock(t), t, 1 / 60, (0.3, 0.2, 0.1))
        ctx.finish()
        if k >= warm:
            out.append(q.elapsed / 1e6)
    return float(np.median(out))


def _label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img[..., :3])
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 8 + 7 * len(text), 16), fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def sheet(out_dir=None, cell=(640, 400), cols=3, jobs=None, name="sheet.png", t0=6.0):
    """Labelled contact sheet of looks -> out_dir/name (+ one PNG per cell)."""
    from pathlib import Path
    from PIL import Image
    out = Path(out_dir or Path(__file__).parent / "test_frames" / "coilnest")
    out.mkdir(parents=True, exist_ok=True)
    ctx, _w = hidden_context()
    cp = CoilNestPass(ctx, cell)
    jobs = jobs or [(f"stage {i}: {s}", dict(coil_prog=0, coil_stage=float(i))) for i, s in enumerate(STAGES)]
    tiles = []
    lay = None
    for label, over in jobs:
        layer = None
        if over.pop("_kene4", False):
            lay = lay or bake_channels(ctx, 256)
            layer = (lay, LAYER_KENE4)
        img = render_frames(cp, default_values(**over), t0=over.pop("_t", t0) if "_t" in over else t0, layer=layer)
        safe = "".join(c if c.isalnum() else "_" for c in label)[:60]
        Image.fromarray(img[..., :3]).save(out / f"{safe}.png")
        tiles.append(_label(img, label))
    rows = -(-len(tiles) // cols)
    S = np.zeros((rows * cell[1], cols * cell[0], 3), np.uint8)
    for k, tl in enumerate(tiles):
        r, c = divmod(k, cols)
        S[r * cell[1]:(r + 1) * cell[1], c * cell[0]:(c + 1) * cell[0]] = tl
    Image.fromarray(S).save(out / name)
    cp.release()
    return out / name


def strip(out_dir=None, cell=(480, 300), n=8, span=16.0, values=None, name="motion_strip.png", t0=6.0):
    """Motion strip: n frames `span` seconds apart (continuous playback between them) + the frame difference."""
    from pathlib import Path
    from PIL import Image
    out = Path(out_dir or Path(__file__).parent / "test_frames" / "coilnest")
    out.mkdir(parents=True, exist_ok=True)
    ctx, _w = hidden_context()
    cp = CoilNestPass(ctx, cell)
    v = default_values(**(values or {}))
    tiles, t, dt = [], t0, 1 / 30
    cp._blank = True
    steps = int(span / dt)
    for k in range(n):
        for _ in range(steps if k else 1):
            tex = cp.render(v, make_clock(t), t, dt, (0.3, 0.2, 0.1))
            t += dt
        img = read_texture(tex)
        x = cp.prog.x
        tiles.append(_label(img, f"t={t:5.1f}s stage {x:.2f} {STAGES[int(x) % 5]}"))
    cols = 4
    rows = -(-n // cols)
    S = np.zeros((rows * cell[1], cols * cell[0], 3), np.uint8)
    for k, tl in enumerate(tiles):
        r, c = divmod(k, cols)
        S[r * cell[1]:(r + 1) * cell[1], c * cell[0]:(c + 1) * cell[0]] = tl
    Image.fromarray(S).save(out / name)
    cp.release()
    return out / name


def bench(out_size=(1280, 800)):
    ctx, _w = hidden_context()
    cp = CoilNestPass(ctx, out_size)
    rows = []
    for n in (6, 10, 14):
        for res in (1.0, 0.75, 0.5):
            for st in (0, 3):
                ms = measure(cp, default_values(coil_count=n, coilnest_res=res, coil_prog=0, coil_stage=float(st)))
                rows.append((n, res, st, ms))
                print(f"N={n:2d} res={res:.2f} stage={st}: {ms:.2f} ms")
    off = cp.draws
    cp.render(default_values(l_coilnest=0.0), make_clock(1), 1.0, 1 / 60)
    print(f"off: draws +{cp.draws - off}")
    cp.release()
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--bench" in args:
        bench()
    elif "--strip" in args:
        print(strip())
    else:
        print(sheet())
