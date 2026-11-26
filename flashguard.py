"""Engine-wide photosensitive limiter ("FLASH GUARD").

One hook at the very end of the picture chain (after filters, limiter, master and blackout; before Spout, the panel
monitor, snapshots, the recorder and the mapping / surfaces pass), so everything the projector shows went through it.

Limits (WCAG 2.3.1 / ITU-R BT.1702 / Harding; research/gemini_neon_support.md section 3):
  no more than 3 general flashes and no more than 3 red flashes in any 1 s, over more than 25 % of the picture.
  A flash is a PAIR of opposing transitions. A general transition is a change of relative luminance (sRGB -> linear,
  Rec.709 weights) of at least 0.10 where the darker state is below 0.80. A red transition is a change of the
  saturated-red energy max(0, R - G - B) (linear) of at least 20 / 320, counted on pixels with R / (R + G + B) >= 0.8.

How it works
  measure   every frame one pass folds the frame into 4 x 4 pixel blocks (every pixel is counted: mean and maximum of
            luminance and red energy), a second tiny one folds the blocks into an 8 x 5 grid. The grid is read back
            through a pixel buffer TWO FRAMES LATER (never a synchronous stall), with the timestamp of the frame it
            belongs to, so a variable frame rate is exact.
  decide    (CPU, `GuardLogic`) opposing transitions per tile in a sliding 1 s window. A tile is "hot" when it is
            flashing faster than the limit allows: 4 transitions whose last three gaps add up to less than 0.48 s
            (above about 3.1 Hz), or 7 in the window. The guard engages when 20 % of the tiles are hot (margin under
            the 25 % of the standards) and lets go when fewer than 10 % are. Red uses the same rule with a lower
            threshold and a stricter rate (above about 2.5 Hz, or 6 in the window). A tile MEAN has to move 0.05
            (red: 0.03) to count: that is half of a tile changing by the standards' 0.10.
  attenuate (GPU, zero latency) while engaged every pixel may fall at once but may only RISE at a capped rate above
            the brightest pixel its own 4 x 4 block showed on the previous OUTPUT frame: 0.10 of luminance per 0.4 s
            (so a visible transition needs 0.4 s: 2.5 flashes / s at the very most, whatever comes in) and 20 / 320
            of red energy per 0.5 s. The picture never stops: a strobe turns into its own dark phase, and everything
            that is steadily lit is left alone bit for bit. The gain is at most 1: the guard never brightens a pixel.
  idle      nothing flashing above the limit: only the two measuring passes run and the frame is not written at all
            (the projector gets the very texture the engine rendered).

Override: "I know what I'm doing, the audience was warned". Armed in the panel with a 2 s hold plus a confirm, checked
here on the laptop's own clock (`arm_begin` ... at least 2 s ... `arm_confirm`). Never saved: a restart is guarded
again. While armed the guard still measures and the panel shows a red badge.

    fg = FlashGuard(ctx, (w, h))
    fg.process(disp, t)        # once per frame, in place; True while it is attenuating
    fg.state()                 # for the panel
    fg.command({"cmd": "flashguard", "op": ...})

Standalone: `python flashguard.py --bench` (cost), `python flashguard.py --demo` (before / after flash counts).
"""
import math
import time

import numpy as np

GRID = (8, 5)                       # tiles across, down
TILES = GRID[0] * GRID[1]
WINDOW = 1.0                        # s
MAX_FLASHES = 3                     # per WINDOW, general and red each
DELTA = 0.10                        # the standards' luminance change
DETECT = 0.05                       # what a TILE MEAN has to move to count here: catches 50 % of a tile changing by 0.10
DARK_MAX = 0.80                     # ... while the darker state is below this
RED_DELTA = 20.0 / 320.0            # the standards' red-energy change
RED_DETECT = 0.03                   # same idea for red energy
RED_RATIO = 0.8                     # saturated red: R / (R + G + B), linear
AREA = 0.25                         # the standards' area
AREA_ON, AREA_OFF = 0.20, 0.10      # engage / let go (fraction of hot tiles)
SLEW_L = DELTA / 0.4                # luminance may rise this fast while engaged (per second)
SLEW_S = RED_DELTA / 0.5            # red energy may rise this fast while engaged (per second)
FREE = 1.0e6                        # "not limited"
RELEASE_T, RELEASE_RATE = 0.6, 2.0  # letting go: a quick ramp back up instead of a cut
ARM_HOLD_S, ARM_EXPIRE_S = 2.0, 15.0
HIST_DIV = 4                        # block history: 1 / 4 resolution, block maximum
LAG = 2                             # frames between a measurement and its readback
OPS = ("arm_begin", "arm_confirm", "arm_cancel", "disarm")


# ------------------------------------------------------------------------------------------------ pure logic
class FlashTracker:
    """Opposing transitions of n series: an excursion of `th` away from the running extreme, in the direction
    opposite to the previous transition (the first one may go either way). `th`, `dark_max`, `span` and `max_n`
    may be per-series arrays."""

    DEPTH = 16

    def __init__(self, n, th, dark_max=None, min_n=4, span=0.48, max_n=7):
        self.n = n
        self.th = np.broadcast_to(np.asarray(th, dtype=np.float64), (n,)).copy()
        self.dark = np.broadcast_to(np.asarray(np.inf if dark_max is None else dark_max, dtype=np.float64), (n,)).copy()
        self.min_n = int(min_n)
        self.span = np.broadcast_to(np.asarray(span, dtype=np.float64), (n,)).copy()
        self.max_n = np.broadcast_to(np.asarray(max_n), (n,)).copy()
        self.ext = np.zeros(n)
        self.lo = np.full(n, np.inf)
        self.hi = np.full(n, -np.inf)
        self.up = np.zeros(n, dtype=bool)          # last transition went up / down (neither: none yet)
        self.dn = np.zeros(n, dtype=bool)
        self.times = np.full((n, self.DEPTH), -1.0e30)
        self.last = -1.0e30

    def push(self, t, v, clean=True):
        if clean:
            v = np.clip(np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        th, dark = self.th, self.dark
        if not (self.up | self.dn).all():          # series still waiting for their first transition
            none = ~(self.up | self.dn)
            np.minimum(self.lo, v, out=self.lo)
            np.maximum(self.hi, v, out=self.hi)
            up0 = none & (v - self.lo >= th) & (self.lo < dark)
            dn0 = none & (self.hi - v >= th) & (v < dark) & ~up0
        else:
            up0 = dn0 = False
        hi = np.where(self.up, np.maximum(self.ext, v), self.ext)
        ext = np.where(self.dn, np.minimum(hi, v), hi)
        up = (self.dn & (v - ext >= th) & (ext < dark)) | up0
        dn = (self.up & (ext - v >= th) & (v < dark)) | dn0
        tr = up | dn
        if tr.any():
            self.up = (self.up & ~dn) | up
            self.dn = (self.dn & ~up) | dn
            ext = np.where(tr, v, ext)
            self.times[tr, :-1] = self.times[tr, 1:]
            self.times[tr, -1] = t
            self.last = float(t)
        self.ext = ext
        return tr

    def counts(self, t):
        return (self.times > t - WINDOW).sum(axis=1)

    def hot(self, t, n=None):
        """Series flashing faster than the limit allows right now."""
        n = self.counts(t) if n is None else n
        k = self.min_n
        return ((n >= k) & (self.times[:, -1] - self.times[:, -k] < self.span)) | (n >= self.max_n)


def area_rate(counts):
    """Flashes per second that at least AREA_ON of the tiles reach (pairs of transitions)."""
    k = max(1, int(math.ceil(AREA_ON * len(counts))))
    return float(np.partition(np.asarray(counts), -k)[-k]) / 2.0


class GuardLogic:
    """Everything that needs no GL: transition counting, the engage / release decision, the override.
    One tracker holds four blocks of TILES + 1 series (the tiles, then the whole frame): luminance in, red energy
    in, luminance out, red energy out."""

    N1 = TILES + 1
    IN_L, IN_S, OUT_L, OUT_S = 0, 1, 2, 3
    METER_EVERY = 6

    def __init__(self):
        n1 = self.N1
        rep = lambda a, b: np.tile(np.repeat(np.array([a, b], dtype=np.float64), n1), 2)      # noqa: E731
        self.tr = FlashTracker(4 * n1, rep(DETECT, RED_DETECT), rep(DARK_MAX, np.inf), 4, rep(0.48, 0.60), rep(7, 6))
        self._v = np.zeros((4, n1))
        self.general = self.red = False
        self.release_until = -1.0e30
        self.armed = False
        self.arm_t0 = None
        self.events = 0                 # times the guard had to step in (or would have, while overridden)
        self.att_s = 0.0
        self.t = 0.0
        self.pushes = 0
        self._n = np.zeros(4 * n1, dtype=int)
        self._hot = (0.0, 0.0)
        self._red_frac = 0.0

    def push(self, t, tiles_in, tiles_out=None):
        """One measured frame: tiles (5, 8, >=3) = mean luminance, mean red energy, saturated-red fraction."""
        t = float(t)
        if not math.isfinite(t):
            return
        n1, v = self.N1, self._v
        a = np.asarray(tiles_in, dtype=np.float64).reshape(TILES, -1)
        b = a if tiles_out is None else np.asarray(tiles_out, dtype=np.float64).reshape(TILES, -1)
        v[0, :TILES], v[1, :TILES], v[2, :TILES], v[3, :TILES] = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        np.nan_to_num(v, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(v, 0.0, 1.0, out=v)
        v[:, TILES] = v[:, :TILES].mean(axis=1)
        tr = self.tr
        tr.push(t, v.ravel(), clean=False)
        self.t = t
        self.pushes += 1
        n = self._n = tr.counts(t)
        hot = tr.hot(t, n)
        hot_l = float(np.count_nonzero(hot[:TILES])) / TILES
        hot_s = float(np.count_nonzero(hot[n1:n1 + TILES])) / TILES
        self._hot = (hot_l, hot_s)
        was = self.general or self.red
        self.general = hot_l >= (AREA_OFF if self.general else AREA_ON)
        self.red = hot_s >= (AREA_OFF if self.red else AREA_ON)
        on = self.general or self.red
        if on and not was:
            self.events += 1
        if was and not on:
            self.release_until = t + RELEASE_T
        if self.pushes % self.METER_EVERY == 0:
            x = np.nan_to_num(a[:, 2], nan=0.0, posinf=1.0, neginf=0.0)
            self._red_frac = float(np.clip(x, 0.0, 1.0).mean())

    def tile_counts(self, which=0):
        """Transitions in the last second per tile: which = IN_L, IN_S, OUT_L or OUT_S."""
        return self._n[which * self.N1: which * self.N1 + TILES]

    @property
    def meter(self):
        n, n1 = self._n, self.N1
        rate = lambda k: area_rate(n[k * n1: k * n1 + TILES])                                 # noqa: E731
        return dict(rate_in=rate(0), rate_out=rate(2), red_in=rate(1), red_out=rate(3),
                    frame_in=float(n[TILES]) / 2.0, area=max(self._hot), red_frac=self._red_frac)

    def engaged(self):
        return self.general or self.red

    def slew(self, t):
        """None = leave the frame alone; else (luminance, red energy) rise limits per second."""
        if self.armed:
            return None
        if self.general or self.red:
            return (SLEW_L if self.general else FREE, SLEW_S if self.red else FREE)
        if t < self.release_until:
            return (RELEASE_RATE, RELEASE_RATE)
        return None

    # -- override ---------------------------------------------------------------------------------------
    def command(self, c, wall=None):
        """arm_begin -> (2 s hold, timed here) -> arm_confirm; arm_cancel; disarm. Returns a short status word."""
        wall = time.monotonic() if wall is None else wall
        op = c.get("op") if isinstance(c, dict) else None
        if op == "arm_begin":
            if not self.armed:
                self.arm_t0 = wall
            return "pending"
        if op == "arm_cancel":
            self.arm_t0 = None
            return "cancelled"
        if op == "arm_confirm":
            t0, self.arm_t0 = self.arm_t0, None
            if t0 is not None and ARM_HOLD_S <= wall - t0 <= ARM_EXPIRE_S:
                self.armed = True
                return "armed"
            return "refused"
        if op == "disarm":
            self.armed, self.arm_t0 = False, None
            return "disarmed"
        return "ignored"

    def mode(self, t):
        if self.armed:
            return "override"
        if self.engaged():
            return "attenuating"
        if t < self.release_until:
            return "releasing"
        return "idle"

    def state(self, t=None, wall=None):
        t = self.t if t is None else t
        wall = time.monotonic() if wall is None else wall
        if self.arm_t0 is not None and wall - self.arm_t0 > ARM_EXPIRE_S:
            self.arm_t0 = None
        return dict(on=True, mode=self.mode(t), armed=self.armed, pending=self.arm_t0 is not None,
                    would=bool(self.armed and self.engaged()), red=bool(self.red), events=int(self.events),
                    att_s=round(self.att_s, 1), limit=MAX_FLASHES, hold_s=ARM_HOLD_S,
                    **{k: round(float(v), 2) for k, v in self.meter.items()})


def validate_command(c):
    """Server thread: normalise {"cmd": "flashguard", "op": ...}; ValueError for anything else."""
    if not isinstance(c, dict) or c.get("op") not in OPS:
        raise ValueError("flashguard: op must be one of " + ", ".join(OPS))
    return {"cmd": "flashguard", "op": c["op"]}


# ------------------------------------------------------------------------------------------------ GL
_VS = """
#version 330
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_COMMON = """
const vec3 LW = vec3(0.2126, 0.7152, 0.0722);
vec3 clean(vec3 c){                       // NaN -> 0, then what the display would show
  c = vec3(isnan(c.r) ? 0.0 : c.r, isnan(c.g) ? 0.0 : c.g, isnan(c.b) ? 0.0 : c.b);
  return clamp(c, 0.0, 1.0);
}
vec3 toLin(vec3 c){ return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
vec3 toSrgb(vec3 l){ return mix(l * 12.92, 1.055 * pow(l, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, l)); }
float redE(vec3 l){ return max(0.0, l.r - l.g - l.b); }
"""

_BLOCK_FS = """
#version 330
uniform sampler2D src;
out vec4 f;
""" + _COMMON + """
void main(){                              // every pixel of a %(div)d x %(div)d block: mean and maximum of luminance and red energy
  ivec2 b = ivec2(gl_FragCoord.xy) * %(div)d, top = textureSize(src, 0) - 1;
  float L = 0.0, S = 0.0, mL = 0.0, mS = 0.0;
  for (int j = 0; j < %(div)d; j++) for (int i = 0; i < %(div)d; i++) {
    vec3 l = toLin(clean(texelFetch(src, min(b + ivec2(i, j), top), 0).rgb));
    float y = dot(l, LW);
    float s = redE(l) * step(%(ratio)s * (l.r + l.g + l.b), l.r);
    L += y; S += s;
    mL = max(mL, y); mS = max(mS, redE(l));
  }
  f = vec4(L / %(n)d.0, S / %(n)d.0, mL, mS);
}
""" % dict(div=HIST_DIV, n=HIST_DIV * HIST_DIV, ratio=RED_RATIO)

_TILE_FS = """
#version 330
uniform sampler2D src;                    // the block texture
uniform ivec2 grid;
out vec4 f;
void main(){
  ivec2 size = textureSize(src, 0), cell = ivec2(gl_FragCoord.xy);
  ivec2 a = cell * size / grid, b = (cell + 1) * size / grid;
  float L = 0.0, S = 0.0, F = 0.0;
  for (int j = a.y; j < b.y; j++) for (int i = a.x; i < b.x; i++) {
    vec2 v = texelFetch(src, ivec2(i, j), 0).rg;
    L += v.x; S += v.y; F += step(0.01, v.y);
  }
  f = vec4(L, S, F, 1.0) / float(max(1, (b.x - a.x) * (b.y - a.y)));
}
"""

_APPLY_FS = """
#version 330
uniform sampler2D src;
uniform sampler2D hist;                   // block texture of the previous OUTPUT frame
uniform vec2 rise;                        // allowed rise this frame: luminance, red energy
out vec4 f;
""" + _COMMON + """
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 c = texelFetch(src, p, 0);
  vec3 l = toLin(clean(c.rgb));
  float L = dot(l, LW), S = redE(l);
  vec2 cap = max(texelFetch(hist, p / %(div)d, 0).ba, vec2(0.0)) + rise;
  float g = 1.0;
  if (L > cap.x) g = cap.x / L;
  if (S > cap.y) g = min(g, cap.y / S);
  f = g < 1.0 ? vec4(toSrgb(l * g), c.a) : c;     // untouched pixels pass through bit for bit
}
""" % dict(div=HIST_DIV)

UNIT_SRC, UNIT_HIST = 24, 25              # private texture units (0-16 belong to the scene and its passes)


class FlashGuard:
    def __init__(self, ctx, size, lag=LAG):
        import moderngl
        self._mgl = moderngl
        self.ctx, self.size, self.lag = ctx, (int(size[0]), int(size[1])), max(1, int(lag))
        self.logic = GuardLogic()
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"))
        self._quad = quad
        mk = lambda fs: ctx.program(vertex_shader=_VS, fragment_shader=fs)           # noqa: E731
        self.p_tile, self.p_apply, self.p_block = mk(_TILE_FS), mk(_APPLY_FS), mk(_BLOCK_FS)
        self.v_tile, self.v_apply, self.v_block = (ctx.vertex_array(p, [(quad, "2f", "in_pos")])
                                                   for p in (self.p_tile, self.p_apply, self.p_block))
        self.p_tile["grid"].value = GRID
        self.t_tile = [ctx.texture(GRID, 4, dtype="f4") for _ in range(2)]            # input, output
        self.f_tile = [ctx.framebuffer([t]) for t in self.t_tile]
        w, h = self.size
        bsize = ((w + HIST_DIV - 1) // HIST_DIV, (h + HIST_DIV - 1) // HIST_DIV)
        self.t_blk = [ctx.texture(bsize, 4, dtype="f2") for _ in range(2)]            # this frame / the previous output
        for t in self.t_blk:
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.f_blk = [ctx.framebuffer([t]) for t in self.t_blk]
        self.k = 0
        self.t_tmp = self.f_tmp = None                                              # full-size scratch, on first use
        self._dst = {}                                                              # texture glo -> framebuffer on it
        nbytes = TILES * 4 * 4
        self._ring = [dict(frame=None, t=0.0, out=False, a=ctx.buffer(reserve=nbytes), b=ctx.buffer(reserve=nbytes))
                      for _ in range(self.lag + 1)]
        self.frame = 0
        self.t_prev = None
        self.attenuating = False
        self.passes = 0                     # frames it rewrote so far (tests: 0 on calm content)
        self.peak = dict(rate_in=0.0, rate_out=0.0, red_in=0.0, red_out=0.0)

    @property
    def t_hist(self):                       # block texture of the last output frame (after process())
        return self.t_blk[1 - self.k]

    # -- internals ----------------------------------------------------------------------------------------
    def _collect(self):
        s = self._ring[(self.frame - self.lag) % len(self._ring)]
        if s["frame"] != self.frame - self.lag:
            return
        s["frame"] = None
        shape = (GRID[1], GRID[0], 4)
        tin = np.frombuffer(s["a"].read(), dtype="f4").reshape(shape)
        tout = np.frombuffer(s["b"].read(), dtype="f4").reshape(shape) if s["out"] else None
        self.logic.push(s["t"], tin, tout)
        if self.frame % 6 == 0:
            m = self.logic.meter
            for k in self.peak:
                self.peak[k] = max(self.peak[k], m[k])

    def _measure(self, tex, which, buf):
        """Fold the frame into 4 x 4 blocks (every pixel counted), then the blocks into the 8 x 5 grid."""
        self.f_blk[self.k].use()
        tex.use(UNIT_SRC)
        self.p_block["src"].value = UNIT_SRC
        self.v_block.render(self._mgl.TRIANGLE_STRIP)
        fbo = self.f_tile[which]
        fbo.use()
        self.t_blk[self.k].use(UNIT_SRC)
        self.p_tile["src"].value = UNIT_SRC
        self.v_tile.render(self._mgl.TRIANGLE_STRIP)
        fbo.read_into(buf, components=4, dtype="f4")          # into a pixel buffer: collected `lag` frames later

    def _target(self, tex):
        fbo = self._dst.get(tex.glo)
        if fbo is None:
            if len(self._dst) > 4:
                for old in self._dst.values():
                    old.release()
                self._dst.clear()
            fbo = self._dst[tex.glo] = self.ctx.framebuffer([tex])
        return fbo

    def _attenuate(self, tex, rise):
        if self.t_tmp is None:
            self.t_tmp = self.ctx.texture(tex.size, 4, dtype="f2")
            self.t_tmp.filter = (self._mgl.NEAREST, self._mgl.NEAREST)
            self.f_tmp = self.ctx.framebuffer([self.t_tmp])
        self.f_tmp.use()
        tex.use(UNIT_SRC)
        self.t_blk[1 - self.k].use(UNIT_HIST)
        self.p_apply["src"].value = UNIT_SRC
        self.p_apply["hist"].value = UNIT_HIST
        self.p_apply["rise"].value = rise
        self.v_apply.render(self._mgl.TRIANGLE_STRIP)
        self.ctx.copy_framebuffer(self._target(tex), self.f_tmp)      # back in place: everything downstream sees it
        self.passes += 1

    # -- the hook -----------------------------------------------------------------------------------------
    def process(self, tex, t):
        """Guard `tex` (the finished frame) in place. `t` = this frame's time in seconds. True while attenuating."""
        ctx = self.ctx
        if tuple(tex.size) != self.size:
            return False
        t = float(t)
        if not math.isfinite(t):
            t = (self.t_prev or 0.0) + 1.0 / 60.0
        dt = 1.0 / 60.0 if self.t_prev is None else min(0.1, max(1e-3, t - self.t_prev))
        self.t_prev = t
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        try:
            self._collect()
            lg = self.logic
            slew = lg.slew(t)
            slot = self._ring[self.frame % len(self._ring)]
            slot.update(frame=self.frame, t=t, out=False)
            self._measure(tex, 0, slot["a"])                  # what came in
            self.attenuating = False
            if slew is not None and self.frame > 0:
                self._attenuate(tex, (min(slew[0] * dt, FREE), min(slew[1] * dt, FREE)))
                self.attenuating = True
                if lg.engaged():
                    lg.att_s += dt
                self._measure(tex, 1, slot["b"])              # what goes out: the meters, and next frame's ceiling
                slot["out"] = True
            self.k ^= 1                                       # this frame's blocks are the next frame's history
        finally:
            self.frame += 1
            try:                              # hand the caller's framebuffer and viewport back
                if prev_fbo is not None and hasattr(prev_fbo.mglo, "use"):
                    prev_fbo.use()
                    ctx.viewport = prev_vp
                else:
                    ctx.screen.use()
            except Exception:
                pass
        return self.attenuating

    def command(self, c):
        return self.logic.command(c)

    def state(self):
        return self.logic.state(self.t_prev)

    def summary(self):
        """Worst flash rates seen since the start (flashes / s over the area): what came in, what went out."""
        return dict(events=self.logic.events, att_s=round(self.logic.att_s, 2), frames=self.frame, passes=self.passes,
                    **{k: round(v, 1) for k, v in self.peak.items()})

    def release(self):
        objs = [self.t_tmp, self.f_tmp, self._quad, *self.t_tile, *self.f_tile, *self.t_blk, *self.f_blk,
                *self._dst.values(), *(s[k] for s in self._ring for k in "ab"),
                self.v_tile, self.v_apply, self.v_block, self.p_tile, self.p_apply, self.p_block]
        for o in objs:
            try:
                if o is not None:
                    o.release()
            except Exception:
                pass
        self._dst.clear()

    close = release


# ------------------------------------------------------------------------------------------------ reference counter
def srgb_to_linear(c):
    c = np.clip(np.nan_to_num(np.asarray(c, dtype=np.float64)), 0.0, 1.0)
    return np.where(c < 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def tile_means(frame):
    """(h, w, 3) sRGB frame -> (TILES, 2): exact mean luminance and red energy per tile (numpy; tests and --demo)."""
    lin = srgb_to_linear(frame[..., :3])
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    tot = lin.sum(axis=-1)
    red = np.where((lin[..., 0] >= RED_RATIO * tot) & (lin[..., 0] > 1e-4), np.maximum(0.0, 2 * lin[..., 0] - tot), 0.0)
    h, w = lum.shape
    ys = np.linspace(0, h, GRID[1] + 1).astype(int)
    xs = np.linspace(0, w, GRID[0] + 1).astype(int)
    out = np.zeros((GRID[1], GRID[0], 2))
    for j in range(GRID[1]):
        for i in range(GRID[0]):
            out[j, i] = (lum[ys[j]:ys[j + 1], xs[i]:xs[i + 1]].mean(), red[ys[j]:ys[j + 1], xs[i]:xs[i + 1]].mean())
    return out.reshape(TILES, 2)


def count_flashes(times, series, th, dark_max=None):
    """The standards' count for a recorded sequence: series (frames, tiles) -> the largest number of flashes
    (pairs of opposing transitions of at least `th`) in any 1 s that MORE than 25 % of the tiles reach."""
    series = np.asarray(series, dtype=np.float64)
    tr = FlashTracker(series.shape[1], th, dark_max)
    stamps = [[] for _ in range(series.shape[1])]
    for t, row in zip(times, series):
        for i in np.nonzero(tr.push(t, row))[0]:
            stamps[i].append(t)
    worst = []
    for ts in stamps:
        ts = np.asarray(ts)
        worst.append(max((int(((ts >= a) & (ts < a + WINDOW)).sum()) for a in ts), default=0) // 2)
    k = int(math.floor(AREA * len(worst))) + 1
    return int(sorted(worst)[-k])


# ------------------------------------------------------------------------------------------------ standalone
def _context():
    import moderngl
    try:
        return moderngl.create_standalone_context(require=330)
    except Exception:
        import fx_stack
        return fx_stack.hidden_context()[0]


def _strobe_frame(w, h, on, rgb=(1.0, 1.0, 1.0)):
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 4), dtype="f4")
    img[..., :3] = 0.08 + 0.10 * (((x // 32) + (y // 32)) % 2)[..., None]
    if on:
        img[..., :3] = np.clip(img[..., :3] + 0.8 * np.array(rgb), 0, 1)
    img[..., 3] = 1.0
    return img


def run_sequence(ctx, size, frames, times, guard=True):
    """Feed frames (h, w, 4 float) through a fresh guard; returns (outputs (h, w, 3) list, the guard)."""
    w, h = size
    tex = ctx.texture(size, 4, dtype="f2")
    fbo = ctx.framebuffer([tex])
    fg = FlashGuard(ctx, size)
    outs = []
    for img, t in zip(frames, times):
        tex.write(np.ascontiguousarray(img, dtype="f4").astype("f2").tobytes())
        if guard:
            fg.process(tex, t)
        outs.append(np.frombuffer(fbo.read(components=3, dtype="f4"), dtype="f4").reshape(h, w, 3).copy())
    return outs, fg


def _demo(ctx):
    size, fps, secs = (320, 200), 60.0, 4.0
    times = [i / fps for i in range(int(fps * secs))]
    rows = []
    for name, hz, rgb in (("strobe 1/16 @ 175 BPM (11.7 Hz)", 175 / 60 * 4, (1, 1, 1)), ("strobe 1/8 @ 175 BPM (5.8 Hz)", 175 / 60 * 2, (1, 1, 1)),
                          ("red flash 4 Hz", 4.0, (1, 0, 0)), ("blackout toggled 6 Hz", 6.0, (1, 1, 1)), ("pulse 2 Hz (legal)", 2.0, (1, 1, 1))):
        frames = [_strobe_frame(*size, (t * hz) % 1.0 < 0.5, rgb) for t in times]
        outs, fg = run_sequence(ctx, size, frames, times)
        a = np.array([tile_means(f[..., :3]) for f in frames]); b = np.array([tile_means(f) for f in outs])
        rows.append((name, count_flashes(times, a[..., 0], DELTA, DARK_MAX), count_flashes(times, b[..., 0], DELTA, DARK_MAX),
                     count_flashes(times, a[..., 1], RED_DELTA), count_flashes(times, b[..., 1], RED_DELTA), fg.logic.events))
        fg.release()
    print(f"{'sequence':34s} general in -> out   red in -> out   (worst flashes in any 1 s over > 25 % of the frame)")
    for name, gi, go, ri, ro, ev in rows:
        print(f"{name:34s} {gi:6d} -> {go:<6d}     {ri:4d} -> {ro:<4d}")


def _bench(ctx, size=(1280, 800), n=600):
    w, h = size
    tex = ctx.texture(size, 4, dtype="f2")
    y, x = np.mgrid[0:h, 0:w]
    calm = np.zeros((h, w, 4), dtype="f4")
    calm[..., 0] = 0.2 + 0.5 * x / w; calm[..., 1] = 0.2 + 0.5 * y / h; calm[..., 2] = 0.3; calm[..., 3] = 1
    calm_b = calm.astype("f2").tobytes()
    bright = np.clip(calm + 0.4, 0, 1).astype("f2").tobytes()
    two = [ctx.texture(size, 4, d, dtype="f2") for d in (calm_b, bright)]
    src = [ctx.framebuffer([t]) for t in two]
    dst = ctx.framebuffer([tex])

    def run(label, flashing):
        fg = FlashGuard(ctx, size)
        for rep in (0, 1):                   # the first round warms up
            ctx.finish()
            t0 = time.perf_counter()
            for i in range(n):
                k = (i // 3) % 2 if flashing else 0
                if flashing:
                    ctx.copy_framebuffer(dst, src[k])
                fg.process(tex, (rep * n + i) / 60.0)
            ctx.finish()
            el = (time.perf_counter() - t0) / n * 1000.0
        base = 0.0
        if flashing:                         # take the test's own frame copy out of the number
            ctx.finish(); t0 = time.perf_counter()
            for i in range(n):
                ctx.copy_framebuffer(dst, src[(i // 3) % 2])
            ctx.finish(); base = (time.perf_counter() - t0) / n * 1000.0
        print(f"{label:38s} {el - base:6.3f} ms / frame   (mode {fg.state()['mode']}, frames rewritten {fg.passes})")
        fg.release()

    tex.write(calm_b)
    print(f"[bench] {ctx.info['GL_RENDERER']}, {w} x {h}, {n} frames, wall time with glFinish at both ends")
    run("idle (calm content: measuring only)", False)
    run("attenuating (10 Hz full-frame strobe)", True)


if __name__ == "__main__":
    import sys
    c = _context()
    if "--bench" in sys.argv:
        _bench(c)
    else:
        _demo(c)
