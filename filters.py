"""Post-FX filter chain: an ordered list of up to six screen-space filters on the finished scene picture.

Where it sits (live_visuals.py, block marked `[filters>` ... `<filters]`): after the scene pass, BEFORE the brightness
limiter, master and blackout, and before mapping, so every surface, Spout, the recorder and the panel monitor get the
filtered picture. While any filter is on, the engine draws the scene with limiter 0 / master 1 / blackout 0 and this
pass applies those three at its end (`FINAL_FS`, same maths as SCENE_FS); it also writes the post-filter pre-limiter
level that the brightness meter reads (`FilterPass.level`). With every filter off (the default) the engine never
calls it: no pass, no allocation, pixel-identical show.

Contract (params.py, group "filters", all MIDI-learnable and mod-matrix modulatable, saved in scenes + live_state):
  flt_slot1..6   stepped, 0 = empty, i = params.FILTER_TYPES[i-1]   (the chain ORDER)
  flt_<type>     0..1 amount of that filter (0 = not run, even when it is in a slot)
  flt_<...>      2-3 params per type (see params.py)
A type that sits in two slots runs once, at its first slot. The panel's FILTERS tab edits exactly these params.

Filters: dream (realtime DeepDream-style, half res + feedback), oil (anisotropic Kuwahara, half res), halftone,
neon (Sobel -> glow), chroma (spectral prism split), posterize (antialiased, optional show palette), sort (span pixel
sort, half res), glitch (block displacement; photosensitive-safe GlitchClock), bloom (Karis-weighted pyramid),
kaleido (mirror / quad / kaleidoscope with spin).

Standalone: python filters.py --gallery [IN.png ...] [--out DIR]   -> stills per filter + contact sheet + ms table
"""
import math

from params import FILTER_TYPES, PARAMS

N_SLOTS = 6
SLOTS = tuple(f"flt_slot{i}" for i in range(1, N_SLOTS + 1))
AMOUNTS = {t: f"flt_{t}" for t in FILTER_TYPES}
ALL_PARAMS = tuple(n for n, v in PARAMS.items() if v[3] == "filters")
HALF_RES = ("dream", "oil", "sort")       # run at half resolution, composited at full resolution
TAU = 2.0 * math.pi
T_WRAP = 1600.0                            # shader time wraps here (all its uses are periodic or noise offsets)
KAL_RATE = 1.2                             # rad/s of kaleido spin at flt_kal_spin = 1
GLITCH_MIN_HZ, GLITCH_MAX_HZ = 0.2, 2.5    # event rate range; 2.5 Hz < the 3 flashes/s limit (WCAG 2.3.1)
GLITCH_RAMP_S = 0.06                       # every glitch fades in / out over at least this long (no hard cuts)


def _f(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def type_of(slot_value):
    """Slot value (1..10) -> filter type id, None for empty / out of range."""
    i = int(round(_f(slot_value)))
    return FILTER_TYPES[i - 1] if 1 <= i <= len(FILTER_TYPES) else None


def chain(vals):
    """[(type, amount)] in slot order: empty slots, amounts <= 0.001 and repeated types are skipped."""
    out, seen = [], set()
    for s in SLOTS:
        ft = type_of(vals.get(s, 0.0))
        if ft is None or ft in seen:
            continue
        seen.add(ft)
        amt = min(_f(vals.get(AMOUNTS[ft], 0.0)), 1.0)
        if amt > 0.001:
            out.append((ft, amt))
    return out


def active(vals):
    return bool(chain(vals))


class GlitchClock:
    """When glitches happen. step(dt, rate, amount) -> (envelope 0..1, seed).

    Photosensitive rules, by construction: at most GLITCH_MAX_HZ events per second (onsets are always at least
    1 / GLITCH_MAX_HZ apart, measured on the clock this is fed with); each event ramps in and out over
    GLITCH_RAMP_S (no single-frame cuts) and is over before the next may start; the block pattern (seed) only changes
    while nothing is displaced. dt 0 (freeze) holds everything."""

    def __init__(self, seed=0):
        self.seed = int(seed) & 0xFFFF
        self.env = 0.0
        self.t_on = None                # time since onset while an event runs, else None
        self.dur = 0.0
        self.wait = 0.5                 # time until the next event may start
        self._r = (self.seed * 2654435761 + 12345) & 0xFFFFFFFF

    def _rand(self):
        self._r = (self._r * 1664525 + 1013904223) & 0xFFFFFFFF
        return self._r / 4294967296.0

    def step(self, dt, rate, amount):
        dt = max(0.0, _f(dt))
        if dt == 0.0:
            return self.env, self.seed
        hz = GLITCH_MIN_HZ + (GLITCH_MAX_HZ - GLITCH_MIN_HZ) * min(max(_f(rate), 0.0), 1.0)
        self.wait -= dt
        if self.t_on is None:
            if self.wait <= 0.0 and _f(amount) > 0.001 and self.env == 0.0:
                self.seed = (self.seed + 1 + int(self._rand() * 997)) & 0xFFFF     # new pattern while invisible
                self.dur = max(2.0 * GLITCH_RAMP_S + 0.04, 0.5 / hz)            # over well before the next onset
                self.t_on = 0.0
                self.wait = (1.0 / hz) * (1.0 + 0.8 * self._rand())            # never sooner than 1 / hz
        if self.t_on is not None:
            self.t_on += dt
            up = self.t_on / GLITCH_RAMP_S
            down = (self.dur - self.t_on) / GLITCH_RAMP_S
            target = min(1.0, up, down)
            if target <= 0.0:
                self.t_on, target = None, 0.0
            lim = dt / GLITCH_RAMP_S                                           # ramp limit per step
            self.env = min(max(target, self.env - lim), self.env + lim)
            self.env = min(max(self.env, 0.0), 1.0)
            if self.t_on is None:
                self.env = 0.0 if self.env <= lim else self.env
        else:
            self.env = max(0.0, self.env - dt / GLITCH_RAMP_S)
        return self.env, self.seed


class ChainState:
    """CPU side of the pass: wrapped time, kaleido spin phase, glitch clock. Freeze stops all of it."""

    def __init__(self):
        self.t = 0.0
        self.kal_phase = 0.0
        self.glitch = GlitchClock(seed=11)
        self.glitch_env, self.glitch_seed = 0.0, 0

    def advance(self, dt, vals):
        fz = min(max(_f(vals.get("freeze", 0.0)), 0.0), 1.0)
        d = max(0.0, _f(dt)) * (1.0 - fz)
        self.t = (self.t + d) % T_WRAP
        self.kal_phase = (self.kal_phase + d * _f(vals.get("flt_kal_spin", 0.0)) * KAL_RATE) % TAU
        self.glitch_env, self.glitch_seed = self.glitch.step(d, vals.get("flt_glitch_rate", 0.3),
                                                             vals.get("flt_glitch", 0.0))
        return d


# =====================================================================================================================
# GLSL
VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos,0,1); }
"""

COMMON = """
#version 330
in vec2 v_uv;
const float TAU = 6.28318530718, PI = 3.14159265359;
float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
float max3(vec3 c){ return max(c.r, max(c.g, c.b)); }
vec3 safe(vec3 c){ return (c.r >= 0.0 && c.g >= 0.0 && c.b >= 0.0 && c.r < 64.0 && c.g < 64.0 && c.b < 64.0) ? c : vec3(0.0); }
vec3 tap(sampler2D s, vec2 uv){ return safe(texture(s, uv).rgb); }
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y)*p3.z); }
vec2 hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*vec3(0.1031, 0.1030, 0.0973)); p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz)*p3.zy); }
vec3 hueRot(vec3 c, float a){ const vec3 k = vec3(0.57735); float ca = cos(a);
  return c*ca + cross(k, c)*sin(a) + k*dot(k, c)*(1.0 - ca); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0 - 2.0*f);
  return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y); }
mat2 rot(float a){ float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
"""

FINAL_FS = COMMON + """
uniform sampler2D src; uniform float lim, master, blackout;
layout(location=0) out vec4 f;
layout(location=1) out vec4 lvl;
void main(){
  vec3 o = safe(texelFetch(src, ivec2(gl_FragCoord.xy), 0).rgb);
  o /= max(1.0, max3(o));                                     // hue-preserving clip: no white-out
  lvl = vec4(0.0, 0.0, 0.0, max3(o));                         // post-filter, pre-limiter level for the meter
  if (lim > 0.0) { float kn = 0.34*lim; o = max(o - kn, 0.0)/(1.0 - kn)*(1.0 - 0.5*lim); }   // = SCENE_FS
  o *= clamp(master, 0.0, 1.0)*(1.0 - clamp(blackout, 0.0, 1.0));
  f = vec4(clamp(o, 0.0, 1.0), 1.0);
}
"""

# ---- realtime DeepDream-style: a half-res state that feeds back on itself. Each frame the previous dream is
# advected along the picture's contours + curl noise and zoomed in a hair (the recursive dive), its band-pass detail is
# amplified (what gradient ascent does to a picture's features, done as an unsharp feedback that grows its own
# texture), it is pulled back toward the live input so it never drifts away, and eyes / fur / scales are grown where
# the picture has light. Everything is gated by the input's light, so black stays black.
DREAM_FS = COMMON + """
uniform sampler2D src, prev;
uniform vec2 res; uniform float asp, scl, swirl, motif, t, fr, amt;
out vec4 f;
vec2 PX;
vec3 S(vec2 uv){ return tap(src, uv); }
float energyAt(vec2 uv){ return max3(S(uv)); }

vec4 eyeAt(vec2 q, float rnd, vec3 tint, float aa){
  float s = 0.8 + 0.45*fract(rnd*7.13);
  q /= s; aa /= s;
  float w = 0.36, hh = 0.17 + 0.05*fract(rnd*3.7);
  float xx = clamp(q.x/w, -1.0, 1.0);
  float lid = abs(q.y) - hh*(1.0 - xx*xx);
  float inside = (1.0 - smoothstep(-aa, aa, lid))*(1.0 - smoothstep(w - aa, w + aa, abs(q.x)));
  float r = length(q), ir = hh*0.92;
  float iris = 1.0 - smoothstep(ir - aa, ir + aa, r);
  float pup = 1.0 - smoothstep(ir*0.45 - aa, ir*0.45 + aa, r);
  float stri = 0.6 + 0.4*sin(atan(q.y, q.x)*19.0 + rnd*40.0);
  float line = (1.0 - smoothstep(0.0, 2.5*aa, abs(lid)))*(1.0 - smoothstep(w - aa, w + 2.0*aa, abs(q.x)));
  float glint = 1.0 - smoothstep(0.0, ir*0.25, length(q - vec2(-ir*0.38, ir*0.36)));
  vec3 irisC = normalize(hueRot(tint, 2.2 + 1.6*rnd) + 0.05)*1.25;
  vec3 sclera = mix(vec3(0.9, 0.85, 0.75), tint, 0.45)*0.75;
  vec3 col = sclera*(1.0 - 0.55*smoothstep(0.0, 1.0, abs(q.x)/w));
  col = mix(col, irisC*stri*(0.45 + 0.9*r/ir), iris);
  col = mix(col, vec3(0.0), pup);
  col += vec3(0.9)*glint*iris;
  col = mix(col, tint*0.12, line*0.85);
  return vec4(col, clamp(max(inside, line), 0.0, 1.0));
}

// eyes on a jittered lattice of `cells` per frame height; returns colour + coverage of the nearest eye
vec4 eyes(vec2 p, float cells, float dens, vec3 tint, float seed){
  vec2 g = p*cells, gi = floor(g), gf = fract(g);
  vec4 best = vec4(0.0); float bd = 9.0;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    vec2 o = vec2(i, j), id = gi + o;
    vec2 h = hash22(id + seed);
    vec2 c = o + 0.5 + (h - 0.5)*0.5 - gf;               // offset from this pixel to the cell's point (cell units)
    float d = dot(c, c);
    if (d < bd) { bd = d; best = vec4(-c, id); }
  }
  vec2 id = best.zw, q = best.xy;
  float rnd = hash12(id*1.37 + seed + 3.1);
  if (rnd > dens) return vec4(0.0);
  vec2 cuv = (id + 0.5)/cells; cuv.x /= asp;              // the cell's point in uv: is there light to grow from?
  float e = smoothstep(0.06, 0.32, energyAt(cuv));
  if (e <= 0.0) return vec4(0.0);
  float ang = (hash12(id + seed + 9.7) - 0.5)*0.9 + 0.25*sin(t*0.4 + rnd*TAU);
  q = rot(ang)*q;
  float aa = 1.2*cells/res.y;
  vec4 ey = eyeAt(q, rnd, tint, aa);
  return vec4(ey.rgb, ey.a*e);
}

void main(){
  PX = 1.0/res;
  vec2 uv = v_uv;
  vec3 c0 = S(uv);
  float lx = luma(S(uv + vec2(1.5*PX.x, 0.0))) - luma(S(uv - vec2(1.5*PX.x, 0.0)));
  float ly = luma(S(uv + vec2(0.0, 1.5*PX.y))) - luma(S(uv - vec2(0.0, 1.5*PX.y)));
  vec2 gr = vec2(lx, ly); float gm = length(gr);
  vec2 n = gm > 1e-4 ? gr/gm : vec2(0.0, 1.0), tg = vec2(-n.y, n.x);
  // light in the neighbourhood: the dream may only live (and spread a little) where the picture has light
  float ew = max(max(max3(c0), max3(S(uv + vec2(5.0*PX.x, 0.0)))), max(max3(S(uv - vec2(5.0*PX.x, 0.0))),
            max(max3(S(uv + vec2(0.0, 5.0*PX.y))), max3(S(uv - vec2(0.0, 5.0*PX.y))))));
  float gate = smoothstep(0.012, 0.10, ew);

  // advection: along the contours (fur / swirls) + curl noise, plus a slow zoom in (the recursive dive)
  vec2 pa = (uv - 0.5)*vec2(asp, 1.0);
  vec2 np = pa*2.3 + vec2(t*0.031, -t*0.023);
  float e1 = 0.05, n0 = vnoise(np), nx = vnoise(np + vec2(e1, 0.0)), ny = vnoise(np + vec2(0.0, e1));
  vec2 curl = vec2(ny - n0, -(nx - n0))/e1;
  vec2 flow = (tg*smoothstep(0.0, 0.12, gm)*1.4 + curl*0.18)*swirl*fr*PX*1.6;
  float zoom = 1.0 - 0.0016*swirl*fr;
  vec2 q = (uv - 0.5)*zoom + 0.5 - flow;
  vec3 fb = tap(prev, q);
  float r = 1.6 + 2.2*scl;                                  // band-pass radius (half-res px): feature size
  vec3 bl = (tap(prev, q + vec2(r*PX.x, 0.0)) + tap(prev, q - vec2(r*PX.x, 0.0)) +
             tap(prev, q + vec2(0.0, r*PX.y)) + tap(prev, q - vec2(0.0, r*PX.y)))*0.25;
  vec3 det = fb - bl;
  float lumd = luma(det);
  vec3 amp = fb + det*1.35 + hueRot(det, 2.1)*0.45*abs(lumd)*6.0;   // amplified features + complementary fringes

  // pull back toward the live picture (persistence per 60 Hz frame)
  float keep = pow(mix(0.62, 0.955, swirl), fr);
  vec3 base = mix(c0*1.05, amp, keep);
  base = max(mix(vec3(luma(base)), base, 1.28), 0.0);       // DeepDream's oversaturated look

  // motifs: m 0 = eyes, 0.5 = fur / dog-slug segments, 1 = scales / arches; eyes never fully vanish
  float wE = clamp(1.0 - 2.0*motif, 0.0, 1.0)*0.75 + 0.25;
  float wF = 1.0 - abs(2.0*motif - 1.0);
  float wS = clamp(2.0*motif - 1.0, 0.0, 1.0);
  vec3 tint = max3(c0) > 0.02 ? c0/max3(c0) : vec3(0.7, 0.5, 0.9);
  float cells = mix(16.0, 5.0, scl);
  vec2 pf = pa;
  float el = smoothstep(0.03, 0.3, max3(c0) + gm*2.0);
  vec3 add = vec3(0.0);
  if (wF > 0.01) {                                           // fur / dog-slug: ridges along the contours, beaded
    vec2 pw = pf*cells*2.2 + vec2(vnoise(pf*3.0 + t*0.05), vnoise(pf*3.0 + 7.0 - t*0.04))*2.0;
    float ph = dot(pw, n)*3.2 + 2.5*vnoise(pw*0.35);
    float rid = pow(0.5 + 0.5*sin(ph), 3.0);
    float seg = 0.55 + 0.45*sin(dot(pw, tg)*1.6 + 1.7*vnoise(pw*0.5 + 3.0));
    add += hueRot(tint, 0.9*sin(ph*0.5))*rid*seg*wF*0.55*el;
  }
  if (wS > 0.01) {                                           // scales / arches: voronoi rims lit from above
    vec2 g = pf*cells*1.7, gi = floor(g), gf = fract(g);
    float d1 = 9.0, d2 = 9.0; vec2 cq = vec2(0.0);
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
      vec2 o = vec2(i, j), h = hash22(gi + o + 17.0), c = o + 0.5 + (h - 0.5)*0.7 - gf;
      float d = dot(c, c);
      if (d < d1) { d2 = d1; d1 = d; cq = c; } else if (d < d2) d2 = d;
    }
    float edge = sqrt(d2) - sqrt(d1);
    float rim = 1.0 - smoothstep(0.02, 0.14, edge);
    float arch = smoothstep(-0.5, 0.45, cq.y)*smoothstep(0.0, 0.25, edge);
    add += (hueRot(tint, 1.3)*rim*0.7 + tint*arch*0.35)*wS*el;
  }
  vec3 o = base + add;
  vec4 ey = eyes(pf, cells, 0.18 + 0.45*wE, tint, 0.0);      // two scales of eyes: self-similar
  vec4 ey2 = eyes(pf, cells*2.35, 0.10 + 0.30*wE, tint, 41.0);
  o = mix(o, ey2.rgb, ey2.a*0.85);
  o = mix(o, ey.rgb, ey.a);
  o *= gate;
  o = o/(1.0 + max(max3(o) - 1.0, 0.0));                     // soft shoulder: the feedback can never run away
  f = vec4(safe(o), 1.0);
}
"""

DREAM_OUT_FS = COMMON + """
uniform sampler2D src, dream; uniform vec2 res; uniform float amt;
out vec4 f;
void main(){
  vec2 px = 1.0/res;
  vec3 c = tap(src, v_uv);
  vec3 b = (tap(src, v_uv + vec2(px.x, 0.0)) + tap(src, v_uv - vec2(px.x, 0.0)) +
            tap(src, v_uv + vec2(0.0, px.y)) + tap(src, v_uv - vec2(0.0, px.y)))*0.25;
  vec3 d = tap(dream, v_uv);
  vec3 o = max(d + (c - b)*0.7, 0.0);                        // full-res crispness back on the half-res dream
  f = vec4(mix(c, o, amt), 1.0);
}
"""

# ---- oil paint: anisotropic generalized Kuwahara (8 overlapping sectors on an ellipse aligned with the structure
# tensor), 48 taps at half res; full-res composite adds impasto relief lit from the upper left.
OIL_FS = COMMON + """
uniform sampler2D src; uniform vec2 res; uniform float rad;
out vec4 f;
void main(){
  vec2 px = 1.0/res, uv = v_uv;
  float tl = luma(tap(src, uv + vec2(-px.x, px.y))), tc = luma(tap(src, uv + vec2(0.0, px.y))), tr = luma(tap(src, uv + px));
  float ml = luma(tap(src, uv - vec2(px.x, 0.0))), mr = luma(tap(src, uv + vec2(px.x, 0.0)));
  float bl = luma(tap(src, uv - px)), bc = luma(tap(src, uv - vec2(0.0, px.y))), br = luma(tap(src, uv + vec2(px.x, -px.y)));
  float gx = (tr + 2.0*mr + br) - (tl + 2.0*ml + bl), gy = (tl + 2.0*tc + tr) - (bl + 2.0*bc + br);
  float E = gx*gx, F = gx*gy, G = gy*gy;
  float tr2 = E + G, dsc = sqrt(max((E - G)*(E - G) + 4.0*F*F, 0.0));
  float l1 = 0.5*(tr2 + dsc), l2 = 0.5*(tr2 - dsc);
  vec2 tv = vec2(l1 - E, -F); tv = dot(tv, tv) > 1e-10 ? normalize(tv) : vec2(1.0, 0.0);
  float A = (l1 + l2) > 1e-6 ? (l1 - l2)/(l1 + l2) : 0.0;
  float ea = rad*(1.0 + 0.9*A), eb = rad/(1.0 + 0.9*A);
  mat2 R = mat2(tv.x, tv.y, -tv.y, tv.x);
  vec3 c0 = tap(src, uv);
  vec3 m[8]; vec3 s[8]; float w[8];
  for (int k = 0; k < 8; k++) { m[k] = c0; s[k] = c0*c0; w[k] = 1.0; }
  for (int ring = 1; ring <= 3; ring++) {
    float rr = float(ring)/3.0, off = (ring == 2) ? 0.5 : 0.0;
    for (int a = 0; a < 16; a++) {
      float ang = (float(a) + off)*TAU/16.0;
      vec2 o = R*vec2(cos(ang)*ea, sin(ang)*eb)*rr*px;
      vec3 c = tap(src, uv + o);
      int k = a/2; int k2 = (a - 2*k == 1) ? (k + 1)%8 : (k + 7)%8;
      m[k] += c; s[k] += c*c; w[k] += 1.0;
      m[k2] += c*0.35; s[k2] += c*c*0.35; w[k2] += 0.35;
    }
  }
  vec3 acc = vec3(0.0); float wsum = 0.0;
  for (int k = 0; k < 8; k++) {
    vec3 mu = m[k]/w[k];
    vec3 v = abs(s[k]/w[k] - mu*mu);
    float sg = sqrt(v.r + v.g + v.b)*9.0;
    float wk = 1.0/(1.0 + sg*sg*sg*sg*sg*sg);
    acc += mu*wk; wsum += wk;
  }
  f = vec4(safe(acc/max(wsum, 1e-6)), 1.0);
}
"""

OIL_OUT_FS = COMMON + """
uniform sampler2D src, oil; uniform vec2 res, hres; uniform float amt, relief;
out vec4 f;
void main(){
  vec2 hp = 1.0/hres;
  vec3 c = tap(src, v_uv), p = tap(oil, v_uv);
  float hx = luma(tap(oil, v_uv + vec2(hp.x, 0.0))) - luma(tap(oil, v_uv - vec2(hp.x, 0.0)));
  float hy = luma(tap(oil, v_uv + vec2(0.0, hp.y))) - luma(tap(oil, v_uv - vec2(0.0, hp.y)));
  float lit = clamp(dot(vec2(hx, hy), normalize(vec2(-0.6, 0.8)))*6.0, -0.45, 0.35);
  float bristle = vnoise(v_uv*res*vec2(0.35, 0.08))*0.08 - 0.04;   // faint canvas / bristle texture, multiplicative
  vec3 o = p*(1.0 + relief*(lit + bristle*step(0.02, max3(p))));
  f = vec4(mix(c, max(o, 0.0), amt), 1.0);
}
"""

HALFTONE_FS = COMMON + """
uniform sampler2D src; uniform vec2 res; uniform float amt, cell, ang, style;
out vec4 f;
vec3 cellColour(vec2 cpx){
  vec2 d = vec2(cell*0.25)/res, u = cpx/res;
  return (tap(src, u)*2.0 + tap(src, u + d) + tap(src, u - d) + tap(src, u + vec2(d.x, -d.y)) + tap(src, u + vec2(-d.x, d.y)))/6.0;
}
float dotCover(vec2 pp, float a, float v){                    // AA dot of area fraction v (<= pi/4) on a grid at angle a
  mat2 R = rot(a);
  vec2 q = (R*pp)/cell, ci = floor(q) + 0.5;
  float r = cell*sqrt(clamp(v, 0.0, 0.785)/PI);
  float d = length(q - ci)*cell;
  return clamp(r - d + 0.5, 0.0, 1.0);
}
vec2 cellCentre(vec2 pp, float a){ mat2 R = rot(a); vec2 ci = floor((R*pp)/cell) + 0.5; return transpose(R)*(ci*cell); }
void main(){
  vec2 pp = gl_FragCoord.xy;
  vec3 c = tap(src, v_uv), o;
  int st = int(style + 0.5);
  if (st == 1) {                                              // RGB screen: one rotated grid per channel
    for (int k = 0; k < 3; k++) {
      float a = ang + float(k)*0.5236;
      vec3 cc = cellColour(cellCentre(pp, a));
      o[k] = dotCover(pp, a, cc[k]);
    }
  } else if (st == 2) {                                       // line engraving: line thickness = tone
    mat2 R = rot(ang);
    vec2 q = (R*pp)/cell;
    vec3 cc = cellColour(pp);
    float v = clamp(max3(cc), 0.0, 1.0);
    float y = q.y + 0.18*sin(q.x*0.7 + v*5.0);
    float dy = abs(fract(y) - 0.5)*cell;
    float cov = clamp(v*cell*0.5 - dy + 0.5, 0.0, 1.0)*step(0.003, v);
    o = (cc/max(max3(cc), 1e-4))*cov;
  } else {                                                    // colour dots: area = brightness, colour at full value
    vec3 cc = cellColour(cellCentre(pp, ang));
    float v = max3(cc);
    o = (cc/max(v, 1e-4))*dotCover(pp, ang, v);
  }
  f = vec4(mix(c, safe(o), amt), 1.0);
}
"""

NEON_EDGE_FS = COMMON + """
uniform sampler2D src; uniform vec2 res; uniform float colormode, t;
uniform vec3 pal[5];
out vec4 f;
vec3 palette(float x){ x = fract(x)*5.0; int i = int(x); float fr = smoothstep(0.0, 1.0, fract(x));
  return mix(pal[i], pal[(i + 1)%5], fr); }
void main(){
  vec2 px = 1.0/res, uv = v_uv;
  vec3 a = tap(src, uv + vec2(-px.x, px.y)), b = tap(src, uv + vec2(0.0, px.y)), cc = tap(src, uv + px);
  vec3 d = tap(src, uv - vec2(px.x, 0.0)), e = tap(src, uv + vec2(px.x, 0.0));
  vec3 g = tap(src, uv - px), h = tap(src, uv - vec2(0.0, px.y)), k = tap(src, uv + vec2(px.x, -px.y));
  vec3 gx = (cc + 2.0*e + k) - (a + 2.0*d + g), gy = (a + 2.0*b + cc) - (g + 2.0*h + k);
  float gm = sqrt(max3(gx*gx + gy*gy));
  float edge = smoothstep(0.08, 0.55, gm);
  vec3 hi = max(max(max(a, b), max(cc, d)), max(max(e, g), max(h, k)));   // colour of the lit side
  vec3 own = max3(hi) > 0.01 ? hi/max3(hi) : vec3(0.0);
  float dir = atan(luma(gy), luma(gx))/TAU;
  vec3 pc = palette(dir + t*0.05);
  vec3 col = mix(own, pc, colormode);
  col = max(mix(vec3(luma(col)), col, 1.35), 0.0);
  f = vec4(col*edge*step(0.01, max3(hi)), 1.0);
}
"""

NEON_OUT_FS = COMMON + """
uniform sampler2D src, edges, glow; uniform float amt, keep, gl;
out vec4 f;
void main(){
  vec3 c = tap(src, v_uv), e = tap(edges, v_uv), g = max(tap(glow, v_uv) - 0.012, 0.0);
  vec3 o = c*keep*0.85 + e + g*gl*2.2;
  f = vec4(mix(c, o, amt), 1.0);
}
"""

# bloom pyramid: down = 4 bilinear taps (box 4x4) with a Karis weight on the first level (no fireflies / flicker)
DOWN_FS = COMMON + """
uniform sampler2D src; uniform vec2 spx; uniform float first, thr;
out vec4 f;
vec3 pre(vec3 c){
  if (first < 0.5) return c;
  if (thr < 0.0) return c;                                   // neon: no threshold, just blur the edges
  float m = max3(c), k = 0.5*thr;
  float soft = clamp(m - thr + k, 0.0, 2.0*k); soft = soft*soft/(4.0*k + 1e-4);
  return c*max(soft, m - thr)/max(m, 1e-4);
}
void main(){
  vec3 a = pre(tap(src, v_uv + vec2(-spx.x, -spx.y))), b = pre(tap(src, v_uv + vec2(spx.x, -spx.y)));
  vec3 c = pre(tap(src, v_uv + vec2(-spx.x, spx.y))), d = pre(tap(src, v_uv + vec2(spx.x, spx.y)));
  if (first > 0.5) {
    float wa = 1.0/(1.0 + luma(a)), wb = 1.0/(1.0 + luma(b)), wc = 1.0/(1.0 + luma(c)), wd = 1.0/(1.0 + luma(d));
    f = vec4((a*wa + b*wb + c*wc + d*wd)/(wa + wb + wc + wd), 1.0);
  } else f = vec4((a + b + c + d)*0.25, 1.0);
}
"""

UP_FS = COMMON + """
uniform sampler2D low, cur; uniform vec2 lpx; uniform float w;
out vec4 f;
void main(){
  vec3 s = tap(low, v_uv)*4.0;
  s += (tap(low, v_uv + vec2(lpx.x, 0.0)) + tap(low, v_uv - vec2(lpx.x, 0.0)) +
        tap(low, v_uv + vec2(0.0, lpx.y)) + tap(low, v_uv - vec2(0.0, lpx.y)))*2.0;
  s += tap(low, v_uv + lpx) + tap(low, v_uv - lpx) + tap(low, v_uv + vec2(lpx.x, -lpx.y)) + tap(low, v_uv + vec2(-lpx.x, lpx.y));
  f = vec4(tap(cur, v_uv) + s/16.0*w, 1.0);
}
"""

BLOOM_OUT_FS = COMMON + """
uniform sampler2D src, bloom; uniform float amt, norm;
out vec4 f;
void main(){
  vec3 c = tap(src, v_uv), b = max(tap(bloom, v_uv)*norm - 0.01, 0.0);   // knee: far halo tails end in true black
  f = vec4(c + b*amt*1.6, 1.0);
}
"""

CHROMA_FS = COMMON + """
uniform sampler2D src; uniform float amt, mode, ang, asp;
out vec4 f;
void main(){
  vec2 p = (v_uv - 0.5)*vec2(asp, 1.0), d;
  int md = int(mode + 0.5);
  if (md == 1) d = vec2(cos(ang*TAU), sin(ang*TAU))*0.016*amt;
  else { d = rot(ang*TAU)*p*0.055*amt*(0.4 + length(p)); }
  vec3 acc = vec3(0.0), wsum = vec3(0.0);
  if (md == 2) {                                              // prism ghosts: three separated copies + the original
    d *= 2.2;
    vec3 o = tap(src, v_uv);
    vec3 r = tap(src, v_uv + d/vec2(asp, 1.0)), b = tap(src, v_uv - d/vec2(asp, 1.0));
    vec3 g2 = tap(src, v_uv + 2.0*d/vec2(asp, 1.0));
    acc = vec3(r.r*0.55 + o.r*0.45, o.g*0.7 + g2.g*0.3, b.b*0.55 + o.b*0.45);
    f = vec4(acc, 1.0);
    return;
  }
  for (int k = 0; k < 7; k++) {                               // spectral taps: weights sum to 1 per channel
    float s = float(k)/3.0 - 1.0;
    vec3 w = vec3(max(0.0, 1.0 - abs(s - 0.75)*1.4), max(0.0, 1.0 - abs(s)*1.4), max(0.0, 1.0 - abs(s + 0.75)*1.4));
    acc += tap(src, v_uv + d*s/vec2(asp, 1.0))*w; wsum += w;
  }
  f = vec4(acc/wsum, 1.0);
}
"""

POSTER_FS = COMMON + """
uniform sampler2D src; uniform float amt, levels, palmix;
uniform vec3 pal[5];
out vec4 f;
float stairs(float x, float n){                               // antialiased staircase (fwidth-wide ramps)
  float y = x*n, fl = floor(y), fr = y - fl, w = clamp(fwidth(y)*0.75, 0.002, 0.5);
  return (fl + smoothstep(0.5 - w, 0.5 + w, fr))/n;
}
void main(){
  vec3 c = tap(src, v_uv);
  float n = max(1.0, floor(levels + 0.5) - 1.0);
  vec3 q = vec3(stairs(c.r, n), stairs(c.g, n), stairs(c.b, n));
  float m = max3(c), ql = stairs(m, n);
  float band = floor(ql*n + 0.5);
  vec3 pc = pal[int(mod(band, 5.0))]*ql;
  pc = pc/max(1.0, max3(pc));
  vec3 o = mix(q, pc, palmix);
  f = vec4(mix(c, o, amt), 1.0);
}
"""

SORT_FS = COMMON + """
uniform sampler2D src; uniform vec2 res; uniform float thr, len, ang;
out vec4 f;
void main(){
  vec2 px = 1.0/res, uv = v_uv;
  vec3 c0 = tap(src, uv);
  float k0 = luma(c0);
  float m0 = smoothstep(thr - 0.03, thr + 0.03, k0);
  if (m0 <= 0.0) { f = vec4(c0, 0.0); return; }
  vec2 dir = vec2(cos(ang*PI), -sin(ang*PI))*px*mix(1.0, 3.0, len);
  vec3 lo = c0, hi = c0; float klo = k0, khi = k0, nb = 0.0, nf = 0.0;
  for (int i = 1; i <= 32; i++) {
    vec3 c = tap(src, uv - dir*float(i)); float k = luma(c);
    if (k < thr) break;
    nb += 1.0; if (k < klo) { klo = k; lo = c; } if (k > khi) { khi = k; hi = c; }
  }
  for (int i = 1; i <= 32; i++) {
    vec3 c = tap(src, uv + dir*float(i)); float k = luma(c);
    if (k < thr) break;
    nf += 1.0; if (k < klo) { klo = k; lo = c; } if (k > khi) { khi = k; hi = c; }
  }
  float fpos = (nb + 0.5)/(nb + nf + 1.0);
  vec3 o = mix(lo, hi, smoothstep(0.0, 1.0, fpos));
  float span = smoothstep(1.0, 6.0, nb + nf);
  f = vec4(o, m0*span);
}
"""

SORT_OUT_FS = COMMON + """
uniform sampler2D src, srt; uniform float amt;
out vec4 f;
void main(){
  vec3 c = tap(src, v_uv); vec4 s = texture(srt, v_uv);
  float a = (s.a >= 0.0 && s.a <= 1.0) ? s.a : 0.0;
  f = vec4(mix(c, safe(s.rgb), a*amt), 1.0);
}
"""

GLITCH_FS = COMMON + """
uniform sampler2D src; uniform vec2 res; uniform float amt, env, seed, blk;
out vec4 f;
void main(){
  vec2 pp = gl_FragCoord.xy;
  float bh = blk, row = floor(pp.y/bh);
  float bw = blk*mix(2.0, 9.0, hash12(vec2(row, seed)));
  vec2 cell = vec2(floor(pp.x/bw), row);
  float h = hash12(cell + seed*13.17);
  float sel = step(1.0 - 0.20*amt, h)*env;                    // at most 20 % of the blocks move
  float sh = floor((hash12(cell + seed*7.71 + 1.3) - 0.5)*2.0*amt*0.08*res.x*sel*0.5)*2.0;
  float split = 6.0*amt*sel;
  float line = step(0.985, hash12(vec2(floor(pp.y/2.0), seed + 5.0)))*env*amt;   // a few thin tearing lines
  float sx = sh + line*24.0*(hash12(vec2(row, seed + 2.0)) - 0.5);
  vec2 u = (pp + vec2(sx, 0.0))/res;
  vec3 o = vec3(tap(src, u + vec2(split, 0.0)/res).r, tap(src, u).g, tap(src, u - vec2(split, 0.0)/res).b);
  f = vec4(o, 1.0);
}
"""

KALEIDO_FS = COMMON + """
uniform sampler2D src; uniform float amt, mode, folds, ph, asp;
out vec4 f;
void main(){
  vec3 c = tap(src, v_uv);
  vec2 p = (v_uv - 0.5)*vec2(asp, 1.0), q;
  int md = int(mode + 0.5);
  if (md == 2) {
    float n = floor(folds + 0.5), seg = TAU/n;
    float a = atan(p.y, p.x) + ph, r = length(p);
    a = mod(a, seg); a = abs(a - 0.5*seg);
    q = r*vec2(cos(a), sin(a));
    q = rot(-0.5*seg)*q;
  } else {
    q = rot(ph)*p;
    q = md == 1 ? abs(q) : vec2(abs(q.x), q.y);
    q = rot(-ph)*q;
  }
  vec3 o = tap(src, clamp(q/vec2(asp, 1.0) + 0.5, 0.0, 1.0));
  f = vec4(mix(c, o, amt), 1.0);
}
"""

PROGRAMS = {
    "final": FINAL_FS, "dream": DREAM_FS, "dream_out": DREAM_OUT_FS, "oil": OIL_FS, "oil_out": OIL_OUT_FS,
    "halftone": HALFTONE_FS, "neon_edge": NEON_EDGE_FS, "neon_out": NEON_OUT_FS, "down": DOWN_FS, "up": UP_FS,
    "bloom_out": BLOOM_OUT_FS, "chroma": CHROMA_FS, "posterize": POSTER_FS, "sort": SORT_FS, "sort_out": SORT_OUT_FS,
    "glitch": GLITCH_FS, "kaleido": KALEIDO_FS,
}
NEEDS = {"dream": ("dream", "dream_out"), "oil": ("oil", "oil_out"), "halftone": ("halftone",),
         "neon": ("neon_edge", "down", "up", "neon_out"), "chroma": ("chroma",), "posterize": ("posterize",),
         "sort": ("sort", "sort_out"), "glitch": ("glitch",), "bloom": ("down", "up", "bloom_out"),
         "kaleido": ("kaleido",)}
DEFAULT_PAL = ((0.0, 0.85, 0.8), (1.0, 0.05, 0.55), (1.0, 0.7, 0.05), (0.28, 0.1, 1.0), (0.1, 0.9, 0.3))


class FilterPass:
    """apply(disp, vals, clock, dt, limiter, master, blackout, pal) runs the chain on `disp` in place (via two
    full-size ping-pong targets) and ends with FINAL_FS writing `disp` + `level`. GL state it touches: viewport (set
    back to full size), texture units 0-3 (the engine re-binds its scene units after the call). Blending is assumed
    off, as everywhere in the engine. Targets are allocated on the first apply(), so a show that never uses a filter
    allocates nothing."""

    def __init__(self, ctx, size):
        import moderngl
        import numpy as np
        self.mgl, self.np = moderngl, np
        self.ctx, self.size = ctx, (int(size[0]), int(size[1]))
        self.state = ChainState()
        self.progs, self.vaos, self.broken = {}, {}, {}
        self.vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        for name, src in PROGRAMS.items():
            try:
                p = ctx.program(vertex_shader=VS, fragment_shader=src)
                self.progs[name] = p
                self.vaos[name] = ctx.vertex_array(p, [(self.vbo, "2f", "in_pos")])
            except Exception as ex:                 # one bad shader disables only the filters that need it
                self.broken[name] = str(ex)
                print(f"[filters] shader {name} failed: {ex}")
        self.T = None                               # targets, allocated lazily
        self.level = None
        self._final = {}
        self._dream_i, self._dream_last = 0, -1e9
        self.ms = {}                                # filter -> last measured GPU ms (bench only)

    # ------------------------------------------------------------------ targets
    def _tex(self, size, comps=4, dtype="f2"):
        t = self.ctx.texture(size, comps, dtype=dtype)
        t.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        t.repeat_x = t.repeat_y = False
        return t

    def _alloc(self):
        w, h = self.size
        hw, hh = max(1, w // 2), max(1, h // 2)
        T = {}
        for k in ("A", "B"):
            T[k] = self._tex((w, h))
        T["E"] = self._tex((w, h))                  # neon edges
        for k in ("H", "D0", "D1"):
            T[k] = self._tex((hw, hh))              # half-res work + dream feedback pair
        sz = (w, h)
        for i in range(1, 5):                       # pyramid 1/2 .. 1/16
            sz = (max(1, sz[0] // 2), max(1, sz[1] // 2))
            T[f"d{i}"] = self._tex(sz)
            if i < 4:
                T[f"u{i}"] = self._tex(sz)
        self.fbo = {k: self.ctx.framebuffer([t]) for k, t in T.items()}
        for k in ("D0", "D1"):
            self.fbo[k].clear(0.0, 0.0, 0.0, 1.0)
        self.T = T
        self.level = self._tex((w, h), 4, "f1")

    def release(self):
        for d in (getattr(self, "fbo", {}), self._final):
            for f in d.values():
                f.release()
        if self.T:
            for t in self.T.values():
                t.release()
        if self.level is not None:
            self.level.release()
        for v in self.vaos.values():
            v.release()
        for p in self.progs.values():
            p.release()
        self.vbo.release()
        self.T, self.level, self._final = None, None, {}

    close = release

    # ------------------------------------------------------------------ helpers
    def _ok(self, ft):
        return all(n in self.progs for n in NEEDS[ft])

    def _draw(self, name, fbo, size, tex=(), **uni):
        p = self.progs[name]
        fbo.use()
        self.ctx.viewport = (0, 0, size[0], size[1])
        for unit, (uname, t) in enumerate(tex):
            t.use(unit)
            if uname in p:
                p[uname].value = unit
        for k, v in uni.items():
            if k in p:
                if k == "pal":
                    p[k].write(self.np.ascontiguousarray(v, dtype="f4").tobytes())
                else:
                    p[k].value = v
        self.vaos[name].render(self.mgl.TRIANGLE_STRIP)

    def _pyramid(self, src, thr, radius):
        """Blur pyramid of `src` -> T['u1'] (half res). thr < 0 = no threshold (plain blur)."""
        T = self.T
        prev, psize = src, self.size
        for i in range(1, 5):
            d = T[f"d{i}"]
            self._draw("down", self.fbo[f"d{i}"], d.size, (("src", prev),),
                       spx=(1.0 / psize[0], 1.0 / psize[1]), first=1.0 if i == 1 else 0.0, thr=thr)
            prev, psize = d, d.size
        w = 0.45 + 0.55 * radius
        low = T["d4"]
        for i in (3, 2, 1):
            u = T[f"u{i}"]
            self._draw("up", self.fbo[f"u{i}"], u.size, (("low", low), ("cur", T[f"d{i}"])),
                       lpx=(1.0 / low.size[0], 1.0 / low.size[1]), w=w)
            low = u
        return T["u1"], 1.0 / (1.0 + w + w * w + w ** 3)

    # ------------------------------------------------------------------ the chain
    def apply(self, disp, vals, clock=None, dt=1 / 60, limiter=0.0, master=1.0, blackout=0.0, pal=None):
        d = self.state.advance(dt, vals)
        ch = [(ft, a) for ft, a in chain(vals) if self._ok(ft)]
        if self.T is None:
            self._alloc()
        T, w, h = self.T, self.size[0], self.size[1]
        pal = DEFAULT_PAL if pal is None else pal
        asp = w / h
        cur = disp
        for i, (ft, amt) in enumerate(ch):
            out = "A" if cur is not T["A"] else "B"
            getattr(self, f"_run_{ft}")(cur, self.fbo[out], amt, vals, d, asp, pal)
            cur = T[out]
        key = disp.glo
        if key not in self._final:
            self._final[key] = self.ctx.framebuffer([disp, self.level])
        src = cur
        if src is disp:                              # nothing ran (all broken): copy so FINAL never reads its target
            self.ctx.copy_framebuffer(self.fbo["A"], self._final[key])
            src = T["A"]
        self._draw("final", self._final[key], (w, h), (("src", src),), lim=float(limiter), master=float(master),
                   blackout=float(blackout))
        self.ctx.viewport = (0, 0, w, h)
        return True

    # one method per filter: (src texture, destination fbo, amount, values, frozen-aware dt, aspect, palette)
    def _run_dream(self, src, dst, amt, v, d, asp, pal):
        T = self.T
        now = self.state.t
        if abs(now - self._dream_last) > 0.5:          # first use, or back after a pause: start from the picture
            for k in ("D0", "D1"):
                self.ctx.copy_framebuffer(self.fbo[k], self._final_src_fbo(src))
        self._dream_last = now
        a, b = ("D0", "D1") if self._dream_i % 2 == 0 else ("D1", "D0")
        if d > 0.0:                                    # frozen: the dream holds
            self._dream_i += 1
            self._draw("dream", self.fbo[b], T[b].size, (("src", src), ("prev", T[a])),
                       res=(float(T[b].size[0]), float(T[b].size[1])), asp=asp, scl=_f(v.get("flt_dream_scale"), 0.5),
                       swirl=_f(v.get("flt_dream_swirl"), 0.5), motif=_f(v.get("flt_dream_motif"), 0.3),
                       t=self.state.t, fr=min(max(d * 60.0, 0.0), 3.0), amt=amt)
        else:
            b = a
        self._draw("dream_out", dst, self.size, (("src", src), ("dream", T[b])), res=(float(self.size[0]),
                   float(self.size[1])), amt=amt)

    def _final_src_fbo(self, tex):
        """A framebuffer around `tex` (cached), for blits."""
        key = ("src", tex.glo)
        if key not in self._final:
            self._final[key] = self.ctx.framebuffer([tex])
        return self._final[key]

    def _run_oil(self, src, dst, amt, v, d, asp, pal):
        T = self.T
        hs = T["H"].size
        self._draw("oil", self.fbo["H"], hs, (("src", src),), res=(float(hs[0]), float(hs[1])),
                   rad=2.0 + 5.0 * _f(v.get("flt_oil_size"), 0.5))
        self._draw("oil_out", dst, self.size, (("src", src), ("oil", T["H"])), res=tuple(map(float, self.size)),
                   hres=(float(hs[0]), float(hs[1])), amt=amt, relief=_f(v.get("flt_oil_relief"), 0.4))

    def _run_halftone(self, src, dst, amt, v, d, asp, pal):
        self._draw("halftone", dst, self.size, (("src", src),), res=tuple(map(float, self.size)), amt=amt,
                   cell=4.0 + 22.0 * _f(v.get("flt_ht_size"), 0.35), ang=_f(v.get("flt_ht_angle"), 0.125) * math.pi,
                   style=float(round(_f(v.get("flt_ht_style"), 0.0))))

    def _run_neon(self, src, dst, amt, v, d, asp, pal):
        T = self.T
        self._draw("neon_edge", self.fbo["E"], self.size, (("src", src),), res=tuple(map(float, self.size)),
                   colormode=_f(v.get("flt_neon_color"), 0.0), t=self.state.t, pal=pal)
        g = _f(v.get("flt_neon_glow"), 0.5)
        glow, norm = self._pyramid(T["E"], -1.0, 0.3 + 0.7 * g)
        self._draw("neon_out", dst, self.size, (("src", src), ("edges", T["E"]), ("glow", glow)), amt=amt,
                   keep=_f(v.get("flt_neon_keep"), 0.25), gl=g * norm * 2.0)

    def _run_bloom(self, src, dst, amt, v, d, asp, pal):
        thr = 0.2 + 0.7 * _f(v.get("flt_bloom_thresh"), 0.55)
        glow, norm = self._pyramid(src, thr, _f(v.get("flt_bloom_radius"), 0.5))
        self._draw("bloom_out", dst, self.size, (("src", src), ("bloom", glow)), amt=amt, norm=norm * 2.0)

    def _run_chroma(self, src, dst, amt, v, d, asp, pal):
        self._draw("chroma", dst, self.size, (("src", src),), amt=amt, asp=asp,
                   mode=float(round(_f(v.get("flt_chroma_mode"), 0.0))), ang=_f(v.get("flt_chroma_angle"), 0.0))

    def _run_posterize(self, src, dst, amt, v, d, asp, pal):
        self._draw("posterize", dst, self.size, (("src", src),), amt=amt,
                   levels=_f(v.get("flt_post_levels"), 4.0), palmix=_f(v.get("flt_post_palette"), 0.0), pal=pal)

    def _run_sort(self, src, dst, amt, v, d, asp, pal):
        T = self.T
        hs = T["H"].size
        self._draw("sort", self.fbo["H"], hs, (("src", src),), res=(float(hs[0]), float(hs[1])),
                   thr=0.05 + 0.75 * _f(v.get("flt_sort_thresh"), 0.35), len=_f(v.get("flt_sort_len"), 0.5),
                   ang=_f(v.get("flt_sort_angle"), 0.25))
        self._draw("sort_out", dst, self.size, (("src", src), ("srt", T["H"])), amt=amt)

    def _run_glitch(self, src, dst, amt, v, d, asp, pal):
        self._draw("glitch", dst, self.size, (("src", src),), res=tuple(map(float, self.size)), amt=amt,
                   env=float(self.state.glitch_env), seed=float(self.state.glitch_seed % 4096),
                   blk=float(round(6.0 + 42.0 * _f(v.get("flt_glitch_block"), 0.4))))

    def _run_kaleido(self, src, dst, amt, v, d, asp, pal):
        self._draw("kaleido", dst, self.size, (("src", src),), amt=amt, asp=asp,
                   mode=float(round(_f(v.get("flt_kal_mode"), 2.0))), folds=_f(v.get("flt_kal_folds"), 6.0),
                   ph=float(self.state.kal_phase))


# =====================================================================================================================
# standalone gallery / bench
def _gallery(inputs, out_dir, frames=50):
    import time
    from pathlib import Path

    import moderngl
    import numpy as np
    from PIL import Image

    import params as P
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = moderngl.create_standalone_context(require=330)
    w, h = 1280, 800
    disp = ctx.texture((w, h), 4, dtype="f2")
    fp = FilterPass(ctx, (w, h))
    looks = [(ft, {}) for ft in P.FILTER_TYPES] + [
        ("dream", {"flt_dream_motif": 0.0}), ("dream", {"flt_dream_motif": 1.0}), ("dream", {"flt_dream_scale": 1.0}),
        ("halftone", {"flt_ht_style": 1}), ("halftone", {"flt_ht_style": 2}), ("neon", {"flt_neon_color": 1.0}),
        ("chroma", {"flt_chroma_mode": 2}), ("posterize", {"flt_post_palette": 1.0}), ("kaleido", {"flt_kal_mode": 0}),
    ]
    q = ctx.query(time=True)
    rows = []
    for ii, path in enumerate(inputs):
        im = Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS).transpose(Image.FLIP_TOP_BOTTOM)
        arr = np.concatenate([np.asarray(im, "f4") / 255.0, np.ones((h, w, 1), "f4")], -1).astype("f2").tobytes()
        for li, (ft, extra) in enumerate(looks):
            v = P.defaults()
            v.update({"flt_slot1": P.FILTER_TYPES.index(ft) + 1, f"flt_{ft}": 1.0}, **extra)
            fp.state = ChainState()
            fp._dream_last = -1e9
            times = []
            for k in range(frames):
                disp.write(arr)
                with q:
                    fp.apply(disp, v, {}, 1 / 60, 0.0, 1.0, 0.0)
                if k >= frames - 20:
                    times.append(q.elapsed / 1e6)
                if ft == "glitch" and k > frames - 4 and fp.state.glitch_env < 0.5:
                    pass
            ctx.finish()
            res = np.frombuffer(disp.read(), dtype="f2").reshape(h, w, 4)[..., :3].astype("f4")
            img = Image.fromarray((np.clip(res, 0, 1) * 255 + 0.5).astype("u1")).transpose(Image.FLIP_TOP_BOTTOM)
            tag = ft + "".join(f"_{k.split('_')[-1]}{v2}" for k, v2 in extra.items())
            name = f"{ii}_{li:02d}_{tag}.png"
            img.save(out_dir / name)
            ms = sorted(times)[len(times) // 2]
            rows.append((Path(path).name, tag, ms, name))
            print(f"[filters] {Path(path).name:40s} {tag:28s} {ms:6.2f} ms")
    # contact sheet per input
    for ii, path in enumerate(inputs):
        names = [r[3] for r in rows if r[3].startswith(f"{ii}_")]
        tw, th = 320, 200
        cols = 5
        sheet = Image.new("RGB", (cols * tw, ((len(names) + 1 + cols - 1) // cols) * (th + 16)), (0, 0, 0))
        src = Image.open(path).convert("RGB").resize((tw, th))
        sheet.paste(src, (0, 0))
        for k, n in enumerate(names, start=1):
            sheet.paste(Image.open(out_dir / n).resize((tw, th), Image.LANCZOS), ((k % cols) * tw, (k // cols) * (th + 16)))
        sheet.save(out_dir / f"contact_{ii}.png")
    fp.release()
    ctx.release()
    return rows


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", nargs="*", default=None, help="input stills (default: two gallery frames)")
    ap.add_argument("--out", default=str(here / "test_frames" / "filters"))
    ap.add_argument("--frames", type=int, default=50)
    a = ap.parse_args()
    if a.gallery is not None:
        ins = a.gallery or [str(here / "test_frames" / "gallery" / "03_5_layers_mandala_all.png"),
                            str(here / "test_frames" / "gallery" / "04_5_layers_flat_all.png")]
        _gallery(ins, a.out, a.frames)
