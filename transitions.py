"""Visual scene transitions (Phase 3a + 3b of research/pro_redesign_plan.md, adapted to the user's "doors" request).

A recall that uses a VISUAL transition (doors, iris, portal, ...) renders the incoming look B a second time, at reduced
scale, into a private framebuffer for the transition's duration only, and composites A -> B through a transition mask.
MORPH (the existing parameter glide) stays the default; nothing here runs, allocates or draws unless a visual
transition is in progress (the engine checks one attribute per frame).

    pure python (no GL):  KINDS / MODES / DEFAULTS, clean_spec(), ease(), mask_ref() (the numpy mirror of the shader's
                          mask maths, used by the tests), Transitioner (the controller-side state machine)
    GL:                   TransitionPass (B framebuffer + the composite shader); created lazily by live_visuals.py

All shader code in this file is original (no gl-transitions ports; THIRD_PARTY_NOTICES.md needs no entry for it).

How a BLEND works (controller side, Transitioner.run):
  1. A is snapshotted (fader bases + modulation), then the ordinary recall runs (scene / theme / preset / clip): it
     writes B's targets the usual way (a morph, a lane morph or a snap).
  2. B's targets are read back out, and the PRIVATE params (the scene shader's own groups: space, motion, shape,
     colour, layers) go back to A: A keeps rendering untouched, B is rendered with B's private values.
  3. SHARED params (offscreen passes: 3D decks, fractal, beings, kene study, coil nest, text, art, camera; master
     and filters) can only have one value per frame: a pass on in both looks MORPHS over the transition; a pass only
     B uses snaps to B at once (A does not show it); a pass only A uses holds until the end. The passes are drawn
     at max(A, B) amount and each render gets its own amount (the pass conflict rule of plan 2.4).
  4. When the mask reaches 1 the private params snap to B (slew skipped): the first frame after equals the last
     composite (same values, same echo input). Echo: B reads the same echo input as A and its echo output goes
     to a scratch attachment, so the feedback buffer only ever sees A.
Safety: PANIC cancels at once (the safe look follows); governor strain at start -> MORPH + a 0.4 s dip through dark;
critical strain mid-way -> finish through a dip; glow is an edge / ring (never full frame) and capped; every visual
transition lasts >= 0.5 s, the portal >= 1 s, the white portal >= 1.5 s; the flash guard still runs on the result.
"""
import copy
import math

try:
    import numpy as np
except Exception:                       # the controller side must import without numpy too
    np = None

# ------------------------------------------------------------------------------------------------ catalogue
KINDS = ("doors", "slide", "curtains", "iris", "portal", "reveal", "shatter", "peel", "dissolve", "wipe", "zoom")
NAMES = {"doors": "Doors", "slide": "Sliding doors", "curtains": "Curtains", "iris": "Iris", "portal": "Portal",
         "reveal": "Pattern reveal", "shatter": "Shatter / tiles", "peel": "Peel", "dissolve": "Dissolve",
         "wipe": "Wipe", "zoom": "Zoom through"}
MODES = {"doors": ("away", "toward"), "slide": ("horizontal", "vertical"), "iris": ("circle", "polygon", "star", "kaleido"),
         "portal": ("ring", "white"), "reveal": ("lines", "dark", "edges"), "shatter": ("flip", "fall")}
EASES = ("lin", "smooth", "in", "out", "inout", "expo")
QUANTS = ("off", "beat", "bar", "phrase")
N_RANGE = {"doors": (1, 4), "slide": (1, 4), "iris": (3, 12), "shatter": (2, 32)}
DEFAULTS = {
    "doors":    dict(beats=4.0, ease="inout", n=2, mode="away", glow=0.6, soft=0.01),
    "slide":    dict(beats=2.0, ease="inout", n=2, mode="horizontal", glow=0.45, soft=0.01),
    "curtains": dict(beats=4.0, ease="inout", glow=0.25, soft=0.01),
    "iris":     dict(beats=2.0, ease="out", n=6, mode="circle", glow=0.45, soft=0.03),
    "portal":   dict(beats=4.0, ease="smooth", mode="ring", glow=0.85, soft=0.02),
    "reveal":   dict(beats=4.0, ease="in", mode="lines", glow=0.5, soft=0.08),
    "shatter":  dict(beats=4.0, ease="lin", n=10, mode="flip", glow=0.3, soft=0.01),
    "peel":     dict(beats=2.0, ease="inout", angle=30.0, glow=0.35, soft=0.01),
    "dissolve": dict(beats=2.0, ease="lin", glow=0.35, soft=0.1),
    "wipe":     dict(beats=2.0, ease="inout", angle=0.0, glow=0.35, soft=0.05),
    "zoom":     dict(beats=2.0, ease="in", glow=0.4, soft=0.03),
}
COMMON = dict(angle=0.0, glow=0.4, soft=0.03, color=(1.0, 0.86, 0.62), origin=None, n=0, mode=None, reverse=False,
              scale=0.5, seed=0.0)
MIN_SECS = 0.5                          # photosensitive floor: no visual transition is shorter (a big change = a flash)
MIN_SECS_KIND = {"portal": 1.0}         # the portal's bright rim swells over >= 1 s (effects spec A7)
MIN_SECS_WHITE = 1.5                    # white portal: up to near-white and back, each half >= 0.75 s
GLOW_CAP = 0.8                          # additive glow (edge / ring only) is scaled by this at most
WHITE_LEVEL = 0.85                      # the white portal never goes to pure white
DIP_S = 0.4                             # MORPH fallback: one dip through dark (down + up = one "flash" pair)
DIP_GAP_S = 1.0                         # never two dips closer than this (<= 1 dip / s, far under 3 Hz)
HURRY_S = 0.35                          # a non-visual recall during a transition: the running one ends in this long
SCALE_STRAINED = 0.35                   # B render scale while the governor reports strain (level 1)

PRIVATE_GROUPS = frozenset(("space", "motion", "shape", "color", "layers"))   # params the scene shader alone reads
GROUP_PASS = {"fractal": ("l_fractal",), "art": ("l_art",), "land": ("l_land", "l_swarm"), "beings": ("l_beings",),
              "text": ("l_text",), "kene": ("l_kene_shipibo", "l_kene_hunikuin"), "coilnest": ("l_coilnest",),
              "camera": ("l_camera",)}
PASS_AMOUNTS = ("l_fractal", "l_art", "l_land", "l_swarm", "l_beings", "l_text", "l_stack", "l_kene_shipibo",
                "l_kene_hunikuin", "l_coilnest", "l_camera")


def _num(v, lo, hi, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return min(hi, max(lo, x))


def clean_spec(raw):
    """A transition spec from a command / cue / scene -> a normalized dict, or None for MORPH (the default glide).

    Accepts a kind name ("doors"), "morph" / None / "" (= MORPH), or a dict {kind, beats | secs, ease, angle, soft,
    glow, color [r,g,b], origin [x,y] | null, n, mode, reverse, scale, quantize}. Unknown keys are dropped, numbers
    are clamped; a bad kind raises ValueError (so a command validator can refuse it)."""
    if raw is None or raw == "" or raw == "morph" or raw is False:
        return None
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError("transition must be a kind name or an object")
    kind = raw.get("kind")
    if kind in (None, "", "morph"):
        return None
    if kind not in KINDS:
        raise ValueError(f"unknown transition '{kind}'")
    d = dict(COMMON, **DEFAULTS[kind])
    out = {"kind": kind}
    if raw.get("secs") is not None:
        out["secs"] = _num(raw.get("secs"), 0.1, 30.0, 2.0)
    else:
        out["beats"] = _num(raw.get("beats", d["beats"]), 0.25, 64.0, d["beats"])
    e = raw.get("ease", d["ease"])
    out["ease"] = e if e in EASES else d["ease"]
    out["angle"] = _num(raw.get("angle", d["angle"]), -360.0, 360.0, d["angle"])
    out["soft"] = _num(raw.get("soft", d["soft"]), 0.0, 0.5, d["soft"])
    out["glow"] = _num(raw.get("glow", d["glow"]), 0.0, 1.0, d["glow"])
    col = raw.get("color", d["color"])
    if isinstance(col, (list, tuple)) and len(col) == 3:
        out["color"] = [_num(c, 0.0, 1.0, 1.0) for c in col]
    else:
        out["color"] = list(d["color"])
    org = raw.get("origin")
    if isinstance(org, (list, tuple)) and len(org) == 2:
        out["origin"] = [_num(org[0], 0.0, 1.0, 0.5), _num(org[1], 0.0, 1.0, 0.5)]
    else:
        out["origin"] = None                       # = the controller's draggable origin
    lo, hi = N_RANGE.get(kind, (0, 0))
    n = raw.get("n", d["n"])
    out["n"] = int(_num(n, lo, hi, d["n"])) if hi else 0
    if kind == "doors" and out["n"] == 3:
        out["n"] = 4
    modes = MODES.get(kind)
    m = raw.get("mode", d["mode"])
    if modes:
        if isinstance(m, int) and not isinstance(m, bool) and 0 <= m < len(modes):
            m = modes[m]
        out["mode"] = m if m in modes else d["mode"]
    out["reverse"] = bool(raw.get("reverse", False)) if kind in ("iris", "portal", "wipe") else False
    out["scale"] = _num(raw.get("scale", d["scale"]), 0.25, 1.0, 0.5)
    out["seed"] = _num(raw.get("seed", 0.0), 0.0, 1000.0, 0.0)
    q = raw.get("quantize")
    if q in QUANTS:
        out["quantize"] = q
    return out


def seconds(spec, bpm):
    """The transition's length in seconds at `bpm`, with the photosensitive floors applied."""
    bpm = bpm if bpm and bpm > 0 else 124.0
    s = spec["secs"] if "secs" in spec else spec["beats"] * 60.0 / bpm
    floor = MIN_SECS_KIND.get(spec["kind"], MIN_SECS)
    if spec["kind"] == "portal" and spec.get("mode") == "white":
        floor = MIN_SECS_WHITE
    return max(floor, float(s))


def ease(name, t):
    t = min(1.0, max(0.0, float(t)))
    if name == "lin":
        return t
    if name == "in":
        return t * t
    if name == "out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if name == "inout":
        return 4.0 * t ** 3 if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0
    if name == "expo":
        if t <= 0.0 or t >= 1.0:
            return t
        return 2.0 ** (20.0 * t - 10.0) / 2.0 if t < 0.5 else (2.0 - 2.0 ** (-20.0 * t + 10.0)) / 2.0
    return t * t * (3.0 - 2.0 * t)              # smooth


def uniforms(spec, e, origin, aspect, b_size=None):
    """Composite-shader uniforms for `spec` at eased progress `e` (also the numpy mirror's inputs)."""
    k = spec["kind"]
    modes = MODES.get(k, ())
    o = spec.get("origin") or origin or (0.5, 0.5)
    return dict(kind=KINDS.index(k), mode=modes.index(spec["mode"]) if spec.get("mode") in modes else 0,
                n=int(spec.get("n") or 0), e=float(min(1.0, max(0.0, e))), soft=max(0.002, float(spec["soft"])),
                glow=float(spec["glow"]) * GLOW_CAP, ang=math.radians(float(spec["angle"])), asp=float(aspect),
                rev=1.0 if spec.get("reverse") else 0.0, seed=float(spec.get("seed", 0.0)),
                org=(float(o[0]), float(o[1])), gcol=tuple(float(c) for c in spec["color"]), white=WHITE_LEVEL)


# ------------------------------------------------------------------------------------------------ shader
VS = """
#version 330
in vec2 in_pos; out vec2 uv;
void main(){ uv = in_pos*0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# One composite program; `kind` is a uniform branch (the whole frame takes the same path). Every kind fills:
#   ua / ub  where A / B are sampled,  m  B's weight (0 = A, 1 = B),  g  glow (edge / ring only),  sa / sb  shading.
# The mask maths is mirrored line for line by mask_ref() below (tests compare the two on the GPU).
COMPOSITE_FS = """
#version 330
uniform sampler2D texA, texB;
uniform int kind, mode, n;
uniform float e, soft, glow, ang, asp, rev, seed, white;
uniform vec2 org, pxA;
uniform vec3 gcol;
in vec2 uv; out vec4 f;
const float PI = 3.14159265;

float env(){ return smoothstep(0.0, 0.12, e) * (1.0 - smoothstep(0.88, 1.0, e)); }
float band(float x, float w){ return exp(-(x*x)/(w*w)); }
vec2 rot(vec2 p, float a){ float c = cos(a), s = sin(a); return vec2(c*p.x - s*p.y, s*p.x + c*p.y); }
float lum(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float hash(vec2 c){ return fract(sin(dot(c, vec2(1.7, 9.2)) + seed*1.3) * 437.585); }
vec2 mirror2(vec2 u){ return 1.0 - abs(1.0 - mod(u, 2.0)); }
float across(vec2 u, float a){
  vec2 d = vec2(cos(a), sin(a)); vec2 p = (u - 0.5)*vec2(asp, 1.0);
  float R = 0.5*(asp*abs(d.x) + abs(d.y)); return (dot(p, d) + R)/(2.0*R);
}
float cornerR(vec2 o){
  vec2 s = vec2(asp, 1.0);
  return max(max(length((vec2(0.0, 0.0) - o)*s), length((vec2(1.0, 0.0) - o)*s)),
             max(length((vec2(0.0, 1.0) - o)*s), length((vec2(1.0, 1.0) - o)*s)));
}
// iris gauge: 1 on the shape's outline at unit size; rmin = the smallest radius on that outline
float gauge(vec2 q, out float rmin){
  float r = length(q), a = atan(q.y, q.x), N = float(max(n, 3));
  if (mode == 1){ float k = 2.0*PI/N; float x = mod(a, k) - 0.5*k; rmin = cos(0.5*k); return r*cos(x)/cos(0.5*k); }
  if (mode == 2){ float t = abs(fract(a*N/(2.0*PI))*2.0 - 1.0); rmin = 0.5; return r/mix(1.0, 0.5, t); }
  if (mode == 3){ float k = 2.0*PI/N; float x = abs(mod(a + 1.5*e, k) - 0.5*k);
                  rmin = 0.5; return r/(0.75 + 0.25*cos(x*N*2.0)); }
  rmin = 1.0; return r;
}

// doors: one hinged panel (hinge x hx, side sx, width W, rows [y0, y1], angle th), true perspective about the centre
const float F = 1.8;
bool door(vec2 p, float hx, float sx, float W, float y0, float y1, float th, float sg,
          inout vec2 ua, inout float sa, inout float g){
  float c = cos(th), s = sin(th);
  float den = W*(c*F - sg*sx*p.x*s);
  float t = den > 1e-5 ? F*(p.x - hx)*sx/den : -1.0;
  float k = F/(F + sg*t*W*s);
  float Y = p.y/max(k, 1e-3);
  float ke = F/(F + sg*W*s);                       // free edge: depth scale and screen x
  float xe = (hx + sx*W*c)*ke;
  float ye0 = y0*ke, ye1 = y1*ke;
  if (p.y >= min(ye0, ye1) - 0.02 && p.y <= max(ye0, ye1) + 0.02) g += band(p.x - xe, 0.018 + soft);
  if (t >= 0.0 && t <= 1.0 && Y >= y0 && Y <= y1 && F + sg*t*W*s > 0.05){
    ua = vec2((hx + sx*t*W)/asp + 0.5, Y + 0.5);
    sa = (sg > 0.0 ? mix(1.0, 0.35, s) : mix(1.0, 0.7, s)) * (1.0 - 0.25*band(1.0 - t, 0.04));
    return true;
  }
  return false;
}

void main(){
  vec2 ua = uv, ub = uv; float m = 0.0, g = 0.0, sa = 1.0, sb = 1.0;
  vec2 S = vec2(asp, 1.0);
  if (kind == 0){                                   // DOORS
    vec2 p = (uv - 0.5)*S; float sg = mode == 1 ? -1.0 : 1.0; bool cov = false;
    if (n <= 1){ cov = door(p, -0.5*asp, 1.0, asp, -0.5, 0.5, e*0.5*PI, sg, ua, sa, g); }
    else if (n == 2){
      cov = p.x < 0.0 ? door(p, -0.5*asp, 1.0, 0.5*asp, -0.5, 0.5, e*0.5*PI, sg, ua, sa, g)
                      : door(p, 0.5*asp, -1.0, 0.5*asp, -0.5, 0.5, e*0.5*PI, sg, ua, sa, g);
    } else {
      float et = clamp(e*1.2, 0.0, 1.0), eb = clamp(e*1.2 - 0.2, 0.0, 1.0);
      float th = (p.y >= 0.0 ? et : eb)*0.5*PI; float y0 = p.y >= 0.0 ? 0.0 : -0.5, y1 = p.y >= 0.0 ? 0.5 : 0.0;
      cov = p.x < 0.0 ? door(p, -0.5*asp, 1.0, 0.5*asp, y0, y1, th, sg, ua, sa, g)
                      : door(p, 0.5*asp, -1.0, 0.5*asp, y0, y1, th, sg, ua, sa, g);
    }
    float fade = 1.0 - smoothstep(0.82, 1.0, e);
    m = cov ? 1.0 - fade : 1.0;
    g *= env();
  } else if (kind == 1){                            // SLIDING DOORS
    bool cov = false; float edge = 1.0;
    if (n <= 1){
      float xn = across(uv, ang); vec2 d = vec2(cos(ang), sin(ang));
      float R = 0.5*(asp*abs(d.x) + abs(d.y));
      cov = xn >= e; ua = uv - d*(e*2.0*R)/S; edge = abs(xn - e)*2.0*R;
    } else if (n == 2){
      float x = mode == 1 ? uv.y : uv.x, sc = mode == 1 ? 1.0 : asp;
      float l = 0.5 - 0.5*e, r = 0.5 + 0.5*e;
      if (x < l){ cov = true; if (mode == 1) ua.y += 0.5*e; else ua.x += 0.5*e; }
      if (x > r){ cov = true; if (mode == 1) ua.y -= 0.5*e; else ua.x -= 0.5*e; }
      edge = min(abs(x - l), abs(x - r))*sc;
    } else {
      vec2 q = sign(uv - 0.5 + 1e-6); vec2 sh = q*0.5*e;
      vec2 o = uv - sh; cov = all(greaterThanEqual((o - 0.5)*q, vec2(0.0)));
      ua = o; vec2 dd = abs(uv - 0.5) - 0.5*e; edge = min(abs(dd.x)*asp, abs(dd.y));
    }
    m = cov ? 0.0 : 1.0;
    g = env()*band(edge, 0.012 + soft);
    if (cov) sa = 1.0 - 0.35*band(edge, 0.012);
  } else if (kind == 2){                            // CURTAINS
    float y = uv.y, pw = 1.0 + 0.8*(1.0 - y);
    float edge = 0.5*pow(max(1.0 - e, 0.0), pw);
    bool left = uv.x < 0.5; float x = left ? uv.x : 1.0 - uv.x;
    bool cov = x < edge;
    float fx = x/max(edge, 1e-4)*0.5;              // fabric coordinate 0..0.5 (gathers as it opens)
    if (cov){ ua = vec2(left ? fx : 1.0 - fx, y); sa = 0.78 + 0.22*cos(fx*2.0*PI*9.0)*(0.35 + 0.65*e); }
    m = cov ? 0.0 : 1.0;
    g = env()*band((x - edge)*asp, 0.01 + soft);
    if (!cov) sb = 1.0 - 0.35*env()*band((x - edge)*asp, 0.05);
  } else if (kind == 3 || kind == 4){               // IRIS / PORTAL
    vec2 q = (uv - org)*S;
    float rmin = 1.0, dn;
    if (kind == 3){ q = rot(q, -ang); dn = gauge(q, rmin); }
    else dn = length(q);
    dn = dn/(cornerR(org)/rmin);
    if (rev > 0.5) dn = 1.0 - dn;
    if (kind == 4 && mode == 1){                    // WHITE PORTAL: a near-white circle swells, then B comes through it
      float r1 = smoothstep(0.0, 0.55, e)*(1.0 + 4.0*soft);
      float w = 1.0 - smoothstep(r1 - 4.0*soft, r1, dn);
      float ph2 = smoothstep(0.45, 1.0, e);
      vec3 ca = texture(texA, uv).rgb, cb = texture(texB, uv).rgb;
      vec3 col = mix(mix(ca, gcol*white, w), cb, ph2);
      f = vec4(max(col, vec3(0.0)), texture(texA, uv).a);
      return;
    }
    float th = e*(1.0 + soft);
    m = 1.0 - smoothstep(th - soft, th, dn);
    float rim = band(dn - (th - 0.5*soft), kind == 4 ? 0.022 + soft : 0.012 + 0.5*soft);
    g = env()*rim;
    if (kind == 4){
      g += 0.35*env()*band(dn - th, 0.09)*m;
      ub = org + (uv - org)*(1.0 - 0.12*env()*band(dn - th, 0.15));
    }
  } else if (kind == 5){                            // PATTERN-AWARE REVEAL (B opens through A's own lines / cells)
    float l0 = lum(textureLod(texA, uv, 0.0).rgb);
    float gx = lum(textureLod(texA, uv + vec2(pxA.x, 0.0), 0.0).rgb) - lum(textureLod(texA, uv - vec2(pxA.x, 0.0), 0.0).rgb);
    float gy = lum(textureLod(texA, uv + vec2(0.0, pxA.y), 0.0).rgb) - lum(textureLod(texA, uv - vec2(0.0, pxA.y), 0.0).rgb);
    float ed = clamp(length(vec2(gx, gy))*4.0, 0.0, 1.0);
    float lb = lum(textureLod(texA, uv, 3.0).rgb);
    float f0, fb;
    if (mode == 1){ f0 = 1.0 - smoothstep(0.05, 0.45, l0); fb = 1.0 - smoothstep(0.05, 0.45, lb); }
    else if (mode == 2){ f0 = ed; fb = smoothstep(0.05, 0.5, lb); }
    else { f0 = max(smoothstep(0.12, 0.65, l0), 0.8*ed); fb = smoothstep(0.05, 0.5, lb); }
    float fl = max(f0, min(1.0, 1.3*e)*fb);
    float th = 1.0 - e*(1.0 + soft);
    m = smoothstep(th, th + soft, fl);
    g = env()*band(fl - th - 0.5*soft, 0.06);
  } else if (kind == 6){                            // SHATTER / TILES
    float cols = float(max(n, 2)), rows = max(1.0, floor(cols/asp + 0.5));
    vec2 G = vec2(cols, rows), cid = floor(uv*G), lc = fract(uv*G);
    float h = hash(cid);
    float dist = clamp(length(((cid + 0.5)/G - org)*S)/1.2, 0.0, 1.0);
    float di = 0.55*(0.5*h + 0.5*dist);
    float pi_ = clamp((e - di)/0.45, 0.0, 1.0);
    m = 1.0;
    if (mode == 1){
      float k = 1.0 - pi_; vec2 off = vec2(0.0, -0.7*pi_*pi_); float a = (h - 0.5)*2.5*pi_;
      vec2 q = rot(lc - 0.5 - off, -a)/max(k, 1e-3);
      if (pi_ < 1.0 && max(abs(q.x), abs(q.y)) < 0.5){ m = 0.0; ua = (cid + q + 0.5)/G; sa = 1.0 - 0.4*pi_; }
      else sb = 1.0 - 0.3*(1.0 - pi_)*step(1e-4, pi_);
    } else {
      float wv = abs(cos(pi_*PI));
      if (abs(lc.x - 0.5) < 0.5*wv){
        vec2 cu = (cid + vec2((lc.x - 0.5)/max(wv, 1e-3) + 0.5, lc.y))/G;
        if (pi_ < 0.5){ m = 0.0; ua = cu; sa = 0.55 + 0.45*wv; } else { ub = cu; sb = 0.55 + 0.45*wv; }
      } else sb = 0.6 + 0.4*pi_;
    }
    float de = min(min(lc.x, 1.0 - lc.x)*asp/cols, min(lc.y, 1.0 - lc.y)/rows);
    g = env()*band(de, 0.006)*step(1e-4, pi_)*(1.0 - pi_);
  } else if (kind == 7){                            // PEEL (the page folds back over itself, B underneath)
    vec2 d = vec2(cos(ang), sin(ang)); float R = 0.5*(asp*abs(d.x) + abs(d.y));
    float xn = across(uv, ang), fp = 1.0 - e*1.01;
    m = smoothstep(fp - 0.002, fp + 0.002, xn);
    if (xn < fp && xn >= 2.0*fp - 1.0){             // the flap: A's back, mirrored about the fold
      float xm = 2.0*fp - xn; ua = uv + d*((xm - xn)*2.0*R)/S;
      sa = 0.5 + 0.35*(fp - xn)/max(1.0 - fp, 1e-3);
    }
    if (xn > fp) sb = 1.0 - 0.5*exp(-(xn - fp)*2.0*R/0.05);
    g = env()*band((xn - fp)*2.0*R, 0.01 + soft);
  } else if (kind == 8){                            // DISSOLVE (smooth noise, burning edge)
    vec2 p = uv*S*3.0;
    float nz = 0.5 + 0.5*(0.5*sin(p.x*2.1 + seed)*sin(p.y*2.7 + 1.3*seed) + 0.3*sin((p.x + p.y)*3.9 + 2.1*seed)
                         + 0.2*sin((p.x - 1.7*p.y)*7.3 + 0.7*seed));
    float th = 1.0 - e*(1.0 + soft);
    m = smoothstep(th, th + soft, nz);
    g = env()*band(nz - th - 0.5*soft, 0.02 + 0.3*soft);
  } else if (kind == 9){                            // WIPE (any angle)
    float xn = across(uv, ang); if (rev > 0.5) xn = 1.0 - xn;
    float th = e*(1.0 + soft);
    m = 1.0 - smoothstep(th - soft, th, xn);
    g = env()*band(xn - (th - 0.5*soft), 0.012 + 0.5*soft);
  } else {                                          // ZOOM THROUGH: fly into a spot of A; B grows out of it
    vec2 c = mix(org, vec2(0.5), e);
    ua = org + (uv - org)/exp(3.0*e);
    float dn = length((uv - c)*S)/cornerR(c);
    float r = pow(e, 1.4)*(1.0 + soft);
    m = 1.0 - smoothstep(r - soft, r, dn);
    ub = mirror2(0.5 + (uv - c)/mix(0.3, 1.0, e));
    g = env()*band(dn - (r - 0.5*soft), 0.02 + soft);
  }
  vec4 ca = texture(texA, ua), cb = texture(texB, ub);
  vec3 col = mix(ca.rgb*sa, cb.rgb*sb, m) + gcol*(g*glow);
  f = vec4(max(col, vec3(0.0)), texture(texA, uv).a);
}
"""


# ------------------------------------------------------------------------------------------------ numpy mirror (tests)
def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _across(u, v, a, asp):
    dx, dy = math.cos(a), math.sin(a)
    R = 0.5 * (asp * abs(dx) + abs(dy))
    return (((u - 0.5) * asp) * dx + (v - 0.5) * dy + R) / (2.0 * R)


def _corner_r(o, asp):
    return max(math.hypot((cx - o[0]) * asp, cy - o[1]) for cx in (0.0, 1.0) for cy in (0.0, 1.0))


def _grid(w, h):
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    return np.meshgrid(u, v)                    # rows = v (bottom row first, like GL)


def mask_ref(spec, e, size=(160, 100), origin=None, field=None):
    """B's weight per pixel (rows bottom-first) for `spec` at eased progress `e`: the numpy mirror of COMPOSITE_FS.

    field: (f0, fb) arrays for "reveal" (the A-derived line field and its blurred version, 0..1).
    Not used at show time: the tests check coverage 0 -> 1 and compare against the GPU."""
    w, h = size
    U = uniforms(spec, e, origin, w / h)
    asp, e, soft, ang, n, mode = U["asp"], U["e"], U["soft"], U["ang"], U["n"], U["mode"]
    o = U["org"]
    u, v = _grid(w, h)
    env = float(_ss(0.0, 0.12, e) * (1.0 - _ss(0.88, 1.0, e)))
    k = spec["kind"]
    if k == "doors":
        px, py = (u - 0.5) * asp, v - 0.5
        sg = -1.0 if mode == 1 else 1.0
        cov = np.zeros_like(u, dtype=bool)

        def door(sel, hx, sx, W, y0, y1, th):
            c, s = math.cos(th), math.sin(th)
            den = W * (c * 1.8 - sg * sx * px * s)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(den > 1e-5, 1.8 * (px - hx) * sx / np.where(den > 1e-5, den, 1.0), -1.0)
            kk = 1.8 / (1.8 + sg * t * W * s)
            Y = py / np.maximum(kk, 1e-3)
            ok = (t >= 0) & (t <= 1) & (Y >= y0) & (Y <= y1) & (1.8 + sg * t * W * s > 0.05)
            return sel & ok
        if n <= 1:
            cov = door(np.ones_like(cov), -0.5 * asp, 1.0, asp, -0.5, 0.5, e * 0.5 * math.pi)
        elif n == 2:
            th = e * 0.5 * math.pi
            cov = door(px < 0, -0.5 * asp, 1.0, 0.5 * asp, -0.5, 0.5, th) | \
                door(px >= 0, 0.5 * asp, -1.0, 0.5 * asp, -0.5, 0.5, th)
        else:
            et, eb = min(1.0, max(0.0, e * 1.2)), min(1.0, max(0.0, e * 1.2 - 0.2))
            for top, ee in ((True, et), (False, eb)):
                rows = (py >= 0) if top else (py < 0)
                y0, y1 = (0.0, 0.5) if top else (-0.5, 0.0)
                th = ee * 0.5 * math.pi
                cov |= door(rows & (px < 0), -0.5 * asp, 1.0, 0.5 * asp, y0, y1, th)
                cov |= door(rows & (px >= 0), 0.5 * asp, -1.0, 0.5 * asp, y0, y1, th)
        fade = 1.0 - float(_ss(0.82, 1.0, e))
        return np.where(cov, 1.0 - fade, 1.0)
    if k == "slide":
        if n <= 1:
            cov = _across(u, v, ang, asp) >= e
        elif n == 2:
            x = v if mode == 1 else u
            cov = (x < 0.5 - 0.5 * e) | (x > 0.5 + 0.5 * e)
        else:
            qx, qy = np.sign(u - 0.5 + 1e-6), np.sign(v - 0.5 + 1e-6)
            ox, oy = u - qx * 0.5 * e, v - qy * 0.5 * e
            cov = ((ox - 0.5) * qx >= 0) & ((oy - 0.5) * qy >= 0)
        return np.where(cov, 0.0, 1.0)
    if k == "curtains":
        edge = 0.5 * np.power(max(1.0 - e, 0.0), 1.0 + 0.8 * (1.0 - v))
        x = np.where(u < 0.5, u, 1.0 - u)
        return np.where(x < edge, 0.0, 1.0)
    if k in ("iris", "portal"):
        qx, qy = (u - o[0]) * asp, v - o[1]
        rmin = 1.0
        if k == "iris":
            c, s = math.cos(-ang), math.sin(-ang)
            qx, qy = c * qx - s * qy, s * qx + c * qy
            r, a = np.hypot(qx, qy), np.arctan2(qy, qx)
            N = float(max(n, 3))
            if mode == 1:
                kk = 2 * math.pi / N
                x = np.mod(a, kk) - 0.5 * kk
                dn, rmin = r * np.cos(x) / math.cos(0.5 * kk), math.cos(0.5 * kk)
            elif mode == 2:
                t = np.abs(np.mod(a * N / (2 * math.pi), 1.0) * 2 - 1)
                dn, rmin = r / (1.0 + (0.5 - 1.0) * t), 0.5
            elif mode == 3:
                kk = 2 * math.pi / N
                x = np.abs(np.mod(a + 1.5 * e, kk) - 0.5 * kk)
                dn, rmin = r / (0.75 + 0.25 * np.cos(x * N * 2.0)), 0.5
            else:
                dn = r
        else:
            dn = np.hypot(qx, qy)
        dn = dn / (_corner_r(o, asp) / rmin)
        if U["rev"] > 0.5:
            dn = 1.0 - dn
        if k == "portal" and mode == 1:
            return np.full_like(u, float(_ss(0.45, 1.0, e)))
        th = e * (1.0 + soft)
        return 1.0 - _ss(th - soft, th, dn)
    if k == "reveal":
        f0, fb = field
        fl = np.maximum(f0, min(1.0, 1.3 * e) * fb)
        th = 1.0 - e * (1.0 + soft)
        return _ss(th, th + soft, fl)
    if k == "shatter":
        cols = float(max(n, 2))
        rows = max(1.0, math.floor(cols / asp + 0.5))
        gx, gy = u * cols, v * rows
        cx, cy = np.floor(gx), np.floor(gy)
        lx, ly = gx - cx, gy - cy
        hsh = np.mod(np.sin(cx * 1.7 + cy * 9.2 + U["seed"] * 1.3) * 437.585, 1.0)
        dist = np.clip(np.hypot(((cx + 0.5) / cols - o[0]) * asp, (cy + 0.5) / rows - o[1]) / 1.2, 0.0, 1.0)
        di = 0.55 * (0.5 * hsh + 0.5 * dist)
        pi_ = np.clip((e - di) / 0.45, 0.0, 1.0)
        if mode == 1:
            kk = 1.0 - pi_
            a = (hsh - 0.5) * 2.5 * pi_
            px, py = lx - 0.5, ly - 0.5 + 0.7 * pi_ * pi_
            c, s = np.cos(-a), np.sin(-a)
            qx, qy = (c * px - s * py) / np.maximum(kk, 1e-3), (s * px + c * py) / np.maximum(kk, 1e-3)
            cov = (pi_ < 1.0) & (np.maximum(np.abs(qx), np.abs(qy)) < 0.5)
        else:
            wv = np.abs(np.cos(pi_ * math.pi))
            cov = (np.abs(lx - 0.5) < 0.5 * wv) & (pi_ < 0.5)
        return np.where(cov, 0.0, 1.0)
    if k == "peel":
        xn = _across(u, v, ang, asp)
        fp = 1.0 - e * 1.01
        return _ss(fp - 0.002, fp + 0.002, xn)
    if k == "dissolve":
        px, py, sd = u * asp * 3.0, v * 3.0, U["seed"]
        nz = 0.5 + 0.5 * (0.5 * np.sin(px * 2.1 + sd) * np.sin(py * 2.7 + 1.3 * sd) + 0.3 * np.sin((px + py) * 3.9 + 2.1 * sd)
                          + 0.2 * np.sin((px - 1.7 * py) * 7.3 + 0.7 * sd))
        th = 1.0 - e * (1.0 + soft)
        return _ss(th, th + soft, nz)
    if k == "wipe":
        xn = _across(u, v, ang, asp)
        if U["rev"] > 0.5:
            xn = 1.0 - xn
        th = e * (1.0 + soft)
        return 1.0 - _ss(th - soft, th, xn)
    # zoom
    cx, cy = o[0] + (0.5 - o[0]) * e, o[1] + (0.5 - o[1]) * e
    dn = np.hypot((u - cx) * asp, v - cy) / _corner_r((cx, cy), asp)
    r = e ** 1.4 * (1.0 + soft)
    return 1.0 - _ss(r - soft, r, dn)


def coverage(spec, e, size=(160, 100), origin=None, field=None):
    return float(np.mean(mask_ref(spec, e, size, origin, field)))


# ------------------------------------------------------------------------------------------------ controller side
class Transitioner:
    """Controller-side transition state machine (no GL). One per Controller (`Controller.trans`).

    idle:  frame() returns None, step() returns at once, nothing else runs (zero cost).
    run:   B's targets are held here; the engine asks frame(values) once per frame for B's values, the pass amounts
           and the eased progress; at progress 1 commit() makes B the live look.
    dip:   the MORPH fallback (governor strain, no engine): master dips through dark once (DIP_S), never twice in
           DIP_GAP_S. A critical strain mid-transition also finishes through a dip (commit at the dark point)."""

    def __init__(self, ctl):
        self.ctl = ctl
        self.default = None                     # global default spec (None = MORPH): panel / MIDI set it
        self.origin = [0.5, 0.5]                # draggable portal / iris / zoom origin
        self.active = None                      # the running BLEND
        self.queued = None                      # (fn, spec, label): a recall that arrived during a BLEND (newest wins)
        self.dip = None                         # {"t", "commit"}: a dip through dark in progress
        self._last_dip = -1e9
        self.engine_ok = False                  # the GL engine sets this once its TransitionPass exists
        self.stats = dict(started=0, finished=0, cancelled=0, fallbacks=0, queued=0, failed=0)
        self.last = None                        # last spec used (panel display)
        self._clock = 0.0                       # own seconds (dt sums): wall-clock jumps cannot skip a transition

    # ---- spec resolution ----------------------------------------------------------------
    def resolve(self, spec, scene=None, use_default=True):
        """Explicit spec > the scene's own default > the global default. "morph" explicitly means MORPH."""
        if spec is not None:
            return clean_spec(spec)
        if scene is not None and scene.get("transition") is not None:
            return clean_spec(scene["transition"])
        return copy.deepcopy(self.default) if use_default else None

    @property
    def busy(self):
        return self.active is not None or (self.dip is not None and self.dip.get("commit"))

    # ---- the wrapper every look change goes through ------------------------------------
    def run(self, fn, spec=None, label=""):
        """Run the look change `fn` (a recall that writes B's targets) with transition `spec` (already resolved,
        None = MORPH). Returns True if fn ran or was queued."""
        if self.busy:                           # overlap: the newest recall waits for the running transition
            self.queued = (fn, spec, label)
            self.stats["queued"] += 1
            if spec is None and self.active is not None:
                self._hurry()                   # a plain morph should not wait a whole door swing
            return True
        if spec is None:
            fn()
            return True
        ctl = self.ctl
        if not self.engine_ok or getattr(ctl, "gov_level", 0) >= 1:
            fn()                                # MORPH fallback (+ a dip only when the governor is the reason)
            self.stats["fallbacks"] += 1
            if self.engine_ok:
                self._start_dip(commit=False)
            return True
        self._begin(fn, spec, label)
        return True

    def _names(self):
        c = self.ctl
        params = _ctl_params(c)
        cont = _ctl_cont(c)
        priv = [n for n in cont if params[n][3] in PRIVATE_GROUPS]
        return params, cont, priv

    def _begin(self, fn, spec, label):
        ctl = self.ctl
        params, cont, priv = self._names()
        a_base = dict(ctl.base)
        a_mods = copy.deepcopy(ctl.mods)
        fn()
        b = {}
        mo = ctl._morph
        for n in cont:
            v = ctl.base[n]
            if mo and n in mo["names"]:
                v = mo["to"][n]
            for lm in ctl._lane_morph.values():
                if n in lm["names"]:
                    v = lm["to"][n]
            b[n] = v
        b_mods = copy.deepcopy(ctl.mods)
        priv_set = set(priv)
        changed = [n for n in priv if abs(b[n] - a_base[n]) > 1e-9 or b_mods.get(n) != a_mods.get(n)]
        if not changed:                         # nothing the scene shader draws differs: the recall's own morph is it
            return
        bpm = getattr(ctl, "bpm", 124.0)
        dur = seconds(spec, bpm)
        beats = dur * bpm / 60.0
        allc = set(cont)
        for lm in ctl._lane_morph.values():     # the BLEND owns every fader now (a lane's stack morph keeps going)
            lm["names"] -= allc
        hold = set(priv_set)
        morph_names = set()
        for n in cont:
            if n in priv_set:
                ctl.base[n] = a_base[n]
                continue
            keys = [k for k in GROUP_PASS.get(params[n][3], ()) if k in params]
            if not keys:
                morph_names.add(n)              # master, filters, unknown groups: glide
                continue
            a_on = any(a_base.get(k, 0.0) > 1e-3 for k in keys)
            b_on = any(b.get(k, 0.0) > 1e-3 for k in keys)
            if a_on and b_on:
                morph_names.add(n)              # both looks show this pass: its params morph (pass conflict rule)
                ctl.base[n] = a_base[n]
            elif a_on:
                ctl.base[n] = a_base[n]         # only A shows it: it holds until the end
                hold.add(n)
            else:
                ctl.base[n] = b[n]              # only B (or nobody) shows it: B's values at once
                ctl._sm[n] = b[n]
        mods = {n: p for n, p in b_mods.items() if n not in hold}
        mods.update({n: p for n, p in a_mods.items() if n in hold})
        ctl.mods = mods
        own = dict(src={n: a_base[n] for n in cont}, to=dict(b), names=morph_names, start=ctl.beat, dur=max(beats, 1e-3),
                   old=[(a_mods, 1.0)] if a_mods else [], t=0.0)
        ctl._morph = own                        # also keeps a snapshot recall's stack morph stepping (_step_morph)
        self.active = dict(spec=spec, label=label, t=0.0, el=0.0, dur=dur, b=b, b_mods=b_mods, a_base=a_base,
                           a_mods=a_mods, hold=hold, priv=priv, own=own, hurry=None, frames=0)
        self.last = spec
        self.stats["started"] += 1

    # ---- per-frame ------------------------------------------------------------------------
    def step(self, dt):
        if self.active is None and self.dip is None:
            return
        self._clock += dt
        d = self.dip
        if d is not None:
            d["t"] += dt
            if d.get("commit") and d["t"] >= 0.5 * DIP_S and self.active is not None:
                self.commit()                   # critical strain: the swap happens at the dark point
            if d["t"] >= DIP_S:
                self.dip = None
                if self.active is None:
                    self._run_queued()
        a = self.active
        if a is None:
            return
        a["el"] += dt
        if a["hurry"] is not None:
            el0, t0 = a["hurry"]
            a["t"] = t0 + (1.0 - t0) * min(1.0, (a["el"] - el0) / HURRY_S)
        else:
            a["t"] = min(1.0, a["el"] / a["dur"])
        if getattr(self.ctl, "gov_level", 0) >= 2 and self.dip is None:
            self.stats["fallbacks"] += 1
            self._start_dip(commit=True)        # critical: no more B renders than needed
            return
        if a["t"] >= 1.0:
            self.commit()
            self.stats["finished"] += 1
            self._run_queued()

    def _run_queued(self):
        q, self.queued = self.queued, None
        if q is not None:
            self.run(*q)

    def _hurry(self):
        a = self.active
        if a is not None and a["hurry"] is None:
            a["hurry"] = (a["el"], a["t"])

    def _start_dip(self, commit):
        now = self.ctl.now() if hasattr(self.ctl, "now") else self._clock
        if now - self._last_dip < DIP_GAP_S and not commit:
            return                              # never two dips within a second (photosensitive)
        self._last_dip = now
        self.dip = dict(t=0.0, commit=commit)

    def dip_mul(self):
        """Master multiplier for the dip (1 when none): smooth down to 0.12 and back over DIP_S."""
        d = self.dip
        if d is None:
            return 1.0
        u = min(1.0, max(0.0, d["t"] / DIP_S))
        return 1.0 - 0.88 * math.sin(math.pi * u) ** 2

    def commit(self):
        """B becomes the live look (private params snap; the slew is skipped so the next frame == the last blend)."""
        a, self.active = self.active, None
        if a is None:
            return
        ctl = self.ctl
        b = a["b"]
        for n in a["hold"]:
            ctl.base[n] = b[n]
            ctl._sm[n] = b[n]
        if "palette" in a["hold"]:
            ctl._pal_target = b["palette"]
        mo = a["own"]
        if ctl._morph is mo:
            for n in mo["names"]:
                ctl.base[n] = mo["to"][n]
            if getattr(ctl, "_stack_morph", False) and ctl.stack is not None:
                try:
                    ctl.stack.morph_to(1.0)
                except Exception:
                    pass
                ctl._stack_morph = False
            ctl._morph = None
        ctl.mods = copy.deepcopy(a["b_mods"])

    def cancel(self, revert=True):
        """Stop the running BLEND now. revert=True: A stays exactly (shared params go back too)."""
        a, self.active = self.active, None
        self.queued = None
        if self.dip is not None and self.dip.get("commit"):
            self.dip = None
        if a is None:
            return False
        ctl = self.ctl
        if ctl._morph is a["own"]:
            ctl._morph = None
        if revert:
            for n, v in a["a_base"].items():
                if n in ctl.base:
                    ctl.base[n] = v
            ctl.mods = copy.deepcopy(a["a_mods"])
        self.stats["cancelled"] += 1
        return True

    def finish(self):
        """Jump to the end (B now)."""
        if self.active is not None:
            self.commit()
            self.stats["finished"] += 1
            self._run_queued()

    def panic(self):
        """PANIC: no transition, queue or dip survives; the safe look follows (Controller.panic)."""
        self.cancel(revert=False)
        self.queued = None
        self.dip = None

    def engine_failed(self):
        """The GL side could not render B: finish at once (B), never leave A frozen half-way."""
        self.stats["failed"] += 1
        self.engine_ok = False
        self.finish()

    # ---- what the engine reads -----------------------------------------------------------
    def frame(self, vals):
        """None when idle. Else {spec, e, t, b (B's frame values), pass (values for the offscreen passes), priv,
        scale, origin}. vals = A's frame values from Controller.update()."""
        a = self.active
        if a is None:
            return None
        ctl = self.ctl
        params = _ctl_params(ctl)
        beat = ctl.beat
        w = {}
        byp = getattr(ctl, "_byp", None)
        if byp:
            cover = _bypass_cover()
            for key, lvl in byp.items():
                for n in cover.get(key, ()):
                    if lvl > w.get(n, 0.0):
                        w[n] = lvl
        neutral = _neutral()
        b = dict(vals)
        bt, bm = a["b"], a["b_mods"]
        for n in a["priv"]:
            lo, hi = params[n][0], params[n][1]
            v = min(hi, max(lo, bt[n] + ctl._mod_sum(bm, n, beat) * (hi - lo)))
            if n in w:
                v += (neutral.get(n, v) - v) * w[n]
            b[n] = v
        pv = dict(vals)
        for k in PASS_AMOUNTS:
            if k in vals:
                pv[k] = max(vals[k], b.get(k, 0.0))
        spec = a["spec"]
        a["frames"] += 1
        return dict(spec=spec, e=ease(spec["ease"], a["t"]), t=a["t"], b=b, pass_vals=pv, priv=a["priv"],
                    scale=SCALE_STRAINED if getattr(ctl, "gov_level", 0) >= 1 else spec["scale"],
                    origin=tuple(spec.get("origin") or self.origin))

    def state(self):
        a = self.active
        st = self.stats                         # compact (the panel polls state() several times a second)
        out = dict(default=self.default, origin=[round(x, 3) for x in self.origin], active=None)
        if self.queued is not None or self.dip is not None or not self.engine_ok:
            out.update(queued=self.queued is not None, dip=self.dip is not None, engine=self.engine_ok)
        if st["started"] or st["fallbacks"]:
            out["n"] = [st[k] for k in ("started", "finished", "cancelled", "fallbacks", "queued", "failed")]
        if a is not None:
            out["active"] = dict(kind=a["spec"]["kind"], t=round(a["t"], 3), dur=round(a["dur"], 2), label=a["label"])
        return out


def _ctl_params(ctl):
    import params as _p
    return _p.PARAMS


def _ctl_cont(ctl):
    import control as _c
    return _c.CONT


def _bypass_cover():
    import control as _c
    return _c.BYPASS_COVER


def _neutral():
    import params as _p
    return _p.NEUTRAL


def meta():
    """For the panel: kinds, names, modes, eases, n ranges, defaults."""
    return dict(kinds=list(KINDS), names=dict(NAMES), modes={k: list(v) for k, v in MODES.items()}, eases=list(EASES),
                n=dict(N_RANGE), defaults={k: dict(v) for k, v in DEFAULTS.items()}, quantize=list(QUANTS))


# ------------------------------------------------------------------------------------------------ GL side
class TransitionPass:
    """B framebuffer + composite. Created by live_visuals.py; allocates its targets on the first transition only.

    render_b(): the scene program drawn once more into the private B target with B's uniforms (only the ones that
    differ are written, then put back, so PreviewTap's later redraws still see A). The scene shader's second output
    (echo energy) lands in a scratch attachment: the echo buffer never sees B.
    composite(): disp -> a private copy (A), then A + B through the transition mask straight back into disp."""

    UNIT_A, UNIT_B = 30, 31                     # private units: never sampled by the scene shader

    def __init__(self, ctx):
        import moderngl
        self.mgl = moderngl
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VS, fragment_shader=COMPOSITE_FS)
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._vao = ctx.vertex_array(self.prog, [(self._quad, "2f", "in_pos")])
        self._b = None                          # (size, fbo, col, scratch)
        self._a = None                          # (size, fbo, tex)
        self._disp = {}                         # id(disp) -> framebuffer([disp])
        self.draws = 0                          # B renders (tests: B only during transitions)
        self.timing = None                      # [(b_ms, comp_ms)] when timing is switched on (bench)
        self._q = None

    def _b_target(self, size):
        if self._b is None or self._b[0] != size:
            if self._b is not None:
                for o in self._b[1:]:
                    o.release()
            col = self.ctx.texture(size, 4, dtype="f2")
            scr = self.ctx.texture(size, 4, dtype="f1")
            col.repeat_x = col.repeat_y = False
            self._b = (size, self.ctx.framebuffer([col, scr]), col, scr)
        return self._b

    def _a_target(self, size):
        if self._a is None or self._a[0] != size:
            if self._a is not None:
                for o in self._a[1:]:
                    o.release()
            tex = self.ctx.texture(size, 4, dtype="f2")
            tex.repeat_x = tex.repeat_y = False
            self._a = (size, self.ctx.framebuffer([tex]), tex)
        return self._a

    @staticmethod
    def _set(prog, k, v):
        if k not in prog:
            return
        if isinstance(v, np.ndarray):
            prog[k].write(v.tobytes())
        else:
            prog[k].value = v

    @staticmethod
    def _same(a, b):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
        return a == b

    def _timer(self):
        if self.timing is None:
            return None
        if self._q is None:
            self._q = (self.ctx.query(time=True), self.ctx.query(time=True))
        return self._q

    def render_b(self, scene_prog, scene_vao, uni, uni_b, pal, pal_b, full_size, scale):
        ctx = self.ctx
        bw, bh = max(16, int(round(full_size[0] * scale))), max(10, int(round(full_size[1] * scale)))
        _, fbo, col, _ = self._b_target((bw, bh))
        diff = [k for k, v in uni_b.items() if k not in uni or not self._same(uni[k], v)]
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        q = self._timer()
        try:
            for k in diff:
                self._set(scene_prog, k, uni_b[k])
            if "pxh" in scene_prog:
                scene_prog["pxh"].value = 1.0 / bh
            if "pal" in scene_prog:
                scene_prog["pal"].write(np.ascontiguousarray(pal_b, dtype="f4").tobytes())
            fbo.use()
            ctx.viewport = (0, 0, bw, bh)
            if q is not None:
                with q[0]:
                    scene_vao.render(self.mgl.TRIANGLE_STRIP)
            else:
                scene_vao.render(self.mgl.TRIANGLE_STRIP)
            self.draws += 1
        finally:
            for k in diff:                      # A's values back: later redraws (panel preview) must see A
                if k in uni:
                    self._set(scene_prog, k, uni[k])
            if "pxh" in scene_prog and "pxh" in uni:
                scene_prog["pxh"].value = uni["pxh"]
            if "pal" in scene_prog:
                scene_prog["pal"].write(np.ascontiguousarray(pal, dtype="f4").tobytes())
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:
                ctx.screen.use()
        return col

    def composite(self, disp, fr, aspect, b_tex=None):
        """disp (A) and the B texture -> disp, through fr["spec"]'s mask at fr["e"]."""
        ctx = self.ctx
        size = disp.size
        _, afbo, atex = self._a_target(size)
        dfbo = self._disp.get(id(disp))
        if dfbo is None or dfbo[0] is not disp:
            dfbo = (disp, ctx.framebuffer([disp]))
            self._disp[id(disp)] = dfbo
        b_tex = b_tex if b_tex is not None else self._b[2]
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        q = self._timer()
        try:
            if q is not None:
                q[1].__enter__()
            ctx.copy_framebuffer(afbo, dfbo[1])
            reveal = fr["spec"]["kind"] == "reveal"
            if reveal:
                atex.build_mipmaps()
                atex.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
            else:
                atex.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
            U = uniforms(fr["spec"], fr["e"], fr.get("origin"), aspect)
            p = self.prog
            for k, v in U.items():
                if k in p:
                    p[k].value = v
            if "pxA" in p:
                p["pxA"].value = (1.0 / size[0], 1.0 / size[1])
            atex.use(self.UNIT_A)
            b_tex.use(self.UNIT_B)
            p["texA"].value, p["texB"].value = self.UNIT_A, self.UNIT_B
            dfbo[1].use()
            ctx.viewport = (0, 0) + tuple(size)
            self._vao.render(self.mgl.TRIANGLE_STRIP)
            if q is not None:
                q[1].__exit__(None, None, None)
                ctx.finish()
                self.timing.append((q[0].elapsed / 1e6, q[1].elapsed / 1e6))
        finally:
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:
                ctx.screen.use()

    def release(self):
        for t in (self._a, self._b):
            if t is not None:
                for o in t[1:]:
                    o.release()
        self._a = self._b = None
        for _, f in self._disp.values():
            f.release()
        self._disp = {}
