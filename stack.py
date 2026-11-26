"""Math stack: up to 8 hands-on procedural layers (stripes, rings, spirals, rays, checkers, waves, polygons,
noise bands), each with its own coordinate space, band shape, motion, colour gradient and blend mode.

Two halves, kept apart so control.py can import this file without any GL:

* the MODEL (pure Python): `StackState` holds the layers as plain JSON-able dicts, runs the validated
  `{"cmd": "stack", "op": ...}` commands, blends two stacks for scene morphs, and once per frame `step()`
  integrates every phase in float64 and resolves each layer to the numbers the shader needs (`.frame`).
* the PASS (`StackPass`, moderngl imported lazily): ONE fragment shader looping over the active layers into an
  RGBA8 texture. rgb = light (already scaled by `l_stack`), a = how much of the engine's own picture underneath
  is hidden (mask / multiply / replace / over modes). live_visuals.py composites it as  col = col*(1-a) + rgb.

Float precision / long shows. Nothing the GPU sees grows with time:
  scroll      raw cycles in float64 (never wrapped on the CPU: 1000 h at the top speed is still exact to 1e-9),
              sent as  raw mod W  where W is a whole number of everything the band index feeds (see `_resolve`):
              the colour cycle, the per-band hash and, in the circular spaces, the angular band count
  spin        radians, wrapped at TAU          warp, hue, free pulse   wrapped at 1
In the circular spaces (polar, tunnel) and for the radial generators every angular frequency is snapped to a
whole number on the CPU, and the shader takes derivatives across both branch cuts of atan, so there is no seam.

Photosensitive limits (pure Python, tested): a layer whose bands are so wide that the whole frame would blink
is capped at SAFE_HZ band passes per second (finer patterns may run faster, they do not change the frame's mean
level); a pulse never repeats faster than SAFE_HZ (a beat-synced one doubles its beat count instead); opacity and
brightness are slewed so even a square LFO cannot blink a layer faster than that; hue cycling is capped at 0.5 Hz.

Run  python stack.py --gallery  for stills of every generator / space (test_frames/stack/, small, headless).
"""
import copy
import math

try:
    from params import MOD_SOURCES
except Exception:                                    # stand-alone use
    MOD_SOURCES = ["none", "bass", "mid", "high", "lfo_sine", "lfo_tri", "lfo_saw", "lfo_square", "random",
                   "seq", "seq_gate"]

TAU = 2.0 * math.pi
MAX_LAYERS = 8
STOP_MIN, STOP_MAX = 2, 6
SAFE_HZ = 2.5                    # full-frame changes per second a layer may make (WCAG 2.3.1 allows 3)
FAST_HZ = 8.0                    # band passes per second for fine patterns (60 fps still resolves them)
HUE_HZ = 0.5
SLEW = 2.0 * SAFE_HZ             # full ranges per second: a full swing there and back takes 1 / SAFE_HZ
OCT_SLEW = 15.0                  # octaves per second for the log-scaled fields (band count, zoom)
# Every numeric field that changes what is on the wall is slewed AFTER modulation (see StackState._limit): a square
# LFO, a MIDI jump or a typed value moves it at most one full range per 1 / SLEW seconds, so nothing can flip the
# frame's level or colour faster than SAFE_HZ. The rates (speed, spin, warp_speed, hue_speed, pulse_div) only set
# how fast a phase runs; they have their own caps in _resolve.
RATE_FIELDS = ("speed", "spin", "warp_speed", "hue_speed", "pulse_div")
OCT_FIELDS = ("freq", "warp_freq")               # limited in log2 space; zoom is already in octaves
PAL_N = 5                        # colours in a show palette (live_visuals.PALETTES)
PAL_FALLBACK = [(0.00, 0.85, 0.80), (1.00, 0.05, 0.55), (1.00, 0.70, 0.05), (0.28, 0.10, 1.00), (0.10, 0.90, 0.30)]
WRAP = 60                        # scroll wrap (cycles) when nothing angular is involved: LCM of 2..6 colour stops
VERSION = 1

GENERATORS = ["Stripes", "Rings", "Spiral", "Rays", "Checker", "Waves", "Polygons", "Noise bands"]
SPACES = ["Straight", "Polar", "Tunnel", "Kaleidoscope", "Mirror X", "Mirror Y", "Mirror XY", "Floor"]
WAVES = ["Sine", "Square", "Triangle", "Saw"]
BLENDS = ["Add", "Screen", "Multiply", "Difference", "Mask below", "Replace", "Over"]
COLORS = ["Gradient", "Show palette"]
SYNCS = ["Free", "Beat"]
G_STRIPES, G_RINGS, G_SPIRAL, G_RAYS, G_CHECKER, G_WAVES, G_POLY, G_NOISE = range(8)
S_STRAIGHT, S_POLAR, S_TUNNEL, S_KALEIDO, S_MIRX, S_MIRY, S_MIRXY, S_FLOOR = range(8)
LINEAR = (G_STRIPES, G_WAVES, G_NOISE)               # band coordinate is linear in the pattern plane
RADIAL = (G_RINGS, G_SPIRAL, G_RAYS, G_POLY)         # ... is built from radius + angle round the layer's centre
PERIODIC = (S_POLAR, S_TUNNEL)                       # the pattern plane's u axis is an angle (period 1)

# numeric fields: name -> (min, max, default, label, hint). Scene morphs interpolate them; MIDI / mods address them.
NUM = {
    "freq":       (0.25, 48.0, 6.0, "Bands", "how many bands fit the frame height"),
    "duty":       (0.02, 0.98, 0.5, "Band width", "lit part of each band"),
    "soft":       (0.0, 1.0, 0.08, "Edge softness", "0 = razor edge (still antialiased)"),
    "angle":      (0.0, 1.0, 0.0, "Direction", "which way the bands run and travel"),
    "phase":      (0.0, 1.0, 0.0, "Phase offset", "slides this layer against the others, in bands"),
    "speed":      (-4.0, 4.0, 0.25, "Scroll speed", "bands per second (per beat with BEAT sync)"),
    "spin":       (-1.0, 1.0, 0.0, "Rotation speed", "turns per second; 0 = settles back upright"),
    "zoom":       (-2.0, 2.0, 0.0, "Zoom", "octaves: +1 = twice as big"),
    "pulse":      (0.0, 1.0, 0.0, "Pulse", "breathing kick in size + brightness"),
    "pulse_div":  (0.25, 16.0, 1.0, "Pulse every", "beats (BEAT sync) or half seconds (FREE)"),
    "warp":       (0.0, 1.0, 0.0, "Wave amount", "bends the bands into ribbons / flowers / stars"),
    "warp_freq":  (0.25, 8.0, 1.5, "Wave length", "ripples per frame height"),
    "warp_speed": (-2.0, 2.0, 0.2, "Wave speed", "ripple cycles per second"),
    "cx":         (-1.0, 1.0, 0.0, "Centre X", ""),
    "cy":         (-1.0, 1.0, 0.0, "Centre Y", ""),
    "hue":        (0.0, 1.0, 0.0, "Hue shift", ""),
    "hue_speed":  (-HUE_HZ, HUE_HZ, 0.0, "Hue cycle speed", "colour-wheel turns per second"),
    "sat":        (0.0, 1.5, 1.0, "Saturation", ""),
    "bright":     (0.0, 2.0, 1.0, "Brightness", ""),
    "grad":       (0.0, 1.0, 0.0, "Colour blend", "0 = each band one flat colour, 1 = smooth gradient"),
    "opacity":    (0.0, 1.0, 1.0, "Opacity", ""),
}
# stepped fields: name -> (min, max, default, label, option names | None). Scene morphs snap them at the midpoint.
INT = {
    "gen":   (0, len(GENERATORS) - 1, 0, "Generator", GENERATORS),
    "space": (0, len(SPACES) - 1, 0, "Space", SPACES),
    "wave":  (0, len(WAVES) - 1, 1, "Waveform", WAVES),
    "blend": (0, len(BLENDS) - 1, 1, "Blend", BLENDS),
    "color": (0, len(COLORS) - 1, 0, "Colour source", COLORS),
    "sync":  (0, len(SYNCS) - 1, 0, "Sync", SYNCS),
    "count": (1, 32, 6, "Arms / rays / sides", None),
    "folds": (2, 24, 6, "Folds", None),
    "mute":  (0, 1, 0, "Mute", None),
    "solo":  (0, 1, 0, "Solo", None),
}
FIELDS = tuple(NUM) + tuple(INT)
OPS = ("add", "set", "del", "move", "dup", "clear", "mod", "reset", "replace")

JEWEL = [[0.0, 0.00, 0.85, 0.80], [0.25, 1.00, 0.05, 0.55], [0.5, 1.00, 0.70, 0.05], [0.75, 0.28, 0.10, 1.00]]
# what "+ ADD LAYER" gives for each generator: a look that reads at once and invites a tweak
GEN_PRESETS = {
    G_STRIPES: dict(freq=5.0, angle=0.08, speed=0.25, soft=0.12, stops=JEWEL),
    G_RINGS: dict(freq=5.0, speed=0.3, soft=0.2, duty=0.45,
                  stops=[[0.0, 1.00, 0.68, 0.08], [0.33, 1.00, 0.30, 0.02], [0.66, 0.55, 0.08, 1.00]]),
    G_SPIRAL: dict(freq=3.0, count=4, speed=0.3, wave=1, soft=0.35, duty=0.5, grad=1.0,
                   stops=[[0.0, 0.05, 0.85, 1.00], [0.5, 0.95, 0.10, 0.75]]),
    G_RAYS: dict(count=12, spin=0.03, speed=0.0, soft=0.06, duty=0.5,
                 stops=[[0.0, 1.00, 0.78, 0.10], [0.5, 1.00, 0.28, 0.04]]),
    G_CHECKER: dict(freq=6.0, speed=0.2, angle=0.125, soft=0.0,
                    stops=[[0.0, 0.95, 0.95, 1.00], [0.5, 0.10, 0.75, 1.00]]),
    G_WAVES: dict(freq=5.0, wave=0, duty=0.3, warp=0.5, warp_freq=1.5, warp_speed=0.25, speed=0.15, grad=1.0,
                  stops=[[0.0, 0.00, 0.90, 0.75], [0.35, 0.05, 0.40, 1.00], [0.7, 0.60, 0.10, 1.00]]),
    G_POLY: dict(freq=4.0, count=6, speed=0.25, soft=0.1, duty=0.4,
                 stops=[[0.0, 0.00, 0.85, 0.35], [0.4, 0.55, 1.00, 0.12], [0.75, 1.00, 0.78, 0.08]]),
    G_NOISE: dict(freq=14.0, speed=0.4, soft=0.05, angle=0.25,
                  stops=[[0.0, 1.00, 0.06, 0.10], [0.2, 1.00, 0.80, 0.00], [0.4, 0.05, 0.95, 0.25],
                         [0.6, 0.05, 0.40, 1.00], [0.8, 0.85, 0.05, 1.00]]),
}
CANVAS_START = dict(gen=G_STRIPES, name="Stripes", freq=5.0, angle=0.08, speed=0.25, soft=0.12, stops=JEWEL)


# ------------------------------------------------------------------------------------------------ validation


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def clean_stops(raw, fallback=None):
    """2..6 colour stops [[pos 0..1, r, g, b], ...] sorted by position; anything malformed -> fallback / JEWEL."""
    out = []
    if isinstance(raw, (list, tuple)):
        for s in raw[:STOP_MAX]:
            if isinstance(s, (list, tuple)) and len(s) == 4 and all(_isnum(x) for x in s):
                out.append([round(min(1.0, max(0.0, float(x))), 4) for x in s])
    if len(out) < STOP_MIN:
        return [list(s) for s in (fallback or JEWEL)]
    out.sort(key=lambda s: s[0])
    return out


def clean_field(name, v):
    """One field value clamped to its range (ints rounded); ValueError for an unknown field or a non-number."""
    if name in NUM:
        if not _isnum(v):
            raise ValueError(f"'{name}' must be a finite number")
        return min(NUM[name][1], max(NUM[name][0], float(v)))
    if name in INT:
        if isinstance(v, bool):
            v = int(v)
        if not _isnum(v):
            raise ValueError(f"'{name}' must be a number")
        return int(min(INT[name][1], max(INT[name][0], round(float(v)))))
    raise ValueError(f"unknown stack field '{str(name)[:24]}'")


def default_layer(gen=0):
    lay = {k: v[2] for k, v in NUM.items()}
    lay.update({k: v[2] for k, v in INT.items()})
    lay["gen"] = clean_field("gen", gen)
    lay["stops"] = [list(s) for s in JEWEL]
    lay["name"] = GENERATORS[lay["gen"]]
    pre = GEN_PRESETS.get(lay["gen"], {})
    for k, v in pre.items():
        lay[k] = clean_stops(v) if k == "stops" else clean_field(k, v)
    return lay


def clean_layer(raw, base=None):
    """A layer dict from untrusted JSON: every field present and in range. Unknown keys are dropped, bad values
    fall back to `base` (or the generator's defaults). Never raises."""
    raw = raw if isinstance(raw, dict) else {}
    gen = raw.get("gen", base["gen"] if base else 0)
    lay = dict(base) if base else default_layer(gen if _isnum(gen) else 0)
    for k in FIELDS:
        if k in raw:
            try:
                lay[k] = clean_field(k, raw[k])
            except ValueError:
                pass
    if "stops" in raw:
        lay["stops"] = clean_stops(raw["stops"], lay.get("stops"))
    else:
        lay["stops"] = [list(s) for s in lay["stops"]]
    name = raw.get("name", lay.get("name"))
    lay["name"] = str(name)[:24] if isinstance(name, str) and name.strip() else GENERATORS[lay["gen"]]
    i = raw.get("id", lay.get("id"))
    if isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= 999999:
        lay["id"] = i
    else:
        lay.pop("id", None)
    return lay


def clean_mod(m):
    """One modulation slot (same shape as control.py's) or None."""
    if not isinstance(m, dict) or m.get("src") not in MOD_SOURCES or m["src"] == "none":
        return None
    out = dict(src=m["src"])
    for k, lo, hi, d in (("amt", -1.0, 1.0, 0.0), ("rate", 1 / 64, 64.0, 1.0), ("phase", 0.0, 1.0, 0.0)):
        v = m.get(k, d)
        if not _isnum(v):
            return None
        out[k] = min(hi, max(lo, float(v)))
    return out


def clean_stack(raw):
    """A whole stack from untrusted JSON -> {"v", "next", "layers", "mods"}; ids unique, at most 8 layers."""
    raw = raw if isinstance(raw, dict) else {}
    layers, seen = [], set()
    src = raw.get("layers")
    for item in (src if isinstance(src, list) else [])[:MAX_LAYERS]:
        if not isinstance(item, dict):
            continue
        lay = clean_layer(item)
        if lay.get("id") in seen:
            lay.pop("id", None)
        if "id" in lay:
            seen.add(lay["id"])
        layers.append(lay)
    nxt = raw.get("next")
    nxt = nxt if isinstance(nxt, int) and not isinstance(nxt, bool) and 1 <= nxt <= 999999 else 1
    nxt = max([nxt] + [i + 1 for i in seen])
    for lay in layers:
        if "id" not in lay:
            lay["id"], nxt = nxt, nxt + 1
    ids = {lay["id"] for lay in layers}
    mods = {}
    if isinstance(raw.get("mods"), dict):
        for key, slots in raw["mods"].items():
            t = parse_key(key)
            if t and t[0] in ids and t[1] in NUM and isinstance(slots, list):
                pair = [clean_mod(m) for m in slots[:2]]
                pair += [None] * (2 - len(pair))
                if any(pair):
                    mods[f"{t[0]}:{t[1]}"] = pair
    return dict(v=VERSION, next=nxt, layers=layers, mods=mods)


def parse_key(key):
    """"<layer id>:<field>" -> (id, field) or None."""
    if not isinstance(key, str):
        return None
    a, _, b = key.partition(":")
    if a.isdigit() and len(a) <= 6 and b in FIELDS and int(a) >= 1:
        return int(a), b
    return None


def parse_target(t):
    """MIDI-learn / modulation target "stack:<layer id>:<field>" -> (id, field) or None."""
    if isinstance(t, str) and t.startswith("stack:"):
        return parse_key(t[6:])
    return None


def _cmd_id(c):
    v = c.get("id")
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 999999:
        raise ValueError("'id' must be a layer id")
    return v


def validate(c):
    """See _validate. Anything malformed (unhashable keys included) is a ValueError, so the panel server answers 400."""
    try:
        return _validate(c)
    except (TypeError, KeyError, AttributeError, IndexError) as ex:
        raise ValueError(f"malformed stack command ({type(ex).__name__})")


def _validate(c):
    """Normalised copy of one {"cmd": "stack", "op": ...} command (clamped, unknown keys dropped); ValueError if
    malformed. Runs on the panel's server thread, so nothing unvalidated reaches the render loop.

      add     {gen?, layer? {fields}, after? id}        new layer (max 8); `layer` overrides the generator's preset
      set     {id, f, v} | {id, set {field: value}}     fields incl. "stops" [[pos, r, g, b] x 2..6] and "name"
      del     {id}        dup {id}        move {id, to index}        clear
      reset   {id, f?}                                  one field (or the whole layer) back to the generator default
      mod     {id, f, slot 0|1, src, amt, rate, phase}  modulation slot for a numeric field
      replace {stack}                                   the whole stack (undo, presets)
    """
    if not isinstance(c, dict) or c.get("op") not in OPS:
        raise ValueError("'op' must be one of " + ", ".join(OPS))
    op = c["op"]
    out = {"cmd": "stack", "op": op}
    if op == "clear":
        return out
    if op == "replace":
        if not isinstance(c.get("stack"), dict):
            raise ValueError("'stack' must be an object")
        out["stack"] = clean_stack(c["stack"])
        return out
    if op == "add":
        gen = c.get("gen", 0)
        lay = c.get("layer") if isinstance(c.get("layer"), dict) else {}
        if "gen" in lay:
            gen = lay["gen"]
        out["gen"] = clean_field("gen", gen)
        out["layer"] = {k: v for k, v in clean_layer(dict(lay, gen=out["gen"])).items()
                        if k in lay and k not in ("id",)}
        if c.get("after") is not None:
            out["after"] = _cmd_id(dict(id=c["after"]))
        return out
    out["id"] = _cmd_id(c)
    if op == "set":
        vals = {}
        src = dict(c["set"]) if isinstance(c.get("set"), dict) else {}
        if "f" in c:
            if not isinstance(c["f"], str):
                raise ValueError("'f' must be a field name")
            src[c["f"]] = c.get("v")
        for k, v in src.items():
            if not isinstance(k, str):
                raise ValueError("field names must be text")
            if k == "stops":
                if not (isinstance(v, list) and STOP_MIN <= len(v) <= STOP_MAX and all(
                        isinstance(s, (list, tuple)) and len(s) == 4 and all(_isnum(x) for x in s) for s in v)):
                    raise ValueError(f"'stops' must be {STOP_MIN}..{STOP_MAX} [pos, r, g, b] entries")
                vals[k] = clean_stops(v)
            elif k == "name":
                if not isinstance(v, str):
                    raise ValueError("'name' must be text")
                vals[k] = v.strip()[:24]
            else:
                vals[str(k)] = clean_field(k, v)
        if not vals:
            raise ValueError("'set' needs at least one field")
        out["set"] = vals
    elif op == "move":
        v = c.get("to")
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < MAX_LAYERS:
            raise ValueError("'to' must be a position 0..7")
        out["to"] = v
    elif op == "reset":
        if c.get("f") is not None:
            if c["f"] not in FIELDS and c["f"] != "stops":
                raise ValueError("unknown stack field")
            out["f"] = c["f"]
    elif op == "mod":
        if c.get("f") not in NUM:
            raise ValueError("'f' must be a numeric stack field")
        slot = c.get("slot")
        if isinstance(slot, bool) or slot not in (0, 1):
            raise ValueError("'slot' must be 0 or 1")
        if "src" not in c and "mod" in c and (c["mod"] is None or isinstance(c["mod"], dict)):
            c = dict(c["mod"] or dict(src="none"), **{k: c[k] for k in ("f", "slot")})     # already normalised once
        if c.get("src") not in MOD_SOURCES:
            raise ValueError("unknown src")
        m = clean_mod(dict(src=c["src"], amt=c.get("amt", 0.0), rate=c.get("rate", 1.0), phase=c.get("phase", 0.0)))
        if m is None and c["src"] != "none":
            raise ValueError("bad modulation values")
        out.update(f=c["f"], slot=slot, mod=m)
    return out


# ------------------------------------------------------------------------------------------------ safety maths


def visible_bands(lay):
    """Roughly how many bands of this layer are on screen at once. Below about 1.5 the layer's scroll changes the
    whole frame's level, so it is rate-limited like a flash; from 3 up it is a moving texture."""
    gen, space = lay["gen"], lay["space"]
    f = lay["freq"] * 2.0 ** (-lay["zoom"])
    if gen == G_RAYS:
        n = float(lay["count"])
    elif gen == G_SPIRAL:
        n = f + lay["count"]
    else:
        n = f
    if space in (S_TUNNEL, S_FLOOR):                 # log / perspective spacing: always many bands in view
        n = max(n, 3.0)
    return n


def safe_rate(bands, hz):
    """Band passes per second this layer may run at: SAFE_HZ while it would blink the frame, FAST_HZ once fine."""
    k = min(1.0, max(0.0, (bands - 1.5) / 1.5))
    cap = SAFE_HZ + (FAST_HZ - SAFE_HZ) * k
    return min(cap, max(-cap, hz))


def pulse_period(div, sync, bpm):
    """Seconds between pulses, never shorter than 1 / SAFE_HZ. Beat-synced pulses stay musical: the beat count
    doubles until it fits. Returns (seconds, beats per pulse or None when free-running)."""
    floor = 1.0 / SAFE_HZ
    if sync:
        beat_s = 60.0 / min(300.0, max(20.0, bpm))
        beats = max(0.25, float(div))
        while beats * beat_s < floor - 1e-9:
            beats *= 2.0
        return beats * beat_s, beats
    return max(floor, 0.5 * float(div)), None


# ------------------------------------------------------------------------------------------------ the model


def blend(a, b, t):
    """Stack `a` morphing into stack `b` (both clean): numeric fields interpolate, stepped fields and names snap
    at the midpoint, a layer only one side has fades in / out, stops interpolate when the counts match."""
    t = min(1.0, max(0.0, float(t)))
    if t <= 0.0:
        return copy.deepcopy(a)
    if t >= 1.0:
        return copy.deepcopy(b)
    la, lb = {x["id"]: x for x in a["layers"]}, {x["id"]: x for x in b["layers"]}
    # Slots: a layer both sides have (same id) glides. The layers only one side has are paired in pile order (ids are
    # never reused, so two scenes built separately share none): a pair dips to nothing at the midpoint and comes back
    # as the other side's layer, the rest fade. So a morph never needs more than max(len a, len b) <= 8 layers.
    only_a = [x for x in a["layers"] if x["id"] not in lb]
    only_b = [x for x in b["layers"] if x["id"] not in la]
    mate_a = {x["id"]: y for x, y in zip(only_a, only_b)}
    mate_b = {y["id"]: x for x, y in zip(only_a, only_b)}
    first_a = t < 0.5
    slots = []
    for i, x in enumerate(a["layers"] if first_a else b["layers"]):
        other = (lb.get(x["id"]) or mate_a.get(x["id"])) if first_a else (la.get(x["id"]) or mate_b.get(x["id"]))
        slots.append((i, 0, x if first_a else other, other if first_a else x))
    for i, x in enumerate(only_b[len(only_a):] if first_a else only_a[len(only_b):]):
        pos = (b["layers"] if first_a else a["layers"]).index(x)
        slots.append((pos, 1, None if first_a else x, x if first_a else None))
    slots.sort(key=lambda s: (s[0], s[1]))
    out = []
    for _, _, xa, xb in slots[:MAX_LAYERS]:
        if xa is None or xb is None:
            lay = copy.deepcopy(xa or xb)
            lay["opacity"] = lay["opacity"] * ((1.0 - t) if xb is None else t)
            out.append(lay)
            continue
        lay = copy.deepcopy(xb if t >= 0.5 else xa)
        for k in NUM:
            lay[k] = xa[k] + (xb[k] - xa[k]) * t
        if len(xa["stops"]) == len(xb["stops"]):
            lay["stops"] = [[round(p + (q - p) * t, 4) for p, q in zip(sa, sb)] for sa, sb in zip(xa["stops"], xb["stops"])]
        if xa["id"] != xb["id"]:                     # two different layers sharing a slot: invisible while they swap
            lay["opacity"] *= abs(2.0 * t - 1.0)
        out.append(lay)
    side = b if t >= 0.5 else a
    return dict(v=VERSION, next=max(a["next"], b["next"]), layers=out, mods=copy.deepcopy(side["mods"]))


class StackState:
    """The stack on show. All methods run on the render thread (commands arrive validated)."""

    def __init__(self):
        self.layers, self.mods, self.next = [], {}, 1
        self._ph = {}                # layer id -> phases (float64): scroll, spin, warp, hue, pulse, op, bright
        self._ghost = []             # deleted layers fading out: [index hint, layer]
        self._morph = None           # (from stack, to stack) while a scene recall is morphing
        self.frame = []              # resolved layers for the shader, bottom first (see _resolve)
        self.live = {}               # "id:field" -> modulated value, for the panel's live read-out
        self.capped = []             # ids whose scroll / spin was held back by the photosensitive limit this frame
        self.edits = 0               # bumps on every structural change (panel: when to rebuild its list)

    # ---- (de)serialisation
    def to_json(self):
        return dict(v=VERSION, next=self.next, layers=copy.deepcopy(self.layers), mods=copy.deepcopy(self.mods))

    def _sig(self):
        return [(x["id"], x["name"], len(x["stops"])) + tuple(x[k] for k in INT) for x in self.layers]

    def load(self, raw, fade=True, trusted=False):
        """Replace the stack. Layers that vanish fade out (unless fade=False); phases of kept ids carry on.
        trusted=True: `raw` is already clean (a blend of two clean stacks) and is taken over without a copy."""
        st = raw if trusted else clean_stack(raw)
        sig = self._sig()
        keep = {x["id"] for x in st["layers"]}
        if fade:
            for i, lay in enumerate(self.layers):
                if lay["id"] not in keep:
                    self._ghost.append([i, lay])
            self._ghost = self._ghost[-MAX_LAYERS:]
        else:
            self._ghost = []
        self._ghost = [g for g in self._ghost if g[1]["id"] not in keep]
        self.layers, self.mods, self.next = st["layers"], st["mods"], max(self.next, st["next"])
        if self._sig() != sig:                       # the panel rebuilds its list on `edits`: not on every morph frame
            self.edits += 1

    def restore(self, raw):
        """From live_state.json: comes back as it was, no fade-in on the projector."""
        self.load(raw, fade=False)
        for lay in self.layers:
            self._phase(lay["id"])["op"] = None

    def state(self):
        """JSON-able snapshot for the panel."""
        return dict(layers=copy.deepcopy(self.layers), mods=copy.deepcopy(self.mods), live=dict(self.live),
                    capped=list(self.capped), edits=self.edits, max=MAX_LAYERS)

    # ---- morphing (scene recall / crossfader)
    def morph_start(self, to):
        self._morph = (self.to_json(), clean_stack(to))

    def morph_to(self, t):
        if self._morph is None:
            return
        a, b = self._morph
        self.load(blend(a, b, t), trusted=True)
        if t >= 1.0:
            self._morph = None

    def morph_cancel(self):
        self._morph = None

    # ---- editing
    def find(self, i):
        return next((x for x in self.layers if x["id"] == i), None)

    def add(self, gen=0, layer=None, after=None):
        if len(self.layers) >= MAX_LAYERS:
            return None
        lay = clean_layer(dict(layer or {}, gen=gen), default_layer(gen))
        lay["id"], self.next = self.next, self.next + 1
        pos = len(self.layers)
        if after is not None:
            pos = next((k + 1 for k, x in enumerate(self.layers) if x["id"] == after), pos)
        self.layers.insert(pos, lay)
        self.edits += 1
        return lay

    def do(self, c):
        """Run one validated command. Returns True if it changed anything."""
        op = c["op"]
        self._morph = None                           # a hand on the stack ends a running scene morph
        if op == "add":
            return self.add(c.get("gen", 0), c.get("layer"), c.get("after")) is not None
        if op == "clear":
            self.load(dict(layers=[]))
            return True
        if op == "replace":
            self.load(c["stack"])
            return True
        lay = self.find(c["id"])
        if lay is None:
            return False
        if op == "set":
            if "gen" in c["set"] and "name" not in c["set"] and lay["name"] == GENERATORS[lay["gen"]]:
                lay["name"] = GENERATORS[c["set"]["gen"]]         # an untouched name follows the generator
            lay.update(copy.deepcopy(c["set"]))
            if "name" in c["set"] and not lay["name"]:
                lay["name"] = GENERATORS[lay["gen"]]
            if any(k in INT or k in ("stops", "name") for k in c["set"]):
                self.edits += 1
        elif op == "del":
            i = self.layers.index(lay)
            self.layers.pop(i)
            self._ghost = (self._ghost + [[i, lay]])[-MAX_LAYERS:]
            self.mods = {k: v for k, v in self.mods.items() if parse_key(k)[0] != lay["id"]}
            self.edits += 1
        elif op == "dup":
            if len(self.layers) >= MAX_LAYERS:
                return False
            new = copy.deepcopy(lay)
            new["id"], self.next = self.next, self.next + 1
            new["solo"] = 0
            self.layers.insert(self.layers.index(lay) + 1, new)
            for k, v in list(self.mods.items()):
                i, f = parse_key(k)
                if i == lay["id"]:
                    self.mods[f"{new['id']}:{f}"] = copy.deepcopy(v)
            src = self._ph.get(lay["id"])
            if src:                                  # the copy starts exactly on top of the original
                self._ph[new["id"]] = dict(src, op=0.0)
            self.edits += 1
        elif op == "move":
            self.layers.remove(lay)
            self.layers.insert(min(c["to"], len(self.layers)), lay)
            self.edits += 1
        elif op == "reset":
            base = default_layer(lay["gen"])
            f = c.get("f")
            if f is None:
                for k in list(NUM) + ["wave", "blend", "color", "sync", "count", "folds", "space", "stops"]:
                    lay[k] = copy.deepcopy(base[k])
                self.mods = {k: v for k, v in self.mods.items() if parse_key(k)[0] != lay["id"]}
            else:
                lay[f] = copy.deepcopy(base[f])
                self.mods.pop(f"{lay['id']}:{f}", None)
            self.edits += 1
        elif op == "mod":
            key = f"{lay['id']}:{c['f']}"
            pair = self.mods.get(key) or [None, None]
            pair[c["slot"]] = c["mod"]
            if any(pair):
                self.mods[key] = pair
            else:
                self.mods.pop(key, None)
            self.edits += 1
        return True

    def midi(self, i, field, val, press=True, rel=None):
        """A bound knob / pad: val 0..1 across the field's range (rel = relative encoder steps); a pad on mute /
        solo toggles on press."""
        lay = self.find(i)
        if lay is None:
            return
        self._morph = None
        if field in ("mute", "solo"):
            if press:
                lay[field] = 0 if lay[field] else 1
                self.edits += 1
            return
        lo, hi = (NUM.get(field) or INT[field])[:2]
        v = lay[field] + rel * (hi - lo) / 127.0 if rel is not None else lo + (hi - lo) * min(1.0, max(0.0, val))
        lay[field] = clean_field(field, v)
        if field in INT:
            self.edits += 1

    # ---- per frame
    def _phase(self, i):
        ph = self._ph.get(i)
        if ph is None:
            ph = self._ph[i] = dict(scroll=0.0, spin=0.0, warp=0.0, hue=0.0, pulse=0.0, env=0.0, op=0.0, bright=1.0)
        return ph

    def _limit(self, ph, v, dt):
        """Slew every look-changing numeric field of one layer toward its (modulated) value. First sight snaps."""
        cur = ph.get("v")
        if cur is None:
            cur = ph["v"] = {f: v[f] for f in NUM if f not in RATE_FIELDS and f != "opacity"}
        for f, x in cur.items():
            want = v[f]
            if dt > 0 and x != want:
                lo, hi = NUM[f][0], NUM[f][1]
                if f in OCT_FIELDS:
                    d = math.log2(want / x)
                    lim = OCT_SLEW * dt
                    x = want if abs(d) <= lim else x * 2.0 ** math.copysign(lim, d)
                else:
                    lim = (OCT_SLEW if f == "zoom" else SLEW if f == "bright" else (hi - lo) * SLEW) * dt
                    x = want if abs(want - x) <= lim else x + math.copysign(lim, want - x)
                cur[f] = x
            v[f] = cur[f]

    def step(self, dt, clock=None, source=None, palette=None, aspect=1.6, hold=False, k=0):
        """Advance every phase by dt seconds (0 or hold=True: freeze) and resolve the layers for the shader.

        clock    {"bpm", "beat"} from the controller        source   fn(mod, key, slot) -> -1..1 (the mod matrix)
        palette  5 show-palette colours (r, g, b) for layers that follow it
        k        tests: whole wrap periods added to every wrapped phase (the picture must not change)
        Returns self.frame: a list of dicts (bottom layer first), ready for StackPass."""
        dt = float(dt) if isinstance(dt, (int, float)) and math.isfinite(dt) and dt > 0 and not hold else 0.0
        dt = min(dt, 0.5)
        clock = clock or {}
        bpm = clock.get("bpm", 124.0)
        bpm = bpm if _isnum(bpm) and bpm > 0 else 124.0
        beat = clock.get("beat", 0.0)
        beat = beat if _isnum(beat) else 0.0
        solo = any(x["solo"] for x in self.layers)
        items = [(i, x, False) for i, x in enumerate(self.layers)]
        for hint, x in self._ghost:
            items.append((hint - 0.5, x, True))
        items.sort(key=lambda it: it[0])
        frame, live, capped, alive = [], {}, [], set()
        # the shader holds 8: while 8 are still visible (ghosts fading out included) a new layer waits at 0 for a slot
        showing = sum(1 for _, x, _g in items if (self._ph.get(x["id"]) or {}).get("op", 0.0) is None
                      or (self._ph.get(x["id"]) or {}).get("op", 0.0) > 1e-4)
        for _, lay, ghost in items:
            i = lay["id"]
            ph = self._phase(i)
            v = dict(lay)
            if not ghost and source is not None:
                for f in NUM:
                    pair = self.mods.get(f"{i}:{f}")
                    if pair:
                        lo, hi = NUM[f][0], NUM[f][1]
                        m = sum(s["amt"] * source(s, f"stack:{i}:{f}", n) for n, s in enumerate(pair) if s)
                        v[f] = min(hi, max(lo, lay[f] + m * (hi - lo)))
                        live[f"{i}:{f}"] = round(v[f], 4)
            want = 0.0 if ghost or v["mute"] or (solo and not v["solo"]) else v["opacity"]
            if ph["op"] is None:                     # restored from disk: already there
                ph["op"] = want
            elif ph["op"] <= 1e-4 and want > 0.0:
                if showing >= MAX_LAYERS:
                    want = 0.0
                else:
                    showing += 1
            if dt > 0:
                lim = SLEW * dt
                ph["op"] += min(lim, max(-lim, want - ph["op"]))
            self._limit(ph, v, dt)
            ph["bright"] = v["bright"]
            if ghost:
                if ph["op"] <= 1e-4:
                    continue
            alive.add(i)
            res = _resolve(v, ph, dt, bpm, beat, hold, palette, aspect, k)
            if res.pop("capped"):
                capped.append(i)
            if ph["op"] > 1e-4:
                res["ghost"] = ghost
                frame.append(res)
        self._ghost = [g for g in self._ghost if g[1]["id"] in alive]
        for i in [i for i in self._ph if i not in alive and self.find(i) is None]:
            del self._ph[i]
        while len(frame) > MAX_LAYERS:               # over the shader's 8: a fading ghost goes first (the faintest),
            gh = [r for r in frame if r["ghost"]]      # never a live layer from the bottom of the pile
            frame.remove(min(gh, key=lambda r: r["F"][0]) if gh else frame[0])
        self.frame, self.live, self.capped = frame, live, capped
        return self.frame


def _snap(x):
    """Nearest whole number (halves away from zero, unlike Python's banker's rounding: no flicker at .5)."""
    return float(math.floor(abs(x) + 0.5)) * (1.0 if x >= 0 else -1.0)


def structure(v):
    """The frequencies the shader runs this layer with, and the periods its band index must respect.

    Returns dict(fu, fv, gu, gv, wu, wv, kc, D, hp, W, xor):
      fu, fv   band coordinate x = fu*u + fv*v (linear / checker) or x = fu*R + fv*turns (radial: fv = arms)
      gu, gv   checker's second axis; radial: gu = polygon sides, gv = flower lobes (whole)
      wu, wv   the warp's own wave vector (wu whole in the circular spaces)
      kc       how far the band index moves in one trip round the circle (0 = it does not)
      D        colour cycle length in bands (index / D = one trip through the stops)
      hp       period of the per-band hash       W = scroll wrap period: whole, and a multiple of D, hp (and 2 for XOR)
    """
    gen, space, n = v["gen"], v["space"], len(v["stops"])
    per = space in PERIODIC
    f, a = v["freq"], v["angle"] * TAU
    xor = gen == G_CHECKER and v["wave"] in (1, 3)
    wu = wv = gu = gv = 0.0
    if gen in RADIAL:
        arms = float(v["count"]) if gen in (G_SPIRAL, G_RAYS) else 0.0
        fu, fv = (0.0 if gen == G_RAYS else f), arms
        gu, gv = float(max(3, v["count"])), float(max(2, int(_snap(v["warp_freq"] * 2.0))))
        kc = arms
    else:
        fu, fv = f * math.sin(a), f * math.cos(a)
        wu, wv = v["warp_freq"] * math.cos(a), -v["warp_freq"] * math.sin(a)
        if per:
            fu, wu = _snap(fu), _snap(wu)
        kc = fu if per else 0.0
        if gen == G_CHECKER:
            gu, gv = f * math.cos(a), -f * math.sin(a)
            if per:
                gu = _snap(gu)
                if xor and (int(fu) + int(gu)) % 2:  # the two-colour parity must close round the circle
                    gu += 1.0 if f * math.cos(a) >= gu else -1.0
                kc = (fu + gu) / 2.0 if xor else fu + gu
    kc = abs(kc)
    if kc >= 1.0:
        m = max(1.0, _snap(kc / n))
        D, hp, W = kc / m, kc, kc
    else:
        kc, D, hp, W = 0.0, float(n), float(WRAP), float(WRAP)
    if xor:
        W *= 2.0
    return dict(fu=fu, fv=fv, gu=gu, gv=gv, wu=wu, wv=wv, kc=kc, D=D, hp=hp, W=W, xor=xor)


def _resolve(v, ph, dt, bpm, beat, hold, palette, aspect, k=0):
    """One layer (modulated values `v`, phases `ph`) -> the uniform block the shader reads. Advances ph by dt."""
    gen, space = v["gen"], v["space"]
    follow = v["color"] == 1
    if follow:                                       # follow the show palette: always PAL_N stops (the count is part
        pal = list(palette)[:PAL_N] if palette is not None and len(palette) >= PAL_N else PAL_FALLBACK   # of the cycle)
        v = dict(v, stops=[[i / PAL_N, float(c[0]), float(c[1]), float(c[2])] for i, c in enumerate(pal)])
    stops = v["stops"]
    s = structure(v)
    bands = visible_bands(v)
    # scroll: bands per second (per beat when synced), capped while the layer would blink the whole frame
    hz = v["speed"] * (bpm / 60.0 if v["sync"] else 1.0)
    hz_ok = safe_rate(bands, hz)
    capped = abs(hz_ok - hz) > 1e-9
    spin = v["spin"]
    if s["kc"] >= 1.0 and abs(spin) * s["kc"] > FAST_HZ:        # rays sweeping past a point
        spin, capped = math.copysign(FAST_HZ / s["kc"], spin), True
    if s["kc"] >= 1.0 and bands < 3.0 and abs(spin) * s["kc"] > SAFE_HZ:
        spin, capped = math.copysign(SAFE_HZ / s["kc"], spin), True
    if dt > 0:
        ph["scroll"] += hz_ok * dt
        if abs(spin) > 1e-6:
            ph["spin"] = (ph["spin"] + spin * TAU * dt) % TAU
        else:                                        # rotation speed 0: settle back upright the short way round
            d = (ph["spin"] + math.pi) % TAU - math.pi
            ph["spin"] = (d * math.exp(-dt / 0.35)) % TAU if abs(d) > 1e-4 else 0.0
        ph["warp"] = (ph["warp"] + v["warp_speed"] * dt) % 1.0
        ph["hue"] = (ph["hue"] + min(HUE_HZ, max(-HUE_HZ, v["hue_speed"])) * dt) % 1.0
    # pulse: one soft kick per period (attack 50 ms, exponential release); never faster than SAFE_HZ
    env = ph["env"]
    if not hold:
        period, beats = pulse_period(v["pulse_div"], v["sync"], bpm)
        if beats is not None:
            p = (beat / beats) % 1.0
        else:
            ph["pulse"] = (ph["pulse"] + dt / period) % 1.0
            p = ph["pulse"]
        tt = p * period
        env = ph["env"] = min(1.0, tt / 0.05) * math.exp(-max(0.0, tt - 0.05) / (0.22 * period + 0.05))
    pulse = v["pulse"]
    scale = 2.0 ** (-v["zoom"]) * (1.0 - 0.10 * pulse * env)
    bright = max(0.0, ph["bright"]) * (1.0 - 0.45 * pulse * (1.0 - env))
    # the band coordinate runs toward the viewer on the floor / in the tunnel, outward and "with the arrow" elsewhere
    sign = 1.0 if (space in (S_TUNNEL, S_FLOOR) and gen not in RADIAL) else -1.0
    W = s["W"]
    raw = sign * (ph["scroll"] + v["phase"])
    if ph.get("W") not in (None, W) and ph.get("sign") == sign:      # W changed (band count / direction moved in a
        ph["base"] = raw - (ph["sent"] % W)                          # circular space): keep the band index, no re-colour
    elif ph.get("sign") not in (None, sign):
        ph["base"] = 0.0
    sent = (raw - ph.get("base", 0.0)) % W
    ph["W"], ph["sign"], ph["sent"] = W, sign, sent
    phase = sent + k * W
    warp_amp = 0.0
    if gen in RADIAL:
        warp_amp = v["warp"]
    elif gen == G_WAVES:
        warp_amp = (0.35 + 0.65 * v["warp"]) * 0.18 * v["freq"]
    elif gen != G_CHECKER:
        warp_amp = v["warp"] * 0.18 * v["freq"]
    return dict(
        id=v["id"], capped=capped, hz=hz_ok, W=s["W"], pal=follow,
        A=(float(gen), float(space), float(v["wave"]), float(v["blend"])),
        B=(s["fu"], s["fv"], s["gu"], s["gv"]),
        C=(phase, ph["spin"] + k * TAU, scale, v["angle"] * TAU),
        D=(v["cx"] * 0.5 * aspect, v["cy"] * 0.5, warp_amp, ph["warp"] + k),
        E=(s["wu"], s["wv"], v["duty"], v["soft"]),
        F=(min(1.0, max(0.0, ph["op"])), v["grad"], (v["hue"] + ph["hue"]) % 1.0 + k, v["sat"]),
        G=(bright, float(len(stops)), s["D"], s["hp"]),
        H=(float(v["folds"]), 1.0 if space in PERIODIC else 0.0, 1.0 if s["xor"] else 0.0, 0.0),
        stops=[(c[1], c[2], c[3], c[0]) for c in stops])


def meta():
    """Static description for the panel: field ranges, option names, presets."""
    return dict(
        max=MAX_LAYERS, stops=[STOP_MIN, STOP_MAX], safe_hz=SAFE_HZ,
        num=[dict(name=k, lo=v[0], hi=v[1], default=v[2], label=v[3], hint=v[4]) for k, v in NUM.items()],
        int=[dict(name=k, lo=v[0], hi=v[1], default=v[2], label=v[3], options=v[4]) for k, v in INT.items()],
        generators=list(GENERATORS), spaces=list(SPACES), waves=list(WAVES), blends=list(BLENDS),
        presets={str(g): default_layer(g) for g in range(len(GENERATORS))})


# ------------------------------------------------------------------------------------------------ the GL pass

VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FS = """
#version 330
// One pass, up to 8 layers. Per layer (all resolved on the CPU, see stack._resolve):
uniform int uN;
uniform vec4 uA[8];      // generator, space, waveform, blend
uniform vec4 uB[8];      // fu, fv, gu, gv   (band coordinate frequencies; whole numbers wherever an angle is involved)
uniform vec4 uC[8];      // scroll phase (wrapped), spin angle (wrapped TAU), scale, pattern angle
uniform vec4 uD[8];      // centre x, y, warp amplitude, warp phase (wrapped 1)
uniform vec4 uE[8];      // warp wave vector u, v, duty, softness
uniform vec4 uF[8];      // opacity, gradient smoothness, hue shift (turns), saturation
uniform vec4 uG[8];      // brightness, stop count, colour cycle length D, hash period
uniform vec4 uH[8];      // folds, circular space flag, xor checker flag, -
uniform vec4 uStops[48]; // 6 per layer: rgb + position
uniform float aspect, amount;
in vec2 v_uv; out vec4 f;
const float TAU = 6.28318530718, PI = 3.14159265359;

float h11(float n){ return fract(sin(n*127.1+311.7)*43758.5453); }
float max3(vec3 c){ return max(c.r, max(c.g, c.b)); }
vec3 hueRot(vec3 c, float a){
  const vec3 k = vec3(0.57735); float ca = cos(a), sa = sin(a);
  return c*ca + cross(k, c)*sa + k*dot(k, c)*(1.0-ca);
}
// the stops as a closed loop: g 0..1 goes through every stop and back to the first
vec3 gradient(int b, int n, float g){
  float p0 = uStops[b].a, pl = uStops[b+n-1].a;
  if (g < p0 || g >= pl) {
    float span = p0+1.0-pl;
    float t = span > 1e-5 ? (g >= pl ? g-pl : g+1.0-pl)/span : 0.0;
    return mix(uStops[b+n-1].rgb, uStops[b].rgb, t);
  }
  for (int i = 0; i < 5; i++) {
    if (i >= n-1) break;
    float a = uStops[b+i].a, c = uStops[b+i+1].a;
    if (g >= a && g < c) return mix(uStops[b+i].rgb, uStops[b+i+1].rgb, (g-a)/max(c-a, 1e-5));
  }
  return uStops[b+n-1].rgb;
}
vec3 stepCol(int b, int n, float k, float D){
  int ci = int(floor(fract((k+0.5)/D)*float(n) + 1e-4));
  return uStops[b + clamp(ci, 0, n-1)].rgb;
}
// brightness across one band: fx 0..1 (band centre at 0.5), w = band coordinate per pixel (for antialiasing)
float profile(int wv, float fx, float w, float duty, float soft){
  float t = abs(fx-0.5);
  if (wv == 0) {                                     // sine: duty sharpens / widens the crest
    float c = clamp(0.5+0.5*cos(TAU*t), 0.0, 1.0);
    return pow(c, (1.0-duty)/max(duty, 0.02));
  }
  if (wv == 1) {                                     // square: crisp band, edge = one pixel or the softness
    float hw = 0.5*duty, s = max(0.75*w, soft*min(hw, 0.5-hw));
    return 1.0 - smoothstep(hw-s, hw+s, t);
  }
  if (wv == 2) {                                     // triangle
    float tri = clamp(1.0 - t/max(0.5*duty*2.0, 0.02), 0.0, 1.0);
    return mix(tri, tri*tri*(3.0-2.0*tri), soft);
  }
  float e = min(duty, 1.0), s = max(1.5*w, 1e-3) + soft*0.3*duty;   // saw: ramp up, then a clean drop
  return clamp(fx/max(duty, 0.02), 0.0, 1.0)*(1.0 - smoothstep(e-s, e, fx));
}

void main(){
  vec2 p0 = (v_uv-0.5)*vec2(aspect, 1.0);
  vec3 acc = vec3(0.0);                              // the stack's own light
  float keep = 1.0;                                  // how much of the engine picture underneath survives
  for (int i = 0; i < 8; i++) {
    if (i >= uN) break;
    int gen = int(uA[i].x+0.5), spc = int(uA[i].y+0.5), wv = int(uA[i].z+0.5), bl = int(uA[i].w+0.5);
    float duty = uE[i].z, soft = uE[i].w, N = uH[i].x;
    bool per = uH[i].y > 0.5;
    int nst = int(uG[i].y+0.5), sb = i*6;
    float D = uG[i].z, hp = uG[i].w;

    // ---- space: frame -> pattern plane (u, v)
    vec2 p = p0 - uD[i].xy;
    float cr = cos(uC[i].y), sr = sin(uC[i].y);
    p = mat2(cr, -sr, sr, cr)*p*uC[i].z;
    float r = length(p);
    float thA = atan(p.y, p.x)/TAU, thB = atan(-p.y, -p.x)/TAU + 0.5;   // the same angle, cut on opposite sides
    vec2 uv = p; float fade = 1.0, uB2 = p.x;
    if (spc == 1) { uv = vec2(thA, r); uB2 = thB; }
    else if (spc == 2) { float d = -log(max(r, 1e-4))/TAU; uv = vec2(thA, d); uB2 = thB; fade = smoothstep(0.003, 0.05, r); }
    else if (spc == 3) { float a = abs(fract(thA*N+0.5)-0.5)/N*TAU; uv = r*vec2(cos(a), sin(a)); uB2 = uv.x; }
    else if (spc == 4) { uv = vec2(abs(p.x), p.y); uB2 = uv.x; }
    else if (spc == 5) { uv = vec2(p.x, abs(p.y)); }
    else if (spc == 6) { uv = abs(p); uB2 = uv.x; }
    else if (spc == 7) { float d = abs(p.y), z = 0.25/max(d, 1e-3); uv = vec2(p.x*z, z); uB2 = uv.x; fade = smoothstep(0.0, 0.10, d); }

    // ---- generator: band coordinate x (one band per unit), its pixel footprint w, band index k
    float x, w, val, k;
    float ph = uC[i].x;
    if (gen == 4) {                                  // checker / grid: two crossed band systems
      float x1 = uB[i].x*uv.x + uB[i].y*uv.y + ph, x1b = uB[i].x*uB2 + uB[i].y*uv.y + ph;
      float x2 = uB[i].z*uv.x + uB[i].w*uv.y,      x2b = uB[i].z*uB2 + uB[i].w*uv.y;
      float w1 = min(fwidth(x1), fwidth(x1b)), w2 = min(fwidth(x2), fwidth(x2b));
      w = max(w1, w2); x = x1;
      float c1 = floor(x1), c2 = floor(x2);
      if (uH[i].z > 0.5) {                           // two-colour checker (square), shaded tiles (saw)
        float a1 = clamp(sin(PI*x1)/(PI*(w1+0.25*soft)+1e-5), -1.0, 1.0);
        float a2 = clamp(sin(PI*x2)/(PI*(w2+0.25*soft)+1e-5), -1.0, 1.0);
        val = clamp(0.5 + 0.5*(a1*a2) + (duty-0.5), 0.0, 1.0);
        if (wv == 3) val *= 0.25 + 0.75*fract(x1);
        k = floor((c1+c2)*0.5);
      } else {
        float l1 = profile(wv, fract(x1), w1, duty, soft), l2 = profile(wv, fract(x2), w2, duty, soft);
        val = wv == 2 ? 1.0-(1.0-l1)*(1.0-l2) : l1*l2;   // triangle: a grid of lines; sine: a lattice of dots
        k = c1+c2;
      }
      val = mix(val, 0.5, smoothstep(0.5, 1.2, w));
    } else {
      float xb;
      if (gen == 0 || gen == 5 || gen == 7) {        // stripes / waves / noise bands: linear in the plane
        float sA = uE[i].x*uv.x + uE[i].y*uv.y, sB = uE[i].x*uB2 + uE[i].y*uv.y;
        float wa = uD[i].z*sin(TAU*(sA+uD[i].w)), wb = uD[i].z*sin(TAU*(sB+uD[i].w));
        if (gen == 5) { wa += 0.35*uD[i].z*sin(TAU*(2.0*sA-uD[i].w)+1.3); wb += 0.35*uD[i].z*sin(TAU*(2.0*sB-uD[i].w)+1.3); }
        float base = uB[i].y*uv.y + ph;
        x = base + uB[i].x*uv.x + wa; xb = base + uB[i].x*uB2 + wb;
      } else {                                       // rings / spiral / rays / polygons: radius + angle
        vec2 q = uv;
        if (per) q = vec2(abs(fract(uv.x*N+0.5)-0.5), spc == 2 ? abs(fract(uv.y*N+0.5)-0.5) : uv.y-0.3);
        float ca = cos(uC[i].w), sa = sin(uC[i].w);
        q = mat2(ca, -sa, sa, ca)*q;
        float rr = length(q), tA = atan(q.y, q.x)/TAU, tB = atan(-q.y, -q.x)/TAU + 0.5;
        float R = rr;
        if (gen == 6) {                              // polygon rings; warp pulls the sides in: stars
          float sides = uB[i].z, a = (fract(tA*sides+0.5)-0.5)/sides*TAU;
          R = rr*cos(a)*(1.0 + 1.6*uD[i].z*(1.0 - abs(a)*sides/PI));
        } else if (uD[i].z > 0.0) {                  // warp: flower lobes (a whole number of them)
          R = rr*(1.0 + 0.3*uD[i].z*sin(TAU*(uB[i].w*tA + uD[i].w)));
        }
        x = uB[i].x*R + uB[i].y*tA + ph; xb = uB[i].x*R + uB[i].y*tB + ph;
      }
      w = min(fwidth(x), fwidth(xb));
      k = floor(x);
      float fx = x-k, dk = duty, gain = 1.0;
      if (gen == 7) {                                // noise bands: every band its own width and level
        float hk = mod(k, hp);
        dk = clamp(duty*(0.35+1.3*h11(hk)), 0.03, 0.97); gain = 0.3+0.7*h11(hk+17.3);
      }
      val = profile(wv, fx, w, dk, soft)*gain;
      val = mix(val, dk*(wv == 1 ? 1.0 : 0.6)*gain, smoothstep(0.5, 1.2, w));   // finer than a pixel: the mean
    }
    val *= fade;
    if (!(val >= 0.0)) val = 0.0;                    // a NaN (exact centre pixel) must not reach the alpha

    // ---- colour: each band takes the next stop (flat), or the stops as one smooth loop
    float fxc = x-floor(x), kk = (gen == 4) ? k : floor(x);
    vec3 cs = stepCol(sb, nst, kk, D);
    if (gen != 4) {
      float e = clamp(0.75*w, 1e-4, 0.5);            // antialias the colour step (shows when the bands touch)
      if (fxc < e) cs = mix(stepCol(sb, nst, kk-1.0, D), cs, 0.5+0.5*fxc/e);
      else if (fxc > 1.0-e) cs = mix(cs, stepCol(sb, nst, kk+1.0, D), 0.5*(fxc-(1.0-e))/e);
    }
    vec3 col = cs;
    if (uF[i].y > 0.0) col = mix(cs, gradient(sb, nst, fract((gen == 4 ? kk+0.5 : x)/D)), uF[i].y);
    if (uF[i].z > 0.0) col = max(hueRot(col, uF[i].z*TAU), 0.0);
    col = max(mix(vec3(dot(col, vec3(0.299, 0.587, 0.114))), col, uF[i].w), 0.0)*uG[i].x;

    // ---- blend onto the layers below
    float o = uF[i].x;
    vec3 c = col*val;
    if (bl == 0) acc += c*o;                                                   // add
    else if (bl == 1) acc = acc + c*o*(1.0-clamp(acc, 0.0, 1.0));              // screen
    else if (bl == 2) { acc *= mix(vec3(1.0), c, o); keep *= mix(1.0, max3(c), o); }        // multiply
    else if (bl == 3) acc = mix(acc, abs(acc-c), o);                           // difference
    else if (bl == 4) { acc *= mix(1.0, val, o); keep *= mix(1.0, val, o); }   // mask: only where this layer is lit
    else if (bl == 5) { acc = mix(acc, c, o); keep *= 1.0-o; }                 // replace: opaque, black included
    else { acc = mix(acc, col, o*val); keep *= 1.0-o*val; }                    // over: the bands cover, gaps are clear
  }
  acc = max(acc, 0.0);
  acc /= max(1.0, max3(acc));                        // hue-preserving clip
  if (!(acc.r >= 0.0 && acc.g >= 0.0 && acc.b >= 0.0)) acc = vec3(0.0);
  if (!(keep >= 0.0)) keep = 1.0;
  f = vec4(acc*amount, clamp((1.0-keep)*amount, 0.0, 1.0));
}
"""


def pack(frame, palette=None):
    """Resolved layers -> (count, {uniform name: float32 bytes}, stops bytes). `palette` (5 rgb colours) replaces the
    stops of the layers that follow the show palette (the controller resolves them without knowing its colours)."""
    import numpy as np
    n = min(len(frame), MAX_LAYERS)
    out = {key: np.zeros((MAX_LAYERS, 4), dtype="f4") for key in "ABCDEFGH"}
    stops = np.zeros((MAX_LAYERS * STOP_MAX, 4), dtype="f4")
    for i, lay in enumerate(frame[:n]):
        for key in "ABCDEFGH":
            out[key][i] = lay[key]
        for j, s in enumerate(lay["stops"][:STOP_MAX]):
            stops[i * STOP_MAX + j] = s
        if lay.get("pal") and palette is not None and len(palette) >= PAL_N:
            for j in range(PAL_N):
                stops[i * STOP_MAX + j, :3] = palette[j][:3]
    return n, {f"u{key}": v.tobytes() for key, v in out.items()}, stops.tobytes()


class StackPass:
    """render(frame, amount) -> texture (RGBA8, output size). `frame` = StackState.frame. Call it only while the
    stack has layers and l_stack is up: an empty stack never reaches the GPU (`draws` counts the passes drawn)."""

    def __init__(self, ctx, out_size=(1280, 800)):
        import moderngl
        import numpy as np
        self._mgl = moderngl
        self.ctx = ctx
        self.out_size = (max(16, int(out_size[0])), max(16, int(out_size[1])))
        self.aspect = self.out_size[0] / self.out_size[1]
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self._vao = ctx.vertex_array(self._prog, [(self._quad, "2f", "in_pos")])
        self._tex = ctx.texture(self.out_size, 4)
        self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._tex.repeat_x = self._tex.repeat_y = False
        self._fbo = ctx.framebuffer(color_attachments=[self._tex])
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.draws = 0
        self._gl = None
        try:                                         # the caller's blend switch is read back with raw GL
            import math_layers as _ml
            self._gl, self._gl_blend = _ml._gl_lib(), _ml._GL_BLEND
        except Exception:
            self._gl = None

    def render(self, frame, amount=1.0, palette=None):
        ctx, prog = self.ctx, self._prog
        n, blocks, stops = pack(frame, palette)
        blend_on = False
        if self._gl is not None:
            try:
                blend_on = bool(self._gl.glIsEnabled(self._gl_blend))
            except Exception:
                blend_on = False
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        ctx.disable(self._mgl.BLEND)
        self._fbo.use()
        ctx.viewport = (0, 0, *self.out_size)
        prog["uN"].value = n
        for name, data in blocks.items():
            if name in prog:
                prog[name].write(data)
        if "uStops" in prog:
            prog["uStops"].write(stops)
        prog["aspect"].value = self.aspect
        prog["amount"].value = float(min(1.0, max(0.0, amount)))
        self._vao.render(self._mgl.TRIANGLE_STRIP)
        self.draws += 1
        if blend_on:
            ctx.enable(self._mgl.BLEND)
        try:                                         # the caller's framebuffer and viewport, as they were
            prev_fbo.use()
            ctx.viewport = prev_vp
        except Exception:
            pass
        return self._tex

    def read(self):
        """The last rendered picture as an RGBA PIL image (tests, gallery)."""
        from PIL import Image
        img = Image.frombytes("RGBA", self.out_size, self._fbo.read(components=4))
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    def release(self):
        for obj in (self._fbo, self._tex, self._vao, self._prog, self._quad):
            try:
                obj.release()
            except Exception:
                pass

    close = release


# ------------------------------------------------------------------------------------------------ gallery

GALLERY = [
    ("canvas_start", [CANVAS_START]),
    ("stripes_crisp", [dict(gen=0, freq=8, soft=0.0, angle=0.0)]),
    ("stripes_gradient", [dict(gen=0, freq=4, soft=0.0, duty=0.98, grad=1.0, angle=0.1)]),
    ("rings", [dict(gen=1)]),
    ("spiral", [dict(gen=2)]),
    ("rays", [dict(gen=3)]),
    ("checker", [dict(gen=4)]),
    ("waves", [dict(gen=5)]),
    ("polygons", [dict(gen=6)]),
    ("stars", [dict(gen=6, count=5, warp=0.7)]),
    ("noise_bands", [dict(gen=7)]),
    ("polar_stripes_spiral", [dict(gen=0, space=1, freq=8, angle=0.12)]),
    ("tunnel_checker", [dict(gen=4, space=2, freq=8, angle=0.0)]),
    ("tunnel_stripes", [dict(gen=0, space=2, freq=10, angle=0.1, grad=1.0, duty=0.7)]),
    ("kaleido_stripes", [dict(gen=0, space=3, folds=8, freq=9, angle=0.07)]),
    ("kaleido_waves", [dict(gen=5, space=3, folds=6)]),
    ("mirror_xy_noise", [dict(gen=7, space=6, angle=0.1)]),
    ("floor_checker", [dict(gen=4, space=7, freq=3, angle=0.0)]),
    ("floor_stripes", [dict(gen=0, space=7, freq=4)]),
    ("layers_add", [dict(gen=0, freq=6, angle=0.1, blend=0), dict(gen=1, freq=4, blend=0, opacity=0.8)]),
    ("layers_mask", [dict(gen=0, freq=4, duty=0.98, grad=1.0, angle=0.25), dict(gen=3, count=10, blend=4)]),
    ("layers_difference", [dict(gen=0, freq=7, angle=0.05), dict(gen=1, freq=5, blend=3)]),
    ("layers_multiply", [dict(gen=5, freq=4), dict(gen=4, freq=5, blend=2, soft=0.3)]),
    ("layers_over", [dict(gen=1, freq=6, grad=1.0, duty=0.9), dict(gen=3, count=8, blend=6, duty=0.35)]),
    ("palette_follow", [dict(gen=0, color=1, freq=5, angle=0.15)]),
]


def render_stack(layers, size=(640, 400), t=1.0, ctx=None, palette=None, k=0):
    """One still of a stack (list of layer dicts) after t seconds of motion -> PIL RGB image. Headless."""
    import moderngl
    from PIL import Image
    own = ctx is None
    if own:
        ctx = moderngl.create_standalone_context()
    st = StackState()
    for lay in layers:
        st.add(lay.get("gen", 0), lay)
    st.step(0.0)
    for x in st.layers:
        st._phase(x["id"])["op"] = None              # stills: no fade-in
    steps = max(1, int(round(t * 60)))
    for _ in range(steps):
        st.step(t / steps, dict(bpm=124.0, beat=0.0), palette=palette, aspect=size[0] / size[1])
    if k:
        st.step(0.0, dict(bpm=124.0, beat=0.0), palette=palette, aspect=size[0] / size[1], hold=True, k=k)
    sp = StackPass(ctx, size)
    sp.render(st.frame, 1.0, palette)
    img = sp.read()
    sp.release()
    if own:
        ctx.release()
    bg = Image.new("RGB", img.size, (0, 0, 0))
    bg.paste(img.convert("RGB"))
    return bg


def _gallery(out, size):
    import moderngl
    from pathlib import Path
    from PIL import Image, ImageDraw
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = moderngl.create_standalone_context()
    pal = [(0.00, 0.85, 0.80), (1.00, 0.05, 0.55), (1.00, 0.70, 0.05), (0.28, 0.10, 1.00), (0.10, 0.90, 0.30)]
    shots = []
    for name, layers in GALLERY:
        img = render_stack(layers, size, 1.3, ctx, pal)
        img.save(out / f"{name}.png")
        shots.append((name, img))
        print(f"[stack] {name}")
    tw, th, cols = 320, 200, 5
    rows = (len(shots) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 16)), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(shots):
        x, y = (i % cols) * tw, (i // cols) * (th + 16)
        sheet.paste(img.resize((tw, th), Image.LANCZOS), (x, y + 16))
        d.text((x + 4, y + 2), name, fill=(220, 220, 220))
    sheet.save(out / "contact_sheet.png")
    ctx.release()
    print(f"[stack] {len(shots)} stills + contact sheet -> {out / 'contact_sheet.png'}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="math stack: headless stills (small; the GPU is shared with the show)")
    ap.add_argument("--gallery", action="store_true", help="every generator / space / blend -> test_frames/stack/")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "test_frames" / "stack"))
    ap.add_argument("--size", default="640x400")
    a = ap.parse_args()
    if a.gallery:
        _gallery(a.out, tuple(int(x) for x in a.size.split("x")))
    else:
        ap.print_help()
