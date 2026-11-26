"""Environment SKINS: put an image (uploaded art, an AI texture, a dreamified picture) on the surfaces of a world.

Three parts, one file:

  * the MODEL (pure python, no GL): `SkinState` = per environment 1-2 material slots, each with a source, a mapping,
    scale / rotation / scroll, tiling, tint, palette-follow, a mix mode with the procedural pattern and an amount.
    Commands `{"cmd": "skin", "op": ...}` are validated by `validate()`. Targets for MIDI learn and the mod matrix are
    named `skin:<env>:<slot>:<field>` (same idea as the math stack's `stack:<id>:<field>`).
  * the LIBRARY (GL): `SkinLibrary` = ONE RGBA8 texture array (8 layers, 512 or 1024 px, mipmapped, hardware
    anisotropic) on ONE private texture unit. Images are decoded, resized AND mip-mapped on a worker thread; the render
    thread only copies about 1 MB per frame (per mip level, glTexSubImage3D), skipped after a late frame; the
    least-recently-used layer is replaced. VRAM is accounted (`vram_bytes`).
  * the ENGINE side: `SkinEngine` = crossfades, float64 phases wrapped at 2 (one mirrored repeat), the per-slot uniform
    block, the governor hook (`govern`), and `GLSL`, the shader chunk the worlds paste in.

COST RULES (the user's priority: "very compute efficient")
  * every slot procedural  -> the host uses its ORIGINAL shader program: no sampler, no bind, no branch, no uniform.
  * a skinned slot         -> ONE hardware-filtered fetch per pixel (two only during a 0.4 s source crossfade).
  * triplanar              -> THREE fetches; it is the expensive mapping and is only run when chosen.
  * no VRAM until the first skin is used (the array is allocated lazily); no float textures; no per-frame uploads;
    nothing is RETAINED per frame (`SkinEngine.update` writes into preallocated arrays). A skinned frame still makes a
    few short-lived Python floats / tuples (the uv matrix); an all-procedural frame makes none.

Environments (`ENVS`): corridor (walls, floor), torus (tube, chambers), vortex (walls), hexdome (domes) = the four
fx_stack.LandPass worlds; tunnel (walls) = the travel modes of the main scene shader.
"""
import math
import os
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np

try:                             # the mod sources the control system knows (unknown ones are refused / dropped)
    from params import MOD_SOURCES as _MOD_SOURCES
    MOD_SOURCES = frozenset(_MOD_SOURCES)
except Exception:
    MOD_SOURCES = None

TAU = 2.0 * math.pi
VERSION = 1
UNIT = 22                        # private texture unit of the skin array (0-17, 20, 21 are taken: see HANDOFF)
UNIT_L = 23                      # private texture unit of the live-layer array (LayerHub)
UNIT_M, UNIT_LM = 24, 25         # the same two arrays again, read through a MIRRORED_REPEAT sampler object
MAX_LAYERS = 8
SIZES = (512, 1024)
FADE_S = 0.4                     # source crossfade / fade in from procedural
PH_WRAP = 2.0                    # every skin phase wraps here: one mirrored repeat = two plain repeats
KINDS = ("art", "ai", "layer")   # art/ uploads (+ dreamified art), textures/ (AI, seed and dreamified textures),
                                 # "layer" = a live engine layer rendered to a small texture (LayerHub, below)
LAYER_RE = re.compile(r"^[a-z0-9_]{1,24}$")
NAME_RE = re.compile(r"^[^\\/:*?\"<>|\x00-\x1f]{1,120}$")
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

MIX_NAMES = ["Replace", "Multiply", "Screen", "Overlay", "Lines reveal the skin"]
TILE_NAMES = ["Repeat", "Mirror (seamless)"]
DECK_MAPS = ["Surface", "Triplanar (3 fetches)", "Screen"]
TUNNEL_MAPS = ["Follow the travel mode", "Log-polar", "Tube", "Screen"]

# env -> label, slots, mapping names. `host` says which pass draws it.
ENVS = {
    "corridor": dict(label="Sacred corridor", slots=("walls", "floor"), maps=DECK_MAPS, host="land", land=0),
    "torus": dict(label="Torus wormhole", slots=("tube", "chambers"), maps=DECK_MAPS, host="land", land=1),
    "vortex": dict(label="Prismatic vortex", slots=("walls",), maps=DECK_MAPS, host="land", land=2),
    "hexdome": dict(label="Hex domes", slots=("domes",), maps=DECK_MAPS, host="land", land=3),
    "tunnel": dict(label="Travel tunnel", slots=("walls",), maps=TUNNEL_MAPS, host="scene"),
    # [beings> the sprite creature (beings.py): "fill" = a skin inside the silhouette (look "Skinned fill"); "sprite" =
    # the user's own image that IS the creature. `own` slots are only a picker: beings.SpriteLibrary loads them (alpha
    # kept, distance field built), the skin array never holds them. <beings]
    "beings": dict(label="Sprite creature", slots=("fill", "sprite"), maps=["Moves with the creature", "Mirrors with the creature", "Fixed to the screen"], host="beings",
                   own=("sprite",)),
    # [coilnest> the coil nest bodies (coilnest.py): the image becomes the cloth ground of every body; u along the
    # body, v round the girth, kené lines / fill / jewels stay on top and keep their gating. <coilnest]
    "coilnest": dict(label="Coil nest bodies", slots=("bodies",), maps=["Along the body"], host="coilnest"),
}
LAND_ENVS = ["corridor", "torus", "vortex", "hexdome"]
MAX_SLOTS = 2

# numeric fields: lo, hi, default, label
NUM = {
    "amount": (0.0, 1.0, 1.0, "Amount"),
    "scale": (0.25, 8.0, 1.0, "Scale"),
    "rot": (-180.0, 180.0, 0.0, "Rotation"),
    "scroll": (0.0, 2.0, 0.0, "Scroll speed"),
    "dir": (0.0, 360.0, 90.0, "Scroll direction"),
    "pal": (0.0, 1.0, 0.0, "Follow the show palette"),
    "bright": (0.0, 2.0, 1.0, "Brightness"),
}
# stepped fields: lo, hi, default, label
INT = {
    "map": (0, 3, 0, "Mapping"),
    "tile": (0, 1, 1, "Tiling"),
    "mix": (0, len(MIX_NAMES) - 1, 0, "Mix with the pattern"),
    "sync": (0, 1, 0, "Scroll on the beat"),
}
FIELDS = tuple(NUM) + tuple(INT)
TINT_DEFAULT = (1.0, 1.0, 1.0)

# ---- skin STACKS: every slot = its base (the fields above) + up to MAX_OVERLAYS overlay layers drawn on top of it.
# An overlay is one more fetch from an image or a live layer, with its own blend, opacity, scale / rotation / scroll,
# per-channel gains (`gate` r g b a), an optional colour remap (`remap` 1: colour = sum of channel x cols[channel], for
# 4-channel line layers: R primary lines, G fine fill, B ground, A accents) and a travelling noise mask ("floating
# patches": noise amount / scale / speed / threshold). `lit` = how much the world's form light shades the layer.
MAX_OVERLAYS = 2
BLEND_NAMES = ["Add", "Screen", "Multiply", "Over (by brightness)", "Mask (base only through the layer)"]
LNUM = {
    "opacity": (0.0, 2.0, 1.0, "Opacity"),
    "scale": (0.25, 8.0, 1.0, "Scale"),
    "rot": (-180.0, 180.0, 0.0, "Rotation"),
    "scroll": (0.0, 2.0, 0.0, "Scroll speed"),
    "dir": (0.0, 360.0, 90.0, "Scroll direction"),
    "noise": (0.0, 1.0, 0.0, "Floating patches"),
    "nscale": (0.25, 8.0, 1.5, "Patch size"),
    "nspeed": (0.0, 2.0, 0.15, "Patch drift"),
    "thr": (0.0, 1.0, 0.5, "Patch threshold"),
    "lit": (0.0, 1.0, 0.6, "Shaded by the world's light"),
}
LINT = {
    "blend": (0, len(BLEND_NAMES) - 1, 0, "Blend"),
    "remap": (0, 1, 0, "Colour by channel"),
    "tile": (0, 1, 1, "Tiling"),
}
LFIELDS = tuple(LNUM) + tuple(LINT)
GATE_DEFAULT = (1.0, 1.0, 1.0, 1.0)
COLS_DEFAULT = ((1.0, 0.93, 0.80), (0.85, 0.90, 1.0), (0.05, 0.16, 0.62), (1.0, 0.10, 0.16))

# Mapping geometry: (env, slot, effective map) -> (kind, base tiles, base z scale, travel follow).
#   "cyl":   the shader gives (angle / TAU, distance along the axis). Tiles round the axis are snapped to a whole
#            number (an even one for mirror tiling), so the seam at the atan cut closes for any scale / rotation.
#   "plane": the shader gives two world / screen coordinates; the matrix is scale x rotation.
# follow = how much of the host's travel (per unit) is added along the second input coordinate, so a skin stays on
# the wall it is painted on while the camera flies.
EFF_TUBE, EFF_LOGPOLAR, EFF_SCREEN, EFF_GROUND = 2, 1, 3, 4
_GEO = {
    ("corridor", "walls", 0): ("cyl", 4.0, 0.26, 1.0),
    ("corridor", "floor", 0): ("plane", 0.26, 0.26, 1.0),
    ("torus", "tube", 0): ("cyl", 4.0, 0.33, 1.0),
    ("torus", "chambers", 0): ("cyl", 4.0, 0.33, 1.0),
    ("vortex", "walls", 0): ("cyl", 4.0, 0.20, 1.0),
    ("hexdome", "domes", 0): ("plane", 0.55, 0.55, 0.0),          # one image per honeycomb cell (no polar stretch)
    ("tunnel", "walls", EFF_LOGPOLAR): ("cyl", 4.0, 0.75, -math.log(2.0)),
    ("tunnel", "walls", EFF_TUBE): ("cyl", 4.0, 0.9, 2.0),
    ("tunnel", "walls", EFF_GROUND): ("plane", 0.9, 0.9, 1.25),
    ("tunnel", "walls", EFF_SCREEN): ("plane", 0.62, 0.62, 0.0),
    ("coilnest", "bodies", 0): ("cyl", 2.0, 1.0, 0.0),           # (girth, 4 cells): 2 images round, one per 4 cells along
}


def geometry(env, slot, eff):
    g = _GEO.get((env, slot, eff))
    if g is not None:
        return g
    if ENVS[env]["host"] == "land":
        follow = 0.0 if env == "hexdome" or eff == 2 else 1.0
        return ("plane", 0.30 if eff == 1 else 1.0, 0.30 if eff == 1 else 1.0, follow)
    return ("plane", 1.0, 1.0, 0.0)


# ------------------------------------------------------------------ tiling maths (pure; mirrored by GLSL below)


def fold(u, tile):
    """Texture coordinate after tiling: 0 = plain repeat, 1 = mirrored repeat (period 2, continuous everywhere)."""
    if tile:
        return 1.0 - abs((u % 2.0) - 1.0)
    return u % 1.0


def _even(x):
    return 2.0 * round(x / 2.0)


def uv_matrix(kind, base, base_z, scale, rot_deg, tile):
    """-> (m00, m01, m10, m11) with uv' = M (a, b).

    plane: scale x rotation.  cyl: a = angle / TAU (0..1 round the axis), b = distance along it. The first column is
    snapped to whole numbers (even for mirror tiling), so uv' moves by a whole period once round: no seam."""
    s = min(max(float(scale), 0.01), 64.0)
    r = math.radians(float(rot_deg))
    c, sn = math.cos(r), math.sin(r)
    if kind == "cyl":
        n = base * s
        snap = _even if tile else round
        lo = 2.0 if tile else 1.0
        a, b = snap(n * c), snap(n * sn)
        if a == 0.0 and b == 0.0:
            a = lo if c >= 0 else -lo
        return (float(a), -sn * base_z * s, float(b), c * base_z * s)
    return (c * base * s, -sn * base_z * s, sn * base * s, c * base_z * s)


def wrap(x):
    return x % PH_WRAP


# ------------------------------------------------------------------ model


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def clean_field(name, v):
    if name in NUM:
        lo, hi, d = NUM[name][:3]
        return min(max(float(v), lo), hi) if _isnum(v) else d
    lo, hi, d = INT[name][:3]
    return int(min(max(round(float(v)), lo), hi)) if _isnum(v) else d


def clean_src(raw):
    """-> {"kind", "name"} or None (procedural). kind "layer": name = a live layer source id (lower-case word)."""
    if not isinstance(raw, dict):
        return None
    kind, name = raw.get("kind"), raw.get("name")
    if kind == "layer":
        return dict(kind=kind, name=name) if isinstance(name, str) and LAYER_RE.match(name) else None
    if kind not in KINDS or not isinstance(name, str) or not NAME_RE.match(name) or name in (".", ".."):
        return None
    if not name.lower().endswith(IMG_EXT):
        return None
    return dict(kind=kind, name=name)


def clean_tint(raw):
    if isinstance(raw, (list, tuple)) and len(raw) == 3 and all(_isnum(x) for x in raw):
        return [min(max(float(x), 0.0), 1.0) for x in raw]
    return list(TINT_DEFAULT)


def default_slot():
    d = {f: NUM[f][2] for f in NUM}
    d.update({f: INT[f][2] for f in INT})
    d["tint"] = list(TINT_DEFAULT)
    d["src"] = None
    d["layers"] = [None] * MAX_OVERLAYS
    return d


def default_layer():
    d = {f: LNUM[f][2] for f in LNUM}
    d.update({f: LINT[f][2] for f in LINT})
    d["gate"] = list(GATE_DEFAULT)
    d["cols"] = [list(c) for c in COLS_DEFAULT]
    d["src"] = None
    return d


def clean_lfield(name, v):
    if name in LNUM:
        lo, hi, d = LNUM[name][:3]
        return min(max(float(v), lo), hi) if _isnum(v) else d
    lo, hi, d = LINT[name][:3]
    return int(min(max(round(float(v)), lo), hi)) if _isnum(v) else d


def _clean_gate(v, strict=False):
    if isinstance(v, (list, tuple)) and len(v) == 4 and all(_isnum(x) for x in v):
        return [min(max(float(x), 0.0), 2.0) for x in v]
    if strict:
        raise ValueError("gate must be [r, g, b, a] gains")
    return list(GATE_DEFAULT)


def _clean_cols(v, strict=False):
    if (isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(c, (list, tuple)) and len(c) == 3
                                                             and all(_isnum(x) for x in c) for c in v)):
        return [[min(max(float(x), 0.0), 1.0) for x in c] for c in v]
    if strict:
        raise ValueError("cols must be four [r, g, b] colours")
    return [list(c) for c in COLS_DEFAULT]


def clean_layer(raw):
    """An overlay layer -> its full canonical dict, or None (empty: no layer)."""
    if not isinstance(raw, dict):
        return None
    out = default_layer()
    for f in LFIELDS:
        if f in raw:
            out[f] = clean_lfield(f, raw[f])
    if "gate" in raw:
        out["gate"] = _clean_gate(raw["gate"])
    if "cols" in raw:
        out["cols"] = _clean_cols(raw["cols"])
    out["src"] = clean_src(raw.get("src"))
    return out


def _layer_diff(lay):
    """Compact form for JSON: only what differs from the default (None stays None)."""
    if lay is None:
        return None
    d = default_layer()
    return {k: v for k, v in lay.items() if v != d[k]}


def clean_slot(raw, env=None):
    out = default_slot()
    if isinstance(raw, dict):
        for f in FIELDS:
            if f in raw:
                out[f] = clean_field(f, raw[f])
        if "tint" in raw:
            out["tint"] = clean_tint(raw["tint"])
        out["src"] = clean_src(raw.get("src"))
        lays = raw.get("layers")
        if isinstance(lays, list):
            out["layers"] = [clean_layer(x) for x in (lays + [None] * MAX_OVERLAYS)[:MAX_OVERLAYS]]
    if env is not None:
        out["map"] = min(out["map"], len(ENVS[env]["maps"]) - 1)
    return out


def parse_key(key):
    """"env:slot:field" -> (env, slot, field) or None."""
    if not isinstance(key, str):
        return None
    p = key.split(":")
    if len(p) != 3 or p[0] not in ENVS or p[1] not in ENVS[p[0]]["slots"] or p[2] not in FIELDS:
        return None
    return p[0], p[1], p[2]


def parse_target(t):
    """MIDI-learn / modulation target "skin:<env>:<slot>:<field>" -> (env, slot, field) or None."""
    if isinstance(t, str) and t.startswith("skin:"):
        return parse_key(t[5:])
    return None


def clean_mod(m, sources=None):
    if not isinstance(m, dict) or not isinstance(m.get("src"), str):
        return None
    sources = MOD_SOURCES if sources is None else sources
    if (sources is not None and m["src"] not in sources) or m["src"] in ("", "none", "off"):
        return None
    amt, rate, phase = m.get("amt", 0.0), m.get("rate", 1.0), m.get("phase", 0.0)
    if not (_isnum(amt) and _isnum(rate) and _isnum(phase)):
        return None
    return dict(src=m["src"][:24], amt=min(max(float(amt), -1.0), 1.0), rate=min(max(float(rate), 1 / 64), 64.0),
                phase=min(max(float(phase), 0.0), 1.0))


def clean_skins(raw, sources=None):
    """Any JSON -> the canonical compact form {"v": 1, "slots": {"env:slot": {changed fields}}, "mods": {...}}.
    Only slots that differ from the default are stored (a show with no skins costs a few bytes)."""
    out = dict(v=VERSION, slots={}, mods={})
    if not isinstance(raw, dict):
        return out
    slots = raw.get("slots")
    if isinstance(slots, dict):
        dflt = default_slot()
        for key, val in slots.items():
            p = key.split(":") if isinstance(key, str) else []
            if len(p) != 2 or p[0] not in ENVS or p[1] not in ENVS[p[0]]["slots"]:
                continue
            s = clean_slot(val, p[0])
            diff = {k: v for k, v in s.items() if v != dflt[k]}
            if "layers" in diff:
                diff["layers"] = [_layer_diff(x) for x in s["layers"]]
            if diff:
                out["slots"][key] = diff
    mods = raw.get("mods")
    if isinstance(mods, dict):
        for key, pair in mods.items():
            t = parse_key(key)
            if t is None or t[2] not in NUM or not isinstance(pair, list):
                continue
            pair = [clean_mod(m, sources) for m in (pair + [None, None])[:2]]
            if any(pair):
                out["mods"][key] = pair
    return out


def _env_slot(c):
    env, slot = c.get("env"), c.get("slot")
    if env not in ENVS:
        raise ValueError("unknown environment")
    if slot not in ENVS[env]["slots"]:
        raise ValueError("unknown slot")
    return env, slot


def validate(c):
    """Clean a {"cmd": "skin", "op": ...} command or raise ValueError. Idempotent (the server thread validates, the
    controller validates again)."""
    if not isinstance(c, dict):
        raise ValueError("command must be an object")
    op = c.get("op")
    out = dict(cmd="skin", op=op)
    if op == "set":
        out["env"], out["slot"] = _env_slot(c)
        sets = c.get("set") if isinstance(c.get("set"), dict) else {c.get("f"): c.get("v")}
        clean = {}
        for f, v in sets.items():
            if f == "tint":
                if not (isinstance(v, (list, tuple)) and len(v) == 3 and all(_isnum(x) for x in v)):
                    raise ValueError("tint must be [r, g, b]")
                clean[f] = clean_tint(v)
            elif f in FIELDS:
                if not _isnum(v):
                    raise ValueError(f"'{f}' must be a number")
                clean[f] = clean_field(f, v)
                if f == "map":
                    clean[f] = min(clean[f], len(ENVS[out["env"]]["maps"]) - 1)
            else:
                raise ValueError("unknown skin field")
        if not clean:
            raise ValueError("nothing to set")
        out["set"] = clean
    elif op == "assign":
        out["env"], out["slot"] = _env_slot(c)
        src = clean_src(dict(kind=c.get("kind"), name=c.get("name")))
        if src is None:
            raise ValueError("source must be {kind: art | ai, name: an image file name} or {kind: layer, name: a layer id}")
        out.update(src)
    elif op in ("layer", "layer_clear"):
        out["env"], out["slot"] = _env_slot(c)
        i = c.get("i")
        if type(i) is not int or not 0 <= i < MAX_OVERLAYS:
            raise ValueError(f"overlay index i must be an integer 0..{MAX_OVERLAYS - 1}")
        out["i"] = i
        if op == "layer":
            sets = c.get("set")
            if not isinstance(sets, dict) or not sets:
                raise ValueError("nothing to set")
            clean = {}
            for f, v in sets.items():
                if f == "src":
                    clean[f] = None if v is None else clean_src(v)
                    if v is not None and clean[f] is None:
                        raise ValueError("bad overlay source")
                elif f == "gate":
                    clean[f] = _clean_gate(v, strict=True)
                elif f == "cols":
                    clean[f] = _clean_cols(v, strict=True)
                elif f in LFIELDS:
                    if not _isnum(v):
                        raise ValueError(f"'{f}' must be a number")
                    clean[f] = clean_lfield(f, v)
                else:
                    raise ValueError("unknown overlay field")
            out["set"] = clean
    elif op == "clear":
        out["env"], out["slot"] = _env_slot(c)
    elif op == "reset":
        if c.get("env") is not None:
            if c["env"] not in ENVS:
                raise ValueError("unknown environment")
            out["env"] = c["env"]
    elif op == "replace":
        if not isinstance(c.get("skins"), dict):
            raise ValueError("'skins' must be an object")
        out["skins"] = clean_skins(c["skins"])
    elif op == "mod":
        out["env"], out["slot"] = _env_slot(c)
        if c.get("f") not in NUM:
            raise ValueError("only numeric skin fields can be modulated")
        out["f"] = c["f"]
        slot_i = c.get("mslot", c.get("slot_i", 0))
        if type(slot_i) is not int or slot_i not in (0, 1):          # 1.0 or True would index the pair wrongly
            raise ValueError("mod slot must be the integer 0 or 1")
        out["mslot"] = slot_i
        src = c.get("src")
        if not isinstance(src, str) or len(src) > 24:
            raise ValueError("bad mod source")
        if MOD_SOURCES is not None and src not in MOD_SOURCES and src not in ("", "none", "off"):
            raise ValueError("unknown mod source")
        out["src"] = src
        for k, lo, hi, d in (("amt", -1.0, 1.0, 0.0), ("rate", 1 / 64, 64.0, 1.0), ("phase", 0.0, 1.0, 0.0)):
            v = c.get(k, d)
            if not _isnum(v):
                raise ValueError(f"'{k}' must be a number")
            out[k] = min(max(float(v), lo), hi)
    elif op == "gen":
        out["env"], out["slot"] = _env_slot(c)
        p = c.get("prompt")
        if not isinstance(p, str) or not p.strip() or len(p) > 400:
            raise ValueError("prompt must be 1..400 characters")
        out["prompt"] = "".join(ch for ch in p if ord(ch) >= 32).strip()
        if not out["prompt"]:
            raise ValueError("prompt must be 1..400 characters")
    elif op == "gen_cancel":
        pass
    elif op == "example":                                # a ready-made skin preset (EXAMPLES): replaces every slot
        if c.get("name") not in EXAMPLES:
            raise ValueError("unknown example")
        out["name"] = c["name"]
    elif op == "enable":
        if not isinstance(c.get("on"), bool):
            raise ValueError("'on' must be true or false")
        out["on"] = c["on"]
    else:
        raise ValueError("unknown skin op")
    return out


GEN_SUFFIX = ", seamless tileable pattern, glowing on a pure black background"
GEN_TIMEOUT_S = 240.0
GEN_WAIT_S = 600.0               # an unsent job is dropped after this long (the show never got calm / the worker never idle)
GEN_LOAD_FREE_MB = 2600          # free VRAM wanted before a click may make the worker LOAD its model


class SkinState:
    """What the performer set: every slot of every environment, plus mod routings. Pure python. The controller owns
    one (saved in live_state.json, scenes and show files); `step()` resolves the modulated values into `frame`, which
    is what the engine reads. `frame` only holds slots that HAVE a source: with none it is empty and nothing runs."""

    def __init__(self, on=True):
        # `on`: the feature switch. The CONTROLLER creates its state with on=False: skins ship INACTIVE (no texture
        # array, no extra shader programs, nothing drawn) until the performer switches them on in the panel
        # ({"cmd": "skin", "op": "enable", "on": true}; remembered in live_state.json as settings.skins_on).
        # While off, `frame` stays empty whatever the slots hold, so the engine never sees a skin.
        self.on = bool(on)
        self.slots = {env: {s: default_slot() for s in e["slots"]} for env, e in ENVS.items()}
        self.mods = {}
        self.frame = {}                  # env -> slot -> resolved dict (reused objects: no per-frame allocation)
        self.edits = 0
        self.gen = None                  # a pending "generate a skin" job: dict(env, slot, prompt, tag, src0, known, t, sent)
        self.gen_note = ""
        self.gen_n = 0                   # job tags count up: results are matched by the worker's job id, not by prompt
        self.cancel_ai = False           # set when a job that was already SENT is cancelled: the owner cancels the worker
        self.cut = 0                     # bumps when the next change must be a cut, not a fade (PANIC)
        self._rebuild()

    # ---- (de)serialise
    def to_json(self):
        raw = dict(slots={f"{e}:{s}": d for e, ss in self.slots.items() for s, d in ss.items()}, mods=self.mods)
        return clean_skins(raw)

    def load(self, raw):
        clean = clean_skins(raw)
        for env, e in ENVS.items():
            for s in e["slots"]:
                self.slots[env][s] = clean_slot(clean["slots"].get(f"{env}:{s}"), env)
        self.mods = {k: [dict(m) if m else None for m in pair] for k, pair in clean["mods"].items()}
        self.edits += 1
        self._rebuild()

    restore = load

    def _rebuild(self):
        """Rebuild `frame` after a structural change (a source set or cleared, a load)."""
        frame = {}
        for env, ss in (self.slots.items() if self.on else ()):
            for s, d in ss.items():
                lays = [None if (x is None or x["src"] is None) else
                        dict(x, gate=list(x["gate"]), cols=[list(c) for c in x["cols"]], src=dict(x["src"]))
                        for x in d["layers"]]
                if d["src"] is not None or any(lays):
                    frame.setdefault(env, {})[s] = dict(d, tint=list(d["tint"]),
                                                        src=dict(d["src"]) if d["src"] else None, layers=lays)
        self.frame = frame
        self._modded = [(k, parse_key(k), "skin:" + k) for k in self.mods]

    def any(self):
        return bool(self.frame)

    def enable(self, on):
        """The feature switch. Off: nothing reaches the engine (it fades out and frees its texture array) and a pending
        generation is dropped; the slots keep what they hold for the next time."""
        self.on = bool(on)
        if not self.on:
            self.gen_cancel()
        self.edits += 1
        self._rebuild()

    # ---- commands
    def assign(self, env, slot, kind, name):
        src = clean_src(dict(kind=kind, name=name))
        if src is None or env not in ENVS or slot not in ENVS[env]["slots"]:
            return False
        self.slots[env][slot]["src"] = src
        self.edits += 1
        self._rebuild()
        return True

    def do(self, c):
        """Apply a VALIDATED command (see validate)."""
        op = c["op"]
        if op == "set":
            d = self.slots[c["env"]][c["slot"]]
            for f, v in c["set"].items():
                d[f] = list(v) if f == "tint" else v
            self.edits += 1
            self._rebuild()
        elif op == "assign":
            self.assign(c["env"], c["slot"], c["kind"], c["name"])
        elif op == "layer":
            lays = self.slots[c["env"]][c["slot"]]["layers"]
            lay = lays[c["i"]] or default_layer()
            for f, v in c["set"].items():
                lay[f] = [list(x) for x in v] if f == "cols" else (list(v) if isinstance(v, list) else v)
            lays[c["i"]] = lay
            self.edits += 1
            self._rebuild()
        elif op == "layer_clear":
            self.slots[c["env"]][c["slot"]]["layers"][c["i"]] = None
            self.edits += 1
            self._rebuild()
        elif op == "clear":
            self.slots[c["env"]][c["slot"]]["src"] = None
            self.edits += 1
            self._rebuild()
        elif op == "reset":
            for env in ([c["env"]] if c.get("env") else list(ENVS)):
                for s in ENVS[env]["slots"]:
                    self.slots[env][s] = default_slot()
                    for f in NUM:
                        self.mods.pop(f"{env}:{s}:{f}", None)
            self.edits += 1
            self._rebuild()
        elif op == "replace":
            self.load(c["skins"])
        elif op == "example":
            self.load(EXAMPLES[c["name"]])
        elif op == "mod":
            key = f"{c['env']}:{c['slot']}:{c['f']}"
            pair = self.mods.get(key) or [None, None]
            pair[c["mslot"]] = (None if c["src"] in ("", "none", "off") or c["amt"] == 0.0 else
                                dict(src=c["src"], amt=c["amt"], rate=c["rate"], phase=c["phase"]))
            if any(pair):
                self.mods[key] = pair
            else:
                self.mods.pop(key, None)
            self._rebuild()
        elif op == "enable":
            self.enable(c["on"])
        elif op == "gen":
            self.gen_cancel()
            if not self.on:
                self.gen_note = "skins are switched off"
                return
            self.gen_n += 1
            src = self.slots[c["env"]][c["slot"]]["src"]
            self.gen = dict(env=c["env"], slot=c["slot"], prompt=c["prompt"], tag=f"skin{self.gen_n}",
                            src0=dict(src) if src else None, known=None, t=None, sent=False)
            self.gen_note = "waiting for the AI worker"
        elif op == "gen_cancel":
            self.gen_cancel()

    def gen_cancel(self, note=""):
        """Drop the pending generate-a-skin job. If it was already sent, `cancel_ai` asks the owner to stop the worker."""
        if self.gen is not None and self.gen["sent"]:
            self.cancel_ai = True
        self.gen, self.gen_note = None, note

    def panic(self):
        """PANIC: every slot procedural NOW (a cut: `cut` tells the engine not to fade), settings kept, generation off."""
        for ss in self.slots.values():
            for d in ss.values():
                d["src"] = None
                for lay in d["layers"]:
                    if lay is not None:
                        lay["src"] = None
        self.gen_cancel()
        self.cut += 1
        self.edits += 1
        self._rebuild()

    def midi(self, env, slot, field, val, rel=None, pad=False, press=True):
        """A bound control. A knob / fader (CC): val 0..1 is ALWAYS the absolute position across the field's range, also
        at its end stops; rel = relative encoder steps (a stepped field moves one step per click, clamped). pad=True
        (a note): a stepped field steps to its next option on press and wraps."""
        d = self.slots[env][slot]
        if field in INT:
            lo, hi = INT[field][:2]
            hi = min(hi, len(ENVS[env]["maps"]) - 1) if field == "map" else hi
            if pad:
                if not press:
                    return
                d[field] = lo if d[field] >= hi else d[field] + 1
            elif rel is not None:
                d[field] = int(min(max(d[field] + (rel > 0) - (rel < 0), lo), hi))
            else:
                d[field] = int(min(max(round(lo + (hi - lo) * min(1.0, max(0.0, val))), lo), hi))
        else:
            lo, hi = NUM[field][:2]
            v = d[field] + rel * (hi - lo) / 127.0 if rel is not None else lo + (hi - lo) * min(1.0, max(0.0, val))
            d[field] = clean_field(field, v)
        self.edits += 1
        self._rebuild()

    # ---- "generate a skin": the controller drives this with its AI link (see control.py)
    def gen_tick(self, now, ai_status, submit, strained=False, result=None, vram_free_mb=None):
        """One step of the pending generate-a-skin job. `ai_status`: AIClient.status(); `submit(job)`: AIClient.submit;
        `result(tag)`: AIClient.tagged (file name of the finished job with that tag, matched by the worker's job id).
        The job is only SENT while the governor is calm and the worker has nothing else to do (lowest priority). If the
        worker has not loaded its model yet, sending would load about 2.5 GB onto the GPU: that only happens while
        calm AND with GEN_LOAD_FREE_MB of VRAM free (unknown = allowed). The finished image goes onto the slot unless
        the performer has put something else there since. Returns the new texture name once, else None."""
        g = self.gen
        if g is None:
            return None
        if not g["sent"]:
            if g["t"] is None:
                g["t"] = now
            if now - g["t"] > GEN_WAIT_S:
                self.gen_cancel("gave up waiting: the show stayed busy. Try again later")
                return None
            if strained:
                self.gen_note = "waiting: the show is under load"
                return None
            if not ai_status.get("available"):
                self.gen_note = "the AI worker is not running"
                if now - g["t"] > 20.0:
                    self.gen_cancel("no AI worker: nothing was generated")
                return None
            if ai_status.get("busy") or ai_status.get("queue") or ai_status.get("pending") or ai_status.get("loading"):
                self.gen_note = "waiting for the AI worker to finish its other jobs"
                return None
            if not ai_status.get("loaded"):
                if vram_free_mb is not None and vram_free_mb < GEN_LOAD_FREE_MB:
                    self.gen_note = (f"waiting: the AI model is not loaded and only {int(vram_free_mb)} MB of VRAM is free "
                                     f"(it needs about {GEN_LOAD_FREE_MB})")
                    return None
                self.gen_note = "loading the AI model, then generating"
            else:
                self.gen_note = "generating"
            g["known"] = {d.get("name") for d in ai_status.get("done") or []}
            g["sent"], g["t"] = True, now
            submit(dict(prompt=(g["prompt"] + GEN_SUFFIX)[:500], negative="", n=1, init=None, strength=0.45, show=False,
                        tag=g["tag"]))
            return None
        name = None
        if result is not None:
            name = result(g["tag"])                      # matched by job id inside the AI link
        else:                                            # a link without tags: a NEW file with exactly our prompt
            want = (g["prompt"] + GEN_SUFFIX)[:80]
            for d in ai_status.get("done") or []:
                if d.get("name") and d["name"] not in g["known"] and str(d.get("prompt") or "")[:80] == want:
                    name = d["name"]
        if name:
            self.gen, self.gen_note = None, ""
            if self.slots[g["env"]][g["slot"]]["src"] != g["src0"]:
                self.gen_note = f"{name} is in the library; the slot was changed meanwhile, so it was left alone"
                return None
            if not self.assign(g["env"], g["slot"], "ai", name):
                return None
            self.slots[g["env"]][g["slot"]]["tile"] = 0      # the local generator makes true seamless tiles: plain
            self._rebuild()                                  # repeat, no mirror creases
            return name
        if now - g["t"] > GEN_TIMEOUT_S:
            self.gen_cancel("the AI worker did not answer")
        return None

    # ---- per frame
    def step(self, dt=0.0, clock=None, source=None):
        """Resolve modulated numeric fields into `frame`. With no skin assigned this returns at once."""
        if not self.frame:
            return self.frame
        for env, ss in self.frame.items():           # copy the plain values (a handful of floats; no new objects)
            live = self.slots[env]
            for s, out in ss.items():
                d = live[s]
                for f in FIELDS:
                    out[f] = d[f]
                out["tint"][0], out["tint"][1], out["tint"][2] = d["tint"]
        if source is not None and self._modded:
            for key, t, target in self._modded:
                if t is None:
                    continue
                env, s, f = t
                fe = self.frame.get(env)
                out = fe.get(s) if fe else None
                if out is None:
                    continue
                lo, hi = NUM[f][:2]
                m = 0.0
                for n, sl in enumerate(self.mods[key]):
                    if sl:
                        m += sl["amt"] * source(sl, target, n)
                out[f] = min(max(out[f] + m * (hi - lo), lo), hi)
        return self.frame

    def state(self):
        """For the panel: only what differs from the defaults (the state JSON has a size budget)."""
        j = self.to_json()
        out = dict(on=self.on, slots=j["slots"], mods=j["mods"], edits=self.edits)
        if self.gen is not None or self.gen_note:
            out["gen"] = dict(env=self.gen["env"], slot=self.gen["slot"], prompt=self.gen["prompt"][:80],
                              note=self.gen_note) if self.gen is not None else dict(note=self.gen_note)
        return out


def meta():
    """Static description for the panel."""
    return dict(
        envs=[dict(id=k, label=e["label"], slots=list(e["slots"]), maps=list(e["maps"]), host=e["host"])
              for k, e in ENVS.items()],
        num={f: dict(lo=v[0], hi=v[1], default=v[2], label=v[3]) for f, v in NUM.items()},
        int={f: dict(lo=v[0], hi=v[1], default=v[2], label=v[3]) for f, v in INT.items()},
        mix=list(MIX_NAMES), tile=list(TILE_NAMES), kinds=list(KINDS), max_layers=MAX_LAYERS,
        overlays=dict(n=MAX_OVERLAYS, blend=list(BLEND_NAMES),
                      num={f: dict(lo=v[0], hi=v[1], default=v[2], label=v[3]) for f, v in LNUM.items()},
                      int={f: dict(lo=v[0], hi=v[1], default=v[2], label=v[3]) for f, v in LINT.items()}),
        layer_sources=dict(LAYER_SOURCES), examples=sorted(EXAMPLES))


# ------------------------------------------------------------------ GLSL chunk (pasted into the host shader)

def glsl(nslots=MAX_SLOTS, pal="palC"):
    """Uniforms + helpers. The host calls skLight (3D worlds), skTap / skTri (ONE fetch / THREE fetches) and skMix.
    `pal` = the host's palette lookup, float -> vec3."""
    return """
// ---- environment skins (materials.py). Per slot: skA = layer A, layer B, crossfade, amount;
// skB = mapping, mix mode, tiling, palette-follow; skC = uv matrix (column major); skD = phase u, v, half texel,
// brightness; skE = tint rgb, unused.
uniform sampler2DArray texSkin, texSkinL;            // images (unit 22) / live engine layers (unit 23, LayerHub)
uniform sampler2DArray texSkinM, texSkinLM;          // the same arrays through a MIRRORED_REPEAT sampler (24 / 25)
uniform float skMS;                                  // 1 = the mirrored samplers are bound (else: fold + clamp in maths)
uniform vec4 skA[%(n)d], skB[%(n)d], skC[%(n)d], skD[%(n)d], skE[%(n)d];
// skF = base present, overlays on, -, -. Overlay k = slot*%(o)d + j: skLA = source code, blend, opacity, remap + 2*mirror;
// skLB = uv matrix; skLC = phase u, v, patch amount, patch size; skLD = channel gains rgba; skLE = patch drift xy,
// threshold, lit; skLR[k*4 + c] = the colour of channel c (remap).
uniform vec4 skF[%(n)d], skLA[%(m)d], skLB[%(m)d], skLC[%(m)d], skLD[%(m)d], skLE[%(m)d], skLR[%(m4)d];
float skZ = 0.5;                                      // depth of this pixel, 0 near .. 1 far (set by the host)
float skSp = 0.0;                                     // specular glint of the form light (set by skLight)
float skDim = 1.0;                                    // the host multiplies its own haze / glow by this
vec2 skU = vec2(0.0), skGx = vec2(0.0), skGy = vec2(0.0);   // the mapping of the last skTap / skTri (overlays reuse it)
// source code < 100: a layer of the image array; >= 100: layer (code - 100) of the live-layer array
vec4 skSample(float code, vec2 uv, vec2 gx, vec2 gy, bool mir){
  if (mir) {                                         // hardware mirrored repeat: seamless for ANY image, at every mip
    if (code > 99.5) return textureGrad(texSkinLM, vec3(uv, code - 100.0), gx, gy);
    return textureGrad(texSkinM, vec3(uv, code), gx, gy);
  }
  if (code > 99.5) return textureGrad(texSkinL, vec3(uv, code - 100.0), gx, gy);
  return textureGrad(texSkin, vec3(uv, code), gx, gy);
}
// "floating patches" field 0..1: two domain-warped sine waves (4 sin, no hash, no fetch); irregular blobs that drift
float skNoise(vec2 p){
  p *= 3.1;
  float a = sin(p.x + 1.7*sin(p.y*0.71 + 0.3)), b = sin(p.y*1.13 + 1.5*sin(p.x*0.83 + 1.9));
  return 0.5 + 0.25*(a + b);
}
vec3 skFetch(int s, vec2 uv, vec2 gx, vec2 gy){
  bool mir = skB[s].z > 0.5 && skMS > 0.5;
  if (skB[s].z > 0.5 && !mir) {                       // fallback mirrored repeat in maths; the fold is kept
    uv = 1.0 - abs(mod(uv, 2.0) - 1.0);               // half a (mip) texel away from the edge so the wrap never bleeds in
    float a = dot(gx, gx), b = dot(gy, gy);
    float e = clamp(max(skD[s].z, sqrt(max(min(a, b), max(a, b)*0.0625))), 0.0, 0.5);
    uv = clamp(uv, e, 1.0 - e);
  }
  vec3 c = skSample(skA[s].x, uv, gx, gy, mir).rgb;
  if (skA[s].z > 0.002) {                             // only while a swap fades: the new image sweeps in from the
    float e = mix(-0.25, 1.25, skA[s].z);             // near end to the far end (a wipe along depth, no grey double exposure)
    c = mix(c, skSample(skA[s].y, uv, gx, gy, mir).rgb, 1.0 - smoothstep(e - 0.18, e + 0.18, skZ));
  }
  return c;
}
// ONE fetch. uv = the mapping's two coordinates, gx / gy = their change across one pixel (analytic: no derivative
// seams at the atan cut or at silhouettes; the hardware does the anisotropic filtering).
vec3 skTap(int s, vec2 uv, vec2 gx, vec2 gy){
  skU = uv; skGx = gx; skGy = gy;
  if (skF[s].x < 0.5) return vec3(0.0);               // no base image: overlays go onto the world's own look
  mat2 M = mat2(skC[s].xy, skC[s].zw);
  return skFetch(s, M*uv + skD[s].xy, M*gx, M*gy);
}
// THREE fetches (the expensive mapping; only run when the slot asks for it). zr = distance along the travel axis.
vec3 skTri(int s, vec3 p, vec3 n, float f){
  mat2 M = mat2(skC[s].xy, skC[s].zw);
  vec3 w = abs(n); w *= w; w *= w; w *= w; w /= (w.x + w.y + w.z + 1e-5);       // ^8: crisp, no double exposure
  skU = w.x > w.y ? (w.x > w.z ? p.yz : p.xy) : (w.y > w.z ? p.xz : p.xy);   // overlays: the dominant plane (1 fetch)
  skGx = vec2(f*1.4, 0.0); skGy = vec2(0.0, f*1.4);
  if (skF[s].x < 0.5) return vec3(0.0);
  vec2 gx = M*vec2(f*1.4, 0.0), gy = M*vec2(0.0, f*1.4), o = skD[s].xy;
  return skFetch(s, M*p.yz + o, gx, gy)*w.x + skFetch(s, M*p.xz + o, gx, gy)*w.y + skFetch(s, M*p.xy + vec2(o.x, 0.0), gx, gy)*w.z;
}
// Form light for a raymarched surface: N = surface normal (towards the camera side), rd = view ray, t = distance,
// steps = march steps taken (cheap ambient occlusion). Returns diffuse x AO x a fog that ends in true black, and
// leaves a soft specular glint in skSp and the depth in skZ. Analytic: no extra march, no fetch.
float skLight(vec3 N, vec3 rd, float t, float steps){
  const vec3 L = vec3(0.2120, 0.7420, -0.6360);
  // key light (wrapped) x a headlamp falloff: walls facing the camera bright, grazing walls and creases fall off
  float ndl = clamp(dot(N, L)*0.50 + 0.72, 0.36, 1.0)*(0.72 + 0.28*clamp(dot(N, -rd), 0.0, 1.0));
  float sp = max(dot(reflect(rd, N), L), 0.0); sp *= sp; sp *= sp; sp *= sp;       // ^16 by squaring (no pow)
  skSp = sp*sp*0.45;
  skZ = clamp(t*0.04, 0.0, 1.0);
  return ndl*exp(-(steps*0.006 + t*t*0.0017));
}
// One overlay layer of a skin stack on top of `o` (ONE fetch; patches = 4 hashes). Same mapping as the base.
vec3 skLayer(int k, vec3 o, float shade){
  vec4 A = skLA[k];
  mat2 M = mat2(skLB[k].xy, skLB[k].zw);
  vec2 uv = M*skU + skLC[k].xy, gx = M*skGx, gy = M*skGy, q = uv;
  bool mir = A.w > 1.5 && skMS > 0.5;                // mirror (any layer is seamless) / plain repeat
  if (A.w > 1.5 && !mir) uv = 1.0 - abs(mod(uv, 2.0) - 1.0);
  vec4 c = skSample(A.x, uv, gx, gy, mir)*skLD[k];
  vec3 col = mod(A.w, 2.0) > 0.5 ? skLR[k*4].rgb*c.r + skLR[k*4 + 1].rgb*c.g + skLR[k*4 + 2].rgb*c.b + skLR[k*4 + 3].rgb*c.a
                                 : c.rgb;
  if (skLC[k].z > 0.001) {                            // floating patches: a drifting noise window on the layer
    float t = skLE[k].z, nz = skNoise(q*skLC[k].w + skLE[k].xy);
    col *= mix(1.0, smoothstep(t - 0.12, t + 0.12, nz), skLC[k].z);
  }
  col *= mix(1.0, clamp(shade*1.6, 0.0, 1.3), skLE[k].w);       // the world's light: crevices stay dark
  float op = A.z, b = A.y, l = dot(col, vec3(0.299, 0.587, 0.114));
  if (b < 0.5) return o + col*op;
  if (b < 1.5) return 1.0 - (1.0 - min(o, 1.0))*(1.0 - min(col*op, 1.0));
  if (b < 2.5) return o*mix(vec3(1.0), 0.15 + 1.7*col, min(op, 1.0));
  if (b < 3.5) return mix(o, col, clamp(l*2.5, 0.0, 1.0)*min(op, 1.0));
  return o*mix(1.0, clamp(l*2.0, 0.0, 1.0), min(op, 1.0));
}
// proc = the world's own colour at this point, S = the skin, lines = the world's line mask (0..1), shade = the form
// shading (light x the world's own relief) so the surface still reads as a lit surface, not a sticker.
vec3 skMix(int s, vec3 proc, vec3 S, float lines, float shade){
  S = max((S - 0.05)*1.0526, 0.0);                    // black floor: dark paint is no light at all
  float l = dot(S, vec3(0.299, 0.587, 0.114));
  S *= (0.72 + 0.72*l)*smoothstep(0.02, 0.24, l);     // local contrast: lights up, murk down (black = no light)
  if (skB[s].w > 0.001) S = mix(S, %(pal)s(l*0.85 + 0.08)*l*1.7, skB[s].w);
  S = min(S*skE[s].rgb*skD[s].w, 1.25);
  l = dot(S, vec3(0.299, 0.587, 0.114));
  vec3 pc = min(proc, 1.0), o;
  float m = skB[s].y, pl = dot(pc, vec3(0.299, 0.587, 0.114));
  if (m < 0.5) o = S*(shade*3.0) + vec3(1.0, 0.95, 0.88)*(skSp*l*shade*1.6);   // Replace: the skin alone, lit
  else if (m < 1.5) o = proc*(0.22 + 2.1*S);          // the world's own light takes the skin's colour and grain
  else if (m < 2.5) { vec3 q = S*(shade*0.85); o = max(pc, q) + 0.25*min(pc, q); }     // lighter of the two: no milky haze
  else if (m < 3.5) o = proc*(0.50 + 1.5*S) + S*(pl*0.50 + 0.10*shade);
  else o = S*(lines*1.9*(0.55 + 0.45*shade)) + proc*(1.0 - lines)*0.10;
  if (skF[s].x < 0.5) o = proc;                       // no base image: the world's own look carries the overlays
  if (skF[s].y > 0.5) {
    for (int j = 0; j < %(o)d; j++) {
      int kk = s*%(o)d + j;
      if (skLA[kk].z > 0.001) o = skLayer(kk, o, shade);
    }
  }
  vec3 k = max(o - 1.15, 0.0);                         // soft knee: a bright skin rolls off, it never clips to white
  o = min(o, 1.15) + k/(1.0 + 2.5*k);                  // (the worlds output up to 1.6 = full white)
  skDim = min(skDim, 1.0 - 0.95*skA[s].w);             // under a skin the world's coloured haze goes (black = no light)
  return mix(proc, o, skA[s].w);
}
""" % dict(n=nslots, pal=pal, o=MAX_OVERLAYS, m=nslots * MAX_OVERLAYS, m4=nslots * MAX_OVERLAYS * 4)


GLSL = glsl()


# ------------------------------------------------------------------ library (GL)

MAX_PIXELS = 60_000_000


def mip_chain(level0):
    """(s, s, 4) uint8 -> [level 0, level 1, ... 1 x 1] by 2 x 2 box filtering (numpy; used when a loader only returns
    the base image)."""
    out = [np.ascontiguousarray(level0)]
    a = out[0]
    while a.shape[0] > 1:
        b = a.astype("u2")
        a = ((b[0::2, 0::2] + b[1::2, 0::2] + b[0::2, 1::2] + b[1::2, 1::2] + 2) >> 2).astype("u1")
        out.append(np.ascontiguousarray(a))
    return out


def load_image(path, size):
    """Decode, resize to size x size RGBA8 AND build the whole mip chain (worker thread: the render thread never
    generates mips). Returns ([level 0, level 1, ...] as uint8 arrays, aspect) or raises."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        if w * h > MAX_PIXELS:
            raise ValueError("image too large")
        im.draft("RGB", (size * 2, size * 2))
        if im.mode == "P":
            im = im.convert("RGBA")
        f = min(im.size) // (size * 2)
        if f >= 2:                                       # shrink BEFORE compositing / filtering at full resolution (RAM)
            im = im.reduce(f)
        if im.mode in ("RGBA", "LA"):
            im = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 255)), im)
        im = im.convert("RGB").resize((size, size), Image.LANCZOS)
        levels, s = [], size
        while True:
            a = np.empty((s, s, 4), dtype="u1")
            a[..., :3] = np.asarray(im)
            a[..., 3] = 255
            levels.append(a)
            if s == 1:
                break
            s //= 2
            im = im.resize((s, s), Image.BOX)
    return levels, w / max(h, 1)


def vram_bytes(size, layers=MAX_LAYERS):
    """RGBA8 texture array with a full mip chain."""
    total, s = 0, size
    while s >= 1:
        total += s * s * 4 * layers
        s //= 2
    return total


class RawGL:
    """glTexImage3D / glTexSubImage3D on the CURRENT context through ctypes: moderngl 5.13 cannot write one mip level
    of one layer of a texture array, and glGenerateMipmap would redo all 8 layers. Create it on the render thread."""

    T2A, RGBA, RGBA8, UBYTE, UNPACK_ALIGN, UNPACK_PBO = 0x8C1A, 0x1908, 0x8058, 0x1401, 0x0CF5, 0x88EF

    def __init__(self):
        import ctypes
        import sys
        if sys.platform != "win32":
            raise OSError("raw GL uploads are only wired up for Windows")
        C = ctypes
        gl = C.windll.opengl32
        get = gl.wglGetProcAddress
        get.restype, get.argtypes = C.c_void_p, [C.c_char_p]

        def proc(name, *args):
            addr = get(name)
            if not addr:
                raise OSError(f"{name.decode()} is not available")
            return C.WINFUNCTYPE(None, *args)(addr)

        i, u, p = C.c_int, C.c_uint, C.c_void_p
        self._sub = proc(b"glTexSubImage3D", u, i, i, i, i, i, i, i, u, u, p)
        self._img = proc(b"glTexImage3D", u, i, i, i, i, i, i, u, u, p)
        self._bind = gl.glBindTexture
        self._bind.restype, self._bind.argtypes = None, [u, u]
        self._store = gl.glPixelStorei
        self._store.restype, self._store.argtypes = None, [u, i]
        self._geti = gl.glGetIntegerv
        self._geti.restype, self._geti.argtypes = None, [u, p]
        self._err = gl.glGetError
        self._err.restype, self._err.argtypes = u, []
        self._c = C

    def _ready(self, glo):
        v = self._c.c_int(0)
        self._geti(self.UNPACK_PBO, self._c.byref(v))
        if v.value:                                      # somebody left a pixel-unpack buffer bound: not ours to touch
            return False
        self._bind(self.T2A, glo)                        # 2D-array target of the active unit: nothing else uses it
        self._store(self.UNPACK_ALIGN, 1)
        return True

    def alloc_levels(self, glo, size, layers):
        """Storage for mip levels 1.. (level 0 exists). False = could not (the caller falls back)."""
        if not self._ready(glo):
            return False
        while self._err():
            pass
        lvl, s = 1, size // 2
        while s >= 1:
            self._img(self.T2A, lvl, self.RGBA8, s, s, layers, 0, self.RGBA, self.UBYTE, None)
            lvl, s = lvl + 1, s // 2
        return self._err() == 0

    def sub(self, glo, level, y, layer, width, rows, arr, offset):
        """Rows y .. y + rows of one layer of one level, from arr (uint8, C order) at byte `offset`."""
        if not self._ready(glo):
            raise OSError("a pixel-unpack buffer is bound")
        self._sub(self.T2A, level, 0, y, layer, width, rows, 1, self.RGBA, self.UBYTE, arr.ctypes.data + offset)


class SkinLibrary:
    """Up to MAX_LAYERS resident skins in ONE texture array, allocated on first use (no skin = no VRAM). The loader
    thread decodes, resizes and builds the mip chain; the render thread only copies bytes: tick() uploads at most
    STRIP_BYTES per frame (level 0 in strips, then the smaller levels), never regenerates mips, and does nothing on
    a frame the caller reports as late."""

    STRIP_BYTES = 1 << 20            # upload at most ~1 MB per frame (256 rows at 1024)
    STARVE_S = 2.0                   # a show that is ALWAYS late still gets its skin after this long

    def __init__(self, ctx, roots, size=1024, layers=MAX_LAYERS, aniso=4.0, loader=load_image, sync=False, raw=None):
        self.ctx, self.roots = ctx, {k: Path(v) for k, v in roots.items()}
        self.layers, self.aniso, self._loader, self.sync = int(layers), float(aniso), loader, bool(sync)
        self.size = min(SIZES, key=lambda s: abs(s - int(size)))
        self.tex = None                  # allocated by the first tick() that has something to upload
        self._raw_arg, self.raw = raw, None      # raw: None = detect, False = moderngl only, else a RawGL-like object
        self.gen = 0                 # bumps whenever layer indices stop being valid (resize / release)
        self.uploads = self.mips = self.evictions = self.deferred = self.binds = self.skipped = self.allocs = 0
        self.upload_ms = 0.0
        self._key = [None] * self.layers         # layer -> key
        self._used = [0] * self.layers           # layer -> last tick it was needed
        self._where = {}                         # key -> layer (ready)
        self._aspect = {}
        self._pending = {}                       # key -> (size, gen) the decode was asked for
        self._failed = {}                        # key -> retry after (monotonic)
        self._job = None                         # the upload in progress: dict(key, layer, levels, steps, i)
        self._tick = 0
        self._late_since = None
        self._q_in, self._q_out = queue.Queue(), queue.Queue()
        self._stop = threading.Event()
        self._thread = None

    # ---- allocation
    def _alloc(self):
        import moderngl
        size = self.size
        self.tex = self.ctx.texture_array((size, size, self.layers), 4, dtype="f1")
        self.tex.repeat_x = self.tex.repeat_y = True
        self.raw = None
        glo = getattr(self.tex, "glo", None)
        if self._raw_arg is not None and self._raw_arg is not False:
            self.raw = self._raw_arg
        elif self._raw_arg is None and isinstance(glo, int):
            try:
                self.raw = RawGL()
            except Exception:
                self.raw = None
        if self.raw is not None:
            try:
                if not self.raw.alloc_levels(glo, size, self.layers):
                    self.raw = None
            except Exception:
                self.raw = None
        if self.raw is None:                             # fallback: let GL make the (empty) chain once
            self.tex.build_mipmaps()
        self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        try:
            self.tex.anisotropy = self.aniso
        except Exception:
            pass
        self.allocs += 1

    def _drop(self):
        if self.tex is not None:
            self.tex.release()
            self.tex = None
        self._key = [None] * self.layers
        self._where.clear()
        self._job = None
        self.gen += 1                                    # pending decodes of the old size are recognised and dropped

    def set_size(self, size):
        """Another resolution (governor / battery). The array is freed; every skin is decoded and uploaded again."""
        size = min(SIZES, key=lambda s: abs(s - int(size)))
        if size != self.size:
            self.size = size
            self._drop()
            self._pending.clear()
        return self.size

    def vram_bytes(self):
        return vram_bytes(self.size, self.layers) if self.tex is not None else 0

    def release(self):
        self._stop.set()
        self._q_in.put(None)
        self._drop()

    # ---- lookups (render thread)
    def path(self, key):
        root = self.roots.get(key[0])
        if root is None:
            return None
        p = root / key[1]
        try:
            if p.resolve().parent != root.resolve():
                return None
        except OSError:
            return None
        return p

    def request(self, key):
        """-> layer index, or -1 while the image is not resident yet (it is then queued for loading). No GL here."""
        lay = self._where.get(key)
        if lay is not None:
            self._used[lay] = self._tick
            return lay
        if key in self._pending or (self._job is not None and self._job["key"] == key):
            return -1
        bad = self._failed.get(key)
        if bad is not None and time.monotonic() < bad:
            return -1
        self._pending[key] = (self.size, self.gen)
        if self.sync:
            self._decode(key, self.size, self.gen)
        else:
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="skin-loader", daemon=True)
                self._thread.start()
            self._q_in.put((key, self.size, self.gen))
        return -1

    def aspect(self, key):
        return self._aspect.get(key, 1.0)

    def _decode(self, key, size, gen):
        try:
            p = self.path(key)
            if p is None or not p.is_file():
                raise FileNotFoundError(str(key[1]))
            data, asp = self._loader(p, size)
            if isinstance(data, (bytes, bytearray, memoryview)):         # a loader that only gives the base image
                data = mip_chain(np.frombuffer(data, dtype="u1").reshape(size, size, 4))
            self._q_out.put((key, size, gen, data, asp, None))
        except Exception as ex:
            self._q_out.put((key, size, gen, None, 1.0, f"{type(ex).__name__}: {ex}"))

    def _work(self):
        try:                                             # decoding must never steal the render thread's core
            import ctypes
            ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), -1)
        except Exception:
            pass
        while not self._stop.is_set():
            item = self._q_in.get()
            if item is None:
                return
            self._decode(*item)

    # ---- uploads (render thread)
    def busy(self):
        return self._job is not None or bool(self._pending) or not self._q_out.empty()

    def _victim(self, pinned):
        for i, k in enumerate(self._key):
            if k is None:
                return i
        best = None
        for i, k in enumerate(self._key):
            if k not in pinned and (best is None or self._used[i] < self._used[best]):
                best = i
        return best

    def _plan(self, levels):
        """Upload steps (level, first row, rows): level 0 in strips of at most STRIP_BYTES, then each smaller level."""
        steps = []
        for lvl, a in enumerate(levels):
            s = a.shape[0]
            rows = max(1, self.STRIP_BYTES // (s * 4))
            y = 0
            while y < s:
                n = min(rows, s - y)
                steps.append((lvl, y, n, n * s * 4))
                y += n
        return steps

    def tick(self, late=False, pinned=(), wanted=None):
        """Do at most ONE small piece of GL work (about STRIP_BYTES of texture upload). `late`: this frame is already
        over budget -> do nothing (unless the show has been late for STARVE_S). `pinned`: keys on screen, never
        evicted. `wanted`: keys some slot still asks for; a decoded image nobody wants any more is dropped without an
        upload. Returns True if it uploaded something."""
        self._tick += 1
        if self._job is None and self._q_out.empty():
            return False
        if late:
            now = time.monotonic()
            if self._late_since is None:
                self._late_since = now
            if now - self._late_since < self.STARVE_S:
                self.deferred += 1
                return False
        else:
            self._late_since = None
        t0 = time.perf_counter()
        job = self._job
        if job is not None and wanted is not None and job["key"] not in wanted:
            self._job = job = None                         # half uploaded and no longer wanted: its layer stays free
            self.skipped += 1
            if self._q_out.empty():
                return False
        if job is None:
            key, size, gen, data, asp, err = self._q_out.get_nowait()
            if self._pending.get(key) == (size, gen):      # only the decode that was asked for clears the request
                del self._pending[key]
            if gen != self.gen or size != self.size:
                return False                               # decoded for an array that no longer exists
            if err is not None:
                self._failed[key] = time.monotonic() + 5.0
                self.last_error = err
                return False
            if key in self._where:
                return False                               # already resident (a duplicate decode)
            if wanted is not None and key not in wanted:
                self.skipped += 1                          # tapped past it: no upload, no eviction
                return False
            if self.tex is None:
                self._alloc()
            lay = self._victim(pinned)
            if lay is None:
                self._failed[key] = time.monotonic() + 1.0  # every layer is on screen: try again in a moment
                return False
            old = self._key[lay]
            if old is not None:
                self._where.pop(old, None)
                self.evictions += 1
            self._key[lay] = None
            self._aspect[key] = asp
            levels = data if self.raw is not None else data[:1]
            self._job = job = dict(key=key, layer=lay, levels=levels, steps=self._plan(levels), i=0)
        s, steps, budget = self.size, job["steps"], self.STRIP_BYTES
        while job["i"] < len(steps):
            lvl, y, rows, nbytes = steps[job["i"]]
            if nbytes > budget and budget < self.STRIP_BYTES:
                break                                      # the next piece goes on the next frame
            a = job["levels"][lvl]
            w = a.shape[1]
            if self.raw is not None:
                self.raw.sub(getattr(self.tex, "glo", 0), lvl, y, job["layer"], w, rows, a, y * w * 4)
            else:
                self.tex.write(memoryview(a).cast("B")[y * w * 4:(y + rows) * w * 4], viewport=(0, y, job["layer"], w, rows, 1))
            self.uploads += 1
            budget -= nbytes
            job["i"] += 1
        if job["i"] >= len(steps):
            if self.raw is None:                           # fallback path only: GL builds the mips (all layers)
                self.tex.build_mipmaps()
                self.mips += 1
            self._key[job["layer"]] = job["key"]
            self._where[job["key"]] = job["layer"]
            self._used[job["layer"]] = self._tick
            self._job = None
        self.upload_ms = (time.perf_counter() - t0) * 1000.0
        return True

    last_error = ""

    def flush(self, pinned=(), timeout=10.0):
        """Tests / stills: finish every pending load now."""
        end = time.monotonic() + timeout
        while self.busy() and time.monotonic() < end:
            if not self.tick(False, pinned):
                time.sleep(0.002)

    def use(self):
        if self.tex is None:                             # only live layers on screen: no image array (none needed)
            return
        self.tex.use(UNIT)
        self.binds += 1

    def status(self):
        return dict(size=self.size, layers=self.layers, resident=len(self._where), loading=len(self._pending) + (self._job is not None),
                    vram_mb=round(self.vram_bytes() / 1048576.0, 1), evictions=self.evictions, deferred=self.deferred,
                    error=self.last_error[:120])


# ------------------------------------------------------------------ engine side

UPSIZE_CALM_S = 60.0             # back to the big array only after the governor has had nothing applied for this long


class _Slot:
    __slots__ = ("cur", "cur_key", "nxt", "nxt_key", "xf", "amt", "pu", "pv", "m", "follow", "gen")

    def __init__(self):
        self.cur = self.nxt = -1
        self.cur_key = self.nxt_key = None
        self.xf = self.amt = self.pu = self.pv = 0.0
        self.m = (1.0, 0.0, 0.0, 1.0)
        self.follow = 0.0
        self.gen = -1


class SkinEngine:
    """Crossfades, phases and uniforms for every skinned slot. One per GL context (the engine shares it between the
    LandPass and the scene shader). Per frame, in this order:

        sk.update(frame, dt, clock, eff, snap)       logic only, NO GL (safe inside a guard); frame = SkinState.frame
        sk.gl_step(late)                             the GL part: at most one upload piece / a resize (call unguarded)
        sk.advance(env, dz)                          the host's travel this frame (keeps the skin on its wall)
        sk.active(env)                               -> bool: does this world need its skinned shader this frame?
        sk.bind(prog, env)                           writes the uniform block (with this frame's phases) + binds the array
    """

    def __init__(self, ctx, roots, size=1024, aniso=4.0, sync=False, loader=load_image, lib=None, clock=time.monotonic):
        self.lib = lib if lib is not None else SkinLibrary(ctx, roots, size=size, aniso=aniso, sync=sync, loader=loader)
        self.want_size = self.lib.size
        self.base_size = self.lib.size
        self.level = 0
        self.suppress = False                       # the governor's last-resort step is applied: everything procedural
        self._clock = clock
        self._calm_since = clock()
        self._slots = {env: [_Slot() for _ in e["slots"]] for env, e in ENVS.items()}
        self._u = {env: np.zeros((5, MAX_SLOTS, 4), dtype="f4") for env in ENVS}       # preallocated uniform blocks
        self._on = {env: False for env in ENVS}
        self._pinned, self._wanted = set(), set()
        self.enabled = True                         # the engine mirrors SkinState.on here (off = free the array)
        self.fade_s = FADE_S
        self.binds = 0
        self.hub = None                             # LayerHub: live engine layers as sources (None = "layer" sources stay off)
        no = MAX_SLOTS * MAX_OVERLAYS
        self._uf = {env: np.zeros((MAX_SLOTS, 4), dtype="f4") for env in ENVS}         # base present, overlays on
        self._ul = {env: np.zeros((5, no, 4), dtype="f4") for env in ENVS}             # skLA..skLE
        self._ulr = {env: np.zeros((no * 4, 4), dtype="f4") for env in ENVS}           # skLR (channel colours)
        self._lph = {env: [[0.0, 0.0, 0.0, 0.0] for _ in range(no)] for env in ENVS}  # overlay phase u, v, patch x, y
        self._lm = {env: [(1.0, 0.0, 0.0, 1.0)] * no for env in ENVS}
        self._lfollow = {env: [0.0] * no for env in ENVS}
        self._ms = None                             # mirrored sampler objects: None = not tried yet, False = unavailable
        self._hg = -1                               # the hub generation the live indices belong to

    def release(self):
        self.lib.release()

    def active(self, env):
        return self._on[env]

    def any_active(self):
        for v in self._on.values():
            if v:
                return True
        return False

    def govern(self, quality):
        """What the resource governor has DECIDED (governor.skin_quality(): its applied reductions, which already
        carry its mode, its 3 s sustain and its 15 s recovery), never the raw strain level:
        0 nothing applied -> the configured size (grown back only after UPSIZE_CALM_S of calm);
        1 something applied -> 512 px; 2 its last-resort step -> every slot procedural until it is restored.
        Changes are made while nothing is on screen (every slot fades out first)."""
        q = min(max(int(quality), 0), 2)
        now = self._clock()
        if q > 0:
            self._calm_since = None
        elif self._calm_since is None:
            self._calm_since = now
        self.level = q
        self.suppress = q >= 2
        if q >= 1:
            self.want_size = SIZES[0]
        elif self.want_size != self.base_size and now - self._calm_since >= UPSIZE_CALM_S:
            self.want_size = self.base_size

    def vram_bytes(self):
        return self.lib.vram_bytes()

    def status(self):
        s = self.lib.status()
        s.update(level=self.level, procedural=self.suppress, active=[e for e, on in self._on.items() if on])
        if self.hub is not None:
            s["layers"] = self.hub.status()
        return s

    def advance(self, env, dz):
        for sl in self._slots[env]:
            if sl.amt > 0.0 and sl.follow != 0.0:
                sl.pu = (sl.pu + sl.m[1] * dz * sl.follow) % PH_WRAP
                sl.pv = (sl.pv + sl.m[3] * dz * sl.follow) % PH_WRAP
        ul = self._ul[env]
        for k, ph in enumerate(self._lph[env]):
            f = self._lfollow[env][k]
            if f != 0.0 and ul[0, k, 2] > 0.0:
                m = self._lm[env][k]
                ph[0] = (ph[0] + m[1] * dz * f) % PH_WRAP
                ph[1] = (ph[1] + m[3] * dz * f) % PH_WRAP

    def _req(self, key):
        """-> source code for the shader: image array layer (0..7), 100 + live layer index, or -1 (not ready)."""
        if key[0] == "layer":
            h = self.hub
            if h is None:
                return -1
            i = h.request(key[1])
            return 100 + i if i >= 0 else -1
        return self.lib.request(key)

    def _hub_gen(self):
        """The live-layer array was freed / reallocated: only slots SHOWING a live layer lose their index (image skins
        are not touched, so they never re-fade)."""
        hg = self.hub.gen if self.hub is not None else -1
        if hg != self._hg:
            self._hg = hg
            for slots in self._slots.values():
                for sl in slots:
                    if sl.cur >= 100 or sl.nxt >= 100:
                        sl.cur = sl.nxt = -1
                        sl.cur_key = sl.nxt_key = None
                        sl.xf = sl.amt = 0.0

    def _overlays(self, env, i, v, kind, base, base_z, follow, tile_base, bpm, dt, pinned, wanted):
        """Overlay uniforms of slot i. -> number of overlays with a ready source."""
        ul, ulr, n = self._ul[env], self._ulr[env], 0
        for j in range(MAX_OVERLAYS):
            k = i * MAX_OVERLAYS + j
            lay = v["layers"][j] if v is not None else None
            code = -1
            if lay is not None:
                key = (lay["src"]["kind"], lay["src"]["name"])
                code = self._req(key)
                if key[0] != "layer":
                    pinned.add(key)
                    wanted.add(key)
            if code < 0:
                ul[0, k, 2] = 0.0
                continue
            n += 1
            m = uv_matrix(kind, base, base_z, lay["scale"], lay["rot"], lay["tile"])
            self._lm[env][k] = m
            self._lfollow[env][k] = follow
            ph = self._lph[env][k]
            sp = lay["scroll"] * dt
            if sp:
                a = math.radians(lay["dir"])
                ph[0] = (ph[0] + math.cos(a) * sp) % PH_WRAP
                ph[1] = (ph[1] + math.sin(a) * sp) % PH_WRAP
            ns = lay["nspeed"] * dt
            if ns:
                ph[2] = (ph[2] + 0.83 * ns) % 1024.0
                ph[3] = (ph[3] + 0.56 * ns) % 1024.0
            ul[0, k, 0], ul[0, k, 1], ul[0, k, 2], ul[0, k, 3] = code, lay["blend"], lay["opacity"], lay["remap"] + 2 * lay["tile"]
            ul[1, k, 0], ul[1, k, 1], ul[1, k, 2], ul[1, k, 3] = m[0], m[2], m[1], m[3]
            ul[2, k, 2], ul[2, k, 3] = lay["noise"], lay["nscale"]
            ul[3, k, 0], ul[3, k, 1], ul[3, k, 2], ul[3, k, 3] = lay["gate"]
            ul[4, k, 2], ul[4, k, 3] = lay["thr"], lay["lit"]
            for c in range(4):
                ulr[k * 4 + c, 0], ulr[k * 4 + c, 1], ulr[k * 4 + c, 2] = lay["cols"][c]
        return n

    def idle(self):
        return not self._pinned and not self.lib.busy() and (self.enabled or self.lib.tex is None)

    def update(self, frame, dt, clock=None, late=False, eff=None, snap=False, gl=True):
        """Logic only. frame: SkinState.frame ({} = nothing skinned: returns after one check). eff: {env: effective
        mapping} from the host (the tunnel's "follow the travel mode"). snap: no fades this frame (PANIC cuts).
        gl=True also runs gl_step(late) (tests, stills); the engine passes gl=False and calls gl_step itself."""
        lib = self.lib
        if not frame and self.idle():
            if gl:
                self.gl_step(late)
            return
        dt = min(max(float(dt), 0.0), 0.1)
        bpm = float(clock.get("bpm", 120.0)) if clock else 120.0
        k = 1.0 if snap or self.fade_s <= 0 else min(1.0, dt / self.fade_s)
        hold = self.suppress or lib.size != self.want_size
        pinned, wanted = self._pinned, self._wanted
        pinned.clear()
        wanted.clear()
        self._hub_gen()
        for env, slots in self._slots.items():
            fe = frame.get(env) if frame else None
            names = ENVS[env]["slots"]
            u = self._u[env]
            on = False
            uf = self._uf[env]
            gen = lib.gen
            for i, sl in enumerate(slots):
                if names[i] in ENVS[env].get("own", ()):        # [beings] picker-only slot: its host loads it itself
                    continue
                v = fe.get(names[i]) if fe else None
                if sl.gen != gen:                             # an array was reallocated: indices are gone
                    sl.cur = sl.nxt = -1
                    sl.cur_key = sl.nxt_key = None
                    sl.xf = sl.amt = 0.0
                    sl.gen = gen
                want = None
                if v is not None and not hold and v["src"] is not None:
                    src = v["src"]
                    want = (src["kind"], src["name"])
                    wanted.add(want)
                nov = 0
                if v is not None and not hold:               # overlay layers (mapping set below, once the base is known)
                    e0 = int(v["map"])
                    if eff is not None and env in eff and e0 == 0:
                        e0 = eff[env]
                    elif ENVS[env]["host"] == "scene":
                        e0 = (EFF_LOGPOLAR, EFF_LOGPOLAR, EFF_TUBE, EFF_SCREEN)[e0]
                    g0 = geometry(env, names[i], e0)
                    nov = self._overlays(env, i, v, g0[0], g0[1], g0[2], g0[3], int(v["tile"]), bpm, dt, pinned, wanted)
                else:
                    self._ul[env][0, i * MAX_OVERLAYS:(i + 1) * MAX_OVERLAYS, 2] = 0.0
                uf[i, 1] = 1.0 if nov else 0.0
                target = 0.0
                if want is None and nov:
                    target = 1.0                              # overlays alone: the world's own look is the base
                if want is not None:
                    if sl.cur_key == want and sl.nxt_key is None:
                        target = 1.0
                    else:
                        if sl.nxt_key is not None and sl.nxt_key != want and sl.xf <= 0.0:
                            sl.nxt, sl.nxt_key = -1, None
                        lay = self._req(want)
                        if sl.cur_key is None:
                            if lay >= 0:
                                sl.cur, sl.cur_key = lay, want
                                target = 1.0
                        else:
                            target = 1.0
                            if sl.cur_key == want:            # went back to the picture on show mid-fade
                                sl.xf = max(0.0, sl.xf - k)
                                if sl.xf <= 0.0:
                                    sl.nxt, sl.nxt_key = -1, None
                            elif lay >= 0:
                                if sl.nxt_key != want and sl.xf <= 0.0:      # (mid-fade to something else: finish that first)
                                    sl.nxt, sl.nxt_key = lay, want
                                if sl.nxt_key is not None:
                                    sl.xf = min(1.0, sl.xf + k)
                                    if sl.xf >= 1.0:
                                        sl.cur, sl.cur_key, sl.nxt, sl.nxt_key, sl.xf = sl.nxt, sl.nxt_key, -1, None, 0.0
                sl.amt = min(target, sl.amt + k) if target > sl.amt else max(target, sl.amt - k)
                if sl.amt <= 0.0 and target <= 0.0:
                    if sl.cur_key is not None:
                        sl.cur, sl.cur_key, sl.nxt, sl.nxt_key, sl.xf = -1, None, -1, None, 0.0
                    u[0, i, 3] = 0.0
                    continue
                if want is None and sl.cur_key is not None and v is not None and sl.amt >= 1.0:
                    sl.cur, sl.cur_key, sl.nxt, sl.nxt_key, sl.xf = -1, None, -1, None, 0.0   # base cleared, overlays stay
                uf[i, 0] = 1.0 if sl.cur_key is not None else 0.0
                if sl.cur_key is not None:
                    pinned.add(sl.cur_key)
                    wanted.add(sl.cur_key)
                    self._req(sl.cur_key)
                if sl.nxt_key is not None:
                    pinned.add(sl.nxt_key)
                    wanted.add(sl.nxt_key)
                if v is None:                                 # fading out after a clear: keep the last look
                    u[0, i, 3] = sl.amt * u[4, i, 3]
                    on = on or sl.amt > 0.0
                    continue
                e = int(v["map"])
                if eff is not None and env in eff and e == 0:
                    e = eff[env]
                elif ENVS[env]["host"] == "scene":
                    e = (EFF_LOGPOLAR, EFF_LOGPOLAR, EFF_TUBE, EFF_SCREEN)[e]
                kind, base, base_z, follow = geometry(env, names[i], e)
                tile = int(v["tile"])
                sl.m = uv_matrix(kind, base, base_z, v["scale"], v["rot"], tile)
                sl.follow = follow
                sp = v["scroll"] * (bpm / 60.0 if v["sync"] else 1.0) * dt
                if sp:
                    a = math.radians(v["dir"])
                    sl.pu = (sl.pu + math.cos(a) * sp) % PH_WRAP
                    sl.pv = (sl.pv + math.sin(a) * sp) % PH_WRAP
                amt = sl.amt * v["amount"]
                u[0, i, 0], u[0, i, 1], u[0, i, 2], u[0, i, 3] = max(sl.cur, 0), max(sl.nxt, 0), sl.xf if sl.nxt >= 0 else 0.0, amt
                u[1, i, 0], u[1, i, 1], u[1, i, 2], u[1, i, 3] = e, v["mix"], tile, v["pal"]
                u[2, i, 0], u[2, i, 1], u[2, i, 2], u[2, i, 3] = sl.m[0], sl.m[2], sl.m[1], sl.m[3]     # column major
                u[3, i, 2], u[3, i, 3] = 0.5 / lib.size, v["bright"]     # the phases (u[3, i, 0:2]) are written in bind()
                u[4, i, 0], u[4, i, 1], u[4, i, 2] = v["tint"]
                u[4, i, 3] = v["amount"]
                on = on or amt > 0.0005
            self._on[env] = on
        if gl:
            self.gl_step(late)

    def gl_step(self, late=False):
        """The GL part of the frame: a pending resize (only while nothing is on screen) and at most one upload piece.
        GL errors raised here are real context trouble: the engine does NOT swallow them."""
        lib = self.lib
        if not self.enabled and lib.tex is not None and not self.any_active():
            lib._drop()                                  # skins were switched off: the VRAM goes back
            lib._pending.clear()
        if lib.size != self.want_size and not self.any_active():
            lib.set_size(self.want_size)
        if lib._job is not None or not lib._q_out.empty():
            lib.tick(late, self._pinned, self._wanted)
        else:
            lib._tick += 1

    def effective(self, env, slot_i=0):
        return int(self._u[env][1, slot_i, 0])

    def bind(self, prog, env):
        """Uniform block + the array on its unit. Call only while active(env), AFTER this frame's advance(): the
        phases are read here, so the skin moves with the geometry of the same frame."""
        u = self._u[env]
        for i, sl in enumerate(self._slots[env]):
            u[3, i, 0], u[3, i, 1] = sl.pu, sl.pv
        n = prog["skA"].array_length
        for j, name in enumerate(("skA", "skB", "skC", "skD", "skE")):
            if name in prog:
                prog[name].write(u[j, :n])
        if "skF" in prog:
            prog["skF"].write(self._uf[env][:n])
            uf = self._uf[env]
            if uf[:n, 1].any():                               # overlays: their block (+ this frame's phases)
                ul = self._ul[env]
                no = n * MAX_OVERLAYS
                for k, ph in enumerate(self._lph[env][:no]):
                    ul[2, k, 0], ul[2, k, 1] = ph[0], ph[1]
                    ul[4, k, 0], ul[4, k, 1] = ph[2], ph[3]
                for j, name in enumerate(("skLA", "skLB", "skLC", "skLD", "skLE")):
                    if name in prog:
                        prog[name].write(ul[j, :no])
                if "skLR" in prog:
                    prog["skLR"].write(self._ulr[env][:no * 4])
        if self._ms is None:                              # created on the render thread, once
            try:
                self._ms = (MirrorSampler(), MirrorSampler())
            except Exception:
                self._ms = False
        for name, unit in (("texSkin", UNIT), ("texSkinL", UNIT_L), ("texSkinM", UNIT_M), ("texSkinLM", UNIT_LM)):
            if name in prog:
                prog[name].value = unit
        if "skMS" in prog:
            prog["skMS"].value = 1.0 if self._ms else 0.0
        self.lib.use()
        if self.hub is not None:
            self.hub.use()
        if self._ms:
            if self.lib.tex is not None:
                self.lib.tex.use(UNIT_M)
                self._ms[0].use(UNIT_M)
            if self.hub is not None and self.hub.be is not None and getattr(self.hub.be, "arr", None) is not None:
                self.hub.be.arr.use(UNIT_LM)
                self._ms[1].use(UNIT_LM)
        self.binds += 1


# ------------------------------------------------------------------ LAYERS AS SKINS (live engine layers -> textures)
#
# Any engine layer (kené / bead studies, Folding Blocks, beadwork, beings, math stack, fractals, text, camera, video,
# the filtered output, the built-in 4-channel line layer) can be a skin source: {"kind": "layer", "name": <id>}.
# The LayerHub turns a referenced layer into one layer of a small RGBA8 texture array (512 px, mipmapped, unit 23):
#   * only while referenced: a source nobody requested for KEEP_S seconds gives its slot back (and its own pass, if
#     one was made for it); with nothing referenced for FREE_S the array is freed. Nothing referenced = run() returns
#     after one dict check: no GL call, no VRAM.
#   * at most once per frame, whatever the number of consumers (deck slots, the tunnel, creature bodies).
#   * reuse first: a layer the show already drew this frame is COPIED (one 512 px draw + mips), never drawn twice;
#     only a layer that is off in the mix is drawn by its own small (512 px) instance.
#   * static sources (the built-in line layer) are drawn once.
#   * governor-aware: quality 1 (any reduction applied) -> every other frame; 2 (last resort) -> frozen.
# One frame of latency: the hub runs after the layers have drawn; consumers sample last frame's copy.
#
# CONTRACT for other modules (beings / serpent bodies):
#   hub = <the engine's LayerHub>              (live_visuals: `skins.hub`, also passed to consumers)
#   i = hub.request("kene")                    every frame you need it; -> array layer index, or -1 (not ready yet)
#   hub.use()                                  binds the array to UNIT_L (23) before your draw
#   GLSL: uniform sampler2DArray texSkinL;  textureGrad(texSkinL, vec3(uv, i), dUVdx, dUVdy)  (RGBA8, mipmapped,
#         clamp-to-edge: fold / fract the uv yourself; mirror fold = 1 - abs(mod(uv, 2) - 1) makes any layer seamless)
#   For full stacks on a body, paste `materials.glsl()` and use skTap / skMix with a slot of your own, or copy skLayer.
#   hub.status() -> dict for the governor / panel.
LAYER_SOURCES = {                # id -> label (what the panel offers; the engine registers what it actually has)
    "lines4": "Line layer, 4 channels (R lines, G fine fill, B ground, A jewels)",
    "kene": "Kené study",
    "kene4": "Kené study, 4 channels (R lines, G fill, B ground, A accents)",
    "beadstudy": "Bead study",
    "blocks": "Folding Blocks",
    "beads": "Beadwork",
    "lines": "Sacred lines",
    "beings": "Beings",
    "stack": "Math stack",
    "fractal": "Math fractals",
    "text": "Text",
    "camera": "Crowd camera (silhouette mask)",
    "video": "Video loops",
    "output": "Filtered output (last frame)",
}

_COPY_FS = """
#version 330
uniform sampler2D src;
uniform vec2 px;                                      // one destination texel in source uv
uniform float mono;                                   // a one-channel source (camera mask): grey
in vec2 v_uv;
out vec4 f;
void main(){                                          // 2x2 box of bilinear taps: a clean downscale to 512 px
  vec2 o = px*0.25;
  f = 0.25*(texture(src, v_uv + vec2(-o.x, -o.y)) + texture(src, v_uv + vec2(o.x, -o.y))
          + texture(src, v_uv + vec2(-o.x, o.y)) + texture(src, v_uv + vec2(o.x, o.y)));
  if (mono > 0.5) f = vec4(f.rrr, 1.0);
}
"""

# The built-in kené-TYPE line layer (an original stepped-lozenge lattice, not a copy of any design), 4 semantic
# channels: R primary outlines, G fine meander fill, B cloth ground, A jewel accents. Seamless (whole cells per tile).
_LINES4_FS = """
#version 330
in vec2 v_uv;
out vec4 f;
uniform float N;
float band(float d, float w, float aa){ return 1.0 - smoothstep(w - aa, w + aa, abs(d)); }
void main(){
  vec2 q = v_uv*N, c = floor(q), g = fract(q) - 0.5;
  float aa = 1.5*N/512.0;
  // stepped lozenge: diamond metric quantised into stairs (8 steps per half cell)
  vec2 st = floor(abs(g)*16.0 + 0.5)/16.0;
  float dia = st.x + st.y, smooth_d = abs(g.x) + abs(g.y);
  float odd = mod(c.x + c.y, 2.0);
  float prim = band(dia - 0.5, 0.045, aa);                           // the lattice outlines
  prim = max(prim, band(dia - 0.25, 0.03, aa)*odd);                  // an inner lozenge on every other cell
  prim = max(prim, band(g.x, 0.03, aa)*step(0.5, 1.0 - odd)*step(dia, 0.5)*step(0.25, abs(g.y)));   // spines
  float inner = step(dia, 0.47);
  float fine = band(fract(smooth_d*9.0 + 0.5*odd) - 0.5, 0.12, 9.0*aa)*inner*(1.0 - prim)*step(0.06, smooth_d);
  float ground = inner*(1.0 - prim);
  vec2 cj = abs(g) - 0.5;                                            // jewels at the lattice junctions + odd centres
  float jew = 1.0 - smoothstep(0.055, 0.055 + aa, length(cj));
  jew = max(jew, (1.0 - smoothstep(0.05, 0.05 + aa, length(g)))*odd);
  f = vec4(prim, fine, ground, jew);
}
"""


class MirrorSampler:
    """A GL sampler object with MIRRORED_REPEAT wrap (moderngl 5.13 has no mirrored wrap), bound to one unit. With
    it the shader reads mirrored tiling straight from the hardware: no fold maths, no edge clamp, no streak bands."""

    def __init__(self, aniso=4.0):
        import ctypes
        import sys
        if sys.platform != "win32":
            raise OSError("raw GL samplers are only wired up for Windows")
        C = ctypes
        get = C.windll.opengl32.wglGetProcAddress
        get.restype, get.argtypes = C.c_void_p, [C.c_char_p]

        def proc(name, *args):
            addr = get(name)
            if not addr:
                raise OSError(f"{name.decode()} is not available")
            return C.WINFUNCTYPE(None, *args)(addr)
        u, i, f = C.c_uint, C.c_int, C.c_float
        gen = proc(b"glGenSamplers", i, C.POINTER(u))
        pi = proc(b"glSamplerParameteri", u, u, i)
        pf = proc(b"glSamplerParameterf", u, u, f)
        self._bind = proc(b"glBindSampler", u, u)
        sid = u(0)
        gen(1, C.byref(sid))
        if not sid.value:
            raise OSError("glGenSamplers gave no sampler")
        self.id = sid.value
        for pname, val in ((0x2802, 0x8370), (0x2803, 0x8370), (0x2801, 0x2703), (0x2800, 0x2601)):
            pi(self.id, pname, val)                      # WRAP_S/T = MIRRORED_REPEAT, trilinear
        pf(self.id, 0x84FE, float(aniso))                # anisotropy

    def use(self, unit):
        self._bind(unit, self.id)


class RawCopy:
    """glCopyImageSubData through ctypes (GL 4.3): a 2D texture level -> one layer of a 2D array level, GPU only."""
    T2D, T2A = 0x0DE1, 0x8C1A

    def __init__(self):
        import ctypes
        import sys
        if sys.platform != "win32":
            raise OSError("raw GL copies are only wired up for Windows")
        C = ctypes
        get = C.windll.opengl32.wglGetProcAddress
        get.restype, get.argtypes = C.c_void_p, [C.c_char_p]
        addr = get(b"glCopyImageSubData")
        if not addr:
            raise OSError("glCopyImageSubData is not available")
        i, u = C.c_int, C.c_uint
        self._copy = C.WINFUNCTYPE(None, u, u, i, i, i, i, u, u, i, i, i, i, i, i, i)(addr)
        self._err = C.windll.opengl32.glGetError
        self._err.restype = u

    def level(self, src, dst, layer, level, size):
        self._copy(src, self.T2D, level, 0, 0, 0, dst, self.T2A, level, 0, 0, layer, size, size, 1)


class _HubGL:
    """The GL side of a LayerHub (tests swap in a fake)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.arr = self.scratch = self.fbo = self.prog = self.vao = self.quad = None
        self.raw = None
        self.size = 0
        self._l4 = None                  # (prog, vao, tex, fbo) of the built-in line layer
        self.copy_fallbacks = 0

    def _quad(self, fs):
        if self.quad is None:
            self.quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        prog = self.ctx.program(vertex_shader=_HUB_VS, fragment_shader=fs)
        return prog, self.ctx.vertex_array(prog, [(self.quad, "2f", "in_pos")])

    def alloc(self, size, layers):
        import moderngl
        self.size = size
        self.arr = self.ctx.texture_array((size, size, layers), 4, dtype="f1")
        self.arr.build_mipmaps()                         # allocates the (empty) chain once
        self.arr.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.arr.repeat_x = self.arr.repeat_y = True     # plain repeat here; mirrored tiling reads unit 25 (sampler)
        try:
            self.arr.anisotropy = 4.0
        except Exception:
            pass
        self.scratch = self.ctx.texture((size, size), 4, dtype="f1")
        self.scratch.build_mipmaps()
        self.fbo = self.ctx.framebuffer([self.scratch])
        if self.prog is None:
            self.prog, self.vao = self._quad(_COPY_FS)
        if self.raw is None:
            try:
                self.raw = RawCopy()
            except Exception:
                self.raw = False

    def blit(self, tex, layer):
        import moderngl
        ctx, s = self.ctx, self.size
        prev, vp = ctx.fbo, ctx.viewport
        self.fbo.use()
        ctx.viewport = (0, 0, s, s)
        tex.use(UNIT_L)
        self.prog["src"].value = UNIT_L
        self.prog["px"].value = (1.0 / s, 1.0 / s)
        self.prog["mono"].value = 1.0 if getattr(tex, "components", 4) == 1 else 0.0
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.scratch.build_mipmaps()
        lv, n = 0, s
        if self.raw:
            while n >= 1:
                self.raw.level(self.scratch.glo, self.arr.glo, layer, lv, n)
                lv, n = lv + 1, n // 2
        else:                                            # no raw GL: through the CPU (slow, correct)
            self.arr.write(self.scratch.read(), viewport=(0, 0, layer, s, s, 1))
            self.arr.build_mipmaps()
            self.copy_fallbacks += 1
        prev.use()
        ctx.viewport = vp

    def use(self, unit=UNIT_L):
        if self.arr is not None:
            self.arr.use(unit)

    def release(self):
        for o in (self.arr, self.scratch, self.fbo):
            if o is not None:
                o.release()
        self.arr = self.scratch = self.fbo = None
        if self._l4 is not None:
            self._l4[2].release()
            self._l4[3].release()
            self._l4 = None

    def vram_bytes(self, layers):
        if self.arr is None:
            return 0
        return int((layers + 1) * self.size * self.size * 4 * 4 / 3)   # array + scratch, with mips

    def lines4(self, size):
        import moderngl
        ctx = self.ctx
        if self._l4 is None or self._l4[2].size != (size, size):
            if self._l4 is not None:
                self._l4[2].release()
                self._l4[3].release()
            prog, vao = self._l4[:2] if self._l4 is not None else self._quad(_LINES4_FS)
            tex = ctx.texture((size, size), 4, dtype="f1")
            self._l4 = (prog, vao, tex, ctx.framebuffer([tex]))
        prog, vao, tex, fbo = self._l4
        prev, vp = ctx.fbo, ctx.viewport
        fbo.use()
        ctx.viewport = (0, 0, size, size)
        prog["N"].value = 6.0
        vao.render(moderngl.TRIANGLE_STRIP)
        prev.use()
        ctx.viewport = vp
        return tex

    def probe(self, layer, level, n=8):
        """Test helper: an n x n grid of texels of one layer / level (uint8 RGBA)."""
        import moderngl
        ctx = self.ctx
        prog, vao = self._quad("""
#version 330
uniform sampler2DArray a; uniform float L, M; in vec2 v_uv; out vec4 f;
void main(){ f = textureLod(a, vec3(v_uv, L), M); }""")
        tex = ctx.texture((n, n), 4, dtype="f1")
        fbo = ctx.framebuffer([tex])
        prev, vp = ctx.fbo, ctx.viewport
        fbo.use()
        ctx.viewport = (0, 0, n, n)
        self.arr.use(UNIT_L)
        prog["a"].value, prog["L"].value, prog["M"].value = UNIT_L, float(layer), float(level)
        vao.render(moderngl.TRIANGLE_STRIP)
        out = np.frombuffer(fbo.read(components=4), dtype="u1").reshape(n, n, 4).copy()
        prev.use()
        ctx.viewport = vp
        for o in (fbo, tex, vao, prog):
            o.release()
        return out


_HUB_VS = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main(){ v_uv = in_pos*0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""


class LayerHub:
    """Live engine layers as skin sources (see the block comment above). One per GL context."""

    KEEP_S = 2.0                     # a source not requested for this long gives its array layer back
    FREE_S = 10.0                    # nothing requested for this long: the array is freed (0 MB)
    MAX_FAILS = 5

    def __init__(self, ctx, size=512, layers=4, backend=None, clock=time.monotonic):
        self.size, self.layers = int(size), int(layers)
        self.be = backend if backend is not None else _HubGL(ctx)
        self.clock = clock
        self._src = {}
        self._want = {}                  # name -> last request time
        self._slot = {}                  # name -> array layer
        self._ready = set()
        self._done = {}                  # name -> tick it was drawn / copied
        self._fails = {}
        self._tick = 0
        self._alloc = False
        self._last_use = None
        self.gen = 0                     # bumps when the array is freed (consumers drop their indices)
        self.copies = self.reused = self.own = self.skipped = 0
        self.ms = 0.0                    # CPU time of the last frame's runs (GPU cost: see the tests' timer queries)
        if hasattr(self.be, "lines4"):
            self.register("lines4", draw=self.builtin_lines4, static=True)

    # ---- registry (the engine registers what it has; consumers only use names)
    def register(self, name, draw=None, visible=None, phase="scene", static=False, release=None, label=None, every=1):
        """draw(size) -> a texture drawn by the source's OWN small instance (called only when the layer is not already
        on screen); visible() -> this frame's on-screen texture or None; phase "scene" (after the layers drew) or
        "post" (after the filters: the output); static: drawn once; release(): free the own instance;
        every: refresh every N frames (expensive sources)."""
        self._src[name] = dict(draw=draw, visible=visible, phase=phase, static=bool(static), release=release,
                               every=max(1, int(every)),
                               label=label or LAYER_SOURCES.get(name, name))
        self._fails.pop(name, None)
        self._ready.discard(name)

    def unregister(self, name):
        self._drop(name)
        self._src.pop(name, None)

    def sources(self):
        return {k: v["label"] for k, v in self._src.items()}

    def builtin_lines4(self, size):
        return self.be.lines4(size)

    # ---- consumers
    def request(self, name):
        """Every frame you need `name`. -> its array layer, or -1 while it holds nothing yet / unknown / no slot free."""
        if name not in self._src or self._fails.get(name, 0) >= self.MAX_FAILS:
            return -1
        self._want[name] = self.clock()
        i = self._slot.get(name)
        if i is None:
            used = set(self._slot.values())
            for j in range(self.layers):
                if j not in used:
                    self._slot[name] = i = j
                    break
            else:
                return -1
        return i if name in self._ready else -1

    def use(self):
        self.be.use(UNIT_L)

    # ---- per frame (GL)
    def run(self, phase="scene", quality=0):
        if not self._want:
            return
        t0 = time.perf_counter()
        for name in self._want:
            src = self._src.get(name)
            if src is None or src["phase"] != phase or name not in self._slot or self._done.get(name) == self._tick:
                continue
            if self._fails.get(name, 0) >= self.MAX_FAILS:
                continue
            ready = name in self._ready
            every = src["every"] * (2 if quality == 1 else 1)
            if ready and (src["static"] or quality >= 2 or self._tick % every):
                self.skipped += 1
                continue
            try:
                tex = src["visible"]() if src["visible"] is not None else None
                if tex is not None:
                    self.reused += 1
                elif src["draw"] is not None:
                    tex = src["draw"](self.size)
                    if tex is not None:
                        self.own += 1
                if tex is None:
                    continue
                if not self._alloc:
                    self.be.alloc(self.size, self.layers)
                    self._alloc = True
                self.be.blit(tex, self._slot[name])
                self._ready.add(name)
                self._done[name] = self._tick
                self._fails.pop(name, None)
                self.copies += 1
            except Exception as ex:
                n = self._fails[name] = self._fails.get(name, 0) + 1
                if n == 1 or n == self.MAX_FAILS:
                    print(f"[skins] layer source '{name}' failed ({n}/{self.MAX_FAILS}): {str(ex)[:200]}")
        self.ms = (time.perf_counter() - t0) * 1000.0

    def end_frame(self):
        self._tick += 1
        if not self._want and not self._alloc:
            return
        now = self.clock()
        for name in [n for n, t in self._want.items() if now - t > self.KEEP_S]:
            self._drop(name)
        if self._want:
            self._last_use = max(self._want.values())
        elif self._alloc and (self._last_use is None or now - self._last_use > self.FREE_S):
            self.be.release()
            self._alloc = False
            self._ready.clear()
            self.gen += 1

    def _drop(self, name):
        self._want.pop(name, None)
        self._slot.pop(name, None)
        self._ready.discard(name)
        self._done.pop(name, None)
        src = self._src.get(name)
        if src is not None and src["release"] is not None:
            try:
                src["release"]()
            except Exception:
                pass

    def release(self):
        for name in list(self._want):
            self._drop(name)
        if self._alloc:
            self.be.release()
            self._alloc = False
            self.gen += 1

    def vram_bytes(self):
        return self.be.vram_bytes(self.layers) if self._alloc else 0

    def probe(self, layer, level=0, n=8):
        return self.be.probe(layer, level, n)

    def status(self):
        return dict(size=self.size, layers=self.layers, live=sorted(self._slot), ready=sorted(self._ready),
                    copies=self.copies, reused=self.reused, own=self.own, skipped=self.skipped,
                    fails=dict(self._fails), vram_mb=round(self.vram_bytes() / 1048576.0, 1), ms=round(self.ms, 3),
                    sources=sorted(self._src))


# ------------------------------------------------------------------ example skin presets
# "Kené cloth world": every part of the torus wormhole + corridor carries a kené-TYPE line layer (the built-in
# 4-channel source, colour-remapped per channel) over an AI image skin; neighbouring parts scroll in opposite
# directions, floating patches gate the fine fill, a gold glow layer rides on top. Load with
# {"cmd": "skin", "op": "replace", "skins": materials.EXAMPLES["kene_cloth_world"]} (or `example` op).
_L4 = dict(kind="layer", name="lines4")
EXAMPLES = {
    "kene_cloth_world": dict(v=VERSION, mods={}, slots={
        "torus:tube": dict(src=dict(kind="ai", name="tex_0000001002.png"), scale=1.0, scroll=0.05, dir=0.0, layers=[
            dict(src=_L4, blend=0, opacity=1.0, remap=1, gate=[1.2, 0.9, 0.0, 1.6], noise=0.55, nscale=1.2, nspeed=0.2,
                 thr=0.45, scroll=0.12, dir=0.0, lit=0.7,
                 cols=[[1.0, 0.95, 0.85], [0.95, 0.96, 1.0], [0.04, 0.14, 0.6], [0.95, 0.06, 0.1]]),
            dict(src=_L4, blend=1, opacity=0.55, remap=1, gate=[1.0, 0.0, 0.0, 0.8], scale=2.0, scroll=0.12, dir=0.0,
                 lit=0.25, cols=[[1.0, 0.62, 0.12], [0, 0, 0], [0, 0, 0], [1.0, 0.8, 0.3]])]),
        "torus:chambers": dict(src=dict(kind="ai", name="tex_0000001005.png"), scroll=0.05, dir=180.0, layers=[
            dict(src=_L4, blend=0, remap=1, gate=[1.1, 0.6, 0.0, 1.4], scroll=0.12, dir=180.0, scale=1.5, lit=0.7,
                 cols=[[0.2, 1.0, 0.6], [0.8, 1.0, 0.9], [0, 0, 0], [0.13, 0.31, 1.0]]), None]),
        "corridor:walls": dict(src=dict(kind="ai", name="tex_0000001002.png"), bright=1.3, layers=[
            dict(src=_L4, blend=0, remap=1, gate=[1.3, 0.7, 0.0, 1.5], noise=0.5, nscale=1.0, nspeed=0.25, thr=0.5,
                 scroll=0.08, dir=0.0, lit=0.6), None]),
        "corridor:floor": dict(src=dict(kind="ai", name="tex_0000001005.png"), bright=1.3, layers=[
            dict(src=_L4, blend=3, remap=1, gate=[1.0, 0.5, 0.6, 1.2], scale=1.5, lit=0.8,
                 cols=[[1.0, 0.9, 0.7], [0.7, 0.8, 1.0], [0.25, 0.02, 0.18], [1.0, 0.1, 0.15]]), None]),
        "vortex:walls": dict(layers=[
            dict(src=_L4, blend=2, remap=1, gate=[1.4, 0.8, 0.5, 1.5], scale=1.0, scroll=0.1, dir=0.0, lit=0.5), None]),
        "hexdome:domes": dict(src=dict(kind="ai", name="tex_0000001003.png"), bright=1.5, layers=[
            dict(src=_L4, blend=0, remap=1, gate=[0.8, 0.0, 0.0, 2.0], scale=1.0, noise=0.4, nscale=2.0, lit=0.4,
                 cols=[[1.0, 0.95, 0.85], [0, 0, 0], [0, 0, 0], [1.0, 0.75, 0.2]]), None]),
        "tunnel:walls": dict(src=dict(kind="ai", name="tex_0000001007.png"), layers=[
            dict(src=_L4, blend=0, remap=1, gate=[1.0, 0.6, 0.0, 1.5], noise=0.5, nscale=1.5, nspeed=0.3, lit=0.0), None]),
    }),
}
EXAMPLES = {k: clean_skins(v) for k, v in EXAMPLES.items()}


def roots_for(textures=None, art=None, here=None):
    here = Path(here) if here else Path(__file__).parent
    return dict(ai=Path(textures) if textures else here / "textures", art=Path(art) if art else here / "art")


# ------------------------------------------------------------------ gallery / bench (standalone)

GALLERY = [  # (label, per-slot settings); image index into the sorted textures/ list
    ("replace", dict(mix=0, scale=1.0)),
    ("lines reveal", dict(mix=4, scale=1.5, pal=0.0)),
    ("multiply + palette", dict(mix=1, scale=0.75, pal=0.6, bright=1.4)),
]


def _images(root, skip=("kene",)):
    return [p.name for p in sorted(Path(root).glob("*")) if p.suffix.lower() in IMG_EXT and not any(s in p.name.lower() for s in skip)]


def gallery(out=None, size=(640, 400), picks=None):
    """Every 3D world: procedural + three skins -> test_frames/skins/*.png + contact_sheet.png."""
    import sys
    import fx_stack as fx
    from PIL import Image, ImageDraw
    here = Path(__file__).parent
    out = Path(out) if out else here / "test_frames" / "skins"
    out.mkdir(parents=True, exist_ok=True)
    names = _images(here / "textures")
    if not names:
        raise SystemExit("no images in textures/")
    picks = picks or [names[i % len(names)] for i in (6, 3, 5, 9, 4, 7, 8, 2, 6, 5, 3, 9)]
    ctx, _win = fx.hidden_context()
    lp = fx.LandPass(ctx, size)
    lp.settle_s = 0.0
    eng = SkinEngine(ctx, roots_for(here / "textures", here / "art"), size=512, sync=True)
    eng.fade_s = 0.0
    lp.skins = eng
    cells = []
    k = 0
    for mode, env in enumerate(LAND_ENVS):
        row = []
        for col in range(len(GALLERY) + 1):
            st = SkinState()
            label = f"{env}: procedural"
            if col:
                gl, sets = GALLERY[col - 1]
                for j, slot in enumerate(ENVS[env]["slots"]):
                    name = picks[(k + j) % len(picks)]
                    st.assign(env, slot, "ai", name)
                    st.do(validate(dict(cmd="skin", op="set", env=env, slot=slot, set=sets)))
                k += 1
                label = f"{env}: {gl}"
            for _ in range(3):
                eng.update(st.step(1 / 60), 1 / 60, dict(bpm=120.0))
                eng.lib.flush()
            img = fx.render_frames(lp, fx.default_values(land_a=mode, land_res=1.0), t0=6.0, frames=2, bands=(0.45, 0.3, 0.2),
                                   palette=fx.GALLERY_PALETTE)
            im = Image.fromarray(np.ascontiguousarray(img[:, :, :3]))
            im.save(out / f"{env}_{col}.png")
            ImageDraw.Draw(im).text((8, 6), label, fill=(255, 255, 255))
            row.append(im)
        cells.append(row)
    lp.release()
    eng.release()
    # the travel tunnel lives in the main scene shader: rendered by the real engine, headless (one short run per shot)
    import json
    import subprocess
    import tempfile
    tun = [("tunnel: procedural", None, dict(travel_mode=0)),
           ("tunnel: replace (log-polar)", dict(src=dict(kind="ai", name=picks[0]), mix=0), dict(travel_mode=0)),
           ("tunnel: screen (tube)", dict(src=dict(kind="ai", name=picks[1]), mix=2, scale=1.0), dict(travel_mode=1)),
           ("tunnel: lines reveal (Droste)", dict(src=dict(kind="ai", name=picks[4]), mix=4, bright=1.6), dict(travel_mode=3))]
    row = []
    with tempfile.TemporaryDirectory() as td:
        for i, (label, slot, sets) in enumerate(tun):
            cmd = [sys.executable, str(here / "live_visuals.py"), "--test", "40", "--out", str(Path(td) / str(i)), "--render",
                   f"{size[0]}x{size[1]}", "--set", "travel=0.6"] + [x for k, v in sets.items() for x in ("--set", f"{k}={v}")]
            if slot:
                (Path(td) / f"s{i}.json").write_text(json.dumps(dict(slots={"tunnel:walls": slot})))
                cmd += ["--skins", str(Path(td) / f"s{i}.json")]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(here))
            shots = sorted((Path(td) / str(i)).glob("frame_*.png"))
            if r.returncode or not shots:
                print("[skins] tunnel shot failed:", (r.stdout + r.stderr)[-300:])
                break
            im = Image.open(shots[-1]).convert("RGB").resize(size, Image.LANCZOS)
            im.save(out / f"tunnel_{i}.png")
            ImageDraw.Draw(im).text((8, 6), label, fill=(255, 255, 255))
            row.append(im)
    if len(row) == len(tun):
        cells.append(row)
    w, h = size[0] // 2, size[1] // 2
    sheet = Image.new("RGB", (w * len(cells[0]), h * len(cells)))
    for r, row in enumerate(cells):
        for c, im in enumerate(row):
            sheet.paste(im.resize((w, h), Image.LANCZOS), (c * w, r * h))
    sheet.save(out / "contact_sheet.png")
    print(f"[skins] gallery -> {out}")
    return out


if __name__ == "__main__":
    import sys
    if "--gallery" in sys.argv:
        gallery(next((a for a in sys.argv[2:] if not a.startswith("-")), None))
    else:
        print(__doc__)
