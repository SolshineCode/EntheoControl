"""Clips, macros, lanes, decks: the playable layer on top of the params registry (pure python, no GL, no I/O).

research/pro_redesign_plan.md section 2 + 3, Phase 1. The core trick:

* Every param belongs to exactly ONE LANE (`LANE_PARAMS`, enforced by tests/test_clips.py like the bypass table).
  Lanes: base (the world), figures (serpents, beings, swarm), text, interact (crowd camera), master (no clips:
  master + triggers, palette / hue / saturation, echo + trails, filters, image cycling).
* A CLIP belongs to one lane and writes only that lane's params into Controller.base, the same values faders write:
  params listed in the clip's `sets` get the clip's value, its macros set their targets, and EVERY other param of the
  lane goes back to its registry default. So a clip always looks the same, whatever was on before. Mods, bypass, the
  governor, scenes and autosave all keep working on top, unchanged.
* A clip has 4-8 MACROS: 0..1 knobs that each drive 1-4 targets of the clip's own lane through a curve
  (lin, exp, log, smooth, bell, inv, step:N, or a list of `values` for stepped params). Never another lane, never a
  master / trigger param (so no macro can strobe or black the wall out).

Clip JSON (`validate_clip` cleans it; unknown keys are dropped, anything wrong is a ValueError with a reason):
    {"id": "mandala", "name": "Folding Blocks Mandala", "lane": "base",
     "sets": {param: value},                        params of the lane fixed by the clip
     "mods": {param: [mod | null, mod | null]},      modulation of lane params (control.py mod format)
     "stack": null | stack JSON,                     BASE only: the math stack the clip brings (null = empty stack)
     "macros": [{"name", "widget", "default" 0..1, "axis"?, "targets": [{"p", "a", "b", "curve"} | {"p", "values": [..]}]}],
     "palette": null | 0..4,                         BASE only: applied unless that lane's PALETTE LOCK is on
     "bars": null | 0..64,                           default MORPH length in bars (null = the performer's morph setting)
     "tags": [..], "handles": [..]}                  handles: reserved for the direct-manipulation gizmos (Phase 2)

Deck JSON (a page of the grid; `validate_decks`): {"id", "name", "columns": [{"id", "name", "bars"?,
    "lanes": {lane: {"clip": id, "macros": {"<1-8>": v}} | "keep" | "stop"}}]}. Column ids are unique across decks
(`scene_go` addresses a column by id). Ids are short stable strings, never list indices, so a timeline or a MIDI
binding keeps working when a deck is reordered.
"""
import copy
import math
import re

from params import MOD_SOURCES, PARAMS, TRIGGERS

VERSION = 1
LANES = ("base", "figures", "text", "interact", "master")
CLIP_LANES = ("base", "figures", "text", "interact")
LANE_NAMES = {"base": "World", "figures": "Figures", "text": "Text", "interact": "Interact", "master": "Master"}

# Ownership by params.py group, then per-param exceptions. A NEW GROUP must be added here (the test fails otherwise).
LANE_BY_GROUP = {
    "master": "master", "space": "base", "motion": "base", "shape": "base", "color": "base", "layers": "base",
    "fractal": "base", "art": "base", "land": "base", "kene": "base", "beings": "figures", "text": "text",
    "camera": "interact", "filters": "master",
}
LANE_EXCEPT = {
    "palette": "master", "hue": "master", "saturation": "master", "trails": "master",     # colour narrative = master
    "l_echo": "master", "img_auto": "master", "img_hold": "master",
    "l_beings": "figures", "l_swarm": "figures", "swarm_vortex": "figures", "swarm_turb": "figures",
    "swarm_trail": "figures", "swarm_tint": "figures",
    "l_text": "text", "l_camera": "interact",
}
LANE_EXCEPT.update({n: "figures" for n in PARAMS if n.startswith("serpent")})          # serpent, serpent_mode, ...
LANE_EXCEPT = {k: v for k, v in LANE_EXCEPT.items() if k in PARAMS}


def lane_of(name):
    """The lane that owns param `name` (KeyError for an unknown param or an undecided group)."""
    if name in LANE_EXCEPT:
        return LANE_EXCEPT[name]
    return LANE_BY_GROUP[PARAMS[name][3]]


LANE_PARAMS = {lane: tuple(n for n in PARAMS if n in LANE_EXCEPT and LANE_EXCEPT[n] == lane
                           or n not in LANE_EXCEPT and LANE_BY_GROUP.get(PARAMS[n][3]) == lane) for lane in LANES}
STACK_LANE = "base"                                  # the math stack is part of the world
STEPPED = frozenset(n for n in PARAMS if n in set(
    __import__("params").OPTIONS) | {"travel_mode", "serpent_mode", "chase_mode", "fractal_mode", "land_a", "land_b",
                                     "land_blend", "land_cam", "land_style", "land_mirror", "art_use", "art_source",
                                     "serpent_count", "being_count", "art_index", "flt_slot1", "text_color"}
                    and n != "palette")
# "off" for a lane STOP: layer amounts to 0 (params.NEUTRAL), everything else to its default
_NEUTRAL = getattr(__import__("params"), "NEUTRAL", {})

CURVES = ("lin", "exp", "log", "smooth", "bell", "inv", "step")
WIDGETS = ("knob", "slider", "xy", "dir", "toggle", "bang", "select")
AXES = ("x", "y", "angle", "length")
ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_MACROS, MAX_TARGETS, MAX_NAME, MAX_TAGS = 8, 4, 32, 8
MAX_CLIPS, MAX_DECKS, MAX_COLUMNS = 256, 16, 32


# ------------------------------------------------------------------ curves


def check_curve(c):
    """'lin' | 'exp' | 'log' | 'smooth' | 'bell' | 'inv' | 'step:N' (N 2..32) -> the same string, else ValueError."""
    if isinstance(c, str):
        if c in CURVES and c != "step":
            return c
        m = re.match(r"^step:(\d{1,2})$", c)
        if m and 2 <= int(m.group(1)) <= 32:
            return c
    raise ValueError("curve must be lin, exp, log, smooth, bell, inv or step:N (N 2..32)")


def curve(c, x):
    """Macro position x (0..1) -> 0..1 through curve c."""
    x = min(1.0, max(0.0, float(x)))
    if c == "lin":
        return x
    if c == "exp":                                   # fine control at the bottom
        return x * x
    if c == "log":
        return math.sqrt(x)
    if c == "smooth":
        return x * x * (3.0 - 2.0 * x)
    if c == "bell":                                  # 0 at both ends, peak at the middle ("a little wobble, then calm")
        return math.sin(math.pi * x) ** 2
    if c == "inv":
        return 1.0 - x
    if c.startswith("step:"):
        n = int(c[5:])
        return round(x * (n - 1)) / (n - 1)
    raise ValueError("unknown curve")


def _num(v, what, lo=-1e9, hi=1e9):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"'{what}' must be a finite number")
    return min(hi, max(lo, float(v)))


def _name(v, what, n=MAX_NAME, required=True):
    if v is None and not required:
        return ""
    if not isinstance(v, str):
        raise ValueError(f"'{what}' must be text")
    v = "".join(ch for ch in v if ord(ch) >= 32).strip()[:n]
    if required and not v:
        raise ValueError(f"'{what}' must not be empty")
    return v


def check_id(v, what="id"):
    if not isinstance(v, str) or not ID.match(v):
        raise ValueError(f"'{what}' must be a short lowercase id (a-z 0-9 - _)")
    return v


def _clamp(p, v):
    lo, hi = PARAMS[p][0], PARAMS[p][1]
    v = min(hi, max(lo, float(v)))
    if p in STEPPED:
        v = float(min(hi, max(lo, round(v))))
    return v


# ------------------------------------------------------------------ macros


def macro_steps(m):
    """Number of detents of a macro (0 = continuous): from a step curve / values list, or 2 for toggle / bang."""
    for t in m["targets"]:
        if "values" in t:
            return len(t["values"])
        if t["curve"].startswith("step:"):
            return int(t["curve"][5:])
    return 2 if m["widget"] in ("toggle", "bang") else 0


def snap(m, x):
    x = min(1.0, max(0.0, float(x)))
    n = macro_steps(m)
    return round(x * (n - 1)) / (n - 1) if n >= 2 else x


def target_value(t, x):
    if "values" in t:
        vs = t["values"]
        return _clamp(t["p"], vs[int(round(min(1.0, max(0.0, x)) * (len(vs) - 1)))])
    return _clamp(t["p"], t["a"] + (t["b"] - t["a"]) * curve(t["curve"], x))


def macro_values(clip, i, x):
    """{param: value} macro i of `clip` writes at position x (snapped first)."""
    m = clip["macros"][i]
    x = snap(m, x)
    return {t["p"]: target_value(t, x) for t in m["targets"]}


def _clean_target(raw, lane):
    if not isinstance(raw, dict):
        raise ValueError("a macro target must be an object")
    p = raw.get("p")
    if not isinstance(p, str) or p not in PARAMS:
        raise ValueError(f"unknown macro target {str(p)[:32]!r} (Phase 1: params only)")
    if p in TRIGGERS or lane_of(p) != lane:
        raise ValueError(f"macro target {p} belongs to lane {lane_of(p)}, not {lane}")
    if raw.get("values") is not None:
        vs = raw["values"]
        if not isinstance(vs, list) or not 2 <= len(vs) <= 32:
            raise ValueError("'values' must be a list of 2..32 numbers")
        return {"p": p, "values": [_clamp(p, _num(v, "values")) for v in vs]}
    lo, hi = PARAMS[p][0], PARAMS[p][1]
    return {"p": p, "a": _clamp(p, _num(raw.get("a"), "a", lo, hi)), "b": _clamp(p, _num(raw.get("b"), "b", lo, hi)),
            "curve": check_curve(raw.get("curve", "lin"))}


def _clean_macro(raw, lane):
    if not isinstance(raw, dict):
        raise ValueError("a macro must be an object")
    w = raw.get("widget", "knob")
    if w not in WIDGETS:
        raise ValueError("widget must be one of " + ", ".join(WIDGETS))
    ts = raw.get("targets")
    if not isinstance(ts, list) or not 1 <= len(ts) <= MAX_TARGETS:
        raise ValueError(f"a macro needs 1..{MAX_TARGETS} targets")
    m = {"name": _name(raw.get("name"), "name", 16), "widget": w, "default": _num(raw.get("default", 0.0), "default", 0, 1),
         "targets": [_clean_target(t, lane) for t in ts]}
    if raw.get("axis") is not None:
        if raw["axis"] not in AXES:
            raise ValueError("axis must be one of " + ", ".join(AXES))
        m["axis"] = raw["axis"]
    steps = {len(t["values"]) if "values" in t else int(t["curve"][5:]) for t in m["targets"]
             if "values" in t or t["curve"].startswith("step:")}
    if len(steps) > 1:
        raise ValueError(f"macro {m['name']}: stepped targets disagree on the number of steps")
    if w == "select" and not steps:
        raise ValueError(f"macro {m['name']}: a select needs a step curve or a values list")
    m["default"] = snap(m, m["default"])
    if len({t["p"] for t in m["targets"]}) != len(m["targets"]):
        raise ValueError(f"macro {m['name']}: a param appears twice")
    return m


# ------------------------------------------------------------------ clips


def _clean_mods(raw, lane):
    out = {}
    if raw is None:
        return out
    if not isinstance(raw, dict):
        raise ValueError("'mods' must be an object")
    for n, pair in raw.items():
        if n not in PARAMS or n in TRIGGERS or lane_of(n) != lane:
            raise ValueError(f"mod on {str(n)[:32]} is outside lane {lane}")
        if not isinstance(pair, list):
            raise ValueError("a mod entry is a list of two slots")
        slots = []
        for m in (pair + [None, None])[:2]:
            if m is None or (isinstance(m, dict) and m.get("src") in (None, "none")):
                slots.append(None)
                continue
            if not isinstance(m, dict) or m.get("src") not in MOD_SOURCES:
                raise ValueError(f"mod on {n}: unknown source")
            slots.append({"src": m["src"], "amt": _num(m.get("amt", 0.0), "amt", -1, 1),
                          "rate": _num(m.get("rate", 1.0), "rate", 1 / 64, 64), "phase": _num(m.get("phase", 0.0), "phase", 0, 1)})
        if any(slots):
            out[n] = slots
    return out


def validate_clip(raw):
    """A clip from the panel / a file -> clean clip (bounded, typed, lane-checked). ValueError with a reason."""
    if not isinstance(raw, dict):
        raise ValueError("a clip must be an object")
    lane = raw.get("lane")
    if lane not in CLIP_LANES:
        raise ValueError("clip 'lane' must be one of " + ", ".join(CLIP_LANES))
    out = {"v": VERSION, "id": check_id(raw.get("id")), "name": _name(raw.get("name", raw.get("id")), "name"), "lane": lane}
    sets = raw.get("sets") or {}
    if not isinstance(sets, dict):
        raise ValueError("'sets' must be an object")
    out["sets"] = {}
    for p, v in sets.items():
        if p not in PARAMS or p in TRIGGERS or lane_of(p) != lane:
            raise ValueError(f"param {str(p)[:32]} is outside lane {lane}")
        out["sets"][p] = _clamp(p, _num(v, p))
    out["mods"] = _clean_mods(raw.get("mods"), lane)
    st = raw.get("stack")
    out["stack"] = None
    if st is not None:
        if lane != STACK_LANE:
            raise ValueError("only a base clip can carry a math stack")
        try:
            import stack as _stack
            out["stack"] = _stack.clean_stack(st)
        except ImportError:
            out["stack"] = None
    ms = raw.get("macros") or []
    if not isinstance(ms, list) or len(ms) > MAX_MACROS:
        raise ValueError(f"a clip has at most {MAX_MACROS} macros")
    out["macros"] = [_clean_macro(m, lane) for m in ms]
    owned = [t["p"] for m in out["macros"] for t in m["targets"]]
    if len(set(owned)) != len(owned):
        raise ValueError("two macros drive the same param")
    both = set(owned) & set(out["sets"])
    if both:
        raise ValueError(f"{sorted(both)[0]} is fixed in 'sets' and driven by a macro: pick one")
    pal = raw.get("palette")
    if pal is not None and lane != "base":
        raise ValueError("only a base clip can carry a palette")
    out["palette"] = None if pal is None else _clamp("palette", _num(pal, "palette")) if "palette" in PARAMS else None
    out["bars"] = None if raw.get("bars") is None else _num(raw["bars"], "bars", 0, 64)
    tags = raw.get("tags") or []
    out["tags"] = []
    for t in (tags if isinstance(tags, list) else [])[:MAX_TAGS]:
        if isinstance(t, str):
            t = "".join(ch for ch in t.lower() if ch.isalnum() or ch in " -_").strip()[:24]
            if t and t not in out["tags"]:
                out["tags"].append(t)
    hs = raw.get("handles") or []
    out["handles"] = [dict(h) for h in hs[:16] if isinstance(h, dict) and isinstance(h.get("type"), str)
                      and re.match(r"^[a-z_]{1,16}$", h["type"])] if isinstance(hs, list) else []
    out["handles"] = [{"type": h["type"]} for h in out["handles"]]     # Phase 2 defines the rest of a handle
    return out


def clean_clips(raw):
    """A list of user clips from a file: bad entries and duplicates are dropped (the rest are kept)."""
    out, seen = [], set(FACTORY_IDS)
    for x in (raw if isinstance(raw, list) else [])[:MAX_CLIPS]:
        try:
            c = validate_clip(x)
        except (ValueError, TypeError, KeyError):
            continue
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)
    return out


def resolve(clip, macros=None, stop=False):
    """What launching `clip` (with macro positions `macros`: list by index, missing = default) writes:
        {"updates": {param: value} for EVERY param of the clip's lane,
         "stack": stack JSON (base lane: the clip's or an empty one) | None (lane does not own the stack),
         "mods": {param: pair} for lane params (all other lane params lose their mods),
         "palette": value | None, "macros": [snapped positions]}
    stop=True: the lane goes dark instead (layer amounts to params.NEUTRAL, the rest to defaults)."""
    lane = clip["lane"]
    up = {n: PARAMS[n][2] for n in LANE_PARAMS[lane] if n not in TRIGGERS}
    if stop:
        for n in up:
            if n in _NEUTRAL and (n.startswith("l_") or n == "serpent"):
                up[n] = _NEUTRAL[n]
        return {"updates": up, "stack": {"v": 1, "layers": []} if lane == STACK_LANE else None, "mods": {},
                "palette": None, "macros": []}
    up.update(clip["sets"])
    pos = []
    for i, m in enumerate(clip["macros"]):
        x = macros[i] if macros is not None and i < len(macros) and macros[i] is not None else m["default"]
        x = snap(m, x)
        pos.append(x)
        for t in m["targets"]:
            up[t["p"]] = target_value(t, x)
    st = None
    if lane == STACK_LANE:
        st = copy.deepcopy(clip["stack"]) if clip.get("stack") else {"v": 1, "layers": []}
    return {"updates": up, "stack": st, "mods": copy.deepcopy(clip["mods"]), "palette": clip.get("palette"),
            "macros": pos}


# ------------------------------------------------------------------ decks


def validate_deck(raw):
    if not isinstance(raw, dict):
        raise ValueError("a deck must be an object")
    out = {"id": check_id(raw.get("id")), "name": _name(raw.get("name", raw.get("id")), "name"), "columns": []}
    cols = raw.get("columns") or []
    if not isinstance(cols, list) or len(cols) > MAX_COLUMNS:
        raise ValueError(f"a deck has at most {MAX_COLUMNS} columns")
    for col in cols:
        if not isinstance(col, dict):
            raise ValueError("a column must be an object")
        o = {"id": check_id(col.get("id"), "column id"), "name": _name(col.get("name", col.get("id")), "name"), "lanes": {}}
        if col.get("bars") is not None:
            o["bars"] = _num(col["bars"], "bars", 0, 64)
        lanes = col.get("lanes") or {}
        if not isinstance(lanes, dict):
            raise ValueError("'lanes' must be an object")
        for lane, v in lanes.items():
            if lane not in CLIP_LANES:
                raise ValueError(f"unknown lane {str(lane)[:16]}")
            if v in ("keep", "stop"):
                o["lanes"][lane] = v
                continue
            if not isinstance(v, dict):
                raise ValueError("a lane cell is 'keep', 'stop' or {clip, macros}")
            cell = {"clip": check_id(v.get("clip"), "clip")}
            mv = v.get("macros")
            if mv:
                if not isinstance(mv, dict):
                    raise ValueError("'macros' must be an object {\"1\".. \"8\": value}")
                cell["macros"] = {}
                for k, x in mv.items():
                    if str(k) not in [str(i) for i in range(1, MAX_MACROS + 1)]:
                        raise ValueError("macro numbers are 1..8")
                    cell["macros"][str(k)] = _num(x, "macro", 0, 1)
            o["lanes"][lane] = cell
        out["columns"].append(o)
    return out


def validate_decks(raw):
    if not isinstance(raw, list) or len(raw) > MAX_DECKS:
        raise ValueError(f"at most {MAX_DECKS} decks")
    out = [validate_deck(d) for d in raw]
    ids = [d["id"] for d in out]
    cols = [c["id"] for d in out for c in d["columns"]]
    if len(set(ids)) != len(ids) or len(set(cols)) != len(cols):
        raise ValueError("deck and column ids must be unique")
    return out


def clean_decks(raw):
    """Decks from a file: bad decks dropped, then duplicate column ids dropped (the first one wins)."""
    out, ids, cols = [], set(), set()
    for d in (raw if isinstance(raw, list) else [])[:MAX_DECKS]:
        try:
            d = validate_deck(d)
        except (ValueError, TypeError):
            continue
        if d["id"] in ids:
            continue
        d["columns"] = [c for c in d["columns"] if c["id"] not in cols]
        ids.add(d["id"])
        cols |= {c["id"] for c in d["columns"]}
        out.append(d)
    return out


def find_column(decks, cid):
    for d in decks:
        for c in d["columns"]:
            if c["id"] == cid:
                return c
    return None


# ------------------------------------------------------------------ factory clips (Phase 2 tunes ranges by eye)
# Rule for every factory clip: at its default macro positions it reproduces its declared look exactly, and the
# Mandala clip at defaults IS the registry default look (golden test).

def _t(p, a, b, curve="lin"):
    return {"p": p, "a": a, "b": b, "curve": curve}


def _m(name, widget, default, *targets, axis=None):
    m = {"name": name, "widget": widget, "default": default, "targets": list(targets)}
    if axis:
        m["axis"] = axis
    return m


_FOLDS = [4, 6, 8, 12, 16]
_FACTORY_RAW = [
    {"id": "mandala", "name": "Folding Blocks Mandala", "lane": "base", "tags": ["default", "mandala"],
     "sets": {"travel_mode": 0},
     "macros": [
         _m("Bloom", "knob", 0.5, _t("zoom", -0.5, 0.5, "smooth"), _t("density", 0.3, 0.7)),
         _m("Petals", "select", 0.5, {"p": "folds", "values": _FOLDS}),
         _m("Spin", "dir", 0.5, _t("spin", -0.5, 0.5), axis="angle"),
         _m("Melt", "knob", 0.0, _t("swim", 0.3, 0.9, "exp"), _t("twirl", 0.0, 0.35, "exp")),
         _m("Weave", "knob", 0.5, _t("l_beads", 0.4, 1.0), _t("l_lines", 0.2, 0.8)),
         _m("Light chase", "knob", 0.5, _t("chase", 0.0, 1.0), _t("chase_width", 0.45, 0.05)),
         _m("Breath", "knob", 0.5, _t("ripple", 0.0, 1.0), _t("bulge", -0.25, 0.25, "smooth")),
         _m("Fly in", "knob", 0.0, _t("travel", 0.0, 0.6, "exp")),
     ]},
    {"id": "crawl", "name": "Folding Blocks Crawl", "lane": "base", "tags": ["flat"], "palette": 3,
     "sets": {"symmetry": 0, "l_beads": 0.3, "l_lines": 0.2, "swim": 0.4},
     "macros": [
         _m("Direction", "dir", 0.125, _t("flow_dir", 0.0, 1.0), axis="angle"),
         _m("Speed", "dir", 0.6, _t("flow", 0.0, 1.0), axis="length"),
         _m("Scale", "knob", 0.5, _t("zoom", -0.6, 0.6)),
         _m("Wave", "knob", 0.0, _t("wave", 0.0, 0.7)),
         _m("Twirl", "knob", 0.5, _t("twirl", -0.6, 0.6)),
         _m("Density", "knob", 0.45 / 0.7, _t("density", 0.2, 0.9)),
         _m("Chase", "knob", 0.5, _t("chase", 0.0, 1.0)),
     ]},
    {"id": "tube", "name": "Tube Flight", "lane": "base", "tags": ["tunnel"], "palette": 1,
     "sets": {"travel_mode": 1, "zoom": 0.3, "rotate": 0.25, "twirl": 0.35, "flow": 0.7},
     "macros": [
         _m("Speed", "knob", math.sqrt(0.5 / 1.0), _t("travel", 0.0, 1.0, "exp")),
         _m("Bend", "knob", 0.4, _t("tunnel_curve", 0.0, 1.0)),
         _m("Distance", "knob", 0.5, _t("depth_fog", 0.8, 0.2)),
         _m("Corkscrew", "knob", 0.0, _t("spin", 0.0, 0.6), _t("spin_twist", 0.0, 0.8)),
         _m("Petals", "select", 0.75, {"p": "folds", "values": _FOLDS}),
         _m("Warp", "knob", 0.375, _t("bulge", -0.3, 0.5)),
     ]},
    {"id": "torus", "name": "Torus Wormhole", "lane": "base", "tags": ["3d"],
     "sets": {"l_land": 1, "land_a": 1, "l_kene": 0, "l_beads": 0, "l_lines": 0, "l_ai": 0},
     "macros": [
         _m("Speed", "knob", 1 / 3, _t("travel", 0.1, 1.0)),
         _m("Twist", "knob", 0.625, _t("land_twist", -1.0, 1.0)),
         _m("Curvature", "knob", 0.4, _t("land_curve", 0.0, 1.0)),
         _m("Geo morph", "knob", 0.3, _t("land_morph", 0.0, 1.0)),
         _m("Lens", "knob", 0.25 / 0.7, _t("land_fov", 0.2, 0.9)),
         _m("Camera", "select", 0.0, {"p": "land_cam", "values": [0, 1, 2]}),
         _m("Prism", "knob", 0.3125, _t("land_disp", 0.0, 0.8)),
     ]},
    {"id": "herd", "name": "Spirit Herd", "lane": "figures", "tags": ["beings"],
     "sets": {"l_beings": 1},
     "macros": [
         _m("Creature", "select", 0.0, {"p": "being", "values": list(range(17))}),
         _m("How many", "select", 2 / 7, {"p": "being_count", "values": list(range(1, 9))}),
         _m("Size", "knob", 0.5, _t("being_size", 0.0, 1.0)),
         _m("Wildness", "knob", 0.2 / 0.7, _t("being_speed", 0.2, 0.9), _t("being_wiggle", 0.3, 1.0)),
         _m("Glow", "knob", 0.5, _t("being_glow", 0.0, 1.0)),
         _m("Variety", "knob", 0.0, _t("being_mix", 0.0, 1.0)),
         _m("Colour", "select", 0.0, {"p": "being_color", "values": list(range(6))}),
     ]},
]
FACTORY = [validate_clip(c) for c in _FACTORY_RAW]
FACTORY_IDS = tuple(c["id"] for c in FACTORY)
_BY_ID = {c["id"]: c for c in FACTORY}


def get(cid, user=None):
    """A clip by id: factory first (factory ids are reserved), then the user's clips {id: clip}."""
    return _BY_ID.get(cid) or (user or {}).get(cid)
