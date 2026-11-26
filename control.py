"""Performance control system: faders, modulation matrix, beat clock, triggers, scenes, autopilot, MIDI.

Implements the API in params.StaticController (the render engine calls it once per frame):

    ctl = Controller(state_dir=Path("."), restore=True)
    frame = ctl.update(dt, bands)        # -> params.Frame(values, events, clock)
    rest  = ctl.apply(remote.drain())    # panel commands; returns the ones the render engine owns
    remote.publish_state(dict(..., ctl=ctl.state()))

Everything runs on the caller's (GL) thread except MIDI input, which arrives through a queue (midi.py).
Files in state_dir: scenes.json (16 scene slots), midi_map.json (bindings), live_state.json (autosave,
restored on start so a crash restart resumes the same look). All writes are atomic and never raise.
"""
import copy
import json
import math
import queue
import random
import re
import threading
import time
import weakref
from collections import deque
from pathlib import Path

from ai_link import AIClient
from midi import MidiManager, atomic_write_json, quarantine
from params import MOD_SOURCES, PARAMS, TRIGGERS, Frame, clamp, defaults

N_SCENES = 16
CONT = [n for n in PARAMS if n not in TRIGGERS]          # fader params (everything that is not a trigger)
QUANT = {"off": 0, "beat": 1, "bar": 4, "phrase": 32}    # recall quantize -> beats
CLOCK_SOURCES = ("internal", "midi", "link", "auto")     # "tap" is accepted as an alias of "internal"
AUTO_MODES = ("random", "next")
SLEW_TAU = 0.06          # s, fader smoothing
ENV_TAU = 0.12           # s, flash / punch release (about 300 ms to fade out)
AUTOSAVE_S = 5.0
BPM_MIN, BPM_MAX = 20.0, 300.0
# Selectable (rounded) params: the panel shows them as tiles. Labels are optional.
OPTIONS = {
    "travel_mode": ["Zoom tunnel", "Tube", "Depth layers", "Droste"],
    "serpent_mode": ["Serpents", "Serpentine warp", "Both"],
    "chase_mode": ["Single", "Dual", "Counter"],
    "fractal_mode": ["Crystal KIFS", "Mandelbox tunnel", "Hyperbolic tiling", "Apollonian", "Julia jewels", "4D polytope"],
    "serpent_count": [],
    "palette": ["Aya Jewel", "Cosmic Gold/Violet", "Gorge Twilight", "Emerald Jungle", "Spectral"],
    # TouchVDMX merge (fx_stack.LandPass)
    "land_a": ["Sacred corridor", "Torus wormhole", "Prismatic vortex", "Hex dome"],
    "land_b": ["Sacred corridor", "Torus wormhole", "Prismatic vortex", "Hex dome"],
    "land_blend": ["Mix", "Add", "Screen", "Difference", "Multiply", "Exclusion", "Divide", "Luma mask"],
    "land_cam": ["Cruise", "Orbit", "Audio dive"],
    "land_style": ["Electric chapel", "Floral glass", "Neon gears"],
    "land_mirror": ["Off", "On"],
    "art_use": ["Fly-through relief", "Inside the lines", "Kaleidoscope", "Tunnel walls", "Displace pattern", "Full frame melt"],
    "art_source": ["Art piece", "AI texture"],
}


def merge_options(base, extra, known):
    """Option labels for stepped params: control's own table plus params.OPTIONS (if a module declares names next
    to its params). Later wins; unknown params and anything that is not a list of labels are dropped."""
    out = {}
    for src in (base, extra if isinstance(extra, dict) else {}):
        for k, v in src.items():
            if k in known and isinstance(v, (list, tuple)):
                out[k] = [str(x)[:40] for x in v]
    return out


import params as _params  # noqa: E402  (params.OPTIONS is optional)
try:                                                 # text layer: the words / phrase list state and its commands
    import text_layer as _textlayer                  # (the pure-python part of text_layer.py; no GL is imported)
except ImportError:
    _textlayer = None
try:                                                 # math stack: the layer model + its commands (pure python part)
    import stack as _stack
except ImportError:
    _stack = None
try:                                                 # environment skins: the slot model + its commands (pure python part)
    import materials as _skins
except ImportError:
    _skins = None
try:                                                 # timeline: moving playhead, cues, automation (timeline.py)
    import timeline as _tl
except ImportError:
    _tl = None
try:                                                 # shows / presets: save and load later (shows.py, worker thread)
    import shows as _shows
except ImportError:
    _shows = None
try:                                                 # ONLINE plugin hooks (pure python; the plugin itself is only
    from plugins.online import protocol as _online   # imported when the performer switches ONLINE on)
except Exception:
    _online = None
try:                                                 # clips / macros / lanes / decks (clips.py, pure python)
    import clips as _clips
except ImportError:
    _clips = None
try:                                                 # [themes] THEMES + SOUND TAGS (themes.py / soundtags.py)
    import themes as _themes
except ImportError:
    _themes = None
try:                                                 # [transitions] visual scene transitions (doors, iris, portal...)
    import transitions as _trans
except ImportError:
    _trans = None
HISTORY_MAX = 50             # engine-side undo steps (Controller.history)
HIST_GAP = 0.6               # s: moves of one control without a gesture id closer than this are one undo step
HIST_G_GAP = 5.0             # s: a gesture id older than this starts a new step (a reloaded panel reuses ids)
OPTIONS = merge_options(OPTIONS, getattr(_params, "OPTIONS", None), PARAMS)
# stepped params: no slew, snap at the morph midpoint (palette is fractional by contract, so it glides)
DISCRETE = [n for n in OPTIONS if n != "palette"] + [n for n in ("art_index",) if n in PARAMS]
_DISCRETE_SET = frozenset(DISCRETE)
MAX_PROMPT = 500
DRIFT_DEFAULT = ["zoom", "pan_x", "pan_y", "rotate", "twirl", "swim", "hue", "flow_dir", "tunnel_curve"]

# Cue channel CC table (FROZEN: a pre-arranged Ableton set depends on these numbers; append only).
CUE_CC = {
    1: "master", 2: "zoom", 3: "pan_x", 4: "pan_y", 5: "rotate", 6: "symmetry", 7: "folds", 8: "speed",
    9: "flow", 10: "flow_dir", 11: "swim", 12: "ripple", 13: "ripple_decay", 14: "twirl", 15: "bulge",
    16: "wave", 17: "density", 18: "palette", 19: "hue", 20: "saturation", 21: "chase", 22: "trails",
    23: "l_kene", 24: "l_beads", 25: "l_lines", 26: "l_ai", 27: "l_echo", 28: "l_video", 29: "luma_key",
    30: "chase_width", 31: "chase_rate", 32: "chase_mode", 33: "travel", 34: "travel_mode", 35: "tunnel_curve",
    36: "depth_fog", 37: "serpent", 38: "serpent_mode", 39: "serpent_count", 40: "serpent_segments",
    41: "serpent_speed", 42: "serpent_wiggle", 43: "serpent_size", 44: "l_fractal", 45: "l_art",
    46: "fractal_mode", 47: "fractal_morph", 48: "fractal_detail", 49: "fractal_glow", 50: "fractal_metal",
    51: "fractal_res", 52: "art_index", 53: "art_relief", 54: "cam_yaw", 55: "cam_pitch", 56: "cam_roll",
    57: "cam_dolly", 58: "cam_drift", 59: "art_melt", 60: "art_liquify", 61: "art_slices", 62: "art_breathe",
    63: "art_fold", 64: "art_droste", 65: "art_smear", 66: "art_glow",
}
CUE_CC = {k: v for k, v in CUE_CC.items() if v in PARAMS}
CUE_NOTE_BASES = (0, 36)     # notes 0-15 and 36-51 (C1.., a drum rack) both recall scenes 1-16
CUE_TRIGGER_NOTE = 60        # notes 60-65 hold flash / strobe / invert / freeze / blackout / punch
CUE_PANIC_NOTE = 127

# ------------------------------------------------------------------ validation


def _num(c, key, lo, hi, default=None):
    v = c.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"'{key}' must be a finite number")
    return min(hi, max(lo, float(v)))


def _int(c, key, lo, hi):
    v = c.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ValueError(f"'{key}' must be an integer in {lo}..{hi}")
    return v


def _bool(c, key):
    v = c.get(key)
    if not isinstance(v, bool):
        raise ValueError(f"'{key}' must be true or false")
    return v


def _name(c, allowed, key="name"):
    v = c.get(key)
    if not isinstance(v, str) or v not in allowed:
        raise ValueError(f"unknown {key}")
    return v


def valid_target(t):
    """MIDI-learn / binding targets."""
    if not isinstance(t, str) or len(t) > 48:
        return False
    if t in ("tap", "autopilot", "drift", "resync", "shift", "panic", "bpm", "next_video", "next", "art_prev",
             "art_next", "xfade", "rec"):
        return True
    p = t.split(":")
    try:
        if p[0] == "param" and len(p) == 2:
            return p[1] in PARAMS
        if p[0] == "trigger" and len(p) == 2:
            return p[1] in TRIGGERS
        if p[0] == "scene" and len(p) == 2:
            return 0 <= int(p[1]) < N_SCENES
        if p[0] == "palette" and len(p) == 2:
            return PARAMS["palette"][0] <= float(p[1]) <= PARAMS["palette"][1]
        if p[0] == "morph" and len(p) == 2:
            return 0 <= float(p[1]) <= 64
        if p[0] == "quantize" and len(p) == 2:
            return p[1] in QUANT
        if p[0] == "bypass" and len(p) == 2:
            return p[1] in BYPASS_COVER
        if p[0] == "stack" and len(p) == 3:          # stack:<layer id>:<field> (a knob on a field, a pad on mute / solo)
            return _stack is not None and _stack.parse_target(t) is not None
        if p[0] == "skin" and len(p) == 4:           # skin:<env>:<slot>:<field> (a knob on a skin slot's field)
            return _skins is not None and _skins.parse_target(t) is not None
        if p[0] == "ripple" and len(p) == 3:
            return 0 <= float(p[1]) <= 1 and 0 <= float(p[2]) <= 1
        if p[0] == "macro" and len(p) == 3:          # macro:<lane>:<1-8>: plays whatever clip is on air in that lane
            return _clips is not None and p[1] in _clips.CLIP_LANES and p[2].isdigit() and 1 <= int(p[2]) <= 8
        if p[0] == "clip" and len(p) == 2:           # clip:<id>: a pad launches it (quantized)
            return _clips is not None and bool(_clips.ID.match(p[1]))
        if p[0] == "theme" and len(p) == 2:          # [themes] theme:<slot>: a pad restores a tray theme
            return _themes is not None and p[1].isdigit() and 1 <= int(p[1]) <= _themes.MAX_TRAY
        if p[0] == "transition" and len(p) == 2:     # [transitions] transition:<kind|morph|len|cancel>: a pad picks
            return _trans is not None and (p[1] in _trans.KINDS or p[1] in ("morph", "len", "cancel"))   # the default
    except ValueError:
        pass
    return False


def _gesture(c, out):
    """Optional gesture id `g` (int or short text): every command of one drag carries the same g = one undo step."""
    g = c.get("g")
    if g is not None:
        if isinstance(g, bool) or not ((isinstance(g, int) and 0 <= g < 2 ** 31) or (
                isinstance(g, str) and 0 < len(g) <= 24 and g.isprintable())):
            raise ValueError("'g' must be a gesture id (a whole number or short text)")
        out["g"] = g
    return out


def _v_set(c):
    name = _name(c, PARAMS)
    return _gesture(c, {"cmd": "set", "name": name, "v": _num(c, "v", PARAMS[name][0], PARAMS[name][1])})


def _v_inc(c):
    name = _name(c, PARAMS)
    span = PARAMS[name][1] - PARAMS[name][0]
    return _gesture(c, {"cmd": "inc", "name": name, "d": _num(c, "d", -span, span)})


def _macro_arg(c):
    """Launch-time macro positions: a list (index 0 = macro 1) or {"1".."8": v} -> {"1".."8": v}."""
    m = c.get("macros")
    if m is None:
        return None
    if isinstance(m, list) and len(m) <= 8:
        m = {str(i + 1): v for i, v in enumerate(m) if v is not None}
    if not isinstance(m, dict) or any(str(k) not in [str(i) for i in range(1, 9)] for k in m):
        raise ValueError("'macros' must be a list of up to 8 values or {\"1\".. \"8\": value}")
    return {str(k): _num(m, k, 0.0, 1.0) for k in m}


def _v_clip(c):
    """{"cmd": "clip", "op": launch {id, macros?, bars?, now?, lane?} | stop {lane, bars?, now?} | save {clip} |
    delete {id} | lock {lane, palette}}."""
    op = _name(c, ("launch", "stop", "save", "delete", "lock"), "op")
    out = {"cmd": "clip", "op": op}
    if op in ("launch", "delete"):
        out["id"] = _clips.check_id(c.get("id"))
    if op in ("stop", "lock") or (op == "launch" and c.get("lane") is not None):
        out["lane"] = _name(c, _clips.CLIP_LANES, "lane")
    if op in ("launch", "stop"):
        if c.get("bars") is not None:
            out["bars"] = _num(c, "bars", 0.0, 64.0)
        if "now" in c:
            out["now"] = _bool(c, "now")
    if op == "launch":
        m = _macro_arg(c)
        if m:
            out["macros"] = m
    if op == "save":
        out["clip"] = _clips.validate_clip(c.get("clip"))
        if out["clip"]["id"] in _clips.FACTORY_IDS:
            raise ValueError("that id belongs to a factory clip: save under a new id")
    if op == "delete" and out["id"] in _clips.FACTORY_IDS:
        raise ValueError("factory clips cannot be deleted")
    if op == "lock":
        out["palette"] = _bool(c, "palette")
    return out


def _v_deck(c):
    op = _name(c, ("save", "delete"), "op")
    if op == "save":
        return {"cmd": "deck", "op": op, "deck": _clips.validate_deck(c.get("deck"))}
    return {"cmd": "deck", "op": op, "id": _clips.check_id(c.get("id"))}


def _v_macro(c):
    out = {"cmd": "macro", "lane": _name(c, _clips.CLIP_LANES, "lane"), "n": _int(c, "n", 1, 8),
           "v": _num(c, "v", 0.0, 1.0)}
    return _gesture(c, out)


def _v_scene_go(c):
    out = {"cmd": "scene_go", "id": _clips.check_id(c.get("id"))}
    if "now" in c:
        out["now"] = _bool(c, "now")
    if c.get("bars") is not None:
        out["bars"] = _num(c, "bars", 0.0, 64.0)
    return out


def _v_mod(c):
    name = c.get("name")
    if isinstance(name, str) and name.startswith("stack:"):      # a math-stack field: stack:<layer id>:<field>
        t = _stack.parse_target(name) if _stack is not None else None
        if t is None or t[1] not in _stack.NUM:
            raise ValueError("unknown stack field")
        return _stack.validate(dict(c, op="mod", id=t[0], f=t[1]))
    if isinstance(name, str) and name.startswith("skin:"):       # a skin slot's field: skin:<env>:<slot>:<field>
        t = _skins.parse_target(name) if _skins is not None else None
        if t is None or t[2] not in _skins.NUM:
            raise ValueError("unknown skin field")
        return _skins.validate(dict(cmd="skin", op="mod", env=t[0], slot=t[1], f=t[2], mslot=_int(c, "slot", 0, 1),
                                    src=_name(c, MOD_SOURCES, "src"), amt=c.get("amt", 0.0), rate=c.get("rate", 1.0),
                                    phase=c.get("phase", 0.0)))
    return {"cmd": "mod", "name": _name(c, CONT), "slot": _int(c, "slot", 0, 1),
            "src": _name(c, MOD_SOURCES, "src"), "amt": _num(c, "amt", -1.0, 1.0, 0.0),
            "rate": _num(c, "rate", 1 / 64, 64.0, 1.0), "phase": _num(c, "phase", 0.0, 1.0, 0.0)}


def _v_autopilot(c):
    out = {"cmd": "autopilot"}
    for k in ("on", "drift"):
        if k in c:
            out[k] = _bool(c, k)
    if "bars" in c:
        out["bars"] = _num(c, "bars", 1.0, 128.0)
    if "mode" in c:
        out["mode"] = _name(c, AUTO_MODES, "mode")
    if "drift_amt" in c:
        out["drift_amt"] = _num(c, "drift_amt", 0.0, 1.0)
    if "pal_hold" in c:
        out["pal_hold"] = _num(c, "pal_hold", 0.0, 64.0)
    if "drift_params" in c:
        v = c["drift_params"]
        if not isinstance(v, list) or len(v) > len(CONT) or any(not isinstance(n, str) or n not in CONT for n in v):
            raise ValueError("'drift_params' must be a list of param names")
        out["drift_params"] = list(dict.fromkeys(v))
    return out


def _v_learn(c):
    t = c.get("target")
    if t is not None and not valid_target(t):
        raise ValueError("unknown learn target")
    return {"cmd": "midi_learn", "target": t}


def _v_clear(c):
    t = c.get("target")
    if t is not None and not valid_target(t):
        raise ValueError("unknown target")
    return {"cmd": "midi_clear", "target": t}


def _v_clock_src(c):
    src = _name(c, CLOCK_SOURCES + ("tap",), "src")
    return {"cmd": "clock_src", "src": "internal" if src == "tap" else src}


def _port(c):
    v = c.get("port")
    if v is not None and (not isinstance(v, str) or not 0 < len(v) <= 200):
        raise ValueError("'port' must be a port name or null")
    return v


def _v_recall(c):
    out = {"cmd": "scene_recall", "i": _int(c, "i", 0, N_SCENES - 1)}
    if "now" in c:
        out["now"] = _bool(c, "now")
    _v_trans_field(c, out)
    return out


def _v_trans_field(c, out):
    """[transitions] optional "transition" on a recall: a kind name / spec object, or "morph" (= the plain glide)."""
    if _trans is not None and c.get("transition") is not None:
        t = c["transition"]
        out["transition"] = "morph" if t == "morph" else (_trans.clean_spec(t) or "morph")
    return out


def _v_transition(c):
    """[transitions] {"cmd": "transition", "op": default | scene | origin | cancel | finish, ...}"""
    op = _name(c, ("default", "scene", "origin", "cancel", "finish"), "op")
    out = {"cmd": "transition", "op": op}
    if op in ("default", "scene"):
        out["spec"] = _trans.clean_spec(c.get("spec"))
    if op == "scene":
        out["i"] = _int(c, "i", 0, N_SCENES - 1)
    if op == "origin":
        out["x"], out["y"] = _num(c, "x", 0.0, 1.0), _num(c, "y", 0.0, 1.0)
    return out


def _text(c, key, required):
    v = c.get(key)
    if v is None and not required:
        return ""
    if not isinstance(v, str) or len(v) > MAX_PROMPT or (required and not v.strip()):
        raise ValueError(f"'{key}' must be text of 1..{MAX_PROMPT} characters")
    return v.strip()


def _v_gen(c):
    """AI image job for ai_worker.py (prompt <= 500 chars, n 1..8, strength clamped to 0.2..0.9)."""
    out = {"cmd": "gen", "prompt": _text(c, "prompt", True), "n": _int(c, "n", 1, 8) if "n" in c else 1,
           "init": None, "show": _bool(c, "show") if "show" in c else True}
    neg = _text(c, "negative", False)
    if neg:
        out["negative"] = neg
    if c.get("seed") is not None:
        out["seed"] = _int(c, "seed", 0, 2 ** 31 - 1)
    if c.get("init") is not None:
        ini = c["init"]
        if isinstance(ini, dict):                      # dream from one library image: {kind: art | ai, name}
            nm = ini.get("name")
            if (ini.get("kind") not in ("art", "ai") or not isinstance(nm, str) or not _TEX_NAME.match(nm)
                    or ".." in nm or nm.lower().endswith(".depth.png")):
                raise ValueError("'init' image must be {kind: art | ai, name: an image file name}")
            out["init"] = {"kind": ini["kind"], "name": nm}
        elif ini != "snapshot":
            raise ValueError("'init' must be null, \"snapshot\" or {kind, name}")
        else:
            out["init"] = "snapshot"
    if "strength" in c or out["init"]:
        out["strength"] = _num(c, "strength", 0.2, 0.9, 0.45)
    return out


SEQ_STEPS = 16
SEQ_DEFAULT = dict(vals=[0.85, 0.2, 0.5, 0.3, 0.9, 0.4, 0.65, 0.25, 0.8, 0.35, 0.55, 0.45, 0.95, 0.3, 0.7, 0.5],
                   gates=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])


def _clean_seq(raw):
    """Step sequencer pattern from untrusted JSON; anything malformed falls back to the default per field."""
    out = dict(vals=list(SEQ_DEFAULT["vals"]), gates=list(SEQ_DEFAULT["gates"]))
    if isinstance(raw, dict):
        v, g = raw.get("vals"), raw.get("gates")
        if isinstance(v, list) and len(v) == SEQ_STEPS and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v):
            out["vals"] = [min(1.0, max(0.0, float(x))) for x in v]
        if isinstance(g, list) and len(g) == SEQ_STEPS and all(x in (0, 1, True, False) for x in g):
            out["gates"] = [int(bool(x)) for x in g]
    return out


def _v_seq(c):
    out = {"cmd": "seq", "i": _int(c, "i", 0, SEQ_STEPS - 1)}
    if "v" in c:
        out["v"] = _num(c, "v", 0.0, 1.0)
    if "g" in c:
        out["g"] = _bool(c, "g")
    if len(out) == 2:
        raise ValueError("'seq' needs 'v' (step value) or 'g' (gate)")
    return out


def _v_xfade(c):
    """Scene A/B crossfader: a / b = deck slots (-1 = empty), x = fader 0 (A) .. 1 (B). Any subset, at least one."""
    out = {"cmd": "xfade"}
    for k in ("a", "b"):
        if k in c:
            out[k] = _int(c, k, -1, N_SCENES - 1)
    if "x" in c:
        out["x"] = _num(c, "x", 0.0, 1.0)
    if len(out) == 1:
        raise ValueError("'xfade' needs 'a', 'b' or 'x'")
    return out


# BYPASS: a layer / tab switched off and on again to exactly the same settings (tables: params.NEUTRAL etc.)
BYPASS_COVER = _params.bypass_cover()
BYPASS_FADE, BYPASS_FADE_MAX = 0.3, 5.0          # s; 0 = instant


# NEW CANVAS: everything the engine draws by itself is bypassed (every layer but the math stack, every effect);
# nothing stored is edited, so RESTORE PREVIOUS brings the look back exactly.
CANVAS_KEYS = tuple(k for k in ["fx"] + [n for n in _params.NEUTRAL if n.startswith("l_") and n != "l_stack"]
                    if k in BYPASS_COVER)
CANVAS_FADE = 1.0                                # s: the crossfade into / out of a clean canvas


def _v_bypass(c):
    """{"cmd": "bypass", "key": layer / tab / "fx", "on": bool}; without "on" it toggles."""
    out = {"cmd": "bypass", "key": _name(c, BYPASS_COVER, "key")}
    if "on" in c and c["on"] is not None:
        out["on"] = _bool(c, "on")
    return out


def passthrough(kind, **extra):
    """A command for the render engine. Carries both "cmd" and "type" so either key can be matched."""
    return dict(extra, cmd=kind, type=kind)


_VALIDATORS = {
    "gen": _v_gen,
    "gen_cancel": lambda c: {"cmd": "gen_cancel"},
    "art_step": lambda c: {"cmd": "art_step", "d": _int(c, "d", -1, 1)},
    "art_deleted": lambda c: {"cmd": "art_deleted", "i": _int(c, "i", 0, 4095)},
    "clock_port": lambda c: {"cmd": "clock_port", "port": _port(c)},
    "set": _v_set,
    "inc": _v_inc,
    "mod": _v_mod,
    "trigger": lambda c: {"cmd": "trigger", "name": _name(c, TRIGGERS), "on": _bool(c, "on")},
    "latch": lambda c: {"cmd": "latch", "name": _name(c, TRIGGERS), "on": _bool(c, "on")},
    "ripple": lambda c: {"cmd": "ripple", "x": _num(c, "x", 0.0, 1.0), "y": _num(c, "y", 0.0, 1.0)},
    "tap": lambda c: {"cmd": "tap"},
    "bpm": lambda c: {"cmd": "bpm", "v": _num(c, "v", BPM_MIN, BPM_MAX)},
    "bpm_nudge": lambda c: {"cmd": "bpm_nudge", "d": _num(c, "d", -50.0, 50.0)},
    "bpm_mult": lambda c: {"cmd": "bpm_mult", "f": _num(c, "f", 0.25, 4.0)},
    "resync": lambda c: {"cmd": "resync"},
    "clock_src": _v_clock_src,
    "strobe_rate": lambda c: {"cmd": "strobe_rate", "v": _num(c, "v", 1 / 16, 4.0)},
    "scene_save": lambda c: {"cmd": "scene_save", "i": _int(c, "i", 0, N_SCENES - 1)},
    "scene_recall": _v_recall,
    "scene_clear": lambda c: {"cmd": "scene_clear", "i": _int(c, "i", 0, N_SCENES - 1)},
    "morph": lambda c: {"cmd": "morph", "bars": _num(c, "bars", 0.0, 64.0)},
    "quantize": lambda c: {"cmd": "quantize", "q": _name(c, QUANT, "q")},
    "glide": lambda c: {"cmd": "glide", "bars": _num(c, "bars", 0.0, 64.0)},
    "autopilot": _v_autopilot,
    "safe": lambda c: {"cmd": "safe", "i": _int(c, "i", 0, N_SCENES - 1)},
    "panic": lambda c: {"cmd": "panic"},
    "init": lambda c: {"cmd": "init"},
    "cue_ch": lambda c: {"cmd": "cue_ch", "ch": _int(c, "ch", 0, 16)},
    "midi_learn": _v_learn,
    "midi_clear": _v_clear,
    "midi_defaults": lambda c: {"cmd": "midi_defaults"},
    "seq": _v_seq,
    "seq_rand": lambda c: {"cmd": "seq_rand"},
    "xfade": _v_xfade,
    "bypass": lambda c: _v_bypass(c),
    "bypass_fade": lambda c: {"cmd": "bypass_fade", "v": _num(c, "v", 0.0, BYPASS_FADE_MAX)},
}
if _stack is not None:                               # {"cmd": "stack", "op": ...}: stack.py; NEW CANVAS / RESTORE PREVIOUS
    _VALIDATORS["stack"] = _stack.validate
    _VALIDATORS["canvas"] = lambda c: {"cmd": "canvas", "op": _name(c, ("new", "restore"), "op")}
if _skins is not None:                               # {"cmd": "skin", "op": ...}: materials.py (environment skins)
    _VALIDATORS["skin"] = _skins.validate
if _textlayer is not None:                           # text, text_phrases, text_add, text_del, text_go, text_cycle
    _VALIDATORS.update(_textlayer.VALIDATORS)
if _clips is not None:                               # clips.py: clip launch / macros / decks / scene columns
    _VALIDATORS.update(clip=_v_clip, deck=_v_deck, macro=_v_macro, scene_go=_v_scene_go)
_VALIDATORS["history"] = lambda c: {"cmd": "history", "op": _name(c, ("undo", "redo"), "op")}   # engine-side undo
if _themes is not None:                              # [themes] {"cmd": "theme", "op": ...}: themes.validate
    _VALIDATORS["theme"] = _themes.validate
if _trans is not None:                               # [transitions] default / per-scene / origin / cancel / finish
    _VALIDATORS["transition"] = _v_transition
if _online is not None:                              # {"cmd": "online", "op": ...}: plugins/online/protocol.py
    _VALIDATORS["online"] = _online.validate
# ---- TIMELINE (timeline.py) + SHOWS / PRESETS (shows.py)
TL_EVENTS = ("bypass", "text", "text_go", "art_step", "ripple", "seq_rand", "next_video", "show_texture", "art_pick",
             "clip", "scene_go", "macro", "theme")     # clip: launch / stop only (by id, never a list index);
# theme: restore only (a tray theme by id; starred themes are also preset cues "theme/<slug>")
_TEX_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(png|jpg|jpeg|webp)$", re.I)
TL_FLASH_GAP = 0.4          # s between two timeline flash-type triggers (<= 2.5 Hz)
TL_SAFE_HZ = 2.5            # timeline strobe only when the strobe rate is at or under this
TL_LUMA_RATE = 5.0          # luminance-critical automation: at most 5 full ranges / s (a full swing >= 0.2 s)
TL_GLIDE = 1.0              # s: hand-back glide after a live override, and after RELEASE
TL_HOLD = 1.5               # s: TOUCH override holds this long after the performer's last move (op "hold")
TL_CHASE_S = 0.15           # s: crossfade of a chased cue (a seek never cuts hard)
SHOW_DEFAULT_PARTS = ("look", "scenes", "timeline", "settings", "recorder")   # venue rig (mapping, MIDI) kept by default
SHOW_DEBOUNCE, SHOW_MAX_WAIT = 15.0, 60.0                                       # s: autosave after the last edit / at most


def _v_tl_event(c):
    """A command a timeline EVENT item may carry: allowlisted, validated like the panel's own. Webcam on / off is
    deliberately not here (a shared show must never switch a camera on); the CAMERA layer's bypass is."""
    if not isinstance(c, dict) or c.get("cmd") not in TL_EVENTS:
        raise ValueError("timeline event must be one of " + ", ".join(TL_EVENTS))
    k = c["cmd"]
    if k in _VALIDATORS:
        out = _VALIDATORS[k](c)
        if k == "bypass" and "on" not in out:
            raise ValueError("a timeline bypass event needs 'on' (true = switch off)")
        if k == "clip" and out["op"] not in ("launch", "stop"):
            raise ValueError("a timeline clip event can only launch or stop")
        if k == "theme" and out["op"] != "restore":
            raise ValueError("a timeline theme event can only restore")
        if k == "clip" and out["op"] == "launch" and "lane" not in out and _clips.get(out["id"]):
            out["lane"] = _clips.get(out["id"])["lane"]          # chase keeps the latest launch per lane
        return out
    if k == "next_video":
        return passthrough(k)
    if k == "show_texture":
        name = c.get("name")
        bare = name[4:] if isinstance(name, str) and name.startswith("art/") else name   # art/<name>: an art upload
        if not isinstance(name, str) or not isinstance(bare, str) or not _TEX_NAME.match(bare) or ".." in name:
            raise ValueError("'name' must be a texture file name")
        return passthrough(k, name=name)
    src, name = c.get("source", "art"), c.get("name")
    if src not in ("art", "ai", "any") or not isinstance(name, str) or not 0 < len(name) <= 200 or name in (".", "..") or any(
            ch in name for ch in '/\\:\0') or any(ord(ch) < 32 for ch in name) or (src == "ai" and not _TEX_NAME.match(name)):
        raise ValueError("'art_pick' needs source art | ai | any and a file name")
    return passthrough(k, source=src, name=name)


def _v_show(c):
    op = _name(c, ("new", "open", "save", "save_as", "delete", "rescan"), "op")
    out = {"cmd": "show", "op": op}
    if op in ("open", "delete") or (op == "new" and c.get("from") == "template"):
        out["slug"] = _shows.check_slug(c.get("slug"))
    if op == "new":
        out["from"] = _name(c, ("clean", "factory", "current", "template"), "from")
    if op == "open":
        out["recover"] = _bool(c, "recover") if "recover" in c else False
        b = c.get("backup")
        if b is not None:
            if not isinstance(b, str) or not re.match(r"^show\.\d{8}-\d{6}\.json$", b):
                raise ValueError("bad backup name")
            out["backup"] = b
    if op in ("open", "new") and c.get("parts") is not None:
        p = c["parts"]
        if not isinstance(p, list) or any(x not in _shows.PARTS for x in p):
            raise ValueError("'parts' must be a list of " + ", ".join(_shows.PARTS))
        out["parts"] = list(dict.fromkeys(p))
    if op in ("save", "save_as"):
        out["media"] = _bool(c, "media") if "media" in c else False
    if op == "save_as":
        out["name"] = _shows.clean_name(c.get("name"))
        out["template"] = _bool(c, "template") if "template" in c else False
    return out


def _groups(c):
    g = c.get("groups")
    if g is None:
        return None
    groups = set(v[3] for v in PARAMS.values())
    if not isinstance(g, list) or len(g) > 64 or any(x not in groups for x in g):
        raise ValueError("'groups' must be a list of param groups")
    return list(dict.fromkeys(g))


def _v_preset(c):
    op = _name(c, ("save", "recall", "set", "delete", "rescan"), "op")
    out = {"cmd": "preset", "op": op}
    if op == "save":
        out["kind"] = _name(c, _shows.KINDS, "kind")
        out["name"] = _shows.clean_name(c.get("name"))
        out["tags"] = _shows.clean_tags(c.get("tags"))
        out["groups"] = _groups(c)
        if out["kind"] == "layer":
            if c.get("stack_id") is not None:
                out["stack_id"] = _int(c, "stack_id", 1, 2 ** 31)
            else:
                out["layer"] = _name(c, [n for n in CONT if n.startswith("l_") or n == "serpent"], "layer")
    elif op in ("recall", "set", "delete"):
        out["ref"] = _tl.preset_ref(c.get("ref")) if _tl is not None else None
        if out["ref"].split("/")[0] not in _shows.KINDS:
            raise ValueError("unknown preset kind")
    if op == "recall":
        out["groups"] = _groups(c)
        out["morph"] = _num(c, "morph", 0.0, 64.0, 0.0)
        _v_trans_field(c, out)                     # [transitions]
    if op == "set":
        if "name" in c:
            out["name"] = _shows.clean_name(c["name"])
        if "tags" in c:
            out["tags"] = _shows.clean_tags(c["tags"])
        if "fav" in c:
            out["fav"] = _bool(c, "fav")
    return out


TL_CTX = None
if _tl is not None:
    TL_CTX = _tl.Ctx({n: PARAMS[n] for n in CONT}, DISCRETE, _v_tl_event, N_SCENES)
    _VALIDATORS["tl"] = lambda c: _tl.validate(c, TL_CTX)
if _shows is not None and _tl is not None:
    _VALIDATORS["show"] = _v_show
    _VALIDATORS["preset"] = _v_preset
LUMA = frozenset(n for n in CONT if n == "master" or n.startswith("l_") or n == "serpent" or "bright" in n
                 or n == "saturation")
CONTROL_CMDS = frozenset(_VALIDATORS)
_BLOB_OWNER = None


def blob(name):
    """(rev, bytes) of a big document of the controller that last published its state (any thread: the tuple is
    immutable and built on the GL thread in state()), or None."""
    owner = _BLOB_OWNER() if _BLOB_OWNER is not None else None
    cache = getattr(owner, "blob_cache", None) if owner is not None else None
    return cache.get(name) if cache else None


_NOT_EDITS = frozenset(("tap", "trigger", "ripple", "panic", "show", "preset", "midi_learn", "gen", "gen_cancel",
                        "online", "resync", "latch", "theme"))     # commands that do not change what a show file holds
                                                  # (theme: themes.py calls _changed() itself when the tray changes)
_TL_TRANSPORT = frozenset(("play", "stop", "seek", "home", "rec", "rec_mode", "ovr_mode", "reenable", "release", "hold"))


def mark_local(c):
    """remote_control: this validated command came from a panel opened on the laptop itself. Only the ONLINE plugin
    cares (its "Allow Google CLI tools" opt-in cannot be given from the network)."""
    return _online.mark_local(c) if _online is not None else c


def validate(c):
    """Normalized copy of one control command (clamped, unknown keys dropped); ValueError if malformed."""
    if not isinstance(c, dict) or not isinstance(c.get("cmd"), str) or c["cmd"] not in _VALIDATORS:
        raise ValueError("not a control command")
    return _VALIDATORS[c["cmd"]](c)


def _clean_mod(m):
    """One mod slot from untrusted JSON -> normalized dict or None."""
    try:
        if not isinstance(m, dict) or m.get("src") not in MOD_SOURCES or m["src"] == "none":
            return None
        return dict(src=m["src"], amt=_num(m, "amt", -1.0, 1.0, 0.0), rate=_num(m, "rate", 1 / 64, 64.0, 1.0),
                    phase=_num(m, "phase", 0.0, 1.0, 0.0))
    except ValueError:
        return None


def _clean_mods(raw):
    out = {}
    if isinstance(raw, dict):
        for name, slots in raw.items():
            if name in CONT and isinstance(slots, list):
                pair = [_clean_mod(m) for m in slots[:2]]
                pair += [None] * (2 - len(pair))
                if any(pair):
                    out[name] = pair
    return out


def _clean_base(raw, fill=None):
    base = dict(fill) if fill else {n: PARAMS[n][2] for n in CONT}
    if isinstance(raw, dict):
        for n in CONT:
            v = raw.get(n)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                base[n] = clamp(n, v)
    return base


def _clean_scene(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("base"), dict):
        return None
    name = raw.get("name")
    out = dict(name=name[:24] if isinstance(name, str) else "", base=_clean_base(raw["base"]),
               mods=_clean_mods(raw.get("mods")))
    if _stack is not None and isinstance(raw.get("stack"), dict):      # scenes saved before the math stack have none:
        out["stack"] = _stack.clean_stack(raw["stack"])               # recalling one leaves the stack alone
    if _skins is not None and isinstance(raw.get("skins"), dict):      # same rule for environment skins
        out["skins"] = _skins.clean_skins(raw["skins"])
    if _trans is not None and raw.get("transition") is not None:       # [transitions] the scene's own default
        try:
            t = _trans.clean_spec(raw["transition"])
        except ValueError:
            t = None
        if t is not None:
            out["transition"] = t
    return out


def _copy_mods(mods):
    return {n: [dict(m) if m else None for m in pair] for n, pair in mods.items()}


ART_EXT = (".png", ".jpg", ".jpeg", ".webp")


def list_art(folder):
    """File names in the art library, in the engine's order (art_layer.list_art when it exists). Never raises."""
    try:
        from art_layer import list_art as engine_list
        return [Path(x).name for x in engine_list(folder)]
    except Exception:
        pass
    try:                                             # same rule as the engine: creation order, oldest first
        out = []
        for f in Path(folder).iterdir():
            if (f.is_file() and f.suffix.lower() in ART_EXT and not f.name.lower().endswith(".depth.png")
                    and not f.name.startswith(".")):
                st = f.stat()
                out.append((getattr(st, "st_birthtime", st.st_ctime), f.name.casefold(), f.name))
        return [name for _, _, name in sorted(out)]
    except OSError:
        return []


def meta():
    """Static description of the control surface; the panel builds every fader from this."""
    return dict(
        params=[dict(name=n, lo=v[0], hi=v[1], default=v[2], group=v[3], label=v[4],
                     trigger=n in TRIGGERS, options=OPTIONS.get(n),
                     stepped=n in DISCRETE) for n, v in PARAMS.items()],
        groups=list(dict.fromkeys(v[3] for v in PARAMS.values())),
        triggers=list(TRIGGERS), sources=list(MOD_SOURCES), n_scenes=N_SCENES, quantize=list(QUANT),
        clock_sources=list(CLOCK_SOURCES), cue_cc={str(k): v for k, v in CUE_CC.items()},
        bypass=dict(keys=list(BYPASS_COVER), cover={k: list(v) for k, v in BYPASS_COVER.items()},
                    group_key=dict(_params.BYPASS_GROUP_KEY)),
        stack=_stack.meta() if _stack is not None else None, canvas_keys=list(CANVAS_KEYS),
        skins=_skins.meta() if _skins is not None else None,
        clips=dict(lanes={ln: list(v) for ln, v in _clips.LANE_PARAMS.items()}, clip_lanes=list(_clips.CLIP_LANES),
                   lane_names=dict(_clips.LANE_NAMES), curves=list(_clips.CURVES), widgets=list(_clips.WIDGETS))
        if _clips is not None else None,
        transitions=_trans.meta() if _trans is not None else None)


# ------------------------------------------------------------------ factory scenes


def _scene(name, mods=None, **over):
    base = {n: PARAMS[n][2] for n in CONT}
    for k, v in over.items():
        if k in base:
            base[k] = clamp(k, v)
    return dict(name=name, base=base, mods=_clean_mods(mods or {}))


def _m(src, amt, rate=1.0, phase=0.0):
    return dict(src=src, amt=amt, rate=rate, phase=phase)


def factory_scenes():
    """Starting looks for slots 1-8 (slot 1 is the default PANIC safe scene). Overwrite freely."""
    s = [None] * N_SCENES
    s[0] = _scene("Home")
    s[1] = _scene("Breathe", zoom=-0.2, rotate=0.04, speed=0.6, swim=0.45, trails=0.92, l_lines=0.3, palette=2,
                  mods={"zoom": [_m("lfo_sine", 0.12, 32)], "hue": [_m("lfo_tri", 0.05, 32)]})
    s[2] = _scene("Tunnel", travel=0.5, travel_mode=1, tunnel_curve=0.4, zoom=0.3, rotate=0.25, folds=12, twirl=0.35, flow=0.7, trails=0.9, palette=1,
                  mods={"zoom": [_m("lfo_saw", 0.25, 16)], "bulge": [_m("bass", 0.4)]})
    s[3] = _scene("Crawl", symmetry=0.0, flow=0.6, flow_dir=0.125, density=0.65, l_kene=1.0, l_beads=0.3,
                  l_lines=0.2, swim=0.4, palette=3,
                  mods={"flow_dir": [_m("lfo_sine", 0.08, 32)], "wave": [_m("mid", 0.5)]})
    s[4] = _scene("Jewel pulse", folds=6, density=0.7, chase=0.75, chase_width=0.35, saturation=1.25, l_beads=1.0, l_kene=0.5,
                  palette=0, mods={"zoom": [_m("bass", 0.15)], "chase": [_m("lfo_saw", 0.3, 4)]})   # was chase 0.9: near-black
    s[5] = _scene("Vortex", travel=0.3, travel_mode=3, twirl=0.7, rotate=-0.3, folds=10, zoom=0.2, swim=0.6, trails=0.94, palette=4,
                  mods={"twirl": [_m("lfo_sine", 0.3, 16)], "hue": [_m("lfo_saw", 0.5, 32)]})
    s[6] = _scene("Serpent liquid", serpent=0.7, serpent_mode=2, serpent_wiggle=0.7, symmetry=0.5, swim=0.9, wave=0.6, bulge=0.3, speed=0.8, l_ai=0.9, l_kene=0.6,
                  palette=2, mods={"bulge": [_m("lfo_sine", 0.4, 8)], "swim": [_m("mid", 0.3)]})
    s[7] = _scene("Peak", folds=16, speed=1.5, rotate=0.4, density=0.8, chase=0.8, chase_width=0.35, saturation=1.3, trails=0.8,
                  palette=1, mods={"zoom": [_m("bass", 0.3), _m("lfo_saw", 0.15, 4)], "twirl": [_m("random", 0.3, 4)],
                                   "hue": [_m("high", 0.15)]})                                      # was chase 1.0: near-black
    return s


# ------------------------------------------------------------------ MIDI clock (24 ppqn)


class MidiClock:
    """24 ppqn clock follower. Tempo = ticks per second over a sliding window (least-squares slope of
    tick number against arrival time), so ticks that arrive bunched together (USB / driver batching)
    do not disturb it the way per-tick intervals would."""
    WINDOW = 48                          # ticks (two beats)

    def __init__(self):
        self.ticks, self.last_t, self.bpm = 0, None, None
        self.running, self.jump, self._armed = True, False, False
        self._ts = deque(maxlen=self.WINDOW + 1)

    def feed(self, kind, val, t):
        if kind == "clock":
            if self.last_t is not None and t - self.last_t >= 0.25:
                self._ts.clear()         # transport gap: relearn the tempo
            self._ts.append(t)
            n = len(self._ts)
            if n >= 7 and self._ts[-1] - self._ts[0] > 1e-3:
                tm, im = sum(self._ts) / n, (n - 1) / 2.0
                var = sum((x - tm) ** 2 for x in self._ts)
                if var > 1e-9:
                    rate = sum((x - tm) * (k - im) for k, x in enumerate(self._ts)) / var     # ticks per second
                    bpm = 60.0 * rate / 24.0
                    if math.isfinite(bpm) and bpm > 0:
                        self.bpm = min(BPM_MAX, max(BPM_MIN, bpm))
            self.last_t = t
            if self._armed:              # the first tick after Start is the downbeat itself
                self._armed = False
            elif self.running:
                self.ticks += 1
        elif kind == "start":
            self.ticks, self.running, self.jump, self._armed = 0, True, True, True
        elif kind == "continue":
            self.running = True
        elif kind == "stop":
            self.running = False
        elif kind == "songpos":
            self.ticks, self.jump = int(val) * 6, True     # song position counts sixteenths = 6 ticks

    def active(self, now):
        return self.bpm is not None and self.last_t is not None and now - self.last_t < 1.0

    def reset(self):
        self.__init__()


# ------------------------------------------------------------------ Ableton Link (optional, `aalink`)


def _make_link(bpm):
    import warnings
    from aalink import Link
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            link = Link(bpm)
        except Exception:
            import asyncio
            link = Link(bpm, asyncio.new_event_loop())
    link.quantum = 4
    return link


class LinkBridge:
    """Owns the one Link session. It is created once, lazily, on a background thread; after that the
    session is only switched on / off (never rebuilt), and every write goes through that same thread.
    The render thread only reads tempo / beat, and non-finite readings are discarded."""

    def __init__(self, bpm=124.0, factory=_make_link):
        self.link, self.error, self.want, self.enabled = None, "", False, False
        self._bpm, self._factory = bpm, factory
        self._cmds = queue.Queue()
        self._wake, self._stop = threading.Event(), threading.Event()
        self._thread = None

    def request(self, on, bpm=None):
        if bpm is not None and self.link is None:
            self._bpm = bpm
        if on == self.want and (not on or self._thread is not None):
            return
        self.want = bool(on)
        if on and self._thread is None and not self.error:
            self._thread = threading.Thread(target=self._run, name="link", daemon=True)
            self._thread.start()
        self._wake.set()

    def set_tempo(self, bpm):
        self._cmds.put(("tempo", float(bpm)))
        self._wake.set()

    def force_beat(self, beat):
        self._cmds.put(("beat", float(beat)))
        self._wake.set()

    def read(self):
        """(bpm, beat) while the session is live and sane, else None. Never raises."""
        link = self.link
        if link is None or not self.enabled:
            return None
        try:
            bpm, beat = float(link.tempo), float(link.beat)
        except Exception:
            return None
        if not (math.isfinite(bpm) and math.isfinite(beat)) or bpm <= 0:
            return None
        return min(BPM_MAX, max(BPM_MIN, bpm)), beat

    def _run(self):
        try:
            self.link = self._factory(self._bpm)
        except Exception as ex:
            self.error = f"{type(ex).__name__}: {ex}"[:120]
            return
        while True:
            try:
                want = self.want and not self._stop.is_set()
                if want != self.enabled:
                    self.link.enabled = want
                    self.enabled = want
                while True:
                    kind, v = self._cmds.get_nowait()
                    if self.enabled and math.isfinite(v):
                        if kind == "tempo":
                            self.link.tempo = v
                        else:
                            self.link.force_beat(v)
            except queue.Empty:
                pass
            except Exception as ex:
                self.error = f"{type(ex).__name__}: {ex}"[:120]
            if self._stop.is_set():
                return
            self._wake.wait(0.5)
            self._wake.clear()

    def close(self):
        self._stop.set()
        self._wake.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=2)


# ------------------------------------------------------------------ controller


class Controller:
    def __init__(self, state_dir=None, restore=True, midi=True, bpm=124.0, time_fn=None, seed=None,
                 ai=True, ai_url="http://127.0.0.1:8766", art_dir=None, autosave=None, threads=True):
        self.dir = Path(state_dir) if state_dir is not None else Path(__file__).resolve().parent
        # Nothing on disk is written, renamed or quarantined by a controller that does not also restore
        # (headless / test runs with restore=False are read-only) unless autosave says otherwise.
        self.autosave = bool(restore) if autosave is None else bool(autosave)
        self.now = time_fn or time.perf_counter
        self.rng = random.Random(seed)
        self._seed = self.rng.random()

        # values
        self.base = defaults()
        self.mods = {}                                 # name -> [mod | None, mod | None]
        self._sm = dict(self.base)                     # smoothed base
        self.values = dict(self.base)
        self.bands = [0.0, 0.0, 0.0]
        self._rand = {}                                # (name, slot) -> (cycle, value) sample-and-hold
        self._pal_speed = 0.0
        self._pal_target = self.base["palette"]
        self.seq = _clean_seq(None)                    # 16-step sequencer feeding the "seq" / "seq_gate" mod sources
        self.xfade = dict(a=-1, b=-1, x=0.0)           # scene A/B crossfader (decks are scene slots)
        self.text = _textlayer.TextState() if _textlayer is not None else None   # words on show, font, phrase list + auto-cycle
        self._xside = None
        self._xf_cc, self._xf_live = None, False       # MIDI crossfader pickup state
        self.bypass = set()                            # bypassed keys (BYPASS_COVER): outputs pulled to params.NEUTRAL
        self.bypass_fade = BYPASS_FADE                 # s to cross-fade a bypass in / out
        self._byp = {}                                 # key -> level 0..1 (the fade)
        self._byp_once = None                          # s: fade time for the transition in progress (NEW CANVAS)
        self.stack = _stack.StackState() if _stack is not None else None      # math stack: hand-built layers
        self._stack_morph = False
        # environment skins (materials.py): INACTIVE until switched on in the panel (settings.skins_on)
        self.skins = _skins.SkinState(on=False) if _skins is not None else None
        self.gov_level = 0                             # the engine copies the governor's level here (0 ok .. 2 critical)
        self.gov_vram_free_mb = None                   # and the free VRAM it measured (None = unknown)
        self.canvas_prev = None                        # the look before NEW CANVAS (base, mods, bypass, stack)
        self.fresh = 0                                 # start time of a --fresh launch (the panel opens in SIMPLE)

        # clock
        self.bpm = min(BPM_MAX, max(BPM_MIN, float(bpm)))
        self.beat = 0.0
        self.clock_src = "internal"
        self.clock_active = "internal"
        self._taps = []
        self.mclock = MidiClock()
        self.clock_port = None                         # None = follow whichever port sends clock first
        self._clock_lock = None
        self._clock_seen = {}                          # port -> last tick time
        self.linkb = LinkBridge(self.bpm)

        # triggers
        self.held = {n: False for n in TRIGGERS}
        self.latch = {n: False for n in TRIGGERS}
        self.env = {"flash": 0.0, "punch": 0.0}
        self.strobe_rate = 0.25                        # beats per strobe cycle

        # scenes
        self.scenes = [None] * N_SCENES
        self.cur = -1
        self.pending = None                            # (slot, fire_beat, quantize beats when queued)
        self.morph_bars = 2.0
        self.quantize = "bar"
        self.glide_bars = 0.0                          # palette glide
        self.safe = 0
        self._morph = None
        self._scenes_ok = True                         # False after a read error: saving is off until a good load
        self.scenes_loaded = self._load_scenes()

        # clips (clips.py): one clip per lane on air, its macro positions; user clips + decks; engine-side undo
        lanes = _clips.CLIP_LANES if _clips is not None else ()
        self.clips = {}                                # user clips {id: clip} (factory clips live in clips.FACTORY)
        self.decks = []                                # pages of scene columns (clips.validate_deck)
        self.comp = {ln: dict(clip=None, macros=[], over=[]) for ln in lanes}
        self.locks = {ln: False for ln in lanes}       # PALETTE LOCK per lane (False = the palette follows the clip)
        self._lane_morph = {}                          # lane -> per-lane MORPH of a launch (src, to, names, start, dur)
        self.trans = _trans.Transitioner(self) if _trans is not None else None   # [transitions] BLEND state machine
        self._pend_trans = None                        # [transitions] the spec of the quantized recall in `pending`
        self._clip_pending = []                        # quantized launches / stops waiting for their beat
        self._macro_map = {}                           # param -> (lane, macro index) of the clips on air
        self._macro_writing = False                    # True while a macro writes its targets (not an override)
        self._clips_rev, self._clips_blob = 1, (0, b"{}")
        self.history = deque(maxlen=HISTORY_MAX)       # undo steps {key, t, old {param: v}, comp?, mods?, stack?}
        self._redo = []
        self._hist_mute, self._hist_n = False, 0

        # autopilot
        self.auto = dict(on=False, bars=8.0, mode="random", drift=False, drift_amt=0.3,
                         drift_params=[n for n in DRIFT_DEFAULT if n in CONT], pal_hold=0.0)
        self._auto_next = None
        self._pal_hold_until = 0.0
        self._drift = {}                               # name -> [ou, smoothed]

        # midi
        self.cue_ch = 16                               # 1..16, 0 = off
        self.learn = None
        self.midi_last = ""
        self._shift = False
        self._cc_hi = {}                               # CC binding -> was it >= 64 last time (rising-edge presses)
        self._out = []                                 # passthrough commands raised by MIDI (e.g. next_video)
        self._led_t = 0.0
        self.events = []
        try:
            self.midi = MidiManager(self.dir / "midi_map.json", scan=bool(midi), valid_target=valid_target,
                                    persist=self.autosave)
        except Exception:
            self.midi = MidiManager(None, scan=False, valid_target=valid_target)

        self.art_dir = Path(art_dir) if art_dir is not None else Path(__file__).resolve().parent / "art"

        # AI worker link (ai_worker.py, separate process); its own thread, never blocks update()
        here = Path(__file__).resolve().parent
        self.ai = AIClient([self.dir / ".remote_token", here / ".remote_token"], url=ai_url, start=bool(ai))

        self.ext = {}                                  # settings owned by other modules (governor.py: ext.governor.mode)
        self.signals = {}                              # external mod sources set each frame (camera_layer: crowd_*, hand_*)
        self.edits, self.panics = {}, 0                # per-param count of direct set() moves; PANIC count

        # timeline (timeline.py): automation lives in its own layer (_auto) over the fader bases, never in them
        self.tl = _tl.Timeline(TL_CTX) if _tl is not None else None
        self.tl_engaged = False                        # automation drives its params (after PLAY / a seek; RELEASE ends it)
        self.tl_hold = TL_HOLD
        self.tl_limited = 0                            # timeline flashes / strobes held back by the photosensitive rules
        self._auto = {}                                # param -> automation value this frame (the target instead of base)
        self._tl_ovr = {}                              # param -> [last performer move (tl clock), glide-from, glide start]
        self._tl_fade = {}                             # param -> [value, start]: RELEASE glides back to the fader
        self._tl_trig = set()                          # trigger names a timeline span holds this frame
        self._tl_buf, self._tl_span_ids, self._tl_blocked = [], set(), set()
        self._tl_clock = 0.0                           # s: sum of update() dt (deterministic, never the wall clock)
        self._tl_beat = None
        self._tl_flash_t = -1e9
        self._tl_cue_t = -1e9                          # last timeline cue (photosensitive: no rapid hard cuts)
        self._tl_byp_t = {}                            # bypass key -> last timeline toggle
        self._tl_firing = False                        # True while the timeline itself acts (nothing is re-recorded)
        self._tl_last_cue = None
        self._rec_g, self._rec_trig, self._rec_ids = {}, {}, set()
        self._rec_dirty, self._rec_compile_t = False, 0.0
        self._tl_blob = (0, b"{}")
        self._tl_blob_t = -1e9
        self.tl_docs, self.tl_i = [None], 0            # every timeline of the show; the active one is live in self.tl

        # shows + presets (shows.py): files on a worker thread
        self.shows = None
        if _shows is not None and _tl is not None:
            try:
                self.shows = _shows.ShowManager(self.dir, enabled=self.autosave, art_dir=self.art_dir, start=bool(threads))
            except Exception:
                self.shows = None
        self._show_dirty_t = self._show_first_t = None
        self._save_req_t = 0.0
        # thumbnails: a scene / preset save asks the engine for one snapshot frame. Only for the real show (state in
        # the code folder); tests and the panel demo (temp folders, no engine) do not get the extra command.
        self.thumbs = bool(restore and threads and self.dir.resolve() == Path(__file__).resolve().parent)
        self.soundtags = None                          # [themes] the engine attaches soundtags.SoundTags (audio tap) here
        self.fg_hold = False                           # [themes] the engine copies "flash guard is holding" here
        self.themes = None
        if _themes is not None:
            try:
                self.themes = _themes.ThemeManager(self)
            except Exception:
                self.themes = None
        self.restored = self._restore() if restore else False
        if restore and self.shows is not None:
            self.shows.submit("session")               # the open show's timeline comes back (transport stopped)
        self._save_t = self.now()
        self.online = None                             # plugins/online: loaded only while ext.online.on is true
        if _online is not None:
            _online.boot(self)

    # ================================================================ public API (params.StaticController)

    def set(self, name, value):
        if name not in PARAMS:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        self.base[name] = clamp(name, v)
        self.edits[name] = self.edits.get(name, 0) + 1  # direct moves only (governor.py: the user wins)
        if self._morph:
            self._morph["names"].discard(name)         # a hand on the fader overrides the morph
        if self._lane_morph:
            for lm in self._lane_morph.values():       # ... and a clip launch's lane morph
                lm["names"].discard(name)
        if not self._macro_writing:
            hit = self._macro_map.get(name)
            if hit is not None:                        # a fader on a macro's target: that macro is "overridden"
                self.comp[hit[0]]["over"][hit[1]] = True
        self._changed()
        if self.tl is not None and not self._tl_firing and name not in TRIGGERS:
            self._tl_touch(name)                       # performer wins over automation; RECORD writes the move

    def nudge(self, name, delta):
        if name in PARAMS:
            self.set(name, self.base[name] + delta)

    def trigger(self, name, on):
        if name not in TRIGGERS:
            return
        on = bool(on)
        if self.latch[name]:
            if on:
                self.held[name] = not self.held[name]
        else:
            self.held[name] = on
        if on and name in self.env:
            self.env[name] = 1.0                       # instant attack: a tap shorter than a frame still shows
        if self.tl is not None and not self._tl_firing and self.tl.recording():
            pos = self.tl.pos
            if self.held[name] and name not in self._rec_trig:
                self._rec_trig[name] = pos
            elif not self.held[name] and name in self._rec_trig:
                t0 = self._rec_trig.pop(name)
                it = self.tl.add_item("trigger", {"t": t0, "len": max(pos - t0, _tl.REC_GAP[self.tl.units]), "name": name})
                if it:
                    self._rec_ids.add(it["id"])

    def ripple(self, x, y):
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return
        if math.isfinite(x) and math.isfinite(y) and len(self.events) < 64:
            self.events.append(("ripple", min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))

    def tap(self, now=None):
        """Tap tempo: averages the last 4 taps, starts over after a 2 s gap; each tap lands on a beat."""
        t = self.now() if now is None else now
        if self._taps and t - self._taps[-1] > 2.0:
            self._taps = []
        self._taps = (self._taps + [t])[-4:]
        if len(self._taps) >= 2:
            iv = (self._taps[-1] - self._taps[0]) / (len(self._taps) - 1)
            if iv > 0:
                self.set_bpm(60.0 / iv)
        if self.clock_active == "internal":
            self.beat = float(round(self.beat))

    def set_bpm(self, bpm):
        if not math.isfinite(bpm):
            return
        self.bpm = min(BPM_MAX, max(BPM_MIN, float(bpm)))
        if self.clock_active == "link":
            self.linkb.set_tempo(self.bpm)

    def resync(self):
        """Downbeat: now is beat 1 of a bar and of a phrase."""
        self.beat = 0.0
        if self.pending:                               # a waiting recall fires on this downbeat
            self.pending = (self.pending[0], 0.0, self.pending[2])
        self._auto_next = None
        self._pal_hold_until = 0.0
        if self.clock_active == "link":
            self.linkb.force_beat(0.0)

    def art_step(self, d):
        """Previous / next piece in the art library (wraps)."""
        if "art_index" not in PARAMS:
            return
        n = min(len(list_art(self.art_dir)), int(PARAMS["art_index"][1]) + 1)
        if n > 0:
            self.set("art_index", (int(round(self.base["art_index"])) + int(d)) % n)

    def _skin_ai_cancel(self):
        """A generate-a-skin job that was already sent to the worker was cancelled: stop the worker's queue too."""
        if self.skins is not None and self.skins.cancel_ai:
            self.skins.cancel_ai = False
            try:
                self.ai.cancel()
            except Exception:
                pass

    def panic(self):
        """Failover: clear every trigger, cancel pending recalls / morphs, snap to the safe scene."""
        self.panics += 1                               # governor.py drops a pending suggestion / countdown
        if self.trans is not None:                     # [transitions] no transition / queue / dip survives PANIC
            self.trans.panic()
        self._pend_trans = None
        for n in TRIGGERS:
            self.held[n] = False
            self.base[n] = PARAMS[n][2]
        self.env = {k: 0.0 for k in self.env}
        self.pending = self._morph = None
        self._comp_clear()                             # no queued launch / lane morph survives PANIC
        self.learn = None
        self.auto["on"] = False                        # nothing may move the look away from the safe scene
        self._auto_next = None
        if self.tl is not None:                        # the timeline stops and lets go of everything it held
            self._tl_pass_end()
            self.tl.playing = self.tl.rec = False
            self.tl_engaged = False
            self._auto.clear()
            self._tl_ovr.clear()
            self._tl_fade.clear()
            self._tl_trig.clear()
            self._tl_span_ids.clear()
        if self.skins is not None:                     # an image on a wall is not part of the safe look either: every slot
            self.skins.panic()                         # procedural NOW (a cut, settings kept; whatever stack.py does)
            self._skin_ai_cancel()                     # and a skin being generated stops using the GPU
        if self.stack is not None:                     # the math stack is not part of the safe look: off, at once,
            self._stack_morph = False                  # and no half-finished stack morph left running
            self.stack.morph_cancel()
            self.stack.load(dict(layers=[]), fade=False)
            if self.canvas_prev is not None:           # on a clean canvas every engine layer is bypassed: with the stack
                keep = set(self.canvas_prev["bypass"])  # gone that would be a black wall. PANIC leaves the canvas: the
                for k in CANVAS_KEYS:                   # switches go back to what they were before NEW CANVAS.
                    if k not in keep:
                        self.bypass.discard(k)
                        self._byp.pop(k, None)
                self.canvas_prev, self._byp_once = None, None
        self.bypass.discard("layers")                  # the one bypass that is a blackout: PANIC must recover from it
        self._byp.pop("layers", None)                  # (single layers / tabs the performer killed stay killed)
        if _online is not None:
            _online.on_panic(self)                     # drops the assistant's queue, its autopilot -> suggest
        safe = self.scenes[self.safe] if 0 <= self.safe < N_SCENES else None
        if safe:
            self._start_recall(self.safe, 0.0)
        else:
            self.base.update({n: PARAMS[n][2] for n in CONT})
            self.mods = {}
            self.cur = -1
        self.base["master"] = max(self.base["master"], PARAMS["master"][2])
        for n in CONT:                                 # instant: skip the slew and any palette glide
            self._sm[n] = self.base[n]
        self._pal_target = self.base["palette"]

    def close(self):
        self._autosave()
        if self.shows is not None:
            try:
                if self._show_dirty_t is not None:
                    self._show_autosave()
                self.shows.close()
            except Exception:
                pass
        if _online is not None:
            _online.close(self)
        for part in (self.midi, self.ai, self.linkb):
            try:
                part.close()
            except Exception:
                pass

    # ---------------------------------------------------------------- modulation

    def set_mod(self, name, slot, src, amt=0.0, rate=1.0, phase=0.0):
        if name not in CONT or slot not in (0, 1) or src not in MOD_SOURCES:
            return
        pair = self.mods.get(name) or [None, None]
        pair[slot] = _clean_mod(dict(src=src, amt=amt, rate=rate, phase=phase))
        if any(pair):
            self.mods[name] = pair
        else:
            self.mods.pop(name, None)

    def _source(self, m, name, slot, beat):
        src = m["src"]
        if src == "bass":
            return self.bands[0]
        if src == "mid":
            return self.bands[1]
        if src == "high":
            return self.bands[2]
        if src in ("seq", "seq_gate"):                 # the 16 steps span rate * 4 beats (rate 1 = sixteenths, one bar)
            pos = beat * 4.0 / m["rate"] + m["phase"] * SEQ_STEPS
            i = int(math.floor(pos)) % SEQ_STEPS
            f = pos - math.floor(pos)
            if src == "seq_gate":
                return math.exp(-6.0 * f) if self.seq["gates"][i] else 0.0
            a, b = self.seq["vals"][i - 1], self.seq["vals"][i]
            return b if f >= 0.15 else a + (b - a) * (f / 0.15)        # short glide into each step: no clicks
        sig = self.signals.get(src)                    # crowd camera (camera_layer.Signals): already 0..1 / -1..1 + slewed
        if sig is not None:
            return sig
        ph = beat / m["rate"] + m["phase"]
        cyc = math.floor(ph)
        f = ph - cyc
        if src == "lfo_sine":
            return math.sin(2 * math.pi * f)
        if src == "lfo_tri":
            return 4 * f if f < 0.25 else (2 - 4 * f if f < 0.75 else 4 * f - 4)
        if src == "lfo_saw":
            return 2 * f - 1
        if src == "lfo_square":
            return 1.0 if f < 0.5 else -1.0
        if src == "random":                            # sample-and-hold: a new value each cycle
            key = (name, slot)
            hit = self._rand.get(key)
            if hit is None or hit[0] != cyc:
                hit = (cyc, random.Random(f"{self._seed}:{name}:{slot}:{cyc}").uniform(-1.0, 1.0))
                self._rand[key] = hit
            return hit[1]
        return 0.0

    def _mod_sum(self, mods, name, beat):
        pair = mods.get(name)
        if not pair:
            return 0.0
        total = 0.0
        for slot, m in enumerate(pair):
            if m:
                total += m["amt"] * self._source(m, name, slot, beat)
        return total

    # ---------------------------------------------------------------- bypass

    def set_bypass(self, key, on=None):
        """Switch a layer / tab / "fx" off (on=True means bypassed) or back on; on=None toggles."""
        if key not in BYPASS_COVER:
            return
        if on is None:
            on = key not in self.bypass
        if on != (key in self.bypass):                 # persist soon (0.5 s), not at the next 5 s autosave: a crash
            self._save_t = min(self._save_t, self.now() - AUTOSAVE_S + 0.5)   # must not bring a killed layer back
            if self.tl is not None and not self._tl_firing and self.tl.recording():
                it = self.tl.add_item("event", {"t": self.tl.pos, "cmd": {"cmd": "bypass", "key": key, "on": bool(on)}})
                if it:
                    self._rec_ids.add(it["id"])
        (self.bypass.add if on else self.bypass.discard)(key)

    def _step_bypass(self, dt):
        """Advance the fades; returns {param: weight 0..1 towards neutral}."""
        fade = self.bypass_fade if self._byp_once is None else self._byp_once
        step = 1.0 if fade <= 0 else dt / fade
        if self._byp_once is not None and all(self._byp.get(k, 0.0) >= 1.0 for k in self.bypass) and not (
                set(self._byp) - self.bypass):
            self._byp_once = None                      # the canvas crossfade has landed: back to the normal fade time
        for key in set(self._byp) | self.bypass:
            lvl = self._byp.get(key, 0.0)
            lvl = min(1.0, lvl + step) if key in self.bypass else max(0.0, lvl - step)
            if lvl <= 0.0 and key not in self.bypass:
                self._byp.pop(key, None)
            else:
                self._byp[key] = lvl
        w = {}
        for key, lvl in self._byp.items():
            for n in BYPASS_COVER[key]:
                if lvl > w.get(n, 0.0):
                    w[n] = lvl
        return w

    # ---------------------------------------------------------------- new canvas / restore previous

    def new_canvas(self, fresh=False):
        """Crossfade to a clean black canvas with ONE moving stripe layer in the math stack. Every engine layer and
        effect is bypassed (CANVAS_KEYS), nothing stored is edited, and the look before is kept for
        restore_canvas(). Pressed again while already on a canvas it starts the stack over and keeps the ORIGINAL
        way back. fresh=True (launch option): already there on the first frame, and the panel opens in SIMPLE."""
        if self.stack is None:
            return
        if self.canvas_prev is None:
            self.canvas_prev = dict(base={n: self.base[n] for n in CONT}, mods=_copy_mods(self.mods),
                                    bypass=sorted(self.bypass), stack=self.stack.to_json(), auto=bool(self.auto["on"]))
        if self.trans is not None:                     # [transitions] a canvas is not a scene: drop any BLEND
            self.trans.cancel(revert=False)
        self.pending = self._morph = None
        self._stack_morph = False
        self.auto["on"] = False                        # a scene step must not paint over the canvas
        self._auto_next = None
        self.bypass |= set(CANVAS_KEYS)
        self.bypass -= {k for k, cover in BYPASS_COVER.items() if "l_stack" in cover}
        if self.base["l_stack"] <= 0.0:
            self.base["l_stack"] = PARAMS["l_stack"][2]
        self.mods.pop("l_stack", None)
        self.stack.morph_cancel()
        self.stack.load(dict(layers=[]), fade=not fresh)
        lay = self.stack.add(_stack.CANVAS_START["gen"], _stack.CANVAS_START)
        if fresh:
            self.fresh = int(time.time())
            self._byp = {k: 1.0 for k in self.bypass}
            self._sm["l_stack"] = self.base["l_stack"]
            self.stack._phase(lay["id"])["op"] = None
        else:
            self._byp_once = CANVAS_FADE

    def restore_canvas(self):
        """RESTORE PREVIOUS: faders, mods, bypass switches and the stack exactly as they were before NEW CANVAS."""
        prev, self.canvas_prev = self.canvas_prev, None
        if prev is None or self.stack is None:
            return
        self.pending = self._morph = None
        self._stack_morph = False
        self.base.update(prev["base"])
        self.mods = _copy_mods(prev["mods"])
        self.bypass = {k for k in prev["bypass"] if k in BYPASS_COVER}
        self.stack.morph_cancel()
        self.stack.load(prev["stack"])
        if prev.get("auto"):
            self.auto["on"], self._auto_next = True, None
        self.cur = -1
        self._byp_once = CANVAS_FADE

    # ---------------------------------------------------------------- clips (clips.py): lanes, launch, macros, decks
    # A clip writes Controller.base for its own lane only (every owned param: the clip's value, a macro's or the
    # registry default). Nothing downstream changes: mods, bypass (never touched by a launch), the governor and
    # scenes work as before. A launch MORPHS per lane (stepped params flip at the middle), quantized like a scene.

    def _comp_clear(self):
        """Nothing is 'on air' as a clip any more (a snapshot recall, the crossfader, PANIC, init)."""
        self._lane_morph.clear()
        self._clip_pending = []
        self._macro_map = {}
        for ln in self.comp:
            self.comp[ln] = dict(clip=None, macros=[], over=[])

    def _macro_map_build(self):
        self._macro_map = {}
        for ln, on in self.comp.items():
            cl = _clips.get(on["clip"], self.clips) if on["clip"] else None
            for i, m in enumerate(cl["macros"] if cl else ()):
                for t in m["targets"]:
                    self._macro_map[t["p"]] = (ln, i)

    def _comp_json(self):
        out = {ln: dict(clip=on["clip"], macros=list(on["macros"])) for ln, on in self.comp.items()}
        out["locks"] = dict(self.locks)
        return out

    def _comp_restore(self, raw):
        """comp from live_state / a show: which clip is on air per lane + its macro positions (values are in base)."""
        raw = raw if isinstance(raw, dict) else {}
        for ln in self.comp:
            d = raw.get(ln) if isinstance(raw.get(ln), dict) else {}
            cid = d.get("clip")
            cl = _clips.get(cid, self.clips) if isinstance(cid, str) else None
            if cl is None or cl["lane"] != ln:
                self.comp[ln] = dict(clip=None, macros=[], over=[])
                continue
            ms = d.get("macros") if isinstance(d.get("macros"), list) else []
            pos = []
            for i, m in enumerate(cl["macros"]):
                x = ms[i] if i < len(ms) else None
                ok = isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                pos.append(_clips.snap(m, x if ok else m["default"]))
            self.comp[ln] = dict(clip=cl["id"], macros=pos, over=[False] * len(pos))
        lk = raw.get("locks") if isinstance(raw.get("locks"), dict) else {}
        for ln in self.locks:
            if isinstance(lk.get(ln), bool):
                self.locks[ln] = lk[ln]
        self._macro_map_build()

    def _comp_state(self):
        r = lambda x: round(x, 3)
        out = {ln: dict(clip=on["clip"], macros=[r(x) for x in on["macros"]], over=[int(o) for o in on["over"]])
               for ln, on in self.comp.items()}
        out["locks"] = [ln for ln, v in self.locks.items() if v]
        out["pending"] = sorted({q["lane"] for q in self._clip_pending})
        out["undo"] = [len(self.history), len(self._redo)]
        out["rev"] = self._clips_rev
        return out

    def _clip_do(self, c):
        op = c["op"]
        if op == "launch":
            self.clip_launch(c["id"], macros=c.get("macros"), bars=c.get("bars"),
                             now=c.get("now", False) or self._tl_firing, lane=c.get("lane"))
        elif op == "stop":
            self.clip_stop(c["lane"], bars=c.get("bars"), now=c.get("now", False) or self._tl_firing)
        elif op == "save":
            self.clips[c["clip"]["id"]] = c["clip"]
            self._clips_rev += 1
            if any(on["clip"] == c["clip"]["id"] for on in self.comp.values()):
                self._macro_map_build()
        elif op == "delete":
            if self.clips.pop(c["id"], None) is not None:
                self._clips_rev += 1
        elif op == "lock":
            self.locks[c["lane"]] = c["palette"]
        self._save_t = min(self._save_t, self.now() - AUTOSAVE_S + 0.5)

    def _deck_do(self, c):
        if c["op"] == "save":
            d = c["deck"]
            others = [x for x in self.decks if x["id"] != d["id"]]
            taken = {col["id"] for x in others for col in x["columns"]}
            if any(col["id"] in taken for col in d["columns"]):
                return                                 # column ids are unique across decks (scene_go addresses them)
            i = next((k for k, x in enumerate(self.decks) if x["id"] == d["id"]), None)
            if i is None:
                if len(self.decks) >= _clips.MAX_DECKS:
                    return
                self.decks.append(d)
            else:
                self.decks[i] = d
        else:
            self.decks = [x for x in self.decks if x["id"] != c["id"]]
        self._clips_rev += 1
        self._save_t = min(self._save_t, self.now() - AUTOSAVE_S + 0.5)

    def _queue_clip(self, now, **job):
        q = QUANT.get(self.quantize, 0)
        if now or q == 0:
            self._clip_fire(**job)
        else:                                          # one waiting job per lane: the newest press wins
            self._clip_pending = [x for x in self._clip_pending if x["lane"] != job["lane"]]
            self._clip_pending.append(dict(job, fire=(math.floor(self.beat / q + 1e-6) + 1) * q))

    def clip_launch(self, cid, macros=None, bars=None, now=False, lane=None, hkey=None):
        """Launch clip `cid` on its lane (quantized unless now). macros = {"1".."8": v} positions (others: default)."""
        cl = _clips.get(cid, self.clips) if _clips is not None else None
        if cl is None or (lane is not None and lane != cl["lane"]):
            return False
        self._queue_clip(now, op="launch", lane=cl["lane"], id=cid, macros=macros, bars=bars, hkey=hkey)
        return True

    def clip_stop(self, lane, bars=None, now=False, hkey=None):
        """Take a lane off the wall: its layer amounts to 0 (params.NEUTRAL), its other params to defaults."""
        if lane not in self.comp:
            return False
        self._queue_clip(now, op="stop", lane=lane, id=None, macros=None, bars=bars, hkey=hkey)
        return True

    def _clip_fire(self, op, lane, id=None, macros=None, bars=None, hkey=None):
        cl = _clips.get(id, self.clips) if id else None
        if op == "launch" and cl is None:
            return                                     # deleted while it waited for its beat
        ref = cl or dict(lane=lane, sets={}, macros=[], mods={}, stack=None, palette=None, bars=None)
        pos = [None] * len(ref["macros"])
        for k, v in (macros or {}).items():
            if int(k) - 1 < len(pos):
                pos[int(k) - 1] = v
        r = _clips.resolve(ref, pos, stop=op == "stop")
        to = r["updates"]
        if r["palette"] is not None and not self.locks.get(lane, False):
            to["palette"] = r["palette"]
        names = list(to)
        if bars is None:
            bars = cl["bars"] if cl is not None and cl.get("bars") is not None else self.morph_bars
        owned = set(_clips.LANE_PARAMS[lane])
        self._hist_push(hkey or self._hkey_once(), names + (["palette"] if "palette" not in to else []), comp=True,
                        mods=owned, stack=r["stack"] is not None)
        old_mods = {n: pair for n, pair in self.mods.items() if n in owned}
        for n in old_mods:
            self.mods.pop(n)
        self.mods.update(_copy_mods(r["mods"]))
        use_stack = r["stack"] is not None and self.stack is not None
        if use_stack:
            self._stack_morph = False
            self.stack.morph_cancel()
            if bars > 0:
                self.stack.morph_start(r["stack"])
            else:
                self.stack.load(r["stack"])
        if self._morph:
            self._morph["names"] -= set(names)         # a running snapshot morph lets go of this lane
        if bars <= 0:
            self.base.update(to)
            self._lane_morph.pop(lane, None)
        else:
            self._lane_morph[lane] = dict(src={n: self.base[n] for n in names}, to=to, names=set(names),
                                          start=self.beat, dur=bars * 4.0, old=old_mods, stack=use_stack, t=0.0)
        self.comp[lane] = dict(clip=id if op == "launch" else None, macros=list(r["macros"]),
                               over=[False] * len(r["macros"]))
        self._macro_map_build()
        self.cur = -1                                  # the faders no longer match a saved snapshot
        if self.tl is not None and not self._tl_firing and self.tl.recording():
            cmd = {"cmd": "clip", "op": op, "lane": lane}
            if op == "launch":
                cmd["id"] = id
            it = self.tl.add_item("event", {"t": self.tl.pos, "cmd": cmd})
            if it:
                self._rec_ids.add(it["id"])
        self._changed()

    def _step_lanes(self):
        disc = _DISCRETE_SET
        for lane, lm in list(self._lane_morph.items()):
            t = (self.beat - lm["start"]) / lm["dur"]
            if t < 0:
                lm["start"], t = self.beat, 0.0
            t = min(1.0, t)
            lm["t"] = t
            src, to = lm["src"], lm["to"]
            for n in lm["names"]:
                a, b = src[n], to[n]
                if n in disc:
                    self.base[n] = b if t >= 0.5 else a
                else:
                    self.base[n] = b if t >= 1.0 else a + (b - a) * t
            if lm["stack"] and self.stack is not None:
                self.stack.morph_to(t)
            if t >= 1.0:
                del self._lane_morph[lane]

    def macro(self, lane, n, v, key=None):
        """Move macro n (1-based) of the clip on air in `lane` to v (0..1, snapped to its steps). The targets are
        written with set(): counted in Controller.edits (the user wins over the governor), recorded by the timeline."""
        on = self.comp.get(lane)
        cl = _clips.get(on["clip"], self.clips) if on and on["clip"] else None
        if cl is None or not 1 <= n <= len(cl["macros"]):
            return False
        i = n - 1
        x = _clips.snap(cl["macros"][i], v)
        vals = _clips.macro_values(cl, i, x)
        self._hist_push(key, list(vals), comp=True)
        self._macro_writing = True
        try:
            for p, val in vals.items():
                self.set(p, val)
        finally:
            self._macro_writing = False
        on["macros"][i] = x
        on["over"][i] = False
        return True

    def scene_go(self, cid, now=False, bars=None):
        """Fire a scene column (by id, any deck): per lane launch its clip, STOP or KEEP. One undo step."""
        col = _clips.find_column(self.decks, cid) if _clips is not None else None
        if col is None:
            return False
        bars = col.get("bars", bars)
        key = self._hkey_once()
        for lane in self.comp:
            cell = col["lanes"].get(lane, "keep")
            if cell == "stop":
                self.clip_stop(lane, bars=bars, now=now, hkey=key)
            elif isinstance(cell, dict):
                self.clip_launch(cell["clip"], macros=cell.get("macros"), bars=bars, now=now, lane=lane, hkey=key)
        return True

    # ---------------------------------------------------------------- engine-side undo (gesture-coalesced)

    def _hkey(self, c, *auto):
        """Undo key of a panel command: its gesture id, else ("auto", what...) = coalesced by time."""
        return ("g", c["g"]) if c.get("g") is not None else ("auto",) + auto

    def _hkey_once(self):
        self._hist_n += 1
        return ("one", self._hist_n)

    def _hist_push(self, key, names, comp=False, mods=None, stack=False):
        """Before a change: remember what `names` (and optionally comp / mods / the stack) were. A step with the same
        key as the last one merges into it (a drag = one step): gesture ids within HIST_G_GAP, "auto" keys within
        HIST_GAP of the last move, "one" keys always (several lanes of one scene_go)."""
        if self._hist_mute or self._tl_firing:
            return
        now = self.now()
        last = self.history[-1] if self.history else None
        if last is not None and key is not None and last["key"] == key and (
                key[0] == "one" or now - last["t"] <= (HIST_G_GAP if key[0] == "g" else HIST_GAP)):
            for n in names:
                last["old"].setdefault(n, self.base[n])
            if comp and last["comp"] is None:
                last["comp"] = copy.deepcopy(self.comp)
            if mods:
                last["mods"] = last["mods"] or {}
                for n in mods:
                    if n not in last["mods"]:
                        last["mods"][n] = copy.deepcopy(self.mods.get(n))
            if stack and last["stack"] is None and self.stack is not None:
                last["stack"] = self.stack.to_json()
            last["t"] = now
            return
        self.history.append(dict(key=key, t=now, old={n: self.base[n] for n in names},
                                 comp=copy.deepcopy(self.comp) if comp else None,
                                 mods={n: copy.deepcopy(self.mods.get(n)) for n in mods} if mods else None,
                                 stack=self.stack.to_json() if stack and self.stack is not None else None))
        self._redo = []

    def _hist_apply(self, st):
        """Put a step back; returns its inverse (for redo / undo of the redo)."""
        inv = dict(key=None, t=self.now(), old={n: self.base[n] for n in st["old"]},
                   comp=copy.deepcopy(self.comp) if st["comp"] is not None else None,
                   mods={n: copy.deepcopy(self.mods.get(n)) for n in st["mods"]} if st["mods"] is not None else None,
                   stack=self.stack.to_json() if st["stack"] is not None and self.stack is not None else None)
        for lm in self._lane_morph.values():
            lm["names"] -= set(st["old"])
        if self._morph:
            self._morph["names"] -= set(st["old"])
        self._hist_mute, self._macro_writing = True, True
        try:
            for n, v in st["old"].items():
                self.set(n, v)
        finally:
            self._hist_mute, self._macro_writing = False, False
        if st["mods"] is not None:
            for n, pair in st["mods"].items():
                if pair:
                    self.mods[n] = copy.deepcopy(pair)
                else:
                    self.mods.pop(n, None)
        if st["stack"] is not None and self.stack is not None:
            self._stack_morph = False
            self.stack.morph_cancel()
            self.stack.load(st["stack"])
        if st["comp"] is not None:
            self.comp = copy.deepcopy(st["comp"])
            self._macro_map_build()
        if st.get("theme") is not None and self.themes is not None:   # [themes] undo of a theme restore
            inv["theme"] = self.themes.undo_apply(st["theme"])
        return inv

    def undo(self):
        if self.history:
            self._redo.append(self._hist_apply(self.history.pop()))
            return True
        return False

    def redo(self):
        if self._redo:
            self.history.append(self._hist_apply(self._redo.pop()))
            return True
        return False

    # ---------------------------------------------------------------- scenes

    def set_xfade(self, a=None, b=None, x=None):
        """Scene A/B crossfader: the faders sit at a hand-set blend of two saved scenes. Continuous params
        interpolate; stepped params and the modulation set flip at the middle. Empty decks leave the look alone."""
        xf = self.xfade
        if a is not None:
            xf["a"] = int(a)
        if b is not None:
            xf["b"] = int(b)
        if a is not None or b is not None:
            self._xside = None
        if x is None:
            return
        xf["x"] = x = min(1.0, max(0.0, float(x)))
        sa = self.scenes[xf["a"]] if 0 <= xf["a"] < N_SCENES else None
        sb = self.scenes[xf["b"]] if 0 <= xf["b"] < N_SCENES else None
        if sa is None or sb is None:
            return
        if self.trans is not None:                     # [transitions] the crossfader takes the look by hand
            self.trans.cancel(revert=False)
        self.pending = self._morph = None
        self._comp_clear()
        for n in CONT:
            va, vb = sa["base"][n], sb["base"][n]
            self.base[n] = (vb if x >= 0.5 else va) if n in DISCRETE else va + (vb - va) * x
        if self.stack is not None and "stack" in sa and "stack" in sb:
            self._stack_morph = False
            self.stack.morph_cancel()
            self.stack.load(_stack.blend(sa["stack"], sb["stack"], x))
        side = "b" if x >= 0.5 else "a"
        if side != self._xside:
            self._xside = side
            self.mods = _copy_mods((sb if side == "b" else sa)["mods"])
        self.cur = xf["a"] if x <= 0.0 else xf["b"] if x >= 1.0 else -1

    def _trans_do(self, c):
        """[transitions] {"cmd": "transition", "op": default | scene | origin | cancel | finish}"""
        tr, op = self.trans, c["op"]
        if op == "default":
            spec = c.get("spec")
            tr.default = _trans.clean_spec(spec) if spec is not None and not isinstance(spec, dict) else spec
            self._changed()
        elif op == "scene":
            sc = self.scenes[c["i"]]
            if sc is not None:
                if c.get("spec") is None:
                    sc.pop("transition", None)
                else:
                    sc["transition"] = c["spec"]
                self._save_scenes()
                self._changed()
        elif op == "origin":
            tr.origin = [c["x"], c["y"]]
        elif op == "cancel":
            tr.cancel()
        elif op == "finish":
            tr.finish()

    def _trans_len(self, x):
        """[transitions] a knob on the default's length: 1 / 2 / 4 / 8 / 16 beats."""
        tr = self.trans
        if tr.default is None:
            return
        beats = (1.0, 2.0, 4.0, 8.0, 16.0)[min(4, int(min(1.0, max(0.0, x)) * 5))]
        tr.default = dict(tr.default, beats=beats)
        tr.default.pop("secs", None)

    def scene_save(self, i):
        if not 0 <= i < N_SCENES:
            return
        snap = dict(name="", base={n: self.base[n] for n in CONT}, mods=_copy_mods(self.mods))
        if self.skins is not None:
            snap["skins"] = self.skins.to_json()
        if self.stack is not None:
            snap["stack"] = self.stack.to_json()
            if self.canvas_prev is not None:           # saved on a clean canvas: scenes do not store bypass switches, so
                for key in self.bypass & set(CANVAS_KEYS):      # the hidden layers / effects are stored at neutral.
                    for n in BYPASS_COVER[key]:        # Recalled anywhere, the scene looks like what was on the wall.
                        snap["base"][n] = _params.NEUTRAL[n]
                        snap["mods"].pop(n, None)
        if not self._scenes_ok:
            self._load_scenes()                        # the file was unreadable earlier: try again before writing
        if self.scenes[i]:
            snap["name"] = self.scenes[i]["name"]
        self.scenes[i] = snap
        self.cur = i
        self._save_scenes()
        self._changed()
        if self.thumbs and self.shows is not None and self.shows.enabled:   # pad thumbnail: one low-res output frame
            self._out.append(passthrough("snapshot"))
            self.shows.submit("scene_thumb", slot=i, snap_t=time.time())

    def scene_clear(self, i):
        if not 0 <= i < N_SCENES:
            return
        if not self._scenes_ok:
            self._load_scenes()
        if self.scenes[i] is not None:
            self.scenes[i] = None
            if self.cur == i:
                self.cur = -1
            self._save_scenes()
            self._changed()
            if self.shows is not None and self.shows.enabled:
                self.shows.submit("scene_thumb_clear", slot=i)

    def scene_recall(self, i, now=False, transition=None):
        """Recall slot i at the next quantize boundary (or immediately), morphing over morph_bars.
        [transitions] transition: a spec / kind / "morph" for this recall (else the scene's own, else the default)."""
        if not 0 <= i < N_SCENES or self.scenes[i] is None:
            return False
        spec = None
        if self.trans is not None:
            try:
                spec = self.trans.resolve(None if transition is None else transition, self.scenes[i])
            except ValueError:
                spec = None
        q = QUANT.get(spec.get("quantize", self.quantize) if spec else self.quantize, 0)
        self._pend_trans = spec
        if now or q == 0:
            self.pending = None
            self._manual_recall(i)
        else:
            self.pending = (i, (math.floor(self.beat / q + 1e-6) + 1) * q, q)
        return True

    def _manual_recall(self, i):
        """A recall the performer asked for: autopilot waits a full period before it changes the look again."""
        spec, self._pend_trans = self._pend_trans, None
        if self.trans is not None:                     # [transitions] BLEND / MORPH / queued behind a running one
            bars = self.morph_bars
            self.trans.run(lambda: self._start_recall(i, bars), spec, f"scene {i + 1}")
        else:
            self._start_recall(i, self.morph_bars)
        if self.auto["on"]:
            self._auto_next = self.beat + self.auto["bars"] * 4.0
        if self.tl is not None and not self._tl_firing and self.tl.recording():
            cue = {"t": self.tl.pos, "kind": "scene", "slot": i, "morph": self.morph_bars}
            if spec is not None:                       # [transitions] the recorded cue replays the same transition
                cue["transition"] = spec
            it = self.tl.add_item("cue", cue)
            if it:
                self._rec_ids.add(it["id"])

    def _start_recall(self, i, bars, keep_palette=False):
        sc = self.scenes[i]
        if sc is None:
            return
        self._comp_clear()                             # a snapshot writes every fader: no clip is "on air" any more
        to = dict(sc["base"])
        if keep_palette:
            to["palette"] = self.base["palette"]
        self.cur = i
        # What is on screen right now is the start of the new morph. Mid-morph that is a blend: the
        # interrupted target's mods at weight t plus whatever was still fading out at (1 - t).
        mo = self._morph
        if mo:
            t = mo["t"]
            old = [(self.mods, t)] + [(m, w * (1.0 - t)) for m, w in mo["old"]]
        else:
            old = [(self.mods, 1.0)]
        old = [(m, w) for m, w in old if w > 0.01 and m][:4]
        self.mods = _copy_mods(sc["mods"])
        if self.skins is not None and "skins" in sc:   # the engine crossfades a changed source by itself (0.4 s)
            self.skins.load(sc["skins"])
        if self.stack is not None:
            self._stack_morph = False
            self.stack.morph_cancel()
            if "stack" in sc:                          # numeric fields glide with the morph, stepped ones snap midway
                if bars <= 0:
                    self.stack.load(sc["stack"])
                else:
                    self.stack.morph_start(sc["stack"])
                    self._stack_morph = True
        if bars <= 0:
            self.base.update(to)
            self._morph = None
        else:
            self._morph = dict(src={n: self.base[n] for n in CONT}, to=to, names=set(CONT),
                               start=self.beat, dur=bars * 4.0, old=old, t=0.0)

    def _step_morph(self):
        mo = self._morph
        if not mo:
            return
        t = (self.beat - mo["start"]) / mo["dur"]
        if t < 0:                                      # the clock jumped backwards: restart the timing
            mo["start"], t = self.beat, 0.0
        t = min(1.0, t)
        mo["t"] = t
        if self._stack_morph:
            self.stack.morph_to(t)
            self._stack_morph = t < 1.0
        for n in mo["names"]:
            a, b = mo["src"][n], mo["to"][n]
            if n in DISCRETE:
                self.base[n] = b if t >= 0.5 else a
            else:
                self.base[n] = b if t >= 1.0 else a + (b - a) * t
        if t >= 1.0:
            self._morph = None

    def _load_scenes(self):
        """scenes.json -> self.scenes. Never raises. Returns True if the file was loaded.

        missing file        factory scenes, saving allowed
        unreadable (OSError: locked, AV scan, network share blip)
                            factory scenes in memory, the file is left alone and saving is DISABLED until
                            a later load works, so a transient error can never overwrite real scenes
        bad JSON / shape    the file is moved aside to scenes.json.<time>.bad (only by a controller that
                            is allowed to write), then factory scenes
        """
        path = self.dir / "scenes.json"
        self._scenes_ok = True
        try:
            text = path.read_text(encoding="utf-8") if path.exists() else None
        except OSError:
            self._scenes_ok = False
            self.scenes = factory_scenes()
            return False
        if text is None:
            self.scenes = factory_scenes()
            return False
        try:
            raw = json.loads(text)["scenes"]
            if not isinstance(raw, list):
                raise ValueError("scenes must be a list")
            sc = [_clean_scene(x) for x in raw[:N_SCENES]]
        except (ValueError, KeyError, TypeError, RecursionError):
            if self.autosave:
                quarantine(path)
            self.scenes = factory_scenes()
            return False
        self.scenes = sc + [None] * (N_SCENES - len(sc))
        return True

    def _save_scenes(self):
        if not self.autosave or not self._scenes_ok:
            return False
        return atomic_write_json(self.dir / "scenes.json", dict(version=1, scenes=self.scenes))

    # ---------------------------------------------------------------- autosave / restore

    def _settings(self):
        return dict(bpm=round(self.bpm, 3), clock_src=self.clock_src, strobe_rate=self.strobe_rate,
                    latch=dict(self.latch), morph=self.morph_bars, quantize=self.quantize, glide=self.glide_bars,
                    safe=self.safe, cue_ch=self.cue_ch, autopilot=dict(self.auto),
                    seq=dict(vals=list(self.seq["vals"]), gates=list(self.seq["gates"])),
                    text=self.text.to_json() if self.text is not None else None,
                    bypass=sorted(self.bypass), bypass_fade=self.bypass_fade, ext=dict(self.ext),
                    stack=self.stack.to_json() if self.stack is not None else None, canvas=self.canvas_prev,
                    skins=self.skins.to_json() if self.skins is not None else None,
                    skins_on=bool(self.skins.on) if self.skins is not None else False,
                    clips=[copy.deepcopy(x) for x in self.clips.values()], decks=copy.deepcopy(self.decks),
                    comp=self._comp_json(), themes=self.themes.to_json() if self.themes is not None else None,
                    trans=dict(default=self.trans.default, origin=list(self.trans.origin)) if self.trans is not None else None)

    def _autosave(self):
        if not self.autosave:
            return False
        return atomic_write_json(self.dir / "live_state.json", dict(
            version=1, base={n: self.base[n] for n in CONT}, mods=self.mods, cur=self.cur, settings=self._settings()))

    def _restore(self):
        """live_state.json -> live state (not triggers: a restart never comes back blacked out). Never raises."""
        try:
            path = self.dir / "live_state.json"
            if not path.exists():
                return False
            d = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                return False
            return self._apply_state(d)
        except Exception:
            return False

    def _apply_state(self, d):
        """A live_state-shaped dict -> the live state; each field validated as at a restart. Keys that are absent are
        left alone (OPEN SHOW applies only the parts it was asked for). Never raises."""
        try:
            if "base" in d:
                self.base.update(_clean_base(d.get("base"), {n: self.base[n] for n in CONT}))
                self._sm = dict(self.base)
                self._pal_target = self.base["palette"]
            if "mods" in d:
                self.mods = _clean_mods(d.get("mods"))
            cur = d.get("cur")
            if isinstance(cur, int) and not isinstance(cur, bool) and 0 <= cur < N_SCENES and self.scenes[cur]:
                self.cur = cur
            s = d.get("settings") if isinstance(d.get("settings"), dict) else {}
            for key, cmd in (("bpm", dict(cmd="bpm", v=s.get("bpm"))), ("clock_src", dict(cmd="clock_src", src=s.get("clock_src"))),
                             ("strobe_rate", dict(cmd="strobe_rate", v=s.get("strobe_rate"))),
                             ("morph", dict(cmd="morph", bars=s.get("morph"))),
                             ("quantize", dict(cmd="quantize", q=s.get("quantize"))), ("glide", dict(cmd="glide", bars=s.get("glide"))),
                             ("safe", dict(cmd="safe", i=s.get("safe"))), ("cue_ch", dict(cmd="cue_ch", ch=s.get("cue_ch"))),
                             ("autopilot", dict(s.get("autopilot") if isinstance(s.get("autopilot"), dict) else {}, cmd="autopilot"))):
                if key not in s:
                    continue
                try:
                    self._do(validate(cmd))
                except Exception:
                    pass                               # one bad field does not spoil the rest
            if "seq" in s:
                self.seq = _clean_seq(s.get("seq"))
            if isinstance(s.get("ext"), dict):         # other modules' settings (small JSON dicts, keyed by module)
                self.ext = {k: v for k, v in s["ext"].items() if isinstance(k, str) and isinstance(v, dict)}
            if self.text is not None and "text" in s:
                self.text.restore(s.get("text"))
            if isinstance(s.get("bypass"), list):      # come back already bypassed: no fade-out on the projector
                self.bypass = {k for k in s["bypass"] if isinstance(k, str) and k in BYPASS_COVER}
                self._byp = {k: 1.0 for k in self.bypass}
            if self.stack is not None and isinstance(s.get("stack"), dict):
                self.stack.restore(s["stack"])         # comes back as it was: no fade-in on the projector
            if self.skins is not None and isinstance(s.get("skins_on"), bool):
                self.skins.on = s["skins_on"]          # the feature switch (a show file never carries it)
            if self.skins is not None and isinstance(s.get("skins"), dict):
                self.skins.restore(s["skins"])
            cv = s.get("canvas")
            if self.stack is not None and isinstance(cv, dict) and isinstance(cv.get("base"), dict):
                self.canvas_prev = dict(base=_clean_base(cv["base"]), mods=_clean_mods(cv.get("mods")),
                                        bypass=sorted(k for k in (cv.get("bypass") if isinstance(cv.get("bypass"), list) else [])
                                                      if isinstance(k, str) and k in BYPASS_COVER),
                                        stack=_stack.clean_stack(cv.get("stack")), auto=bool(cv.get("auto")))
            if _clips is not None:                     # user clips + decks first: "comp" refers to them
                if "clips" in s:
                    self.clips = {x["id"]: x for x in _clips.clean_clips(s.get("clips"))}
                    self._clips_rev += 1
                if "decks" in s:
                    self.decks = _clips.clean_decks(s.get("decks"))
                    self._clips_rev += 1
                if "comp" in s:
                    self._comp_restore(s.get("comp"))
            if self.themes is not None and isinstance(s.get("themes"), dict):
                self.themes.restore(s["themes"])       # [themes] session tray (validated in themes.clean_tray)
            if self.trans is not None and isinstance(s.get("trans"), dict):   # [transitions] default + origin
                try:
                    self.trans.default = _trans.clean_spec(s["trans"].get("default"))
                except ValueError:
                    self.trans.default = None
                o = s["trans"].get("origin")
                if isinstance(o, list) and len(o) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in o):
                    self.trans.origin = [min(1.0, max(0.0, float(x))) for x in o]
            f = s.get("bypass_fade")
            if isinstance(f, (int, float)) and not isinstance(f, bool) and math.isfinite(f):
                self.bypass_fade = min(BYPASS_FADE_MAX, max(0.0, float(f)))
            if isinstance(s.get("latch"), dict):
                for n in TRIGGERS:
                    if isinstance(s["latch"].get(n), bool):
                        self.latch[n] = s["latch"][n]
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------- commands

    def apply(self, cmds):
        """Run the panel commands this class owns; return the rest untouched (corner pin, grid, next...).

        Also returns passthrough commands raised by MIDI (e.g. {"cmd": "next_video"}), so call it every
        frame even with an empty list. Malformed control commands are dropped, never raised.
        """
        rest = []
        try:
            for c in cmds or ():
                if not isinstance(c, dict) or c.get("cmd") not in CONTROL_CMDS:
                    rest.append(c)
                    continue
                try:
                    v = validate(c)
                    self._do(v)
                    if self.themes is not None:
                        self.themes.saw(v)             # [themes] a performer action holds sound-tag recall
                    if v["cmd"] not in _NOT_EDITS and not (v["cmd"] == "tl" and v["op"] in _TL_TRANSPORT):
                        self._changed()
                except Exception:
                    pass
        except TypeError:
            pass
        if self._out:
            rest += self._out
            self._out = []
        if self.themes is not None and rest:
            self.themes.note_media(rest)               # [themes] remember the AI texture the engine is sent
        return rest

    def _do(self, c):
        cmd = c["cmd"]
        if cmd == "set":
            self._hist_push(self._hkey(c, "set", c["name"]), (c["name"],))
            self.set(c["name"], c["v"])
        elif cmd == "inc":
            self._hist_push(self._hkey(c, "set", c["name"]), (c["name"],))
            self.nudge(c["name"], c["d"])
        elif cmd == "clip":
            self._clip_do(c)
        elif cmd == "deck":
            self._deck_do(c)
        elif cmd == "macro":
            self.macro(c["lane"], c["n"], c["v"], key=self._hkey(c, "macro", c["lane"], c["n"]))
        elif cmd == "scene_go":
            self.scene_go(c["id"], now=c.get("now", False) or self._tl_firing, bars=c.get("bars"))
        elif cmd == "history":
            (self.undo if c["op"] == "undo" else self.redo)()
        elif cmd == "mod":
            self.set_mod(c["name"], c["slot"], c["src"], c["amt"], c["rate"], c["phase"])
        elif cmd == "stack":                           # math stack edit (also what a "mod" on stack:<id>:<field> becomes)
            self._stack_morph = False
            self.stack.do(c)
            self._save_t = min(self._save_t, self.now() - AUTOSAVE_S + 0.5)   # a crash restart resumes the build
        elif cmd == "skin":                            # environment skins (also what a "mod" on skin:<env>:<slot>:<f> becomes)
            if self.skins is not None:
                self.skins.do(c)
                self._skin_ai_cancel()
                self._save_t = min(self._save_t, self.now() - AUTOSAVE_S + 0.5)
        elif cmd == "canvas":
            (self.new_canvas if c["op"] == "new" else self.restore_canvas)()
            self._save_t = min(self._save_t, self.now() - AUTOSAVE_S + 0.5)
        elif cmd == "tl":
            self._tl_do(c)
        elif cmd == "show":
            self._show_do(c)
        elif cmd == "preset":
            self._preset_do(c)
        elif cmd == "theme" and self.themes is not None:
            self.themes.do(c)
        elif cmd == "trigger":
            self.trigger(c["name"], c["on"])
        elif cmd == "latch":
            self.latch[c["name"]] = c["on"]
            self.held[c["name"]] = False
        elif cmd == "ripple":
            self.ripple(c["x"], c["y"])
        elif cmd == "tap":
            self.tap()
        elif cmd == "bpm":
            self.set_bpm(c["v"])
        elif cmd == "bpm_nudge":
            self.set_bpm(self.bpm + c["d"])
        elif cmd == "bpm_mult":
            if BPM_MIN <= self.bpm * c["f"] <= BPM_MAX:
                self.set_bpm(self.bpm * c["f"])
        elif cmd == "resync":
            self.resync()
        elif cmd == "clock_src":
            self.clock_src = c["src"]
        elif cmd == "strobe_rate":
            self.strobe_rate = c["v"]
        elif cmd == "scene_save":
            self.scene_save(c["i"])
        elif cmd == "scene_recall":
            self.scene_recall(c["i"], c.get("now", False), c.get("transition"))
        elif cmd == "transition" and self.trans is not None:   # [transitions]
            self._trans_do(c)
        elif cmd == "scene_clear":
            self.scene_clear(c["i"])
        elif cmd == "morph":
            self.morph_bars = c["bars"]
        elif cmd == "quantize":
            self.quantize = c["q"]
            q = QUANT[c["q"]]
            if self.pending and q:                     # a waiting recall moves to the new grid; it never fires early
                self.pending = (self.pending[0], (math.floor(self.beat / q + 1e-6) + 1) * q, q)
        elif cmd == "glide":
            self.glide_bars = c["bars"]
        elif cmd == "autopilot":
            was = self.auto["on"]
            self.auto.update({k: v for k, v in c.items() if k != "cmd"})
            if self.auto["on"] != was or "bars" in c:
                self._auto_next = None
            if not self.auto["drift"]:
                self._drift = {}
            for n in [n for n in self._drift if n not in self.auto["drift_params"]]:
                del self._drift[n]                     # a param taken off the list lets go of its offset
        elif cmd == "safe":
            self.safe = c["i"]
        elif cmd == "panic":
            self.panic()
        elif cmd == "init":
            self.base.update({n: PARAMS[n][2] for n in CONT})
            self.mods, self._morph, self.cur = {}, None, -1
            self._comp_clear()
            if self.stack is not None:
                self._stack_morph = False
                self.stack.load(dict(layers=[]))
        elif cmd == "cue_ch":
            self.cue_ch = c["ch"]
        elif cmd == "midi_learn":
            self.learn = c["target"]
        elif cmd == "midi_clear":
            self.midi.clear(c["target"])
        elif cmd == "midi_defaults":
            self.midi.reset_defaults()
        elif cmd == "seq":
            if "v" in c:
                self.seq["vals"][c["i"]] = c["v"]
            if "g" in c:
                self.seq["gates"][c["i"]] = int(c["g"])
        elif cmd == "seq_rand":
            self.seq = dict(vals=[round(self.rng.random(), 3) for _ in range(SEQ_STEPS)],
                            gates=[int(self.rng.random() > 0.45) for _ in range(SEQ_STEPS)])
        elif cmd == "bypass":
            self.set_bypass(c["key"], c.get("on"))
        elif cmd == "bypass_fade":
            self.bypass_fade = c["v"]
        elif self.text is not None and cmd in _textlayer.TEXT_CMDS:
            self.text.do(c)
        elif cmd == "xfade":
            self.set_xfade(c.get("a"), c.get("b"), c.get("x"))
            self._xf_cc, self._xf_live = None, False   # the panel moved it: a MIDI fader must pick it up again
        elif cmd == "gen":
            job = {k: v for k, v in c.items() if k != "cmd"}
            if _online is not None and _online.take_gen(self, job):
                pass                                   # Nano Banana engine (ONLINE on); falls back to self.ai itself
            elif job["init"] == "snapshot":            # the engine writes snapshots/current.png first
                self._out.append(passthrough("snapshot"))
                self.ai.submit(job, delay=0.3)
            else:
                self.ai.submit(job)
        elif cmd == "gen_cancel":
            self.ai.cancel()
            if _online is not None:
                _online.cancel_gen(self)
        elif cmd == "online":
            _online.do(self, c)
        elif cmd == "art_step":
            self.art_step(c["d"])
        elif cmd == "art_deleted":                     # piece i left the library: keep the same piece on screen
            cur = int(round(self.base.get("art_index", 0)))
            if "art_index" in PARAMS and c["i"] < cur:
                self.set("art_index", cur - 1)
        elif cmd == "clock_port":
            self.clock_port = c["port"]
            self._clock_lock = None
            self.mclock.reset()

    # ================================================================ TIMELINE (timeline.py)
    # Automation is a layer of its own (self._auto: param -> value, the target instead of the fader's base), so the
    # faders, scenes and morphs are never written by the timeline: RELEASE glides every param back to its fader.
    # Order in update(): clock -> pending recall -> autopilot -> morph -> TIMELINE -> slew + mods -> bypass -> triggers,
    # so slew, mods, bypass, the governor (frame values) and PANIC all still apply on top of it.

    def _changed(self):
        """Something the show file holds changed (autosave is debounced in _step_shows)."""
        if self.shows is not None:
            t = self.now()
            self._show_dirty_t = t
            if self._show_first_t is None:
                self._show_first_t = t
            self.shows.dirty = True

    def _tl_touch(self, name):
        """A performer move (set() from the panel, MIDI, cue channel). On an automated param while the timeline drives
        it: that lane is overridden (TOUCH: until TL_HOLD s after the last move, then it glides back; LATCH: until
        RE-ENABLE). While RECORD is armed and the transport runs: the move is written into the lane."""
        tl = self.tl
        if name not in TL_CTX.params:
            return
        if self.tl_engaged and name in tl.by_param:
            self._tl_ovr[name] = [self._tl_clock, None, 0.0, False]
        if not tl.recording():
            return
        pos = tl.pos
        g = self._rec_g.get(name)
        if g is None:
            lane = tl.by_param.get(name)                # compiled before any write: the old curve, kept for the anchors
            tr = tl.ensure_lane(name)
            if tr is None:
                return
            eps = _tl.EPS[tl.units]
            old = _tl.lane_value(lane, pos) if lane is not None else None
            if old is not None and pos - eps > 0:     # the curve before the gesture stays as it was
                tl.write_key(tr, pos - eps, old, pos - eps - 1e-9)
            g = self._rec_g[name] = dict(tr=tr, a=max(0.0, pos - eps), last=pos - eps, w=-1e18, old=lane)
            self._tl_ovr[name] = [self._tl_clock, None, 0.0, False]
        g["touch"] = self._tl_clock
        self._tl_rec_write(name, g, force=True)

    def _tl_rec_write(self, name, g, force=False):
        tl = self.tl
        pos = tl.pos
        if pos <= g["last"] or (not force and pos - g["w"] < _tl.REC_GAP[tl.units]):
            return
        if force and pos - g["w"] < _tl.REC_GAP[tl.units] and g["w"] > -1e17:
            for it in reversed(g["tr"]["items"]):     # thin: move the newest key of this gesture instead of adding one
                if it["t"] == g["w"]:
                    it["v"] = self.base[name]
                    self._rec_dirty = True
                    return
        tl.write_key(g["tr"], pos, self.base[name], g["last"])
        g["last"] = g["w"] = pos
        self._rec_dirty = True

    def _tl_gesture_end(self, name, g, pos):
        tl = self.tl
        if tl.ovr_mode == "latch" and pos > g["last"]:   # LATCH: the held value runs right up to the end of the take
            tl.write_key(g["tr"], pos, self.base[name], g["last"])
        if g["old"] is not None and g["old"].ts:         # after the take (touch release, punch-out, stop) the old
            eps = _tl.EPS[tl.units]                      # curve carries on exactly as it was
            t = min(tl.data["length"], pos + eps)
            if t > g["last"]:
                tl.write_key(g["tr"], t, _tl.lane_value(g["old"], t), max(g["last"], pos))
        tl.simplify_range(g["tr"], g["a"] + 1e-9, pos)
        self._rec_dirty = True

    def _tl_pass_start(self):
        """A record pass begins (RECORD armed and the transport running): one undo step for the whole pass."""
        self.tl.snapshot()
        self._rec_g, self._rec_trig, self._rec_ids = {}, {}, set()

    def _tl_pass_end(self, pos=None):
        tl = self.tl
        if tl is None:
            return
        pos = tl.pos if pos is None else pos
        for name, g in list(self._rec_g.items()):
            self._tl_gesture_end(name, g, pos)
            self._tl_ovr.pop(name, None)               # recorded = what the lane now holds: hand it to the lane
        for name, t0 in self._rec_trig.items():
            it = tl.add_item("trigger", {"t": t0, "len": max(pos - t0, _tl.REC_GAP[tl.units]), "name": name})
            if it:
                self._rec_ids.add(it["id"])
        self._rec_g, self._rec_trig = {}, {}
        if self._rec_dirty:
            tl.compile()
            self._rec_dirty = False
            self._changed()

    def _step_timeline(self, dt):
        tl = self.tl
        self._tl_clock += dt
        clock = self._tl_clock
        beat = self.beat
        if tl.units == "beats":                        # follow the show clock (tap / MIDI clock / Link) beat for beat
            prev, self._tl_beat = self._tl_beat, beat
            d = 0.0 if prev is None else beat - prev
            if not 0.0 <= d <= dt * BPM_MAX / 60.0 * 2.0 + 1e-9:
                d = dt * self.bpm / 60.0               # the clock jumped (DOWNBEAT, song position): run at tempo
        else:
            d = dt
        p0 = tl.pos
        rec = tl.recording()
        if tl.playing:
            self._tl_firing = True
            try:
                for tr, it in tl.advance(d):
                    self._tl_fire(tr, it)
            finally:
                self._tl_firing = False
        wrapped = tl.pos < p0
        if rec and (wrapped or not tl.recording()):    # loop wrap, punch-out, end: close the take where it ended
            lp, pu = tl.data["loop"], tl.data["punch"]
            end = lp["b"] if wrapped else (pu["b"] if pu["on"] and tl.pos >= pu["b"] else tl.pos)
            if wrapped and tl.rec_mode == "replace":
                tl.clear_range(("cue", "trigger"), p0, end, self._rec_ids)
            self._tl_pass_end(end)
            if wrapped and tl.recording():
                self._rec_ids = set()
        # trigger spans (declarative: a pure function of the position; only while the transport runs)
        th = self._tl_trig
        th.clear()
        if tl.playing and tl.trig:
            ids = set()
            for it in tl.active_triggers(tl.pos, self._tl_buf):
                i, n = it["id"], it["name"]
                ids.add(i)
                if i not in self._tl_span_ids:         # the span starts now
                    ok = True
                    if n in _tl.FLASHY:
                        ok = clock - self._tl_flash_t >= TL_FLASH_GAP and not (
                            n == "strobe" and self.bpm / 60.0 / max(self.strobe_rate, 1e-6) > TL_SAFE_HZ)
                        if ok:
                            self._tl_flash_t = clock
                    if not ok:
                        self._tl_blocked.add(i)
                        self.tl_limited += 1
                    elif n in self.env:
                        self.env[n] = 1.0
                if n == "strobe" and self.bpm / 60.0 / max(self.strobe_rate, 1e-6) > TL_SAFE_HZ and i not in self._tl_blocked:
                    self._tl_blocked.add(i)
                    self.tl_limited += 1
                if i not in self._tl_blocked:
                    th.add(n)
            self._tl_span_ids = ids
            self._tl_blocked &= ids
        elif self._tl_span_ids:
            self._tl_span_ids, self._tl_blocked = set(), set()
        # automation layer
        au = self._auto
        if self.tl_engaged:
            for lane in tl.lanes:
                n = lane.param
                v = None if lane.mute else _tl.lane_value(lane, tl.pos)
                if v is None:
                    if n in au and n in LUMA:
                        pv, b, step = au[n], self.base[n], TL_LUMA_RATE * (lane.hi - lane.lo) * dt
                        if abs(b - pv) > step:
                            au[n] = pv + (step if b > pv else -step)
                            continue
                    au.pop(n, None)
                    continue
                o = self._tl_ovr.get(n)
                if o is not None:
                    if not o[3] and (tl.ovr_mode == "latch" or clock - o[0] < self.tl_hold):
                        au.pop(n, None)                # the performer has it
                        continue
                    if o[1] is None:
                        o[1], o[2] = self.base[n], clock
                    f = (clock - o[2]) / TL_GLIDE
                    if f >= 1.0:
                        del self._tl_ovr[n]
                    else:
                        v = o[1] + (v - o[1]) * f * f * (3.0 - 2.0 * f)
                if n in LUMA:                          # photosensitive: no luminance swing faster than 0.2 s
                    pv = au[n] if n in au else self.base[n]
                    step = TL_LUMA_RATE * (lane.hi - lane.lo) * dt
                    v = pv + step if v > pv + step else (pv - step if v < pv - step else v)
                au[n] = v
            if len(au) > len(tl.lanes) or any(n not in tl.by_param for n in au):
                for n in [n for n in au if n not in tl.by_param]:
                    del au[n]
        if self._tl_fade:                              # RELEASE: glide from the automation value back to the fader
            for n, (v0, t0) in list(self._tl_fade.items()):
                f = (clock - t0) / TL_GLIDE
                if f >= 1.0 or self.tl_engaged:
                    del self._tl_fade[n]
                    if not self.tl_engaged:
                        au.pop(n, None)
                else:
                    au[n] = v0 + (self.base[n] - v0) * f * f * (3.0 - 2.0 * f)
        # recording bookkeeping
        if rec and tl.playing and not wrapped:
            for name, g in list(self._rec_g.items()):
                if tl.ovr_mode == "touch" and clock - g.get("touch", clock) >= self.tl_hold:
                    self._tl_gesture_end(name, g, tl.pos)
                    del self._rec_g[name]
                else:
                    self._tl_rec_write(name, g)        # held still: keeps writing (and replacing) as the playhead runs
            if tl.rec_mode == "replace" and tl.pos > p0:
                if tl.clear_range(("cue", "trigger"), p0, tl.pos, self._rec_ids):
                    self._rec_dirty = False             # clear_range compiled
        if self._rec_dirty and (not rec or clock - self._rec_compile_t >= 0.25):
            self._rec_compile_t = clock
            self._rec_dirty = False
            tl.compile()
            self._changed()

    def _tl_fire(self, tr, it):
        """A discrete item the playhead crossed (cue, event, text). Already validated when it was placed / loaded."""
        k = tr["kind"]
        if k == "cue":
            self._tl_last_cue = it["id"]
            if self._tl_clock - self._tl_cue_t < TL_FLASH_GAP:     # photosensitive: no rapid hard cuts
                it = dict(it, morph=max(it["morph"], TL_FLASH_GAP * self.bpm / 60.0 / 4.0))
            self._tl_cue_t = self._tl_clock
            if it["kind"] == "scene":
                if self.scenes[it["slot"]] is not None:
                    if self.trans is not None:             # [transitions] a cue's own transition only (old shows: MORPH)
                        try:
                            spec = self.trans.resolve(it["transition"]) if it.get("transition") is not None else None
                        except ValueError:
                            spec = None
                        self.trans.run(lambda: self._start_recall(it["slot"], it["morph"]), spec, "cue")
                    else:
                        self._start_recall(it["slot"], it["morph"])
                    if self.auto["on"]:
                        self._auto_next = self.beat + self.auto["bars"] * 4.0
            elif self.shows is not None:
                p = self.shows.cache.get(it["preset"])
                po = {"groups": it.get("groups"), "morph": it["morph"]}
                if it.get("transition") is not None:    # [transitions]
                    po["transition"] = it["transition"]
                if p is not None:
                    self._apply_preset(p, po)
                else:                                   # not prefetched (yet): load now, applied when it arrives
                    self.shows.submit("preset_load", ref=it["preset"], opts=po)
        elif k == "event":
            c = dict(it["cmd"])
            if c["cmd"] == "bypass":                       # photosensitive: a layer cannot blink faster than 2.5 Hz
                last = self._tl_byp_t.get(c["key"], -1e9)
                if self._tl_clock - last < TL_FLASH_GAP:
                    self.tl_limited += 1
                    return
                self._tl_byp_t[c["key"]] = self._tl_clock
            if c["cmd"] in CONTROL_CMDS:
                self._do(c)
            else:
                self._out.append(c)
        elif k == "text" and self.text is not None:
            try:
                self.text.do(_VALIDATORS["text"]({"cmd": "text", "text": it["text"]}))
            except Exception:
                pass

    def _tl_chase(self):
        """Play / seek: bring back the state the timeline would have at this position (latest cue with a short
        crossfade, latest text, latest bypass per key, latest media pick). Flashes and one-shots are never chased."""
        tl = self.tl
        if not tl.data["chase"]:
            return
        self._tl_firing = True
        try:
            for tr, it in tl.chase_items(tl.pos, inclusive=not tl.playing):
                if tr["kind"] == "cue":
                    if it["id"] == self._tl_last_cue:
                        continue
                    it = dict(it, morph=min(it["morph"], TL_CHASE_S * self.bpm / 60.0 / 4.0))
                self._tl_fire(tr, it)
        finally:
            self._tl_firing = False

    def _tl_home(self):
        lp = self.tl.data["loop"]
        return lp["a"] if lp["on"] else 0.0

    def _tl_release(self):
        """Hand every automated param back to its fader (gliding over TL_GLIDE s)."""
        for n, v in self._auto.items():
            self._tl_fade[n] = [v, self._tl_clock]
        self.tl_engaged = False
        self._tl_ovr.clear()

    def _tl_docs_all(self):
        docs = list(self.tl_docs)
        docs[self.tl_i] = self.tl.to_json()
        return docs

    def _tl_load_docs(self, docs, active=0):
        docs = [d for d in docs if isinstance(d, dict)][:_tl.MAX_TIMELINES] or [_tl.empty()]
        active = min(max(0, int(active)), len(docs) - 1)
        self._tl_pass_end()
        self.tl.playing = self.tl.rec = False
        self.tl.pos = 0.0
        self.tl.load(docs[active])
        docs[active] = None
        self.tl_docs, self.tl_i = docs, active
        self.tl_engaged = False
        self._tl_release()
        self._tl_last_cue = None
        self._tl_prefetch()

    def _tl_prefetch(self):
        if self.shows is not None and self.shows.enabled:
            refs = self.shows._preset_cues(self.tl.data)
            if any(r not in self.shows.cache for r in refs):
                self.shows.submit("prefetch", refs=refs)

    def _tl_do(self, c):
        tl, op = self.tl, c["op"]
        if op == "play":
            if "t" in c:
                self._tl_pass_end()
                tl.seek(c["t"])
            if tl.pos >= tl.data["length"]:
                tl.seek(self._tl_home())
            if not tl.playing:
                tl.playing, self.tl_engaged, self._tl_beat = True, True, None
                self._tl_last_cue = None
                self._tl_chase()
                if tl.rec:
                    self._tl_pass_start()
        elif op == "stop":                             # first STOP pauses, the second returns to the start
            if tl.playing:
                self._tl_pass_end()
                tl.playing = False
            else:
                tl.seek(self._tl_home())
                self._tl_last_cue = None
        elif op == "home":
            self._tl_pass_end()
            tl.seek(self._tl_home())
            self._tl_last_cue = None
            if tl.playing:
                self._tl_chase()
                if tl.rec:
                    self._tl_pass_start()
        elif op == "seek":
            self._tl_pass_end()
            tl.seek(c["t"])
            self.tl_engaged = True
            self._tl_fade.clear()
            self._tl_chase()
            if tl.playing and tl.rec:
                self._tl_pass_start()
        elif op == "rec":
            if c["on"] and not tl.rec:
                tl.rec = True
                if tl.playing:
                    self._tl_pass_start()
            elif not c["on"] and tl.rec:
                self._tl_pass_end()
                tl.rec = False
        elif op == "rec_mode":
            tl.rec_mode = c["mode"]
        elif op == "ovr_mode":
            tl.ovr_mode = c["mode"]
        elif op == "reenable":                         # every overridden lane glides back to its automation
            for o in self._tl_ovr.values():
                o[3] = True
        elif op == "release":
            self._tl_release()
        elif op == "hold":
            self.tl_hold = c["s"]
        elif op == "units":
            self._tl_pass_end()
            if tl.convert_units(c["units"], self.bpm):
                self._tl_beat = None
        elif op in ("tl_new", "tl_dup", "tl_select", "tl_delete"):
            docs = self._tl_docs_all()
            i = self.tl_i
            if op == "tl_new" and len(docs) < _tl.MAX_TIMELINES:
                d = _tl.empty(tl.units)
                d["name"] = f"Timeline {len(docs) + 1}"
                docs.insert(i + 1, d)
                i += 1
            elif op == "tl_dup" and len(docs) < _tl.MAX_TIMELINES:
                d = json.loads(json.dumps(docs[i]))
                d["name"] = (d.get("name") or "Timeline")[:34] + " copy"
                docs.insert(i + 1, d)
                i += 1
            elif op == "tl_select" and c["i"] < len(docs):
                if c["i"] == i:
                    return
                i = c["i"]
            elif op == "tl_delete" and len(docs) > 1 and c["i"] < len(docs):
                del docs[c["i"]]
                i = min(i if c["i"] > i else max(0, i - 1), len(docs) - 1)
            self._tl_load_docs(docs, i)
        else:
            self._tl_pass_end()
            if tl.do(c):
                self._tl_prefetch()

    def _tl_state(self):
        tl = self.tl
        r = lambda x: round(x, 4)
        d = tl.data
        out = dict(playing=tl.playing, rec=tl.rec, pos=r(tl.pos), units=d["units"], length=d["length"], rev=tl.rev,
                   engaged=self.tl_engaged, rec_mode=tl.rec_mode, ovr_mode=tl.ovr_mode, name=d.get("name", ""),
                   next=[[r(t), lab] for t, lab in tl.upcoming(3)], active=self.tl_i)
        # only what is not at its default (a phone polls this once a second)
        for k in ("loop", "punch"):
            if d[k]["on"]:
                out[k] = dict(d[k])
        opt = dict(recording=tl.recording(), auto=sorted(self._auto), overrides=sorted(self._tl_ovr),
                   held=sorted(self._tl_trig), limited=self.tl_limited, section=tl.section(), audio=d["audio"],
                   chase=not d["chase"] and "off", hold=self.tl_hold != TL_HOLD and self.tl_hold)
        out.update({k: v for k, v in opt.items() if v})
        if len(self.tl_docs) > 1:
            out["timelines"] = [x.get("name", "") if x is not None else d.get("name", "") for x in self.tl_docs]
        return out

    def _show_state(self):
        if self.shows is None:
            return None
        st = self.shows.state()
        bc = getattr(self, "blob_cache", None) or {}
        for k in ("shows", "presets"):                 # the rev the panel sees always matches what GET /<k> serves
            if k in bc and isinstance(st.get("rev"), dict):
                st["rev"][k] = bc[k][0]
        return st

    def _blobs(self):
        """Big documents the panel fetches only when their rev changes: built here at most every 0.5 s while a
        record pass runs (json of the timeline), the listings on the shows worker."""
        out = {}
        if self.tl is not None:
            rev = self.tl.rev
            if rev != self._tl_blob[0] and (not self.tl.recording() or self._tl_clock - self._tl_blob_t >= 0.5):
                self._tl_blob_t = self._tl_clock
                try:
                    self._tl_blob = (rev, json.dumps(self.tl.data, allow_nan=False, separators=(",", ":")).encode())
                except (TypeError, ValueError):
                    pass
            out["timeline"] = self._tl_blob
        if self.shows is not None:
            out.update(self.shows.blobs)
        if _clips is not None:                         # GET /clips: factory + user clips + decks (rev in state.comp)
            if self._clips_blob[0] != self._clips_rev:
                try:
                    self._clips_blob = (self._clips_rev, json.dumps(dict(
                        factory=_clips.FACTORY, user=list(self.clips.values()), decks=self.decks),
                        allow_nan=False, separators=(",", ":")).encode())
                except (TypeError, ValueError):
                    pass
            out["clips"] = self._clips_blob
        if self.themes is not None:                    # [themes] GET /themes: the tray, log, suggestions
            out["themes"] = self.themes.blob()
        return out

    # ================================================================ SHOWS + PRESETS (shows.py)

    def _show_snapshot(self):
        """Cheap copies of everything the controller owns, for the worker to serialize (GL thread, ~1 ms)."""
        docs = []
        if self.tl is not None:                        # the inactive timelines are never mutated: no copy needed
            docs = list(self.tl_docs)
            rev, data = self._tl_blob
            docs[self.tl_i] = json.loads(data) if rev == self.tl.rev and data != b"{}" else self.tl.to_json()
        st = copy.deepcopy(self._settings())
        st["ext"] = {k: v for k, v in st.get("ext", {}).items() if k != "online"}
        return dict(controller=dict(base={n: self.base[n] for n in CONT}, mods=_copy_mods(self.mods), cur=self.cur,
                                    scenes=copy.deepcopy(self.scenes), settings=st),
                    midi=[dict(b) for b in self.midi.bindings], timeline=docs[self.tl_i] if docs else None,
                    timelines=docs, timeline_active=self.tl_i)

    def _show_autosave(self):
        m = self.shows
        self._show_dirty_t = self._show_first_t = None
        m.submit("autosave", snap=self._show_snapshot(), slug=m.current, name=m.current_name)

    def _step_shows(self, now):
        m = self.shows
        if not m.enabled:
            return
        if m._thread is None:
            m.run_sync()
        for kind, payload in m.poll():
            if kind == "saved":
                m.current, m.current_name = payload["slug"], payload["name"]
                if self._show_dirty_t is None or self._show_dirty_t <= self._save_req_t:
                    m.dirty = False
                    self._show_dirty_t = self._show_first_t = None
            elif kind == "opened":
                self._apply_show(payload)
            elif kind == "session":
                if payload.get("timelines") and self.tl is not None:
                    self._tl_load_docs(payload["timelines"], payload.get("timeline_active", 0))
                m.current, m.current_name = payload["slug"], payload["name"]
                m.thumbs = {i: 1 for i in payload.get("thumbs", [])}
            elif kind == "preset":
                ref, p, opts = payload
                self._apply_preset(p, opts)
            elif kind == "thumb":
                m.rev["thumbs"] += 1
                m.thumbs[payload] = m.rev["thumbs"]
            elif kind == "thumb_gone":
                m.thumbs.pop(payload, None)
            elif kind == "theme_thumb" and self.themes is not None:     # [themes]
                self.themes.thumb_done(payload)
            elif kind == "theme_starred" and self.themes is not None:
                self.themes.starred(*payload)
        t = self._show_dirty_t
        if t is not None and (now - t >= SHOW_DEBOUNCE or now - self._show_first_t >= SHOW_MAX_WAIT):
            if not (self.tl is not None and self.tl.recording()) or now - self._show_first_t >= 2 * SHOW_MAX_WAIT:
                self._show_autosave()

    def _show_do(self, c):
        m, op = self.shows, c["op"]
        if m is None:
            return
        self._save_req_t = self.now()
        if op == "save":
            if m.current is None:
                op, c = "save_as", dict(c, name=m.current_name, template=False)
            else:
                m.submit("save", snap=self._show_snapshot(), name=m.current_name, slug=m.current, media=c["media"])
                return
        if op == "save_as":
            m.submit("save", snap=self._show_snapshot(), name=c["name"], slug=None, media=c["media"],
                     template=c["template"])
        elif op == "open":
            m.submit("open", slug=c["slug"], recover=c["recover"], backup=c.get("backup"),
                     parts=c.get("parts") or list(SHOW_DEFAULT_PARTS))
        elif op == "new":
            self._new_show(c["from"], c.get("slug"), c.get("parts"))
        elif op == "delete":
            if c["slug"] == m.current:
                m.current, m.current_name = None, "Untitled"
                m.submit("session_clear")
            m.submit("delete_show", slug=c["slug"])
        elif op == "rescan":
            m.submit("rescan")

    def _new_show(self, src, slug=None, parts=None):
        m = self.shows
        if src == "template":
            m.submit("open", slug=slug, parts=parts or [p for p in _shows.PARTS if p not in ("midi", "mapping")],
                     template=True)
            return
        if self.tl is not None:
            self._tl_load_docs([_tl.empty(self.tl.units)])
        if src == "factory":
            self._do(dict(cmd="init"))
            self.bypass, self._byp = set(), {}
        elif src == "clean":
            self.new_canvas()
        if src in ("factory", "clean"):
            self.scenes, self._scenes_ok = factory_scenes(), True
            self.cur = -1
            self._save_scenes()
            for i in list(m.thumbs):
                m.submit("scene_thumb_clear", slot=i)
        m.current, m.current_name, m.missing = None, "Untitled", []
        m.submit("session_clear")
        self._changed()
        m.say("ok", "new show" + {"clean": " on a clean canvas", "factory": " from the factory looks",
                                  "current": " from what is on now"}.get(src, ""))

    def _morph_base(self, to, names, bars):
        """Glide the fader bases of `names` to `to` over `bars` (0 = snap), like a scene recall that is no scene."""
        if bars <= 0:
            for n in names:
                self.base[n] = to[n]
            if self._morph:
                self._morph["names"] -= set(names)
            return
        tgt = dict(self.base)
        tgt.update({n: to[n] for n in names})
        self._morph = dict(src={n: self.base[n] for n in CONT}, to=tgt, names=set(names), start=self.beat,
                           dur=bars * 4.0, old=[(self.mods, 1.0)] if self.mods else [], t=0.0)

    def _apply_show(self, plan):
        """An opened show (cleaned on the worker) -> live. Only the asked-for parts; the rig (mapping, MIDI) only when
        asked. Commands for the engine (surfaces, recorder, camera, governor) go out validated, like the panel's."""
        m, parts = self.shows, set(plan.get("parts") or SHOW_DEFAULT_PARTS)
        if self.trans is not None:                     # [transitions] an opened show replaces whatever was blending
            self.trans.cancel(revert=False)
        bars = self.morph_bars                         # the performer's crossfade now, not the one the show brings
        self.pending = None
        if "timeline" in parts and self.tl is not None and plan.get("timelines"):
            self._tl_load_docs(plan["timelines"], plan.get("timeline_active", 0))
        st = dict(plan.get("settings") or {})
        look_keys = ("stack", "bypass", "canvas", "skins", "comp")
        scene_keys = ("clips", "decks", "themes")      # the clip library + decks (+ the theme tray) travel with the scenes
        if "look" not in parts:
            st = {k: v for k, v in st.items() if k not in look_keys}
        if "scenes" not in parts:
            st = {k: v for k, v in st.items() if k not in scene_keys}
        if "settings" not in parts:
            st = {k: v for k, v in st.items() if k in look_keys + scene_keys}
        ext = st.pop("ext", None)
        if st:
            self._apply_state({"settings": st})
        if "settings" in parts and isinstance(ext, dict):
            for k, v in ext.items():
                self.ext[k] = v
            if "camera" in ext:
                try:
                    import camera_layer
                    self._out.append(camera_layer.validate_command(dict(ext["camera"], cmd="camera", op="set")))
                except Exception:
                    pass
            if "governor" in ext:
                try:
                    import governor
                    self._out.append(governor.validate_command({"cmd": "gov", "op": "mode", "mode": ext["governor"]["mode"]}))
                except Exception:
                    pass
        if "look" in parts and plan.get("look"):
            lk = plan["look"]
            old_mods = self.mods
            self._morph_base(lk["base"], CONT, bars)
            if self._morph:
                self._morph["old"] = [(old_mods, 1.0)] if old_mods else []
            self.mods = _copy_mods(lk["mods"])
            self.cur = lk["cur"]
        if "scenes" in parts and plan.get("scenes"):
            self.scenes, self._scenes_ok = plan["scenes"], True
            if not 0 <= self.cur < N_SCENES or self.scenes[self.cur] is None:
                self.cur = -1
            self._save_scenes()
            m.rev["thumbs"] += 1
            m.thumbs = {i: m.rev["thumbs"] for i in plan.get("thumbs", [])}
        if "midi" in parts and plan.get("midi") is not None:
            self.midi.bindings = [dict(b) for b in plan["midi"]]
            self.midi.save()
        if "mapping" in parts and plan.get("mapping"):
            self._out.append(dict(plan["mapping"]))
        if "recorder" in parts and plan.get("recorder"):
            self._out.append(dict(plan["recorder"]))
        if plan.get("template"):
            m.current, m.current_name = None, plan["name"] + " (new)"
            self._changed()
        elif not plan.get("switch", True):
            self._changed()                            # parts of another show merged into this one
        else:
            m.current, m.current_name, m.dirty = plan["slug"], plan["name"], False
            self._show_dirty_t = self._show_first_t = None
        m.missing = list(plan.get("missing") or [])
        m.say("warn" if m.missing else "ok", f"opened '{plan['name']}'" + (
            f": {len(m.missing)} media file(s) missing" if m.missing else ""))

    def _preset_do(self, c):
        m, op = self.shows, c["op"]
        if m is None:
            return
        if op == "save":
            p = self._preset_data(c)
            if p is None:
                m.say("error", "nothing to save for that preset")
                return
            snap_t = None
            if self.thumbs and c["kind"] in ("look", "layer", "fx", "mapping"):
                self._out.append(passthrough("snapshot"))  # thumbnail: one low-res frame of the output
                snap_t = time.time()
            m.submit("preset_save", preset=p, snap_t=snap_t)
        elif op == "recall":
            opts = {"groups": c.get("groups"), "morph": c["morph"], "ref": c["ref"]}
            if self.trans is not None:                 # [transitions] the command's own, else the default (manual)
                opts["transition"] = c["transition"] if c.get("transition") is not None else (self.trans.default or "morph")
            p = m.cache.get(c["ref"])
            if p is not None:
                self._apply_preset(p, opts)
            else:
                m.submit("preset_load", ref=c["ref"], opts=opts)
        elif op == "set":
            m.submit("preset_set", ref=c["ref"], name=c.get("name"), tags=c.get("tags"), fav=c.get("fav"))
        elif op == "delete":
            m.submit("preset_delete", ref=c["ref"])
        elif op == "rescan":
            m.submit("rescan", what=("presets",))

    def _preset_data(self, c):
        """The preset as JSON-able data (copies), cleaned again on the worker (shows.clean_preset)."""
        kind, g = c["kind"], c.get("groups")
        keep = (lambda n: PARAMS[n][3] in g) if g else (lambda n: True)
        data = {}
        if kind == "look":
            names = [n for n in CONT if n != "master" and keep(n)]
            data["values"] = {n: self.base[n] for n in names}
            data["mods"] = {n: p for n, p in _copy_mods(self.mods).items() if n in data["values"]}
            if self.stack is not None and (not g or "layers" in g):
                data["stack"] = self.stack.to_json()
            if self.skins is not None and self.skins.any() and (not g or "layers" in g):
                data["skins"] = self.skins.to_json()   # "skin preset": a look preset carries the skins it was saved with
        elif kind == "layer":
            if c.get("stack_id") is not None:
                lay = self.stack.find(c["stack_id"]) if self.stack is not None else None
                if lay is None:
                    return None
                data["stack_layer"] = copy.deepcopy(lay)
                data["values"] = {}
            else:
                names = _shows.layer_params(c["layer"])
                data["layer"] = c["layer"]
                data["values"] = {n: self.base[n] for n in names}
            data["mods"] = {n: p for n, p in _copy_mods(self.mods).items() if n in data["values"]}
        elif kind == "fx":
            data["values"] = {n: self.base[n] for n in _shows.fx_params() if keep(n)}
            data["mods"] = {n: p for n, p in _copy_mods(self.mods).items() if n in data["values"]}
        elif kind == "mod":
            data["mods"] = {n: p for n, p in _copy_mods(self.mods).items() if keep(n)}
        elif kind == "mapping":
            data["surfaces"] = None                    # read from surfaces.json on the worker
        elif kind == "midi":
            data["bindings"] = [dict(b) for b in self.midi.bindings]
        return {"format": _shows.PRESET_FORMAT, "version": _shows.PRESET_VERSION, "kind": kind, "name": c["name"],
                "tags": c.get("tags") or [], "fav": False, "created": time.time(), "data": data}

    def _apply_preset(self, p, opts):
        """A clean preset -> live. groups = partial recall (only those param groups); morph in bars (0 = snap)."""
        if self.trans is not None and opts.get("transition") is not None and p.get("kind") in ("look", "layer", "fx"):
            try:                                       # [transitions] BLEND the preset in (or queue / MORPH)
                spec = self.trans.resolve(opts["transition"])
            except ValueError:
                spec = None
            inner = dict(opts)
            inner.pop("transition")
            self.trans.run(lambda: self._apply_preset(p, inner), spec, "preset")
            return
        kind, d = p["kind"], p["data"]
        groups = opts.get("groups")
        keep = (lambda n: PARAMS[n][3] in groups) if groups else (lambda n: True)
        bars = float(opts.get("morph") or 0.0)
        if kind in ("look", "layer", "fx"):
            vals = {n: v for n, v in d.get("values", {}).items() if n in CONT and keep(n)}
            if vals:
                old_mods = _copy_mods(self.mods)
                self._morph_base(vals, list(vals), bars)
                if self._morph:
                    self._morph["old"] = [(old_mods, 1.0)] if old_mods else []
                for n in vals:                         # the preset's modulation replaces what those params had
                    pair = d.get("mods", {}).get(n)
                    if pair:
                        self.mods[n] = [dict(x) if x else None for x in pair]
                    else:
                        self.mods.pop(n, None)
                self.cur = -1
            if self.stack is not None and kind == "look" and "stack" in d and (not groups or "layers" in groups):
                self._stack_morph = False
                self.stack.morph_cancel()
                if bars > 0 and self._morph:
                    self.stack.morph_start(d["stack"])
                    self._stack_morph = True
                else:
                    self.stack.load(d["stack"])
            if self.skins is not None and kind == "look" and "skins" in d and (not groups or "layers" in groups):
                self.skins.load(d["skins"])
            if self.stack is not None and kind == "layer" and "stack_layer" in d:
                lay = d["stack_layer"]
                self.stack.add(lay.get("gen", 0), lay)
        elif kind == "mod":
            mods = {n: pr for n, pr in d.get("mods", {}).items() if n in CONT and keep(n)}
            if not groups:
                self.mods = {}
            else:
                for n in [n for n in self.mods if keep(n)]:
                    self.mods.pop(n)
            for n, pair in mods.items():
                self.mods[n] = [dict(x) if x else None for x in pair]
        elif kind == "mapping":
            m = _shows.clean_surfaces(d.get("surfaces"))
            if m is not None:
                self._out.append(m)
        elif kind == "midi":
            self.midi.bindings = [dict(b) for b in d.get("bindings", [])]
            self.midi.save()
        elif kind == "theme" and self.themes is not None:   # [themes] a starred theme from the library / a cue
            self.themes.restore_preset(p, opts)
        self._changed()

    # ---------------------------------------------------------------- MIDI (drained inside update)

    def _midi_in(self, now):
        for port, kind, ch, num, val, t in self.midi.poll():
            try:
                if kind in ("clock", "start", "stop", "continue", "songpos"):
                    if self._clock_from(port, kind, t):
                        self.mclock.feed(kind, val, t)
                    continue
                self.midi_last = f"{'cc' if kind == 'cc' else 'note'} {num} ch{ch + 1} = {val}"
                typ = "cc" if kind == "cc" else "note"
                if self.learn is not None:
                    if kind != "note_off":
                        self.midi.bind(typ, ch, num, self.learn, port)
                        self.learn = None
                    continue
                if self.cue_ch and ch == self.cue_ch - 1:
                    self._cue(kind, num, val)
                    continue
                for b in self.midi.lookup(port, typ, ch, num):
                    self._act(b, kind, val)
            except Exception:
                pass                                   # a bad message never reaches the render loop

    def _clock_from(self, port, kind, t):
        """True if clock messages from `port` drive the beat: the chosen port, or (auto) the first one heard,
        re-chosen only after that one has been silent for a second. Two clock sources never get mixed."""
        if kind == "clock":
            self._clock_seen[port] = t
        if self.clock_port is not None:
            want = self.clock_port
        else:
            last = self.mclock.last_t
            if self._clock_lock is None or (kind == "clock" and (last is None or t - last > 1.0)):
                if self._clock_lock != port:
                    self._clock_lock = port
                    self.mclock.reset()
            want = self._clock_lock
        return port == want

    def _cue(self, kind, num, val):
        """Cue channel: note N -> scene N (quantized + morphed per settings), CC -> CUE_CC param."""
        if kind == "cc":
            name = CUE_CC.get(num)
            if name:
                lo, hi = PARAMS[name][0], PARAMS[name][1]
                self.set(name, lo + (hi - lo) * val / 127.0)
            return
        if CUE_TRIGGER_NOTE <= num < CUE_TRIGGER_NOTE + len(TRIGGERS):
            self.trigger(TRIGGERS[num - CUE_TRIGGER_NOTE], kind == "note_on")
        elif kind == "note_on":
            if num == CUE_PANIC_NOTE:
                self.panic()
            for b in CUE_NOTE_BASES:
                if b <= num < b + N_SCENES:
                    self.scene_recall(num - b)

    def _cc_rise(self, b, val):
        """Shared CC press helper: True only when this binding's CC goes from < 64 to >= 64. A fader or knob swept
        through the top half (or a pad re-sending 127) fires a toggle once, not on every message."""
        key = (b.get("type"), b.get("ch"), b.get("num"), b.get("port"), b.get("target"))
        hi, was = val >= 64, self._cc_hi.get(key, False)
        self._cc_hi[key] = hi
        return hi and not was

    def _act(self, b, kind, val):
        target = b["target"]
        p = target.split(":")
        if self.themes is not None:
            self.themes.manual()                       # [themes] any bound MIDI control is a performer action
        press = kind == "note_on" or (kind == "cc" and val >= 64)
        edge = self._cc_rise(b, val) if kind == "cc" else press   # one-shot actions: a CC fires once per crossing of 64
        if p[0] == "param":
            name = p[1]
            lo, hi = PARAMS[name][0], PARAMS[name][1]
            if kind in ("cc", "note_on"):
                self._hist_push(("auto", "midi", name), (name,))
            if kind == "cc":
                if b.get("mode") == "rel":             # relative encoder, two's complement: 1 = +1, 127 = -1
                    self.nudge(name, (val if val < 64 else val - 128) * (hi - lo) / 127.0)
                else:
                    self.set(name, lo + (hi - lo) * val / 127.0)
            elif kind == "note_on":                    # a pad on a fader target toggles min / max
                self.set(name, lo if self.base[name] > (lo + hi) / 2 else hi)
        elif p[0] == "trigger":
            if kind == "cc" or kind in ("note_on", "note_off"):
                self.trigger(p[1], press)
        elif target == "shift":
            self._shift = press
        elif target == "bpm":
            if kind == "cc":
                self.set_bpm(60.0 + 120.0 * val / 127.0)
        elif p[0] == "bypass":
            if edge:                                   # a pad kills / restores a layer or a tab
                self.set_bypass(p[1])
        elif p[0] == "stack":                          # a knob on a math-stack field, a pad on mute / solo
            t = _stack.parse_target(target) if self.stack is not None else None
            if t and (kind == "cc" or kind == "note_on"):
                self._stack_morph = False
                if kind == "cc" and t[1] not in ("mute", "solo"):
                    rel = (val if val < 64 else val - 128) if b.get("mode") == "rel" else None
                    self.stack.midi(t[0], t[1], val / 127.0, rel=rel)
                else:
                    self.stack.midi(t[0], t[1], 1.0 if press else 0.0, press=edge)
        elif p[0] == "skin":                           # a knob on a skin slot's field (a pad steps a stepped field)
            t = _skins.parse_target(target) if self.skins is not None else None
            if t and kind == "cc":
                rel = (val if val < 64 else val - 128) if b.get("mode") == "rel" else None
                self.skins.midi(t[0], t[1], t[2], val / 127.0, rel=rel)
            elif t and kind == "note_on" and t[2] in _skins.INT:
                self.skins.midi(t[0], t[1], t[2], 1.0, pad=True, press=edge)
        elif p[0] == "macro":                          # macro:<lane>:<n>: whatever clip is on air in that lane
            lane, n = p[1], int(p[2])
            on = self.comp.get(lane, {})
            if kind == "cc":
                x = val / 127.0
                if b.get("mode") == "rel" and on.get("macros") and n <= len(on["macros"]):
                    x = on["macros"][n - 1] + (val if val < 64 else val - 128) / 127.0
                self.macro(lane, n, x, key=("auto", "macro", lane, n))
            elif kind in ("note_on", "note_off") and on.get("clip"):
                cl = _clips.get(on["clip"], self.clips)
                m = cl["macros"][n - 1] if cl and n <= len(cl["macros"]) else None
                if m is not None and m["widget"] == "bang":          # momentary: held = 1
                    self.macro(lane, n, 1.0 if press else 0.0, key=("auto", "macro", lane, n))
                elif m is not None and kind == "note_on":            # a pad toggles the macro min / max
                    self.macro(lane, n, 0.0 if on["macros"][n - 1] > 0.5 else 1.0, key=None)
        elif p[0] == "transition" and p[1] == "len":    # [transitions] a knob: the default's length (continuous)
            if kind == "cc" and self.trans is not None:
                self._trans_len(val / 127.0)
        elif target == "xfade":
            if kind == "cc":                           # soft takeover: a physical fader that is somewhere else does
                v, last, cur = val / 127.0, self._xf_cc, self.xfade["x"]     # nothing until it reaches the position
                self._xf_cc = v
                if self._xf_live or abs(v - cur) <= 0.05 or (last is not None and (last - cur) * (v - cur) <= 0):
                    self._xf_live = True
                    self.set_xfade(x=v)
        elif not edge:
            return                                     # everything below fires on press only (CC: rising edge)
        elif p[0] == "scene":
            (self.scene_save if self._shift else self.scene_recall)(int(p[1]))
        elif p[0] == "clip":
            self.clip_launch(p[1])
        elif p[0] == "theme" and self.themes is not None:   # [themes] restore; Shift + pad = capture the look there
            self.themes.midi(int(p[1]), shift=self._shift)
        elif p[0] == "transition" and self.trans is not None:   # [transitions] a pad: pick the default / cancel
            if p[1] == "cancel":
                self.trans.cancel()
            else:
                self._trans_do({"cmd": "transition", "op": "default", "spec": None if p[1] == "morph" else p[1]})
        elif p[0] == "palette":
            self.set("palette", float(p[1]))
        elif p[0] == "morph":
            self.morph_bars = float(p[1])
        elif p[0] == "quantize":
            self.quantize = p[1]
        elif p[0] == "ripple":
            self.ripple(float(p[1]), float(p[2]))
        elif target == "tap":
            self.tap()
        elif target == "autopilot":
            self._do(dict(cmd="autopilot", on=not self.auto["on"]))
        elif target == "drift":
            self._do(dict(cmd="autopilot", drift=not self.auto["drift"]))
        elif target == "resync":
            self.resync()
        elif target == "panic":
            self.panic()
        elif target in ("next_video", "next"):
            self._out.append(passthrough(target))
        elif target == "rec":                          # show recorder start / stop (recorder.py)
            self._out.append(passthrough("rec", op="toggle"))
        elif target in ("art_prev", "art_next"):
            self.art_step(-1 if target == "art_prev" else 1)

    def _leds(self):
        """Paint the APC mini mk2 pads: scenes (blue = saved, green = live, yellow = queued), held triggers."""
        want = {}
        pend = self.pending[0] if self.pending else -1
        for i in range(N_SCENES):
            note = 56 + i if i < 8 else 40 + i
            want[note] = 21 if i == self.cur else 13 if i == pend else 45 if self.scenes[i] else 0
        for i, n in enumerate(("flash", "punch", "strobe", "invert", "freeze", "blackout")):
            want[i] = 5 if self.held[n] else 1
        want[6] = 3 if self.beat % 1.0 < 0.25 else 0
        want[7] = 21 if self.auto["on"] else 1
        self.midi.leds(want)

    # ---------------------------------------------------------------- clock

    def _step_clock(self, dt, now):
        src, mc = self.clock_src, self.mclock
        if src == "auto":
            src = "midi" if mc.active(now) else "internal"
        self.linkb.request(src == "link", self.bpm)    # created once on its own thread, then only toggled
        link = self.linkb.read() if src == "link" else None
        if src == "midi" and mc.active(now):
            self.bpm = mc.bpm
            adv = dt * self.bpm / 60.0
            if mc.running:
                target = mc.ticks / 24.0 + min(max(now - mc.last_t, 0.0), 0.05) * self.bpm / 60.0
                if mc.jump or abs(target - self.beat) > 1.0:
                    self.beat, mc.jump = target, False
                else:                                  # phase-locked: run at tempo, pull gently onto the ticks
                    self.beat += adv
                    self.beat += (target - self.beat) * min(1.0, dt * 8.0)
            else:
                self.beat += adv                       # transport stopped: keep the visuals moving
        elif link is not None:
            self.bpm, self.beat = link
        else:
            src = "internal"                           # also the fallback while MIDI clock / Link is not there
            self.beat += dt * self.bpm / 60.0
        if not math.isfinite(self.beat):
            self.beat = 0.0
        self.clock_active = src

    # ---------------------------------------------------------------- autopilot

    def _step_autopilot(self, dt):
        a = self.auto
        if a["on"]:
            period = a["bars"] * 4.0
            if self._auto_next is None or self._auto_next - self.beat > period + 1e-6:
                self._auto_next = (math.floor(self.beat / period + 1e-6) + 1) * period
            if self.beat >= self._auto_next:
                while self._auto_next <= self.beat:
                    self._auto_next += period
                filled = [i for i, s in enumerate(self.scenes) if s is not None and i != self.cur]
                if filled and not self.pending:        # a queued manual recall wins; autopilot skips this turn
                    if a["mode"] == "next":
                        i = min((j for j in filled if j > self.cur), default=filled[0])
                    else:
                        i = self.rng.choice(filled)
                    keep = a["pal_hold"] > 0 and self.beat < self._pal_hold_until - 1e-6
                    if a["pal_hold"] > 0 and not keep:
                        self._pal_hold_until = self.beat + a["pal_hold"] * 32.0
                    if self.trans is not None:         # [transitions] the default transition; never over a running one
                        bars = self.morph_bars
                        self.trans.run(lambda: self._start_recall(i, bars, keep_palette=keep),
                                       self.trans.resolve(None, self.scenes[i]), "autopilot")
                    else:
                        self._start_recall(i, self.morph_bars, keep_palette=keep)
        if a["drift"]:
            k_ou, k_lp = dt / 10.0, 1.0 - math.exp(-dt / 1.5)
            for n in a["drift_params"]:
                d = self._drift.setdefault(n, [0.0, 0.0])
                d[0] = min(1.0, max(-1.0, d[0] - d[0] * k_ou + 0.3 * math.sqrt(dt) * self.rng.gauss(0.0, 1.0)))
                d[1] += (d[0] - d[1]) * k_lp

    # ---------------------------------------------------------------- per-frame

    def update(self, dt, bands=None):
        try:
            dt = float(dt)
        except (TypeError, ValueError):
            dt = 0.0
        if not math.isfinite(dt) or dt < 0:
            dt = 0.0
        dt = min(dt, 0.5)
        now = self.now()
        try:
            self._midi_in(now)
        except Exception:
            pass

        self._step_clock(dt, now)
        beat = self.beat

        if self.pending and self.pending[1] - beat > self.pending[2] + 1e-6:
            self.pending = (self.pending[0], beat, self.pending[2])     # the clock jumped back: do not wait a whole phrase
        if self.pending and beat >= self.pending[1] - 1e-9:
            i, self.pending = self.pending[0], None
            self._manual_recall(i)                     # before autopilot, which then stands down for a period
        if self._clip_pending:                         # quantized clip launches / stops whose beat has come
            due = [q for q in self._clip_pending if beat >= q["fire"] - 1e-9]
            if due:
                self._clip_pending = [q for q in self._clip_pending if q not in due]
                for q in due:
                    self._clip_fire(**{k: v for k, v in q.items() if k != "fire"})
        if self.themes is not None:                    # [themes] quantized theme restores + SOUND TAGS decisions
            try:
                self.themes.step(now, beat)
            except Exception:
                pass
        self._step_autopilot(dt)
        self._step_morph()
        if self._lane_morph:
            self._step_lanes()
        if self.trans is not None and (self.trans.active is not None or self.trans.dip is not None):
            self.trans.step(dt)                        # [transitions] progress / commit / dip / queued recall
        if self.tl is not None:
            try:
                self._step_timeline(dt)
            except Exception:
                pass

        try:
            b = [float(x) for x in list(bands)[:3]] if bands is not None else []
        except (TypeError, ValueError):
            b = []
        b += [0.0] * (3 - len(b))
        self.bands = [min(1.0, max(0.0, x)) if math.isfinite(x) else 0.0 for x in b]

        k = 1.0 - math.exp(-dt / SLEW_TAU) if dt > 0 else 0.0
        mo = self._morph
        w_new = mo["t"] if mo else 1.0
        drift_amt = self.auto["drift_amt"] if self.auto["drift"] else 0.0
        vals = {}
        au = self._auto
        lane_old = None                                # a clip launch's lane morph crossfades the lane's old mods
        if self._lane_morph:
            lane_old = {n: (pair, lm["t"]) for lm in self._lane_morph.values() for n, pair in lm["old"].items()}
        for n in CONT:
            lo, hi = PARAMS[n][0], PARAMS[n][1]
            target = au[n] if n in au else self.base[n]   # timeline automation (its own layer) wins over the fader
            if n == "palette" and self.glide_bars > 0:
                if target != self._pal_target:         # new destination: cross the gap in glide_bars
                    self._pal_target = target
                    self._pal_speed = abs(target - self._sm[n]) / (self.glide_bars * 4.0)
                step = self._pal_speed * dt * self.bpm / 60.0
                d = target - self._sm[n]
                self._sm[n] = target if abs(d) <= step else self._sm[n] + math.copysign(step, d)
            elif n in DISCRETE:
                self._sm[n] = target
            else:
                self._sm[n] += (target - self._sm[n]) * k
                if abs(target - self._sm[n]) < 1e-6:
                    self._sm[n] = target
            m = self._mod_sum(self.mods, n, beat)
            if mo:                                     # crossfade the outgoing modulation (possibly itself a blend)
                m = m * w_new + (1.0 - w_new) * sum(w * self._mod_sum(om, n, beat) for om, w in mo["old"])
            if lane_old and n in lane_old:
                pair, lt = lane_old[n]
                m = m * lt + (1.0 - lt) * self._mod_sum({n: pair}, n, beat)
            if drift_amt and n in self._drift and n in self.auto["drift_params"]:
                m += self._drift[n][1] * drift_amt * 0.5
            vals[n] = min(hi, max(lo, self._sm[n] + m * (hi - lo)))

        # bypass: the finished values of covered params are pulled to neutral; nothing stored is touched
        if self.bypass or self._byp:
            for n, w in self._step_bypass(dt).items():
                vals[n] += (_params.NEUTRAL[n] - vals[n]) * w

        decay = math.exp(-dt / ENV_TAU)
        th = self._tl_trig                             # trigger spans the timeline holds this frame
        for n in self.env:
            self.env[n] = 1.0 if (self.held[n] or n in th) else (self.env[n] * decay if self.env[n] > 2e-3 else 0.0)
        gate = 1.0 if (beat / self.strobe_rate) % 1.0 < 0.5 else 0.0
        for n in TRIGGERS:
            hn = self.held[n] or n in th
            if n in self.env:
                v = max(self.base[n], self.env[n])
            elif n == "strobe":
                v = gate if (hn or self.base[n] > 0) else 0.0
            else:
                v = 1.0 if hn else self.base[n]
            vals[n] = min(1.0, max(0.0, v))
        if self.trans is not None and self.trans.dip is not None:
            vals["master"] *= self.trans.dip_mul()     # [transitions] MORPH fallback: one dip through dark
        self.values = {n: vals[n] for n in PARAMS}     # PARAMS order
        if self.stack is not None:                     # math stack: phases (float64) + modulated fields -> stack.frame
            try:
                fz = self.values["freeze"]
                self.stack.step(dt * (1.0 - fz), dict(bpm=self.bpm, beat=beat), hold=fz >= 0.5,
                                source=lambda m, name, slot: self._source(m, name, slot, beat))
            except Exception:
                pass
        if self.skins is not None and (self.skins.frame or self.skins.gen is not None):   # nothing skinned: not even a call
            try:
                self.skins.step(dt, None, source=lambda m, name, slot: self._source(m, name, slot, beat))
                if self.skins.gen is not None:         # "generate a skin": sent only when calm + the worker is idle
                    self.skins.gen_tick(now, self.ai.status(), self.ai.submit, strained=self.gov_level > 0,
                                        result=getattr(self.ai, "tagged", None), vram_free_mb=self.gov_vram_free_mb)
                    self._skin_ai_cancel()
            except Exception:
                pass

        if now - self._save_t >= AUTOSAVE_S:
            self._save_t = now
            self._autosave()
        if self.shows is not None:
            try:
                self._step_shows(now)
            except Exception:
                pass
        if now - self._led_t >= 0.08:
            self._led_t = now
            try:
                self._leds()
            except Exception:
                pass

        if self.text is not None:                      # phrase auto-cycle on the beat; new words go to the engine
            try:
                self.text.tick(self.beat)
                show = self.text.pop_show()
                if show is not None:
                    self._out.append(passthrough("text_show", **show))
            except Exception:
                pass
        for name in self.ai.pop_shown():               # a finished "show when ready" job: crossfade to it
            self._out.append(passthrough("show_texture", name=name))
        if self.online is not None:                    # ONLINE plugin: finished assistant replies / online images
            try:
                _online.tick(self)
            except Exception:
                pass

        ev, self.events = self.events, []
        return Frame(values=dict(self.values), events=ev, clock=self._clock())

    def _clock(self):
        b = self.beat
        return dict(bpm=self.bpm, beat=b, beat_phase=b % 1.0, bar_phase=(b / 4.0) % 1.0,
                    phrase_phase=(b / 32.0) % 1.0)

    # ---------------------------------------------------------------- panel snapshot

    def state(self):
        global _BLOB_OWNER
        self.blob_cache = self._blobs()                # GET /timeline, /shows, /presets read these (see blob())
        _BLOB_OWNER = weakref.ref(self)
        r = lambda x: round(x, 4)
        c = self._clock()
        mo = self._morph
        return dict(
            t=time.time(),
            values={n: r(v) for n, v in self.values.items()},
            base={n: r(self.base[n]) for n in CONT},
            mods={n: [dict(m, amt=r(m["amt"]), rate=r(m["rate"]), phase=r(m["phase"])) if m else None for m in pair]
                  for n, pair in self.mods.items()},
            clock=dict(bpm=round(c["bpm"], 2), beat=r(c["beat"]), beat_phase=r(c["beat_phase"]),
                       bar_phase=r(c["bar_phase"]), phrase_phase=r(c["phrase_phase"]),
                       src=self.clock_src, active=self.clock_active, midi_clock=self.mclock.active(self.now()),
                       link=self.linkb.enabled, link_error=self.linkb.error, clock_port=self.clock_port,
                       clock_ports=sorted(p for p, t in self._clock_seen.items() if self.now() - t < 5.0),
                       clock_from=self._clock_lock),
            trig={n: [int(self.held[n]), int(self.latch[n])] for n in TRIGGERS},
            strobe_rate=self.strobe_rate,
            scenes=dict(occ=[int(s is not None) for s in self.scenes],
                        names=[s["name"] if s else "" for s in self.scenes],
                        cur=self.cur, pending=self.pending[0] if self.pending else -1,
                        morph_t=r(mo["t"]) if mo else None, safe=self.safe),
            morph=self.morph_bars, quantize=self.quantize, glide=self.glide_bars,
            autopilot=dict(self.auto),
            midi=dict(available=self.midi.available, error=self.midi.error, ports=list(self.midi.ports), learn=self.learn,
                      last=self.midi_last, cue_ch=self.cue_ch, n=len(self.midi.bindings),
                      bound=self.midi.targets()),
            ai=self.ai.status(),
            seq=dict(vals=[r(x) for x in self.seq["vals"]], gates=list(self.seq["gates"])),
            xfade=dict(self.xfade),
            text=self.text.state() if self.text is not None else None,
            bypass=sorted(self.bypass), bypass_fade=self.bypass_fade,
            stack=self.stack.state() if self.stack is not None else None,
            skins=self.skins.state() if self.skins is not None else None,
            canvas=dict(active=self.canvas_prev is not None, fresh=self.fresh),
            online=_online.state(self) if _online is not None else dict(on=False, missing=True),
            tl=self._tl_state() if self.tl is not None else None,
            show=self._show_state(),
            comp=self._comp_state() if _clips is not None else None,
            themes=self.themes.state() if self.themes is not None else None,
            trans=self.trans.state() if self.trans is not None else None,
        )
