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
import json
import math
import queue
import random
import threading
import time
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
    "palette": ["Aya Jewel", "Cosmic Gold/Violet", "Gorge Twilight", "Emerald Shipibo", "Spectral"],
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
try:                                                 # ONLINE plugin hooks (pure python; the plugin itself is only
    from plugins.online import protocol as _online   # imported when the performer switches ONLINE on)
except Exception:
    _online = None
OPTIONS = merge_options(OPTIONS, getattr(_params, "OPTIONS", None), PARAMS)
# stepped params: no slew, snap at the morph midpoint (palette is fractional by contract, so it glides)
DISCRETE = [n for n in OPTIONS if n != "palette"] + [n for n in ("art_index",) if n in PARAMS]
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
        if p[0] == "ripple" and len(p) == 3:
            return 0 <= float(p[1]) <= 1 and 0 <= float(p[2]) <= 1
    except ValueError:
        pass
    return False


def _v_set(c):
    name = _name(c, PARAMS)
    return {"cmd": "set", "name": name, "v": _num(c, "v", PARAMS[name][0], PARAMS[name][1])}


def _v_inc(c):
    name = _name(c, PARAMS)
    span = PARAMS[name][1] - PARAMS[name][0]
    return {"cmd": "inc", "name": name, "d": _num(c, "d", -span, span)}


def _v_mod(c):
    name = c.get("name")
    if isinstance(name, str) and name.startswith("stack:"):      # a math-stack field: stack:<layer id>:<field>
        t = _stack.parse_target(name) if _stack is not None else None
        if t is None or t[1] not in _stack.NUM:
            raise ValueError("unknown stack field")
        return _stack.validate(dict(c, op="mod", id=t[0], f=t[1]))
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
        if c["init"] != "snapshot":
            raise ValueError("'init' must be null or \"snapshot\"")
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
if _textlayer is not None:                           # text, text_phrases, text_add, text_del, text_go, text_cycle
    _VALIDATORS.update(_textlayer.VALIDATORS)
if _online is not None:                              # {"cmd": "online", "op": ...}: plugins/online/protocol.py
    _VALIDATORS["online"] = _online.validate
CONTROL_CMDS = frozenset(_VALIDATORS)


def mark_local(c):
    """remote_control: this validated command came from the laptop itself (127.0.0.1). Only the ONLINE plugin cares."""
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
        stack=_stack.meta() if _stack is not None else None, canvas_keys=list(CANVAS_KEYS))


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
                 ai=True, ai_url="http://127.0.0.1:8766", art_dir=None, autosave=None):
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
        self.restored = self._restore() if restore else False
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

    def panic(self):
        """Failover: clear every trigger, cancel pending recalls / morphs, snap to the safe scene."""
        self.panics += 1                               # governor.py drops a pending suggestion / countdown
        for n in TRIGGERS:
            self.held[n] = False
            self.base[n] = PARAMS[n][2]
        self.env = {k: 0.0 for k in self.env}
        self.pending = self._morph = None
        self.learn = None
        self.auto["on"] = False                        # nothing may move the look away from the safe scene
        self._auto_next = None
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
        self.pending = self._morph = None
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

    def scene_save(self, i):
        if not 0 <= i < N_SCENES:
            return
        snap = dict(name="", base={n: self.base[n] for n in CONT}, mods=_copy_mods(self.mods))
        if self.stack is not None:
            snap["stack"] = self.stack.to_json()
        if not self._scenes_ok:
            self._load_scenes()                        # the file was unreadable earlier: try again before writing
        if self.scenes[i]:
            snap["name"] = self.scenes[i]["name"]
        self.scenes[i] = snap
        self.cur = i
        self._save_scenes()

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

    def scene_recall(self, i, now=False):
        """Recall slot i at the next quantize boundary (or immediately), morphing over morph_bars."""
        if not 0 <= i < N_SCENES or self.scenes[i] is None:
            return False
        q = QUANT.get(self.quantize, 0)
        if now or q == 0:
            self.pending = None
            self._manual_recall(i)
        else:
            self.pending = (i, (math.floor(self.beat / q + 1e-6) + 1) * q, q)
        return True

    def _manual_recall(self, i):
        """A recall the performer asked for: autopilot waits a full period before it changes the look again."""
        self._start_recall(i, self.morph_bars)
        if self.auto["on"]:
            self._auto_next = self.beat + self.auto["bars"] * 4.0

    def _start_recall(self, i, bars, keep_palette=False):
        sc = self.scenes[i]
        if sc is None:
            return
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
                    stack=self.stack.to_json() if self.stack is not None else None, canvas=self.canvas_prev)

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
            self.base.update(_clean_base(d.get("base"), {n: self.base[n] for n in CONT}))
            self.mods = _clean_mods(d.get("mods"))
            self._sm = dict(self.base)
            self._pal_target = self.base["palette"]
            cur = d.get("cur")
            if isinstance(cur, int) and not isinstance(cur, bool) and 0 <= cur < N_SCENES and self.scenes[cur]:
                self.cur = cur
            s = d.get("settings") if isinstance(d.get("settings"), dict) else {}
            for cmd in (dict(cmd="bpm", v=s.get("bpm")), dict(cmd="clock_src", src=s.get("clock_src")),
                        dict(cmd="strobe_rate", v=s.get("strobe_rate")), dict(cmd="morph", bars=s.get("morph")),
                        dict(cmd="quantize", q=s.get("quantize")), dict(cmd="glide", bars=s.get("glide")),
                        dict(cmd="safe", i=s.get("safe")), dict(cmd="cue_ch", ch=s.get("cue_ch")),
                        dict(s.get("autopilot") if isinstance(s.get("autopilot"), dict) else {}, cmd="autopilot")):
                try:
                    self._do(validate(cmd))
                except Exception:
                    pass                               # one bad field does not spoil the rest
            self.seq = _clean_seq(s.get("seq"))
            if isinstance(s.get("ext"), dict):         # other modules' settings (small JSON dicts, keyed by module)
                self.ext = {k: v for k, v in s["ext"].items() if isinstance(k, str) and isinstance(v, dict)}
            if self.text is not None:
                self.text.restore(s.get("text"))
            if isinstance(s.get("bypass"), list):      # come back already bypassed: no fade-out on the projector
                self.bypass = {k for k in s["bypass"] if isinstance(k, str) and k in BYPASS_COVER}
                self._byp = {k: 1.0 for k in self.bypass}
            if self.stack is not None and isinstance(s.get("stack"), dict):
                self.stack.restore(s["stack"])         # comes back as it was: no fade-in on the projector
            cv = s.get("canvas")
            if self.stack is not None and isinstance(cv, dict) and isinstance(cv.get("base"), dict):
                self.canvas_prev = dict(base=_clean_base(cv["base"]), mods=_clean_mods(cv.get("mods")),
                                        bypass=sorted(k for k in (cv.get("bypass") if isinstance(cv.get("bypass"), list) else [])
                                                      if isinstance(k, str) and k in BYPASS_COVER),
                                        stack=_stack.clean_stack(cv.get("stack")), auto=bool(cv.get("auto")))
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
                    self._do(validate(c))
                except Exception:
                    pass
        except TypeError:
            pass
        if self._out:
            rest += self._out
            self._out = []
        return rest

    def _do(self, c):
        cmd = c["cmd"]
        if cmd == "set":
            self.set(c["name"], c["v"])
        elif cmd == "inc":
            self.nudge(c["name"], c["d"])
        elif cmd == "mod":
            self.set_mod(c["name"], c["slot"], c["src"], c["amt"], c["rate"], c["phase"])
        elif cmd == "stack":                           # math stack edit (also what a "mod" on stack:<id>:<field> becomes)
            self._stack_morph = False
            self.stack.do(c)
        elif cmd == "canvas":
            (self.new_canvas if c["op"] == "new" else self.restore_canvas)()
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
            self.scene_recall(c["i"], c.get("now", False))
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
        press = kind == "note_on" or (kind == "cc" and val >= 64)
        edge = self._cc_rise(b, val) if kind == "cc" else press   # one-shot actions: a CC fires once per crossing of 64
        if p[0] == "param":
            name = p[1]
            lo, hi = PARAMS[name][0], PARAMS[name][1]
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
        self._step_autopilot(dt)
        self._step_morph()

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
        for n in CONT:
            lo, hi = PARAMS[n][0], PARAMS[n][1]
            target = self.base[n]
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
            if drift_amt and n in self._drift and n in self.auto["drift_params"]:
                m += self._drift[n][1] * drift_amt * 0.5
            vals[n] = min(hi, max(lo, self._sm[n] + m * (hi - lo)))

        # bypass: the finished values of covered params are pulled to neutral; nothing stored is touched
        if self.bypass or self._byp:
            for n, w in self._step_bypass(dt).items():
                vals[n] += (_params.NEUTRAL[n] - vals[n]) * w

        decay = math.exp(-dt / ENV_TAU)
        for n in self.env:
            self.env[n] = 1.0 if self.held[n] else (self.env[n] * decay if self.env[n] > 2e-3 else 0.0)
        gate = 1.0 if (beat / self.strobe_rate) % 1.0 < 0.5 else 0.0
        for n in TRIGGERS:
            if n in self.env:
                v = max(self.base[n], self.env[n])
            elif n == "strobe":
                v = gate if (self.held[n] or self.base[n] > 0) else 0.0
            else:
                v = 1.0 if self.held[n] else self.base[n]
            vals[n] = min(1.0, max(0.0, v))
        self.values = {n: vals[n] for n in PARAMS}     # PARAMS order
        if self.stack is not None:                     # math stack: phases (float64) + modulated fields -> stack.frame
            try:
                fz = self.values["freeze"]
                self.stack.step(dt * (1.0 - fz), dict(bpm=self.bpm, beat=beat), hold=fz >= 0.5,
                                source=lambda m, name, slot: self._source(m, name, slot, beat))
            except Exception:
                pass

        if now - self._save_t >= AUTOSAVE_S:
            self._save_t = now
            self._autosave()
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
            canvas=dict(active=self.canvas_prev is not None, fresh=self.fresh),
            online=_online.state(self) if _online is not None else dict(on=False, missing=True),
        )
