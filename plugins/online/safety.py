"""What the assistant LLM is allowed to do: parse its reply, then pass every command through an allowlist.

The LLM never talks to the engine. Its reply is text; `parse_reply` pulls one JSON object out of it and
`validate_batch` turns each entry into a normal control command (the same dicts the panel sends) or rejects it
with a reason. Nothing here imports subprocess, sockets or the filesystem.

Hard rules (photosensitive + show safety), none of them configurable from the panel:
  * triggers (flash, strobe, invert, freeze, blackout, punch) and strobe_rate are performer-only: an LLM can
    never flash the room. Combined with the apply interval in assistant.py (one batch at most every
    MIN_APPLY_S, one value per param per batch) no value the assistant controls can alternate faster than
    MAX_FLASH_HZ.
  * `master` is clamped to MASTER_MIN or more and the whole-`layers` switch cannot be turned off: no blackout
    by the back door. PANIC, blackout, mapping / surfaces, MIDI, clock, recorder, governor: not in the allowlist.
  * the light chase: sweeps a second x how often each point lights up per sweep (chase_mode 0 = 1, 1 = 2,
    2 = up to 4) stays at or under MAX_FLASH_HZ. Checked against the chase_mode the batch leaves behind and against
    the BPM at the moment of APPLY (`finalize`), not only when the reply arrived.
    A recalled scene is applied at once (never queued to the next bar, where it would fire after the clamps), and
    selector values are snapped the way the shader reads them (floor(v + 0.5)); switching the chase on counts too.
  * no blackout by the back door: a batch whose net effect leaves no layer visible (every l_* under LAYER_MIN or
    switched off) is refused as a whole.
  * GPU load: the assistant may lower but never raise render scales / detail / counts (COST), and may not switch on
    more heavy offscreen layers than HEAVY_MAX (none at all while the resource governor reports strain). Its fader
    moves do not count as "the performer's hand" (assistant.quiet_set), so the governor keeps its reductions.
  * no hard cuts: a layer the assistant switches fades over at least MIN_FADE_S, a scene it recalls morphs over at
    least MIN_MORPH_S (`fade_guards`, applied at the moment of apply and reported as clamps).
  * saturated red: the assistant has no flashing control at all and its changes land at most once a second, which is
    inside the "no more than 1 Hz while the picture is saturated red" guidance; it cannot oscillate the full frame.
Every clamp is written to logs/online_*.log (kind "clamp") and shown in the proposal.
"""
import json
import math
import re

import control
from params import PARAMS, TRIGGERS

ALLOWED = ("set", "nudge", "palette", "morph", "scene_recall", "scene_save", "bypass", "art_pick", "text", "gen")
MAX_FLASH_HZ = 2.5          # WCAG 2.3.1 / ITU-R BT.1702 allow at most 3 flashes a second; a dark room with a
                            # wall-sized picture is harsher than a monitor, so stay under it (WCAG 2.2 SC 2.3.1)
MASTER_MIN = 0.15
MIN_FADE_S = 0.35           # shortest layer on / off fade the assistant may cause
MIN_MORPH_S = 0.5           # shortest scene morph the assistant may cause
MAX_CMDS = 24               # per reply
MAX_GEN = 2                 # image jobs per reply
MAX_SAY = 600
MAX_REPLY = 16384           # characters of a reply that are looked at (the JSON we ask for is a few hundred)
MAX_PARSE_TRIES = 48        # "{" positions tried as the start of the reply object
CHASE_MULT = {0: 1.0, 1: 2.0, 2: 4.0}      # times each point lights up per sweep, by chase_mode (live_visuals chaseKey)
LAYER_MIN = 0.15            # a layer below this does not count as visible
HEAVY = ("l_fractal", "l_land", "l_swarm", "l_beings", "l_art")      # offscreen passes: the expensive layers
HEAVY_MAX = 2               # heavy layers the assistant may have on at once (never forces the performer's own off)
COST_EXTRA = ("being_count", "serpent_count")
TEX_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(png|jpg|jpeg|webp)$", re.I)


def parse_reply(text):
    """LLM output -> (say, raw command list, error or ""). Never raises. Prose around the JSON and ``` fences are
    tolerated; anything that is not one JSON object with "say" / "commands" comes back as plain chat text."""
    if not isinstance(text, str) or not text.strip():
        return "", [], "empty reply"
    text = text.strip()
    obj = None
    dec = json.JSONDecoder(strict=False)
    for k, m in enumerate(re.finditer(r"\{", text)):
        if k >= MAX_PARSE_TRIES:                      # bounded work, whatever the model printed
            break
        try:
            cand, _ = dec.raw_decode(text, m.start())
        except (ValueError, RecursionError):
            continue
        if isinstance(cand, dict) and ("commands" in cand or "say" in cand):
            obj = cand
            break
    if obj is None:
        return text[:MAX_SAY], [], "no JSON command object in the reply"
    say = obj.get("say")
    say = say.strip()[:MAX_SAY] if isinstance(say, str) else ""
    cmds = obj.get("commands", [])
    if not isinstance(cmds, list):
        return say, [], "'commands' is not a list"
    return say, cmds, ""


def context(ctl, tex=(), art=()):
    """Everything validation needs to know about the show right now (plain data, so tests can build one by hand)."""
    gov = None
    try:
        fn = getattr(ctl, "gov_status", None)        # optional hook: the engine may publish governor.status here
        gov = fn() if callable(fn) else None
    except Exception:
        gov = None
    strained = bool(isinstance(gov, dict) and (gov.get("active") or gov.get("level") in ("strained", "critical", 1, 2)))
    return dict(bpm=float(ctl.bpm), base=dict(ctl.base), scenes=[s is not None for s in ctl.scenes],
                scene_names=[(s or {}).get("name", "") for s in ctl.scenes], tex=list(tex), art=list(art),
                scene_bases=[dict(s.get("base") or {}) if s else None for s in ctl.scenes],
                bypass=set(ctl.bypass), has_text=ctl.text is not None, strained=strained)


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _param(c):
    name = c.get("name")
    if not isinstance(name, str) or name not in PARAMS:
        raise ValueError(f"unknown param {str(name)[:40]!r}")
    if name in TRIGGERS:
        raise ValueError(f"'{name}' is a performer-only trigger (photosensitive safety)")
    return name


def max_chase_rate(bpm, mode=0):
    """Largest |chase_rate| (sweeps per bar) at which no point of the picture lights up more than MAX_FLASH_HZ
    times a second at this tempo: chase_mode 1 lights each point twice per sweep, mode 2 up to four times."""
    mult = CHASE_MULT.get(snap(mode), 4.0)
    return MAX_FLASH_HZ * 240.0 / max(1.0, float(bpm)) / mult


def snap(v):
    """A selector value as the SHADER reads it: floor(v + 0.5). (Python's round() sends 0.5 to 0, the shader to 1.)"""
    return int(math.floor(float(v) + 0.5))


def is_cost(name):
    return name.endswith("_res") or name.endswith("_detail") or name in COST_EXTRA


def _covered(name, bypass):
    return any(name in control.BYPASS_COVER.get(k, ()) for k in bypass)


def _visible(base, bypass):
    """Is any picture-making layer up? (l_echo is feedback of the others, not a source.)"""
    return any(n.startswith("l_") and n != "l_echo" and base.get(n, 0.0) >= LAYER_MIN and not _covered(n, bypass)
               for n in PARAMS)


def _heavy_on(base, bypass):
    return [n for n in HEAVY if n in PARAMS and base.get(n, 0.0) > 0.01 and not _covered(n, bypass)]


def _after(cmds, ctx):
    """(base, bypass) the batch leaves behind: a recalled scene first, the batch's own sets on top."""
    base, byp = dict(ctx["base"]), set(ctx["bypass"])
    for c in cmds:
        if c["cmd"] == "scene_recall":
            base.update((ctx.get("scene_bases") or [None] * (c["i"] + 1))[c["i"]] or {})
    for c in cmds:
        if c["cmd"] == "set":
            base[c["name"]] = c["v"]
        elif c["cmd"] == "bypass":
            (byp.add if c["on"] else byp.discard)(c["key"])
    return base, byp


def finalize(cmds, ctx):
    """Whole-batch rules, run when the reply arrives AND again at the moment of apply (the tempo, the faders and the
    governor may have moved in between). -> (commands, notes, refusal): refusal != "" means apply nothing."""
    cmds, notes = [dict(c) for c in cmds], []
    # ---- order: a recalled scene first (it rewrites every fader), then the batch's own moves on top of it. That is
    #      the order _after() assumes, so a clamp appended below really is the last word.
    first = [c for c in cmds if c["cmd"] == "morph"] + [c for c in cmds if c["cmd"] == "scene_recall"]
    cmds = first + [c for c in cmds if c["cmd"] not in ("morph", "scene_recall")]
    for c in cmds:
        if c["cmd"] == "scene_recall":
            c["now"] = True          # never queued to the next bar: a queued recall would fire AFTER our clamps
    # ---- GPU cost: never raise render scale / detail / counts
    keep = []
    for c in cmds:
        if c["cmd"] == "set" and is_cost(c["name"]) and c["v"] > ctx["base"].get(c["name"], PARAMS[c["name"]][2]) + 1e-9:
            notes.append(f"{c['name']} not raised (GPU load is the performer's and the governor's call)")
            continue
        keep.append(c)
    cmds = keep
    # ---- heavy layers: no more than HEAVY_MAX on (or than are on already); none new while the governor is strained
    before = _heavy_on(ctx["base"], ctx["bypass"])
    allowed = len(before) if ctx.get("strained") else max(len(before), HEAVY_MAX)
    while True:
        base, byp = _after(cmds, ctx)
        new = [n for n in _heavy_on(base, byp) if n not in before]
        if len(before) + len(new) <= allowed or not new:
            break
        victim = new[-1]
        drop = next((c for c in reversed(cmds) if c["cmd"] == "set" and c["name"] == victim), None) \
            or next((c for c in reversed(cmds) if c["cmd"] == "bypass" and not c["on"]
                     and victim in control.BYPASS_COVER.get(c["key"], ())), None) \
            or next((c for c in cmds if c["cmd"] == "scene_recall"), None)
        if drop is None:
            break
        cmds.remove(drop)
        notes.append(f"{victim} left off: too many heavy layers at once"
                     + (" (the governor is already reducing load)" if ctx.get("strained") else ""))
    # ---- light chase: flashes per second = sweeps per second x multiplicity of the mode the batch leaves behind
    touches = any((c["cmd"] == "set" and c["name"].startswith("chase")) or c["cmd"] == "scene_recall"
                  or (c["cmd"] == "bypass" and not c["on"] and "chase" in control.BYPASS_COVER.get(c["key"], ()))
                  for c in cmds)       # switching the chase ON (fader or bypass) counts as much as changing its rate
    base, byp = _after(cmds, ctx)
    if touches and "chase_rate" in PARAMS:
        lim = max_chase_rate(ctx["bpm"], base.get("chase_mode", 0.0))
        rate = base.get("chase_rate", 0.0)
        if abs(rate) > lim + 1e-9:
            v = lim if rate > 0 else -lim
            cmds = [c for c in cmds if not (c["cmd"] == "set" and c["name"] == "chase_rate")]
            cmds.append({"cmd": "set", "name": "chase_rate", "v": v})
            notes.append(f"chase_rate clamped {rate:g} -> {v:.3g} (at most {MAX_FLASH_HZ:g} flashes a second in chase "
                         f"mode {snap(base.get('chase_mode', 0.0))} at {ctx['bpm']:.0f} BPM)")
            base, byp = _after(cmds, ctx)
    # ---- a recalled scene may carry a master below the floor: lift it
    if any(c["cmd"] == "scene_recall" for c in cmds) and base.get("master", 1.0) < MASTER_MIN - 1e-9:
        cmds = [c for c in cmds if not (c["cmd"] == "set" and c["name"] == "master")]
        cmds.append({"cmd": "set", "name": "master", "v": MASTER_MIN})
        notes.append(f"master lifted to {MASTER_MIN:g} (the recalled scene is darker than the assistant may go)")
        base, byp = _after(cmds, ctx)
    # ---- net effect: never leave the room dark
    if _visible(ctx["base"], ctx["bypass"]) and not _visible(base, byp):
        return [], notes, "this would switch every layer off (a blackout is the performer's call)"
    if base.get("master", 1.0) < MASTER_MIN - 1e-9 and ctx["base"].get("master", 1.0) >= MASTER_MIN:
        return [], notes, "this would take the master below its floor"
    return cmds, notes, ""


def _set(name, v, ctx, notes):
    lo, hi = PARAMS[name][0], PARAMS[name][1]
    if name == "master":
        lo = max(lo, MASTER_MIN)
    if name == "chase_rate":
        lim = max_chase_rate(ctx["bpm"])
        lo, hi = max(lo, -lim), min(hi, lim)
    out = min(hi, max(lo, float(v)))
    if name in control.DISCRETE:
        out = float(min(math.floor(hi + 0.5), max(math.floor(lo + 0.5), snap(out))))    # selectors: whole steps only
    if abs(out - float(v)) > 1e-9:
        notes.append(f"{name} clamped {float(v):g} -> {out:g}")
    return {"cmd": "set", "name": name, "v": out}


def _slot(c, ctx):
    if c.get("slot") is not None:
        s = c["slot"]
        if isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= control.N_SCENES:
            raise ValueError(f"'slot' must be 1..{control.N_SCENES}")
        return s - 1
    name = c.get("name")
    if isinstance(name, str) and name.strip():
        low = name.strip().lower()
        for i, n in enumerate(ctx["scene_names"]):
            if ctx["scenes"][i] and n.lower() == low:
                return i
        raise ValueError(f"no saved scene named {name[:40]!r}")
    raise ValueError(f"scene needs 'slot' (1..{control.N_SCENES}) or 'name'")


def _one(c, ctx, notes):
    """One raw LLM command -> one control / engine command. ValueError(reason) if it is not allowed."""
    if not isinstance(c, dict):
        raise ValueError("not an object")
    cmd = c.get("cmd", c.get("type"))
    if not isinstance(cmd, str) or cmd not in ALLOWED:
        raise ValueError(f"'{str(cmd)[:32]}' is not in the assistant's allowlist")
    if cmd == "set":
        name = _param(c)
        if not _finite(c.get("v")):
            raise ValueError("'v' must be a finite number")
        return _set(name, c["v"], ctx, notes)
    if cmd == "nudge":
        name = _param(c)
        if not _finite(c.get("d")):
            raise ValueError("'d' must be a finite number")
        return _set(name, ctx["base"].get(name, PARAMS[name][2]) + c["d"], ctx, notes)
    if cmd == "palette":
        v = c.get("i", c.get("v"))
        if not _finite(v):
            raise ValueError("'i' must be a number")
        return _set("palette", v, ctx, notes)
    if cmd == "morph":
        if not _finite(c.get("bars")):
            raise ValueError("'bars' must be a number")
        return {"cmd": "morph", "bars": min(64.0, max(0.0, float(c["bars"])))}
    if cmd == "scene_recall":
        i = _slot(c, ctx)
        if not ctx["scenes"][i]:
            raise ValueError(f"scene slot {i + 1} is empty")
        return {"cmd": "scene_recall", "i": i}
    if cmd == "scene_save":
        if c.get("slot") is None:
            raise ValueError(f"'slot' must be 1..{control.N_SCENES}")
        return {"cmd": "scene_save", "i": _slot({"slot": c["slot"]}, ctx)}
    if cmd == "bypass":
        key = c.get("key")
        if not isinstance(key, str) or key not in control.BYPASS_COVER:
            raise ValueError(f"unknown bypass key {str(key)[:40]!r}")
        en = c.get("enabled")
        if not isinstance(en, bool):
            raise ValueError("'enabled' must be true or false")
        if key == "layers" and not en:
            raise ValueError("switching every layer off is a blackout: performer-only")
        return {"cmd": "bypass", "key": key, "on": not en}
    if cmd == "art_pick":
        src, name = c.get("source"), c.get("name")
        if src not in ("art", "ai"):
            raise ValueError("'source' must be \"art\" or \"ai\"")
        pool = ctx["tex"] if src == "ai" else ctx["art"]
        if not isinstance(name, str) or name not in pool:
            raise ValueError(f"{str(name)[:60]!r} is not in the {'texture' if src == 'ai' else 'art'} library")
        if src == "ai" and not TEX_NAME.match(name):
            raise ValueError("bad texture name")
        out = {"source": src, "name": name}
        if c.get("use") is not None:
            u = c["use"]
            hi = int(PARAMS["art_use"][1]) if "art_use" in PARAMS else 0
            if isinstance(u, bool) or not isinstance(u, int) or not 0 <= u <= hi:
                raise ValueError(f"'use' must be 0..{hi}")
            out["use"] = u
        return control.passthrough("art_pick", **out)
    if cmd == "text":
        if not ctx.get("has_text") or "text" not in control.CONTROL_CMDS:
            raise ValueError("the text layer is not available")
        if not isinstance(c.get("text"), str):
            raise ValueError("'text' must be a string")
        return control.validate({"cmd": "text", "text": c["text"]})          # words only: no font / phrase list
    if cmd == "gen":
        raw = {"cmd": "gen", "prompt": c.get("prompt"), "n": 1, "show": True,
               "init": "snapshot" if c.get("from_frame") is True or c.get("init") == "snapshot" else None}
        return control.validate(raw)
    raise ValueError("not allowed")


ONCE = ("scene_save", "scene_recall", "text", "morph", "art_pick")


def validate_batch(raw, ctx):
    """-> (accepted control commands, rejected [{"cmd", "why"}], notes [str]). One value per param (last wins),
    at most MAX_CMDS commands, MAX_GEN image jobs, one each of ONCE."""
    ok, rejected, notes = [], [], []
    if not isinstance(raw, list):
        return ok, [{"cmd": "?", "why": "'commands' is not a list"}], notes
    seen_once = set()
    n_gen = 0
    for n, c in enumerate(raw):
        label = str(c.get("cmd", c.get("type", "?")))[:32] if isinstance(c, dict) else "?"
        if n >= MAX_CMDS:
            rejected.append({"cmd": label, "why": f"more than {MAX_CMDS} commands in one reply"})
            continue
        try:
            out = _one(c, ctx, notes)
            kind = out["cmd"]
            if kind == "gen":
                if n_gen >= MAX_GEN:
                    raise ValueError(f"more than {MAX_GEN} image jobs in one reply")
                n_gen += 1
            if kind in ONCE:
                if kind in seen_once:
                    raise ValueError(f"only one {kind} per reply")
                seen_once.add(kind)
        except ValueError as ex:
            rejected.append({"cmd": label, "why": str(ex)[:160]})
            continue
        except Exception as ex:                                  # a validator bug must never reach the show
            rejected.append({"cmd": label, "why": type(ex).__name__})
            continue
        if kind == "set":                                        # one value per param per batch: last wins
            ok = [o for o in ok if not (o["cmd"] == "set" and o["name"] == out["name"])]
        if kind == "bypass":
            ok = [o for o in ok if not (o["cmd"] == "bypass" and o["key"] == out["key"])]
        ok.append(out)
    if ok:
        ok, more, refusal = finalize(ok, ctx)
        notes += more
        if refusal:
            rejected.append({"cmd": "batch", "why": refusal})
            ok = []
        elif not ok:
            rejected.append({"cmd": "batch", "why": "nothing left after the safety rules: " + "; ".join(more)[:120]})
    return ok, rejected, notes


def fade_guards(ctl, cmds):
    """Called right before a batch is applied: lengthen the controller's fades if this batch would cut hard.
    Returns the notes to show / log. The old values are part of the batch's undo record."""
    notes = []
    kinds = {c["cmd"] for c in cmds}
    if "bypass" in kinds and ctl.bypass_fade < MIN_FADE_S:
        notes.append(f"layer fade lengthened {ctl.bypass_fade:g} -> {MIN_FADE_S:g} s (no hard cuts)")
        ctl.bypass_fade = MIN_FADE_S
    if "scene_recall" in kinds:
        bars = next((c["bars"] for c in cmds if c["cmd"] == "morph"), ctl.morph_bars)
        need = MIN_MORPH_S * max(1.0, float(ctl.bpm)) / 240.0            # bars that last MIN_MORPH_S at this tempo
        if bars < need:
            notes.append(f"scene morph lengthened {bars:g} -> {need:.2f} bars ({MIN_MORPH_S:g} s, no hard cuts)")
            for c in cmds:
                if c["cmd"] == "morph":
                    c["bars"] = need
            ctl.morph_bars = need
    return notes
