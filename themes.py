"""THEMES: one-tap capture / exact restore of a layer, a tab (params group) or the whole look, plus SOUND TAGS
(a theme comes back by itself when similar music plays). Pure python; owned by control.Controller (GL thread).

A theme is a PRESET WITH A SCOPE: starred themes are ordinary presets of kind "theme" in presets/theme/<slug>.json
(shows.py store, same files, thumbnails, import / export, timeline preset cues `theme/<slug>`). The session TRAY
(unstarred + starred captures, at most MAX_TRAY) lives in live_state.json settings["themes"] and in show files
(settings key "themes", travels with the "scenes" part).

Scopes (`parse_scope`): "look" (every visual fader except master; not mapping, MIDI, camera ON, triggers),
"layer:<l_* param or serpent>" (shows.layer_params: the fader + the tab its bypass switch covers),
"group:<params group>" (one tab; "shape" includes "space"; + the layers that tab's switch covers),
"stack:<math stack layer id>" (one stack layer).

What a capture holds (`data`), restored EXACTLY by a cut (and by a morph once it has finished):
  values   base value of every param in scope (stepped ones such as kene_seed, being_count, being_form, travel_mode,
           art_index are params, so the design seed / form / count come back exactly)
  mods     the modulation pair of each of those params (absent = no modulation; restore removes extra mods)
  bypass   {key: on} for every bypass key in scope (manual restore only: a SOUND TAG never switches a layer on or
           off, the performer's kills stand)
  stack    look / l_stack / layers tab: the whole math stack (ids, fields, stops, mods) + `stack_ph` (each layer's
           float64 scroll / spin / warp / hue / pulse phases); stack:<id>: that layer + its mods + its phases
  skins    look / layers tab: every environment skin slot; land layer / tab: the deck worlds' slots (not the tunnel)
  text     text layer / tab / look: words, font, align, phrase list, auto-cycle (text_layer.TextState JSON)
  media    the AI texture last sent to the engine (show_texture / art_pick ai) for look / l_ai / l_art / art / layers
NOT restorable (free-running, by design): engine phases integrated from wall time (mandala rotation, flow / travel
distance, serpent and beings positions on their paths, swarm trails, kene pen progress, echo / feedback buffer),
beat-synced LFO phases (they follow the show clock, so they line up again with the beat, not with the moment),
the AI worker's queue. A MORPHED restore keeps the stack's running phases (no visible jump); a CUT (hold) jumps
them to the captured values.

Restore: tap = morph over the performer's morph length at the performer's quantize; hold = instant cut
(`now: true`). Only the scope is written (a running scene / clip morph lets go of exactly those params); one undo
step (Controller.history entry with a "theme" pre-capture).

SOUND TAGS: see soundtags.py for features + matching. This module records examples (TAG), compiles templates for
the engine, runs soundtags.Decider at the engine's 2-4 Hz and fires restores, gated by: the global ARM switch
(off at every start, never saved), PANIC (cancels anything pending, holds PANIC_HOLD_S), blackout / master down /
the "layers" bypass, a performer action (holds `override` s, default 20, and uses up a trigger that was ready),
the flash guard holding, the governor at critical; an auto restore never cuts (>= 1 beat morph) and never
touches bypass. Every fire / hold is logged (panel log + console).
"""
import copy
import math
import re
import time

try:
    import soundtags as _st
except ImportError:                                   # numpy missing: themes still work, sound tags do not
    _st = None

VERSION = 1
MAX_TRAY = 32
PALETTE_NAMES = ("Jewel", "Gold", "Twilight", "Emerald", "Spectral")
ID = re.compile(r"^[a-z0-9]{1,16}$")
OVERRIDE_S = 20.0           # a performer action holds sound-tag recall this long
GAP_S = 8.0                 # at least this long between two automatic recalls
PANIC_HOLD_S = 30.0
TEST_S = 15.0               # learn / test meters stay on this long after the panel's last {"op": "test"}
TAG_LEN = 10.0
TAG_LEN_MIN, TAG_LEN_MAX = 5.0, 15.0
AUTO_MIN_BARS = 0.25        # an automatic recall morphs at least one beat: never a cut
LOG_MAX = 40
LAND_ENVS_EXCLUDE = ("tunnel",)
PERFORMER_CMDS = frozenset(("set", "inc", "scene_recall", "scene_go", "clip", "macro", "preset", "bypass", "mod",
                            "stack", "skin", "xfade", "canvas", "init", "text", "text_go", "text_cycle", "art_step",
                            "history", "seq", "seq_rand"))
STACK_PH = ("scroll", "spin", "warp", "hue", "pulse", "env")
OPS = ("capture", "restore", "star", "rename", "recapture", "delete", "tag", "tag_set", "tag_clear", "arm", "test",
       "opts", "suggest", "suggest_apply")


def _ctl():
    import control                                    # lazy: control imports this module
    return control


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ------------------------------------------------------------------ scopes


def layer_keys():
    c = _ctl()
    return [n for n in c.CONT if n.startswith("l_")] + (["serpent"] if "serpent" in c.CONT else [])


def group_keys():
    c = _ctl()
    seen = []
    for n in c.CONT:
        g = c.PARAMS[n][3]
        if g not in ("master", "space") and g not in seen:
            seen.append(g)
    return seen


def parse_scope(s):
    """'look' | 'layer:<key>' | 'group:<g>' | 'stack:<id>' (or the dict form) -> {"kind", "key"}; ValueError."""
    if isinstance(s, dict):
        kind, key = s.get("kind"), s.get("key")
    elif isinstance(s, str):
        kind, _, key = s.partition(":")
    else:
        raise ValueError("'scope' must be look, layer:<name>, group:<name> or stack:<id>")
    if kind == "look":
        return {"kind": "look", "key": None}
    if kind == "layer" and key in layer_keys():
        return {"kind": "layer", "key": key}
    if kind == "group":
        key = "shape" if key == "space" else key
        if key in group_keys():
            return {"kind": "group", "key": key}
    if kind == "stack":
        try:
            i = int(key)
        except (TypeError, ValueError):
            i = 0
        if 1 <= i <= 999999:
            return {"kind": "stack", "key": i}
    raise ValueError("'scope' must be look, layer:<name>, group:<name> or stack:<id>")


def scope_names(scope):
    """The params a scope captures (fader values + their mods)."""
    c = _ctl()
    import params as P
    kind, key = scope["kind"], scope["key"]
    if kind == "look":
        return [n for n in c.CONT if n != "master"]
    if kind == "layer":
        import shows
        return [n for n in shows.layer_params(key) if n in c.CONT and n != "master"]
    if kind == "group":
        groups = {key, "space"} if key == "shape" else {key}
        names = [n for n in c.CONT if c.PARAMS[n][3] in groups]
        names += [n for n in getattr(P, "BYPASS_LINKS", {}).get(key, ()) if n in c.CONT and n not in names]
        return names
    return []


def scope_bypass(scope):
    """Bypass keys a scope owns."""
    c = _ctl()
    import params as P
    cover, links = c.BYPASS_COVER, getattr(P, "BYPASS_LINKS", {})
    kind, key = scope["kind"], scope["key"]
    if kind == "look":
        return sorted(cover)
    out = set()
    if kind == "layer":
        if key in cover:
            out.add(key)
        out |= {g for g, ls in links.items() if key in ls and g in cover}
    elif kind == "group":
        g = P.BYPASS_GROUP_KEY.get(key, key) if hasattr(P, "BYPASS_GROUP_KEY") else key
        if g in cover:
            out.add(g)
        out |= {n for n in links.get(key, ()) if n in cover}
    return sorted(out)


def _wants(scope):
    """Which extra state a scope carries: (stack, skins: None | 'all' | 'land', text, media)."""
    kind, key = scope["kind"], scope["key"]
    look = kind == "look"
    layers = kind == "group" and key == "layers"
    stack = look or layers or (kind == "layer" and key == "l_stack")
    skins = "all" if (look or layers) else ("land" if (kind == "layer" and key == "l_land") or (
        kind == "group" and key == "land") else None)
    text = look or (kind == "layer" and key == "l_text") or (kind == "group" and key == "text")
    media = look or layers or (kind == "layer" and key in ("l_ai", "l_art")) or (kind == "group" and key == "art")
    return stack, skins, text, media


def scope_label(scope):
    c = _ctl()
    kind, key = scope["kind"], scope["key"]
    if kind == "look":
        return "Look"
    if kind == "layer":
        lab = c.PARAMS[key][4] if key in c.PARAMS else key
        return str(lab)[:24]
    if kind == "group":
        return str(key).capitalize()
    return f"Stack #{key}"


def palette_name(v):
    try:
        return PALETTE_NAMES[int(round(float(v))) % len(PALETTE_NAMES)]
    except (TypeError, ValueError):
        return ""


def _skin_env(k):
    return str(k).split(":", 1)[0]


# ------------------------------------------------------------------ cleaning (untrusted JSON: live_state, shows, presets)


def clean_tag(raw):
    """Tag settings + examples. None when there is no tag."""
    if not isinstance(raw, dict) or _st is None:
        return None
    d = dict(_st.TAG_DEFAULTS)
    if raw.get("mode") in _st.MODES:
        d["mode"] = raw["mode"]
    for k in ("key_inv", "on", "only_auto"):
        if isinstance(raw.get(k), bool):
            d[k] = raw[k]
    for k, lo, hi in (("sens", 0.0, 1.0), ("hold", 0.5, 30.0), ("cooldown", 0.0, 600.0)):
        if _isnum(raw.get(k)):
            d[k] = round(min(hi, max(lo, float(raw[k]))), 3)
    q = "none" if raw.get("quant") == "off" else raw.get("quant")
    if q in _st.QUANTS:
        d["quant"] = q
    m = raw.get("morph")
    d["morph"] = None if m is None or not _isnum(m) else round(min(64.0, max(0.0, float(m))), 3)
    if _isnum(raw.get("prio")):
        d["prio"] = int(min(9, max(0, round(raw["prio"]))))
    ex = raw.get("ex") if isinstance(raw.get("ex"), list) else []
    d["ex"] = [e for e in (_st.clean_example(x) for x in ex[-_st.MAX_EXAMPLES:]) if e]
    return d


def clean_data(raw, scope):
    """The captured state, every field through its owner's validator (nothing is trusted from a file)."""
    c = _ctl()
    import shows
    raw = raw if isinstance(raw, dict) else {}
    names = set(scope_names(scope))
    vals = {n: v for n, v in shows.clean_values(raw.get("values")).items() if n in names}
    mods = {n: p for n, p in c._clean_mods(raw.get("mods")).items() if n in vals}
    keys = set(scope_bypass(scope))
    byp = raw.get("bypass") if isinstance(raw.get("bypass"), dict) else {}
    out = {"values": vals, "mods": mods, "bypass": {k: v for k, v in byp.items() if k in keys and isinstance(v, bool)}}
    ph_raw = raw.get("stack_ph") if isinstance(raw.get("stack_ph"), dict) else {}

    def phases(ids):
        ph = {}
        for i in ids:
            p = ph_raw.get(str(i))
            if isinstance(p, dict):
                q = {k: float(p[k]) for k in STACK_PH if _isnum(p.get(k)) and abs(p[k]) < 1e9}
                if q:
                    ph[str(i)] = q
        return ph

    if c._stack is not None:
        if isinstance(raw.get("stack"), dict):
            out["stack"] = c._stack.clean_stack(raw["stack"])
            out["stack_ph"] = phases(x["id"] for x in out["stack"]["layers"])
        elif scope["kind"] == "stack" and isinstance(raw.get("stack_layer"), dict):
            lay = dict(raw["stack_layer"], id=scope["key"])
            st = c._stack.clean_stack(dict(layers=[lay], mods=raw.get("stack_mods"), next=scope["key"] + 1))
            if st["layers"] and st["layers"][0]["id"] == scope["key"]:
                out["stack_layer"] = st["layers"][0]
                out["stack_mods"] = st["mods"]
                out["stack_ph"] = phases([scope["key"]])
    if getattr(c, "_skins", None) is not None and isinstance(raw.get("skins"), dict):
        out["skins"] = c._skins.clean_skins(raw["skins"])
    if isinstance(raw.get("text"), dict):                     # TextState.restore validates every field again
        t = raw["text"]
        out["text"] = {k: copy.deepcopy(t[k]) for k in ("text", "font", "align", "phrases", "cycle") if k in t}
    md = raw.get("media") if isinstance(raw.get("media"), dict) else {}
    tex = md.get("tex")
    if isinstance(tex, str) and c._TEX_NAME.match(tex) and ".." not in tex:
        out["media"] = {"tex": tex}
    return out


def clean_theme(raw):
    """One tray / preset theme -> clean dict, or ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("not a theme")
    scope = parse_scope(raw.get("scope"))
    tid = raw.get("id")
    tid = tid if isinstance(tid, str) and ID.match(tid) else None
    name = raw.get("name")
    name = name.strip()[:48] if isinstance(name, str) and name.strip() else scope_label(scope)
    slot = raw.get("slot")
    slot = int(slot) if _isnum(slot) and 1 <= slot <= MAX_TRAY else None
    ref = raw.get("ref")
    ref = ref if isinstance(ref, str) and re.match(r"^theme/[a-z0-9][a-z0-9_-]{0,47}$", ref) else None
    t = raw.get("t")
    pal = raw.get("pal")
    return {"id": tid, "slot": slot, "name": name, "scope": scope, "t": float(t) if _isnum(t) else time.time(),
            "star": raw.get("star") is True, "ref": ref, "thumb": 0,
            "pal": round(float(pal), 3) if _isnum(pal) else None,
            "data": clean_data(raw.get("data"), scope), "tag": clean_tag(raw.get("tag"))}


def clean_tray(raw):
    raw = raw if isinstance(raw, dict) else {}
    items = raw.get("tray") if isinstance(raw.get("tray"), list) else []
    out, ids, slots = [], set(), set()
    for x in items[:MAX_TRAY]:
        try:
            th = clean_theme(x)
        except ValueError:
            continue
        if th["id"] is None or th["id"] in ids:
            continue
        if th["slot"] in slots:
            th["slot"] = None
        ids.add(th["id"])
        if th["slot"]:
            slots.add(th["slot"])
        out.append(th)
    for th in out:                                            # every theme has a MIDI slot
        if th["slot"] is None:
            th["slot"] = next(i for i in range(1, MAX_TRAY + 1) if i not in slots)
            slots.add(th["slot"])
    return out


# ------------------------------------------------------------------ command validation


def validate(c):
    """{"cmd": "theme", "op": ...} -> clean copy; ValueError for anything malformed."""
    ctl = _ctl()
    op = c.get("op")
    if op not in OPS:
        raise ValueError("theme 'op' must be one of " + ", ".join(OPS))
    out = {"cmd": "theme", "op": op}

    def tid(required=True):
        v = c.get("id")
        if v is None and not required:
            return None
        if not isinstance(v, str) or not ID.match(v):
            raise ValueError("'id' must be a theme id")
        return v

    if op == "capture":
        out["scope"] = parse_scope(c.get("scope", "look"))
        if isinstance(c.get("name"), str) and c["name"].strip():
            out["name"] = c["name"].strip()[:48]
        if _isnum(c.get("slot")) and 1 <= c["slot"] <= MAX_TRAY:
            out["slot"] = int(c["slot"])
    elif op == "restore":
        if "slot" in c and "id" not in c:
            out["slot"] = ctl._int(c, "slot", 1, MAX_TRAY)
        else:
            out["id"] = tid()
        if c.get("now") is not None:
            out["now"] = ctl._bool(c, "now")
        if c.get("bars") is not None:
            out["bars"] = ctl._num(c, "bars", 0.0, 64.0)
    elif op in ("star", "arm", "test"):
        if op == "star":
            out["id"] = tid()
        out["on"] = ctl._bool(c, "on")
    elif op == "rename":
        out["id"] = tid()
        name = c.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be text")
        out["name"] = name.strip()[:48]
    elif op in ("recapture", "delete"):
        out["id"] = tid()
    elif op == "tag":
        out["id"] = tid()
        out["on"] = ctl._bool(c, "on")
        if c.get("len") is not None:
            out["len"] = ctl._num(c, "len", TAG_LEN_MIN, TAG_LEN_MAX)
    elif op == "tag_set":
        out["id"] = tid()
        s = c.get("set")
        if not isinstance(s, dict) or not s:
            raise ValueError("'set' must be an object of tag settings")
        t = clean_tag(dict(s))
        if t is None:
            raise ValueError("sound tags are unavailable")
        out["set"] = {k: t[k] for k in s if k in _st.TAG_DEFAULTS}
    elif op == "tag_clear":
        out["id"] = tid()
        if c.get("n") is not None:
            out["n"] = ctl._int(c, "n", 0, 63)
    elif op == "opts":
        if c.get("override") is not None:
            out["override"] = ctl._num(c, "override", 0.0, 300.0)
        if c.get("gap") is not None:
            out["gap"] = ctl._num(c, "gap", 0.0, 120.0)
    elif op == "suggest_apply":
        out["sid"] = ctl._int(c, "sid", 0, 63)
        v = tid(required=False)
        if v is not None:
            out["id"] = v
    return out


# ------------------------------------------------------------------ the manager


class ThemeManager:
    """Tray + capture / restore + sound-tag decisions for one Controller. Every method runs on the GL thread."""

    def __init__(self, ctl):
        self.ctl = ctl
        self.tray = []
        self.rev = 1
        self.next = 1
        self.armed = False                       # SOUND TAGS arm: off at every start (never saved)
        self.test_until = -1e9
        self.override_s, self.gap_s = OVERRIDE_S, GAP_S
        self.pending = None                      # quantized restore: {id, fire (beat), q, bars, auto}
        self.rec = None                          # TAG recording: {id, start (frame index), t0, len}
        self.current = None                      # id of the last restored theme
        self.log = []                            # [t (epoch), kind, id, name, sim, note]
        self.meters, self.block = [], None
        self.dec = _st.Decider() if _st is not None else None
        self.media = {}                          # last AI texture sent to the engine
        self.suggestions = []
        self._blob = (0, b"{}")
        self._panics = getattr(ctl, "panics", 0)
        self._panic_until = -1e9
        self._manual_t = -1e9
        self._edits = None
        self._byp_seen = None
        self._eng = None
        self._sims_seen = None
        self._held_logged = {}
        self._last_fire = None                   # (wall t, id, name, sim, auto)

    # ---- persistence
    def to_json(self):
        out = []
        for th in self.tray:
            d = {k: th[k] for k in ("id", "slot", "name", "scope", "t", "star", "ref", "pal", "data", "tag")}
            out.append(d)
        return {"v": VERSION, "next": self.next, "tray": out, "override": self.override_s, "gap": self.gap_s}

    def restore(self, raw):
        """live_state / show settings -> tray (validated). Never raises."""
        try:
            self.tray = clean_tray(raw)
            nx = raw.get("next") if isinstance(raw, dict) else None
            self.next = max([int(nx) if _isnum(nx) and nx >= 1 else 1] + [self._num(th["id"]) + 1 for th in self.tray])
            if isinstance(raw, dict):
                if _isnum(raw.get("override")):
                    self.override_s = min(300.0, max(0.0, float(raw["override"])))
                if _isnum(raw.get("gap")):
                    self.gap_s = min(120.0, max(0.0, float(raw["gap"])))
            self.pending = self.rec = None
            self._compile()
            self._bump()
            return True
        except Exception:
            return False

    @staticmethod
    def _num(tid):
        m = re.match(r"^t(\d+)$", tid or "")
        return int(m.group(1)) if m else 0

    def _bump(self):
        self.rev += 1

    def find(self, tid):
        return next((th for th in self.tray if th["id"] == tid), None)

    def by_slot(self, slot):
        return next((th for th in self.tray if th["slot"] == slot), None)

    def say(self, level, text):
        m = getattr(self.ctl, "shows", None)
        if m is not None:
            m.say(level, text)

    def _log(self, kind, th, sim=None, note=""):
        self.log.append([round(time.time(), 2), kind, th["id"] if th else None, th["name"] if th else "",
                         None if sim is None else round(float(sim), 3), str(note)[:80]])
        del self.log[:-LOG_MAX]
        self._bump()
        if kind in ("fire", "block"):
            print(f"[soundtags] {kind} {th['name'] if th else '-'} sim={sim if sim is None else round(sim, 3)} {note}")

    # ---- capture
    def capture_data(self, scope, missing_ok=False):
        ctl = self.ctl
        names = scope_names(scope)
        data = {"values": {n: ctl.base[n] for n in names},
                "mods": {n: copy.deepcopy(ctl.mods[n]) for n in names if n in ctl.mods},
                "bypass": {k: k in ctl.bypass for k in scope_bypass(scope)}}
        stack, skins, text, media = _wants(scope)
        st = ctl.stack
        if st is not None and stack:
            data["stack"] = st.to_json()
            data["stack_ph"] = {str(x["id"]): self._ph(x["id"]) for x in st.layers}
        if st is not None and scope["kind"] == "stack":
            lay = st.find(scope["key"])
            if lay is None and missing_ok:                    # undo of bringing a deleted stack layer back
                data["stack_absent"] = scope["key"]
                return data
            if lay is None:
                raise ValueError(f"there is no stack layer #{scope['key']}")
            data["stack_layer"] = copy.deepcopy(lay)
            pre = f"{scope['key']}:"
            data["stack_mods"] = {k: copy.deepcopy(v) for k, v in st.mods.items() if k.startswith(pre)}
            data["stack_ph"] = {str(scope["key"]): self._ph(scope["key"])}
        sk = getattr(ctl, "skins", None)
        if sk is not None and skins:
            j = sk.to_json()
            if skins == "land":
                j = dict(j, slots={k: v for k, v in j.get("slots", {}).items() if _skin_env(k) not in LAND_ENVS_EXCLUDE},
                         mods={k: v for k, v in j.get("mods", {}).items() if _skin_env(k) not in LAND_ENVS_EXCLUDE})
            data["skins"] = copy.deepcopy(j)
        if text and ctl.text is not None:
            t = ctl.text.to_json()
            data["text"] = {k: copy.deepcopy(t[k]) for k in ("text", "font", "align", "phrases", "cycle") if k in t}
        if media and self.media.get("tex"):
            data["media"] = {"tex": self.media["tex"]}
        return data

    def _ph(self, i):
        ph = self.ctl.stack._ph.get(i) if self.ctl.stack is not None else None
        return {k: float(ph[k]) for k in STACK_PH if ph and k in ph}

    def _free_slot(self, want=None):
        used = {th["slot"] for th in self.tray}
        if want and want not in used:
            return want
        return next((i for i in range(1, MAX_TRAY + 1) if i not in used), None)

    def capture(self, scope, name=None, slot=None, thumb=True):
        """Instant: a new tray theme of `scope` (auto-named). Returns it, or None (tray full of kept themes)."""
        ctl = self.ctl
        if slot is not None and self.by_slot(slot) is not None:
            return self.recapture(self.by_slot(slot)["id"])
        if len(self.tray) >= MAX_TRAY:                        # drop the oldest theme that is neither starred nor tagged
            old = next((th for th in self.tray if not th["star"] and not th["tag"]), None)
            if old is None:
                self.say("warn", "the theme tray is full of starred / tagged themes: delete one first")
                return None
            self._delete(old)
        try:
            data = self.capture_data(scope)
        except ValueError as ex:
            self.say("error", str(ex))
            return None
        tid = f"t{self.next}"
        self.next += 1
        pal = ctl.base.get("palette")
        auto = f"{scope_label(scope)} {time.strftime('%H:%M')} {palette_name(pal)}".strip()
        th = {"id": tid, "slot": self._free_slot(slot), "name": (name or auto)[:48], "scope": dict(scope),
              "t": time.time(), "star": False, "ref": None, "thumb": 0, "pal": round(float(pal), 3), "data": data,
              "tag": None}
        self.tray.append(th)
        if thumb:
            self._thumb(th)
        self._bump()
        ctl._changed()
        return th

    def recapture(self, tid):
        th = self.find(tid)
        if th is None:
            return None
        try:
            th["data"] = self.capture_data(th["scope"])
        except ValueError as ex:
            self.say("error", str(ex))
            return None
        th["pal"] = round(float(self.ctl.base.get("palette", 0.0)), 3)
        self._thumb(th)
        if th["star"]:
            self._star_save(th)
        self._bump()
        self.ctl._changed()
        return th

    def _thumb(self, th):
        ctl, m = self.ctl, getattr(self.ctl, "shows", None)
        if not getattr(ctl, "thumbs", False) or m is None or not hasattr(m, "_do_theme_thumb"):
            return
        ctl._out.append(_ctl().passthrough("snapshot"))
        m.submit("theme_thumb", tid=th["id"], snap_t=time.time())

    def thumb_done(self, tid):
        th = self.find(tid)
        if th is not None:
            th["thumb"] = self.rev + 1
            self._bump()

    def _delete(self, th):
        self.tray = [x for x in self.tray if x is not th]
        if self.pending and self.pending["id"] == th["id"]:
            self.pending = None
        if self.rec and self.rec["id"] == th["id"]:
            self.rec = None
        m = getattr(self.ctl, "shows", None)
        if m is not None and hasattr(m, "_do_theme_thumb"):
            m.submit("theme_thumb_clear", tid=th["id"])
        self._compile()

    # ---- star (= kept in the preset library as kind "theme")
    def preset_of(self, th):
        import shows
        body = {k: copy.deepcopy(th[k]) for k in ("name", "scope", "pal", "data", "tag")}
        return {"format": shows.PRESET_FORMAT, "version": shows.PRESET_VERSION, "kind": "theme", "name": th["name"],
                "tags": ["theme", th["scope"]["kind"]], "fav": True, "created": th["t"], "data": {"theme": body}}

    def _star_save(self, th):
        m = getattr(self.ctl, "shows", None)
        if m is None or not hasattr(m, "_do_theme_star"):
            return
        slug = th["ref"].split("/", 1)[1] if th["ref"] else None
        m.submit("theme_star", preset=self.preset_of(th), tid=th["id"], slug=slug)

    def star(self, tid, on):
        th = self.find(tid)
        if th is None or th["star"] == on:
            return
        th["star"] = on
        m = getattr(self.ctl, "shows", None)
        if on:
            self._star_save(th)
        elif th["ref"] and m is not None:
            m.submit("preset_delete", ref=th["ref"])
            th["ref"] = None
        self._bump()
        self.ctl._changed()

    def starred(self, tid, ref):
        th = self.find(tid)
        if th is not None:
            th["ref"] = ref
            self._bump()

    # ---- restore
    def restore_theme(self, th, now=False, bars=None, auto=False, quant=None):
        """Tap: quantized (performer's quantize, or the tag's) + morphed; now=True: an instant cut."""
        ctl = self.ctl
        c = _ctl()
        q = c.QUANT.get("off" if (quant or ctl.quantize) == "none" else (quant or ctl.quantize), 0)
        if now or q == 0:
            self.pending = None
            self.apply(th, bars=bars, cut=bool(now), auto=auto)
        else:
            self.pending = dict(id=th["id"], fire=(math.floor(ctl.beat / q + 1e-6) + 1) * q, q=q, bars=bars, auto=auto)
        return True

    def apply(self, th, bars=None, cut=False, auto=False, hist=True):
        """Write a theme's scope into the live state now (morphing over `bars`, or a cut)."""
        ctl = self.ctl
        c = _ctl()
        scope, d = th["scope"], th["data"]
        bars = ctl.morph_bars if bars is None else float(bars)
        if cut:
            bars = 0.0
        if auto:
            bars = max(bars, AUTO_MIN_BARS)
        if hist and not ctl._tl_firing and not ctl._hist_mute:
            ctl.history.append(dict(key=ctl._hkey_once(), t=ctl.now(), old={}, comp=None, mods=None, stack=None,
                                    theme=dict(scope=dict(scope), data=self.capture_data(scope, missing_ok=True))))
            ctl._redo = []
        if scope["kind"] == "look":
            ctl._comp_clear()                                 # like a snapshot recall: no clip is on air any more
        vals = {n: v for n, v in d.get("values", {}).items() if n in c.CONT}
        names = list(vals)
        old_mods = {n: ctl.mods[n] for n in names if n in ctl.mods}
        for n in names:                                       # modulation: exactly what was captured
            pair = d.get("mods", {}).get(n)
            if pair:
                ctl.mods[n] = copy.deepcopy(pair)
            else:
                ctl.mods.pop(n, None)
        gone = set(names)
        if ctl._morph:
            ctl._morph["names"] -= gone                       # a running scene morph lets go of exactly these
        for lm in ctl._lane_morph.values():
            lm["names"] -= gone
        st = ctl.stack
        use_stack = st is not None and "stack" in d
        if bars <= 0:
            ctl._lane_morph.pop("theme", None)
            ctl.base.update(vals)
            if cut:                                           # a real cut: skip the fader slew and the palette glide
                for n in names:
                    ctl._sm[n] = vals[n]
                if "palette" in vals:
                    ctl._pal_target = vals["palette"]
        elif names or use_stack:
            ctl._lane_morph["theme"] = dict(src={n: ctl.base[n] for n in names}, to=vals, names=set(names),
                                            start=ctl.beat, dur=bars * 4.0, old=old_mods, stack=use_stack, t=0.0)
        if use_stack:
            ctl._stack_morph = False
            st.morph_cancel()
            if bars > 0:
                st.morph_start(d["stack"])
            else:
                st.load(d["stack"])
                if cut:
                    self._set_ph(d.get("stack_ph", {}))
        elif st is not None and "stack_layer" in d:
            self._apply_stack_layer(d, cut)
        sk = getattr(ctl, "skins", None)
        if sk is not None and "skins" in d:
            j = d["skins"]
            if scope["kind"] in ("layer", "group") and scope["key"] in ("l_land", "land"):
                cur = sk.to_json()                            # the deck worlds only: the tunnel skin stays
                keep = lambda k: _skin_env(k) in LAND_ENVS_EXCLUDE                          # noqa: E731
                j = dict(cur, slots=dict({k: v for k, v in cur.get("slots", {}).items() if keep(k)}, **{
                    k: v for k, v in j.get("slots", {}).items() if not keep(k)}),
                    mods=dict({k: v for k, v in cur.get("mods", {}).items() if keep(k)}, **{
                        k: v for k, v in j.get("mods", {}).items() if not keep(k)}))
            sk.load(copy.deepcopy(j))
        if ctl.text is not None and "text" in d:
            ctl.text.restore(copy.deepcopy(d["text"]))
        tex = d.get("media", {}).get("tex")
        if tex and tex != self.media.get("tex"):
            ctl._out.append(c.passthrough("show_texture", name=tex))
            self.media["tex"] = tex
        if not auto:                                          # a sound tag never switches a layer on / off
            for k, on in d.get("bypass", {}).items():
                if k in c.BYPASS_COVER:
                    ctl.set_bypass(k, on)
        if names:
            ctl.cur = -1                                      # the faders no longer match a saved snapshot
        if auto and ctl.auto.get("on") and scope["kind"] == "look":
            ctl._auto_next = ctl.beat + ctl.auto["bars"] * 4.0   # autopilot waits a period, like after a manual recall
        self._byp_seen = frozenset(ctl.bypass)
        if th.get("id"):
            self.current = th["id"]
        ctl._changed()

    def _set_ph(self, phs):
        st = self.ctl.stack
        for k, p in (phs or {}).items():
            try:
                i = int(k)
            except ValueError:
                continue
            if st.find(i) is not None:
                ph = st._phase(i)
                for f, v in p.items():
                    if f in STACK_PH:
                        ph[f] = float(v)

    def _apply_stack_layer(self, d, cut):
        st = self.ctl.stack
        lay = copy.deepcopy(d["stack_layer"])
        i = lay["id"]
        cur = st.find(i)
        new = cur is None
        if new:
            if len(st.layers) >= _ctl()._stack.MAX_LAYERS:
                self.say("warn", "the math stack is full: that stack theme cannot come back")
                return
            st.layers.append(lay)
            st.next = max(st.next, i + 1)
        else:
            st.layers[st.layers.index(cur)] = lay
        pre = f"{i}:"
        st.mods = {k: v for k, v in st.mods.items() if not k.startswith(pre)}
        st.mods.update(copy.deepcopy(d.get("stack_mods", {})))
        st.edits += 1
        if cut or new:
            self._set_ph(d.get("stack_ph", {}))
            if new:
                st._phase(i)["op"] = None                    # no fade-in from nothing on a cut

    def undo_apply(self, snap):
        """Controller.undo of a theme restore: put the pre-restore capture back (a cut). Returns the inverse."""
        inv = dict(scope=dict(snap["scope"]), data=self.capture_data(snap["scope"], missing_ok=True))
        self.ctl._lane_morph.pop("theme", None)
        gone = snap["data"].get("stack_absent")
        st = self.ctl.stack
        if gone is not None and st is not None:               # the restore had brought that layer back: remove it
            st.layers = [x for x in st.layers if x["id"] != gone]
            st.mods = {k: v for k, v in st.mods.items() if not k.startswith(f"{gone}:")}
            st.edits += 1
        else:
            self.apply(dict(id=None, scope=snap["scope"], data=snap["data"]), cut=True, hist=False)
        return inv

    def restore_preset(self, p, opts):
        """A theme preset (presets drawer / timeline cue) -> restore; it also joins the tray if it is not there."""
        body = (p.get("data") or {}).get("theme")
        if not isinstance(body, dict):
            return
        try:
            th = clean_theme(body)
        except ValueError:
            return
        ref = opts.get("ref")
        hit = next((x for x in self.tray if ref and x["ref"] == ref), None)
        bars = opts.get("morph")
        if hit is None:
            hit = th
            hit["id"] = None
        self.apply(hit, bars=bars, cut=bars is not None and float(bars) <= 0)

    # ---- performer / media / engine notes from the controller
    def manual(self, now=None):
        now = self.ctl.now() if now is None else now
        self._manual_t = now                                  # held while now < _manual_t + override (live setting)
        if self.pending and self.pending.get("auto"):
            self._log("block", self.find(self.pending["id"]), note="performer moved: pending recall dropped")
            self.pending = None

    def saw(self, c):
        """A validated command from the panel / assistant (Controller.apply)."""
        cmd = c.get("cmd")
        if cmd in PERFORMER_CMDS or (cmd == "theme" and c.get("op") == "restore"):     # a capture moves nothing
            self.manual()

    def note_media(self, cmds):
        for c in cmds:
            if not isinstance(c, dict):
                continue
            k = c.get("cmd") or c.get("type")
            if k == "show_texture" and isinstance(c.get("name"), str):
                self.media["tex"] = c["name"]
            elif k == "art_pick" and c.get("source") == "ai" and isinstance(c.get("name"), str) and c["name"]:
                self.media["tex"] = c["name"]

    def engine(self):
        return getattr(self.ctl, "soundtags", None)

    # ---- sound tags
    def _tags(self):
        return {th["id"]: th["tag"] for th in self.tray if th["tag"] and th["tag"]["ex"]}

    def _compile(self):
        eng = self.engine()
        if eng is None or _st is None:
            return
        eng.set_templates({th["id"]: _st.compile_tag(th["tag"]["ex"], th["tag"]["mode"], th["tag"]["key_inv"])
                           for th in self.tray if th["tag"] and th["tag"]["ex"] and th["tag"]["on"]})

    def tag(self, tid, on, length=None):
        th, eng = self.find(tid), self.engine()
        if th is None:
            return
        if on:
            if _st is None or eng is None or not eng.receiving(eng.clock()):
                self.say("error", "no audio is coming in: sound tags need the show's loopback capture")
                return
            self.rec = dict(id=tid, start=eng.count, t0=self.ctl.now(), len=float(length or TAG_LEN))
            self._bump()
        elif self.rec and self.rec["id"] == tid:
            self._rec_finish()

    def _rec_finish(self):
        r, self.rec = self.rec, None
        th, eng = self.find(r["id"]), self.engine()
        self._bump()
        if th is None or eng is None:
            return
        fr, loud = eng.frames_since(r["start"])
        try:
            ex = _st.make_example(fr, loud)
        except ValueError as e:
            self.say("warn", f"tag not saved: {e}")
            self._log("tag", th, note=f"not saved: {e}")
            return
        self._add_example(th, ex)
        self._log("tag", th, note=f"example {len(th['tag']['ex'])} ({ex['secs']:.0f} s)")
        self.say("ok", f"'{th['name']}' tagged ({len(th['tag']['ex'])} example(s))")

    def _add_example(self, th, ex):
        if th["tag"] is None:
            th["tag"] = dict(_st.TAG_DEFAULTS, ex=[])
        th["tag"]["ex"] = (th["tag"]["ex"] + [ex])[-_st.MAX_EXAMPLES:]
        self._compile()
        if th["star"]:
            self._star_save(th)
        self._bump()
        self.ctl._changed()

    def tag_set(self, tid, s):
        th = self.find(tid)
        if th is None or _st is None:
            return
        if th["tag"] is None:
            th["tag"] = dict(_st.TAG_DEFAULTS, ex=[])
        th["tag"].update(s)
        self._compile()
        if th["star"]:
            self._star_save(th)
        self._bump()
        self.ctl._changed()

    def tag_clear(self, tid, n=None):
        th = self.find(tid)
        if th is None or th["tag"] is None:
            return
        if n is None:
            th["tag"] = None
        elif 0 <= n < len(th["tag"]["ex"]):
            del th["tag"]["ex"][n]
        self._compile()
        if th["star"]:
            self._star_save(th)
        self._bump()
        self.ctl._changed()

    def blocked(self, now):
        ctl = self.ctl
        if now < self._panic_until:
            return "panic"
        if (ctl.held.get("blackout") or ctl.base.get("blackout", 0.0) > 0.0
                or (ctl.values or {}).get("blackout", 0.0) > 0.05 or "layers" in ctl.bypass
                or ctl.base.get("master", 1.0) < 0.02 or "blackout" in ctl._tl_trig):
            return "blackout"
        if now < self._manual_t + self.override_s:
            return "performer"
        if getattr(ctl, "fg_hold", False):
            return "flashguard"
        if getattr(ctl, "gov_level", 0) >= 2:
            return "governor"
        return None

    def suggest(self):
        eng = self.engine()
        if _st is None or eng is None:
            self.say("error", "no audio engine: nothing to suggest from")
            return
        fr, loud = eng.frames_since(0)
        self.suggestions = _st.suggest(fr, loud)
        self.say("ok" if self.suggestions else "warn", f"{len(self.suggestions)} recurring section(s) found"
                 if self.suggestions else "no section came back yet: let more of the set play")
        self._bump()

    def suggest_apply(self, sid, tid=None):
        if not 0 <= sid < len(self.suggestions):
            return
        th = self.find(tid) if tid else self.capture({"kind": "look", "key": None})
        if th is None:
            return
        self._add_example(th, copy.deepcopy(self.suggestions[sid]["example"]))
        self._log("tag", th, note=f"from suggestion {self.suggestions[sid].get('label', '')}")

    # ---- commands
    def do(self, c):
        op = c["op"]
        if op == "capture":
            self.capture(c["scope"], c.get("name"), c.get("slot"))
        elif op == "restore":
            th = self.find(c["id"]) if "id" in c else self.by_slot(c["slot"])
            if th is not None and self.ctl._tl_firing:       # a timeline event fires exactly where it was placed
                self.apply(th, bars=c.get("bars"), cut=bool(c.get("now")))
            elif th is not None:
                self._log("restore", th, note="cut" if c.get("now") else "")
                self.restore_theme(th, now=c.get("now", False), bars=c.get("bars"))
        elif op == "star":
            self.star(c["id"], c["on"])
        elif op == "rename":
            th = self.find(c["id"])
            if th is not None:
                th["name"] = c["name"]
                if th["star"]:
                    self._star_save(th)
                self._bump()
                self.ctl._changed()
        elif op == "recapture":
            self.recapture(c["id"])
        elif op == "delete":
            th = self.find(c["id"])
            if th is not None:                                # a starred theme's library copy stays (PRESETS)
                self._delete(th)
                self._bump()
                self.ctl._changed()
        elif op == "tag":
            self.tag(c["id"], c["on"], c.get("len"))
        elif op == "tag_set":
            self.tag_set(c["id"], c["set"])
        elif op == "tag_clear":
            self.tag_clear(c["id"], c.get("n"))
        elif op == "arm":
            if self.armed != c["on"]:
                self.armed = c["on"]
                if self.dec is not None:
                    self.dec.reset()                          # nothing fires the moment it is armed: re-arm first
                self._log("arm", None, note="armed" if c["on"] else "disarmed")
        elif op == "test":
            self.test_until = self.ctl.now() + TEST_S if c["on"] else -1e9
        elif op == "opts":
            if "override" in c:
                self.override_s = c["override"]
            if "gap" in c:
                self.gap_s = c["gap"]
            self._bump()
        elif op == "suggest":
            self.suggest()
        elif op == "suggest_apply":
            self.suggest_apply(c["sid"], c.get("id"))

    def midi(self, slot, shift=False):
        """MIDI target theme:<slot>: press = restore (quantized + morphed); Shift + press = capture the look there."""
        th = self.by_slot(slot)
        if shift:
            if th is not None:
                self.recapture(th["id"])
            else:
                self.capture({"kind": "look", "key": None}, slot=slot)
        elif th is not None:
            self._log("restore", th, note="midi")
            self.restore_theme(th)

    # ---- per frame (Controller.update, after the clip queue)
    def step(self, now, beat):
        ctl = self.ctl
        if ctl.panics != self._panics:                       # PANIC: nothing pending survives; auto recall holds
            self._panics = ctl.panics
            if self.pending:
                self._log("block", self.find(self.pending["id"]), note="PANIC: pending recall dropped")
            self.pending = None
            self._panic_until = now + PANIC_HOLD_S
            if self.dec is not None:
                self.dec.reset()
        e = sum(ctl.edits.values()) if ctl.edits else 0      # MIDI / panel fader moves (set() counts them)
        if self._edits is None:
            self._edits = e
        elif e != self._edits:
            self._edits = e
            self.manual(now)
        b = frozenset(ctl.bypass)                            # a bypass pad / switch is a performer action too
        if self._byp_seen is None:
            self._byp_seen = b
        elif b != self._byp_seen:
            self._byp_seen = b
            if not ctl._tl_firing:
                self.manual(now)
        if self.rec and now - self.rec["t0"] >= self.rec["len"]:
            self._rec_finish()
        p = self.pending
        if p:
            if p["fire"] - beat > p["q"] + 1e-6:             # the clock jumped back: do not wait a whole phrase
                p["fire"] = beat
            th = self.find(p["id"])
            if th is None:
                self.pending = None
            elif p["auto"] and self.blocked(now) is not None:
                self._log("block", th, note=f"{self.blocked(now)}: pending recall dropped")
                self.pending = None
            elif beat >= p["fire"] - 1e-9:
                self.pending = None
                self.apply(th, bars=p["bars"], auto=p["auto"])
        eng = self.engine()
        if eng is not self._eng:
            self._eng = eng
            self._compile()
        if eng is None or self.dec is None:
            self.meters, self.block = [], None
            return
        tags = self._tags()
        want = (self.armed or now < self.test_until) and bool(tags)
        eng.active = want
        if not want:
            self.meters, self.block = [], None if self.armed else "disarmed"
            return
        new = {}
        if eng.sims_t != self._sims_seen:
            self._sims_seen = eng.sims_t
            new = dict(eng.sims)
        blk = self.blocked(now)
        self.block = blk if self.armed else "disarmed"
        order = [th["id"] for th in self.tray if th["id"] in tags]
        fire, self.meters, held = self.dec.step(now, new, tags, armed=self.armed, block=blk,
                                                consume=blk in ("performer", "panic", "blackout"),
                                                autopilot=bool(ctl.auto.get("on")), gap=self.gap_s, order=order)
        for k, why in held:
            if self._held_logged.get(k) != why:
                self._held_logged[k] = why
                th = self.find(k)
                self._log("block", th, self.dec.st[k]["s"], why)
        if fire:
            th = self.find(fire)
            self._held_logged.pop(fire, None)
            sim = self.dec.st[fire]["s"]
            cfg = th["tag"]
            self._last_fire = (time.time(), th["id"], th["name"], sim, True)
            self._log("fire", th, sim, f"quant {cfg['quant']}")
            self.restore_theme(th, bars=cfg["morph"], auto=True, quant=cfg["quant"])

    # ---- panel
    def state(self):
        """Compact: the phone polls it every second (the tray itself is the /themes blob)."""
        now = self.ctl.now()
        out = {"rev": self.rev, "n": len(self.tray)}
        eng = self.engine()
        if self.armed:
            out["armed"] = True
        if now < self.test_until:
            out["test"] = True
        if eng is not None and eng.receiving(eng.clock()):
            out["audio"] = True
        if self.rec:
            out["rec"] = {"id": self.rec["id"], "t": round(now - self.rec["t0"], 1), "len": self.rec["len"]}
        if self.meters:
            out["m"] = self.meters
        if self.block and (self.armed or self.meters):
            out["block"] = self.block
            if self.block == "performer":
                out["susp"] = round(self._manual_t + self.override_s - now, 1)
        if self._last_fire:
            t, i, name, sim, auto = self._last_fire
            out["last"] = [round(time.time() - t, 1), i, name, round(sim, 3), auto]
        if self.pending:
            out["pend"] = self.pending["id"]
        return out

    def blob(self):
        if self._blob[0] != self.rev:
            import json
            tray = []
            for th in self.tray:
                tg = th["tag"]
                tray.append({"id": th["id"], "slot": th["slot"], "name": th["name"], "scope": th["scope"],
                             "label": scope_label(th["scope"]), "star": th["star"], "ref": th["ref"],
                             "t": round(th["t"], 1), "thumb": th["thumb"], "pal": th["pal"],
                             "tag": None if not tg else dict({k: tg[k] for k in _st.TAG_DEFAULTS}, n=len(tg["ex"]),
                                                             secs=round(sum(e["secs"] for e in tg["ex"]), 1))})
            sug = [{"sid": i, "label": s.get("label", ""), "secs": s["secs"], "n": s["n"], "at": s["at"]}
                   for i, s in enumerate(self.suggestions)]
            try:
                c = _ctl()
                scopes = {"layers": [[k, str(c.PARAMS[k][4])[:24]] for k in layer_keys()],
                          "groups": [[g, g.capitalize()] for g in group_keys()]}
                self._blob = (self.rev, json.dumps(dict(rev=self.rev, tray=tray, log=self.log, suggest=sug,
                                                        scopes=scopes, opts={"override": self.override_s,
                                                                             "gap": self.gap_s}),
                                                   allow_nan=False, separators=(",", ":")).encode())
            except (TypeError, ValueError):
                pass
        return self._blob
