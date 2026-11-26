"""Timeline: a moving playhead with tracks of things placed on it (pure python, no GL, no I/O).

The controller (control.py) owns one `Timeline`. Each frame it calls `advance(delta)` (delta = the clock's beat delta in
"beats" units, so the playhead follows tap / MIDI clock / Link; seconds otherwise), gets back the discrete items the
playhead crossed (cues, triggers, events, text) and asks `lane_value(lane, pos)` for every automation lane.

Data (JSON, saved inside a show file):
    {"v": 1, "units": "beats" | "seconds", "length": 256, "chase": true,
     "loop": {"on": false, "a": 0, "b": 32}, "punch": {"on": false, "a": 0, "b": 32},
     "audio": null | {"name": "song.mp3", "offset": 0.0},
     "next": <next free item / track id>,
     "tracks": [{"id", "kind", "name", "mute", "items": [...]} ...]}
Track kinds and their items (every item has a unique int "id" and a time "t"):
    marker   {t, name}                                   sections: intro / build / drop
    cue      {t, kind: "scene", slot, morph}             scene recall; morph in bars (0 = cut)
             {t, kind: "preset", preset: "look/<slug>", morph, groups: [..] | null}
    auto     {t, v, ease}  (the track has "param")       automation keyframe; ease = how to go on to the NEXT key
    trigger  {t, len, name}                              flash / punch / invert / freeze / blackout / strobe held for len
    event    {t, cmd: {...}}                             an allowlisted command (validated by the caller's validator)
    text     {t, text}                                   words on the text layer
Everything that comes from a file or the panel goes through `clean()` / `validate()`: pure data, bounded, typed.

Determinism: `lane_value` is a pure function of (lane, position); `advance` only adds floats. The same timeline driven by
the same sequence of deltas gives the same values and the same fired items, every time (tests/test_timeline.py).
The per-frame path allocates nothing new beyond Python floats: compiled lanes are reused, the fired list is reused.
"""
import bisect
import json
import math

VERSION = 1
UNITS = ("beats", "seconds")
EASES = ("hold", "linear", "smooth", "in", "out")
KINDS = ("marker", "cue", "auto", "trigger", "event", "text")
TRIGGERS = ("flash", "punch", "invert", "freeze", "blackout", "strobe")
FLASHY = ("flash", "invert", "blackout", "strobe")        # photosensitive: these change the whole frame's light
MAX_T = 1.0e6                                              # beats or seconds
MAX_TRACKS = 64
MAX_KEYS = 20000                                           # automation keys per lane
MAX_ITEMS = 4000                                           # items per discrete track
MAX_NAME, MAX_TEXT = 40, 200
MAX_UNDO = 50
MAX_TIMELINES = 16                                         # per show ("songs" / sets); one is active
REC_GAP = {"beats": 1.0 / 32.0, "seconds": 0.02}           # thinnest spacing of recorded keys
EPS = {"beats": 1.0 / 256.0, "seconds": 0.002}             # anchor offset around a recorded gesture
SIMPLIFY = 0.004                                           # RDP tolerance, fraction of the param's range
OPS = ("play", "stop", "seek", "home", "rec", "rec_mode", "ovr_mode", "reenable", "loop", "punch", "length", "units",
       "chase", "audio", "name", "release", "hold", "tl_new", "tl_dup", "tl_select", "tl_delete", "track_add", "track_del", "track_set", "track_move", "item_add", "items_add", "items_set",
       "items_del", "undo", "redo", "clear", "replace")
REC_MODES = ("overdub", "replace")
OVR_MODES = ("touch", "latch")
_EASE_I = {e: i for i, e in enumerate(EASES)}
PRESET_KINDS = ("look", "layer", "fx", "mod", "mapping", "midi")      # == shows.KINDS
_SLUG = __import__("re").compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")      # == shows.SLUG
_MEDIA = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,119}$")   # == shows.MEDIA_NAME


# ------------------------------------------------------------------ validation helpers


def _f(v, lo, hi, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"'{what}' must be a finite number")
    return min(hi, max(lo, float(v)))


def _i(v, lo, hi, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or int(v) != v or not lo <= v <= hi:
        raise ValueError(f"'{what}' must be a whole number {lo}..{hi}")
    return int(v)


def _b(v, what):
    if not isinstance(v, bool):
        raise ValueError(f"'{what}' must be true or false")
    return v


def _s(v, n, what, required=False):
    if v is None and not required:
        return ""
    if not isinstance(v, str) or len(v) > n or any(ord(ch) < 32 for ch in v) or (required and not v.strip()):
        raise ValueError(f"'{what}' must be text of at most {n} characters")
    return v.strip()


def preset_ref(v):
    """'<kind>/<slug>' with a plain slug (the preset store resolves it inside presets/)."""
    if not isinstance(v, str) or len(v) > 90 or v.count("/") != 1:
        raise ValueError("'preset' must be '<kind>/<name>'")
    kind, slug = v.split("/")
    if kind not in PRESET_KINDS or not _SLUG.match(slug):
        raise ValueError("'preset' must be '<kind>/<name>'")
    return v


class Ctx:
    """What validation needs to know about the rest of the system (injected by control.py, so this module imports
    nothing of the engine). params: name -> (lo, hi, default, group, label); discrete: stepped names;
    event: callable(cmd dict) -> clean cmd dict (raises ValueError for anything not allowlisted)."""

    def __init__(self, params, discrete=(), event=None, n_scenes=16, groups=None):
        self.params = params
        self.discrete = frozenset(discrete)
        self.event = event or (lambda c: (_ for _ in ()).throw(ValueError("events are not available")))
        self.n_scenes = n_scenes
        self.groups = frozenset(groups if groups is not None else (v[3] for v in params.values()))


def clean_item(kind, raw, ctx, track=None):
    """One item of a track of `kind` -> normalized dict (id kept if it is a valid int). ValueError if malformed."""
    if not isinstance(raw, dict):
        raise ValueError("item must be an object")
    out = {"t": _f(raw.get("t"), 0.0, MAX_T, "t")}
    if isinstance(raw.get("id"), int) and not isinstance(raw.get("id"), bool) and raw["id"] > 0:
        out["id"] = raw["id"]
    if kind == "marker":
        out["name"] = _s(raw.get("name", ""), MAX_NAME, "name")
    elif kind == "cue":
        k = raw.get("kind", "scene")
        out["morph"] = _f(raw.get("morph", 0.0), 0.0, 64.0, "morph")
        if k == "scene":
            out["kind"], out["slot"] = "scene", _i(raw.get("slot"), 0, ctx.n_scenes - 1, "slot")
        elif k == "preset":
            out["kind"], out["preset"] = "preset", preset_ref(raw.get("preset"))
            g = raw.get("groups")
            if g is not None:
                if not isinstance(g, list) or len(g) > 64 or any(x not in ctx.groups for x in g):
                    raise ValueError("'groups' must be a list of param groups")
                out["groups"] = list(dict.fromkeys(g))
        else:
            raise ValueError("cue 'kind' must be scene or preset")
    elif kind == "auto":
        name = track.get("param") if isinstance(track, dict) else None
        if name not in ctx.params:
            raise ValueError("automation lane has no param")
        lo, hi = ctx.params[name][0], ctx.params[name][1]
        out["v"] = _f(raw.get("v"), lo, hi, "v")
        e = raw.get("ease", "linear")
        if e not in EASES:
            raise ValueError("'ease' must be one of " + ", ".join(EASES))
        out["ease"] = "hold" if name in ctx.discrete else e
    elif kind == "trigger":
        if raw.get("name") not in TRIGGERS:
            raise ValueError("'name' must be a trigger")
        out["name"] = raw["name"]
        out["len"] = _f(raw.get("len", 0.25), 0.0, 4096.0, "len")
    elif kind == "event":
        out["cmd"] = ctx.event(raw.get("cmd"))
    elif kind == "text":
        out["text"] = _s(raw.get("text"), MAX_TEXT, "text", required=True)
    else:
        raise ValueError("unknown track kind")
    return out


def _clean_range(raw, length, default_b):
    r = raw if isinstance(raw, dict) else {}
    out = {"on": bool(r.get("on")) if isinstance(r.get("on"), bool) else False}
    try:
        a = _f(r.get("a", 0.0), 0.0, MAX_T, "a")
        b = _f(r.get("b", default_b), 0.0, MAX_T, "b")
    except ValueError:
        a, b = 0.0, default_b
    if b <= a:
        a, b = 0.0, max(default_b, 1e-3)
    out["a"], out["b"] = a, b
    return out


def empty(units="beats"):
    units = units if units in UNITS else "beats"
    return {"v": VERSION, "name": "Timeline 1", "units": units, "length": 256.0 if units == "beats" else 300.0, "chase": True,
            "loop": {"on": False, "a": 0.0, "b": 32.0 if units == "beats" else 30.0},
            "punch": {"on": False, "a": 0.0, "b": 32.0 if units == "beats" else 30.0},
            "audio": None, "next": 3,
            "tracks": [{"id": 1, "kind": "marker", "name": "Sections", "mute": False, "items": []},
                       {"id": 2, "kind": "cue", "name": "Cues", "mute": False, "items": []}]}


def clean(raw, ctx):
    """Untrusted timeline JSON -> a complete, valid timeline. Never raises: malformed parts are dropped, a malformed
    root gives the empty timeline. Ids are made unique; items are sorted by time."""
    if not isinstance(raw, dict):
        return empty()
    units = raw.get("units") if raw.get("units") in UNITS else "beats"
    out = empty(units)
    try:
        out["length"] = _f(raw.get("length", out["length"]), 1.0, MAX_T, "length")
    except ValueError:
        pass
    out["chase"] = raw["chase"] if isinstance(raw.get("chase"), bool) else True
    try:
        out["name"] = _s(raw.get("name", ""), MAX_NAME, "name") or out["name"]
    except ValueError:
        pass
    out["loop"] = _clean_range(raw.get("loop"), out["length"], out["loop"]["b"])
    out["punch"] = _clean_range(raw.get("punch"), out["length"], out["punch"]["b"])
    au = raw.get("audio")
    if isinstance(au, dict):
        try:
            out["audio"] = clean_audio(au)
        except ValueError:
            out["audio"] = None
    used, tracks, lanes = set(), [], set()
    for tr in (raw.get("tracks") if isinstance(raw.get("tracks"), list) else [])[:MAX_TRACKS]:
        try:
            t = clean_track(tr, ctx)
        except ValueError:
            continue
        if t["kind"] == "auto":
            if t["param"] in lanes:
                continue                                    # one lane per param
            lanes.add(t["param"])
        cap = MAX_KEYS if t["kind"] == "auto" else MAX_ITEMS
        items = []
        for it in (tr.get("items") if isinstance(tr.get("items"), list) else [])[:cap]:
            try:
                items.append(clean_item(t["kind"], it, ctx, t))
            except ValueError:
                continue
        t["items"] = items
        tracks.append(t)
    # unique ids (tracks + items share one counter)
    nxt = 1
    for t in tracks:
        for obj in [t] + t["items"]:
            i = obj.get("id")
            if not isinstance(i, int) or i in used:
                obj["id"] = None
            else:
                used.add(i)
                nxt = max(nxt, i + 1)
    for t in tracks:
        for obj in [t] + t["items"]:
            if obj["id"] is None:
                obj["id"] = nxt
                used.add(nxt)
                nxt += 1
        t["items"].sort(key=lambda it: (it["t"], it["id"]))
    if not any(t["kind"] == "marker" for t in tracks):
        tracks.insert(0, {"id": nxt, "kind": "marker", "name": "Sections", "mute": False, "items": []})
        nxt += 1
    if not any(t["kind"] == "cue" for t in tracks):
        tracks.insert(1, {"id": nxt, "kind": "cue", "name": "Cues", "mute": False, "items": []})
        nxt += 1
    out["tracks"], out["next"] = tracks, max(nxt, _i(raw.get("next", 1), 1, 2 ** 31, "next") if isinstance(
        raw.get("next"), int) and not isinstance(raw.get("next"), bool) and 0 < raw["next"] < 2 ** 31 else 1)
    return out


def clean_audio(au):
    name = au.get("name")
    if not isinstance(name, str) or not _MEDIA.match(name):
        raise ValueError("'audio.name' must be a media file name")
    return {"name": name, "offset": _f(au.get("offset", 0.0), -3600.0, 3600.0, "offset")}


def clean_track(tr, ctx):
    if not isinstance(tr, dict) or tr.get("kind") not in KINDS:
        raise ValueError("track 'kind' must be one of " + ", ".join(KINDS))
    t = {"id": tr["id"] if isinstance(tr.get("id"), int) and not isinstance(tr.get("id"), bool) and tr["id"] > 0 else None,
         "kind": tr["kind"], "name": _s(tr.get("name", ""), MAX_NAME, "name"),
         "mute": tr["mute"] if isinstance(tr.get("mute"), bool) else False, "items": []}
    if t["kind"] == "auto":
        if tr.get("param") not in ctx.params:
            raise ValueError("unknown param")
        t["param"] = tr["param"]
        if not t["name"]:
            t["name"] = str(ctx.params[t["param"]][4])[:MAX_NAME]
    return t


def validate(c, ctx):
    """{"cmd": "tl", "op": ...} from the panel -> clean copy. ValueError if malformed."""
    if not isinstance(c, dict) or c.get("op") not in OPS:
        raise ValueError("'op' must be one of " + ", ".join(OPS))
    op = c["op"]
    out = {"cmd": "tl", "op": op}
    if op == "seek":
        out["t"] = _f(c.get("t"), 0.0, MAX_T, "t")
    elif op in ("play",):
        if c.get("t") is not None:
            out["t"] = _f(c.get("t"), 0.0, MAX_T, "t")
    elif op == "rec":
        out["on"] = _b(c.get("on"), "on")
    elif op == "rec_mode":
        if c.get("mode") not in REC_MODES:
            raise ValueError("'mode' must be overdub or replace")
        out["mode"] = c["mode"]
    elif op == "ovr_mode":
        if c.get("mode") not in OVR_MODES:
            raise ValueError("'mode' must be touch or latch")
        out["mode"] = c["mode"]
    elif op in ("loop", "punch"):
        for k in ("a", "b"):
            if k in c:
                out[k] = _f(c[k], 0.0, MAX_T, k)
        if "on" in c:
            out["on"] = _b(c["on"], "on")
    elif op == "length":
        out["t"] = _f(c.get("t"), 1.0, MAX_T, "t")
    elif op == "units":
        if c.get("units") not in UNITS:
            raise ValueError("'units' must be beats or seconds")
        out["units"] = c["units"]
    elif op == "chase":
        out["on"] = _b(c.get("on"), "on")
    elif op == "name":
        out["name"] = _s(c.get("name"), MAX_NAME, "name", required=True)
    elif op == "hold":
        out["s"] = _f(c.get("s"), 0.25, 10.0, "s")
    elif op in ("tl_select", "tl_delete"):
        out["i"] = _i(c.get("i"), 0, MAX_TIMELINES - 1, "i")
    elif op == "audio":
        out["audio"] = None if c.get("audio") is None else clean_audio(c["audio"] if isinstance(c["audio"], dict) else {})
    elif op == "track_add":
        tr = clean_track(dict(kind=c.get("kind"), param=c.get("param"), name=c.get("name", "")), ctx)
        out.update(kind=tr["kind"], name=tr["name"])
        if "param" in tr:
            out["param"] = tr["param"]
    elif op in ("track_del", "track_set", "track_move"):
        out["id"] = _i(c.get("id"), 1, 2 ** 31, "id")
        if op == "track_set":
            if "name" in c:
                out["name"] = _s(c["name"], MAX_NAME, "name")
            if "mute" in c:
                out["mute"] = _b(c["mute"], "mute")
        if op == "track_move":
            out["to"] = _i(c.get("to"), 0, MAX_TRACKS - 1, "to")
    elif op in ("item_add", "items_add", "items_set", "items_del"):
        out["track"] = _i(c.get("track"), 1, 2 ** 31, "track")
        if op == "items_del":
            ids = c.get("ids")
            if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_KEYS:
                raise ValueError("'ids' must be a list")
            out["ids"] = [_i(x, 1, 2 ** 31, "id") for x in ids]
        elif op == "item_add":
            if not isinstance(c.get("item"), dict):
                raise ValueError("'item' must be an object")
            out["item"] = dict(c["item"])               # validated against the track's kind when applied
        else:
            items = c.get("items")
            if not isinstance(items, list) or not 1 <= len(items) <= MAX_KEYS or any(not isinstance(x, dict) for x in items):
                raise ValueError("'items' must be a list of objects")
            if op == "items_set" and any(not isinstance(x.get("id"), int) or isinstance(x.get("id"), bool) for x in items):
                raise ValueError("every item to change needs its 'id'")
            out["items"] = [dict(x) for x in items]
    elif op == "replace":
        if not isinstance(c.get("timeline"), dict):
            raise ValueError("'timeline' must be an object")
        out["timeline"] = clean(c["timeline"], ctx)
    return out


# ------------------------------------------------------------------ easing + simplification


def ease(e, f):
    """e = ease index. f in 0..1 -> 0..1."""
    if e == 0:                       # hold
        return 0.0
    if e == 1:                       # linear
        return f
    if e == 2:                       # smooth (smoothstep)
        return f * f * (3.0 - 2.0 * f)
    if e == 3:                       # in (accelerate)
        return f * f
    return 1.0 - (1.0 - f) * (1.0 - f)   # out (decelerate)


def simplify(pts, tol):
    """Ramer-Douglas-Peucker on [(t, v)] (sorted by t): keeps the end points, drops keys within tol of the line."""
    if len(pts) <= 2:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (ta, va), (tb, vb) = pts[a], pts[b]
        worst, wi = -1.0, -1
        for i in range(a + 1, b):
            t, v = pts[i]
            f = (t - ta) / (tb - ta) if tb > ta else 0.0
            d = abs(v - (va + (vb - va) * f))
            if d > worst:
                worst, wi = d, i
        if wi > 0 and worst > tol:
            keep[wi] = True
            stack.append((a, wi))
            stack.append((wi, b))
    return [p for p, k in zip(pts, keep) if k]


# ------------------------------------------------------------------ compiled lanes


class Lane:
    """An automation lane compiled for evaluation: parallel sorted lists (bisect, no per-frame allocation)."""
    __slots__ = ("tid", "param", "ts", "vs", "es", "mute", "lo", "hi")

    def __init__(self, track, lo, hi):
        self.tid, self.param, self.mute, self.lo, self.hi = track["id"], track["param"], track["mute"], lo, hi
        self.ts = [it["t"] for it in track["items"]]
        self.vs = [it["v"] for it in track["items"]]
        self.es = [_EASE_I[it["ease"]] for it in track["items"]]


def lane_value(lane, pos):
    """Automation value at `pos` (None for an empty lane). Before the first key = first value, after the last = last."""
    ts = lane.ts
    n = len(ts)
    if n == 0:
        return None
    i = bisect.bisect_right(ts, pos) - 1
    if i < 0:
        return lane.vs[0]
    if i >= n - 1:
        return lane.vs[n - 1]
    t0, t1 = ts[i], ts[i + 1]
    a = lane.vs[i]
    if t1 <= t0:
        return lane.vs[i + 1]
    return a + (lane.vs[i + 1] - a) * ease(lane.es[i], (pos - t0) / (t1 - t0))


# ------------------------------------------------------------------ the timeline


class Timeline:
    """Model + transport. Not thread-safe: the GL thread (controller) is the only caller."""

    def __init__(self, ctx, data=None):
        self.ctx = ctx
        self.data = clean(data, ctx) if data is not None else empty()
        self.playing = False
        self.pos = 0.0
        self.rec = False
        self.rec_mode = "overdub"
        self.ovr_mode = "touch"
        self.rev = 0                                   # bumped on every edit (the panel re-fetches the document)
        self.undo_stack, self.redo_stack = [], []
        self.lanes, self.discrete, self.trig = [], [], []   # compiled
        self.by_param = {}
        self._fired = []
        self.compile()

    # ---------------------------------------------------------------- compile / json
    def compile(self):
        self.lanes, self.discrete, self.by_param, self.trig = [], [], {}, []
        for t in self.data["tracks"]:
            if t["kind"] == "auto":
                p = self.ctx.params[t["param"]]
                lane = Lane(t, p[0], p[1])
                self.lanes.append(lane)
                self.by_param[t["param"]] = lane
            elif t["kind"] == "trigger":
                self.trig.append((t, [it["t"] for it in t["items"]], max((it["len"] for it in t["items"]), default=0.0)))
            elif t["kind"] != "marker":
                self.discrete.append((t, [it["t"] for it in t["items"]]))
        self.rev += 1

    def to_json(self):
        return _copy(self.data)

    def load(self, raw):
        self.data = clean(raw, self.ctx)
        self.undo_stack, self.redo_stack = [], []
        self.pos = min(self.pos, self.data["length"])
        self.compile()

    @property
    def units(self):
        return self.data["units"]

    def track(self, tid):
        for t in self.data["tracks"]:
            if t["id"] == tid:
                return t
        return None

    def first(self, kind):
        for t in self.data["tracks"]:
            if t["kind"] == kind:
                return t
        return None

    def _new_id(self):
        i = self.data["next"]
        self.data["next"] = i + 1
        return i

    # ---------------------------------------------------------------- undo
    def snapshot(self):
        self.undo_stack.append(_copy(self.data))
        del self.undo_stack[:-MAX_UNDO]
        self.redo_stack = []

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.data)
            self.data = self.undo_stack.pop()
            self.compile()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.data)
            self.data = self.redo_stack.pop()
            self.compile()

    # ---------------------------------------------------------------- transport
    def seek(self, t):
        self.pos = min(max(0.0, float(t)), self.data["length"])

    def in_punch(self, pos=None):
        p = self.data["punch"]
        pos = self.pos if pos is None else pos
        return not p["on"] or p["a"] <= pos < p["b"]

    def recording(self):
        return self.rec and self.playing and self.in_punch()

    def advance(self, d):
        """Move the playhead by d (>= 0) while playing; returns the reused list of (track, item) crossed, in time order.
        Half-open windows [from, to): an item exactly at the start of play fires once. Loop wraps; the end stops."""
        fired = self._fired
        fired.clear()
        if not self.playing or not d > 0.0 or not math.isfinite(d):
            return fired
        lp, L = self.data["loop"], self.data["length"]
        p0, p1 = self.pos, self.pos + d
        if lp["on"] and lp["b"] > lp["a"] and p0 < lp["b"] <= p1:
            span = lp["b"] - lp["a"]
            self._collect(p0, lp["b"])
            rest = (p1 - lp["b"]) % span
            self._collect(lp["a"], lp["a"] + rest)
            self.pos = lp["a"] + rest
        elif p1 >= L:
            self._collect(p0, L + 1e-9)
            self.pos = L
            self.playing = False
        else:
            self._collect(p0, p1)
            self.pos = p1
        return fired

    def _collect(self, a, b):
        if b <= a:
            return
        start = len(self._fired)
        for tr, ts in self.discrete:
            if tr["mute"] or not ts:
                continue
            i = bisect.bisect_left(ts, a)
            items = tr["items"]
            while i < len(ts) and ts[i] < b:
                self._fired.append((tr, items[i]))
                i += 1
        if len(self.discrete) > 1 and len(self._fired) - start > 1:
            seg = self._fired[start:]
            seg.sort(key=lambda x: x[1]["t"])
            self._fired[start:] = seg

    def active_triggers(self, pos, out):
        """Declarative trigger spans: fills `out` (a reused list) with the items where t <= pos < t + len. A span is a
        pure function of the position, so stop / seek / loop can never leave a trigger stuck on."""
        out.clear()
        for tr, ts, maxlen in self.trig:
            if tr["mute"] or not ts:
                continue
            items = tr["items"]
            i = bisect.bisect_right(ts, pos) - 1
            while i >= 0 and ts[i] + maxlen > pos:
                it = items[i]
                if pos < it["t"] + max(it["len"], 1e-6):
                    out.append(it)
                i -= 1
        return out

    def chase_items(self, pos, inclusive=False):
        """What a seek / play-from-here must re-establish (Ableton chases, Millumin evaluates at t): the latest cue,
        the latest text and, per bypass key / media kind, the latest event before pos. One-shots (flash, next video,
        ripples) are never chased."""
        best = {}
        for tr, ts in self.discrete:
            if tr["mute"] or not ts:
                continue
            i = (bisect.bisect_right if inclusive else bisect.bisect_left)(ts, pos) - 1
            while i >= 0:
                it = tr["items"][i]
                if tr["kind"] in ("cue", "text"):
                    key = tr["kind"]
                else:
                    c = it["cmd"]
                    key = ("bypass", c.get("key")) if c["cmd"] == "bypass" else (
                        "media" if c["cmd"] in ("show_texture", "art_pick") else None)
                if key is not None and (key not in best or it["t"] > best[key][1]["t"]):
                    best[key] = (tr, it)
                if tr["kind"] in ("cue", "text"):
                    break
                i -= 1
        return sorted(best.values(), key=lambda x: x[1]["t"])

    def last_cue_before(self, pos):
        """The cue (track, item) at or before pos on the first unmuted cue track, for chase-on-seek."""
        for tr, ts in self.discrete:
            if tr["kind"] != "cue" or tr["mute"] or not ts:
                continue
            i = bisect.bisect_right(ts, pos) - 1
            if i >= 0:
                return tr, tr["items"][i]
        return None

    def upcoming(self, n=3):
        """Next n cues / markers after the playhead: [(t, label)] for the panel's next-cue strip."""
        out = []
        for t in self.data["tracks"]:
            if t["kind"] not in ("cue", "marker") or t["mute"]:
                continue
            for it in t["items"]:
                if it["t"] >= self.pos - 1e-9:
                    out.append((it["t"], label(t, it)))
        out.sort(key=lambda x: x[0])
        return out[:n]

    def section(self):
        """Name of the marker at or before the playhead ('' if none)."""
        m = self.first("marker")
        name = ""
        for it in (m["items"] if m else ()):
            if it["t"] <= self.pos + 1e-9:
                name = it["name"]
            else:
                break
        return name

    # ---------------------------------------------------------------- edits
    def do(self, c):
        """Apply one validated EDIT op (transport ops are handled by the controller). Returns True if it changed."""
        op = c["op"]
        if op == "undo":
            self.undo()
            return True
        if op == "redo":
            self.redo()
            return True
        d = self.data
        if op == "replace":
            self.snapshot()
            self.data = clean(c["timeline"], self.ctx)
        elif op == "clear":
            self.snapshot()
            units = d["units"]
            self.data = empty(units)
        elif op in ("loop", "punch"):
            r = dict(d[op], **{k: c[k] for k in ("a", "b", "on") if k in c})
            if r["b"] <= r["a"]:
                return False
            self.snapshot()
            d[op] = r
        elif op == "length":
            self.snapshot()
            d["length"] = c["t"]
            self.pos = min(self.pos, d["length"])
        elif op == "chase":
            d["chase"] = c["on"]
        elif op == "name":
            d["name"] = c["name"]
        elif op == "audio":
            self.snapshot()
            d["audio"] = c["audio"]
        elif op == "track_add":
            if len(d["tracks"]) >= MAX_TRACKS or (c["kind"] == "auto" and c["param"] in self.by_param):
                return False
            self.snapshot()
            tr = {"id": self._new_id(), "kind": c["kind"], "name": c["name"], "mute": False, "items": []}
            if c["kind"] == "auto":
                tr["param"] = c["param"]
            d["tracks"].append(tr)
        elif op == "track_del":
            tr = self.track(c["id"])
            if tr is None:
                return False
            self.snapshot()
            d["tracks"].remove(tr)
        elif op == "track_set":
            tr = self.track(c["id"])
            if tr is None:
                return False
            self.snapshot()
            for k in ("name", "mute"):
                if k in c:
                    tr[k] = c[k]
        elif op == "track_move":
            tr = self.track(c["id"])
            if tr is None:
                return False
            self.snapshot()
            d["tracks"].remove(tr)
            d["tracks"].insert(min(c["to"], len(d["tracks"])), tr)
        elif op in ("item_add", "items_add"):
            tr = self.track(c["track"])
            if tr is None:
                return False
            raws = [c["item"]] if op == "item_add" else c["items"]
            cap = MAX_KEYS if tr["kind"] == "auto" else MAX_ITEMS
            new = []
            for raw in raws:
                try:
                    it = clean_item(tr["kind"], raw, self.ctx, tr)
                except ValueError:
                    continue
                it["id"] = self._new_id()
                new.append(it)
            if not new or len(tr["items"]) + len(new) > cap:
                return False
            self.snapshot()
            if tr["kind"] == "auto":                   # a key at the same time as an existing one replaces it
                times = {it["t"] for it in new}
                tr["items"] = [it for it in tr["items"] if it["t"] not in times]
            tr["items"] += new
            tr["items"].sort(key=lambda it: (it["t"], it["id"]))
            self.last_added = [it["id"] for it in new]
        elif op == "items_set":
            tr = self.track(c["track"])
            if tr is None:
                return False
            by_id = {it["id"]: it for it in tr["items"]}
            changes = []
            for patch in c["items"]:
                old = by_id.get(patch["id"])
                if old is None:
                    continue
                try:
                    it = clean_item(tr["kind"], dict(old, **{k: v for k, v in patch.items() if k != "id"}), self.ctx, tr)
                except ValueError:
                    continue
                it["id"] = old["id"]
                changes.append(it)
            if not changes:
                return False
            self.snapshot()
            for it in changes:
                by_id[it["id"]] = it
            tr["items"] = sorted(by_id.values(), key=lambda it: (it["t"], it["id"]))
        elif op == "items_del":
            tr = self.track(c["track"])
            if tr is None:
                return False
            ids = set(c["ids"])
            keep = [it for it in tr["items"] if it["id"] not in ids]
            if len(keep) == len(tr["items"]):
                return False
            self.snapshot()
            tr["items"] = keep
        elif op == "units":
            return False                               # needs the tempo: Controller converts and calls convert_units
        else:
            return False
        self.compile()
        return True

    def convert_units(self, units, bpm):
        """Switch beats <-> seconds, converting every time at the given (constant) tempo."""
        d = self.data
        if units == d["units"] or not bpm > 0:
            return False
        f = 60.0 / bpm if units == "seconds" else bpm / 60.0
        self.snapshot()
        d["units"] = units
        d["length"] = min(MAX_T, max(1.0, d["length"] * f))
        for k in ("loop", "punch"):
            d[k]["a"], d[k]["b"] = d[k]["a"] * f, d[k]["b"] * f
        for t in d["tracks"]:
            for it in t["items"]:
                it["t"] = min(MAX_T, it["t"] * f)
                if "len" in it:
                    it["len"] *= f
        self.pos = min(self.pos * f, d["length"])
        self.compile()
        return True

    # ---------------------------------------------------------------- recording (called by the controller)
    def ensure_lane(self, param):
        lane = self.by_param.get(param)
        if lane is not None:
            return self.track(lane.tid)
        if len(self.data["tracks"]) >= MAX_TRACKS or param not in self.ctx.params:
            return None
        tr = {"id": self._new_id(), "kind": "auto", "name": str(self.ctx.params[param][4])[:MAX_NAME],
              "param": param, "mute": False, "items": []}
        self.data["tracks"].append(tr)
        self.compile()
        return tr

    def write_key(self, tr, t, v, sweep_from):
        """Recording: drop this lane's keys in (sweep_from, t] and put (t, v) there. No compile (the caller batches)."""
        items = tr["items"]
        if len(items) >= MAX_KEYS:
            return None
        ts = [it["t"] for it in items]
        a = bisect.bisect_right(ts, sweep_from)
        b = bisect.bisect_right(ts, t)
        key = {"id": self._new_id(), "t": t, "v": v, "ease": "hold" if tr["param"] in self.ctx.discrete else "linear"}
        items[a:b] = [key]
        return key

    def add_item(self, kind, item):
        """Recording a cue / trigger: append to the first track of that kind (created if missing)."""
        tr = self.first(kind)
        if tr is None:
            if len(self.data["tracks"]) >= MAX_TRACKS:
                return None
            tr = {"id": self._new_id(), "kind": kind, "name": kind.title() + "s", "mute": False, "items": []}
            self.data["tracks"].append(tr)
        try:
            it = clean_item(kind, item, self.ctx, tr)
        except ValueError:
            return None
        if len(tr["items"]) >= MAX_ITEMS:
            return None
        it["id"] = self._new_id()
        ts = [x["t"] for x in tr["items"]]
        tr["items"].insert(bisect.bisect_right(ts, it["t"]), it)
        self.compile()
        return it

    def clear_range(self, kinds, a, b, keep_ids):
        """Replace-mode recording: remove items of these track kinds with a <= t < b (except this pass's own)."""
        changed = False
        for tr in self.data["tracks"]:
            if tr["kind"] in kinds and tr["items"]:
                keep = [it for it in tr["items"] if not (a <= it["t"] < b) or it["id"] in keep_ids]
                if len(keep) != len(tr["items"]):
                    tr["items"] = keep
                    changed = True
        if changed:
            self.compile()
        return changed

    def simplify_range(self, tr, a, b):
        """After a recorded gesture: thin its keys in [a, b] (RDP), keeping both ends."""
        if tr["param"] in self.ctx.discrete:
            return
        lo, hi = self.ctx.params[tr["param"]][0], self.ctx.params[tr["param"]][1]
        items = tr["items"]
        inside = [it for it in items if a <= it["t"] <= b]
        if len(inside) <= 2:
            return
        kept = simplify([(it["t"], it["v"]) for it in inside], SIMPLIFY * (hi - lo))
        keep_t = {p[0] for p in kept}
        tr["items"] = [it for it in items if not (a <= it["t"] <= b) or it["t"] in keep_t]


def _copy(d):
    """Deep copy of plain JSON data (the C json codec beats copy.deepcopy several times on 20k keys)."""
    return json.loads(json.dumps(d, separators=(",", ":")))


def label(track, it):
    k = track["kind"]
    if k == "marker":
        return it.get("name") or "marker"
    if k == "cue":
        if it.get("kind") == "scene":
            return f"scene {it['slot'] + 1}"
        return "preset " + it.get("preset", "").split("/")[-1]
    if k == "trigger":
        return it.get("name", "")
    if k == "text":
        return '"' + it.get("text", "")[:20] + '"'
    if k == "event":
        return str(it.get("cmd", {}).get("cmd", "event"))
    return k
