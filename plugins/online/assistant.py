"""Chat assistant: turns "make it calmer and greener" into validated control commands.

Threads: `ask()` (GL thread, via a panel command) builds the prompt from the live state and queues a job. ONE worker
thread runs the backend CLI (seconds to minutes). `tick()` (GL thread, every frame, costs nothing when idle) collects
finished replies, validates them (safety.py) against the state AT THAT MOMENT, and either shows a proposal (SUGGEST)
or applies it (AUTOPILOT). Only queues cross threads; the controller is only ever touched on the GL thread. The reply
is capped, scrubbed and parsed on the worker thread, and folder listings / CLI checks never run on the GL thread.

Every applied batch records what it replaced, so it can be undone from the history list.

SEE THE OUTPUT (see.py): with `see`, the GL thread asks the engine for ONE picture of the guarded output (`capture`
hook -> {"cmd": "assist_snap"}); it arrives through `snapshot_in` (any thread, a queue) and is attached to the request
in `tick`. A picture that could show people is held until the performer confirms it. The LOOK-AND-ADJUST loop is a
small state machine in `tick`: look -> ask -> apply (SUGGEST: the performer; AUTOPILOT: at once) -> settle -> look ...
at most n model calls, each one through the same request limit, allowlist, clamps and undo history.
"""
import collections
import copy
import json
import queue
import re
import threading
import time
from pathlib import Path

import control
from params import PARAMS, TRIGGERS

from . import safety, see

MIN_APPLY_S = 1.0            # s between applied batches (with one value per param per batch: <= 1 change / s / param)
ASK_LIMIT = (6, 60.0)        # chat requests per window
GEN_LIMIT = (4, 60.0)        # assistant-started image jobs per window
MAX_QUEUE = 3
MAX_MSGS = 40
MAX_HISTORY = 20
SHOWN_MSGS = 12


class RateLimiter:
    def __init__(self, n, per_s, clock=time.monotonic):
        self.n, self.per, self.clock, self._t = int(n), float(per_s), clock, collections.deque()

    def allow(self):
        now = self.clock()
        while self._t and now - self._t[0] >= self.per:
            self._t.popleft()
        if len(self._t) >= self.n:
            return False
        self._t.append(now)
        return True


class OnlineLog:
    """Append-only JSON lines in logs/online_<date>.log, written by its own thread (never by the render loop)."""

    def __init__(self, folder=None):
        self.folder = Path(folder) if folder else None
        self.recent = collections.deque(maxlen=200)
        self._q = queue.Queue()
        self._thread = None
        if self.folder is not None:
            self._thread = threading.Thread(target=self._run, name="online-log", daemon=True)
            self._thread.start()

    def log(self, kind, **fields):
        rec = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), kind=kind, **fields)
        self.recent.append(rec)
        if self._thread is not None:
            self._q.put(rec)

    def _run(self):
        while True:
            rec = self._q.get()
            if rec is None:
                return
            try:
                self.folder.mkdir(parents=True, exist_ok=True)
                with open(self.folder / f"online_{time.strftime('%Y%m%d')}.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass

    def close(self):
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=2)
            self._thread = None


# ------------------------------------------------------------------------------------------------ the prompt

SCHEMA = """Reply with ONE JSON object and nothing else (no markdown, no prose outside it):
{"say": "<one or two short sentences for the performer>", "commands": [ ... ]}

Allowed commands (anything else is discarded):
{"cmd":"set","name":"<param>","v":<number>}             set a fader / selector (clamped to its range)
{"cmd":"nudge","name":"<param>","d":<number>}           relative change
{"cmd":"palette","i":<number>}                          colour palette (fractions blend two palettes)
{"cmd":"morph","bars":<0..64>}                          how long scene recalls morph
{"cmd":"scene_recall","slot":<1..16>}                   only a saved slot from the list below
{"cmd":"scene_save","slot":<1..16>}                     ONLY when the performer asks to save
{"cmd":"bypass","key":"<key>","enabled":true|false}     switch a layer / tab off (false) or back on (true)
{"cmd":"art_pick","source":"art"|"ai","name":"<file>","use":<0..5>}   put a listed image on show
{"cmd":"text","text":"<words>"}                         words on the text layer (raise l_text to see them)
{"cmd":"gen","prompt":"<image prompt>","from_frame":false}            start an AI image (at most 2)

Rules: use only param names, scene slots, bypass keys and file names listed below. Small, tasteful changes unless asked
for more. A layer shows only while its l_* fader is up. You cannot flash, strobe, invert, freeze, black out, touch
PANIC, tempo, mapping or files: do not try. If the request needs none of the commands, answer in "say" with
"commands": []. Never put anything but the JSON object in your reply."""


def _fmt(v):
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"


def describe_params(base):
    lines = []
    for name, (lo, hi, _d, group, label) in PARAMS.items():
        if name in TRIGGERS:
            continue
        opts = control.OPTIONS.get(name) or []
        line = f"{name} [{group}] {_fmt(lo)}..{_fmt(hi)} now {_fmt(base.get(name, _d))} \"{label}\""
        if opts:
            line += " options: " + ", ".join(f"{i + int(lo)}={o}" for i, o in enumerate(opts))
        lines.append(line)
    return "\n".join(lines)


def build_prompt(ctl, user_text, tex=(), art=(), chat=(), see_txt=""):
    """The whole request as one text: role, strict schema, params registry + current values, scenes, libraries."""
    scenes = [f'{i + 1} "{(s.get("name") or "").strip() or "unnamed"}"' for i, s in enumerate(ctl.scenes) if s]
    groups = sorted(k for k in control.BYPASS_COVER if k not in PARAMS)
    text_now = ""
    try:
        if ctl.text is not None:
            text_now = str((ctl.text.state() or {}).get("text", ""))[:120]
    except Exception:
        pass
    parts = [
        "You are the lighting / visuals assistant inside a live psychedelic projection-mapping show. The performer "
        "types a wish; you translate it into control commands for the visual engine. You have no tools: your only "
        "output is the JSON reply described here.",
        SCHEMA,
        "PARAMS (name [group] min..max now <current fader value> \"label\"):\n" + describe_params(ctl.base),
        f"TEMPO: {ctl.bpm:.1f} BPM (read-only).  SCENE MORPH: {ctl.morph_bars:g} bars.",
        "SAVED SCENES (slot \"name\"): " + (", ".join(scenes) or "none")
        + (f".  Current: slot {ctl.cur + 1}" if ctl.cur >= 0 else ""),
        "BYPASS KEYS: any l_* layer name or bypassable param, or a whole tab: " + ", ".join(groups)
        + ".  Switched off now: " + (", ".join(sorted(ctl.bypass)) or "nothing"),
        "AI TEXTURES (source \"ai\", newest first): " + (", ".join(list(tex)[:16]) or "none"),
        "UPLOADED ART (source \"art\"): " + (", ".join(list(art)[:16]) or "none"),
    ]
    if text_now:
        parts.append(f"TEXT ON SHOW: {text_now!r}")
    recent = [f"{m['role']}: {m['text'][:200]}" for m in list(chat)[-4:] if m.get("role") in ("user", "assistant")]
    if recent:
        parts.append("EARLIER IN THIS CHAT:\n" + "\n".join(recent))
    parts.append("PERFORMER: " + user_text.strip())
    if see_txt:
        parts.append(see_txt)
    return "\n\n".join(parts)


# ------------------------------------------------------------------------------------------------ diff + undo

def _opt(name, v):
    opts = control.OPTIONS.get(name) or []
    i = int(round(v - PARAMS[name][0]))
    return opts[i] if 0 <= i < len(opts) else f"{v:.2f}"


def describe(ctl, cmds):
    """Readable before -> after lines for a validated batch, against the controller's state right now."""
    out = []
    for c in cmds:
        k = c["cmd"]
        if k == "set":
            n = c["name"]
            out.append(f"{PARAMS[n][4]} ({n}): {_opt(n, ctl.base[n])} → {_opt(n, c['v'])}")
        elif k == "morph":
            out.append(f"Scene morph time: {ctl.morph_bars:g} → {c['bars']:g} bars")
        elif k == "scene_recall":
            nm = (ctl.scenes[c["i"]] or {}).get("name", "")
            out.append(f"Recall scene {c['i'] + 1}" + (f' "{nm}"' if nm else ""))
        elif k == "scene_save":
            old = ctl.scenes[c["i"]]
            out.append(f"SAVE the current look into scene slot {c['i'] + 1}"
                       + (f' (overwrites "{old.get("name") or "unnamed"}")' if old else " (empty slot)"))
        elif k == "bypass":
            out.append(f"Switch {'OFF' if c['on'] else 'ON'}: {c['key']}")
        elif k == "art_pick":
            use = f", use: {_opt('art_use', c['use'])}" if "use" in c and "art_use" in PARAMS else ""
            out.append(f"Show image {c['name']} ({'AI texture' if c['source'] == 'ai' else 'art piece'}{use})")
        elif k == "text":
            out.append(f"Text on show: \"{c.get('text', '')[:80]}\"")
        elif k == "gen":
            out.append(f"Generate an AI image{' from the current frame' if c.get('init') else ''}: "
                       f"\"{c['prompt'][:100]}\" (cannot be undone)")
    return out


ART_PARAMS = ("art_source", "art_index", "art_use", "l_art", "l_ai")
EMAIL_RX = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def redact(text):
    """Scrub what a CLI may know about the person from text that goes to panels and the log: e-mail addresses (the
    Claude CLI puts the account's address into the model's context by itself) and the home folder path."""
    text = EMAIL_RX.sub("[email]", text)
    home = str(Path.home())
    for h in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:
        if len(h) > 3:
            text = text.replace(h, "~")
    return text


def quiet_set(ctl, name, v):
    """A fader move that is NOT the performer's hand: no `edits` count, so the resource governor keeps whatever it
    has reduced (Controller.set would make it stand down). Same clamp, same "a move overrides the morph" rule."""
    ctl.base[name] = min(PARAMS[name][1], max(PARAMS[name][0], float(v)))
    if ctl._morph:
        ctl._morph["names"].discard(name)


def capture(ctl, cmds):
    """What this batch is about to replace (only what it touches), and what it will leave behind (`after`), so an
    undo can tell which of it the performer has changed since."""
    b = dict(base={}, after={}, bypass={}, scenes={})
    for c in cmds:
        k = c["cmd"]
        if k == "set":
            b["base"].setdefault(c["name"], ctl.base[c["name"]])
            b["after"][c["name"]] = c["v"]
        elif k == "morph":
            b.setdefault("morph", ctl.morph_bars)
        elif k == "scene_recall":
            sc = ctl.scenes[c["i"]] or {}
            b.setdefault("full", dict(base={n: ctl.base[n] for n in control.CONT}, mods=copy.deepcopy(ctl.mods),
                                      cur=ctl.cur, slot=c["i"], to=dict(sc.get("base") or {}),
                                      to_mods=copy.deepcopy(sc.get("mods") or {})))
            b.setdefault("morph", ctl.morph_bars)        # fade_guards may lengthen it
        elif k == "scene_save":
            b["scenes"].setdefault(c["i"], copy.deepcopy(ctl.scenes[c["i"]]))
            b.setdefault("cur", ctl.cur)
        elif k == "bypass":
            b["bypass"].setdefault(c["key"], (c["key"] in ctl.bypass, c["on"]))
        elif k == "text" and ctl.text is not None:
            b.setdefault("text", copy.deepcopy(ctl.text.to_json()))
    return b


def restore(ctl, b):
    """Undo one batch WITHOUT cuts and without touching what the performer changed afterwards.

    Faders (and selectors, which flip at the half-way point like in any scene morph) travel back through a morph of
    at least MIN_MORPH_S. A param is only restored if it still holds the value the batch gave it. A recall the
    performer has queued stays queued; a morph the performer started since is never replaced. Layers come back
    through the bypass fade (at least MIN_FADE_S). Returns (notes, fade the caller should put back later or None).
    """
    notes = []
    mo = ctl._morph
    full = b.get("full")
    target = {}                                          # param -> value to travel back to
    after = dict(full["to"]) if full else {}
    after.update(b.get("after", {}))
    before = dict(full["base"]) if full else {}
    before.update(b.get("base", {}))
    ours = bool(mo) and all(abs(mo["to"].get(n, 1e9) - after.get(n, -1e9)) < 1e-6 for n in mo["names"])
    foreign = bool(mo) and not ours
    for n, old in before.items():
        if n not in after or n not in ctl.base:
            continue
        cur = ctl.base[n]
        in_flight = ours and n in mo["names"]
        if in_flight or abs(cur - after[n]) < 1e-6:
            if abs(cur - old) > 1e-9 or in_flight:
                target[n] = old
    skipped = [n for n in before if n in after and n not in target and abs(ctl.base.get(n, 0.0) - before[n]) > 1e-6]
    if foreign and target:
        notes.append("a scene change is in progress: faders were left alone")
        target = {}
    elif skipped:
        notes.append(f"{len(skipped)} fader(s) you moved since were left alone")
    if target:
        bars = max(safety.MIN_MORPH_S * max(1.0, float(ctl.bpm)) / 240.0, 1e-3)
        ctl._morph = dict(src={n: ctl.base[n] for n in target}, to=dict(target), names=set(target),
                          start=ctl.beat, dur=bars * 4.0, old=[], t=0.0)
    if full and ctl.pending is not None and ctl.pending[0] == full["slot"] and ctl.cur != full["slot"]:
        ctl.pending = None                               # the batch's own recall, still waiting for its bar: cancel it
    if full and not foreign:
        if ctl.cur == full["slot"]:
            ctl.cur = full["cur"]
        if ctl.mods == full["to_mods"]:                  # modulation untouched since the recall: put the old set back
            ctl.mods = copy.deepcopy(full["mods"])
    if "morph" in b:
        ctl.morph_bars = b["morph"]
    fade_back = None
    flips = [(k, was) for k, (was, now) in b.get("bypass", {}).items() if (k in ctl.bypass) == now and was != now]
    if flips:
        orig = ctl.bypass_fade
        if orig < safety.MIN_FADE_S:                     # the layer comes back as a fade, never as a cut ...
            ctl.bypass_fade = safety.MIN_FADE_S
            fade_back = orig                             # ... and the performer's own fade time returns afterwards
        for k, was in flips:
            ctl.set_bypass(k, was)
    if b.get("scenes"):
        for i, old in b["scenes"].items():
            ctl.scenes[i] = copy.deepcopy(old)
        if "cur" in b:
            ctl.cur = b["cur"]
        ctl._save_scenes()
    if "text" in b and ctl.text is not None:
        ctl.text.restore(b["text"])
    return notes, fade_back


# ------------------------------------------------------------------------------------------------ the assistant

class Assistant:
    def __init__(self, backends, log=None, lister=None, clock=time.monotonic, sync=False, mode="suggest",
                 backend="claude"):
        self.backends = dict(backends)                 # name -> object with .label, .available(), .ask(prompt, ...)
        self.log = log or OnlineLog(None)
        self.lister = lister or (lambda: ([], []))     # -> CACHED (texture names newest first, art names): no disk
        self.rescan = lambda: None                     # plugin hook, called on the WORKER thread before each job
        self.clock, self.sync = clock, bool(sync)
        self.mode, self.backend = mode, backend
        self.msgs = collections.deque(maxlen=MAX_MSGS)
        self.props = {}                                # id -> proposal
        self.history = collections.deque(maxlen=MAX_HISTORY)
        self.note = ""
        self._id = 0
        self._jobs = queue.Queue()
        self._done = queue.Queue()
        self._queued = 0
        self._busy = None                              # dict(id, backend, t0, chars) while a CLI runs
        self._events = set()                           # cancel events of every job queued or running (one per job)
        self._gen = 0                                  # bumped by cancel / panic: older replies are dropped
        self._last_apply = -1e9
        self._fade_back = None                         # (deadline, performer's bypass_fade, the value we set meanwhile)
        self._ask_rate = RateLimiter(*ASK_LIMIT, clock=clock)
        self._gen_rate = RateLimiter(*GEN_LIMIT, clock=clock)
        self._avail = None                             # name -> bool, filled off the render thread (refresh)
        self._stop = False
        self._thread = None
        # SEE THE OUTPUT
        self.snaps = see.SnapStore()                   # pictures, memory only (the panel shows the exact bytes sent)
        self.capture = lambda ctl: False               # plugin hook (GL thread): ask the engine for one picture
        self._snap_q = queue.Queue(maxsize=4)          # engine -> GL thread
        self._await = None                             # the request waiting for its picture
        self._held = {}                                # confirm id -> request held back (people could be in it)
        self.loop = None                               # the look-and-adjust loop, or None

    # ---------------------------------------------------------------- helpers
    def _next(self):
        self._id += 1
        return self._id

    def _msg(self, role, text, **kw):
        m = dict(id=self._next(), role=role, text=str(text)[:2000], **kw)
        self.msgs.append(m)
        return m

    def refresh(self):
        """Which backends can run. May start `claude --help` once, so it is called off the render thread."""
        out = {}
        for name, be in self.backends.items():
            try:
                out[name] = bool(be.available())
            except Exception:
                out[name] = False
        self._avail = out
        return out

    def available(self, name):
        return bool((self._avail or {}).get(name))

    # ---------------------------------------------------------------- GL thread: requests
    def ask(self, ctl, text, backend=None, see_out=False, loop=0):
        name = backend or self.backend
        um = self._msg("user", text)
        if name not in self.backends:
            self._msg("system", f"Unknown backend '{name}'.")
            return False
        if name in getattr(self, "blocked", {}):
            self._msg("system", f"{self.backends[name].label} is switched off. {self.blocked[name]}")
            return False
        if self._avail is None:
            self._msg("system", "The assistant is still starting: try again in a moment.")
            return False
        if not self.available(name):
            why = getattr(self.backends[name], "problem", "") or "it is not installed on this laptop"
            self._msg("system", f"{self.backends[name].label} cannot be used: {why}.")
            return False
        if self._queued >= MAX_QUEUE:
            self._msg("system", "Still working on earlier requests: wait or press CANCEL.")
            return False
        be = self.backends[name]
        if see_out and not getattr(be, "sees", False):
            self._msg("system", f"{be.label} is text only here (it cannot be shown a picture without tools): this "
                                "message goes without the picture" + (" and without the look-and-adjust loop." if loop > 1 else "."))
            see_out, loop = False, 0
        if see_out and (self._await is not None or self.loop is not None or self._held):
            self._msg("system", "The assistant is still looking at the output (or a loop is running): wait, or press "
                                "STOP LOOP / CANCEL.")
            return False
        if not self._ask_rate.allow():
            self._msg("system", f"Slow down: at most {ASK_LIMIT[0]} requests a minute.")
            self.log.log("rate_limited", what="ask")
            return False
        if see_out:
            n = max(1, min(int(loop or 1), see.LOOP_MAX))
            if n > 1:
                um["step"] = 1
                self.loop = dict(id=self._next(), n=n, step=1, text=text, backend=name, phase="look", t=self.clock(),
                                 pid=None, before=None, prev="", sig=None, why="", ev=None)
                self.log.log("loop", what="start", n=n, backend=name)
            return self._look(ctl, dict(text=text, backend=name, step=1, n=n, msg=um, chat=list(self.msgs)[:-1]))
        return self._queue(ctl, text, name, chat=list(self.msgs)[:-1])

    def _queue(self, ctl, text, name, chat=(), images=None, see_txt="", loop_id=None):
        """Build the prompt from the state RIGHT NOW and hand the job to the worker."""
        tex, art = self.lister()                       # cached listing: no disk access on this thread
        prompt = build_prompt(ctl, text, tex, art, chat=chat, see_txt=see_txt)
        ev = threading.Event()
        job = dict(id=self._next(), text=text, backend=name, prompt=prompt, gen=self._gen, cancel=ev,
                   images=list(images or []), loop=loop_id)
        if loop_id is not None and self.loop is not None:
            self.loop["ev"] = ev
        self._events.add(ev)
        self.log.log("request", id=job["id"], backend=name, text=text, prompt_chars=len(prompt),
                     pictures=[(n, len(b)) for n, b in job["images"]], loop=loop_id)      # sizes only, never the picture
        self._queued += 1
        if self.sync:
            self._run_job(job)
        else:
            self._jobs.put(job)
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="online-assistant", daemon=True)
                self._thread.start()
        return True

    # ---------------------------------------------------------------- worker thread
    def _work(self):
        while not self._stop:
            job = self._jobs.get()
            if job is None:
                return
            self._run_job(job)

    def _run_job(self, job):
        ev = job["cancel"]                             # this job's own event: set once, never cleared, so a cancel
        if ev.is_set() or job["gen"] != self._gen:     # that lands at any moment is seen
            self._done.put((job, None, "cancelled", 0.0))
            return
        busy = dict(id=job["id"], backend=job["backend"], t0=time.monotonic(), chars=0)
        self._busy = busy
        parsed = err = None
        try:
            kw = dict(images=job["images"]) if job.get("images") else {}      # text-only call: the old signature
            raw = self.backends[job["backend"]].ask(job["prompt"], cancel=ev,
                                                   progress=lambda n: busy.__setitem__("chars", n), **kw)
            raw = redact(str(raw)[:safety.MAX_REPLY])         # bounded, scrubbed, and parsed here: not on the render thread
            parsed = (raw,) + tuple(safety.parse_reply(raw))
        except Exception as ex:
            err = redact(f"{type(ex).__name__}: {ex}")[:400]
        try:
            self.rescan()                              # folder listings for the next prompt / validation
        except Exception:
            pass
        secs = time.monotonic() - busy["t0"]
        self._busy = None
        self._done.put((job, parsed, err, secs))

    # ---------------------------------------------------------------- GL thread: results
    def tick(self, ctl):
        while True:
            try:
                job, parsed, err, secs = self._done.get_nowait()
            except queue.Empty:
                break
            self._queued = max(0, self._queued - 1)
            self._events.discard(job["cancel"])
            self._handle(ctl, job, parsed, err, secs)
        if self._await is not None or self.loop is not None or not self._snap_q.empty():
            self._see_tick(ctl)
        fb = self._fade_back
        if fb and self.clock() >= fb[0]:
            self._fade_back = None
            if abs(ctl.bypass_fade - fb[2]) < 1e-9:     # only if nobody changed it in the meantime
                ctl.bypass_fade = fb[1]
        if self.mode == "autopilot":
            for p in list(self.props.values()):
                if p["status"] == "pending" and p.get("auto"):
                    self.apply(ctl, p["id"], by="autopilot")

    def _handle(self, ctl, job, parsed, err, secs):
        label = getattr(self.backends.get(job["backend"]), "label", job["backend"])
        if job["gen"] != self._gen or job["cancel"].is_set():
            self.log.log("dropped", id=job["id"], why="cancelled")
            return
        if err:
            self.log.log("error", id=job["id"], backend=job["backend"], secs=round(secs, 2), error=err)
            self._msg("system", f"{label} failed after {secs:.1f} s: {err}", backend=job["backend"], secs=round(secs, 1))
            self._loop_reply(ctl, job, None, [], err)
            return
        text, say, raw, perr = parsed
        tex, art = self.lister()
        ok, rejected, notes = safety.validate_batch(raw, safety.context(ctl, tex, art))
        if notes:
            self.log.log("clamp", id=job["id"], notes=notes)
        self.log.log("response", id=job["id"], backend=job["backend"], secs=round(secs, 2), raw=text[:4000],
                     parse_error=perr, accepted=ok, rejected=rejected, notes=notes)
        m = self._msg("assistant", say or ("(no message)" if ok else text.strip()[:600] or "(empty reply)"),
                      backend=job["backend"], label=label, secs=round(secs, 1))
        if perr and not ok:
            m["warn"] = perr
        if not ok and not rejected:
            self._loop_reply(ctl, job, None, ok, "")
            return
        pid = self._next()
        self.props[pid] = dict(id=pid, status="pending" if ok else "rejected", cmds=ok, diff=describe(ctl, ok),
                               rejected=rejected, notes=notes, backend=job["backend"], why="",
                               auto=self.mode == "autopilot", request=job["text"][:120])
        m["prop"] = pid
        for old in sorted(self.props)[:-MAX_MSGS]:
            del self.props[old]
        self._loop_reply(ctl, job, pid, ok, "")
        if ok and self.mode == "autopilot":
            self.apply(ctl, pid, by="autopilot")

    # ---------------------------------------------------------------- apply / undo
    def _hold_fade(self, ctl, orig):
        """bypass_fade was lengthened for a fade of ours: give the performer's value back once that fade is over."""
        if self._fade_back is None:
            self._fade_back = (self.clock() + safety.MIN_FADE_S * 1.5 + 0.1, orig, ctl.bypass_fade)

    def apply(self, ctl, pid, by="performer"):
        p = self.props.get(pid)
        if not p or p["status"] != "pending" or not p["cmds"]:
            return False
        now = self.clock()
        if by == "autopilot" and ctl.held.get("blackout"):
            p["why"] = "blackout is on: waiting"
            return False
        if now - self._last_apply < MIN_APPLY_S:
            p["why"] = "rate limit: one batch a second"
            return False
        # the show may have moved since the reply arrived (tempo, faders, governor): run the whole-batch rules again
        cmds, renotes, refusal = safety.finalize(p["cmds"], safety.context(ctl))
        if renotes:
            p["notes"] += [n for n in renotes if n not in p["notes"]]
            self.log.log("clamp", prop=pid, notes=renotes)
        if refusal or not cmds:
            p["status"], p["auto"], p["why"] = "rejected", False, refusal or "nothing left to apply"
            p["rejected"] = p["rejected"] + [{"cmd": "batch", "why": p["why"]}]
            self.log.log("refused", prop=pid, why=p["why"])
            return False
        kept = []
        for c in cmds:
            if c["cmd"] == "gen" and not self._gen_rate.allow():
                p["notes"].append(f"image job skipped: at most {GEN_LIMIT[0]} a minute")
                self.log.log("rate_limited", what="gen", prop=pid)
                continue
            kept.append(c)
        cmds = kept
        p["diff"] = describe(ctl, cmds)                  # what is really applied, against the state right now
        before = capture(ctl, cmds)
        fade0 = ctl.bypass_fade
        guards = safety.fade_guards(ctl, cmds)           # photosensitive: no hard cuts from the assistant
        if guards:
            p["notes"] += guards
            self.log.log("clamp", prop=pid, notes=guards)
        if ctl.bypass_fade != fade0:
            self._hold_fade(ctl, fade0)
        for c in cmds:
            try:
                if c["cmd"] == "set":
                    quiet_set(ctl, c["name"], c["v"])    # not a performer edit: the governor keeps its reductions
                elif c["cmd"] in control.CONTROL_CMDS:
                    ctl._do(dict(c))
                else:
                    ctl._out.append(dict(c))             # engine command (art_pick): out through Controller.apply
            except Exception as ex:
                p["notes"].append(f"{c['cmd']} failed: {type(ex).__name__}")
        self._last_apply = now
        p["status"], p["why"], p["auto"] = "applied", "", False
        summary = "; ".join(p["diff"])[:300] or "no change"
        self.history.append(dict(id=pid, t=time.strftime("%H:%M:%S"), summary=summary, undone=False, stale=False,
                                 before=before, by=by, n=len(cmds)))
        self.log.log("apply", prop=pid, by=by, commands=cmds, diff=p["diff"])
        L = self.loop
        if L is not None and L["pid"] == pid:          # the loop's own batch: let it settle, then look again
            L.update(phase="settle", t=now, t_still=None, sig=see.touch_sig(ctl), prev="; ".join(p["diff"])[:400], why="")
        return True

    def dismiss(self, pid):
        p = self.props.get(pid)
        if p and p["status"] == "pending":
            p["status"], p["auto"] = "dismissed", False
            self.log.log("dismiss", prop=pid)

    def undo(self, ctl, pid):
        for hrec in self.history:
            if hrec["id"] == pid and not hrec["undone"]:
                if hrec.get("stale"):                    # applied before the last PANIC: an undo would reverse PANIC
                    self.note = "that batch is from before the last PANIC and can no longer be undone"
                    return False
                try:
                    notes, fade = restore(ctl, hrec["before"])
                except Exception as ex:
                    self.note = f"undo failed: {type(ex).__name__}"
                    self.log.log("error", what="undo", prop=pid, error=str(ex)[:200])
                    return False
                if fade is not None:
                    self._hold_fade(ctl, fade)
                hrec["undone"] = True
                if pid in self.props:
                    self.props[pid]["status"] = "undone"
                    self.props[pid]["notes"] = self.props[pid]["notes"] + notes
                self.note = "; ".join(notes)
                self.log.log("undo", prop=pid, notes=notes)
                return True
        return False

    # ---------------------------------------------------------------- cancel / panic / close
    def cancel(self, why="cancelled"):
        self._see_reset("cancelled" if why == "cancelled" else why)
        self._gen += 1
        for ev in list(self._events):                   # the running job (its CLI tree is killed) and every queued one
            ev.set()
        try:
            while True:
                job = self._jobs.get_nowait()
                if job is None:
                    self._jobs.put(None)
                    break
                self._events.discard(job["cancel"])
                self._queued = max(0, self._queued - 1)
        except queue.Empty:
            pass
        self.log.log("cancel", why=why)

    def on_panic(self):
        """PANIC wins: nothing the assistant has in flight may move the look away from the safe scene, and nothing it
        did before can be 'undone' on top of the safe scene afterwards."""
        self.cancel("panic")
        for p in self.props.values():
            if p["status"] == "pending":
                p["status"], p["auto"], p["why"] = "dismissed", False, "PANIC"
            elif p["status"] == "applied":
                p["status"], p["why"] = "locked", "applied before PANIC: cannot be undone"
        for hrec in self.history:
            hrec["stale"] = True
        self._fade_back = None
        self.mode = "suggest"
        self._msg("system", "PANIC: assistant requests dropped, mode back to SUGGEST.")

    def clear(self):
        self.msgs.clear()
        self.props = {k: p for k, p in self.props.items() if p["status"] == "pending"}

    def close(self):
        self._stop = True
        self.cancel("closing")
        self._jobs.put(None)
        self.snaps.clear()                             # pictures never outlive the ONLINE switch

    # ---------------------------------------------------------------- panel
    def state(self):
        msgs = list(self.msgs)[-SHOWN_MSGS:]
        want = {m["prop"] for m in msgs if "prop" in m} | {k for k, p in self.props.items() if p["status"] == "pending"}
        props = [dict(id=p["id"], status=p["status"], diff=p["diff"][:30], rejected=p["rejected"][:12],
                      notes=p["notes"][:8], why=p["why"]) for k, p in sorted(self.props.items()) if k in want]
        b = self._busy
        return dict(
            mode=self.mode, backend=self.backend,
            backends={n: dict(label=getattr(be, "label", n), ok=self.available(n), optin=n in getattr(self, "blocked", {}),
                              sees=bool(getattr(be, "sees", False)),
                              why=(getattr(be, "problem", "") or "") if self._avail is not None else "checking")
                      for n, be in self.backends.items()},
            busy=dict(id=b["id"], backend=b["backend"], secs=round(time.monotonic() - b["t0"], 1), chars=b["chars"]) if b else None,
            queue=self._queued,
            msgs=[dict(m, text=m["text"][:600]) for m in msgs],
            props=props,
            history=[dict(id=x["id"], t=x["t"], summary=x["summary"][:160], undone=x["undone"], by=x["by"],
                          stale=bool(x.get("stale"))) for x in list(self.history)[-SHOWN_MSGS:]],
            note=self.note,
            looking=self._await is not None,
            loop=None if self.loop is None else dict(step=self.loop["step"], n=self.loop["n"], phase=self.loop["phase"],
                                                     why=self.loop["why"], secs=round(self.clock() - self.loop["t"], 1)))

    # ================================================================ SEE THE OUTPUT
    def snapshot_in(self, jpg):
        """The engine's picture (encoder thread). Only a JPEG of sane size is taken; the GL thread decides the rest."""
        if see.is_jpeg(jpg):
            try:
                self._snap_q.put_nowait(bytes(jpg))
            except queue.Full:
                pass

    def _look(self, ctl, req):
        """Ask the engine for one picture for this request (GL thread). The request waits in self._await."""
        req["t0"] = self.clock()
        self._await = req
        try:
            asked = bool(self.capture(ctl))
        except Exception:
            asked = False
        if not asked:
            self._await = None
            self._msg("system", "No picture of the output can be taken right now (the engine is not capturing): nothing "
                                "was sent.")
            self._loop_end("no picture of the output")
            return False
        if self.loop is not None:
            self.loop.update(phase="look", t=self.clock(), why="")
        return True

    def _see_tick(self, ctl):
        now = self.clock()
        while True:
            try:
                jpg = self._snap_q.get_nowait()
            except queue.Empty:
                break
            if self._await is not None:                # otherwise nobody asked for it: dropped
                req, self._await = self._await, None
                self._got_picture(ctl, req, jpg)
        if self._await is not None and now - self._await["t0"] > see.SNAP_TIMEOUT_S:
            self._await = None
            self._msg("system", "No picture of the output arrived (is the engine running with its preview?): nothing "
                                "was sent.")
            self._loop_end("no picture of the output arrived")
        L = self.loop
        if L is None:
            return
        if L["sig"] is not None and see.touch_sig(ctl) != L["sig"]:
            self._loop_end("you took over (a fader or switch moved)")
            return
        ph = L["phase"]
        if ph == "apply":
            p = self.props.get(L["pid"])
            if p is None or p["status"] in ("dismissed", "rejected", "undone", "locked"):
                self._loop_end("its proposal was not applied")
            elif now - L["t"] > see.APPLY_WAIT_S:
                self._loop_end("nobody applied its proposal")
        elif ph == "settle":
            busy = ctl._morph is not None or ctl.pending is not None or self._fading(ctl)
            if busy and now - L["t"] < see.SETTLE_MAX_S:
                L["t_still"] = None
                return
            if L.get("t_still") is None:
                L["t_still"] = now
            if now - L["t_still"] < see.SETTLE_S and now - L["t"] < see.SETTLE_MAX_S:
                return
            if L["step"] >= L["n"]:
                self._loop_end(f"all {L['n']} steps done", done=True)
                return
            L.update(phase="rate", t_rate=now, why="")
            self._loop_next(ctl)
        elif ph == "rate":
            self._loop_next(ctl)

    @staticmethod
    def _fading(ctl):
        byp = getattr(ctl, "_byp", None) or {}
        return (any((k in ctl.bypass and v < 1.0) or k not in ctl.bypass for k, v in byp.items())
                or any(k not in byp for k in ctl.bypass))

    def _loop_next(self, ctl):
        L = self.loop
        now = self.clock()
        if self._queued >= MAX_QUEUE or not self._ask_rate.allow():
            L["why"] = f"request limit ({ASK_LIMIT[0]} a minute): waiting"
            if now - L.get("t_rate", now) > see.RATE_WAIT_S:
                self._loop_end("the request limit held it back too long")
            return
        L["step"] += 1
        self._look(ctl, dict(text=L["text"], backend=L["backend"], step=L["step"], n=L["n"], before=L["before"],
                             prev=L["prev"], chat=list(self.msgs)))

    def _got_picture(self, ctl, req, jpg):
        sid = self.snaps.put(jpg)
        if req["step"] <= 1:
            pics = [("output.jpg", sid)]                   # shown on the message once it is really sent
        else:
            imgs = [dict(id=req["before"], label="before"), dict(id=sid, label="after")]
            req["msg"] = self._msg("loop", f"Look-and-adjust step {req['step']} of {req['n']}: the output before and "
                                           f"after the last change goes to the assistant.", imgs=imgs, step=req["step"])
            pics = [("before.jpg", req["before"]), ("output.jpg", sid)]
        req["pics"], req["sid"] = pics, sid
        risk = see.people_risk(getattr(ctl, "values", None) or ctl.base, ctl.bypass)
        if risk:
            hid = self._next()
            req["confirm_msg"] = self._msg("system", "People could be in this picture (" + "; ".join(risk) + "). It is "
                                           "NOT sent unless you press SEND THIS PICTURE.", confirm=hid, people=risk,
                                           imgs=[dict(id=sid, label="to send")])
            self._held[hid] = req
            if self.loop is not None:
                self.loop.update(phase="confirm", t=self.clock(), why="waiting for your confirmation")
            self.log.log("see_held", people=risk)
            return
        self._send_picture(ctl, req)

    def _send_picture(self, ctl, req):
        images = []
        for name, sid in req["pics"]:
            b = self.snaps.get(sid)
            if b is None:
                self._msg("system", "That picture is no longer held: nothing was sent.")
                self._loop_end("picture lost")
                return False
            images.append((name, b))
        if req["step"] <= 1:
            req["msg"]["imgs"] = [dict(id=req["sid"], label="sent")]      # exactly these bytes go out
        txt = see.see_text(req["step"], req["n"], req.get("prev", ""))
        lid = self.loop["id"] if self.loop is not None else None
        if self.loop is not None:
            self.loop.update(phase="ask", t=self.clock(), before=req["sid"], why="", sig=see.touch_sig(ctl))
        self._queue(ctl, req["text"], req["backend"], chat=req.get("chat", ()), images=images, see_txt=txt, loop_id=lid)
        return True

    def see_ok(self, ctl, hid):
        req = self._held.pop(hid, None)
        if req is None:
            return False
        req["confirm_msg"].pop("confirm", None)
        req["confirm_msg"]["text"] += "  (sent)"
        self.log.log("see_confirmed")
        return self._send_picture(ctl, req)

    def see_drop(self, hid):
        req = self._held.pop(hid, None)
        if req is None:
            return False
        self.snaps.drop(req["sid"])
        for m in (req["confirm_msg"], req["msg"]):
            m.pop("confirm", None)
            m["imgs"] = [i for i in m.get("imgs", []) if i["id"] != req["sid"]]
        req["confirm_msg"]["text"] = "Picture discarded: nothing was sent."
        self.log.log("see_dropped")
        self._loop_end("the picture was not sent")
        return True

    def _loop_reply(self, ctl, job, pid, ok, err):
        """A reply to one of the loop's requests arrived (GL thread)."""
        L = self.loop
        if L is None or job.get("loop") != L["id"]:
            return
        L["ev"] = None
        if err:
            self._loop_end("the assistant failed")
        elif not ok:
            self._loop_end("the assistant is happy with the look" if pid is None else "nothing it proposed could be applied",
                           done=pid is None)
        else:
            L.update(pid=pid, phase="apply", t=self.clock(), why="", sig=see.touch_sig(ctl))

    def _loop_end(self, why, done=False):
        L, self.loop = self.loop, None
        if L is None:
            return
        if L.get("ev") is not None:                    # its request still in flight: that reply is dropped
            L["ev"].set()
        if self._await is not None and self._await.get("step", 1) > 1:
            self._await = None
        self._msg("loop" if done else "system", ("Look-and-adjust loop finished: " if done else "Look-and-adjust loop stopped: ") + why
                  + f" (step {L['step']} of {L['n']}). Every applied step can be undone from the list.")
        self.log.log("loop", what="end", why=why, step=L["step"], n=L["n"])

    def loop_stop(self):
        self._loop_end("stopped by you")

    def _see_reset(self, why):
        self._loop_end(why)
        self._await = None
        for hid in list(self._held):
            self.see_drop(hid)
        try:
            while True:
                self._snap_q.get_nowait()
        except queue.Empty:
            pass
