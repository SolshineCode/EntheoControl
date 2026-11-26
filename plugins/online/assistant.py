"""Chat assistant: turns "make it calmer and greener" into validated control commands.

Threads: `ask()` (GL thread, via a panel command) builds the prompt from the live state and queues a job. ONE worker
thread runs the backend CLI (seconds to minutes). `tick()` (GL thread, every frame, costs nothing when idle) collects
finished replies, parses + validates them (safety.py) against the state AT THAT MOMENT, and either shows a proposal
(SUGGEST) or applies it (AUTOPILOT). Only queues cross threads; the controller is only ever touched on the GL thread.

Every applied batch records what it replaced, so it can be undone from the history list.
"""
import collections
import copy
import json
import queue
import threading
import time
from pathlib import Path

import control
from params import PARAMS, TRIGGERS

from . import safety

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


def build_prompt(ctl, user_text, tex=(), art=(), chat=()):
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


def capture(ctl, cmds):
    """What this batch is about to replace (only what it touches)."""
    b = dict(base={}, bypass={}, scenes={})
    for c in cmds:
        k = c["cmd"]
        if k == "set":
            b["base"].setdefault(c["name"], ctl.base[c["name"]])
        elif k == "morph":
            b.setdefault("morph", ctl.morph_bars)
        elif k == "scene_recall":
            b.setdefault("full", dict(base={n: ctl.base[n] for n in control.CONT}, mods=copy.deepcopy(ctl.mods),
                                      cur=ctl.cur))
            b.setdefault("morph", ctl.morph_bars)        # fade_guards may lengthen it
        elif k == "scene_save":
            b["scenes"].setdefault(c["i"], copy.deepcopy(ctl.scenes[c["i"]]))
            b.setdefault("cur", ctl.cur)
        elif k == "bypass":
            b["bypass"].setdefault(c["key"], c["key"] in ctl.bypass)
            b.setdefault("bypass_fade", ctl.bypass_fade)
        elif k == "art_pick":
            for n in ART_PARAMS:
                if n in ctl.base:
                    b["base"].setdefault(n, ctl.base[n])
        elif k == "text" and ctl.text is not None:
            b.setdefault("text", copy.deepcopy(ctl.text.to_json()))
    return b


def restore(ctl, b):
    """Put back what capture() saw. Faders glide back through the controller's normal slew."""
    full = b.get("full")
    if full:
        ctl.pending = None
        ctl._morph = None
        ctl.base.update(full["base"])
        ctl.mods = copy.deepcopy(full["mods"])
        ctl.cur = full["cur"]
    for n, v in b.get("base", {}).items():
        ctl.set(n, v)
    if "morph" in b:
        ctl.morph_bars = b["morph"]
    if "bypass_fade" in b:
        ctl.bypass_fade = b["bypass_fade"]
    for key, was in b.get("bypass", {}).items():
        ctl.set_bypass(key, was)
    if b.get("scenes"):
        for i, old in b["scenes"].items():
            ctl.scenes[i] = copy.deepcopy(old)
        if "cur" in b:
            ctl.cur = b["cur"]
        ctl._save_scenes()
    if "text" in b and ctl.text is not None:
        ctl.text.restore(b["text"])


# ------------------------------------------------------------------------------------------------ the assistant

class Assistant:
    def __init__(self, backends, log=None, lister=None, clock=time.monotonic, sync=False, mode="suggest",
                 backend="claude"):
        self.backends = dict(backends)                 # name -> object with .label, .available(), .ask(prompt, ...)
        self.log = log or OnlineLog(None)
        self.lister = lister or (lambda: ([], []))     # -> (texture names newest first, art names)
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
        self._cancel = threading.Event()
        self._gen = 0                                  # bumped by cancel / panic: older replies are dropped
        self._last_apply = -1e9
        self._ask_rate = RateLimiter(*ASK_LIMIT, clock=clock)
        self._gen_rate = RateLimiter(*GEN_LIMIT, clock=clock)
        self._avail = {}
        self.optin = {}                                # backend name -> why it is refused (opt-in not given)
        self._stop = False
        self._thread = None

    # ---------------------------------------------------------------- helpers
    def _next(self):
        self._id += 1
        return self._id

    def _msg(self, role, text, **kw):
        m = dict(id=self._next(), role=role, text=str(text)[:2000], **kw)
        self.msgs.append(m)
        return m

    def available(self, name):
        if name not in self._avail:                    # shutil.which once per session, not per frame
            try:
                self._avail[name] = bool(self.backends[name].available())
            except Exception:
                self._avail[name] = False
        return self._avail[name]

    # ---------------------------------------------------------------- GL thread: requests
    def ask(self, ctl, text, backend=None):
        name = backend or self.backend
        self._msg("user", text)
        if name not in self.backends:
            self._msg("system", f"Unknown backend '{name}'.")
            return False
        if name in self.optin:
            self._msg("system", f"{self.backends[name].label} is opt-in and switched off. {self.optin[name]}")
            return False
        if not self.available(name):
            self._msg("system", f"{self.backends[name].label} is not installed on this laptop.")
            return False
        if self._queued >= MAX_QUEUE:
            self._msg("system", "Still working on earlier requests: wait or press CANCEL.")
            return False
        if not self._ask_rate.allow():
            self._msg("system", f"Slow down: at most {ASK_LIMIT[0]} requests a minute.")
            self.log.log("rate_limited", what="ask")
            return False
        tex, art = self.lister()
        prompt = build_prompt(ctl, text, tex, art, chat=list(self.msgs)[:-1])
        job = dict(id=self._next(), text=text, backend=name, prompt=prompt, gen=self._gen)
        self.log.log("request", id=job["id"], backend=name, text=text, prompt_chars=len(prompt))
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
        if job["gen"] != self._gen:
            self._done.put((job, None, "cancelled", 0.0))
            return
        self._cancel.clear()
        busy = dict(id=job["id"], backend=job["backend"], t0=time.monotonic(), chars=0)
        self._busy = busy
        text = err = None
        try:
            text = self.backends[job["backend"]].ask(job["prompt"], cancel=self._cancel,
                                                    progress=lambda n: busy.__setitem__("chars", n))
        except Exception as ex:
            err = f"{type(ex).__name__}: {ex}"[:400]
        secs = time.monotonic() - busy["t0"]
        self._busy = None
        self._done.put((job, text, err, secs))

    # ---------------------------------------------------------------- GL thread: results
    def tick(self, ctl):
        while True:
            try:
                job, text, err, secs = self._done.get_nowait()
            except queue.Empty:
                break
            self._queued = max(0, self._queued - 1)
            self._handle(ctl, job, text, err, secs)
        if self.mode == "autopilot":
            for p in list(self.props.values()):
                if p["status"] == "pending" and p.get("auto"):
                    self.apply(ctl, p["id"], by="autopilot")

    def _handle(self, ctl, job, text, err, secs):
        label = getattr(self.backends.get(job["backend"]), "label", job["backend"])
        if job["gen"] != self._gen:
            self.log.log("dropped", id=job["id"], why="cancelled")
            return
        if err:
            self.log.log("error", id=job["id"], backend=job["backend"], secs=round(secs, 2), error=err)
            self._msg("system", f"{label} failed after {secs:.1f} s: {err}", backend=job["backend"], secs=round(secs, 1))
            return
        say, raw, perr = safety.parse_reply(text)
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
            return
        pid = self._next()
        self.props[pid] = dict(id=pid, status="pending" if ok else "rejected", cmds=ok, diff=describe(ctl, ok),
                               rejected=rejected, notes=notes, backend=job["backend"], why="",
                               auto=self.mode == "autopilot", request=job["text"][:120])
        m["prop"] = pid
        for old in sorted(self.props)[:-MAX_MSGS]:
            del self.props[old]
        if ok and self.mode == "autopilot":
            self.apply(ctl, pid, by="autopilot")

    # ---------------------------------------------------------------- apply / undo
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
        cmds = []
        for c in p["cmds"]:
            if c["cmd"] == "gen" and not self._gen_rate.allow():
                p["notes"].append(f"image job skipped: at most {GEN_LIMIT[0]} a minute")
                self.log.log("rate_limited", what="gen", prop=pid)
                continue
            cmds.append(c)
        before = capture(ctl, cmds)
        guards = safety.fade_guards(ctl, cmds)           # photosensitive: no hard cuts from the assistant
        if guards:
            p["notes"] += guards
            self.log.log("clamp", prop=pid, notes=guards)
        for c in cmds:
            try:
                if c["cmd"] in control.CONTROL_CMDS:
                    ctl._do(dict(c))
                else:
                    ctl._out.append(dict(c))             # engine command (art_pick): out through Controller.apply
            except Exception as ex:
                p["notes"].append(f"{c['cmd']} failed: {type(ex).__name__}")
        self._last_apply = now
        p["status"], p["why"], p["auto"] = "applied", "", False
        summary = "; ".join(p["diff"])[:300] or "no change"
        self.history.append(dict(id=pid, t=time.strftime("%H:%M:%S"), summary=summary, undone=False, before=before,
                                 by=by, n=len(cmds)))
        self.log.log("apply", prop=pid, by=by, commands=cmds, diff=p["diff"])
        return True

    def dismiss(self, pid):
        p = self.props.get(pid)
        if p and p["status"] == "pending":
            p["status"], p["auto"] = "dismissed", False
            self.log.log("dismiss", prop=pid)

    def undo(self, ctl, pid):
        for hrec in self.history:
            if hrec["id"] == pid and not hrec["undone"]:
                try:
                    restore(ctl, hrec["before"])
                except Exception as ex:
                    self.note = f"undo failed: {type(ex).__name__}"
                    self.log.log("error", what="undo", prop=pid, error=str(ex)[:200])
                    return False
                hrec["undone"] = True
                if pid in self.props:
                    self.props[pid]["status"] = "undone"
                self.log.log("undo", prop=pid)
                return True
        return False

    # ---------------------------------------------------------------- cancel / panic / close
    def cancel(self, why="cancelled"):
        self._gen += 1
        self._cancel.set()
        try:
            while True:
                self._jobs.get_nowait()
                self._queued = max(0, self._queued - 1)
        except queue.Empty:
            pass
        self.log.log("cancel", why=why)

    def on_panic(self):
        """PANIC wins: nothing the assistant has in flight may move the look away from the safe scene."""
        self.cancel("panic")
        for p in self.props.values():
            if p["status"] == "pending":
                p["status"], p["auto"], p["why"] = "dismissed", False, "PANIC"
        self.mode = "suggest"
        self._msg("system", "PANIC: assistant requests dropped, mode back to SUGGEST.")

    def clear(self):
        self.msgs.clear()
        self.props = {k: p for k, p in self.props.items() if p["status"] == "pending"}

    def close(self):
        self._stop = True
        self.cancel("closing")
        self._jobs.put(None)

    # ---------------------------------------------------------------- panel
    def state(self):
        msgs = list(self.msgs)[-SHOWN_MSGS:]
        want = {m["prop"] for m in msgs if "prop" in m} | {k for k, p in self.props.items() if p["status"] == "pending"}
        props = [dict(id=p["id"], status=p["status"], diff=p["diff"][:30], rejected=p["rejected"][:12],
                      notes=p["notes"][:8], why=p["why"]) for k, p in sorted(self.props.items()) if k in want]
        b = self._busy
        return dict(
            mode=self.mode, backend=self.backend,
            backends={n: dict(label=getattr(be, "label", n), ok=self.available(n), optin=n in self.optin)
                      for n, be in self.backends.items()},
            busy=dict(id=b["id"], backend=b["backend"], secs=round(time.monotonic() - b["t0"], 1), chars=b["chars"]) if b else None,
            queue=self._queued,
            msgs=[dict(m, text=m["text"][:600]) for m in msgs],
            props=props,
            history=[dict(id=x["id"], t=x["t"], summary=x["summary"][:160], undone=x["undone"], by=x["by"])
                     for x in list(self.history)[-SHOWN_MSGS:]],
            note=self.note)
