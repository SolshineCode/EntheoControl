"""The only part of the ONLINE plugin that is imported while ONLINE is off: pure Python, no subprocess, no sockets.

control.py calls these hooks (each is a no-op while the plugin is not loaded):

    validate(c)        validator for {"cmd": "online", "op": ...}           (control._VALIDATORS)
    do(ctl, c)         run one validated "online" command                    (Controller._do)
    boot(ctl)          load the plugin if the saved switch says ON           (end of Controller.__init__)
    tick(ctl)          collect finished jobs, apply autopilot batches        (Controller.update, GL thread)
    state(ctl)         the "online" block of Controller.state()
    take_gen(ctl, job) True = the Nano Banana engine took this image job     (Controller._do "gen")
    on_panic(ctl)      PANIC: drop everything queued, autopilot -> suggest   (Controller.panic)
    close(ctl)         stop worker threads, kill child processes             (Controller.close)

Settings live in ctl.ext["online"] = {on, mode, backend, engine}, saved with live_state.json like the governor's.

Commands (all {"cmd": "online", "op": ...}):
    enable {on: bool}                      the master ONLINE switch (default OFF)
    mode {mode: "suggest" | "autopilot"}   suggest = show a diff with Apply; autopilot = apply, undoable
    backend {backend: "claude" | "agy"}    which CLI answers the chat
    engine {engine: "local" | "nano"}      AI DREAM image engine (nano = Nano Banana, only while ONLINE is on)
    cli_google {on: bool}                  opt-in (default OFF) to driving Google's CLIs (agy / gemini CLI) from this
                                           plugin: the agy chat backend and the CLI image providers. The user must
                                           review Antigravity's terms first; without it only Claude Code (chat) and
                                           the Gemini API with the user's own key (images) are used.
    key {key: str}                         the user's own Gemini API key for Nano Banana ("" = forget it); kept in
                                           online_keys.json next to the code, never logged, never sent back to a panel
    ask {text, backend?}                   chat message (<= MAX_ASK characters)
    apply {id} / dismiss {id}              a pending proposal
    undo {id}                              an applied batch (history)
    cancel                                 stop the running request and clear the queue
    clear                                  clear the chat (history of applied batches is kept)
"""
import re
from pathlib import Path

MODES = ("suggest", "autopilot")
BACKENDS = ("claude", "agy")
ENGINES = ("local", "nano")
MAX_ASK = 1000
MAX_ID = 10 ** 9
DEFAULTS = dict(on=False, mode="suggest", backend="claude", engine="local", cli_google=False)
OPS = ("enable", "mode", "backend", "engine", "cli_google", "key", "ask", "apply", "dismiss", "undo", "cancel", "clear")
KEY_RX = re.compile(r"^[A-Za-z0-9_\-]{20,200}$")


def _choice(c, key, allowed):
    v = c.get(key)
    if not isinstance(v, str) or v not in allowed:
        raise ValueError(f"'{key}' must be one of {list(allowed)}")
    return v


def _id(c):
    v = c.get("id")
    if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= MAX_ID:
        raise ValueError("'id' must be a whole number")
    return v


def validate(c):
    """Normalized copy of one "online" command; ValueError if malformed. Unknown keys are dropped."""
    op = c.get("op")
    if not isinstance(op, str) or op not in OPS:
        raise ValueError(f"'op' must be one of {list(OPS)}")
    out = {"cmd": "online", "op": op}
    if op in ("enable", "cli_google"):
        if not isinstance(c.get("on"), bool):
            raise ValueError("'on' must be true or false")
        out["on"] = c["on"]
    elif op == "key":
        k = c.get("key")
        if not isinstance(k, str) or (k != "" and not KEY_RX.match(k)):
            raise ValueError("'key' must be an API key (20..200 letters, digits, - or _) or an empty string to forget it")
        out["key"] = k
    elif op == "mode":
        out["mode"] = _choice(c, "mode", MODES)
    elif op == "backend":
        out["backend"] = _choice(c, "backend", BACKENDS)
    elif op == "engine":
        out["engine"] = _choice(c, "engine", ENGINES)
    elif op == "ask":
        t = c.get("text")
        if not isinstance(t, str) or not t.strip() or len(t) > MAX_ASK:
            raise ValueError(f"'text' must be 1..{MAX_ASK} characters")
        out["text"] = "".join(ch for ch in t if ch == "\n" or ord(ch) >= 32).strip()
        if not out["text"]:
            raise ValueError(f"'text' must be 1..{MAX_ASK} characters")
        if c.get("backend") is not None:
            out["backend"] = _choice(c, "backend", BACKENDS)
    elif op in ("apply", "dismiss", "undo"):
        out["id"] = _id(c)
    return out


def settings(ctl):
    """Cleaned copy of ctl.ext["online"] (whatever a hand-edited live_state.json holds, this is valid)."""
    raw = ctl.ext.get("online") if isinstance(getattr(ctl, "ext", None), dict) else None
    s = dict(DEFAULTS)
    if isinstance(raw, dict):
        for key in ("on", "cli_google"):
            if isinstance(raw.get(key), bool):
                s[key] = raw[key]
        for key, allowed in (("mode", MODES), ("backend", BACKENDS), ("engine", ENGINES)):
            if raw.get(key) in allowed:
                s[key] = raw[key]
    if s["backend"] == "agy" and not s["cli_google"]:
        s["backend"] = "claude"                       # agy is opt-in
    return s


def _load(ctl):
    """Import the real plugin and attach it. Only ever called with the switch ON."""
    if getattr(ctl, "online", None) is not None:
        return ctl.online
    from .plugin import OnlinePlugin                 # subprocess-capable code is imported here and nowhere else
    opts = dict(getattr(ctl, "online_opts", None) or {})
    opts.setdefault("root", Path(__file__).resolve().parents[2])
    opts.setdefault("art_dir", getattr(ctl, "art_dir", None))
    ctl.online = OnlinePlugin(settings=settings(ctl), **opts)
    return ctl.online


def _unload(ctl):
    p, ctl.online = getattr(ctl, "online", None), None
    if p is not None:
        try:
            p.close()
        except Exception:
            pass


def boot(ctl):
    try:
        if settings(ctl)["on"]:
            _load(ctl)
    except Exception:
        ctl.online = None


def do(ctl, c):
    s = settings(ctl)
    op = c["op"]
    if op == "enable":
        s["on"] = c["on"]
        ctl.ext["online"] = s
        if c["on"]:
            _load(ctl)
        else:
            _unload(ctl)
        return
    if op in ("mode", "backend", "engine", "cli_google"):
        if op == "backend" and c["backend"] == "agy" and not s["cli_google"]:
            return                                    # not opted in: the choice is refused
        s[op] = c["on"] if op == "cli_google" else c[op]
        if op == "cli_google" and not c["on"] and s["backend"] == "agy":
            s["backend"] = "claude"
        ctl.ext["online"] = s
        if getattr(ctl, "online", None) is not None:
            ctl.online.configure(s)
        return
    if getattr(ctl, "online", None) is not None:      # ask / apply / dismiss / undo / cancel / clear: OFF = ignored
        ctl.online.command(ctl, c)


def tick(ctl):
    p = getattr(ctl, "online", None)
    if p is not None:
        p.tick(ctl)


def state(ctl):
    s = settings(ctl)
    p = getattr(ctl, "online", None)
    if p is None:
        return dict(on=False, mode=s["mode"], backend=s["backend"], engine=s["engine"], cli_google=s["cli_google"])
    return p.state()


def take_gen(ctl, job):
    p = getattr(ctl, "online", None)
    return bool(p is not None and p.take_gen(ctl, job))


def cancel_gen(ctl):
    p = getattr(ctl, "online", None)
    if p is not None:
        p.cancel_gen()


def on_panic(ctl):
    p = getattr(ctl, "online", None)
    if p is not None:
        p.on_panic(ctl)


def close(ctl):
    _unload(ctl)
