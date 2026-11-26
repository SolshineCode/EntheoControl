"""OnlinePlugin: the object behind the ONLINE switch. Created by protocol._load() when the switch goes ON,
closed (worker threads stopped, child processes killed) when it goes OFF. Importing this module is what makes
subprocess-capable code reachable; control.py never imports it directly.
"""
import os
import threading
import time
from pathlib import Path

from . import protocol
from .assistant import Assistant, OnlineLog
from .backends import ClaudeBackend
from .images import GeminiApiProvider, KeyStore, NanoBanana, default_key_path

PRIVACY_NOTE = ("ONLINE is on: chat messages + the current settings go to the chosen assistant (Anthropic / Google); "
                "image prompts (and frames, for dream-from-frame) are sent to Google; usage limits apply.")
CLI_NOTE = ("Google's CLIs (agy, gemini) are not used: they are coding agents with no tool-less mode. Online images "
            "and AUTO MAP vision need your own Gemini API key.")
AGY_CHAT_NOTE = CLI_NOTE
TEX_EXT = (".png", ".jpg", ".jpeg", ".webp")


def list_names(folder, cap=24, prefix=None):
    """Image file names in a folder, newest first (plain os.scandir: a few dozen entries, microseconds)."""
    out = []
    try:
        with os.scandir(folder) as it:
            for e in it:
                n = e.name
                if (e.is_file() and n.lower().endswith(TEX_EXT) and not n.lower().endswith(".depth.png")
                        and not n.startswith(".") and (prefix is None or n.startswith(prefix))):
                    out.append((e.stat().st_mtime, n))
    except OSError:
        return []
    return [n for _, n in sorted(out, reverse=True)[:cap]]


class OnlinePlugin:
    def __init__(self, root, art_dir=None, settings=None, backends=None, image_providers=None, log_dir="auto",
                 tex_dir=None, snapshot=None, work_root=None, clock=time.monotonic, sync=False, keys_path=None):
        self.root = Path(root)
        self.tex_dir = Path(tex_dir) if tex_dir else self.root / "textures"
        self.art_dir = Path(art_dir) if art_dir else self.root / "art"
        s = dict(protocol.DEFAULTS, **(settings or {}))
        self.engine = s["engine"]
        self._vision = None                                        # plugins.online.vision.Vision, one per session
        self.keys = KeyStore(keys_path or default_key_path())      # outside the project folder
        self.sync = bool(sync)
        self._lists = ([], [])                                     # cached (textures, art), refreshed by _scan
        self._scanning = threading.Lock()
        self.notice = ""
        self._fixed_providers = image_providers is not None
        self.log = OnlineLog(self.root / "logs" if log_dir == "auto" else log_dir)
        work = Path(work_root) if work_root else self.root / "logs" / "online_work"
        if backends is None:                                       # chat is Claude Code only (see backends.py)
            backends = {"claude": ClaudeBackend(work, model=os.environ.get("ONLINE_CLAUDE_MODEL") or "sonnet")}
        self.assistant = Assistant(backends, log=self.log, lister=self._lister, clock=clock, sync=sync,
                                   mode=s["mode"], backend=s["backend"])
        self.images = NanoBanana(self.tex_dir, Path(snapshot) if snapshot else self.root / "snapshots" / "current.png",
                                 work, providers=image_providers if self._fixed_providers else self._providers(),
                                 log=self.log, clock=clock, sync=sync)
        self.assistant.rescan = self._scan_lists
        self.images.rescan = self._scan_lists
        self.log.log("online", on=True, mode=s["mode"], backend=s["backend"], engine=s["engine"])
        self.rescan()

    def _lister(self):
        return self._lists                                         # cached: the render thread never lists folders

    def _scan_lists(self):
        self._lists = (list_names(self.tex_dir), list_names(self.art_dir))

    def _scan(self):
        """Everything that touches the disk or starts a process just to find out what is there: folder listings,
        `claude --help` (flag check), shutil.which, the key file. Never on the render thread."""
        with self._scanning:
            try:
                self._scan_lists()
                self.keys.get()
                self.images.refresh()
                self.assistant.refresh()
            except Exception:
                pass

    def rescan(self):
        if self.sync:
            self._scan()
        else:
            threading.Thread(target=self._scan, name="online-scan", daemon=True).start()

    def _providers(self):
        """The Gemini API with the user's own key. Nothing else (see images.py)."""
        return [GeminiApiProvider(self.keys)]

    def configure(self, s):
        self.assistant.mode, self.assistant.backend, self.engine = s["mode"], s["backend"], s["engine"]
        self.log.log("settings", mode=s["mode"], backend=s["backend"], engine=s["engine"])

    # ---------------------------------------------------------------- hooks (GL thread)
    def command(self, ctl, c):
        op = c["op"]
        a = self.assistant
        if op == "ask":
            a.ask(ctl, c["text"], c.get("backend"))
        elif op == "apply":
            a.apply(ctl, c["id"])
        elif op == "dismiss":
            a.dismiss(c["id"])
        elif op == "undo":
            a.undo(ctl, c["id"])
        elif op == "cancel":
            a.cancel()
            self.images.cancel()
        elif op == "clear":
            a.clear()
        elif op == "key":
            ok = self.keys.set(c["key"])
            self.rescan()                                             # availability changed
            self.log.log("key", saved=bool(c["key"]) and ok, forgotten=not c["key"])   # never the key itself

    def tick(self, ctl):
        self.assistant.tick(ctl)
        self.images.tick(ctl)

    def take_gen(self, ctl, job):
        """AI DREAM image job: True = Nano Banana has it (it falls back to ctl.ai by itself on any failure)."""
        if self.engine != "nano":
            return False
        delay = 0.0
        if job.get("init") == "snapshot":                # the engine writes snapshots/current.png first
            import control
            ctl._out.append(control.passthrough("snapshot"))
            delay = 0.6
        self.images.submit(job, delay=delay)
        return True

    def cancel_gen(self):
        self.images.cancel()

    def refused(self, what, why):
        self.notice = f"{what} refused: {why}"
        self.log.log("refused", what=what, why=why)

    def on_panic(self, ctl):
        self.assistant.on_panic()
        self.images.cancel()                             # an image in flight must not cross-fade in after PANIC
        s = protocol.settings(ctl)
        s["mode"] = "suggest"
        ctl.ext["online"] = s

    def state(self):
        st = self.assistant.state()
        st.update(on=True, engine=self.engine, nano=self.images.state(), privacy=PRIVACY_NOTE,
                  cli_note=CLI_NOTE, has_key=bool(self.keys._key),
                  agy_chat=AGY_CHAT_NOTE, notice=self.notice)
        return st

    def close(self):
        self.log.log("online", on=False)
        for part in (self.assistant, self.images, self.log):
            try:
                part.close()
            except Exception:
                pass
