"""Nano Banana (Google's Gemini image model) as an AI DREAM engine, next to the local SD-Turbo worker.

Same contract as ai_worker.py: an atomic write of textures/gen_<time>_<seed>.png with the prompt in the PNG text
chunks, so live_visuals hot-loads it, the art picker lists it, and a "show when ready" job cross-fades to it
(`show_texture`). "Dream from current frame" hands snapshots/current.png to the image tool as its input image.

Providers are tried in order:
  GeminiApiProvider    RECOMMENDED. Google's Gemini API (Interactions endpoint) with an API key the USER typed into the
                       panel, kept in online_keys.json next to the code (see KeyStore). Uses the google-genai SDK when
                       it is installed in the venv, plain HTTPS otherwise. NOT yet exercised against the live API.
  GeminiNanoProvider   gemini CLI + its `nanobanana` extension (the user's nano-banana skill uses the same route)
  AgyImageProvider     Antigravity CLI (`agy`), asked to use its image tool and save out.png
The two CLI providers use the CLI's own login and exist only after the explicit opt-in (`cli_google`): Antigravity's
terms may not allow other software to drive the CLI, and that is the user's call (plugin.CLI_NOTE).
If every provider fails, is missing, or the rate limit is hit, the job goes to the local worker unchanged
(`ctl.ai.submit`) and the panel says so.

No key from any other project, environment variable or CLI config is ever read here.

PRIVACY: the prompt, and for dream-from-frame the snapshot image, are sent to Google. Usage limits apply.
"""
import base64
import json
import os
import queue
import random
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import control

from .backends import BackendError, agy_pty, fresh_dir, run_cli

MAX_N = 2                    # images per job on this engine (each one is a metered online call)
IMG_LIMIT = (6, 60.0)        # online images per window; beyond it jobs fall back to the local engine
MAX_SIDE = 768               # stored texture size (4 GB GPU: keep textures modest)
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def clean_prompt(p):
    p = re.sub(r"\s+", " ", "".join(ch for ch in str(p) if ord(ch) >= 32 or ch in "\n\t")).strip()
    return p[:500]


def newest_image(folder, skip=()):
    best, best_t = None, -1.0
    for f in Path(folder).rglob("*"):
        try:
            if f.is_file() and f.suffix.lower() in IMG_EXT and f.name not in skip and f.stat().st_size > 0:
                t = f.stat().st_mtime
                if t > best_t:
                    best, best_t = f, t
        except OSError:
            pass
    return best


def save_texture(src, job, seed, tex_dir, provider="", max_side=MAX_SIDE):
    """Any image file -> textures/gen_<time>_<seed>.png (RGB, <= max_side, atomic, prompt metadata). Returns the name."""
    from PIL import Image, PngImagePlugin
    tex_dir = Path(tex_dir)
    tex_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.load()
        im = im.convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(8, int(im.width * s)), max(8, int(im.height * s))), Image.LANCZOS)
    name = f"gen_{time.strftime('%Y%m%d_%H%M%S')}_{seed}.png"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("prompt", job.get("prompt", ""))
    meta.add_text("negative", job.get("negative", "") or "")
    meta.add_text("seed", str(seed))
    meta.add_text("engine", f"nano-banana/{provider}")
    tmp = tex_dir / (name + ".tmp")
    im.save(tmp, format="PNG", pnginfo=meta)
    os.replace(tmp, tex_dir / name)
    return name


class KeyStore:
    """The user's own Gemini API key, typed into the panel. One small JSON file next to the code, written
    atomically, listed in .gitignore. The key is never logged and never part of any state sent to a panel."""

    def __init__(self, path):
        self.path = Path(path)
        self._key = None

    def get(self):
        if self._key is None:
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                k = d.get("gemini_api_key") if isinstance(d, dict) else ""
                self._key = k if isinstance(k, str) else ""
            except (OSError, ValueError):
                self._key = ""
        return self._key

    def has(self):
        return bool(self.get())

    def set(self, key):
        self._key = str(key or "")
        try:
            if not self._key:
                if self.path.exists():
                    self.path.unlink()
                return True
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"gemini_api_key": self._key}) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False


API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_REVISION = "2026-05-20"
API_MODEL = "gemini-3.1-flash-lite-image"      # "Nano Banana 2 Lite": the fastest / cheapest per Google's image docs


def find_image_b64(obj):
    """The last image block's base64 data anywhere in an Interactions reply (steps[].content[] of type "image")."""
    found = None
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if x.get("type") == "image" and isinstance(x.get("data"), str) and x["data"]:
                found = x["data"]
            stack.extend(reversed(list(x.values())))
        elif isinstance(x, list):
            stack.extend(reversed(x))
    return found


class GeminiApiProvider:
    name = "api"

    def __init__(self, keys, model=API_MODEL, timeout=90.0):
        self.keys, self.model, self.timeout = keys, model, float(timeout)

    def available(self):
        return self.keys.has()

    def _input(self, prompt, init):
        parts = [{"type": "text", "text": clean_prompt(prompt)}]
        if init is not None:
            parts.append({"type": "image", "mime_type": "image/png",
                          "data": base64.b64encode(Path(init).read_bytes()).decode("ascii")})
        return parts

    def _sdk(self, key, parts):
        from google import genai                         # only if the user installed google-genai in the venv
        it = genai.Client(api_key=key).interactions.create(model=self.model, input=parts)
        return it.output_image.data

    def _rest(self, key, parts):
        req = urllib.request.Request(API_URL, data=json.dumps({"model": self.model, "input": parts}).encode(),
                                     headers={"x-goog-api-key": key, "Content-Type": "application/json",
                                              "Api-Revision": API_REVISION})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                reply = json.loads(r.read(64 << 20).decode("utf-8"))
        except urllib.error.HTTPError as ex:
            try:
                msg = json.loads(ex.read(8192).decode("utf-8", "replace")).get("error", {}).get("message", "")
            except Exception:
                msg = ""
            raise BackendError(f"Gemini API said {ex.code} {str(msg)[:200]}".strip()) from None
        except (OSError, ValueError) as ex:
            raise BackendError(f"Gemini API unreachable: {type(ex).__name__}") from None
        return find_image_b64(reply)

    def generate(self, prompt, init, work, cancel=None):
        key = self.keys.get()
        if not key:
            raise BackendError("no Gemini API key saved")
        parts = self._input(prompt, init)
        try:
            data = self._sdk(key, parts)
        except ImportError:
            data = self._rest(key, parts)
        except BackendError:
            raise
        except Exception as ex:
            raise BackendError(f"Gemini API (SDK) failed: {type(ex).__name__}: {str(ex)[:160]}".replace(key, "***")) from None
        if not data:
            raise BackendError("the Gemini API reply had no image in it")
        out = Path(work) / "api_out.png"
        out.write_bytes(base64.b64decode(data))
        return out


class GeminiNanoProvider:
    name = "gemini"

    def __init__(self, timeout=240.0, exe=None, ext_dir=None):
        self.timeout, self._exe = float(timeout), exe
        self.ext_dir = Path(ext_dir) if ext_dir else Path.home() / ".gemini" / "extensions" / "nanobanana"

    def exe(self):
        return self._exe or shutil.which("gemini")

    def available(self):
        return bool(self.exe()) and self.ext_dir.is_dir()

    def generate(self, prompt, init, work, cancel=None):
        # only the nanobanana extension is loaded (no mail / drive tools); the instruction rides on stdin, so no
        # part of the prompt is ever parsed by a shell
        argv = [self.exe(), "-e", "nanobanana", "--yolo", "--skip-trust", "-o", "text",
                "-p", "Follow the instructions given above exactly."]
        if init is not None:
            shutil.copyfile(init, Path(work) / "frame.png")
            task = ("Call the edit_image tool exactly once with file frame.png (it is in the current folder) and the "
                    "prompt below.")
        else:
            task = "Call the generate_image tool exactly once with the prompt below and no other options."
        stdin = (f"{task} Do not run shell commands, do not read or write any other file, do not call any other "
                 f"tool. Afterwards reply with only the saved file path.\nPROMPT: {clean_prompt(prompt)}\n")
        rc, out, err = run_cli(argv, stdin, cwd=work, timeout=self.timeout, cancel=cancel)
        img = newest_image(work, skip=("frame.png",))
        if img is None:
            raise BackendError(f"gemini made no image (exit {rc}): {(err or out).strip()[-240:] or 'no output'}")
        return img


class AgyImageProvider:
    name = "agy"

    def __init__(self, timeout=600.0, exe=None, model="gemini-4-hydrogen-low"):
        # the model here is only the AGENT that dispatches agy's built-in image tool (Nano Banana): the cheapest one
        self.timeout, self._exe, self.model = float(timeout), exe, model

    def exe(self):
        return self._exe or shutil.which("agy")

    def available(self):
        return bool(self.exe())

    def generate(self, prompt, init, work, cancel=None):
        work = Path(work)
        src = ""
        if init is not None:
            shutil.copyfile(init, work / "frame.png")
            src = " Use the file frame.png in the current folder as the input image and transform it."
        text = (f"Use your image generation tool (Nano Banana) to create exactly one image and save it as out.png in "
                f"the current folder.{src} Do not run other commands and do not touch any other file. "
                f"Image prompt: {clean_prompt(prompt)}")
        screen = agy_pty(self.exe(), text, work, timeout=self.timeout, done_file=work / "out.png", settle=2.0,
                         idle_after=45.0, cancel=cancel, model=self.model, extra=("--effort", "low"))
        img = newest_image(work, skip=("frame.png",))
        if img is None:
            raise BackendError(f"agy made no image: {screen[-240:] or 'no output'}")
        return img


def default_providers():
    return [GeminiNanoProvider(), AgyImageProvider()]


class NanoBanana:
    """Job queue + one worker thread. submit() and tick() are called on the GL thread; nothing here blocks it."""

    def __init__(self, tex_dir, snapshot, work_root, providers=None, log=None, clock=time.monotonic, sync=False):
        self.tex_dir, self.snapshot, self.work_root = Path(tex_dir), Path(snapshot), Path(work_root)
        self.providers = list(default_providers() if providers is None else providers)
        self.log, self.clock, self.sync = log, clock, bool(sync)
        self._jobs, self._out = queue.Queue(), queue.Queue()
        self._cancel = threading.Event()
        self._gen = 0
        self._times = []
        self._thread = None
        self._stop = False
        self.busy = None             # dict(prompt, t0, provider)
        self.pending = 0
        self.done = []               # last results for the panel
        self.note = ""
        self._avail = None

    def set_providers(self, providers):
        self.providers = list(providers)
        self._avail = None

    def available(self):
        if self._avail is None:
            self._avail = []
            for p in self.providers:
                try:
                    if p.available():
                        self._avail.append(p.name)
                except Exception:
                    pass
        return list(self._avail)

    def _log(self, kind, **kw):
        if self.log is not None:
            self.log.log(kind, **kw)

    def _allow(self, n):
        now = self.clock()
        self._times = [t for t in self._times if now - t < IMG_LIMIT[1]]
        if len(self._times) + n > IMG_LIMIT[0]:
            return False
        self._times += [now] * n
        return True

    # ---------------------------------------------------------------- GL thread
    def submit(self, job, delay=0.0):
        job = dict(job)
        n = max(1, min(MAX_N, int(job.get("n", 1))))
        if not self.available():
            self._out.put(("fallback", job, "no online image tool is installed"))
            return
        if not self._allow(n):
            self._out.put(("fallback", job, f"limit of {IMG_LIMIT[0]} online images a minute"))
            return
        job["n"] = n
        item = (job, time.monotonic() + max(0.0, delay), self._gen)
        self.pending += 1
        self._log("image_request", prompt=job.get("prompt", "")[:200], n=n, from_frame=job.get("init") == "snapshot")
        if self.sync:
            self._run(item)
        else:
            self._jobs.put(item)
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="online-images", daemon=True)
                self._thread.start()

    def tick(self, ctl):
        while True:
            try:
                kind, a, b = self._out.get_nowait()
            except queue.Empty:
                return
            if kind == "done":
                self.done = (self.done + [a])[-8:]
                self.note = ""
                if a.get("show"):
                    ctl._out.append(control.passthrough("show_texture", name=a["name"]))
            elif kind == "end":
                self.pending = max(0, self.pending - 1)
            elif kind == "fallback":
                self.note = f"Nano Banana: {b}. Generated locally instead."
                self._log("image_fallback", why=b, prompt=a.get("prompt", "")[:200])
                local = {k: a[k] for k in ("prompt", "negative", "n", "seed", "init", "show", "strength") if k in a}
                try:
                    ctl.ai.submit(local)
                except Exception:
                    pass

    def cancel(self):
        self._gen += 1
        self._cancel.set()

    def close(self):
        self._stop = True
        self.cancel()
        self._jobs.put(None)

    def state(self):
        b = self.busy
        return dict(available=self.available(), providers=[p.name for p in self.providers], busy=bool(b),
                    pending=self.pending,
                    current=dict(prompt=b["prompt"][:80], secs=round(time.monotonic() - b["t0"], 1),
                                 provider=b["provider"]) if b else None,
                    done=list(self.done), note=self.note)

    # ---------------------------------------------------------------- worker thread
    def _work(self):
        while not self._stop:
            item = self._jobs.get()
            if item is None:
                return
            self._run(item)

    def _run(self, item):
        job, due, gen = item
        try:
            while not self.sync and time.monotonic() < due and gen == self._gen:
                time.sleep(0.05)
            if gen != self._gen:
                return
            self._cancel.clear()
            init = None
            if job.get("init") == "snapshot":
                if not self.snapshot.is_file():
                    self._out.put(("fallback", job, "no snapshot of the current frame"))
                    return
                init = self.snapshot
            seed0 = job["seed"] if isinstance(job.get("seed"), int) else random.randrange(2 ** 31 - 16)
            for i in range(job["n"]):
                if gen != self._gen:
                    return
                err = self._one(job, init, seed0 + i)
                if err == "cancelled":
                    return
                if err:
                    if i == 0:
                        self._out.put(("fallback", dict(job), err))
                    else:
                        self._log("image_error", error=err)
                    return
        except Exception as ex:                          # never let the worker thread die
            self._out.put(("fallback", dict(job), f"{type(ex).__name__}: {ex}"[:200]))
        finally:
            self.busy = None
            self._out.put(("end", None, None))

    def _one(self, job, init, seed):
        """One image through the first provider that works. Returns "" or the last error."""
        err = "no provider available"
        for p in self.providers:
            if p.name not in self.available():
                continue
            work = fresh_dir(self.work_root, "nano")
            t0 = time.monotonic()
            self.busy = dict(prompt=job.get("prompt", ""), t0=t0, provider=p.name)
            try:
                src = p.generate(job["prompt"], init, work, cancel=self._cancel)
                name = save_texture(src, job, seed, self.tex_dir, provider=p.name)
                secs = round(time.monotonic() - t0, 2)
                self._out.put(("done", dict(name=name, prompt=job["prompt"][:80], secs=secs, provider=p.name,
                                            show=bool(job.get("show", True))), None))
                self._log("image", name=name, provider=p.name, secs=secs, prompt=job["prompt"][:200],
                          from_frame=init is not None)
                return ""
            except Exception as ex:
                err = f"{p.name}: {type(ex).__name__}: {ex}"[:300]
                self._log("image_error", provider=p.name, error=err, secs=round(time.monotonic() - t0, 2))
                if self._cancel.is_set():
                    return "cancelled"
            finally:
                shutil.rmtree(work, ignore_errors=True)
        return err
