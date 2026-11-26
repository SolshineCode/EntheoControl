"""Background AI texture worker: keeps the diffusion model loaded and turns panel prompts into textures.

  python ai_worker.py                       # default: SD-Turbo, http://127.0.0.1:8766, low priority
  python ai_worker.py --base sd15-lcm --lora my_lora.safetensors

Runs as its own process so an out-of-memory or driver hiccup can never take down the show.
Output: textures/gen_<time>_<seed>.png (atomic write; prompt stored in the PNG text chunk),
which live_visuals.py hot-loads. "Dream from current frame" reads snapshots/current.png
(written by live_visuals on request) and runs img2img on it.

HTTP (localhost only, header X-Token = contents of .remote_token):
  POST /job    {prompt, negative?, n 1..8, seed?, init: null|"snapshot", strength 0.2..0.9, show?}
  GET  /status
  POST /depth   {name}  -> art/<stem>.depth.png (Depth Anything V2 Small, 16-bit, white = near)
  POST /cancel
  POST /dreamify {name, source: "art"|"ai", strength 0.05..1, octaves 1..10, preset}  -> real DeepDream of a library
        image (dreamify.py: torchvision GoogLeNet, octaves, jitter, tiles, fp16) written as a NEW image:
        art -> art/<stem>_dream_<preset>_<time>.png (+ a depth map job), ai -> textures/gen_dream_<time>_<preset>.png
        (atomic, PNG text dream_*). VRAM: SD-Turbo is moved to system RAM while it runs if free VRAM is short, then
        moved back; no room at all -> CPU at 384 px. /status "dream" = live progress {pct, octave, device, ...}.
  POST /dream-cancel   stops a running / queued dreamify (nothing is written)
"""
import argparse
import json
import math
import os
import queue
import random
import re
import secrets
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps, PngImagePlugin

import dreamify

HERE = Path(__file__).parent
MAX_QUEUE = 16
ART_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(png|jpg|jpeg|webp)$", re.I)
MAX_BODY = 8192


def read_token(path):
    try:
        t = Path(path).read_text(encoding="utf-8").strip()
        return t or None
    except OSError:
        return None


def ensure_token(path):
    t = read_token(path)
    if t:
        return t
    t = secrets.token_urlsafe(6)
    try:
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(t + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass
    return t


def validate_job(body):
    """Return a clean job dict or raise ValueError."""
    if not isinstance(body, dict):
        raise ValueError("job must be an object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 500:
        raise ValueError("prompt must be 1..500 chars")
    neg = body.get("negative") or ""
    if not isinstance(neg, str) or len(neg) > 500:
        raise ValueError("negative must be <= 500 chars")
    n = body.get("n", 1)
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 8:
        raise ValueError("n must be 1..8")
    seed = body.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31):
        raise ValueError("seed must be an int")
    init = body.get("init")
    if isinstance(init, dict):                       # dream from one library image (image_library ids)
        nm = init.get("name")
        if init.get("kind") not in ("art", "ai") or not isinstance(nm, str) or not ART_NAME.match(nm) or ".." in nm                 or nm.lower().endswith(".depth.png"):
            raise ValueError("init image must be {kind: art | ai, name}")
        init = {"kind": init["kind"], "name": nm}
    elif init not in (None, "snapshot"):
        raise ValueError("init must be null, 'snapshot' or {kind, name}")
    strength = body.get("strength", 0.45)
    if not isinstance(strength, (int, float)) or isinstance(strength, bool) or not 0.2 <= float(strength) <= 0.9:
        raise ValueError("strength must be 0.2..0.9")
    return dict(prompt=prompt.strip(), negative=neg.strip(), n=n,
                seed=seed if seed is not None else random.randrange(2**31 - 16),
                init=init, strength=float(strength), show=bool(body.get("show", True)))


class RealPipes:
    """diffusers pipelines (txt2img + img2img sharing weights), seamless-tiling, 4 GB friendly."""

    def __init__(self, base="sd-turbo", lora=None, vae_fp32=False):
        import torch
        from diffusers import AutoPipelineForImage2Image
        import ai_gen
        self.torch = torch
        self.pipe, self.cfg = ai_gen.load_pipe(base, lora, torch.float16, vae_fp32)
        self.i2i = AutoPipelineForImage2Image.from_pipe(self.pipe)
        self.base = base

    on_gpu = True

    def offload(self):
        """Move the weights to system RAM (DREAMIFY needs the VRAM); restore() brings them back."""
        self.pipe.to("cpu")
        self.on_gpu = False
        self.torch.cuda.empty_cache()

    def restore(self):
        """Back to the GPU. On failure (OOM) the WHOLE pipe goes back to the CPU (never half on each) and the
        next txt2img / img2img tries again; raises so the caller can report it."""
        try:
            self.pipe.to("cuda")
            self.on_gpu = True
        except Exception:
            try:
                self.pipe.to("cpu")
            finally:
                self.on_gpu = False
                _free_cuda()
            raise

    def _ready(self):
        if not self.on_gpu:
            self.restore()

    def txt2img(self, prompt, negative, seed, size):
        self._ready()
        g = self.torch.Generator("cuda").manual_seed(seed)
        kw = dict(num_inference_steps=self.cfg["steps"], guidance_scale=self.cfg["cfg"],
                  height=size, width=size, generator=g)
        if negative and self.cfg["cfg"] > 1.0:
            kw["negative_prompt"] = negative
        return self.pipe(prompt, **kw).images[0]

    def img2img(self, prompt, negative, seed, init, strength):
        self._ready()
        g = self.torch.Generator("cuda").manual_seed(seed)
        steps = max(self.cfg["steps"], math.ceil(1.0 / strength), 2)   # steps*strength must be >= 1
        return self.i2i(prompt, image=init, strength=strength, num_inference_steps=steps,
                        guidance_scale=self.cfg["cfg"], generator=g).images[0]


class RealDepth:
    """Depth Anything V2 Small (~25M params, ~100 MB VRAM). Returns float32 HxW, 1 = near."""

    def __init__(self, model="depth-anything/Depth-Anything-V2-Small-hf"):
        import torch
        from transformers import pipeline
        self.pipe = pipeline("depth-estimation", model=model, device=0 if torch.cuda.is_available() else -1)

    def depth(self, img):
        import numpy as np
        out = self.pipe(img.convert("RGB"))
        d = out["predicted_depth"]
        d = d.squeeze().float().cpu().numpy() if hasattr(d, "cpu") else np.asarray(d, "float32")
        d = np.asarray(Image.fromarray(d.astype("float32"), mode="F").resize(img.size, Image.BICUBIC))
        return d                                   # relative inverse depth: larger = nearer


def validate_art_name(body, art_dir):
    if not isinstance(body, dict):
        raise ValueError("body must be an object")
    name = body.get("name")
    if not isinstance(name, str) or not ART_NAME.match(name) or name.lower().endswith(".depth.png"):
        raise ValueError("bad art name")
    path = Path(art_dir) / name
    if Path(name).name != name or not path.is_file():
        raise ValueError("art file not found")
    return dict(kind="depth", name=name, prompt=f"depth: {name}", n=1)


def init_path(init, tex_dir, art_dir):
    """{kind: art | ai, name} -> the image file, which must lie directly in its folder (FileNotFoundError)."""
    folder = Path(art_dir if init.get("kind") == "art" else tex_dir)
    name = str(init.get("name") or "")
    p = folder / name
    if not ART_NAME.match(name) or Path(name).name != name or not p.is_file() or p.resolve().parent != folder.resolve():
        raise FileNotFoundError(f"init image not found: {name}")
    return p


def save_depth(arr, path):
    """float depth (any range, larger = nearer) -> normalized 16-bit PNG, white = near. Atomic."""
    import numpy as np
    a = np.nan_to_num(np.asarray(arr, "float64"))
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    img = Image.fromarray((a * 65535 + 0.5).astype("uint16"))
    tmp = Path(str(path) + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def fit_init(img, max_side=768):
    img = img.convert("RGB")
    s = min(1.0, max_side / max(img.size))
    w, h = (max(64, int(img.width * s) // 8 * 8), max(64, int(img.height * s) // 8 * 8))
    return img.resize((w, h), Image.LANCZOS)


class Worker:
    def __init__(self, out_dir, snapshot, token, port=8766, pipes_factory=None, size=512,
                 art_dir=None, depth_factory=None, dream_factory=None, vram_free=None):
        self.out_dir, self.snapshot, self.token, self.size = Path(out_dir), Path(snapshot), token, size
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.art_dir = Path(art_dir) if art_dir else Path(out_dir).parent / "art"
        self.pipes_factory = pipes_factory or RealPipes
        self.depth_factory = depth_factory          # None -> use the diffusion pipes object's .depth
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.pending = []            # [(id, job)] mirrors the queue for /status and cancel
        self.cancel_gen = 0
        self.st = dict(loaded=False, loading=False, busy=False, current=None, done=[], error="", base="", dream=None)
        self.dream_factory = dream_factory          # device -> object with .dream(...) / .release() (dreamify.RealDream)
        self.vram_free = vram_free or dreamify.free_vram_mb
        self.dream_cancel_gen = 0
        self.running = False
        self._srv = None
        self.port = port

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        worker = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def _auth(self):
                ok = secrets.compare_digest(self.headers.get("X-Token", ""), worker.token)
                if not ok:
                    self._send(403, {"error": "forbidden"})
                return ok

            def do_GET(self):
                if not self._auth():
                    return
                if self.path == "/status":
                    self._send(200, worker.status())
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                if not self._auth():
                    return
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    n = -1
                if not 0 <= n <= MAX_BODY:
                    return self._send(413, {"error": "body too large"})
                raw = self.rfile.read(n) if n else b"{}"
                try:
                    body = json.loads(raw or b"{}", parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
                except ValueError:
                    return self._send(400, {"error": "bad json"})
                if self.path == "/job":
                    try:
                        job = validate_job(body)
                    except ValueError as ex:
                        return self._send(400, {"error": str(ex)})
                    jid = worker.submit(job)
                    if jid is None:
                        return self._send(503, {"error": "queue full"})
                    return self._send(200, {"id": jid})
                if self.path == "/depth":
                    try:
                        job = validate_art_name(body, worker.art_dir)
                    except ValueError as ex:
                        return self._send(400, {"error": str(ex)})
                    jid = worker.submit(job)
                    if jid is None:
                        return self._send(503, {"error": "queue full"})
                    return self._send(200, {"id": jid})
                if self.path == "/dreamify":
                    if worker.dream_factory is None:
                        return self._send(503, {"error": "dreamify is not available in this worker"})
                    try:
                        job = dreamify.validate(body, worker.art_dir, worker.out_dir)
                    except ValueError as ex:
                        return self._send(400, {"error": str(ex)})
                    jid = worker.submit(job)
                    if jid is None:
                        return self._send(503, {"error": "queue full"})
                    return self._send(200, {"id": jid})
                if self.path == "/dream-cancel":
                    worker.cancel_dream()
                    return self._send(200, {"ok": True})
                if self.path == "/cancel":
                    worker.cancel()
                    return self._send(200, {"ok": True})
                if self.path == "/offload":              # governor.py power balance: idle model -> system RAM
                    worker.q.put(("offload", None, worker.cancel_gen))
                    return self._send(200, {"ok": True})
                self._send(404, {"error": "not found"})

        self._srv = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self._srv.daemon_threads = True
        self.port = self._srv.server_address[1]
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def stop(self):
        self.running = False
        self.q.put(None)
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()

    # ---------------------------------------------------------------- queue
    def submit(self, job):
        with self.lock:
            if len(self.pending) >= MAX_QUEUE:
                return None
            jid = secrets.token_hex(4)
            self.pending.append((jid, job))
            gens = (self.cancel_gen, self.dream_cancel_gen)
            self.q.put((jid, job, gens[0], gens[1]))       # inside the lock: no cancel can slip in between
        return jid

    def cancel(self):
        with self.lock:
            self.pending.clear()
            self.cancel_gen += 1

    def cancel_dream(self):
        with self.lock:
            self.pending = [(j, jb) for j, jb in self.pending if jb.get("kind") != "dream"]
            self.dream_cancel_gen += 1

    def status(self):
        with self.lock:
            s = dict(self.st)
            s["dream"] = dict(self.st["dream"]) if self.st["dream"] else None
            s["done"] = list(self.st["done"][-20:])
            p = getattr(self, "_pipes", None)
            s["on_gpu"] = bool(getattr(p, "on_gpu", True)) if p is not None else False
            s["queue"] = [dict(id=j, prompt=job["prompt"][:80], n=job["n"], kind=job.get("kind", "gen"))
                          for j, job in self.pending]
        return s

    def _set(self, **kw):
        with self.lock:
            self.st.update(kw)

    # ---------------------------------------------------------------- work
    def _load(self):
        self._set(loading=True, error="")
        t = time.time()
        pipes = self.pipes_factory()
        self._set(loading=False, loaded=True, base=getattr(pipes, "base", "fake"))
        self._pipes = pipes                          # status() reports whether the weights sit on the GPU
        print(f"[ai] model ready in {time.time() - t:.1f}s", flush=True)
        return pipes

    def _save(self, img, job, seed):
        name = f"gen_{time.strftime('%Y%m%d_%H%M%S')}_{seed}.png"
        meta = PngImagePlugin.PngInfo()
        meta.add_text("prompt", job["prompt"])
        meta.add_text("negative", job["negative"])
        meta.add_text("seed", str(seed))
        tmp = self.out_dir / (name + ".tmp")
        img.save(tmp, format="PNG", pnginfo=meta)
        os.replace(tmp, self.out_dir / name)
        return name

    def _loop(self):
        pipes = depth = None
        while self.running:
            item = self.q.get()
            if item is None:
                break
            jid, job, gen = item[:3]
            dgen = item[3] if len(item) > 3 else self.dream_cancel_gen
            if jid == "offload":                         # on this thread: never races a running job
                if pipes is not None and getattr(pipes, "on_gpu", False) and hasattr(pipes, "offload"):
                    try:
                        pipes.offload()
                        print("[ai] idle: model moved to system RAM (back on the next image)", flush=True)
                    except Exception as ex:
                        print(f"[ai] offload failed: {ex}", flush=True)
                continue
            if jid == "warmup":
                if pipes is None:
                    try:
                        pipes = self._load()
                    except Exception as ex:
                        self._set(loading=False, error=f"model load failed: {type(ex).__name__}: {ex}")
                        print(f"[ai] model load failed: {ex}", flush=True)
                continue
            with self.lock:
                if gen != self.cancel_gen or (job.get("kind") == "dream" and dgen != self.dream_cancel_gen):
                    self.pending = [(j, jb) for j, jb in self.pending if j != jid]
                    continue                             # cancelled while it waited in the queue
                self.pending = [(j, jb) for j, jb in self.pending if j != jid]
            self._set(busy=True, current=dict(id=jid, prompt=job["prompt"][:80]), error="")
            if job.get("kind") == "dream":
                try:
                    self._dream(jid, job, gen, pipes, dgen)
                finally:
                    self._set(busy=False, current=None, dream=None)
                continue
            if job.get("kind") == "depth":
                try:
                    t = time.time()
                    if depth is None:
                        depth = self.depth_factory() if self.depth_factory else (pipes or self._load())
                        if pipes is None and not self.depth_factory:
                            pipes = depth
                    src = self.art_dir / job["name"]
                    with Image.open(src) as im:
                        im.load()
                        im = ImageOps.exif_transpose(im)      # match art_layer's orientation
                        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                            im = im.point(lambda v: v / 256).convert("L")
                        d = depth.depth(im.convert("RGB"))
                    out = self.art_dir / (Path(job["name"]).stem + ".depth.png")
                    save_depth(d, out)
                    with self.lock:
                        self.st["done"].append(dict(id=jid, name=out.name, prompt=job["prompt"], kind="depth",
                                                    secs=round(time.time() - t, 2), show=False))
                        self.st["done"] = self.st["done"][-50:]
                    print(f"[ai] depth map {out.name} in {time.time() - t:.2f}s", flush=True)
                except Exception as ex:
                    msg = f"depth failed: {type(ex).__name__}: {ex}"
                    print(f"[ai] {msg}", flush=True)
                    self._set(error=msg)
                finally:
                    self._set(busy=False, current=None)
                continue
            try:
                if pipes is None:
                    pipes = self._load()
                init = None
                if job["init"] == "snapshot":
                    if not self.snapshot.exists():
                        raise FileNotFoundError(f"snapshot not found: {self.snapshot}")
                    with Image.open(self.snapshot) as im:
                        init = fit_init(im)
                elif isinstance(job["init"], dict):     # dream from a library image (art upload / AI / dreamed)
                    with Image.open(init_path(job["init"], self.out_dir, self.art_dir)) as im:
                        init = fit_init(im.convert("RGB"))
                for i in range(job["n"]):
                    if gen != self.cancel_gen:
                        break
                    seed = job["seed"] + i
                    t = time.time()
                    if init is not None:
                        img = pipes.img2img(job["prompt"], job["negative"], seed, init, job["strength"])
                    else:
                        img = pipes.txt2img(job["prompt"], job["negative"], seed, self.size)
                    name = self._save(img, job, seed)
                    secs = round(time.time() - t, 2)
                    with self.lock:
                        self.st["done"].append(dict(id=jid, name=name, prompt=job["prompt"][:80], kind="gen",
                                                    secs=secs, show=job["show"]))
                        self.st["done"] = self.st["done"][-50:]
                    print(f"[ai] {secs:.2f}s {name} :: {job['prompt'][:60]}", flush=True)
            except Exception as ex:  # report, never die
                msg = f"{type(ex).__name__}: {ex}"
                print(f"[ai] job failed: {msg}", flush=True)
                self._set(error=msg)                  # report first; cleanup below can be slow
                if "out of memory" in str(ex).lower():
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    traceback.print_exc()
            finally:
                self._set(busy=False, current=None)


    def _dream(self, jid, job, gen, pipes, dgen=None):
        """One DREAMIFY job: pick the device by free VRAM (moving SD-Turbo to RAM if that makes room), dream, save."""
        g0 = self.dream_cancel_gen if dgen is None else dgen

        def cancelled():
            return gen != self.cancel_gen or g0 != self.dream_cancel_gen
        t = time.time()
        folder = self.art_dir if job["source"] == "art" else self.out_dir
        offloaded, dreamer = False, None
        try:
            if cancelled():
                return
            free = self.vram_free()
            if free is not None and free < dreamify.NEED_MB[768] and pipes is not None and hasattr(pipes, "offload"):
                pipes.offload()                      # SD-Turbo -> system RAM (about 2 GB of VRAM back)
                offloaded = True
                free = self.vram_free()
            if free is not None and free >= dreamify.NEED_MB[512]:
                device = "cuda"
                side = dreamify.GPU_SIDE if free >= dreamify.NEED_MB[768] else 512
            else:
                device, side = "cpu", dreamify.CPU_SIDE
            info = dict(id=jid, name=job["name"], preset=job["preset"], source=job["source"], device=device, side=side,
                        octave=0, octaves=job["octaves"], step=0, steps=0, pct=0.0,
                        vram_free_mb=None if free is None else round(free))
            self._set(dream=dict(info))

            def progress(octave, step, total):
                info.update(octave=octave + 1, step=step, steps=total, pct=round(100.0 * step / max(total, 1), 1))
                self._set(dream=dict(info))
            with Image.open(folder / job["name"]) as im:
                if im.width * im.height > 60_000_000:
                    raise ValueError(f"image too large ({im.width}x{im.height})")
                im.load()
                im = ImageOps.exif_transpose(im)
                if im.mode in ("RGBA", "LA", "P", "PA"):
                    im = im.convert("RGBA")
                    im = Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 255)), im)
                src = im.convert("RGB")
            dreamer = self.dream_factory(device)
            out = dreamer.dream(src, job["preset"], job["strength"], job["octaves"], side, progress, cancelled)
            if out is None or cancelled():
                self._set(error="dreamify cancelled")
                print(f"[ai] dreamify cancelled: {job['name']}", flush=True)
                return
            name = dreamify.out_name(job["source"], job["name"], job["preset"], time.strftime("%Y%m%d_%H%M%S"), folder)
            secs = round(time.time() - t, 1)
            dreamify.save_png(out, folder / name, dict(
                prompt=job["prompt"], dream_source=job["name"], dream_preset=job["preset"],
                dream_strength=job["strength"], dream_octaves=job["octaves"], dream_model=dreamify.MODEL_NAME,
                dream_layers=",".join(dreamify.PRESETS[job["preset"]]["layers"]), dream_device=device, dream_secs=secs))
            with self.lock:
                self.st["done"].append(dict(id=jid, name=name, prompt=job["prompt"][:80], kind="dream",
                                            source=job["source"], secs=secs, show=False))
                self.st["done"] = self.st["done"][-50:]
            print(f"[ai] dreamify {job['preset']} {job['name']} -> {name} on {device} in {secs:.1f}s", flush=True)
            if job["source"] == "art" and self.depth_factory is not None:
                try:                                 # the fly-through wants a depth map for the new piece
                    self.submit(validate_art_name({"name": name}, self.art_dir))
                except ValueError:
                    pass
        except Exception as ex:
            msg = f"dreamify failed: {type(ex).__name__}: {ex}"
            print(f"[ai] {msg}", flush=True)
            self._set(error=msg[:300])
            if "out of memory" in str(ex).lower():
                traceback.print_exc()
        finally:
            if dreamer is not None:
                try:
                    dreamer.release()
                except Exception:
                    pass
                dreamer = None
            _free_cuda()                                 # GoogLeNet + its activations gone before SD-Turbo returns
            if offloaded:
                try:
                    pipes.restore()
                except Exception as ex:                  # RealPipes stays whole on the CPU and retries on its next job
                    self._set(error=f"SD-Turbo stays in RAM for now (restore failed, retried on the next image): {ex}"[:300])


def _free_cuda():
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def lower_priority():
    try:
        import ctypes
        BELOW_NORMAL = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--base", default="sd-turbo", choices=["sd-turbo", "sd15-lcm"])
    ap.add_argument("--lora", default=None)
    ap.add_argument("--vae-fp32", action="store_true")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no-preload", action="store_true", help="load the model on the first job instead of at start")
    a = ap.parse_args()
    lower_priority()
    token = ensure_token(HERE / ".remote_token")
    w = Worker(HERE / "textures", HERE / "snapshots" / "current.png", token, a.port,
               pipes_factory=lambda: RealPipes(a.base, a.lora, a.vae_fp32), size=a.size,
               art_dir=HERE / "art", depth_factory=RealDepth, dream_factory=dreamify.RealDream)
    w.start()
    print(f"[ai] worker on http://127.0.0.1:{w.port}  base={a.base}", flush=True)
    if not a.no_preload:   # warm the model now so the first prompt is fast
        w.q.put(("warmup", None, -1))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        w.stop()


if __name__ == "__main__":
    main()
