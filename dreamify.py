"""DREAMIFY: real DeepDream (gradient ascent on a pretrained ImageNet network) for stills, run OFFLINE by ai_worker.py.

Not realtime: a 768 px image takes tens of seconds on the GTX 1650 Ti. The live, shader-only approximation is the
"dream" filter in filters.py.

Network: torchvision GoogLeNet (Inception v1, the original DeepDream network), weights GoogLeNet_Weights.IMAGENET1K_V1
(googlenet-1378be20.pth, about 50 MB, downloaded once into the torch hub cache: %USERPROFILE%\\.cache\\torch\\hub\\checkpoints).
Method: octave pyramid (detail carried up between octaves), random jitter (roll) per step, tiled gradients (512 px
tiles on the GPU, 384 on the CPU) with the jitter hiding tile seams, L2 objective on the preset's layers, gradient
normalised by its mean magnitude. fp16 autocast on CUDA; a non-finite gradient switches the rest of the job to fp32.
The forward pass stops after the deepest layer it needs. Between steps it sleeps for half the step's GPU time so the
live show keeps its frames (`duty`).
"""
import math
import random
import re
import time
from pathlib import Path

PRESETS = {
    # name: layers {torchvision module name: weight}, octaves, octave scale, step multiplier
    "dogslugs": dict(layers={"inception4c": 1.0, "inception4d": 0.6}, octaves=4, scale=1.4, step=1.0),
    "eyes": dict(layers={"inception4d": 0.5, "inception4e": 1.0}, octaves=4, scale=1.4, step=1.1),
    "architecture": dict(layers={"inception3b": 1.0, "inception4a": 0.25}, octaves=4, scale=1.4, step=0.9),
    "psychedelic swirl": dict(layers={"inception3a": 0.5, "inception3b": 0.8, "inception4a": 0.6}, octaves=7,
                              scale=1.5, step=1.0),
}
ORDER = ["conv1", "maxpool1", "conv2", "conv3", "maxpool2", "inception3a", "inception3b", "maxpool3", "inception4a",
         "inception4b", "inception4c", "inception4d", "inception4e", "maxpool4", "inception5a", "inception5b"]
MODEL_NAME = "torchvision googlenet IMAGENET1K_V1"
WEIGHTS_MB = 50
MIN_SIDE = 96                       # smallest octave side (deeper inception blocks need some spatial extent)
GPU_SIDE, CPU_SIDE = 768, 384       # longest side of the result
GPU_TILE, CPU_TILE = 512, 384
NEED_MB = {768: 1100, 512: 800}     # free VRAM wanted before dreaming on the GPU at that size (headroom included)
EXTS = (".png", ".jpg", ".jpeg", ".webp")
TEX_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(png|jpg|jpeg|webp)$", re.I)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ pure helpers (tested without torch)
def iterations(strength):
    return int(round(6 + 18 * min(max(float(strength), 0.0), 1.0)))


def fit_size(w, h, max_side):
    s = min(1.0, max_side / float(max(w, h)))
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def octave_sizes(w, h, octaves, scale):
    """Smallest first, ending exactly at (w, h); no octave below MIN_SIDE (those are dropped)."""
    out = []
    for i in range(int(octaves) - 1, 0, -1):
        f = scale ** i
        sw, sh = int(round(w / f)), int(round(h / f))
        if min(sw, sh) >= MIN_SIDE:
            out.append((sw, sh))
    out.append((int(w), int(h)))
    return out


def tile_boxes(w, h, tile):
    """(x0, y0, x1, y1) boxes, each at most tile x tile, covering the image (even split, no sliver tiles)."""
    nx, ny = max(1, math.ceil(w / tile)), max(1, math.ceil(h / tile))
    xs = [round(i * w / nx) for i in range(nx + 1)]
    ys = [round(j * h / ny) for j in range(ny + 1)]
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(ny) for i in range(nx)]


def _plain_name(name):
    return (isinstance(name, str) and 0 < len(name) <= 160 and "\0" not in name and Path(name).name == name
            and "/" not in name
            and "\\" not in name and ":" not in name and not name.startswith(".") and ".." not in name
            and name.lower().endswith(EXTS) and not name.lower().endswith(".depth.png"))


def _num(body, key, lo, hi, default):
    v = body.get(key, default)
    if v is None:
        v = default
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not lo <= v <= hi:
        raise ValueError(f"{key} must be {lo}..{hi}")
    return v


def validate(body, art_dir, tex_dir):
    """POST /dreamify body -> job dict, or ValueError. The image must already be in the library."""
    if not isinstance(body, dict):
        raise ValueError("body must be an object")
    src = body.get("source", "art")
    if src not in ("art", "ai"):
        raise ValueError("source must be 'art' or 'ai'")
    name = body.get("name")
    if not _plain_name(name) or (src == "ai" and not TEX_NAME.match(name)):
        raise ValueError("bad image name")
    folder = Path(art_dir if src == "art" else tex_dir)
    f = folder / name
    if not f.is_file() or f.resolve().parent != folder.resolve():
        raise ValueError("image not found in the library")
    preset = body.get("preset") or "dogslugs"
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}")
    strength = float(_num(body, "strength", 0.05, 1.0, 0.5))
    octaves = int(_num(body, "octaves", 1, 10, PRESETS[preset]["octaves"]))
    return dict(kind="dream", name=name, source=src, strength=strength, octaves=octaves, preset=preset,
                prompt=f"dreamify {preset}: {name}"[:120], n=1)


def _slug(s, n=48):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_-")[:n].strip("_-") or "image"


def out_name(source, name, preset, stamp, folder=None):
    """New file name; with `folder`, a short suffix is added while the name is taken (two dreams in one second)."""
    for k in range(100):
        sfx = "" if k == 0 else f"_{k}"
        if source == "ai":
            n = f"gen_dream_{stamp}_{_slug(preset, 20)}{sfx}.png"        # gen_ prefix: the picker lists it as AI
        else:
            n = f"{_slug(Path(name).stem)}_dream_{_slug(preset, 20)}_{stamp}{sfx}.png"
        if folder is None or not (Path(folder) / n).exists():
            return n
    return n


def save_png(img, path, meta):
    """Atomic: <name>.tmp then os.replace (the texture / art scanners ignore .tmp)."""
    import os
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        info.add_text(str(k), str(v))
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        img.save(tmp, format="PNG", pnginfo=info)
        os.replace(tmp, path)
    finally:
        if tmp.exists():                        # failed half-way: never leave a .tmp behind
            try:
                tmp.unlink()
            except OSError:
                pass


def free_vram_mb():
    """Free VRAM on the device in MB (all processes counted), None when CUDA is not usable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 2 ** 20
    except Exception:
        return None


def light_mask(img, grow=None):
    """0..1 float mask (H, W): 1 where the picture has light, spread a little (dream detail may overhang its edges),
    0 in true black. DeepDream otherwise grows texture out of black, which would light up the wall."""
    import numpy as np
    from PIL import ImageFilter
    g = grow or max(3, (max(img.size) // 96) | 1)
    m = img.convert("RGB").point(lambda v: min(255, v * 4)).convert("L")
    m = m.filter(ImageFilter.MaxFilter(g)).filter(ImageFilter.GaussianBlur(g * 0.6))
    a = np.asarray(m, "float32") / 255.0
    return np.clip((a - 0.04) / 0.5, 0.0, 1.0)


# ------------------------------------------------------------------ the real thing
class _Stop(Exception):
    pass


class RealDream:
    def __init__(self, device="cuda", fp16=True, duty=0.5):
        import torch
        from torchvision.models import GoogLeNet_Weights, googlenet
        self.torch, self.device = torch, device
        if device == "cpu":
            import os
            torch.set_num_threads(max(2, (os.cpu_count() or 4) // 4))   # leave the show's cores alone
        m = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        m.eval().requires_grad_(False)
        self.model = m.to(device)
        self.fp16 = bool(fp16 and device == "cuda")
        self.duty = duty
        self.acts = {}
        self._deepest = None
        for name in ORDER:
            getattr(m, name).register_forward_hook(self._hook(name))

    def _hook(self, name):
        def fn(_mod, _inp, out):
            if name in self.want:
                self.acts[name] = out
            if name == self._deepest:
                raise _Stop()
        return fn

    want = ()

    def release(self):
        self.acts = {}
        try:
            if self.model is not None:
                self.model.to("cpu")
        except Exception:
            pass
        self.model = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def _grad(self, x, layers):
        torch = self.torch
        x = x.detach().requires_grad_(True)
        self.acts = {}
        try:
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                    self.model(x)
            except _Stop:
                pass
            loss = sum(w * self.acts[n].float().pow(2).mean() for n, w in layers.items())
            loss.backward()
            return x.grad.detach()
        finally:
            self.acts = {}                          # no activation / autograd graph outlives the step

    def dream(self, img, preset, strength, octaves, max_side, progress, cancelled):
        """PIL image -> dreamed PIL image (long side max_side), or None when cancelled."""
        import torch.nn.functional as F
        from PIL import Image
        torch = self.torch
        p = PRESETS[preset]
        layers = p["layers"]
        self.want = tuple(layers)
        self._deepest = max(layers, key=ORDER.index)
        w, h = fit_size(img.width, img.height, max_side)
        img = img.convert("RGB").resize((w, h), Image.LANCZOS)
        mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)
        x0 = torch.from_numpy(__import__("numpy").asarray(img, "float32") / 255.0).permute(2, 0, 1)[None]
        base = (x0.to(self.device) - mean) / std
        lo, hi = (0 - mean) / std, (1 - mean) / std
        sizes = octave_sizes(w, h, octaves, p["scale"])
        iters = iterations(strength)
        step = 0.045 * (0.5 + strength) * p["step"]
        tile = GPU_TILE if self.device == "cuda" else CPU_TILE
        total, done = len(sizes) * iters, 0
        detail = torch.zeros_like(F.interpolate(base, size=sizes[0][::-1], mode="bilinear", align_corners=False))
        rnd = random.Random(1234)
        for oi, (sw, sh) in enumerate(sizes):
            src = F.interpolate(base, size=(sh, sw), mode="bilinear", align_corners=False)
            x = src + F.interpolate(detail, size=(sh, sw), mode="bilinear", align_corners=False)
            for _ in range(iters):
                if cancelled():
                    return None
                t0 = time.perf_counter()
                ox, oy = rnd.randint(-32, 32), rnd.randint(-32, 32)
                xr = torch.roll(x, (oy, ox), (2, 3))
                g = torch.zeros_like(xr)
                for x0_, y0_, x1_, y1_ in tile_boxes(sw, sh, tile):
                    g[..., y0_:y1_, x0_:x1_] = self._grad(xr[..., y0_:y1_, x0_:x1_], layers)
                if not torch.isfinite(g).all():                     # fp16 overflow: redo this step in fp32
                    if not self.fp16:
                        g = torch.nan_to_num(g, 0.0, 0.0, 0.0)
                    else:
                        self.fp16 = False
                        for x0_, y0_, x1_, y1_ in tile_boxes(sw, sh, tile):
                            g[..., y0_:y1_, x0_:x1_] = self._grad(xr[..., y0_:y1_, x0_:x1_], layers)
                        g = torch.nan_to_num(g, 0.0, 0.0, 0.0)
                g = torch.roll(g, (-oy, -ox), (2, 3))
                x = torch.max(torch.min(x + step * g / (g.abs().mean() + 1e-8), hi), lo).detach()
                done += 1
                progress(oi, done, total)
                if self.device == "cuda" and self.duty > 0:
                    torch.cuda.synchronize()
                    time.sleep(min(0.25, (time.perf_counter() - t0) * self.duty))   # give the show its GPU time
            detail = x - src
        import numpy as np
        out = (x * std + mean).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        out *= light_mask(img)[..., None]          # projection rule: where the picture was black, it stays black
        return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))
