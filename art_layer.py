"""Uploaded art as a living, flyable 3D relief (Jonathan Singer / CoSM style).

A flat painting from the `art/` folder becomes a displaced grid mesh lit by a moving key light. The
performer flies a perspective camera over it and melts, liquifies, slices, folds, Droste-spirals and
slit-scan-smears it live. The piece floats on pure black with feathered edges.

Art library format (shared with control.py / ai_worker.py):
    art/<stem>.png|jpg|jpeg|webp      the painting
    art/<stem>.depth.png              optional depth map, 8/16-bit grayscale, white = near (Depth Anything V2)

    list_art(folder) -> ["a.png", "b.jpg", ...]   file names in CREATION order (oldest first, so uploads
                                                  append at the end), depth maps excluded
    values["art_index"] -> list_art(folder)[round(art_index) % len], resolved when art_index changes; the
                           piece on screen is then held by NAME, so the list changing never swaps it

API:
    ap = ArtPass(ctx, (1280, 800), Path("art"))
    tex = ap.render(values, clock, t, travel_z, dt)    # RGBA8 moderngl.Texture, premultiplied on black
    ap.release()

Look checks:  python art_layer.py --gallery   ->  test_frames/art_gallery/contact_sheet.png
Timings:      python art_layer.py --bench
"""
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import moderngl

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
DEPTH_SUFFIX = ".depth.png"
MAX_COLOR_DIM = 1536          # longest side of the colour texture
MAX_GEO_DIM = 768             # longest side of the depth / normal texture
CACHE_MAX = 3                 # pieces kept on the GPU
CROSSFADE_S = 1.5
RESCAN_S = 2.0
SETTLE_S = 2.0                # a file must be this old (mtime) before it is decoded: half-written uploads
MAX_SOURCE_PIXELS = 60_000_000   # larger files dropped straight into art/ are refused, not decoded
STRIPE_BYTES = 384 * 1024     # texture uploads go up in row stripes of about this size ...
UPLOAD_BUDGET_MS = 1.0        # ... and a frame stops starting new stripes after this long
GL_RETRY_S = 0.4              # pause before a failed GPU upload is tried again
DECODE_TIMEOUT_S = 30.0       # a decode server stuck on one file is killed after this long
N_SLICES = 6
HIST_COLS, HIST_ROWS = 8, 4   # slit-scan ring: 32 frames in one atlas texture
HIST_N = HIST_COLS * HIST_ROWS
HIST_DT = 1.0 / 30.0          # ring cadence (time based, so the smear length does not depend on fps)
GRID_LONG, GRID_SHORT = 320, 200
FOV_DEG = 38.0
DETAIL_GAIN = 0.14            # how much luminance emboss rides on top of the base depth
UNIT0 = 9                     # texture units 9..15 are used and left bound: a compositor sampler must not live there
                              # (live_visuals: 0..5 scene, 6 beings, 7 land, 8 flat art; tests/test_beings.py checks beings)

# copy of live_visuals.PALETTES (5 RGB stops each): only used by the optional light chase
PALETTES = [
    [(0.00, 0.85, 0.80), (1.00, 0.05, 0.55), (1.00, 0.70, 0.05), (0.28, 0.10, 1.00), (0.10, 0.90, 0.30)],
    [(1.00, 0.68, 0.08), (1.00, 0.30, 0.02), (0.55, 0.08, 1.00), (0.10, 0.75, 1.00), (0.85, 0.10, 0.75)],
    [(0.05, 0.22, 1.00), (1.00, 0.52, 0.05), (0.00, 0.90, 0.75), (0.90, 0.15, 0.42), (0.30, 0.10, 0.90)],
    [(0.00, 0.85, 0.35), (0.55, 1.00, 0.12), (1.00, 0.78, 0.08), (0.00, 0.62, 0.60), (1.00, 0.16, 0.08)],
    [(1.00, 0.06, 0.10), (1.00, 0.80, 0.00), (0.05, 0.95, 0.25), (0.05, 0.40, 1.00), (0.85, 0.05, 1.00)],
]

DEFAULTS = dict(
    art_index=0.0, art_relief=0.4, cam_yaw=0.0, cam_pitch=0.0, cam_roll=0.0, cam_dolly=0.0, cam_drift=0.3,
    art_melt=0.0, art_liquify=0.0, art_slices=0.0, art_breathe=0.3, art_fold=0.0, art_droste=0.0,
    art_smear=0.0, art_glow=0.3, zoom=0.0, swim=0.0, twirl=0.0, ripple=0.0, symmetry=1.0, folds=8.0,
    rotate=0.0, chase=0.0, chase_width=0.25, chase_rate=1.0, palette=0.0, hue=0.0, saturation=1.0,
    tunnel_curve=0.2, punch=0.0, freeze=0.0)


# ----------------------------------------------------------------------------------------------------------
# library
# ----------------------------------------------------------------------------------------------------------
def _is_art(name):
    low = name.lower()
    return low.endswith(IMAGE_EXT) and not low.endswith(DEPTH_SUFFIX) and not name.startswith(".")


def _created_ns(st):
    """Creation time: st_birthtime where the platform has it, else st_ctime (creation time on Windows)."""
    v = getattr(st, "st_birthtime_ns", None)
    if v is None:
        b = getattr(st, "st_birthtime", None)
        v = int(b * 1e9) if b is not None else st.st_ctime_ns
    return v


def _scan_entries(folder):
    """[(name, stat)] of the paintings in creation order (oldest first, ties by name). Never raises."""
    out = []
    try:
        with os.scandir(folder) as it:
            for e in it:
                try:
                    if _is_art(e.name) and e.is_file():
                        out.append((e.name, e.stat()))
                except OSError:
                    pass
    except OSError:
        return []
    out.sort(key=lambda ns: (_created_ns(ns[1]), ns[0].casefold(), ns[0]))
    return out


def list_art(folder):
    """File names of the paintings in `folder` (depth maps excluded) in CREATION order, oldest first, so a
    new upload always appends at the end and existing art_index values keep their piece.

    Never raises: a missing or unreadable folder is an empty library.
    """
    return [n for n, _ in _scan_entries(folder)]


def depth_path(folder, name):
    """Where the depth map of painting `name` lives (it may not exist)."""
    return Path(folder) / (Path(name).stem + DEPTH_SUFFIX)


def _stat_sig(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def scan_library(folder):
    """-> (names, {name: (sig, newest_mtime_seconds)}); sig changes whenever the image or its depth map does."""
    names, info = [], {}
    for n, st in _scan_entries(folder):
        a = (st.st_mtime_ns, st.st_size)
        d = _stat_sig(depth_path(folder, n))
        info[n] = ((a, d), max(a[0], d[0] if d else 0) / 1e9)
        names.append(n)
    return names, info


# ----------------------------------------------------------------------------------------------------------
# decode + depth (worker thread, numpy / PIL only)
# ----------------------------------------------------------------------------------------------------------
def _box1(a, r, axis):
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi, lo = [slice(None)] * a.ndim, [slice(None)] * a.ndim
    hi[axis], lo[axis] = slice(2 * r + 1, 2 * r + 1 + n), slice(0, n)
    return ((c[tuple(hi)] - c[tuple(lo)]) / (2 * r + 1)).astype(np.float32)


def blur(a, sigma):
    """Gaussian-ish blur (three box passes), edge-clamped, pure numpy. Wide blurs run on a block-averaged
    copy and are scaled back up: same soft result at a fraction of the cost."""
    a = np.asarray(a, dtype=np.float32)
    if sigma <= 0.3:
        return a
    f = int(sigma // 3)
    if f >= 2 and a.ndim == 2 and min(a.shape) >= 8 * f:
        h, w = a.shape
        ph, pw = (-h) % f, (-w) % f
        b = np.pad(a, ((0, ph), (0, pw)), mode="edge")
        b = b.reshape(b.shape[0] // f, f, b.shape[1] // f, f).mean(axis=(1, 3))
        b = blur(b, sigma / f)
        up = Image.fromarray(b, mode="F").resize((b.shape[1] * f, b.shape[0] * f), Image.BICUBIC)
        return np.asarray(up, dtype=np.float32)[:h, :w]
    r = max(1, int(round(sigma * 0.94)))
    out = a
    for _ in range(3):
        out = _box1(_box1(out, r, 0), r, 1)
    return out


def _norm(a, lo_pct=1.0, hi_pct=99.0):
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.full(a.shape, 0.5, np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def pseudo_depth(rgb):
    """Soft relief from a painting with no depth map: blurred luminance + saturation + edge density.

    rgb: float32 (h, w, 3) in 0..1. Returns float32 (h, w) in 0..1, 1 = near. Deliberately low-frequency:
    the fine emboss is added separately (detail_map), so this never turns into noise.
    """
    h, w = rgb.shape[:2]
    dim = float(max(h, w))
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    gy, gx = np.gradient(blur(lum, dim * 0.004))
    edge = _norm(blur(np.hypot(gx, gy), dim * 0.03), 2, 98)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = ((xx / max(w - 1, 1) - 0.5) ** 2 + (yy / max(h - 1, 1) - 0.5) ** 2) * 4.0
    base = (0.42 * blur(lum, dim * 0.035) + 0.22 * blur(lum, dim * 0.012) + 0.20 * blur(sat, dim * 0.03)
            + 0.10 * edge + 0.14 * (1.0 - np.clip(rr, 0, 1)))
    return blur(_norm(base, 2, 98), dim * 0.006)


def detail_map(rgb):
    """High-frequency emboss (-1..1): luminance minus its blur plus a little saturation, so beads and lines
    stand up even when the depth map is only a smooth bowl."""
    dim = float(max(rgb.shape[:2]))
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    d = (lum - blur(lum, dim * 0.02)) + 0.35 * (sat - blur(sat, dim * 0.02))
    d = blur(d, dim * 0.0016)
    s = float(d.std())
    if s < 1e-5:
        return np.zeros_like(d)
    return np.tanh(d / (2.2 * s)).astype(np.float32)


def _read_depth(path, size):
    """Depth map file -> float32 (h, w) 0..1 at `size` (w, h), robustly normalised. Raises on a bad file."""
    with Image.open(path) as im:
        im.load()
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.asarray(im).astype(np.float32)
        elif im.mode == "F":
            a = np.asarray(im, dtype=np.float32)
        else:
            a = np.asarray(im.convert("L")).astype(np.float32)
    a = np.nan_to_num(a)
    a = np.asarray(Image.fromarray(a, mode="F").resize(size, Image.BILINEAR), dtype=np.float32)
    return _norm(a, 0.5, 99.5)


def _to_8bit(im):
    """16-bit / 32-bit / float grayscale -> 8-bit "L" with its tones kept (PIL's own convert clips to white)."""
    a = np.nan_to_num(np.asarray(im).astype(np.float32))
    top = float(a.max()) if a.size else 0.0
    if im.mode.startswith("I;16"):
        hi = 65535.0
    elif im.mode == "F":
        hi = 1.0 if top <= 1.0 else (255.0 if top <= 255.0 else top)
    else:                                                   # "I": whatever range the file really uses
        hi = 255.0
        for bits in (255.0, 4095.0, 65535.0):
            hi = bits
            if top <= bits:
                break
        else:
            hi = top
    return Image.fromarray(np.clip(a / max(hi, 1e-6) * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")


def _mips(a):
    """Box-filtered mip chain below `a` (h, w, c), down to 1x1, with GL's floor(n / 2) level sizes."""
    out, cur = [], a
    is_int = np.issubdtype(a.dtype, np.integer)
    while max(cur.shape[:2]) > 1:
        h, w, c = cur.shape
        h2, w2 = max(1, h // 2), max(1, w // 2)
        f = cur[:h2 * 2 if h > 1 else 1, :w2 * 2 if w > 1 else 1].astype(np.float32)
        if h > 1:
            f = f.reshape(h2, 2, f.shape[1], c).mean(axis=1)
        if w > 1:
            f = f.reshape(h2, w2, 2, c).mean(axis=2)
        cur = np.ascontiguousarray((f + 0.5).astype(a.dtype) if is_int else f.astype(a.dtype))
        out.append(cur)
    return out


def decode_piece(img_path, dpath=None, max_pixels=None):
    """Decode one painting (+ optional depth map) into GPU-ready arrays. Raises on corrupt / oversized input.

    Returns dict(color=uint8 (H, W, 4) bottom-up PREMULTIPLIED, geo=float16 (h, w, 4) bottom-up,
    color_mips=[...], geo_mips=[...] (levels 1.. down to 1x1), aspect, has_depth).
    geo = (base depth, d height/du, d height/dv, detail) with height = base + DETAIL_GAIN * detail.
    """
    cap = MAX_SOURCE_PIXELS if max_pixels is None else max_pixels
    if Image.MAX_IMAGE_PIXELS:
        cap = min(cap, Image.MAX_IMAGE_PIXELS)
    with Image.open(img_path) as im:
        if im.width * im.height > cap:
            raise ValueError(f"image too large: {im.width}x{im.height} > {cap} px")
        try:
            im.draft("RGB", (MAX_COLOR_DIM, MAX_COLOR_DIM))       # JPEG: decode at 1/2, 1/4, 1/8 scale directly
        except Exception:
            pass
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
            im = _to_8bit(im)
        k = max(im.size) // MAX_COLOR_DIM
        if k >= 2:                                                # cheap integer box reduce before the LANCZOS pass
            if im.mode not in ("L", "LA", "RGB", "RGBA"):
                im = im.convert("RGBA")
            im = im.reduce(k)
        im = im.convert("RGBA")
    if im.width < 2 or im.height < 2:
        raise ValueError("image too small")
    if max(im.size) > MAX_COLOR_DIM:
        s = MAX_COLOR_DIM / max(im.size)
        im = im.resize((max(2, round(im.width * s)), max(2, round(im.height * s))), Image.LANCZOS)
    aspect = im.width / im.height
    color = np.array(np.asarray(im)[::-1])
    if color[..., 3].min() < 255:                                 # premultiply: no colour fringes in the mips
        a16 = color[..., 3:4].astype(np.uint16)
        color[..., :3] = ((color[..., :3].astype(np.uint16) * a16 + 127) // 255).astype(np.uint8)
    color = np.ascontiguousarray(color)

    s = min(1.0, MAX_GEO_DIM / max(im.size))
    gsize = (max(2, round(im.width * s)), max(2, round(im.height * s)))
    small = np.asarray(im.resize(gsize, Image.BILINEAR), dtype=np.float32) / 255.0
    rgb = small[..., :3] * small[..., 3:4]
    dim = float(max(gsize))
    has_depth = False
    base = None
    if dpath is not None and os.path.exists(dpath):
        try:
            base = blur(_read_depth(dpath, gsize), dim * 0.003)
            has_depth = True
        except Exception:
            base = None                      # a broken depth map must not take the painting down with it
    if base is None:
        base = pseudo_depth(rgb)
    detail = detail_map(rgb)
    height = blur(base + DETAIL_GAIN * detail, 1.0)
    gy, gx = np.gradient(height)
    geo = np.stack([base, gx * gsize[0], -gy * gsize[1], detail], axis=2)[::-1]   # v axis points up after the flip
    geo = np.ascontiguousarray(np.nan_to_num(geo).astype(np.float16))
    return dict(color=color, geo=geo, color_mips=_mips(color), geo_mips=_mips(geo), aspect=float(aspect),
                has_depth=has_depth)


# ----------------------------------------------------------------------------------------------------------
# decode server: a child process, so PIL / numpy never hold the render process' GIL
# ----------------------------------------------------------------------------------------------------------
class DecodeError(Exception):
    """The file itself is bad (corrupt, too large, ...): do not retry until it changes."""


def _serve_decodes():
    """Child side: one JSON request per stdin line -> one JSON header line + raw array bytes on stdout."""
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                                   # nothing else may ever write into the pipe
    for line in sys.stdin.buffer:
        try:
            req = json.loads(line)
        except Exception:
            continue
        try:
            d = decode_piece(req["img"], req.get("depth"), req.get("max_pixels"))
            cols, geos = [d["color"]] + d["color_mips"], [d["geo"]] + d["geo_mips"]
            head = dict(ok=True, aspect=d["aspect"], has_depth=d["has_depth"],
                        color=[list(a.shape[:2]) for a in cols], geo=[list(a.shape[:2]) for a in geos])
            out.write((json.dumps(head) + "\n").encode())
            for a in cols + geos:
                out.write(a.data)
        except Exception as ex:
            out.write((json.dumps(dict(ok=False, err=repr(ex)[:300])) + "\n").encode())
        out.flush()


class _Decoder:
    """Parent side of the decode server. mode "process" normally; falls back to in-thread decoding
    ("thread") if the child cannot be started twice in a row."""

    def __init__(self):
        self.proc = None
        self.mode = "process"
        self.closed = False
        self.timed_out = False
        self._plock = threading.Lock()

    def start(self):
        with self._plock:
            if self.closed or self.mode != "process" or (self.proc is not None and self.proc.poll() is None):
                return
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--decode-server"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags)

    def kill(self, timed_out=False):
        """Safe from any thread: a blocked read in decode() sees EOF and gives up."""
        proc = self.proc
        if timed_out:
            self.timed_out = True
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def close(self):
        self.closed = True
        self.kill()

    def _ask(self, img, dpath, max_pixels):
        proc = self.proc
        req = dict(img=str(img), depth=str(dpath) if dpath is not None else None, max_pixels=max_pixels)
        proc.stdin.write((json.dumps(req) + "\n").encode())
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise EOFError("decode server went away")
        head = json.loads(line)
        if not head.get("ok"):
            raise DecodeError(head.get("err", "decode failed"))

        def grab(shapes, dtype):
            arrs = []
            for h, w in shapes:
                n = h * w * 4 * np.dtype(dtype).itemsize
                buf = proc.stdout.read(n)                      # blocking pipe read: releases the GIL
                if len(buf) != n:
                    raise EOFError("decode server went away mid-piece")
                arrs.append(np.frombuffer(buf, dtype=dtype).reshape(h, w, 4))
            return arrs
        cols, geos = grab(head["color"], np.uint8), grab(head["geo"], np.float16)
        return dict(color=cols[0], color_mips=cols[1:], geo=geos[0], geo_mips=geos[1:],
                    aspect=float(head["aspect"]), has_depth=bool(head["has_depth"]))

    def decode(self, img, dpath=None, max_pixels=None):
        if self.mode == "process":
            self.timed_out = False
            for _ in range(2):
                if self.closed:
                    raise RuntimeError("decoder closed")
                try:
                    self.start()
                    return self._ask(img, dpath, max_pixels)
                except DecodeError:
                    raise
                except Exception:
                    self.kill()
                    if self.timed_out:
                        raise DecodeError("decode timed out")
            if self.closed:
                raise RuntimeError("decoder closed")
            self.mode = "thread"
        try:
            return decode_piece(img, dpath, max_pixels)
        except Exception as ex:
            raise DecodeError(repr(ex))


# ----------------------------------------------------------------------------------------------------------
# shaders
# ----------------------------------------------------------------------------------------------------------
GLSL_COMMON = """
#version 330
uniform sampler2D u_colA, u_geoA, u_colB, u_geoB;
uniform float u_mix, u_wA, u_wB, u_asp, u_t;
uniform float u_twirl, u_fold, u_folds, u_foldrot, u_droste, u_dls, u_dzoom, u_feather;
const float PI = 3.14159265, TAU = 6.2831853;

float hash21(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
  return mix(mix(hash21(i), hash21(i+vec2(1,0)), f.x), mix(hash21(i+vec2(0,1)), hash21(i+vec2(1,1)), f.x), f.y);
}
vec2 uv2p(vec2 uv){ return (uv-0.5)*2.0*vec2(u_asp, 1.0); }
vec2 p2uv(vec2 p){ return p/(2.0*vec2(u_asp, 1.0))+0.5; }

// soft squircle: the piece never ends in a hard rectangle
float edgemask(vec2 uv){
  vec2 q = abs(uv-0.5)*2.0; q *= q; q *= q;
  float d = sqrt(sqrt(q.x+q.y));
  return 1.0-smoothstep(1.0-u_feather, 1.0, d);
}

// twirl + kaleidoscope fold, in centred picture space
vec2 warp_pre(vec2 p){
  float r = length(p);
  if (u_twirl != 0.0) {
    float k = 1.0-smoothstep(0.0, 1.6, r);
    float a = u_twirl*2.6*k*k, c = cos(a), s = sin(a);
    p = mat2(c, -s, s, c)*p;
  }
  if (u_fold > 0.001) {
    float seg = TAU/u_folds;
    float a = atan(p.x, p.y+1e-6);
    a = abs(mod(a+0.5*seg, seg)-0.5*seg)+u_foldrot;       // mirrored wedge around "up": continuous everywhere
    p = mix(p, r*vec2(sin(a), cos(a)), u_fold);
  }
  return p;
}

// conformal Escher / Droste log-spiral: p1 = this level, p2 = the next copy in (for the seam crossfade)
void droste(vec2 p, out vec2 p1, out vec2 p2, out float bw, out float s1){
  float R = 0.84*min(u_asp, 1.0);
  float r = max(length(p), 1e-5);
  float L = log(r/R), th = atan(p.y, p.x+1e-6), k = u_dls/TAU;
  float L2 = L+th*k-u_dzoom, th2 = th-L*k;
  float Lw = -mod(-L2, u_dls);
  float r1 = R*exp(Lw);
  vec2 d = vec2(cos(th2), sin(th2));
  p1 = r1*d; s1 = r1/r;
  p2 = p1*exp(-u_dls);
  bw = smoothstep(-0.35*u_dls, 0.0, Lw);
}
"""

MESH_VS = GLSL_COMMON + """
in vec2 in_uv;
uniform mat4 u_vp;
uniform float u_scale, u_amp, u_nsl, u_slices, u_sep, u_melt, u_liq, u_swim, u_flowt, u_ripple, u_beat, u_lod;
out vec2 v_uv, v_off, v_vel;
out float v_ph;
out vec3 v_wpos;
flat out float v_k;

vec4 geo_at(vec2 uv){
  vec4 g = vec4(0.5, 0.0, 0.0, 0.0);
  if (u_mix < 0.999) g = textureLod(u_geoA, uv, u_lod);
  if (u_mix > 0.001) g = mix(g, textureLod(u_geoB, uv, u_lod), u_mix);
  return g;
}
// height (base + emboss) and base depth at a mesh uv, through the same warps the fragment shader uses
vec2 height_at(vec2 uv){
  vec2 p = warp_pre(uv2p(uv));
  vec2 q = p2uv(p);
  vec4 g = geo_at(q);
  vec2 hb = mix(vec2(0.5), vec2(g.x+%(DG)s*0.6*g.w, g.x), edgemask(q));
  if (u_droste > 0.001) {
    vec2 p1, p2; float bw, s1;
    droste(p, p1, p2, bw, s1);
    vec2 q1 = p2uv(p1), q2 = p2uv(p2);
    vec4 g1 = mix(geo_at(q1), geo_at(q2), bw);
    vec2 hd = mix(vec2(0.5), vec2(g1.x+%(DG)s*0.6*g1.w, g1.x), mix(edgemask(q1), edgemask(q2), bw));
    hb = mix(hb, hd, smoothstep(0.0, 0.3, u_droste));
  }
  return hb;
}
float psi(vec2 p){ return vnoise(p)+0.5*vnoise(p*2.03+7.1); }

void main(){
  vec2 uv = in_uv;
  vec2 p = uv2p(uv);
  // gentle whole-picture swim (direct offset) + liquid wobble
  vec2 off = u_swim*0.012*vec2(sin(p.y*3.1+u_t*0.7)+0.5*sin(p.y*7.3-u_t*0.9), cos(p.x*2.9+u_t*0.6)+0.5*cos(p.x*6.7+u_t*0.8));
  vec2 vel = vec2(0.0);
  float ph = 0.0;
  if (u_liq > 0.001) {                                   // curl noise: divergence-free swirl
    vec2 q = p*1.7+vec2(u_t*0.045, -u_t*0.037); float e = 0.03;
    float a = psi(q), bx = psi(q+vec2(e, 0.0)), by = psi(q+vec2(0.0, e));
    vec2 curl = vec2(by-a, -(bx-a))/e;
    vel += curl*u_liq*0.085;
    off += curl*u_liq*0.012*sin(u_t*0.5+a*6.0);
  }
  if (u_melt > 0.001) {                                  // drips: column strands, near (raised) paint runs faster
    float d0 = height_at(uv).y;
    float s1 = vnoise(vec2(p.x*4.5, u_t*0.04)), s2 = vnoise(vec2(p.x*11.0+3.0, 1.7+u_t*0.02));
    float strand = 0.18+0.82*s1*s1*(0.55+0.9*s2);
    float sway = (vnoise(vec2(p.x*2.0+u_t*0.1, p.y*1.5))-0.5)*0.22;
    vel += u_melt*vec2(sway, -1.0)*strand*(0.35+1.1*d0)*0.42;
  }
  ph = 0.3*vnoise(p*1.3+11.0);
  v_uv = uv; v_off = off; v_vel = vel; v_ph = ph;

  vec2 hb;
  if (dot(vel, vel) > 1e-10) {
    float f1 = fract(u_flowt+ph), f2 = fract(u_flowt+ph+0.5), w1 = 1.0-abs(2.0*f1-1.0);
    hb = height_at(uv+off-vel*f1)*w1+height_at(uv+off-vel*f2)*(1.0-w1);
  } else hb = height_at(uv+off);

  float rr = length(p);
  float z = (hb.x-0.5)*u_amp*(1.0-0.7*u_slices);
  z += u_ripple*0.035*sin(rr*14.0-TAU*u_beat)*exp(-1.6*rr);          // a ring rides outward on each beat
  vec3 pos = vec3(p*u_scale, z);
  float k = float(gl_InstanceID);
  if (u_nsl > 1.5) {
    float c = (k+0.5)/u_nsl-0.5;
    pos.z += c*u_sep*u_slices;
    float ra = c*u_slices*0.16*sin(u_t*0.31+k*1.7), cs = cos(ra), sn = sin(ra);
    pos.xy = mat2(cs, -sn, sn, cs)*pos.xy+u_slices*0.045*vec2(sin(u_t*0.27+k*2.1), cos(u_t*0.23+k*1.3));
  }
  v_k = k; v_wpos = pos;
  gl_Position = u_vp*vec4(pos, 1.0);
}
""" % dict(DG=DETAIL_GAIN)

MESH_FS = GLSL_COMMON + """
in vec2 v_uv, v_off, v_vel;
in float v_ph;
in vec3 v_wpos;
flat in float v_k;
uniform float u_scale, u_amp, u_nsl, u_slices, u_flowt, u_glow, u_light, u_lt, u_chase, u_chw, u_chph, u_close;
uniform vec3 u_cam;
uniform vec2 u_res;
uniform vec3 u_pal[5];
uniform mat3 u_colmat;
layout(location = 0) out vec4 o_col;
layout(location = 1) out vec4 o_aux;

vec2 G_dx, G_dy;

void tap(vec2 uv, float gs, out vec4 c, out vec4 g){
  float m = edgemask(uv);
  c = vec4(0.0); g = vec4(0.5, 0.0, 0.0, 0.0);
  if (m <= 0.0) return;
  vec2 dx = G_dx*gs, dy = G_dy*gs;
  vec4 ca = vec4(0.0), cb = vec4(0.0), ga = g, gb = g;
  // colour textures are premultiplied at decode time (no fringes from transparent texels in the mips)
  if (u_mix < 0.999) { ca = textureGrad(u_colA, uv, dx, dy)*u_wA; ga = textureGrad(u_geoA, uv, dx, dy); }
  if (u_mix > 0.001) { cb = textureGrad(u_colB, uv, dx, dy)*u_wB; gb = textureGrad(u_geoB, uv, dx, dy); }
  c = mix(ca, cb, u_mix)*m;
  g = mix(ga, gb, u_mix);
}

void sample_w(vec2 uv, out vec4 c, out vec4 g){
  vec2 p = warp_pre(uv2p(uv));
  float f = smoothstep(0.0, 0.3, u_droste);
  c = vec4(0.0); g = vec4(0.5, 0.0, 0.0, 0.0);
  if (f < 0.999) tap(p2uv(p), 1.0, c, g);
  if (f > 0.0) {
    vec2 p1, p2; float bw, s1; vec4 c1, g1, c2, g2;
    droste(p, p1, p2, bw, s1);
    tap(p2uv(p1), s1, c1, g1);
    if (bw > 0.0) { tap(p2uv(p2), s1*exp(-u_dls), c2, g2); c1 = mix(c1, c2, bw); g1 = mix(g1, g2, bw); }
    c = mix(c, c1, f); g = mix(g, g1, f);
  }
}

vec3 palette(float x){
  x = fract(x)*5.0;
  int i = int(floor(x)); float f = fract(x); f = f*f*(3.0-2.0*f);
  return mix(u_pal[i%5], u_pal[(i+1)%5], f);
}

void main(){
  G_dx = dFdx(v_uv); G_dy = dFdy(v_uv);
  vec2 uv = v_uv+v_off;
  vec4 c, g;
  if (dot(v_vel, v_vel) > 1e-10) {                       // flow-map: two phases half a cycle apart, crossfaded
    float f1 = fract(u_flowt+v_ph), f2 = fract(u_flowt+v_ph+0.5), w1 = 1.0-abs(2.0*f1-1.0);
    vec4 c2, g2;
    sample_w(uv-v_vel*f1, c, g);
    sample_w(uv-v_vel*f2, c2, g2);
    c = c*w1+c2*(1.0-w1); g = g*w1+g2*(1.0-w1);
  } else sample_w(uv, c, g);
  c *= edgemask(v_uv);

  float alpha = c.a, d = clamp(g.x, 0.0, 1.0), sedge = 0.0, shadow = 1.0;
  if (u_nsl > 1.5) {                                     // depth segmentation: this instance owns one depth band
    float si = min(d, 0.9999)*u_nsl-v_k;
    float fw = clamp(fwidth(d)*u_nsl*1.5, 0.02, 0.25);
    float lo = v_k < 0.5 ? 1.0 : smoothstep(-fw, fw, si);
    float hi = v_k > u_nsl-1.5 ? 1.0 : 1.0-smoothstep(1.0-fw, 1.0+fw, si);
    float own = lo*hi;
    sedge = (1.0-abs(2.0*own-1.0))*step(0.02, alpha);
    shadow = mix(0.16, 1.0, hi);                          // paint owned by a nearer layer: dim underlay
    alpha *= lo; c *= lo;
  }
  if (alpha < 0.004) discard;
  vec3 base = c.rgb/max(c.a, 1e-4);

  // ---- relief lighting: moving key light, specular varnish, rim
  vec2 slope = g.yz*u_amp/(2.0*u_scale*vec2(u_asp, 1.0));
  slope /= sqrt(1.0+0.35*dot(slope, slope));             // soft limit: cliffs do not turn into noise
  vec3 n = normalize(vec3(-slope, 1.0));
  vec3 V = normalize(u_cam-v_wpos);
  vec3 L = normalize(vec3(0.72*cos(u_lt), 0.72*sin(u_lt*0.83+0.6), 0.62));
  float ndl = max(dot(n, L), 0.0)/L.z;
  float bump = clamp(length(slope)*1.6, 0.0, 1.0);
  float floor_ = mix(0.30, 0.10, u_close);                // close up the shading gets deeper: form, not a flat wall
  vec3 col = base*mix(1.0, floor_+(1.08-floor_)*ndl, u_light);
  col *= mix(1.0, 0.50+0.50*d, u_close*0.8);             // far parts of the relief fall away
  float spec = pow(max(dot(n, normalize(L+V)), 0.0), 36.0)*bump;
  float lum = dot(base, vec3(0.299, 0.587, 0.114));
  col += mix(vec3(1.0, 0.95, 0.85), base, 0.45)*spec*u_light*(0.35+0.9*lum)*0.9*smoothstep(0.0, 0.12, lum);   // black paint stays black
  float rim = pow(1.0-max(dot(n, V), 0.0), 2.2);
  // ---- luminous edges: ridges of the relief and slice outlines glow in the painting's own colour
  vec3 hot = base*base*1.6+base*0.6;
  float gl = u_glow*(1.0-0.8*u_close);
  col += hot*(rim*1.6+bump*bump*0.55+sedge*1.4)*gl;
  col += hot*gl*0.22*smoothstep(0.35, 1.0, lum);
  // ---- optional TAS light chase: palette bands sweeping along the depth contours
  if (u_chase > 0.001) {
    float ph = fract(d*3.0+0.12*g.w-u_chph);
    float w = 1.0-smoothstep(0.0, u_chw, abs(ph-0.5)*2.0);
    w *= u_chase;
    col = col*(1.0+0.8*w)+palette(d*0.6+u_chph*0.37)*w*(0.10+0.90*smoothstep(0.0, 0.5, lum))*1.1;
  }
  col = max(u_colmat*col, 0.0)*shadow;
  if (u_close > 0.0) {                                    // camera headlamp: the light pools mid-frame and falls off
    vec2 sp = (gl_FragCoord.xy/u_res-0.5)*vec2(u_res.x/u_res.y, 1.0)*2.0;
    col *= (1.0-0.22*u_close)*(1.0-u_close*0.80*smoothstep(0.0, 1.75, length(sp)));
  }
  col = min(col, 0.86)+0.14*tanh(max(col-0.86, 0.0)/0.14);     // soft shoulder: highlights keep their shape
  o_col = vec4(col*alpha, alpha);
  o_aux = vec4(d*alpha, 0.0, 0.0, alpha);
}
"""

QUAD_VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

BLIT_FS = """
#version 330
in vec2 v_uv; uniform sampler2D u_scene; out vec4 o;
void main(){ o = clamp(texture(u_scene, v_uv), 0.0, 1.0); }
"""

POST_FS = """
#version 330
in vec2 v_uv;
uniform sampler2D u_scene, u_aux, u_hist;
uniform float u_smear, u_valid, u_head, u_age;
uniform vec2 u_tiles, u_half;
out vec4 o;
vec4 hist(float j, vec2 uv){
  float idx = mod(u_head-j+%(N)d.0, %(N)d.0);
  vec2 cell = vec2(mod(idx, u_tiles.x), floor(idx/u_tiles.x));
  return texture(u_hist, (cell+clamp(uv, u_half, 1.0-u_half))/u_tiles);
}
void main(){
  vec4 cur = texture(u_scene, v_uv);
  if (u_smear > 0.001 && u_valid > 1.5) {
    // slit-scan: far parts of the relief and the top of the frame lag behind in time
    vec2 px = 6.0/vec2(textureSize(u_aux, 0));
    float dep = 0.25*(texture(u_aux, v_uv+px).r+texture(u_aux, v_uv-px).r
                      +texture(u_aux, v_uv+px*vec2(1.0, -1.0)).r+texture(u_aux, v_uv+px*vec2(-1.0, 1.0)).r);
    float gsel = clamp(0.35*(1.0-dep)+0.65*v_uv.y, 0.0, 1.0);
    float dly = u_smear*(u_valid-1.0)*gsel;              // in ring frames behind "now"
    float j = dly-u_age;
    if (j <= 0.0) cur = mix(cur, hist(0.0, v_uv), clamp(dly/max(u_age, 1e-3), 0.0, 1.0));
    else {
      j = min(j, u_valid-1.0);
      float j0 = floor(j);
      cur = mix(hist(j0, v_uv), hist(min(j0+1.0, u_valid-1.0), v_uv), j-j0);
    }
  }
  o = clamp(cur, 0.0, 1.0);
}
""" % dict(N=HIST_N)


# ----------------------------------------------------------------------------------------------------------
# mip level storage without glGenerateMipmap
# ----------------------------------------------------------------------------------------------------------
_GL11 = None


def _gl11():
    """The GL 1.1 entry points of opengl32.dll (Windows), or False. They forward to the driver of the
    current context, which is all we need to declare empty mip levels."""
    global _GL11
    if _GL11 is None:
        try:
            import ctypes
            gl = ctypes.windll.opengl32
            gl.glGetIntegerv.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
            gl.glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]
            gl.glTexParameteri.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_int]
            gl.glTexImage2D.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
            gl.glTexSubImage2D.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
            gl.glPixelStorei.argtypes = [ctypes.c_uint, ctypes.c_int]
            gl.glGetError.restype = ctypes.c_uint
            for fn in (gl.glGetIntegerv, gl.glBindTexture, gl.glTexParameteri, gl.glTexImage2D, gl.glTexSubImage2D,
                       gl.glPixelStorei):
                fn.restype = None
            _GL11 = (gl, ctypes)
        except Exception:
            _GL11 = False
    return _GL11


def alloc_mip_levels(tex, half_float):
    """Give `tex` (RGBA8, or RGBA16F when half_float) empty storage for every mip level, so the levels can
    be filled with tex.write(level=n) a stripe at a time.

    moderngl can only do this through build_mipmaps(), i.e. glGenerateMipmap, which costs 4-6 ms on a
    1536 x 1536 texture even when it is empty. On Windows we declare the levels directly (well under 1 ms)
    and put the texture binding back; elsewhere (or if that fails) we fall back to build_mipmaps().
    Returns "direct" or "generate".
    """
    g = _gl11()
    if g:
        gl, ctypes = g
        TEX2D, BINDING, MAX_LEVEL = 0x0DE1, 0x8069, 0x813D
        ifmt, typ = (0x881A, 0x140B) if half_float else (0x8058, 0x1401)       # RGBA16F/HALF_FLOAT, RGBA8/UBYTE
        prev = ctypes.c_int(0)
        try:
            while gl.glGetError():
                pass
            gl.glGetIntegerv(BINDING, ctypes.byref(prev))
            gl.glBindTexture(TEX2D, tex.glo)
            w, h = tex.size
            lvl = 0
            while max(w, h) > 1:
                w, h, lvl = max(1, w // 2), max(1, h // 2), lvl + 1
                gl.glTexImage2D(TEX2D, lvl, ifmt, w, h, 0, 0x1908, typ, None)
            gl.glTexParameteri(TEX2D, MAX_LEVEL, lvl)
            ok = gl.glGetError() == 0
            gl.glBindTexture(TEX2D, max(prev.value, 0))
            if ok:
                return "direct"
        except Exception:
            pass
    tex.build_mipmaps()
    return "generate"


def write_rows(tex, level, arr, y0, half_float, direct):
    """Upload the rows `arr` (n, w, 4) into mip `level` of `tex` starting at row y0, straight from the numpy
    buffer. `direct` levels (see alloc_mip_levels) are unknown to moderngl, so they go through
    glTexSubImage2D; otherwise tex.write does the same thing."""
    n, w = arr.shape[:2]
    if not direct:
        tex.write(arr, viewport=(0, y0, w, n), level=level)
        return
    gl, ctypes = _gl11()
    prev, align = ctypes.c_int(0), ctypes.c_int(4)
    gl.glGetIntegerv(0x8069, ctypes.byref(prev))                  # TEXTURE_BINDING_2D
    gl.glGetIntegerv(0x0CF5, ctypes.byref(align))                 # UNPACK_ALIGNMENT
    gl.glBindTexture(0x0DE1, tex.glo)
    gl.glPixelStorei(0x0CF5, 1)
    gl.glTexSubImage2D(0x0DE1, level, 0, y0, w, n, 0x1908, 0x140B if half_float else 0x1401,
                       ctypes.c_void_p(arr.ctypes.data))
    gl.glPixelStorei(0x0CF5, align.value)
    gl.glBindTexture(0x0DE1, max(prev.value, 0))


# ----------------------------------------------------------------------------------------------------------
# small maths
# ----------------------------------------------------------------------------------------------------------
def _perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    m = np.zeros((4, 4), np.float64)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye, target, up):
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-9)
    s = np.cross(f, up)
    s /= max(np.linalg.norm(s), 1e-9)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def palette_stops(x):
    """Fractional palette value -> 5 blended RGB stops (glides between neighbours, wraps)."""
    x = float(x) if math.isfinite(float(x)) else 0.0
    n = len(PALETTES)
    i = int(math.floor(x)) % n
    f = x - math.floor(x)
    a, b = np.array(PALETTES[i], np.float32), np.array(PALETTES[(i + 1) % n], np.float32)
    return a * (1 - f) + b * f


def color_matrix(hue, sat):
    """Gentle hue rotation (half a turn at hue = 1) and saturation around the art's own colours."""
    a = float(hue) * math.pi
    s = 1.0 + (float(sat) - 1.0) * 0.6
    c, sn = math.cos(a), math.sin(a)
    w = np.array([0.299, 0.587, 0.114])
    rot = (c * np.eye(3) + (1 - c) * np.full((3, 3), 1 / 3.0)
           + sn / math.sqrt(3.0) * np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]))
    satm = s * np.eye(3) + (1 - s) * np.tile(w, (3, 1))
    return satm @ rot


def _f(values, name):
    try:
        v = float(values.get(name, DEFAULTS[name]))
    except (TypeError, ValueError):
        return DEFAULTS[name]
    return v if math.isfinite(v) else DEFAULTS[name]


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


DISPLACERS = ("art_melt", "art_liquify", "art_slices", "art_smear")   # they move pixels: their sum is what turns to mush
REMAPS = ("art_fold", "art_droste")                                   # structural: they re-map the picture, kept as set


def limit_distortion(values):
    """Internal limiter for the art distortions -> {name: effective amount 0..1} for the six of them.

    Any one displacer at full, or two at 0.75, plays exactly as set. Beyond a combined budget of 1.5 (less when
    fold / Droste are up: 0.8 with both at full) the displacers are scaled down together, keeping their
    proportions, so the piece stays legible however many faders are pushed. Continuous: nothing jumps."""
    out = {k: _clamp(_f(values, k), 0.0, 1.0) for k in DISPLACERS + REMAPS}
    total = sum(out[k] for k in DISPLACERS)
    budget = 1.5 - 0.35 * (out["art_fold"] + out["art_droste"])
    if total > budget:
        for k in DISPLACERS:
            out[k] *= budget / total
    return out


# ----------------------------------------------------------------------------------------------------------
# the pass
# ----------------------------------------------------------------------------------------------------------
class _Piece:
    __slots__ = ("color", "geo", "aspect", "has_depth")

    def __init__(self, color, geo, aspect, has_depth):
        self.color, self.geo, self.aspect, self.has_depth = color, geo, aspect, has_depth

    def release(self):
        for tx in (self.color, self.geo):
            try:
                tx.release()
            except Exception:
                pass


class ArtPass:
    def __init__(self, ctx, out_size=(1280, 800), folder=Path("art")):
        self.ctx = ctx
        self.size = (max(16, min(int(out_size[0]), 1280)), max(16, min(int(out_size[1]), 800)))
        self.folder = Path(folder)
        self.settle_s = SETTLE_S
        self.rescan_s = RESCAN_S
        self.stats = dict(gl_allocs=0, uploads=0, evictions=0, decode_fail=0, gl_fail=0, ms=0.0, upload_ms=0.0,
                          upload_ms_max=0.0, decoder="process")
        w, h = self.size

        # ---- programs + static geometry
        self._mesh_prog = self._alloc(ctx.program, vertex_shader=MESH_VS, fragment_shader=MESH_FS)
        self._post_prog = self._alloc(ctx.program, vertex_shader=QUAD_VS, fragment_shader=POST_FS)
        self._blit_prog = self._alloc(ctx.program, vertex_shader=QUAD_VS, fragment_shader=BLIT_FS)
        self._quad = self._alloc(ctx.buffer, np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._post_vao = self._alloc(ctx.vertex_array, self._post_prog, [(self._quad, "2f", "in_pos")])
        self._blit_vao = self._alloc(ctx.vertex_array, self._blit_prog, [(self._quad, "2f", "in_pos")])
        self._meshes = []                                     # [landscape, portrait]: (vbo, ibo, vao)
        for nx, ny in ((GRID_LONG, GRID_SHORT), (GRID_SHORT, GRID_LONG)):
            u, v = np.meshgrid(np.linspace(0, 1, nx + 1, dtype="f4"), np.linspace(0, 1, ny + 1, dtype="f4"))
            vbo = self._alloc(ctx.buffer, np.stack([u, v], axis=2).astype("f4").tobytes())
            i = (np.arange(ny)[:, None] * (nx + 1) + np.arange(nx)[None, :]).astype("u4")
            tri = np.stack([i, i + 1, i + nx + 1, i + 1, i + nx + 2, i + nx + 1], axis=2)
            ibo = self._alloc(ctx.buffer, tri.astype("u4").tobytes())
            vao = self._alloc(ctx.vertex_array, self._mesh_prog, [(vbo, "2f", "in_uv")], index_buffer=ibo)
            self._meshes.append((vbo, ibo, vao))

        # ---- render targets (all <= out_size)
        self._scene = self._alloc(ctx.texture, (w, h), 4, dtype="f2")
        self._aux = self._alloc(ctx.texture, (w, h), 2, dtype="f1")
        self._depth_rb = self._alloc(ctx.depth_renderbuffer, (w, h))
        self._scene_fbo = self._alloc(ctx.framebuffer, [self._scene, self._aux], self._depth_rb)
        self._out = self._alloc(ctx.texture, (w, h), 4, dtype="f1")
        self._out_fbo = self._alloc(ctx.framebuffer, [self._out])
        self._tile = (max(8, w // 2), max(8, h // 2))        # 32 x 640x400 RGBA8 = 33 MB at 1280x800
        self._hist = self._alloc(ctx.texture, (self._tile[0] * HIST_COLS, self._tile[1] * HIST_ROWS), 4, dtype="f1")
        self._hist_fbo = self._alloc(ctx.framebuffer, [self._hist])
        self._black = self._alloc(ctx.texture, (1, 1), 4, bytes(4), dtype="f1")
        self._flat = self._alloc(ctx.texture, (1, 1), 4, np.array([0.5, 0, 0, 0], "f2").tobytes(), dtype="f2")
        for tx in (self._scene, self._aux, self._out, self._hist, self._black, self._flat):
            tx.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tx.repeat_x = tx.repeat_y = False
        self._guard(lambda: (self._out_fbo.clear(0, 0, 0, 0), self._hist_fbo.clear(0, 0, 0, 0)))

        # ---- constant uniforms
        mp, pp = self._mesh_prog, self._post_prog
        for i, n in enumerate(("u_colA", "u_geoA", "u_colB", "u_geoB")):
            self._set(mp, n, UNIT0 + i)
        self._set(pp, "u_scene", UNIT0 + 4); self._set(pp, "u_aux", UNIT0 + 5); self._set(pp, "u_hist", UNIT0 + 6)
        self._set(self._blit_prog, "u_scene", UNIT0 + 4)
        self._set(pp, "u_tiles", (float(HIST_COLS), float(HIST_ROWS)))
        self._set(pp, "u_half", (0.5 / self._tile[0], 0.5 / self._tile[1]))
        self._set(mp, "u_feather", 0.14)
        self._set(mp, "u_res", (float(w), float(h)))

        # ---- animation state
        self._cam = None                 # smoothed yaw, pitch, roll, dolly
        self._flow_t = 0.0
        self._fold_rot = 0.0
        self._dzoom = 0.0
        self._chase_ph = 0.0
        self._last_beat = None
        self._last_travel = None
        self._pulse = 0.0
        self._hist_head, self._hist_valid, self._hist_acc = 0, 0, 0.0
        self._last_t = None

        # ---- library / loader state
        self._cache = OrderedDict()      # key=(name, sig) -> _Piece, LRU
        self._a = None                   # key on show
        self._b = None                   # key fading in (None may also be a target: fade to black)
        self._fading = False
        self._mixv = 0.0
        self._staging = None             # dict: a decoded piece on its way onto the GPU, a few stripes per frame
        self._failed = {}                # name -> sig that would not decode
        self._retry_at = 0.0             # no GPU upload attempts before this (monotonic) after a GL failure
        self._idx_seen = None            # rounded art_index the current name was resolved from
        self._name = None                # the piece we hold on to, by NAME
        self._n_seen, self._wrapped = 0, False
        self._busy_since = 0.0
        self._decoder = _Decoder()
        _gl11()                          # resolve the GL 1.1 entry points now, not in the middle of an upload
        self._lock = threading.Lock()
        self._lib = ([], {})
        self._want = None                # key the worker should decode
        self._result = None              # (key, decoded dict | None)
        self._busy = None
        self._stop = False
        self._wake = threading.Event()
        self._scan()
        self._thread = threading.Thread(target=self._worker, name="art-loader", daemon=True)
        self._thread.start()

    # ---- helpers -------------------------------------------------------------------------------------
    def _alloc(self, fn, *a, **k):
        self.stats["gl_allocs"] += 1
        return fn(*a, **k)

    @staticmethod
    def _set(prog, name, value):
        if name in prog:
            prog[name].value = value

    def _guard(self, fn):
        """Run GL work on our own framebuffers, then put the caller's framebuffer + viewport back."""
        ctx = self.ctx
        prev = ctx.fbo
        try:
            vp = prev.viewport
        except Exception:                # the caller's framebuffer was released under us: fall back to the screen
            prev = ctx.screen
            vp = prev.viewport
        try:
            return fn()
        finally:
            ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            prev.use()
            prev.viewport = vp

    # ---- library + worker -----------------------------------------------------------------------------
    def _scan(self):
        lib = scan_library(self.folder)
        with self._lock:
            self._lib = lib

    def _worker(self):
        last_scan = time.monotonic()
        try:
            self._decoder.start()                              # warm: the first piece does not pay for the spawn
        except Exception:
            pass
        while not self._stop:
            self._wake.wait(0.1)
            self._wake.clear()
            if self._stop:
                break
            now = time.monotonic()
            if now - last_scan >= self.rescan_s:
                last_scan = now
                try:
                    self._scan()
                except Exception:
                    pass
            with self._lock:
                key = self._want
                if key is None or self._result is not None or key == self._busy:
                    continue
                self._busy = key
                self._busy_since = time.monotonic()
            data, bad = None, False
            try:
                name = key[0]
                data = self._decoder.decode(self.folder / name, depth_path(self.folder, name), MAX_SOURCE_PIXELS)
            except DecodeError:
                bad = True
            except Exception:
                data = None                                    # shutting down / decoder gone: not the file's fault
            self.stats["decoder"] = self._decoder.mode
            with self._lock:
                if data is not None or bad:
                    self._result = (key, data)
                self._busy = None

    def source(self, values, dt):
        """Keep the library / uploads going without drawing the relief -> (colour texture, aspect) of the piece
        on show, or None while there is none. For the compositor's other ways of using the image (art_use)."""
        try:
            dt = _clamp(float(dt) if math.isfinite(float(dt)) else 0.0, 0.0, 0.25)
            self._update_library(values, dt)
            key = self._b if (self._fading and self._b and self._mixv > 0.5) else self._a
            piece = self._cache.get(key) if key else None
            if piece is None and self._a:
                piece = self._cache.get(self._a)
            self._hist_valid = 0                                   # the relief's slit-scan history is stale when it comes back
            return (piece.color, float(piece.aspect)) if piece is not None else None
        except Exception as ex:
            self.stats["error"] = repr(ex)
            return None

    def library(self):
        """Names currently in the library, in art_index order."""
        with self._lock:
            return list(self._lib[0])

    def status(self):
        """For a panel: what is showing / loading."""
        return dict(count=len(self.library()), showing=self._a[0] if self._a else None,
                    incoming=self._b[0] if (self._fading and self._b) else None,
                    cached=[k[0] for k in self._cache], depth=bool(self._a and self._cache.get(self._a)
                                                                   and self._cache[self._a].has_depth))

    def _target(self, values):
        """-> ("show", key) | ("black", None) | ("keep", None)."""
        with self._lock:
            names, info = self._lib
        idx = int(round(_f(values, "art_index")))
        if (idx != self._idx_seen or self._name not in info
                or (self._wrapped and len(names) != self._n_seen)):
            # art_index -> name only when the VALUE moves (or our piece is gone): the list may grow, shrink or
            # reorder underneath without swapping what is on screen. One exception: an index that pointed past
            # the end (the panel selected a fresh upload before our rescan saw the file) was only a wrapped
            # guess, so it is resolved again when the list changes size.
            self._idx_seen, self._n_seen = idx, len(names)
            self._wrapped = not (0 <= idx < len(names))
            self._name = names[idx % len(names)] if names else None
        name = self._name
        if name is None:
            return "black", None
        sig, newest = info[name]
        if self._failed.get(name) == sig:
            return "black", None
        if sig[0][1] <= 0:
            return "black", None                               # empty file
        if time.time() - newest < self.settle_s:
            # still being written (or its depth map just landed): hold what is on screen until it settles,
            # but an older, already-uploaded version of the same piece may stay up
            return "keep", None
        return "show", (name, sig)

    def _evict(self, keep):
        while len(self._cache) >= CACHE_MAX:
            victim = next((k for k in self._cache if k not in keep), None)
            if victim is None:
                return False
            self._cache.pop(victim).release()
            self.stats["evictions"] += 1
        return True

    def _upload_step(self):
        """Move a decoded piece onto the GPU a little per frame: one texture allocation, or row stripes
        until UPLOAD_BUDGET_MS is spent. Mip levels come ready-made from the decoder, so there is no
        glGenerateMipmap hitch on real data either."""
        t0 = time.perf_counter()
        try:
            self._upload_work(t0)
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.stats["upload_ms"] = ms
            self.stats["upload_ms_max"] = max(self.stats["upload_ms_max"], ms)

    def _upload_work(self, t0):
        if self._busy is not None and time.monotonic() - self._busy_since > DECODE_TIMEOUT_S:
            self._decoder.kill(timed_out=True)                  # stuck on a pathological file
        if self._staging is None:
            with self._lock:
                res, self._result = self._result, None
                if res is not None:
                    self._want = None                           # the worker must not start the same job again
            if res is None:
                return
            key, data = res
            if data is None:
                self._failed[key[0]] = key[1]
                self.stats["decode_fail"] += 1
                return
            if key in self._cache or key != self._wanted_key:
                return                                          # stale: the performer moved on
            levels = [[data["color"]] + list(data["color_mips"]), [data["geo"]] + list(data["geo_mips"])]
            plan = [(which, lvl, arr) for which in (0, 1) for lvl, arr in enumerate(levels[which])]
            self._staging = dict(key=key, data=data, tex=[None, None], direct=[False, False], plan=plan, pos=0,
                                 row=0, tries=0)
            return
        st = self._staging
        if st["key"] != self._wanted_key and st["tex"][0] is None:
            self._staging = None                                # moved on before anything was allocated
            return
        if time.monotonic() < self._retry_at:
            return
        try:
            plan = st["plan"]
            while st["pos"] < len(plan):
                which, lvl, arr = plan[st["pos"]]
                tex = st["tex"][which]
                if tex is None:
                    if which == 0 and not self._evict({self._a, self._b if self._fading else self._a}):
                        return
                    h, w = arr.shape[:2]
                    tex = self._alloc(self.ctx.texture, (w, h), 4, dtype="f1" if which == 0 else "f2")
                    st["tex"][which] = tex
                    st["direct"][which] = alloc_mip_levels(tex, which == 1) == "direct"
                    self.stats["mip_alloc"] = "direct" if st["direct"][which] else "generate"
                    return                                      # an allocation is this frame's whole share
                h, w = arr.shape[:2]
                rows = max(1, STRIPE_BYTES // max(1, w * arr.strides[1]))
                y0 = st["row"]
                y1 = min(h, y0 + rows)
                write_rows(tex, lvl, arr[y0:y1], y0, which == 1, st["direct"][which])
                if y1 >= h:
                    st["pos"] += 1
                    st["row"] = 0
                else:
                    st["row"] = y1
                if (time.perf_counter() - t0) * 1000.0 >= UPLOAD_BUDGET_MS:
                    return
            col, geo = st["tex"]
            for tex in (col, geo):
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                tex.repeat_x = tex.repeat_y = False
            try:
                col.anisotropy = 4.0
            except Exception:
                pass
            self._cache[st["key"]] = _Piece(col, geo, st["data"]["aspect"], st["data"]["has_depth"])
            self.stats["uploads"] += 1
            self._staging = None
        except Exception as ex:
            # a GL failure says nothing about the file: free BOTH textures, wait a moment, try again
            for i, tex in enumerate(st["tex"]):
                if tex is not None:
                    try:
                        tex.release()
                    except Exception:
                        pass
                    st["tex"][i] = None
            st["pos"], st["row"] = 0, 0
            st["tries"] += 1
            self.stats["gl_fail"] += 1
            self.stats["gl_error"] = repr(ex)
            self._retry_at = time.monotonic() + GL_RETRY_S * min(st["tries"], 8)

    _wanted_key = None

    def _update_library(self, values, dt):
        kind, key = self._target(values)
        if kind == "keep":
            key = self._b if self._fading else self._a
            kind = "show" if key is not None else "black"
        self._wanted_key = key if kind == "show" else None
        self._upload_step()
        staged = self._staging["key"] if self._staging else None
        need = (self._wanted_key is not None and self._wanted_key not in self._cache and self._wanted_key != staged
                and self._failed.get(self._wanted_key[0]) != self._wanted_key[1])
        with self._lock:
            self._want = self._wanted_key if need else None
        if need:
            self._wake.set()

        showing = self._b if self._fading else self._a
        ready = kind == "black" or key in self._cache
        if not self._fading and ready and key != self._a:
            self._b, self._fading, self._mixv = key, True, 0.0
        if self._fading:
            hurry = 4.0 if (ready and key != showing) else 1.0      # scrubbing through pieces: do not queue up
            self._mixv += dt / CROSSFADE_S * hurry
            if self._mixv >= 1.0:
                self._a, self._b, self._fading, self._mixv = self._b, None, False, 0.0
        for k in (self._a, self._b):
            if k in self._cache:
                self._cache.move_to_end(k)

    # ---- frame ----------------------------------------------------------------------------------------
    def render(self, values, clock, t, travel_z, dt):
        t0 = time.perf_counter()
        try:
            t = float(t) if math.isfinite(float(t)) else 0.0
            dt = float(dt) if math.isfinite(float(dt)) else 0.0
            dt = _clamp(dt, 0.0, 0.25)
            travel_z = float(travel_z) if math.isfinite(float(travel_z)) else 0.0
            self._update_library(values, dt)
            self._guard(lambda: self._draw(values, clock or {}, t, travel_z, dt))
        except Exception as ex:                                   # a show must never die on a bad frame
            self.stats["error"] = repr(ex)
            try:
                self._guard(lambda: self._out_fbo.clear(0, 0, 0, 0))
            except Exception:
                pass
        self.stats["ms"] = (time.perf_counter() - t0) * 1000.0
        return self._out

    def _draw(self, values, clock, t, travel_z, dt):
        ctx = self.ctx
        pa = self._cache.get(self._a) if self._a else None
        pb = self._cache.get(self._b) if (self._fading and self._b) else None
        if self._last_t is not None and abs(t - self._last_t) > 0.5:
            self._hist_valid = 0                                 # time jumped (layer was off): old ring frames are stale
        self._last_t = t
        if pa is None and pb is None:
            self._hist_valid = 0
            self._out_fbo.use()
            self._out_fbo.clear(0, 0, 0, 0)
            return
        mixv = self._mixv if self._fading else 0.0
        mixv = mixv * mixv * (3 - 2 * mixv)
        asp_a = pa.aspect if pa else (pb.aspect if pb else 1.0)
        asp_b = pb.aspect if pb else asp_a
        asp = asp_a + (asp_b - asp_a) * mixv
        w, h = self.size
        out_asp = w / h
        frozen = _f(values, "freeze") > 0.5
        adt = 0.0 if frozen else dt

        # ---- clock-driven phases (integrated on the CPU so nothing jumps when a fader moves)
        def fin(x, default):
            try:
                x = float(x)
            except (TypeError, ValueError):
                return default
            return x if math.isfinite(x) else default
        for name in ("_pulse", "_flow_t", "_fold_rot", "_dzoom", "_chase_ph", "_hist_acc"):
            if not math.isfinite(getattr(self, name)):           # never let one bad frame live on in the state
                setattr(self, name, 0.0)
        if self._cam is not None and not np.isfinite(self._cam).all():
            self._cam = None
        if self._last_beat is not None and not math.isfinite(self._last_beat):
            self._last_beat = None
        beat = fin(clock.get("beat"), self._last_beat if self._last_beat is not None else 0.0)
        bphase = fin(clock.get("beat_phase"), beat % 1.0)
        dbeat = 0.0 if self._last_beat is None else beat - self._last_beat
        self._last_beat = beat
        if not (0.0 <= dbeat < 2.0):
            dbeat = 0.0
        self._chase_ph = (self._chase_ph + dbeat / 4.0 * _f(values, "chase_rate")) % 64.0
        kick = math.exp(-4.0 * (bphase % 1.0))
        self._pulse += (kick - self._pulse) * (1.0 - math.exp(-adt / 0.035)) if adt > 0 else 0.0
        lim = limit_distortion(values)                             # many distortions at once share one budget
        melt, liq = lim["art_melt"], lim["art_liquify"]
        self._flow_t = (self._flow_t + adt * (0.10 + 0.10 * melt + 0.08 * liq)) % 1024.0
        self._fold_rot = (self._fold_rot + adt * 0.12 * _f(values, "rotate")) % math.tau
        dtravel = 0.0 if self._last_travel is None else travel_z - self._last_travel
        self._last_travel = travel_z
        if abs(dtravel) > 2.0:
            dtravel = 0.0
        droste = _clamp(_f(values, "art_droste"), 0, 1)
        dls = 2.3 - 1.3 * droste                                   # ln(scale ratio): 10x -> 2.7x per turn
        self._dzoom = (self._dzoom + dtravel * 0.55 + adt * 0.045 * droste) % (dls * 64.0)

        # ---- camera: orbit around the piece, smoothed, with handheld drift and travel push
        tgt = np.array([_f(values, "cam_yaw"), _f(values, "cam_pitch"), _f(values, "cam_roll"),
                        _f(values, "cam_dolly") + 0.5 * _f(values, "zoom")])
        tgt = np.clip(tgt, -1.5, 1.5)
        if self._cam is None:
            self._cam = tgt.copy()
        else:
            self._cam += (tgt - self._cam) * (1.0 - math.exp(-dt / 0.18))
        yaw, pitch, roll, dolly = self._cam
        drift = _clamp(_f(values, "cam_drift"), 0, 1)
        yaw_a = yaw * 0.95 + drift * 0.24 * math.sin(t * 0.23)
        pitch_a = pitch * 0.75 + drift * 0.15 * math.sin(t * 0.31 + 1.0)
        roll_a = roll * math.pi * 0.5 + drift * 0.05 * math.sin(t * 0.17 + 2.0)
        try:                                                       # the engine's barrel roll (param "spin", already an angle)
            spin = float(values.get("spin_phase", 0.0))
            roll_a += spin if math.isfinite(spin) else 0.0
        except (TypeError, ValueError):
            pass
        relief = _clamp(_f(values, "art_relief"), 0, 1)
        breathe = _clamp(_f(values, "art_breathe"), 0, 1)
        slices = lim["art_slices"]
        amp = relief * 0.85 * (1.0 + breathe * (0.55 * self._pulse - 0.12 + 0.10 * math.sin(t * 0.8)))
        tanf = math.tan(math.radians(FOV_DEG) * 0.5)
        d0 = 1.0 / (tanf * 0.95)
        scale = min(1.0, out_asp * d0 * tanf * 0.95 / max(asp, 1e-3))
        curve = _clamp(_f(values, "tunnel_curve"), 0, 1)
        push = 0.5 - 0.5 * math.cos(travel_z * 0.8)                # travel glides the camera in and back out
        dist = d0 * math.exp(-dolly * 1.0) * (1.0 - 0.58 * push) * (1.0 + drift * 0.05 * math.sin(t * 0.19))
        dist *= 1.0 - 0.15 * _clamp(_f(values, "punch"), 0, 1)
        sep = 1.15
        front = 0.5 * amp + (0.5 * sep * slices if slices > 0.01 else 0.0)
        dist = _clamp(dist, front + 0.35, 14.0)
        look = np.array([curve * 0.30 * push * math.sin(travel_z * 0.37), curve * 0.22 * push * math.sin(travel_z * 0.53), 0.0])
        cy, sy, cp, sp = math.cos(yaw_a), math.sin(yaw_a), math.cos(pitch_a), math.sin(pitch_a)
        eye = look + dist * np.array([sy * cp, sp, cy * cp])
        up = np.array([math.sin(roll_a), math.cos(roll_a), 0.0])
        vp = _perspective(FOV_DEG, out_asp, 0.05, 40.0) @ _look_at(eye, look, up)

        # ---- mesh pass
        mp = self._mesh_prog
        (pa.color if pa else self._black).use(UNIT0)
        (pa.geo if pa else self._flat).use(UNIT0 + 1)
        (pb.color if pb else self._black).use(UNIT0 + 2)
        (pb.geo if pb else self._flat).use(UNIT0 + 3)
        nsl = float(N_SLICES) if slices > 0.01 else 1.0
        fold = _clamp(_f(values, "art_fold"), 0, 1) * _clamp(_f(values, "symmetry") * 2.0, 0, 1)
        glow = _clamp(_f(values, "art_glow"), 0, 1)
        uni = dict(
            u_mix=mixv if self._fading else 0.0, u_wA=1.0 if pa else 0.0, u_wB=1.0 if pb else 0.0, u_asp=asp,
            u_t=t % 4096.0, u_twirl=_clamp(_f(values, "twirl"), -1, 1), u_fold=fold,
            u_folds=float(_clamp(round(_f(values, "folds")), 2, 24)), u_foldrot=self._fold_rot,
            u_droste=droste, u_dls=dls, u_dzoom=self._dzoom % dls, u_scale=scale, u_amp=amp, u_nsl=nsl,
            u_slices=slices, u_sep=sep, u_melt=melt, u_liq=liq, u_swim=_clamp(_f(values, "swim"), 0, 1),
            u_flowt=self._flow_t, u_ripple=breathe * (0.35 + _clamp(_f(values, "ripple"), 0, 1)) * min(1.0, relief * 4.0),
            u_beat=bphase % 1.0, u_lod=1.8, u_glow=glow, u_light=0.9 * min(1.0, relief * 3.0),
            u_lt=(t * 0.37) % (math.tau * 100.0), u_chase=_clamp(_f(values, "chase"), 0, 1),
            u_chw=_clamp(_f(values, "chase_width"), 0.05, 0.95), u_chph=self._chase_ph,
            u_cam=tuple(float(x) for x in eye), u_close=_clamp(1.0 - dist / (0.80 * d0), 0.0, 1.0))
        for k, v in uni.items():
            if k in mp:
                mp[k].value = v
        if "u_vp" in mp:
            mp["u_vp"].write(vp.T.astype("f4").tobytes())
        if "u_pal" in mp:
            mp["u_pal"].write(palette_stops(_f(values, "palette")).astype("f4").tobytes())
        if "u_colmat" in mp:
            mp["u_colmat"].write(color_matrix(_f(values, "hue"), _f(values, "saturation")).T.astype("f4").tobytes())

        self._scene_fbo.use()
        self._scene_fbo.viewport = (0, 0, w, h)
        self._scene_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self._meshes[0 if asp >= 1.0 else 1][2].render(moderngl.TRIANGLES, instances=int(nsl))
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)

        # ---- slit-scan ring: record the art pass at half res on a fixed time base
        self._scene.use(UNIT0 + 4); self._aux.use(UNIT0 + 5); self._hist.use(UNIT0 + 6)
        self._hist_acc += adt
        if self._hist_valid == 0 or self._hist_acc >= HIST_DT:
            self._hist_acc = min(self._hist_acc - HIST_DT, HIST_DT) if self._hist_valid else 0.0
            self._hist_acc = max(self._hist_acc, 0.0)
            self._hist_head = (self._hist_head + 1) % HIST_N
            self._hist_valid = min(self._hist_valid + 1, HIST_N)
            tw, th = self._tile
            self._hist_fbo.use()
            self._hist_fbo.viewport = ((self._hist_head % HIST_COLS) * tw, (self._hist_head // HIST_COLS) * th, tw, th)
            self._blit_vao.render(moderngl.TRIANGLE_STRIP)

        # ---- output
        pp = self._post_prog
        for k, v in dict(u_smear=lim["art_smear"], u_valid=float(self._hist_valid),
                         u_head=float(self._hist_head), u_age=self._hist_acc / HIST_DT).items():
            if k in pp:
                pp[k].value = v
        self._out_fbo.use()
        self._out_fbo.viewport = (0, 0, w, h)
        self._post_vao.render(moderngl.TRIANGLE_STRIP)

    def read(self):
        """Last output frame as uint8 (H, W, 4), top row first. For tests and the gallery (slow: glReadPixels)."""
        w, h = self.size
        raw = self._guard(lambda: self._out_fbo.read(components=4))
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)[::-1]

    def release(self):
        self._stop = True
        self._decoder.close()                                   # kills the decode server: a blocked read returns
        self._wake.set()
        try:
            self._thread.join(timeout=0.3)                      # daemon thread: never hold the show up for it
        except Exception:
            pass
        for p in self._cache.values():
            p.release()
        self._cache.clear()
        for tex in (self._staging["tex"] if self._staging else ()):
            if tex is not None:
                try:
                    tex.release()
                except Exception:
                    pass
        self._staging = None
        objs = [self._scene_fbo, self._out_fbo, self._hist_fbo, self._scene, self._aux, self._depth_rb, self._out,
                self._hist, self._black, self._flat, self._post_vao, self._blit_vao, self._quad]
        for vbo, ibo, vao in self._meshes:
            objs += [vao, vbo, ibo]
        objs += [self._mesh_prog, self._post_prog, self._blit_prog]
        for o in objs:
            try:
                o.release()
            except Exception:
                pass


# ----------------------------------------------------------------------------------------------------------
# look check: procedural "paintings", gallery, benchmark
# ----------------------------------------------------------------------------------------------------------
def _hsv(h, s, v):
    h = (h % 1.0) * 6.0
    i = np.floor(h)
    f = h - i
    p, q, t_ = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.astype(int) % 6
    r = np.choose(i, [v, q, p, p, t_, v])
    g = np.choose(i, [t_, v, v, q, p, p])
    b = np.choose(i, [p, p, t_, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _grid(w, h):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return (x - w / 2) / (h / 2), (h / 2 - y) / (h / 2)


def _ss(a, b, x):
    t_ = np.clip((x - a) / (b - a), 0, 1)
    return t_ * t_ * (3 - 2 * t_)


def paint_cosmic_eye(w=1600, h=1000):
    """An all-seeing eye in a rainbow petal mandala. Returns (rgb uint8, synthetic depth uint16)."""
    x, y = _grid(w, h)
    r, th = np.hypot(x, y), np.arctan2(y, x)
    rng = np.random.default_rng(7)
    col = _hsv(0.72 + 0.08 * r, 0.9, 0.10 + 0.16 * np.exp(-r * 1.2))          # indigo night
    stars = (rng.random((h, w)) > 0.9985).astype(np.float32)
    col += blur(stars, 1.2)[..., None] * 6.0
    depth = 0.10 + 0.10 * np.exp(-r * 1.5)
    for k, (rad, n, hue) in enumerate([(1.25, 16, 0.78), (1.0, 12, 0.55), (0.78, 12, 0.08), (0.58, 8, 0.95)]):
        pet = rad * (0.72 + 0.28 * np.abs(np.cos(n * 0.5 * (th + 0.13 * k)))) - r
        m = _ss(0.0, 0.02, pet)
        line = _ss(0.035, 0.0, np.abs(pet - 0.02))
        shade = 0.55 + 0.45 * _ss(0.0, 0.35, pet)
        pc = _hsv(hue + 0.10 * np.sin(n * th) + 0.25 * pet, 0.85, 0.9 * shade)
        col = col * (1 - m[..., None]) + pc * m[..., None]
        col += line[..., None] * np.array([1.0, 0.85, 0.4]) * 0.8
        depth = np.where(m > 0.5, 0.22 + 0.13 * k + 0.10 * _ss(0, 0.3, pet), depth)
    # almond eye
    lid = 1.0 - (np.abs(y) / (0.36 * np.cos(np.clip(x / 0.74, -1, 1) * np.pi / 2) + 1e-4))
    eye = _ss(0.0, 0.06, lid) * (np.abs(x) < 0.74)
    sclera = _hsv(0.12 + 0 * r, 0.25, 0.98 - 0.35 * np.abs(x))
    col = col * (1 - eye[..., None]) + sclera * eye[..., None]
    col += (_ss(0.08, 0.0, np.abs(lid)) * (np.abs(x) < 0.78))[..., None] * np.array([1.0, 0.75, 0.2])
    iris = _ss(0.31, 0.29, r)
    fib = 0.5 + 0.5 * np.sin(th * 46 + 4 * np.sin(r * 30))
    ic = _hsv(0.48 - 0.9 * r + 0.05 * fib, 0.95, 0.45 + 0.55 * fib * _ss(0.05, 0.3, r))
    col = col * (1 - (iris * eye)[..., None]) + ic * (iris * eye)[..., None]
    pupil = _ss(0.115, 0.10, r)
    col = col * (1 - pupil[..., None]) + pupil[..., None] * np.array([0.02, 0.0, 0.05])
    glint = np.exp(-((x + 0.06) ** 2 + (y - 0.07) ** 2) / 0.0012)
    col += glint[..., None]
    depth = np.where(eye > 0.5, 0.74 + 0.12 * lid.clip(0, 1), depth)
    depth = np.where((iris * eye) > 0.5, 0.86 + 0.14 * np.sqrt(np.clip(1 - (r / 0.31) ** 2, 0, 1)), depth)
    depth = np.where(pupil > 0.5, 0.80, depth)
    depth = blur(depth.astype(np.float32), 5.0)
    return (np.clip(col, 0, 1) * 255).astype(np.uint8), (np.clip(depth, 0, 1) * 65535).astype(np.uint16)


def paint_meditator(w=1040, h=1400):
    """A seated figure of light with chakras, aura and a lotus (portrait format, no depth map)."""
    x, y = _grid(w, h)
    y = y + 0.05
    r, th = np.hypot(x, y - 0.1), np.arctan2(y - 0.1, x)
    col = _hsv(0.70 - 0.22 * (y * 0.5 + 0.5), 0.85, 0.12 + 0.20 * np.exp(-r * 1.6))
    # aura: rainbow shells of light around the body, radiating lines
    for k in range(7):
        rad = 0.42 + 0.085 * k
        shell = np.exp(-((np.hypot(x * 1.25, y - 0.12) - rad) / 0.022) ** 2)
        col += shell[..., None] * _hsv(np.full_like(x, 0.0 + k * 0.125), 0.9, 0.75)
    rays = (0.5 + 0.5 * np.cos(th * 36)) ** 6 * _ss(0.35, 0.6, r) * _ss(1.25, 0.7, r)
    col += rays[..., None] * np.array([1.0, 0.8, 0.35]) * 0.45

    def ell(cx, cy, rx, ry):
        return _ss(1.0, 0.94, np.hypot((x - cx) / rx, (y - cy) / ry))
    body = np.maximum.reduce([ell(0, 0.02, 0.21, 0.33), ell(0, 0.52, 0.125, 0.15), ell(0, -0.33, 0.47, 0.13),
                              ell(-0.27, -0.10, 0.075, 0.26), ell(0.27, -0.10, 0.075, 0.26), ell(0, 0.33, 0.07, 0.08)])
    skin = _hsv(0.56 + 0.10 * y + 0.03 * np.sin(x * 40) * np.sin(y * 40), 0.75, 0.55 + 0.35 * np.exp(-(x / 0.16) ** 2))
    weave = 0.5 + 0.5 * np.sin((x + y) * 70) * np.sin((x - y) * 70)             # energy-body lattice
    skin = skin * (0.75 + 0.25 * weave[..., None])
    col = col * (1 - body[..., None]) + skin * body[..., None]
    for k, cy in enumerate(np.linspace(-0.24, 0.62, 7)):                        # chakras
        dd = np.hypot(x, y - cy)
        c = _hsv(np.full_like(x, k / 7.5), 0.95, 1.0)
        col += (np.exp(-(dd / 0.032) ** 2) * 1.2 + np.exp(-(dd / 0.09) ** 2) * 0.35)[..., None] * c
        col += (np.exp(-((dd - 0.055) / 0.006) ** 2) * (0.5 + 0.5 * np.cos(np.arctan2(y - cy, x) * 8)))[..., None] * c
    halo = np.exp(-((np.hypot(x, y - 0.52) - 0.22) / 0.012) ** 2)
    col += halo[..., None] * np.array([1.0, 0.85, 0.3])
    for k in range(9):                                                          # lotus petals under the figure
        a = (k - 4) * 0.36
        px, py = x - 0.50 * np.sin(a), y + 0.52 - 0.10 * np.cos(a)
        ca, sa = np.cos(a), np.sin(a)
        u, v = px * ca - py * sa, px * sa + py * ca
        pet = _ss(1.0, 0.9, np.hypot(u / 0.085, (v - 0.0) / 0.2)) * (1 - body)
        pc = _hsv(0.88 + 0.05 * np.abs(k - 4) + 0 * x, 0.8, 0.5 + 0.5 * _ss(-0.2, 0.2, v))
        col = col * (1 - pet[..., None]) + pc * pet[..., None]
    return (np.clip(col, 0, 1) * 255).astype(np.uint8)


def paint_lattice(w=1200, h=1200):
    """Fine sacred-geometry beadwork on black + the smooth 'bowl' depth Depth Anything gives abstract art."""
    x, y = _grid(w, h)
    r, th = np.hypot(x, y), np.arctan2(y, x)
    col = np.zeros((h, w, 3), np.float32)
    for k in range(1, 9):                                                       # beaded rings
        rad = 0.11 * k
        n = 6 * k
        ring = np.exp(-((r - rad) / 0.016) ** 2)
        beads = (0.5 + 0.5 * np.cos(th * n + k)) ** 3
        col += (ring * beads)[..., None] * _hsv(0.08 * k + 0.1 * np.cos(th * 3), 0.9, 1.0)
    for k in range(6):                                                          # flower of life arcs
        a = k * np.pi / 3
        dd = np.hypot(x - 0.44 * np.cos(a), y - 0.44 * np.sin(a))
        for rad in (0.44, 0.22):
            col += np.exp(-((dd - rad) / 0.006) ** 2)[..., None] * np.array([1.0, 0.8, 0.3]) * 0.8
    star = np.abs(np.cos(th * 6)) ** 40 * _ss(0.95, 0.2, r)
    col += star[..., None] * np.array([0.2, 0.9, 1.0]) * 0.7
    col *= _ss(0.98, 0.9, r)[..., None]
    bowl = np.clip(0.15 + 0.8 * r ** 1.5, 0, 1)                                 # centre far, edges near
    return (np.clip(col, 0, 1) * 255).astype(np.uint8), (bowl * 65535).astype(np.uint16)


def write_test_art(folder):
    """Procedural test paintings -> folder. Returns the file names in list_art order."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    eye, eye_d = paint_cosmic_eye()
    Image.fromarray(eye).save(folder / "a_cosmic_eye.png")
    Image.fromarray(eye_d).save(folder / "a_cosmic_eye.depth.png")
    Image.fromarray(paint_meditator()).save(folder / "b_meditator.jpg", quality=93)
    lat, lat_d = paint_lattice()
    Image.fromarray(lat).save(folder / "c_lattice.png")
    Image.fromarray(lat_d).save(folder / "c_lattice.depth.png")
    return list_art(folder)


def make_context(size=(1280, 800)):
    """Hidden GLFW window + moderngl context (never visible: a show may be on the projector)."""
    import glfw
    if not glfw.init():
        raise RuntimeError("glfw init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.VISIBLE, False)
    win = glfw.create_window(size[0], size[1], "art_layer (hidden)", None, None)
    if not win:
        raise RuntimeError("could not create a hidden GLFW window")
    glfw.make_context_current(win)
    glfw.swap_interval(0)
    return win, moderngl.create_context()


def wait_for(ap, values, name=None, timeout=20.0):
    """Pump render() until the wanted piece is fully on show (gallery / tests). Returns True when it is."""
    clock = dict(bpm=120.0, beat=0.0, beat_phase=0.0, bar_phase=0.0, phrase_phase=0.0)
    end = time.time() + timeout
    while time.time() < end:
        ap.render(values, clock, 0.0, 0.0, 0.25)
        if ap._a is not None and not ap._fading and (name is None or ap._a[0] == name) and ap._a == ap._wanted_key:
            return True
        time.sleep(0.005)
    return False


def _clock(t, bpm=120.0):
    b = t * bpm / 60.0
    return dict(bpm=bpm, beat=b, beat_phase=b % 1.0, bar_phase=(b / 4) % 1.0, phrase_phase=(b / 32) % 1.0)


def run_shot(ap, values, frames=40, t0=3.0, travel=(0.0, 0.0), cam_path=None, fps=60.0):
    """Simulate `frames` of show time so smoothing / flow / the slit-scan ring develop; return the last frame."""
    vals = dict(values)
    for i in range(frames):
        t = t0 + i / fps
        if cam_path:
            vals.update(cam_path(i / max(frames - 1, 1)))
        tz = travel[0] + (travel[1] - travel[0]) * i / max(frames - 1, 1)
        ap.render(vals, _clock(t), t, tz, 1.0 / fps)
    return ap.read()


def base_values(**over):
    v = dict(DEFAULTS)
    v.update(over)
    return v


def gallery(out_dir=Path("test_frames/art_gallery"), size=(1280, 800), extra_art=None):
    out_dir = Path(out_dir)
    art_dir = out_dir / "art"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = write_test_art(art_dir)
    if extra_art:                                             # e.g. a real Depth Anything pair from the AI worker
        import shutil
        src = Path(extra_art)
        if src.exists():
            shutil.copy(src, art_dir / "d_beadwork.png")
            dsrc = src.with_name(src.stem + DEPTH_SUFFIX)
            if dsrc.exists():
                shutil.copy(dsrc, art_dir / "d_beadwork.depth.png")
        names = list_art(art_dir)
    print("[gallery] library:", names)
    win, ctx = make_context(size)
    ap = ArtPass(ctx, size, art_dir)
    ap.settle_s = 0.0
    EYE, MED, LAT = 0, 1, 2
    BEAD = 3 if len(names) > 3 else LAT
    side = dict(cam_yaw=0.55, cam_pitch=0.25, cam_drift=0.0)

    def sway(u):
        return dict(cam_yaw=0.6 * math.sin(u * 9.0), cam_pitch=0.25 * math.cos(u * 7.0), cam_roll=0.3 * math.sin(u * 6.0))
    shots = [
        ("eye baseline (depth map)", base_values(art_index=EYE), {}),
        ("meditator baseline (pseudo-depth)", base_values(art_index=MED), {}),
        ("lattice baseline (bowl depth+emboss)", base_values(art_index=LAT), {}),
        ("beadwork baseline (Depth Anything)", base_values(art_index=BEAD), {}),
        ("eye relief 1.0 side view", base_values(art_index=EYE, art_relief=1.0, **side), {}),
        ("meditator relief 1.0 side view", base_values(art_index=MED, art_relief=1.0, **side), {}),
        ("lattice relief 1.0 close", base_values(art_index=LAT, art_relief=1.0, cam_dolly=0.6, cam_yaw=-0.4, cam_pitch=-0.3, cam_drift=0), {}),
        ("eye relief 0 (flat, true colour)", base_values(art_index=EYE, art_relief=0.0, art_glow=0.0, cam_drift=0), {}),
        ("eye breathe 1.0 on the kick", base_values(art_index=EYE, art_breathe=1.0, art_relief=0.7, ripple=1.0, **side), dict(frames=31)),
        ("eye melt 0.7", base_values(art_index=EYE, art_melt=0.7), dict(frames=150)),
        ("meditator melt 1.0", base_values(art_index=MED, art_melt=1.0), dict(frames=150)),
        ("eye liquify 0.7", base_values(art_index=EYE, art_liquify=0.7), dict(frames=150)),
        ("meditator liquify 1.0", base_values(art_index=MED, art_liquify=1.0), dict(frames=150)),
        ("eye slices 0.8", base_values(art_index=EYE, art_slices=0.8, cam_yaw=0.6, cam_pitch=0.3, cam_drift=0), {}),
        ("meditator slices 1.0", base_values(art_index=MED, art_slices=1.0, cam_yaw=-0.6, cam_pitch=0.2, cam_drift=0), {}),
        ("eye fold 1.0 (8)", base_values(art_index=EYE, art_fold=1.0, folds=8), {}),
        ("meditator fold 1.0 (6)", base_values(art_index=MED, art_fold=1.0, folds=6), {}),
        ("eye fold 0.5 (mid-fold)", base_values(art_index=EYE, art_fold=0.5, folds=8), {}),
        ("eye droste 0.8", base_values(art_index=EYE, art_droste=0.8), {}),
        ("meditator droste 1.0", base_values(art_index=MED, art_droste=1.0), {}),
        ("eye droste 0.15 (fading in)", base_values(art_index=EYE, art_droste=0.15), {}),
        ("eye smear 0.9 (camera swinging)", base_values(art_index=EYE, art_smear=0.9, cam_drift=0), dict(frames=90, cam_path=sway)),
        ("meditator smear 0.9 + travel", base_values(art_index=MED, art_smear=0.9), dict(frames=90, travel=(0.0, 2.5), cam_path=sway)),
        ("eye glow 1.0", base_values(art_index=EYE, art_glow=1.0, art_relief=0.7, **side), {}),
        ("lattice chase 1.0", base_values(art_index=LAT, chase=1.0, chase_width=0.3), dict(frames=50)),
        ("eye chase 1.0 palette 2", base_values(art_index=EYE, chase=1.0, palette=2.0), dict(frames=50)),
        ("eye twirl 0.6 + swim 1", base_values(art_index=EYE, twirl=0.6, swim=1.0), {}),
        ("eye hue 0.5 sat 1.5", base_values(art_index=EYE, hue=0.5, saturation=1.5), {}),
        ("COMBO eye slices+smear+droste", base_values(art_index=EYE, art_slices=0.7, art_smear=0.6, art_droste=0.7, cam_drift=0), dict(frames=90, cam_path=sway)),
        ("COMBO meditator melt+fold", base_values(art_index=MED, art_melt=0.6, art_fold=1.0, folds=6), dict(frames=150)),
        ("COMBO eye liquify+droste+glow", base_values(art_index=EYE, art_liquify=0.6, art_droste=0.7, art_glow=0.9), dict(frames=150)),
        ("COMBO lattice fold+slices", base_values(art_index=LAT, art_fold=1.0, folds=12, art_slices=0.8, cam_yaw=0.5, cam_pitch=0.3, cam_drift=0), {}),
        ("COMBO beadwork droste+fold+relief", base_values(art_index=BEAD, art_droste=0.6, art_fold=0.8, folds=6, art_relief=0.9, cam_yaw=0.4, cam_drift=0), {}),
        ("COMBO meditator everything 0.5", base_values(art_index=MED, art_melt=0.5, art_liquify=0.5, art_slices=0.5, art_fold=0.5, art_droste=0.5, art_smear=0.5, art_glow=0.6, chase=0.5), dict(frames=120, cam_path=sway)),
    ]
    fly = [dict(cam_yaw=-0.7, cam_pitch=0.35, cam_dolly=-0.3, tz=0.0), dict(cam_yaw=-0.35, cam_pitch=0.2, cam_dolly=0.2, tz=0.8),
           dict(cam_yaw=0.0, cam_pitch=-0.1, cam_dolly=0.6, tz=1.8), dict(cam_yaw=0.4, cam_pitch=-0.45, cam_dolly=0.9, tz=2.9),
           dict(cam_yaw=0.8, cam_pitch=-0.2, cam_dolly=0.5, cam_roll=0.25, tz=3.9), dict(cam_yaw=0.3, cam_pitch=0.5, cam_dolly=-0.1, cam_roll=-0.2, tz=5.2)]
    for i, f in enumerate(fly):
        f = dict(f)
        tz = f.pop("tz")
        piece = EYE if i % 2 == 0 else MED
        shots.append((f"FLY {i + 1}/6", base_values(art_index=piece, art_relief=0.8, cam_drift=0.2, **f), dict(travel=(tz, tz))))

    results, cur_piece = [], None
    for label, vals, opt in shots:
        piece = int(vals["art_index"])
        if piece != cur_piece:
            ok = wait_for(ap, vals, names[piece % len(names)])
            cur_piece = piece
            if not ok:
                print(f"[gallery] piece {piece} did not load")
        ap._cam = None                                           # each shot starts settled on its own camera
        img = run_shot(ap, vals, **{"frames": 40, **opt})
        rgb = img[..., :3]
        fn = out_dir / ("%02d_" % len(results) + "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") + ".png")
        Image.fromarray(rgb).save(fn)
        corner = max(rgb[:12, :12].max(), rgb[:12, -12:].max(), rgb[-12:, :12].max(), rgb[-12:, -12:].max())
        results.append((label, rgb))
        print(f"[gallery] {label:42s} mean {rgb.mean():6.1f}  black {100 * (rgb.max(axis=2) < 8).mean():5.1f}%  corner max {corner}")
    # crossfade mid-morph
    vals = base_values(art_index=EYE)
    wait_for(ap, vals, names[EYE])
    vals["art_index"] = MED
    wait_for(ap, vals, names[MED])           # make sure both are cached
    vals["art_index"] = EYE
    wait_for(ap, vals, names[EYE])
    vals["art_index"] = MED
    for i in range(46):
        ap.render(vals, _clock(3 + i / 60), 3 + i / 60, 0.0, 1 / 60)
    rgb = ap.read()[..., :3]
    Image.fromarray(rgb).save(out_dir / ("%02d_crossfade_morph.png" % len(results)))
    results.append(("crossfade eye -> meditator (0.75 s in)", rgb))

    from PIL import ImageDraw
    cols, tw, th = 5, 384, 240
    rows = (len(results) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 16)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for i, (label, rgb) in enumerate(results):
        cx, cy = (i % cols) * tw, (i // cols) * (th + 16)
        sheet.paste(Image.fromarray(rgb).resize((tw, th), Image.LANCZOS), (cx, cy + 16))
        dr.rectangle([cx, cy, cx + tw - 1, cy + 15], fill=(18, 18, 18))
        dr.text((cx + 4, cy + 2), label, fill=(210, 210, 210))
    sheet.save(out_dir / "contact_sheet.png")
    print("[gallery] contact sheet:", (out_dir / "contact_sheet.png").resolve())
    ap.release()
    import glfw
    glfw.destroy_window(win)
    return out_dir / "contact_sheet.png"


def bench(art_dir=Path("test_frames/art_gallery/art"), size=(1280, 800), frames=150):
    """GPU ms per frame (ctx.finish() timed) for the single effects and the heavy combos."""
    if not list_art(art_dir):
        write_test_art(art_dir)
    win, ctx = make_context(size)
    ap = ArtPass(ctx, size, art_dir)
    ap.settle_s = 0.0
    cases = [
        ("relief only", {}),
        ("relief + melt", dict(art_melt=0.7)),
        ("relief + liquify", dict(art_liquify=0.7)),
        ("relief + slices", dict(art_slices=0.8, cam_yaw=0.5)),
        ("relief + fold", dict(art_fold=1.0)),
        ("relief + droste", dict(art_droste=0.8)),
        ("relief + smear", dict(art_smear=0.8)),
        ("relief + chase + glow", dict(chase=1.0, art_glow=1.0)),
        ("close-up dolly 1.0 (max fill)", dict(cam_dolly=1.0)),
        ("HEAVY slices + smear + droste", dict(art_slices=0.8, art_smear=0.8, art_droste=0.8)),
        ("HEAVY everything on", dict(art_slices=0.8, art_smear=0.8, art_droste=0.8, art_melt=0.6, art_liquify=0.6,
                                     art_fold=0.8, chase=1.0, art_glow=1.0, cam_dolly=0.5)),
    ]
    out = []
    for label, over in cases:
        vals = base_values(art_index=0, **over)
        wait_for(ap, vals)
        for i in range(20):
            ap.render(vals, _clock(i / 60), i / 60, 0.0, 1 / 60)
        ctx.finish()
        ts = []
        for i in range(frames):
            t = 1 + i / 60
            a = time.perf_counter()
            ap.render(vals, _clock(t), t, t * 0.3, 1 / 60)
            ctx.finish()
            ts.append((time.perf_counter() - a) * 1000)
        ts = np.array(ts)
        out.append((label, float(np.median(ts)), float(np.percentile(ts, 95))))
        print(f"[bench] {label:34s} median {np.median(ts):5.2f} ms   p95 {np.percentile(ts, 95):5.2f} ms")
    ap.release()
    import glfw
    glfw.destroy_window(win)
    return out


if __name__ == "__main__":
    import argparse
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--gallery", action="store_true", help="render the look-check gallery + contact sheet (headless)")
    ap_.add_argument("--bench", action="store_true", help="GPU ms/frame for single effects and heavy combos")
    ap_.add_argument("--out", default="test_frames/art_gallery")
    ap_.add_argument("--extra-art", default=None, help="extra painting to include (its .depth.png sibling is copied too)")
    ap_.add_argument("--decode-server", action="store_true", help=argparse.SUPPRESS)
    a_ = ap_.parse_args()
    if a_.decode_server:
        _serve_decodes()
        sys.exit(0)
    if a_.gallery:
        gallery(Path(a_.out), extra_art=a_.extra_art)
    if a_.bench:
        bench(Path(a_.out) / "art")
    if not (a_.gallery or a_.bench or a_.decode_server):
        ap_.print_help()
