"""Text layer: words typed in the panel, drawn as crisp signed-distance-field letters and woven into the show.

Three parts, one file:

* ``TextState`` (pure python, no GL): the words, the font, the phrase list and its beat / bar auto-cycle. It lives in
  ``control.Controller`` (``controller.text``), is saved in live_state.json with the other settings and shows up in
  ``controller.state()["text"]``. Panel commands (validated here, registered in control's validator table):

      {"cmd": "text", "text": "LOVE\\nIS THE WAY", "font": "Impact", "align": "center"}   any subset, at least one
      {"cmd": "text_phrases", "phrases": ["one", "two"]}        replace the phrase list
      {"cmd": "text_add", "text": "..."}                        append (no text = the words on show)
      {"cmd": "text_del", "i": 2}
      {"cmd": "text_go", "i": 2}  or  {"cmd": "text_go", "d": 1}   show phrase i / step through the list
      {"cmd": "text_cycle", "on": true, "every": "bar", "order": "next"}   lyrics / mantra mode

  Whenever the words on show change the controller raises the engine command
  ``{"cmd": "text_show", "type": "text_show", "text", "font", "align", "warm": [phrases]}``.

* ``TextPass(ctx, out_size).render(values, clock, t, dt, bands, palette_rgb) -> Texture``: glyphs are rasterized on
  the CPU with Pillow into signed distance fields (a worker thread, only when the words or the font change; glyphs
  are cached), packed into an atlas, laid out every frame (size, tracking, line spacing, movement are live faders)
  and drawn as instanced quads into the LETTER FIELD, an RGBA16F texture the size of the output:
      r = signed distance to the letter edge in letter heights (em), positive inside, -PAD far away
      g = letter index 0..1 (colour per letter), b = letter alpha, a = letter height in frame heights
  With text_mode 7 it also keeps its own half-resolution trail buffer (``trail_tex``, texture unit TRAIL_UNIT).

* ``SCENE_GLSL``: the compositor, injected into live_visuals.SCENE_FS at the ``//@TEXT_GLSL`` marker. It colours
  the field (palette / fixed / rainbow / light chase, outline, glow) and integrates it per text_mode. Everything is
  display-only (after the echo stage), except the two displacement modes which bend the pattern's coordinates.

Fonts: every .ttf / .otf / .ttc in C:\\Windows\\Fonts, the per-user font folder and ``fonts/`` next to this file.

    python text_layer.py --gallery      # every integration mode x 2 fonts through the real engine -> test_frames/text/
    python text_layer.py --bench        # fps of the pass alone and of the engine with the layer up
"""
import io
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAU = 2.0 * math.pi

# ------------------------------------------------------------------------------------------------ contract

PARAM_NAMES = ["l_text", "text_size", "text_x", "text_y", "text_track", "text_line", "text_tilt", "text_color",
               "text_hue", "text_sat", "text_fill", "text_outline", "text_glow", "text_move", "text_speed",
               "text_wave", "text_pulse", "text_mode", "text_weave", "text_dissolve"]
# read by SCENE_GLSL as p_<name>; the rest are consumed on the CPU by TextPass (live_visuals.CPU_ONLY)
GLSL_PARAMS = ("l_text", "text_color", "text_hue", "text_sat", "text_fill", "text_outline", "text_glow", "text_mode",
               "text_weave", "text_dissolve")
CPU_ONLY_PARAMS = frozenset(n for n in PARAM_NAMES if n not in GLSL_PARAMS)
MODES = ["Overlay", "Woven", "Stencil mask", "Knockout", "Displace", "Rides the scene", "Kaleidoscope", "Echo trails"]
COLOR_MODES = ["Show palette", "Fixed colour", "Rainbow letters", "Light chase"]
MOVES = ["Static", "Drift", "Orbit ring", "Marquee", "Letter wave", "Tunnel spiral", "Beat bounce"]
FIELD_UNIT, TRAIL_UNIT, ATLAS_UNIT = 10, 11, 12        # texture units (the engine uses 0-5, 7, 8, 9)

MAX_TEXT, MAX_LINES, MAX_PHRASES, MAX_FONT = 240, 8, 32, 80
ALIGNS = ("center", "left", "right")
EVERY = {"beat": 1, "bar": 4, "bar2": 8, "bar4": 16, "phrase": 32}     # auto-cycle period in beats
ORDERS = ("next", "random")
DEFAULT_FONTS = ("Segoe UI Black", "Arial Black", "Impact", "Segoe UI Bold", "Arial Bold", "Arial", "Verdana")
GALLERY_PALETTE = [(0.00, 0.85, 0.80), (1.00, 0.05, 0.55), (1.00, 0.70, 0.05), (0.28, 0.10, 1.00), (0.10, 0.90, 0.30)]


# ------------------------------------------------------------------------------------------------ text state

def clean_text(s, required=False):
    """Words from untrusted input: str, control characters dropped, at most MAX_LINES lines / MAX_TEXT characters."""
    if not isinstance(s, str):
        raise ValueError("'text' must be text")
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    s = "".join(ch for ch in s if ch == "\n" or (ord(ch) >= 32 and ord(ch) != 127))
    lines = [ln.rstrip() for ln in s.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    s = "\n".join(lines[:MAX_LINES])[:MAX_TEXT]
    if required and not s.strip():
        raise ValueError("'text' is empty")
    return s


def clean_font(s):
    if not isinstance(s, str) or len(s) > MAX_FONT or any(ord(ch) < 32 for ch in s):
        raise ValueError(f"'font' must be a font name of at most {MAX_FONT} characters")
    return s.strip()


def _v_text(c):
    out = {"cmd": "text"}
    if "text" in c and c["text"] is not None:
        out["text"] = clean_text(c["text"])
    if c.get("font") is not None:
        out["font"] = clean_font(c["font"])
    if c.get("align") is not None:
        if c["align"] not in ALIGNS:
            raise ValueError(f"'align' must be one of {ALIGNS}")
        out["align"] = c["align"]
    if len(out) == 1:
        raise ValueError("'text' needs 'text', 'font' or 'align'")
    return out


def _v_phrases(c):
    p = c.get("phrases")
    if not isinstance(p, list) or len(p) > MAX_PHRASES:
        raise ValueError(f"'phrases' must be a list of at most {MAX_PHRASES} texts")
    return {"cmd": "text_phrases", "phrases": [clean_text(x, required=True) for x in p]}


def _v_index(c, key, lo, hi):
    v = c.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or int(v) != v or not lo <= v <= hi:
        raise ValueError(f"'{key}' must be a whole number {lo}..{hi}")
    return int(v)


def _v_add(c):
    out = {"cmd": "text_add"}
    if c.get("text") is not None:
        out["text"] = clean_text(c["text"], required=True)
    return out


def _v_go(c):
    if "i" in c:
        return {"cmd": "text_go", "i": _v_index(c, "i", 0, MAX_PHRASES - 1)}
    if "d" in c:
        return {"cmd": "text_go", "d": _v_index(c, "d", -1, 1)}
    raise ValueError("'text_go' needs 'i' (phrase) or 'd' (step)")


def _v_cycle(c):
    out = {"cmd": "text_cycle"}
    if "on" in c:
        if not isinstance(c["on"], bool):
            raise ValueError("'on' must be true or false")
        out["on"] = c["on"]
    if "every" in c:
        if c["every"] not in EVERY:
            raise ValueError(f"'every' must be one of {list(EVERY)}")
        out["every"] = c["every"]
    if "order" in c:
        if c["order"] not in ORDERS:
            raise ValueError(f"'order' must be one of {ORDERS}")
        out["order"] = c["order"]
    if len(out) == 1:
        raise ValueError("'text_cycle' needs 'on', 'every' or 'order'")
    return out


VALIDATORS = {
    "text": _v_text,
    "text_phrases": _v_phrases,
    "text_add": _v_add,
    "text_del": lambda c: {"cmd": "text_del", "i": _v_index(c, "i", 0, MAX_PHRASES - 1)},
    "text_go": _v_go,
    "text_cycle": _v_cycle,
}
TEXT_CMDS = frozenset(VALIDATORS)


class TextState:
    """The words on show, the font, the phrase list and its auto-cycle. Pure python; owned by the controller."""

    def __init__(self, seed=None):
        self.text, self.font, self.align = "", "", "center"
        self.phrases, self.idx = [], -1                       # idx = which phrase is on show (-1 = free text)
        self.cycle = dict(on=False, every="bar", order="next")
        self._rng = random.Random(seed)
        self._slot = None                                     # last cycle boundary seen
        self._dirty = False                                   # the engine has not been told about the current words

    # -- commands (already validated)
    def do(self, c):
        cmd = c["cmd"]
        if cmd == "text":
            if "text" in c:
                self.text = c["text"]
                self.idx = self.phrases.index(self.text) if self.text in self.phrases else -1
            if "font" in c:
                self.font = c["font"]
            if "align" in c:
                self.align = c["align"]
            self._dirty = True
        elif cmd == "text_phrases":
            self.phrases = list(c["phrases"])
            self.idx = self.phrases.index(self.text) if self.text in self.phrases else -1
            self._dirty = True                                # the engine pre-rasterizes the list
        elif cmd == "text_add":
            t = c.get("text", self.text)
            if t.strip() and len(self.phrases) < MAX_PHRASES:
                self.phrases.append(t)
                if t == self.text and self.idx < 0:
                    self.idx = len(self.phrases) - 1
                self._dirty = True
        elif cmd == "text_del":
            if 0 <= c["i"] < len(self.phrases):
                del self.phrases[c["i"]]
                self.idx = -1 if self.idx == c["i"] else self.idx - (1 if self.idx > c["i"] else 0)
        elif cmd == "text_go":
            if not self.phrases:
                return
            if "i" in c:
                if c["i"] >= len(self.phrases):
                    return
                self._show(c["i"])
            else:
                self._show((self.idx + c["d"]) % len(self.phrases) if self.idx >= 0 else (0 if c["d"] >= 0 else len(self.phrases) - 1))
        elif cmd == "text_cycle":
            for k in ("on", "every", "order"):
                if k in c:
                    self.cycle[k] = c[k]
            self._slot = None

    def _show(self, i):
        self.idx = i
        if self.phrases[i] != self.text:
            self.text = self.phrases[i]
            self._dirty = True

    def tick(self, beat):
        """Call once per frame with the beat counter: steps through the phrase list on its beat / bar boundary."""
        if not self.cycle["on"] or not self.phrases:
            self._slot = None
            return
        try:
            slot = math.floor(float(beat) / EVERY.get(self.cycle["every"], 4))
        except (TypeError, ValueError, OverflowError):
            return
        if self._slot is None:
            self._slot = slot
            if self.idx < 0:                                  # turning the cycle on with free text up: start the list
                self._show(0)
            return
        if slot == self._slot:
            return
        self._slot = slot
        n = len(self.phrases)
        if self.cycle["order"] == "random" and n > 1:
            i = self._rng.randrange(n - 1)
            self._show(i + (1 if i >= self.idx >= 0 else 0))
        else:
            self._show((self.idx + 1) % n)

    def pop_show(self):
        """The engine command payload if the words / font changed since the last call, else None."""
        if not self._dirty:
            return None
        self._dirty = False
        return dict(text=self.text, font=self.font, align=self.align, warm=list(self.phrases))

    # -- panel + persistence
    def state(self):
        return dict(text=self.text, font=self.font, align=self.align, phrases=list(self.phrases), idx=self.idx,
                    cycle=dict(self.cycle))

    to_json = state

    def restore(self, d):
        """Settings from live_state.json (untrusted): every field is validated on its own; never raises."""
        if not isinstance(d, dict):
            return False
        for cmd in (dict(cmd="text_phrases", phrases=d.get("phrases")),
                    dict(cmd="text", text=d.get("text"), font=d.get("font"), align=d.get("align")),
                    dict(d.get("cycle") if isinstance(d.get("cycle"), dict) else {}, cmd="text_cycle")):
            try:
                self.do(VALIDATORS[cmd["cmd"]](cmd))
            except Exception:
                if cmd["cmd"] == "text":                      # one bad field does not spoil the others
                    for k in ("text", "font", "align"):
                        try:
                            self.do(_v_text({k: d.get(k)}))
                        except Exception:
                            pass
        self._dirty = bool(self.text.strip())
        return True


# ------------------------------------------------------------------------------------------------ fonts

FONT_EXT = (".ttf", ".otf", ".ttc", ".otc")
_FONTS, _FONTS_SIG, _FONTS_LOCK = None, None, threading.Lock()
_PREVIEWS, _PREVIEWS_LOCK = OrderedDict(), threading.Lock()


def font_dirs():
    """(folder, is_user_font) in priority order: the project's own fonts/ first, so a dropped-in file wins a name clash."""
    out = [(HERE / "fonts", True)]
    out.append((Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts", False))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        out.append((Path(local) / "Microsoft" / "Windows" / "Fonts", False))
    if sys.platform != "win32":
        out += [(Path("/usr/share/fonts"), False), (Path("/Library/Fonts"), False)]
    return out


def _font_files(folder):
    try:
        if sys.platform == "win32":
            return sorted(p for p in folder.iterdir() if p.suffix.lower() in FONT_EXT and p.is_file())
        return sorted(p for p in folder.rglob("*") if p.suffix.lower() in FONT_EXT and p.is_file())
    except OSError:
        return []


def _dirs_signature(dirs):
    sig = []
    for folder, _ in dirs:
        try:
            st = folder.stat()
            sig.append((str(folder), st.st_mtime_ns))
        except OSError:
            sig.append((str(folder), None))
    return tuple(sig)


def list_fonts(dirs=None, refresh=False):
    """[{"name", "path", "user"}], sorted by name. Names are "Family Style" ("Regular" is dropped). Cached; a change
    in any font folder (a file dropped into fonts/) rescans. Costs about half a second the first time: never call it
    on the render thread (TextPass resolves fonts on its worker)."""
    global _FONTS, _FONTS_SIG
    from PIL import ImageFont
    custom = dirs is not None
    dirs = [(Path(d), bool(u)) for d, u in dirs] if custom else font_dirs()
    sig = _dirs_signature(dirs)
    with _FONTS_LOCK:
        if not custom and not refresh and _FONTS is not None and sig == _FONTS_SIG:
            return _FONTS
        seen, out = set(), []
        for folder, user in dirs:
            for fp in _font_files(folder):
                try:
                    fam, style = ImageFont.truetype(str(fp), 12).getname()
                except Exception:
                    continue
                name = " ".join(x for x in (fam or fp.stem, "" if (style or "").lower() in ("regular", "") else style) if x)
                name = "".join(ch for ch in name if ord(ch) >= 32).strip()[:MAX_FONT]
                key = name.casefold()
                if not name or key in seen:
                    continue
                seen.add(key)
                out.append(dict(name=name, path=str(fp), user=user))
        out.sort(key=lambda f: (not f["user"], f["name"].casefold()))
        if not custom:
            _FONTS, _FONTS_SIG = out, sig
        return out


def font_names():
    return [f["name"] for f in list_fonts()]


def resolve_font(name=None):
    """Font name -> {"name", "path"}; an unknown / empty name falls back to a bold default. None if no font at all."""
    fonts = list_fonts()
    by = {f["name"].casefold(): f for f in fonts}
    for cand in ((name or "").strip(),) + DEFAULT_FONTS:
        if cand and cand.casefold() in by:
            return by[cand.casefold()]
    return fonts[0] if fonts else None


def font_preview_png(name, height=30, max_width=360):
    """Small PNG of the font's own name set in that font (white on transparent), for the panel's font list. None
    if the font is unknown. Cached in memory."""
    from PIL import Image, ImageDraw, ImageFont
    f = next((x for x in list_fonts() if x["name"].casefold() == str(name).casefold()), None)
    if f is None:
        return None
    key = (f["path"], height)
    with _PREVIEWS_LOCK:
        if key in _PREVIEWS:
            _PREVIEWS.move_to_end(key)
            return _PREVIEWS[key]
    try:
        font = ImageFont.truetype(f["path"], int(height * 0.72))
        label = f["name"]
        box = font.getbbox(label)
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:        # symbol font: its name has no glyphs in it
            label = "ABC abc 123"
            box = font.getbbox(label)
        w = min(max_width, max(8, box[2] - box[0] + 8))
        img = Image.new("LA", (w, height), (255, 0))
        ImageDraw.Draw(img).text((4 - box[0], (height - (box[3] - box[1])) // 2 - box[1]), label, font=font, fill=(255, 255))
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True)
        data = buf.getvalue()
    except Exception:
        return None
    with _PREVIEWS_LOCK:
        _PREVIEWS[key] = data
        while len(_PREVIEWS) > 700:
            _PREVIEWS.popitem(last=False)
    return data


# ------------------------------------------------------------------------------------------------ SDF glyphs

EM = 96                 # atlas pixels per letter height (em)
SS = 3                  # the glyph is rasterized SS times larger and the distance transform runs at that size
PAD = 0.25              # distance range kept around every glyph, in em (glow / outline / displacement reach)
PAD_PX = int(EM * PAD)  # 24 atlas pixels
ATLAS_W, ATLAS_MAX_H = 2048, 4096
_GLYPHS, _GLYPHS_LOCK, GLYPH_CACHE = OrderedDict(), threading.Lock(), 2000
_PILFONTS = OrderedDict()
RASTER_COUNT = 0        # glyphs rasterized so far (tests: the cache works)


def _nearest_1d(mask):
    """For every pixel, the distance along axis 0 to the nearest True pixel of its column (inf if none)."""
    import numpy as np
    h, w = mask.shape
    rows = np.arange(h, dtype=np.float32)[:, None]
    up = np.where(mask, rows, -np.inf)
    np.maximum.accumulate(up, axis=0, out=up)
    down = np.where(mask, rows, np.inf)
    down = np.minimum.accumulate(down[::-1], axis=0)[::-1]
    return np.minimum(rows - up, down - rows)


def _edt_sub(mask, radius, ys, xs):
    """Exact Euclidean distance to the nearest True pixel of ``mask``, capped at ``radius``, evaluated only at the
    pixel grid ys x xs (numpy, separable: a 1D pass down the columns, then a min over +-radius along the rows)."""
    import numpy as np
    ys, xs = np.asarray(ys), np.asarray(xs)
    step = int(xs[1] - xs[0]) if len(xs) > 1 else 1
    g = np.minimum(_nearest_1d(mask[:, :]), radius + 1.0)[ys, :].astype(np.float32)
    g2 = np.pad(g * g, ((0, 0), (radius, radius)), constant_values=float((radius + 1.0) ** 2))
    best = np.full((len(ys), len(xs)), float((radius + 1.0) ** 2), dtype=np.float32)
    x0, n = int(xs[0]) + radius, len(xs)
    tmp = np.empty_like(best)
    for dx in range(-radius, radius + 1):                    # strided views: no fancy indexing in the loop
        a = x0 + dx
        np.add(g2[:, a:a + (n - 1) * step + 1:step], float(dx * dx), out=tmp)
        np.minimum(best, tmp, out=best)
    return np.minimum(np.sqrt(best), float(radius))


def sdf_from_mask(inside, ss=SS, radius=None):
    """Signed distance field of a binary glyph image (True = ink), sampled at every ``ss``-th pixel centre.
    Returns float32 in units of ONE OUTPUT pixel: positive inside, negative outside, clipped to +-radius/ss."""
    import numpy as np
    inside = np.asarray(inside, dtype=bool)
    radius = int(radius if radius is not None else PAD_PX * ss)
    h, w = inside.shape
    ys, xs = np.arange(ss // 2, h, ss), np.arange(ss // 2, w, ss)
    if not inside.any():
        return np.full((len(ys), len(xs)), -radius / ss, dtype=np.float32)
    d_out = _edt_sub(inside, radius + 1, ys, xs)             # outside pixels: distance to the ink
    d_in = _edt_sub(~inside, radius + 1, ys, xs)             # ink pixels: distance to the background
    sd = np.where(inside[np.ix_(ys, xs)], d_in - 0.5, 0.5 - d_out)
    return (np.clip(sd, -radius, radius) / ss).astype(np.float32)


def _pil_font(path):
    from PIL import ImageFont
    f = _PILFONTS.get(path)
    if f is None:
        f = ImageFont.truetype(path, EM * SS)
        _PILFONTS[path] = f
        while len(_PILFONTS) > 6:
            _PILFONTS.popitem(last=False)
    else:
        _PILFONTS.move_to_end(path)
    return f


def glyph_sdf(path, ch):
    """One glyph as a signed distance field. Returns None for blanks, else a dict:
    sdf float16 [h, w] in em (row 0 = top), x0 / y0 = the cell's lower-left corner relative to the pen position on
    the baseline (em, y up), w / h = cell size in em. Cached per (font file, character)."""
    global RASTER_COUNT
    import numpy as np
    from PIL import Image, ImageDraw
    key = (path, ch)
    with _GLYPHS_LOCK:
        if key in _GLYPHS:
            _GLYPHS.move_to_end(key)
            return _GLYPHS[key]
    font = _pil_font(path)
    try:
        l, t, r, b = font.getbbox(ch, anchor="ls")
    except Exception:
        l = t = r = b = 0
    out = None
    S = PAD_PX * SS
    if r - l >= 1 and b - t >= 1 and (r - l) <= EM * SS * 4 and (b - t) <= EM * SS * 4:
        w = -(-(r - l + 2 * S) // SS) * SS
        h = -(-(b - t + 2 * S) // SS) * SS
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).text((S - l, S - t), ch, font=font, fill=255, anchor="ls")
        inside = np.asarray(img) >= 128
        if inside.any():
            sd = sdf_from_mask(inside, SS, S) / EM
            out = dict(sdf=sd.astype(np.float16), x0=(l - S) / (EM * SS), y0=-(t - S + h) / (EM * SS),
                       w=w / (EM * SS), h=h / (EM * SS))
            RASTER_COUNT += 1
    with _GLYPHS_LOCK:
        _GLYPHS[key] = out
        while len(_GLYPHS) > GLYPH_CACHE:
            _GLYPHS.popitem(last=False)
    return out


class Layout:
    """One text block ready to draw: per-letter pen positions (em), atlas cells, and the atlas itself (numpy)."""
    __slots__ = ("text", "font", "align", "n", "x", "adv", "line", "pos", "quad", "uv", "widths", "counts",
                 "n_lines", "atlas", "key")


def build_layout(text, font_path, font_name="", align="center"):
    """Rasterize (or fetch from the cache) every glyph of ``text`` and lay the lines out. CPU only, any thread."""
    import numpy as np
    font = _pil_font(font_path)
    scale = 1.0 / (EM * SS)
    lines = text.split("\n")[:MAX_LINES]
    cells, order = {}, []                                    # char -> glyph dict; chars in atlas order
    xs, advs, lns, poss, chars, widths, counts = [], [], [], [], [], [], []
    for j, line in enumerate(lines):
        pen, prev, k = 0.0, None, 0
        for ch in line:
            try:
                adv = float(font.getlength(ch))
                if prev is not None:                         # pair kerning: what the pair measures minus its parts
                    pen += float(font.getlength(prev + ch)) - float(font.getlength(prev)) - adv
            except Exception:
                adv = EM * SS * 0.5
            if ch not in cells:
                g = glyph_sdf(font_path, ch)
                cells[ch] = g
                if g is not None:
                    order.append(ch)
            if cells[ch] is not None:
                xs.append(pen * scale); advs.append(adv * scale); lns.append(j); poss.append(k); chars.append(ch)
            pen += adv
            prev, k = ch, k + 1
        widths.append(pen * scale)
        counts.append(k)
    # ---- pack the atlas (shelves, widest rows first is not needed: cells are all about one em tall)
    rects, x, y, row_h = {}, 0, 0, 0
    for ch in order:
        gh, gw = cells[ch]["sdf"].shape
        if x + gw > ATLAS_W:
            x, y, row_h = 0, y + row_h, 0
        if y + gh > ATLAS_MAX_H:
            break                                            # out of room: the remaining glyphs are not drawn
        rects[ch] = (x, y, gw, gh)
        x, row_h = x + gw, max(row_h, gh)
    H = max(4, y + row_h)
    atlas = np.full((H, ATLAS_W), -PAD, dtype=np.float16)
    for ch, (ax, ay, gw, gh) in rects.items():
        atlas[ay:ay + gh, ax:ax + gw] = cells[ch]["sdf"]
    keep = [i for i, ch in enumerate(chars) if ch in rects]
    lay = Layout()
    lay.text, lay.font, lay.align, lay.key = text, font_name, align, (text, font_path, align)
    lay.n = len(keep)
    lay.x = np.array([xs[i] for i in keep], dtype=np.float64)
    lay.adv = np.array([advs[i] for i in keep], dtype=np.float64)
    lay.line = np.array([lns[i] for i in keep], dtype=np.int64)
    lay.pos = np.array([poss[i] for i in keep], dtype=np.float64)
    lay.quad = np.array([[cells[chars[i]][k] for k in ("x0", "y0", "w", "h")] for i in keep], dtype=np.float64).reshape(-1, 4)
    uv = []
    for i in keep:
        ax, ay, gw, gh = rects[chars[i]]
        uv.append((ax / ATLAS_W, (ay + gh) / H, (ax + gw) / ATLAS_W, ay / H))     # u0, v(bottom), u1, v(top)
    lay.uv = np.array(uv, dtype=np.float64).reshape(-1, 4)
    lay.widths = np.array(widths, dtype=np.float64)
    lay.counts = np.array(counts, dtype=np.float64)
    lay.n_lines = len(lines)
    lay.atlas = atlas
    return lay


# ------------------------------------------------------------------------------------------------ shaders

FIELD_VS = """
#version 330
in vec2 in_c;
in vec2 i_p; in float i_s; in float i_r; in vec4 i_q; in vec4 i_uv; in float i_id; in float i_a;
uniform float aspect;
out vec2 v_uv; out vec3 v_d;
void main(){
  vec2 e = (i_q.xy + in_c*i_q.zw)*i_s; float c = cos(i_r), s = sin(i_r);
  vec2 p = i_p + vec2(c*e.x - s*e.y, s*e.x + c*e.y);
  v_uv = mix(i_uv.xy, i_uv.zw, in_c); v_d = vec3(i_id, i_a, i_s);
  gl_Position = vec4(p.x/(aspect*0.5), p.y*2.0, 0.0, 1.0);
}
"""

FIELD_FS = """
#version 330
uniform sampler2D atlas;
in vec2 v_uv; in vec3 v_d;
out vec4 f;
void main(){
  float d = texture(atlas, v_uv).r;
  if (!(d > -1.0 && d < 1.0)) d = -%(PAD)s;
  f = vec4(d, v_d.x, v_d.y, v_d.z);
  gl_FragDepth = clamp(0.5 - d, 0.0, 1.0);          // where padded cells overlap, the nearer letter wins every channel
}
""" % dict(PAD=PAD)

QUAD_VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# text_mode 7: the letters' own feedback buffer. r = trail energy, g = the letter index it came from.
TRAIL_FS = """
#version 330
uniform sampler2D field, prev;
uniform float decay, zoom, turn, aspect, grow;
uniform vec2 drift;
in vec2 v_uv;
out vec4 f;
void main(){
  vec4 F = texture(field, v_uv);
  float d = (F.r > -1.0 && F.r < 1.0) ? F.r : -1.0;
  float aa = clamp(fwidth(d), 1e-4, 0.08);
  float c = smoothstep(-aa, aa, d + grow)*clamp(F.b, 0.0, 1.0);
  vec2 q = (v_uv - 0.5)*vec2(aspect, 1.0);
  float cs = cos(turn), sn = sin(turn);
  q = vec2(cs*q.x - sn*q.y, sn*q.x + cs*q.y)*zoom - drift;
  q = q/vec2(aspect, 1.0) + 0.5;
  vec4 P = texture(prev, q);
  float I = (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) ? 0.0 : P.r*decay - 0.004;
  if (!(I > 0.0)) I = 0.0;
  f = c >= I ? vec4(c, F.g, 0.0, 1.0) : vec4(I, P.g, 0.0, 1.0);
}
"""

# ---- the compositor, injected into live_visuals.SCENE_FS just before its main(). It may use everything SCENE_FS
# declares above that point (palette(), hueRot(), fbm(), rot(), max3(), t, pxh, chasePh, chaseCol, trZ, travelAct ...).
SCENE_GLSL = """
// ================================================================ text layer (text_layer.py)
uniform sampler2D texT, texTT;      // letter field (r = signed distance in em, g = letter index, b = alpha, a = em size); trails
uniform float p_l_text, p_text_mode, p_text_color, p_text_hue, p_text_sat, p_text_fill, p_text_outline, p_text_glow;
uniform float p_text_weave, p_text_dissolve;
const float TXPAD = %(PAD)s;
vec4 txField(vec2 uv){
  vec4 F = texture(texT, uv);
  if (!(F.r > -1.0 && F.r < 1.0) || !(F.b >= 0.0 && F.b <= 1.0) || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
    return vec4(-TXPAD, 0.0, 0.0, 0.0);
  return F;
}
vec3 txHue(float h){ return clamp(abs(fract(h+vec3(0.0, 2.0/3.0, 1.0/3.0))*6.0-3.0)-1.0, 0.0, 1.0); }
// colour of the letter with index id (0..1 along the text)
vec3 txInk(float id){
  float cm = floor(p_text_color+0.5); vec3 c;
  if (cm < 0.5) c = palette(0.12 + t*0.03 + chaseCol + 0.22*id);
  else if (cm < 1.5) c = txHue(p_text_hue);
  else if (cm < 2.5) c = txHue(p_text_hue + id*0.83 + 0.12*phrase_phase);
  else {
    float k = floor(p_chase_mode+0.5) < 0.5 ? 1.0 : 2.0;
    float w = 1.0 - abs(fract(id*0.92*k - chasePh*k) - 0.5)*2.0;
    float key = smoothstep(1.0 - max(p_chase_width, 0.22), 1.0, w);
    c = palette(id*0.6 + t*0.03 + chaseCol);
    c = c/max(max3(c), 0.3)*(0.10 + 0.90*key*(1.0 + 0.25*exp(-beat_phase*5.0)));
    return mix(vec3(max3(c)), c, p_text_sat);
  }
  if (cm < 0.5) {                                                    // palette colours follow the show's grade
    c = max(mix(vec3(dot(c, vec3(.299,.587,.114))), c, min(p_saturation, 1.0)), 0.0);
    if (p_hue > 0.0) c = max(hueRot(c, p_hue*TAU), 0.0);
  }
  c /= max(max3(c), 0.3);
  return mix(vec3(1.0), c, p_text_sat);
}
// x = lit body, y = outline stroke, z = glow, w = silhouette (body + stroke); all carry the letter alpha
vec4 txShape(vec4 F, vec2 np, float pat, out float d){
  d = F.r;
  float D = clamp(p_text_dissolve, 0.0, 1.0), gate = 1.0;
  if (D > 0.0) {                                                     // erosion by drifting noise + the pattern shows through
    float n = fbm(np*7.0 + vec2(swN[1], -swN[2])*0.6)/0.875;
    d -= D*(0.008 + 0.20*n*n + 0.03*D);
    gate = mix(1.0, smoothstep(0.03, 0.26, pat), D*0.85);
  }
  float aa = clamp(fwidth(d), 1e-4, 0.06)*0.75;
  float fill = clamp(p_text_fill, 0.0, 1.0);
  float ow = max(p_text_outline*0.07, 0.0);
  float body = smoothstep(-aa, aa, d), sil = smoothstep(-aa, aa, d + ow);
  float inner = smoothstep(-aa, aa, d - max(0.026, ow*0.6));
  float lit = mix(body - inner, body, fill);
  float pad = smoothstep(-TXPAD, -TXPAD*0.55, d);
  float glow = p_text_glow*exp(min(d + ow, 0.0)/(0.030 + 0.075*p_text_glow))*pad*(1.0 - sil);
  return vec4(lit*gate, (sil - body)*gate, glow*gate, sil)*clamp(F.b, 0.0, 1.0);
}
// modes 1 and 4: the letters bend the pattern's coordinates (added to p before any pattern is drawn)
vec2 textWarp(vec2 pscr, vec2 asp){
  float m = floor(p_text_mode+0.5);
  if (m != 1.0 && m != 4.0) return vec2(0.0);
  vec2 uv = pscr/asp + 0.5;
  vec4 F = txField(uv);
  if (F.b <= 0.0) return vec2(0.0);
  vec2 ex = vec2(2.0*pxh, 0.0)/asp, ey = vec2(0.0, 2.0*pxh)/asp;
  vec2 g = vec2(txField(uv+ex).r - txField(uv-ex).r, txField(uv+ey).r - txField(uv-ey).r);
  float gl = length(g); vec2 n = gl > 1e-5 ? g/gl : vec2(0.0);      // points into the letter
  float A = clamp(p_l_text, 0.0, 1.0)*F.b, W = clamp(p_text_weave, 0.0, 1.0), s = clamp(F.a, 0.0, 1.0), d = F.r;
  if (m == 1.0) {                                                    // woven: the lines crowd up against the letters
    float band = smoothstep(-TXPAD, -0.02, d)*(1.0 - smoothstep(-0.02, 0.03, d));
    return n*band*band*A*W*0.20*s;
  }
  float aa = 2.0*pxh/max(s, 1e-3);
  float in_ = smoothstep(-aa, aa, d), bevel = exp(-abs(d)/0.035);    // displace: a glass letter (lens body + refracting rim)
  return (-pscr*0.42 + vec2(0.09, 0.06))*in_*A*W + n*bevel*A*W*0.07*s;
}
vec2 txToUv(vec2 q, vec2 asp){ return q/asp + 0.5; }
// display stage (after the echo): colour the field and integrate it per text_mode
void textMix(inout vec3 o, inout float cov, vec2 pscr, vec2 pt, vec2 asp, float edge){
  float A = clamp(p_l_text, 0.0, 1.0), m = floor(p_text_mode+0.5), W = clamp(p_text_weave, 0.0, 1.0);
  float pat = max3(o), d = -TXPAD;
  vec2 uv = txToUv(pscr, asp);
  vec4 F = vec4(-TXPAD, 0.0, 0.0, 0.0), S = vec4(0.0);
  if (m == 5.0) {
    // rides the scene: sampled through the pattern's own warp chain (ripple, wave, pan, rotate / spin, punch, bulge,
    // twirl, zoom). Flying (travel) sends copies of the words down the tunnel, far -> near, like the depth layers.
    vec2 q = pt*exp2(-p_zoom*1.2);
    float act = clamp(travelAct, 0.0, 1.0);
    int nk = act > 0.02 ? 2 : 1;                                     // two copies, cross-faded: one is always nearly alone
    for (int k = 0; k < nk; k++) {
      float fk = fract(trZ*0.5 + float(k)*0.5);
      float sc = exp2((fk - 0.5)*3.4*act);
      float win = sin(PI*fk); win = win*win*win;
      float al = mix(k == 0 ? 1.0 : 0.0, win, act);
      vec4 Fk = txField(txToUv(q/sc, asp)); float dk;
      vec4 Sk = txShape(Fk, q/sc, pat, dk)*al;
      if (Sk.w + Sk.z >= S.w + S.z) { F = Fk; d = dk; }
      S = max(S, Sk);
    }
  } else if (m == 6.0) {
    // kaleidoscope: the words repeat round the fold count, one copy per wedge, reading outward along each spoke.
    // The wedge is unrolled log-polar, so the letters taper toward the centre and a block that fits the wedge at
    // the rim fits it all the way in (no letters cut by the fold seams); more folds = smaller words.
    float hs = PI/floor(clamp(p_folds, 2.0, 24.0)+0.5);
    float r = length(pt)+1e-5, v = (fract(atan(pt.y, pt.x)/(2.0*hs) + 0.5) - 0.5)*2.0*hs;
    float k = 0.3927/hs;                                             // 8 folds = life size
    vec2 kp = vec2(log(r/0.27)*0.27, v*0.27)*k;
    F = txField(txToUv(kp, asp));
    S = txShape(F, kp, pat, d)*smoothstep(0.02, 0.07, r);
  } else {
    F = txField(uv);
    S = txShape(F, pscr, pat, d);
  }
  vec3 ink = txInk(F.g), rim = mix(ink, vec3(1.0), 0.62);
  vec3 L = ink*S.x + rim*S.y + ink*S.z*0.75;
  float fa = clamp(F.b, 0.0, 1.0);
  if (m == 1.0) {
    // woven / engraved: the letter is a channel cut through the pattern, drawn as nested contour lines (the way the
    // kene line field is built); the pattern lines crowd up against it (textWarp) and pass over it in alternate bands
    float sp = 0.050, lw = 0.011;
    float aa = clamp(fwidth(d), 1e-4, 0.06)*0.75;
    float cd = abs(fract(d/sp + 0.5) - 0.5)*sp;                      // distance to the nearest contour d = k*sp
    float lines = (1.0 - smoothstep(lw - aa, lw + aa, cd))*smoothstep(-lw - aa, -lw + aa, d)*(1.0 - smoothstep(2.5*sp, 2.5*sp + 2.0*aa, d));
    lines = mix(lines, 0.55*smoothstep(-aa, aa, d), smoothstep(sp*0.30, sp*0.55, aa))*fa;   // too small to resolve: solid
    float first = 1.0 - smoothstep(lw - aa, lw + aa, abs(d));
    vec3 lc = mix(ink, rim, first)*lines;
    vec2 wv = rot(0.7854)*pscr;
    float over = smoothstep(0.35, 0.65, abs(fract(wv.x*5.0) - 0.5)*2.0)*W;   // bands where the pattern crosses over
    float cut = S.w*(1.0 - over*smoothstep(0.10, 0.35, pat));
    o = o*(1.0 - A*0.96*cut) + (lc*(1.0 - over*smoothstep(0.10, 0.35, pat)) + ink*S.z*0.5)*A*edge;
  } else if (m == 2.0) {
    // stencil: the visual shows only inside the letters; a hairline keeps them readable where the visual is dark
    float aa = clamp(fwidth(d), 1e-4, 0.06)*0.75;
    float hair = (1.0 - smoothstep(0.006, 0.006 + 2.0*aa, abs(d)))*fa;
    vec3 inside = min(o*1.25, vec3(1.0))*S.w + rim*S.y + ink*hair*(0.10 + 0.55*W);
    o = mix(o, inside*edge, A);
  } else if (m == 3.0) {
    // knockout: black letters cut out of the visual, with a thin corona so they also read over black
    float corona = exp(min(d, 0.0)/(0.012 + 0.05*W))*smoothstep(-TXPAD, -TXPAD*0.55, d)*(1.0 - S.w)*fa;
    o = o*(1.0 - A*S.w) + ink*corona*(0.15 + 0.75*W)*A*edge;
  } else if (m == 4.0) {
    // displace: the letters are glass (textWarp bends the pattern); only a faint bright rim is drawn
    float aa = clamp(fwidth(d), 1e-4, 0.06)*0.75;
    float hair = (1.0 - smoothstep(0.007, 0.007 + 2.0*aa, abs(d)))*fa;
    o = o*(1.0 - 0.9*A*hair)*(1.0 + 0.25*A*S.w) + (rim*hair*0.62 + rim*S.y + ink*S.z*0.35)*A*edge;
  } else {
    if (m == 7.0) {                                                  // echo trails: the letters' own feedback buffer
      vec4 T = texture(texTT, uv);
      float I = clamp(T.r, 0.0, 1.0);
      vec3 tc = txInk(T.g + 0.45*(1.0 - I))*I*I*0.85;
      o += tc*(1.0 - clamp(o, 0.0, 1.0))*A*edge;
    }
    // overlay (0), rides the scene (5), kaleidoscope (6): opaque letters over a dark moat, lit additively
    float moat = max(S.w, W*0.9*smoothstep(-0.11, -0.01, d)*fa);
    moat *= clamp(max3(ink)*1.3, 0.0, 1.0);                          // a letter the light chase has not reached is a ghost, not a hole
    o = o*(1.0 - A*0.94*moat) + L*A*edge;
  }
  o /= max(1.0, max3(o));
  cov = max(cov, S.w*A);
}
""" % dict(PAD=PAD)

# what live_visuals falls back to if this module cannot be imported: the hooks compile to nothing
SCENE_GLSL_STUB = """
uniform float p_l_text;
vec2 textWarp(vec2 pscr, vec2 asp){ return vec2(0.0); }
void textMix(inout vec3 o, inout float cov, vec2 pscr, vec2 pt, vec2 asp, float edge){}
"""


# ------------------------------------------------------------------------------------------------ the pass

def _num(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _val(values, name, default, lo, hi):
    return min(max(_num((values or {}).get(name, default), default), lo), hi)


def em_size(size):
    """text_size 0..1 -> letter height as a fraction of the frame height (log scale)."""
    return 0.03 * 20.0 ** min(max(size, 0.0), 1.0)


INST_FLOATS = 14                         # i_p 2, i_s 1, i_r 1, i_q 4, i_uv 4, i_id 1, i_a 1
MAX_INST = MAX_TEXT
FADE_IN = 0.14                           # s: new words fade / pop in


class TextPass:
    """render(values, clock, t, dt, bands, palette_rgb) -> the letter field texture (RGBA16F, output size)."""

    def __init__(self, ctx, out_size=(1280, 800)):
        import moderngl
        import numpy as np
        self._mgl, self._np = moderngl, np
        self.ctx = ctx
        self.out_size = (max(16, int(out_size[0])), max(16, int(out_size[1])))
        self.aspect = self.out_size[0] / self.out_size[1]
        self._corner = ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype="f4").tobytes())
        self._quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._inst = ctx.buffer(reserve=MAX_INST * INST_FLOATS * 4, dynamic=True)
        self._prog = ctx.program(vertex_shader=FIELD_VS, fragment_shader=FIELD_FS)
        self._vao = ctx.vertex_array(self._prog, [
            (self._corner, "2f", "in_c"),
            (self._inst, "2f 1f 1f 4f 4f 1f 1f/i", "i_p", "i_s", "i_r", "i_q", "i_uv", "i_id", "i_a")])
        self._tex = ctx.texture(self.out_size, 4, dtype="f2")
        self._tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._tex.repeat_x = self._tex.repeat_y = False
        self._depth = ctx.depth_renderbuffer(self.out_size)
        self._fbo = ctx.framebuffer(color_attachments=[self._tex], depth_attachment=self._depth)
        self._fbo.clear(-PAD, 0.0, 0.0, 0.0, depth=1.0)
        # trails (text_mode 7): half resolution ping-pong
        self.trail_size = (max(8, self.out_size[0] // 2), max(8, self.out_size[1] // 2))
        self._tprog = ctx.program(vertex_shader=QUAD_VS, fragment_shader=TRAIL_FS)
        self._tvao = ctx.vertex_array(self._tprog, [(self._quad, "2f", "in_pos")])
        self._ttex = [ctx.texture(self.trail_size, 4) for _ in range(2)]
        for tx in self._ttex:
            tx.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tx.repeat_x = tx.repeat_y = False
        self._tfbo = [ctx.framebuffer(color_attachments=[tx]) for tx in self._ttex]
        for fb in self._tfbo:
            fb.clear(0.0, 0.0, 0.0, 0.0)
        self._ti, self._trail_blank = 0, True
        self._atlas = None                               # GL texture of the layout on show
        self._atlas_size = None
        self._lay = None                                 # Layout on show
        self._blank = True
        self.draws = 0                                   # field passes drawn (tests / cost checks)
        self.trail_draws = 0
        self.uploads = 0                                 # atlas uploads (tests: only when the words / font change)
        self.instances = 0
        self._last_t = None
        self._ph = 0.0                                   # movement phase (double, wrapped)
        self._since = 1e9                                # s since the words on show changed
        self.font_name = ""                              # the font actually in use (after fallback)
        self.error = None
        # worker: rasterizes off the render thread; only the newest request matters
        self._lock = threading.Lock()
        self._want = None                                # (text, font, align) requested
        self._req = None                                 # request the worker has not picked up yet
        self._warm = []
        self._ready = None                               # Layout waiting for upload
        self._idle = threading.Event(); self._idle.set()
        self._wake = threading.Event()
        self._stop = False
        self._thread = None
        self._gl = None
        try:                                             # the caller's blend / depth switches are read back with raw GL
            import math_layers as _ml
            self._gl, self._gl_blend = _ml._gl_lib(), _ml._GL_BLEND
        except Exception:
            self._gl = None

    # ---------------------------------------------------------------- words

    def set_text(self, text, font=None, align=None, warm=None, wait=False, timeout=20.0):
        """Show these words (in this font). Returns at once; the glyphs are rasterized on the worker thread and the
        old words stay up until the new ones are ready. ``warm`` = other phrases to pre-rasterize (the phrase list).
        ``wait=True`` blocks until the layout is ready (headless renders / tests)."""
        if isinstance(text, str):
            text = clean_text(text)
        else:                                            # no words given (a font-only change): keep what is up
            text = self._want[0] if self._want else ""
        font = font.strip()[:MAX_FONT] if isinstance(font, str) else (self._want[1] if self._want else "")
        align = align if align in ALIGNS else (self._want[2] if self._want else "center")
        want = (text, font, align)
        with self._lock:
            if isinstance(warm, (list, tuple)):
                self._warm = [w for w in warm if isinstance(w, str)][:MAX_PHRASES]
            if want != self._want:
                self._want = want
                self._req = want
                self._idle.clear()
            elif self._warm:
                self._idle.clear()
            busy = not self._idle.is_set()
        if not busy:
            return True
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name="text-raster", daemon=True)
            self._thread.start()
        self._wake.set()
        return self._idle.wait(timeout) if wait else True

    def _work(self):
        while not self._stop:
            self._wake.wait(0.5)
            self._wake.clear()
            while not self._stop:
                with self._lock:
                    req, self._req = self._req, None
                    warm, self._warm = (self._warm, []) if req is None else ([], self._warm)
                    if req is None and not warm:
                        self._idle.set()
                        break
                try:
                    f = resolve_font(req[1] if req else (self._want[1] if self._want else ""))
                    if f is None:
                        raise RuntimeError("no fonts found")
                    if req is not None:
                        lay = build_layout(req[0], f["path"], f["name"], req[2]) if req[0].strip() else None
                        with self._lock:
                            if self._req is None:        # nothing newer arrived meanwhile
                                self._ready = (req, lay, f["name"])
                    else:
                        for phrase in warm:              # pre-rasterize the phrase list into the glyph cache
                            if self._stop or self._req is not None:
                                break
                            for ch in set(phrase):
                                if not ch.isspace():
                                    glyph_sdf(f["path"], ch)
                    self.error = None
                except Exception as ex:                  # a broken font file must not take the layer down
                    self.error = f"{type(ex).__name__}: {ex}"
                    if req is not None:
                        with self._lock:
                            if self._req is None:
                                self._ready = (req, None, "")

    def _adopt(self):
        """Render thread: upload a finished layout (one texture write; only when the words / font changed)."""
        with self._lock:
            got, self._ready = self._ready, None
        if got is None:
            return
        _, lay, name = got
        self.font_name = name
        if lay is None or lay.n == 0:
            self._lay = None
            return
        size = (lay.atlas.shape[1], lay.atlas.shape[0])
        if self._atlas is None or self._atlas_size != size:
            if self._atlas is not None:
                self._atlas.release()
            self._atlas = self.ctx.texture(size, 1, dtype="f2")
            self._atlas.repeat_x = self._atlas.repeat_y = False
            self._atlas_size = size
        self._atlas.write(lay.atlas.tobytes())
        self._atlas.build_mipmaps(0, 3)
        self._atlas.filter = (self._mgl.LINEAR_MIPMAP_LINEAR, self._mgl.LINEAR)
        self.uploads += 1
        self._lay = lay
        self._since = 0.0

    @property
    def text(self):
        return self._lay.text if self._lay is not None else ""

    @property
    def trail_tex(self):
        return self._ttex[self._ti]

    def release(self):
        self._stop = True
        self._wake.set()
        objs = [self._fbo, self._depth, self._tex, self._vao, self._prog, self._inst, self._corner, self._quad,
                self._tvao, self._tprog, *self._tfbo, *self._ttex]
        if self._atlas is not None:
            objs.append(self._atlas)
        for o in objs:
            try:
                o.release()
            except Exception:
                pass

    close = release

    # ---------------------------------------------------------------- layout -> instances

    def instances_for(self, values, clock, dt, bands):
        """Per-letter instance rows [n, 14] for this frame (numpy float32). Pure CPU: movement, size, tracking."""
        np, lay, asp = self._np, self._lay, self.aspect
        v = values or {}
        move = int(round(_val(v, "text_move", 0.0, 0.0, 6.0)))
        speed = _val(v, "text_speed", 0.4, -1.0, 1.0)
        amt = _val(v, "text_wave", 0.5, 0.0, 1.0)
        pulse = _val(v, "text_pulse", 0.3, 0.0, 1.0)
        track = _val(v, "text_track", 0.0, -0.2, 1.0)
        pitch = 1.15 * _val(v, "text_line", 1.0, 0.6, 2.5)
        tilt = _val(v, "text_tilt", 0.0, -1.0, 1.0) * math.pi
        cx = _val(v, "text_x", 0.0, -1.0, 1.0) * asp * 0.5
        cy = _val(v, "text_y", 0.0, -1.0, 1.0) * 0.5
        s = em_size(_val(v, "text_size", 0.45, 0.0, 1.0))
        bp = _num((clock or {}).get("beat_phase", 0.0)) % 1.0
        bass = min(max(_num(bands[0]), 0.0), 1.0)
        self._ph = (self._ph + dt * speed) % (TAU * 64.0)
        ph = self._ph
        n = lay.n
        # ---- block layout in em: tracking widens every line, lines are aligned, the block is centred on its pivot
        widths = lay.widths + np.maximum(lay.counts - 1.0, 0.0) * track
        maxw = float(widths.max()) if len(widths) else 0.0
        lw = widths[lay.line]
        x = lay.x + lay.pos * track
        if lay.align == "left":
            x = x - maxw * 0.5
        elif lay.align == "right":
            x = x + maxw * 0.5 - lw
        else:
            x = x - lw * 0.5
        y = ((lay.n_lines - 1) * 0.5 - lay.line) * pitch - 0.35          # baseline; 0.35 em ~ half a capital
        px, py = x + lay.adv * 0.5, y + 0.35                              # each letter's pivot (its centre)
        if move not in (3, 5) and maxw > 1e-6:                            # keep the block inside the frame
            s = min(s, 0.94 * asp / maxw)
        kick = math.exp(-bp * 4.0)
        s *= 1.0 + pulse * (0.16 * bass + 0.07 * kick)
        rot = np.full(n, tilt)
        sc = np.full(n, s)
        alpha = np.ones(n)
        idn = np.arange(n, dtype=np.float64) / max(n - 1, 1)
        ct, st_ = math.cos(tilt), math.sin(tilt)
        if move == 2:
            # orbit: the first line bent into a ring that turns round the block position (further lines = inner rings)
            R = (0.16 + 0.26 * amt) - lay.line * pitch * s
            R = np.maximum(R, 0.04)
            circ = TAU * R
            fit = np.minimum(1.0, circ * 0.92 / np.maximum(lw * s, 1e-6))
            sc = sc * fit
            ang = math.pi * 0.5 - (px * s * fit) / R + ph * 0.6 + tilt
            X, Y = cx + R * np.cos(ang), cy + R * np.sin(ang)
            rot = ang - math.pi * 0.5
        elif move == 5:
            # spiral into the tunnel: the words ride a log spiral toward the centre, shrinking as they go
            R0, turns = 0.56, 1.5 + 2.0 * amt
            TH = TAU * turns
            b = math.log(R0 / 0.012) / TH
            q = s / (R0 * 0.72)                                           # radians per em at the rim
            # the words repeat along the spiral so it is never empty (as many copies as fit, lines interleaved)
            span = (maxw + 2.5) * q * max(lay.n_lines, 1)
            reps = int(min(max(TH // max(span, 1e-6), 1), max(MAX_INST // max(n, 1), 1), 12))
            if reps > 1:
                px = np.tile(px, reps)
                idn = np.tile(idn, reps)
                rep_off = np.repeat(np.arange(reps) * (TH / reps), n)
                line_of = np.tile(lay.line, reps)
                self._rep = reps
            else:
                rep_off, line_of, self._rep = 0.0, lay.line, 1
            line_off = line_of * (TH / reps / max(lay.n_lines, 1))
            th = (ph * 0.55 - px * q - line_off - rep_off) % TH
            r = R0 * np.exp(-b * th)
            sc = s * r / (R0 * 0.72)
            ang = math.pi * 0.5 + th + tilt
            X, Y = cx + r * np.cos(ang), cy + r * np.sin(ang)
            rot = ang - math.pi * 0.5 - math.atan(b)
            u = th / TH
            alpha = np.clip(u / 0.06, 0.0, 1.0) * np.clip((1.0 - u) / 0.12, 0.0, 1.0)
        else:
            ox, oy = np.zeros(n), np.zeros(n)
            if move == 1:                                                 # drift: a slow float around the position
                cx += 0.16 * asp * amt * math.sin(ph * 0.61)
                cy += 0.20 * amt * math.sin(ph * 0.83 + 1.0)
                rot = rot + 0.10 * amt * math.sin(ph * 0.47 + 2.0)
                ct, st_ = math.cos(tilt + 0.10 * amt * math.sin(ph * 0.47 + 2.0)), math.sin(tilt + 0.10 * amt * math.sin(ph * 0.47 + 2.0))
            elif move == 3:                                               # marquee: endless horizontal scroll, wrapped
                span = max(maxw + 2.0, (asp + 0.4) / max(s, 1e-6) + 2.0)
                ox = ((px - ph * 1.6 + span * 0.5) % span) - span * 0.5 - px
            elif move == 4:                                               # wave: every letter rides its own sine
                w = ph * 3.2 - np.arange(n) * 0.62
                oy = 0.55 * amt * np.sin(w)
                rot = rot + 0.30 * amt * np.cos(w)
                sc = sc * (1.0 + 0.16 * amt * np.sin(w + 0.8))
            elif move == 6:                                               # bounce: up on the beat, letters a touch apart
                lag = np.clip(bp - 0.06 * (np.arange(n) % 5) * abs(speed), 0.0, 1.0)
                oy = 0.70 * amt * 4.0 * lag * (1.0 - lag)
                sc = sc * (1.0 + 0.10 * amt * np.exp(-lag * 7.0))
            ex, ey = (px + ox) * s, (py + oy) * s
            X, Y = cx + ct * ex - st_ * ey, cy + st_ * ex + ct * ey
        fade = min(1.0, self._since / FADE_IN)
        alpha = alpha * fade
        sc = sc * (1.0 + 0.10 * (1.0 - fade) ** 2)
        reps = len(X) // max(n, 1)                                        # > 1 only for the spiral's repeated copies
        quad, uv, adv = (np.tile(lay.quad, (reps, 1)), np.tile(lay.uv, (reps, 1)), np.tile(lay.adv, reps)) if reps > 1 else (lay.quad, lay.uv, lay.adv)
        inst = np.empty((len(X), INST_FLOATS), dtype=np.float32)
        inst[:, 0], inst[:, 1], inst[:, 2], inst[:, 3] = X, Y, sc, rot
        inst[:, 4] = quad[:, 0] - adv * 0.5                              # cell corner relative to the letter's pivot
        inst[:, 5] = quad[:, 1] - 0.35
        inst[:, 6:8] = quad[:, 2:4]
        inst[:, 8:12] = uv
        inst[:, 12], inst[:, 13] = idn, alpha
        inst = inst[alpha > 0.002]
        if not np.isfinite(inst).all():
            inst = inst[np.isfinite(inst).all(axis=1)]
        return inst[:MAX_INST]

    # ---------------------------------------------------------------- render

    def render(self, values, clock=None, t=0.0, dt=None, bands=(0.0, 0.0, 0.0), palette_rgb=None):
        ctx, mgl = self.ctx, self._mgl
        amt = _val(values, "l_text", 0.0, 0.0, 1.0)
        t = _num(t)
        if dt is None:
            dt = 0.0 if self._last_t is None else t - self._last_t
        dt = min(max(_num(dt), 0.0), 0.25)
        self._last_t = t
        bands = [min(max(_num(b), 0.0), 1.0) for b in (list(bands if bands is not None else ()) + [0, 0, 0])[:3]]
        if amt <= 0.001:                                 # off: nothing is laid out, uploaded or drawn
            if not self._blank or not self._trail_blank:
                self._clear_all()
            return self._tex
        self._adopt()
        self._since += dt
        mode = int(round(_val(values, "text_mode", 0.0, 0.0, 7.0)))
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        blend_on = bool(self._gl.glIsEnabled(self._gl_blend)) if self._gl is not None else False
        depth_on = bool(self._gl.glIsEnabled(0x0B71)) if self._gl is not None else False
        try:
            self._fbo.use()
            ctx.viewport = (0, 0, *self.out_size)
            self._fbo.clear(-PAD, 0.0, 0.0, 0.0, depth=1.0)
            self._blank = True
            if self._lay is not None and self._atlas is not None:
                inst = self.instances_for(values, clock, dt, bands)
                self.instances = len(inst)
                if len(inst):
                    self._inst.write(inst.tobytes())
                    ctx.disable(mgl.BLEND)
                    ctx.enable(mgl.DEPTH_TEST)
                    ctx.depth_func = "<"
                    self._atlas.use(ATLAS_UNIT)
                    self._prog["atlas"].value = ATLAS_UNIT
                    self._prog["aspect"].value = self.aspect
                    self._vao.render(mgl.TRIANGLE_STRIP, instances=len(inst))
                    ctx.disable(mgl.DEPTH_TEST)
                    self.draws += 1
                    self._blank = False
            if mode == 7:
                self._trails(values, dt, bands)
            elif not self._trail_blank:
                for fb in self._tfbo:
                    fb.clear(0.0, 0.0, 0.0, 0.0)
                self._trail_blank = True
            self._ttex[self._ti].use(TRAIL_UNIT)
        finally:
            if blend_on:
                ctx.enable(mgl.BLEND)
            if depth_on:
                ctx.enable(mgl.DEPTH_TEST)
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:
                ctx.screen.use()
        return self._tex

    def _trails(self, values, dt, bands):
        ctx, mgl = self.ctx, self._mgl
        W = _val(values, "text_weave", 0.5, 0.0, 1.0)
        speed = _val(values, "text_speed", 0.4, -1.0, 1.0)
        k = dt * 60.0                                    # the buffer steps per frame: scale everything by the frame time
        src, dst = self._ti, 1 - self._ti
        self._tfbo[dst].use()
        ctx.viewport = (0, 0, *self.trail_size)
        ctx.disable(mgl.BLEND)
        self._tex.use(FIELD_UNIT)
        self._ttex[src].use(TRAIL_UNIT)
        p = self._tprog
        p["field"].value, p["prev"].value = FIELD_UNIT, TRAIL_UNIT
        p["decay"].value = (0.90 + 0.085 * W) ** k if k > 0 else 1.0
        p["zoom"].value = (1.0 - (0.006 + 0.012 * W) * (1.0 + 1.2 * bands[0])) ** k if k > 0 else 1.0
        p["turn"].value = 0.006 * speed * k
        p["aspect"].value = self.aspect
        p["grow"].value = _val(values, "text_outline", 0.0, 0.0, 1.0) * 0.07
        if "drift" in p:
            p["drift"].value = (0.0, 0.0)
        self._tvao.render(mgl.TRIANGLE_STRIP)
        self._ti = dst
        self.trail_draws += 1
        self._trail_blank = False

    def _clear_all(self):
        ctx = self.ctx
        prev_fbo, prev_vp = ctx.fbo, ctx.viewport
        try:
            self._fbo.use()
            self._fbo.clear(-PAD, 0.0, 0.0, 0.0, depth=1.0)
            for fb in self._tfbo:
                fb.clear(0.0, 0.0, 0.0, 0.0)
        finally:
            try:
                prev_fbo.use()
                ctx.viewport = prev_vp
            except Exception:
                ctx.screen.use()
        self._blank = self._trail_blank = True


# ------------------------------------------------------------------------------------------------ headless helpers

def hidden_context(size=(64, 64)):
    from math_layers import hidden_context as hc
    return hc(size)


def default_values(**over):
    import params
    v = params.defaults()
    v["l_text"] = 1.0
    v.update(over)
    return v


def make_clock(t, bpm=124.0):
    b = t * bpm / 60.0
    return dict(bpm=bpm, beat=b, beat_phase=b % 1.0, bar_phase=(b / 4) % 1.0, phrase_phase=(b / 32) % 1.0)


def read_field(tp):
    """The letter field as float32 [h, w, 4], row 0 = top."""
    import numpy as np
    return np.frombuffer(tp._tex.read(), dtype=np.float16).reshape(tp.out_size[1], tp.out_size[0], 4)[::-1].astype(np.float32)


def read_trail(tp):
    import numpy as np
    return np.frombuffer(tp.trail_tex.read(), dtype=np.uint8).reshape(tp.trail_size[1], tp.trail_size[0], 4)[::-1]


GALLERY_FONTS = ("Impact", "Georgia Bold Italic")
GALLERY_TEXT = "LOVE IS\nTHE WAY"


def gallery_jobs(fonts=GALLERY_FONTS, text=GALLERY_TEXT, frames=100):
    """Batch jobs for live_visuals --batch: every integration mode x the fonts, then colour / movement / style looks."""
    jobs = []

    def job(name, font=fonts[0], words=text, n=frames, **sets):
        base = dict(l_text=1.0, text_size=0.62)
        base.update(sets)
        jobs.append(dict(name=name, sets=base, frames=n, script={"0": [["text", words, font]]}))

    for fi, font in enumerate(fonts):
        for m, label in enumerate(MODES):
            extra = {}
            if m == 6:
                extra = dict(text_size=0.36)
            if m == 5:
                extra = dict(twirl=0.45, bulge=0.3, zoom=0.15)
            job(f"mode{m}_{label.lower().replace(' ', '_')}_font{fi + 1}", font, text_mode=float(m), **extra)
    job("flat_mode1_woven", symmetry=0.0, text_mode=1.0)
    job("flat_mode2_stencil", symmetry=0.0, text_mode=2.0)
    job("flat_mode4_displace", symmetry=0.0, text_mode=4.0)
    job("travel_mode5_rides", text_mode=5.0, travel=0.6, text_size=0.5)
    for c, label in enumerate(COLOR_MODES):
        job(f"color{c}_{label.lower().replace(' ', '_')}", text_color=float(c), text_outline=0.4 if c == 1 else 0.0)
    for mv, label in enumerate(MOVES):
        job(f"move{mv}_{label.lower().replace(' ', '_')}", words="THE WAY IS LOVE" if mv in (2, 3, 5) else text,
            text_move=float(mv), text_size=0.5, text_color=2.0)
    job("style_hollow_outline", text_fill=0.0, text_outline=0.5, text_glow=0.6)
    job("style_dissolve_04", text_dissolve=0.4)
    job("style_dissolve_08", text_dissolve=0.8)
    job("style_tracking_tilt", text_track=0.35, text_tilt=0.08, text_line=1.4)
    return jobs


def run_engine_batch(jobs, out_dir, timeout=900):
    """Render jobs through the real engine, headless (never a visible window). Returns batch.json's content."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = out_dir / "jobs.json"
    spec.write_text(json.dumps(jobs), encoding="utf-8")
    r = subprocess.run([sys.executable, "-u", str(HERE / "live_visuals.py"), "--batch", str(spec), "--out", str(out_dir)],
                       cwd=str(HERE), capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"engine batch failed ({r.returncode}):\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}")
    stats = out_dir / "batch.json"
    return json.loads(stats.read_text(encoding="utf-8")) if stats.exists() else []


def gallery(out_dir=None):
    out_dir = Path(out_dir or HERE / "test_frames" / "text")
    jobs = gallery_jobs()
    stats = run_engine_batch(jobs, out_dir)
    try:
        from PIL import Image, ImageDraw
        names = [j["name"] for j in jobs]
        cols, tw, th = 4, 320, 200
        rows = -(-len(names) // cols)
        sheet = Image.new("RGB", (cols * tw, rows * (th + 16)), (0, 0, 0))
        dr = ImageDraw.Draw(sheet)
        for i, n in enumerate(names):
            fp = out_dir / f"{n}.png"
            if fp.exists():
                sheet.paste(Image.open(fp).convert("RGB").resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * (th + 16) + 16))
            dr.text(((i % cols) * tw + 4, (i // cols) * (th + 16) + 2), n, fill=(210, 210, 210))
        sheet.save(out_dir / "contact_sheet.png")
    except Exception as ex:
        print(f"[text] contact sheet failed: {ex}")
    print(f"[text] {len(jobs)} frames -> {out_dir}")
    return stats


def bench_engine(out_dir=None, frames=600):
    """Uncapped headless fps of the real engine at 1280x800: default look, then with the text layer up per mode."""
    import re
    import tempfile
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="text_bench_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [("base", []), ("text_overlay", ["l_text=1"]), ("text_woven", ["l_text=1", "text_mode=1"]),
            ("text_rides_travel", ["l_text=1", "text_mode=5", "travel=0.5"]), ("text_kaleido", ["l_text=1", "text_mode=6"]),
            ("text_trails_wave", ["l_text=1", "text_mode=7", "text_move=4"])]
    res = {}
    for name, sets in runs:
        cmd = [sys.executable, "-u", str(HERE / "live_visuals.py"), "--test", str(frames), "--out", str(out_dir / name),
               "--text", "LOVE IS|THE WAY"]
        for s in sets:
            cmd += ["--set", s]
        r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, timeout=300)
        m = re.search(r"= (\d+) fps", r.stdout)
        if r.returncode != 0 or not m:
            raise RuntimeError(f"engine run '{name}' failed:\n{r.stdout[-800:]}\n{r.stderr[-2000:]}")
        res[name] = float(m.group(1))
        print(f"[text bench] engine {name:20s} {res[name]:.0f} fps")
    (out_dir / "bench.json").write_text(json.dumps(res), encoding="utf-8")
    return res


def bench(out_size=(1280, 800), frames=300):
    """Cost of the pass alone (ms per frame, GPU finished), per movement mode and with trails."""
    ctx, _win = hidden_context()
    tp = TextPass(ctx, out_size)
    tp.set_text("LOVE IS THE WAY\nALL IS ONE", "Impact", wait=True)
    out = {}
    for label, over in (("off", dict(l_text=0.0)), ("static", {}), ("wave", dict(text_move=4.0)),
                        ("spiral", dict(text_move=5.0)), ("trails", dict(text_mode=7.0))):
        vals = default_values(**over)
        for i in range(10):
            tp.render(vals, make_clock(i / 60), i / 60, 1 / 60, (0.5, 0.3, 0.2), GALLERY_PALETTE)
        ctx.finish()
        t0 = time.perf_counter()
        for i in range(frames):
            tp.render(vals, make_clock(i / 60), i / 60, 1 / 60, (0.5, 0.3, 0.2), GALLERY_PALETTE)
        ctx.finish()
        out[label] = (time.perf_counter() - t0) / frames * 1000.0
        print(f"[text bench] {label:8s} {out[label]:.3f} ms / frame")
    tp.release()
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="text layer: gallery / bench / font list (always headless)")
    ap.add_argument("--gallery", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-engine", action="store_true", help="fps of the real engine with and without the layer")
    ap.add_argument("--fonts", action="store_true", help="print the font list")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.fonts:
        for f in list_fonts():
            print(("* " if f["user"] else "  ") + f["name"])
    if a.bench:
        bench()
    if a.bench_engine:
        bench_engine(a.out)
    if a.gallery:
        gallery(a.out)
