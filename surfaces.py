"""Multi-surface projection mapping: the model (pure Python: no numpy, no GL).

A SurfaceSet is an ordered list of surfaces. Each surface is a quad (4 output corners in projector
pixels) that shows a part of the rendered visual, optionally cut by an N-point polygon mask, so several
boxes and sculpture faces each get their own warped piece of the picture and everything else stays black.

Coordinates
  projector pixels, origin TOP-LEFT, x right, y down, in the set's `output` size (1280 x 800).
  Quad corner order: 0 TL, 1 TR, 2 BR, 3 BL (clockwise on screen). The legacy corner pin (mapping.json,
  the `corner` / `nudge` commands, `state.corners`) is BL, BR, TR, TL in 0..1 with v up; see legacy_corners().

surfaces.json
  {"version": 1, "output": [1280, 800], "sel": <id>, "surfaces": [{
      "id": 1, "name": "Box front", "order": 0, "enabled": true,
      "shape": "quad" | "poly",            poly = the quad's picture cut by `mask`
      "corners": [[x, y] x 4],             TL, TR, BR, BL
      "mask": [[x, y] x 3..8],             projector pixels, straight edges (used when shape is "poly")
      "source": {"mode": "full" | "rect" | "span", "rect": [x, y, w, h]},   rect in 0..1 of the frame, y down
      "content": "main" | "solid" | "glow" | "black",
      "color": [r, g, b],                  0..1: identification colour, solid fill, edge-glow tint
      "opacity": 0..1, "brightness": 0..2, "glow": 0..1, "feather": px,
      "layers": [{"src", "on", "opacity", "blend", "scale", "scroll", "dir", "hue"} x 0..3],   content "layers"
      "maskfile": "name.png", "label": 1..255, "group": "Grass"}]}      shape "mask" only (see MASK SURFACES)

  content "layers" (PER-SURFACE LAYERS): the surface shows its OWN stack of up to 3 engine layers (LayerHub sources,
  or "main" = the show's picture), each with EYE (on), opacity, blend, scale, scroll speed + direction and hue shift,
  laid out in the surface's own 0..1 square (mask surfaces: along each part's flow uv). The list is kept when the
  surface shows something else, so "same as main" <-> "own layers" is a switch.

  shape "mask" (MASK SURFACES, thin parts such as grass blades): the outline is a projector-space label image
  masks/<maskfile> next to surfaces.json, RGBA = label (1..255), flow u (base -> tip), flow v (across), coverage.
  Every part of one group shares one file (one draw for a whole group); `corners` = the part's bounding box.

  source "span": every surface in span mode is a window onto ONE picture laid over the bounding box of all of
  them, so the image runs continuously from face to face as if they were one canvas.

Loading follows the scenes.json rules: a missing file migrates mapping.json into surface #1 (or gives one
full-frame surface), an unreadable file is left alone with saving disabled, a corrupt file is moved aside as
surfaces.json.<time>.bad.
"""
import copy
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

VERSION = 1
DEFAULT_OUTPUT = (1280, 800)
MAX_SURFACES = 64       # 24 -> 64 (2026-11-26): thin mask parts (grass blades) are one surface each
MAX_MASK = 8
MAX_POLY = 16            # clipped polygon points the renderer accepts
UNDO_DEPTH = 20
MAX_NAME = 40
MAX_FEATHER = 200.0
OVERSCAN = 0.5           # corners may sit this far (in frame sizes) outside the projector frame
MIN_CROSS = 4.0          # px^2: smallest corner cross product of a usable quad (about 2 px x 2 px)
CORNER_NAMES = ("TL", "TR", "BR", "BL")
SHAPES = ("quad", "poly", "mask")         # "mask" appended 2026-11-26
SOURCE_MODES = ("full", "rect", "span")
CONTENTS = ("main", "solid", "glow", "black", "layers")     # "layers" appended 2026-11-26 (order = shader index)
MAX_SURF_LAYERS = 3
BLENDS = ("normal", "add", "screen", "multiply")
# per-surface layer sources: "main" = the show's own picture, the rest = materials.LayerHub source ids (the engine
# registers what it actually has; an unknown / missing one draws nothing). Panel mirror: panel.html SURF_SRC.
LAYER_SOURCES = (("main", "Main picture"), ("kene", "Kené study"), ("beadstudy", "Bead study"),
                 ("blocks", "Folding Blocks"), ("beads", "Beadwork"), ("lines", "Sacred lines"), ("beings", "Beings"),
                 ("fractal", "Math fractals"), ("land", "3D deck"), ("stack", "Math stack look"), ("art", "Art image"),
                 ("ai", "AI image"), ("text", "Text"), ("lines4", "Line layer (4 ch)"), ("kene4", "Kené 4 channels"),
                 ("video", "Video loops"), ("camera", "Crowd camera"))
SOURCE_IDS = tuple(k for k, _ in LAYER_SOURCES)
LOOK_ROTA = ("kene", "fractal", "blocks", "beadstudy", "lines", "beings", "land", "beads")   # all moving
LAYER_DEFAULT = dict(src="kene", on=True, opacity=1.0, blend="normal", scale=1.0, scroll=0.0, dir=0.0, hue=0.0)
MASK_DIR = Path(__file__).resolve().parent / "masks"
OVERLAY_MODES = ("off", "sel", "all")
# distinct, projector-friendly identification colours (assigned by id)
COLORS = [(1.0, 0.25, 0.70), (0.10, 0.85, 1.0), (1.0, 0.80, 0.10), (0.35, 1.0, 0.35), (0.65, 0.40, 1.0), (1.0, 0.45, 0.15),
          (0.20, 0.55, 1.0), (0.85, 1.0, 0.20), (1.0, 0.30, 0.30), (0.20, 1.0, 0.75), (1.0, 0.55, 0.90), (0.75, 0.75, 0.75)]


# ---------------------------------------------------------------------------------------------- geometry

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def edge_lengths(pts):
    """Length of edge i (from point i to point i+1), in the units of the points."""
    n = len(pts)
    return [math.hypot(pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1]) for i in range(n)]


def angle_at(a, b, c):
    """Angle ABC in degrees (0..180)."""
    v1, v2 = _sub(a, b), _sub(c, b)
    d = math.hypot(*v1) * math.hypot(*v2)
    if d <= 0.0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / d))))


def signed_area(pts):
    """Shoelace area; positive for clockwise-on-screen winding (y down)."""
    n = len(pts)
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))


def interior_angles(pts):
    """Interior angle at every vertex, degrees. Reflex corners of a concave polygon read above 180."""
    n, sign = len(pts), 1.0 if signed_area(pts) >= 0 else -1.0
    out = []
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        ang = angle_at(a, b, c)
        out.append(360.0 - ang if _cross(_sub(b, a), _sub(c, b)) * sign < 0 else ang)
    return out


def centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def is_convex(pts, eps=MIN_CROSS):
    """Strictly convex polygon, either winding (a mirrored quad is a legitimate mapping)."""
    n = len(pts)
    if n < 3:
        return False
    zs = [_cross(_sub(pts[(i + 1) % n], pts[i]), _sub(pts[(i + 2) % n], pts[(i + 1) % n])) for i in range(n)]
    return all(z > eps for z in zs) or all(z < -eps for z in zs)


def is_simple(pts):
    """No two non-adjacent edges cross, and the polygon has some area."""
    n = len(pts)
    if n < 3 or abs(signed_area(pts)) < MIN_CROSS:
        return False

    def hit(p1, p2, p3, p4):
        d1, d2 = _cross(_sub(p4, p3), _sub(p1, p3)), _cross(_sub(p4, p3), _sub(p2, p3))
        d3, d4 = _cross(_sub(p2, p1), _sub(p3, p1)), _cross(_sub(p2, p1), _sub(p4, p1))
        return d1 * d2 < 0 and d3 * d4 < 0

    for i in range(n):
        for j in range(i + 2, n):
            if (j + 1) % n == i:
                continue
            if hit(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                return False
    return True


def point_in_poly(p, pts):
    inside, n = False, len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


def rotation(quad):
    """Tilt of the quad in degrees: mean direction of the top (TL-TR) and bottom (BL-BR) edges.
    Positive = clockwise on screen."""
    t, b = _sub(quad[1], quad[0]), _sub(quad[2], quad[3])
    return math.degrees(math.atan2(t[1] + b[1], t[0] + b[0]))


def skew(quad):
    """How far the quad's two mid-lines are from perpendicular, degrees (0 for any rectangle or keystone-only
    trapezoid; positive = leaning right at the top)."""
    mt, mb = centroid(quad[0:2]), centroid(quad[2:4])
    ml, mr = centroid((quad[0], quad[3])), centroid((quad[1], quad[2]))
    h, v = _sub(mr, ml), _sub(mb, mt)
    d = (math.hypot(*h) * math.hypot(*v)) or 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (h[0] * v[0] + h[1] * v[1]) / d)))) - 90.0


def size(quad):
    """(mean width, mean height) of the quad."""
    e = edge_lengths(quad)
    return ((e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0)


def aspect(quad):
    w, h = size(quad)
    return w / h if h > 0 else 0.0


def keystone(quad):
    """(horizontal %, vertical %): top vs bottom edge and right vs left edge, as a percentage of the mean."""
    e = edge_lengths(quad)
    w, h = (e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0
    return ((e[0] - e[2]) / (w or 1.0) * 100.0, (e[1] - e[3]) / (h or 1.0) * 100.0)


def rotate_pts(pts, deg, about=None):
    """Rotate points by deg (clockwise on screen) about a point (default: their centroid)."""
    cx, cy = about if about is not None else centroid(pts)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c] for x, y in pts]


def square_up(quad, level=False):
    """The nearest rectangle: same centre, area, aspect and tilt (level=True also removes the tilt)."""
    cx, cy = centroid(quad)
    asp, area = aspect(quad) or 1.0, abs(signed_area(quad))
    h = math.sqrt(area / asp)
    w = h * asp
    flip = -1.0 if signed_area(quad) < 0 else 1.0          # keep a mirrored quad mirrored
    rect = [[cx - w / 2, cy - flip * h / 2], [cx + w / 2, cy - flip * h / 2], [cx + w / 2, cy + flip * h / 2], [cx - w / 2, cy + flip * h / 2]]
    return rect if level else rotate_pts(rect, rotation(quad), (cx, cy))


def with_angle(quad, i, deg):
    """Corner i slides along the line from the midpoint of its two neighbours until its interior angle is deg."""
    a, c = quad[(i - 1) % 4], quad[(i + 1) % 4]
    m = ((a[0] + c[0]) / 2.0, (a[1] + c[1]) / 2.0)
    d = _sub(quad[i], m)
    n = math.hypot(*d) or 1.0
    d = (d[0] / n, d[1] / n)
    lo, hi = 1e-3, 1e5
    for _ in range(70):
        mid = (lo + hi) / 2.0
        if angle_at(a, (m[0] + d[0] * mid, m[1] + d[1] * mid), c) > deg:
            lo = mid
        else:
            hi = mid
    out = [list(p) for p in quad]
    out[i] = [m[0] + d[0] * lo, m[1] + d[1] * lo]
    return out


def with_edge(quad, i, length):
    """Edge i keeps its start corner and direction; its end corner moves to make the length."""
    a, j = quad[i], (i + 1) % 4
    b = quad[j]
    n = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
    out = [list(p) for p in quad]
    out[j] = [a[0] + (b[0] - a[0]) / n * length, a[1] + (b[1] - a[1]) / n * length]
    return out


def describe(quad, output=DEFAULT_OUTPUT):
    """Every live readout of one quad, for the panel and for tests."""
    w, h = size(quad)
    kh, kv = keystone(quad)
    return dict(px=[[float(x), float(y)] for x, y in quad],
                pct=[[x / output[0] * 100.0, y / output[1] * 100.0] for x, y in quad],
                angles=interior_angles(quad), edges=edge_lengths(quad), rotation=rotation(quad), skew=skew(quad),
                aspect=aspect(quad), keystone=[kh, kv], width=w, height=h, area=abs(signed_area(quad)),
                center=list(centroid(quad)))


# ---- 3x3 projective maths (row-major nested lists)

def _solve(a, b):
    """Gaussian elimination with partial pivoting. Raises ValueError when singular."""
    n = len(b)
    m = [list(map(float, a[i])) + [float(b[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            if f:
                for k in range(col, n + 1):
                    m[r][k] -= f * m[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][k] * x[k] for k in range(i + 1, n))) / m[i][i]
    return x


def homography(src, dst):
    """3x3 matrix H with H * (x, y, 1) ~ (u, v, 1) for the four point pairs src[i] -> dst[i].
    Raises ValueError for a degenerate set (three points in a line)."""
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = _solve(a, b)
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]


def apply_h(H, p):
    x, y = p
    w = H[2][0] * x + H[2][1] * y + H[2][2]
    if abs(w) < 1e-12:
        return (math.inf, math.inf)
    return ((H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w)


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def invert3(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-18:
        raise ValueError("singular")
    return [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
            [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]


def clip_poly(subject, clip):
    """Sutherland-Hodgman: the part of polygon `subject` inside the CONVEX polygon `clip`."""
    sign = 1.0 if signed_area(clip) >= 0 else -1.0
    out = [tuple(map(float, p)) for p in subject]
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        e = _sub(b, a)
        inp, out = out, []
        if not inp:
            break

        def side(p):
            return _cross(e, _sub(p, a)) * sign

        for k in range(len(inp)):
            p, q = inp[k], inp[(k + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    clean = []
    for p in out:                                         # drop repeated points
        if not clean or math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 1e-6:
            clean.append(p)
    if len(clean) > 1 and math.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1]) <= 1e-6:
        clean.pop()
    return clean


# ---------------------------------------------------------------------------------------------- validation

def _num(v, lo, hi, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"'{what}' must be a finite number")
    return min(hi, max(lo, float(v)))


def _int(v, lo, hi, what):
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ValueError(f"'{what}' must be an integer in {lo}..{hi}")
    return v


def _pt(p, output, what="point"):
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        raise ValueError(f"'{what}' must be [x, y]")
    return [_num(p[0], -OVERSCAN * output[0], (1 + OVERSCAN) * output[0], what),
            _num(p[1], -OVERSCAN * output[1], (1 + OVERSCAN) * output[1], what)]


def _pts(raw, output, lo, hi, what):
    if not isinstance(raw, (list, tuple)) or not lo <= len(raw) <= hi:
        raise ValueError(f"'{what}' must be a list of {lo}..{hi} points" if lo != hi else f"'{what}' must be {lo} points")
    return [_pt(p, output, what) for p in raw]


def _name(v):
    if not isinstance(v, str):
        raise ValueError("'name' must be text")
    v = "".join(ch for ch in v if ch.isprintable()).strip()[:MAX_NAME]
    if not v:
        raise ValueError("'name' must not be empty")
    return v


def _color(v):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError("'color' must be [r, g, b]")
    return [_num(x, 0.0, 1.0, "color") for x in v]


def _rect(v):
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise ValueError("'rect' must be [x, y, w, h]")
    x, y = _num(v[0], 0.0, 0.98, "rect"), _num(v[1], 0.0, 0.98, "rect")
    return [x, y, min(_num(v[2], 0.02, 1.0, "rect"), 1.0 - x), min(_num(v[3], 0.02, 1.0, "rect"), 1.0 - y)]


def _choice(v, options, what):
    if v not in options:
        raise ValueError(f"'{what}' must be one of {', '.join(options)}")
    return v


def _flag(v, what):
    if not isinstance(v, bool):
        raise ValueError(f"'{what}' must be true or false")
    return v


def _layer(v):
    """One per-surface layer row (missing fields take their defaults)."""
    if not isinstance(v, dict):
        raise ValueError("a layer must be an object")
    out = dict(LAYER_DEFAULT)
    if "src" in v:
        out["src"] = _choice(v["src"], SOURCE_IDS, "src")
    if "on" in v:
        out["on"] = _flag(v["on"], "on")
    if "blend" in v:
        out["blend"] = _choice(v["blend"], BLENDS, "blend")
    for k, lo, hi in (("opacity", 0.0, 1.0), ("scale", 0.1, 8.0), ("scroll", -2.0, 2.0), ("dir", -360.0, 360.0),
                      ("hue", -180.0, 180.0)):
        if k in v:
            out[k] = _num(v[k], lo, hi, k)
    return out


def _layers(v):
    if not isinstance(v, (list, tuple)) or len(v) > MAX_SURF_LAYERS:
        raise ValueError(f"'layers' must be a list of 0..{MAX_SURF_LAYERS} layers")
    return [_layer(x) for x in v]


def _maskfile(v):
    if not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_\-]{1,48}\.png", v):
        raise ValueError("'maskfile' must be a plain file name like grass_1.png")
    return v


def _group(v):
    if not isinstance(v, str):
        raise ValueError("'group' must be text")
    return "".join(ch for ch in v if ch.isprintable()).strip()[:MAX_NAME]


# field name in a "set" command -> cleaner
_FIELDS = {
    "enabled": lambda v: _flag(v, "enabled"),
    "opacity": lambda v: _num(v, 0.0, 1.0, "opacity"),
    "brightness": lambda v: _num(v, 0.0, 2.0, "brightness"),
    "glow": lambda v: _num(v, 0.0, 1.0, "glow"),
    "feather": lambda v: _num(v, 0.0, MAX_FEATHER, "feather"),
    "content": lambda v: _choice(v, CONTENTS, "content"),
    "shape": lambda v: _choice(v, SHAPES, "shape"),
    "src": lambda v: _choice(v, SOURCE_MODES, "src"),
    "rect": _rect,
    "color": _color,
    "layers": _layers,
}
_ID = 1 << 30
EDIT_OPS = ("add", "dup", "del", "rename", "move", "set", "corner", "nudge", "corners", "shift", "mask", "mask_add",
            "mask_del", "square", "angle", "edge", "rotate", "reset", "replace")
EDIT_OPS += ("set_layers", "looks")                 # per-surface layers (2026-11-26)
OPS = EDIT_OPS + ("select", "overlay", "undo", "redo", "save")
BIG = 1e5


def validate_cmd(c, output=DEFAULT_OUTPUT):
    """Normalized copy of one {"cmd": "surf", "op": ...} panel command; ValueError if malformed.
    Runs on the server thread: only finite, typed, bounded values reach the engine. Positions are bounded
    loosely here and clamped to the real output size when applied."""
    if not isinstance(c, dict) or c.get("op") not in OPS:
        raise ValueError("'op' must be one of " + ", ".join(OPS))
    op = c["op"]
    out = {"cmd": "surf", "type": "surf", "op": op}
    wide = (BIG / (1 + OVERSCAN), BIG / (1 + OVERSCAN))     # _pt clamps to +-1.5x this: just "finite and sane"
    if op in ("undo", "redo", "save"):
        return out
    if op == "overlay":
        if "mode" in c:
            out["mode"] = _choice(c["mode"], OVERLAY_MODES, "mode")
        if "grid" in c:
            out["grid"] = _flag(c["grid"], "grid")
        return out
    g = c.get("g")
    if g is not None:
        if isinstance(g, bool) or not isinstance(g, (int, str)) or len(str(g)) > 24:
            raise ValueError("'g' must be a short gesture id")
        out["g"] = g
    if op == "add":
        if c.get("name") is not None:
            out["name"] = _name(c["name"])
        return out
    if op == "looks":                                     # every surface a different moving look (round robin)
        return out
    if op == "replace":                                   # whole-set import (the panel's "paste layout")
        if not isinstance(c.get("surfaces"), list) or not 1 <= len(c["surfaces"]) <= MAX_SURFACES:
            raise ValueError("'surfaces' must be a list")
        out["surfaces"] = [clean_surface(s, wide, strict=True) for s in c["surfaces"]]
        return out
    out["id"] = _int(c.get("id"), 0, _ID, "id")
    if c.get("group_all") is True:                       # set / set_layers on every surface of the id's group
        out["group_all"] = True
    if op == "set_layers":
        out["layers"] = _layers(c.get("layers"))
        if "content" in c:
            out["content"] = _choice(c["content"], CONTENTS, "content")
    elif op == "rename":
        out["name"] = _name(c.get("name"))
    elif op == "move":
        out["to"] = _int(c.get("to"), 0, MAX_SURFACES - 1, "to")
    elif op == "set":
        n = 0
        for k, fn in _FIELDS.items():
            if k in c:
                out[k] = fn(c[k])
                n += 1
        if not n:
            raise ValueError("'set' needs at least one field")
    elif op in ("corner", "nudge"):
        out["m"] = bool(c.get("m", False)) if isinstance(c.get("m", False), bool) else False
        out["i"] = _int(c.get("i"), 0, (MAX_MASK if out["m"] else 4) - 1, "i")
        if op == "corner":
            out["x"], out["y"] = _num(c.get("x"), -BIG, BIG, "x"), _num(c.get("y"), -BIG, BIG, "y")
        else:
            out["dx"], out["dy"] = _num(c.get("dx"), -BIG, BIG, "dx"), _num(c.get("dy"), -BIG, BIG, "dy")
    elif op == "corners":
        out["pts"] = _pts(c.get("pts"), wide, 4, 4, "pts")
    elif op == "mask":
        out["pts"] = _pts(c.get("pts"), wide, 3, MAX_MASK, "pts")
    elif op == "shift":
        out["dx"], out["dy"] = _num(c.get("dx"), -BIG, BIG, "dx"), _num(c.get("dy"), -BIG, BIG, "dy")
    elif op == "mask_add":
        if c.get("i") is not None:
            out["i"] = _int(c["i"], 0, MAX_MASK - 1, "i")
    elif op == "mask_del":
        out["i"] = _int(c.get("i"), 0, MAX_MASK - 1, "i")
    elif op == "square":
        out["level"] = bool(c.get("level", False)) if isinstance(c.get("level", False), bool) else False
    elif op == "angle":
        out["i"], out["deg"] = _int(c.get("i"), 0, 3, "i"), _num(c.get("deg"), 1.0, 179.0, "deg")
    elif op == "edge":
        out["i"], out["len"] = _int(c.get("i"), 0, 3, "i"), _num(c.get("len"), 4.0, BIG, "len")
    elif op == "rotate":
        out["deg"] = _num(c.get("deg"), -360.0, 360.0, "deg")
    return out


def full_quad(output):
    w, h = float(output[0]), float(output[1])
    return [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]


def new_surface(sid, output=DEFAULT_OUTPUT, name=None, corners=None):
    return dict(id=int(sid), name=name or f"Surface {sid}", order=0, enabled=True, shape="quad",
                corners=[list(map(float, p)) for p in (corners or full_quad(output))], mask=[],
                source=dict(mode="full", rect=[0.0, 0.0, 1.0, 1.0]), content="main",
                color=list(COLORS[(int(sid) - 1) % len(COLORS)]), opacity=1.0, brightness=1.0, glow=0.0, feather=0.0,
                layers=[])


def clean_surface(raw, output=DEFAULT_OUTPUT, strict=False):
    """One surface from untrusted JSON -> normalized dict. Raises ValueError if it cannot be used.
    Optional fields fall back to their defaults (strict=True: a present but malformed field raises)."""
    if not isinstance(raw, dict):
        raise ValueError("surface must be an object")
    corners = _pts(raw.get("corners"), output, 4, 4, "corners")
    if not is_convex(corners):
        raise ValueError("corners must form a convex quad")
    sid = raw.get("id", 1)
    if isinstance(sid, bool) or not isinstance(sid, int) or not 0 <= sid <= _ID:
        if strict:
            raise ValueError("'id' must be a small positive integer")
        sid = 0                                            # from_dict gives it a free id
    s = new_surface(sid, output, corners=corners)
    if "name" not in raw and sid == 0:
        s["name"] = "Surface"

    def opt(key, fn, dst=None, sub=None):
        src = raw.get(sub, {}) if sub else raw
        if not isinstance(src, dict) or key not in src:
            return
        try:
            val = fn(src[key])
        except ValueError:
            if strict:
                raise
            return
        if sub:
            s[sub][dst or key] = val
        else:
            s[dst or key] = val

    opt("name", _name)
    for k in ("enabled", "opacity", "brightness", "glow", "feather", "content", "shape", "color"):
        opt(k, _FIELDS[k])
    opt("order", lambda v: _int(v, -_ID, _ID, "order"))
    opt("mode", _FIELDS["src"], sub="source")
    opt("rect", _rect, sub="source")
    opt("layers", _layers)
    opt("group", _group)
    opt("mask", lambda v: [] if v == [] else _pts(v, output, 3, MAX_MASK, "mask"))
    if s["shape"] == "mask":                             # a mask surface needs its label image + label
        try:
            s["maskfile"] = _maskfile(raw.get("maskfile"))
            s["label"] = _int(raw.get("label"), 1, 255, "label")
        except ValueError:
            if strict:
                raise
            s["shape"] = "quad"                          # unusable: it falls back to its bounding quad
        s["mask"] = []
        return s
    if s["mask"] and not is_simple(s["mask"]):
        if strict:
            raise ValueError("mask must not cross itself")
        s["mask"] = []
    if s["shape"] == "poly" and not s["mask"]:
        s["mask"] = [list(p) for p in corners]
    return s


# ---------------------------------------------------------------------------------------------- files

def _quarantine(path):
    try:
        path = Path(path)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for n in range(100):
            bad = path.with_name(f"{path.name}.{stamp}{'-%d' % n if n else ''}.bad")
            if not bad.exists():
                os.replace(path, bad)
                return bad
    except OSError:
        pass
    return None


def _atomic_write(path, obj):
    tmp = None
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(obj, allow_nan=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if tmp is not None and tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def _backup(path):
    """Copy a file aside as <name>.<time>.bak before it is overwritten. True if there is nothing to lose
    (no file) or the copy was made."""
    try:
        path = Path(path)
        if not path.exists():
            return True
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for n in range(100):
            bak = path.with_name(f"{path.name}.{stamp}{'-%d' % n if n else ''}.bak")
            if not bak.exists():
                shutil.copy2(path, bak)
                return True
    except OSError:
        pass
    return False


def legacy_to_px(corners, output=DEFAULT_OUTPUT):
    """mapping.json corners (BL, BR, TR, TL in 0..1, v up) -> TL, TR, BR, BL in pixels (y down)."""
    w, h = output
    bl, br, tr, tl = [(float(c[0]) * w, (1.0 - float(c[1])) * h) for c in corners]
    return [list(tl), list(tr), list(br), list(bl)]


def px_to_legacy(corners, output=DEFAULT_OUTPUT):
    w, h = output
    tl, tr, br, bl = [[c[0] / w, 1.0 - c[1] / h] for c in corners]
    return [bl, br, tr, tl]


def _read_legacy(path, output):
    """mapping.json -> pixel corners, or None when missing / unusable. Never raises, never writes."""
    try:
        data = json.loads(Path(path).read_text())
        if isinstance(data, list) and len(data) == 4:
            pts = legacy_to_px(data, output)
            if all(math.isfinite(v) for p in pts for v in p) and is_convex(pts):
                return pts
    except (OSError, ValueError, TypeError, IndexError):
        pass
    return None


# ---------------------------------------------------------------------------------------------- the set

class SurfaceSet:
    """Ordered surfaces + selection + overlay + undo. All edits go through apply() (a validated command)."""

    def __init__(self, output=DEFAULT_OUTPUT, surfaces=None, sel=None):
        self.output = (int(output[0]), int(output[1]))
        self.surfaces = surfaces if surfaces else [new_surface(1, self.output)]
        self._renumber()
        self.sel = sel if any(s["id"] == sel for s in self.surfaces) else self.surfaces[0]["id"]
        self.overlay = dict(mode="off", grid=False)        # runtime only: never saved
        self.rev = 0                                       # bumps on every change (render / panel caches)
        self.dirty = False                                 # unsaved edits
        self.can_save = True
        self.status = "default"                            # how load() got here
        self.backup_first = False                          # copy the file on disk aside before the first save
        self.mask_dir = MASK_DIR                           # where shape "mask" label images live (Mapper: next to the file)
        self._undo, self._redo, self._gesture = [], [], None

    # ---- construction / persistence
    @classmethod
    def from_dict(cls, raw, output=None):
        """Raises ValueError when the document is not a usable surfaces file."""
        if not isinstance(raw, dict) or not isinstance(raw.get("surfaces"), list):
            raise ValueError("not a surfaces file")
        out = raw.get("output", output or DEFAULT_OUTPUT)
        if (not isinstance(out, (list, tuple)) or len(out) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or not 16 <= v <= 16384 for v in out)):
            raise ValueError("bad output size")
        surfs, seen, dropped = [], set(), max(0, len(raw["surfaces"]) - MAX_SURFACES)
        for item in raw["surfaces"][:MAX_SURFACES]:
            try:
                s = clean_surface(item, out)
            except ValueError:
                dropped += 1
                continue                                   # one broken surface does not take the others down
            while s["id"] in seen or s["id"] < 1:
                s["id"] = max(seen, default=0) + 1
            seen.add(s["id"])
            surfs.append(s)
        if not surfs:
            raise ValueError("no usable surface")
        surfs.sort(key=lambda s: s["order"])
        obj = cls(out, surfs, raw.get("sel"))
        obj.backup_first = dropped > 0                     # the file holds more than we could use: keep a copy of it
        return obj

    def to_dict(self):
        return dict(version=VERSION, output=list(self.output), sel=self.sel, surfaces=copy.deepcopy(self.surfaces))

    @classmethod
    def load(cls, path, legacy=None, output=DEFAULT_OUTPUT, writable=True):
        """surfaces.json -> SurfaceSet. Never raises.

        path None / missing   mapping.json (`legacy`) becomes surface #1, else one full-frame surface
        unreadable (OSError)  defaults in memory, the file is left alone and saving is DISABLED
        bad JSON / shape      moved aside to surfaces.json.<time>.bad (only when writable), then as missing
        """
        text, status, can_save, stuck = None, "default", True, False
        if path is not None:
            path = Path(path)
            try:
                text = path.read_text(encoding="utf-8") if path.exists() else None
            except OSError as ex:
                print(f"[map] could not read {path} ({ex}) -> defaults, saving is off")
                can_save, status = False, "unreadable"
        if text is not None:
            try:
                obj = cls.from_dict(json.loads(text))
                obj.status = "loaded"
                return obj
            except (ValueError, TypeError, KeyError, RecursionError) as ex:
                moved = _quarantine(path) if writable else None
                print(f"[map] {path} is not a valid surfaces file ({ex})" + (f" -> moved to {moved.name}" if moved else ""))
                status = "quarantined"
                stuck = writable and moved is None
        obj = cls(output)
        # the old corner pin is used whenever surfaces.json gave nothing, also when it could not be read
        # (saving is off then anyway): the show must not fall back to the full frame if a pin exists
        pts = _read_legacy(legacy, obj.output) if legacy is not None else None
        if pts is not None:
            obj.surfaces[0]["corners"] = pts
            status = "migrated" if status == "default" else status
        obj.status, obj.can_save, obj.backup_first = status, can_save, stuck
        return obj

    def save(self, path, legacy=None):
        """Write surfaces.json atomically (and mapping.json for surface #1, so the old corner pin stays in step)."""
        if not self.can_save or path is None:
            return False
        if self.backup_first:                              # lossy load, or a corrupt file that could not be moved aside
            if not _backup(path):
                return False                               # never overwrite the only copy
            self.backup_first = False
        ok = _atomic_write(path, self.to_dict())
        if ok:
            self.dirty = False
            if legacy is not None:
                _atomic_write(legacy, px_to_legacy(self.surfaces[0]["corners"], self.output))
        return ok

    # ---- lookups
    def _renumber(self):
        for i, s in enumerate(self.surfaces):
            s["order"] = i

    def get(self, sid):
        return next((s for s in self.surfaces if s["id"] == sid), None)

    def selected(self):
        return self.get(self.sel) or self.surfaces[0]

    def enabled(self):
        return [s for s in self.surfaces if s["enabled"] and s["opacity"] > 0.0]

    def polygon(self, s):
        """The visible outline of a surface in pixels: the quad, or the mask cut by the quad."""
        if s["shape"] == "poly" and len(s["mask"]) >= 3:
            poly = clip_poly(s["mask"], s["corners"])
            if len(poly) >= 3 and abs(signed_area(poly)) >= MIN_CROSS:
                return poly[:MAX_POLY]
            return []
        return [tuple(p) for p in s["corners"]]

    def trivial(self):
        """True when the output is exactly the old single corner pin (the engine then uses its old path)."""
        if self.overlay["mode"] != "off":
            return False
        en = [s for s in self.surfaces if s["enabled"]]
        if len(en) != 1:
            return False
        s = en[0]
        return (s["shape"] == "quad" and s["source"]["mode"] == "full" and s["content"] == "main" and s["opacity"] == 1.0
                and s["brightness"] == 1.0 and s["glow"] == 0.0 and s["feather"] == 0.0)

    def legacy_corners(self, sid=None):
        """Corners of one surface (default: the one the old path draws / the selection) as BL, BR, TR, TL in 0..1."""
        s = self.get(sid) if sid is not None else None
        if s is None:
            en = [x for x in self.surfaces if x["enabled"]]
            s = en[0] if self.trivial() and en else self.selected()
        return px_to_legacy(s["corners"], self.output)

    def set_legacy_corners(self, corners, sid=None, g="legacy"):
        s = self.get(sid) if sid is not None else None
        if s is None:
            en = [x for x in self.surfaces if x["enabled"]]
            s = en[0] if self.trivial() and en else self.selected()
        return self.apply(dict(op="corners", id=s["id"], pts=legacy_to_px(corners, self.output), g=g))

    def span_box(self):
        """(x0, y0, w, h) in pixels of the virtual canvas all span-mode surfaces share: the smallest box of the
        output's aspect ratio that covers every one of them (so the picture is never stretched)."""
        # every span-mode surface counts, on or off: switching one face off mid-show must not rescale or shift
        # the picture on the others
        pts = [p for s in self.surfaces if s["source"]["mode"] == "span" for p in self.polygon(s)]
        if not pts:
            return (0.0, 0.0, float(self.output[0]), float(self.output[1]))
        x0, y0, x1, y1 = bbox(pts)
        w, h, asp = max(x1 - x0, 1.0), max(y1 - y0, 1.0), self.output[0] / self.output[1]
        cw, ch = (w, w / asp) if w / h >= asp else (h * asp, h)
        return ((x0 + x1 - cw) / 2.0, (y0 + y1 - ch) / 2.0, cw, ch)

    def source_matrix(self, s):
        """3x3 M: projector pixel (x, y down) -> source (u, v) with (0, 0) the TOP-left of the rendered frame."""
        mode = s["source"]["mode"]
        if s["shape"] == "mask" and mode == "full":      # a stencil: the picture as it falls on the projector
            return [[1.0 / self.output[0], 0.0, 0.0], [0.0, 1.0 / self.output[1], 0.0], [0.0, 0.0, 1.0]]
        if mode == "span":
            x0, y0, cw, ch = self.span_box()
            return [[1.0 / cw, 0.0, -x0 / cw], [0.0, 1.0 / ch, -y0 / ch], [0.0, 0.0, 1.0]]
        H = homography(s["corners"], [(0, 0), (1, 0), (1, 1), (0, 1)])
        if mode == "rect":
            x, y, w, h = s["source"]["rect"]
            return matmul([[w, 0.0, x], [0.0, h, y], [0.0, 0.0, 1.0]], H)
        return H

    def local_matrix(self, s):
        """Projector pixel -> the surface's own 0..1 square (TL origin), whatever its source mode."""
        return homography(s["corners"], [(0, 0), (1, 0), (1, 1), (0, 1)])

    def pick(self, x, y):
        """Topmost enabled surface under a projector pixel -> (surface, (u, v) in the source frame) or None."""
        for s in reversed(self.enabled()):
            poly = self.polygon(s)
            if len(poly) >= 3 and point_in_poly((x, y), poly):
                try:
                    return s, apply_h(self.source_matrix(s), (x, y))
                except ValueError:
                    return None
        return None

    def state(self):
        """What the panel needs: the document plus runtime flags."""
        d = self.to_dict()
        d.update(rev=self.rev, overlay=dict(self.overlay), undo=len(self._undo), redo=len(self._redo),
                 dirty=self.dirty, can_save=self.can_save, trivial=self.trivial())
        return d

    # ---- undo
    def _snap(self):
        return (copy.deepcopy(self.surfaces), self.sel)

    def _push(self, gesture):
        """Remember the state before an edit. Edits that share a gesture key (one drag, one key-repeat run,
        one typed field) collapse into a single undo step."""
        if gesture is not None and gesture == self._gesture and self._undo:
            return
        self._undo.append(self._snap())
        del self._undo[:-UNDO_DEPTH]
        self._redo.clear()
        self._gesture = gesture

    def _changed(self):
        self.rev += 1
        self.dirty = True

    def undo(self):
        if not self._undo:
            return False
        self._redo.append(self._snap())
        self.surfaces, self.sel = self._undo.pop()
        self._gesture = None
        self._changed()
        return True

    def redo(self):
        if not self._redo:
            return False
        self._undo.append(self._snap())
        self.surfaces, self.sel = self._redo.pop()
        self._gesture = None
        self._changed()
        return True

    # ---- edits
    def _clamp(self, p):
        w, h = self.output
        return [min((1 + OVERSCAN) * w, max(-OVERSCAN * w, float(p[0]))), min((1 + OVERSCAN) * h, max(-OVERSCAN * h, float(p[1])))]

    def apply(self, c):
        """Apply one validated command (see validate_cmd). Returns True if anything changed. Never raises:
        an edit that would fold a quad, cross a mask or leave nothing on screen is refused."""
        try:
            return bool(self._apply(c))
        except (KeyError, ValueError, TypeError, IndexError):
            return False

    def _apply(self, c):
        op = c.get("op")
        if op == "undo":
            return self.undo()
        if op == "redo":
            return self.redo()
        if op == "overlay":
            new = dict(self.overlay)
            if c.get("mode") in OVERLAY_MODES:
                new["mode"] = c["mode"]
            if isinstance(c.get("grid"), bool):
                new["grid"] = c["grid"]
            if new == self.overlay:
                return False
            self.overlay = new
            self.rev += 1
            return True
        if op == "select":
            if self.get(c["id"]) is None or self.sel == c["id"]:
                return False
            self.sel = c["id"]
            self.rev += 1
            return True
        if op not in EDIT_OPS:
            return False
        sid = c.get("id")
        gesture = None if c.get("g") is None else (op, sid, c.get("i"), c.get("m"), c["g"])
        before = self._snap()
        if not self._edit(op, c):
            self.surfaces, self.sel = before
            return False
        after, (self.surfaces, self.sel) = self._snap(), before
        if after[0] == before[0] and after[1] == before[1]:
            return False
        self._push(gesture)
        self.surfaces, self.sel = after
        self._renumber()
        self._changed()
        return True

    def _edit(self, op, c):
        W, H = self.output
        if op == "add":
            if len(self.surfaces) >= MAX_SURFACES:
                return False
            sid = max(s["id"] for s in self.surfaces) + 1
            k = (len(self.surfaces) % 6) * 0.04              # new surfaces cascade so they never hide each other
            x0, y0, w, h = (0.30 + k) * W, (0.28 + k) * H, 0.34 * W, 0.34 * H
            s = new_surface(sid, self.output, c.get("name"), [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])
            self.surfaces.append(s)
            self.sel = sid
            return True
        if op == "replace":
            surfs, seen = [], set()
            for raw in c["surfaces"]:
                s = clean_surface(raw, self.output, strict=True)
                while s["id"] in seen or s["id"] < 1:
                    s["id"] = max(seen, default=0) + 1
                seen.add(s["id"])
                surfs.append(s)
            self.surfaces = surfs
            self.sel = surfs[0]["id"]
            return True
        if op == "looks":                                # every surface its own moving source, hue and drift
            n = len(LOOK_ROTA)
            for k, t in enumerate(self.surfaces):
                t["content"] = "layers"
                t["layers"] = [dict(LAYER_DEFAULT, src=LOOK_ROTA[k % n], scroll=0.06 + 0.03 * (k % 3),
                                    dir=float((k * 137) % 360 - 180), hue=float(((k // n) * 67) % 360 - 180) if k >= n else 0.0)]
            return True
        s = self.get(c["id"])
        if s is None:
            return False
        if op == "dup":
            if len(self.surfaces) >= MAX_SURFACES:
                return False
            d = copy.deepcopy(s)
            d["id"] = max(x["id"] for x in self.surfaces) + 1
            d["name"] = (s["name"] + " copy")[:MAX_NAME]
            d["color"] = list(COLORS[(d["id"] - 1) % len(COLORS)])
            off = 24.0
            d["corners"] = [self._clamp((p[0] + off, p[1] + off)) for p in d["corners"]]
            d["mask"] = [self._clamp((p[0] + off, p[1] + off)) for p in d["mask"]]
            self.surfaces.insert(self.surfaces.index(s) + 1, d)
            self.sel = d["id"]
            return True
        if op == "del":
            if len(self.surfaces) <= 1:
                return False
            i = self.surfaces.index(s)
            self.surfaces.pop(i)
            if self.sel == s["id"]:
                self.sel = self.surfaces[min(i, len(self.surfaces) - 1)]["id"]
            return True
        if op == "rename":
            s["name"] = c["name"]
            return True
        if op == "move":
            self.surfaces.remove(s)
            self.surfaces.insert(min(max(int(c["to"]), 0), len(self.surfaces)), s)
            return True
        if op in ("set", "set_layers") and c.get("group_all") and s.get("group"):
            ok = False                                   # one setting for the whole group: one undo step
            for t in [x for x in self.surfaces if x.get("group") == s["group"]]:
                ok = self._edit(op, dict(c, id=t["id"], group_all=False)) or ok
            return ok
        if op == "set_layers":
            s["layers"] = copy.deepcopy(c["layers"])
            if "content" in c:
                s["content"] = c["content"]
            return True
        if op == "set":
            if c.get("shape") == "mask" and not s.get("maskfile"):
                return False                             # only AUTO MAP makes mask surfaces (it writes the label image)
            if c.get("shape") in ("quad", "poly") and s["shape"] == "mask":
                s.pop("maskfile", None)
                s.pop("label", None)
            for k in ("enabled", "opacity", "brightness", "glow", "feather", "content", "shape", "color", "layers"):
                if k in c:
                    s[k] = copy.deepcopy(c[k])
            if "src" in c:
                s["source"]["mode"] = c["src"]
            if "rect" in c:
                s["source"]["rect"] = list(c["rect"])
            if s["shape"] == "poly" and len(s["mask"]) < 3:
                s["mask"] = [list(p) for p in s["corners"]]
            if c.get("content") == "glow" and s["glow"] <= 0.0:
                s["glow"] = 1.0                              # "outline only" with the outline at 0 would be a black face
            return True
        if op in ("corner", "nudge"):
            pts = s["mask"] if c.get("m") else s["corners"]
            i = c["i"]
            if i >= len(pts):
                return False
            new = [c["x"], c["y"]] if op == "corner" else [pts[i][0] + c["dx"], pts[i][1] + c["dy"]]
            trial = [list(p) for p in pts]
            trial[i] = self._clamp(new)
            return self._put(s, trial, bool(c.get("m")))
        if op == "corners":
            return self._put(s, [self._clamp(p) for p in c["pts"]], False)
        if op == "mask":
            ok = self._put(s, [self._clamp(p) for p in c["pts"]], True)
            if ok:
                s["shape"] = "poly"
            return ok
        if op == "shift":
            dx, dy = c["dx"], c["dy"]
            for key in ("corners", "mask"):                  # the whole surface moves rigidly, or not at all
                moved = [[p[0] + dx, p[1] + dy] for p in s[key]]
                if any(self._clamp(p) != p for p in moved):
                    return False
                s[key] = moved
            return True
        if op == "mask_add":
            if len(s["mask"]) < 3:
                s["mask"] = [list(p) for p in s["corners"]]
            if len(s["mask"]) >= MAX_MASK:
                return False
            lens = edge_lengths(s["mask"])
            i = c.get("i")
            i = lens.index(max(lens)) if i is None or i >= len(lens) else i     # default: split the longest edge
            a, b = s["mask"][i], s["mask"][(i + 1) % len(s["mask"])]
            s["mask"].insert(i + 1, [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0])
            s["shape"] = "poly"
            return True
        if op == "mask_del":
            if len(s["mask"]) <= 3 or c["i"] >= len(s["mask"]):
                return False
            trial = [p for k, p in enumerate(s["mask"]) if k != c["i"]]
            return self._put(s, trial, True)
        if op == "square":
            return self._put(s, [self._clamp(p) for p in square_up(s["corners"], bool(c.get("level")))], False)
        if op == "angle":
            return self._put(s, [self._clamp(p) for p in with_angle(s["corners"], c["i"], c["deg"])], False)
        if op == "edge":
            return self._put(s, [self._clamp(p) for p in with_edge(s["corners"], c["i"], c["len"])], False)
        if op == "rotate":                                    # absolute tilt
            turned = rotate_pts(s["corners"], c["deg"] - rotation(s["corners"]))
            if any(self._clamp(p) != [float(p[0]), float(p[1])] for p in turned):
                return False
            return self._put(s, turned, False)
        if op == "reset":
            s["corners"], s["mask"], s["shape"] = full_quad(self.output), [], "quad"
            return True
        return False

    def _put(self, s, pts, mask):
        if mask:
            if not 3 <= len(pts) <= MAX_MASK or not is_simple(pts):
                return False
            s["mask"] = pts
        else:
            if len(pts) != 4 or not is_convex(pts):
                return False
            s["corners"] = pts
        return True

    # ---- render data
    def render_items(self, fb_w, fb_h):
        """Everything the renderer needs per enabled surface, in draw order, in FRAMEBUFFER pixels with y UP
        (gl_FragCoord): polygon, scissor box, matrices (row-major 9-tuples: pixel -> source uv with v up, and
        pixel -> the surface's own 0..1 square), look values. Pure maths, so it is tested without a GPU."""
        if fb_w <= 0 or fb_h <= 0:
            return []                                      # minimised window: nothing to draw, nothing to divide by
        sx, sy = fb_w / self.output[0], fb_h / self.output[1]
        to_px = [[1.0 / sx, 0.0, 0.0], [0.0, -1.0 / sy, fb_h / sy], [0.0, 0.0, 1.0]]     # gl pixel -> output pixel (y down)
        flip = [[1.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 1.0]]                      # v down -> v up
        items = []
        for s in self.enabled():
            poly = self.polygon(s)
            if len(poly) < 3:
                continue
            try:
                M = matmul(flip, matmul(self.source_matrix(s), to_px))
                L = matmul(flip, matmul(self.local_matrix(s), to_px))
            except ValueError:
                continue
            gl = [(x * sx, fb_h - y * sy) for x, y in poly]
            x0, y0, x1, y1 = bbox(gl)
            box = (max(0, int(math.floor(x0)) - 1), max(0, int(math.floor(y0)) - 1),
                   min(fb_w, int(math.ceil(x1)) + 1), min(fb_h, int(math.ceil(y1)) + 1))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue                                     # wholly off the projector
            area = abs(signed_area(gl))
            glow = s["glow"] if s["content"] != "glow" or s["glow"] > 0 else 1.0
            lay = []
            if s["content"] == "layers":
                for ly in s.get("layers", ()):
                    if ly["on"] and ly["opacity"] > 0.0:
                        a = math.radians(ly["dir"])
                        lay.append((ly["src"], ly["opacity"], BLENDS.index(ly["blend"]), ly["scale"], ly["scroll"],
                                    math.cos(a), -math.sin(a), math.radians(ly["hue"])))
            extra = dict(layers=tuple(lay))
            if s["shape"] == "mask":
                extra.update(maskfile=s["maskfile"], label=s["label"])
                glow = 0.0                                 # the edge glow follows polygons; a mask has none
            items.append(dict(extra, id=s["id"], name=s["name"], poly=gl, n=len(gl), box=box,
                              M=tuple(v for row in M for v in row), L=tuple(v for row in L for v in row),
                              content=CONTENTS.index(s["content"]), color=tuple(s["color"]), opacity=s["opacity"],
                              brightness=s["brightness"], glow=glow, feather=s["feather"] * (sx + sy) / 2.0,
                              perimeter=sum(edge_lengths(gl)), area=area, center=centroid(gl),
                              line=min(5.0, max(1.5, 0.012 * math.sqrt(area))), selected=s["id"] == self.sel))
        return items


# ---------------------------------------------------------------------------------------------- engine side

class OutputGate:
    """Which output path the engine takes this frame, so one bad moment never costs the mapping for the night.

      "skip"   the framebuffer is 0 x 0 (window minimised / mid re-plug): draw nothing, count nothing
      "multi"  draw every surface
      "pin"    the old single corner pin: either that is all there is (plain mode "pin"), or the surface renderer
               failed MAX_FAILS frames in a row (mode "fallback"). The pin shader lights only the inside of that
               one quad, so the rest of the projector stays black. From a fallback the surface path is tried
               again every RETRY_S, for as long as it takes.
    """
    MAX_FAILS = 5
    RETRY_S = 2.0

    def __init__(self):
        self.fails, self.retry_at, self.mode, self.error = 0, 0.0, "pin", ""

    def plan(self, fb, trivial, have_renderer, now):
        if fb[0] <= 0 or fb[1] <= 0:
            return "skip"
        if trivial or not have_renderer:
            self.mode = "pin"
            return "pin"
        if self.fails >= self.MAX_FAILS and now < self.retry_at:
            self.mode = "fallback"
            return "pin"
        return "multi"

    def ok(self):
        self.fails, self.mode, self.error = 0, "multi", ""

    def failed(self, now, error=""):
        """Record a failed surface draw. Returns the line to log."""
        self.fails += 1
        self.error = str(error)[:200]
        if self.fails >= self.MAX_FAILS:
            self.fails, self.retry_at, self.mode = self.MAX_FAILS, now + self.RETRY_S, "fallback"
            return f"[map] surface output failed: corner pin only, trying again in {self.RETRY_S:.0f} s ({self.error})"
        return f"[map] surface output failed ({self.fails}/{self.MAX_FAILS}), corner pin this frame ({self.error})"


class Mapper:
    """What the render engine holds: the set, where it lives on disk, autosave, and the bridge that keeps the
    old corner-pin commands / mouse calibration working on one surface."""
    AUTOSAVE_S = 1.5                                       # quiet time after the last edit before writing
    RETRY_MAX_S = 30.0                                     # a failed write is retried: 1.5, 3, 6 ... 30 s apart

    def __init__(self, path=None, legacy=None, output=DEFAULT_OUTPUT, writable=True):
        self.path = Path(path) if path is not None else None
        self.legacy = Path(legacy) if legacy is not None else None
        self.writable = bool(writable) and self.path is not None
        self.set = SurfaceSet.load(self.path, self.legacy, output, writable=self.writable)
        if self.path is not None:
            self.set.mask_dir = self.path.parent / "masks"
        self._edit_t = None                                # when the pending write became due-able
        self._wait = self.AUTOSAVE_S                       # current delay (grows while writes fail)
        self.save_failed = False
        self.gate = OutputGate()

    def state(self):
        """The panel's view: the set plus what the engine is doing with it."""
        d = self.set.state()
        d.update(out=self.gate.mode, save_failed=self.save_failed)
        return d

    def command(self, c):
        """One validated {"cmd": "surf"} command from the panel. Returns True if it did something."""
        if c.get("op") == "save":
            return self.save()
        changed = self.set.apply(c)
        if changed and self.set.dirty:
            self._touch()
        return changed

    def _touch(self):
        self._edit_t = time.monotonic()
        if not self.save_failed:
            self._wait = self.AUTOSAVE_S

    def save(self):
        """Write now. A failed write (Windows: a virus scanner or the indexer holding the file for a moment)
        stays pending and is retried by tick() with a growing delay; the edits are never dropped."""
        if not self.writable or not self.set.can_save:     # headless, or the file was unreadable at start: by design
            self._edit_t = None
            return False
        ok = self.set.save(self.path, self.legacy)
        if ok:
            if self.save_failed:
                print(f"[map] saving works again -> {self.path.name}")
            self._edit_t, self._wait, self.save_failed = None, self.AUTOSAVE_S, False
            print(f"[map] saved {len(self.set.surfaces)} surface(s) -> {self.path.name}")
        else:
            self._wait = self.AUTOSAVE_S if not self.save_failed else min(self.RETRY_MAX_S, self._wait * 2.0)
            self._edit_t, self.save_failed = time.monotonic(), True
            print(f"[map] could NOT save {self.path}: trying again in {self._wait:.1f} s")
        return ok

    def tick(self):
        """Call every frame: writes the file once edits have been quiet for AUTOSAVE_S (longer after a failure)."""
        if self._edit_t is not None and time.monotonic() - self._edit_t >= self._wait:
            self.save()

    def flush(self, tries=4, pause=0.25):
        """At exit: anything unsaved is written, with a few quick retries. Returns True if nothing is left unsaved."""
        if not self.writable or not self.set.can_save or not (self.set.dirty or self._edit_t is not None):
            return True
        for k in range(max(1, tries)):
            if self.save():
                return True
            if k + 1 < tries:
                time.sleep(pause)
        print(f"[map] UNSAVED mapping edits could not be written to {self.path}")
        return False

    # ---- the old corner pin (BL, BR, TR, TL in 0..1, v up) is a view of one surface
    def target_id(self):
        s = self.set
        en = [x for x in s.surfaces if x["enabled"]]
        return en[0]["id"] if s.trivial() and en else s.selected()["id"]

    def corners(self):
        return self.set.legacy_corners(self.target_id())

    def push_corners(self, corners, sid=None, g="legacy"):
        sid = self.target_id() if sid is None else sid
        if self.set.set_legacy_corners(corners, sid, g=g):
            self._touch()
            return True
        return False

    def nearest_corner(self, u, v):
        """Projector-window calibration: the enabled surface corner nearest a click (u, v in 0..1, v up)
        -> (surface id, legacy corner index), and that surface becomes the selection."""
        x, y = u * self.set.output[0], (1.0 - v) * self.set.output[1]
        best = None
        for s in self.set.surfaces:
            if not s["enabled"]:
                continue
            for i, p in enumerate(s["corners"]):
                d = (p[0] - x) ** 2 + (p[1] - y) ** 2
                if best is None or d < best[0]:
                    best = (d, s["id"], 3 - i)
        if best is None:
            return self.target_id(), 0
        if not self.set.trivial():
            self.set.apply(dict(op="select", id=best[1]))
        return best[1], best[2]

    def ripple_uv(self, u, v):
        """Where a click at window (u, v) lands in the picture (0..1, v up), or None if it hit no surface."""
        hit = self.set.pick(u * self.set.output[0], (1.0 - v) * self.set.output[1])
        if hit is None:
            return None
        su, sv = hit[1]
        if not (math.isfinite(su) and math.isfinite(sv) and 0.0 <= su <= 1.0 and 0.0 <= sv <= 1.0):
            return None
        return su, 1.0 - sv
