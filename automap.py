"""AUTO MAP: camera-assisted projection mapping (research/automap_spec.md), adapted to ONE system.

No second app or window. The engine draws the calibration patterns over its own output (AutoMapOverlay, one fullscreen
draw after the mapping pass); camera_worker.py captures stills on request (GET /still); this module does the maths;
the results become ordinary editable surfaces (surfaces.py "replace" / "corners" commands, so the MAPPING tab's undo
covers them). Classical CV only; the optional AI assist is a validated entry point (op "ai_polys"), see HANDOFF.md.

Stages
  1 calibrate   ArUco grid (planar homography, fast) + Gray-code structured light (dense camera -> projector map, for
                non-planar sculptures). Reprojection error reported; low-quality results are refused with a reason.
  2 detect      projector-lit difference image (grey minus black); faces = lit regions split by camera edges; click a
                face in the panel -> region -> quad / polygon fit; candidates listed with a confidence.
  3 verify      project thin outlines of each surface, find them in the camera, move the corners until the projected
                outline sits on the physical edge (closed loop through the calibration; a few iterations).

Coordinates: camera px (x right, y down, the worker's still size, 640x360); projector px (x right, y down, the
output size, as in surfaces.json).

Photosensitive safety: patterns are static, at most PATTERN_LEVEL brightness, each held >= HOLD_S (<= 2 per second);
the panel warns that nobody should be looking at the projector while it calibrates.
"""
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CALIB_DIR = HERE / "calibration"
PATTERN_LEVEL = 170            # white of the calibration patterns (of 255): bright enough, not blinding
LEVEL_MAX = 190                # never brighter, whatever the camera needs (patterns are static, but full-frame)
HOLD_S = 0.5                   # every pattern stays at least this long (<= 2 changes per second)
STRIPE = 4                     # Gray code resolution, projector px per code step (camera can't resolve finer)
ARUCO_GRID = (6, 4)            # markers across x down
MIN_MARKERS = 8
MAX_RMS = 4.0                  # projector px: calibration refused above this
MIN_CONTRAST = 18              # grey levels between a bit pattern and its inverse
MIN_FACE_FRAC = 0.004          # a face must cover at least this fraction of the camera frame
REFINE_TOL = 0.8               # camera px: stop refining below this corner error
AI_MAX_NUDGE = 6.0             # camera px: the most an AI verify correction may move one vertex per press
AI_MAX_POLYS = 12


# per-camera profiles (calibration/<profile>/pose<n>/): how to capture with that kind of camera
PROFILES = {
    "builtin": dict(label="Built-in webcam", n_avg=3, settle=1.6, level=190),   # auto exposure, noisy, often mirrored
    "usb": dict(label="USB webcam", n_avg=1, settle=1.0, level=PATTERN_LEVEL),
}
BUILTIN_NAME = r"(integrated|built-?in|facetime|internal|hd (web)?cam|front)"


class CalibError(ValueError):
    """A calibration / detection result that is refused; str() is the reason shown in the panel."""


# ------------------------------------------------------------------ homographies

def fit_h(src, dst, ransac=3.0):
    """H (3x3) with dst ~ H src, RANSAC; returns (H, inlier mask, rms of inliers in dst units)."""
    import cv2
    src, dst = np.asarray(src, np.float64).reshape(-1, 2), np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 4:
        raise CalibError("not enough points for a homography")
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC if ransac else 0, ransac or 3.0)
    if H is None:
        raise CalibError("the points do not fit a plane")
    inl = inl.ravel().astype(bool) if inl is not None else np.ones(len(src), bool)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise CalibError("the points do not fit a plane")
    if ransac and inl.sum() < max(4, min(8, len(src)), len(src) // 2):   # RANSAC fitted a lucky few: not a calibration
        raise CalibError(f"only {int(inl.sum())} of {len(src)} points agree: is the camera still and the view sharp?")
    err = np.linalg.norm(apply_h(H, src[inl]) - dst[inl], axis=1)
    rms = float(np.sqrt((err ** 2).mean())) if len(err) else float("inf")
    if not math.isfinite(rms):
        raise CalibError("the points do not fit a plane")
    return H, inl, rms


def apply_h(H, pts):
    """Points through a homography. Points on / past the vanishing line (w <= 0) come back as NaN, never as a
    wrapped-around position; callers drop or refuse non-finite results."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ np.asarray(H, np.float64).T
    w = q[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(w > 1e-12, q[:, :2] / np.where(w > 1e-12, w, 1.0), np.nan)


# ------------------------------------------------------------------ patterns (projector images, uint8)

def gray_nbits(n, stripe=STRIPE):
    return max(1, int(math.ceil(math.log2(max(2, int(math.ceil(n / stripe)))))))


def gray_sequence(w, h, stripe=STRIPE, level=PATTERN_LEVEL):
    """[(key, image)]: white, black, then for x and y every Gray-code bit (MSB first) and its inverse.
    stripe = projector px per code step (wider for a low-res / far camera), level = white (dim room: brighter)."""
    level = int(min(max(level, 60), LEVEL_MAX))
    seq = [("white", np.full((h, w), level, np.uint8)), ("black", np.zeros((h, w), np.uint8))]
    for axis, n in (("x", w), ("y", h)):
        nb = gray_nbits(n, stripe)
        idx = (np.arange(n) // stripe)
        g = idx ^ (idx >> 1)
        for b in range(nb - 1, -1, -1):
            bit = ((g >> b) & 1).astype(np.uint8) * level
            img = np.broadcast_to(bit[None, :], (h, w)) if axis == "x" else np.broadcast_to(bit[:, None], (h, w))
            seq.append((f"{axis}{b}", np.ascontiguousarray(img)))
            seq.append((f"{axis}{b}i", np.ascontiguousarray(level - img)))
    return seq


def gray_decode(caps, w, h, min_contrast=MIN_CONTRAST, stripe=STRIPE):
    """caps {key: camera image (float or uint8)} from gray_sequence -> (proj_x, proj_y, valid), camera sized."""
    white, black = caps["white"].astype(np.float32), caps["black"].astype(np.float32)
    valid = (white - black) > 2 * min_contrast
    out = []
    for axis, n in (("x", w), ("y", h)):
        nb = gray_nbits(n, stripe)
        g = np.zeros(white.shape, np.int32)
        for b in range(nb - 1, -1, -1):
            pos, neg = caps[f"{axis}{b}"].astype(np.float32), caps[f"{axis}{b}i"].astype(np.float32)
            valid &= np.abs(pos - neg) > min_contrast
            g |= (pos > neg).astype(np.int32) << b
        code = g.copy()                                  # Gray -> binary
        s = g >> 1
        while s.any():
            code ^= s
            s >>= 1
        coord = code * stripe + stripe * 0.5 - 0.5
        valid &= coord < n
        out.append(coord.astype(np.float32))
    return out[0], out[1], valid


def aruco_board(w, h, grid=ARUCO_GRID, level=PATTERN_LEVEL):
    """Projector image with a grid of 4x4 ArUco markers + {id: 4 corners in projector px (TL, TR, BR, BL)}."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    nx, ny = grid
    cell = min(w / nx, h / ny)
    size = int(cell * 0.62)
    img = np.zeros((h, w), np.uint8)
    corners = {}
    ox, oy = (w - cell * nx) / 2, (h - cell * ny) / 2
    for j in range(ny):
        for i in range(nx):
            mid = j * nx + i
            x0 = int(round(ox + (i + 0.5) * cell - size / 2))
            y0 = int(round(oy + (j + 0.5) * cell - size / 2))
            q = int(size * 0.12)                         # white quiet zone around the marker
            img[y0 - q:y0 + size + q, x0 - q:x0 + size + q] = level
            m = cv2.aruco.generateImageMarker(d, mid, size)
            img[y0:y0 + size, x0:x0 + size] = (m > 127).astype(np.uint8) * level
            corners[mid] = np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]], np.float64) - 0.5
    return img, corners


def detect_aruco(gray):
    """{id: 4 camera corners}"""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    c, ids, _ = cv2.aruco.ArucoDetector(d, p).detectMarkers(np.asarray(gray, np.uint8))
    return {} if ids is None else {int(i): cc.reshape(4, 2).astype(np.float64) for i, cc in zip(ids.ravel(), c)}


# ------------------------------------------------------------------ calibration result

# ---- wide-angle lens: one radial model (k1, k2) about the image centre, fitted from the calibration's own points
def lens_K(size, f_scale=0.85):
    w, h = size
    f = f_scale * w
    return np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1]], np.float64)


def undistort_pts(pts, K, dist):
    import cv2
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K, np.array([dist[0], dist[1], 0, 0, 0], np.float64), P=K).reshape(-1, 2)


def distort_pts(pts, K, dist):
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    x, y = (p[:, 0] - cx) / f, (p[:, 1] - cy) / f
    r2 = x * x + y * y
    s = 1 + dist[0] * r2 + dist[1] * r2 * r2
    return np.c_[x * s * f + cx, y * s * f + cy]


def fit_lens(cam, proj, size, max_pts=600):
    """Radial distortion that makes the camera->projector correspondences most planar. -> (dist (k1, k2), rms before,
    rms after); (0, 0) unless it clearly helps (a flat wall seen through a wide-angle lens bends near the edges)."""
    import cv2
    cam, proj = np.asarray(cam, np.float64).reshape(-1, 2), np.asarray(proj, np.float64).reshape(-1, 2)
    if len(cam) > max_pts:
        k = np.random.default_rng(0).choice(len(cam), max_pts, replace=False)
        cam, proj = cam[k], proj[k]
    K = lens_K(size)

    def cost(k1, k2):
        u = undistort_pts(cam, K, (k1, k2)) if (k1 or k2) else cam
        H, _ = cv2.findHomography(u, proj, 0)
        if H is None:
            return float("inf")
        return float(np.sqrt(((apply_h(H, u) - proj) ** 2).sum(axis=1).mean()))
    c0 = cost(0.0, 0.0)
    best = (c0, 0.0, 0.0)
    for k1 in np.arange(-0.5, 0.501, 0.05):
        for k2 in (-0.2, -0.1, 0.0, 0.1, 0.2):
            c = cost(k1, k2)
            if c < best[0]:
                best = (c, k1, k2)
    step1, step2 = 0.025, 0.05
    for _ in range(5):
        c, k1, k2 = best
        for a, b in ((step1, 0), (-step1, 0), (0, step2), (0, -step2)):
            cc = cost(k1 + a, k2 + b)
            if cc < best[0]:
                best = (cc, k1 + a, k2 + b)
        step1, step2 = step1 / 2, step2 / 2
    c, k1, k2 = best
    if c0 - c > max(0.25, 0.15 * c0):
        return (float(k1), float(k2)), c0, c
    return (0.0, 0.0), c0, c0


class Calibration:
    """camera px -> projector px. H always (planar fit, in lens-corrected camera px); dense Gray-code map when it was
    captured; lens (k1, k2) for wide-angle webcams; which part of the projection the camera saw (coverage 0..1)."""

    def __init__(self, H, rms, cam_size, proj_size, method, dense=None, seen=None, dist=(0.0, 0.0), coverage=1.0,
                 warning="", mirrored=False):
        self.H = np.asarray(H, np.float64)
        if self.H.shape != (3, 3) or not np.isfinite(self.H).all() or abs(np.linalg.det(self.H)) < 1e-12 \
                or not math.isfinite(float(rms)):
            raise CalibError("calibration matrix is not usable (degenerate or not finite)")
        self.Hinv = np.linalg.inv(self.H)
        self.rms, self.cam_size, self.proj_size, self.method = float(rms), tuple(cam_size), tuple(proj_size), method
        self.dense = dense                               # (proj_x, proj_y, valid) camera-sized, or None
        self.seen = seen or ""
        self.dist = (float(dist[0]), float(dist[1]))
        self.K = lens_K(self.cam_size)
        self.coverage, self.warning, self.mirrored = float(coverage), warning, bool(mirrored)

    def _lens(self):
        return self.dist[0] != 0.0 or self.dist[1] != 0.0

    def cam_to_proj(self, pts, dense=True):
        """Camera px -> projector px. With a dense map, each point uses the median of valid correspondences within 3
        camera px (non-planar surfaces); the homography (after the lens correction) where there are none."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        out = apply_h(self.H, undistort_pts(pts, self.K, self.dist) if self._lens() else pts)
        if dense and self.dense is not None:
            px, py, ok = self.dense
            hh, ww = ok.shape
            for k, (x, y) in enumerate(pts):
                x0, x1, y0, y1 = int(max(x - 3, 0)), int(min(x + 4, ww)), int(max(y - 3, 0)), int(min(y + 4, hh))
                if x1 <= x0 or y1 <= y0:
                    continue
                m = ok[y0:y1, x0:x1]
                if m.sum() >= 5:
                    out[k] = (float(np.median(px[y0:y1, x0:x1][m])), float(np.median(py[y0:y1, x0:x1][m])))
        return out

    def proj_to_cam(self, pts):
        q = apply_h(self.Hinv, pts)
        return distort_pts(q, self.K, self.dist) if self._lens() else q

    def summary(self):
        return dict(method=self.method, rms=round(self.rms, 2), seen=self.seen, dense=self.dense is not None,
                    cam=list(self.cam_size), proj=list(self.proj_size), lens=[round(self.dist[0], 3), round(self.dist[1], 3)],
                    coverage=round(self.coverage, 3), warning=self.warning, mirrored=self.mirrored)

    def save(self, folder=CALIB_DIR):
        """Numbers only (homography + correspondence map), never camera pictures."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        meta = dict(self.summary(), H=self.H.tolist(), lens=list(self.dist), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        tmp = folder / "calibration.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        os.replace(tmp, folder / "calibration.json")
        if self.dense is not None:
            px, py, ok = self.dense
            np.savez_compressed(folder / "graycode_map.npz", px=px.astype(np.float32), py=py.astype(np.float32), ok=ok)

    @classmethod
    def load(cls, folder=CALIB_DIR):
        folder = Path(folder)
        try:
            meta = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
            dense = None
            if meta.get("dense") and (folder / "graycode_map.npz").is_file():
                z = np.load(folder / "graycode_map.npz")
                dense = (z["px"], z["py"], z["ok"].astype(bool))
            return cls(meta["H"], meta["rms"], meta["cam"], meta["proj"], meta["method"], dense, meta.get("seen"),
                       tuple(meta.get("lens") or (0.0, 0.0)), meta.get("coverage", 1.0), meta.get("warning", ""),
                       meta.get("mirrored", False))
        except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError):   # CalibError is a ValueError
            return None


MIN_PARTIAL = 4                # markers: below this nothing is calibrated; MIN_PARTIAL..MIN_MARKERS = partial, with a warning


def proj_coverage(proj_pts, proj_size):
    """Fraction of the projector frame inside the hull of the points the camera saw."""
    import cv2
    W, H = proj_size
    p = np.asarray(proj_pts, np.float32).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    hull = cv2.convexHull(p)
    return float(min(max(cv2.contourArea(hull) / float(W * H), 0.0), 1.0))


def _with_lens(cam, proj, cam_size, ransac):
    """Lens fit on the RANSAC inliers, then the homography in corrected coordinates -> (H, dist, rms, inlier mask)."""
    H0, inl, rms0 = fit_h(cam, proj, ransac=ransac)
    dist, _, _ = fit_lens(cam[inl], proj[inl], cam_size) if inl.sum() >= 24 else ((0.0, 0.0), 0, 0)
    if dist != (0.0, 0.0):
        u = undistort_pts(cam, lens_K(cam_size), dist)
        H, inl, rms = fit_h(u, proj, ransac=ransac)
        if rms < rms0:
            return H, dist, rms, inl
    return H0, (0.0, 0.0), rms0, inl


def calib_from_aruco(board, found, cam_size, proj_size, need=MIN_MARKERS, mirrored=False):
    n = len(board)
    ids = [i for i in found if i in board]
    if len(ids) < MIN_PARTIAL:
        raise CalibError(f"only {len(ids)} of {n} markers seen: point the camera so it sees the projection (at least "
                         f"part of it), dim the room lights, and try again")
    cam = np.concatenate([found[i] for i in ids])
    proj = np.concatenate([board[i] for i in ids])
    if len(ids) >= need:                                 # enough markers spread out: fit the lens too
        H, dist, rms, _ = _with_lens(cam, proj, cam_size, 3.0)
    else:                                                # partial view: plain homography only
        H, _, rms = fit_h(cam, proj)
        dist = (0.0, 0.0)
    if rms > MAX_RMS:
        raise CalibError(f"markers do not line up (error {rms:.1f} px): is the wall flat and the camera still?")
    cov = proj_coverage(proj, proj_size)
    warn = ""
    if len(ids) < need or cov < 0.75:
        warn = (f"the camera sees about {cov * 100:.0f}% of the projection ({len(ids)}/{n} markers): outside that part the "
                f"mapping is extrapolated. Better: aim the camera so it sees all of it, or calibrate a second pose")
    return Calibration(H, rms, cam_size, proj_size, "aruco", seen=f"{len(ids)}/{n} markers", dist=dist, coverage=cov,
                       warning=warn, mirrored=mirrored)


def calib_from_gray(px, py, ok, cam_size, proj_size, step=4, stripe=STRIPE):
    cover = float(ok.mean())
    if cover < 0.02:
        raise CalibError("the structured light was not seen: is the camera looking at the projection, lights dimmed?")
    ys, xs = np.nonzero(ok[::step, ::step])
    ys, xs = ys * step, xs * step
    if len(xs) < 50:
        raise CalibError("too few decoded points")
    cam = np.c_[xs, ys].astype(np.float64)
    proj = np.c_[px[ys, xs], py[ys, xs]].astype(np.float64)
    H, dist, rms, inl = _with_lens(cam, proj, cam_size, stripe * 1.5)
    return Calibration(H, rms, cam_size, proj_size, "graycode", dense=(px, py, ok), dist=dist,
                       coverage=proj_coverage(proj[inl], proj_size),
                       seen=f"{cover * 100:.0f}% of the camera decoded, {inl.mean() * 100:.0f}% planar")


def union_coverage(calibs, proj_size, grid=(64, 40)):
    """Fraction of the projector frame seen by at least one calibration pose (sampled on a grid)."""
    W, H = proj_size
    xs, ys = np.meshgrid((np.arange(grid[0]) + 0.5) * W / grid[0], (np.arange(grid[1]) + 0.5) * H / grid[1])
    pts = np.c_[xs.ravel(), ys.ravel()]
    seen = np.zeros(len(pts), bool)
    for c in calibs:
        if c is None:
            continue
        q = c.proj_to_cam(pts)
        w, h = c.cam_size
        inside = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        if c.dense is not None:                           # where the structured light was actually decoded
            ok = c.dense[2]
            qi = np.clip(np.round(q).astype(int), 0, [w - 1, h - 1])
            inside &= ok[qi[:, 1], qi[:, 0]]
        seen |= inside
    return float(seen.mean())


def framing_advice(lit):
    """Live framing check for a webcam turned toward the wall: (coverage of the camera view, [advice]).
    `lit` is the projector-lit mask in UNMIRRORED camera coordinates (x right, y down)."""
    h, w = lit.shape
    frac = float(lit.mean())
    if frac < 0.01:
        return frac, ["the camera does not see the projection: turn the laptop lid toward the wall"]
    ys, xs = np.nonzero(lit)
    adv = []
    m = 2
    if ys.min() <= m:
        adv.append("tilt the camera UP a little (the top of the projection is cut off)")
    if ys.max() >= h - 1 - m:
        adv.append("tilt the camera DOWN a little (the bottom is cut off)")
    if xs.min() <= m:
        adv.append("turn the camera LEFT a little (the left side is cut off)")
    if xs.max() >= w - 1 - m:
        adv.append("turn the camera RIGHT a little (the right side is cut off)")
    if len(adv) >= 3:
        adv = ["move the laptop further back, or use a USB webcam with a wider view (most edges are cut off)"]
    if not adv and frac < 0.12:
        adv.append("the projection looks small: move the laptop closer, or point it straight at the wall")
    if not adv:
        adv.append("good: the whole projection is in view. Keep the laptop still from now on")
    return frac, adv


# ------------------------------------------------------------------ stage 2: faces

def lit_mask(grey, black, thr=20):
    """Where the projector light lands (projector grey minus ambient)."""
    import cv2
    d = cv2.GaussianBlur(np.asarray(grey, np.float32) - np.asarray(black, np.float32), (5, 5), 0)
    return d > thr


def _regions(grey, lit):
    """Connected lit regions split by camera edges (box edges / creases): (n, labels, stats)."""
    import cv2
    g = cv2.GaussianBlur(np.asarray(grey, np.uint8), (5, 5), 0)
    e = cv2.dilate(cv2.Canny(g, 30, 90), np.ones((3, 3), np.uint8))
    reg = (lit & (e == 0)).astype(np.uint8)
    reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(reg, connectivity=4)
    return n, lab, st


def fit_poly(region, max_pts=8):
    """Binary region -> (polygon camera px, quad?) : approxPolyDP with growing epsilon; 4 points preferred."""
    import cv2
    cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None, False
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    best = None
    for f in (0.01, 0.02, 0.03, 0.05, 0.08):
        p = cv2.approxPolyDP(hull, f * peri, True).reshape(-1, 2)
        if len(p) == 4:
            return p.astype(np.float64), True
        if 3 <= len(p) <= max_pts and best is None:
            best = p
    if best is not None:
        return best.astype(np.float64), False
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64), True


def order_quad(q):
    """TL, TR, BR, BL (image coordinates, y down)."""
    q = np.asarray(q, np.float64)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]                             # clockwise in y-down coordinates, starting near -180 deg
    k = int(np.argmin(q.sum(axis=1)))                  # top-left = smallest x + y
    return np.roll(q, -k, axis=0)


def candidates(grey, black, max_n=12):
    """[{poly, quad, conf, area}] faces found in the camera picture, biggest first."""
    lit = lit_mask(grey, black)
    n, lab, st = _regions(grey, lit)
    tot = lit.size
    out = []
    for i in range(1, n):
        area = st[i, 4]
        if area < MIN_FACE_FRAC * tot:
            continue
        poly, quad = fit_poly(lab == i)
        if poly is None:
            continue
        pa = abs(_area(poly))
        conf = float(min(area / max(pa, 1.0), 1.0))     # how well the fitted outline fills the region
        if quad:
            poly = order_quad(poly)
        f = dict(poly=poly, quad=quad, conf=round(conf, 2), area=float(area / tot))
        if area > 0.6 * max(int(lit.sum()), 1):          # the lit wall around everything: its outline is the
            f.update(wall=True, conf=round(conf * 0.5, 2), name="Wall (whole projection)")   # projection's own edge
        out.append(f)
    out.sort(key=lambda d: (bool(d.get("wall")), -d["area"]))      # real faces first, the wall last
    return out[:max_n]


def pick_face(grey, black, x, y):
    """The face under a click (camera px), or the nearest lit region; None if nothing is lit near it."""
    lit = lit_mask(grey, black)
    n, lab, st = _regions(grey, lit)
    h, w = lab.shape
    xi, yi = int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1))
    k = lab[yi, xi]
    if k == 0:
        ys, xs = np.nonzero(lab)
        if not len(xs):
            return None
        j = int(np.argmin((xs - xi) ** 2 + (ys - yi) ** 2))
        if (xs[j] - xi) ** 2 + (ys[j] - yi) ** 2 > (0.08 * w) ** 2:
            return None
        k = lab[ys[j], xs[j]]
    poly, quad = fit_poly(lab == k)
    if poly is None:
        return None
    return dict(poly=order_quad(poly) if quad else poly, quad=quad, conf=1.0, area=float(st[k, 4] / lab.size))


def _area(p):
    p = np.asarray(p, np.float64)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def validate_polys(raw, cam_size, max_polys=12):
    """AI-assist (or any outside) polygons in camera px -> clean list; ValueError with a reason if unusable.
    Each: 3..8 finite points inside the frame, simple (no self-crossing), not tiny."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= max_polys:
        raise ValueError(f"expected 1..{max_polys} polygons")
    w, h = cam_size
    out = []
    for p in raw:
        pts = p.get("points") if isinstance(p, dict) else p
        if not isinstance(pts, list) or not 3 <= len(pts) <= 8:
            raise ValueError("each polygon needs 3..8 points")
        a = []
        for q in pts:
            if (not isinstance(q, (list, tuple)) or len(q) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in q)):
                raise ValueError("points must be [x, y] numbers")
            if not (-1 <= q[0] <= w + 1 and -1 <= q[1] <= h + 1):
                raise ValueError("a point lies outside the camera picture")
            a.append([float(q[0]), float(q[1])])
        a = np.array(a)
        if abs(_area(a)) < MIN_FACE_FRAC * w * h:
            raise ValueError("a polygon is too small")
        if not _simple(a):
            raise ValueError("a polygon crosses itself")
        name = p.get("name") if isinstance(p, dict) else ""
        name = name.strip()[:40] if isinstance(name, str) else ""
        quad = len(a) == 4 and _convex(a)                  # a concave 4-gon is a polygon mask, never re-ordered
        out.append(dict(poly=order_quad(a) if quad else a, quad=quad, conf=1.0, name=name,
                        area=abs(_area(a)) / (w * h)))
    return out


def _convex(p):
    p = np.asarray(p, np.float64)
    n = len(p)
    def x2(a, b):
        return a[0] * b[1] - a[1] * b[0]
    cr = [x2(p[(i + 1) % n] - p[i], p[(i + 2) % n] - p[(i + 1) % n]) for i in range(n)]
    return all(c > 1e-6 for c in cr) or all(c < -1e-6 for c in cr)


def _simple(p):
    n = len(p)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            a, b, c, d = p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]
            if cross(a, b, c) * cross(a, b, d) <= 0 and cross(c, d, a) * cross(c, d, b) <= 0:
                return False                             # crossing, touching or collinear-overlapping edges
    return True


def to_surfaces(found, calib, existing, proj_size, names=None):
    """Camera polygons -> surfaces.py surface dicts (projector px), appended to `existing` (a list of surface dicts).
    A lone untouched full-frame surface #1 is replaced (black outside the found faces). Quads -> shape quad; other
    polygons -> shape poly with the polygon as mask and its bounding quad as corners."""
    import surfaces
    W, H = proj_size
    keep = list(existing)
    if len(keep) == 1 and np.allclose(np.array(keep[0]["corners"], float), np.array(surfaces.full_quad(proj_size)), atol=2.0) \
            and not keep[0].get("mask"):
        keep = []
    nid = max([s["id"] for s in keep], default=0) + 1
    made = []
    for k, f in enumerate(found):
        if len(keep) + len(made) >= surfaces.MAX_SURFACES:
            break
        p = calib.cam_to_proj(f["poly"])
        if not np.isfinite(p).all():
            continue                                     # beyond the calibration's vanishing line: skip, never wrap
        p = np.clip(p, [-0.4 * W, -0.4 * H], [1.4 * W, 1.4 * H])
        nm = names[k] if names and k < len(names) and isinstance(names[k], str) else ""
        nm = nm.strip() or (f.get("name") or "").strip()
        name = nm[:40] or f"Auto {k + 1}"
        if f["quad"] and _convex(order_quad(p)):
            s = surfaces.new_surface(nid, proj_size, name, order_quad(p).tolist())
        else:
            x0, y0 = p.min(axis=0)
            x1, y1 = p.max(axis=0)
            s = surfaces.new_surface(nid, proj_size, name, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            if len(p) > surfaces.MAX_MASK or not _simple(p):
                continue                                 # a warped outline that folds over itself: skip it
            s["shape"], s["mask"] = "poly", p.tolist()
        made.append(s)
        nid += 1
    return keep + made, [s["id"] for s in made]


# ------------------------------------------------------------------ stage 3: verify & refine

def outline_image(quads, size, width=3, level=PATTERN_LEVEL):
    """Projector image with thin outlines of the given projector quads (the verify pattern)."""
    import cv2
    img = np.zeros((size[1], size[0]), np.uint8)
    for q in quads:
        cv2.polylines(img, [np.round(np.asarray(q)).astype(np.int32).reshape(-1, 1, 2)], True, int(level), width,
                      cv2.LINE_AA)
    return img


def _edge_lines(img, quad_cam, search, mode):
    """For each side of quad_cam, sample along the side and look +-search px along its normal for the strongest
    response (mode 'grad' = image gradient magnitude, 'ridge' = brightest), fit a line. -> [(point, dir)] or None."""
    import cv2
    f = np.asarray(img, np.float32)
    if mode == "grad":
        gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        f = np.hypot(gx, gy)
    else:
        f = cv2.GaussianBlur(f, (3, 3), 0)
    h, w = f.shape
    lines = []
    q = np.asarray(quad_cam, np.float64)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        L = float(np.hypot(*d))
        if L < 8:
            return None
        d /= L
        nrm = np.array([-d[1], d[0]])
        pts = []
        for s in np.linspace(0.12, 0.88, 24):
            c = a + (b - a) * s
            offs = np.arange(-search, search + 0.5, 0.5)
            xy = c[None, :] + offs[:, None] * nrm[None, :]
            xi, yi = np.round(xy[:, 0]).astype(int), np.round(xy[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if ok.sum() < 3:
                continue
            v = np.where(ok, f[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], -1)
            j = int(np.argmax(v))
            if v[j] <= 0:
                continue
            pts.append(c + offs[j] * nrm)
        if len(pts) < 6:
            return None
        P = np.array(pts)
        off = (P - a) @ nrm                              # robust: drop samples far from the median offset
        P = P[np.abs(off - np.median(off)) <= max(1.5, 2.5 * np.median(np.abs(off - np.median(off))))]
        if len(P) < 4:
            return None
        c0 = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c0)
        if abs(float(vt[0] @ d)) < 0.8:                  # the fitted line must run along the side, not across it
            return None
        lines.append((c0, vt[0]))
    return lines


def _intersect(l1, l2):
    (p, d), (q, e) = l1, l2
    A = np.array([d, -e]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, q - p)
    return p + t[0] * d


def _corners(lines):
    out = []
    for i in range(4):
        c = _intersect(lines[i - 1], lines[i])
        if c is None:
            return None
        out.append(c)
    return np.array(out)


def physical_quad(edge_img, quad_cam, search=12):
    """Camera image of the scene (grey-lit) + a rough quad in camera px -> the quad of the real edges, or None."""
    lines = _edge_lines(edge_img, quad_cam, search, "grad")
    return None if lines is None else _corners(lines)


def projected_quad(outline_img, quad_cam, search=12):
    """Camera image of the projected outline (outline minus black) + where it should be -> where it really is."""
    lines = _edge_lines(outline_img, quad_cam, search, "ridge")
    return None if lines is None else _corners(lines)


def refine_step(quad_proj, calib, target_cam, seen_cam, gain=0.7):
    """One closed-loop correction. target_cam = physical corners, seen_cam = where the projected corners appeared
    (both camera px). Each corner moves in projector px by the calibration's local mapping of the camera error, so
    a biased calibration still converges. -> (new quad_proj, max camera error before)."""
    quad_proj = np.asarray(quad_proj, np.float64)
    err = np.linalg.norm(np.asarray(target_cam) - np.asarray(seen_cam), axis=1)
    step = calib.cam_to_proj(target_cam, dense=False) - calib.cam_to_proj(seen_cam, dense=False)
    return quad_proj + gain * step, float(err.max())   # damped: a face at another depth than the wall cannot overshoot


# ------------------------------------------------------------------ AI assist helpers (pure)

def redact(gray, person_mask=None, thr=0.25):
    """The picture that may leave the laptop: people (segmentation mask, any size, 0..1) greyed out, generously."""
    import cv2
    g = np.clip(np.asarray(gray, np.float32), 0, 255).astype(np.uint8)
    if person_mask is None:
        return g, 0.0
    m = cv2.resize(np.asarray(person_mask, np.float32), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR) > thr
    m = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    frac = float(m.mean())
    if frac > 0:
        g = g.copy()
        g[m] = int(np.median(g[~m])) if (~m).any() else 0
    return g, frac


def png_bytes(gray):
    import cv2
    ok, b = cv2.imencode(".png", np.asarray(gray, np.uint8))
    if not ok:
        raise CalibError("could not encode the picture")
    return b.tobytes()


def ai_polygons(obj, cam_size, max_n=AI_MAX_POLYS):
    """Model reply {"polygons": [...]} -> validated candidates marked ai=True (invalid ones dropped, reasons kept)."""
    raw = obj.get("polygons") if isinstance(obj, dict) else None
    if not isinstance(raw, list):
        raise CalibError("the AI reply had no polygon list")
    good, bad = [], []
    for k, p in enumerate(raw[:max_n]):
        try:
            f = validate_polys([p], cam_size)[0]
            f["ai"] = True
            f["name"] = f.get("name") or f"AI {k + 1}"
            good.append(f)
        except ValueError as ex:
            bad.append(f"#{k + 1}: {ex}")
    if len(raw) > max_n:
        bad.append(f"{len(raw) - max_n} more ignored (max {max_n})")
    return good, bad


def ai_corrections(obj, shapes, max_d=AI_MAX_NUDGE):
    """Model reply {"corrections": [...]} -> {surface id: (4, 2) camera px deltas}, each clamped to max_d (length),
    unknown ids / vertices / non-numbers dropped, one correction per vertex (the first)."""
    raw = obj.get("corrections") if isinstance(obj, dict) else None
    if not isinstance(raw, list):
        raise CalibError("the AI reply had no correction list")
    n = {int(s["id"]): len(s["points"]) for s in shapes}
    out, seen = {}, set()
    for c in raw[:64]:
        if not isinstance(c, dict):
            continue
        try:
            sid, v, dx, dy = int(c.get("id")), int(c.get("vertex")), float(c.get("dx")), float(c.get("dy"))
        except (TypeError, ValueError):
            continue
        if sid not in n or not 0 <= v < n[sid] or (sid, v) in seen or not (math.isfinite(dx) and math.isfinite(dy)):
            continue
        seen.add((sid, v))
        L = math.hypot(dx, dy)
        if L > max_d:
            dx, dy = dx * max_d / L, dy * max_d / L
        out.setdefault(sid, np.zeros((n[sid], 2)))[v] = (dx, dy)
    return out


# ------------------------------------------------------------------ the wizard (engine side, its own thread)

class AutoMap:
    """Runs the wizard steps on a daemon thread. The engine provides:
         show(img or None)      the overlay to draw (uint8 projector-size image), drawn until replaced
         shown_id()             counter of overlay images actually drawn (so a capture waits for its pattern)
         still(skip)            a fresh camera still (uint8 grey, camera px) or raises
         surfaces()             (list of surface dicts, output size)     surf(cmd)   one validated surfaces command
    State for the panel: state(); camera pictures for the panel: image_jpeg() (memory only, never saved)."""

    def __init__(self, show, shown_id, still, surfaces_fn, surf_cmd, calib_dir=CALIB_DIR, hold=HOLD_S,
                 clock=time.monotonic, sleep=time.sleep, exposure=None, ready=None, vision=None, person=None,
                 profile="usb", stills=None):
        self._stills = stills or (lambda seconds: None)     # camera: open / close the /still gate
        self.touched = clock()                               # last wizard activity (pictures are dropped when idle)
        self._vision = vision or (lambda: None)              # -> plugins.online.vision.Vision while ONLINE is on, else None
        self._person = person or (lambda: None)              # -> person mask (0..1, camera aspect) or None
        self.ai = None                                       # pending AI request: shown in the panel, sent on SEND
        self._ready = ready or (lambda: None)                # -> reason the camera cannot be used, or None
        self._exposure = exposure or (lambda on: None)     # lock the camera's auto exposure during calibration
        self._show, self._shown_id, self._still, self._surfaces, self._surf = show, shown_id, still, surfaces_fn, surf_cmd
        self.calib_dir, self.hold, self.clock, self.sleep = Path(calib_dir), hold, clock, sleep
        self.profile = profile if profile in PROFILES else "usb"
        self.pose = 1
        self.poses = {}                                  # pose -> Calibration (two-pose capture for a partial view)
        self.mirrored = None                             # None = unknown; True = the camera delivers a mirror image
        self.adapt = dict(stripe=STRIPE, level=PROFILES[self.profile]["level"], snr=None, scale=None)
        self.framing = None                              # live framing check: {coverage, advice}
        self._load_profile()
        self.step, self.busy, self.msg, self.err, self.progress = "idle", False, "", "", 0.0
        self.scene = None                                # (grey, black) camera stills of the scene
        self.scene_person = (None, False)                # (person mask, segmentation trusted) taken with the scene
        self.found = []                                  # candidate faces (camera px)
        self.made = []                                   # surface ids created by the last "create"
        self.report = []                                 # per-surface verify result
        self.image = None                                # (kind, uint8 BGR) picture for the panel
        self._cancel = threading.Event()
        self._why = ""
        self._thread = None
        self._pid = 0

    # ---- plumbing
    def _run(self, name, fn, *a):
        if self.busy:
            self.err = "busy: wait for the current step or press CANCEL"
            return False
        why = self._ready()
        if why:
            self.err = why
            return False
        self.busy, self.err, self.msg, self.progress, self.step = True, "", "", 0.0, name
        self._cancel.clear()
        self._why = ""
        self.touched = self.clock()

        def work():
            try:
                self._stills(600)                        # the camera serves stills only while a step runs
                fn(*a)
            except (CalibError, RuntimeError) as ex:
                self.err = (self._why if self._cancel.is_set() and self._why else str(ex))
            except Exception as ex:                      # a CV / camera failure must never reach the show
                self.err = f"{type(ex).__name__}: {ex}"[:240]
            finally:
                self._show(None)
                self._stills(0)
                self.busy = False
        self._thread = threading.Thread(target=work, name=f"automap-{name}", daemon=True)
        self._thread.start()
        return True

    def _project(self, img, settle=None):
        """Show a pattern and wait until it has been drawn and held (photosensitive: >= hold per pattern)."""
        if self._cancel.is_set():
            raise CalibError("cancelled")
        n0 = self._shown_id()
        self._show(img)
        if self._cancel.is_set():                        # cancelled while we were putting it up: take it down again
            self._show(None)
            raise CalibError("cancelled")
        t0 = self.clock()
        while self._shown_id() <= n0:
            if self._cancel.is_set():
                raise CalibError("cancelled")
            if self.clock() - t0 > 3.0:
                raise CalibError("the projector frame is not being drawn (is the projector window up, blackout off?)")
            self.sleep(0.01)
        t0 = self.clock()                                # the hold counts from when the pattern is really up
        rest = (self.hold * PROFILES[self.profile]["settle"] if settle is None else settle) - (self.clock() - t0)
        if rest > 0:
            if self.sleep is time.sleep:                 # the real clock: CANCEL / PANIC ends the hold at once
                if self._cancel.wait(rest):
                    raise CalibError("cancelled")
            else:                                        # tests with a fake clock
                self.sleep(rest)

    def _cap(self, img, skip=2, n=None):
        """Show a pattern, then the average of n stills (noisy built-in webcams), unmirrored if the camera mirrors."""
        self._project(img)
        n = PROFILES[self.profile]["n_avg"] if n is None else n
        acc = np.asarray(self._still(skip), np.float32)
        for _ in range(n - 1):
            acc = acc + np.asarray(self._still(1), np.float32)
        acc /= n
        return acc[:, ::-1].copy() if self.mirrored else acc

    # ---- profiles / poses (calibration/<profile>/pose<n>/: numbers only)
    def _folder(self, pose=None):
        return self.calib_dir / self.profile / f"pose{pose or self.pose}"

    def _load_profile(self):
        self.poses = {}
        for k in (1, 2):
            c = Calibration.load(self._folder(k))
            if c is not None:
                self.poses[k] = c
        if not self.poses and self.profile == "usb":
            legacy = Calibration.load(self.calib_dir)    # a calibration saved before profiles existed
            if legacy is not None:
                self.poses[1] = legacy
        self.calib = self.poses.get(self.pose) or self.poses.get(1)
        self.mirrored = self.calib.mirrored if self.calib is not None else None
        self.adapt["level"] = PROFILES[self.profile]["level"]

    def set_profile(self, name):
        if name not in PROFILES or self.busy:
            return False
        self.profile, self.pose = name, 1
        self._load_profile()
        self.msg = f"camera profile: {PROFILES[name]['label']}" + (" (calibration loaded)" if self.calib else " (not calibrated yet)")
        return True

    def set_pose(self, n):
        if n not in (1, 2) or self.busy:
            return False
        self.pose = n
        self.calib = self.poses.get(n)
        self.scene, self.found = None, []                # a new pose sees a different picture: detect again
        self.msg = (f"pose {n}: " + ("calibrated" if self.calib else "move / turn the laptop so the camera sees the other "
                                     "part of the projection (with some overlap), keep it still, then 2 CALIBRATE"))
        return True

    # ---- steps
    def check(self):
        """Step 1: does the camera see the whole projection? White frame vs black."""
        return self._run("check", self._check)

    def _detect_mirror(self, W, H):
        """Built-in webcams often deliver a mirror image: the ArUco markers only decode the right way round."""
        board, _ = aruco_board(W, H, level=self.adapt["level"])
        was, self.mirrored = self.mirrored, False
        u8 = np.clip(self._cap(board), 0, 255).astype(np.uint8)
        a, b = len(detect_aruco(u8)), len(detect_aruco(u8[:, ::-1].copy()))
        self.mirrored = (b > a) if max(a, b) >= MIN_PARTIAL else was
        return max(a, b)

    def _check(self):
        W, H = self._surfaces()[1]
        if self.mirrored is None:
            self._detect_mirror(W, H)
        black = self._cap(np.zeros((H, W), np.uint8))
        black2 = self._cap(np.zeros((H, W), np.uint8))
        white = self._cap(np.full((H, W), self.adapt["level"], np.uint8))
        lit = lit_mask(white, black)
        frac, advice = framing_advice(lit)
        self.framing = dict(coverage=round(frac, 3), advice=advice)
        self._picture("check", white, lit)
        if frac < 0.01:
            raise CalibError(advice[0])
        noise = float(np.std(black2 - black)) / math.sqrt(2.0) + 1e-3
        signal = float(np.median((white - black)[lit]))
        snr = signal / noise
        scale = math.sqrt(lit.sum() / float(W * H))        # camera px per projector px (along one axis)
        stripe = STRIPE
        while stripe < 32 and stripe * scale < 2.5:        # each Gray-code stripe >= ~2.5 camera px
            stripe *= 2
        level = PROFILES[self.profile]["level"]
        if snr < 8:
            level = LEVEL_MAX                              # dim / noisy: brightest pattern we use (still static)
        self.adapt = dict(stripe=stripe, level=level, snr=round(snr, 1), scale=round(scale, 3))
        self.msg = (("camera image is mirrored (handled automatically). " if self.mirrored else "")
                    + f"the projection covers {frac * 100:.0f}% of the camera view; signal/noise {snr:.0f}"
                    f"{' (low: dim the room)' if snr < 8 else ''}; Gray-code stripes {stripe} px. " + " ".join(advice))

    def framing_live(self, on=True, seconds=90.0):
        """Live framing check while the laptop is being aimed: a steady dim white frame (no flashing), the camera
        view with the lit area marked and advice, refreshed about twice a second; stops after `seconds` or CANCEL."""
        if not on:
            self.cancel()
            return True
        return self._run("framing", self._framing, seconds)

    def _framing(self, seconds):
        W, H = self._surfaces()[1]
        black = self._cap(np.zeros((H, W), np.uint8), n=1)
        self._project(np.full((H, W), 120, np.uint8))
        t0 = self.clock()
        while self.clock() - t0 < seconds and not self._cancel.is_set():
            g = np.asarray(self._still(1), np.float32)
            g = g[:, ::-1] if self.mirrored else g
            lit = lit_mask(g, black, thr=12)
            frac, advice = framing_advice(lit)
            self.framing = dict(coverage=round(frac, 3), advice=advice)
            self._picture("check", g, lit)
            self.msg = f"live framing: {frac * 100:.0f}% of the camera view is projection. " + " ".join(advice)
            self.progress = min(1.0, (self.clock() - t0) / seconds)
            self.sleep(0.4)

    def calibrate(self, gray=True):
        return self._run("calibrate", self._calibrate, gray)

    def _calibrate(self, gray):
        W, H = self._surfaces()[1]
        self._project(np.full((H, W), int(self.adapt["level"] * 0.8), np.uint8))   # settle on a neutral frame,
        self._exposure(True)                                                          # then freeze the exposure
        try:
            self._calibrate_locked(gray)
        finally:
            self._exposure(False)

    def _calibrate_locked(self, gray):
        W, H = self._surfaces()[1]
        lv = self.adapt["level"]
        board, corners = aruco_board(W, H, level=lv)
        img = self._cap(board)
        cam_size = (img.shape[1], img.shape[0])
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        found = detect_aruco(u8)
        if self.mirrored is None or len(found) < MIN_MARKERS:      # a mirrored camera: markers only read when flipped
            flip = detect_aruco(u8[:, ::-1].copy())
            if len(flip) > len(found):
                self.mirrored = not bool(self.mirrored)
                img, found = img[:, ::-1].copy(), flip
            elif self.mirrored is None:
                self.mirrored = False
        cal = calib_from_aruco(corners, found, cam_size, (W, H), mirrored=bool(self.mirrored))
        self.msg = f"markers: {cal.seen}, error {cal.rms:.2f} px"
        if gray:
            st = self.adapt["stripe"]
            seq = gray_sequence(W, H, st, lv)
            caps = {}
            for k, (key, pat) in enumerate(seq):
                caps[key] = self._cap(pat)
                self.progress = (k + 1) / len(seq)
            px, py, ok = gray_decode(caps, W, H, stripe=st)
            try:
                g = calib_from_gray(px, py, ok, cam_size, (W, H), stripe=st)
                if not cal.warning or g.rms > cal.rms * 1.5 + st:   # markers fully seen (sub-pixel corners): their
                    g.H, g.Hinv, g.rms, g.dist = cal.H, cal.Hinv, cal.rms, cal.dist   # planar fit + lens win; the
                    g.K = cal.K                                         # stripes only add the dense map
                g.method, g.mirrored = "aruco+graycode", cal.mirrored
                g.coverage, g.warning = max(g.coverage, cal.coverage), cal.warning if g.coverage < 0.75 else ""
                g.seen = cal.seen + "; " + g.seen
                cal = g
            except CalibError as ex:
                self.msg += f" (structured light skipped: {ex})"
        self.calib = cal
        self.poses[self.pose] = cal
        cal.save(self._folder())
        self._picture("calib", img, None)
        both = union_coverage(list(self.poses.values()), (W, H)) if len(self.poses) > 1 else None
        self.msg = (f"calibrated pose {self.pose} ({cal.method}): {cal.seen}, error {cal.rms:.2f} px"
                    + (f", lens k1 {cal.dist[0]:+.2f}" if cal.dist[0] else "") + (", camera image is mirrored (handled)" if cal.mirrored else "")
                    + (f"; both poses together see {both * 100:.0f}% of the projection" if both is not None else "")
                    + (f". NOTE: {cal.warning}" if cal.warning else ""))

    def detect(self):
        return self._run("detect", self._detect)

    def _need_calib(self):
        if self.calib is None:
            raise CalibError("calibrate first (step 2)")

    def _detect(self):
        self._need_calib()
        W, H = self._surfaces()[1]
        black = self._cap(np.zeros((H, W), np.uint8))
        grey = self._cap(np.full((H, W), int(PATTERN_LEVEL * 0.6), np.uint8))
        self.scene = (grey, black)
        self.scene_person = self._person_now()           # who was in front of the camera when THIS photo was taken
        self.found = candidates(grey, black)
        self._picture("scene", grey, None)
        self.msg = f"{len(self.found)} face(s) found; click a face in the picture to add one, untick the ones you don't want"

    def pick(self, x, y):
        """Click-to-pick (camera uv 0..1, y down): runs on the wizard thread (OpenCV never on the GL thread)."""
        if self.scene is None:
            self.err = "detect surfaces first (step 3)"
            return False
        return self._run("pick", self._pick, x, y)

    def _pick(self, x, y):
        grey, black = self.scene
        h, w = grey.shape
        f = pick_face(grey, black, x * w, y * h)
        if f is None:
            raise CalibError("nothing lit there")
        self.found = self.found + [f]
        self._picture("scene", grey, None)

    def add_polys(self, polys):
        """Validated outside polygons (AI assist) in camera px."""
        if self.scene is None:
            self.err = "detect surfaces first (step 3)"
            return False
        if self.busy:
            self.err = "busy: wait for the current step"
            return False
        h, w = self.scene[0].shape
        try:
            self.found = self.found + validate_polys(polys, (w, h))
        except ValueError as ex:
            self.err = f"AI result refused: {ex}"
            return False
        self._picture("scene", self.scene[0], None)
        return True

    def create(self, keep=None, names=None):
        """Selected candidates -> surfaces (ONE undo step via surfaces 'replace')."""
        if self.busy:
            self.err = "busy: wait for the current step"
            return False
        if self.calib is None or not self.found:
            self.err = "nothing to create: calibrate and detect first"
            return False
        sel = [f for i, f in enumerate(self.found) if keep is None or i in keep]
        if not sel:
            self.err = "no face selected"
            return False
        cur, size = self._surfaces()
        new, ids = to_surfaces(sel, self.calib, cur, size, names)
        if not ids:
            self.err = "none of the selected faces could be turned into a surface (outside the calibrated area?)"
            return False
        import surfaces
        cmd = {"cmd": "surf", "op": "replace", "surfaces": new}
        try:
            surfaces.validate_cmd(cmd, size)               # refuse here, with the reason, rather than silently later
        except ValueError as ex:
            self.err = f"could not create the surfaces: {ex}"
            return False
        self._surf(cmd)
        self.made, self.step = ids, "created"
        self.msg = f"{len(ids)} surface(s) created" + (f" ({len(sel) - len(ids)} skipped)" if len(ids) < len(sel) else "") \
            + ": edit them by hand as usual, or VERIFY & REFINE"
        return True

    def verify(self, ids=None, iters=3):
        return self._run("verify", self._verify, ids, iters)

    def _verify(self, ids, iters):
        self._need_calib()
        cur, size = self._surfaces()
        W, H = size
        want = ids or self.made or [x["id"] for x in cur]
        todo = [s for s in cur if s["id"] in want and s.get("shape", "quad") == "quad"]   # polygon masks: by hand
        black = self._cap(np.zeros((H, W), np.uint8))
        grey = self._cap(np.full((H, W), int(PATTERN_LEVEL * 0.6), np.uint8))
        lit = np.clip(grey - black, 0, 255)               # projector light only: no ambient / wall texture edges
        report = []
        start = {s["id"]: np.array(s["corners"], np.float64) for s in todo}
        quads = {k: v.copy() for k, v in start.items()}
        targets = {}
        for sid, q in quads.items():
            t = physical_quad(lit, self.calib.proj_to_cam(q))
            if t is not None and np.isfinite(t).all():
                targets[sid] = t
        first, best = {}, {}
        for n, sid in enumerate(targets):                 # one surface at a time: neighbours' outlines can't confuse it
            for it in range(iters + 1):
                out = self._cap(outline_image([quads[sid]], size))
                seen = projected_quad(np.clip(out - black, 0, 255), self.calib.proj_to_cam(quads[sid]))
                if seen is None or not np.isfinite(seen).all():
                    break
                e = float(np.linalg.norm(targets[sid] - seen, axis=1).max())
                first.setdefault(sid, e)
                if sid not in best or e < best[sid][0]:
                    best[sid] = (e, quads[sid].copy())
                if e <= REFINE_TOL or it == iters or e > best[sid][0] + 1.0:   # converged / out of steps / diverging
                    break
                newq, _ = refine_step(quads[sid], self.calib, targets[sid], seen)
                if not _convex(newq) or np.abs(newq - start[sid]).max() > 0.25 * W:
                    break
                quads[sid] = newq                        # the loop projects its own outline; surfaces change once, below
            self.progress = (n + 1) / max(len(targets), 1)
        out = grey
        for sid in targets:
            e, q = best.get(sid, (None, start[sid]))
            quads[sid] = q
            report.append(dict(id=sid, before=None if sid not in first else round(first[sid], 2),
                               after=None if e is None else round(e, 2)))
        moved = [sid for sid in targets if not np.allclose(quads[sid], start[sid])]
        if moved:                                        # ONE undo step, merged onto the surfaces as they are NOW
            latest, _ = self._surfaces()
            new = [dict(s, corners=quads[s["id"]].tolist()) if s["id"] in moved and
                   np.allclose(np.array(s["corners"], np.float64), start[s["id"]]) else s for s in latest]
            self._surf({"cmd": "surf", "op": "replace", "surfaces": new})
        missing = [s["id"] for s in todo if s["id"] not in targets]
        self.report = report
        self._picture("verify", out, None)
        self.msg = ("refined: " + ", ".join(f"#{r['id']} {r['before']} -> {r['after']} cam px" for r in report)
                    + (f"; no clear physical edges found for #{', #'.join(map(str, missing))} (left as they were)" if missing else "")
                    + ". UNDO (Ctrl+Z) puts the corners back.")

    # ---- AI assist (ONLINE only, button press only, the exact picture shown first)
    def ai_prepare(self, kind, instruction=""):
        v = self._vision()
        if v is None:
            self.err = "AI assist needs ONLINE (top bar); nothing is sent while it is off"
            return False
        if kind == "detect":
            if self.scene is None:
                self.err = "run 3 DETECT SURFACES first (it takes the photo the AI looks at)"
                return False
            self._ai_stage("detect", self.scene[0], instruction, None, self.scene_person)
            return True
        if self.calib is None:
            self.err = "calibrate first (step 2)"
            return False
        return self._run("ai_prepare", self._ai_prepare_verify, instruction)

    def _person_now(self):
        """(mask or None, segmentation trusted?) at this moment, un-mirrored like the stills."""
        got = self._person()
        mask, seg = got if isinstance(got, tuple) else (got, got is not None)
        if mask is not None:
            mask = np.array(mask, np.float32)
            if self.mirrored:
                mask = mask[:, ::-1]
        return mask, bool(seg)

    def _ai_stage(self, kind, gray, instruction, extra, person=None):
        from plugins.online import vision as V
        mask, seg = person if person is not None else (None, False)
        g, frac = redact(gray, mask)
        h, w = g.shape
        if kind == "detect":
            prompt = V.detect_prompt(w, h, instruction, AI_MAX_POLYS)
        else:
            prompt = V.verify_prompt(w, h, extra["shapes"], instruction, AI_MAX_NUDGE)
        # no person segmentation for this photo (MediaPipe missing, camera just started): always ask
        person = (not seg) or (mask is not None and float(np.max(mask)) > 0.5)
        self.ai = dict(kind=kind, png=png_bytes(g), prompt=prompt, instruction=V.clean_instruction(instruction),
                       person=person, redacted=round(frac, 3), extra=extra, size=(w, h))
        self._picture("ai_send", g, None)
        v = self._vision()
        names = v.available() if v is not None else []
        self.msg = ("This exact picture (people greyed out) and the question below will be sent to "
                    + (" / ".join(names) or "no provider yet: save a Gemini API key on AI DREAM") + ". Press SEND, or DISCARD.")
        self.err = ""

    def _ai_prepare_verify(self, instruction):
        cur, size = self._surfaces()
        W, H = size
        want = self.made or [x["id"] for x in cur]
        todo = [s for s in cur if s["id"] in want and s.get("shape", "quad") == "quad"]
        if not todo:
            raise CalibError("no quad surfaces to check")
        quads = {s["id"]: np.array(s["corners"], np.float64) for s in todo}
        black = self._cap(np.zeros((H, W), np.uint8))
        grey = self._cap(np.full((H, W), int(PATTERN_LEVEL * 0.6), np.uint8))
        pat = np.maximum(np.full((H, W), int(PATTERN_LEVEL * 0.3), np.uint8), outline_image(list(quads.values()), size))
        photo = self._cap(pat)
        person = self._person_now()
        shapes = [dict(id=sid, points=self.calib.proj_to_cam(q).tolist()) for sid, q in quads.items()]
        self._ai_stage("verify", photo, instruction, dict(shapes=shapes, quads={k: v.tolist() for k, v in quads.items()},
                                                          grey=grey, black=black), person)

    def ai_send(self, confirm_person=False):
        if self.ai is None:
            self.err = "nothing prepared"
            return False
        if self._vision() is None:
            self.ai = None
            self.err = "ONLINE was switched off: nothing sent"
            return False
        if self.ai["person"] and not confirm_person:
            self.err = "someone may be in the picture (greyed out): tick 'send anyway' to confirm"
            return False
        return self._run("ai_send", self._ai_send)

    def _ai_send(self):
        req, self.ai = self.ai, None
        v = self._vision()
        if v is None:
            raise CalibError("ONLINE was switched off: nothing sent")
        from plugins.online.backends import BackendError
        try:
            obj, prov = v.ask(req["kind"], req["prompt"], req["png"], cancel=self._cancel)
        except BackendError as ex:
            raise CalibError(f"AI assist: {ex}") from None
        if req["kind"] == "detect":
            good, bad = ai_polygons(obj, req["size"])
            self.found += good
            if self.scene is not None:
                self._picture("scene", self.scene[0], None)
            self.msg = (f"{prov}: {len(good)} surface(s) suggested (AI badge)" + (f"; refused {'; '.join(bad)}" if bad else "")
                        + ". Tick the ones you want and CREATE.")
            return
        self._ai_apply_verify(obj, req, prov)

    def _ai_apply_verify(self, obj, req, prov):
        ex = req["extra"]
        corr = ai_corrections(obj, ex["shapes"])
        if not corr:
            self.msg = f"{prov}: no corrections suggested"
            return
        cur, size = self._surfaces()
        W, H = size
        black, grey = ex["black"], ex["grey"]
        old = {sid: np.array(q) for sid, q in ex["quads"].items()}
        new = {}
        for sid, d in corr.items():
            cam = np.array(next(s["points"] for s in ex["shapes"] if s["id"] == sid))
            new[sid] = old[sid] + self.calib.cam_to_proj(cam + d, dense=False) - self.calib.cam_to_proj(cam, dense=False)

        def errors(quads):
            out = self._cap(outline_image(list(quads.values()), size))
            diff = np.clip(out - black, 0, 255)
            e = {}
            for sid, q in quads.items():
                tgt = physical_quad(grey, self.calib.proj_to_cam(q))
                seen = projected_quad(diff, self.calib.proj_to_cam(q))
                e[sid] = None if tgt is None or seen is None else float(np.linalg.norm(tgt - seen, axis=1).max())
            return e
        before = errors({sid: old[sid] for sid in new})
        after = errors(new)
        keep, rep = {}, []
        for sid in new:
            b, a = before.get(sid), after.get(sid)
            ok = a is None or b is None or a <= b + 0.3        # stop if the error did not improve
            if ok:
                keep[sid] = new[sid]
            rep.append(dict(id=sid, before=None if b is None else round(b, 2), after=None if a is None else round(a, 2),
                            kept=ok, ai=True))
        if keep:                                               # ONE undo step
            self._surf({"cmd": "surf", "op": "replace",
                        "surfaces": [dict(s, corners=keep[s["id"]].tolist()) if s["id"] in keep else s for s in cur]})
        self.report = rep
        self.msg = (f"{prov}: " + ", ".join(f"#{r['id']} {r['before']} -> {r['after']} ({'kept' if r['kept'] else 'worse: reverted'})"
                                             for r in rep) + ". UNDO puts the corners back.")

    def ai_discard(self):
        self.ai = None
        self.msg = "AI request discarded: nothing was sent"

    def cancel(self, why=""):
        self._why = why
        self._cancel.set()
        self._show(None)                                 # the pattern leaves the wall NOW, not when the step unwinds

    def undo(self):
        self._surf({"cmd": "surf", "op": "undo"})

    # ---- panel
    def _picture(self, kind, img, lit):
        """Camera picture for the wizard (memory only): stills + found faces / calibration. Never saved."""
        import cv2
        g = np.clip(np.asarray(img, np.float32), 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if lit is not None:
            bgr[lit] = (bgr[lit] * 0.5 + np.array([0, 90, 0])).astype(np.uint8)
        if kind == "scene":
            for i, f in enumerate(self.found):
                p = np.round(f["poly"]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(bgr, [p], True, (60, 220, 255) if f["quad"] else (255, 120, 220), 2)
                c = f["poly"].mean(axis=0).astype(int)
                cv2.putText(bgr, str(i + 1), (int(c[0]) - 6, int(c[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self._pid += 1
        self.image = (kind, bgr, self._pid)

    def image_jpeg(self):
        img = self.image                                 # one read: another thread may clear it meanwhile
        if img is None:
            return None
        import cv2
        ok, jpg = cv2.imencode(".jpg", img[1], [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpg.tobytes() if ok else None

    def clear_pictures(self, why=""):
        """Drop every camera picture held for the panel (camera off, idle, reset). Calibration numbers stay."""
        if self.busy:
            self.cancel()
        had = self.image is not None or self.scene is not None or self.ai is not None
        self.image, self.scene, self.ai, self.scene_person = None, None, None, (None, False)
        if had and why:
            self.msg = f"camera pictures dropped ({why})"

    def state(self):
        return dict(step=self.step, busy=self.busy, msg=self.msg, err=self.err, progress=round(self.progress, 2),
                    calib=self.calib.summary() if self.calib is not None else None,
                    found=[dict(quad=f["quad"], conf=f["conf"], area=round(f["area"], 3), name=f.get("name", ""), ai=bool(f.get("ai")),
                                wall=bool(f.get("wall")))
                           for f in self.found],
                    made=list(self.made), report=[dict(r) for r in self.report], image=(self.image or (0, 0, 0))[2],
                    ai=self._ai_state(), profile=self.profile, profiles={k: v["label"] for k, v in PROFILES.items()},
                    pose=self.pose, poses=sorted(self.poses), mirrored=self.mirrored, adapt=dict(self.adapt),
                    framing=None if self.framing is None else dict(self.framing, advice=list(self.framing["advice"])))

    def _ai_state(self):
        v = self._vision()
        st = dict(online=v is not None, providers=v.available() if v is not None else [])
        if self.ai is not None:
            st["pending"] = dict(kind=self.ai["kind"], person=self.ai["person"], redacted=self.ai["redacted"],
                                 prompt=self.ai["prompt"], bytes=len(self.ai["png"]))
        return st

    def command(self, c):
        op = c.get("op")
        if op == "check":
            self.check()
        elif op == "calibrate":
            self.calibrate(bool(c.get("gray", True)))
        elif op == "detect":
            self.detect()
        elif op == "pick":
            self.pick(c["x"], c["y"])
        elif op == "drop":
            i = c["i"]
            if self.busy:
                self.err = "busy: wait for the current step"
            elif 0 <= i < len(self.found):
                self.found = self.found[:i] + self.found[i + 1:]
                if self.scene is not None:
                    self._picture("scene", self.scene[0], None)
        elif op == "create":
            self.create(c.get("keep"), c.get("names"))
        elif op == "verify":
            self.verify(c.get("ids"))
        elif op == "ai_polys":
            self.add_polys(c["polys"])
        elif op == "ai_prepare":
            self.ai_prepare(c["kind"], c.get("instruction", ""))
        elif op == "ai_send":
            self.ai_send(bool(c.get("confirm_person")))
        elif op == "ai_discard":
            self.ai_discard()
        elif op == "profile":
            self.set_profile(c["name"])
        elif op == "pose":
            self.set_pose(c["n"])
        elif op == "framing":
            self.framing_live(bool(c.get("on", True)))
        elif op == "cancel":
            self.cancel()
        elif op == "undo":
            self.undo()
        elif op == "reset":
            self.cancel()
            self.scene, self.found, self.made, self.report, self.image, self.msg, self.err = None, [], [], [], None, "", ""
            self.ai, self.scene_person = None, (None, False)
            self.step = "idle"


def validate_command(c):
    """{"cmd": "automap", "op": ...} from the panel -> clean copy (server thread). ValueError if malformed."""
    if not isinstance(c, dict):
        raise ValueError("automap command must be an object")
    op = c.get("op")
    out = {"cmd": "automap", "type": "automap", "op": op}
    if op in ("check", "detect", "cancel", "undo", "reset"):
        return out
    if op == "calibrate":
        out["gray"] = c.get("gray", True) is True
        return out
    if op == "pick":
        for k in ("x", "y"):
            v = c.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                raise ValueError(f"'{k}' must be 0..1")
            out[k] = float(v)
        return out
    if op == "drop":
        v = c.get("i")
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 64:
            raise ValueError("'i' must be an index")
        out["i"] = v
        return out
    if op in ("create", "verify"):
        key = "keep" if op == "create" else "ids"
        v = c.get(key)
        if v is not None:
            if not isinstance(v, list) or len(v) > 64 or not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 1 << 30 for x in v):
                raise ValueError(f"'{key}' must be a list of indices")
            out[key] = list(v)
        if op == "create" and c.get("names") is not None:
            n = c["names"]
            if not isinstance(n, list) or len(n) > 64:
                raise ValueError("'names' must be a list")
            out["names"] = [str(x)[:40] if isinstance(x, str) else "" for x in n]
        return out
    if op == "ai_prepare":
        if c.get("kind") not in ("detect", "verify"):
            raise ValueError("'kind' must be detect or verify")
        out["kind"] = c["kind"]
        t = c.get("instruction", "")
        if not isinstance(t, str) or len(t) > 1000:
            raise ValueError("'instruction' must be a short text")
        out["instruction"] = t[:300]
        return out
    if op == "ai_send":
        out["confirm_person"] = c.get("confirm_person") is True
        return out
    if op == "ai_discard":
        return out
    if op == "profile":
        if c.get("name") not in PROFILES:
            raise ValueError("'name' must be one of " + ", ".join(PROFILES))
        out["name"] = c["name"]
        return out
    if op == "pose":
        if c.get("n") not in (1, 2) or isinstance(c.get("n"), bool):
            raise ValueError("'n' must be 1 or 2")
        out["n"] = c["n"]
        return out
    if op == "framing":
        out["on"] = c.get("on", True) is True
        return out
    if op == "ai_polys":
        p = c.get("polys")
        if not isinstance(p, list) or not 1 <= len(p) <= AI_MAX_POLYS:
            raise ValueError(f"'polys' must be a list of 1..{AI_MAX_POLYS} polygons")
        clean = []
        for poly in p:
            pts = poly.get("points") if isinstance(poly, dict) else poly
            if not isinstance(pts, list) or not 3 <= len(pts) <= 8 or not all(
                    isinstance(q, (list, tuple)) and len(q) == 2 and all(
                        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) < 1e5
                        for v in q) for q in pts):
                raise ValueError("each polygon needs 3..8 finite [x, y] points")
            name = poly.get("name") if isinstance(poly, dict) else ""
            clean.append({"points": [[float(a), float(b)] for a, b in pts],
                          "name": name.strip()[:40] if isinstance(name, str) else ""})
        out["polys"] = clean                             # checked against the camera size in AutoMap.add_polys
        return out
    raise ValueError("automap: unknown op")


# ------------------------------------------------------------------ engine bridge (GL thread side)

ACTIVE = None                                            # the engine's bridge (remote_control's /automap-image)


class EngineBridge:
    """What live_visuals.py holds. tick(mapper) on the GL thread applies the wizard's surfaces commands (validated)
    and refreshes the surfaces snapshot the wizard thread reads; draw() paints the pattern over the output."""

    def __init__(self, ctx, cam, output, controller=None):
        global ACTIVE
        import copy
        self._copy = copy.deepcopy
        self.overlay = AutoMapOverlay(ctx) if ctx is not None else None
        self._q, self._lock = [], threading.Lock()
        self._snap, self._rev, self.output = ([], tuple(output)), None, tuple(output)
        show = self.overlay.show if self.overlay is not None else (lambda img: None)
        shown = self.overlay.shown_id if self.overlay is not None else (lambda: 0)
        self.wizard = AutoMap(show, shown, lambda skip: cam.still(skip), lambda: self._snap, self._enqueue,
                              exposure=getattr(cam, "lock_exposure", None),
                              ready=lambda: _camera_ready(cam),
                              vision=lambda: _vision_for(controller), person=lambda: _person_of(cam),
                              profile=camera_profile(cam), stills=getattr(cam, "allow_stills", None))
        self.cam, self.controller = cam, controller
        self._panics = getattr(controller, "panics", 0)
        ACTIVE = self

    def _enqueue(self, c):
        with self._lock:
            self._q.append(c)

    IDLE_DROP_S = 600.0

    def tick(self, mapper, suppress=False):
        """GL thread, every frame. suppress = blackout held / master down: a running pattern sequence is cancelled
        (the overlay is not drawn either). PANIC cancels too. Camera OFF or 10 idle minutes drop the pictures."""
        import surfaces
        w = self.wizard
        p = getattr(self.controller, "panics", 0)
        if p != self._panics or (suppress and w.busy):
            self._panics = p
            if self.overlay is not None:
                self.overlay.show(None)
            if w.busy:
                w.cancel("stopped by PANIC / BLACKOUT: run the step again when the room is ready")
        if not getattr(self.cam, "on", True) and (w.image is not None or w.scene is not None or w.ai is not None):
            w.clear_pictures("camera off")
        elif not w.busy and w.image is not None and w.clock() - w.touched > self.IDLE_DROP_S:
            w.clear_pictures("10 minutes idle")
        with self._lock:
            q, self._q = self._q, []
        out = tuple(getattr(mapper.set, "output", self.output))
        for c in q:
            try:
                mapper.command(surfaces.validate_cmd(c, out))
            except ValueError as ex:
                self.wizard.err = f"surface edit refused: {ex}"
        rev = getattr(mapper.set, "rev", None)
        if rev != self._rev or q:
            self._rev = rev
            self._snap = (self._copy(mapper.set.surfaces), out)

    def draw(self, target, size, suppress=False):
        if self.overlay is None:
            return False
        if suppress:                                     # blackout / master down: never light the wall with patterns
            self.overlay.show(None)
            return False
        return self.overlay.draw(target, size)

    def command(self, c):
        self.wizard.touched = self.wizard.clock()
        self.wizard.command(c)

    def state(self):
        return self.wizard.state()

    def close(self):
        global ACTIVE
        self.wizard.cancel()
        if self.overlay is not None:
            self.overlay.release()
        if ACTIVE is self:
            ACTIVE = None


def _person_of(cam):
    """(mask, trusted): trusted only with MediaPipe segmentation (the motion fallback misses people standing still)."""
    mask = getattr(cam, "mask", None)
    link = getattr(cam, "link", None)
    backend = str(((getattr(link, "status", None) or {}).get("backend")) or "")
    return (None if mask is None else np.array(mask, np.float32)), backend.startswith("mediapipe") and mask is not None


def _camera_ready(cam):
    if not getattr(cam, "on", True):
        return "switch the CAMERA on (CAMERA tab) and point it at the projection"
    try:
        import cv2
        if not hasattr(cv2, "aruco"):
            return ("this OpenCV has no ArUco markers: install opencv-contrib-python (not opencv-python) in the venv")
    except ImportError:
        return "OpenCV is not installed in the venv (opencv-contrib-python)"
    try:
        dev = int((getattr(cam, "s", None) or {}).get("device", 0))
        for d in getattr(cam, "devices", None) or []:
            if d.get("index") == dev and d.get("ir"):
                return "the selected camera is an infrared camera: it cannot see the projection, pick another device"
    except Exception:
        pass
    return None


def camera_profile(cam):
    """builtin for the laptop's own webcam (by device name, or device 0 when names are unknown), usb otherwise."""
    import re
    try:
        dev = int((getattr(cam, "s", None) or {}).get("device", 0))
        names = {d["index"]: d["name"] for d in (getattr(cam, "devices", None) or [])}
        name = names.get(dev, "")
        if name:
            return "builtin" if re.search(BUILTIN_NAME, name, re.I) else "usb"
        return "builtin" if dev == 0 else "usb"
    except Exception:
        return "builtin"


def _vision_for(controller):
    """AI vision only while the ONLINE plugin is loaded (ONLINE on); None otherwise (nothing imported, nothing sent)."""
    plugin = getattr(controller, "online", None)
    if plugin is None:
        return None
    try:
        from plugins.online.vision import Vision
        return Vision.from_plugin(plugin)
    except Exception:
        return None


def image_jpeg():
    b = ACTIVE
    return b.wizard.image_jpeg() if b is not None else None


# ------------------------------------------------------------------ engine overlay (GL)

class AutoMapOverlay:
    """Draws the current pattern full-frame over the projector output (mapping bypassed). One texture, one draw;
    nothing while there is no pattern."""
    VS = "#version 330\nin vec2 in_pos; out vec2 v_uv; void main(){ v_uv = in_pos*0.5+0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }"
    FS = ("#version 330\nuniform sampler2D pat; in vec2 v_uv; out vec4 f;\n"
          "void main(){ float v = texture(pat, vec2(v_uv.x, 1.0-v_uv.y)).r; f = vec4(v, v, v, 1.0); }")

    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=self.VS, fragment_shader=self.FS)
        self.vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])
        self.tex = None
        self._lock = threading.Lock()
        self._pending = None                              # (image,) set from the wizard thread
        self._active = False
        self.drawn = 0

    def show(self, img):
        with self._lock:
            self._pending = (None if img is None else np.ascontiguousarray(img, np.uint8),)

    def shown_id(self):
        return self.drawn

    def draw(self, target, size):
        with self._lock:
            p, self._pending = self._pending, None
        if p is not None:
            img = p[0]
            if img is None:
                self._active = False
            else:
                h, w = img.shape
                if self.tex is None or self.tex.size != (w, h):
                    if self.tex is not None:
                        self.tex.release()
                    self.tex = self.ctx.texture((w, h), 1, dtype="f1", alignment=1)
                    self.tex.filter = (0x2600, 0x2600)        # NEAREST: crisp stripes / markers
                self.tex.write(img.tobytes())
                self._active = True
        if not self._active or size[0] <= 0 or size[1] <= 0:
            return False
        target.use()
        self.ctx.viewport = (0, 0, size[0], size[1])
        self.tex.use(14)
        self.prog["pat"].value = 14
        self.vao.render(5)                                    # TRIANGLE_STRIP
        if p is not None:
            self.drawn += 1                                   # this frame carries the new pattern
        return True

    def release(self):
        for o in (self.tex, self.vao, self.vbo, self.prog):
            if o is not None:
                try:
                    o.release()
                except Exception:
                    pass
