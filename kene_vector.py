"""Vector generators for two kené STUDIES (GRAMMAR_SPEC A painted, C embroidered): continuous geometry
(polylines, arcs, polygons) built on a worker thread and rasterised into antialiased coverage textures.

Nothing here copies an artwork and nothing here is kené. These are procedural studies of construction rules written
down in research/kene_reference/user_refs/GRAMMAR_SPEC.md (the reference photos themselves are private and are never
read by this code). See kene_real.CREDIT for the credit line every output carries in the panel.

Output of `vector_design(kind, seed, ...)` -> dict with three (h, w, 4) uint8 arrays, rows TOP-DOWN:
    cov   R primary line coverage, G fine fill (meander) coverage, B ground (cloth) coverage, A accent coverage
    prim  RGB = primary colour premultiplied by its coverage (painter's order inside the layer), A = cell id hash
    acc   RGB = accent colour premultiplied by its coverage, A = 0
These four semantic layers can be lit / masked independently (kene_real "gate" params, the skins channel output).

Geometry is in u (one primary line width), origin at the field centre, y up. Each layer is drawn with PIL at SS x the
texture resolution and box-filtered down (true area coverage). The meander is a TRUE space-filling path: the outline
of a random spanning tree on a sub-grid, clipped per cell by everything already drawn (+ a clearance), with rounded
turns, so it never branches, never crosses a line and never leaves stray dots.
"""
import math

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:                                     # the engine still runs; the vector studies fall back to tiles
    Image = ImageDraw = None

ERASE = "erase"
SS = 2                                                  # supersampling of the rasteriser (per axis)

# spec A (painted): cream ground, black-brown ink, rust / ochre accents
A_GROUND, A_INK = (233, 228, 219), (42, 37, 34)
A_RED, A_OCHRE = (174, 60, 35), (207, 167, 41)
# spec C (embroidered): raspberry ground, violet meander, orange outer + yellow inner outline, navy channels, twigs
C_GROUND, C_FILL = (200, 40, 94), (75, 59, 168)
C_ORANGE, C_YELLOW, C_NAVY = (242, 138, 29), (245, 208, 32), (30, 42, 110)
C_TWIGS = ((25, 160, 181), (57, 181, 74), (240, 106, 196), (176, 127, 224), (200, 240, 200))
C_EYE = (24, 96, 48)


# =====================================================================================================================
# rasteriser
# =====================================================================================================================
class Canvas:
    """Five supersampled 'L' layers (prim, fill, ground, acc, block = meander keep-out only) + RGB for prim / acc."""
    LAYERS = ("prim", "fill", "ground", "acc", "block")
    COLOURED = ("prim", "acc")

    def __init__(self, W, H, tex_h, ss=SS):
        self.W, self.H, self.ss = float(W), float(H), int(ss)
        self.h1 = int(tex_h)
        self.w1 = int(round(self.W / self.H * self.h1))
        self.k = self.h1 / self.H                       # px per u (1x)
        self.s = self.k * self.ss                       # px per u (supersampled)
        size = (self.w1 * self.ss, self.h1 * self.ss)
        self.size = size
        self.img = {n: Image.new("L", size, 0) for n in self.LAYERS}
        self.col = {n: Image.new("RGB", size, (0, 0, 0)) for n in self.COLOURED}
        self.d = {n: ImageDraw.Draw(im) for n, im in self.img.items()}
        self.dc = {n: ImageDraw.Draw(im) for n, im in self.col.items()}

    def P(self, pts):
        s, w2, h2 = self.s, self.W / 2.0, self.H / 2.0
        return [((x + w2) * s, (h2 - y) * s) for x, y in pts]

    def _targets(self, layer, colour, block):
        er = colour == ERASE
        out = [(self.d[layer], 0 if er else 255)]
        if layer in self.dc:
            out.append((self.dc[layer], (0, 0, 0) if er else tuple(int(c) for c in colour)))
        if block and layer != "block":
            out.append((self.d["block"], 255))
        return out

    def stroke(self, layer, pts, w, colour=(255, 255, 255), join="round", cap="round", closed=False, block=False):
        if len(pts) < 2:
            return
        ps = self.P(pts)
        wpx = max(1.0, w * self.s)
        h = wpx / 2.0
        for d, fv in self._targets(layer, colour, block):
            if join == "round":
                q = list(ps)
                if closed:
                    q.append(q[0])
                elif cap == "square":
                    q[0] = _ext(q[1], q[0], h)
                    q[-1] = _ext(q[-2], q[-1], h)
                d.line(q, fill=fv, width=max(1, int(round(wpx))), joint="curve")
                ends = q[:1] + q[-1:] if (closed or cap == "round") else []
                for x, y in ends:
                    d.ellipse([x - h, y - h, x + h, y + h], fill=fv)
            else:                                       # miter for orthogonal paths: rectangles with square ends
                n = len(ps) - 1 + (1 if closed else 0)
                for i in range(n):
                    a, b = ps[i], ps[(i + 1) % len(ps)]
                    ex, ey = b[0] - a[0], b[1] - a[1]
                    L = math.hypot(ex, ey)
                    if L < 1e-9:
                        continue
                    ex, ey = ex / L, ey / L
                    e0 = 0.0 if (cap == "butt" and not closed and i == 0) else h
                    e1 = 0.0 if (cap == "butt" and not closed and i == n - 1) else h
                    ax, ay, bx, by = a[0] - ex * e0, a[1] - ey * e0, b[0] + ex * e1, b[1] + ey * e1
                    nx_, ny_ = -ey * h, ex * h
                    d.polygon([(ax + nx_, ay + ny_), (bx + nx_, by + ny_), (bx - nx_, by - ny_), (ax - nx_, ay - ny_)], fill=fv)

    def poly(self, layer, pts, colour=(255, 255, 255), block=False):
        ps = self.P(pts)
        for d, fv in self._targets(layer, colour, block):
            d.polygon(ps, fill=fv)

    def disc(self, layer, c, r, colour=(255, 255, 255), block=False):
        (x, y), = self.P([c])
        rr = r * self.s
        for d, fv in self._targets(layer, colour, block):
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fv)

    def rect(self, layer, x0, y0, x1, y1, colour=(255, 255, 255), block=False):
        self.poly(layer, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], colour, block)

    def band(self, chains, off, hw, colour, layer="prim", join="miter"):
        """A line `off` u to both sides of every chain (an offset curve, hw = half width): the parallel hairline."""
        tmp = Image.new("L", self.size, 0)
        dt = ImageDraw.Draw(tmp)
        for w, v in ((2.0 * (off + hw), 255), (2.0 * (off - hw), 0)):
            for ch in chains:
                ps = self.P(ch)
                wpx = max(1, int(round(w * self.s)))
                dt.line(ps, fill=v, width=wpx, joint="curve")
                for x, y in (ps[0], ps[-1]):
                    r = w * self.s / 2.0
                    dt.ellipse([x - r, y - r, x + r, y + r], fill=v)
        self.img[layer].paste(255, mask=tmp)
        if layer in self.col:
            self.col[layer].paste(tuple(colour), mask=tmp)

    def down(self, name, rgb=False):
        im = (self.col if rgb else self.img)[name]
        return np.asarray(im.resize((self.w1, self.h1), Image.BOX))

    def blocked1x(self):
        """Everything the meander must keep clear of, at 1x (bool)."""
        m = np.zeros((self.h1, self.w1), bool)
        for n in ("prim", "acc", "block"):
            m |= self.down(n) > 8
        return m

    def finish(self, cell_id, border=0.0):
        """border: width (u) of a frame band round the panel that never scrolls (the 'inner' rect moves)."""
        cov = np.stack([self.down(n) for n in ("prim", "fill", "ground", "acc")], axis=-1)
        prim = np.concatenate([self.down("prim", True), cell_id[..., None]], axis=-1)
        acc = np.concatenate([self.down("acc", True), np.zeros((self.h1, self.w1, 1), np.uint8)], axis=-1)
        return dict(cov=np.ascontiguousarray(cov), prim=np.ascontiguousarray(prim), acc=np.ascontiguousarray(acc),
                    W=self.W, H=self.H, inner=(border, border, self.W - border, self.H - border))


def _ext(a, b, h):
    """b moved h further along a->b."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    L = math.hypot(ex, ey) or 1.0
    return (b[0] + ex / L * h, b[1] + ey / L * h)


# =====================================================================================================================
# meander (spec A2.3 / B / C): the outline of a random spanning tree, clipped per cell
# =====================================================================================================================
def _dilate(m, r):
    out = m.copy()
    for k in range(1, r + 1):
        out[k:, :] |= m[:-k, :]
        out[:-k, :] |= m[k:, :]
    m2 = out.copy()
    for k in range(1, r + 1):
        out[:, k:] |= m2[:, :-k]
        out[:, :-k] |= m2[:, k:]
    return out


def _find(par, a):
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a


def meander_paths(ok, rng):
    """ok: (nr, nc) bool cells a path may use. Returns (loops, comp): closed loops as lists of (row, col) lattice
    corners (all turns 90 degrees), and a component id per cell (0 = not usable). Nodes sit on even cells; a random
    spanning forest (Kruskal) joins them through the odd cells between; each loop is the outline of one tree, so
    parallel runs are exactly one cell apart and the path never branches or crosses. Trees of a single node are
    dropped (no stray dots)."""
    nr, nc = ok.shape
    NR, NC = (nr + 1) // 2, (nc + 1) // 2
    okp = np.zeros((2 * NR + 1, 2 * NC + 1), bool)
    okp[:nr, :nc] = ok
    node = okp[0:2 * NR:2, 0:2 * NC:2]
    eh = node[:, :-1] & node[:, 1:] & okp[0:2 * NR:2, 1:2 * NC - 1:2]
    ev = node[:-1, :] & node[1:, :] & okp[1:2 * NR - 1:2, 0:2 * NC:2]
    hr, hc = np.nonzero(eh)
    vr, vc = np.nonzero(ev)
    A = np.concatenate([hr * NC + hc, vr * NC + vc])
    B = np.concatenate([hr * NC + hc + 1, (vr + 1) * NC + vc])
    MR = np.concatenate([2 * hr, 2 * vr + 1])
    MC = np.concatenate([2 * hc + 1, 2 * vc])
    order = rng.permutation(len(A))
    par = list(range(NR * NC))
    deg = np.zeros(NR * NC, int)
    R = np.zeros((2 * NR + 1, 2 * NC + 1), bool)
    for e in order.tolist():
        a, b = _find(par, int(A[e])), _find(par, int(B[e]))
        if a != b:
            par[a] = b
            R[MR[e], MC[e]] = True
            deg[A[e]] += 1
            deg[B[e]] += 1
    R[0:2 * NR:2, 0:2 * NC:2] = node & (deg.reshape(NR, NC) > 0)
    R = R[:nr, :nc]
    # component per usable cell (4-connectivity over `ok`): cell id for gating
    comp = np.zeros((nr, nc), np.int32)
    cpar = {}
    idx = np.arange(nr * nc).reshape(nr, nc)
    cp = list(range(nr * nc))
    for dj, di in ((0, 1), (1, 0)):
        a = ok[:nr - dj, :nc - di] & ok[dj:, di:]
        for x, y in zip(idx[:nr - dj, :nc - di][a].tolist(), idx[dj:, di:][a].tolist()):
            ra, rb = _find(cp, x), _find(cp, y)
            if ra != rb:
                cp[ra] = rb
    roots = np.array([_find(cp, i) for i in range(nr * nc)]).reshape(nr, nc)
    for n, r in enumerate(np.unique(roots[ok]).tolist(), start=1):
        cpar[r] = n
    comp[ok] = [cpar[r] for r in roots[ok].tolist()]
    # outline of R: directed boundary edges (clockwise in image coords), one outgoing edge per corner (no pinches)
    Rp = np.pad(R, 1)
    c = Rp[1:-1, 1:-1]
    up, dn, lf, rt = ~Rp[:-2, 1:-1], ~Rp[2:, 1:-1], ~Rp[1:-1, :-2], ~Rp[1:-1, 2:]
    W1 = nc + 1
    nxt = np.full((nr + 1) * W1, -1, np.int64)
    rr, cc = np.nonzero(c & up)
    nxt[rr * W1 + cc] = rr * W1 + cc + 1
    rr, cc = np.nonzero(c & rt)
    nxt[rr * W1 + cc + 1] = (rr + 1) * W1 + cc + 1
    rr, cc = np.nonzero(c & dn)
    nxt[(rr + 1) * W1 + cc + 1] = (rr + 1) * W1 + cc
    rr, cc = np.nonzero(c & lf)
    nxt[(rr + 1) * W1 + cc] = rr * W1 + cc
    seen = np.zeros(len(nxt), bool)
    loops = []
    nl = nxt.tolist()
    for s in np.nonzero(nxt >= 0)[0].tolist():
        if seen[s]:
            continue
        loop, v = [], s
        while not seen[v] and nl[v] >= 0:
            seen[v] = True
            loop.append(v)
            v = nl[v]
        if len(loop) >= 4:
            loops.append([(q // W1, q % W1) for q in loop])
    return loops, comp


def _round_loop(pts, r, n=4):
    """Closed orthogonal loop (array of corners incl. collinear points) -> point list with every corner rounded."""
    P = np.asarray(pts, float)
    d = np.roll(P, -1, 0) - P
    dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    pd = np.roll(dn, 1, 0)
    turn = np.abs(pd[:, 0] * dn[:, 1] - pd[:, 1] * dn[:, 0]) > 0.5
    C = P[turn]
    if len(C) < 3:
        return [tuple(p) for p in P]
    a = C - np.roll(C, 1, 0)
    b = np.roll(C, -1, 0) - C
    la, lb = np.linalg.norm(a, axis=1, keepdims=True), np.linalg.norm(b, axis=1, keepdims=True)
    ra = np.minimum(r, 0.5 * la)
    rb = np.minimum(r, 0.5 * lb)
    p0 = C - a / la * ra
    p2 = C + b / lb * rb
    out = []
    ts = np.linspace(0.0, 1.0, n)
    for t in ts:
        out.append(((1 - t) ** 2)[..., None] * p0 + (2 * t * (1 - t)) * C + (t * t) * p2)
    Q = np.stack(out, axis=1).reshape(-1, 2)
    return [tuple(q) for q in Q.tolist()]


def draw_meander(cv, pitch, lw, clear, rng, allowed=None, wob=0.0):
    """Fill every free cell of the canvas (not prim / acc / block, minus `clear` u) with the meander. Returns the
    cell-id hash map (h1, w1) uint8 for the gating channel."""
    blocked = cv.blocked1x()
    if allowed is not None:
        blocked |= ~allowed
    bl = _dilate(blocked, max(1, int(round(clear * cv.k))))
    cp = pitch * cv.k
    nr, nc = int(cv.h1 // cp), int(cv.w1 // cp)
    re = np.floor(np.arange(nr + 1) * cp).astype(int)
    ce = np.floor(np.arange(nc + 1) * cp).astype(int)
    sub = bl[:re[-1], :ce[-1]]
    cellb = np.logical_or.reduceat(np.logical_or.reduceat(sub, re[:-1], axis=0), ce[:-1], axis=1)
    loops, comp = meander_paths(~cellb, rng)
    s = cv.ss * cp
    dimg = cv.d["fill"]
    wpx = max(1, int(round(lw * cv.s)))
    for lp in loops:
        q = _round_loop([(c * s, r * s) for r, c in lp], 0.46 * s)
        if wob > 0.0:
            ph = rng.uniform(0, 6.28, 2)
            q = [(x + wob * s * math.sin(y / s * 1.3 + ph[0]), y + wob * s * math.sin(x / s * 1.1 + ph[1])) for x, y in q]
        q.append(q[0])
        dimg.line(q, fill=255, width=wpx, joint="curve")
    # cell ids: grow into the blocked cells (lines) so every pixel has its cell's id, then hash and upsample
    cid = comp.copy()
    for _ in range(4):
        z = cid == 0
        if not z.any():
            break
        g = np.zeros_like(cid)
        g[1:, :] = np.maximum(g[1:, :], cid[:-1, :])
        g[:-1, :] = np.maximum(g[:-1, :], cid[1:, :])
        g[:, 1:] = np.maximum(g[:, 1:], cid[:, :-1])
        g[:, :-1] = np.maximum(g[:, :-1], cid[:, 1:])
        cid[z] = g[z]
    hsh = np.where(cid > 0, (cid.astype(np.int64) * 97 + 13) % 251 + 4, 0).astype(np.uint8)
    ri = np.minimum((np.arange(cv.h1) / cp).astype(int), nr - 1)
    ci = np.minimum((np.arange(cv.w1) / cp).astype(int), nc - 1)
    return hsh[ri][:, ci]


# =====================================================================================================================
# lattice partitions -> boundary chains
# =====================================================================================================================
def boundary_chains(lab, x0, y0, st, frame=False):
    """lab: (ny, nx) labels on cells of size st, row 0 at y0 (BOTTOM), col 0 at x0. Returns (chains, junctions):
    chains = lists of (x, y) corner points in u running between junctions (or closed loops); junctions = points
    where three or more labels meet. frame=True also draws the outer edge of the grid."""
    ny, nx = lab.shape
    if frame:
        L = np.pad(lab, 1, constant_values=-1)
        x0, y0 = x0 - st, y0 - st
        ny, nx = ny + 2, nx + 2
    else:
        L = lab
    adj = {}

    def add(a, b):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    vj, vi = np.nonzero(L[:, :-1] != L[:, 1:])          # vertical edge at x = i + 1, from y = j to j + 1
    for j, i in zip(vj.tolist(), vi.tolist()):
        add((i + 1, j), (i + 1, j + 1))
    hj, hi = np.nonzero(L[:-1, :] != L[1:, :])          # horizontal edge at y = j + 1
    for j, i in zip(hj.tolist(), hi.tolist()):
        add((i, j + 1), (i + 1, j + 1))
    deg = {v: len(n) for v, n in adj.items()}
    used = set()
    chains, junctions = [], []
    for v, dg in deg.items():
        if dg >= 3:
            junctions.append(v)
    starts = [v for v, dg in deg.items() if dg != 2]
    for s in starts:
        for n in adj[s]:
            e = (min(s, n), max(s, n))
            if e in used:
                continue
            used.add(e)
            ch, prev, cur = [s, n], s, n
            while deg[cur] == 2:
                a, b = adj[cur]
                nx_ = b if a == prev else a
                e = (min(cur, nx_), max(cur, nx_))
                if e in used:
                    break
                used.add(e)
                ch.append(nx_)
                prev, cur = cur, nx_
            chains.append(ch)
    for v in deg:                                        # closed loops without junctions
        for n in adj[v]:
            e = (min(v, n), max(v, n))
            if e in used:
                continue
            used.add(e)
            ch, prev, cur = [v, n], v, n
            while cur != v:
                a, b = adj[cur]
                nx_ = b if a == prev else a
                e = (min(cur, nx_), max(cur, nx_))
                if e in used:
                    break
                used.add(e)
                ch.append(nx_)
                prev, cur = cur, nx_
            chains.append(ch)
    tou = lambda p: (x0 + p[0] * st, y0 + p[1] * st)
    return [[tou(p) for p in _corners(ch)] for ch in chains], [tou(p) for p in junctions]


def fillet(ch, r, minlen, n=7):
    """Round the corners of a polyline whose two neighbouring segments are both at least `minlen` long."""
    if len(ch) < 3:
        return list(ch)
    out = [ch[0]]
    for a, b, c in zip(ch, ch[1:], ch[2:]):
        la, lb = math.hypot(b[0] - a[0], b[1] - a[1]), math.hypot(c[0] - b[0], c[1] - b[1])
        if la < minlen or lb < minlen:
            out.append(b)
            continue
        rr = min(r, 0.45 * la, 0.45 * lb)
        p0 = (b[0] + (a[0] - b[0]) / la * rr, b[1] + (a[1] - b[1]) / la * rr)
        p2 = (b[0] + (c[0] - b[0]) / lb * rr, b[1] + (c[1] - b[1]) / lb * rr)
        for k in range(n):
            t = k / (n - 1)
            out.append(((1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * b[0] + t * t * p2[0],
                        (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * b[1] + t * t * p2[1]))
    out.append(ch[-1])
    return out


def _corners(ch):
    """Drop collinear points of a lattice path."""
    if len(ch) <= 2:
        return ch
    out = [ch[0]]
    for a, b, c in zip(ch, ch[1:], ch[2:]):
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            out.append(b)
    out.append(ch[-1])
    return out


def simplify(pts, eps):
    """Douglas-Peucker (keeps the ends)."""
    if len(pts) < 3:
        return list(pts)
    P = np.asarray(pts, float)
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = P[b] - P[a]
        L = np.hypot(*d)
        seg = P[a + 1:b] - P[a]
        dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / L if L > 1e-9 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > eps:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return [tuple(p) for p in P[keep].tolist()]


# =====================================================================================================================
# occupancy (placement of motifs without collisions)
# =====================================================================================================================
class Occ:
    """Grid over a box (res u): lab = which medallion owns the point, occ: 0 free, 1 network line, 2 motif."""
    def __init__(self, x0, y0, x1, y1, labf, res=0.25):
        self.x0, self.y0, self.res = x0, y0, res
        xs = np.arange(x0, x1, res) + res / 2
        ys = np.arange(y0, y1, res) + res / 2
        X, Y = np.meshgrid(xs, ys)
        self.lab = labf(X, Y)
        self.occ = np.zeros(self.lab.shape, np.uint8)
        self._masks = {}

    def ij(self, x, y):
        return int((y - self.y0) / self.res), int((x - self.x0) / self.res)

    def _disc(self, x, y, r):
        j, i = self.ij(x, y)
        k = int(math.ceil(r / self.res))
        j0, j1, i0, i1 = j - k, j + k + 1, i - k, i + k + 1
        if j0 < 0 or i0 < 0 or j1 > self.occ.shape[0] or i1 > self.occ.shape[1]:
            return None
        key = (k, round(r / self.res, 1))
        m = self._masks.get(key)
        if m is None:
            yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
            m = self._masks[key] = (yy ** 2 + xx ** 2) * self.res ** 2 <= r * r
        return (slice(j0, j1), slice(i0, i1)), m

    def free(self, foot, lab, cross=False):
        """cross=True: the motif may cross network lines (an arm through a frame) but not motifs."""
        for x, y, r in foot:
            q = self._disc(x, y, r)
            if q is None:
                return False
            sl, m = q
            o = self.occ[sl][m]
            if ((o == 2) if cross else (o != 0)).any() or (self.lab[sl][m] != lab).any():
                return False
        return True

    def mark(self, foot, v=2):
        for x, y, r in foot:
            q = self._disc(x, y, r)
            if q is None:
                continue
            sl, m = q
            o = self.occ[sl]
            o[m] = np.maximum(o[m], v)

    def ray(self, x, y, dx, dy, lab, maxlen, step=0.25):
        """March from (x, y) along (dx, dy) until a network line; None if blocked / too far / left the cell."""
        t = 0.0
        while t <= maxlen:
            j, i = self.ij(x + dx * t, y + dy * t)
            if not (0 <= j < self.occ.shape[0] and 0 <= i < self.occ.shape[1]):
                return None
            o = self.occ[j, i]
            if o == 1:
                return t
            if o == 2 or self.lab[j, i] != lab:
                return None
            t += step
        return None


def _samples(pts, r, step=0.5):
    out = []
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        L = math.hypot(bx - ax, by - ay)
        n = max(1, int(L / step))
        for k in range(n + 1):
            t = k / n
            out.append((ax + (bx - ax) * t, ay + (by - ay) * t, r))
    return out


def _mir(pts, sx, sy, cx=0.0, cy=0.0):
    return [(cx + sx * x, cy + sy * y) for x, y in pts]


# =====================================================================================================================
# A: painted kené study
# =====================================================================================================================
def _arrow(tip, d, length, half):
    """Filled triangle: tip, pointing along unit d."""
    tx, ty = tip
    dx, dy = d
    bx, by = tx - dx * length, ty - dy * length
    return [(tx, ty), (bx - dy * half, by + dx * half), (bx + dy * half, by - dx * half)]


def _spiral(c, R, theta0, sigma, sweep, shrink=0.4, n=48):
    pts = []
    for k in range(n + 1):
        f = k / n
        th = theta0 + sigma * sweep * f
        r = R * (1.0 - shrink * f * f)
        pts.append((c[0] + r * math.cos(th), c[1] + r * math.sin(th)))
    return pts


def painted_design(seed, H=140.0, W=214.0, tex_h=1280, weight=0.5):
    rng = np.random.default_rng([int(seed) % 100003, 51, int(H), int(W)])
    cv = Canvas(W, H, tex_h)
    lw = 0.9 + 0.3 * weight                             # primary line width (u)
    Lx2 = 0.38 * H * rng.uniform(0.94, 1.06)            # rhombic lattice: centres at (i Lx2, j Ly2), i + j even
    Ly2 = 0.27 * H * rng.uniform(0.94, 1.08)
    oct_c = rng.uniform(1.35, 1.7)                      # octagon metric: orthogonal runs + diagonal (stepped) runs
    delta = rng.uniform(0.16, 0.30)                     # cross medallions (i even) are bigger than bowtie ones
    st = rng.uniform(2.5, 3.3)                          # staircase step (u)
    hw, hh = W / 2.0 + Lx2 * 1.2, H / 2.0 + Ly2 * 1.2

    def centres():
        out = []
        for i in range(-int(hw / Lx2) - 2, int(hw / Lx2) + 3):
            for j in range(-int(hh / Ly2) - 2, int(hh / Ly2) + 3):
                if (i + j) % 2 == 0:
                    out.append((i, j))
        return out
    C = centres()

    def labf(X, Y):
        best = np.full(X.shape, 1e9)
        lab = np.zeros(X.shape, np.int32)
        for n, (i, j) in enumerate(C):
            a, b = np.abs(X - i * Lx2) / Lx2, np.abs(Y - j * Ly2) / Ly2
            if a.min() > 3.0 or b.min() > 3.0:
                continue
            m = np.maximum(np.maximum(a, b), (a + b) / oct_c) - (delta if i % 2 == 0 else 0.0)
            t = m < best - 1e-9
            best[t], lab[t] = m[t], n
        return lab
    # the partition on the staircase grid (symmetric about 0: cell edges at multiples of st)
    nx2, ny2 = int(math.ceil(hw / st)), int(math.ceil(hh / st))
    xs = (np.arange(-nx2, nx2) + 0.5) * st
    ys = (np.arange(-ny2, ny2) + 0.5) * st
    X, Y = np.meshgrid(xs, ys)
    lab = labf(X, Y)
    chains, junc = boundary_chains(lab, -nx2 * st, -ny2 * st, st)
    look = lambda X_, Y_: lab[np.clip(np.floor(Y_ / st).astype(int) + ny2, 0, 2 * ny2 - 1),
                              np.clip(np.floor(X_ / st).astype(int) + nx2, 0, 2 * nx2 - 1)]
    occ = Occ(-hw, -hh, hw, hh, look)
    for ch in chains:
        occ.mark(_samples(ch, lw / 2 + 0.15, 0.25), 1)

    # ---- the network: bold primary along every boundary (sharp steps, but a big turn between two long runs is
    #      rounded, radius ~2-3 u, as the line wraps a lobe), a fine hairline both sides (A2.1 / A2.2, A3)
    rr = rng.uniform(2.0, 3.0)
    drawn = [fillet(ch, rr, 1.6 * st) for ch in chains]
    for ch in drawn:
        cv.stroke("prim", ch, lw, A_INK, join="miter", cap="square")
    cv.band(drawn, 0.55 * lw + 0.75, 0.13, A_INK)
    for ch in chains:
        occ.mark(_samples(ch, lw / 2 + 0.95, 0.25), 1)

    acc_ops = []
    # junction gems where channels meet (A4), red / black alternating by position
    for n, (x, y) in enumerate(sorted(junc)):
        g = 0.95 * lw + 0.35
        col = A_INK if (round(abs(x) / st) + round(abs(y) / st)) % 3 == 0 else A_RED
        acc_ops.append(("rect", (x - g, y - g, x + g, y + g), col))
    # red channel segments: some straight runs of the network become red bars (A4); mirrored by the field's D2
    runs = []
    for ch in chains:
        for a, b in zip(ch, ch[1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            if L >= 2.5 * st and a[0] >= -1e-6 and a[1] >= -1e-6 and b[0] >= -1e-6 and b[1] >= -1e-6:
                runs.append((a, b, L))
    rng.shuffle(runs)
    for a, b, L in runs[:max(3, len(runs) // 2)]:
        ex, ey = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        m = min(L * 0.12, 0.6 * st)
        seg = [(a[0] + ex * m, a[1] + ey * m), (b[0] - ex * m, b[1] - ey * m)]
        for sx in (1, -1):
            for sy in (1, -1):
                acc_ops.append(("bar", _mir(seg, sx, sy), A_RED))
    # ochre staircase blocks lying along stepped (diagonal) runs, inside a cell, 3-4 steps long (A4)
    stairs = []
    for ch in chains:
        if len(ch) < 6:
            continue
        P = np.asarray(ch)
        d = np.diff(P, axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        short = L <= 1.01 * st
        k = 0
        while k < len(short):
            if short[k]:
                e = k
                while e < len(short) and short[e]:
                    e += 1
                if e - k >= 5:
                    stairs.append(P[k:e + 1])
                k = e
            else:
                k += 1
    for S in stairs:
        mid = len(S) // 2
        n_st = int(rng.integers(6, 10))
        seg = S[max(0, mid - n_st // 2):mid + (n_st + 1) // 2 + 1]
        if seg[:, 0].min() < 0 or seg[:, 1].min() < 0:
            continue
        D = seg[-1] - seg[0]
        nrm = np.array([-D[1], D[0]]) / (np.hypot(*D) + 1e-9)
        nd = np.sign(nrm) * (0.5 * lw + 1.55)
        for side in (1, -1):
            off = seg + side * nd
            pts = [tuple(p) for p in off.tolist()]
            jq, iq = occ.ij(*pts[len(pts) // 2])
            if not (0 <= jq < occ.lab.shape[0] and 0 <= iq < occ.lab.shape[1]):
                continue
            lab0 = occ.lab[jq, iq]
            if not occ.free([(x, y, 0.65) for x, y, _r in _samples(pts, 0, 0.5)], lab0):
                continue
            occ.mark(_samples(pts, 1.4, 0.5))
            for sx in (1, -1):
                for sy in (1, -1):
                    acc_ops.append(("stair", _mir(pts, sx, sy), A_OCHRE))
            break

    # ---- motifs inside the medallions: one template per medallion type, built at a reference centre, stamped
    #      on every lattice point of that type, D2 mirrored (A3)
    templates = {0: [], 1: []}
    for typ, (ci, cj) in ((0, (0, 0)), (1, (1, 1))):
        cx, cy = ci * Lx2, cj * Ly2
        L0 = C.index((ci, cj))
        T = templates[typ]

        def put(ops, foot, mirrors=((1, 1), (-1, 1), (1, -1), (-1, -1)), cross=False, markv=2, _T=T, _cx=cx, _cy=cy, _L0=L0):
            feet = [[(_cx + sx * x, _cy + sy * y, r) for x, y, r in foot] for sx, sy in mirrors]
            for f in feet:
                if not occ.free(f, _L0, cross):
                    return False
            for f in feet:
                occ.mark([(x, y, max(0.3, r - 1.3)) for x, y, r in f], markv)
            _T.append((ops, mirrors))
            return True

        def reach(x, y, dx, dy, maxlen=24.0):
            return occ.ray(cx + x, cy + y, dx, dy, L0, maxlen)
        ext_x, ext_y = reach(0, 0, 1, 0, 80.0) or 8.0, reach(0, 0, 0, 1, 80.0) or 8.0
        mg = 1.45                                        # clearance margin round a motif (room for the meander)

        pcache = {}

        def pick(near=False, _cx=cx, _cy=cy, _L0=L0, _pc=pcache):
            """A random free point of this medallion's first quadrant (local coords); near = next to a network line.
            The candidate list is refreshed every 40 calls (a stale point is rejected by occ.free anyway)."""
            c = _pc.get(near)
            if c is None or c[3] >= 40:
                j0, i0 = occ.ij(_cx + 0.3, _cy + 0.3)
                j1, i1 = occ.ij(_cx + ext_x + 14, _cy + ext_y + 14)
                j1, i1 = min(j1, occ.occ.shape[0]), min(i1, occ.occ.shape[1])
                m = (occ.lab[j0:j1, i0:i1] == _L0) & (occ.occ[j0:j1, i0:i1] == 0)
                if near:                                 # right next to a network line
                    m &= _dilate(occ.occ[j0:j1, i0:i1] == 1, 3)
                jj, ii = np.nonzero(m)
                c = [j0 + jj, i0 + ii, None, 0]
            c[3] += 1
            _pc[near] = c
            if not len(c[0]):
                return None
            k = int(rng.integers(0, len(c[0])))
            return (occ.x0 + (c[1][k] + 0.5) * occ.res - _cx, occ.y0 + (c[0][k] + 0.5) * occ.res - _cy)
        if typ == 0:
            # stepped Greek cross of filled squares round an open centre square; thin arms out to arrowheads
            s0 = 1.45 * lw
            sq = [(i, j) for i in range(-2, 3) for j in range(-2, 3) if abs(i) + abs(j) <= 2 and (i, j) != (0, 0)]
            ops = [("rect", (i * s0 - s0 / 2, j * s0 - s0 / 2, i * s0 + s0 / 2, j * s0 + s0 / 2), A_INK, "prim") for i, j in sq]
            ops.append(("loop", [(-0.28 * s0, -0.28 * s0), (0.28 * s0, -0.28 * s0), (0.28 * s0, 0.28 * s0), (-0.28 * s0, 0.28 * s0)], 0.22, A_INK))
            put(ops, [(0, 0, 2.6 * s0 + mg)], mirrors=((1, 1),))
            # frame: a rounded rectangle round the cross
            fx, fy = min(0.42 * ext_x, 9.0 + 2.5 * rng.random()), min(0.42 * ext_y, 7.0 + 2.0 * rng.random())
            fr = 0.45 * min(fx, fy)
            q = _rrect(fx, fy, fr)
            if put([("loop", q, lw, A_INK)], [(x, y, lw / 2 + 0.9) for x, y, _ in _samples(q + q[:1], 0, 0.5)], mirrors=((1, 1),), markv=1):
                occ.mark(_samples(q + q[:1], lw / 2 + 0.2, 0.25), 1)
            for ax_, (ex_, ey_) in ((0, (1, 0)), (1, (0, 1))):
                ext = ext_x if ax_ == 0 else ext_y
                x0_ = 2.5 * s0
                L = min(ext * rng.uniform(0.55, 0.7), x0_ + 16.0)
                tip = (ex_ * L, ey_ * L)
                arm = [(ex_ * x0_, ey_ * x0_), (ex_ * (L - 1.6), ey_ * (L - 1.6))]
                ops = [("line", arm, 0.7 * lw, A_INK), ("tri", _arrow(tip, (ex_, ey_), 2.0, 1.15), A_INK)]
                blk = L + 1.0
                ops.append(("rect", (ex_ * blk - 0.7 - ey_ * 0.0, ey_ * blk - 0.7, ex_ * (blk + 1.2) + 0.7 * ey_, ey_ * (blk + 1.2) + 0.7 * ex_), A_RED, "acc"))
                foot = [(x, y, 1.2 + mg * 0.6) for x, y, _ in _samples([arm[0], tip, (ex_ * (blk + 1.4), ey_ * (blk + 1.4))], 0, 0.5)]
                mir = ((1, 1), (-1, 1)) if ax_ == 0 else ((1, 1), (1, -1))
                put(ops, [f for f in foot if abs(f[0]) + abs(f[1]) > 2.6 * s0 + mg + 1.4], mirrors=mir, cross=True)
        else:
            if rng.random() < 0.6:                        # bowtie / double axe round the centre (A3)
                bw, bh = rng.uniform(3.6, 4.6), rng.uniform(2.6, 3.2)
                q = _bowtie(bw, bh)
                ops = [("loop", q, lw, A_INK), ("fillb", q)]
                ops.append(("rect", (-0.9 * bw, -0.45, 0.9 * bw, 0.45), A_RED, "acc"))
                put(ops, [(0, 0, bw + mg)], mirrors=((1, 1),))
                x0_ = bw + 0.3
            else:                                        # red bar crossed by a vertical arm with small triangles
                ops = [("rect", (-3.2, -0.55, 3.2, 0.55), A_RED, "acc"), ("line", [(0, -3.4), (0, 3.4)], 0.8 * lw, A_INK),
                       ("tri", _arrow((0, 1.9), (0, -1), 1.3, 0.9), A_INK), ("tri", _arrow((0, -1.9), (0, 1), 1.3, 0.9), A_INK)]
                put(ops, [(0, 0, 3.6 + mg)], mirrors=((1, 1),))
                x0_ = 3.4
            # horizontal arms out to arrow terminals, an ochre block, and a vertical arm with a small T
            L = min(ext_x * rng.uniform(0.6, 0.75), x0_ + 14.0)
            arm = [(x0_, 0), (L - 1.6, 0)]
            ops = [("line", arm, 0.75 * lw, A_INK), ("tri", _arrow((L, 0), (1, 0), 2.0, 1.15), A_INK),
                   ("rect", (x0_ + 1.2, -0.8, x0_ + 2.6, 0.8), A_OCHRE, "acc")]
            put(ops, [(x, 0, 1.2 + 0.9) for x in np.arange(x0_ + 3.2, L + 0.5, 0.5)], mirrors=((1, 1), (-1, 1)), cross=True)
            Lv = min(ext_y * rng.uniform(0.5, 0.65), 11.0)
            if Lv > 6.0:
                ops = [("line", [(0, 3.0), (0, Lv)], 0.75 * lw, A_INK), ("line", [(-1.3, Lv), (1.3, Lv)], 0.75 * lw, A_INK),
                       ("rect", (-0.8, Lv + 0.9, 0.8, Lv + 2.0), A_OCHRE, "acc")]
                put(ops, [(0, y, 1.6 + 0.9) for y in np.arange(6.4, Lv + 2.5, 0.5)], mirrors=((1, 1), (1, -1)), cross=True)
        # bulbs: C-loops curling back to an arrowhead, joined to the network by a straight stem (A3), mirrored pairs
        nb = 0
        for _try in range(260):
            if nb >= (3 if typ == 0 else 2):
                break
            R = rng.uniform(2.1, 3.0)
            q = pick()
            if q is None:
                break
            x, y = q
            t = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(0, 4))]
            side = 1 if rng.random() < 0.5 else -1
            px_, py_ = -t[1] * side, t[0] * side           # start point S = c + R * perp, tangent t there
            S = (x + R * px_, y + R * py_)
            th0 = math.atan2(py_, px_)
            sigma = 1 if (-math.sin(th0) * t[0] + math.cos(th0) * t[1]) > 0 else -1
            sweep = math.radians(rng.uniform(290, 330))
            sp = _spiral((x, y), R, th0, sigma, sweep, shrink=0.42)
            stem = reach(S[0], S[1], -t[0], -t[1], 14.0)
            if stem is None or stem < 1.0:
                continue
            N = (S[0] - t[0] * stem, S[1] - t[1] * stem)
            ex_, ey_ = sp[-1][0] - sp[-3][0], sp[-1][1] - sp[-3][1]
            el = math.hypot(ex_, ey_) or 1.0
            d = (ex_ / el, ey_ / el)
            tip = (sp[-1][0] + d[0] * 1.3, sp[-1][1] + d[1] * 1.3)
            path = [N] + sp
            foot = [(px, py, lw / 2 + mg) for px, py, _ in _samples(sp, 0, 0.4)]
            foot += [(px, py, lw / 2 + 0.6) for px, py, _ in _samples([S, (N[0] + t[0] * 1.6, N[1] + t[1] * 1.6)], 0, 0.4)]
            foot.append((tip[0], tip[1], 1.4))
            if min(f[0] for f in foot) - 0.3 < 0 or min(f[1] for f in foot) - 0.3 < 0:
                continue                                 # stays in its quadrant (the mirror images never touch)
            ops = [("line", path, lw, A_INK), ("tri", _arrow(tip, d, 1.6, 1.05), A_INK)]
            if put(ops, foot):
                nb += 1
        # stepped arms from the network inward, ending in an L block (red / ochre) or a T-bar
        na = 0
        for _try in range(300):
            if na >= 5:
                break
            q = pick()
            if q is None:
                break
            x, y = q
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            d1 = dirs[int(rng.integers(0, 4))]
            stem = reach(x, y, -d1[0], -d1[1], 10.0)
            if stem is None or stem < 0.5:
                continue
            start = (x - d1[0] * stem, y - d1[1] * stem)
            d2 = (d1[1], d1[0]) if rng.random() < 0.5 else (-d1[1], -d1[0])
            run, rise = rng.uniform(1.8, 3.0), rng.uniform(1.6, 2.4)
            pts, p = [start, (x, y)], (x, y)
            for _k in range(int(rng.integers(2, 4))):
                p = (p[0] + d2[0] * rise, p[1] + d2[1] * rise)
                pts.append(p)
                p = (p[0] + d1[0] * run, p[1] + d1[1] * run)
                pts.append(p)
            e = d1
            end = pts[-1]
            ops = [("pline", pts, lw, A_INK)]
            kind = rng.random()
            if kind < 0.75:                              # L block just past the end (A4)
                col = A_RED if rng.random() < 0.5 else A_OCHRE
                b0 = (end[0] + e[0] * 0.9, end[1] + e[1] * 0.9)
                l1 = [b0, (b0[0] + e[0] * 2.0, b0[1] + e[1] * 2.0)]
                l2 = [l1[1], (l1[1][0] + d2[0] * 1.4, l1[1][1] + d2[1] * 1.4)]
                ops += [("bar", l1, 1.1, col), ("bar", l2, 1.1, col)]
                tail = l2[1]
            elif kind < 0.85:                            # T-bar / hammer
                bar = [(end[0] - d2[0] * 1.4, end[1] - d2[1] * 1.4), (end[0] + d2[0] * 1.4, end[1] + d2[1] * 1.4)]
                ops += [("bar", bar, lw, A_INK)]
                tail = end
            else:                                        # arrow terminal
                ops[0] = ("pline", pts[:-1] + [(end[0] - e[0] * 1.4, end[1] - e[1] * 1.4)], lw, A_INK)
                ops += [("tri", _arrow((end[0] + e[0] * 0.5, end[1] + e[1] * 0.5), e, 1.9, 1.1), A_INK)]
                tail = end
            body = pts[1:] + [tail]
            foot = [(px, py, lw / 2 + mg) for px, py, _ in _samples(body, 0, 0.4)]
            foot += [(px, py, lw / 2 + 0.6) for px, py, _ in _samples([(x, y), (start[0] + d1[0] * 1.6, start[1] + d1[1] * 1.6)], 0, 0.4)]
            if min(f[0] for f in foot) - 0.3 < 0 or min(f[1] for f in foot) - 0.3 < 0:
                continue
            if put(ops, foot):
                na += 1

        # free-standing L / step accents tucked against the network (red / ochre, A4: they carry most of the colour)
        nl = 0
        for _try in range(900):
            if nl >= 8:
                break
            q = pick(near=True)
            if q is None:
                break
            x, y = q
            best = None
            for d1 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                gp = reach(x, y, d1[0], d1[1], 1.2)
                if gp is not None and (best is None or gp < best[1]):
                    best = (d1, gp)
            if best is None:
                continue
            d1, gap = best
            gap -= 0.1
            d2 = (d1[1], d1[0]) if rng.random() < 0.5 else (-d1[1], -d1[0])
            a0 = (x + d1[0] * (gap - 0.72), y + d1[1] * (gap - 0.72))      # the long leg runs along the line
            L1, L2 = rng.uniform(2.2, 3.4), rng.uniform(1.2, 2.0)
            leg1 = [a0, (a0[0] + d2[0] * L1, a0[1] + d2[1] * L1)]
            leg2 = [leg1[1], (leg1[1][0] - d1[0] * L2, leg1[1][1] - d1[1] * L2)]
            if rng.random() < 0.4:                       # a 2-step staircase block instead of an L
                leg2 = [leg1[1], (leg1[1][0] - d1[0] * 1.2, leg1[1][1] - d1[1] * 1.2),
                        (leg1[1][0] - d1[0] * 1.2 + d2[0] * 1.8, leg1[1][1] - d1[1] * 1.2 + d2[1] * 1.8)]
            col = A_RED if rng.random() < 0.72 else A_OCHRE
            pts = leg1 + leg2[1:]
            foot = [(px, py, 0.75 + 1.0) for px, py, _ in _samples(pts, 0, 0.4)]
            foot = [f for f in foot if math.hypot(f[0] - a0[0], f[1] - a0[1]) > 0.1]
            # the leg along the line may sit close to it: check that side with a smaller radius
            foot = [(fx, fy, min(r, 0.6 + max(0.0, -((fx - a0[0]) * d1[0] + (fy - a0[1]) * d1[1])))) for fx, fy, r in foot]
            if min(f[0] for f in foot) - 0.3 < 0 or min(f[1] for f in foot) - 0.3 < 0:
                continue
            if put([("bar", pts, 1.3, col)], foot):
                nl += 1

    # ---- draw: stamp templates on every lattice point of their type
    for n, (i, j) in enumerate(C):
        cx, cy = i * Lx2, j * Ly2
        if abs(cx) > W / 2 + Lx2 or abs(cy) > H / 2 + Ly2:
            continue
        for ops, mirrors in templates[i % 2]:
            for sx, sy in mirrors:
                for op in ops:
                    _draw_op(cv, op, sx, sy, cx, cy)
    for op in acc_ops:
        kind, geo, col = op
        if kind == "rect":
            cv.rect("acc", *geo, colour=col)
        elif kind == "bar":
            cv.stroke("acc", geo, lw * 1.05, col, join="miter", cap="butt")
        elif kind == "stair":
            cv.stroke("acc", geo, 1.5, col, join="miter", cap="butt")
    # notched edge: a fine sawtooth along one side of the longest straight runs (A3, sparingly)
    longest = sorted(runs, key=lambda r: -r[2])[:1]
    for a, b, L in longest:
        ex, ey = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        nx_, ny_ = -ey, ex
        k = 0.55
        pts = []
        for m in np.arange(0.2 * L, 0.8 * L, k):
            base = (a[0] + ex * m, a[1] + ey * m)
            o = lw / 2
            pts = [(base[0] + nx_ * o, base[1] + ny_ * o), (base[0] + nx_ * (o + 0.5) + ex * k / 2, base[1] + ny_ * (o + 0.5) + ey * k / 2),
                   (base[0] + nx_ * o + ex * k, base[1] + ny_ * o + ey * k)]
            for sx in (1, -1):
                for sy in (1, -1):
                    cv.poly("prim", _mir(pts, sx, sy), A_INK)
    cv.rect("ground", -W / 2, -H / 2, W / 2, H / 2)
    cell = draw_meander(cv, pitch=rng.uniform(1.0, 1.12), lw=0.17 + 0.05 * weight, clear=0.32, rng=rng)
    return cv.finish(cell)


def _rrect(fx, fy, r, n=8):
    pts = []
    for cx, cy, a0 in ((fx - r, fy - r, 0.0), (-fx + r, fy - r, 90.0), (-fx + r, -fy + r, 180.0), (fx - r, -fy + r, 270.0)):
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _bowtie(bw, bh, n=10):
    """Two concave-sided trapezoids mirrored about x = 0 (a double axe on its side)."""
    pts = [(-bw, -bh), (-bw, bh)]
    for k in range(1, n):
        t = k / n
        x = -bw + 2 * bw * t
        pts.append((x, bh - (bh * 0.68) * math.sin(math.pi * t)))
    pts += [(bw, bh), (bw, -bh)]
    for k in range(1, n):
        t = k / n
        x = bw - 2 * bw * t
        pts.append((x, -bh + (bh * 0.68) * math.sin(math.pi * t)))
    return pts


def _draw_op(cv, op, sx, sy, cx, cy):
    kind = op[0]
    M = lambda pts: _mir(pts, sx, sy, cx, cy)
    if kind == "rect":
        x0, y0, x1, y1 = op[1]
        cv.poly(op[3] if len(op) > 3 else "prim", M([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), op[2])
    elif kind == "loop":
        cv.stroke("prim", M(op[1]), op[2], op[3], join="round", closed=True)
    elif kind == "line":
        cv.stroke("prim", M(op[1]), op[2], op[3], join="round", cap="round")
    elif kind == "pline":
        cv.stroke("prim", M(op[1]), op[2], op[3], join="miter", cap="square")
    elif kind == "bar":
        cv.stroke("acc" if op[3] != A_INK else "prim", M(op[1]), op[2], op[3], join="miter", cap="square")
    elif kind == "tri":
        cv.poly("prim", M(op[1]), op[2])
    elif kind == "fillb":
        cv.poly("block", M(op[1]), (255, 255, 255))


# =====================================================================================================================
# C: embroidered kené study
# =====================================================================================================================
def embroidered_design(seed, H=110.0, W=168.0, tex_h=1280, weight=0.5):
    rng = np.random.default_rng([int(seed) % 100003, 73, int(H), int(W)])
    cv = Canvas(W, H, tex_h)
    bw = 0.10 * H                                       # border band width
    ix, iy = W / 2 - bw, H / 2 - bw                     # inner field half size
    # ---- amoeba cells: clusters of 1-3 lobes, octagon metric (45 degree chamfers), bilateral mirror
    sp = rng.uniform(11.5, 13.5)
    seeds = []
    for gy in np.arange(-iy + sp * 0.5, iy, sp * 0.9):
        for gx in np.arange(sp * 0.25, ix, sp):
            x = gx + rng.uniform(-0.35, 0.35) * sp + (0.5 * sp if int(round(gy / sp)) % 2 else 0.0)
            y = gy + rng.uniform(-0.3, 0.3) * sp
            if x > ix + 1 or abs(y) > iy + 1:
                continue
            lobes = [(x, y, 0.0)]
            for _ in range(int(rng.choice([1, 1, 2, 2, 3]))):
                a = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(3.2, 5.4)
                lobes.append((x + r * math.cos(a), y + r * math.sin(a), rng.uniform(-0.8, 0.6)))
            seeds.append(lobes)
    cells, partner = [], []
    for lobes in seeds:
        if abs(lobes[0][0]) < 0.3 * sp:
            cells.append(lobes + [(-x, y, w) for x, y, w in lobes])
            partner.append(len(cells) - 1)
        else:
            cells.append(lobes)
            cells.append([(-x, y, w) for x, y, w in lobes])
            partner += [len(cells) - 1, len(cells) - 2]
    st = 0.5
    nx2, ny2 = int(math.ceil(ix / st)), int(math.ceil(iy / st))
    xs = (np.arange(-nx2, nx2) + 0.5) * st
    ys = (np.arange(-ny2, ny2) + 0.5) * st
    X, Y = np.meshgrid(xs, ys)
    best = np.full(X.shape, 1e9)
    lab = np.zeros(X.shape, np.int32)
    q = rng.uniform(1.25, 1.45)
    for n, lobes in enumerate(cells):
        d = np.full(X.shape, 1e9)
        for x, y, w in lobes:
            a, b = np.abs(X - x), np.abs(Y - y)
            d = np.minimum(d, np.maximum(np.maximum(a, b), (a + b) / q) - w)
        t = d < best
        best[t], lab[t] = d[t], n
    for l in np.unique(lab[:, nx2]).tolist():            # a cell and its mirror image meeting on the axis are ONE cell
        if partner[l] != l:                              # (no straight seam down the middle)
            lab[lab == partner[l]] = l
    chains, junc = boundary_chains(lab, -nx2 * st, -ny2 * st, st, frame=True)
    chains = [simplify(ch, 0.45) for ch in chains]
    # the frame edge of the inner field (its own chain set: outlines only on the inside)
    c_half, wo = 0.55, 0.42 + 0.12 * weight             # channel half width; width of each outline (orange, yellow)
    for w_, col in ((2 * (c_half + 2 * wo), C_YELLOW), (2 * (c_half + wo), C_ORANGE)):
        for ch in chains:
            cv.stroke("prim", ch, w_, col, join="round", cap="round")
    for ch in chains:
        cv.stroke("prim", ch, 2 * c_half, ERASE, join="round", cap="round")
        cv.stroke("block", ch, 2 * c_half + 0.2, (255, 255, 255), join="round", cap="round")
    # navy channel lines on most chains (a network with free ends), with stitched look added in the shader
    inner = [ch for ch in chains if not _on_frame(ch, ix, iy)]
    navy = [ch for ch in inner if rng.random() < 0.62]
    for ch in navy:
        cv.stroke("prim", ch, 0.8, C_NAVY, join="round", cap="round")
    # ---- border band (C): cream zig-zag, purple band, diamond chain on white with pink crosses and a lime stepped
    #      outline, navy stripe, crimson stripe, yellow-green overcast zig-zag edge
    _emb_border(cv, W, H, bw, rng)
    # twigs: short coloured T-bars and crosses at channel junctions (2-3 u, perpendicular end bars)
    jun = [p for p in junc if abs(p[0]) < ix - 2 and abs(p[1]) < iy - 2 and p[0] >= 0]
    rng.shuffle(jun)
    for (x, y) in jun[:max(3, len(jun) // 3)]:
        col = C_TWIGS[int(rng.integers(0, len(C_TWIGS)))]
        ops = []
        if rng.random() < 0.35:                          # a cross
            L = rng.uniform(1.8, 2.6)
            dg = rng.random() < 0.5
            for a in (0, 90):
                ang = math.radians(a + (45 if dg else 0))
                ex, ey = math.cos(ang), math.sin(ang)
                ops.append([(x - ex * L, y - ey * L), (x + ex * L, y + ey * L)])
                for s in (1, -1):
                    bx, by = x + s * ex * L, y + s * ey * L
                    ops.append([(bx - ey * 0.8, by + ex * 0.8), (bx + ey * 0.8, by - ex * 0.8)])
        else:                                            # a T-bar running out along one of the channels
            ch = min(chains, key=lambda c: min(math.hypot(px - x, py - y) for px, py in (c[0], c[-1])))
            a, b = (ch[0], ch[1]) if math.hypot(ch[0][0] - x, ch[0][1] - y) < 0.1 else (ch[-1], ch[-2])
            ex, ey = b[0] - a[0], b[1] - a[1]
            L0 = math.hypot(ex, ey) or 1.0
            ex, ey = ex / L0, ey / L0
            L = min(rng.uniform(2.2, 3.2), 0.8 * L0)
            tip = (x + ex * L, y + ey * L)
            ops.append([(x, y), tip])
            ops.append([(tip[0] - ey * 1.1, tip[1] + ex * 1.1), (tip[0] + ey * 1.1, tip[1] - ex * 1.1)])
        for sx in ((1, -1) if x > 0.5 else (1,)):
            for seg in ops:
                cv.stroke("acc", _mir(seg, sx, 1), 0.6, col, join="round", cap="round")
    # eyes: a navy stub from the channel into some cells ending in a dark green dot (C)
    cy_, cx_ = np.nonzero(np.ones_like(lab, bool))
    for n in range(0, len(cells), 2):
        if rng.random() > 0.4:
            continue
        m = lab == n
        if m.sum() < 60:
            continue
        jj, ii = np.nonzero(m)
        gx, gy = xs[ii].mean(), ys[jj].mean()
        if gx < 0:
            continue
        # stub from the cell's edge (along the axis to the nearest edge) toward the centre
        best_d, best_e = None, None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            t = 0.0
            while True:
                px, py = gx + dx * t, gy + dy * t
                i_, j_ = int((px + nx2 * st) / st), int((py + ny2 * st) / st)
                if not (0 <= i_ < lab.shape[1] and 0 <= j_ < lab.shape[0]) or lab[j_, i_] != n:
                    break
                t += st
            if best_d is None or t < best_d:
                best_d, best_e = t, (dx, dy)
        if best_d is None or best_d < 3.0:
            continue
        dx, dy = best_e
        e0 = (gx + dx * best_d, gy + dy * best_d)
        e1 = (gx + dx * min(1.2, best_d * 0.3), gy + dy * min(1.2, best_d * 0.3))
        for sx in (1, -1):
            cv.stroke("prim", _mir([e0, e1], sx, 1), 0.6, C_NAVY, join="round", cap="round")
            cv.disc("acc", (sx * e1[0], e1[1]), 0.8, C_EYE)
    cv.rect("ground", -W / 2, -H / 2, W / 2, H / 2)
    allowed = np.zeros((cv.h1, cv.w1), bool)
    m0 = int(round((W / 2 - ix) * cv.k)) + 1
    allowed[m0:cv.h1 - m0, m0:cv.w1 - m0] = True
    cell = draw_meander(cv, pitch=rng.uniform(0.6, 0.66), lw=0.3, clear=0.12, rng=rng, allowed=allowed, wob=0.06)
    return cv.finish(cell, bw)


def _on_frame(ch, ix, iy):
    return all(abs(abs(x) - ix) < 0.6 or abs(abs(y) - iy) < 0.6 for x, y in ch)


def _emb_border(cv, W, H, bw, rng):
    PURPLE, WHITE, PINK, LIME, NAVY, CRIMSON = (110, 36, 135), (238, 235, 222), (240, 120, 196), (115, 205, 50), C_NAVY, (200, 22, 58)
    DIA = (250, 170, 25)
    w2, h2 = W / 2, H / 2
    # (inner offset, outer offset) of each band, as fractions of bw from the inner field edge outward
    def frame(o0, o1, col):
        a0, a1 = bw - o0 * bw, bw - o1 * bw              # distance in from the outer edge
        cv.poly("acc", [(-w2 + a1, -h2 + a1), (w2 - a1, -h2 + a1), (w2 - a1, h2 - a1), (-w2 + a1, h2 - a1)], col)
        cv.poly("acc", [(-w2 + a0, -h2 + a0), (w2 - a0, -h2 + a0), (w2 - a0, h2 - a0), (-w2 + a0, h2 - a0)], ERASE)
    # paint from the outside in (each band erases the inside of the next one first)
    frame(0.0, 1.0, CRIMSON)
    frame(0.0, 0.86, NAVY)
    frame(0.0, 0.76, WHITE)
    frame(0.0, 0.22, PURPLE)
    frame(0.0, 0.10, (0, 0, 0))
    cv.poly("acc", [(-w2 + bw * 0.95, -h2 + bw * 0.95), (w2 - bw * 0.95, -h2 + bw * 0.95), (w2 - bw * 0.95, h2 - bw * 0.95),
                    (-w2 + bw * 0.95, h2 - bw * 0.95)], ERASE)
    # diamond chain band between 0.22 and 0.76: filled diamonds, lime stepped outline, pink crosses between
    mid, half = bw * (1 - 0.49), bw * 0.27
    pitch = 2.1 * half
    for side in range(4):
        horiz = side < 2
        span = (w2 if horiz else h2) - bw * 0.2
        sgn = 1 if side % 2 == 0 else -1
        for t in np.arange(-span + pitch * 0.5, span, pitch):
            if horiz:
                c = (t, sgn * (h2 - mid))
            else:
                c = (sgn * (w2 - mid), t)
            if abs(c[0]) > w2 - mid + 0.1 or abs(c[1]) > h2 - mid + 0.1:
                continue
            st_ = half / 3.0
            outer = _stepped_diamond(c, half, st_)
            cv.stroke("acc", outer + outer[:1], 0.55, LIME, join="miter", cap="square")
            r = half - 1.4 * st_
            cv.poly("acc", [(c[0], c[1] + r), (c[0] + r, c[1]), (c[0], c[1] - r), (c[0] - r, c[1])], DIA)
            k = pitch / 2
            pc = (c[0] + k, c[1]) if horiz else (c[0], c[1] + k)
            if abs(pc[0]) > w2 - mid - k * 0.9 or abs(pc[1]) > h2 - mid - k * 0.9:
                if (horiz and abs(pc[0]) > w2 - mid - k * 0.9) or (not horiz and abs(pc[1]) > h2 - mid - k * 0.9):
                    continue
            for a, b in (((-0.6, 0), (0.6, 0)), ((0, -0.6), (0, 0.6))):
                cv.stroke("acc", [(pc[0] + a[0], pc[1] + a[1]), (pc[0] + b[0], pc[1] + b[1])], 0.4, PINK, cap="round")
    for sx in (1, -1):                                  # a diamond on each corner of the chain
        for sy in (1, -1):
            c = (sx * (w2 - mid), sy * (h2 - mid))
            st_ = half / 3.0
            outer = _stepped_diamond(c, half, st_)
            cv.stroke("acc", outer + outer[:1], 0.55, LIME, join="miter", cap="square")
            r = half - 1.4 * st_
            cv.poly("acc", [(c[0], c[1] + r), (c[0] + r, c[1]), (c[0], c[1] - r), (c[0] - r, c[1])], DIA)
    # zig-zags: cream inside the purple band's inner edge; yellow-green overcast on the outer edge
    for off, amp, col, wz in ((bw * 0.9, bw * 0.05, (245, 240, 205), 0.35), (bw * 0.04, bw * 0.05, (215, 240, 90), 0.45)):
        for side in range(4):
            horiz = side < 2
            sgn = 1 if side % 2 == 0 else -1
            span = (w2 if horiz else h2) - off
            pts = []
            for n_, t in enumerate(np.arange(-span, span + 1e-6, 0.9)):
                o = off + (amp if n_ % 2 else -amp)
                pts.append((t, sgn * (h2 - o)) if horiz else (sgn * (w2 - o), t))
            cv.stroke("acc", pts, wz, col, join="round", cap="round")


def _stepped_diamond(c, r, st):
    n = max(2, int(round(r / st)))
    pts = []
    for q in range(4):
        for k in range(n):
            # walk one quadrant edge as a staircase from the top point clockwise
            a = (0, r) if q == 0 else ((r, 0) if q == 1 else ((0, -r) if q == 2 else (-r, 0)))
            dx = (1, -1) if q == 0 else ((-1, -1) if q == 1 else ((-1, 1) if q == 2 else (1, 1)))
            x, y = a[0] + dx[0] * k * st, a[1] + dx[1] * k * st
            if q in (0, 2):
                pts += [(x, y), (x + dx[0] * st, y)]
            else:
                pts += [(x, y), (x, y + dx[1] * st)]
    return [(c[0] + x, c[1] + y) for x, y in pts]


# =====================================================================================================================
# B: cloth kené study (portrait panel, a spine with a lozenge, hooks ending in open squares, coarse hand-made fill)
# =====================================================================================================================
B_FIELD, B_INK, B_BORDER = (217, 207, 192), (43, 35, 32), (59, 53, 48)


def _open_arrow(tip, d, length, half):
    t = _arrow(tip, d, length, half)
    return t + t[:1]


def _diamond(c, r):
    return [(c[0], c[1] + r), (c[0] + r, c[1]), (c[0], c[1] - r), (c[0] - r, c[1])]


def cloth_design(seed, H=124.0, W=62.0, tex_h=1280, weight=0.5):
    """W x H = the whole cloth (dark border included), portrait."""
    rng = np.random.default_rng([int(seed) % 100003, 37, int(H), int(W)])
    cv = Canvas(W, H, tex_h)
    lw = 0.95 + 0.3 * weight
    bw = 0.12 * W
    ix, iy = W / 2 - bw, H / 2 - bw                     # inner (cream) field half size
    # D2-mirrored cells round a central lozenge; their shared edges are the channels (the spine is the x = 0 edge)
    sy = rng.uniform(0.29, 0.34)
    seeds = [(0.0, sy * 2 * iy, rng.uniform(0.0, 3.0)), (ix * rng.uniform(0.5, 0.6), 0.0, rng.uniform(0.0, 3.0)),
             (ix * rng.uniform(0.62, 0.8), iy * rng.uniform(0.74, 0.86), rng.uniform(-3.0, -1.0))]
    pts = []
    for x, y, w in seeds:
        for mx in ((1, -1) if x > 0 else (1,)):
            for my in ((1, -1) if y > 0 else (1,)):
                pts.append((mx * x, my * y, w))
    st = rng.uniform(2.4, 3.0)
    oc = rng.uniform(1.3, 1.6)
    nx2, ny2 = int(math.ceil(ix / st)), int(math.ceil(iy / st))
    xs = (np.arange(-nx2, nx2) + 0.5) * st
    ys = (np.arange(-ny2, ny2) + 0.5) * st
    X, Y = np.meshgrid(xs, ys)
    best = np.full(X.shape, 1e9)
    lab = np.zeros(X.shape, np.int32)
    for n, (x, y, w) in enumerate(pts):
        a, b = np.abs(X - x), np.abs(Y - y) * 0.55
        m = np.maximum(np.maximum(a, b), (a + b) / oc) - w
        t = m < best
        best[t], lab[t] = m[t], n
    chains, junc = boundary_chains(lab, -nx2 * st, -ny2 * st, st)
    clip = lambda ch: [(min(max(x, -ix), ix), min(max(y, -iy), iy)) for x, y in ch]
    chains = [clip(ch) for ch in chains]
    for ch in chains:
        cv.stroke("prim", ch, lw, B_INK, join="miter", cap="square")
    # a thin line round the cream field, just inside the border (as on the cloth)
    fr = [(-ix + 0.9, -iy + 0.9), (ix - 0.9, -iy + 0.9), (ix - 0.9, iy - 0.9), (-ix + 0.9, iy - 0.9)]
    cv.stroke("prim", fr, 0.35, B_INK, join="miter", closed=True)
    # the lozenge (elongated hexagon) on the spine at the centre, cream inside
    lw2, lh = rng.uniform(3.6, 4.6), rng.uniform(8.0, 10.0)
    loz = [(0, lh), (lw2, lh - lw2), (lw2, -lh + lw2), (0, -lh), (-lw2, -lh + lw2), (-lw2, lh - lw2)]
    cv.poly("prim", loz, ERASE)
    cv.stroke("prim", loz, lw, B_INK, join="miter", closed=True)
    cv.stroke("prim", [(0, lh), (0, -lh)], lw, ERASE)
    cv.stroke("prim", loz, lw, B_INK, join="miter", closed=True)
    # small diamonds where the spine meets a channel (junctions on x = 0)
    dia = [(x, y) for x, y in junc if abs(x) < 0.5 * st and abs(y) > lh + 2 and abs(y) < iy - 3]
    for x, y in dia:
        q = _diamond((0.0, y), 1.9)
        cv.poly("prim", q, ERASE)
        cv.stroke("prim", q, 0.8 * lw, B_INK, join="miter", closed=True)
    occ = Occ(-ix - 1, -iy - 1, ix + 1, iy + 1,
              lambda X_, Y_: lab[np.clip(np.floor(Y_ / st).astype(int) + ny2, 0, 2 * ny2 - 1),
                                 np.clip(np.floor(X_ / st).astype(int) + nx2, 0, 2 * nx2 - 1)])
    for ch in chains + [fr + fr[:1]]:
        occ.mark(_samples(ch, lw / 2 + 1.0, 0.25), 1)
    occ.mark(_samples(loz + loz[:1], lw / 2 + 1.0, 0.25), 1)
    occ.mark([(0.0, 0.0, lw2 + 0.5)], 2)
    for x, y in dia:
        occ.mark([(0.0, y, 3.0)], 1)
    ops = []

    def place(geo, foot, mir=((1, 1), (-1, 1), (1, -1), (-1, -1)), cross=False):
        for sx, sy_ in mir:
            f = [(sx * x, sy_ * y, r) for x, y, r in foot]
            lab0 = occ.lab[occ.ij(f[0][0], f[0][1])] if f else -1
            if not occ.free(f, lab0, cross):
                return False
        for sx, sy_ in mir:
            occ.mark([(sx * x, sy_ * y, max(0.3, r - 1.2)) for x, y, r in foot])
            ops.append((geo, sx, sy_))
        return True
    # top / bottom cells: a stem on the axis with an outline arrow at each end and a small diamond between
    ty = sy * 2 * iy
    L = rng.uniform(6.0, 8.5)
    geo = [("line", [(0, ty - L + 2.2), (0, ty + L - 2.2)], 0.8 * lw), ("oarrow", _open_arrow((0, ty + L), (0, 1), 2.6, 1.6)),
           ("oarrow", _open_arrow((0, ty - L), (0, -1), 2.6, 1.6)), ("dia", (0, ty + rng.uniform(-1.5, 1.5)), 1.3)]
    place(geo, [(0, y, 2.4) for y in np.arange(ty - L, ty + L + 0.1, 0.5)], mir=((1, 1), (1, -1)), cross=True)
    # side middle cells: an outline arrow pointing in, on a stem from the edge
    ax = rng.uniform(0.45, 0.62) * ix
    geo = [("line", [(ix - 1.0, 0), (ax + 2.2, 0)], 0.8 * lw), ("oarrow", _open_arrow((ax, 0), (-1, 0), 2.6, 1.6)),
           ("line", [(ax + 4.5, -1.3), (ax + 4.5, 1.3)], 0.7 * lw)]
    place(geo, [(x, 0, 2.2) for x in np.arange(ax, ix - 3.0, 0.5)], mir=((1, 1), (-1, 1)), cross=True)
    # stepped hooks from the channels, ending in an open square (a "key")
    nh = 0
    for _try in range(500):
        if nh >= 4:
            break
        x, y = rng.uniform(2.0, ix - 2.0), rng.uniform(lh + 1.0, iy - 2.0)
        d1 = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(0, 4))]
        lab0 = occ.lab[occ.ij(x, y)]
        stem = occ.ray(x, y, -d1[0], -d1[1], lab0, 6.0)
        if stem is None or stem < 0.5:
            continue
        start = (x - d1[0] * stem, y - d1[1] * stem)
        d2 = (d1[1], d1[0]) if rng.random() < 0.5 else (-d1[1], -d1[0])
        p = (x, y)
        path = [start, p]
        for _k in range(int(rng.integers(2, 4))):
            p = (p[0] + d1[0] * rng.uniform(2.0, 3.2), p[1] + d1[1] * rng.uniform(2.0, 3.2))
            path.append(p)
            p = (p[0] + d2[0] * rng.uniform(1.6, 2.4), p[1] + d2[1] * rng.uniform(1.6, 2.4))
            path.append(p)
        end = (p[0] + d1[0] * 1.2, p[1] + d1[1] * 1.2)
        path.append(end)
        k = 1.0
        sq = [(end[0] + d1[0] * 0.0 - d2[0] * k + d1[0] * 0.0, end[1] - d2[1] * k)]
        c = (end[0] + d1[0] * 1.3, end[1] + d1[1] * 1.3)
        sq = [(c[0] - 1.25, c[1] - 1.25), (c[0] + 1.25, c[1] - 1.25), (c[0] + 1.25, c[1] + 1.25), (c[0] - 1.25, c[1] + 1.25)]
        body = path[1:] + [c]
        foot = [(px, py, lw / 2 + 1.6) for px, py, _ in _samples(body, 0, 0.4)]
        foot += [(px, py, lw / 2 + 0.5) for px, py, _ in _samples([(x, y), (start[0] + d1[0] * 1.5, start[1] + d1[1] * 1.5)], 0, 0.4)]
        foot.append((c[0], c[1], 3.0))
        if min(f[0] for f in foot) < 0.3 or min(f[1] for f in foot) < 0.3:
            continue
        if place([("pline", path, lw), ("osq", sq)], foot):
            nh += 1
    # dumbbells floating in the outer cells: a bar with end bars (and a small box in the middle)
    nd = 0
    for _try in range(300):
        if nd >= 2:
            break
        x, y = rng.uniform(0.45 * ix, ix - 4.0), rng.uniform(0.5 * iy, iy - 3.0)
        horiz = rng.random() < 0.7
        L2 = rng.uniform(2.6, 3.6)
        e = (1, 0) if horiz else (0, 1)
        n = (-e[1], e[0])
        geo = [("line", [(x - e[0] * L2, y - e[1] * L2), (x + e[0] * L2, y + e[1] * L2)], 0.8 * lw)]
        for sgn in (1, -1):
            b = (x + sgn * e[0] * L2, y + sgn * e[1] * L2)
            geo.append(("line", [(b[0] - n[0] * 1.3, b[1] - n[1] * 1.3), (b[0] + n[0] * 1.3, b[1] + n[1] * 1.3)], 0.8 * lw))
        geo.append(("osq", [(x - 0.9, y - 0.9), (x + 0.9, y - 0.9), (x + 0.9, y + 0.9), (x - 0.9, y + 0.9)]))
        foot = [(px, py, 2.6) for px, py, _ in _samples([(x - e[0] * L2, y - e[1] * L2), (x + e[0] * L2, y + e[1] * L2)], 0, 0.5)]
        if place(geo, foot):
            nd += 1
    for geo, sx, sy_ in ops:
        for g in geo:
            k = g[0]
            if k == "line":
                cv.stroke("prim", _mir(g[1], sx, sy_), g[2], B_INK)
            elif k == "pline":
                cv.stroke("prim", _mir(g[1], sx, sy_), g[2], B_INK, join="miter", cap="square")
            elif k == "oarrow":
                q = _mir(g[1], sx, sy_)
                cv.poly("prim", q, ERASE)
                cv.poly("block", q)
                cv.stroke("prim", q, 0.7 * lw, B_INK, join="miter", closed=True)
            elif k == "osq":
                q = _mir(g[1], sx, sy_)
                cv.poly("prim", q, ERASE)
                cv.poly("block", q)
                cv.stroke("prim", q, 0.7 * lw, B_INK, join="miter", closed=True)
            elif k == "dia":
                q = _mir(_diamond(g[1], g[2]), sx, sy_)
                cv.poly("prim", q, ERASE)
                cv.poly("block", q)
                cv.stroke("prim", q, 0.6 * lw, B_INK, join="miter", closed=True)
    # the dark cloth border (accent layer: a textile frame), the cream ground
    cv.poly("acc", [(-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)], B_BORDER)
    cv.poly("acc", [(-ix, -iy), (ix, -iy), (ix, iy), (-ix, iy)], ERASE)
    cv.poly("block", [(-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)])
    cv.poly("block", [(-ix + 1.3, -iy + 1.3), (ix - 1.3, -iy + 1.3), (ix - 1.3, iy - 1.3), (-ix + 1.3, iy - 1.3)], ERASE)
    cv.rect("ground", -W / 2, -H / 2, W / 2, H / 2)
    cell = draw_meander(cv, pitch=rng.uniform(1.15, 1.3), lw=0.3 + 0.06 * weight, clear=0.3, rng=rng, wob=0.1)
    return cv.finish(cell, bw)


# =====================================================================================================================
def field_dims(kind, scale, aspect):
    """(W, H) in u for a single-panel field. scale 0..1 = line density (kene_scale)."""
    s = min(max(float(scale), 0.0), 1.0)
    if kind == "A":
        H = 100.0 + 70.0 * s
    elif kind == "B":                                    # portrait cloth 1 : 2 (+ border), whatever the screen
        H = 110.0 + 40.0 * s
        return round(H * 0.5), round(H)
    else:
        H = 80.0 + 50.0 * s
    return round(H * aspect * 0.955), round(H)


def vector_design(kind, seed, scale=0.5, aspect=1.6, tex_h=1280, weight=0.5):
    W, H = field_dims(kind, scale, aspect)
    if kind == "A":
        return painted_design(seed, H, W, tex_h, weight)
    if kind == "B":
        return cloth_design(seed, H, W, tex_h, weight)
    return embroidered_design(seed, H, W, tex_h, weight)


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    out = Path(__file__).resolve().parent / "test_frames" / "kene_real"
    out.mkdir(parents=True, exist_ok=True)
    for kind in sys.argv[1:] or ["A", "C"]:
        t0 = time.perf_counter()
        d = vector_design(kind, 3)
        print(kind, f"{time.perf_counter() - t0:.2f}s", d["cov"].shape)
        cov = d["cov"].astype(float) / 255.0
        g = np.array({"A": A_GROUND, "B": B_FIELD, "C": C_GROUND}[kind], float) / 255.0
        fc = np.array({"A": A_INK, "B": B_INK, "C": C_FILL}[kind], float) / 255.0
        col = g[None, None] * cov[..., 2:3]
        col = col * (1 - cov[..., 1:2]) + fc * cov[..., 1:2]
        col = col * (1 - cov[..., 0:1]) + d["prim"][..., :3] / 255.0
        col = col * (1 - cov[..., 3:4]) + d["acc"][..., :3] / 255.0
        Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).save(out / f"_vec_{kind}.png")
