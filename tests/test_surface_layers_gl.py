"""PER-SURFACE LAYERS + MASK SURFACES on a real (hidden) GL context: own source per surface, one render per shared
source, zero cost unused, bypass = black, blends / hue, mask groups as one draw, cost. Run under .locks/gpu_suite."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
moderngl = pytest.importorskip("moderngl")
import materials  # noqa: E402
import surfaces as S  # noqa: E402
import surface_render as SR  # noqa: E402

W, H = 640, 400
COL = dict(kene=(255, 0, 0), fractal=(0, 255, 0), lines=(0, 0, 255), text=None)


def cmd(op, **kw):
    return S.validate_cmd(dict(kw, cmd="surf", op=op))


@pytest.fixture(scope="module")
def gl():
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception:
        import fx_stack
        ctx, _win = fx_stack.hidden_context()
    scene = ctx.texture((W, H), 4, bytes([128, 128, 128, 255]) * (W * H))
    fbo = ctx.simple_framebuffer((W, H))
    yield dict(ctx=ctx, scene=scene, fbo=fbo)


def solid(ctx, rgb, n=64):
    return ctx.texture((n, n), 4, bytes(list(rgb) + [255]) * (n * n))


def make(gl):
    ctx = gl["ctx"]
    hub = materials.LayerHub(ctx, size=64, layers=8)
    calls = {k: 0 for k in COL}
    texs = {k: solid(ctx, c) for k, c in COL.items() if c}
    g = np.zeros((64, 64, 4), np.uint8)
    g[..., 0] = np.linspace(0, 255, 64)[None, :].astype(np.uint8)    # red grows with u
    g[..., 3] = 255
    texs["text"] = ctx.texture((64, 64), 4, g.tobytes())
    for k in COL:
        def draw(size, k=k):
            calls[k] += 1
            return texs[k]
        hub.register(k, draw=draw)
    return hub, calls, SR.SurfaceRenderer(ctx)


def frame(gl, r, hub, sset, t=0.0, blocked=(), n=1):
    for _ in range(n):
        r.render(sset, gl["scene"], gl["fbo"], (W, H), t, 0.0, hub=hub, blocked=set(blocked))
        hub.run("scene")
        hub.end_frame()
    r.render(sset, gl["scene"], gl["fbo"], (W, H), t, 0.0, hub=hub, blocked=set(blocked))
    return np.frombuffer(gl["fbo"].read(components=3), dtype=np.uint8).reshape(H, W, 3).astype(int)


def px(a, x, y):
    return a[H - 1 - int(y * H / 800), int(x * W / 1280)]


BOXES = [[[100, 100], [400, 100], [400, 400], [100, 400]], [[500, 100], [800, 100], [800, 400], [500, 400]],
         [[900, 100], [1200, 100], [1200, 400], [900, 400]]]


def three(srcs):
    st = S.SurfaceSet()
    for k, (box, src) in enumerate(zip(BOXES, srcs)):
        if k:
            st.apply(cmd("add"))
        sid = st.surfaces[-1]["id"]
        st.apply(cmd("corners", id=sid, pts=box))
        if src:
            st.apply(cmd("set_layers", id=sid, content="layers", layers=[{"src": s} for s in src]))
    return st


def test_each_surface_shows_its_own_source(gl):
    hub, calls, r = make(gl)
    st = three([["kene"], ["fractal"], ["lines"]])
    a = frame(gl, r, hub, st)
    assert tuple(px(a, 250, 250)) == (255, 0, 0)
    assert tuple(px(a, 650, 250)) == (0, 255, 0)
    assert tuple(px(a, 1050, 250)) == (0, 0, 255)
    assert px(a, 450, 250).sum() == 0 and px(a, 640, 600).sum() == 0          # black between / outside


def test_a_source_shared_by_three_surfaces_renders_once_per_frame(gl):
    hub, calls, r = make(gl)
    st = three([["kene"], ["kene"], ["kene", "fractal"]])
    frame(gl, r, hub, st)
    c0, q0 = dict(calls), r.hub_requests
    frame(gl, r, hub, st, n=5)
    assert calls["kene"] - c0["kene"] == 5 and calls["fractal"] - c0["fractal"] == 5     # once per frame each
    assert r.hub_requests - q0 == 6 * 2                                                 # one request per source per frame


def test_zero_cost_when_no_surface_uses_layers(gl):
    class NoHub:
        def request(self, name):
            raise AssertionError("must not be asked")

        def use(self):
            raise AssertionError("must not be bound")
    r = SR.SurfaceRenderer(gl["ctx"])
    st = three([None, None, None])
    st.apply(cmd("set", id=2, content="glow"))
    st.get(1)["layers"] = [dict(S.LAYER_DEFAULT)]                      # a stored stack while showing "main": idle
    r.render(st, gl["scene"], gl["fbo"], (W, H), 0.0, 0.0, hub=NoHub())
    assert r.hub_requests == 0
    hub, calls, _ = make(gl)
    frame(gl, r, hub, st, n=3)
    assert sum(calls.values()) == 0 and hub.status()["live"] == [] and hub.vram_bytes() == 0


def test_bypassed_source_is_black_and_blends_and_hue(gl):
    hub, calls, r = make(gl)
    st = three([["kene"], ["fractal"], ["lines"]])
    a = frame(gl, r, hub, st, blocked=SR.blocked_sources({"kene"}, {"kene": ("l_kene_shipibo", "l_kene_hunikuin")}))
    assert px(a, 250, 250).sum() == 0 and tuple(px(a, 650, 250)) == (0, 255, 0)
    assert SR.blocked_sources({"layers"}, {"layers": ("l_kene", "l_beads")}) == {"blocks", "beads"}
    st.apply(cmd("set_layers", id=1, layers=[{"src": "main"}, {"src": "fractal", "blend": "add", "opacity": 0.5}]))
    st.apply(cmd("set_layers", id=2, layers=[{"src": "main"}, {"src": "kene", "blend": "multiply"}]))
    st.apply(cmd("set_layers", id=3, layers=[{"src": "kene", "hue": 120}]))
    a = frame(gl, r, hub, st)
    r1, g1, b1 = px(a, 250, 250)
    assert abs(r1 - 128) <= 2 and abs(g1 - 128 - 128) <= 3 and abs(b1 - 128) <= 2          # main + 0.5 green
    assert tuple(np.abs(px(a, 650, 250) - (128, 0, 0)) <= 2) == (True,) * 3               # main x red
    r3, g3, b3 = px(a, 1050, 250)
    assert g3 > 2 * r3 and g3 > b3                                                         # red turned green


def test_scroll_moves_the_texture(gl):
    hub, calls, r = make(gl)
    st = three([["text"], None, None])
    st.apply(cmd("set_layers", id=1, layers=[{"src": "text", "scroll": 0.25, "dir": 0}]))
    a = frame(gl, r, hub, st, t=0.0)
    b = frame(gl, r, hub, st, t=1.0)
    assert px(a, 120, 250)[0] < 40 and abs(int(px(b, 120, 250)[0]) - 64) < 20            # u shifted by 0.25


def mask_set(tmp_path):
    img = np.zeros((800, 1280, 4), np.uint8)
    for lab, x in ((1, 300), (2, 600), (3, 900)):
        img[100:700, x:x + 12, 0] = lab
        img[100:700, x:x + 12, 1] = np.linspace(255, 0, 600)[:, None].astype(np.uint8)   # u: base (bottom) -> tip
        img[100:700, x:x + 12, 3] = 255
    (tmp_path / "masks").mkdir()
    from PIL import Image
    Image.fromarray(img, "RGBA").save(tmp_path / "masks" / "grass_1.png")
    surfs = []
    for k, x in enumerate((300, 600, 900), 1):
        s = S.new_surface(k, name=f"Grass blade {k}", corners=[[x, 100], [x + 12, 100], [x + 12, 700], [x, 700]])
        s.update(shape="mask", maskfile="grass_1.png", label=k, group="Grass blade")
        surfs.append(s)
    st = S.SurfaceSet()
    st.mask_dir = tmp_path / "masks"
    assert st.apply(cmd("replace", surfaces=surfs))
    return st


def test_mask_group_is_one_draw_and_lights_only_the_parts(gl, tmp_path):
    hub, calls, r = make(gl)
    st = mask_set(tmp_path)
    a = frame(gl, r, hub, st)
    assert r.draw_calls == 1                                            # three blades, one draw
    assert tuple(px(a, 606, 400)) == (128, 128, 128) and px(a, 450, 400).sum() == 0 and px(a, 606, 60).sum() == 0
    st.apply(cmd("set", id=2, opacity=0.5))                             # one blade on its own: the run splits
    frame(gl, r, hub, st)
    assert r.draw_calls == 3
    st.apply(cmd("set", id=2, opacity=1.0))
    st.apply(cmd("set_layers", id=1, group_all=True, content="layers", layers=[{"src": "text"}]))
    a = frame(gl, r, hub, st)
    assert r.draw_calls == 1
    low, high = px(a, 906, 650)[0], px(a, 906, 150)[0]                  # the layer flows along each blade
    assert low < 60 and high > 190


def test_cost_three_surfaces_with_layers(gl):
    ctx = gl["ctx"]
    hub, calls, r = make(gl)
    big = ctx.simple_framebuffer((1280, 800))
    st = three([["kene", "fractal", "lines"], ["fractal", "lines"], ["kene", "main"]])
    plain = three([None, None, None])

    def gpu(sset, n=40):
        q = ctx.query(time=True)
        out = []
        for _ in range(n):
            with q:
                r.render(sset, gl["scene"], big, (1280, 800), 1.0, 0.0, hub=hub)
            hub.run("scene")
            hub.end_frame()
            out.append(q.elapsed / 1e6)
        return float(np.median(out[5:]))
    gpu(st, 5)
    ms_layers, ms_plain = gpu(st), gpu(plain)
    print(f"[surf-layers cost] 3 surfaces with layers {ms_layers:.3f} ms, plain {ms_plain:.3f} ms (1280x800)")
    assert ms_layers <= 0.5
    big.release()


def test_two_surfaces_with_different_rects_sample_different_regions(gl):
    """Real-wall report (Neon loop): per-surface src rect / full must not collapse into one projector-space span."""
    ctx = gl["ctx"]
    y, x = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 4), np.uint8)
    img[..., 0] = (x * 255 // (W - 1)).astype(np.uint8)                  # red = frame x
    img[..., 1] = (y * 255 // (H - 1)).astype(np.uint8)                  # green = frame v (up)
    img[..., 3] = 255
    scene = ctx.texture((W, H), 4, img.tobytes())
    r = SR.SurfaceRenderer(ctx)
    st = three([None, None, None])
    st.apply(cmd("set", id=1, src="rect", rect=[0.0, 0.0, 0.25, 0.25]))      # top-left corner of the show
    st.apply(cmd("set", id=2, src="rect", rect=[0.75, 0.75, 0.25, 0.25]))    # bottom-right corner
    st.apply(cmd("set", id=3, src="full"))
    r.render(st, scene, gl["fbo"], (W, H))
    a = np.frombuffer(gl["fbo"].read(components=3), dtype=np.uint8).reshape(H, W, 3).astype(int)
    c1, c2, c3 = px(a, 250, 250), px(a, 650, 250), px(a, 1050, 250)        # each surface's centre
    assert c1[0] < 50 and c1[1] > 200                                         # frame (0.125, 0.125 from top)
    assert c2[0] > 200 and c2[1] < 50                                         # frame (0.875, 0.875 from top)
    assert abs(c3[0] - 128) < 12 and abs(c3[1] - 128) < 12                    # whole frame: its centre
    st.apply(cmd("set", id=1, brightness=0.1))                                # dark-cloth vs drywall balance
    r.render(st, scene, gl["fbo"], (W, H))
    b = np.frombuffer(gl["fbo"].read(components=3), dtype=np.uint8).reshape(H, W, 3).astype(int)
    assert abs(px(b, 250, 250)[1] - c1[1] * 0.1) < 4
    scene.release()
    r.release()
