"""Multi-surface output stage on a real (hidden) GL context: black outside, warps, modes, overlay, speed."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
moderngl = pytest.importorskip("moderngl")
import surfaces as S  # noqa: E402
import surface_render as SR  # noqa: E402

W, H = 640, 400


def cmd(op, **kw):
    return S.validate_cmd(dict(kw, cmd="surf", op=op))


@pytest.fixture(scope="module")
def gl():
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception:
        import fx_stack
        ctx, _win = fx_stack.hidden_context()
    # a scene with no flat areas: smooth gradients (so any shift or flip shows) under a coarse checker
    y, x = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 4), dtype="f4")
    img[..., 0] = 0.15 + 0.8 * x / (W - 1)
    img[..., 1] = 0.15 + 0.8 * y / (H - 1)                # row 0 = bottom (GL): green grows upward
    img[..., 2] = 0.25 + 0.5 * (((x // 16) + (y // 16)) % 2)
    img[..., 3] = 1.0
    scene = ctx.texture((W, H), 4, img.astype("f2").tobytes(), dtype="f2")
    scene.filter = (moderngl.LINEAR, moderngl.LINEAR)
    scene.repeat_x = scene.repeat_y = False
    fbo = ctx.simple_framebuffer((W, H))
    r = SR.SurfaceRenderer(ctx)
    ref = np.clip(np.rint(img[..., :3].astype("f2").astype("f4") * 255), 0, 255).astype(np.int16)
    yield dict(ctx=ctx, scene=scene, fbo=fbo, r=r, ref=ref)


def draw(gl, sset, t=0.0, bass=0.0, grid=False):
    """-> int16 array [row from the BOTTOM, x, rgb], as GL reads it."""
    gl["r"].render(sset, gl["scene"], gl["fbo"], (W, H), t, bass, grid)
    return np.frombuffer(gl["fbo"].read(components=3), dtype=np.uint8).reshape(H, W, 3).astype(np.int16)


def px(a, x, y):
    """Pixel at projector position (x, y) in the 1280x800 output frame (y down)."""
    return a[H - 1 - int(y * H / 800), int(x * W / 1280)]


def faces():
    st = S.SurfaceSet()
    st.apply(cmd("corners", id=1, pts=[[200, 300], [500, 300], [500, 600], [200, 600]]))
    st.apply(cmd("add")); st.apply(cmd("corners", id=2, pts=[[500, 300], [700, 200], [700, 500], [500, 600]]))
    st.apply(cmd("add")); st.apply(cmd("corners", id=3, pts=[[200, 300], [500, 300], [700, 200], [400, 200]]))
    return st


def test_one_full_frame_surface_is_the_frame_itself(gl):
    out = draw(gl, S.SurfaceSet())
    assert np.abs(out - gl["ref"]).max() <= 1, "the general path must not shift, flip or dim the picture"


def test_everything_outside_the_surfaces_is_pure_black(gl):
    st = faces()
    out = draw(gl, st)
    mask = np.zeros((H, W), dtype=bool)
    for yy in range(H):
        for xx in range(0, W, 3):
            p = ((xx + 0.5) * 1280 / W, (H - 1 - yy + 0.5) * 800 / H)
            mask[yy, xx] = any(S.point_in_poly(p, s["corners"]) for s in st.surfaces)
    lit = out.sum(axis=2) > 0
    cols = np.zeros((H, W), dtype=bool)
    cols[:, ::3] = True
    grow = np.zeros_like(mask)                                    # allow the 1 px anti-aliased rim
    for dy in (-1, 0, 1):
        for dx in (-3, 0, 3):
            grow |= np.roll(np.roll(mask, dy, 0), dx, 1)
    assert not (lit & cols & ~grow).any(), "light leaked outside a surface"
    assert (lit & mask).sum() > 0.9 * mask.sum(), "the faces themselves are lit"
    assert tuple(px(out, 60, 60)) == (0, 0, 0) and tuple(px(out, 1000, 700)) == (0, 0, 0)


def test_each_quad_shows_the_whole_frame_warped(gl):
    out = draw(gl, faces())
    # surface 1 is a 300x300 px square at (200,300): red grows to its right, green to its top
    tl, br = px(out, 206, 306), px(out, 494, 594)
    assert tl[0] < 60 and tl[1] > 200 and br[0] > 200 and br[1] < 60
    c1 = px(out, 350, 450)
    assert abs(c1[0] - 140) <= 6 and abs(c1[1] - 140) <= 6, "its centre shows the frame centre"
    c2 = px(out, 600, 400)                                        # surface 2 is a parallelogram: same at its centre
    assert abs(c2[0] - 140) <= 8 and abs(c2[1] - 140) <= 8


def test_sub_rect_source_shows_only_that_part(gl):
    st = S.SurfaceSet()
    st.apply(cmd("set", id=1, src="rect", rect=[0.5, 0.0, 0.5, 0.5]))             # right half, top half of the frame
    out = draw(gl, st)
    assert out[..., 0].min() >= 135 and out[..., 1].min() >= 135, "only the top-right quarter is on screen"
    assert out[..., 0].max() >= 235 and out[..., 1].max() >= 235


def test_spanned_surfaces_show_one_continuous_unwarped_picture(gl):
    st = faces()
    for i in (1, 2, 3):
        st.apply(cmd("set", id=i, src="span"))
    out = draw(gl, st)
    x0, y0, cw, ch = st.span_box()
    for (x, y) in [(350, 450), (600, 400), (450, 250), (496, 450), (504, 450)]:    # the last two straddle a shared edge
        u, v = (x - x0) / cw, 1 - (y - y0) / ch
        want = gl["ref"][min(H - 1, int(v * H)), min(W - 1, int(u * W))]
        assert np.abs(px(out, x, y)[:2] - want[:2]).max() <= 4, (x, y)


def test_content_modes_opacity_brightness(gl):
    st = S.SurfaceSet()
    st.apply(cmd("set", id=1, content="solid", color=[1, 0.5, 0], brightness=0.5))
    got = draw(gl, st)[H // 2, W // 2]
    assert abs(got[0] - 128) <= 1 and abs(got[1] - 64) <= 1 and got[2] == 0
    st.apply(cmd("set", id=1, content="black", brightness=1))
    assert draw(gl, st).max() == 0
    st.apply(cmd("set", id=1, content="main", opacity=0.5))
    assert np.abs(draw(gl, st) * 2 - gl["ref"]).max() <= 3
    st.apply(cmd("set", id=1, enabled=False))
    assert draw(gl, st).max() == 0, "no enabled surface = a black projector"


def test_later_surfaces_cover_earlier_ones(gl):
    st = S.SurfaceSet()
    st.apply(cmd("set", id=1, content="solid", color=[1, 0, 0]))
    st.apply(cmd("add"))
    st.apply(cmd("set", id=2, content="solid", color=[0, 0, 1]))
    out = draw(gl, st)
    c = st.get(2)["corners"]
    mid = ((c[0][0] + c[2][0]) / 2, (c[0][1] + c[2][1]) / 2)
    assert tuple(px(out, *mid)) == (0, 0, 255) and tuple(px(out, 20, 20)) == (255, 0, 0)
    st.apply(cmd("move", id=2, to=0))
    assert tuple(px(draw(gl, st), *mid)) == (255, 0, 0)


def test_feather_fades_the_edge_inward_only(gl):
    st = S.SurfaceSet()
    st.apply(cmd("corners", id=1, pts=[[320, 200], [960, 200], [960, 600], [320, 600]]))
    st.apply(cmd("set", id=1, content="solid", color=[1, 1, 1], feather=80))
    out = draw(gl, st)
    row = [int(px(out, x, 400)[0]) for x in (300, 324, 340, 360, 380, 420, 640)]
    assert row[0] == 0 and row[1] < 20 and row == sorted(row) and row[-2] == 255 and row[-1] == 255
    assert 90 < row[3] < 170, "half way into the feather is about half bright"


def test_poly_mask_cuts_the_quad(gl):
    st = S.SurfaceSet()
    st.apply(cmd("set", id=1, content="solid", color=[1, 1, 1]))
    st.apply(cmd("mask", id=1, pts=[[640, 40], [1240, 760], [40, 760]]))
    out = draw(gl, st)
    assert tuple(px(out, 640, 500)) == (255, 255, 255) and tuple(px(out, 100, 100)) == (0, 0, 0)
    assert tuple(px(out, 1180, 100)) == (0, 0, 0)


def test_edge_glow_is_a_bright_frame_inside_a_dark_face(gl):
    st = S.SurfaceSet()
    st.apply(cmd("corners", id=1, pts=[[320, 200], [960, 200], [960, 600], [320, 600]]))
    st.apply(cmd("set", id=1, content="glow"))
    out = draw(gl, st, t=1.0, bass=0.5)
    lum = out.max(axis=2)
    top = H - 1 - 100                                             # fbo row of the top edge (y = 200 px of 800)
    assert lum[top - 4:top + 1, 200:440].max(axis=0).min() > 150, "a bright line runs along the whole top edge"
    assert lum[H // 2, W // 2] < 12, "the middle of the face stays dark"
    assert tuple(px(out, 300, 400)) == (0, 0, 0) and tuple(px(out, 640, 180)) == (0, 0, 0), "glow never leaves the face"
    assert px(out, 326, 206).max() > 180, "corner jewel"
    a, b = draw(gl, st, t=0.0), draw(gl, st, t=2.5)
    assert np.abs(a - b).max() > 30, "the light chases round the frame"
    st.apply(cmd("set", id=1, content="main", glow=0.6))
    both = draw(gl, st, t=1.0)
    assert np.abs(px(both, 640, 400) - gl["ref"][H // 2, W // 2]).max() <= 60, "glow over the picture leaves the middle alone"
    assert both.max(axis=2)[top - 4:top + 1, 200:440].max(axis=0).min() > 120


def test_overlay_marks_each_surface_in_its_colour_with_a_label(gl):
    st = faces()
    plain = draw(gl, st)
    st.apply(cmd("overlay", mode="all", grid=True))
    out = draw(gl, st)
    assert np.abs(out - plain).mean() > 5
    for s in st.surfaces:                                         # the outline carries the surface's own colour
        c = np.array(s["color"]) * 255
        x, y = (s["corners"][0][0] + s["corners"][1][0]) / 2, (s["corners"][0][1] + s["corners"][1][1]) / 2
        cx, cy = S.centroid(s["corners"])
        d = max(1.0, ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5)
        near = px(out, x + (cx - x) / d * 3, y + (cy - y) / d * 3)        # 3 output px inside the top edge
        assert np.abs(near - c).max() < 70 or near.min() > 180, (s["id"], near, c)
    assert tuple(px(out, 60, 60)) == (0, 0, 0), "the overlay is black outside too"
    cx, cy = S.centroid(st.get(1)["corners"])
    r0, c0 = H - 1 - int(cy * H / 800), int(cx * W / 1280)
    box = out[r0 - 6:r0 + 6, c0 - 20:c0 + 20]
    assert (box.max(axis=2) < 40).mean() > 0.2, "a dark label plate sits at the centre"
    assert (box.max(axis=2) > 150).mean() > 0.03, "with the name written on it"
    st.apply(cmd("overlay", mode="sel", grid=False))
    only = draw(gl, st)
    assert np.abs(px(only, 350, 380) - px(plain, 350, 380)).max() <= 1, "unselected surfaces keep playing"
    st.apply(cmd("overlay", mode="off"))
    assert np.abs(draw(gl, st) - plain).max() == 0
    assert np.abs(draw(gl, st, grid=True) - out).max() <= 1, "the old GRID switch = overlay + grid on every surface"


def test_twelve_surfaces_cost_little_and_blend_state_is_restored(gl):
    st = S.SurfaceSet()
    st.apply(cmd("set", id=1, glow=0.5, feather=6))
    for k in range(11):
        st.apply(cmd("add"))
        st.apply(cmd("set", id=k + 2, glow=1.0, feather=4, content=("main", "glow", "solid")[k % 3]))
        st.apply(cmd("mask_add", id=k + 2))
    draw(gl, st)
    gl["ctx"].finish()
    t0 = time.perf_counter()
    for i in range(30):
        gl["r"].render(st, gl["scene"], gl["fbo"], (W, H), i / 60.0, 0.3)
    gl["ctx"].finish()
    ms = (time.perf_counter() - t0) / 30 * 1000
    assert ms < 8.0, f"{ms:.2f} ms for 12 surfaces"
    out = draw(gl, S.SurfaceSet())                                # a later plain draw is unaffected by leftover state
    assert np.abs(out - gl["ref"]).max() <= 1
