"""art_layer.ArtPass: library handling, robustness and GL hygiene. Headless (hidden GLFW window)."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import art_layer as al  # noqa: E402

moderngl = pytest.importorskip("moderngl")
SIZE = (320, 200)


# ---------------------------------------------------------------------------------------------- helpers
def painting(seed, w=240, h=150):
    """A small bright 'painting': coloured rings on a dark ground (bright in the middle)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((x - w / 2) / (h / 2), (y - h / 2) / (h / 2))
    rgb = np.stack([0.5 + 0.5 * np.cos(r * (6 + seed) + k * 2.1 + seed) for k in range(3)], axis=2)
    rgb = rgb * np.clip(1.25 - r, 0.15, 1)[..., None]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def save_art(folder, name, seed=0, depth=False, size=(240, 150)):
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(painting(seed, *size)).save(folder / name)
    if depth:
        y, x = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
        d = np.clip(1 - np.hypot(x / size[0] - 0.5, y / size[1] - 0.5) * 2, 0, 1)
        Image.fromarray((d * 65535).astype(np.uint16)).save(folder / (Path(name).stem + ".depth.png"))


def clock(t=0.0, bpm=120.0):
    b = t * bpm / 60
    return dict(bpm=bpm, beat=b, beat_phase=b % 1, bar_phase=(b / 4) % 1, phrase_phase=(b / 32) % 1)


def pump(ap, vals, seconds=0.5, dt=0.25):
    """Render for a while in wall time so the loader thread can do its work."""
    end = time.time() + seconds
    while time.time() < end:
        ap.render(vals, clock(), 0.0, 0.0, dt)
        time.sleep(0.005)


def make(ctx, folder, settle=0.0):
    ap = al.ArtPass(ctx, SIZE, folder)
    ap.settle_s = settle
    return ap


def corners(img, n=6):
    return max(img[:n, :n, :3].max(), img[:n, -n:, :3].max(), img[-n:, :n, :3].max(), img[-n:, -n:, :3].max())


def scene_float(ap):
    """The pre-quantisation float render target, where a NaN would actually be visible."""
    prev = ap.ctx.fbo
    raw = ap._scene_fbo.read(components=4, dtype="f2")
    prev.use()
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


@pytest.fixture(scope="module")
def ctx():
    try:
        win, c = al.make_context(SIZE)
    except Exception as ex:                       # no GL on this machine / session
        pytest.skip(f"no GL context: {ex}")
    yield c
    import glfw
    glfw.destroy_window(win)


@pytest.fixture()
def lib(tmp_path):
    folder = tmp_path / "art"
    save_art(folder, "a_one.png", 0, depth=True)
    save_art(folder, "b_two.jpg", 1)
    return folder


# ---------------------------------------------------------------------------------------------- library (no GL)
def test_list_art_is_in_creation_order_and_excludes_depth_maps(tmp_path):
    f = tmp_path / "art"
    order = ["zeta.png", "Alpha.jpg", "beta.webp", "gamma.JPEG"]
    for i, n in enumerate(order):
        save_art(f, n, i)
        time.sleep(0.03)
    save_art(f, "delta.png", 5, depth=True)
    (f / "notes.txt").write_text("x")
    (f / "ZETA.DEPTH.PNG").write_bytes(b"junk")
    (f / "sub.png").mkdir()
    got = al.list_art(f)
    assert got == order + ["delta.png"]                             # oldest first: uploads append at the end
    assert all(not n.lower().endswith(".depth.png") for n in got)
    assert got == al.list_art(f) == al.list_art(str(f))            # stable, str or Path
    assert al.depth_path(f, "delta.png") == f / "delta.depth.png"
    time.sleep(0.03)
    save_art(f, "aaa_new_upload.png", 9)                            # sorts first by name, still lands last
    Image.fromarray(painting(3)).save(f / "zeta.png")              # re-saving a piece does not move it
    assert al.list_art(f) == order + ["delta.png", "aaa_new_upload.png"]
    assert al.scan_library(f)[0] == al.list_art(f)


def test_list_art_ties_break_by_name(tmp_path, monkeypatch):
    for n in ("b.png", "C.png", "a.png"):
        save_art(tmp_path, n)
    monkeypatch.setattr(al, "_created_ns", lambda st: 42)
    assert al.list_art(tmp_path) == ["a.png", "b.png", "C.png"]


def test_list_art_missing_folder_is_empty(tmp_path):
    assert al.list_art(tmp_path / "nope") == []
    assert al.scan_library(tmp_path / "nope") == ([], {})


def test_pseudo_depth_is_a_soft_relief_not_noise():
    rng = np.random.default_rng(0)
    rgb = rng.random((120, 192, 3)).astype(np.float32)             # worst case: pure noise in
    rgb[30:90, 60:130] = (1.0, 0.8, 0.2)                           # with one bright saturated shape
    d = al.pseudo_depth(rgb)
    assert d.shape == (120, 192) and np.isfinite(d).all()
    assert d.min() >= 0.0 and d.max() <= 1.0
    gy, gx = np.gradient(d)
    assert np.abs(gx).mean() < 0.03 and np.abs(gy).mean() < 0.03   # smooth
    assert d[45:75, 80:110].mean() > d[:15, :30].mean() + 0.15     # the bright shape stands up
    det = al.detail_map(rgb)
    assert np.isfinite(det).all() and det.min() >= -1 and det.max() <= 1
    flat = al.pseudo_depth(np.zeros((40, 60, 3), np.float32))      # nothing to go on: flat, not NaN
    assert np.isfinite(flat).all()


def test_decode_piece_with_and_without_depth(lib):
    a = al.decode_piece(lib / "a_one.png", al.depth_path(lib, "a_one.png"))
    b = al.decode_piece(lib / "b_two.jpg", al.depth_path(lib, "b_two.jpg"))
    assert a["has_depth"] and not b["has_depth"]
    for p in (a, b):
        assert p["color"].dtype == np.uint8 and p["color"].shape == (150, 240, 4)
        assert [m.shape[:2] for m in p["color_mips"]] == [(75, 120), (37, 60), (18, 30), (9, 15), (4, 7), (2, 3), (1, 1)]
        assert all(m.dtype == np.uint8 and m.flags.c_contiguous for m in p["color_mips"])
        assert p["geo_mips"][-1].shape == (1, 1, 4) and all(m.dtype == np.float16 for m in p["geo_mips"])
        assert abs(float(p["color_mips"][-1][0, 0, 0]) - p["color"][..., 0].mean()) < 12       # a real box filter
        assert p["geo"].dtype == np.float16 and p["geo"].shape[2] == 4
        assert np.isfinite(p["geo"].astype(np.float32)).all()
        assert abs(p["aspect"] - 1.6) < 1e-6
    g = a["geo"].astype(np.float32)
    assert g[75, 120, 0] > g[5, 5, 0] + 0.3                        # white = near: the centre of the cone is near
    (lib / "b_two.depth.png").write_bytes(b"not a png")            # a broken depth map falls back, never raises
    assert not al.decode_piece(lib / "b_two.jpg", al.depth_path(lib, "b_two.jpg"))["has_depth"]


def test_decode_piece_downsizes_huge_images(tmp_path):
    Image.new("RGB", (4000, 1000), (200, 40, 90)).save(tmp_path / "big.png")
    p = al.decode_piece(tmp_path / "big.png")
    assert max(p["color"].shape[:2]) == al.MAX_COLOR_DIM and max(p["geo"].shape[:2]) <= al.MAX_GEO_DIM


def test_source_pixel_cap(tmp_path, monkeypatch):
    Image.new("RGB", (400, 300), (10, 200, 90)).save(tmp_path / "ok.png")
    assert al.MAX_SOURCE_PIXELS <= 60_000_000
    al.decode_piece(tmp_path / "ok.png")
    with pytest.raises(Exception):
        al.decode_piece(tmp_path / "ok.png", max_pixels=400 * 300 - 1)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)            # PIL's own limit is respected as well
    with pytest.raises(Exception):
        al.decode_piece(tmp_path / "ok.png")


def test_16_bit_grayscale_images_keep_their_tones(tmp_path):
    ramp = np.tile(np.linspace(0, 65535, 256).astype(np.uint16), (64, 1))
    Image.fromarray(ramp).save(tmp_path / "ramp16.png")
    c = al.decode_piece(tmp_path / "ramp16.png")["color"]
    assert c[:, 0, 0].max() <= 2 and c[:, -1, 0].min() >= 253
    assert abs(float(c[:, 128, 0].mean()) - 128) < 4                # not clipped to white
    assert (c[..., 0] == c[..., 1]).all() and (c[..., 3] == 255).all()
    twelve = Image.fromarray((ramp // 16).astype(np.int32))         # 12-bit data in an "I" image
    assert twelve.mode == "I"
    assert abs(float(al._to_8bit(twelve).getpixel((128, 5))) - 128) < 4


def test_alpha_is_premultiplied_before_upload(tmp_path):
    rgba = np.zeros((40, 60, 4), np.uint8)
    rgba[..., 0] = 255                                              # red everywhere, also under alpha 0
    rgba[:, 30:, 3] = 255
    rgba[:, 20:30, 3] = 128
    Image.fromarray(rgba).save(tmp_path / "cut.png")
    p = al.decode_piece(tmp_path / "cut.png")
    c = p["color"]
    assert c[:, :20, :3].max() == 0 and (c[:, 30:, 0] == 255).all()
    assert abs(int(c[5, 25, 0]) - 128) <= 1
    m = p["color_mips"][0].astype(int)                              # no red fringe bleeding out of the transparent side
    assert (m[..., 0] <= m[..., 3] + 1).all()


def test_decode_server_round_trip(lib):
    dec = al._Decoder()
    try:
        a = dec.decode(lib / "a_one.png", al.depth_path(lib, "a_one.png"))
        ref = al.decode_piece(lib / "a_one.png", al.depth_path(lib, "a_one.png"))
        assert dec.mode == "process" and dec.proc is not None and dec.proc.poll() is None
        assert a["has_depth"] and a["aspect"] == ref["aspect"]
        assert np.array_equal(a["color"], ref["color"]) and np.array_equal(a["geo"], ref["geo"])
        assert len(a["color_mips"]) == len(ref["color_mips"]) and np.array_equal(a["geo_mips"][2], ref["geo_mips"][2])
        (lib / "bad.png").write_bytes(b"nope")
        with pytest.raises(Exception):
            dec.decode(lib / "bad.png", None)
        assert dec.proc.poll() is None                              # a bad file does not kill the server
        b = dec.decode(lib / "b_two.jpg", None, max_pixels=10 ** 7)
        assert not b["has_depth"]
        with pytest.raises(Exception):
            dec.decode(lib / "b_two.jpg", None, max_pixels=10)
        dec.proc.kill()                                             # a crashed server is restarted transparently
        dec.proc.wait(5)
        assert np.array_equal(dec.decode(lib / "a_one.png", None)["color"], ref["color"])
    finally:
        t0 = time.perf_counter()
        dec.close()
        assert time.perf_counter() - t0 < 0.5


def test_palette_and_colour_matrix_are_sane():
    assert al.palette_stops(0.0).shape == (5, 3)
    assert np.allclose(al.palette_stops(0.5), (al.palette_stops(0.0) + al.palette_stops(1.0)) / 2, atol=1e-6)
    assert np.isfinite(al.palette_stops(float("nan"))).all()
    assert np.allclose(al.color_matrix(0.0, 1.0), np.eye(3), atol=1e-9)        # defaults keep the true colours
    assert np.allclose(al.color_matrix(0.3, 1.2) @ np.ones(3), np.ones(3), atol=1e-6)   # white stays white


# ---------------------------------------------------------------------------------------------- rendering
def test_render_returns_rgba_texture_piece_in_the_middle_black_outside(ctx, lib):
    ap = make(ctx, lib)
    try:
        vals = al.base_values(art_index=0)
        assert al.wait_for(ap, vals, "a_one.png", timeout=15)
        tex = ap.render(vals, clock(1.0), 1.0, 0.0, 1 / 60)
        assert isinstance(tex, moderngl.Texture) and tex.size == SIZE and tex.components == 4
        assert ap.render(vals, clock(1.02), 1.02, 0.0, 1 / 60) is tex
        img = ap.read()
        assert img[80:120, 130:190, :3].mean() > 30                 # the painting is there
        assert corners(img) <= 2                                    # floating in darkness
        assert img[:, :3, :3].max() <= 2 and img[:, -3:, :3].max() <= 2      # no hard rectangle at the frame edge
        assert ap.status()["showing"] == "a_one.png" and ap.status()["depth"] is True
        assert "error" not in ap.stats
    finally:
        ap.release()


def test_art_index_picks_rounds_and_wraps(ctx, lib):
    ap = make(ctx, lib)
    try:
        for idx, want in ((0.0, "a_one.png"), (0.6, "b_two.jpg"), (1.4, "b_two.jpg"), (2.0, "a_one.png"), (63.0, "b_two.jpg")):
            assert al.wait_for(ap, al.base_values(art_index=idx), want, timeout=15), (idx, want)
    finally:
        ap.release()


EXTREMES = {
    "all max": dict(art_relief=1, cam_yaw=1, cam_pitch=1, cam_roll=1, cam_dolly=1, cam_drift=1, art_melt=1, art_liquify=1,
                    art_slices=1, art_breathe=1, art_fold=1, art_droste=1, art_smear=1, art_glow=1, zoom=1, swim=1,
                    twirl=1, ripple=1, folds=24, rotate=1, chase=1, chase_width=0.95, chase_rate=4, palette=4, hue=1,
                    saturation=1.5, tunnel_curve=1, punch=1),
    "all min": dict(art_relief=0, cam_yaw=-1, cam_pitch=-1, cam_roll=-1, cam_dolly=-1, cam_drift=0, art_melt=0, art_liquify=0,
                    art_slices=0, art_breathe=0, art_fold=0, art_droste=0, art_smear=0, art_glow=0, zoom=-1, swim=0,
                    twirl=-1, ripple=0, folds=2, rotate=-1, chase=0, chase_width=0.05, chase_rate=-4, palette=0, hue=0,
                    saturation=0, symmetry=0),
    "droste + fold deep": dict(art_droste=1, art_fold=1, folds=2, cam_dolly=1, art_relief=1, art_glow=1),
    "slices side on": dict(art_slices=1, cam_yaw=1, cam_pitch=-1, art_relief=1, cam_dolly=1),
    "out of range + junk": dict(art_relief=50, cam_yaw=-99, cam_dolly=1e9, art_melt=float("nan"), art_fold=float("inf"),
                                folds=0, art_droste=-3, chase="x", palette=1e12, art_index=1e6),
}


@pytest.mark.parametrize("name", list(EXTREMES))
def test_no_nan_and_black_background_at_param_extremes(ctx, lib, name):
    ap = make(ctx, lib)
    try:
        assert al.wait_for(ap, al.base_values(art_index=0), timeout=15)
        vals = al.base_values(**EXTREMES[name])
        for i, (t, tz) in enumerate([(0.0, 0.0), (0.016, 0.3), (7.7, -40.0), (36000.0, 1e7), (36000.02, 1e7)] * 3):
            tex = ap.render(vals, clock(t), t, tz, 1 / 60)
            f = scene_float(ap)
            assert np.isfinite(f).all(), (name, i)
            assert f.min() >= -1e-3, (name, i)
        assert tex.size == SIZE and "error" not in ap.stats
        img = ap.read()
        if name == "all min":                                       # far away and flat: must float on black
            assert corners(img) <= 2
        for bad in (float("nan"), float("inf")):                    # junk time / travel / dt never raises either
            ap.render(vals, {}, bad, bad, bad)
            ap.render(vals, None, 1.0, 0.0, -5.0)
        assert np.isfinite(scene_float(ap)).all() and "error" not in ap.stats
    finally:
        ap.release()


def test_gl_state_is_restored(ctx, lib):
    ap = make(ctx, lib)
    tex = ctx.texture((64, 48), 4)
    drb = ctx.depth_renderbuffer((64, 48))
    fbo = ctx.framebuffer([tex], drb)
    prog = ctx.program(
        vertex_shader="#version 330\nin vec2 p; void main(){ gl_Position = vec4(p, 0.5, 1.0); }",
        fragment_shader="#version 330\nout vec4 o; void main(){ o = vec4(1.0, 0.0, 0.0, 0.25); }")
    buf = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"))
    vao = ctx.vertex_array(prog, [(buf, "2f", "p")])
    try:
        assert al.wait_for(ap, al.base_values(art_index=0), timeout=15)
        fbo.use()
        ctx.viewport = (3, 5, 40, 30)
        fbo.clear(0.25, 0.5, 0.75, 1.0)
        for vals in (al.base_values(), al.base_values(art_slices=1, art_smear=1, art_droste=1), al.base_values(art_index=1)):
            for i in range(4):
                ap.render(vals, clock(i / 60), i / 60, 0.0, 1 / 60)
                assert ctx.fbo is fbo or ctx.fbo == fbo
                assert tuple(ctx.viewport) == (3, 5, 40, 30)
                assert tuple(fbo.viewport) == (3, 5, 40, 30)
        px = np.frombuffer(fbo.read(viewport=(0, 0, 64, 48), components=4), dtype=np.uint8)
        assert np.abs(px.reshape(-1, 4).astype(int) - np.array([64, 128, 191, 255])).max() <= 1   # never drawn on
        ap.read()
        assert ctx.fbo is fbo or ctx.fbo == fbo
        # depth test and blending are really off afterwards: a quad at z = 0.5 with alpha 0.25 must land
        # unblended even though the depth buffer says "nearer than everything" (cleared to 0)
        ctx.viewport = (0, 0, 64, 48)
        fbo.clear(0.0, 0.0, 1.0, 1.0, depth=0.0)
        ap.render(al.base_values(art_slices=1), clock(1.0), 1.0, 0.0, 1 / 60)
        vao.render(moderngl.TRIANGLE_STRIP)
        px = np.frombuffer(fbo.read(viewport=(0, 0, 64, 48), components=4), dtype=np.uint8).reshape(-1, 4)
        assert (px == np.array([255, 0, 0, 64])).all(), np.unique(px, axis=0)
        # and the check itself can tell: with the state our mesh pass uses, the same quad is rejected
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        fbo.clear(0.0, 0.0, 1.0, 1.0, depth=0.0)
        vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        px = np.frombuffer(fbo.read(viewport=(0, 0, 64, 48), components=4), dtype=np.uint8).reshape(-1, 4)
        assert (px == np.array([0, 0, 255, 255])).all()
    finally:
        ap.release()
        ctx.screen.use()
        for o in (vao, buf, prog, fbo, drb, tex):
            o.release()


def test_survives_the_callers_framebuffer_being_released(ctx, lib):
    tex = ctx.texture((32, 32), 4)
    fbo = ctx.framebuffer([tex])
    fbo.use()
    fbo.release()
    tex.release()
    ap = make(ctx, lib)                                                      # ctx.fbo is a dead object right now
    try:
        assert al.wait_for(ap, al.base_values(), timeout=15)
        assert "error" not in ap.stats and ap.read()[..., :3].max() > 60
    finally:
        ap.release()
        ctx.screen.use()


def test_no_gl_allocations_per_frame(ctx, lib, monkeypatch):
    ap = make(ctx, lib)
    try:
        vals = al.base_values(art_index=0, art_slices=0.5, art_smear=0.5, art_droste=0.5, art_melt=0.5)
        assert al.wait_for(ap, vals, timeout=15)
        made = []
        cls = type(ctx)
        for meth in ("texture", "texture_array", "texture3d", "texture_cube", "framebuffer", "simple_framebuffer",
                     "buffer", "program", "vertex_array", "simple_vertex_array", "renderbuffer", "depth_renderbuffer",
                     "depth_texture", "sampler", "query", "scope", "compute_shader"):
            orig = getattr(cls, meth, None)
            if orig is None:
                continue

            def wrap(self, *a, _o=orig, _m=meth, **k):
                made.append(_m)
                return _o(self, *a, **k)
            monkeypatch.setattr(cls, meth, wrap)
        before = ap.stats["gl_allocs"]
        for i in range(90):
            t = i / 60
            v = dict(vals, cam_yaw=np.sin(t), art_fold=i / 90, art_liquify=(i % 30) / 30)
            ap.render(v, clock(t), t, t * 0.5, 1 / 60)
        assert made == []
        assert ap.stats["gl_allocs"] == before
    finally:
        ap.release()


# ---------------------------------------------------------------------------------------------- bad input
def test_missing_and_empty_folder_render_black(ctx, tmp_path):
    for folder in (tmp_path / "does_not_exist", tmp_path):
        ap = make(ctx, folder)
        try:
            pump(ap, al.base_values(), 0.2)
            tex = ap.render(al.base_values(), clock(), 0.0, 0.0, 1 / 60)
            assert isinstance(tex, moderngl.Texture)
            assert ap.read().max() == 0 and "error" not in ap.stats
        finally:
            ap.release()


def test_corrupt_and_empty_files_render_black_and_do_not_block_good_ones(ctx, tmp_path):
    f = tmp_path / "art"
    f.mkdir()
    (f / "a_corrupt.png").write_bytes(os.urandom(4096))
    (f / "b_empty.jpg").write_bytes(b"")
    (f / "c_text.webp").write_text("definitely not an image")
    save_art(f, "d_good.png", 3)
    ap = make(ctx, f)
    try:
        for idx in (0, 1, 2):
            vals = al.base_values(art_index=idx)
            pump(ap, vals, 0.6)
            assert ap.read().max() == 0, idx
            assert ap.status()["showing"] is None
        assert ap.stats["decode_fail"] == 2 and "error" not in ap.stats      # the empty file is never even opened
        fails = ap.stats["decode_fail"]
        pump(ap, al.base_values(art_index=0), 0.4)
        assert ap.stats["decode_fail"] == fails                              # a bad file is not retried every frame
        assert al.wait_for(ap, al.base_values(art_index=3), "d_good.png", timeout=15)
        assert ap.read()[..., :3].max() > 60
        pump(ap, al.base_values(art_index=0), 1.0)                           # back to the corrupt one: fades to black
        assert ap.read().max() == 0
    finally:
        ap.release()


def test_half_written_file_is_black_then_loads_once_complete(ctx, tmp_path):
    f = tmp_path / "art"
    save_art(f, "piece.png", 2)
    whole = (f / "piece.png").read_bytes()
    (f / "piece.png").write_bytes(whole[: len(whole) // 3])                  # upload in progress
    ap = make(ctx, f)
    try:
        vals = al.base_values(art_index=0)
        pump(ap, vals, 0.6)
        assert ap.read().max() == 0 and ap.stats["decode_fail"] == 1
        (f / "piece.png").write_bytes(whole)                                 # upload finished
        ap._scan()
        assert al.wait_for(ap, vals, "piece.png", timeout=15)
        assert ap.read()[..., :3].max() > 60
    finally:
        ap.release()


def test_mtime_guard_holds_off_fresh_files(ctx, tmp_path):
    f = tmp_path / "art"
    save_art(f, "fresh.png", 1)
    ap = make(ctx, f, settle=2.0)
    try:
        vals = al.base_values(art_index=0)
        pump(ap, vals, 0.5)
        assert ap.read().max() == 0 and ap.stats["uploads"] == 0             # younger than 2 s: not touched
        assert ap.stats["decode_fail"] == 0
        old = time.time() - 10
        os.utime(f / "fresh.png", (old, old))
        ap._scan()
        assert al.wait_for(ap, vals, "fresh.png", timeout=15)
    finally:
        ap.release()


def test_piece_on_show_survives_while_its_depth_map_is_being_written(ctx, tmp_path):
    f = tmp_path / "art"
    save_art(f, "piece.png", 1)
    old = time.time() - 10
    os.utime(f / "piece.png", (old, old))
    ap = make(ctx, f, settle=2.0)
    try:
        vals = al.base_values(art_index=0)
        assert al.wait_for(ap, vals, "piece.png", timeout=15)
        assert ap.status()["depth"] is False
        save_art(f, "piece.png", 1, depth=True)                              # the AI worker drops a depth map
        os.utime(f / "piece.png", (old, old))
        ap._scan()
        pump(ap, vals, 0.4)
        assert ap.status()["showing"] == "piece.png" and ap.read()[..., :3].max() > 60   # still up, old version
        os.utime(f / "piece.depth.png", (old, old))                          # settled
        ap._scan()
        end = time.time() + 15
        while time.time() < end and not ap.status()["depth"]:
            pump(ap, vals, 0.1)
        assert ap.status()["depth"] is True and ap.status()["showing"] == "piece.png"
    finally:
        ap.release()


def test_folder_rescan_picks_up_new_and_deleted_files(ctx, tmp_path):
    f = tmp_path / "art"
    save_art(f, "a.png", 1)
    ap = make(ctx, f)
    ap.rescan_s = 0.1
    try:
        assert al.wait_for(ap, al.base_values(art_index=0), "a.png", timeout=15)
        save_art(f, "b.png", 2)
        assert al.wait_for(ap, al.base_values(art_index=1), "b.png", timeout=15)   # found by the background rescan
        assert ap.library() == ["a.png", "b.png"]
        os.remove(f / "a.png")
        os.remove(f / "b.png")
        end = time.time() + 10
        while time.time() < end and ap.library():
            time.sleep(0.05)
        pump(ap, al.base_values(art_index=1), 1.0)
        assert ap.library() == [] and ap.read().max() == 0                   # library emptied: fade to black
    finally:
        ap.release()


# ---------------------------------------------------------------------------------------------- cache / crossfade
def test_cache_keeps_at_most_three_pieces_and_releases_the_rest(ctx, tmp_path):
    f = tmp_path / "art"
    for i in range(6):
        save_art(f, f"p{i}.png", i, depth=(i % 2 == 0))
    ap = make(ctx, f)
    try:
        seen = []
        for i in list(range(6)) + [0, 3]:
            assert al.wait_for(ap, al.base_values(art_index=i), f"p{i}.png", timeout=15)
            assert len(ap._cache) <= al.CACHE_MAX
            seen += [p for p in ap._cache.values() if all(p is not q for q in seen)]
        assert ap.stats["uploads"] == 8 and ap.stats["evictions"] == 5
        live = list(ap._cache.values())
        assert len(live) == al.CACHE_MAX and len(seen) == 8
        for piece in seen:
            dead = isinstance(piece.color.mglo, moderngl.InvalidObject) and isinstance(piece.geo.mglo, moderngl.InvalidObject)
            assert dead == all(piece is not q for q in live)
        pieces = list(ap._cache.values())
    finally:
        ap.release()
    assert all(isinstance(p.color.mglo, moderngl.InvalidObject) for p in pieces)       # release() frees the cache


def test_piece_change_crossfades_over_about_a_second_and_a_half(ctx, lib):
    ap = make(ctx, lib)
    try:
        v0, v1 = al.base_values(art_index=0, cam_drift=0), al.base_values(art_index=1, cam_drift=0)
        assert al.wait_for(ap, v1, "b_two.jpg", timeout=15)
        assert al.wait_for(ap, v0, "a_one.png", timeout=15)             # both on the GPU now
        ap.render(v0, clock(), 0.0, 0.0, 1 / 60)
        start = ap.read().astype(np.float32)
        frames = []
        for i in range(120):                                            # 2 s of show time at 60 fps
            ap.render(v1, clock(), 0.0, 0.0, 1 / 60)
            if i in (1, 44, 119):
                frames.append(ap.read().astype(np.float32))
            if i == 44:
                assert ap._fading and ap.status()["incoming"] == "b_two.jpg"
            if i == 80:
                assert ap._fading                                       # 1.35 s in: still blending
        assert not ap._fading and ap.status()["showing"] == "b_two.jpg"
        d_early, d_mid, d_end = (np.abs(fr - start).mean() for fr in frames)
        assert d_early < 0.25 * d_end                                   # no pop at the start
        assert 0.15 * d_end < d_mid < 0.95 * d_end                      # genuinely in between
    finally:
        ap.release()


def test_release_stops_the_loader_thread_and_is_idempotent(ctx, lib):
    ap = make(ctx, lib)
    assert al.wait_for(ap, al.base_values(), timeout=15)
    ap.release()
    assert not ap._thread.is_alive()
    ap.release()


# ---------------------------------------------------------------------------------------------- review fixes
def big_piece(folder, name="big.png", n=1536):
    """A full-size (1536 x 1536) piece: the worst case for uploads."""
    folder.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    rgb = np.stack([0.5 + 0.5 * np.sin(x * 40), 0.5 + 0.5 * np.cos(y * 33), 0.5 + 0.5 * np.sin((x + y) * 21)], axis=2)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(folder / name, compress_level=1)


def test_upload_never_costs_a_frame_more_than_3ms(ctx, tmp_path):
    f = tmp_path / "art"
    big_piece(f)
    ap = make(ctx, f)
    try:
        times, orig = [], ap._upload_step

        def timed():
            ctx.finish()
            t0 = time.perf_counter()
            orig()
            ctx.finish()                                            # count the driver's share as well
            times.append((time.perf_counter() - t0) * 1000.0)
        ap._upload_step = timed
        vals = al.base_values(art_index=0)
        assert al.wait_for(ap, vals, "big.png", timeout=30)
        piece = next(iter(ap._cache.values()))
        assert piece.color.size == (1536, 1536) and piece.geo.size == (768, 768)
        busy = [x for x in times if x > 0.15]
        assert len(busy) >= 8                                       # spread over many frames, not one hitch
        # the scheduler can steal a slice from any one frame; two slow frames would be a real hitch
        assert sorted(times)[-2] < 3.0 and max(times) < 6.0, sorted(times)[-5:]
        assert ap.stats["upload_ms_max"] < 3.0
        img = ap.read()
        assert img[80:120, 130:190, :3].mean() > 40                 # and it really is on screen, mipmapped
        assert corners(img) <= 2
    finally:
        ap.release()


def test_tight_render_loop_without_sleeps_still_loads_quickly(ctx, tmp_path):
    f = tmp_path / "art"
    big_piece(f)
    ap = make(ctx, f)
    try:
        vals, t0, frames = al.base_values(art_index=0), time.perf_counter(), 0
        while ap._a is None and time.perf_counter() - t0 < 12:
            ap.render(vals, clock(frames / 60), frames / 60, 0.0, 1 / 60)      # no sleep: the loader gets no help
            frames += 1
        took = time.perf_counter() - t0
        assert ap._a is not None and took < 6.0, took
        assert ap.stats["decoder"] == "process"                     # decoded outside the render process' GIL
    finally:
        ap.release()


def test_oversized_file_dropped_into_the_folder_is_black_not_a_stall(ctx, tmp_path, monkeypatch):
    f = tmp_path / "art"
    save_art(f, "huge.png", 1, size=(400, 300))
    monkeypatch.setattr(al, "MAX_SOURCE_PIXELS", 400 * 300 - 1)
    ap = make(ctx, f)
    try:
        pump(ap, al.base_values(art_index=0), 1.5)
        assert ap.read().max() == 0 and ap.stats["decode_fail"] == 1 and ap.stats["uploads"] == 0
    finally:
        ap.release()


def test_piece_on_screen_is_kept_by_name_when_the_list_changes(ctx, tmp_path):
    f = tmp_path / "art"
    for i, n in enumerate(("m_first.png", "n_second.png", "o_third.png")):
        save_art(f, n, i)
        time.sleep(0.03)
    ap = make(ctx, f)
    try:
        vals = al.base_values(art_index=1)
        assert al.wait_for(ap, vals, "n_second.png", timeout=15)
        time.sleep(0.03)
        save_art(f, "a_upload.png", 7)                              # an upload: appended, nothing shifts
        ap._scan()
        assert ap.library() == ["m_first.png", "n_second.png", "o_third.png", "a_upload.png"]
        pump(ap, vals, 0.3)
        assert ap.status()["showing"] == "n_second.png"
        os.remove(f / "m_first.png")                                # indices shift under the piece on screen
        ap._scan()
        assert ap.library()[1] == "o_third.png"
        pump(ap, vals, 0.5)
        assert ap.status()["showing"] == "n_second.png" and not ap._fading    # same art_index: stay on the same NAME
        pump(ap, dict(vals, art_index=1.3), 0.3)                    # still rounds to 1: no re-resolve
        assert ap.status()["showing"] == "n_second.png"
        assert al.wait_for(ap, al.base_values(art_index=2), "a_upload.png", timeout=15)   # the value moved: re-resolve
        assert al.wait_for(ap, al.base_values(art_index=1), "o_third.png", timeout=15)
        os.remove(f / "o_third.png")                                # the piece on screen itself is deleted
        ap._scan()
        assert al.wait_for(ap, al.base_values(art_index=1), "a_upload.png", timeout=15)   # falls back to the index
    finally:
        ap.release()


def test_index_past_the_end_is_resolved_again_when_the_upload_shows_up(ctx, tmp_path):
    f = tmp_path / "art"
    save_art(f, "one.png", 1)
    time.sleep(0.03)
    save_art(f, "two.png", 2)
    ap = make(ctx, f)
    try:
        vals = al.base_values(art_index=2)                          # the panel already points at the new upload
        assert al.wait_for(ap, vals, "one.png", timeout=15)         # 2 % 2: a wrapped stand-in for now
        time.sleep(0.03)
        save_art(f, "three.png", 3)
        ap._scan()
        assert al.wait_for(ap, vals, "three.png", timeout=15)       # the file arrived: the index means it
        time.sleep(0.03)
        save_art(f, "four.png", 4)
        ap._scan()
        pump(ap, vals, 0.4)
        assert ap.status()["showing"] == "three.png"                # in range now: held by name from here on
    finally:
        ap.release()


def test_library_filling_up_later_shows_the_indexed_piece(ctx, tmp_path):
    f = tmp_path / "art"
    f.mkdir()
    ap = make(ctx, f)
    try:
        vals = al.base_values(art_index=0)
        pump(ap, vals, 0.2)
        assert ap.read().max() == 0
        save_art(f, "late.png", 4)
        ap._scan()
        assert al.wait_for(ap, vals, "late.png", timeout=15)
    finally:
        ap.release()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_clock_does_not_poison_the_pass(ctx, lib, bad):
    ap = make(ctx, lib)
    try:
        vals = al.base_values(art_index=0, art_breathe=1.0, art_relief=0.8, chase=1.0, art_droste=0.5, art_melt=0.5)
        assert al.wait_for(ap, vals, timeout=15)
        for i in range(5):
            ap.render(vals, dict(bpm=bad, beat=bad, beat_phase=bad, bar_phase=bad, phrase_phase=bad), 1.0 + i / 60, 0.0, 1 / 60)
            assert np.isfinite(scene_float(ap)).all()
        ap.render(vals, dict(beat="x", beat_phase=None), 1.2, 0.0, 1 / 60)
        for name in ("_pulse", "_flow_t", "_fold_rot", "_dzoom", "_chase_ph", "_hist_acc"):
            assert np.isfinite(getattr(ap, name)), name
        assert np.isfinite(ap._cam).all()
        for i in range(30):                                         # a healthy clock afterwards: everything is back
            t = 2.0 + i / 60
            ap.render(vals, clock(t), t, 0.0, 1 / 60)
        assert np.isfinite(scene_float(ap)).all() and ap.read()[..., :3].max() > 60
        assert ap._pulse > 0.01 and "error" not in ap.stats
        for name in ("_pulse", "_flow_t", "_dzoom", "_chase_ph"):   # poisoned state is repaired, not just avoided
            setattr(ap, name, bad)
        ap._cam = np.array([bad, 0.0, 0.0, 0.0])
        ap.render(vals, clock(3.0), 3.0, 0.0, 1 / 60)
        ap.render(vals, clock(3.02), 3.02, 0.0, 1 / 60)
        assert np.isfinite(scene_float(ap)).all() and ap.read()[..., :3].max() > 60
        assert all(np.isfinite(getattr(ap, n)) for n in ("_pulse", "_flow_t", "_dzoom", "_chase_ph")) and np.isfinite(ap._cam).all()
    finally:
        ap.release()


def test_gl_upload_failure_releases_textures_and_is_retried(ctx, lib):
    ap = make(ctx, lib)
    try:
        made, orig, state = [], ap._alloc, dict(n=0)

        def flaky(fn, *a, **k):
            state["n"] += 1
            if state["n"] == 2:                                     # the depth/normal texture of the first attempt
                raise RuntimeError("GL_OUT_OF_MEMORY (simulated)")
            tex = orig(fn, *a, **k)
            made.append(tex)
            return tex
        ap._alloc = flaky
        vals = al.base_values(art_index=0)
        assert al.wait_for(ap, vals, "a_one.png", timeout=15)       # the retry got through
        assert state["n"] == 4 and len(made) == 3
        assert isinstance(made[0].mglo, moderngl.InvalidObject)     # the orphaned colour texture was released
        assert not ap._failed and ap.stats["uploads"] == 1 and ap.stats["gl_fail"] == 1
        assert ap.read()[..., :3].max() > 60
    finally:
        ap.release()


def test_persistent_gl_failure_backs_off_instead_of_hammering(ctx, lib):
    ap = make(ctx, lib)
    try:
        calls = dict(n=0)

        def broken(fn, *a, **k):
            calls["n"] += 1
            raise RuntimeError("no texture memory (simulated)")
        ap._alloc = broken
        pump(ap, al.base_values(art_index=0), 1.2, dt=1 / 60)
        assert 1 <= calls["n"] <= 6                                 # a handful of spaced attempts, not one per frame
        assert ap.read().max() == 0 and "error" not in ap.stats and not ap._failed
    finally:
        ap.release()


def test_release_is_fast_even_in_the_middle_of_a_decode(ctx, tmp_path):
    f = tmp_path / "art"
    big_piece(f)
    ap = make(ctx, f)
    vals = al.base_values(art_index=0)
    end = time.time() + 5
    while time.time() < end and ap._busy is None:
        ap.render(vals, clock(), 0.0, 0.0, 1 / 60)
    t0 = time.perf_counter()
    ap.release()
    assert time.perf_counter() - t0 < 0.5
    proc = ap._decoder.proc
    assert proc is None or proc.wait(3) is not None                 # no orphaned decode server
    end = time.time() + 3
    while time.time() < end and ap._thread.is_alive():
        time.sleep(0.02)
    assert not ap._thread.is_alive()


def test_extreme_close_up_of_a_bright_painting_is_not_a_flat_white_wall(ctx, tmp_path):
    f = tmp_path / "art"
    f.mkdir()
    y, x = np.mgrid[0:300, 0:480].astype(np.float32)
    r = np.hypot(x - 240, y - 150) / 150
    lum = 0.93 + 0.07 * np.cos(r * 9)                               # ivory with faint rings: nearly white everywhere
    Image.fromarray((np.stack([lum, lum * 0.98, lum * 0.92], axis=2) * 255).astype(np.uint8)).save(f / "ivory.png")
    ap = make(ctx, f)
    try:
        vals = al.base_values(art_index=0, cam_dolly=1.0, art_relief=0.8, art_glow=1.0, cam_drift=0.0, art_breathe=1.0)
        assert al.wait_for(ap, vals, timeout=15)
        for i in range(40):
            ap.render(vals, clock(i / 60), i / 60, 2.0, 1 / 60)      # travel pushes in even further
        rgb = ap.read()[..., :3].astype(np.float32)
        lumo = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        assert (rgb.min(axis=2) >= 250).mean() < 0.30                # not blown out
        assert lumo.std() > 12 and lumo.mean() > 90                  # falloff across the frame, still a bright piece
        assert lumo[70:130, 110:210].mean() > lumo[:20, :30].mean() + 15       # lit in the middle, falling away
    finally:
        ap.release()
