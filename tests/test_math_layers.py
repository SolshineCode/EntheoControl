"""math_layers.FractalPass: every mode renders finite, projector-safe frames and leaves GL state alone.

Uses one hidden GLFW window for the whole module (never visible on the projector).
"""
import ctypes
import gc
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import math_layers as ml  # noqa: E402
import params  # noqa: E402

PAL = ml.GALLERY_PALETTES[0][1]
MODES = list(range(6))
OUT = (640, 400)              # small canvas keeps the suite fast; fractal_res scales it further


@pytest.fixture(scope="module")
def ctx():
    c, _win = ml.hidden_context()
    yield c


@pytest.fixture()
def fp(ctx):
    p = ml.FractalPass(ctx, OUT)
    yield p
    p.release()


def render(fp, mode, t=5.0, z=0.0, **over):
    tex = fp.render(ml.default_values(fractal_mode=float(mode), **over), ml.make_clock(t), t, z, PAL)
    return tex, ml.read_texture(tex)


def test_api_contract(fp):
    import moderngl
    tex, a = render(fp, 0)
    assert isinstance(tex, moderngl.Texture)
    assert tex.size == (320, 200) and tex.components == 4          # out_size * fractal_res (default 0.5), RGBA
    assert tex.filter == (moderngl.LINEAR, moderngl.LINEAR)
    assert a.shape == (200, 320, 4)
    assert set(params.PARAMS) <= set(ml.default_values())          # the pass is fed every PARAMS key


@pytest.mark.parametrize("mode", MODES)
def test_mode_renders_something_finite(fp, mode):
    _, a = render(fp, mode)
    assert np.isfinite(a.astype(np.float32)).all()
    assert a[..., :3].max() > 60, f"mode {mode} rendered (nearly) nothing"
    assert a[..., :3].std() > 3


@pytest.mark.parametrize("mode", MODES)
def test_float_readback_has_no_nan(ctx, fp, mode):
    """Render the same shader into a float target: NaN/inf would survive there (RGBA8 would hide them)."""
    tex = ctx.texture((160, 100), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[tex])
    real_tex, real_fbo, size = fp._tex, fp._fbo, fp._size
    try:
        fp._tex, fp._fbo, fp._size = tex, fbo, (160, 100)
        fp.out_size = (320, 200)
        for over in ({}, {"zoom": 1.0, "twirl": 1.0, "swim": 1.0}, {"zoom": -1.0, "twirl": -1.0}):
            fp.render(ml.default_values(fractal_mode=float(mode), **over), ml.make_clock(7.0), 7.0, 3.3, PAL)
            a = np.frombuffer(tex.read(), dtype="f4")
            assert np.isfinite(a).all(), f"mode {mode} {over}"
            assert a.min() >= 0.0 and a.max() <= 1.0001
    finally:
        fp._tex, fp._fbo, fp._size = real_tex, real_fbo, size
        fp.out_size = OUT
        fbo.release()
        tex.release()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("t,z", [(3.0, 0.0), (21.0, 2.0), (47.0, 4.7)])
def test_projection_safe_at_defaults(ctx, mode, t, z):
    """Black is no light: at least a quarter of the frame stays near-black, and nothing blows out to white."""
    a = ml.render_once(ctx, mode, t, z, PAL, out_size=(1280, 800))
    black, blown, mean, sat = ml.frame_stats(a)
    assert black >= 0.25, f"mode {mode} t={t}: only {black:.1%} near-black"
    assert blown < 0.01, f"mode {mode} t={t}: {blown:.2%} blown"
    assert mean > 2.0, f"mode {mode} t={t}: frame is nearly empty (mean {mean:.1f})"
    assert sat > 0.4, f"mode {mode} t={t}: colours are washed out ({sat:.0%} saturated)"


def _gl_state(ctx):
    lib = ml._gl_lib()
    ints = [ml.gl_get_int(lib, e) for e in (ml._GL_DRAW_FBO,) + ml._GL_BLEND_ENUMS]
    vp = (ctypes.c_int * 4)()
    lib.glGetIntegerv(0x0BA2, vp)                                   # GL_VIEWPORT
    return ints, tuple(vp), bool(lib.glIsEnabled(ml._GL_BLEND))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("blend", [False, True])
def test_gl_state_restored(ctx, fp, mode, blend):
    import moderngl
    host_tex = ctx.texture((96, 64), 4)
    host = ctx.framebuffer(color_attachments=[host_tex])
    try:
        host.use()
        ctx.viewport = (3, 5, 70, 40)
        if blend:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE, moderngl.ONE)
            ctx.blend_equation = moderngl.FUNC_ADD
        else:
            ctx.disable(moderngl.BLEND)
        before = _gl_state(ctx)
        render(fp, mode)
        assert _gl_state(ctx) == before
        assert before[0][0] == host.glo and before[1] == (3, 5, 70, 40) and before[2] == blend
        assert ctx.fbo is host and ctx.viewport == (3, 5, 70, 40)
    finally:
        ctx.disable(moderngl.BLEND)
        ctx.blend_func = moderngl.DEFAULT_BLENDING
        ctx.screen.use()
        host.release()
        host_tex.release()


def _n_gl_objects():
    import moderngl
    gc.collect()
    kinds = (moderngl.Texture, moderngl.Framebuffer, moderngl.Buffer, moderngl.Program, moderngl.VertexArray)
    return sum(isinstance(o, kinds) for o in gc.get_objects())


def test_no_per_frame_gl_allocations(fp):
    for m in MODES:                                                 # warm up every mode
        render(fp, m)
    tex0, fbo0, n0, alloc0 = fp._tex.glo, fp._fbo.glo, _n_gl_objects(), fp.alloc_count
    for i in range(60):
        tex, _ = render(fp, i % 6, t=5.0 + i / 60.0, z=i * 0.05)
        assert tex.glo == tex0 and fp._fbo.glo == fbo0
    assert fp.alloc_count == alloc0 == 1
    assert _n_gl_objects() == n0


def test_resolution_change_reallocates_exactly_once(fp):
    fp.settle_s = 0.0                                               # no settling delay: exact counts
    tex, _ = render(fp, 2)
    assert fp.alloc_count == 1 and tex.size == (320, 200)
    for _ in range(5):
        tex, _ = render(fp, 2, fractal_res=1.0)
    assert fp.alloc_count == 2 and tex.size == (640, 400)
    for _ in range(5):
        tex, _ = render(fp, 3, fractal_res=1.0)                     # a mode change is not a reallocation
    assert fp.alloc_count == 2
    tex, _ = render(fp, 3, fractal_res=0.25)
    assert fp.alloc_count == 3 and tex.size == (160, 100)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("which", ["min", "max"])
def test_extreme_params_do_not_nan(ctx, fp, mode, which):
    idx = 0 if which == "min" else 1
    v = {k: float(spec[idx]) for k, spec in params.PARAMS.items()}
    v["fractal_mode"], v["fractal_res"] = float(mode), 0.5
    tex = ctx.texture((160, 100), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[tex])
    real = fp._tex, fp._fbo, fp._size
    try:
        fp._tex, fp._fbo, fp._size = tex, fbo, (160, 100)
        fp.out_size = (320, 200)
        for t, z in ((0.0, 0.0), (9.0, -250.0), (1234.5, 1.0e4)):
            fp.render(v, ml.make_clock(t), t, z, PAL)
            a = np.frombuffer(tex.read(), dtype="f4")
            assert np.isfinite(a).all(), f"mode {mode} {which} t={t} z={z}"
    finally:
        fp._tex, fp._fbo, fp._size = real
        fp.out_size = OUT
        fbo.release()
        tex.release()


def test_bad_inputs_are_tolerated(fp):
    v = ml.default_values(fractal_mode=9.0)                         # out-of-range mode clamps
    fp.render(v, {}, 1.0, float("nan"), PAL)                        # empty clock, NaN travel
    fp.render(ml.default_values(fractal_mode=-3.0), ml.make_clock(2.0), 2.0, float("inf"), PAL[:3])   # short palette
    _, a = render(fp, 2)
    assert a[..., :3].max() > 60


def test_freeze_holds_the_frame(fp):
    """Engine model: on freeze the show time T stops but the clock keeps counting beats. Nothing may move."""
    for mode in (0, 4):                                             # chase phase (all modes) and Julia's bar-driven c
        v = ml.default_values(fractal_mode=float(mode), chase=1.0)
        fp.render(v, ml.make_clock(3.9), 3.9, 1.0, PAL)
        a = ml.read_texture(fp.render(v, ml.make_clock(4.0), 4.0, 1.0, PAL)).copy()
        v["freeze"] = 1.0
        for k in range(1, 5):                                       # 2 s of wall clock: beat advances, T does not
            b = ml.read_texture(fp.render(v, ml.make_clock(4.0 + 0.5 * k), 4.0, 1.0, PAL))
            assert np.array_equal(a, b), f"mode {mode}: frame drifted during freeze"
        v["freeze"] = 0.0
        fp.render(v, ml.make_clock(6.1), 4.1, 1.0, PAL)
        c = ml.read_texture(fp.render(v, ml.make_clock(6.6), 4.6, 1.4, PAL))
        assert not np.array_equal(a, c)


def test_speed_is_not_applied_twice(fp):
    """t is already speed-scaled show time (the engine integrates p_speed): the pass must not scale it again."""
    for i in range(31):
        fp.render(ml.default_values(speed=2.0, rotate=1.0), ml.make_clock(i * 0.1), i * 0.1, 0.0, PAL)
    assert fp._T == pytest.approx(3.0)
    assert fp._rot == pytest.approx(3.0 * 0.6)
    slow = ml.FractalPass(fp.ctx, OUT)
    try:
        for i in range(31):
            slow.render(ml.default_values(speed=0.0, rotate=1.0), ml.make_clock(i * 0.1), i * 0.1, 0.0, PAL)
        assert slow._T == pytest.approx(3.0) and slow._rot == pytest.approx(fp._rot)
    finally:
        slow.release()


def test_beat_resync_does_not_jump(fp):
    v = ml.default_values(chase_rate=1.0)
    clock = lambda beat: dict(bpm=124.0, beat=beat, beat_phase=beat % 1, bar_phase=(beat / 4) % 1, phrase_phase=0.0)
    fp.render(v, clock(2000.0), 1.0, 0.0, PAL)                      # first sight of a large beat: no jump either
    assert fp._bars == 0.0 and fp._chase == 0.0
    fp.render(v, clock(2000.5), 1.1, 0.0, PAL)
    bars, chase = fp._bars, fp._chase
    assert bars == pytest.approx(0.125) and chase == pytest.approx(0.125)
    fp.render(v, clock(0.0), 1.2, 0.0, PAL)                         # tap-tempo / resync: beat snaps back to 0
    assert (fp._bars, fp._chase) == (bars, chase)
    fp.render(v, clock(37.0), 1.3, 0.0, PAL)                        # and a forward jump
    assert (fp._bars, fp._chase) == (bars, chase)
    fp.render(v, clock(37.25), 1.4, 0.0, PAL)                       # normal running resumes
    assert fp._bars == pytest.approx(bars + 0.0625)


def test_fractal_res_modulation_does_not_churn(fp):
    """A modulator wobbling fractal_res must not reallocate every frame: res snaps to 0.05 and has to settle."""
    render(fp, 0)
    assert fp.alloc_count == 1
    for i in range(40):
        tex, _ = render(fp, 0, t=5.0 + i / 60.0, fractal_res=0.5 + 0.02 * math.sin(i * 0.7))   # inside one 0.05 step
    assert fp.alloc_count == 1
    for i in range(40):
        tex, _ = render(fp, 0, t=6.0 + i / 60.0, fractal_res=0.3 + 0.6 * (i % 7) / 6.0)        # wild sweep, never settles
    assert fp.alloc_count == 1 and tex.size == (320, 200)
    assert ml.snap_res(0.52) == pytest.approx(0.5) and ml.snap_res(0.53) == pytest.approx(0.55)
    assert ml.snap_res(0.0) == pytest.approx(0.25) and ml.snap_res(7.0) == pytest.approx(1.0)


def test_high_res_caps_raymarch_cost(fp):
    """Above res 0.75 the raymarchers cap effective detail at 0.6 (frame-time guard on the 1650 Ti)."""
    fp.settle_s = 0.0
    for mode in (0, 1):
        render(fp, mode, fractal_res=0.5, fractal_detail=1.0)
        full = (fp._uni[mode]["uSteps"].value, fp._uni[mode]["uIters"].value)
        render(fp, mode, fractal_res=1.0, fractal_detail=1.0)
        capped = (fp._uni[mode]["uSteps"].value, fp._uni[mode]["uIters"].value)
        render(fp, mode, fractal_res=1.0, fractal_detail=0.6)
        at06 = (fp._uni[mode]["uSteps"].value, fp._uni[mode]["uIters"].value)
        assert capped == at06 and capped[0] < full[0]


@pytest.mark.parametrize("mode", MODES)
def test_raw_prefinish_buffer_has_no_nan(ctx, fp, mode):
    """finish() has a NaN guard and a clamp that would hide bad maths: read the colour BEFORE it (debug_raw)."""
    tex = ctx.texture((160, 100), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[tex])
    real = fp._tex, fp._fbo, fp._size
    try:
        fp._tex, fp._fbo, fp._size = tex, fbo, (160, 100)
        fp.out_size = (320, 200)
        fp.debug_raw = True
        lo = {k: float(spec[0]) for k, spec in params.PARAMS.items()}
        hi = {k: float(spec[1]) for k, spec in params.PARAMS.items()}
        for base in (ml.default_values(), lo, hi, ml.default_values(zoom=1.0, twirl=1.0, swim=1.0)):
            v = dict(base, fractal_mode=float(mode), fractal_res=0.5)
            for t, z in ((0.0, 0.0), (7.0, 3.3), (1234.5, -250.0)):
                fp.render(v, ml.make_clock(t), t, z, PAL)
                a = np.frombuffer(tex.read(), dtype="f4").reshape(-1, 4)[:, :3]
                assert np.isfinite(a).all(), f"mode {mode} t={t} z={z}: {(~np.isfinite(a)).sum()} non-finite raw values"
                assert a.min() >= 0.0
        assert a.max() > 0.0 or mode in (0, 1)
    finally:
        fp.debug_raw = False
        fp._tex, fp._fbo, fp._size = real
        fp.out_size = OUT
        fbo.release()
        tex.release()


def test_blend_state_restored_when_draw_raises(ctx, fp):
    import moderngl
    render(fp, 5)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE, moderngl.ONE)
    before = _gl_state(ctx)

    class Boom:
        def render(self, *a, **k):
            raise RuntimeError("draw failed")
    real = fp._vaos[5]
    fp._vaos[5] = Boom()
    try:
        with pytest.raises(RuntimeError):
            render(fp, 5)
        assert _gl_state(ctx) == before
    finally:
        fp._vaos[5] = real
        ctx.disable(moderngl.BLEND)
        ctx.blend_func = moderngl.DEFAULT_BLENDING


def test_gl_boolean_signature():
    lib = ml._gl_lib()
    assert lib.glIsEnabled.restype is ctypes.c_ubyte and lib.glIsEnabled.argtypes == [ctypes.c_uint]


@pytest.mark.parametrize("mode", MODES)
def test_travel_moves_the_camera(fp, mode):
    _, a = render(fp, mode, t=5.0, z=0.0)
    a = a.copy()
    _, b = render(fp, mode, t=5.0, z=0.37)
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 0.5, f"mode {mode}: travel_z has no visible effect"


def test_hyperbolic_view_wraps_seamlessly():
    """The Moebius view is periodic: travelling one period returns the identical transform (endless walk)."""
    for P, Q in ml.TILINGS:
        _, _, period = ml.hyperbolic_view(P, Q, 0.0)
        for d in (0.3, 5.0, 123.4):
            a0, b0, _ = ml.hyperbolic_view(P, Q, d)
            a1, b1, _ = ml.hyperbolic_view(P, Q, d + 7 * period)
            assert abs(a0 - a1) < 1e-9 and abs(b0 - b1) < 1e-9
            assert abs(abs(a0) ** 2 - abs(b0) ** 2 - 1.0) < 1e-9     # a disk automorphism
            assert abs(b0) < math.sinh(period / 4) * 1.5 + 1e-9      # wrapped: never loses float precision


def test_hyperbolic_period_is_a_tiling_symmetry(ctx):
    """Frames one period apart must match (the wrap in hyperbolic_view is invisible)."""
    P, Q = ml.TILINGS[2]
    _, _, period = ml.hyperbolic_view(P, Q, 0.0)
    eps = 1e-3
    a = ml.render_once(ctx, 2, 5.0, (period / 2 - eps) / 0.7, PAL, out_size=OUT, fractal_res=1.0)
    b = ml.render_once(ctx, 2, 5.0, (period / 2 + eps) / 0.7, PAL, out_size=OUT, fractal_res=1.0)
    c = ml.render_once(ctx, 2, 5.0, (period / 4) / 0.7, PAL, out_size=OUT, fractal_res=1.0)
    wrap = np.abs(a.astype(int) - b.astype(int)).mean()            # only sub-pixel line jitter may differ
    moved = np.abs(a.astype(int) - c.astype(int)).mean()
    assert wrap < 0.2 * moved, (wrap, moved)


def test_polytope_edge_counts():
    e = ml._polytope_edges()
    ids = e[:, 8].astype(int)
    assert [(ids == i).sum() for i in range(3)] == [32, 96, 720]    # tesseract, 24-cell, 600-cell
    assert np.allclose(np.linalg.norm(e[:, :4], axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("morph", [0.0, 0.3, 0.5, 0.7, 1.0, float("nan")])
@pytest.mark.parametrize("phase", [0.0, 2.0, 4.5])
def test_julia_zooms_into_a_repelling_spiral_centre(morph, phase):
    """The Julia zoom centre is the alpha fixed point of a p/q bulb: repelling (|lambda| > 1, or the zoom loop
    never leaves it), turning p/q of a turn per period (q spiral arms), and the set is connected (the critical orbit
    stays bounded: filaments, not dust and straight escape lines)."""
    import cmath
    c, a, lam = ml.julia_constants(morph, phase)
    assert abs(a * a + c - a) < 1e-9 and abs(lam - 2 * a) < 1e-12
    assert abs(lam) > 1.02
    i = min(int((0.5 if morph != morph else morph) * len(ml.JULIA_BULBS)), len(ml.JULIA_BULBS) - 1)
    p, q = ml.JULIA_BULBS[i]
    assert abs((cmath.phase(lam) / (2 * math.pi)) % 1.0 - p / q) < 0.01
    z = 0j
    for _ in range(500):
        z = z * z + c
    assert abs(z) < 2.0
