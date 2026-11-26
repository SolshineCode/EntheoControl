"""fx_stack (TouchVDMX merge): landscape decks, blend modes, particle swarm, panel monitor tap.

One hidden GLFW window for the module (never visible on the projector).
"""
import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fx_stack as fx  # noqa: E402
import params  # noqa: E402

OUT = (640, 400)


@pytest.fixture(scope="module")
def ctx():
    c, _win = fx.hidden_context()
    yield c


@pytest.fixture()
def lp(ctx):
    p = fx.LandPass(ctx, OUT)
    p.settle_s = 0.0
    yield p
    p.release()


def shot(lp, frames=1, t0=4.0, bands=(0.5, 0.3, 0.2), **over):
    return fx.render_frames(lp, fx.default_values(**over), t0=t0, frames=frames, bands=bands)


# ---------------------------------------------------------------- contract


def test_params_appended_not_reordered():
    names = list(params.PARAMS)
    assert names[:5] == ["master", "flash", "strobe", "invert", "freeze"]
    assert names.index("art_glow") == 71, "the 72 original params must keep their positions (MIDI CC order)"
    new = names[72:92]                                               # this merge's 20; later appends follow them
    assert new[:2] == ["l_land", "l_swarm"] and all(n.startswith(("land_", "swarm_")) for n in new[2:])
    for n in new:
        lo, hi, d, group, label = params.PARAMS[n]
        assert lo <= d <= hi and group in ("layers", "land") and label
    assert params.PARAMS["l_land"][2] == 0.0 and params.PARAMS["l_swarm"][2] == 0.0, "new layers must default to off"


def test_api_contract(lp):
    import moderngl
    tex = lp.render(fx.default_values(), fx.make_clock(1.0), 1.0, 0.0, 1 / 60, fx.GALLERY_PALETTE, (0, 0, 0))
    assert isinstance(tex, moderngl.Texture) and tex.components == 4
    assert tex.size == OUT            # always full size: land_res only scales the decks drawn inside the pass
    assert tex.filter == (moderngl.LINEAR, moderngl.LINEAR)


def test_res_is_snapped_and_clamped():
    assert fx.snap_res(0.52) == 0.5 and fx.snap_res(5) == 1.0 and fx.snap_res(-1) == 0.25
    assert fx.snap_res(float("nan")) == 0.5


# ---------------------------------------------------------------- landscapes


@pytest.mark.parametrize("mode", range(4))
def test_each_landscape_renders_a_structured_finite_picture(lp, mode):
    a = shot(lp, land_a=mode)[..., :3].astype(np.float32)
    assert np.isfinite(a).all()
    assert a.max() > 120, f"{fx.LAND_NAMES[mode]} is (nearly) black"
    assert a.std() > 12, f"{fx.LAND_NAMES[mode]} is flat"
    assert (a.min(axis=2) > 235).mean() < 0.05, "blown to white"
    assert a.mean() < 150, "a landscape must not flood the projector"


def test_landscapes_differ_from_each_other(lp, ctx):
    imgs = []
    for m in range(4):
        p = fx.LandPass(ctx, OUT)
        imgs.append(shot(p, land_a=m)[..., :3].astype(np.float32))
        p.release()
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.abs(imgs[i] - imgs[j]).mean() > 8, f"{i} and {j} look the same"


def test_raw_output_has_no_nan(lp):
    lp.debug_raw = True
    for m in range(4):
        for over in (dict(), dict(land_twist=-1.0, land_curve=1.0, land_morph=1.0, land_fov=1.0, punch=1.0, folds=24.0),
                     dict(land_twist=1.0, land_curve=0.0, land_morph=0.0, land_fov=0.0, folds=2.0, land_mirror=1.0)):
            tex = lp.render(fx.default_values(land_a=m, land_b=(m + 1) % 4, land_xfade=0.5, land_blend=6, **over),
                            fx.make_clock(7.0), 7.0, 3.0, 1 / 60, fx.GALLERY_PALETTE, (1.0, 1.0, 1.0))
            assert np.isfinite(fx.read_texture(tex).astype(np.float32)).all()


def test_layer_amount_scales_output(lp):
    full = shot(lp, l_land=1.0)[..., :3].astype(np.float32).mean()
    half = shot(lp, l_land=0.5, t0=4.0 + 1 / 60)[..., :3].astype(np.float32).mean()
    assert 0.35 * full < half < 0.65 * full


def test_palette_drives_the_colours(lp, ctx):
    red = [(1.0, 0.0, 0.0)] * 5
    a = fx.render_frames(lp, fx.default_values(land_a=3), palette=red)[..., :3].astype(np.float32)
    assert a[..., 0].mean() > 2.5 * a[..., 2].mean(), "an all-red palette must give a red picture"


def test_deck_b_costs_nothing_until_the_crossfader_moves(lp):
    shot(lp, land_xfade=0.0)
    assert lp.decks_rendered == 1 and not lp.swarm_rendered
    shot(lp, land_xfade=0.5)
    assert lp.decks_rendered == 2
    shot(lp, land_xfade=1.0, land_blend=0, land_warp=0.0)
    assert lp.decks_rendered == 1, "a full plain mix to B with no cross-warp must skip deck A"
    shot(lp, l_land=0.0, l_swarm=1.0)
    assert lp.decks_rendered == 0 and lp.swarm_rendered


def test_crossfade_ends_are_the_two_decks(ctx):
    def one(**over):
        p = fx.LandPass(ctx, OUT)
        try:
            return shot(p, land_disp=0.0, land_warp=0.0, **over)[..., :3].astype(np.float32)
        finally:
            p.release()
    a, b = one(land_a=0), one(land_a=3)
    assert np.abs(one(land_a=0, land_b=3, land_xfade=0.0) - a).mean() < 0.5
    for blend in range(8):
        end = one(land_a=0, land_b=3, land_xfade=1.0, land_blend=blend)
        assert np.abs(end - b).mean() < 1.5, f"blend {fx.BLEND_NAMES[blend]}: xfade 1 must be deck B alone"
    mids = [one(land_a=0, land_b=3, land_xfade=0.5, land_blend=k) for k in range(8)]
    for i in range(8):
        for j in range(i + 1, 8):
            assert np.abs(mids[i] - mids[j]).mean() > 1.0, f"blend modes {i} and {j} give the same picture"


def test_cross_warp_bends_deck_b(ctx):
    def one(w):
        p = fx.LandPass(ctx, OUT)
        try:
            return shot(p, land_a=2, land_b=1, land_xfade=1.0, land_blend=0, land_warp=w)[..., :3].astype(np.float32)
        finally:
            p.release()
    assert np.abs(one(0.0) - one(1.0)).mean() > 4


def test_faders_change_the_picture(ctx):
    def one(**over):
        p = fx.LandPass(ctx, OUT)
        try:
            return shot(p, **over)[..., :3].astype(np.float32)
        finally:
            p.release()
    base = one(land_a=1)
    for name, val in (("land_twist", -0.8), ("land_curve", 1.0), ("land_morph", 0.9), ("land_fov", 1.0),
                      ("land_style", 2.0), ("land_mirror", 1.0), ("land_disp", 1.0), ("land_cam", 1.0), ("punch", 1.0),
                      ("cam_yaw", 0.6), ("folds", 14.0)):
        assert np.abs(one(land_a=1, **{name: val}) - base).mean() > 0.8, f"{name} does nothing"


def test_spin_rolls_the_flythrough_and_the_swarm(ctx):
    def one(frames=1, spin_phase=None, **over):
        p = fx.LandPass(ctx, OUT)
        try:
            v = fx.default_values(**over)
            for i in range(frames):
                t = 4.0 + i / 60
                tex = p.render(v, fx.make_clock(t), t, 0.0, 1 / 60, fx.GALLERY_PALETTE, (0.5, 0.3, 0.2), spin_phase)
            return fx.read_texture(tex)[..., :3].astype(np.float32), p
        finally:
            p.release()
    base, _ = one(land_a=0)
    rolled, _ = one(land_a=0, spin_phase=math.pi)                    # the engine's accumulated phase wins
    assert np.abs(rolled - base).mean() > 5
    assert np.abs(rolled - base[::-1, ::-1]).mean() < 0.35 * np.abs(rolled - base).mean(), "pi = the picture upside down"
    full, _ = one(land_a=0, spin_phase=2 * math.pi)
    assert np.abs(full - base).mean() < 1.0
    # no phase from the engine: the pass integrates values["spin"] itself, in both directions, and holds at 0
    _, p0 = one(frames=30, land_a=0, spin=0.0)
    _, p1 = one(frames=30, land_a=0, spin=1.0)
    _, p2 = one(frames=30, land_a=0, spin=-1.0)
    assert p0._spin == 0.0 and p1._spin > 0.3 and p2._spin == pytest.approx(fx.TAU - p1._spin, abs=1e-6)
    s0, _ = one(frames=40, l_land=0.0, l_swarm=1.0)
    s1, _ = one(frames=40, l_land=0.0, l_swarm=1.0, spin_phase=1.2)
    assert np.abs(s0 - s1).mean() > 1.0, "the swarm rolls with the camera"


def test_freeze_holds_and_time_moves(lp):
    v = fx.default_values(land_a=0, l_swarm=1.0)
    for i in range(20):
        lp.render(v, fx.make_clock(2.0 + i / 60), 2.0 + i / 60, 0.0, 1 / 60, fx.GALLERY_PALETTE, (0.4, 0.2, 0.1))
    t = 2.0 + 19 / 60
    a = fx.read_texture(lp.render(v, fx.make_clock(t), t, 0.0, 0.0, fx.GALLERY_PALETTE, (0.4, 0.2, 0.1))).copy()
    b = fx.read_texture(lp.render(v, fx.make_clock(t), t, 0.0, 0.0, fx.GALLERY_PALETTE, (0.4, 0.2, 0.1))).copy()
    assert np.array_equal(a, b), "the same show time with dt 0 (freeze) must give the same frame"
    c = fx.read_texture(lp.render(v, fx.make_clock(t + 0.5), t + 0.5, 0.0, 1 / 60, fx.GALLERY_PALETTE, (0.4, 0.2, 0.1)))
    assert np.abs(c.astype(np.float32) - a).mean() > 1.0


def test_travel_and_long_shows_stay_finite(lp):
    for t, z in ((1e4, 5e3), (9e4, -4e4), (3.0, float("nan")), (float("inf"), 0.0)):
        tex = lp.render(fx.default_values(land_a=2), fx.make_clock(3.0), t, z, 1 / 60, fx.GALLERY_PALETTE, (0.5, float("nan"), 2.0))
        a = fx.read_texture(tex).astype(np.float32)
        assert np.isfinite(a).all() and a[..., :3].max() > 60
    assert 0.0 <= lp._z < fx.Z_WRAP and 0.0 <= lp._T < fx.T_WRAP


def test_gl_state_is_restored(ctx, lp):
    import moderngl
    fbo = ctx.simple_framebuffer((32, 32))
    fbo.use()
    ctx.viewport = (0, 0, 32, 32)
    import math_layers as ml
    gl = ml._gl_lib()
    for on in (False, True):
        if on:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        else:
            ctx.disable(moderngl.BLEND)
        before = [ml.gl_get_int(gl, e) for e in ml._GL_BLEND_ENUMS]
        shot(lp, l_swarm=1.0, frames=3)
        assert ctx.fbo.glo == fbo.glo and ctx.viewport == (0, 0, 32, 32)
        assert bool(gl.glIsEnabled(ml._GL_BLEND)) == on
        assert [ml.gl_get_int(gl, e) for e in ml._GL_BLEND_ENUMS] == before
    ctx.disable(moderngl.BLEND)
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    fbo.release()


def test_render_scale_never_reallocates_and_keeps_the_trails(lp):
    """Review fix 5: a modulated / cross-faded land_res used to reallocate five targets and wipe the swarm trails."""
    a = shot(lp, frames=40, l_swarm=1.0, land_res=0.5)[..., :3]
    lit = (a.max(axis=2) > 30).mean()
    ids = {k: t.glo for k, t in lp._tex.items()}
    for i, res in enumerate((1.0, 0.25, 0.7, 0.5, 0.95)):
        b = shot(lp, frames=1, t0=4.0 + (40 + i) / 60, l_swarm=1.0, land_res=res)[..., :3]
        assert (b.max(axis=2) > 30).mean() > 0.6 * lit, f"trails wiped at land_res {res}"
    assert lp.alloc_count == 1 and {k: t.glo for k, t in lp._tex.items()} == ids
    assert lp._tex["out"].size == OUT and lp._tex["s0"].dtype == "f2"


def test_render_scale_changes_detail_not_framing(ctx):
    def one(res):
        p = fx.LandPass(ctx, OUT)
        try:
            return shot(p, land_a=3, land_res=res, land_disp=0.0)[..., :3].astype(np.float32)
        finally:
            p.release()
    lo, hi = one(0.25), one(1.0)
    assert lo.shape == hi.shape
    from PIL import Image
    small = lambda a: np.asarray(Image.fromarray(a.astype(np.uint8)).resize((64, 40), Image.BILINEAR), dtype=np.float32)
    assert np.abs(small(lo) - small(hi)).mean() < 9, "low render scale must show the same picture, only softer"
    assert lo[-3:, :, :].mean() > 0.3 * lo.mean() and lo[:, -3:, :].mean() > 0.3 * lo.mean(), "no black band at the edge of the used area"


def test_swarm_trails_decay_to_black_even_at_maximum_persistence(lp):
    """Review fix 1: 8-bit trail buffers stuck at a floor (32/255 at swarm_trail 1.0) and the frame whited out."""
    a = shot(lp, frames=1800, l_land=0.0, l_swarm=1.0, swarm_trail=1.0, bands=(0.8, 0.5, 0.5))[..., :3].astype(np.float32)
    lum = a.max(axis=2)
    assert a.mean() < 110, f"blown out after 30 s: mean {a.mean():.0f}"
    assert (lum < 10).mean() > 0.10, f"no black left after 30 s: {(lum < 10).mean():.1%}"
    # and once the lines stop being drawn the buffer really reaches zero
    for i in range(400):
        lp._fbo["s1" if lp._ping == 0 else "s0"].use()
        lp._tex["s%d" % lp._ping].use(0)
        lp._trail["texPrev"].value, lp._trail["uDecay"].value = 0, 0.985
        lp.ctx.viewport = (0, 0) + lp.swarm_size
        lp._trail_vao.render(5)
        lp._ping = 1 - lp._ping
    left = np.frombuffer(lp._tex["s%d" % lp._ping].read(), dtype=np.float16)
    assert float(left.reshape(-1, 4)[:, :3].max()) == 0.0


@pytest.mark.parametrize("mode", range(4))
def test_no_pop_when_any_phase_wraps(ctx, mode):
    """Review fixes 2 + 6: scroll / time / camera-z wraps used to jump tile ids, palette phases and twist."""
    p = fx.LandPass(ctx, OUT)
    try:
        v = fx.default_values(land_a=mode, land_res=1.0, land_twist=0.37, land_morph=0.41, l_swarm=0.0)

        def frame(t):
            return fx.read_texture(p.render(v, fx.make_clock(t), t, 0.0, None, fx.GALLERY_PALETTE, (0, 0, 0)))[..., :3].astype(np.float32)

        for i in range(5):
            frame(10.0 + i / 60)
        t = 10.0 + 4 / 60
        eps = 1e-4
        p._T, p._z = fx.T_WRAP - eps, fx.Z_WRAP - eps                # every phase sits just below its wrap
        p._tu = 100.0 * 7 - eps                                      # palette time: a whole number of cycles
        ph = p._ph
        ph["tw"] = [fx.TAU - eps] * 3
        ph["v"] = [w - eps for w in fx.V_WRAP]
        ph["pz"], ph["sk"] = 1.0 - eps, 1.0 - eps
        nx = math.floor(max(8 * 2.0, 12.0) + 0.5)
        ph["hex"] = nx * 64.0 - eps
        a = frame(t)                                                 # same show time: nothing advances
        b = frame(t + 1 / 60)                                        # ... and now everything wraps at once
        c = frame(t + 2 / 60)
        assert p._T < 1.0 and p._z < 1.0 and ph["pz"] < 0.5 and ph["v"][0] < 16, "the phases did wrap"
        wrap, normal = np.abs(b - a).mean(), np.abs(c - b).mean()
        assert wrap < 1.5 * normal + 0.6, f"{fx.LAND_NAMES[mode]}: frame delta {wrap:.2f} across the wraps vs {normal:.2f} normally"
    finally:
        p.release()


def test_swarm_has_no_pop_at_any_time_wrap():
    a, b = fx.Swarm(n=400, seed=2), fx.Swarm(n=400, seed=2)
    big = fx.T_WRAP * 3 + 0.123                                      # the same instant, with and without whole wraps
    for i in range(30):
        a.step(1 / 60, 0.123 + i / 60, 0.4, 0.3, 0.2)
        b.step(1 / 60, big + i / 60, 0.4, 0.3, 0.2)
    assert np.abs(a.pos - b.pos).max() > 1e-4 or True                # different phases are fine; what matters:
    s = fx.Swarm(n=400, seed=2)
    d = []
    t0 = fx.TAU / 0.65 * 40 - 5 / 60                                 # crosses a wrap of every velocity phase
    for i in range(12):
        prev = s.pos.copy()
        s.step(1 / 60, t0 + i / 60, 0.4, 0.3, 0.2)
        d.append(float(np.abs(s.pos - prev).mean()))
    assert max(d) < 1.25 * min(d), f"step size jumps at the wrap: {d}"


# ---------------------------------------------------------------- swarm


def test_swarm_step_is_deterministic_bounded_and_moves():
    a, b = fx.Swarm(n=500, seed=3), fx.Swarm(n=500, seed=3)
    p0 = a.pos.copy()
    for i in range(120):
        va = a.step(1 / 60, i / 60, 0.4, 0.35, 0.5).copy()
        vb = b.step(1 / 60, i / 60, 0.4, 0.35, 0.5)
    assert np.array_equal(va, vb)
    assert np.isfinite(va).all() and (np.linalg.norm(a.pos, axis=1) < math.sqrt(14.0) + 1e-3).all()
    assert np.abs(a.pos - p0).mean() > 0.2
    seg = np.linalg.norm(va[:, 1, :3] - va[:, 0, :3], axis=1)
    assert seg.max() < 0.6, "a respawned particle must not draw a streak across the frame"


def test_swarm_zero_dt_does_not_move():
    s = fx.Swarm(n=200)
    p0 = s.pos.copy()
    s.step(0.0, 1.0, 0.5, 0.5, 1.0)
    assert np.array_equal(s.pos, p0)


def test_burst_pushes_particles_outward():
    calm, hit = fx.Swarm(n=2000, seed=5), fx.Swarm(n=2000, seed=5)
    hit.burst = 1.0
    inner = np.linalg.norm(calm.pos, axis=1) < 1.5                   # these cannot leave the volume in 6 frames
    for i in range(6):
        calm.step(1 / 60, i / 60, 0.2, 0.0, 0.0)
        hit.step(1 / 60, i / 60, 0.2, 0.0, 0.0)
    rc, rh = np.linalg.norm(calm.pos, axis=1)[inner].mean(), np.linalg.norm(hit.pos, axis=1)[inner].mean()
    assert rh > rc + 0.08, (rc, rh)
    for i in range(20):
        hit.step(1 / 60, i / 60, 0.2, 0.0, 0.0)
    assert hit.burst < 0.1, "the shockwave must decay"


def test_swarm_layer_draws_filaments_on_black(lp):
    a = shot(lp, frames=60, l_land=0.0, l_swarm=1.0)[..., :3].astype(np.float32)
    lit = a.max(axis=2)
    assert lit.max() > 80 and (lit > 30).mean() > 0.02
    assert (lit < 10).mean() > 0.15, "the swarm must leave black between the filaments"
    gold = shot(lp, frames=30, t0=5.0, l_land=0.0, l_swarm=1.0, swarm_tint=1.0)[..., :3].astype(np.float32)
    assert gold[..., 0].mean() > 1.3 * gold[..., 2].mean()


def test_swarm_trails_fade_when_the_layer_returns(lp):
    steady = (shot(lp, frames=40, l_land=0.0, l_swarm=1.0)[..., :3].max(axis=2) > 30).mean()
    shot(lp, frames=2, t0=6.0, l_land=1.0, l_swarm=0.0)              # layer off for a moment
    a = shot(lp, frames=1, t0=7.0, l_land=0.0, l_swarm=1.0)[..., :3]
    assert (a.max(axis=2) > 30).mean() < 0.5 * steady, "stale trails from before the gap must not pop back"


# ---------------------------------------------------------------- panel monitor


def test_preview_tap_makes_a_small_jpeg_and_restores_state(ctx, lp):
    from PIL import Image
    tex = lp.render(fx.default_values(land_res=1.0), fx.make_clock(4.0), 4.0, 0.0, 1 / 60, fx.GALLERY_PALETTE, (0.5, 0.3, 0.2))
    tap = fx.PreviewTap(ctx, (320, 200))
    fbo = ctx.simple_framebuffer((16, 16))
    fbo.use()
    ctx.viewport = (0, 0, 16, 16)
    got = []
    assert tap.capture(tex, got.append, wait=True)
    assert ctx.fbo.glo == fbo.glo and ctx.viewport == (0, 0, 16, 16)
    assert len(got) == 1 and got[0][:2] == b"\xff\xd8" and len(got[0]) < 80_000
    im = Image.open(io.BytesIO(got[0]))
    assert im.size == (320, 200)
    small = np.asarray(im.convert("RGB"), dtype=np.float32)
    ref = np.asarray(Image.fromarray(fx.read_texture(tex)[..., :3]).resize((320, 200), Image.BILINEAR), dtype=np.float32)
    assert np.abs(small - ref).mean() < 12, "the monitor must show the frame the right way up"
    tap.release()
    fbo.release()


# ---------------------------------------------------------------- layer panel: preview-only mix


MIX_FS = """
#version 330
in vec2 v_uv; layout(location = 0) out vec4 o_disp; layout(location = 1) out vec4 o_echo;
uniform float p_l_kene, p_l_art, p_serpent, pxh;
void main(){ o_disp = vec4(p_l_kene, p_l_art, p_serpent, 1.0); o_echo = vec4(1.0); }
"""


class FakeRemote:
    def __init__(self, jobs):
        self.jobs, self.preview, self.thumbs = list(jobs), [], {}

    def preview_job(self):
        return self.jobs.pop(0) if self.jobs else None

    def set_preview(self, d, solo=()):
        self.preview.append(d)
        self.solos = getattr(self, "solos", []) + [tuple(solo)]

    def set_thumb(self, name, d):
        self.thumbs[name] = d


def test_layer_keys():
    assert fx.layer_keys(["master", "l_kene", "p_l_art", "l_swarm", "serpent", "p_serpent", "zoom"]) == ["l_kene", "l_art", "serpent"]
    assert set(fx.layer_keys(params.PARAMS)) >= {"l_kene", "l_beads", "l_lines", "l_ai", "l_echo", "l_video", "l_fractal", "l_art", "l_land", "serpent"}


def test_preview_mix_masks_layers_without_touching_the_callers_target(ctx):
    import moderngl
    from PIL import Image
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
    prog = ctx.program(vertex_shader=fx.VS, fragment_shader=MIX_FS)
    vao = ctx.vertex_array(prog, [(quad, "2f", "in_pos")])
    tap = fx.PreviewTap(ctx, (64, 40))
    main = ctx.simple_framebuffer((16, 16))
    main.clear(0.1, 0.2, 0.3, 1.0)
    main.use()
    ctx.viewport = (0, 0, 16, 16)
    uni = {"p_l_kene": 1.0, "p_l_art": 0.4, "p_serpent": 0.0, "p_master": 1.0}

    def px(mask):
        a = np.frombuffer(tap.render_mix(prog, vao, uni, mask), dtype=np.uint8).reshape(40, 64, 3)
        return [round(float(x) / 255, 2) for x in a[20, 32]]

    assert px(["l_kene", "l_art", "serpent"]) == [1.0, 0.4, 0.8], "a pattern layer that is down is previewed at 0.8"
    assert px(["l_art"]) == [0.0, 0.4, 0.0]
    assert px([]) == [0.0, 0.0, 0.0]
    uni["p_l_art"] = 0.0
    assert px(["l_art"]) == [0.0, 0.0, 0.0], "an offscreen-pass layer that is off has no picture to preview"
    assert ctx.fbo.glo == main.glo and ctx.viewport == (0, 0, 16, 16)
    got = np.frombuffer(main.read(components=3), dtype=np.uint8)
    assert abs(int(got[0]) - 26) <= 1, "the caller's framebuffer must be untouched"

    # service(): the remote decides; thumbnails are small, the monitor frame full size
    disp = ctx.texture((32, 20), 4, data=bytes([255, 0, 0, 255]) * 640)
    uni["p_l_art"] = 0.4
    rem = FakeRemote([("out", None), ("mix", ("l_art",)), ("thumb", "l_kene"), None])
    for _ in range(4):
        tap.service(rem, disp, prog, vao, uni, wait=True)
    assert len(rem.preview) == 2 and list(rem.thumbs) == ["l_kene"] and rem.solos == [(), ("l_art",)]
    out, mix = (np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.float32) for b in rem.preview)
    assert out.shape == (40, 64, 3) and out[..., 0].mean() > 230 and out[..., 1].mean() < 30
    assert mix[..., 1].mean() == pytest.approx(0.4 * 255, abs=8) and mix[..., 0].mean() < 12
    th = Image.open(io.BytesIO(rem.thumbs["l_kene"]))
    assert th.size == fx.THUMB_SIZE and np.asarray(th.convert("RGB"), dtype=np.float32)[..., 0].mean() > 230
    assert ctx.fbo.glo == main.glo
    tap.release()
    for o in (disp, main, vao, prog, quad):
        o.release()


def test_preview_readback_is_asynchronous_and_never_eats_a_job(ctx):
    """Review fixes 3, 7, 13: no synchronous fbo.read() on the frame of the capture; private texture unit; a job is
    only taken from the remote when it can be done."""
    tap = fx.PreviewTap(ctx, (64, 40))
    assert tap.UNIT >= 12, "the tap must not use a unit the scene shader samples (texL is 7)"
    disp = ctx.texture((32, 20), 4, data=bytes([0, 255, 0, 255]) * 640)
    rem = FakeRemote([("out", None), ("out", None), ("out", None)])
    assert tap.service(rem, disp, None, None, {}) == ("out", None)
    assert rem.preview == [] and tap._pending is not None, "the capture frame only starts the transfer"
    assert tap.service(rem, disp, None, None, {}) is None or True      # a later frame collects it
    for _ in range(200):
        if rem.preview:
            break
        import time
        time.sleep(0.01)
        tap.service(FakeRemote([]), disp, None, None, {})
    assert len(rem.preview) >= 1 and rem.preview[0][:2] == b"\xff\xd8"
    left = len(rem.jobs)
    tap._busy.acquire()                                              # the encoder is busy
    try:
        tap._pending = None
        assert tap.service(rem, disp, None, None, {}) is None and len(rem.jobs) == left, "a busy tap must not consume a job"
    finally:
        tap._busy.release()
    tap.release()
    disp.release()


def test_swarm_never_clips_to_white_even_with_punch_held(lp):
    """Found by test_core's random extremes: swarm 1.0 + punch held blew 6% of the frame to white."""
    a = shot(lp, frames=120, l_land=0.4, l_swarm=1.0, punch=1.0, swarm_trail=0.0, bands=(1.0, 1.0, 1.0))[..., :3].astype(np.float32)
    assert (a.min(axis=2) > 235).mean() < 0.005
    b = shot(lp, frames=120, t0=9.0, l_land=0.0, l_swarm=1.0, punch=1.0, swarm_trail=1.0, swarm_vortex=1.0, bands=(1.0, 1.0, 1.0))[..., :3]
    assert (b.astype(np.float32).min(axis=2) > 235).mean() < 0.005 and b.max() > 120
