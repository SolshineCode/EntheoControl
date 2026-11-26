"""Coil nest layer (coilnest.CoilNestPass): param contract, off = zero cost, NaN guard, black floor, gating programme
flash / slew caps, UV continuity along every body (incl. the wrap of closed coils), deterministic phases over hours,
skin sources (procedural, 4-channel layer, colour layer), cost.

GL tests share one hidden GLFW window (never visible on the projector).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coilnest as cn  # noqa: E402
import params  # noqa: E402

OUT = (640, 400)


# ---------------------------------------------------------------- param contract (CPU)


def test_params_are_one_appended_block_default_off():
    names = list(params.PARAMS)
    assert names.index("art_glow") == 71, "the original params keep their positions (MIDI CC order)"
    i = names.index("l_coilnest")
    assert names[i:i + len(cn.PARAM_NAMES)] == list(cn.PARAM_NAMES), "the coil nest params are one appended block"
    assert params.PARAMS["l_coilnest"][2] == 0.0, "off by default"
    assert params.PARAMS["l_coilnest"][3] == "layers"
    for n in cn.PARAM_NAMES[1:]:
        assert params.PARAMS[n][3] == "coilnest", n
    lo, hi, d = params.PARAMS["coil_count"][:3]
    assert (lo, hi) == (6.0, 14.0) and lo <= d <= hi


def test_options_contract():
    for name, opts in cn.OPTIONS.items():
        assert params.OPTIONS[name] == opts, name
        lo, hi = params.PARAMS[name][:2]
        assert len(opts) == int(hi - lo) + 1, name
    assert params.OPTIONS["coil_stage"] == cn.STAGES and len(cn.STAGES) == 5


def test_bypass_classified_and_tab_switch_drops_the_layer():
    assert params.NEUTRAL["l_coilnest"] == 0.0
    for n in cn.PARAM_NAMES[1:]:
        assert n in params.BYPASS_KEEP and n not in params.NEUTRAL, n
    assert params.BYPASS_LINKS["coilnest"] == ["l_coilnest"]
    assert "l_coilnest" in params.bypass_cover()["coilnest"]


def test_engine_treats_the_group_as_cpu_only_and_has_a_lane():
    import clips
    import live_visuals as lv
    for n in cn.PARAM_NAMES[1:]:
        assert n in lv.CPU_ONLY, n
    assert clips.lane_of("l_coilnest") == "figures" and clips.lane_of("coil_count") == "figures"


def test_governor_knows_the_count_and_the_render_scale():
    import governor
    ids = {s.id for s in governor.build_steps()}
    assert "count:coil_count" in ids and "res:coilnest_res" in ids


# ---------------------------------------------------------------- geometry (CPU)


def _nest(n=10, **over):
    v = cn.default_values(coil_count=n, **over)
    nest = cn.Nest()
    nest.step(0.0, v, cn.make_clock(0.0))
    return nest, v


def test_knots_are_finite_and_frame_filling_at_every_count():
    for n in range(6, 15):
        nest, v = _nest(n)
        a, b = nest.knots
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert nest.n == n
        assert (a[:n, :, 2] > 0).all(), "screen radius must stay positive (taper floor)"


def test_uv_continuity_along_every_body_including_the_wrap():
    nest, v = _nest(14)
    for i in range(nest.n):
        s = np.linspace(0.0, nest.last[i] + (1 if nest.closed[i] else 0), 4001)
        P, D, r, z, U, R = cn.hermite(nest, i, s)
        assert np.isfinite(P).all() and (np.linalg.norm(D, axis=1) > 1e-5).all(), i
        du = np.diff(U)
        assert (du >= -1e-9).all(), f"u must run monotonic along body {i}"
        step = np.linalg.norm(np.diff(P, axis=0), axis=1)
        # u advances like arc length (world units): no jumps anywhere
        assert du.max() < 6.0 * step.max() / max(nest.scale_min, 1e-3) + 1e-6, i
        q = U * nest.cells[i]                        # pattern coordinate in cells
        if nest.closed[i]:
            n_cells = q[-1] - q[0]
            assert abs(n_cells - round(n_cells)) < 1e-6, "a closed coil holds a whole number of cells"
            assert round(n_cells) % 2 == 0 and round(n_cells) == nest.wrap[i]
            assert np.allclose(P[0], P[-1], atol=1e-6), "closed coil closes"
        else:
            assert nest.wrap[i] == cn.OPEN_WRAP


def test_scroll_phases_wrap_on_whole_pattern_periods():
    nest, v = _nest(14)
    clock = cn.make_clock(0.0)
    for _ in range(200):
        nest.step(1.0, dict(v, coil_slither=1.0), clock)
    for i in range(nest.n):
        assert 0.0 <= nest.scroll[i] < nest.wrap[i]
        assert nest.wrap[i] % 2 == 0


def test_phases_deterministic_over_hours():
    v = cn.default_values()
    a, b = cn.Nest(), cn.Nest()
    clock = cn.make_clock(0.0)
    for k in range(6 * 3600):                        # six hours in 1 s steps
        a.step(1.0, v, clock, geometry=False)
    for k in range(6 * 3600 * 4):                    # the same six hours in 0.25 s steps
        b.step(0.25, v, clock, geometry=False)
    a.step(0.0, v, clock)
    circ = lambda x, y, p: abs((x - y + p / 2) % p - p / 2)
    assert circ(a.ph_writhe, b.ph_writhe, cn.TAU) < 1e-6 and circ(a.ph_dolly, b.ph_dolly, cn.TAU) < 1e-6
    assert 0 <= a.ph_writhe < cn.TAU and 0 <= a.ph_dolly < cn.TAU
    for i in range(a.n):
        assert circ(a.scroll[i], b.scroll[i], a.wrap[i]) < 1e-5
    c = cn.Nest()                                    # geometry depends only on the wrapped phases
    c.ph_writhe, c.ph_dolly = a.ph_writhe, a.ph_dolly
    c.step(0.0, v, clock)
    assert np.allclose(c.knots[0], a.knots[0], atol=1e-5)
    assert np.abs(a.knots[0]).max() < 50 and np.abs(a.knots[1]).max() < 1e4    # float32 friendly


def test_nan_and_inf_values_are_guarded():
    bad = {n: float("nan") for n in cn.PARAM_NAMES}
    bad.update(coil_count=float("inf"), coil_thick=-1e30)
    nest = cn.Nest()
    prog = cn.Programme()
    clock = dict(beat=float("nan"), bar_phase=float("inf"), bpm=float("nan"))
    nest.step(float("nan"), bad, clock)
    g = prog.step(float("inf"), bad, clock, float("nan"))
    assert np.isfinite(nest.knots[0]).all() and np.isfinite(nest.knots[1]).all()
    assert np.isfinite(g).all() and 6 <= nest.n <= 14


# ---------------------------------------------------------------- gating programme (CPU)


def _lum(g):
    return float(np.dot(g[:5], cn.LUM_W))


def test_programme_walks_the_five_stages_on_the_bar():
    prog = cn.Programme()
    v = cn.default_values(coil_prog=1, coil_bars=1)  # 4 bars per stage
    seen = []
    for k in range(5):
        beat = (k * 4 + 2) * 4.0                     # the middle of stage k (bars 2 of 4)
        seen.append(prog.target(v, cn.make_clock_beats(beat))[0])
    assert [int(x) for x in seen] == [0, 1, 2, 3, 4]
    g0 = prog.target(v, cn.make_clock_beats(8.0))[1]
    g3 = prog.target(v, cn.make_clock_beats(56.0))[1]
    assert g0[cn.G_GROUND] < 0.05 and g0[cn.G_LINE] > 0.9, "stage 0 = linework only"
    assert g3[cn.G_GROUND] > 0.9 and g3[cn.G_FILL] > 0.5, "stage 3 = full embroidered cloth"


def test_manual_stage_and_intensity():
    prog = cn.Programme()
    g = prog.target(cn.default_values(coil_prog=0, coil_stage=1.0), cn.make_clock(0))[1]
    assert g[cn.G_MONO] > 0.7 and g[cn.G_GROUND] < 0.05, "white engraving"
    full = prog.target(cn.default_values(coil_prog=0, coil_stage=1.0, coil_prog_amt=0.0), cn.make_clock(0))[1]
    assert full[cn.G_GROUND] > 0.9, "intensity 0 = the full cloth, no gating"


def test_gains_slew_and_never_flash():
    """Tempo jumps, stage jumps every frame and a MIDI fader wiggled at 5-15 Hz: the luminance proxy changes slowly
    and has at most 3 opposing >=10 % swings in any 1 s window (WCAG 2.3.1 general flash threshold)."""
    rng = np.random.default_rng(3)
    prog = cn.Programme()
    dt, lum, beat = 1 / 60, [], 0.0
    for k in range(60 * 120):
        tt = k * dt
        v = cn.default_values(coil_prog=int(rng.random() < 0.5), coil_stage=float(rng.integers(0, 5)),
                              coil_bars=int(rng.integers(0, 4)))
        beat = beat + dt * 2.0 if rng.random() > 0.01 else rng.random() * 1e4      # tempo / position jumps
        bass = 1.0 if math.sin(tt * math.tau * (5 + 10 * rng.random())) > 0 else 0.0
        g = prog.step(dt, v, cn.make_clock_beats(beat), bass)
        lum.append(_lum(g))
    lum = np.array(lum)
    rate = np.abs(np.diff(lum)) / dt
    assert rate.max() <= cn.MAX_LUM_RATE + 1e-6, f"luminance moved {rate.max():.2f}/s"
    # count opposing swings of >= 0.1 (relative to the proxy's full range) inside every 1 s window
    full = float(np.sum(cn.LUM_W))
    ext, last, dirn = [], lum[0], 0
    for i, x in enumerate(lum):
        d = x - last
        if abs(d) >= 0.1 * full:
            nd = 1 if d > 0 else -1
            if nd != dirn:
                ext.append(i)
                dirn = nd
            last = x
        elif (dirn > 0 and x > last) or (dirn < 0 and x < last):
            last = x
    ext = np.array(ext) * dt
    for t0 in ext:
        assert np.sum((ext >= t0) & (ext < t0 + 1.0)) <= 3 * 2, "more than 3 flashes in a second"


def test_bass_lift_is_small_and_smooth():
    prog = cn.Programme()
    v = cn.default_values(coil_prog=0, coil_stage=3.0)
    for _ in range(300):
        prog.step(1 / 60, v, cn.make_clock(0), 0.0)
    base = prog.lift
    vals = []
    for k in range(600):
        g = prog.step(1 / 60, v, cn.make_clock(0), 1.0 if (k // 4) % 2 else 0.0)     # 7.5 Hz bass square
        vals.append(prog.lift)
    vals = np.array(vals[120:])
    assert vals.max() - base <= cn.BASS_LIFT + 1e-6
    assert vals.max() - vals.min() < 0.08, "a fast bass square must not pump the brightness"


# ---------------------------------------------------------------- GL


@pytest.fixture(scope="module")
def ctx():
    c, _win = cn.hidden_context()
    yield c


@pytest.fixture()
def cp(ctx):
    p = cn.CoilNestPass(ctx, OUT)
    yield p
    p.release()


def shot(cp, t=4.0, frames=1, **over):
    return cn.render_frames(cp, cn.default_values(**over), t0=t, frames=frames)


def test_off_is_zero_cost(cp):
    v = cn.default_values(l_coilnest=0.0)
    tex = cp.render(v, cn.make_clock(1.0), 1.0, 1 / 60)
    assert cp.draws == 0
    img = cn.read_texture(tex)
    assert img.max() == 0


def test_renders_a_frame_filling_nest_with_black_crevices(cp):
    img = shot(cp, coil_prog=0, coil_stage=3.0)
    L = img[..., :3].max(axis=2) / 255.0
    assert cp.draws >= 1
    lit = (L > 0.06).mean()
    assert lit > 0.35, f"the nest should fill the frame (lit {lit:.2f})"
    dark = (L < 3 / 255).mean()
    assert 0.02 < dark < 0.6, f"pitch-black crevices must exist (dark {dark:.2f})"
    assert L.min() == 0.0
    assert (L > 0.98).mean() < 0.01, "never blown out"
    # every quadrant has bodies in it (no empty background)
    h, w = L.shape
    for q in (L[: h // 2, : w // 2], L[: h // 2, w // 2:], L[h // 2:, : w // 2], L[h // 2:, w // 2:]):
        assert (q > 0.06).mean() > 0.2


def test_outside_the_bodies_is_exactly_black(cp):
    img = shot(cp, coil_prog=0, coil_stage=3.0, coil_floor=0.0, coil_count=6)
    cov = img[..., 3]
    rgb = img[..., :3].max(axis=2)
    assert (rgb[cov == 0] == 0).all(), "light outside every body"


def test_every_stage_reads_differently_and_stays_in_range(cp):
    means, sats = [], []
    for s in range(5):
        img = shot(cp, coil_prog=0, coil_stage=float(s)).astype(np.float32) / 255
        rgb = img[..., :3]
        means.append(rgb.max(axis=2).mean())
        sats.append((rgb.max(axis=2) - rgb.min(axis=2)).mean())
        assert (rgb.max(axis=2) > 0.98).mean() < 0.01
    assert sats[3] > 1.5 * sats[1], "embroidered cloth is far more saturated than white engraving"
    assert means[4] < means[3] * 0.6, "the rim sweep stage is a dark stage"
    assert means[0] < means[3], "linework only is darker than the full cloth"


def test_nan_values_render_finite(cp):
    bad = cn.default_values(**{n: float("nan") for n in cn.PARAM_NAMES[1:]})
    tex = cp.render(bad, dict(beat=float("nan")), float("nan"), float("nan"), (float("nan"),) * 3)
    img = cn.read_texture(tex)
    assert img[..., :3].max() > 0 and (img[..., :3].max(axis=2) > 250).mean() < 0.02


def test_heads_and_caps_change_the_picture(cp):
    a = shot(cp, coil_prog=0, coil_stage=3.0, coil_heads=0, coil_caps=0).astype(np.float32)
    b = shot(cp, coil_prog=0, coil_stage=3.0, coil_heads=1, coil_caps=1).astype(np.float32)
    assert np.abs(a - b).mean() > 0.2


def test_motion_is_smooth(cp):
    v = cn.default_values(coil_prog=0, coil_stage=3.0)
    a = cn.read_texture(cp.render(v, cn.make_clock(4.0), 4.0, 1 / 60)).astype(np.float32)
    b = cn.read_texture(cp.render(v, cn.make_clock(4.0 + 1 / 60), 4.0 + 1 / 60, 1 / 60)).astype(np.float32)
    d = np.abs(a - b)[..., :3].mean()
    assert 0.02 < d < 6.0, f"frame-to-frame change {d:.2f} (still, or jumping)"


def test_uv_debug_is_continuous_across_the_wrap(cp):
    """Debug output: r,g = cos / sin of the pattern phase (cells / wrap): continuous everywhere on a body, also where
    a closed coil's u wraps back to 0."""
    cp.debug = 1
    try:
        img = shot(cp, coil_prog=0, coil_stage=3.0).astype(np.float32) / 255
    finally:
        cp.debug = 0
    cov = img[..., 3] > 0.99
    c, s = img[..., 0] * 2 - 1, img[..., 1] * 2 - 1
    same = (np.abs(img[:, 1:, 2] - img[:, :-1, 2]) < 0.5 / 255) & cov[:, 1:] & cov[:, :-1]
    ang = np.arctan2(s, c)
    dang = np.abs(np.angle(np.exp(1j * (ang[:, 1:] - ang[:, :-1]))))
    assert np.percentile(dang[same], 99.9) < 0.35, "the u coordinate jumps on a body"


def test_kene4_layer_skin_gates_per_channel(cp, ctx):
    tex = cn.bake_channels(ctx, 256)
    try:
        v = cn.default_values(coil_prog=0, coil_stage=0.0, coil_skin=2)
        a = cn.render_frames(cp, v, layer=(tex, cn.LAYER_KENE4)).astype(np.float32)
        b = cn.render_frames(cp, dict(v, coil_stage=3.0), layer=(tex, cn.LAYER_KENE4)).astype(np.float32)
        p = cn.render_frames(cp, dict(v, coil_skin=0)).astype(np.float32)
        assert np.abs(a - p).mean() > 0.5, "the layer's channels are used, not the procedural ones"
        sat = lambda im: (im[..., :3].max(axis=2) - im[..., :3].min(axis=2)).mean()
        assert b[..., :3].mean() > a[..., :3].mean() * 1.2, "ground channel switched on in stage 3"
        assert sat(b) > sat(a)
    finally:
        tex.release()


def test_colour_layer_skin_is_used(cp, ctx):
    rng = np.random.default_rng(1)
    arr = (rng.random((64, 64, 4)) * 255).astype("u1")
    arr[..., 3] = 255
    arr[..., 1:3] = 0                                # a red image
    tex = ctx.texture((64, 64), 4, arr.tobytes())
    try:
        v = cn.default_values(coil_prog=0, coil_stage=3.0, coil_skin=3)
        img = cn.render_frames(cp, v, layer=(tex, cn.LAYER_COLOUR)).astype(np.float32)
        ref = shot(cp, coil_prog=0, coil_stage=3.0).astype(np.float32)
        assert np.abs(img - ref).mean() > 1.0
    finally:
        tex.release()


def test_missing_layer_falls_back_to_procedural(cp):
    a = shot(cp, coil_prog=0, coil_stage=3.0, coil_skin=2)
    b = shot(cp, coil_prog=0, coil_stage=3.0, coil_skin=0)
    assert np.array_equal(a, b)


def test_cost_budget(ctx):
    p = cn.CoilNestPass(ctx, (1280, 800))
    try:
        ms = cn.measure(p, cn.default_values(), frames=40)
        print(f"[coilnest cost] default N=10 at 1280x800: {ms:.2f} ms")
        assert ms < 3.0 * 1.5, "budget is 3 ms (x1.5 slack for a shared / throttling GPU)"
    finally:
        p.release()


def test_serpent_skin_bake_is_tileable_and_gated(cp):
    v = cn.default_values(l_coilnest=0.0, coil_prog=0, coil_stage=3.0)
    cp.reset(4.0)
    img = cn.read_texture(cp.bake(v, cn.make_clock(4.0), 4.0, 1 / 60)).astype(np.float32)
    assert cp.bakes == 1 and cp.draws == 0, "the bake works with the coil nest layer off"
    assert img[..., :3].mean() > 10
    left, right = img[:, 0, :3], img[:, -1, :3]
    assert np.abs(left - right).mean() < 12, "the strip tiles along the body"
    cp.reset(4.0)
    dark = cn.read_texture(cp.bake(dict(v, coil_stage=4.0), cn.make_clock(4.0), 4.0, 1 / 60)).astype(np.float32)
    assert dark[..., :3].mean() < img[..., :3].mean() * 0.6, "the gating programme reaches the serpents too"


def test_scene_shader_declares_the_hooks():
    import live_visuals as lv
    assert "p_l_coilnest" in lv.SCENE_FS and "texC" in lv.SCENE_FS
    assert "p_serpent_skin" in lv.SCENE_FS and "texSS" in lv.SCENE_FS
    assert "p_serpent_skin" in lv.SCENE_FS_PLAIN


def test_engine_off_is_never_drawn_and_on_draws(tmp_path):
    """Engine level: l_coilnest 0 (default) -> the pass exists but never draws; on -> it draws every frame; the
    serpents' skin bakes only while used."""
    import re
    import subprocess

    def run(*sets):
        r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--test", "12", "--out", str(tmp_path / "f"),
                            *sets], capture_output=True, text=True, timeout=300, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        m = re.search(r"\[l_coilnest\] draws=(\d+)", r.stdout)
        assert m, r.stdout[-800:]
        return int(m.group(1))
    assert run() == 0
    assert run("--set", "l_coilnest=1") >= 12
    assert run("--set", "serpent=1", "--set", "serpent_skin=1") == 0


def test_no_body_folds_onto_itself():
    """The tube radius stays below the radius of curvature of its centre line (x1.0 margin at the extremes, thick 1 +
    writhe 1 + dolly 1, every writhe phase): a thicker tube folds, its closest point is ambiguous and the skin tears.
    Tail needles (last 35 %) and stub caps (last 15 %, drawn as a medallion disc) are exempt."""
    worst = {}
    for th, w in ((1.0, 1.0), (0.5, 0.45)):
        for ph in np.linspace(0, 2 * math.pi, 9)[:-1]:
            for dol in (0.0, 1.6, 4.7):
                n = cn.Nest()
                n.ph_writhe, n.ph_dolly = ph, dol
                n.step(0.0, cn.default_values(coil_count=14, coil_thick=th, coil_writhe=w, coil_dolly=1.0), cn.make_clock(0))
                for i in range(14):
                    hi = n.last[i] + (1 if n.closed[i] else 0)
                    s = np.linspace(0, hi, 2500)
                    P, D, r, z, U, R = cn.hermite(n, i, s)
                    dd = np.gradient(D, s[1] - s[0], axis=0)
                    cr = np.abs(D[:, 0] * dd[:, 1] - D[:, 1] * dd[:, 0])
                    k = np.linalg.norm(D, axis=1) ** 3 / np.maximum(cr, 1e-12) / r
                    keep = s < hi * (0.65 if n.kind[i] == 2 else 0.85 if n.kind[i] == 3 else 2.0)
                    worst[i] = min(worst.get(i, 9e9), float(k[keep].min()))
    bad = {i: round(v, 2) for i, v in worst.items() if v < 1.0}
    assert not bad, f"bodies thicker than their bend: {bad}"
