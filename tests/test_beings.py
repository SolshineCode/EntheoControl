"""Beings layer (beings.BeingsPass): param contract, every form visible / on screen / never blown out, audio
reaction, non-finite guard, off = zero cost, engine hook.

One hidden GLFW window for the module (never visible on the projector).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import beings as bg  # noqa: E402
import params  # noqa: E402

OUT = (640, 400)
NAMES = ["l_beings", "being", "being_count", "being_size", "being_speed", "being_wiggle", "being_glow",
         "being_detail", "being_mix"]


@pytest.fixture(scope="module")
def ctx():
    c, _win = bg.hidden_context()
    yield c


@pytest.fixture()
def bp(ctx):
    p = bg.BeingsPass(ctx, OUT)
    yield p
    p.release()


def shot(bp, frames=1, t0=4.0, bands=(0.5, 0.3, 0.2), warm=0.0, **over):
    bp.herd.snap = True                                      # a still of these values, not a fade from the last ones
    return bg.render_frames(bp, bg.default_values(**over), t0=t0, frames=frames, bands=bands, warm=warm)


def frame(bp, vals, t, dt, bands=(0.5, 0.3, 0.2)):
    return bg.read_frame(bp, bp.render(vals, bg.make_clock(t), t, dt, bands, bg.GALLERY_PALETTE))


def lum(img):
    return img[..., :3].max(axis=2).astype(np.float32) / 255.0


# ---------------------------------------------------------------- contract


def test_params_appended_not_reordered():
    names = list(params.PARAMS)
    assert names[:5] == ["master", "flash", "strobe", "invert", "freeze"]
    assert names.index("art_glow") == 71, "the original params must keep their positions (MIDI CC order)"
    i = names.index("l_beings")
    assert i >= 92 and names[i:i + 9] == NAMES, "the nine beings params are one appended block"
    for n in NAMES:
        lo, hi, d, group, label = params.PARAMS[n]
        assert lo <= d <= hi and label
        assert group == ("layers" if n == "l_beings" else "beings")
    assert params.PARAMS["l_beings"][2] == 0.0, "the layer must default to off (the default look is unchanged)"


def test_being_options_contract():
    lo, hi = params.PARAMS["being"][:2]
    opts = params.OPTIONS["being"]
    assert lo == 0 and len(opts) == int(hi) + 1 == len(bg.FORMS) and 12 <= len(opts) <= 24
    assert opts == bg.FORM_LABELS and len(set(opts)) == len(opts)
    assert len(set(bg.FORM_KEYS)) == len(bg.FORM_KEYS)


def test_panel_meta_carries_the_option_names():
    control = pytest.importorskip("control")
    assert control.OPTIONS.get("being") == params.OPTIONS["being"], "control.py merges params.OPTIONS for the panel"
    assert "being" in control.DISCRETE                       # stepped: no slew through the forms in between


def test_api_contract(bp, ctx):
    import moderngl
    fbo = ctx.simple_framebuffer((32, 32))
    fbo.use()
    ctx.viewport = (0, 0, 32, 32)
    tex = bp.render(bg.default_values(), bg.make_clock(1.0), 1.0, 1 / 60, (0, 0, 0), bg.GALLERY_PALETTE)
    assert isinstance(tex, moderngl.Texture) and tex.components == 4 and tex.size == OUT
    assert tex.filter == (moderngl.LINEAR, moderngl.LINEAR)
    assert ctx.fbo.glo == fbo.glo and ctx.viewport == (0, 0, 32, 32), "the caller's target and viewport come back"
    assert bp.render(bg.default_values(), None, 2.0) is tex   # clock / dt / bands / palette are optional
    assert callable(bp.release) and callable(bp.close)
    ctx.screen.use()
    fbo.release()


# ---------------------------------------------------------------- every form


@pytest.mark.parametrize("form", range(len(bg.FORMS)), ids=bg.FORM_KEYS)
def test_form_is_visible_dark_floored_and_never_blown_out(bp, form):
    img = shot(bp, frames=2, warm=10.0 + form, being=form, being_size=0.6)
    l, a = lum(img), img[..., 3] / 255.0
    lit = (l > 0.25).mean()
    assert lit > 0.003, f"{bg.FORM_KEYS[form]}: nothing to see ({lit:.4f})"
    assert (l < 0.04).mean() > 0.55, f"{bg.FORM_KEYS[form]}: black must stay black around the beings"
    white = (img[..., :3].min(axis=2) > 230).mean()
    assert l.mean() < 0.22 and white < 0.01 and (l > 0.98).mean() < 0.04, f"{bg.FORM_KEYS[form]}: blown out"
    if bg.FORM_KEYS[form] != "fireflies":                    # fireflies are light, hardly any body
        assert (a > 0.5).mean() > 0.003, "a silhouette for the compositor to darken behind"
    sat = img[..., :3].max(axis=2).astype(int) - img[..., :3].min(axis=2)
    assert (sat[l > 0.25] > 60).mean() > 0.25, "palette colour, not white line-art"


def test_forms_are_distinct(ctx):
    small = []
    for f in range(len(bg.FORMS)):
        p = bg.BeingsPass(ctx, (320, 200))
        try:
            small.append(lum(bg.render_frames(p, bg.default_values(being=f, being_count=1, being_speed=0.0, being_size=1.0))))
        finally:
            p.release()
    for i in range(len(small)):
        for j in range(i + 1, len(small)):
            assert np.abs(small[i] - small[j]).mean() > 0.004, f"{bg.FORM_KEYS[i]} and {bg.FORM_KEYS[j]} look the same"


@pytest.mark.parametrize("form", range(len(bg.FORMS)), ids=bg.FORM_KEYS)
def test_form_stays_mostly_on_screen(bp, form):
    """Seen from further out (view_zoom), most of the light is inside the real frame, at any moment of a long run."""
    vals = bg.default_values(being=form, being_count=4, being_size=0.5, being_speed=1.0)
    bp.view_zoom = 1.6
    w, h = OUT
    x0, x1 = int(w * (0.5 - 0.5 / 1.6)), int(w * (0.5 + 0.5 / 1.6))
    y0, y1 = int(h * (0.5 - 0.5 / 1.6)), int(h * (0.5 + 0.5 / 1.6))
    fracs = []
    for k in range(12):
        l = lum(bg.render_frames(bp, vals, t0=4.0 + k, warm=17.0))
        total = l.sum()
        assert total > 1.0
        fracs.append(l[y0:y1, x0:x1].sum() / total)
    assert np.mean(fracs) > 0.80 and min(fracs) > 0.6, f"{bg.FORM_KEYS[form]}: {np.round(fracs, 2)}"


def test_centres_never_leave_the_frame():
    herd = bg.Herd(1.6)
    for form in range(len(bg.FORMS)):
        for size in (0.0, 0.5, 1.0):
            vals = bg.default_values(being=form, being_count=8, being_size=size, being_speed=1.0)
            for _ in range(300):
                ua, ub, uc, n = herd.step(0.1, vals, (1.0, 0.0, 0.0))
                assert n == 8 and np.isfinite(ua).all() and np.isfinite(ub).all() and np.isfinite(uc).all()
                assert (np.abs(ua[:, 0]) <= 0.8).all() and (np.abs(ua[:, 1]) <= 0.5).all()
                assert (ub[:, 0] >= 0).all() and (ub[:, 0] <= bg.PHASE_WRAP).all()


# ---------------------------------------------------------------- life, audio, controls


def test_beings_move_and_hold_still_when_time_stops(bp):
    vals = bg.default_values(being=7)
    a = lum(bg.render_frames(bp, vals, t0=4.0, warm=3.0))
    b = lum(frame(bp, vals, 4.0, 0.0))
    assert np.abs(a - b).mean() < 1e-4, "dt 0 (freeze): the same picture"
    c = lum(bg.render_frames(bp, vals, t0=4.5, frames=30))
    assert np.abs(a - c).mean() > 0.003, "half a second later the fish have swum on"


def test_bass_pulses_and_highs_sparkle(bp):
    vals = bg.default_values(being=5, being_speed=0.0)
    quiet = lum(frame(bp, vals, 4.0, 0.0, (0.0, 0.0, 0.0)))
    bass = lum(frame(bp, vals, 4.0, 0.0, (1.0, 0.0, 0.0)))
    high = lum(frame(bp, vals, 4.0, 0.0, (0.0, 0.0, 1.0)))
    assert bass.sum() > quiet.sum() * 1.08, "bass: bigger and brighter"
    assert high.sum() > quiet.sum() * 1.005 and np.abs(high - quiet).max() > 0.2, "highs: sparkle"


def test_amount_count_size_and_palette(bp):
    base = dict(being=0, being_speed=0.0)
    full = lum(shot(bp, **base)).sum()
    assert lum(shot(bp, l_beings=0.5, **base)).sum() < full * 0.7
    assert lum(shot(bp, being_count=8, **base)).sum() > full * 1.5
    assert lum(shot(bp, being_size=1.0, **base)).sum() > lum(shot(bp, being_size=0.0, **base)).sum() * 1.5
    red = bg.render_frames(bp, bg.default_values(**base), palette=[(1.0, 0.0, 0.0)])
    assert red[..., 0].astype(int).sum() > 3 * red[..., 2].astype(int).sum(), "colours come from the show palette"


def test_variety_mixes_forms():
    assert {bg.form_of(i, 5, 0.0) for i in range(bg.N_MAX)} == {5}
    mixed = [bg.form_of(i, 5, 1.0) for i in range(bg.N_MAX)]
    assert mixed[0] == 5 and len(set(mixed)) >= 6 and all(0 <= f < len(bg.FORMS) for f in mixed)
    half = [bg.form_of(i, 5, 0.5) for i in range(bg.N_MAX)]
    assert 2 <= sum(f != 5 for f in half) <= 6


# ---------------------------------------------------------------- safety


def test_non_finite_inputs_never_reach_the_picture(bp):
    nan, inf = float("nan"), float("inf")
    vals = bg.default_values(being=nan, being_count=inf, being_size=nan, being_speed=-inf, being_wiggle=inf,
                             being_glow=nan, being_detail=inf, being_mix=nan)
    for t, dt, bands, pal in ((nan, nan, (nan, inf, -inf), [(nan, 0, 0)] * 5), (inf, inf, None, "nope"),
                              (4.0, -5.0, (), None), (4.0, 1e9, (2.0, 2.0, 2.0), [(9.0, -3.0, 0.5)])):
        tex = bp.render(vals, None, t, dt, bands, pal)
        img = bg.read_texture(tex)
        assert img.shape == (OUT[1], OUT[0], 4)
    assert math.isfinite(bp.herd.s) and math.isfinite(bp.herd.a) and np.isfinite(bp.herd.gait).all()
    good = lum(shot(bp, being=0))
    assert 0.003 < (good > 0.25).mean() < 0.3, "and the pass still draws properly afterwards"
    assert lum(bg.read_texture(bp.render({}, None, 5.0))).sum() == 0.0, "no values at all = layer off"


def test_extreme_values_stay_bounded(bp):
    for form in range(len(bg.FORMS)):
        img = shot(bp, bands=(1.0, 1.0, 1.0), being=form, being_count=8, being_size=1.0, being_glow=1.0, being_detail=1.0,
                   being_wiggle=1.0, being_speed=1.0, warm=5.0)
        l = lum(img)
        white = (img[..., :3].min(axis=2) > 230).mean()
        assert l.mean() < 0.62 and (l < 0.04).mean() > 0.10 and white < 0.15, f"{bg.FORM_KEYS[form]} floods the frame at maximum"


def test_off_costs_nothing(bp):
    off = bg.default_values(l_beings=0.0)
    for i in range(5):
        tex = bp.render(off, None, 1.0 + i, 1 / 60, (1, 1, 1), bg.GALLERY_PALETTE)
    assert bp.draws == 0 and bp.herd.s == 0.0 and bp.herd.a == 0.0, "off: no shader pass, no motion"
    assert bg.read_texture(tex).sum() == 0
    shot(bp, frames=2)
    assert bp.draws == 2
    tex = bp.render(off, None, 9.0, 1 / 60)
    assert bp.draws == 2 and bg.read_texture(tex).sum() == 0, "switched off again: cleared, not left on show"


def test_engine_hook_contract():
    lv = pytest.importorskip("live_visuals")
    assert lv.BeingsPass is bg.BeingsPass
    assert all(n in lv.CPU_ONLY for n in NAMES[1:]) and "l_beings" not in lv.CPU_ONLY
    assert "p_l_beings" in lv.SCENE_FS and "texG" in lv.SCENE_FS


def test_engine_composites_beings(tmp_path):
    """Default look untouched with the layer at 0; with it up the beings are in the frame, over the pattern."""
    from PIL import Image
    spec = tmp_path / "jobs.json"
    K = dict(l_beads=0.0, l_lines=0.0, l_ai=0.0, l_echo=0.0, swim=0.0, rotate=0.0, flow=0.0, speed=0.0, chase=0.0)
    jobs = [dict(name="plain", sets=K, frames=20), dict(name="zero", sets=dict(K, being=8, being_count=8), frames=20),
            dict(name="beings", sets=dict(K, l_beings=1.0, being=8), frames=20),
            dict(name="alone", sets=dict(K, l_kene=0.0, l_beings=1.0, being=8), frames=20)]
    spec.write_text(json.dumps(jobs))
    r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--batch", str(spec), "--out", str(tmp_path),
                        "--test-audio", "silent"], capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "[l_beings] disabled" not in r.stdout and "render failed" not in r.stdout
    done = {d["name"]: d for d in json.loads((tmp_path / "batch.json").read_text())}
    img = {n: np.asarray(Image.open(tmp_path / d["file"]).convert("RGB")).astype(np.float32) / 255.0 for n, d in done.items()}
    assert all(d.get("nonfinite", 0) == 0 for d in done.values())
    assert np.abs(img["plain"] - img["zero"]).mean() < 0.002, "l_beings 0: the other being_* params change nothing"
    assert np.abs(img["plain"] - img["beings"]).mean() > 0.004, "the beings are in the picture"
    alone = img["alone"].max(axis=2)
    assert 0.003 < (alone > 0.25).mean() < 0.3 and (alone < 0.04).mean() > 0.5, "beings alone: light on black"


# ================================================================ review round (Opus, 2026-11-25): regressions


def _herd_vals(**over):
    return bg.default_values(**over)


def test_phase_multipliers_are_wrap_safe():
    """Phases reach the shader wrapped at PHASE_WRAP = TAU*64: sin(ph*m) needs m*64 whole, fract(ph*TURN*k) too."""
    import re
    assert abs(bg.PHASE_WRAP - bg.TAU * 64.0) < 1e-9
    turns = [float(m) for m in re.findall(r"\bph\*TURN\*([0-9.]+)", bg.FS)]
    plain = [float(m) for m in re.findall(r"\bph\*([0-9][0-9.]*)", bg.FS)]
    assert len(plain) > 40 and len(turns) >= 2
    for m in plain:
        assert abs(m * 8 - round(m * 8)) < 1e-9, f"ph*{m}: not a multiple of 1/8, jumps at the phase wrap"
    for k in turns:
        assert abs(k * 64 - round(k * 64)) < 1e-9, f"ph*TURN*{k}: the cycle does not close at the phase wrap"
    assert not re.findall(r"fract\([^;]*\bph\*[0-9]", bg.FS), "fract cycles must count whole turns (ph*TURN*k)"
    assert "uT*0." not in bg.FS and "% 1024" not in Path(bg.__file__).read_text(encoding="utf-8")
    assert "pow((cyc" not in bg.FS, "pow(negative, 2.0) is NaN on some drivers: square by multiplying"


@pytest.mark.parametrize("form", range(len(bg.FORMS)), ids=bg.FORM_KEYS)
def test_no_jump_when_the_animation_phase_wraps(ctx, form):
    """Two frames that straddle a multiple of PHASE_WRAP differ no more than two ordinary neighbouring frames."""
    m, off = 0.85 + 0.3 * bg.HERD_H[0, 0], 40.0 * bg.HERD_H[0, 1]
    rate, dt = 0.45 + 1.5 * 0.4, 1 / 60
    vals = bg.default_values(being=form, being_count=1, being_speed=0.4, being_size=0.8)
    diffs = []
    for k in (3.0, 3.0 - 30.0 / bg.PHASE_WRAP, 3.0 - 71.0 / bg.PHASE_WRAP):   # the wrap, then two ordinary moments
        bp = bg.BeingsPass(ctx, OUT)
        try:
            target = bg.PHASE_WRAP * k - off - 0.5 * dt * rate * m
            bp.herd.a, bp.herd.s = target / m - dt * rate, 50.0
            a = lum(frame(bp, vals, 100.0, dt, (0, 0, 0)))
            ph_a = bp.herd.a * m + off
            b = lum(frame(bp, vals, 100.0, dt, (0, 0, 0)))                    # same show time: only the phase moves
            ph_b = bp.herd.a * m + off
            if k == 3.0:
                assert ph_a // bg.PHASE_WRAP != ph_b // bg.PHASE_WRAP, "the pair must straddle the wrap"
            diffs.append(float(np.abs(a - b).mean()))
        finally:
            bp.release()
    assert diffs[0] <= 1.6 * max(diffs[1], diffs[2]) + 2e-4, f"{bg.FORM_KEYS[form]}: jump at the wrap {np.round(diffs, 5)}"


@pytest.mark.parametrize("wrap_t", [bg.PHASE_WRAP, 64.0 / 5.0 / 0.12, 64.0 / 9.0 / 0.30],
                         ids=["twinkle clock", "ember scroll", "breath scroll"])
def test_no_jump_when_the_sparkle_clock_wraps(bp, wrap_t):
    """Sparkle / embers run on show time, wrapped in double precision where their lattice and rates repeat."""
    dt = 1 / 60
    for form in (8, 15, 0):                                   # firebird embers, whale breath, plain sparkle
        vals = bg.default_values(being=form, being_count=2, being_speed=0.0, being_size=0.9)
        bp.herd.snap = True

        def pair(t):
            a = lum(frame(bp, vals, t - dt / 2, 0.0, (0.0, 0.0, 1.0)))
            return float(np.abs(a - lum(frame(bp, vals, t + dt / 2, 0.0, (0.0, 0.0, 1.0)))).mean())
        across = pair(wrap_t * 3.0)
        normal = max(pair(wrap_t * 3.0 - 0.71), pair(wrap_t * 3.0 + 1.37))
        assert across <= 1.6 * normal + 2e-4, f"{bg.FORM_KEYS[form]}: sparkle jumps at t={wrap_t * 3:.1f} ({across:.5f} vs {normal:.5f})"


# ---------------------------------------------------------------- cost at maximum


MAXED = dict(being_count=8, being_size=1.0, being_glow=1.0, being_detail=1.0, being_wiggle=1.0)


@pytest.mark.parametrize("form", range(len(bg.FORMS)), ids=bg.FORM_KEYS)
def test_cost_is_capped_at_maximum(bp, form):
    """8 beings at full size: the area budget shrinks them together and the pass drops to LOW_RES, so the pixel work
    (cull-box area on show x resolution) stays near that of a default herd. Structural on purpose: milliseconds on a
    GPU shared with the running show are not a stable thing to assert."""
    vals = bg.default_values(being=form, **MAXED)
    shot(bp, frames=2, warm=2.0, bands=(1.0, 1.0, 1.0), **{k: v for k, v in vals.items() if k.startswith("being")})
    anchored = bg.FORMS[form][2] == bg.ANCHOR
    if not anchored:
        assert bp.herd.load <= bg.AREA_BUDGET * 1.001
        raw = sum(4 * bg.FORM_EXT[form][0] * bg.FORM_EXT[form][1] * bg.Herd.raw_scale(i, form, 1.0) ** 2 for i in range(8))
        if raw > 1.15 * bg.AREA_BUDGET:                       # (little forms, the kittens, fit the budget as they are)
            assert bp.herd.cap < 0.95
            assert raw > 2.0 * bp.herd.load * bp.res ** 2, "at least halved against drawing it as asked"
    assert bp.res == (bg.MIN_RES if anchored else bg.LOW_RES), "heavy herd: drawn at a reduced resolution"
    work = bp.herd.load * bp.res ** 2
    assert work <= (3.0 if anchored else bg.AREA_BUDGET * bg.LOW_RES ** 2 + 0.01), f"{bg.FORM_KEYS[form]}: pixel work {work:.2f} content-units at maximum"
    shot(bp, frames=2, warm=2.0, being=form)
    assert bp.res == 1.0 and bp.herd.cap == 1.0, "a default herd is drawn at full resolution, at the size asked for"


def test_size_still_matters_under_the_cap(bp):
    small = lum(shot(bp, being=7, being_count=8, being_size=0.0, being_speed=0.0)).sum()
    big = lum(shot(bp, being=7, being_count=8, being_size=1.0, being_speed=0.0)).sum()
    assert big > small * 1.3
    three = [bg.Herd.raw_scale(i, 7, 1.0) for i in range(3)]
    h = bg.Herd(1.6)
    ua, _, _, _ = h.step(0.0, bg.default_values(being=7, being_count=3, being_size=1.0))
    assert np.allclose(ua[:3, 3], three, rtol=1e-5) and h.cap == 1.0, "three big beings are not shrunk at all"


def test_low_res_switch_has_hysteresis(bp):
    seen = []
    for size in list(np.linspace(0.3, 1.0, 15)) + list(np.linspace(1.0, 0.3, 15)):
        shot(bp, being=0, being_count=8, being_size=float(size), being_speed=0.0)
        seen.append((round(bp.herd.load / bg.AREA_BUDGET, 3), bp.res))
    flips = sum(1 for a, b in zip(seen, seen[1:]) if a[1] != b[1])
    assert flips == 2, f"one switch down, one back up: {seen}"
    assert all(r == bg.LOW_RES for l, r in seen if l > bg.LOW_RES_ON) and all(r == 1.0 for l, r in seen if l < bg.LOW_RES_OFF)
    assert bg.LOW_RES_OFF < bg.LOW_RES_ON


@pytest.mark.parametrize("form", range(len(bg.FORMS)), ids=bg.FORM_KEYS)
def test_cull_box_never_clips_the_picture(ctx, form):
    """The per-form cull box is an optimisation only: same picture with it and without it."""
    vals = bg.default_values(being=form, being_count=3, being_size=1.0, being_glow=1.0, being_wiggle=1.0)
    out = []
    for cull in (1.0, 1e3):
        p = bg.BeingsPass(ctx, OUT)
        p.cull = cull
        try:
            out.append([lum(bg.render_frames(p, vals, t0=7.0 + k, warm=3.0 if k == 0 else 1.3, bands=(1, 1, 1))) for k in range(4)])
        finally:
            p.release()
    for a, b in zip(*out):
        d = np.abs(a - b)
        assert d.max() < 0.08 and d.mean() < 2e-4, f"{bg.FORM_KEYS[form]}: the cull box cuts into the picture ({d.max():.3f})"


# ---------------------------------------------------------------- mid-show switches


def _run_herd(h, vals, seconds, dt=1 / 60):
    rows = []
    for _ in range(int(round(seconds / dt))):
        ua, ub, uc, _n = h.step(dt, vals)
        rows.append((ua.copy(), ub.copy(), uc.copy()))
    return rows


def _assert_smooth(rows, dt=1 / 60):
    max_step = 1.5 * dt / bg.FADE_S + 1e-4                    # steepest slope of the smoothstep envelope
    for (a0, b0, c0), (a1, b1, c1) in zip(rows, rows[1:]):
        for i in range(bg.N_MAX):
            f0, f1 = c0[i, 3], c1[i, 3]
            if bg.FORMS[int(b1[i, 1])][2] != bg.ANCHOR:       # (the vine has its own grow / dissolve cycle on top)
                assert abs(f1 - f0) <= max_step, f"slot {i}: visibility jumps {f0:.3f} -> {f1:.3f}"
            if f0 > 0.0 and f1 > 0.0:
                assert b0[i, 1] == b1[i, 1], f"slot {i} changed form while visible"
                assert np.hypot(*(a1[i, :2] - a0[i, :2])) < 0.03, f"slot {i} teleported while visible"
                assert abs(a1[i, 3] - a0[i, 3]) < 0.08 * a0[i, 3] + 1e-6, f"slot {i} changed size abruptly"


def test_form_change_fades_out_swaps_and_fades_in():
    h = bg.Herd(1.6)
    rows = _run_herd(h, _herd_vals(being=0), 0.5)
    assert all(r[2][:3, 3].min() == 1.0 and r[2][3:, 3].max() == 0.0 for r in rows), "steady: three fully there"
    sw = _run_herd(h, _herd_vals(being=7), 1.2)
    _assert_smooth(rows[-1:] + sw)
    forms0 = [int(r[1][0, 1]) for r in sw]
    fade0 = [float(r[2][0, 3]) for r in sw]
    k = forms0.index(7)
    assert all(f == 0 for f in forms0[:k]) and all(f == 7 for f in forms0[k:]), "one swap"
    assert abs(k / 60 - bg.FADE_S) < 0.06 and fade0[k - 1] < 0.02, "swapped at the bottom of the fade"
    assert min(fade0) == 0.0 or min(fade0) < 0.02
    back = next(j for j, r in enumerate(sw) if j >= k and r[2][0, 3] == 1.0)
    assert fade0[-1] == 1.0 and abs(back / 60 - 2 * bg.FADE_S) < 0.08


def test_count_and_variety_changes_fade():
    h = bg.Herd(1.6)
    big = dict(being=5, being_size=1.0)                       # big: the area budget rescales the herd as slots fade
    rows = _run_herd(h, _herd_vals(being_count=3, **big), 0.3)
    rows += _run_herd(h, _herd_vals(being_count=8, **big), 1.0)
    assert rows[-1][2][:, 3].min() == 1.0, "eight there after the fade in"
    rows += _run_herd(h, _herd_vals(being_count=8, being_mix=1.0, **big), 1.2)
    assert len({int(f) for f in rows[-1][1][:, 1]}) >= 6
    rows += _run_herd(h, _herd_vals(being_count=2, being_mix=1.0, **big), 1.0)
    assert rows[-1][2][2:, 3].max() == 0.0 and rows[-1][2][:2, 3].min() == 1.0
    _assert_smooth(rows)


def test_modulated_count_breathes_instead_of_strobing():
    """being_count is a fader (not stepped): an LFO on it must not flash beings on and off."""
    h, rows = bg.Herd(1.6), []
    for k in range(240):                                      # 4 s of a fast square wave between 2 and 8
        rows += _run_herd(h, _herd_vals(being=4, being_count=8 if (k // 6) % 2 else 2, being_size=1.0), 1 / 60)
    _assert_smooth(rows)
    vis = np.array([r[2][5, 3] for r in rows[60:]])
    assert vis.max() < 0.3, "a slot toggled every 0.1 s never gets far: no strobe"


def test_freeze_holds_a_fade_and_layer_off_resets_it(bp):
    h = bp.herd
    shot(bp, being=0)
    frame(bp, bg.default_values(being=9), 5.0, 1 / 60)
    mid = h.env.copy()
    assert 0.9 < mid[0] < 1.0
    for _ in range(30):
        frame(bp, bg.default_values(being=9), 5.0, 0.0)
    assert np.array_equal(h.env, mid) and h.form[0] == 0, "dt 0: the fade waits"
    bp.render(bg.default_values(l_beings=0.0, being=9), None, 6.0, 1 / 60)
    frame(bp, bg.default_values(being=12, being_count=6), 7.0, 1 / 60)
    assert h.form[0] == 12 and list(h.env) == [1.0] * 6 + [0.0] * 2, "layer brought back: the wanted herd at once"


def test_switching_form_does_not_pop_the_picture(bp):
    vals = bg.default_values(being=1, being_size=0.8)
    shot(bp, frames=2, warm=2.0, being=1, being_size=0.8)
    a = lum(frame(bp, vals, 6.0, 1 / 60, (0, 0, 0)))
    steady = np.abs(a - lum(frame(bp, vals, 6.0, 1 / 60, (0, 0, 0)))).mean()
    switched = np.abs(a - lum(frame(bp, dict(vals, being=10), 6.0, 1 / 60, (0, 0, 0)))).mean()
    assert switched < 3.0 * steady + 0.002, f"first frame after the switch: {switched:.4f} vs steady {steady:.4f}"


# ---------------------------------------------------------------- GL state, texture units, engine


def test_blend_state_is_read_without_other_modules_and_restored(bp, ctx):
    import moderngl
    src = Path(bg.__file__).read_text(encoding="utf-8")
    assert "import math_layers" not in src, "the blend query must not depend on another layer's module"
    assert bp._blend_on is not None and bp.blend_assumed_off is False
    vals = bg.default_values(being=6, being_speed=0.0)
    plain = frame(bp, vals, 4.0, 0.0)
    ctx.enable(moderngl.BLEND)
    try:
        assert bp._blend_on() is True
        blended = frame(bp, vals, 4.0, 0.0)
        assert bp._blend_on() is True, "the caller's blending is switched back on"
    finally:
        ctx.disable(moderngl.BLEND)
    assert np.array_equal(plain, blended), "and it never leaks into the pass's own picture"
    assert bp._blend_on() is False
    bp._blend_on = None                                       # raw GL not available: draw anyway, blending stays off
    assert np.array_equal(frame(bp, vals, 4.0, 0.0), plain)


def test_texture_unit_is_not_shared():
    import re
    lv = pytest.importorskip("live_visuals")
    src = Path(lv.__file__).read_text(encoding="utf-8")
    reg = re.search(r'param="l_beings".*?uniform="texG", unit=(\d+)', src)
    uni = re.search(r"uni = dict\(([^\n]*)", src).group(1)
    samplers = {k: int(v) for k, v in re.findall(r"\b(tex[A-Z]+|prevFrame)=(\d+)", uni)}
    unit = int(reg.group(1))
    assert samplers["texG"] == unit, "the sampler uniform and the bound unit must agree"
    others = {k: v for k, v in samplers.items() if k != "texG"}
    assert unit not in others.values(), f"unit {unit} is also {[k for k, v in others.items() if v == unit]}"
    assert unit != 8, "unit 8 is the flat art image (texS)"
    art = pytest.importorskip("art_layer")
    assert unit not in range(art.UNIT0, art.UNIT0 + 7), "ArtPass leaves units UNIT0..UNIT0+6 bound to its own textures"
    fx = Path(lv.__file__).with_name("fx_stack.py").read_text(encoding="utf-8")
    assert f"unit={unit}" not in fx, "fx_stack's preview read uses its own unit"


def test_engine_never_draws_the_pass_while_the_layer_is_off(tmp_path):
    """Engine level: with l_beings at 0 (the default) the pass object exists but has not drawn once."""
    def run(*sets):
        r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--test", "12", "--out", str(tmp_path / "f"),
                            *sets], capture_output=True, text=True, timeout=300, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        import re
        m = re.search(r"\[l_beings\] draws=(\d+)", r.stdout)
        assert m, r.stdout[-800:]
        return int(m.group(1))
    assert run("--set", "being=9", "--set", "being_count=8") == 0
    assert run("--set", "l_beings=0.6") >= 12


# ================================================================ kittens + colour modes (2026-11-25)

COLOUR_NAMES = ["being_color", "being_hue", "being_hue2", "being_sat", "being_bright",
                "serpent_color", "serpent_hue", "serpent_hue2", "serpent_sat", "serpent_bright"]
GOLDEN = ROOT / "tests" / "golden"


def _hues(img, floor=0.35):
    """Hue (0..1) of the clearly coloured, clearly lit pixels."""
    rgb = img[..., :3].astype(np.float32) / 255.0
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sel = (mx > floor) & ((mx - mn) > 0.45 * mx)
    r, g, b = (rgb[..., k][sel] for k in range(3))
    d = np.maximum(mx[sel] - mn[sel], 1e-6)
    h = np.where(mx[sel] == r, ((g - b) / d) % 6.0, np.where(mx[sel] == g, (b - r) / d + 2.0, (r - g) / d + 4.0)) / 6.0
    return h % 1.0


def _near(h, target, width=0.07):
    return (np.abs((h - target + 0.5) % 1.0 - 0.5) < width).mean()


def test_kitten_is_form_16_and_colour_params_are_appended():
    names = list(params.PARAMS)
    assert bg.FORM_KEYS[16] == "kitten" and bg.FORM_KEYS[:16] == [
        "serpent", "dragon", "jaguar", "hummingbird", "butterflies", "jellyfish", "manta", "koi", "phoenix", "eyes",
        "angel", "jester", "octopus", "vine", "fireflies", "whale"], "forms are appended, never reordered"
    assert params.OPTIONS["being"][16] == "Kittens" and params.PARAMS["being"][1] >= 16.0
    i = names.index("being_color")
    assert names[i:i + 10] == COLOUR_NAMES and i > names.index("being_mix"), "one appended block, after everything older"
    for n in COLOUR_NAMES:
        lo, hi, d, group, label = params.PARAMS[n]
        assert lo <= d <= hi and label and group == ("beings" if n.startswith("being") else "motion")
    for n in ("being_color", "serpent_color"):
        assert params.PARAMS[n][:3] == (0.0, 5.0, 0.0) and len(params.OPTIONS[n]) == 6 == len(bg.COLOR_MODES)
        assert params.OPTIONS[n] == bg.COLOR_MODES
    for n in ("being_sat", "being_bright", "serpent_sat", "serpent_bright"):
        assert params.PARAMS[n][2] == 1.0, "the defaults are the look as it was"
    assert len(bg.FORM_EXT) == len(bg.FORMS) >= 17


@pytest.mark.parametrize("key", ["serpent", "jaguar", "jellyfish", "phoenix", "vine", "whale"])
def test_default_look_is_pixel_identical_to_before_the_colour_modes(ctx, key):
    """Golden frames were rendered before the kitten / colour-mode change (tests/golden). 2/255 covers the shader
    being recompiled with more code around it; any real change of look is far above that."""
    from PIL import Image
    gold = np.asarray(Image.open(GOLDEN / f"beings_default_{key}.png")).astype(int)
    p = bg.BeingsPass(ctx, (320, 200))
    try:
        img = bg.render_frames(p, bg.default_values(being=bg.FORM_KEYS.index(key)), t0=4.0, frames=2, warm=3.0).astype(int)
    finally:
        p.release()
    d = np.abs(img - gold)
    assert d.max() <= 2 and d.mean() < 0.02, f"{key}: default look changed (max {d.max()}, mean {d.mean():.4f})"


def test_explicit_colour_defaults_change_nothing(bp):
    base = dict(being=8, being_speed=0.0)
    shot(bp, **base)
    a = frame(bp, bg.default_values(**base), 4.0, 0.0)
    b = frame(bp, bg.default_values(being_color=0, being_hue=0.11, being_hue2=0.77, being_sat=1.0, being_bright=1.0, **base), 4.0, 0.0)
    assert np.array_equal(a, b), "mode 0: the hue faders are ignored, depth 1 and brightness 1 are the identity"


def test_colour_modes(bp):
    base = dict(being=0, being_count=3, being_size=0.9, being_speed=0.0, being_glow=0.3)
    show = shot(bp, **base)
    pics = [shot(bp, being_color=m, being_hue=0.33, being_hue2=0.83, **base) for m in range(6)]
    for m in range(1, 6):
        assert np.abs(pics[m].astype(int) - show.astype(int)).mean() > 0.3, f"{bg.COLOR_MODES[m]} looks like the show palette"
    single, two, rainbow, each = (_hues(pics[m]) for m in (1, 2, 3, 4))
    assert _near(single, 0.33, 0.08) > 0.9, "single hue: green and nothing else"
    assert _near(two, 0.33, 0.14) > 0.12 and _near(two, 0.83, 0.14) > 0.12, "two-tone: both ends are there"
    assert _near(two, 0.33, 0.14) + _near(two, 0.83, 0.14) + _near(two, 0.58, 0.14) > 0.85, "and the blend between them"
    assert (np.histogram(rainbow, bins=6, range=(0, 1))[0] > 0.04 * len(rainbow)).sum() >= 5, "rainbow: all round the wheel"
    assert _near(single, 0.33, 0.08) > _near(each, 0.33, 0.08) + 0.3, "each its own hue: not one colour"
    assert (np.histogram(each, bins=12, range=(0, 1))[0] > 0.08 * len(each)).sum() >= 2
    moved = shot(bp, being_color=1, being_hue=0.70, **base)
    assert _near(_hues(moved), 0.70, 0.08) > 0.9, "the hue fader moves it"


def test_two_tone_runs_head_to_tail(ctx):
    p = bg.BeingsPass(ctx, OUT)
    try:
        ua, ub, uc = (np.zeros((bg.N_MAX, 4), dtype="f4") for _ in range(3))
        ua[0], ub[0], uc[0] = (0.0, 0.0, 0.0, 0.42), (1.0, 0, 0.3, 1.0), (0.0, 0.0, 1.0, 1.0)   # one serpent, head to the right
        p.herd.step = lambda dt, values, bands, clock=None: (ua, ub, uc, 1)
        img = frame(p, bg.default_values(being_color=2, being_hue=0.0, being_hue2=0.6, being_glow=0.2), 4.0, 0.0)
    finally:
        p.release()
    w = OUT[0]
    head, tail = _hues(img[:, int(w * 0.68):]), _hues(img[:, :int(w * 0.32)])
    assert _near(head, 0.0, 0.10) > 0.7 and _near(tail, 0.6, 0.12) > 0.7, "hue 1 at the head, hue 2 at the tail"


def test_light_chase_mode_follows_the_chase_phase(bp):
    base = dict(being=7, being_count=2, being_size=1.0, being_speed=0.0, being_color=5, chase_width=0.3)
    shot(bp, **base)

    def still(**over):
        return lum(frame(bp, bg.default_values(**dict(base, **over)), 4.0, 0.0))
    show, a, b = still(being_color=0), still(chase_phase=0.20), still(chase_phase=0.70)
    assert a.sum() < 0.8 * show.sum(), "only the window is lit"
    assert np.abs(a - b).mean() > 0.2 * a.mean(), "and it travels with the chase"
    again = still(chase_phase=1.20)
    assert np.abs(a - again).mean() < 1e-3, "phase is cyclic"
    clocked = lum(bg.read_frame(bp, bp.render(bg.default_values(**base), dict(bar_phase=0.2, beat=0.8), 4.0, 0.0, (0.5, 0.3, 0.2),
                                              bg.GALLERY_PALETTE)))
    assert np.abs(a - clocked).mean() < 1e-3, "without the engine's phase it rides the bar (chase_rate 1)"


def test_colour_depth_and_brightness(bp):
    base = dict(being=9, being_speed=0.0)
    full = shot(bp, **base)
    grey = shot(bp, being_sat=0.0, **base)[..., :3].astype(int)
    assert (grey.max(axis=2) - grey.min(axis=2)).max() <= 12, "colour depth 0: white light only"
    half = lum(shot(bp, being_bright=0.5, **base)).sum()
    assert 0.3 * lum(full).sum() < half < 0.8 * lum(full).sum()
    assert lum(shot(bp, being_bright=0.0, **base)).sum() == 0.0
    hot = shot(bp, being_bright=2.0, bands=(1, 1, 1), **base)
    assert (hot[..., :3].min(axis=2) > 230).mean() < 0.15, "brightness 2 glows harder, it does not flood white"


def test_kittens_blink_pounce_on_the_beat_and_chase():
    h = bg.Herd(1.6)
    vals = bg.default_values(being=16, being_count=3, being_speed=0.6)
    ys, xs = [], []
    for k in range(240):                                       # 4 s at 120 bpm: 8 beats
        ua, ub, uc, _ = h.step(1 / 60, vals, (0, 0, 0), dict(beat=k / 30.0))
        ys.append(ua[:3, 1].copy()); xs.append(ua[:3, :2].copy())
    ys = np.array(ys)
    h2 = bg.Herd(1.6)
    flat = np.array([h2.step(1 / 60, vals, (0, 0, 0), dict(beat=0.0))[0][:3, 1].copy() for _ in range(240)])
    hop = ys - flat
    assert hop.max() > 0.04 and hop.min() > -1e-6, "kittens hop upward on the beat"
    assert np.abs(np.diff(ys, axis=0)).max() < 0.03, "a hop is a smooth arc"
    on = hop[:, 0] > 0.01
    assert 0.25 < on.mean() < 0.6 and (hop[:, 1] > 0.01)[on].mean() < 0.1, "they take turns (every other beat each)"
    # the chase: kitten 1 is where kitten 0 was a moment ago
    steps, trail = [], []
    for k in range(600):
        ua, _, _, _ = h2.step(1 / 60, vals, (0, 0, 0), dict(beat=0.0))
        trail.append(ua[0, :2].copy()); steps.append(ua[1, :2].copy())
    trail, steps = np.array(trail), np.array(steps)
    best = min(np.abs(steps[lag:] - trail[:len(trail) - lag]).max() for lag in range(1, 300))
    assert best < 0.02, f"kitten 1 runs kitten 0's track behind it ({best:.3f})"
    assert (np.abs(ys) <= 0.5).all()


def test_kitten_eyes_blink(ctx):
    p = bg.BeingsPass(ctx, OUT)
    try:
        ua, ub, uc = (np.zeros((bg.N_MAX, 4), dtype="f4") for _ in range(3))
        ua[0], uc[0] = (0.0, 0.0, 0.0, 0.42), (0.0, 0.0, 1.0, 1.0)
        white = []
        for cyc in (0.30, 0.50):                               # eyes open, mid-blink (phase in blink cycles: TAU*4 each)
            ub[0] = ((cyc - (0.0 * 7.3) % 1.0) * bg.TAU * 4.0, 16, 0.0, 1.0)
            p.herd.step = lambda dt, values, bands, clock=None: (ua, ub, uc, 1)
            img = frame(p, bg.default_values(being=16, being_glow=0.0), 4.0, 0.0, (0, 0, 0))
            white.append((img[..., :3].min(axis=2) > 175).sum())
    finally:
        p.release()
    assert white[0] > 150 and white[1] < 0.6 * white[0], f"the big eye rims close in a blink: {white}"


# ---------------------------------------------------------------- serpents (engine)


def _engine_batch(script, out, jobs):
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "jobs.json"
    spec.write_text(json.dumps(jobs))
    r = subprocess.run([sys.executable, str(script), "--batch", str(spec), "--out", str(out), "--render", "640x400",
                        "--test-audio", "silent"], capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    from PIL import Image
    done = {d["name"]: d for d in json.loads((out / "batch.json").read_text())}
    assert all(d.get("nonfinite", 0) == 0 for d in done.values())
    return {n: np.asarray(Image.open(out / d["file"]).convert("RGB")) for n, d in done.items()}


SERP = dict(l_kene=0.0, l_beads=0.0, l_lines=0.0, l_ai=0.0, l_echo=0.0, serpent=1.0, serpent_size=0.7, chase=0.0)


def test_serpent_defaults_are_pixel_identical_to_the_engine_without_the_colour_hook(tmp_path):
    """'Before' = this very engine with the marked serpent-colour blocks cut out again, run side by side."""
    import re
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    plain, n = re.subn(r"[ \t]*// \[serpent-colour>.*?// <serpent-colour\]\n", "", src, flags=re.S)
    assert n == 2 and "serpColour" not in plain and "p_serpent_color" not in plain
    jobs = [dict(name="serpents", sets=SERP, frames=40), dict(name="over all", sets=dict(serpent=1.0, serpent_mode=2), frames=40),
            dict(name="chased", sets=dict(SERP, chase=0.8, serpent_count=8), frames=40)]
    tmp = ROOT / "_serpent_plain_tmp.py"
    tmp.write_text(plain, encoding="utf-8")
    try:
        before = _engine_batch(tmp, tmp_path / "before", jobs)
    finally:
        tmp.unlink()
    after = _engine_batch(ROOT / "live_visuals.py", tmp_path / "after", jobs)
    for name in before:
        d = np.abs(before[name].astype(int) - after[name].astype(int))
        assert before[name].max() > 100, "there are serpents in the frame"
        assert d.max() <= 1, f"{name}: the default serpents changed (max {d.max()}, mean {d.mean():.4f})"


def test_serpent_colour_modes(tmp_path):
    jobs = [dict(name=f"mode{m}", sets=dict(SERP, serpent_color=m, serpent_hue=0.33, serpent_hue2=0.83), frames=40) for m in range(6)]
    jobs += [dict(name="grey", sets=dict(SERP, serpent_sat=0.0), frames=40), dict(name="dim", sets=dict(SERP, serpent_bright=0.4), frames=40),
             dict(name="blue", sets=dict(SERP, serpent_color=1, serpent_hue=0.62), frames=40)]
    pic = _engine_batch(ROOT / "live_visuals.py", tmp_path, jobs)
    for m in range(1, 6):
        assert np.abs(pic[f"mode{m}"].astype(int) - pic["mode0"].astype(int)).mean() > 0.3, f"serpent mode {m} = mode 0"
    assert _near(_hues(pic["mode1"]), 0.33, 0.09) > 0.9 and _near(_hues(pic["blue"]), 0.62, 0.09) > 0.9
    two = _hues(pic["mode2"])
    assert _near(two, 0.33, 0.14) > 0.04 and _near(two, 0.83, 0.14) > 0.04, "both ends of the two-tone are on screen"
    assert (np.histogram(_hues(pic["mode3"]), bins=6, range=(0, 1))[0] > 0.04 * len(_hues(pic["mode3"]))).sum() >= 5
    assert _near(_hues(pic["mode4"]), 0.33, 0.09) < 0.7, "each serpent its own hue"
    g = pic["grey"].astype(int)
    lit = g.max(axis=2) > 60
    assert lit.mean() > 0.005 and np.percentile((g.max(axis=2) - g.min(axis=2))[lit], 90) <= 40, "depth 0: white serpents"
    assert pic["dim"].astype(int).sum() < 0.7 * pic["mode0"].astype(int).sum()


# ================================================================ sprite creature (2026-11-26): the user's own image
#
# Only generic generated test images are used here (discs, the built-in stand-in figure); user images live in art/
# (gitignored) and are never part of the repository.

SPRITE_NAMES = ["being_dance", "being_bounce", "being_spin", "being_look", "being_tint", "being_frames", "being_xfade",
                "being_faces", "being_sheet"]


def _disc_png(path, size=(400, 300), r=0.42, rgb=(40, 200, 255), soft=False, mode="RGBA"):
    """A generic test image: a coloured disc on transparent (PNG) or on white (JPG)."""
    from PIL import Image
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(xx - w / 2 + 0.5, yy - h / 2 + 0.5) - r * min(w, h)
    a = np.clip(0.5 - d, 0.0, 1.0) if soft else (d < 0).astype(np.float32)
    if mode == "RGBA":
        img = np.zeros((h, w, 4), np.uint8)
        img[..., :3] = rgb
        img[..., 3] = (a * 255).astype(np.uint8)
        Image.fromarray(img, "RGBA").save(path)
    else:
        img = np.full((h, w, 3), 255, np.float32)
        img = img * (1 - a[..., None]) + np.array(rgb, np.float32) * a[..., None]
        Image.fromarray(img.astype(np.uint8), "RGB").save(path, quality=95)
    return path


def test_sprite_form_and_params_are_appended():
    names = list(params.PARAMS)
    assert bg.FORM_KEYS[17] == "sprite" and bg.FORM_KEYS[16] == "kitten"
    assert params.OPTIONS["being"][17] == "Sprite (your image)" and params.PARAMS["being"][1] == 17.0
    i = names.index("being_dance")
    assert names[i:i + len(SPRITE_NAMES)] == SPRITE_NAMES and i > names.index("serpent_bright")
    for n in SPRITE_NAMES:
        lo, hi, d, group, label = params.PARAMS[n]
        assert lo <= d <= hi and group == "beings" and label and n in params.BYPASS_KEEP
    assert params.OPTIONS["being_dance"] == bg.DANCES and params.OPTIONS["being_look"] == bg.LOOKS
    assert params.OPTIONS["being_frames"] == bg.POSE_STEPS and params.OPTIONS["being_faces"] == bg.FACES
    for n in ("being_dance", "being_look", "being_frames", "being_faces"):
        assert len(params.OPTIONS[n]) == int(params.PARAMS[n][1] - params.PARAMS[n][0]) + 1


def test_no_third_party_names_anywhere_in_the_feature():
    banned = ["".join(reversed(w)) for w in ("lufetarg", "samoht bob", "aicrag", "ecaf ruoy laets", "daehdaed")]
    for f in ("beings.py", "params.py", "materials.py", "tests/test_beings.py", "panel.html"):
        txt = (ROOT / f).read_text(encoding="utf-8").lower()
        for w in banned:
            assert w not in txt, f"{f} names a third-party mark"
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert "art/" in gi, "user images (art/) never go into the repository"


def test_sequence_and_sheet_and_jpg_key(tmp_path):
    for k in (3, 1, 2):
        _disc_png(tmp_path / f"walk_{k}.png", r=0.2 + 0.05 * k)
    _disc_png(tmp_path / "other.png")
    assert [q.name for q in bg.sequence_for(tmp_path / "walk_2.png")] == ["walk_1.png", "walk_2.png", "walk_3.png"]
    assert bg.sequence_for(tmp_path / "other.png") == [tmp_path / "other.png"]
    poses, asp, _ = bg.decode_sprite(tmp_path / "walk_1.png", 1, 128)
    pad = 2 * (bg.SPR_SPREAD + 2)                            # the texture carries a border for the field / glow
    assert poses.shape[0] == 3 and poses.dtype == np.uint8 and asp == poses.shape[2] / poses.shape[1]
    assert abs((poses.shape[2] - pad) / (poses.shape[1] - pad) - 400 / 300) < 0.05
    inside = (poses[..., 3] > 127).mean(axis=(1, 2))
    assert inside[0] < inside[1] < inside[2], "the poses keep their number order"
    from PIL import Image                                    # a sheet: four poses side by side
    Image.fromarray(np.concatenate([np.asarray(Image.open(tmp_path / "other.png"))] * 4, axis=1)).save(tmp_path / "sheet.png")
    poses, asp, _ = bg.decode_sprite(tmp_path / "sheet.png", 4, 128)
    assert poses.shape[0] == 4 and abs((poses.shape[2] - pad) / (poses.shape[1] - pad) - 400 / 300) < 0.05
    jpg = _disc_png(tmp_path / "photo.jpg", mode="RGB")      # no alpha: keyed on the border colour
    poses, _, _ = bg.decode_sprite(jpg, 1, 128)
    a = poses[0, ..., 3] > 127
    h, w = a.shape
    assert a[h // 2, w // 2] and not a[2, 2] and 0.08 < a.mean() < 0.25, "the white background is keyed out"


def test_distance_field_keeps_the_exact_outline(tmp_path):
    """A disc of known radius: the field's 0.5 contour sits where the alpha edge was (sub-pixel), the colour is bled
    past the edge (no fringe when filtered)."""
    src = _disc_png(tmp_path / "disc.png", size=(300, 300), r=0.4, rgb=(250, 40, 90), soft=True)
    poses, _, _ = bg.decode_sprite(src, 1, 256)
    f = poses[0]
    H, W = f.shape[:2]
    pad = bg.SPR_SPREAD + 2
    scale = (W - 2 * pad) / 300.0
    row = f[H // 2, :, 3].astype(np.float32) / 255.0
    sd = (0.5 - row) * 2.0 * bg.SPR_SPREAD                   # texels, > 0 outside
    xs = np.arange(W) + 0.5
    k = np.nonzero((sd[:-1] < 0) & (sd[1:] >= 0))[0][-1]     # right-hand edge crossing
    edge = xs[k] + (0 - sd[k]) / (sd[k + 1] - sd[k])
    want = W / 2 + 0.4 * 300 * scale
    assert abs(edge - want) < 0.8, f"outline moved: {edge:.2f} vs {want:.2f}"
    assert np.abs(f[H // 2, -3, :3].astype(int) - np.array([250, 40, 90])).max() < 12, "colour bled past the edge"
    assert f[H // 2, -3, 3] < 40 and f[H // 2, W // 2, 3] > 220


def test_huge_and_bad_images_never_hurt(tmp_path, monkeypatch, ctx):
    from PIL import Image
    big = tmp_path / "big.png"
    Image.new("RGBA", (6000, 4000), (255, 0, 0, 255)).save(big)
    poses, asp, _ = bg.decode_sprite(big, 1, 256)
    pad = 2 * (bg.SPR_SPREAD + 2)
    assert max(poses.shape[1:3]) <= 256 and abs((poses.shape[2] - pad) / (poses.shape[1] - pad) - 1.5) < 0.03, "decoded small"
    monkeypatch.setattr(bg, "SPR_MAX_PIXELS", 1000)
    lib = bg.SpriteLibrary(ctx, sync=True)
    try:
        lib.want(big)
        lib.tick()
        assert lib.tex is not None and "could not use" in lib.note, "too big: refused, the stand-in shows"
        lib.want(tmp_path / "missing.png")
        lib.tick()
        assert "could not use" in lib.note
    finally:
        lib.release()


def test_library_decodes_off_the_render_thread(tmp_path, ctx):
    import threading
    import time
    src = _disc_png(tmp_path / "a.png")
    lib = bg.SpriteLibrary(ctx)
    try:
        assert lib._thread is None and lib.tex is None, "nothing until asked"
        lib.want(src)
        assert lib._thread is not None and lib._thread is not threading.current_thread()
        t0 = time.time()
        while lib.key is None and time.time() - t0 < 10:
            lib.tick()
            time.sleep(0.02)
        assert lib.key == (str(src), 1) and lib.frames == 1 and lib.uploads == 1
    finally:
        lib.release()


def test_sprite_costs_nothing_until_shown(bp):
    shot(bp, frames=2, being=3)
    assert bp.sprites is None, "no sprite on show: no library, no thread, no texture"
    src = bg.shader_source()
    assert "fSprite" in src and "#define SKIN" not in src and "skTap" not in src, "the plain program carries no skin code"


def _herd_run(vals, seconds=4.0, dt=1 / 60, bpm=120.0):
    h = bg.Herd(1.6)
    rows = []
    for k in range(int(round(seconds / dt))):
        ua, ub, uc, _ = h.step(dt, vals, (0, 0, 0), dict(beat=k * dt * bpm / 60.0))
        rows.append((ua.copy(), h.ud.copy()))
    return h, rows


def test_march_line_hops_on_every_beat_and_faces_its_way():
    vals = bg.default_values(being=bg.SPRITE, being_count=4, being_dance=1, being_speed=0.6, being_bounce=1.0, being_spin=0.0)
    _, rows = _herd_run(vals)
    _, base = _herd_run(dict(vals, being_bounce=0.0))
    ys = np.array([r[0][:4, 1] for r in rows]) - np.array([r[0][:4, 1] for r in base])
    per_beat = ys[:, 0].reshape(-1, 30).max(axis=1)
    assert (per_beat > 0.02).all(), "one hop on every beat"
    assert np.abs(np.diff(ys, axis=0)).max() < 0.03, "hops are smooth arcs"
    sy = np.array([r[1][0, 1] for r in rows])
    assert sy.min() < 0.95 and sy.max() > 1.04, "squash on landing, stretch in the air"
    assert np.abs(np.diff(sy)).max() < 0.08, "squash / stretch is continuous"
    xs = np.array([r[0][:4, 0] for r in base])
    vx = np.diff(xs[:, 0])
    w = np.array([r[1][0, 0] for r in base])[1:]
    settled = np.abs(vx) > 1e-4
    assert (np.sign(w[settled]) == np.sign(vx[settled])).mean() > 0.85, "the image faces the way it walks"
    lane = np.array([r[0][:4, 1] for r in base])
    assert np.ptp(lane[-1]) < 1e-6, "one line: everybody on the same lane"


def test_conga_follows_the_leader_and_mirror_pairs_face_each_other():
    vals = bg.default_values(being=bg.SPRITE, being_count=4, being_dance=2, being_speed=0.6, being_bounce=0.0, being_spin=0.0)
    _, rows = _herd_run(vals, seconds=10.0)
    lead = np.array([r[0][0, :2] for r in rows])
    foll = np.array([r[0][1, :2] for r in rows])
    best = min(np.abs(foll[lag:] - lead[:len(lead) - lag]).max() for lag in range(1, 400))
    assert best < 0.01, "the second dancer walks the leader's track behind it"
    vals = dict(vals, being_dance=3)
    _, rows = _herd_run(vals, seconds=3.0)
    ua, ud = rows[-1]
    assert ua[0, 0] < ua[1, 0] and ud[0, 0] > 0.9 and ud[1, 0] < -0.9, "partners face each other"
    _, rows = _herd_run(dict(vals, being_faces=2), seconds=3.0)
    assert (rows[-1][1][:4, 0] > 0.9).all(), "never flip: always as drawn"
    _, rows = _herd_run(dict(vals, being_faces=1), seconds=3.0)
    assert rows[-1][1][0, 0] < -0.9, "an image that faces left is turned the other way"


def test_bounce_in_place_spins_and_flock():
    vals = bg.default_values(being=bg.SPRITE, being_count=4, being_dance=4, being_speed=0.6, being_spin=0.0)
    _, rows = _herd_run(vals, seconds=4.0)
    xs = np.array([r[0][:4, 0] for r in rows])
    assert np.ptp(xs, axis=0).max() < 0.05, "bounce in place: they stay where they are"
    still = np.array([r[1][:4, 0] for r in rows[60:]])          # (after the first turn to face the floor)
    assert (np.abs(still) > 0.75).all() and (np.sign(still) == np.sign(still[-1])).all(), "no spins, no turning at 0"
    spun = np.array([r[1][:4, 0] for r in _herd_run(dict(vals, being_spin=1.0), seconds=16.0)[1]])
    assert (spun * np.sign(spun[0]) < -0.5).any() and np.abs(np.diff(spun, axis=0)).max() < 0.2, "spins, smoothly"
    fl = dict(vals, being_dance=5, being_count=6, being_size=0.4)
    _, rows = _herd_run(fl, seconds=12.0)
    for ua, _ in rows[240:]:
        p = ua[:6, :2]
        assert (np.abs(p[:, 0]) <= 0.8).all() and (np.abs(p[:, 1]) <= 0.5).all()
    p = rows[-1][0][:6, :2]
    dmin = min(np.hypot(*(p[i] - p[j])) for i in range(6) for j in range(i + 1, 6))
    assert dmin > 0.05, "the flock keeps apart"
    assert np.abs(rows[-1][0][:6, :2] - rows[-200][0][:6, :2]).max() > 0.02, "and moves"


def test_poses_step_on_the_beat_with_optional_crossfade():
    h = bg.Herd(1.6)
    h.frames = 4
    vals = bg.default_values(being=bg.SPRITE, being_count=2, being_dance=1, being_frames=0, being_xfade=0.0)
    seen = []
    for k in range(240):
        h.step(1 / 60, vals, (0, 0, 0), dict(beat=k / 30.0))
        seen.append(h.ud[0, 2:].copy())
    pa = [int(s[0]) for s in seen]
    assert sorted(set(pa)) == [0, 1, 2, 3] and sum(a != b for a, b in zip(pa, pa[1:])) == 7, "one pose per beat"
    assert all(abs(s[1] - round(s[1])) < 1e-6 for s in seen), "no crossfade asked: a cut"
    h2 = bg.Herd(1.6)
    h2.frames = 4
    xs = []
    for k in range(120):
        h2.step(1 / 60, dict(vals, being_frames=2, being_xfade=0.5), (0, 0, 0), dict(beat=k / 30.0))
        xs.append(h2.ud[0, 2:].copy())
    assert len({int(s[0]) for s in xs}) == 1, "every bar: one pose over these 4 beats"
    fr = [s[1] % 1.0 for s in xs]
    assert max(fr) > 0.5 and min(fr) == 0.0, "the crossfade ramps over the last half of the step"


def test_sprite_looks_and_image(tmp_path, ctx):
    src = _disc_png(tmp_path / "disc.png", size=(300, 300), r=0.4, rgb=(250, 60, 40))
    pics = []
    for look in range(4):
        p = bg.BeingsPass(ctx, OUT)
        p.sprite_sync, p.sprite_path = True, str(src)
        try:
            pics.append(bg.render_frames(p, bg.default_values(being=bg.SPRITE, being_look=look, being_count=1, being_size=0.9,
                                                              being_speed=0.0, being_color=1, being_hue=0.62), frames=2))
            assert p.sprites.key[0] == str(src) and p.sprites.frames == 1
        finally:
            p.release()
    img = pics[0][..., :3].astype(int)
    lit = img.max(axis=2) > 80
    assert ((img[..., 0] > img[..., 2] + 60) & lit).sum() > 0.6 * lit.sum(), "image look: the picture's own colour"
    assert _near(_hues(pics[2]), 0.62, 0.10) > 0.6, "solid silhouette: the colour mode"
    assert (lum(pics[1]) > 0.25).mean() < 0.5 * (lum(pics[2]) > 0.25).mean(), "glowing outline: light on the edge"
    assert np.abs(pics[3].astype(int) - pics[2].astype(int)).mean() < 1.0, "skinned fill with no skin = solid"


def test_sprite_edge_is_crisp_when_big(tmp_path, ctx):
    src = _disc_png(tmp_path / "d.png", size=(200, 200), r=0.45, rgb=(255, 255, 255))
    p = bg.BeingsPass(ctx, (1280, 800))
    p.sprite_sync, p.sprite_path = True, str(src)
    try:
        img = bg.render_frames(p, bg.default_values(being=bg.SPRITE, being_count=1, being_size=1.0, being_speed=0.0,
                                                    being_glow=0.0, being_bounce=0.0), frames=2)
    finally:
        p.release()
    a = img[..., 3].astype(np.float32) / 255.0
    row = a[np.argmax((a > 0.9).sum(axis=1))]
    ramp = ((row > 0.1) & (row < 0.9)).sum()
    assert (row > 0.9).sum() > 250 and ramp <= 8, f"silhouette edge stays sharp far above the decode size ({ramp} px ramp)"


def test_sprite_skin_slot_is_picker_only_and_fill_skins():
    mt = pytest.importorskip("materials")
    env = mt.ENVS[bg.SKIN_ENV]
    assert env["slots"] == ("fill", "sprite") and "sprite" in env["own"] and env["host"] == "beings"
    assert 'if names[i] in ENVS[env].get("own", ()):' in Path(mt.__file__).read_text(encoding="utf-8")
    skinned = bg.shader_source(mt.glsl(1, "pal"))
    assert "texSkin" in skinned and "#define SKIN" in skinned


def test_skinned_sprite_program_compiles(ctx):
    pytest.importorskip("materials")
    p = bg.BeingsPass(ctx, OUT)
    try:
        class Eng:
            def active(self, env):
                return False
        p.skins = Eng()
        assert p._skin_prog, "the skinned variant compiles"
    finally:
        p.release()


def test_sprite_source_comes_from_the_skin_slot(ctx, tmp_path):
    p = bg.BeingsPass(ctx, OUT)
    try:
        class Lib:
            def path(self, key):
                return tmp_path / key[1]

        class Eng:
            lib = Lib()

            def active(self, env):
                return False
        p._skins = Eng()
        p.skin_frame = {"beings": {"sprite": {"src": {"kind": "art", "name": "me.png"}}}}
        assert p._sprite_source() == tmp_path / "me.png"
        p.skin_frame = {"beings": {"sprite": {"src": {"kind": "layer", "name": "kene"}}}}
        assert p._sprite_source() is None, "a live layer is not an image"
        p.skin_frame = None
        assert p._sprite_source() is None
    finally:
        p.release()


def test_engine_and_panel_hooks():
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    assert '_bx["obj"].skins = skins' in src and 'ex["obj"].skin_frame' in src
    html = (ROOT / "panel.html").read_text(encoding="utf-8")
    assert "e.host==='beings'" in html and "mkCard('beings'" in html
