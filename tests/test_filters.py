"""Post-FX filter chain (filters.py): contract, chain logic, photosensitive glitch clock, engine / panel hooks.

Pure-Python tests run anywhere. The GL tests (class TestGL) need an OpenGL 3.3 standalone context; they render
small (320x200) frames and are skipped when no context can be made.
"""
import math
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import params  # noqa: E402
import filters  # noqa: E402


# ------------------------------------------------------------------ contract
def test_every_filter_type_has_an_amount_param_defaulting_to_off():
    assert len(params.FILTER_TYPES) == 10
    for ft in params.FILTER_TYPES:
        name = f"flt_{ft}"
        lo, hi, default, group, _ = params.PARAMS[name]
        assert (lo, hi, default, group) == (0.0, 1.0, 0.0, "filters"), name


def test_slots_are_stepped_selectors_with_names():
    names = params.OPTIONS["flt_slot1"]
    assert names[0] == "Empty" and len(names) == len(params.FILTER_TYPES) + 1
    for i in range(1, filters.N_SLOTS + 1):
        lo, hi, default, group, _ = params.PARAMS[f"flt_slot{i}"]
        assert (lo, hi, default, group) == (0.0, float(len(params.FILTER_TYPES)), 0.0, "filters")
        assert params.OPTIONS[f"flt_slot{i}"] == names
    assert filters.N_SLOTS == 6


def test_filter_params_are_appended_after_everything_that_existed():
    names = list(params.PARAMS)
    first = min(i for i, n in enumerate(names) if n.startswith("flt_"))
    assert names.index("serpent_bright") < first
    assert all(n.startswith("flt_") for n in names[first:first + len(filters.ALL_PARAMS)])


def test_every_filter_param_belongs_to_the_filters_group():
    for n in filters.ALL_PARAMS:
        assert params.PARAMS[n][3] == "filters", n
    assert set(filters.ALL_PARAMS) == {n for n, v in params.PARAMS.items() if v[3] == "filters"}


def test_bypass_classification_amounts_neutral_rest_kept():
    for ft in params.FILTER_TYPES:
        assert params.NEUTRAL[f"flt_{ft}"] == 0.0
    for n in filters.ALL_PARAMS:
        if n not in filters.AMOUNTS.values():
            assert n in params.BYPASS_KEEP, n
    cover = params.bypass_cover()
    assert set(cover["filters"]) == set(filters.AMOUNTS.values())
    assert set(filters.AMOUNTS.values()) <= set(cover["fx"])          # ALL FX OFF also kills the filters


def test_control_treats_slots_and_modes_as_stepped():
    import control
    for n in ("flt_slot1", "flt_slot6", "flt_ht_style", "flt_chroma_mode", "flt_kal_mode"):
        assert n in control.DISCRETE, n
    meta = {p["name"]: p for p in control.meta()["params"]}
    assert meta["flt_slot3"]["options"][1] == "Deep dream"
    assert "filters" in control.BYPASS_COVER


# ------------------------------------------------------------------ chain logic
def _vals(**kw):
    v = params.defaults()
    v.update(kw)
    return v


def test_defaults_are_inactive_zero_cost():
    assert filters.chain(_vals()) == []
    assert not filters.active(_vals())


def test_chain_follows_slot_order_and_skips_empty_off_and_duplicates():
    t = {ft: i + 1 for i, ft in enumerate(params.FILTER_TYPES)}
    v = _vals(flt_slot1=t["bloom"], flt_slot2=0, flt_slot3=t["dream"], flt_slot4=t["bloom"], flt_slot5=t["oil"],
              flt_slot6=t["glitch"], flt_bloom=0.5, flt_dream=1.0, flt_oil=0.0, flt_glitch=0.3)
    assert filters.chain(v) == [("bloom", 0.5), ("dream", 1.0), ("glitch", 0.3)]
    assert filters.active(v)


def test_chain_rounds_slot_values_and_clamps_garbage():
    v = _vals(flt_slot1=1.4, flt_dream=0.7, flt_slot2=float("nan"), flt_slot3=99.0, flt_slot4=-3)
    assert filters.chain(v) == [("dream", 0.7)]
    v = _vals(flt_slot1=1.0, flt_dream=float("nan"))
    assert filters.chain(v) == []


def test_amount_without_a_slot_does_nothing():
    assert filters.chain(_vals(flt_bloom=1.0)) == []


def test_slot_value_names_round_trip():
    for i, ft in enumerate(params.FILTER_TYPES):
        assert filters.type_of(i + 1) == ft
    assert filters.type_of(0) is None and filters.type_of(11) is None


# ------------------------------------------------------------------ glitch clock: photosensitive-safe by construction
def _run_glitch(rate, amount, seconds=60.0, fps=60.0, dt_fn=None):
    g = filters.GlitchClock(seed=7)
    out, t = [], 0.0
    n = int(seconds * fps)
    for i in range(n):
        dt = dt_fn(i) if dt_fn else 1.0 / fps
        t += dt
        out.append((t, g.step(dt, rate, amount)))
    return out


@pytest.mark.parametrize("rate", [0.0, 0.3, 0.7, 1.0])
def test_glitch_events_never_exceed_2_5_per_second(rate):
    samples = _run_glitch(rate, 1.0)
    onsets = [t for (t, (env, seed)), (_, (penv, _ps)) in zip(samples[1:], samples[:-1]) if env > 0.0 and penv == 0.0]
    for i, t0 in enumerate(onsets):
        assert sum(1 for t in onsets[i:] if t - t0 < 1.0) <= 3      # WCAG 2.3.1: no more than 3 in any 1 s
    assert all(b - a >= 1.0 / filters.GLITCH_MAX_HZ - 1e-6 for a, b in zip(onsets, onsets[1:]))
    assert len(onsets) >= 5                                          # it does glitch


def test_glitch_envelope_ramps_never_cut():
    samples = _run_glitch(1.0, 1.0)
    dt = 1.0 / 60.0
    envs = [e for _, (e, _s) in samples]
    worst = max(abs(b - a) for a, b in zip(envs, envs[1:]))
    assert worst <= dt / filters.GLITCH_RAMP_S + 1e-6
    assert max(envs) <= 1.0 and min(envs) >= 0.0


def test_glitch_seed_only_changes_while_invisible():
    samples = _run_glitch(1.0, 1.0)
    for (_, (pe, ps)), (_, (e, s)) in zip(samples, samples[1:]):
        if s != ps:
            assert pe == 0.0


def test_glitch_off_at_amount_zero_and_freeze_holds():
    assert all(e == 0.0 for _, (e, _s) in _run_glitch(1.0, 0.0))
    g = filters.GlitchClock(seed=1)
    for _ in range(200):
        g.step(1 / 60, 1.0, 1.0)
    before = g.step(0.0, 1.0, 1.0)
    for _ in range(50):
        assert g.step(0.0, 1.0, 1.0) == before


def test_glitch_survives_huge_dt_without_bursts():
    samples = _run_glitch(1.0, 1.0, dt_fn=lambda i: 2.0 if i % 50 == 0 else 1 / 60)
    envs = [e for _, (e, _s) in samples]
    assert max(abs(b - a) for a, b in zip(envs, envs[1:])) <= 1.0


# ------------------------------------------------------------------ CPU state of the pass (no GL)
def test_spin_phase_wraps_and_freeze_stops_it():
    st = filters.ChainState()
    for _ in range(10000):
        st.advance(0.05, _vals(flt_kal_spin=1.0))
    assert 0.0 <= st.kal_phase < 2 * math.pi
    p = st.kal_phase
    st.advance(0.0, _vals(flt_kal_spin=1.0))
    assert st.kal_phase == p
    st.advance(0.05, _vals(flt_kal_spin=1.0, freeze=1.0))
    assert st.kal_phase == p
    assert 0.0 <= st.t < filters.T_WRAP


# ------------------------------------------------------------------ engine + panel hooks (source checks, no GL)
def test_engine_hook_is_wired():
    src = (HERE / "live_visuals.py").read_text(encoding="utf-8")
    assert "from filters import FilterPass" in src
    assert "[filters>" in src and "<filters]" in src
    import live_visuals
    for n in filters.ALL_PARAMS:
        assert n in live_visuals.CPU_ONLY, n


def test_panel_has_the_filters_block_and_dreamify():
    html = (HERE / "panel.html").read_text(encoding="utf-8")
    assert "/* [filters>" in html and "<filters] */" in html
    assert "dreamify" in html and "dream-status" in html


# ------------------------------------------------------------------ GL (small renders; skipped without a context)
def _ctx():
    try:
        import moderngl
        return moderngl.create_standalone_context(require=330)
    except Exception as ex:   # pragma: no cover - depends on the machine
        pytest.skip(f"no GL context: {ex}")


def _test_image(w, h):
    import numpy as np
    y, x = np.mgrid[0:h, 0:w].astype("f4")
    cx, cy = (x - w / 2) / h, (y - h / 2) / h
    r = np.hypot(cx, cy)
    a = np.arctan2(cy, cx)
    rings = (np.sin(r * 60.0 + np.sin(a * 8.0) * 2.0) > 0.3).astype("f4")
    img = np.zeros((h, w, 4), "f4")
    img[..., 0] = rings * (0.5 + 0.5 * np.sin(a * 3.0))
    img[..., 1] = rings * (0.5 + 0.5 * np.sin(a * 3.0 + 2.1))
    img[..., 2] = rings * (0.5 + 0.5 * np.sin(a * 3.0 + 4.2))
    img[..., :3] *= (r < 0.45)[..., None]                      # black margin (the 25 % black floor)
    img[..., 3] = 1.0
    return img


@pytest.fixture(scope="module")
def gl():
    import numpy as np
    ctx = _ctx()
    w, h = 320, 200
    disp = ctx.texture((w, h), 4, dtype="f2")
    fp = filters.FilterPass(ctx, (w, h))
    yield ctx, fp, disp, w, h, np
    fp.release()
    ctx.release()


def _render(gl, vals, img, frames=12):
    ctx, fp, disp, w, h, np = gl
    for _ in range(frames):
        disp.write(img.astype("f2").tobytes())
        fp.apply(disp, vals, dict(beat=0.0, bar_phase=0.0), 1 / 60, limiter=0.0, master=1.0, blackout=0.0)
    out = np.frombuffer(disp.read(), dtype="f2").reshape(h, w, 4).astype("f4")[..., :3]
    return out


def _one(ft, amount=1.0, **kw):
    i = params.FILTER_TYPES.index(ft) + 1
    return _vals(flt_slot1=i, **{f"flt_{ft}": amount}, **kw)


@pytest.mark.parametrize("ft", params.FILTER_TYPES)
def test_each_filter_changes_the_picture_keeps_black_and_never_blows_out(gl, ft):
    np = gl[5]
    img = _test_image(gl[3], gl[4])
    out = _render(gl, _one(ft), img)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-3
    diff = np.abs(out - img[..., :3]).mean()
    assert diff > 0.004, f"{ft} did nothing ({diff:.4f})"
    black_in = (img[..., :3].max(-1) < 0.04).mean()
    black_out = (out.max(-1) < 0.04).mean()
    assert black_out >= min(0.25, black_in * 0.8), f"{ft}: black floor {black_out:.2f} (in {black_in:.2f})"
    assert out.mean() <= img[..., :3].mean() * 1.6 + 0.02, f"{ft}: brighter than allowed"
    black = np.zeros_like(img)
    black[..., 3] = 1.0
    assert _render(gl, _one(ft), black, frames=30).max() < 0.02, f"{ft} makes light from black"


@pytest.mark.parametrize("ft", params.FILTER_TYPES)
def test_each_filter_is_nan_safe(gl, ft):
    np = gl[5]
    img = _test_image(gl[3], gl[4])
    img[50:60, 50:60, :3] = np.nan
    img[80:90, 80:90, :3] = np.inf
    out = _render(gl, _one(ft), img)
    assert np.isfinite(out).all(), ft


def test_final_stage_applies_limiter_master_and_blackout(gl):
    np = gl[5]
    ctx, fp, disp, w, h, _ = gl
    img = _test_image(w, h)
    v = _one("posterize", 0.001 + 1e-6)
    a = _render(gl, v, img, frames=1)
    disp.write(img.astype("f2").tobytes())
    fp.apply(disp, v, {}, 1 / 60, limiter=0.0, master=0.5, blackout=0.0)
    b = np.frombuffer(disp.read(), dtype="f2").reshape(h, w, 4).astype("f4")[..., :3]
    assert abs(b.mean() - a.mean() * 0.5) < 0.01
    disp.write(img.astype("f2").tobytes())
    fp.apply(disp, v, {}, 1 / 60, limiter=0.0, master=1.0, blackout=1.0)
    assert np.frombuffer(disp.read(), dtype="f2").astype("f4").reshape(h, w, 4)[..., :3].max() == 0.0
    disp.write(img.astype("f2").tobytes())
    fp.apply(disp, v, {}, 1 / 60, limiter=0.8, master=1.0, blackout=0.0)
    c = np.frombuffer(disp.read(), dtype="f2").reshape(h, w, 4).astype("f4")[..., :3]
    assert c.mean() < a.mean()
    assert fp.level is not None                                     # the meter reads the post-filter level


def test_full_chain_of_six_is_finite_and_bounded(gl):
    np = gl[5]
    t = {ft: i + 1 for i, ft in enumerate(params.FILTER_TYPES)}
    v = _vals(flt_slot1=t["dream"], flt_slot2=t["oil"], flt_slot3=t["neon"], flt_slot4=t["chroma"],
              flt_slot5=t["bloom"], flt_slot6=t["kaleido"], flt_dream=1, flt_oil=1, flt_neon=1, flt_chroma=1,
              flt_bloom=1, flt_kaleido=1)
    out = _render(gl, v, _test_image(gl[3], gl[4]), frames=30)
    assert np.isfinite(out).all() and out.max() <= 1.0 + 1e-3
