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
    """A frame stall (dt 0.1 .. 2 s) must not turn a ramp into a cut: at most half the range per step."""
    for big in (0.1, 2.0):
        samples = _run_glitch(1.0, 1.0, dt_fn=lambda i: big if i % 7 == 0 else 1 / 60)
        envs = [e for _, (e, _s) in samples]
        assert max(abs(b - a) for a, b in zip(envs, envs[1:])) <= 0.5 + 1e-9


def test_glitch_knobs_are_latched_for_the_whole_event():
    """A MOD / LFO on flt_glitch or flt_glitch_block cannot re-cut the blocks while a glitch is visible."""
    st = filters.ChainState()
    v = _vals(flt_glitch=1.0, flt_glitch_rate=1.0, flt_glitch_block=0.2)
    seen = []
    for i in range(2000):
        v["flt_glitch_block"] = 0.5 + 0.5 * math.sin(i * 0.9)        # wild modulation every frame
        v["flt_glitch"] = 0.5 + 0.5 * math.sin(i * 1.3)
        prev = (st.glitch_seed, st.glitch_blk, st.glitch_amt)
        st.advance(1 / 60, v)
        if st.glitch_env > 0.0 and st.glitch_seed == prev[0]:
            assert (st.glitch_blk, st.glitch_amt) == prev[1:]
        seen.append(st.glitch_env)
    assert max(seen) > 0.9


# ------------------------------------------------------------------ CPU state of the pass (no GL)
def test_dream_drift_phases_wrap_at_the_noise_period():
    st = filters.ChainState()
    for _ in range(200000):
        st.advance(0.05, _vals())
    assert all(0.0 <= x < filters.NOISE_P for x in st.drift) and 0.0 <= st.wob < 2 * math.pi
    assert filters.NOISE_P == 64.0                                       # = the shader's hashP period


def test_run_reports_failure_so_the_engine_can_fall_back(monkeypatch):
    class Boom:
        def apply(self, *a, **k):
            raise MemoryError("VRAM")
    logs = []
    assert filters.run(Boom(), None, {}, {}, 1 / 60, 0.5, 1.0, 0.0, log=logs.append) is False
    assert logs and "MemoryError" in logs[0]

    class Fine:
        def apply(self, *a, **k):
            self.k = k
    f = Fine()
    assert filters.run(f, None, {}, {}, 1 / 60, 0.5, 0.7, 0.2) is True
    assert f.k["limiter"] == 0.5 and f.k["master"] == 0.7 and f.k["blackout"] == 0.2
    monkeypatch.setenv("FILTERS_TEST_FAULT", "1")
    assert filters.run(f, None, {}, {}, 1 / 60, 0.5, 0.7, 0.2, log=lambda m: None) is False


def test_engine_falls_back_and_skips_while_blacked_out():
    src = (HERE / "live_visuals.py").read_text(encoding="utf-8")
    i = src.index("filt_ok = filters_run(")
    block = src[i:i + 1600]
    assert "scene_vao.render(moderngl.TRIANGLE_STRIP)" in block           # the plain redraw on failure
    assert 'scene_prog[kname].value = v' in block                         # real limiter / master / blackout back first
    assert 'bo < 0.999 and vals["master"] > 0.001' in src


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


def _render(gl, vals, img, frames=12, reset=True):
    ctx, fp, disp, w, h, np = gl
    if reset:
        fp.reset()
    for _ in range(frames):
        disp.write(img.astype("f2").tobytes())
        fp.apply(disp, vals, dict(beat=0.0, bar_phase=0.0), 1 / 60, limiter=0.0, master=1.0, blackout=0.0)
    out = np.frombuffer(disp.read(), dtype="f2").reshape(h, w, 4).astype("f4")[..., :3]
    return out


def _one(ft, amount=1.0, **kw):
    i = params.FILTER_TYPES.index(ft) + 1
    return _vals(flt_slot1=i, **{f"flt_{ft}": amount}, **kw)


VARIANTS = [(ft, {}) for ft in params.FILTER_TYPES] + [
    ("halftone", {"flt_ht_style": 1}), ("halftone", {"flt_ht_style": 2}), ("chroma", {"flt_chroma_mode": 1}),
    ("chroma", {"flt_chroma_mode": 2}), ("posterize", {"flt_post_palette": 1.0}), ("neon", {"flt_neon_color": 1.0}),
    ("kaleido", {"flt_kal_mode": 0}), ("kaleido", {"flt_kal_mode": 1}), ("dream", {"flt_dream_motif": 1.0}),
    ("dream", {"flt_dream_motif": 0.0, "flt_dream_scale": 1.0}), ("sort", {"flt_sort_len": 1.0})]


@pytest.mark.parametrize("ft,extra", VARIANTS, ids=[f"{f}{''.join(f'-{k[4:]}{v}' for k, v in e.items())}" for f, e in VARIANTS])
def test_each_filter_changes_the_picture_keeps_black_and_never_blows_out(gl, ft, extra):
    np = gl[5]
    img = _test_image(gl[3], gl[4])
    frames = 45 if ft == "glitch" else 12                                # the first glitch may start after 0.5 s
    out = _render(gl, _one(ft, **extra), img, frames=frames)
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
    assert _render(gl, _one(ft, **extra), black, frames=60).max() < 0.02, f"{ft} {extra} makes light from black"


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
    lvl = np.frombuffer(fp.level.read(), dtype="u1").reshape(h, w, 4)[..., 3].astype("f4") / 255.0
    assert abs(lvl.mean() - a.max(-1).mean()) < 0.01                 # alpha = post-filter level BEFORE the limiter


def test_safe_clamps_negative_channels_and_blanks_nan(gl):
    """H4: a slightly negative channel (hueRot of a saturated colour) must not blank the whole pixel."""
    np = gl[5]
    ctx = gl[0]
    prog = ctx.program(vertex_shader=filters.VS, fragment_shader=filters.COMMON + """
uniform float nanv; out vec4 f;
void main(){ vec3 c = gl_FragCoord.x < 1.0 ? vec3(-0.1, 0.5, 0.7) : gl_FragCoord.x < 2.0 ? vec3(nanv) : vec3(0.0, -1e6, 0.0);
  f = vec4(safe(c), 1.0); }""")
    prog["nanv"].value = float("nan")
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])
    fbo = ctx.simple_framebuffer((3, 1), dtype="f4")
    fbo.use()
    vao.render(5)
    px = np.frombuffer(fbo.read(components=3, dtype="f4"), dtype="f4").reshape(3, 3)
    assert np.allclose(px[0], [0.0, 0.5, 0.7], atol=1e-3)
    assert np.allclose(px[1], 0.0) and np.allclose(px[2], 0.0)
    for o in (fbo, vao, vbo, prog):
        o.release()


def test_dream_reseeds_after_a_pause(gl):
    """M2: dream unused for > DREAM_RESEED_S (wall clock: filters off, or out of the chain) restarts from the picture."""
    np = gl[5]
    a = _test_image(gl[3], gl[4])
    b = a[::-1].copy()                                                # a different picture
    fresh = _render(gl, _one("dream"), b, frames=1)
    _render(gl, _one("dream"), a, frames=30)
    gl[1]._dream_last -= 10.0                                         # as if 10 s went by with the chain off
    after = _render(gl, _one("dream"), b, frames=1, reset=False)
    assert np.abs(after - fresh).mean() < 0.01


def test_full_chain_of_six_is_finite_and_bounded(gl):
    np = gl[5]
    t = {ft: i + 1 for i, ft in enumerate(params.FILTER_TYPES)}
    v = _vals(flt_slot1=t["dream"], flt_slot2=t["oil"], flt_slot3=t["neon"], flt_slot4=t["chroma"],
              flt_slot5=t["bloom"], flt_slot6=t["kaleido"], flt_dream=1, flt_oil=1, flt_neon=1, flt_chroma=1,
              flt_bloom=1, flt_kaleido=1)
    out = _render(gl, v, _test_image(gl[3], gl[4]), frames=30)
    assert np.isfinite(out).all() and out.max() <= 1.0 + 1e-3


# ------------------------------------------------------------------ panel in a headless browser (needs playwright)
def test_panel_filters_tab_and_dream_button(tmp_path):
    pw_api = pytest.importorskip("playwright.sync_api")
    import socket
    import threading
    import time
    from PIL import Image
    import control
    from remote_control import RemoteControl
    (tmp_path / "art").mkdir()
    (tmp_path / "tex").mkdir()
    Image.new("RGB", (64, 48), (200, 50, 120)).save(tmp_path / "art" / "piece.png")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art",
                       tex_dir=tmp_path / "tex")
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            ctl.apply(rc.drain())
            ctl.update(1 / 60, [0.5, 0.3, 0.1])
            rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60))
            time.sleep(1 / 60)
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    shots = HERE / "test_frames" / "filters"
    shots.mkdir(parents=True, exist_ok=True)
    try:
        with pw_api.sync_playwright() as pw:
            br = None
            for ch in ("msedge", "chrome", None):
                try:
                    br = pw.chromium.launch(channel=ch, headless=True) if ch else pw.chromium.launch(headless=True)
                    break
                except Exception:
                    continue
            if br is None:
                pytest.skip("no browser")
            page = br.new_page(viewport=dict(width=1280, height=900))
            errs = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/?k=tok")
            page.click("#tabs button[data-tab=filters]")
            page.click("#fltc button:has-text('+ ADD FILTER')")
            page.click("#fltadd button:has-text('Deep dream')")
            page.click("#fltc button:has-text('+ ADD FILTER')")
            page.click("#fltadd button:has-text('Bloom')")
            end = time.time() + 5
            while time.time() < end and not (ctl.base["flt_slot2"] == 9 and ctl.base["flt_dream"] >= 0.7):
                time.sleep(0.05)
            assert ctl.base["flt_slot1"] == 1 and ctl.base["flt_slot2"] == 9 and ctl.base["flt_dream"] >= 0.7
            page.wait_for_timeout(400)
            page.locator("#fltc .fslot").nth(1).locator("button:has-text('▲')").click()     # bloom first now
            end = time.time() + 5
            while time.time() < end and not (ctl.base["flt_slot1"] == 9 and ctl.base["flt_slot2"] == 1):
                time.sleep(0.05)
            assert ctl.base["flt_slot1"] == 9 and ctl.base["flt_slot2"] == 1
            page.wait_for_timeout(400)
            page.screenshot(path=str(shots / "panel_filters_tab.png"))
            page.click("#tabs button[data-tab=art]")
            page.wait_for_selector("#pg-art .pick .art .dmb", timeout=8000)
            page.locator("#pg-art .pick .art .dmb").first.click()
            assert page.locator("#dpop.on").count() == 1
            page.click("#dpop button:has-text('Eyes')")
            page.wait_for_timeout(300)
            page.screenshot(path=str(shots / "panel_dreamify.png"))
            assert not errs, errs
            br.close()
    finally:
        stop.set()
        th.join(timeout=2)
        ctl.close()
        rc.stop()


# ------------------------------------------------------------------ H1 through the real engine (headless, 320x200)
def _batch(tmp_path, name, jobs, env_extra):
    import json
    import os
    import subprocess
    out = tmp_path / name
    jf = tmp_path / f"{name}.json"
    jf.write_text(json.dumps(jobs), encoding="utf-8")
    env = dict(os.environ, **env_extra)
    r = subprocess.run([sys.executable, str(HERE / "live_visuals.py"), "--batch", str(jf), "--out", str(out),
                        "--render", "320x200"], cwd=str(HERE), env=env, capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return out, r.stdout


def test_engine_never_shows_an_unlimited_frame_when_the_filters_fail(tmp_path):
    pytest.importorskip("moderngl")
    import numpy as np
    from PIL import Image
    dream = {"flt_slot1": 1, "flt_dream": 1.0}
    jobs = [{"name": "plain_m03", "sets": {"master": 0.3}, "frames": 40},
            {"name": "filt_m03", "sets": dict(dream, master=0.3), "frames": 40},
            {"name": "filt_bo", "sets": dict(dream, blackout=1.0), "frames": 40}]
    ok, _ = _batch(tmp_path, "ok", jobs, {"FILTERS_TEST_FAULT": "0"})
    bad, log = _batch(tmp_path, "bad", jobs, {"FILTERS_TEST_FAULT": "1"})
    assert "fell back to the plain scene" in log
    img = lambda d, n: np.asarray(Image.open(d / f"{n}.png"), "f4") / 255.0   # noqa: E731
    plain = img(ok, "plain_m03")
    assert np.abs(img(bad, "filt_m03") - plain).max() < 2 / 255        # failure = the normal limited, mastered scene
    assert img(bad, "filt_bo").max() == 0.0 and img(ok, "filt_bo").max() == 0.0   # blackout is ALWAYS black
    assert np.abs(img(ok, "filt_m03") - plain).mean() > 0.002           # (and without the fault the dream does run)
    assert img(ok, "filt_m03").mean() < img(ok, "plain_m03").mean() * 3.3 + 0.01
