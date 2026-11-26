import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import live_visuals as lv  # noqa: E402

UNIT = [(0, 0), (1, 0), (1, 1), (0, 1)]


def apply(H, x, y):
    q = H @ np.array([x, y, 1.0])
    return q[:2] / q[2]


def test_homography_identity():
    assert np.allclose(lv.homography(UNIT, UNIT), np.eye(3), atol=1e-9)


def test_homography_maps_corners_to_keystoned_quad():
    dst = [(0.1, 0.05), (0.92, 0.0), (1.0, 0.97), (0.0, 0.9)]
    H = lv.homography(UNIT, dst)
    for s, d in zip(UNIT, dst):
        assert np.allclose(apply(H, *s), d, atol=1e-9)
    # inverse sends screen corners back to content unit square
    Hi = np.linalg.inv(H)
    for s, d in zip(UNIT, dst):
        assert np.allclose(apply(Hi, *d), s, atol=1e-9)


def test_synthetic_audio_fallback_in_range_and_pulses():
    a = lv.Audio("none")
    vals = np.array([a.get(t) for t in np.linspace(0, 4, 400)])
    assert vals.shape == (400, 3)
    assert (vals >= 0).all() and (vals <= 1.0001).all()
    assert vals[:, 0].max() > 0.9 and vals[:, 0].min() < 0.1  # kick pulses


def _stats(path):
    img = np.asarray(Image.open(path)).astype(float)
    mx, mn = img.max(axis=2), img.min(axis=2)
    lit = mx > 40
    return dict(
        mean=img.mean(),
        black=(mx < 8).mean(),                       # true black: the projector emits nothing here
        lit=lit.mean(),
        blown=(mn > 235).mean(),                     # near-white blowout
        saturated=(((mx - mn) / np.maximum(mx, 1) > 0.5) & lit).sum() / max(1, lit.sum()),
        std=img.std(axis=(0, 1)).mean(),
        hue_spread=np.ptp(img.reshape(-1, 3)[lit.ravel()].mean(axis=0)) if lit.any() else 0.0,
    )


def _check_projection_frame(path):
    s = _stats(path)
    assert s["mean"] > 5, f"frame is (nearly) black: {s}"
    assert s["mean"] < 70, f"frame is too bright / washed out for projection: {s}"
    assert s["black"] > 0.25, f"needs at least 25% true black (negative space): {s}"
    assert s["lit"] > 0.08, f"too little geometry on screen: {s}"
    assert s["blown"] < 0.01, f"blown out to white: {s}"
    assert s["saturated"] > 0.5, f"lit pixels should be saturated jewel tones, not pastel: {s}"
    assert s["std"] > 10, f"frame lacks variation: {s}"


def _run(tmp_path, *args, frames=90, timeout=240):
    """One headless engine run -> (stdout, output dir). Frames land in tmp_path, not in the project."""
    out = tmp_path / "frames"
    r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--test", str(frames), "--out", str(out), *map(str, args)],
                       capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "[test]" in r.stdout and "nonfinite=0" in r.stdout, r.stdout[-600:]
    return r.stdout, out


def _batch(out, jobs, *args, timeout=600):
    """Many looks in ONE engine process (--batch): {name: dict(path=..., nonfinite=..., **job)}.
    jobs: {name: dict(sets={param: value}, frames=N, script={frame: [action, ...]})}."""
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "jobs.json"
    spec.write_text(json.dumps([dict(name=n, **j) for n, j in jobs.items()]))
    r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--batch", str(spec), "--out", str(out), *map(str, args)],
                       capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    done = {d["name"]: d for d in json.loads((out / "batch.json").read_text())}
    assert set(done) == set(jobs)
    return {n: dict(d, path=out / d["file"]) for n, d in done.items()}


def _sets(**kw):
    return [x for k, v in kw.items() for x in ("--set", f"{k}={v}")]


def _preset(name, **over):
    return dict(zip(lv.PRESET_KEYS, lv.MODE_W[lv.MODES.index(name)]), **over)


def test_headless_render_produces_nonblack_colorful_frames(tmp_path):
    stdout, out = _run(tmp_path, frames=120)
    assert "unbound: none" in stdout                     # every param reaches the shader (or is CPU-side)
    assert "[control] StaticController" in stdout        # headless default: deterministic stand-in
    active = re.search(r"\[uniforms\] clock: (.*)", stdout).group(1).split(", ")
    assert set(active) == set(lv.CLOCK_UNIFORMS), f"clock uniforms optimised out of the shader: {active}"
    # frame 30 = early, 60 = mid, 120 = after 2 s of feedback (where the first shader had blown out)
    for name in ("frame_0030.png", "frame_0060.png", "frame_0120.png"):
        _check_projection_frame(out / name)


def test_feedback_does_not_accumulate_at_max_amount_or_long_runs(tmp_path):
    # 10 s of the synthetic kick in the densest preset with trails at maximum: brightness stays bounded, black survives
    _, out = _run(tmp_path, "--mode", "all", *_sets(trails=0.98, l_echo=1.0), frames=600)
    a, b = _stats(out / "frame_0300.png"), _stats(out / "frame_0600.png")
    _check_projection_frame(out / "frame_0600.png")
    assert b["mean"] < a["mean"] * 1.6 + 5, f"brightness keeps growing: {a['mean']:.1f} -> {b['mean']:.1f}"


TRAVEL = dict(travel=0.7)
LOOKS = {
    # every layer preset in the mandala
    **{f"mandala {m}": _preset(m) for m in lv.MODES},
    # the flat field (symmetry 0) tiles the frame and must still keep its negative space
    **{f"flat {m}": _preset(m, symmetry=0) for m in ("kene", "kene+beads", "beads", "both", "all")},
    "flat kene dense 45deg": _preset("kene", symmetry=0, density=0.85, flow_dir=0.125, palette=2.5),
    "half symmetry": _preset("kene", symmetry=0.5),
    "zoomed in": _preset("all", zoom=0.6),
    "contorted": _preset("all", twirl=0.7, bulge=0.5, wave=0.6, swim=0.8),
    "contorted flat": _preset("kene", symmetry=0, twirl=-0.7, bulge=-0.6, wave=1.0),
    "tube": _preset("kene+beads", travel_mode=1, **TRAVEL),
    "ground plane": _preset("kene", symmetry=0, travel_mode=1, **TRAVEL),
    "depth layers": _preset("kene+beads", travel_mode=2, **TRAVEL),
    "depth layers flat": _preset("kene", symmetry=0, travel_mode=2, **TRAVEL),
    "droste": _preset("all", travel_mode=3, **TRAVEL),
    "droste flat": _preset("kene", symmetry=0, travel_mode=3, **TRAVEL),
    "zoom tunnel": _preset("all", travel_mode=0, **TRAVEL),
    "serpents + warp": _preset("all", serpent=1.0, serpent_mode=2, serpent_count=8),
    "chase 0": _preset("kene+beads", chase=0.0),
    "inverted": _preset("all", invert=1.0),
    "inverted flat kene": _preset("kene", symmetry=0, invert=1.0),
    "strobe pulse": _preset("all", strobe=1.0),
    "frozen": _preset("all", freeze=1.0),
    "punch": _preset("kene", punch=1.0),
}
LOOKS = {k: {p: v for p, v in s.items() if p in lv.PARAMS} for k, s in LOOKS.items()}


@pytest.fixture(scope="module")
def looks(tmp_path_factory):
    return _batch(tmp_path_factory.mktemp("looks"), {n: dict(sets=s, frames=90) for n, s in LOOKS.items()})


@pytest.mark.parametrize("look", sorted(LOOKS))
def test_every_look_keeps_black_and_colour(look, looks):
    assert looks[look]["nonfinite"] == 0
    _check_projection_frame(looks[look]["path"])          # includes "frozen": held from frame 0, still a full picture
    if "flat" in look and "invert" not in look:
        assert looks[look]["limiter"] == 0, "a full-frame flat field must not be pinned by the limiter"


def test_flash_and_blackout(tmp_path):
    res = _batch(tmp_path, {"flash": dict(sets=dict(flash=1.0), frames=60),
                            "blackout": dict(sets=dict(blackout=1.0), frames=60),
                            "blackout 100ms": dict(sets={}, frames=66, script={"60": [["trigger", "blackout", True]]}),
                            "master 0": dict(sets=dict(master=0.0), frames=30)})
    assert all(r["nonfinite"] == 0 for r in res.values())
    assert _stats(res["flash"]["path"])["blown"] < 0.10                   # a bloom, not a white rectangle
    assert _stats(res["flash"]["path"])["mean"] > _stats(res["master 0"]["path"])["mean"] + 20
    for name in ("blackout", "master 0"):
        assert _stats(res[name]["path"])["mean"] < 0.5, name
    assert _stats(res["blackout 100ms"]["path"])["mean"] < 2.5, "blackout must be (nearly) there after 100 ms"


def test_layer_presets_cover_every_layer_fader():
    assert len(lv.MODE_W) == len(lv.MODES) and len(set(lv.MODES)) == len(lv.MODES)
    assert all(k in lv.PARAMS for k in lv.PRESET_KEYS)
    for w in lv.MODE_W:
        assert len(w) == len(lv.PRESET_KEYS) and all(0 <= x <= 1 for x in w) and max(w) > 0


# ---------------------------------------------------------------- params contract

def test_every_param_maps_to_a_shader_uniform_or_is_cpu_only():
    code = re.sub(r"//[^\n]*", "", lv.SCENE_FS)             # comments are not code
    decl = " ".join(re.findall(r"uniform\s+float\s+([^;]*);", code))
    for name in lv.PARAMS:
        if name not in lv.CPU_ONLY:
            assert re.search(rf"\bp_{name}\b", decl), f"p_{name} is not declared"
            assert len(re.findall(rf"\bp_{name}\b", code)) >= 2, f"p_{name} is declared but never used"
    for name in lv.PARAMS:
        if name in lv.CPU_ONLY:
            assert not re.search(rf"\bp_{name}\b", lv.SCENE_FS), f"{name} is listed CPU-only but the shader declares it"
        else:
            assert re.search(rf"\bp_{name}\b", lv.SCENE_FS), f"params.py '{name}' has no p_{name} uniform in the shader"
    assert lv.CPU_ONLY <= set(lv.PARAMS), "CPU_ONLY lists params that no longer exist"
    for clock in lv.CLOCK_UNIFORMS:
        assert len(re.findall(rf"\b{clock}\b", code)) >= 2, f"{clock} is declared but never used"


def test_static_controller_is_the_fallback():
    assert isinstance(lv.make_controller(headless=True), lv.StaticController)


def test_real_controller_drives_the_engine_when_present(tmp_path):
    if lv.Controller is None:
        pytest.skip("control.py not present")
    state = ROOT / "live_state.json"
    before = state.read_bytes() if state.exists() else None
    stdout, out = _run(tmp_path, "--live-controller", frames=60)
    assert "[control] Controller" in stdout
    assert _stats(out / "frame_0060.png")["blown"] < 0.01
    assert (state.read_bytes() if state.exists() else None) == before, "a headless run must not touch the show's live state"


def test_no_nan_at_param_extremes(tmp_path):
    jobs = {}
    for kind in ("min", "max", "rand1", "rand2", "rand3", "rand4", "rand5", "rand6"):
        ex = lv.extreme_values(kind)
        ripples = {"3": [["ripple", 0.5, 0.5], ["ripple", 0.0, 1.0]]}
        jobs[kind] = dict(sets=ex, frames=40, script=ripples)
        # the same extremes but visible and moving ("max" alone is frozen and blacked out)
        jobs[kind + " live"] = dict(sets=dict(ex, blackout=0, freeze=0, master=1, speed=2), frames=40, script=ripples)
    res = _batch(tmp_path, jobs, "--no-nan-guard")       # raw shader output: the NaN zeroing is switched off
    bad = {n: r["nonfinite"] for n, r in res.items() if r["nonfinite"]}
    assert not bad, f"NaN/Inf in the displayed frame or the echo buffer: {bad}"
    for n, r in res.items():
        assert _stats(r["path"])["blown"] < 0.05 or jobs[n]["sets"].get("flash") or jobs[n]["sets"].get("strobe"), n   # light pulses are exempt


def test_travel_is_stable_for_ten_seconds(tmp_path):
    # travel +-1 in every mode for 600 frames (10 s); symmetry 0.5 runs both topologies at once
    jobs = {f"mode{m} travel{tr:+d}": dict(sets=dict(travel=tr, travel_mode=m, symmetry=0.5, tunnel_curve=1), frames=600)
            for m in range(4) for tr in (-1, 1)}
    res = _batch(tmp_path, jobs, "--render", "640x400", "--no-nan-guard")
    for n, r in res.items():
        s = _stats(r["path"])
        assert r["nonfinite"] == 0, n
        assert s["blown"] < 0.01 and 2 < s["mean"] < 80, (n, s)


def test_palette_value_is_fractional_and_wraps():
    P = [np.array(p[1]) for p in lv.PALETTES]
    assert np.allclose(lv.palette_stops(0), P[0]) and np.allclose(lv.palette_stops(4), P[4])
    assert np.allclose(lv.palette_stops(1.5), (P[1] + P[2]) / 2, atol=1e-6)
    assert np.allclose(lv.palette_stops(4.5), (P[4] + P[0]) / 2, atol=1e-6)      # wraps 4 -> 0
    assert lv.palette_stops(2.25).shape == (5, 3)


# ---------------------------------------------------------------- remote token, snapshot, video

def test_remote_token_is_persistent_single_line_and_never_raises(tmp_path):
    f = tmp_path / ".remote_token"
    assert lv.load_remote_token(f, "given-token") == "given-token" and not f.exists()
    tok = lv.load_remote_token(f)
    assert tok and f.read_text() == tok + "\n"
    assert lv.load_remote_token(f) == tok                                # stable across restarts
    f.write_text("  hand_edited-1  \n")
    assert lv.load_remote_token(f) == "hand_edited-1"
    f.write_text("bad token with spaces\nand lines")
    fresh = lv.load_remote_token(f)
    assert fresh and " " not in fresh and f.read_text() == fresh + "\n"  # unusable file is replaced
    assert lv.load_remote_token(tmp_path / "no" / "such" / "dir" / "t")   # unwritable location: still a token


def test_snapshot_is_written_atomically_and_downscaled(tmp_path):
    _, out = _run(tmp_path, "--snapshot", frames=40)
    snap = out / "snapshots" / "current.png"
    with Image.open(snap) as im:
        assert im.size == (768, 480)
        assert np.asarray(im).mean() > 3
    assert not list(snap.parent.glob("*.tmp*"))
    assert lv.save_snapshot(tmp_path / "x" / "y.png", b"too short", (1280, 800)) is False   # never raises


def test_video_decoder_loops_and_survives_garbage(tmp_path):
    clip = tmp_path / "clip.mp4"
    assert lv.make_test_clip(clip, frames=12)
    d = lv._Decoder(clip, (64, 40))
    try:
        seen = 0
        for _ in range(40):                                              # 12-frame clip: must loop to deliver 40
            buf, k = d.take(1, wait=5.0)
            assert buf is not None and buf.shape == (64 * 40 * 3,)
            seen += k
        assert seen == 40 and not d.failed and 1 <= d.fps <= 240
        assert buf.max() > 40                                            # real picture, not zeros
    finally:
        d.stop()
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"this is not a video" * 50)
    d2 = lv._Decoder(bad, (64, 40))
    d2.join(timeout=30)
    assert d2.failed and d2.take(1)[0] is None
    d3 = lv._Decoder(tmp_path / "missing.mp4", (64, 40))
    d3.join(timeout=30)
    assert d3.failed


def test_video_layer_renders_through_the_engine(tmp_path):
    vids = tmp_path / "videos"
    assert lv.make_test_clip(vids / "clip.mp4")
    (vids / "corrupt.mp4").write_bytes(b"nope")
    base = ["--videos", str(vids), "--mode", "kene", *_sets(l_kene=0, l_video=1, luma_key=0.1, symmetry=0, chase=0)]
    _, out = _run(tmp_path, *base, frames=60)
    assert _stats(out / "frame_0060.png")["mean"] > 3, "video layer shows nothing"
    _, out = _run(tmp_path, "--videos", str(tmp_path / "empty"), *base[2:], frames=20)     # no folder: just black
    assert _stats(out / "frame_0020.png")["mean"] < 3


# ---------------------------------------------------------------- show safety (review fixes)

def _img(path):
    return np.asarray(Image.open(path)).astype(float)


def test_freeze_holds_the_exact_picture(tmp_path):
    hold = {"40": [["trigger", "freeze", True]]}
    res = _batch(tmp_path, {"frozen +20": dict(sets={}, frames=60, script=hold),
                            "frozen +80": dict(sets={}, frames=120, script=hold),
                            "running +80": dict(sets={}, frames=120)})
    a, b, c = (_img(res[n]["path"]) for n in ("frozen +20", "frozen +80", "running +80"))
    _check_projection_frame(res["frozen +80"]["path"])                   # frozen is a picture, not black
    assert np.abs(a - b).mean() < 0.6, "the frame keeps changing while frozen (audio, clock or echo leaking through)"
    assert np.abs(c - b).mean() > 3.0, "sanity: without freeze the picture moves on"


def test_quit_needs_confirmation_on_the_projector(capsys):
    g = lv.QuitGuard(confirm=True, window=1.0)
    assert g.press(10.0) is False                         # a single ESC does not end the show
    assert "again" in capsys.readouterr().out             # hint goes to the console
    assert g.press(10.6) is True                          # second press within 1 s quits
    g = lv.QuitGuard(confirm=True)
    assert g.press(10.0) is False and g.press(11.5) is False and g.press(12.0) is True    # too slow re-arms
    assert lv.QuitGuard(confirm=True).press(5.0, shift=True) is True                      # Shift+ESC quits at once
    assert lv.QuitGuard(confirm=False).press(5.0) is True                                 # preview window: plain ESC


def test_projector_is_recognised_by_name_and_size_across_replug():
    laptop, proj = ("Laptop", 1920, 1080, 0, 0), ("BenQ MW632ST", 1280, 800, 1920, 0)
    assert lv.pick_monitor([laptop, proj], 1) == 1
    assert lv.pick_monitor([laptop], 1) is None                           # not plugged in at start: wait, no preview
    remembered = proj[:3]
    assert lv.pick_monitor([laptop], 1, remembered) is None               # unplugged mid-show
    assert lv.pick_monitor([proj, laptop], 1, remembered) == 0            # back, but Windows reordered the displays
    assert lv.pick_monitor([laptop, ("Generic PnP", 1280, 800, 1920, 0)], 1, remembered) == 1   # renamed by the driver
    assert lv.pick_monitor([laptop, ("Other", 3840, 2160, 1920, 0)], 1, remembered) is None     # a different screen
    assert lv.pick_monitor([], 0) is None and lv.pick_monitor([laptop], -1) is None


def test_limiter_converges_without_oscillation_and_spares_flat_fields():
    lim = lv.Limiter(mean=0.36, black=0.15)
    hist = []
    for i in range(600):                                  # constant bright look, metered every 4th frame
        if i % 4 == 0:
            lim.measure(0.60, 0.40, symmetry=1.0)
        hist.append(lim.step(1 / 60))
    assert abs(hist[-1] - 2 * (1 - 0.36 / 0.60)) < 1e-3
    assert max(hist[300:]) - min(hist[300:]) < 0.01       # settled within 5 s, then flat
    assert all(b >= a - 1e-9 for a, b in zip(hist, hist[1:])), "monotone approach, no sawtooth"
    for _ in range(600):                                  # jitter inside the deadband does not move the target
        lim.measure(0.60 + 0.004 * np.sin(_), 0.40, 1.0)
        lim.step(1 / 60)
    assert abs(lim.value - hist[-1]) < 1e-3
    up = lv.Limiter(); up.measure(0.72, 0.5); t_up = next(i for i in range(999) if up.step(1 / 60) > 0.5 * up.target)
    up.measure(0.1, 0.5); v0 = up.value; t_dn = next(i for i in range(999) if up.step(1 / 60) < 0.5 * v0)
    assert abs(t_up - t_dn) <= 2, "attack and release use the same time constant"
    flat = lv.Limiter()
    assert flat.measure(0.20, 0.02, symmetry=0.0) == 0.0  # full-frame flat field: little black is legitimate
    assert flat.measure(0.20, 0.02, symmetry=1.0) > 0.5   # a mandala that lost its black is pulled back
    assert lv.Limiter().measure(0.10, 0.60) == 0.0        # ordinary looks are left alone


def _limiter_line(stdout):
    m = re.search(r"limiter=([0-9.]+) limiter_span=([0-9.]+)", stdout)
    return float(m.group(1)), float(m.group(2))


def test_limiter_gain_is_steady_on_a_constant_look(tmp_path):
    # a frozen picture with the ceiling set low enough to engage: 15 s, the last 10 s are judged
    still = _sets(freeze=1)                              # freeze = a perfectly constant picture
    out, frames = _run(tmp_path, "--test-audio", "silent", "--limit-mean", "0.02", *still, frames=900)
    value, span = _limiter_line(out)
    assert value > 0.05, "the limiter never engaged"
    assert span <= 0.01 * max(value, 1e-6) + 1e-4, f"limiter gain oscillates: {value} +- {span}"
    limited = _stats(frames / "frame_0900.png")["mean"]
    out2, frames2 = _run(tmp_path / "free", "--test-audio", "silent", "--limit-mean", "1", *still, frames=120)
    assert _limiter_line(out2)[0] == 0.0
    assert limited < _stats(frames2 / "frame_0120.png")["mean"] * 0.9, "the limiter did not dim the frame"
    out3, _ = _run(tmp_path / "default", frames=300)      # the default look with a kick never touches it
    assert _limiter_line(out3) == (0.0, 0.0)


def test_phases_stay_small_after_hours_and_wrap_to_the_same_values():
    ph = lv.Phases()
    vals = dict(lv.defaults(), travel=1.0, serpent_speed=1.0, flow=1.0, speed=2.0, rotate=1.0, punch=1.0)
    bands = np.array([1.0, 1.0, 1.0])
    for _ in range(144000):                               # 4 h of show time in 0.1 s steps at full tilt
        ph.advance(0.1, vals, bands)
    assert ph.travel > 1e4 and ph.swim > 1e4              # the raw (double) phases really are large by now
    u = ph.uniforms(vals)
    big = {k: float(np.abs(np.asarray(v)).max()) for k, v in u.items() if k != "t"}
    assert max(big.values()) <= 1600.0, f"a phase handed to the GPU grew with show length: {big}"
    assert u["trZ"] < 256 and u["flowM"] < 320 * np.pi / 8 + 1e-9 and (u["serpB"] < 2 * np.pi + 1e-6).all()
    k = ph.uniforms(vals, 3)                              # +3 whole periods
    assert np.isclose(k["flowF"] - u["flowF"], 3 * 320 * np.pi / 8) and np.allclose(k["swN"] - u["swN"], 4800, atol=1e-2)
    frozen = lv.Phases()
    frozen.advance(1.0, dict(vals, freeze=1.0), bands)
    assert frozen.T == 0 and frozen.travel == 0 and frozen.act == 0 and frozen.tex == 0   # freeze stops everything


PERIODIC = {
    "serpents + warp": _preset("kene", serpent=1.0, serpent_mode=2, serpent_count=8, serpent_speed=1.0),
    "swim wave flat": _preset("all", symmetry=0, swim=1.0, wave=1.0, flow=1.0),
    "swim mandala": _preset("all", swim=1.0, flow=-1.0),
    **{f"travel {m} sym {sy}": _preset("all", travel=1.0, travel_mode=m, symmetry=sy, tunnel_curve=1.0, flow=0.6)
       for m in range(4) for sy in (0, 1)},
}


def test_wrapped_phases_are_seamless(tmp_path):
    # The CPU wraps every long-running phase at a period where the pattern is supposed to repeat. Adding 12 whole
    # periods on the GPU side (hours of crawl / travel / slither) must therefore give the same picture.
    jobs = {n: dict(sets={p: v for p, v in s.items() if p in lv.PARAMS}, frames=75) for n, s in PERIODIC.items()}
    base = _batch(tmp_path / "k0", jobs)
    far = _batch(tmp_path / "k12", jobs, "--phase-offset", "12")
    for n in jobs:
        a, b = _img(base[n]["path"]), _img(far[n]["path"])
        assert a.mean() > 4, n
        diff = np.abs(a - b)
        assert diff.mean() < 1.0 and (diff.max(axis=2) > 40).mean() < 0.01, f"{n}: wrap is visible (mean diff {diff.mean():.2f})"


def test_audio_bands_are_always_finite():
    assert np.array_equal(lv.clean_bands([np.nan, np.inf, -np.inf]), [0.0, 1.0, 0.0])
    assert np.array_equal(lv.clean_bands(np.array([0.5, 2.0, -1.0])), [0.5, 1.0, 0.0])
    assert lv.clean_bands([0.3]).shape == (3,) and lv.clean_bands(np.zeros(5)).shape == (3,)
    a = lv.Audio("none")
    a.ok, a.raw = True, np.array([np.nan, 0.5, np.inf])
    assert np.isfinite(a.get(0.0)).all()


def test_guarded_steps_never_raise_and_log_sparingly(capsys, monkeypatch):
    monkeypatch.setattr(lv, "STRICT", False)
    now = [0.0]
    monkeypatch.setattr(lv, "LOG", lv.RateLog(every=5.0, clock=lambda: now[0]))

    def boom():
        raise RuntimeError("midi exploded")

    for _ in range(300):                                  # 5 s of failing every frame
        assert lv.guarded("control", boom, default="fallback") == "fallback"
        now[0] += 1 / 60
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "midi exploded" in ln]
    assert 1 <= len(lines) <= 2, lines
    assert lv.guarded("control", lambda x: x + 1, 2) == 3
    monkeypatch.setattr(lv, "STRICT", True)               # headless / tests: failures surface
    with pytest.raises(RuntimeError):
        lv.guarded("control", boom)


def test_texture_library_is_capped_to_the_most_recent_and_keeps_what_is_showing():
    files = [(f"tex_{i:03d}.png", 1000.0 + i) for i in range(100)]
    got = lv.select_recent(files, cap=64, keep=("tex_003.png", "tex_090.png"))
    assert len(got) == 65 and got == sorted(got)
    assert "tex_003.png" in got and "tex_035.png" not in got and "tex_036.png" in got and "tex_099.png" in got
    assert lv.select_recent(files[:10], cap=64) == [n for n, _ in files[:10]]
    assert lv.select_recent([], cap=64) == []


def test_bad_textures_are_retried_only_when_rewritten(tmp_path):
    import os
    import time
    good, broken = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(good)
    broken.write_bytes(b"not a png")
    old = time.time() - 60
    for f in (good, broken):
        os.utime(f, (old, old))
    bad = {"b.png": broken.stat().st_mtime}
    assert lv.wanted_textures(tmp_path, bad) == ["a.png"]                  # known-bad file is left alone
    os.utime(broken, (old + 10, old + 10))                                 # ...until it is rewritten
    assert lv.wanted_textures(tmp_path, bad) == ["a.png", "b.png"]
    assert lv.wanted_textures(tmp_path / "missing", bad) == []


def test_concurrent_snapshots_use_unique_temp_files(tmp_path):
    import threading
    data, out = bytes(64 * 40 * 3), tmp_path / "snap" / "current.png"
    seen, real_replace = [], lv.os.replace

    def spy(src, dst):
        seen.append(str(src))
        real_replace(src, dst)

    lv.os.replace = spy
    try:
        ts = [threading.Thread(target=lv.save_snapshot, args=(out, data, (64, 40))) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    finally:
        lv.os.replace = real_replace
    assert len(seen) == 8 and len(set(seen)) == 8, "snapshot temp names collide"
    assert out.exists() and not list(out.parent.glob("*.tmp*"))


def test_palettes_are_five_saturated_stops():
    assert len(lv.PALETTES) == 5
    for name, stops in lv.PALETTES:
        assert len(stops) == 5, name
        for c in stops:
            assert len(c) == 3 and 0 <= min(c) and max(c) <= 1
            assert max(c) >= 0.6 and max(c) - min(c) >= 0.5, f"{name}: {c} is not a jewel tone"


def test_gallery_audio_has_kick_section_then_breakdown():
    kick = np.array([lv.gallery_audio(t) for t in np.linspace(0, 3.9, 400)])
    brk = np.array([lv.gallery_audio(t) for t in np.linspace(4.0, 8, 400)])
    assert kick[:, 0].max() > 0.9 and brk[:, 0].max() == 0
    assert (kick >= 0).all() and (kick <= 1).all() and (brk >= 0).all() and (brk <= 1).all()


# ---------------------------------------------------------------- corner-pin robustness

def test_load_corners_falls_back_on_bad_mapping(tmp_path):
    unit = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    f = tmp_path / "mapping.json"
    assert lv.load_corners(f) == unit                                   # missing file
    good = [[0.1, 0.05], [0.92, 0.0], [1.0, 0.97], [0.0, 0.9]]
    f.write_text(__import__("json").dumps(good))
    assert lv.load_corners(f) == good
    for bad in ("[[0,0],[1,0],[1,1]]",                                  # too few points
                "[[0,0],[1,0],[0,0],[1,0]]",                            # collapsed
                "[[0,0],[1,1],[1,0],[0,1]]",                            # bow-tie
                "[[0,0],[1,0],[1,1],[0]]", "[[0,0],[1,0],[1,1],[0,\"x\"]]", "{\"a\":1}", "not json", "[[0,0],[1,0],[1,1],[0,NaN]]"):
        f.write_text(bad)
        assert lv.load_corners(f) == unit, bad


def test_inverse_homography_never_raises_and_keeps_last_good():
    good = [[0.1, 0.05], [0.92, 0.0], [1.0, 0.97], [0.0, 0.9]]
    Hi = lv.inverse_homography(good)
    assert np.allclose(Hi @ lv.homography(UNIT, good), np.eye(3), atol=1e-6)
    collapsed = [[0.5, 0.5]] * 4
    out = lv.inverse_homography(collapsed, Hi)
    assert out is Hi                                                   # degenerate -> last good, not garbage
    assert np.isfinite(lv.inverse_homography(collapsed)).all()
    assert lv.inverse_homography([[0, 0]], Hi) is Hi                     # wrong length -> last good
    assert not lv.corners_ok(collapsed) and lv.corners_ok(good)
