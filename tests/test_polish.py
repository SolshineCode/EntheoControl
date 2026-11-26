"""Visual polish pass: serpents stay on screen, the ground-plane horizon is lit, the art layer limits its
combined distortion, the factory scenes are projection-safe."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import live_visuals as lv  # noqa: E402

ASPECT = 1.6                                             # 1280x800


def _batch(out, jobs, *args, timeout=600):
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "jobs.json"
    spec.write_text(json.dumps([dict(name=n, **j) for n, j in jobs.items()]))
    r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--batch", str(spec), "--out", str(out), *map(str, args)],
                       capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    done = {d["name"]: d for d in json.loads((out / "batch.json").read_text())}
    return {n: dict(d, path=out / d["file"]) for n, d in done.items()}


def _img(path):
    return np.asarray(Image.open(path)).astype(np.float32)


# ---------------------------------------------------------------- serpents

def _serpent_run(seconds=400, dt=0.1, **over):
    """Head positions over a run: array (steps, 8, 2) in content units (x +-aspect/2, y +-0.5)."""
    vals = dict(lv.defaults(), serpent=1.0, **over)
    ph = lv.Phases()
    out = []
    for _ in range(int(seconds / dt)):
        ph.advance(dt, vals, np.zeros(3))
        out.append(ph.serpent_heads(vals, ASPECT))
    return np.array(out)


def _in_frame(heads):
    return (np.abs(heads[..., 0]) <= ASPECT / 2) & (np.abs(heads[..., 1]) <= 0.5)


@pytest.mark.parametrize("over", [
    dict(),                                              # defaults: size 0.4, wiggle 0.5, speed 0.3
    dict(serpent_size=0.2, serpent_speed=1.0),
    dict(serpent_size=0.9, serpent_segments=40),
    dict(serpent_wiggle=1.0, serpent_speed=0.6),
    dict(serpent_speed=-0.5),                            # backwards
])
def test_serpent_heads_stay_in_frame_most_of_the_time(over):
    inside = _in_frame(_serpent_run(**over))
    per = inside.mean(axis=0)
    assert per.mean() >= 0.78, f"heads are on screen only {per.mean():.0%} of the time: {np.round(per, 2)}"
    assert per.min() >= 0.72, f"one serpent wanders off for long stretches: {np.round(per, 2)}"


def test_serpent_heads_never_jump_while_on_screen():
    heads = _serpent_run(seconds=120, dt=1 / 60, serpent_speed=1.0)
    inside = _in_frame(heads)
    step = np.linalg.norm(np.diff(heads, axis=0), axis=2)
    both = inside[1:] & inside[:-1]
    assert step[both].max() < 0.08, f"a visible head teleported {step[both].max():.3f} units in one frame"
    # a head that is on screen in one frame is never somewhere else entirely in the next one
    assert step[inside[:-1]].max() < 0.08


def test_serpent_uniforms_are_bounded_and_survive_hours():
    vals = dict(lv.defaults(), serpent=1.0, serpent_speed=1.0, speed=2.0)
    ph = lv.Phases()
    for _ in range(144000):                              # 4 h of show time
        ph.advance(0.1, vals, np.zeros(3))
    u = ph.uniforms(vals)
    for k in ("serpA", "serpB", "serpC"):
        assert u[k].shape == (8, 4) and np.isfinite(u[k]).all() and np.abs(u[k]).max() < 16.0, k


def test_serpents_are_visible_at_any_moment(tmp_path):
    # three serpents on black, sampled across 20 s: there is always a clearly visible amount of serpent on screen
    sets = dict(l_kene=0, l_beads=0, l_lines=0, l_ai=0, l_echo=0, symmetry=0, serpent=1.0, serpent_speed=0.6)
    jobs = {f"t{f}": dict(sets=sets, frames=f) for f in (60, 240, 420, 600, 780, 960, 1200)}
    res = _batch(tmp_path, jobs, "--render", "640x400")
    for n, r in res.items():
        lit = (_img(r["path"]).max(axis=2) > 40).mean()
        assert lit > 0.012, f"{n}: serpents are (nearly) off screen, lit {lit:.2%}"


# ---------------------------------------------------------------- art layer: combined distortion limiter

def test_art_distortion_limiter_leaves_single_effects_alone_and_tames_the_pile_up():
    import art_layer as al
    zero = dict(art_melt=0.0, art_liquify=0.0, art_slices=0.0, art_smear=0.0, art_fold=0.0, art_droste=0.0)
    for name in ("art_melt", "art_liquify", "art_slices", "art_smear"):
        for other in ({}, dict(art_fold=1.0), dict(art_droste=1.0)):          # one effect, even over a fold, is untouched
            out = al.limit_distortion(dict(zero, **{name: 1.0}, **other))
            assert out[name] == pytest.approx(1.0), (name, other)
    two = al.limit_distortion(dict(zero, art_melt=0.7, art_liquify=0.7))
    assert two["art_melt"] == pytest.approx(0.7) and two["art_liquify"] == pytest.approx(0.7)
    full = al.limit_distortion({k: 1.0 for k in zero})
    disp = [full[k] for k in ("art_melt", "art_liquify", "art_slices", "art_smear")]
    assert sum(disp) <= 0.85 and min(disp) > 0.1, disp                        # tamed, but every effect still plays
    assert full["art_fold"] == 1.0 and full["art_droste"] == 1.0              # the structural remaps are not scaled
    mixed = al.limit_distortion(dict(zero, art_melt=1.0, art_liquify=0.5, art_slices=1.0, art_smear=0.5))
    assert mixed["art_melt"] / mixed["art_liquify"] == pytest.approx(2.0)     # proportions survive
    prev = None                                                              # no jump while one fader sweeps up
    for x in np.linspace(0, 1, 101):
        out = al.limit_distortion(dict(zero, art_melt=0.8, art_liquify=0.8, art_smear=float(x), art_fold=0.5))
        vec = np.array([out[k] for k in sorted(zero)])
        assert prev is None or np.abs(vec - prev).max() < 0.03
        prev = vec
    assert al.limit_distortion({}) == {k: 0.0 for k in zero}                  # missing / bad values are zeros
    assert al.limit_distortion(dict(art_melt=float("nan"), art_smear=9.0))["art_smear"] == pytest.approx(1.0)


# ---------------------------------------------------------------- ground plane horizon + factory scenes

def test_ground_plane_horizon_is_lit_not_a_dead_band(tmp_path):
    sets = dict(l_kene=1, l_beads=0, l_lines=0, l_ai=0, symmetry=0, travel=0.7, travel_mode=1, rotate=0, tunnel_curve=0)
    res = _batch(tmp_path, {"plane": dict(sets=sets, frames=90), "plane fog 1": dict(sets=dict(sets, depth_fog=1.0), frames=90)})
    for n, r in res.items():
        a = _img(r["path"]).max(axis=2)
        h, w = a.shape
        mid = a[int(h * 0.40):int(h * 0.60), int(w * 0.35):int(w * 0.65)]   # the horizon sways a few degrees: take a band
        assert mid.max(axis=0).mean() > 110, f"{n}: no glow on the horizon ({mid.max(axis=0).mean():.0f})"
        assert (mid > 8).mean() > 0.45, f"{n}: dead band at the horizon, only {(mid > 8).mean():.0%} lit"
        assert (a < 8).mean() > 0.25 and (_img(r["path"]).min(axis=2) > 235).mean() < 0.01, n   # still black around it, no blowout


def test_factory_scenes_are_projection_safe_and_not_near_black(tmp_path):
    import control
    scenes = [s for s in control.factory_scenes() if s is not None]
    assert len(scenes) == 8
    jobs = {f"{i} {s['name']}": dict(sets={k: v for k, v in s["base"].items() if k in lv.PARAMS}, frames=120)
            for i, s in enumerate(scenes)}
    res = _batch(tmp_path, jobs)
    for n, r in res.items():
        a = _img(r["path"])
        mx, mn = a.max(axis=2), a.min(axis=2)
        assert r["nonfinite"] == 0, n
        assert (mx < 8).mean() > 0.15, f"{n}: too little true black"
        assert (mn > 235).mean() < 0.01, f"{n}: blown out"
        assert 4.0 < a.mean() < 70.0, f"{n}: too dim / too bright, mean {a.mean():.1f}"
        assert (mx > 40).mean() > 0.04, f"{n}: too little on screen"


# ---------------------------------------------------------------- art_use modes, spin, widened faders

def test_art_use_modes_each_look_different(tmp_path):
    # AI texture as the source (no uploads needed): every mode must be a clearly different picture
    names = ("relief", "stencil", "kaleido", "tunnel", "displace", "melt")
    jobs = {n: dict(sets=dict(l_art=1.0, art_use=i, art_source=1.0), frames=120) for i, n in enumerate(names)}
    jobs["off"] = dict(sets={}, frames=120)
    res = _batch(tmp_path, jobs, "--render", "640x400")
    imgs = {n: _img(r["path"]) for n, r in res.items()}
    assert all(r["nonfinite"] == 0 for r in res.values())
    for a in imgs:
        for b in imgs:
            if a < b:
                d = np.abs(imgs[a] - imgs[b]).mean() / 255 * 100
                assert d > 2.0, f"art_use '{a}' and '{b}' look the same ({d:.2f} % apart)"
        assert (imgs[a].min(axis=2) > 235).mean() < 0.02, f"{a}: blown out"


def test_art_use_params_are_appended_and_stepped():
    import control
    names = list(lv.PARAMS)
    for n in ("art_use", "art_source", "spin", "spin_twist"):
        assert n in names and names.index(n) > names.index("art_glow"), n     # appended, nothing reordered
    assert lv.PARAMS["art_use"][:3] == (0.0, 5.0, 0.0) and lv.PARAMS["spin"][:3] == (-1.0, 1.0, 0.0)
    assert len(control.OPTIONS["art_use"]) == 6 and len(control.OPTIONS["art_source"]) == 2
    assert "art_use" in control.DISCRETE and "art_source" in control.DISCRETE
    assert control.CUE_CC[66] == "art_glow" and control.CUE_CC[1] == "master"   # the frozen CC table did not move


def test_spin_phase_accumulates_wraps_and_stacks_with_rotate():
    ph = lv.Phases()
    vals = dict(lv.defaults(), spin=0.3, rotate=0.0)
    for _ in range(100):
        ph.advance(0.01, vals, np.zeros(3))
    assert ph.spin == pytest.approx(0.3 * lv.SPIN_RATE, rel=1e-6) and ph.rot == 0.0   # 1 s at 0.3: 27 degrees, visible
    assert ph.spin > 0.4
    u = ph.uniforms(vals)
    assert u["spinPh"] == pytest.approx(ph.spin) and u["spinV"] == pytest.approx(0.3)
    for _ in range(36000):                                                  # an hour at full tilt, backwards
        ph.advance(0.1, dict(vals, spin=-1.0, rotate=1.0), np.zeros(3))
    assert 0.0 <= ph.spin < 2 * np.pi and 0.0 <= ph.rot < 2 * np.pi
    frozen = lv.Phases()
    frozen.advance(1.0, dict(vals, freeze=1.0), np.zeros(3))
    assert frozen.spin == 0.0


def test_spin_turns_the_picture_in_every_travel_mode(tmp_path):
    jobs = {}
    for m in range(4):
        for sy in (0, 1):
            base = dict(travel=0.6, travel_mode=m, symmetry=sy, rotate=0.0, folds=3.0)
            jobs[f"m{m} s{sy} still"] = dict(sets=base, frames=60)
            jobs[f"m{m} s{sy} spin"] = dict(sets=dict(base, spin=0.3), frames=60)
            jobs[f"m{m} s{sy} twist"] = dict(sets=dict(base, spin=0.3, spin_twist=1.0), frames=60)
    res = _batch(tmp_path, jobs, "--render", "640x400")
    for m in range(4):
        for sy in (0, 1):
            still, spin, twist = (_img(res[f"m{m} s{sy} {k}"]["path"]) for k in ("still", "spin", "twist"))
            assert np.abs(spin - still).mean() / 255 * 100 > 2.0, f"mode {m} symmetry {sy}: spin 0.3 is not visible"
            assert np.abs(twist - spin).mean() / 255 * 100 > 1.0, f"mode {m} symmetry {sy}: spin twist is not visible"


def test_widened_faders_change_the_picture_and_keep_their_default(tmp_path):
    # the top of these faders used to do next to nothing; the default position must stay exactly the old look
    code = lv.SCENE_FS
    for needle in ("max(p_l_ai-0.5, 0.0)", "max(p_ripple-0.5, 0.0)", "max(p_trails-0.94, 0.0)", "max(p_l_echo-0.6, 0.0)"):
        assert needle in code, needle                         # extra gain only above the default value
    d = lv.defaults()
    assert (d["l_ai"], d["ripple"], d["trails"], d["l_echo"]) == (0.5, 0.5, 0.85, 0.6)
    move = dict(rotate=0.6)
    jobs = {"base": dict(sets=move, frames=120), "ai 1": dict(sets=dict(move, l_ai=1.0), frames=120),
            "echo 1": dict(sets=dict(move, l_echo=1.0), frames=120), "trails max": dict(sets=dict(move, trails=0.98), frames=120)}
    res = _batch(tmp_path, jobs, "--render", "640x400")
    base = _img(res["base"]["path"])
    for n in ("ai 1", "echo 1", "trails max"):
        a = _img(res[n]["path"])
        assert np.abs(a - base).mean() / 255 * 100 > 1.0, f"{n}: still barely visible"
        assert (a.max(axis=2) < 8).mean() > 0.25 and (a.min(axis=2) > 235).mean() < 0.01, n
