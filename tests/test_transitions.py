"""Visual scene transitions (transitions.py + the controller / engine hooks).

CPU tests: spec cleaning, easing, the mask maths of every transition (coverage 0 -> 1, monotonic), the controller
state machine (start / finish / cancel / PANIC / overlap / governor fallback / dip rate), idle = zero cost.
GL tests (names end in _gl; run them under .locks/gpu_suite): the shader matches the numpy mirror, B is drawn only
during a transition, the echo target never sees B, the engine is pixel-identical while idle.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import control  # noqa: E402
import themes  # noqa: E402
import timeline  # noqa: E402
import transitions as T  # noqa: E402
from params import PARAMS  # noqa: E402

DT = 1 / 60


class FakeTime:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(tmp_path, engine=True, **kw):
    ft = FakeTime()
    kw.setdefault("restore", False)
    c = control.Controller(tmp_path, midi=False, ai=False, time_fn=ft, seed=1, threads=False, **kw)
    c.ft = ft
    c.quantize = "off"
    c.morph_bars = 0.0
    c.trans.engine_ok = engine
    return c


def run(c, seconds=0.1, dt=DT, frames=None):
    f = None
    for _ in range(max(1, int(round(seconds / dt)))):
        c.ft.t += dt
        f = c.update(dt, (0.0, 0.0, 0.0))
        if frames is not None:
            frames.append(c.trans.frame(f.values))
    return f.values


B_LOOK = dict(zoom=0.5, hue=0.4, symmetry=0.0, twirl=0.6, palette=2.0, l_beads=0.9)


def two_scenes(c, b=None, a=None):
    """Slot 0 = the default look (A), slot 1 = B_LOOK (B), slot 2 / 3 = other looks. Returns c (A on air)."""
    base = dict(c.base)
    for i, over in enumerate((a or {}, b or B_LOOK, dict(zoom=-0.6, swim=0.8), dict(rotate=0.7, density=0.9))):
        c.base.update(base)
        c.base.update(over)
        c.mods = {}
        c.scene_save(i)
    c.base.update(base)
    c.base.update(a or {})
    c.mods = {}
    run(c, 0.2)
    return c


def priv(c):
    return [n for n in control.CONT if PARAMS[n][3] in T.PRIVATE_GROUPS]


# ------------------------------------------------------------------------------------------------ specs
def test_clean_spec_kinds_morph_and_bad_input():
    assert T.clean_spec(None) is None and T.clean_spec("morph") is None and T.clean_spec({"kind": "morph"}) is None
    for k in T.KINDS:
        s = T.clean_spec(k)
        assert s["kind"] == k and 0.25 <= s["beats"] <= 64 and s["ease"] in T.EASES
        assert json.loads(json.dumps(s)) == s
    with pytest.raises(ValueError):
        T.clean_spec("teleport")
    with pytest.raises(ValueError):
        T.clean_spec(42)
    s = T.clean_spec({"kind": "doors", "beats": 999, "glow": 7, "soft": -1, "n": 3, "mode": "toward", "junk": 1,
                      "origin": [2, -1], "color": [2, 0.5, "x"], "scale": 0.01, "quantize": "bar"})
    assert s["beats"] == 64 and s["glow"] == 1 and s["soft"] == 0 and s["n"] == 4 and s["mode"] == "toward"
    assert s["origin"] == [1.0, 0.0] and s["color"] == [1.0, 0.5, 1.0] and s["scale"] == 0.25 and "junk" not in s
    assert s["quantize"] == "bar"
    assert T.clean_spec({"kind": "iris", "mode": 2})["mode"] == "star"
    assert T.clean_spec({"kind": "iris", "mode": "hexagon"})["mode"] == "circle"
    assert "secs" in T.clean_spec({"kind": "wipe", "secs": 1.5}) and "beats" not in T.clean_spec({"kind": "wipe", "secs": 1})


def test_photosensitive_duration_floors():
    fast = {"beats": 0.25}
    assert T.seconds(T.clean_spec(dict(fast, kind="wipe")), 180) == T.MIN_SECS == 0.5
    assert T.seconds(T.clean_spec(dict(fast, kind="portal")), 180) == 1.0
    assert T.seconds(T.clean_spec(dict(fast, kind="portal", mode="white")), 180) == 1.5
    assert T.seconds(T.clean_spec({"kind": "doors", "beats": 4}), 120) == pytest.approx(2.0)
    assert T.seconds(T.clean_spec({"kind": "doors", "secs": 3}), 120) == 3.0


@pytest.mark.parametrize("name", T.EASES)
def test_easing_ends_at_0_and_1_and_never_goes_back(name):
    xs = [T.ease(name, i / 100) for i in range(101)]
    assert xs[0] == pytest.approx(0.0, abs=1e-9) and xs[-1] == pytest.approx(1.0, abs=1e-9)
    assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:]))
    assert T.ease(name, -3) == xs[0] and T.ease(name, 7) == xs[-1]


# ------------------------------------------------------------------------------------------------ mask maths
VARIANTS = [("doors", {}), ("doors", {"n": 1}), ("doors", {"n": 4}), ("doors", {"mode": "toward"}),
            ("doors", {"n": 4, "mode": "toward"}),
            ("slide", {}), ("slide", {"n": 1, "angle": 35}), ("slide", {"n": 4}), ("slide", {"mode": "vertical"}),
            ("curtains", {}),
            ("iris", {}), ("iris", {"mode": "polygon", "n": 5}), ("iris", {"mode": "star", "n": 7}),
            ("iris", {"mode": "kaleido", "n": 8}), ("iris", {"reverse": True}), ("iris", {"origin": [0.15, 0.8]}),
            ("portal", {}), ("portal", {"origin": [0.9, 0.2]}), ("portal", {"mode": "white"}),
            ("shatter", {}), ("shatter", {"mode": "fall", "n": 16}), ("shatter", {"n": 4, "origin": [0, 0]}),
            ("peel", {}), ("peel", {"angle": 210}), ("dissolve", {}), ("dissolve", {"seed": 7.0}),
            ("wipe", {}), ("wipe", {"angle": 123}), ("wipe", {"reverse": True}), ("zoom", {}), ("zoom", {"origin": [0.2, 0.3]})]


@pytest.mark.parametrize("kind,over", VARIANTS, ids=[f"{k}-{'-'.join(map(str, o.values())) or 'default'}" for k, o in VARIANTS])
def test_mask_coverage_goes_from_A_to_B_and_never_back(kind, over):
    spec = T.clean_spec(dict(over, kind=kind))
    cov = [T.coverage(spec, i / 40) for i in range(41)]
    assert cov[0] <= 0.005, cov[0]                    # the first frame is A
    assert cov[-1] >= 0.999, cov[-1]                  # the last frame is B: no leftover sliver of A
    tol = 0.01 if (kind == "shatter" and spec["mode"] == "fall") or (kind == "iris" and spec["mode"] == "kaleido") else 1e-6
    assert all(b >= a - tol for a, b in zip(cov, cov[1:])), cov
    mid = cov[30] if spec.get("mode") == "white" else cov[20]   # the white portal shows B in its second half only
    assert mid > cov[0] + 0.05 and mid < cov[-1] - 0.02           # something visibly happens mid-way


def test_reveal_opens_along_the_lines_first():
    w, h = 160, 100
    u, _ = T._grid(w, h)
    lines = (np.abs(np.sin(u * 40.0)) > 0.92).astype(float)          # thin bright stripes = the current pattern's lines
    blur = np.clip(np.abs(np.sin(u * 40.0)) ** 2, 0, 1)
    spec = T.clean_spec("reveal")
    cov = [float(np.mean(T.mask_ref(spec, i / 40, (w, h), field=(lines, blur)))) for i in range(41)]
    assert cov[0] <= 0.005 and cov[-1] >= 0.999 and all(b >= a - 1e-9 for a, b in zip(cov, cov[1:]))
    m = T.mask_ref(spec, 0.15, (w, h), field=(lines, blur))
    assert m[lines > 0.5].mean() > 0.9 and m[lines < 0.5].mean() < 0.1    # through the lines first, then it widens
    m = T.mask_ref(spec, 0.6, (w, h), field=(lines, blur))
    assert 0.2 < m[lines < 0.5].mean() < 0.99


def test_uniforms_cap_the_glow_and_carry_the_origin():
    U = T.uniforms(T.clean_spec({"kind": "portal", "glow": 1.0}), 0.5, (0.2, 0.7), 1.6)
    assert U["glow"] == pytest.approx(T.GLOW_CAP) and U["org"] == (0.2, 0.7) and U["kind"] == T.KINDS.index("portal")
    U = T.uniforms(T.clean_spec({"kind": "portal", "origin": [0.9, 0.1]}), 0.5, (0.2, 0.7), 1.6)
    assert U["org"] == (0.9, 0.1)                     # the spec's own origin wins over the draggable one


# ------------------------------------------------------------------------------------------------ state machine
def test_idle_costs_nothing_and_changes_nothing(tmp_path):
    a, b = two_scenes(make(tmp_path / "a")), two_scenes(make(tmp_path / "b"))
    b.trans = None                                    # the same controller without the transitions module
    for _ in range(90):
        a.ft.t += DT
        b.ft.t += DT
        va, vb = a.update(DT, (0.3, 0.2, 0.1)).values, b.update(DT, (0.3, 0.2, 0.1)).values
        assert va == vb
        assert a.trans.frame(va) is None
    a.scene_recall(1)                                 # no default transition: a recall is the plain MORPH
    b.scene_recall(1)
    assert a.trans.active is None and run(a, 0.3) == run(b, 0.3)


def test_blend_holds_A_renders_B_then_commits_seamlessly(tmp_path):
    c = two_scenes(make(tmp_path))
    a_vals = run(c, 0.05)
    assert c.apply([{"cmd": "scene_recall", "i": 1, "transition": {"kind": "doors", "secs": 1.0}}]) == []
    assert c.trans.active is not None and c.cur == 1
    frames = []
    v = run(c, 0.5, frames=frames)
    for n in priv(c):                                 # A keeps rendering untouched
        assert v[n] == pytest.approx(a_vals[n], abs=1e-6), n
    fr = frames[-1]
    for n, x in B_LOOK.items():                       # B is rendered with B's values
        assert fr["b"][n] == pytest.approx(x, abs=1e-6), n
    assert 0.3 < fr["t"] < 0.7 and fr["e"] == pytest.approx(T.ease("inout", fr["t"]))
    es = [f["e"] for f in frames]
    assert all(y >= x for x, y in zip(es, es[1:]))
    frames = []
    run(c, 0.6, frames=frames)
    done = [i for i, f in enumerate(frames) if f is None]
    assert done and c.trans.active is None
    last_b, after = frames[done[0] - 1]["b"], run(c, DT)
    for n in priv(c):                                 # the first frame after == the last composite's B
        assert after[n] == pytest.approx(last_b[n], abs=1e-6), n
    assert c.trans.stats["started"] == 1 and c.trans.stats["finished"] == 1


def test_beats_follow_the_tempo_and_quantize_holds_the_spec(tmp_path):
    c = two_scenes(make(tmp_path))
    c.set_bpm(120.0)
    c.quantize = "bar"
    c.scene_recall(1, transition={"kind": "wipe", "beats": 2})
    assert c.pending and c.trans.active is None
    run(c, 2.1)                                       # the bar comes (<= 2 s at 120 BPM): the wipe starts there
    assert c.trans.active is not None and c.trans.active["dur"] == pytest.approx(1.0)
    run(c, 1.1)
    assert c.trans.active is None and c.base["zoom"] == pytest.approx(B_LOOK["zoom"])


def test_cancel_reverts_to_A_and_finish_jumps_to_B(tmp_path):
    c = two_scenes(make(tmp_path))
    a_base = dict(c.base)
    c.scene_recall(1, transition="iris")
    run(c, 0.2)
    c.apply([{"cmd": "transition", "op": "cancel"}])
    assert c.trans.active is None and all(c.base[n] == a_base[n] for n in control.CONT)
    c.scene_recall(1, transition="iris")
    run(c, 0.2)
    c.apply([{"cmd": "transition", "op": "finish"}])
    assert c.trans.active is None and c.base["hue"] == pytest.approx(B_LOOK["hue"])


def test_panic_cancels_instantly_to_the_safe_look(tmp_path):
    c = two_scenes(make(tmp_path))
    c.safe = 2
    c.scene_recall(1, transition={"kind": "portal", "beats": 8})
    run(c, 0.3)
    c.scene_recall(3, transition="doors")             # queued behind it
    assert c.trans.queued is not None
    c.apply([{"cmd": "panic"}])
    assert c.trans.active is None and c.trans.queued is None and c.trans.dip is None
    v = run(c, DT)
    assert c.trans.frame(v) is None and c.cur == 2
    assert v["zoom"] == pytest.approx(-0.6) and v["swim"] == pytest.approx(0.8)
    run(c, 3.0)
    assert c.cur == 2 and c.trans.active is None      # nothing queued comes back after PANIC


def test_overlapping_recalls_queue_the_newest(tmp_path):
    c = two_scenes(make(tmp_path))
    c.scene_recall(1, transition={"kind": "doors", "secs": 1.0})
    run(c, 0.2)
    c.scene_recall(2, transition="wipe")
    c.scene_recall(3, transition={"kind": "dissolve", "secs": 1.0})   # newest wins
    assert c.trans.active["label"] == "scene 2" and c.trans.stats["queued"] == 2
    seen = set()
    for _ in range(150):
        run(c, DT)
        if c.trans.active is not None:
            seen.add(c.trans.active["label"])
    assert seen == {"scene 2", "scene 4"} and c.cur == 3 and c.base["rotate"] == pytest.approx(0.7)
    assert c.trans.active is None


def test_a_plain_morph_recall_hurries_the_running_transition(tmp_path):
    c = two_scenes(make(tmp_path))
    c.scene_recall(1, transition={"kind": "doors", "secs": 8.0})
    run(c, 0.5)
    c.scene_recall(2, transition="morph")
    run(c, T.HURRY_S + 0.05)
    assert c.trans.active is None and c.cur == 2      # ended within HURRY_S, then the morph recall ran


def test_governor_strain_falls_back_to_morph_with_one_dip(tmp_path):
    c = two_scenes(make(tmp_path))
    c.gov_level = 1
    c.scene_recall(1, transition="doors")
    assert c.trans.active is None and c.trans.dip is not None and c.trans.stats["fallbacks"] == 1
    m = []
    for _ in range(int(0.6 / DT)):
        m.append(run(c, DT)["master"])
    assert min(m) < 0.2 and m[-1] == pytest.approx(1.0)
    assert c.base["zoom"] == pytest.approx(B_LOOK["zoom"])   # the MORPH (here a cut: morph_bars 0) went through
    dips, prev_low = 0, False                          # hammering recalls under strain: <= 1 dip per second
    for k in range(int(4.0 / DT)):
        if k % 6 == 0:
            c.scene_recall(1 + (k // 6) % 3, transition="iris")
        low = run(c, DT)["master"] < 0.5
        dips += low and not prev_low
        prev_low = low
    assert 1 <= dips <= 4


def test_critical_strain_mid_transition_finishes_through_a_dip(tmp_path):
    c = two_scenes(make(tmp_path))
    c.scene_recall(1, transition={"kind": "doors", "secs": 4.0})
    run(c, 0.3)
    c.gov_level = 2
    run(c, T.DIP_S * 0.6)
    assert c.trans.active is None and c.base["zoom"] == pytest.approx(B_LOOK["zoom"])
    assert run(c, T.DIP_S)["master"] == pytest.approx(1.0)


def test_no_engine_means_morph(tmp_path):
    c = two_scenes(make(tmp_path, engine=False))
    c.scene_recall(1, transition="doors")
    assert c.trans.active is None and c.trans.dip is None and c.base["zoom"] == pytest.approx(B_LOOK["zoom"])


def test_engine_failure_finishes_at_once(tmp_path):
    c = two_scenes(make(tmp_path))
    c.scene_recall(1, transition="doors")
    run(c, 0.1)
    c.trans.engine_failed()
    assert c.trans.active is None and not c.trans.engine_ok and c.base["zoom"] == pytest.approx(B_LOOK["zoom"])


def test_a_pass_only_B_shows_snaps_and_draws_at_the_max_amount(tmp_path):
    land_b = dict(B_LOOK, l_land=1.0, land_a=3.0, land_twist=0.8)
    c = two_scenes(make(tmp_path), b=land_b, a=dict(l_fractal=0.8, fractal_morph=0.2))
    c.scene_recall(1, transition={"kind": "iris", "secs": 2.0})
    v = run(c, 0.2)
    fr = c.trans.frame(v)
    assert v["l_land"] == 0.0 and fr["b"]["l_land"] == 1.0 and fr["pass_vals"]["l_land"] == 1.0
    assert c.base["land_a"] == 3.0 and c.base["land_twist"] == pytest.approx(0.8)   # B only: at once
    assert fr["pass_vals"]["l_fractal"] == pytest.approx(0.8, abs=0.01) and fr["b"]["l_fractal"] == 0.0
    assert c.base["fractal_morph"] == pytest.approx(0.2)                           # A only: held
    run(c, 2.0)
    assert c.trans.active is None and c.base["l_land"] == 1.0 and c.base["l_fractal"] == 0.0
    assert c.base["fractal_morph"] == pytest.approx(PARAMS["fractal_morph"][2])


def test_a_pass_both_looks_show_morphs_over_the_transition(tmp_path):
    c = two_scenes(make(tmp_path), b=dict(B_LOOK, l_land=1.0, land_twist=0.9), a=dict(l_land=1.0, land_twist=0.1))
    c.scene_recall(1, transition={"kind": "wipe", "secs": 2.0})
    xs = []
    for _ in range(int(2.2 / DT)):
        xs.append(run(c, DT)["land_twist"])
    assert xs[0] < 0.2 and 0.3 < xs[len(xs) // 2] < 0.7 and xs[-1] == pytest.approx(0.9, abs=1e-3)


def test_resolution_order_explicit_then_scene_then_default(tmp_path):
    c = two_scenes(make(tmp_path))
    c.apply([{"cmd": "transition", "op": "default", "spec": {"kind": "wipe", "secs": 1}},
             {"cmd": "transition", "op": "scene", "i": 1, "spec": "iris"}])
    assert c.trans.default["kind"] == "wipe" and c.scenes[1]["transition"]["kind"] == "iris"
    c.scene_recall(1)
    assert c.trans.active["spec"]["kind"] == "iris"   # the scene's own
    c.trans.finish()
    c.scene_recall(2)
    assert c.trans.active["spec"]["kind"] == "wipe"   # the default
    c.trans.finish()
    c.scene_recall(1, transition="morph")
    assert c.trans.active is None                     # explicit MORPH beats the scene's default
    c.apply([{"cmd": "transition", "op": "scene", "i": 1, "spec": None}])
    assert "transition" not in c.scenes[1]
    c2 = make(tmp_path)                               # the scene default survives a save / load (scenes.json)
    c2._load_scenes()
    assert "transition" not in c2.scenes[1]


def test_commands_validate_and_refuse_junk():
    v = control.validate
    assert v({"cmd": "scene_recall", "i": 2, "transition": "doors"})["transition"]["kind"] == "doors"
    assert v({"cmd": "scene_recall", "i": 2, "transition": "morph"})["transition"] == "morph"
    with pytest.raises(ValueError):
        v({"cmd": "scene_recall", "i": 2, "transition": "nope"})
    assert v({"cmd": "transition", "op": "origin", "x": 9, "y": -1}) == {"cmd": "transition", "op": "origin", "x": 1.0, "y": 0.0}
    assert v({"cmd": "transition", "op": "default", "spec": None})["spec"] is None
    with pytest.raises(ValueError):
        v({"cmd": "transition", "op": "explode"})
    assert "transition" in control.CONTROL_CMDS
    assert control.valid_target("transition:doors") and control.valid_target("transition:len")
    assert not control.valid_target("transition:x")
    assert control.meta()["transitions"]["kinds"] == list(T.KINDS)


def test_midi_pads_pick_the_default_and_a_knob_its_length(tmp_path):
    c = make(tmp_path)
    c._act({"target": "transition:portal"}, "note_on", 127)
    assert c.trans.default["kind"] == "portal"
    c._act({"target": "transition:len"}, "cc", 127)
    assert c.trans.default["beats"] == 16.0
    c._act({"target": "transition:len"}, "cc", 0)
    assert c.trans.default["beats"] == 1.0
    c._act({"target": "transition:morph"}, "note_on", 127)
    assert c.trans.default is None


def test_default_and_origin_persist_in_the_settings(tmp_path):
    c = make(tmp_path)
    c.apply([{"cmd": "transition", "op": "default", "spec": {"kind": "zoom", "beats": 8}},
             {"cmd": "transition", "op": "origin", "x": 0.25, "y": 0.75}])
    s = json.loads(json.dumps(c._settings()))
    d = make(tmp_path)
    d._apply_state({"settings": s})
    assert d.trans.default == c.trans.default and d.trans.origin == [0.25, 0.75]
    d._apply_state({"settings": dict(s, trans={"default": {"kind": "bogus"}, "origin": "x"})})
    assert d.trans.default is None and d.trans.origin == [0.25, 0.75]


def test_state_is_small_and_json(tmp_path):
    c = two_scenes(make(tmp_path))
    assert len(json.dumps(c.state()["trans"])) < 120
    c.scene_recall(1, transition="doors")
    st = json.loads(json.dumps(c.state()["trans"]))
    assert st["active"]["kind"] == "doors" and len(json.dumps(st)) < 300


def test_timeline_cue_carries_a_transition(tmp_path):
    ctx = control.TL_CTX or timeline.Ctx(PARAMS, n_scenes=control.N_SCENES)
    it = timeline.clean_item("cue", {"t": 1.0, "kind": "scene", "slot": 1, "morph": 0, "transition": "shatter"}, ctx)
    assert it["transition"]["kind"] == "shatter"
    assert "transition" not in timeline.clean_item("cue", {"t": 1.0, "kind": "scene", "slot": 1}, ctx)
    with pytest.raises(ValueError):
        timeline.clean_item("cue", {"t": 1.0, "kind": "scene", "slot": 1, "transition": "warp9"}, ctx)
    c = two_scenes(make(tmp_path))
    c.apply([{"cmd": "transition", "op": "default", "spec": "doors"}])
    c._tl_fire({"kind": "cue"}, dict(id=1, kind="scene", slot=1, morph=0.0))
    assert c.trans.active is None                     # a cue without its own transition stays a MORPH (old shows)
    c._tl_fire({"kind": "cue"}, dict(id=2, kind="scene", slot=2, morph=0.0, transition=T.clean_spec("peel")))
    assert c.trans.active["spec"]["kind"] == "peel"


def test_theme_restore_uses_its_own_or_the_default_transition(tmp_path):
    c = two_scenes(make(tmp_path))
    c.base.update(B_LOOK)
    run(c, 0.1)
    c.apply([{"cmd": "theme", "op": "capture", "scope": "look", "slot": 1}])
    c.base.update({n: PARAMS[n][2] for n in control.CONT})
    run(c, 0.2)
    assert themes.validate({"cmd": "theme", "op": "restore", "slot": 1, "transition": "curtains"})["transition"]["kind"] == "curtains"
    c.apply([{"cmd": "theme", "op": "restore", "slot": 1, "transition": "curtains"}])
    assert c.trans.active is not None and c.trans.active["spec"]["kind"] == "curtains" and c.trans.active["label"] == "theme"
    run(c, 6.0)
    assert c.trans.active is None and c.base["hue"] == pytest.approx(B_LOOK["hue"])
    c.base.update({n: PARAMS[n][2] for n in control.CONT})
    run(c, 0.2)
    c.apply([{"cmd": "theme", "op": "restore", "slot": 1}])
    assert c.trans.active is None                     # no default set: the theme MORPHs as before
    c.base.update({n: PARAMS[n][2] for n in control.CONT})
    run(c, 0.2)
    c.apply([{"cmd": "transition", "op": "default", "spec": "dissolve"}, {"cmd": "theme", "op": "restore", "slot": 1}])
    assert c.trans.active is not None and c.trans.active["spec"]["kind"] == "dissolve"


def test_autopilot_uses_the_default_and_never_overlaps(tmp_path):
    c = two_scenes(make(tmp_path))
    c.apply([{"cmd": "transition", "op": "default", "spec": {"kind": "wipe", "beats": 8}}])
    c.auto.update(on=True, bars=1, mode="next")
    labels = set()
    for _ in range(int(12 / DT)):
        run(c, DT)
        if c.trans.active is not None:
            labels.add(c.trans.active["label"])
    assert "autopilot" in labels and c.trans.stats["started"] >= 2


def test_crossfader_new_canvas_and_show_drop_a_blend(tmp_path):
    c = two_scenes(make(tmp_path))
    c.scene_recall(1, transition="doors")
    c.set_xfade(a=2, b=3, x=0.5)
    assert c.trans.active is None
    c.scene_recall(1, transition="doors")
    c.new_canvas()
    assert c.trans.active is None


# ------------------------------------------------------------------------------------------------ GL (under the GPU lock)
def _gl():
    mgl = pytest.importorskip("moderngl")
    try:
        return mgl.create_standalone_context(require=330)
    except Exception as ex:                            # no GPU here
        pytest.skip(f"no GL: {ex}")


def _tex(ctx, img):
    t = ctx.texture((img.shape[1], img.shape[0]), 4, np.ascontiguousarray(img, dtype="f2").tobytes(), dtype="f2")
    return t


@pytest.mark.parametrize("kind,over", [v for v in VARIANTS if v[0] != "portal" or v[1].get("mode") != "white"],
                         ids=[f"{k}-{'-'.join(map(str, o.values())) or 'default'}" for k, o in VARIANTS
                              if k != "portal" or o.get("mode") != "white"])
def test_shader_mask_matches_the_numpy_mirror_gl(kind, over):
    ctx = _gl()
    w, h = 160, 100
    tp = T.TransitionPass(ctx)
    a = np.zeros((h, w, 4), "f4")
    a[..., 3] = 1
    b = np.ones((h, w, 4), "f4")
    disp, bt = _tex(ctx, a), _tex(ctx, b)
    spec = T.clean_spec(dict(over, kind=kind, glow=0.0))
    for e in (0.25, 0.5, 0.75):
        disp.write(np.ascontiguousarray(a, dtype="f2").tobytes())
        tp.composite(disp, dict(spec=spec, e=e, origin=None), w / h, bt)
        out = np.frombuffer(disp.read(), dtype="f2").reshape(h, w, 4)[..., 0].astype("f4")
        ref = T.mask_ref(spec, e, (w, h))
        got = float(np.mean(out > 0.25))               # B weight, through any shading of B (>= 0.5)
        want = float(np.mean(ref > 0.5))
        assert abs(got - want) < 0.025, (e, got, want)
    tp.release()
    ctx.release()


def test_b_target_private_and_echo_untouched_gl():
    """A two-output 'scene' program (like SCENE_FS: colour + echo energy): render_b() writes only the private B
    target (both of its attachments), never the engine's colour / echo pair, and puts A's uniforms back."""
    ctx = _gl()
    import moderngl as mgl
    prog = ctx.program(vertex_shader=T.VS, fragment_shader="""
#version 330
uniform float p_zoom; uniform float pxh; uniform vec3 pal[5];
in vec2 uv; layout(location=0) out vec4 f; layout(location=1) out vec4 fbE;
void main(){ f = vec4(p_zoom, pal[0].r, pxh, 1.0); fbE = vec4(p_zoom*0.5); }
""")
    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
    vao = ctx.vertex_array(prog, [(quad, "2f", "in_pos")])
    size = (64, 40)
    disp, echo = ctx.texture(size, 4, dtype="f2"), ctx.texture(size, 4, dtype="f2")
    fbo = ctx.framebuffer([disp, echo])
    pal_a, pal_b = np.zeros((5, 3), "f4"), np.full((5, 3), 0.75, "f4")
    uni = {"p_zoom": 0.25, "pxh": 1.0 / size[1]}
    fbo.use()
    ctx.viewport = (0, 0) + size
    prog["p_zoom"].value, prog["pxh"].value = 0.25, uni["pxh"]
    prog["pal"].write(pal_a.tobytes())
    vao.render(mgl.TRIANGLE_STRIP)
    before = (disp.read(), echo.read())
    tp = T.TransitionPass(ctx)
    bt = tp.render_b(prog, vao, uni, dict(uni, p_zoom=1.0), pal_a, pal_b, size, 0.5)
    assert (disp.read(), echo.read()) == before        # the engine's pair: untouched
    assert bt.size == (32, 20) and tp.draws == 1
    bpx = np.frombuffer(bt.read(), dtype="f2").reshape(20, 32, 4)
    assert bpx[..., 0].mean() == pytest.approx(1.0) and bpx[..., 1].mean() == pytest.approx(0.75, abs=1e-3)
    scr = np.frombuffer(tp._b[3].read(), dtype="u1").reshape(20, 32, 4)
    assert scr[..., 0].mean() == pytest.approx(128, abs=1)   # B's echo output went to the scratch attachment
    assert prog["p_zoom"].value == 0.25 and prog["pxh"].value == pytest.approx(uni["pxh"])   # A's uniforms back
    assert np.allclose(np.frombuffer(prog["pal"].read(), "f4")[:3], 0.0)
    assert ctx.fbo == fbo                             # the engine's target is current again
    tp.release()
    ctx.release()


DRIVER = ROOT / "tests" / "transitions_engine_driver.py"


def _engine(tmp_path, name, *args, env=None):
    out = tmp_path / name
    e = dict(os.environ, **(env or {}))
    r = subprocess.run([sys.executable, str(DRIVER), *map(str, args), "--test", "150", "--out", str(out),
                        "--test-audio", "silent"], capture_output=True, text=True, timeout=600, cwd=str(ROOT), env=e)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("[transition] draws=")]
    stats = {k: float(v) for k, v in (kv.split("=") for kv in line[-1].split()[1:])} if line else None
    return out, stats, r.stdout


def test_engine_idle_is_pixel_identical_and_b_draws_only_during_a_transition_gl(tmp_path):
    _gl()
    plain, none, _ = _engine(tmp_path, "plain", "none", env={"LV_TRANSITIONS": "0"})
    idle, st, _ = _engine(tmp_path, "idle", "none")
    assert none is None and st["draws"] == 0
    files = sorted(f.name for f in plain.glob("frame_*.png"))
    assert files and all((plain / f).read_bytes() == (idle / f).read_bytes() for f in files)
    # a 1 s doors transition starting at frame 30 of 150 (60 fps): B is drawn on those ~60 frames only
    _, st, log = _engine(tmp_path, "doors", "doors:30:1.0")
    assert 55 <= st["draws"] <= 62, log[-1500:]


# ------------------------------------------------------------------------------------------------ panel (headless browser, CPU)
def test_panel_picks_a_transition_and_a_scene_pad_uses_it_browser(tmp_path):
    pw_api = pytest.importorskip("playwright.sync_api")
    import socket
    import threading
    import time
    from remote_control import RemoteControl
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art", tex_dir=tmp_path / "tex")
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
    ctl.quantize, ctl.morph_bars = "off", 0.0
    ctl.trans.engine_ok = True
    ctl.scene_save(0)
    ctl.base.update(B_LOOK)
    ctl.scene_save(1)
    ctl.base.update({n: PARAMS[n][2] for n in control.CONT})
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            ctl.apply(rc.drain())
            ctl.update(1 / 60, [0.5, 0.3, 0.1])
            rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60))
            time.sleep(1 / 60)

    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def until(cond, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cond():
                return True
            time.sleep(0.03)
        return False
    errors = []
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
                pytest.skip("no Chromium-family browser available")
            page = br.new_page(viewport=dict(width=1920, height=1080))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(f"http://127.0.0.1:{port}/?k=tok")
            page.wait_for_function("window.__panel && window.__panel.state().online", timeout=10000)
            page.click('#tabs button[data-tab="scenes"]')
            page.click('#pg-scenes [data-mt="transition:doors"]')
            assert until(lambda: (ctl.trans.default or {}).get("kind") == "doors")
            box = page.locator("#pg-scenes .trbox").first
            box.locator(".trrow .seg").nth(0).locator("button", has_text="8").first.click()
            assert until(lambda: ctl.trans.default.get("beats") == 8.0)
            box.locator("button", has_text="TOWARD").click()
            assert until(lambda: ctl.trans.default.get("mode") == "toward")
            o = box.locator("canvas.trorg").bounding_box()
            page.mouse.click(o["x"] + o["width"] * 0.25, o["y"] + o["height"] * 0.25)
            assert until(lambda: abs(ctl.trans.origin[0] - 0.25) < 0.05 and abs(ctl.trans.origin[1] - 0.75) < 0.05)
            page.click('#pg-scenes [data-mt="scene:1"]')
            assert until(lambda: ctl.trans.active is not None and ctl.trans.active["spec"]["kind"] == "doors")
            assert until(lambda: "Doors" in box.locator(".trst").inner_text())
            box.locator("button", has_text="END NOW").click()
            assert until(lambda: ctl.trans.active is None)
            page.click('#tabs button[data-tab="perform"]')                   # the compact picker on PERFORM
            page.select_option('#pg-perform .pftr select[title^="Transition"]', "portal")
            assert until(lambda: (ctl.trans.default or {}).get("kind") == "portal")
            page.set_viewport_size(dict(width=390, height=844))
            page.wait_for_timeout(300)
            assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 2
            br.close()
    finally:
        stop.set()
        th.join(timeout=2)
        ctl.close()
        rc.stop()
    assert errors == []
