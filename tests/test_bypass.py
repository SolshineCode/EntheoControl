"""BYPASS: click a layer / tab entirely off and on again to exactly the same settings (controller level)."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import control  # noqa: E402
import params  # noqa: E402
from control import Controller, validate  # noqa: E402
from params import NEUTRAL, PARAMS, TRIGGERS  # noqa: E402


class FakeTime:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(tmp_path, **kw):
    ft = FakeTime()
    kw.setdefault("restore", False)
    kw.setdefault("autosave", True)
    c = Controller(tmp_path, midi=False, ai=False, time_fn=ft, seed=1, **kw)
    c.ft = ft
    c.quantize = "off"
    c.morph_bars = 0.0
    return c


def run(c, seconds, dt=1 / 60, bands=(0.6, 0.4, 0.2)):
    f = None
    for _ in range(int(round(seconds / dt))):
        c.ft.t += dt
        f = c.update(dt, bands)
    return f.values


def do(c, **cmd):
    assert c.apply([cmd]) == []


COVER = params.bypass_cover()


# ---------------------------------------------------------------- the table


def test_every_param_is_classified_exactly_once():
    neutral, keep = set(NEUTRAL), set(params.BYPASS_KEEP)
    assert not neutral & keep, f"both neutral and keep: {sorted(neutral & keep)}"
    missing = set(PARAMS) - neutral - keep
    assert not missing, (f"new params without a bypass decision: {sorted(missing)}. Add each to params.NEUTRAL (value that "
                         "means 'this does nothing') or to params.BYPASS_KEEP (bypass leaves it alone).")
    assert not (neutral | keep) - set(PARAMS), "the bypass tables name params that do not exist"
    for n, v in NEUTRAL.items():
        assert PARAMS[n][0] <= v <= PARAMS[n][1], n
    assert all(NEUTRAL[n] == 0.0 for n in PARAMS if n.startswith("l_")), "every layer's neutral is opacity 0"
    assert not set(TRIGGERS) & neutral and "master" in keep and "palette" in keep and "hue" in keep


def test_keys_cover_layers_tabs_and_the_fx_master():
    for key in ("l_kene", "l_art", "l_land", "l_swarm", "serpent", "chase", "motion", "shape", "color", "layers",
                "fractal", "art", "land", "fx"):
        assert key in COVER and COVER[key], key
    assert "master" not in COVER and "mapping" not in COVER and "space" not in COVER
    assert {"zoom", "twirl", "rotate"} <= set(COVER["shape"]), "the SHAPE tab holds the space and shape groups"
    assert set(COVER["layers"]) == {n for n in PARAMS if n.startswith("l_")}
    assert "l_art" in COVER["art"] and {"l_land", "l_swarm"} <= set(COVER["land"]) and COVER["fractal"] == ("l_fractal",)
    assert not any(n.startswith("l_") for n in COVER["fx"]) and {"twirl", "travel", "chase", "spin"} <= set(COVER["fx"])
    for g in {v[3] for v in PARAMS.values()} - {"master"}:           # tabs other agents add get a key by themselves
        key = params.BYPASS_GROUP_KEY.get(g, g)
        assert key in COVER or not any(n in NEUTRAL for n, v in PARAMS.items() if v[3] == g) and g not in params.BYPASS_LINKS
    m = control.meta()["bypass"]
    assert m["keys"] == list(COVER) and m["cover"]["l_kene"] == ["l_kene"]


# ---------------------------------------------------------------- behaviour


def test_bypass_off_and_on_again_is_exactly_the_same_look(tmp_path):
    c = make(tmp_path)
    c.set("l_kene", 0.73); c.set("twirl", -0.4); c.set("zoom", 0.31); c.set("folds", 12); c.set("palette", 2.5)
    c.bpm = 120.0
    before = dict(run(c, 2.0, bands=(0, 0, 0)))
    base, mods = dict(c.base), json.dumps(c.mods)
    do(c, cmd="bypass", key="l_kene", on=True)
    do(c, cmd="bypass", key="shape", on=True)
    v = run(c, 1.0, bands=(0, 0, 0))
    assert v["l_kene"] == 0.0 and v["twirl"] == 0.0 and v["zoom"] == 0.0
    assert v["folds"] == 12 and v["palette"] == 2.5 and v["l_beads"] == before["l_beads"]
    assert c.base == base and json.dumps(c.mods) == mods, "bypass must not touch stored values"
    do(c, cmd="bypass", key="l_kene", on=False)
    do(c, cmd="bypass", key="shape", on=False)
    after = run(c, 1.0, bands=(0, 0, 0))
    assert after == before


def test_mods_keep_running_underneath(tmp_path):
    a, b = make(tmp_path / "a"), make(tmp_path / "b")
    for c in (a, b):
        c.bpm = 120.0
        c.set("twirl", 0.2)
        c.set_mod("twirl", 0, "lfo_sine", amt=0.3, rate=4.0)
        c.set_mod("l_kene", 0, "bass", amt=-0.5)
    run(a, 1.0); run(b, 1.0)
    do(a, cmd="bypass", key="fx", on=True)
    do(a, cmd="bypass", key="l_kene", on=True)
    va = run(a, 3.3)
    vb = run(b, 3.3)
    assert va["twirl"] == 0.0 and va["l_kene"] == 0.0 and abs(vb["twirl"] - 0.2) > 0.01 and vb["l_kene"] < 1.0
    assert a.mods == b.mods and a.base == b.base
    do(a, cmd="bypass", key="fx", on=False)
    do(a, cmd="bypass", key="l_kene", on=False)
    va, vb = run(a, 1.0), run(b, 1.0)
    assert va == pytest.approx(vb, abs=1e-9), "back on: the modulation is where it would have been anyway"


def test_bypass_crossfades_and_the_time_is_configurable(tmp_path):
    c = make(tmp_path)
    assert c.bypass_fade == pytest.approx(0.3)
    run(c, 1.0)
    do(c, cmd="bypass", key="l_kene", on=True)
    mid = run(c, 0.15)["l_kene"]
    assert 0.3 < mid < 0.7, "about half way through a 0.3 s fade"
    assert run(c, 0.2)["l_kene"] == 0.0
    do(c, cmd="bypass", key="l_kene", on=False)
    assert 0.3 < run(c, 0.15)["l_kene"] < 0.7
    assert run(c, 0.2)["l_kene"] == 1.0
    do(c, cmd="bypass_fade", v=0.0)
    do(c, cmd="bypass", key="l_kene", on=True)
    assert run(c, 1 / 60)["l_kene"] == 0.0, "fade 0 = instant"
    do(c, cmd="bypass_fade", v=99.0)
    assert c.bypass_fade == 5.0


def test_toggle_groups_links_and_fx_master(tmp_path):
    c = make(tmp_path)
    c.set("l_art", 0.9); c.set("l_land", 0.8); c.set("l_swarm", 0.7); c.set("l_fractal", 0.6); c.set("travel", 0.5)
    c.set("chase", 0.9)
    run(c, 1.5)                                                      # let the fader slew settle
    do(c, cmd="bypass_fade", v=0.0)
    do(c, cmd="bypass", key="art")                                   # no "on": toggle
    assert "art" in c.bypass and run(c, 0.1)["l_art"] == 0.0
    do(c, cmd="bypass", key="art")
    assert "art" not in c.bypass and run(c, 0.1)["l_art"] == pytest.approx(0.9)
    do(c, cmd="bypass", key="land", on=True); do(c, cmd="bypass", key="fractal", on=True)
    v = run(c, 0.1)
    assert v["l_land"] == 0.0 and v["l_swarm"] == 0.0 and v["l_fractal"] == 0.0 and v["l_art"] == pytest.approx(0.9)
    do(c, cmd="bypass", key="fx", on=True)
    v = run(c, 0.1)
    assert v["travel"] == 0.0 and v["chase"] == 0.0 and v["l_kene"] == 1.0 and v["l_art"] == pytest.approx(0.9)
    do(c, cmd="bypass", key="layers", on=True)
    assert all(run(c, 0.1)[n] == 0.0 for n in PARAMS if n.startswith("l_"))
    s = c.state()
    assert s["bypass"] == sorted(["land", "fractal", "fx", "layers"]) and s["bypass_fade"] == 0.0


def test_triggers_and_master_are_never_bypassed(tmp_path):
    c = make(tmp_path)
    do(c, cmd="bypass_fade", v=0.0)
    for key in COVER:
        do(c, cmd="bypass", key=key, on=True)
    c.trigger("flash", True)
    v = run(c, 0.1)
    assert v["flash"] == 1.0 and v["master"] == 1.0 and v["folds"] == PARAMS["folds"][2] and v["speed"] == PARAMS["speed"][2]


def test_validation():
    assert validate(dict(cmd="bypass", key="l_kene", on=True, type="bypass", junk=1)) == dict(cmd="bypass", key="l_kene", on=True)
    assert validate(dict(cmd="bypass", key="fx")) == dict(cmd="bypass", key="fx")
    for bad in (dict(key="master"), dict(key="mapping"), dict(key="nope"), dict(key=3), dict(), dict(key="fx", on=1),
                dict(key="palette")):
        with pytest.raises(ValueError):
            validate(dict(bad, cmd="bypass"))
    with pytest.raises(ValueError):
        validate(dict(cmd="bypass_fade", v="x"))
    import remote_control
    assert remote_control.validate_command({"type": "bypass", "key": "art", "on": False}) == dict(cmd="bypass", key="art", on=False)


def test_persists_across_a_restart_and_survives_junk(tmp_path):
    c = make(tmp_path)
    do(c, cmd="bypass", key="l_beads", on=True)
    do(c, cmd="bypass", key="motion", on=True)
    do(c, cmd="bypass_fade", v=1.5)
    assert c._autosave()
    c.close()
    c2 = Controller(tmp_path, midi=False, ai=False, restore=True)
    assert c2.bypass == {"l_beads", "motion"} and c2.bypass_fade == 1.5
    assert c2.update(1 / 60, [0, 0, 0]).values["l_beads"] == 0.0, "a restart comes back already bypassed, no fade-out on screen"
    assert c2.base["l_beads"] == PARAMS["l_beads"][2]
    c2.close()
    p = tmp_path / "live_state.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["settings"]["bypass"] = ["l_beads", "bogus", 5, "master"]
    d["settings"]["bypass_fade"] = "soon"
    p.write_text(json.dumps(d), encoding="utf-8")
    c3 = Controller(tmp_path, midi=False, ai=False, restore=True)
    assert c3.bypass == {"l_beads"} and c3.bypass_fade == pytest.approx(0.3)
    c3.close()


def test_midi_pad_toggles_a_bypass(tmp_path):
    c = make(tmp_path)
    assert control.valid_target("bypass:l_kene") and control.valid_target("bypass:fx")
    assert not control.valid_target("bypass:master") and not control.valid_target("bypass:")
    c._act(dict(target="bypass:l_kene"), "note_on", 127)
    assert c.bypass == {"l_kene"}
    c._act(dict(target="bypass:l_kene"), "note_off", 0)              # release does nothing
    assert c.bypass == {"l_kene"}
    c._act(dict(target="bypass:l_kene"), "note_on", 100)
    assert c.bypass == set()
    c._act(dict(target="bypass:l_kene"), "cc", 127)                  # a button sending CC works too
    assert c.bypass == {"l_kene"}


def test_scenes_and_panic_leave_bypass_alone(tmp_path):
    c = make(tmp_path)
    do(c, cmd="bypass_fade", v=0.0)
    c.set("l_kene", 0.5)
    c.scene_save(3)
    do(c, cmd="bypass", key="l_kene", on=True)
    c.set("l_kene", 0.9)
    c.scene_save(4)                                                  # saved while bypassed: stores the fader, not the 0
    assert c.scenes[4]["base"]["l_kene"] == pytest.approx(0.9) and "bypass" not in c.scenes[4]
    c.scene_recall(3, now=True)
    assert c.bypass == {"l_kene"} and run(c, 0.2)["l_kene"] == 0.0 and c.base["l_kene"] == pytest.approx(0.5)
    c.panic()
    assert c.bypass == {"l_kene"}, "PANIC is unchanged: it does not touch bypass"
    do(c, cmd="bypass", key="l_kene", on=False)
    assert run(c, 0.2)["l_kene"] == pytest.approx(c.base["l_kene"])
    assert math.isfinite(sum(run(c, 0.1).values()))
