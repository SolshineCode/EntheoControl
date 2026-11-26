"""Math stack + NEW CANVAS: pure-Python tests (no GL, no GPU). The GL side lives in tests/test_stack_gl.py."""
import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import control  # noqa: E402
import params  # noqa: E402
import stack  # noqa: E402
from control import Controller, validate  # noqa: E402
from stack import StackState  # noqa: E402


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


def sdo(st, **cmd):
    return st.do(stack.validate(dict(cmd, cmd="stack")))


# ---------------------------------------------------------------- params contract


def test_master_param_is_appended_last_and_classified():
    order = list(params.PARAMS)
    assert order.index("l_stack") == order.index("serpent_bright") + 1      # appended where the registry ended that day
    lo, hi, default, group, _ = params.PARAMS["l_stack"]
    assert (lo, hi, group) == (0.0, 1.0, "layers")
    assert params.NEUTRAL["l_stack"] == 0.0 and "l_stack" in control.BYPASS_COVER
    assert "l_stack" not in control.CUE_CC.values()          # frozen cue table untouched


def test_default_controller_has_an_empty_stack(tmp_path):
    c = make(tmp_path)
    run(c, 0.2)
    assert c.stack.layers == [] and c.stack.frame == []      # nothing to draw: today's look is unchanged
    assert c.state()["stack"]["layers"] == []


# ---------------------------------------------------------------- model: validation + ops


def test_add_set_del_move_dup_clear():
    st = StackState()
    assert sdo(st, op="add", gen=0)
    assert sdo(st, op="add", gen=3)
    a, b = st.layers
    assert (a["id"], b["id"]) == (1, 2) and b["name"] == "Rays"
    sdo(st, op="set", id=1, f="speed", v=99)                 # clamped
    assert st.find(1)["speed"] == stack.NUM["speed"][1]
    sdo(st, op="set", id=1, set=dict(freq=12, wave=3, blend=4.4, name="  Main  "))
    assert (a["freq"], a["wave"], a["blend"], a["name"]) == (12.0, 3, 4, "Main")
    sdo(st, op="dup", id=1)
    assert [x["id"] for x in st.layers] == [1, 3, 2] and st.layers[1]["freq"] == 12.0
    sdo(st, op="move", id=2, to=0)
    assert [x["id"] for x in st.layers] == [2, 1, 3]
    sdo(st, op="del", id=1)
    assert [x["id"] for x in st.layers] == [2, 3]
    sdo(st, op="add", gen=1)
    assert st.layers[-1]["id"] == 4                           # ids are never reused: bindings stay meaningful
    sdo(st, op="clear")
    assert st.layers == []
    assert sdo(st, op="set", id=7, f="freq", v=3) is False    # unknown layer: ignored


def test_at_most_eight_layers():
    st = StackState()
    for _ in range(12):
        sdo(st, op="add", gen=0)
    assert len(st.layers) == stack.MAX_LAYERS
    assert sdo(st, op="dup", id=1) is False


@pytest.mark.parametrize("bad", [
    dict(op="nope"), dict(op="set", id=1), dict(op="set", id=1, f="bogus", v=1), dict(op="set", id="1", f="freq", v=1),
    dict(op="set", id=1, f="freq", v=float("nan")), dict(op="set", id=1, f="freq", v="3"), dict(op="del"),
    dict(op="move", id=1, to=99), dict(op="set", id=1, f="stops", v=[[0, 1, 1, 1]]),
    dict(op="set", id=1, f="stops", v=[[0, 1, 1, 1]] * 7), dict(op="set", id=1, f="stops", v=[[0, 1, 1], [1, 0, 0, 0]]),
    dict(op="mod", id=1, f="gen", slot=0, src="bass"), dict(op="mod", id=1, f="speed", slot=2, src="bass"),
    dict(op="mod", id=1, f="speed", slot=0, src="nope"), dict(op="replace", stack=[1, 2]), dict(op="set", id=True, f="freq", v=1),
])
def test_malformed_commands_are_rejected(bad):
    with pytest.raises(ValueError):
        stack.validate(dict(bad, cmd="stack"))
    with pytest.raises(ValueError):
        validate(dict(bad, cmd="stack"))                       # and through the controller's validator


def test_validation_is_idempotent():
    """The panel server validates, then Controller.apply validates the normalised command again."""
    st = StackState()
    sdo(st, op="add", gen=0)
    for raw in (dict(op="add", gen=2, layer=dict(freq=3, stops=[[0, 1, 0, 0], [0.5, 0, 0, 1]]), after=1),
                dict(op="set", id=1, f="freq", v=7), dict(op="set", id=1, set=dict(name="x", stops=[[0, 1, 1, 1], [1, 0, 0, 0]])),
                dict(op="mod", id=1, f="speed", slot=1, src="lfo_saw", amt=0.3, rate=4, phase=0.5),
                dict(op="mod", id=1, f="speed", slot=1, src="none"), dict(op="move", id=1, to=0), dict(op="reset", id=1, f="duty"),
                dict(op="dup", id=1), dict(op="replace", stack=dict(layers=[dict(gen=1)])), dict(op="del", id=1), dict(op="clear")):
        once = stack.validate(dict(raw, cmd="stack"))
        assert stack.validate(once) == once, raw
        assert validate(validate(dict(raw, cmd="stack"))) == once
    assert validate(validate(dict(cmd="mod", name="stack:1:speed", slot=0, src="bass", amt=0.2)))["mod"]["amt"] == 0.2


def test_stops_are_sorted_and_clamped():
    c = stack.validate(dict(cmd="stack", op="set", id=1, f="stops", v=[[0.9, 2, -1, 0.5], [0.1, 0, 0, 1]]))
    assert c["set"]["stops"] == [[0.1, 0.0, 0.0, 1.0], [0.9, 1.0, 0.0, 0.5]]


def test_clean_stack_survives_garbage():
    for raw in (None, 3, [], dict(layers="x"), dict(layers=[1, None, dict(gen="x", freq="y", stops=7, id=-4)]),
                dict(layers=[dict(id=2), dict(id=2)], mods={"2:speed": [dict(src="bass", amt=5)], "9:speed": [], "x": 1})):
        st = stack.clean_stack(raw)
        ids = [x["id"] for x in st["layers"]]
        assert len(ids) == len(set(ids)) and st["next"] > max(ids + [0])
        for lay in st["layers"]:
            assert set(stack.FIELDS) <= set(lay) and stack.STOP_MIN <= len(lay["stops"]) <= stack.STOP_MAX
        json.dumps(st)
    assert stack.clean_stack(dict(layers=[dict(id=2)], mods={"2:speed": [dict(src="bass", amt=5)]}))["mods"]["2:speed"][0]["amt"] == 1.0


def test_reset_field_and_layer():
    st = StackState()
    sdo(st, op="add", gen=1)
    base = stack.default_layer(1)
    sdo(st, op="set", id=1, set=dict(freq=20, duty=0.9, stops=[[0, 1, 1, 1], [1, 0, 0, 0]]))
    sdo(st, op="mod", id=1, f="freq", slot=0, src="bass", amt=0.5)
    sdo(st, op="reset", id=1, f="freq")
    assert st.find(1)["freq"] == base["freq"] and st.find(1)["duty"] == 0.9 and "1:freq" not in st.mods
    sdo(st, op="reset", id=1)
    assert {k: st.find(1)[k] for k in stack.NUM} == {k: base[k] for k in stack.NUM} and st.find(1)["stops"] == base["stops"]


# ---------------------------------------------------------------- resolve: seamless structure


def random_layer(rng):
    lay = stack.default_layer(rng.randrange(8))
    for k, (lo, hi, *_r) in stack.NUM.items():
        lay[k] = rng.uniform(lo, hi)
    for k, (lo, hi, *_r) in stack.INT.items():
        lay[k] = rng.randint(lo, hi)
    lay["mute"] = lay["solo"] = 0
    n = rng.randint(stack.STOP_MIN, stack.STOP_MAX)
    lay["stops"] = stack.clean_stops([[rng.random() for _ in range(4)] for _ in range(n)])
    return lay


def whole(x, tol=1e-9):
    return abs(x - round(x)) < tol


def test_every_wrap_period_closes_the_band_index():
    """The scroll phase is wrapped at W. For the picture not to pop, W must be a whole number of bands, of colour
    cycles (D) and of hash periods; in the circular spaces every angular frequency must be a whole number and one
    trip round the circle (kc bands) must also be a whole number of colour cycles and hash periods."""
    rng = random.Random(7)
    for _ in range(4000):
        lay = random_layer(rng)
        s = stack.structure(lay)
        n = len(lay["stops"])
        assert whole(s["W"]) and s["W"] >= 1
        shift = s["W"] / 2 if s["xor"] else s["W"]            # how far the colour / hash index moves per wrap
        assert whole(shift / s["D"]), (lay, s)
        assert whole(shift / s["hp"]), (lay, s)
        assert s["D"] > 0
        if s["xor"]:
            assert s["W"] % 2 == 0
        per = lay["space"] in stack.PERIODIC
        if lay["gen"] in stack.RADIAL:
            assert whole(s["fv"]) and whole(s["gu"]) and whole(s["gv"])      # arms, sides, lobes
        elif per:
            assert whole(s["fu"]) and whole(s["wu"]) and whole(s["gu"])
            if s["xor"]:
                assert (int(s["fu"]) + int(s["gu"])) % 2 == 0
        if s["kc"]:
            assert whole(s["kc"]) and whole(s["kc"] / s["D"]) and whole(s["kc"] / s["hp"])
        else:
            assert s["D"] == n and s["hp"] == stack.WRAP


def test_phase_sent_to_the_gpu_stays_small_for_a_ten_hour_show():
    st = StackState()
    sdo(st, op="add", gen=0, layer=dict(speed=4.0, freq=20, spin=0.7, warp_speed=2.0, hue_speed=0.5, warp=0.5))
    sdo(st, op="add", gen=3, layer=dict(count=7, speed=3.0, spin=-0.4))
    for lay in st.layers:                                     # jump ten hours ahead, then run across the wraps
        st._phase(lay["id"])["scroll"] = 4.0 * 36000.0 - 0.01
    seen = []
    for _ in range(240):
        fr = st.step(1 / 60)
        for res in fr:
            ph, spin = res["C"][0], res["C"][1]
            assert 0.0 <= ph < res["W"] + 1e-9 and 0.0 <= spin < stack.TAU + 1e-9
            assert 0.0 <= res["D"][3] < 1.0 and 0.0 <= res["F"][2] < 1.0
        seen.append(fr[0]["C"][0])
    steps = [(b - a) % fr[0]["W"] for a, b in zip(seen, seen[1:])]
    assert max(steps) - min(steps) < 1e-6                     # float64 on the CPU: the step is still exact out there


def test_k_whole_periods_only_shift_wrapped_phases():
    st = StackState()
    sdo(st, op="add", gen=2, layer=dict(space=1, hue=0.3, warp=0.4))
    for _ in range(30):
        st.step(1 / 60)
    a = st.step(0.0)[0]
    b = st.step(0.0, k=3)[0]
    assert b["C"][0] == pytest.approx(a["C"][0] + 3 * a["W"]) and b["C"][1] == pytest.approx(a["C"][1] + 3 * stack.TAU)
    assert b["D"][3] == pytest.approx(a["D"][3] + 3) and b["F"][2] == pytest.approx(a["F"][2] + 3)
    assert a["B"] == b["B"] and a["G"] == b["G"]


def test_freeze_holds_every_phase():
    st = StackState()
    sdo(st, op="add", gen=5, layer=dict(speed=2, spin=0.5, hue_speed=0.3, pulse=1.0, sync=1))
    for i in range(30):
        st.step(1 / 60, dict(bpm=120, beat=i / 30))
    a = json.dumps(st.step(0.0, dict(bpm=120, beat=5.3), hold=True))
    b = json.dumps(st.step(1 / 60, dict(bpm=120, beat=9.9), hold=True))
    assert a == b


def test_rotation_settles_back_upright_when_speed_is_zero():
    st = StackState()
    sdo(st, op="add", gen=0, layer=dict(spin=0.5))
    for _ in range(40):
        st.step(1 / 60)
    assert st.frame[0]["C"][1] > 0.5
    sdo(st, op="set", id=1, f="spin", v=0.0)
    for _ in range(600):
        st.step(1 / 60)
    assert st.frame[0]["C"][1] == 0.0


# ---------------------------------------------------------------- photosensitive limits


def test_coarse_layers_cannot_blink_the_frame_faster_than_the_limit():
    st = StackState()
    sdo(st, op="add", gen=0, layer=dict(freq=0.5, speed=4.0))          # one band wider than the frame
    sdo(st, op="add", gen=0, layer=dict(freq=12, speed=4.0))           # a fine texture may run fast
    sdo(st, op="add", gen=0, layer=dict(freq=0.5, speed=4.0, sync=1))  # beat-synced at 300 BPM = 20 bands / s asked
    fr = st.step(1 / 60, dict(bpm=300.0, beat=0.0))
    assert abs(fr[0]["hz"]) <= stack.SAFE_HZ + 1e-9 and abs(fr[2]["hz"]) <= stack.SAFE_HZ + 1e-9
    assert fr[1]["hz"] == pytest.approx(4.0)
    assert st.capped == [1, 3]
    for bands in (0.1, 1.0, 1.5, 2.0, 3.0, 40.0):
        for hz in (-50, -3, 0, 3, 50):
            assert abs(stack.safe_rate(bands, hz)) <= (stack.SAFE_HZ if bands <= 1.5 else stack.FAST_HZ) + 1e-9
    assert stack.SAFE_HZ <= 3.0


def test_pulses_never_repeat_faster_than_the_limit():
    for bpm in (20, 60, 124, 174, 300):
        for div in (0.25, 0.5, 1, 2, 4, 16):
            sec, beats = stack.pulse_period(div, True, bpm)
            assert sec >= 1 / stack.SAFE_HZ - 1e-9
            assert math.log2(beats / div) == round(math.log2(beats / div))    # still a musical multiple
            assert stack.pulse_period(div, False, bpm)[0] >= 1 / stack.SAFE_HZ - 1e-9
    assert stack.pulse_period(1, True, 124)[1] == 1                    # 2.07 Hz: allowed as asked
    assert stack.pulse_period(1, True, 174)[1] == 2                    # 2.9 Hz: every second beat instead


def test_opacity_and_brightness_cannot_square_wave_faster_than_the_limit():
    st = StackState()
    sdo(st, op="add", gen=0)
    lay = st.find(1)
    levels, flips, last = [], 0, None
    for i in range(600):                                               # a 15 Hz square wave on opacity for 10 s
        lay["opacity"] = 1.0 if (i // 2) % 2 else 0.0
        lay["bright"] = 2.0 if (i // 2) % 2 else 0.0
        fr = st.step(1 / 60)
        op = fr[0]["F"][0] if fr else 0.0
        levels.append(op)
    assert max(levels) - min(levels) < 0.2                             # it cannot follow: a small ripple, no blink
    lay["opacity"] = 0.0
    for _ in range(60):
        st.step(1 / 60)
    lay["opacity"], t = 1.0, 0
    while (st.step(1 / 60) or [dict(F=[0])])[0]["F"][0] < 0.999:
        t += 1
    assert t / 60 >= 0.5 / stack.SAFE_HZ - 0.05                        # a full fade takes at least half a period


def test_mute_solo_and_delete_fade_instead_of_cutting():
    st = StackState()
    sdo(st, op="add", gen=0)
    sdo(st, op="add", gen=1)
    for _ in range(60):
        st.step(1 / 60)
    assert [r["id"] for r in st.frame] == [1, 2]
    sdo(st, op="set", id=2, f="solo", v=1)
    st.step(1 / 60)
    assert 0.5 < st.frame[0]["F"][0] < 1.0                             # layer 1 is on its way out, not gone
    for _ in range(60):
        st.step(1 / 60)
    assert [r["id"] for r in st.frame] == [2]
    sdo(st, op="del", id=2)
    st.step(1 / 60)
    assert [r["id"] for r in st.frame] == [1, 2] and [x["id"] for x in st.layers] == [1]
    assert 0.5 < st.frame[1]["F"][0] < 1.0 and st.frame[0]["F"][0] < 0.5     # a ghost fades out where the layer was; 1 returns
    for _ in range(60):
        st.step(1 / 60)
    assert [r["id"] for r in st.frame] == [1] and not st._ghost and set(st._ph) == {1}


# ---------------------------------------------------------------- modulation, MIDI


def test_stack_fields_are_modulated_by_the_mod_matrix(tmp_path):
    c = make(tmp_path)
    do(c, cmd="stack", op="add", gen=0)
    do(c, cmd="mod", name="stack:1:speed", slot=0, src="bass", amt=0.25, rate=1, phase=0)
    assert c.stack.mods["1:speed"][0]["src"] == "bass" and "stack:1:speed" not in c.mods
    run(c, 0.2, bands=(1.0, 0, 0))
    base = c.stack.find(1)["speed"]
    assert c.stack.live["1:speed"] == pytest.approx(base + 0.25 * 8.0)  # amount x range x source
    assert c.stack.find(1)["speed"] == base                            # the stored value is never touched
    assert c.state()["stack"]["live"]["1:speed"] == pytest.approx(2.25)
    do(c, cmd="mod", name="stack:1:speed", slot=0, src="none")
    assert c.stack.mods == {}
    for bad in ("stack:1:gen", "stack:x:speed", "stack:1:nope", "stack:1"):
        with pytest.raises(ValueError):
            validate(dict(cmd="mod", name=bad, slot=0, src="bass"))


def test_lfo_modulation_follows_the_beat(tmp_path):
    c = make(tmp_path)
    do(c, cmd="stack", op="add", gen=0)
    do(c, cmd="stack", op="mod", id=1, f="freq", slot=1, src="lfo_sine", amt=0.2, rate=1, phase=0)
    seen = []
    for _ in range(120):
        run(c, 1 / 60)
        seen.append(c.stack.live["1:freq"])
    assert max(seen) - min(seen) > 5.0


def test_midi_learn_targets_and_cc(tmp_path):
    assert control.valid_target("stack:2:speed") and control.valid_target("stack:12:mute")
    for bad in ("stack:2:nope", "stack:0:speed", "stack:-1:speed", "stack:2", "stack:2:speed:x", "stack:a:speed"):
        assert not control.valid_target(bad)
    assert validate(dict(cmd="midi_learn", target="stack:2:speed"))["target"] == "stack:2:speed"
    c = make(tmp_path)
    do(c, cmd="stack", op="add", gen=0)
    do(c, cmd="stack", op="add", gen=1)
    c._act(dict(target="stack:2:speed"), "cc", 127)
    assert c.stack.find(2)["speed"] == stack.NUM["speed"][1]
    c._act(dict(target="stack:2:speed"), "cc", 0)
    assert c.stack.find(2)["speed"] == stack.NUM["speed"][0]
    c._act(dict(target="stack:2:gen"), "cc", 127)
    assert c.stack.find(2)["gen"] == 7
    c._act(dict(target="stack:1:mute"), "note_on", 127)
    assert c.stack.find(1)["mute"] == 1
    c._act(dict(target="stack:1:mute"), "note_off", 0)
    assert c.stack.find(1)["mute"] == 1
    c._act(dict(target="stack:1:mute"), "note_on", 127)
    assert c.stack.find(1)["mute"] == 0
    c._act(dict(target="stack:9:speed"), "cc", 64)                     # a layer that is gone: nothing happens


# ---------------------------------------------------------------- persistence, scenes, morphs


def test_stack_persists_in_live_state_and_comes_back_without_a_fade(tmp_path):
    c = make(tmp_path)
    do(c, cmd="stack", op="add", gen=2, layer=dict(freq=9, stops=[[0, 1, 0, 0], [0.5, 0, 0, 1]]))
    do(c, cmd="stack", op="mod", id=1, f="spin", slot=0, src="lfo_tri", amt=0.3, rate=4, phase=0.25)
    c._autosave()
    saved = json.loads((tmp_path / "live_state.json").read_text())["settings"]["stack"]
    assert saved["layers"][0]["freq"] == 9 and saved["mods"]["1:spin"][0]["src"] == "lfo_tri"
    c2 = make(tmp_path, restore=True)
    assert c2.stack.to_json() == c.stack.to_json()
    c2.ft.t += 1 / 60
    c2.update(1 / 60, (0, 0, 0))
    assert c2.stack.frame[0]["F"][0] == pytest.approx(1.0)             # already there on the first frame
    do(c2, cmd="stack", op="add", gen=0)
    assert c2.stack.layers[-1]["id"] == 2


def test_a_corrupt_saved_stack_never_stops_the_restore(tmp_path):
    c = make(tmp_path)
    do(c, cmd="set", name="zoom", v=0.4)
    c._autosave()
    p = tmp_path / "live_state.json"
    d = json.loads(p.read_text())
    d["settings"]["stack"] = dict(layers=[3, dict(gen="x", stops="no"), dict(freq=1e99)], mods=7, next="a")
    d["settings"]["canvas"] = "garbage"
    p.write_text(json.dumps(d))
    c2 = make(tmp_path, restore=True)
    assert c2.restored and c2.base["zoom"] == 0.4 and len(c2.stack.layers) == 2 and c2.canvas_prev is None


def test_scenes_store_the_stack_and_morph_it(tmp_path):
    c = make(tmp_path)
    do(c, cmd="stack", op="add", gen=0, layer=dict(freq=4, wave=1, opacity=1.0))      # id 1
    do(c, cmd="stack", op="add", gen=1, layer=dict(freq=4))                           # id 2 (only in scene A)
    do(c, cmd="scene_save", i=9)
    do(c, cmd="stack", op="del", id=2)
    do(c, cmd="stack", op="set", id=1, set=dict(freq=12, wave=3))
    do(c, cmd="stack", op="add", gen=3)                                               # id 3 (only in scene B)
    do(c, cmd="scene_save", i=10)
    saved = json.loads((tmp_path / "scenes.json").read_text())["scenes"]
    assert [x["id"] for x in saved[9]["stack"]["layers"]] == [1, 2] and saved[10]["stack"]["layers"][0]["freq"] == 12
    c.morph_bars = 1.0                                                                # 4 beats
    c.bpm = 120.0
    do(c, cmd="scene_recall", i=9, now=True)
    run(c, 0.5)                                                                       # 1 beat in: t = 0.25
    L = {x["id"]: x for x in c.stack.layers}
    assert L[1]["freq"] == pytest.approx(10.0, abs=0.3) and L[1]["wave"] == 3         # numeric glides, stepped waits
    assert L[2]["opacity"] == pytest.approx(0.25, abs=0.05) and L[3]["opacity"] == pytest.approx(0.75, abs=0.05)
    run(c, 1.0)                                                                       # t = 0.75: stepped has snapped
    assert c.stack.find(1)["wave"] == 1
    run(c, 1.0)
    assert c.stack.to_json()["layers"] == saved[9]["stack"]["layers"]
    do(c, cmd="scene_recall", i=10, now=True)
    run(c, 0.5)
    do(c, cmd="stack", op="set", id=1, f="freq", v=20)                                # a hand on the stack ends its morph
    run(c, 3.0)
    assert c.stack.find(1)["freq"] == 20


def test_old_scenes_without_a_stack_leave_it_alone(tmp_path):
    c = make(tmp_path)
    do(c, cmd="stack", op="add", gen=0)
    assert "stack" not in c.scenes[1]                                                 # factory scene
    do(c, cmd="scene_recall", i=1, now=True)
    run(c, 0.2)
    assert len(c.stack.layers) == 1
    (tmp_path / "scenes.json").write_text(json.dumps(dict(version=1, scenes=[dict(name="x", base={}, stack="junk")])))
    c._load_scenes()
    assert c.scenes[0] is not None and "stack" not in c.scenes[0]


def test_blend_endpoints_and_stops():
    a = stack.clean_stack(dict(layers=[dict(id=1, gen=0, freq=2, stops=[[0, 0, 0, 0], [1, 1, 1, 1]])]))
    b = stack.clean_stack(dict(layers=[dict(id=1, gen=1, freq=6, stops=[[0.2, 1, 1, 1], [0.6, 0, 0, 0]])]))
    assert stack.blend(a, b, 0) == a and stack.blend(a, b, 1) == b
    m = stack.blend(a, b, 0.25)["layers"][0]
    assert m["freq"] == pytest.approx(3.0) and m["gen"] == 0 and m["stops"][0] == [0.05, 0.25, 0.25, 0.25]
    assert stack.blend(a, b, 0.75)["layers"][0]["gen"] == 1
    stack.clean_stack(stack.blend(a, b, 0.5))


def test_scene_crossfader_blends_two_stacks(tmp_path):
    c = make(tmp_path)
    do(c, cmd="stack", op="add", gen=0, layer=dict(freq=4))
    do(c, cmd="scene_save", i=9)
    do(c, cmd="stack", op="set", id=1, f="freq", v=8)
    do(c, cmd="scene_save", i=10)
    do(c, cmd="xfade", a=9, b=10, x=0.5)
    assert c.stack.find(1)["freq"] == pytest.approx(6.0)


def test_panic_and_init(tmp_path):
    c = make(tmp_path)
    do(c, cmd="stack", op="add", gen=0)
    do(c, cmd="init")
    assert c.stack.layers == []


# ---------------------------------------------------------------- NEW CANVAS / RESTORE PREVIOUS


def lit(values):
    return {n: v for n, v in values.items() if n.startswith("l_") and n != "l_stack" and v > 1e-6}


def test_new_canvas_is_black_but_for_one_stripe_layer_and_restore_is_exact(tmp_path):
    c = make(tmp_path)
    do(c, cmd="set", name="l_beads", v=0.9)
    do(c, cmd="set", name="twirl", v=0.6)
    do(c, cmd="set", name="serpent", v=0.7)
    do(c, cmd="mod", name="zoom", slot=0, src="bass", amt=0.4, rate=1, phase=0)
    do(c, cmd="bypass", key="l_lines", on=True)
    do(c, cmd="stack", op="add", gen=4)
    before = dict(base=dict(c.base), mods=json.dumps(c.mods), bypass=set(c.bypass), stack=c.stack.to_json())
    v0 = run(c, 0.5)
    assert v0["l_kene"] == 1.0 and v0["twirl"] == pytest.approx(0.6, abs=1e-3)

    do(c, cmd="canvas", op="new")
    assert c.state()["canvas"]["active"]
    mid = run(c, 0.2)
    assert 0.0 < mid["l_kene"] < 1.0                                   # a crossfade, not a cut
    v = run(c, 3.0)
    assert lit(v) == {} and v["serpent"] == 0.0 and v["twirl"] == 0.0 and v["zoom"] == 0.0
    assert v["l_stack"] == 1.0 and v["master"] == 1.0
    assert [(x["gen"], x["speed"] != 0) for x in c.stack.layers] == [(0, True)]
    assert len(c.stack.frame) == 1 and c.stack.frame[0]["F"][0] == pytest.approx(1.0)
    assert c.base["l_beads"] == 0.9 and c.base["twirl"] == 0.6         # nothing stored was edited

    # build on it, bring a layer in, press NEW CANVAS again by accident: the way back must survive all of it
    do(c, cmd="stack", op="add", gen=1)
    do(c, cmd="bypass", key="l_kene", on=False)
    do(c, cmd="set", name="l_kene", v=0.35)
    run(c, 1.0)
    do(c, cmd="canvas", op="new")
    run(c, 1.0)
    assert len(c.stack.layers) == 1

    do(c, cmd="canvas", op="restore")
    v = run(c, 3.0)
    assert not c.state()["canvas"]["active"]
    assert dict(c.base) == before["base"] and json.dumps(c.mods) == before["mods"]
    assert set(c.bypass) == before["bypass"] and c.stack.to_json()["layers"] == before["stack"]["layers"]
    for n in ("l_kene", "l_beads", "serpent", "twirl"):
        assert v[n] == pytest.approx(v0[n], abs=1e-3)
    assert v["l_lines"] == 0.0                                         # it was switched off before: still off
    do(c, cmd="canvas", op="restore")                                  # nothing to restore: a no-op
    assert dict(c.base) == before["base"]


def test_canvas_and_the_way_back_survive_a_restart(tmp_path):
    c = make(tmp_path)
    do(c, cmd="set", name="l_beads", v=0.9)
    do(c, cmd="canvas", op="new")
    run(c, 1.0)
    c._autosave()
    c2 = make(tmp_path, restore=True)
    assert c2.state()["canvas"]["active"] and len(c2.stack.layers) == 1
    assert lit(run(c2, 0.1)) == {}
    do(c2, cmd="canvas", op="restore")
    v = run(c2, 3.0)
    assert v["l_beads"] == pytest.approx(0.9, abs=1e-3) and c2.stack.layers == []


def test_fresh_start_flag(tmp_path):
    c = make(tmp_path)
    do(c, cmd="set", name="l_beads", v=0.9)
    c.new_canvas(fresh=True)
    s = c.state()["canvas"]
    assert s["active"] and s["fresh"]
    v = run(c, 0.1)
    assert lit(v) == {}                                                # launched that way: no fade from the old look
    assert not make(tmp_path / "x").state()["canvas"]["fresh"]


def test_canvas_commands_validate():
    assert validate(dict(cmd="canvas", op="new")) == dict(cmd="canvas", op="new")
    for bad in (dict(cmd="canvas"), dict(cmd="canvas", op="wipe")):
        with pytest.raises(ValueError):
            validate(bad)


def test_meta_describes_the_stack_for_the_panel():
    m = control.meta()["stack"]
    assert m["max"] == 8 and len(m["generators"]) == 8 and {x["name"] for x in m["num"]} == set(stack.NUM)
    assert all(set(stack.FIELDS) <= set(p) for p in m["presets"].values())
    json.dumps(m)
