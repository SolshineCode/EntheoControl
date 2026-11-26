"""THEMES (themes.py + its control.py / shows.py hooks): capture, EXACT restore per scope, the tray, persistence,
MIDI / timeline addressing, and SOUND TAGS end to end through the controller with synthetic audio (gating:
PANIC, blackout, performer override, flash guard, governor, bypass, quantize, priority, cooldown). No GL."""
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import control  # noqa: E402
import shows  # noqa: E402
import soundtags as S  # noqa: E402
import themes as T  # noqa: E402
from control import CONT, Controller  # noqa: E402
from params import PARAMS  # noqa: E402
from synth_audio import noise_bursts, song_a, song_b  # noqa: E402


class FakeTime:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(tmp_path, **kw):
    ft = FakeTime()
    kw.setdefault("restore", False)
    kw.setdefault("autosave", True)
    c = Controller(tmp_path, midi=False, ai=False, time_fn=ft, seed=1, threads=False, **kw)
    c.ft = ft
    c.quantize = "off"
    c.morph_bars = 0.0
    return c


def run(c, seconds, dt=1 / 60):
    for _ in range(int(round(seconds / dt))):
        c.ft.t += dt
        c.update(dt, (0, 0, 0))


def cmd(c, **k):
    c.apply([dict(k, cmd="theme")])


def last(c):
    return c.themes.tray[-1]


def rnd_value(n, rng):
    lo, hi = PARAMS[n][0], PARAMS[n][1]
    v = rng.uniform(lo, hi)
    return float(round(v)) if n in control._DISCRETE_SET else round(v, 4)


def scramble(c, rng, with_stack=True):
    """A different value for every fader, some modulation, some bypass switches, text, skins, the stack."""
    for n in CONT:
        if n != "master":
            c.base[n] = control.clamp(n, rnd_value(n, rng))
    c.mods = {}
    for n in rng.sample([n for n in CONT if n != "master"], 12):
        c.mods[n] = [dict(src=rng.choice(["bass", "lfo_sine", "random"]), amt=round(rng.uniform(-1, 1), 3),
                          rate=1.0, phase=0.0), None]
    c.bypass = set(rng.sample(sorted(k for k in control.BYPASS_COVER if k != "layers"), 6))
    c._byp = {k: 1.0 for k in c.bypass}
    c.apply([{"cmd": "text", "text": f"WORDS {rng.randint(0, 999)}"}])
    c.apply([{"cmd": "skin", "op": "set", "env": "corridor", "slot": "walls", "f": "scale", "v": rng.uniform(0.3, 7)},
             {"cmd": "skin", "op": "set", "env": "tunnel", "slot": "walls", "f": "rot", "v": rng.uniform(-170, 170)}])
    if with_stack and c.stack is not None:
        c.stack.load(dict(layers=[]), fade=False)
        for g in (0, 1):
            lay = c.stack.add(g)
            lay["opacity"] = round(rng.uniform(0.2, 1.0), 3)
            ph = c.stack._phase(lay["id"])
            ph["scroll"], ph["spin"] = rng.uniform(0, 50), rng.uniform(0, 6)


def norm(d):
    """Captured data without the stack's id allocator (it only counts up, so a new layer never reuses an id a
    theme refers to; it is not part of the look)."""
    d = copy.deepcopy(d)
    if "stack" in d:
        d["stack"].pop("next", None)
    return d


def outside(c, scope):
    names = set(T.scope_names(scope))
    keys = set(T.scope_bypass(scope))
    return ({n: c.base[n] for n in CONT if n not in names},
            {k for k in c.bypass if k not in keys})


SCOPES = ["look", "layer:l_beings", "layer:l_kene_shipibo", "layer:l_text", "layer:l_land", "layer:l_stack",
          "layer:l_ai", "layer:serpent", "group:color", "group:shape", "group:kene", "group:text", "group:land",
          "group:layers", "group:filters"]


# ---------------------------------------------------------------- capture + exact restore


@pytest.mark.parametrize("scope_s", SCOPES)
def test_exact_restore_round_trip_per_scope(tmp_path, scope_s):
    c = make(tmp_path)
    rng = random.Random(hash(scope_s) & 0xFFFF)
    scramble(c, rng)
    c.themes.note_media([{"cmd": "show_texture", "name": "tex_0001.png"}])    # what apply() sees going to the engine
    cmd(c, op="capture", scope=scope_s)
    th = last(c)
    want = copy.deepcopy(th["data"])
    scope = th["scope"]
    assert set(want["values"]) == set(T.scope_names(scope))
    scramble(c, random.Random(99))                           # everything changes
    c.themes.media["tex"] = "other.png"
    out_vals, out_byp = outside(c, scope)
    cmd(c, op="restore", id=th["id"], now=True)               # hold = an instant cut
    got = c.themes.capture_data(scope)
    if "stack_ph" in want:                                     # a cut puts the captured phases back
        assert got["stack_ph"] == want["stack_ph"]
    assert norm(got) == norm(want)
    assert outside(c, scope) == (out_vals, out_byp)           # nothing outside the scope moved
    for n in want["values"]:
        assert c._sm[n] == want["values"][n]                  # a cut skips the slew
    run(c, 0.5)
    assert {n: c.base[n] for n in want["values"]} == want["values"]


def test_stack_layer_scope_restores_that_layer_with_its_id_mods_and_phases(tmp_path):
    c = make(tmp_path)
    a = c.stack.add(0)
    b = c.stack.add(1)
    c.stack.mods[f"{b['id']}:opacity"] = [dict(src="bass", amt=0.5, rate=1.0, phase=0.0), None]
    ph = c.stack._phase(b["id"])
    ph["scroll"], ph["hue"] = 12.5, 0.25
    cmd(c, op="capture", scope=f"stack:{b['id']}")
    th = last(c)
    assert th["name"].startswith(f"Stack #{b['id']}")
    c.stack.find(b["id"])["opacity"] = 0.1
    c.stack.mods.pop(f"{b['id']}:opacity")
    ph["scroll"] = 99.0
    cmd(c, op="restore", id=th["id"], now=True)
    assert norm(c.themes.capture_data(th["scope"])) == norm(th["data"])
    assert [x["id"] for x in c.stack.layers] == [a["id"], b["id"]]          # same place, same id
    c.stack.load(dict(layers=[copy.deepcopy(c.stack.find(a["id"]))], next=c.stack.next), fade=False)
    cmd(c, op="restore", id=th["id"], now=True)                              # deleted meanwhile: comes back by id
    assert c.stack.find(b["id"]) is not None
    assert norm(c.themes.capture_data(th["scope"])) == norm(th["data"])
    c.apply([{"cmd": "history", "op": "undo"}])                              # undo takes it away again
    assert c.stack.find(b["id"]) is None and c.stack.find(a["id"]) is not None


def test_tap_restore_is_quantized_and_morphed_only_in_scope(tmp_path):
    c = make(tmp_path)
    c.base["l_beings"], c.base["being_color"], c.base["zoom"] = 0.9, 4.0, 0.4
    cmd(c, op="capture", scope="layer:l_beings")
    th = last(c)
    c.base["l_beings"], c.base["being_color"] = 0.1, 1.0
    c.quantize, c.morph_bars = "bar", 1.0
    run(c, 60 / c.bpm * 1.3)                                  # mid-bar
    cmd(c, op="restore", id=th["id"])
    assert c.themes.pending and c.base["l_beings"] == 0.1
    c.base["zoom"] = 0.77                                      # outside the scope: untouched by the morph
    beat0 = c.beat
    while c.themes.pending:
        run(c, 1 / 60)
    assert c.beat % 4 < 0.1 and c.beat > beat0
    assert "theme" in c._lane_morph
    run(c, 60 / c.bpm * 2.2)                                  # past half way: continuous params in between
    assert 0.1 < c.base["l_beings"] < 0.9
    assert c.base["being_color"] == 4.0                       # stepped: snaps at the midpoint
    run(c, 60 / c.bpm * 2.2)
    assert c.base["l_beings"] == 0.9 and "theme" not in c._lane_morph
    assert c.base["zoom"] == 0.77


def test_restore_takes_its_params_out_of_a_running_scene_morph(tmp_path):
    c = make(tmp_path)
    c.base["l_beings"] = 0.8
    cmd(c, op="capture", scope="layer:l_beings")
    th = last(c)
    c.scene_save(3)
    c.scenes[3]["base"]["l_beings"] = 0.0
    c.scenes[3]["base"]["zoom"] = 0.2
    c.morph_bars = 2.0
    c.scene_recall(3, now=True)
    run(c, 0.3)
    cmd(c, op="restore", id=th["id"], now=True)
    run(c, 60 / c.bpm * 9)
    assert c.base["l_beings"] == 0.8 and c.base["zoom"] == pytest.approx(0.2)


def test_undo_and_redo_of_a_restore(tmp_path):
    c = make(tmp_path)
    scramble(c, random.Random(1))
    cmd(c, op="capture", scope="look")
    th = last(c)
    scramble(c, random.Random(2))
    before = c.themes.capture_data(th["scope"])
    cmd(c, op="restore", id=th["id"], now=True)
    assert norm(c.themes.capture_data(th["scope"])) == norm(th["data"])
    c.apply([{"cmd": "history", "op": "undo"}])
    after = c.themes.capture_data(th["scope"])
    assert after["values"] == before["values"] and after["mods"] == before["mods"]
    assert after["bypass"] == before["bypass"] and after["text"] == before["text"]
    assert norm(after)["stack"] == norm(before)["stack"]
    c.apply([{"cmd": "history", "op": "redo"}])
    assert c.themes.capture_data(th["scope"])["values"] == th["data"]["values"]


def test_group_scopes(tmp_path):
    c = make(tmp_path)
    sh = T.scope_names(T.parse_scope("group:shape"))
    assert any(PARAMS[n][3] == "space" for n in sh) and any(PARAMS[n][3] == "shape" for n in sh)
    assert T.parse_scope("group:space") == {"kind": "group", "key": "shape"}
    fr = T.scope_names(T.parse_scope("group:fractal"))
    assert "l_fractal" in fr                                   # the tab's switch covers its layer: so does the theme
    assert "fractal" in T.scope_bypass(T.parse_scope("group:fractal"))
    assert "master" not in T.scope_names(T.parse_scope("look"))
    assert not set(T.scope_names(T.parse_scope("look"))) & set(control.TRIGGERS)
    for bad in ("", "layer:zoom", "layer:master", "group:master", "group:nope", "stack:0", "stack:x", 5, "mapping"):
        with pytest.raises(ValueError):
            T.parse_scope(bad)


def test_capture_is_instant_auto_named_and_has_slots(tmp_path):
    c = make(tmp_path)
    c.base["palette"] = 2.0
    cmd(c, op="capture", scope="layer:l_beings")
    th = last(c)
    assert th["name"].startswith("Beings ") and th["name"].endswith("Twilight")
    assert th["slot"] == 1 and th["star"] is False and th["tag"] is None
    cmd(c, op="capture", scope="look")
    assert last(c)["slot"] == 2 and last(c)["name"].startswith("Look ")
    cmd(c, op="rename", id=th["id"], name="  Herd  ")
    assert c.themes.find(th["id"])["name"] == "Herd"
    cmd(c, op="delete", id=th["id"])
    cmd(c, op="capture", scope="look")
    assert last(c)["slot"] == 1                                # the freed slot is reused


def test_tray_is_bounded_and_keeps_starred_and_tagged(tmp_path):
    c = make(tmp_path)
    for _ in range(T.MAX_TRAY):
        cmd(c, op="capture", scope="layer:l_kene")
    first = c.themes.tray[0]["id"]
    c.themes.tray[1]["star"] = True
    cmd(c, op="capture", scope="look")
    assert len(c.themes.tray) == T.MAX_TRAY and c.themes.find(first) is None
    for th in c.themes.tray:
        th["star"] = True
    n = len(c.themes.tray)
    cmd(c, op="capture", scope="look")
    assert len(c.themes.tray) == n                            # full of kept themes: refused, nothing dropped


def test_recapture_updates_in_place(tmp_path):
    c = make(tmp_path)
    c.base["l_beings"] = 0.3
    cmd(c, op="capture", scope="layer:l_beings")
    th = last(c)
    c.base["l_beings"] = 0.6
    cmd(c, op="recapture", id=th["id"])
    assert c.themes.find(th["id"])["data"]["values"]["l_beings"] == 0.6 and len(c.themes.tray) == 1


def test_validation(tmp_path):
    v = control.validate
    assert v({"cmd": "theme", "op": "capture"})["scope"] == {"kind": "look", "key": None}
    assert v({"cmd": "theme", "op": "restore", "id": "t3", "now": True, "junk": 2}) == {
        "cmd": "theme", "op": "restore", "id": "t3", "now": True}
    assert v({"cmd": "theme", "op": "restore", "slot": 4})["slot"] == 4
    assert v({"cmd": "theme", "op": "tag_set", "id": "t1", "set": {"sens": 7, "quant": "off", "prio": 3.6}})["set"] == {
        "sens": 1.0, "quant": "none", "prio": 4}
    for bad in ({"op": "nope"}, {"op": "restore"}, {"op": "restore", "id": "../x"}, {"op": "rename", "id": "t1"},
                {"op": "tag_set", "id": "t1", "set": {}}, {"op": "capture", "scope": "layer:zoom"},
                {"op": "tag", "id": "t1"}):
        with pytest.raises(ValueError):
            v(dict(bad, cmd="theme"))


# ---------------------------------------------------------------- persistence, presets, shows, MIDI, timeline


def test_tray_survives_a_restart_and_arm_does_not(tmp_path):
    c = make(tmp_path)
    scramble(c, random.Random(4))
    cmd(c, op="capture", scope="look")
    cmd(c, op="capture", scope="layer:l_beings")
    cmd(c, op="arm", on=True)
    th = copy.deepcopy(c.themes.tray)
    c._autosave()
    c.close()
    d = make(tmp_path, restore=True)
    assert [x["id"] for x in d.themes.tray] == [x["id"] for x in th]
    assert d.themes.tray[0]["data"] == th[0]["data"]
    assert d.themes.armed is False
    cmd(d, op="capture", scope="look")
    assert last(d)["id"] not in [x["id"] for x in th]          # ids keep counting up


def test_corrupt_tray_entries_are_dropped_not_fatal(tmp_path):
    c = make(tmp_path)
    good = dict(id="t5", slot=3, name="ok", scope={"kind": "layer", "key": "l_beings"},
                data={"values": {"l_beings": 0.5, "zoom": 0.9, "bogus": 1}, "mods": {"l_beings": "x"},
                      "bypass": {"beings": True, "nope": True}}, tag={"ex": [{"bad": 1}], "sens": "x"})
    raw = {"tray": [good, {"id": "t6", "scope": "layer:zoom"}, "junk", dict(good, id="t5")], "next": "x"}
    assert c.themes.restore(raw)
    assert [x["id"] for x in c.themes.tray] == ["t5"]
    th = c.themes.tray[0]
    assert th["data"]["values"] == {"l_beings": 0.5} and th["data"]["mods"] == {}
    assert th["data"]["bypass"] == {"beings": True} and th["tag"]["ex"] == [] and th["tag"]["sens"] == 0.5


def test_star_saves_a_theme_preset_and_recalling_it_restores(tmp_path):
    c = make(tmp_path)
    c.base["l_beings"] = 0.7
    cmd(c, op="capture", scope="layer:l_beings")
    th = last(c)
    cmd(c, op="star", id=th["id"], on=True)
    run(c, 0.1)
    ref = c.themes.find(th["id"])["ref"]
    assert ref and ref.startswith("theme/")
    f = tmp_path / "presets" / "theme" / (ref.split("/")[1] + ".json")
    p = shows.clean_preset(json.loads(f.read_text(encoding="utf-8")))
    assert p["kind"] == "theme" and p["data"]["theme"]["data"]["values"]["l_beings"] == 0.7
    c.base["l_beings"] = 0.0
    c.apply([{"cmd": "preset", "op": "recall", "ref": ref, "morph": 0}])
    run(c, 0.1)
    assert c.base["l_beings"] == 0.7
    assert control._tl.preset_ref(ref) == ref                   # a timeline preset cue can fire it
    cmd(c, op="star", id=th["id"], on=False)
    run(c, 0.1)
    assert c.themes.find(th["id"])["ref"] is None and not f.exists()


def test_show_files_carry_the_tray(tmp_path):
    c = make(tmp_path)
    cmd(c, op="capture", scope="look")
    st = shows.clean_settings(c._settings())
    assert st["themes"]["tray"][0]["id"] == last(c)["id"]
    d = make(tmp_path / "other")
    d._apply_show({"settings": st, "parts": ["scenes"], "name": "x", "slug": "x", "switch": False})
    assert [x["id"] for x in d.themes.tray] == [last(c)["id"]]
    e = make(tmp_path / "third")
    cmd(e, op="capture", scope="look")
    keep = copy.deepcopy(e.themes.tray)
    e._apply_show({"settings": {"bpm": 120}, "parts": ["scenes", "settings"], "name": "y", "slug": "y",
                   "switch": False})
    assert e.themes.tray == keep                                # an older show (no key) leaves the tray alone


def test_midi_theme_slots(tmp_path):
    c = make(tmp_path)
    assert control.valid_target("theme:1") and control.valid_target("theme:32")
    assert not control.valid_target("theme:0") and not control.valid_target("theme:33")
    c._shift = True
    c._act({"target": "theme:5"}, "note_on", 127)              # Shift + pad = capture the look into slot 5
    c._shift = False
    th = c.themes.by_slot(5)
    assert th is not None and th["scope"]["kind"] == "look"
    want = th["data"]["values"]["zoom"]
    c.base["zoom"] = 0.123 if want != 0.123 else 0.5
    c._act({"target": "theme:5"}, "note_on", 127)
    run(c, 0.05)
    assert c.base["zoom"] == want


def test_timeline_theme_event(tmp_path):
    assert control._v_tl_event({"cmd": "theme", "op": "restore", "id": "t2"})["op"] == "restore"
    with pytest.raises(ValueError):
        control._v_tl_event({"cmd": "theme", "op": "capture"})
    c = make(tmp_path)
    c.base["l_kene"] = 0.6
    cmd(c, op="capture", scope="layer:l_kene")
    th = last(c)
    c.base["l_kene"] = 0.0
    c.quantize = "phrase"
    c._tl_firing = True
    try:
        c._do(control.validate({"cmd": "theme", "op": "restore", "id": th["id"], "now": True}))
    finally:
        c._tl_firing = False
    assert c.base["l_kene"] == 0.6                              # exactly where it was placed: no quantize wait


def test_state_is_small_and_the_blob_has_the_tray(tmp_path):
    c = make(tmp_path)
    s0 = json.dumps(c.state()["themes"])
    assert len(s0) < 40
    for _ in range(5):
        cmd(c, op="capture", scope="look")
    c.state()
    rev, data = control.blob("themes")
    doc = json.loads(data)
    assert rev == c.themes.rev and len(doc["tray"]) == 5 and "data" not in doc["tray"][0]
    assert any(k == "l_beings" for k, _ in doc["scopes"]["layers"])
    assert "theme" in control._NOT_EDITS                       # a meter heartbeat never dirties the show


# ---------------------------------------------------------------- SOUND TAGS through the controller


def attach(c):
    eng = S.SoundTags(clock=c.ft)
    c.soundtags = eng
    return eng


def play(c, eng, y, dt=1 / 60):
    y = np.asarray(y, np.float32)
    n = int(round(S.SR * dt))
    for i in range(0, len(y) - n + 1, n):
        eng.feed(y[i:i + n])
        c.ft.t += dt
        c.update(dt, (0, 0, 0))


@pytest.fixture
def tagged(tmp_path):
    """A controller with two look themes, A tagged with song A, B tagged with song B; armed; override 0."""
    c = make(tmp_path)
    eng = attach(c)
    c.base.update(zoom=0.2, l_beings=0.9, palette=1.0)
    cmd(c, op="capture", scope="look")
    a = last(c)
    play(c, eng, song_a(1))
    cmd(c, op="tag", id=a["id"], on=True, len=8)
    play(c, eng, song_a(9))
    c.base.update(zoom=0.8, l_beings=0.1, palette=3.0)
    cmd(c, op="capture", scope="look")
    b = last(c)
    cmd(c, op="tag", id=b["id"], on=True, len=8)
    play(c, eng, song_b(9))
    for th in (a, b):
        cmd(c, op="tag_set", id=th["id"], set={"hold": 2.0, "cooldown": 20.0, "quant": "none"})
    cmd(c, op="opts", override=0.0, gap=4.0)
    cmd(c, op="arm", on=True)
    c.ft.t += 1.0
    return c, eng, a["id"], b["id"]


def fires(c):
    return [x[2] for x in c.themes.log if x[1] == "fire"]


def test_tag_records_an_example_from_the_stream(tagged):
    c, eng, a, b = tagged
    ta = c.themes.find(a)["tag"]
    assert len(ta["ex"]) == 1 and 7 <= ta["ex"][0]["secs"] <= 9
    assert set(eng.templates) == {a, b}
    assert any(x[1] == "tag" for x in c.themes.log)


def test_same_music_recalls_its_theme_different_music_the_other(tagged):
    c, eng, a, b = tagged
    play(c, eng, song_a(10, offset=6))
    assert fires(c) == [a]
    assert c.base["zoom"] == 0.2 and c.base["l_beings"] == 0.9
    play(c, eng, song_b(10))
    assert fires(c) == [a, b]
    assert c.base["zoom"] == 0.8
    play(c, eng, noise_bursts(8))
    assert fires(c) == [a, b]                                   # neither theme for noise


def test_volume_change_still_recalls(tagged):
    c, eng, a, b = tagged
    play(c, eng, song_a(10, gain=0.1))
    assert fires(c) == [a]


def test_no_recall_while_disarmed_and_meters_only_in_test_mode(tagged):
    c, eng, a, b = tagged
    cmd(c, op="arm", on=False)
    play(c, eng, song_a(8))
    assert fires(c) == [] and "m" not in c.state()["themes"] and eng.active is False
    cmd(c, op="test", on=True)
    play(c, eng, song_a(4))
    st = c.state()["themes"]
    assert eng.active and {m[0] for m in st["m"]} == {a, b}
    sim = {m[0]: m[1] for m in st["m"]}
    assert sim[a] > 0.8 > sim[b] and fires(c) == []


def test_cooldown_then_the_section_coming_back_fires_again(tagged):
    c, eng, a, b = tagged
    for th in (a, b):
        cmd(c, op="tag_set", id=th, set={"cooldown": 30.0})
    play(c, eng, song_a(8))                                     # A arrives: fires (~3 s in)
    play(c, eng, song_b(8))                                     # B arrives: fires
    play(c, eng, song_a(8))                                     # A is back inside its 30 s cooldown: held
    play(c, eng, song_b(6))                                     # B too
    assert fires(c) == [a, b]
    play(c, eng, song_a(12))                                    # A is back after its cooldown: fires again
    assert fires(c) == [a, b, a]


def test_performer_move_suspends_auto_recall(tagged):
    c, eng, a, b = tagged
    cmd(c, op="opts", override=20.0)
    play(c, eng, song_a(2))
    c.apply([{"cmd": "set", "name": "zoom", "v": 0.55}])        # the performer takes over
    play(c, eng, song_a(12))
    assert fires(c) == [] and c.base["zoom"] == 0.55
    assert c.state()["themes"]["block"] == "performer"
    assert any(x[1] == "block" and x[5] == "performer" for x in c.themes.log)
    play(c, eng, song_a(10))                                    # the override has ended, but that arrival was used
    assert fires(c) == []
    play(c, eng, song_b(10))
    assert fires(c) == [b]                                      # new music after the override: recall works again


def test_panic_and_blackout_gate(tagged):
    c, eng, a, b = tagged
    c.apply([{"cmd": "panic"}])
    play(c, eng, song_a(12))
    assert fires(c) == []
    c.ft.t += 40
    c.apply([{"cmd": "trigger", "name": "blackout", "on": True}])
    play(c, eng, song_b(10))
    assert fires(c) == []
    c.apply([{"cmd": "trigger", "name": "blackout", "on": False}])
    play(c, eng, song_a(10))
    assert fires(c) == [a]


def test_panic_drops_a_pending_quantized_recall(tagged):
    c, eng, a, b = tagged
    cmd(c, op="tag_set", id=a, set={"quant": "phrase"})
    play(c, eng, song_a(5))
    for _ in range(80):                                         # at most 20 s more
        if c.themes.pending or fires(c):
            break
        play(c, eng, song_a(0.25))
    assert c.themes.pending and c.themes.pending["auto"]
    c.apply([{"cmd": "panic"}])
    run(c, 0.1)
    assert c.themes.pending is None


def test_flash_guard_and_governor_hold_it(tagged):
    c, eng, a, b = tagged
    c.fg_hold = True
    play(c, eng, song_a(8))
    assert fires(c) == []
    c.fg_hold = False
    play(c, eng, song_a(2))
    assert fires(c) == [a]                                      # not used up: fires once the guard lets go
    c.gov_level = 2
    play(c, eng, song_b(10))
    assert fires(c) == [a]
    c.gov_level = 0
    play(c, eng, song_b(3))
    assert fires(c) == [a, b]


def test_auto_recall_never_switches_bypass_and_never_cuts(tagged):
    c, eng, a, b = tagged
    c.themes.find(a)["data"]["bypass"]["beings"] = False
    c.bypass.add("beings")
    c.themes._byp_seen = frozenset(c.bypass)
    c.themes._manual_t = -1e9
    c.morph_bars = 0.0
    lane = []
    orig = c.themes.apply

    def spy(th, **kw):
        orig(th, **kw)
        lane.append(c._lane_morph.get("theme"))
    c.themes.apply = spy
    play(c, eng, song_a(8))
    assert fires(c) == [a] and "beings" in c.bypass             # the performer's kill stands
    assert lane and lane[0] is not None and lane[0]["dur"] >= 1.0   # at least one beat of morph


def test_quantize_to_bar(tagged):
    c, eng, a, b = tagged
    cmd(c, op="tag_set", id=a, set={"quant": "bar"})
    seen = []
    orig = c.themes.apply

    def spy(th, **kw):
        seen.append(c.beat)
        orig(th, **kw)
    c.themes.apply = spy
    play(c, eng, song_a(10))
    assert fires(c) == [a] and len(seen) == 1
    assert seen[0] % 4 < 0.05 or seen[0] % 4 > 3.95


def test_priority_when_two_tags_match(tagged):
    c, eng, a, b = tagged
    cmd(c, op="capture", scope="layer:l_beings")
    x = last(c)["id"]
    c.themes._add_example(c.themes.find(x), copy.deepcopy(c.themes.find(a)["tag"]["ex"][0]))
    cmd(c, op="tag_set", id=x, set={"hold": 2.0, "quant": "none", "prio": 5})
    c.themes._manual_t = -1e9
    play(c, eng, song_a(10))
    assert fires(c)[0] == x


def test_only_auto_flag_needs_autopilot(tagged):
    c, eng, a, b = tagged
    cmd(c, op="tag_set", id=a, set={"only_auto": True})
    play(c, eng, song_a(10))
    assert fires(c) == []


def test_tagging_without_audio_is_refused(tmp_path):
    c = make(tmp_path)
    cmd(c, op="capture", scope="look")
    cmd(c, op="tag", id=last(c)["id"], on=True)
    assert c.themes.rec is None
    eng = attach(c)                                             # attached but silent for > 2 s: still refused
    c.ft.t += 5
    cmd(c, op="tag", id=last(c)["id"], on=True)
    assert c.themes.rec is None and eng.count == 0


def test_suggest_and_apply_a_suggestion(tmp_path):
    c = make(tmp_path)
    eng = attach(c)
    play(c, eng, np.concatenate([song_a(16), song_b(16), song_a(16, offset=8), song_b(16)]))
    cmd(c, op="suggest")
    assert len(c.themes.suggestions) >= 2
    doc = json.loads(c.themes.blob()[1])
    assert doc["suggest"][0]["n"] >= 2 and "example" not in doc["suggest"][0]
    cmd(c, op="suggest_apply", sid=0)
    th = last(c)
    assert th["scope"]["kind"] == "look" and len(th["tag"]["ex"]) == 1


def test_sound_tag_runs_are_deterministic(tmp_path):
    def once(p):
        c = make(p)
        eng = attach(c)
        c.base["zoom"] = 0.3
        cmd(c, op="capture", scope="look")
        a = last(c)["id"]
        cmd(c, op="tag", id=a, on=True, len=6)
        play(c, eng, song_a(7))
        cmd(c, op="tag_set", id=a, set={"hold": 1.0, "quant": "beat", "cooldown": 5.0})
        cmd(c, op="opts", override=0.0, gap=0.0)
        cmd(c, op="arm", on=True)
        c.base["zoom"] = 0.9
        play(c, eng, np.concatenate([song_b(6), song_a(8), song_b(8), song_a(8)]))
        return [(x[1], x[2], x[4]) for x in c.themes.log if x[1] in ("fire", "block")], c.base["zoom"]
    assert once(tmp_path / "1") == once(tmp_path / "2")


def test_controller_side_cost_is_tiny(tagged):
    c, eng, a, b = tagged
    t0 = time.perf_counter()
    for _ in range(600):
        c.ft.t += 1 / 60
        c.themes.step(c.ft.t, c.beat)
    per = (time.perf_counter() - t0) / 600 * 1000
    print(f"[themes step] {per:.4f} ms")
    assert per < 0.05
