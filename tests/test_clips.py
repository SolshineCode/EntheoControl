"""Clip model (research/pro_redesign_plan.md Phase 1): lanes own params, clips write base values, 4-8 named macros per
clip, decks of scene columns, engine-side undo. Pure python, no GL: python -m pytest -q tests/test_clips.py"""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import clips as C  # noqa: E402
import control  # noqa: E402
import params  # noqa: E402
import remote_control  # noqa: E402
import shows as S  # noqa: E402
from params import PARAMS, TRIGGERS  # noqa: E402

DT = 1 / 60


class FakeTime:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(tmp_path, **kw):
    ft = FakeTime()
    kw.setdefault("restore", False)
    c = control.Controller(tmp_path, midi=False, ai=False, time_fn=ft, seed=1, threads=False, **kw)
    c.ft = ft
    c.quantize = "off"
    c.morph_bars = 0.0
    return c


def run(c, seconds=0.1, dt=DT):
    f = None
    for _ in range(max(1, int(round(seconds / dt)))):
        c.ft.t += dt
        f = c.update(dt, (0.0, 0.0, 0.0))
    return f.values


def do(c, **cmd):
    return c.apply([cmd])


def clip(**over):
    d = {"id": "t1", "name": "Test", "lane": "base", "sets": {"zoom": 0.4},
         "macros": [{"name": "Bloom", "widget": "knob", "default": 0.0,
                     "targets": [{"p": "twirl", "a": 0.0, "b": 0.6, "curve": "lin"}]}]}
    d.update(over)
    return d


# ---------------------------------------------------------------- the lane table


def test_every_param_is_in_exactly_one_lane():
    seen = {}
    for lane in C.LANES:
        for n in C.LANE_PARAMS[lane]:
            assert n not in seen, f"{n} is in lanes {seen[n]} and {lane}"
            seen[n] = lane
    missing = set(PARAMS) - set(seen)
    assert not missing, (f"params without a lane: {sorted(missing)}. Add the group to clips.LANE_BY_GROUP or the param "
                         "to clips.LANE_EXCEPT (base / figures / text / interact / master).")
    assert not set(seen) - set(PARAMS)
    groups = {v[3] for v in PARAMS.values()}
    assert groups <= set(C.LANE_BY_GROUP), f"param groups without a lane decision: {sorted(groups - set(C.LANE_BY_GROUP))}"
    assert not set(C.LANE_EXCEPT) - set(PARAMS), "LANE_EXCEPT names params that do not exist"


def test_lanes_follow_the_plan():
    L = C.lane_of
    assert L("master") == L("flash") == L("palette") == L("hue") == L("saturation") == L("trails") == "master"
    assert L("l_echo") == L("flt_dream") == L("img_hold") == "master"
    assert all(L(n) == "master" for n in TRIGGERS)
    assert L("zoom") == L("folds") == L("travel") == L("chase") == L("l_kene") == L("l_land") == L("land_a") == "base"
    assert L("l_stack") == L("fractal_mode") == L("art_use") == L("l_kene_shipibo") == L("kene_seed") == "base"
    assert L("serpent") == L("serpent_hue2") == L("l_beings") == L("being") == L("l_swarm") == L("swarm_tint") == "figures"
    assert L("l_text") == L("text_size") == "text"
    assert L("l_camera") == L("crowd_mode") == "interact"
    assert C.CLIP_LANES == ("base", "figures", "text", "interact")
    assert set(control.DISCRETE) <= C.STEPPED, "a stepped param that clips would not round"


# ---------------------------------------------------------------- macros: curves, steps, targets


@pytest.mark.parametrize("curve, x, want", [
    ("lin", 0.25, 0.25), ("exp", 0.5, 0.25), ("log", 0.25, 0.5), ("smooth", 0.25, 0.15625),
    ("bell", 0.5, 1.0), ("bell", 0.0, 0.0), ("bell", 1.0, 0.0), ("bell", 0.25, 0.5), ("inv", 0.25, 0.75),
    ("step:5", 0.3, 0.25), ("step:5", 0.4, 0.5), ("step:2", 0.49, 0.0), ("step:2", 0.51, 1.0),
])
def test_all_seven_curves(curve, x, want):
    assert C.curve(curve, x) == pytest.approx(want, abs=1e-9)


def test_curves_stay_in_0_1_and_are_exact_at_the_ends():
    for cv in C.CURVES:
        name = cv if cv != "step" else "step:4"
        for i in range(101):
            assert 0.0 <= C.curve(name, i / 100) <= 1.0
        if cv not in ("bell", "inv"):
            assert C.curve(name, 0.0) == 0.0 and C.curve(name, 1.0) == 1.0
    assert C.curve("inv", 0.0) == 1.0 and C.curve("inv", 1.0) == 0.0
    with pytest.raises(ValueError):
        C.check_curve("step:1")
    with pytest.raises(ValueError):
        C.check_curve("cubic")


def test_step_snapping_of_macro_and_stepped_targets():
    cl = C.validate_clip(clip(macros=[
        {"name": "Petals", "widget": "select", "default": 0.5, "targets": [{"p": "folds", "values": [4, 6, 8, 12, 16]}]},
        {"name": "Mode", "widget": "knob", "default": 0.0, "targets": [{"p": "travel_mode", "a": 0, "b": 3}]},
        {"name": "On", "widget": "toggle", "default": 0.0, "targets": [{"p": "wave", "a": 0.0, "b": 0.7}]},
    ]))
    assert C.macro_steps(cl["macros"][0]) == 5 and C.snap(cl["macros"][0], 0.6) == 0.5
    assert C.macro_values(cl, 0, 0.6) == {"folds": 8.0}
    assert C.macro_values(cl, 0, 0.9) == {"folds": 16.0}
    assert C.macro_values(cl, 1, 0.4) == {"travel_mode": 1.0}       # stepped param: whole numbers only
    assert C.macro_values(cl, 1, 0.55) == {"travel_mode": 2.0}
    assert C.snap(cl["macros"][2], 0.3) == 0.0 and C.snap(cl["macros"][2], 0.7) == 1.0
    assert C.macro_values(cl, 2, 0.7) == {"wave": pytest.approx(0.7)}


def test_macro_targets_outside_the_lane_are_refused():
    for p in ("serpent", "l_beings", "master", "flash", "strobe", "palette", "l_text", "no_such"):
        with pytest.raises(ValueError):
            C.validate_clip(clip(macros=[{"name": "X", "widget": "knob", "default": 0,
                                          "targets": [{"p": p, "a": 0, "b": 1}]}]))
    with pytest.raises(ValueError):                    # sets outside the lane too
        C.validate_clip(clip(sets={"being": 3}))
    with pytest.raises(ValueError):                    # a FIGURES clip cannot drive the world
        C.validate_clip(clip(lane="figures", sets={"l_beings": 1},
                             macros=[{"name": "X", "widget": "knob", "default": 0, "targets": [{"p": "zoom", "a": 0, "b": 1}]}]))
    with pytest.raises(ValueError):                    # one param, one owner inside a clip
        C.validate_clip(clip(macros=[{"name": "A", "widget": "knob", "default": 0, "targets": [{"p": "twirl", "a": 0, "b": 1}]},
                                     {"name": "B", "widget": "knob", "default": 0, "targets": [{"p": "twirl", "a": 0, "b": -1}]}]))
    with pytest.raises(ValueError):
        C.validate_clip(clip(sets={"twirl": 0.2}))     # fixed in sets AND driven by a macro
    with pytest.raises(ValueError):
        C.validate_clip(clip(macros=[dict(clip()["macros"][0], name=f"M{i}") for i in range(9)]))
    with pytest.raises(ValueError):
        C.validate_clip(clip(id="Bad Id"))


def test_validate_clip_cleans_and_is_idempotent():
    raw = clip(sets={"zoom": 5, "density": 0.3}, tags=["Chill", 3], junk=1,
               mods={"swim": [{"src": "bass", "amt": 0.5}], "hue": [{"src": "bass", "amt": 1}]})
    with pytest.raises(ValueError):                    # hue is a master param: no mods on another lane either
        C.validate_clip(raw)
    raw["mods"].pop("hue")
    a = C.validate_clip(raw)
    assert a["sets"]["zoom"] == 1.0 and "junk" not in a and a["tags"] == ["chill"]
    assert a["mods"]["swim"][0]["src"] == "bass" and a["mods"]["swim"][1] is None
    assert C.validate_clip(a) == a
    json.dumps(a, allow_nan=False)


# ---------------------------------------------------------------- resolve


def test_resolve_writes_every_owned_param_and_nothing_else():
    cl = C.validate_clip(clip())
    r = C.resolve(cl)
    assert set(r["updates"]) == set(C.LANE_PARAMS["base"])
    assert r["updates"]["zoom"] == 0.4 and r["updates"]["twirl"] == 0.0
    for n in C.LANE_PARAMS["base"]:                    # unlisted owned params go to their registry default
        if n not in ("zoom", "twirl"):
            assert r["updates"][n] == PARAMS[n][2], n
    assert r["stack"] == {"v": 1, "layers": []} or r["stack"]["layers"] == []      # BASE owns the math stack
    r = C.resolve(cl, [1.0])
    assert r["updates"]["twirl"] == pytest.approx(0.6)
    fig = C.validate_clip({"id": "f", "name": "F", "lane": "figures", "sets": {"l_beings": 1}})
    rf = C.resolve(fig)
    assert set(rf["updates"]) == set(C.LANE_PARAMS["figures"]) and rf["stack"] is None


# ---------------------------------------------------------------- factory clips


def test_first_five_factory_clips():
    ids = [c["id"] for c in C.FACTORY]
    assert ids[:5] == ["mandala", "crawl", "tube", "torus", "herd"]
    lanes = {c["id"]: c["lane"] for c in C.FACTORY}
    assert lanes["herd"] == "figures" and all(lanes[i] == "base" for i in ids[:4])
    for cl in C.FACTORY:
        assert C.validate_clip(cl) == cl
        assert 4 <= len(cl["macros"]) <= 8, cl["id"]
        names = [m["name"] for m in cl["macros"]]
        assert len(set(names)) == len(names) and all(2 <= len(n) <= 16 for n in names)
        for i, m in enumerate(cl["macros"]):           # every macro really moves something across its travel
            lo, hi = C.macro_values(cl, i, 0.0), C.macro_values(cl, i, 1.0)
            assert lo != hi, (cl["id"], m["name"])


def test_golden_mandala_clip_at_defaults_is_the_default_look(tmp_path):
    """The default show look must not move: Mandala at its default macros = a fresh controller, value for value."""
    r = C.resolve(C.get("mandala"))
    for n, v in r["updates"].items():
        assert v == PARAMS[n][2], f"Mandala at defaults moves {n}: {v} != {PARAMS[n][2]}"
    ref, c = make(tmp_path / "a"), make(tmp_path / "b")
    do(c, cmd="clip", op="launch", id="mandala", now=True)
    a, b = run(ref, 0.5), run(c, 0.5)
    assert a == b
    assert c.comp["base"]["clip"] == "mandala" and c.stack.to_json()["layers"] == []


# ---------------------------------------------------------------- controller: launch


def test_base_launch_never_touches_figures_or_master(tmp_path):
    c = make(tmp_path)
    do(c, cmd="clip", op="lock", lane="base", palette=True)     # palette is a master param: it follows only unlocked
    before = {n: c.base[n] for n in PARAMS if C.lane_of(n) != "base"}
    for n in before:
        if n not in TRIGGERS:
            lo, hi = PARAMS[n][0], PARAMS[n][1]
            c.base[n] = lo + (hi - lo) * 0.37
    snap = {n: c.base[n] for n in before}
    for cid in ("mandala", "crawl", "tube", "torus"):
        do(c, cmd="clip", op="launch", id=cid, now=True)
        run(c, 0.05)
        assert {n: c.base[n] for n in before} == snap, cid


def test_launch_puts_unlisted_owned_params_back_to_defaults(tmp_path):
    c = make(tmp_path)
    c.set("twirl", 0.9)
    c.set("l_fractal", 1.0)
    c.set("fractal_mode", 3)
    do(c, cmd="clip", op="launch", id="crawl", now=True)
    run(c)
    assert c.base["twirl"] == 0.0 and c.base["l_fractal"] == 0.0 and c.base["fractal_mode"] == 0.0
    assert c.base["symmetry"] == 0.0


def test_launch_morphs_per_lane_and_two_lanes_do_not_freeze_each_other(tmp_path):
    c = make(tmp_path)
    do(c, cmd="clip", op="launch", id="tube", now=True, bars=1)
    run(c, 0.5)                                         # 120 bpm default? 124: one bar = ~1.9 s
    mid = c.base["travel"]
    assert 0.0 < mid < C.resolve(C.get("tube"))["updates"]["travel"]
    do(c, cmd="clip", op="launch", id="herd", now=True, bars=1)
    run(c, 0.5)
    assert c.base["travel"] > mid                       # the BASE morph keeps going
    run(c, 4.0)
    assert c.base["travel"] == pytest.approx(C.resolve(C.get("tube"))["updates"]["travel"])
    assert c.base["l_beings"] == 1.0


def test_launch_does_not_change_bypass(tmp_path):
    c = make(tmp_path)
    do(c, cmd="bypass", key="l_beads", on=True)
    do(c, cmd="bypass", key="motion", on=True)
    do(c, cmd="clip", op="launch", id="mandala", now=True)
    do(c, cmd="clip", op="launch", id="herd", now=True)
    do(c, cmd="clip", op="stop", lane="figures")
    assert c.bypass == {"l_beads", "motion"}


def test_launch_is_quantized_like_a_scene(tmp_path):
    c = make(tmp_path)
    c.quantize = "bar"
    c.beat = 0.5
    do(c, cmd="clip", op="launch", id="crawl")
    assert c.base["symmetry"] == 1.0 and c.comp["base"]["clip"] is None
    run(c, 2.5)                                         # past beat 4 at 124 bpm
    assert c.base["symmetry"] == 0.0 and c.comp["base"]["clip"] == "crawl"


def test_palette_follows_a_base_clip_unless_locked(tmp_path):
    c = make(tmp_path)
    do(c, cmd="clip", op="launch", id="crawl", now=True)
    assert c.base["palette"] == C.get("crawl")["palette"]
    c.set("palette", 1.0)
    do(c, cmd="clip", op="lock", lane="base", palette=True)
    do(c, cmd="clip", op="launch", id="crawl", now=True)
    assert c.base["palette"] == 1.0
    do(c, cmd="clip", op="launch", id="mandala", now=True)       # no palette in the clip: never touched
    assert c.base["palette"] == 1.0


def test_stop_takes_the_lane_off_the_wall(tmp_path):
    c = make(tmp_path)
    do(c, cmd="clip", op="launch", id="herd", now=True)
    assert c.base["l_beings"] == 1.0
    do(c, cmd="clip", op="stop", lane="figures", now=True)
    assert c.base["l_beings"] == 0.0 and c.base["serpent"] == 0.0 and c.comp["figures"]["clip"] is None


def test_launch_replaces_lane_mods_only(tmp_path):
    c = make(tmp_path)
    c.set_mod("swim", 0, "bass", 0.5)
    c.set_mod("hue", 0, "lfo_sine", 0.2)
    c.set_mod("being_size", 0, "bass", 0.2)
    do(c, cmd="clip", op="launch", id="mandala", now=True)
    assert "swim" not in c.mods and "hue" in c.mods and "being_size" in c.mods


# ---------------------------------------------------------------- controller: macros


def test_macro_moves_write_targets_and_count_as_edits(tmp_path):
    c = make(tmp_path)
    do(c, cmd="clip", op="launch", id="mandala", now=True)
    e0 = dict(c.edits)
    i = next(k for k, m in enumerate(C.get("mandala")["macros"]) if m["name"] == "Melt")
    do(c, cmd="macro", lane="base", n=i + 1, v=1.0)
    want = C.macro_values(C.get("mandala"), i, 1.0)
    for p, v in want.items():
        assert c.base[p] == pytest.approx(v)
        assert c.edits.get(p, 0) == e0.get(p, 0) + 1        # the user wins over the governor
    assert c.comp["base"]["macros"][i] == 1.0


def test_macro_on_an_empty_lane_or_bad_index_does_nothing(tmp_path):
    c = make(tmp_path)
    base = dict(c.base)
    do(c, cmd="macro", lane="figures", n=1, v=1.0)
    do(c, cmd="clip", op="launch", id="herd", now=True)
    do(c, cmd="macro", lane="figures", n=8, v=1.0)          # herd has 7 macros
    assert {n: c.base[n] for n in base if C.lane_of(n) == "base"} == {n: base[n] for n in base if C.lane_of(n) == "base"}
    with pytest.raises(ValueError):
        control.validate({"cmd": "macro", "lane": "master", "n": 1, "v": 0.5})
    with pytest.raises(ValueError):
        control.validate({"cmd": "macro", "lane": "base", "n": 9, "v": 0.5})


def test_fader_move_on_a_target_marks_the_macro_overridden(tmp_path):
    c = make(tmp_path)
    do(c, cmd="clip", op="launch", id="mandala", now=True)
    i = next(k for k, m in enumerate(C.get("mandala")["macros"]) if m["name"] == "Bloom")
    do(c, cmd="set", name="zoom", v=0.9)
    assert c.comp["base"]["over"][i] is True
    do(c, cmd="macro", lane="base", n=i + 1, v=0.2)
    assert c.comp["base"]["over"][i] is False and c.base["zoom"] != 0.9


def test_macro_midi_target_plays_whatever_is_on_air(tmp_path):
    c = make(tmp_path)
    assert control.valid_target("macro:base:3") and control.valid_target("clip:mandala")
    assert not control.valid_target("macro:master:1") and not control.valid_target("macro:base:9")
    assert not control.valid_target("clip:Bad Id")
    c.midi.bindings = [dict(type="cc", ch=0, num=21, target="macro:base:1", port=None),
                       dict(type="note", ch=0, num=40, target="clip:crawl", port=None)]
    c.midi.inject("note_on", 0, 40, 127)
    run(c)
    assert c.comp["base"]["clip"] == "crawl"
    c.midi.inject("cc", 0, 21, 127)
    run(c)
    assert c.comp["base"]["macros"][0] == 1.0
    m0 = C.macro_values(C.get("crawl"), 0, 1.0)
    assert all(c.base[p] == pytest.approx(v) for p, v in m0.items())


def test_comp_state_is_compact_and_survives_a_restart(tmp_path):
    c = make(tmp_path, restore=True)
    do(c, cmd="clip", op="launch", id="tube", now=True)
    do(c, cmd="macro", lane="base", n=1, v=0.25)
    s = c.state()["comp"]
    assert s["base"]["clip"] == "tube" and len(json.dumps(s)) < 600
    c._autosave()
    c2 = make(tmp_path, restore=True)
    assert c2.comp["base"]["clip"] == "tube" and c2.comp["base"]["macros"][0] == 0.25


def test_scene_recall_and_panic_clear_what_is_on_air(tmp_path):
    c = make(tmp_path)
    do(c, cmd="clip", op="launch", id="tube", now=True)
    do(c, cmd="scene_recall", i=1, now=True)
    assert c.comp["base"]["clip"] is None
    do(c, cmd="clip", op="launch", id="herd", now=True)
    c._clip_pending.append(dict(fire=1e9, op="launch", lane="base", id="crawl", macros=None, bars=0.0))
    c.panic()
    assert c.comp["figures"]["clip"] is None and c._clip_pending == [] and not c._lane_morph


# ---------------------------------------------------------------- user clips, decks, scene_go


def test_user_clip_save_delete_and_factory_ids_are_reserved(tmp_path):
    c = make(tmp_path)
    do(c, cmd="clip", op="save", clip=clip(id="mine"))
    assert "mine" in c.clips
    do(c, cmd="clip", op="launch", id="mine", now=True)
    assert c.base["zoom"] == 0.4
    with pytest.raises(ValueError):
        control.validate({"cmd": "clip", "op": "save", "clip": clip(id="mandala")})
    do(c, cmd="clip", op="delete", id="mine")
    assert "mine" not in c.clips


def test_scene_go_fires_a_column(tmp_path):
    c = make(tmp_path)
    deck = {"id": "warmup", "name": "Warm-up", "columns": [
        {"id": "drop", "name": "Drop", "lanes": {"base": {"clip": "tube", "macros": {"1": 1.0}}, "figures": {"clip": "herd"},
                                                 "text": "stop", "interact": "keep"}},
        {"id": "calm", "name": "Calm", "lanes": {"base": {"clip": "mandala"}, "figures": "stop"}}]}
    do(c, cmd="deck", op="save", deck=deck)
    do(c, cmd="scene_go", id="drop", now=True)
    assert c.comp["base"]["clip"] == "tube" and c.comp["figures"]["clip"] == "herd"
    assert c.comp["base"]["macros"][0] == 1.0
    do(c, cmd="scene_go", id="calm", now=True)
    assert c.comp["base"]["clip"] == "mandala" and c.comp["figures"]["clip"] is None and c.base["l_beings"] == 0.0
    with pytest.raises(ValueError):                     # column ids are unique across decks
        C.validate_decks([deck, dict(deck, id="other")])


# ---------------------------------------------------------------- undo


def test_undo_coalesces_a_gesture(tmp_path):
    c = make(tmp_path)
    z0 = c.base["zoom"]
    for i in range(20):                                 # one drag = many sets with the same gesture id
        do(c, cmd="set", name="zoom", v=i / 40, g=7)
    do(c, cmd="set", name="zoom", v=0.9, g=8)
    assert len(c.history) == 2
    do(c, cmd="history", op="undo")
    assert c.base["zoom"] == pytest.approx(19 / 40)
    do(c, cmd="history", op="undo")
    assert c.base["zoom"] == z0
    do(c, cmd="history", op="redo")
    assert c.base["zoom"] == pytest.approx(19 / 40)


def test_undo_of_a_launch_and_a_macro(tmp_path):
    c = make(tmp_path)
    c.set("twirl", 0.5)
    before = dict(c.base)
    do(c, cmd="clip", op="launch", id="crawl", now=True)
    do(c, cmd="macro", lane="base", n=4, v=1.0, g=3)
    do(c, cmd="macro", lane="base", n=4, v=0.9, g=3)
    do(c, cmd="history", op="undo")
    assert c.comp["base"]["macros"][3] == pytest.approx(C.get("crawl")["macros"][3]["default"])
    do(c, cmd="history", op="undo")
    assert c.base == before and c.comp["base"]["clip"] is None


def test_undo_history_is_capped_and_ungestured_sets_coalesce_by_time(tmp_path):
    c = make(tmp_path)
    for i in range(80):
        c.ft.t += 1.0
        do(c, cmd="set", name="zoom", v=(i % 10) / 10)
    assert len(c.history) == control.HISTORY_MAX
    n = len(c.history)
    for i in range(5):                                  # quick nudges of one fader without a gesture id: one step
        c.ft.t += 0.05
        do(c, cmd="set", name="hue", v=i / 10)
    assert len(c.history) == n                          # capped, so the count stays; the newest step is the hue drag
    assert set(c.history[-1]["old"]) == {"hue"}


# ---------------------------------------------------------------- validation (server thread) + timeline ids


def test_remote_validation_of_the_new_commands():
    v = remote_control.validate_command
    assert v({"cmd": "clip", "op": "launch", "id": "mandala"})["id"] == "mandala"
    assert v({"cmd": "macro", "lane": "base", "n": 2, "v": 7})["v"] == 1.0
    assert v({"cmd": "scene_go", "id": "drop"})["id"] == "drop"
    assert v({"cmd": "history", "op": "undo"})["op"] == "undo"
    assert v({"cmd": "set", "name": "zoom", "v": 0.1, "g": 12})["g"] == 12
    for bad in ({"cmd": "clip", "op": "launch", "id": "../x"}, {"cmd": "clip", "op": "boom"},
                {"cmd": "macro", "lane": "base", "n": 0, "v": 0}, {"cmd": "scene_go", "id": 3},
                {"cmd": "history", "op": "wipe"}, {"cmd": "set", "name": "zoom", "v": 0, "g": [1]},
                {"cmd": "clip", "op": "save", "clip": {"id": "x", "lane": "base", "sets": {"master": 0}}}):
        with pytest.raises(ValueError):
            v(bad)


def test_timeline_can_carry_clip_scene_go_and_macro_events(tmp_path):
    assert {"clip", "scene_go", "macro"} <= set(control.TL_EVENTS)
    c = make(tmp_path)
    ev = c.tl.first("event") or None
    do(c, cmd="tl", op="track_add", kind="event")
    tr = c.tl.first("event")["id"]
    do(c, cmd="tl", op="items_add", track=tr, items=[{"t": 1, "cmd": {"cmd": "clip", "op": "launch", "id": "crawl",
                                                                       "now": True}}])
    do(c, cmd="tl", op="play")
    run(c, 1.5)
    assert c.comp["base"]["clip"] == "crawl"
    assert ev is None or ev


# ---------------------------------------------------------------- show files


def test_show_round_trip_carries_clips_decks_and_comp(tmp_path):
    c = make(tmp_path, restore=True)
    do(c, cmd="clip", op="save", clip=clip(id="mine"))
    do(c, cmd="deck", op="save", deck={"id": "d1", "name": "D", "columns": [{"id": "c1", "name": "C", "lanes": {"base": {"clip": "mine"}}}]})
    do(c, cmd="clip", op="launch", id="tube", now=True)
    do(c, cmd="show", op="save_as", name="Clips")
    run(c)
    m = json.loads((tmp_path / "shows" / "clips" / "show.json").read_text())
    assert m["version"] == S.SHOW_VERSION == 2
    st = m["controller"]["settings"]
    assert st["clips"][0]["id"] == "mine" and st["decks"][0]["id"] == "d1" and st["comp"]["base"]["clip"] == "tube"
    do(c, cmd="clip", op="delete", id="mine")
    do(c, cmd="deck", op="delete", id="d1")
    do(c, cmd="clip", op="launch", id="mandala", now=True)
    do(c, cmd="show", op="open", slug="clips")
    run(c)
    assert "mine" in c.clips and c.decks[0]["id"] == "d1" and c.comp["base"]["clip"] == "tube"


def test_v1_show_migrates_keeping_all_16_snapshots_and_cue_notes(tmp_path):
    scenes = [control._scene(f"S{i}", zoom=i / 20) for i in range(16)]
    v1 = {"format": "entheo-show", "version": 1, "name": "Old",
          "controller": {"base": {"zoom": 0.3}, "scenes": scenes, "cur": 4,
                         "settings": {"cue_ch": 9, "safe": 2}},
          "timeline": {"v": 1, "tracks": [{"id": 1, "kind": "cue", "name": "Cues", "mute": False,
                                           "items": [{"id": 2, "t": 4, "kind": "scene", "slot": 15, "morph": 0}]}]}}
    m = S.migrate_show(json.loads(json.dumps(v1)))
    assert m["version"] == 2 and m["controller"]["settings"]["clips"] == [] and m["controller"]["settings"]["decks"] == []
    plan = S.clean_show(m)
    assert [s["name"] for s in plan["scenes"]] == [f"S{i}" for i in range(16)]
    assert plan["settings"]["cue_ch"] == 9 and plan["settings"]["safe"] == 2
    cue = next(t for t in plan["timelines"][0]["tracks"] if t["kind"] == "cue")
    assert cue["items"][0]["slot"] == 15                # scene cues still address snapshots by slot number
    d = tmp_path / "shows" / "old"
    d.mkdir(parents=True)
    (d / "show.json").write_text(json.dumps(v1))
    c = make(tmp_path, restore=True)
    do(c, cmd="show", op="open", slug="old")
    run(c)
    assert all(c.scenes[i]["name"] == f"S{i}" for i in range(16)) and c.cue_ch == 9
    c.quantize = "off"
    c._cue("note_on", 3, 100)                           # cue channel note 3 still recalls snapshot 4
    assert c.cur == 3


def test_corrupt_show_is_quarantined_and_corrupt_clips_are_dropped(tmp_path):
    d = tmp_path / "shows" / "bad"
    d.mkdir(parents=True)
    (d / "show.json").write_text("{nope")
    c = make(tmp_path, restore=True)
    c.set("zoom", -0.7)
    do(c, cmd="show", op="open", slug="bad")
    run(c)
    assert c.base["zoom"] == -0.7 and list(d.glob("show.json.*.bad"))
    good = C.validate_clip(clip(id="ok"))
    plan_clips = C.clean_clips([good, {"id": "x", "lane": "base", "sets": {"master": 1}}, "junk", dict(good)])
    assert [x["id"] for x in plan_clips] == ["ok"]      # bad and duplicate entries dropped, the rest kept
