"""Timeline: model, validation, evaluator, transport, live override, recording, photosensitive limits, PANIC,
determinism. Pure python (no GL): python -m pytest -q tests/test_timeline.py"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import control  # noqa: E402
import timeline as T  # noqa: E402
from params import PARAMS  # noqa: E402

DT = 1 / 60


def make(tmp_path, **kw):
    kw.setdefault("threads", False)
    c = control.Controller(tmp_path, restore=kw.pop("restore", True), midi=False, ai=False, bpm=120.0, seed=1, **kw)
    c.clock_src = "internal"
    return c


def do(c, **cmd):
    c.apply([cmd])


def tl(c, **cmd):
    c.apply([dict(cmd, cmd="tl")])


def run(c, seconds, bands=(0.0, 0.0, 0.0)):
    for _ in range(int(round(seconds / DT))):
        c.update(DT, bands)


def lane(c, param):
    tl(c, op="track_add", kind="auto", param=param)
    return c.tl.track(c.tl.by_param[param].tid)


def keys(c, tr, pts):
    tl(c, op="items_add", track=tr["id"], items=[dict(t=t, v=v, ease=e) for t, v, e in pts])


# ------------------------------------------------------------------ model + validation


def test_ease_and_lane_value_are_pure():
    ctx = T.Ctx({"zoom": PARAMS["zoom"]})
    t = T.Timeline(ctx)
    tr = {"id": 9, "kind": "auto", "param": "zoom", "mute": False,
          "items": [dict(id=1, t=0.0, v=0.0, ease="linear"), dict(id=2, t=4.0, v=1.0, ease="hold"),
                    dict(id=3, t=8.0, v=-1.0, ease="smooth")]}
    ln = T.Lane(tr, -1, 1)
    assert T.lane_value(ln, -5) == 0.0
    assert T.lane_value(ln, 2.0) == pytest.approx(0.5)
    assert T.lane_value(ln, 6.0) == 1.0           # hold
    assert T.lane_value(ln, 100) == -1.0
    assert [T.ease(i, 0.5) for i in range(5)] == [0.0, 0.5, 0.5, 0.25, 0.75]
    assert t.lanes == []


def test_clean_drops_garbage_and_never_raises():
    ctx = control.TL_CTX
    for raw in (None, 3, [], "x", {"tracks": "no"}, {"tracks": [1, {"kind": "nope"}, {"kind": "auto", "param": "nope"}]}):
        d = T.clean(raw, ctx)
        assert [t["kind"] for t in d["tracks"]] == ["marker", "cue"]
    d = T.clean({"units": "seconds", "length": float("nan"), "tracks": [
        {"kind": "auto", "param": "zoom", "items": [{"t": 1, "v": 99}, {"t": "x", "v": 0}, {"t": 2, "v": 0, "ease": "bogus"}]},
        {"kind": "auto", "param": "zoom", "items": []},                          # a second lane for zoom is dropped
        {"kind": "event", "items": [{"t": 1, "cmd": {"cmd": "panic"}}, {"t": 2, "cmd": {"cmd": "camera", "op": "on"}},
                                    {"t": 3, "cmd": {"cmd": "bypass", "key": "l_kene", "on": True}}]},
        {"kind": "cue", "items": [{"t": 0, "kind": "scene", "slot": 99}, {"t": 1, "kind": "preset", "preset": "../x"},
                                  {"t": 2, "kind": "scene", "slot": 3, "morph": 2}]}]}, ctx)
    autos = [t for t in d["tracks"] if t["kind"] == "auto"]
    assert len(autos) == 1 and [it["v"] for it in autos[0]["items"]] == [1.0]   # clamped; bad items dropped
    ev = [t for t in d["tracks"] if t["kind"] == "event"][0]
    assert [it["cmd"]["cmd"] for it in ev["items"]] == ["bypass"]               # no panic, no webcam from a file
    cue = [t for t in d["tracks"] if t["kind"] == "cue"][0]
    assert [it["slot"] for it in cue["items"]] == [3]
    ids = [o["id"] for t in d["tracks"] for o in [t] + t["items"]]
    assert len(ids) == len(set(ids)) and d["next"] > max(ids)
    json.dumps(d, allow_nan=False)


def test_validate_rejects_bad_commands():
    ctx = control.TL_CTX
    for c in ({"op": "nope"}, {"op": "seek", "t": float("inf")}, {"op": "track_add", "kind": "auto", "param": "flash"},
              {"op": "items_set", "track": 1, "items": [{"t": 1}]}, {"op": "rec", "on": 1}, {"op": "units", "units": "ms"}):
        with pytest.raises(ValueError):
            T.validate(dict(c, cmd="tl"), ctx)
    with pytest.raises(ValueError):
        control.validate({"cmd": "tl", "op": "item_add", "track": 2, "item": "x"})


# ------------------------------------------------------------------ transport + automation


def test_automation_drives_its_own_layer_and_release_gives_the_fader_back(tmp_path):
    c = make(tmp_path)
    tr = lane(c, "zoom")
    keys(c, tr, [(0, -0.5, "linear"), (8, 0.5, "linear")])
    c.set("zoom", 0.2)
    tl(c, op="play")
    run(c, 2.0)                                       # 120 bpm: 4 beats
    assert c.tl.pos == pytest.approx(4.0, abs=0.05)
    assert c.values["zoom"] == pytest.approx(0.0, abs=0.03)
    assert c.base["zoom"] == 0.2                      # the fader is never written by the timeline
    tl(c, op="stop")
    run(c, 0.5)
    assert c.values["zoom"] == pytest.approx(0.0, abs=0.03)   # stopped: the automation value holds
    tl(c, op="release")
    run(c, 1.5)
    assert c.values["zoom"] == pytest.approx(0.2, abs=1e-3) and not c.tl_engaged


def test_follows_the_beat_clock_and_loops(tmp_path):
    c = make(tmp_path)
    tl(c, op="loop", on=True, a=0, b=4)
    tl(c, op="play")
    run(c, 2.5)                                       # 5 beats -> wrapped once
    assert c.tl.playing and c.tl.pos == pytest.approx(1.0, abs=0.05)
    do(c, cmd="bpm", v=60.0)
    p = c.tl.pos
    run(c, 1.0)
    assert c.tl.pos - p == pytest.approx(1.0, abs=0.05)
    do(c, cmd="resync")                               # the clock jumps back: the playhead runs on at tempo
    p = c.tl.pos
    run(c, 0.5)
    assert c.tl.pos - p == pytest.approx(0.5, abs=0.05)


def test_seconds_units_and_conversion(tmp_path):
    c = make(tmp_path)
    tr = lane(c, "zoom")
    keys(c, tr, [(8, 1.0, "linear")])
    tl(c, op="units", units="seconds")                # 120 bpm: beat 8 = 4 s
    assert c.tl.units == "seconds" and c.tl.track(tr["id"])["items"][0]["t"] == pytest.approx(4.0)
    tl(c, op="play")
    run(c, 1.0)
    assert c.tl.pos == pytest.approx(1.0, abs=0.02)


def test_cues_fire_once_exactly_where_placed_and_end_stops(tmp_path):
    c = make(tmp_path)
    do(c, cmd="quantize", q="phrase")                 # timeline cues ignore the recall quantize
    cue = c.tl.first("cue")
    tl(c, op="items_add", track=cue["id"], items=[dict(t=2, kind="scene", slot=2, morph=0), dict(t=4, kind="scene", slot=5, morph=0)])
    tl(c, op="length", t=6)
    tl(c, op="play")
    run(c, 1.1)
    assert c.cur == 2
    run(c, 1.0)
    assert c.cur == 5
    run(c, 2.0)
    assert not c.tl.playing and c.tl.pos == 6


def test_chase_on_seek_recalls_the_latest_cue_but_never_a_flash(tmp_path):
    c = make(tmp_path)
    cue = c.tl.first("cue")
    tl(c, op="items_add", track=cue["id"], items=[dict(t=2, kind="scene", slot=3, morph=1)])
    tl(c, op="track_add", kind="trigger")
    trg = c.tl.first("trigger")
    tl(c, op="items_add", track=trg["id"], items=[dict(t=1, len=8, name="flash")])
    tl(c, op="seek", t=5)
    run(c, 0.05)
    assert c.cur == 3                                 # chased
    assert c.values["flash"] == 0.0                   # stopped: spans do nothing


def test_trigger_spans_are_declarative_and_photosensitive_limited(tmp_path):
    c = make(tmp_path)
    tl(c, op="track_add", kind="trigger")
    trg = c.tl.first("trigger")
    tl(c, op="items_add", track=trg["id"], items=[dict(t=0.5, len=0.25, name="flash"), dict(t=0.75, len=0.25, name="flash"),
                                                 dict(t=2, len=1, name="freeze")])
    tl(c, op="play")
    run(c, 0.3)                                       # beat 0.6: inside the first flash
    assert c.values["flash"] == 1.0
    run(c, 0.15)                                      # beat 0.9: second flash only 0.125 s later -> held back
    assert c.tl_limited == 1 and "flash" not in c._tl_trig
    tl(c, op="seek", t=2.5)
    run(c, 0.05)
    assert c.values["freeze"] == 1.0
    tl(c, op="stop")
    run(c, 0.05)
    assert c.values["freeze"] == 0.0                  # stop / seek can never leave a trigger stuck on


def test_strobe_on_the_timeline_respects_the_safe_rate(tmp_path):
    c = make(tmp_path)
    do(c, cmd="strobe_rate", v=0.125)                 # 120 bpm: 16 flashes / s
    tl(c, op="track_add", kind="trigger")
    trg = c.tl.first("trigger")
    tl(c, op="items_add", track=trg["id"], items=[dict(t=0.1, len=4, name="strobe")])
    tl(c, op="play")
    run(c, 0.5)
    assert c.tl_limited == 1 and c.values["strobe"] == 0.0


def test_luminance_lanes_are_slew_limited(tmp_path):
    c = make(tmp_path)
    tr = lane(c, "master")
    keys(c, tr, [(0, 1.0, "hold"), (0.1, 0.0, "hold"), (0.2, 1.0, "hold"), (0.3, 0.0, "hold")])   # 20 Hz square
    tl(c, op="play")
    vals = []
    for _ in range(40):
        c.update(DT, (0, 0, 0))
        vals.append(c._auto["master"])
    steps = [abs(b - a) for a, b in zip(vals, vals[1:])]
    assert max(steps) <= control.TL_LUMA_RATE * DT + 1e-9


def test_events_text_and_bypass_fire_and_bypass_is_honoured(tmp_path):
    c = make(tmp_path)
    tl(c, op="track_add", kind="event")
    ev = c.tl.first("event")
    tl(c, op="items_add", track=ev["id"], items=[dict(t=1, cmd={"cmd": "bypass", "key": "l_kene", "on": True}),
                                                dict(t=1.5, cmd={"cmd": "next_video"})])
    tr = lane(c, "l_kene")
    keys(c, tr, [(0, 1.0, "linear")])
    tl(c, op="play")
    out = []
    for _ in range(int(2 / DT)):
        c.update(DT, (0, 0, 0))
        out += c.apply([])
    assert "l_kene" in c.bypass and c.values["l_kene"] == pytest.approx(0.0, abs=1e-6)   # bypass wins over automation
    assert any(x.get("cmd") == "next_video" for x in out)


def test_panic_stops_the_timeline_and_lets_go(tmp_path):
    c = make(tmp_path)
    tr = lane(c, "zoom")
    keys(c, tr, [(0, 0.9, "linear")])
    tl(c, op="track_add", kind="trigger")
    tl(c, op="items_add", track=c.tl.first("trigger")["id"], items=[dict(t=0, len=16, name="freeze")])
    tl(c, op="rec", on=True)
    tl(c, op="play")
    run(c, 0.5)
    assert c.values["freeze"] == 1.0 and c._auto
    do(c, cmd="panic")
    run(c, 0.1)
    assert not c.tl.playing and not c.tl.rec and not c._auto and c.values["freeze"] == 0.0


def test_governor_and_bypass_paths_untouched(tmp_path):
    c = make(tmp_path)
    tr = lane(c, "fractal_res") if "fractal_res" in PARAMS else lane(c, "zoom")
    keys(c, tr, [(0, 0.3, "linear")])
    tl(c, op="play")
    run(c, 0.5)
    assert c.edits == {}                              # the governor's "user moved it" signal is not touched


# ------------------------------------------------------------------ live override + recording


def test_touch_override_wins_while_held_then_glides_back(tmp_path):
    c = make(tmp_path)
    tr = lane(c, "zoom")
    keys(c, tr, [(0, 0.5, "linear")])
    tl(c, op="play")
    run(c, 0.2)
    do(c, cmd="set", name="zoom", v=-0.8)
    run(c, 0.5)
    assert c.values["zoom"] == pytest.approx(-0.8, abs=1e-3) and "zoom" in c._tl_ovr
    run(c, control.TL_HOLD + control.TL_GLIDE + 0.3)
    assert c.values["zoom"] == pytest.approx(0.5, abs=1e-3) and not c._tl_ovr


def test_latch_override_waits_for_reenable(tmp_path):
    c = make(tmp_path)
    tr = lane(c, "zoom")
    keys(c, tr, [(0, 0.5, "linear")])
    tl(c, op="ovr_mode", mode="latch")
    tl(c, op="play")
    run(c, 0.1)
    do(c, cmd="set", name="zoom", v=-0.8)
    run(c, 4.0)
    assert c.values["zoom"] == pytest.approx(-0.8, abs=1e-3)
    tl(c, op="reenable")
    run(c, control.TL_GLIDE + 0.5)
    assert c.values["zoom"] == pytest.approx(0.5, abs=1e-3)


def test_record_fader_moves_scene_recalls_and_triggers(tmp_path):
    c = make(tmp_path)
    tl(c, op="rec", on=True)
    tl(c, op="play")
    for i in range(60):                               # a 1 s sweep of zoom
        c.set("zoom", -1 + 2 * i / 59)
        c.update(DT, (0, 0, 0))
    do(c, cmd="scene_recall", i=4, now=True)
    do(c, cmd="trigger", name="punch", on=True)
    run(c, 0.25)
    do(c, cmd="trigger", name="punch", on=False)
    run(c, control.TL_HOLD + 0.2)
    tl(c, op="stop")
    zl = c.tl.track(c.tl.by_param["zoom"].tid)["items"]
    assert 2 <= len(zl) < 30                          # thinned (RDP)
    ln = c.tl.by_param["zoom"]
    assert T.lane_value(ln, 0.5) == pytest.approx(-1 + 2 * 15 / 59, abs=0.08)
    cues = c.tl.first("cue")["items"]
    assert [(x["slot"]) for x in cues] == [4]
    trg = c.tl.first("trigger")["items"]
    assert trg[0]["name"] == "punch" and trg[0]["len"] == pytest.approx(0.5, abs=0.06)
    tl(c, op="undo")                                  # one undo step for the whole pass
    assert "zoom" not in c.tl.by_param and not c.tl.first("cue")["items"]


def test_record_touch_keeps_the_old_curve_outside_the_gesture(tmp_path):
    c = make(tmp_path)
    tr = lane(c, "zoom")
    keys(c, tr, [(0, 0.0, "linear"), (16, 0.0, "linear")])
    tl(c, op="rec", on=True)
    tl(c, op="play")
    run(c, 1.0)                                       # beat 2
    for _ in range(15):
        c.set("zoom", 0.9)
        c.update(DT, (0, 0, 0))
    run(c, control.TL_HOLD + 0.5)
    tl(c, op="stop")
    ln = c.tl.by_param["zoom"]
    assert T.lane_value(ln, 1.0) == pytest.approx(0.0, abs=1e-6)
    assert T.lane_value(ln, 2.3) == pytest.approx(0.9, abs=1e-6)
    assert T.lane_value(ln, 12.0) == pytest.approx(0.0, abs=1e-6)


def test_punch_region_limits_recording_and_replace_clears_cues(tmp_path):
    c = make(tmp_path)
    cue = c.tl.first("cue")
    tl(c, op="items_add", track=cue["id"], items=[dict(t=1, kind="scene", slot=1), dict(t=6, kind="scene", slot=2)])
    tl(c, op="punch", on=True, a=4, b=8)
    tl(c, op="rec_mode", mode="replace")
    tl(c, op="rec", on=True)
    tl(c, op="play")
    c.set("zoom", 0.5)                                # beat 0: outside the punch -> not recorded
    run(c, 4.0)                                       # to beat 8
    assert "zoom" not in c.tl.by_param
    assert [x["slot"] for x in c.tl.first("cue")["items"]] == [1]   # the cue at 6 was inside the punch: replaced


def test_items_edit_undo_redo(tmp_path):
    c = make(tmp_path)
    m = c.tl.first("marker")
    tl(c, op="items_add", track=m["id"], items=[dict(t=0, name="intro"), dict(t=32, name="drop")])
    ids = [x["id"] for x in c.tl.first("marker")["items"]]
    tl(c, op="items_set", track=m["id"], items=[dict(id=ids[1], t=64)])
    assert [x["t"] for x in c.tl.first("marker")["items"]] == [0, 64]
    tl(c, op="items_del", track=m["id"], ids=[ids[0]])
    tl(c, op="undo")
    tl(c, op="undo")
    assert [x["t"] for x in c.tl.first("marker")["items"]] == [0, 32]
    tl(c, op="redo")
    assert [x["t"] for x in c.tl.first("marker")["items"]] == [0, 64]
    c.tl.seek(40)
    assert c.tl.section() == "intro"


def test_several_timelines_per_show(tmp_path):
    c = make(tmp_path)
    tl(c, op="name", name="Song A")
    tl(c, op="tl_new")
    assert c.tl_i == 1 and len(c.tl_docs) == 2
    tl(c, op="name", name="Song B")
    tl(c, op="tl_select", i=0)
    assert c.tl.data["name"] == "Song A"
    assert [x for x in c.state()["tl"]["timelines"]] == ["Song A", "Song B"]


def test_state_and_blob(tmp_path):
    c = make(tmp_path)
    st = c.state()
    json.dumps(st, allow_nan=False)
    rev, data = control.blob("timeline")
    assert rev == c.tl.rev and json.loads(data)["v"] == 1


# ------------------------------------------------------------------ determinism


def _drive(tmp_path):
    c = make(tmp_path)
    c.rng.seed(7)
    tr = lane(c, "zoom")
    keys(c, tr, [(0, -1, "smooth"), (4, 1, "in"), (8, 0, "out"), (12, 0.4, "hold")])
    tr2 = lane(c, "hue")
    keys(c, tr2, [(0, 0, "linear"), (16, 1, "linear")])
    cue = c.tl.first("cue")
    tl(c, op="items_add", track=cue["id"], items=[dict(t=3, kind="scene", slot=2, morph=1), dict(t=9, kind="scene", slot=6, morph=0.5)])
    tl(c, op="track_add", kind="trigger")
    tl(c, op="items_add", track=c.tl.first("trigger")["id"], items=[dict(t=5, len=0.5, name="punch")])
    tl(c, op="loop", on=True, a=2, b=14)
    tl(c, op="play")
    frames = []
    dts = [1 / 60, 1 / 59, 1 / 61, 1 / 30, 1 / 120]
    for i in range(1500):
        f = c.update(dts[i % 5], (0.3, 0.2, 0.1))
        frames.append(tuple(f.values.values()))
    return frames


def test_same_timeline_twice_gives_identical_frames(tmp_path):
    a = _drive(tmp_path / "a")
    b = _drive(tmp_path / "b")
    assert a == b
    assert len(set(a)) > 100 and all(all(math.isfinite(x) for x in f) for f in a)


# ------------------------------------------------------------------ Neon review regressions (research/timeline_review_neon.md)


def test_n22_punch_out_ends_the_take_and_keeps_automation_after_it(tmp_path):
    c = make(tmp_path)
    tr = lane(c, "zoom")
    keys(c, tr, [(0, 0.0, "linear"), (16, 0.0, "linear")])
    tl(c, op="punch", on=True, a=2, b=4)
    tl(c, op="ovr_mode", mode="latch")
    tl(c, op="rec", on=True)
    tl(c, op="seek", t=2.0)
    tl(c, op="play")
    c.set("zoom", 0.8)
    run(c, 3.0)                                       # through the punch-out at beat 4 to beat 8
    tl(c, op="stop")
    ln = c.tl.by_param["zoom"]
    assert T.lane_value(ln, 3.0) == pytest.approx(0.8, abs=1e-6)
    assert T.lane_value(ln, 6.0) == pytest.approx(0.0, abs=0.12)     # untouched after the punch-out
    assert max(it["t"] for it in c.tl.track(tr["id"])["items"] if it["v"] > 0.5) <= 4.0 + 1e-6


def test_n22_loop_wrap_closes_the_take_at_the_loop_end(tmp_path):
    c = make(tmp_path)
    tl(c, op="loop", on=True, a=0, b=4)
    tl(c, op="ovr_mode", mode="latch")
    tl(c, op="rec", on=True)
    tl(c, op="seek", t=2.0)
    tl(c, op="play")
    c.set("zoom", 0.5)
    run(c, 1.2)                                       # wraps at beat 4
    tl(c, op="stop")
    its = c.tl.track(c.tl.by_param["zoom"].tid)["items"]
    assert its and max(it["t"] for it in its) <= 4.0 + 1e-6 and T.lane_value(c.tl.by_param["zoom"], 3.9) == pytest.approx(0.5)


def test_n41_held_strobe_is_rechecked_and_cue_cuts_get_a_crossfade(tmp_path):
    c = make(tmp_path)
    do(c, cmd="strobe_rate", v=1.0)                   # 2 Hz at 120 bpm: allowed
    tl(c, op="track_add", kind="trigger")
    tl(c, op="items_add", track=c.tl.first("trigger")["id"], items=[dict(t=0.1, len=8, name="strobe")])
    cue = c.tl.first("cue")
    tl(c, op="items_add", track=cue["id"], items=[dict(t=0.2, kind="scene", slot=2, morph=0), dict(t=0.25, kind="scene", slot=5, morph=0)])
    tl(c, op="play")
    run(c, 0.3)
    assert "strobe" in c._tl_trig
    assert c._morph is not None                       # the second cut came 25 ms later: crossfaded, not a hard cut
    do(c, cmd="strobe_rate", v=0.125)                 # now 16 Hz
    run(c, 0.1)
    assert "strobe" not in c._tl_trig and c.tl_limited >= 1


def test_n23_bad_preset_refs_are_rejected_at_the_door():
    for bad in ("Look/X", "foo/bar", "look/-x", "look/" + "a" * 60):
        with pytest.raises(ValueError):
            T.preset_ref(bad)
    assert T.preset_ref("look/deep-blue") == "look/deep-blue"
