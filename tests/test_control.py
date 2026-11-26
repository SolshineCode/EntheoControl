"""Tests for the performance control system (control.py, midi.py, ai_link.py). No GL, no MIDI hardware."""
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import control  # noqa: E402
import midi as midimod  # noqa: E402
from ai_link import AIClient  # noqa: E402
from control import CONT, Controller, validate  # noqa: E402
from params import PARAMS, TRIGGERS  # noqa: E402


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


def run(c, seconds, dt=1 / 60, bands=(0, 0, 0)):
    f = None
    for _ in range(int(round(seconds / dt))):
        c.ft.t += dt
        f = c.update(dt, bands)
    return f


def beats(c, n, dt=1 / 120):
    return run(c, n * 60.0 / c.bpm, dt)


# ---------------------------------------------------------------- basics / API

def test_api_and_defaults(tmp_path):
    c = make(tmp_path)
    f = c.update(1 / 60, [0, 0, 0])
    assert set(f.values) == set(PARAMS)
    for n in CONT:
        assert f.values[n] == pytest.approx(PARAMS[n][2])
    for n in TRIGGERS:
        assert f.values[n] == 0.0
    assert set(f.clock) >= {"bpm", "beat", "beat_phase", "bar_phase", "phrase_phase"}
    assert f.clock["bpm"] == 124.0 and f.events == []
    json.dumps(c.state(), allow_nan=False)
    c.close()


def test_set_clamps_and_slews(tmp_path):
    c = make(tmp_path)
    c.set("zoom", 5)
    assert c.base["zoom"] == 1.0
    c.set("zoom", float("nan"))
    c.set("nope", 1)
    c.set("zoom", "x")
    assert c.base["zoom"] == 1.0
    f = run(c, 1 / 60)
    assert 0.0 < f.values["zoom"] < 0.5          # slewed: no pop on the first frame
    assert run(c, 0.5).values["zoom"] == pytest.approx(1.0, abs=1e-3)
    c.nudge("zoom", -0.25)
    assert c.base["zoom"] == pytest.approx(0.75)


def test_update_survives_bad_inputs(tmp_path):
    c = make(tmp_path)
    for dt, bands in ((float("nan"), None), (-1, [float("nan")]), (1e9, "abc"), ("x", [1, 2, 3, 4]), (0, [])):
        f = c.update(dt, bands)
        assert all(math.isfinite(v) for v in f.values.values())


def test_ripple_events_one_shot(tmp_path):
    c = make(tmp_path)
    c.ripple(0.25, 2.0)
    c.ripple(float("nan"), 0)
    assert c.update(0.01, None).events == [("ripple", 0.25, 1.0)]
    assert c.update(0.01, None).events == []


# ---------------------------------------------------------------- modulation

def test_mod_audio_maths_and_clamp(tmp_path):
    c = make(tmp_path)
    c.set("swim", 0.3)
    c.set_mod("swim", 0, "bass", 0.5)
    c.set_mod("swim", 1, "high", -0.2)
    f = run(c, 0.5, bands=(0.8, 0.0, 0.5))
    assert f.values["swim"] == pytest.approx(0.3 + 0.5 * 0.8 - 0.2 * 0.5, abs=1e-3)
    c.set_mod("swim", 1, "none")
    f = run(c, 0.1, bands=(1.0, 0, 0))
    assert f.values["swim"] == pytest.approx(0.8, abs=1e-3)
    c.set("swim", 0.9)
    assert run(c, 0.5, bands=(1.0, 0, 0)).values["swim"] == 1.0      # clamped to the range
    # range scaling: zoom spans 2.0, so amount 0.25 at full mids moves it by 0.5
    c.set_mod("zoom", 0, "mid", 0.25)
    assert run(c, 0.2, bands=(0, 1.0, 0)).values["zoom"] == pytest.approx(0.5, abs=1e-3)
    assert c.base["zoom"] == 0.0                                    # modulation never moves the fader


def test_lfo_synced_to_clock(tmp_path):
    c = make(tmp_path, bpm=120.0)
    c.set("hue", 0.5)
    c._sm["hue"] = 0.5
    c.set_mod("hue", 0, "lfo_sine", 0.25, rate=4.0)                 # one cycle per bar
    assert beats(c, 1).values["hue"] == pytest.approx(0.75, abs=5e-3)   # quarter cycle: peak
    assert beats(c, 1).values["hue"] == pytest.approx(0.50, abs=1e-2)
    assert beats(c, 1).values["hue"] == pytest.approx(0.25, abs=5e-3)
    c.set_bpm(60.0)                                                 # tempo change keeps the LFO beat-locked
    assert beats(c, 2).values["hue"] == pytest.approx(0.75, abs=1e-2)   # beat 5 = quarter cycle again
    for src, at_quarter in (("lfo_tri", 1.0), ("lfo_saw", -0.5), ("lfo_square", 1.0)):
        assert c._source(dict(src=src, rate=4.0, phase=0.0), "hue", 0, 1.0) == pytest.approx(at_quarter)
    assert c._source(dict(src="lfo_sine", rate=4.0, phase=0.25), "hue", 0, 0.0) == pytest.approx(1.0)


def test_random_is_sample_and_hold(tmp_path):
    c = make(tmp_path)
    m = dict(src="random", rate=1.0, phase=0.0)
    a = [c._source(m, "twirl", 0, b) for b in (0.1, 0.5, 0.99)]
    b = [c._source(m, "twirl", 0, x) for x in (1.01, 1.9)]
    assert a[0] == a[1] == a[2] and b[0] == b[1] and a[0] != b[0]
    assert -1.0 <= a[0] <= 1.0


def test_mod_limits(tmp_path):
    c = make(tmp_path)
    c.set_mod("zoom", 2, "bass", 1)          # only 2 slots
    c.set_mod("flash", 0, "bass", 1)         # triggers are not modulatable
    c.set_mod("zoom", 0, "bogus", 1)
    assert c.mods == {}
    c.set_mod("zoom", 0, "bass", 9, rate=0)
    assert c.mods["zoom"][0]["amt"] == 1.0 and c.mods["zoom"][0]["rate"] > 0


# ---------------------------------------------------------------- clock

def test_clock_phases(tmp_path):
    c = make(tmp_path, bpm=120.0)
    f = run(c, 3.0)                           # 6 beats
    assert f.clock["beat"] == pytest.approx(6.0, abs=1e-6)
    assert f.clock["bar_phase"] == pytest.approx(0.5, abs=1e-6)
    assert f.clock["phrase_phase"] == pytest.approx(6 / 32, abs=1e-6)
    run(c, 0.25)
    assert c._clock()["beat_phase"] == pytest.approx(0.5, abs=1e-6)


def test_tap_tempo_average_and_reset(tmp_path):
    c = make(tmp_path)
    for t in (0.0, 0.5, 1.0, 1.5):
        c.tap(t)
    assert c.bpm == pytest.approx(120.0)
    # only the last 4 taps count
    for t in (1.9, 2.3, 2.7):
        c.tap(t)
    assert c.bpm == pytest.approx(60.0 / 0.4)
    # a gap over 2 s starts over: one tap alone changes nothing, the next one sets the tempo
    c.tap(10.0)
    assert c.bpm == pytest.approx(150.0)
    c.tap(10.6)
    assert c.bpm == pytest.approx(100.0)
    c.tap(10.6001)                            # absurd double-tap is clamped, not a crash
    assert control.BPM_MIN <= c.bpm <= control.BPM_MAX


def test_tap_lands_on_a_beat_and_resync(tmp_path):
    c = make(tmp_path)
    c.beat = 5.4
    c.tap(0.0)
    assert c.beat == 5.0
    c.beat = 5.6
    c.tap(0.5)
    assert c.beat == 6.0
    c.apply([{"cmd": "resync"}])
    assert c.beat == 0.0
    c.apply([{"cmd": "bpm", "v": 9999}, {"cmd": "bpm_nudge", "d": -1}])
    assert c.bpm == control.BPM_MAX - 1


def feed_clock(c, bpm, n_ticks, start=False):
    iv = 60.0 / bpm / 24
    if start:
        c.midi.inject("start", t=c.ft.t)
    for _ in range(n_ticks):
        c.midi.inject("clock", t=c.ft.t)
        c.ft.t += iv
        c.update(iv, None)


def test_midi_clock_drives_bpm_and_beat(tmp_path):
    c = make(tmp_path)
    c.apply([{"cmd": "clock_src", "src": "midi"}])
    feed_clock(c, 140.0, 24 * 4 + 1, start=True)
    assert c.bpm == pytest.approx(140.0, abs=0.01)
    assert c.clock_active == "midi"
    assert c.beat == pytest.approx(4.0 + 1 / 24, abs=0.02)   # 96 ticks after Start, plus the tick interval since
    c.midi.inject("songpos", val=32, t=c.ft.t)          # 32 sixteenths = beat 8: the beat jumps there
    c.midi.inject("clock", t=c.ft.t)
    c.ft.t += 0.001
    c.update(0.001, None)
    assert c.beat == pytest.approx(8.0, abs=0.1)
    # clock goes silent: falls back to the internal clock at the last tempo and keeps moving
    b0 = c.beat
    run(c, 2.0)
    assert c.clock_active == "internal" and c.beat > b0 + 2 and c.bpm == pytest.approx(140.0, abs=0.01)


def test_clock_auto_follow_and_tap_alias(tmp_path):
    c = make(tmp_path)
    c.apply([{"cmd": "clock_src", "src": "auto"}])
    run(c, 0.1)
    assert c.clock_active == "internal" and c.bpm == 124.0
    feed_clock(c, 100.0, 60)
    assert c.clock_active == "midi" and c.bpm == pytest.approx(100.0, abs=0.01)
    assert validate({"cmd": "clock_src", "src": "tap"})["src"] == "internal"
    c.apply([{"cmd": "clock_src", "src": "internal"}])
    feed_clock(c, 100.0, 30)
    c.set_bpm(124.0)
    feed_clock(c, 100.0, 30)
    assert c.clock_active == "internal" and c.bpm == 124.0   # internal ignores incoming clock


# ---------------------------------------------------------------- triggers

def test_flash_envelope(tmp_path):
    c = make(tmp_path)
    c.trigger("flash", True)
    assert c.update(0.001, None).values["flash"] == 1.0      # instant attack
    assert run(c, 0.5).values["flash"] == 1.0                # held
    c.trigger("flash", False)
    assert run(c, 0.05).values["flash"] > 0.5
    assert run(c, 0.30).values["flash"] < 0.1                # gone in roughly 300-400 ms
    assert run(c, 1.0).values["flash"] == 0.0
    c.trigger("punch", True)
    c.trigger("punch", False)                                # tap shorter than a frame still shows
    assert c.update(1 / 60, None).values["punch"] > 0.8


def test_held_triggers_strobe_and_latch(tmp_path):
    c = make(tmp_path, bpm=120.0)
    for n in ("invert", "freeze", "blackout"):
        c.trigger(n, True)
        assert c.update(0.001, None).values[n] == 1.0
        c.trigger(n, False)
        assert c.update(0.001, None).values[n] == 0.0
    c.trigger("strobe", True)
    seen = {run(c, 1 / 60).values["strobe"] for _ in range(60)}
    assert seen == {0.0, 1.0}                                # gated by the clock
    c.trigger("strobe", False)
    assert {run(c, 1 / 60).values["strobe"] for _ in range(30)} == {0.0}
    c.apply([{"cmd": "latch", "name": "invert", "on": True}])
    c.trigger("invert", True)
    c.trigger("invert", False)
    assert c.update(0.001, None).values["invert"] == 1.0     # latched on
    c.trigger("invert", True)
    c.trigger("invert", False)
    assert c.update(0.001, None).values["invert"] == 0.0
    c.trigger("nope", True)                                  # unknown: ignored


# ---------------------------------------------------------------- scenes

def test_scene_save_recall_instant(tmp_path):
    c = make(tmp_path)
    c.set("zoom", 0.5)
    c.set_mod("hue", 0, "bass", 0.3)
    c.scene_save(9)
    c.set("zoom", -0.5)
    c.set_mod("hue", 0, "none")
    assert c.scene_recall(9)
    assert c.base["zoom"] == 0.5 and c.mods["hue"][0]["src"] == "bass" and c.cur == 9
    c.scene_clear(9)
    assert not c.scene_recall(9)                              # empty slot: ignored
    assert not c.scene_recall(99)
    st = c.state()["scenes"]
    assert st["occ"][9] == 0 and len(st["occ"]) == 16


def test_scene_morph_interpolates_and_snaps_discrete(tmp_path):
    c = make(tmp_path, bpm=120.0)
    c.scenes = [None] * 16
    c.set("zoom", 1.0)
    c.set("travel_mode", 3)
    c.set("palette", 4)
    c.scene_save(0)
    c.apply([{"cmd": "init"}])
    c.set("zoom", -1.0)
    c.morph_bars = 2.0                                        # 8 beats
    c.scene_recall(0)
    beats(c, 2)
    assert c.base["zoom"] == pytest.approx(-0.5, abs=0.02)
    assert c.base["travel_mode"] == 0.0                       # discrete: not yet
    assert c.base["palette"] == pytest.approx(1.0, abs=0.05)  # palette is fractional: it glides
    beats(c, 2.2)
    assert c.base["travel_mode"] == 3.0                       # snapped at the midpoint
    assert c.base["zoom"] == pytest.approx(0.05, abs=0.03)
    assert c.state()["scenes"]["morph_t"] == pytest.approx(0.525, abs=0.02)
    beats(c, 4)
    assert c.base["zoom"] == 1.0 and c._morph is None


def test_morph_crossfades_mods_and_fader_overrides(tmp_path):
    c = make(tmp_path, bpm=120.0)
    c.scenes = [None] * 16
    c.scene_save(0)                                           # scene 0: no mods
    c.set_mod("swim", 0, "bass", 0.4)
    c.morph_bars = 1.0
    c.scene_recall(0)
    beats(c, 2, dt=1 / 120)
    f = c.update(0, (1.0, 0, 0))
    assert f.values["swim"] == pytest.approx(0.3 + 0.4 * 0.5, abs=0.02)   # outgoing mod at half weight
    c.set("zoom", 0.7)                                        # hand on a fader mid-morph wins
    beats(c, 3)
    assert c.base["zoom"] == 0.7


def test_quantized_recall_timing(tmp_path):
    c = make(tmp_path, bpm=120.0)
    c.set("zoom", 1.0)
    c.scene_save(3)
    c.set("zoom", 0.0)
    c.apply([{"cmd": "quantize", "q": "bar"}])
    beats(c, 1.5)
    c.apply([{"cmd": "scene_recall", "i": 3}])
    assert c.state()["scenes"]["pending"] == 3
    beats(c, 2.3)                                             # beat 3.8: still waiting
    assert c.base["zoom"] == 0.0 and c.pending is not None
    beats(c, 0.3)                                             # crossed beat 4
    assert c.base["zoom"] == 1.0 and c.pending is None
    # phrase quantize, and "now" bypasses it
    c.set("zoom", 0.0)
    c.apply([{"cmd": "quantize", "q": "phrase"}, {"cmd": "scene_recall", "i": 3}])
    beats(c, 8)
    assert c.base["zoom"] == 0.0
    c.apply([{"cmd": "scene_recall", "i": 3, "now": True}])
    assert c.base["zoom"] == 1.0
    # beat quantize
    c.set("zoom", 0.0)
    c.apply([{"cmd": "quantize", "q": "beat"}, {"cmd": "scene_recall", "i": 3}])
    beats(c, 1.05)
    assert c.base["zoom"] == 1.0


def test_scenes_persist_and_corrupt_file_falls_back(tmp_path):
    c = make(tmp_path)
    assert c.scenes_loaded is False and c.scenes[0] is not None     # factory scenes when there is no file
    c.set("twirl", 0.42)
    c.scene_save(12)
    data = json.loads((tmp_path / "scenes.json").read_text())
    assert data["scenes"][12]["base"]["twirl"] == 0.42
    assert not list(tmp_path.glob("*.tmp"))
    c2 = make(tmp_path)
    assert c2.scenes_loaded and c2.scenes[12]["base"]["twirl"] == 0.42
    # hostile content is cleaned, not trusted
    data["scenes"][12]["base"]["twirl"] = 99
    data["scenes"][12]["base"]["bogus"] = 1
    data["scenes"][12]["mods"] = {"zoom": [{"src": "evil", "amt": 5}], "hue": [{"src": "bass", "amt": 5}]}
    data["scenes"][3] = "junk"
    (tmp_path / "scenes.json").write_text(json.dumps(data))
    c3 = make(tmp_path)
    assert c3.scenes[12]["base"]["twirl"] == 1.0 and "bogus" not in c3.scenes[12]["base"]
    assert c3.scenes[12]["mods"] == {"hue": [dict(src="bass", amt=1.0, rate=1.0, phase=0.0), None]}
    assert c3.scenes[3] is None
    for junk in ("{not json", "[]", '{"scenes": 5}', ""):
        (tmp_path / "scenes.json").write_text(junk)
        c4 = make(tmp_path)
        assert c4.scenes_loaded is False and c4.scenes[0]["name"] == "Home"
        c4.update(1 / 60, None)


def test_scene_save_never_raises_when_dir_is_unwritable(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    c = Controller(blocker / "sub", midi=False, ai=False, restore=True)   # state_dir cannot exist
    c.scene_save(0)
    c.close()


# ---------------------------------------------------------------- autopilot

def test_autopilot_steps_every_n_bars_skipping_current(tmp_path):
    c = make(tmp_path, bpm=240.0)
    c.scenes = [None] * 16
    for i, z in ((2, 0.2), (5, 0.5), (11, -0.7)):
        c.set("zoom", z)
        c.scene_save(i)
    c.apply([{"cmd": "autopilot", "on": True, "bars": 4, "mode": "next"}])
    assert c.cur == 11
    beats(c, 0.5)
    seq = []
    for _ in range(5):
        beats(c, 16)
        seq.append(c.cur)
    assert seq == [2, 5, 11, 2, 5]
    c.apply([{"cmd": "autopilot", "mode": "random"}])
    prev, seen = c.cur, set()
    for _ in range(12):
        beats(c, 16)
        assert c.cur != prev                                  # never "changes" to the scene already showing
        prev = c.cur
        seen.add(prev)
    assert seen == {2, 5, 11}
    c.apply([{"cmd": "autopilot", "on": False}])
    beats(c, 40)
    assert c.cur == prev


def test_autopilot_morphs_and_holds_palette(tmp_path):
    c = make(tmp_path, bpm=240.0)
    c.scenes = [None] * 16
    for i, pal in ((0, 0.0), (1, 3.0)):
        c.set("palette", pal)
        c.set("zoom", i * 1.0)
        c.scene_save(i)
    c.morph_bars = 1.0
    c.apply([{"cmd": "autopilot", "on": True, "bars": 4, "mode": "next", "pal_hold": 1.5}])   # hold colour 48 beats
    beats(c, 16.5)          # first step (to scene 0): palette allowed, starts the hold (until beat 64)
    assert c.cur == 0 and c._morph is not None
    beats(c, 16)            # second step (to scene 1) is inside the hold: geometry changes, colour stays
    beats(c, 6)
    assert c.cur == 1 and c.base["zoom"] == 1.0 and c.base["palette"] == 0.0
    beats(c, 32)            # the next visit to scene 1 is past the hold
    assert c.cur == 1 and c.base["palette"] == 3.0


def test_drift_wanders_within_range_without_moving_faders(tmp_path):
    c = make(tmp_path)
    c.apply([{"cmd": "autopilot", "drift": True, "drift_amt": 1.0, "drift_params": ["hue", "zoom"]}])
    vals = [run(c, 0.5).values for _ in range(120)]
    zs = [v["zoom"] for v in vals]
    assert max(zs) - min(zs) > 0.05 and all(-1.0 <= z <= 1.0 for z in zs)
    assert all(0.0 <= v["hue"] <= 1.0 for v in vals)
    assert c.base["zoom"] == 0.0 and vals[-1]["twirl"] == 0.0
    c.apply([{"cmd": "autopilot", "drift": False}])
    assert run(c, 0.5).values["zoom"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- palette glide / panic / autosave

def test_palette_glide(tmp_path):
    c = make(tmp_path, bpm=120.0)
    c.apply([{"cmd": "glide", "bars": 2}])                   # 8 beats = 4 s
    c.set("palette", 4)
    assert run(c, 1.0).values["palette"] == pytest.approx(1.0, abs=0.05)
    assert run(c, 2.0).values["palette"] == pytest.approx(3.0, abs=0.05)
    assert run(c, 1.2).values["palette"] == 4.0
    c.apply([{"cmd": "glide", "bars": 0}])
    c.set("palette", 0)
    assert run(c, 0.5).values["palette"] == pytest.approx(0.0, abs=1e-3)


def test_panic_recalls_safe_scene_and_clears_triggers(tmp_path):
    c = make(tmp_path)
    c.set("zoom", 0.33)
    c.scene_save(4)
    c.apply([{"cmd": "safe", "i": 4}, {"cmd": "quantize", "q": "phrase"}, {"cmd": "morph", "bars": 8},
             {"cmd": "latch", "name": "blackout", "on": True}])
    c.set("zoom", -1.0)
    c.set("master", 0.0)
    c.trigger("blackout", True)
    c.trigger("strobe", True)
    c.scene_recall(1)
    c.apply([{"cmd": "panic"}])
    f = c.update(1 / 60, None)
    assert f.values["zoom"] == pytest.approx(0.33) and c.cur == 4       # instantly: no morph, no quantize, no slew
    assert f.values["blackout"] == 0.0 and f.values["strobe"] == 0.0 and f.values["master"] == 1.0
    assert c.pending is None and c._morph is None
    c.scene_clear(4)
    c.set("zoom", 1.0)
    c.panic()                                                          # no safe scene: plain defaults
    assert c.update(1 / 60, None).values["zoom"] == 0.0


def test_autosave_and_restore(tmp_path):
    c = make(tmp_path)
    c.set("twirl", 0.6)
    c.set_mod("zoom", 1, "lfo_saw", -0.4, rate=8)
    c.apply([{"cmd": "bpm", "v": 133}, {"cmd": "morph", "bars": 4}, {"cmd": "quantize", "q": "phrase"},
             {"cmd": "autopilot", "on": True, "bars": 16}, {"cmd": "glide", "bars": 4}, {"cmd": "cue_ch", "ch": 10},
             {"cmd": "latch", "name": "freeze", "on": True}])
    c.trigger("blackout", True)
    assert not (tmp_path / "live_state.json").exists()
    run(c, 5.2)                                                         # autosaves every ~5 s
    assert (tmp_path / "live_state.json").exists()
    r = Controller(tmp_path, midi=False, ai=False)                      # restore=True is the default
    assert r.restored and r.base["twirl"] == 0.6 and r.bpm == 133
    assert r.mods["zoom"][1] == dict(src="lfo_saw", amt=-0.4, rate=8.0, phase=0.0)
    assert (r.morph_bars, r.quantize, r.glide_bars, r.cue_ch) == (4.0, "phrase", 4.0, 10)
    assert r.auto["on"] and r.auto["bars"] == 16 and r.latch["freeze"]
    f = r.update(1 / 60, None)
    assert f.values["blackout"] == 0.0 and f.values["twirl"] == 0.6     # triggers are never restored; no slew-in
    fresh = Controller(tmp_path, midi=False, ai=False, restore=False)
    assert fresh.base["twirl"] == 0.0
    before = (tmp_path / "live_state.json").read_text()
    fresh.set("twirl", -0.9)
    fresh.close()                                                       # restore=False never overwrites the saved show state
    assert (tmp_path / "live_state.json").read_text() == before
    for junk in ("{nope", "[1,2]", '{"base": 7, "mods": 3, "settings": {"bpm": "fast", "morph": 2}}'):
        (tmp_path / "live_state.json").write_text(junk)
        r = Controller(tmp_path, midi=False, ai=False)
        assert r.base["twirl"] == 0.0 and r.bpm == 124.0
        r.update(1 / 60, None)
    assert r.morph_bars == 2.0 and r.restored


# ---------------------------------------------------------------- apply(): validation + passthrough

def test_apply_passthrough_untouched(tmp_path):
    c = make(tmp_path)
    foreign = [{"cmd": "corner", "i": 1, "u": 0.5, "v": 0.5}, {"cmd": "nudge", "i": 0, "du": 0.01, "dv": 0},
               {"cmd": "grid", "on": True}, {"cmd": "calib", "on": False}, {"cmd": "next"}, {"cmd": "save"},
               {"cmd": "reset"}, {"cmd": "next_video", "type": "next_video"}, {"type": "x"}, "junk", None]
    mine = [{"cmd": "set", "name": "zoom", "v": 0.5}, {"cmd": "tap"}, {"cmd": "ripple", "x": 0.1, "y": 0.9}]
    rest = c.apply(foreign[:4] + mine + foreign[4:])
    assert rest == foreign and all(a is b for a, b in zip(rest, foreign))
    assert c.base["zoom"] == 0.5 and c.events == [("ripple", 0.1, 0.9)]
    assert c.apply([]) == [] and c.apply(None) == []


def test_apply_validates_everything(tmp_path):
    c = make(tmp_path)
    before = json.dumps(c.state()["base"])
    bad = [
        {"cmd": "set", "name": "nope", "v": 1}, {"cmd": "set", "name": "zoom"}, {"cmd": "set", "name": "zoom", "v": "1"},
        {"cmd": "set", "name": "zoom", "v": float("nan")}, {"cmd": "set", "name": "zoom", "v": float("inf")},
        {"cmd": "set", "name": "zoom", "v": True}, {"cmd": "set", "name": ["zoom"], "v": 1},
        {"cmd": "mod", "name": "zoom", "slot": 2, "src": "bass"}, {"cmd": "mod", "name": "zoom", "slot": 0, "src": "x"},
        {"cmd": "mod", "name": "flash", "slot": 0, "src": "bass"}, {"cmd": "trigger", "name": "zoom", "on": True},
        {"cmd": "trigger", "name": "flash", "on": 1}, {"cmd": "ripple", "x": None, "y": 0},
        {"cmd": "bpm", "v": None}, {"cmd": "scene_recall", "i": 16}, {"cmd": "scene_save", "i": -1},
        {"cmd": "scene_save", "i": 1.5}, {"cmd": "quantize", "q": "never"}, {"cmd": "morph", "bars": "2"},
        {"cmd": "autopilot", "on": "yes"}, {"cmd": "autopilot", "drift_params": ["zoom", "bogus"]},
        {"cmd": "midi_learn", "target": "param:bogus"}, {"cmd": "midi_learn", "target": 5},
        {"cmd": "clock_src", "src": "moon"}, {"cmd": "cue_ch", "ch": 17}, {"cmd": "safe", "i": "0"},
    ]
    for b in bad:
        with pytest.raises(ValueError):
            validate(b)
    assert c.apply(bad) == []                                 # owned but malformed: dropped, never raised
    assert json.dumps(c.state()["base"]) == before and c.mods == {} and c.learn is None
    c.apply([{"cmd": "set", "name": "zoom", "v": 77, "junk": 1}, {"cmd": "inc", "name": "hue", "d": 0.25},
             {"cmd": "mod", "name": "swim", "slot": 1, "src": "lfo_tri", "amt": -3, "rate": 1e9},
             {"cmd": "trigger", "name": "freeze", "on": True}, {"cmd": "strobe_rate", "v": 0.5},
             {"cmd": "morph", "bars": 999}])
    assert c.base["zoom"] == 1.0 and c.base["hue"] == 0.25
    assert c.mods["swim"][1] == dict(src="lfo_tri", amt=-1.0, rate=64.0, phase=0.0)
    assert c.held["freeze"] and c.strobe_rate == 0.5 and c.morph_bars == 64.0
    assert validate({"cmd": "set", "name": "zoom", "v": 77, "junk": 1}) == {"cmd": "set", "name": "zoom", "v": 1.0}


def test_state_is_compact_json_and_meta_covers_params(tmp_path):
    c = make(tmp_path)
    c.set_mod("zoom", 0, "lfo_sine", 0.5, 4)
    run(c, 0.1)
    s = c.state()
    blob = json.dumps(s, allow_nan=False)
    assert len(blob) < 12000
    for key in ("values", "base", "mods", "clock", "trig", "scenes", "autopilot", "midi", "ai", "morph", "quantize"):
        assert key in s
    assert s["mods"] == {"zoom": [dict(src="lfo_sine", amt=0.5, rate=4.0, phase=0.0), None]}
    assert s["ai"]["available"] is False
    m = control.meta()
    json.dumps(m)
    assert [p["name"] for p in m["params"]] == list(PARAMS)   # generated from PARAMS: new params appear by themselves
    assert next(p for p in m["params"] if p["name"] == "travel_mode")["options"][1] == "Tube"


# ---------------------------------------------------------------- MIDI

def test_midi_learn_cc_to_param(tmp_path):
    c = make(tmp_path)
    c.midi.clear()
    c.apply([{"cmd": "midi_learn", "target": "param:zoom"}])
    assert c.state()["midi"]["learn"] == "param:zoom"
    c.midi.inject("cc", ch=2, num=21, val=127)
    c.update(1 / 60, None)
    assert c.learn is None and c.base["zoom"] == 0.0          # the learning message binds, it does not act
    assert c.state()["midi"]["bound"] == {"param:zoom": ["cc 21 ch3"]}
    c.midi.inject("cc", ch=2, num=21, val=127)
    c.midi.inject("cc", ch=5, num=21, val=0)                  # other channel: not this binding
    c.update(1 / 60, None)
    assert c.base["zoom"] == 1.0
    c.midi.inject("cc", ch=2, num=21, val=0)
    c.update(1 / 60, None)
    assert c.base["zoom"] == -1.0
    saved = json.loads((tmp_path / "midi_map.json").read_text())
    assert saved["bindings"] == [dict(type="cc", ch=2, num=21, target="param:zoom", port=None, mode="abs")]
    c2 = make(tmp_path)                                       # persisted
    c2.midi.inject("cc", ch=2, num=21, val=64)
    c2.update(1 / 60, None)
    assert c2.base["zoom"] == pytest.approx(64 / 127 * 2 - 1)
    c2.apply([{"cmd": "midi_clear", "target": "param:zoom"}])
    assert c2.midi.bindings == []


def test_midi_learn_note_to_trigger_scene_tap_autopilot(tmp_path):
    c = make(tmp_path)
    c.midi.clear()
    for note, target in ((40, "trigger:flash"), (41, "scene:5"), (42, "tap"), (43, "autopilot")):
        c.apply([{"cmd": "midi_learn", "target": target}])
        c.midi.inject("note_off", num=99)                     # a stray release must not get bound
        c.midi.inject("note_on", num=note, val=100)
        c.update(1 / 60, None)
    assert len(c.midi.bindings) == 4
    c.midi.inject("note_on", num=40, val=90)
    assert c.update(1 / 60, None).values["flash"] == 1.0
    c.midi.inject("note_off", num=40)
    run(c, 1.0)
    assert c.values["flash"] == 0.0
    c.set("zoom", 0.9)
    c.scene_save(5)
    c.set("zoom", 0.0)
    c.midi.inject("note_on", num=41, val=1)
    c.update(1 / 60, None)
    assert c.base["zoom"] == 0.9
    c.midi.inject("note_on", num=43, val=1)
    c.update(1 / 60, None)
    assert c.auto["on"] is True
    for t in (0.0, 0.5, 1.0):
        c.ft.t = 200 + t
        c.midi.inject("note_on", num=42, val=1)
        c.update(1 / 60, None)
    assert c.bpm == pytest.approx(120.0, abs=0.5)
    # re-learning a control replaces what it did before
    c.apply([{"cmd": "midi_learn", "target": "trigger:invert"}])
    c.midi.inject("note_on", num=40, val=1)
    c.update(1 / 60, None)
    assert [b["target"] for b in c.midi.lookup("test", "note", 0, 40)] == ["trigger:invert"]


def test_default_map_apc_mini_mk2_and_launch_control(tmp_path):
    c = make(tmp_path)
    apc = "APC mini mk2 0"
    c.midi.inject("cc", num=56, val=0, port=apc)              # 9th fader = master
    c.midi.inject("cc", num=48, val=127, port=apc)            # fader 1 = zoom
    c.midi.inject("note_on", num=0, val=127, port=apc)        # bottom-left pad = flash
    c.midi.inject("note_on", num=20, val=127, port=apc)       # a ripple pad
    f = c.update(1 / 60, None)
    assert c.base["master"] == 0.0 and c.base["zoom"] == 1.0 and f.values["flash"] == 1.0
    assert f.events == [("ripple", pytest.approx(4.5 / 8), pytest.approx(0.3))]
    c.set("twirl", 0.5)
    c.midi.inject("note_on", num=122, val=127, port=apc)      # shift + top-left pad = save scene 1
    c.midi.inject("note_on", num=56, val=127, port=apc)
    c.midi.inject("note_off", num=122, port=apc)
    c.update(1 / 60, None)
    assert c.scenes[0]["base"]["twirl"] == 0.5
    c.set("twirl", 0.0)
    c.midi.inject("note_on", num=56, val=127, port=apc)       # without shift = recall
    c.update(1 / 60, None)
    assert c.base["twirl"] == 0.5
    c.midi.inject("note_on", num=106, val=127, port=apc)      # track button 7 = next video (passed to the engine)
    c.update(1 / 60, None)
    assert c.apply([]) == [{"cmd": "next_video", "type": "next_video"}] and c.apply([]) == []
    # same numbers from an unknown device do nothing; the Launch Control XL has its own layout
    c.midi.inject("cc", num=56, val=127, port="Some Keyboard")
    c.midi.inject("cc", num=77, val=64, port="Launch Control XL 1")
    c.update(1 / 60, None)
    assert c.base["master"] == pytest.approx(64 / 127)
    kinds = {b["port"] for b in midimod.default_bindings()}
    assert kinds == {"apc mini", "launch control"}
    assert all(control.valid_target(b["target"]) for b in midimod.default_bindings())


def test_cue_channel_notes_recall_scenes_and_ccs_set_params(tmp_path):
    c = make(tmp_path, bpm=120.0)
    c.set("zoom", 0.8)
    c.scene_save(2)
    c.set("zoom", 0.0)
    c.apply([{"cmd": "quantize", "q": "bar"}])
    c.midi.inject("note_on", ch=15, num=2, val=100, port="loopMIDI Port")     # channel 16, note 2 -> scene 3
    c.update(0.001, None)
    assert c.pending is not None and c.pending[0] == 2          # quantized per the settings
    beats(c, 4.1)
    assert c.base["zoom"] == 0.8
    c.midi.inject("cc", ch=15, num=19, val=127)                 # CUE_CC 19 = hue
    c.midi.inject("note_on", ch=14, num=2, val=100)             # channel 15 is not the cue channel
    c.update(0.001, None)
    assert c.base["hue"] == 1.0 and c.pending is None
    assert control.CUE_CC[1] == "master" and control.CUE_CC[19] == "hue"      # frozen numbering
    c.apply([{"cmd": "cue_ch", "ch": 0}])                       # off
    c.set("hue", 0.0)
    c.midi.inject("cc", ch=15, num=19, val=127)
    c.update(0.001, None)
    assert c.base["hue"] == 0.0


def test_midi_map_corrupt_falls_back_to_defaults(tmp_path):
    n_default = len(midimod.default_bindings())
    for junk in ("{broken", "[]", '{"bindings": "x"}', ""):
        (tmp_path / "midi_map.json").write_text(junk)
        c = make(tmp_path)
        assert len(c.midi.bindings) == n_default
        c.update(1 / 60, None)
    (tmp_path / "midi_map.json").write_text(json.dumps({"bindings": [
        {"type": "cc", "ch": 0, "num": 7, "target": "param:hue"}, {"type": "cc", "num": 999, "target": "tap"},
        {"type": "sysex", "num": 1, "target": "tap"}, {"type": "note", "num": 1, "target": "rm -rf"}, 7]}))
    c = make(tmp_path)
    assert [b["target"] for b in c.midi.bindings] == ["param:hue"]            # bad entries dropped
    c.apply([{"cmd": "midi_defaults"}])
    assert len(c.midi.bindings) == n_default


def test_midi_never_raises_into_update(tmp_path):
    c = make(tmp_path)
    c.midi.q.put(("p", "cc", 0, 48, "bad", 0.0))
    c.midi.q.put(("too", "short"))
    c.midi.q.put(None)
    c.midi.bindings.append(dict(type="cc", ch=None, num=1, target="scene:zzz", port=None, mode="abs"))
    c.midi.inject("cc", num=1, val=127)
    c.update(1 / 60, None)
    c.midi.poll = lambda: 1 / 0
    c.update(1 / 60, None)


def test_midi_manager_real_backend_starts_and_stops_without_devices(tmp_path):
    m = midimod.MidiManager(tmp_path / "midi_map.json", scan=True, rescan_s=0.05)
    time.sleep(0.3)
    assert isinstance(m.ports, list)          # usually empty: no hardware in the test environment
    m.leds({0: 5})
    m.close()


def test_relative_encoder_mode(tmp_path):
    c = make(tmp_path)
    c.midi.bindings = [dict(type="cc", ch=None, num=9, target="param:hue", port=None, mode="rel")]
    for v in (1, 1, 3, 127):
        c.midi.inject("cc", num=9, val=v)
    c.update(1 / 60, None)
    assert c.base["hue"] == pytest.approx(4 / 127)


# ---------------------------------------------------------------- AI worker link

class FakeWorker:
    def __init__(self, token="secret", slow=0.0):
        self.jobs, self.cancels, self.done, self.token, self.slow = [], 0, [], token, slow
        fw = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.headers.get("X-Token") != fw.token:
                    return self._send(403, {"error": "forbidden"})
                time.sleep(fw.slow)
                self._send(200, dict(loaded=True, loading=False, busy=False, queue=[], done=fw.done, error="", base="fake"))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.headers.get("X-Token") != fw.token:
                    return self._send(403, {"error": "forbidden"})
                if self.path == "/cancel":
                    fw.cancels += 1
                    return self._send(200, {"ok": True})
                fw.jobs.append((time.monotonic(), body))
                self._send(200, {"id": f"j{len(fw.jobs)}"})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


def wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def worker():
    w = FakeWorker()
    yield w
    w.stop()


def ai_controller(tmp_path, url):
    c = Controller(tmp_path, midi=False, ai=False, restore=False)
    c.ai = AIClient([tmp_path / ".remote_token"], url=url, poll_s=0.05, start=True)
    return c


def test_gen_validation():
    ok = validate({"cmd": "gen", "prompt": "  kene lattice  ", "n": 4, "seed": 7, "show": False, "junk": 1})
    assert ok == {"cmd": "gen", "prompt": "kene lattice", "n": 4, "init": None, "show": False, "seed": 7}
    d = validate({"cmd": "gen", "prompt": "x"})
    assert d["n"] == 1 and d["show"] is True and d["init"] is None and "strength" not in d
    s = validate({"cmd": "gen", "prompt": "x", "init": "snapshot"})
    assert s["init"] == "snapshot" and s["strength"] == 0.45
    assert validate({"cmd": "gen", "prompt": "x", "init": "snapshot", "strength": 5})["strength"] == 0.9
    assert validate({"cmd": "gen", "prompt": "x", "strength": 0.0})["strength"] == 0.2
    assert validate({"cmd": "gen", "prompt": "x" * 500})["prompt"] == "x" * 500
    assert validate({"cmd": "gen", "prompt": "x", "negative": " blurry "})["negative"] == "blurry"
    for bad in ({"cmd": "gen"}, {"cmd": "gen", "prompt": ""}, {"cmd": "gen", "prompt": "   "},
                {"cmd": "gen", "prompt": "x" * 501}, {"cmd": "gen", "prompt": 5}, {"cmd": "gen", "prompt": "x", "n": 0},
                {"cmd": "gen", "prompt": "x", "n": 9}, {"cmd": "gen", "prompt": "x", "n": 2.0},
                {"cmd": "gen", "prompt": "x", "seed": -1}, {"cmd": "gen", "prompt": "x", "seed": "1"},
                {"cmd": "gen", "prompt": "x", "init": "file.png"}, {"cmd": "gen", "prompt": "x", "show": 1},
                {"cmd": "gen", "prompt": "x", "negative": "y" * 501},
                {"cmd": "gen", "prompt": "x", "strength": float("nan")}):
        with pytest.raises(ValueError):
            validate(bad)


def test_gen_forwarded_to_worker_and_show_texture_emitted(tmp_path, worker):
    (tmp_path / ".remote_token").write_text("secret\n")
    c = ai_controller(tmp_path, worker.url)
    try:
        assert wait_for(lambda: c.state()["ai"].get("available"))
        s = c.state()["ai"]
        assert s["loaded"] is True and s["base"] == "fake" and s["queue"] == 0
        assert c.apply([{"cmd": "gen", "prompt": "beadwork mandala", "n": 2, "show": True}]) == []
        assert wait_for(lambda: worker.jobs)
        assert worker.jobs[0][1] == dict(prompt="beadwork mandala", n=2, init=None, show=True)
        worker.done.append(dict(id="j1", name="gen_1.png", prompt="beadwork mandala", secs=2.5))
        worker.done.append(dict(id="other", name="gen_0.png", prompt="someone else", secs=1.0))
        assert wait_for(lambda: (c.update(0.01, None), c._out)[1])
        assert c.apply([]) == [{"cmd": "show_texture", "type": "show_texture", "name": "gen_1.png"}]
        time.sleep(0.2)
        c.update(0.01, None)
        assert c.apply([]) == []                              # announced once
        assert [d["name"] for d in c.state()["ai"]["done"]] == ["gen_1.png", "gen_0.png"]
        c.apply([{"cmd": "gen_cancel"}])
        assert wait_for(lambda: worker.cancels == 1)
    finally:
        c.close()


def test_gen_from_snapshot_requests_a_frame_first(tmp_path, worker):
    (tmp_path / ".remote_token").write_text("secret")
    c = ai_controller(tmp_path, worker.url)
    try:
        t0 = time.monotonic()
        rest = c.apply([{"cmd": "gen", "prompt": "dream", "init": "snapshot", "strength": 0.6, "show": False}])
        assert rest == [{"cmd": "snapshot", "type": "snapshot"}]          # the engine writes snapshots/current.png
        assert wait_for(lambda: worker.jobs)
        assert worker.jobs[0][0] - t0 >= 0.28                             # the job goes out ~300 ms later
        assert worker.jobs[0][1]["init"] == "snapshot" and worker.jobs[0][1]["strength"] == 0.6
        worker.done.append(dict(id="j1", name="gen_2.png", prompt="dream", secs=1))
        time.sleep(0.3)
        c.update(0.01, None)
        assert c.apply([]) == []                                          # show=False: no crossfade request
    finally:
        c.close()


def test_ai_unavailable_without_token_or_worker(tmp_path, worker):
    c = ai_controller(tmp_path, worker.url)                               # no .remote_token yet
    try:
        time.sleep(0.2)
        assert c.state()["ai"]["available"] is False
        c.apply([{"cmd": "gen", "prompt": "x"}])
        t0 = time.perf_counter()
        for _ in range(30):
            c.update(1 / 60, None)
        assert time.perf_counter() - t0 < 0.5 and worker.jobs == []
        (tmp_path / ".remote_token").write_text("wrong")
        c.ai._wake.set()
        assert wait_for(lambda: "403" in c.state()["ai"].get("error", ""))
        assert c.state()["ai"]["available"] is False
    finally:
        c.close()
    (tmp_path / ".remote_token").write_text("secret")
    dead = AIClient([tmp_path / ".remote_token"], url="http://127.0.0.1:9", timeout=0.5, start=False)
    assert dead.poll_once() is False and dead.status()["available"] is False
    off = Controller(tmp_path, midi=False, ai=False, restore=False)
    off.apply([{"cmd": "gen", "prompt": "x"}, {"cmd": "gen_cancel"}])     # AI off: accepted, nothing happens
    assert off.state()["ai"]["available"] is False


def test_ai_timeout_never_blocks_update(tmp_path):
    slow = FakeWorker(slow=1.5)
    (tmp_path / ".remote_token").write_text("secret")
    c = Controller(tmp_path, midi=False, ai=False, restore=False)
    c.ai = AIClient([tmp_path / ".remote_token"], url=slow.url, poll_s=0.05, timeout=0.3, start=True)
    try:
        c.apply([{"cmd": "gen", "prompt": "x"}])
        t0 = time.perf_counter()
        for _ in range(60):
            c.update(1 / 60, None)
            c.state()
        assert time.perf_counter() - t0 < 0.5                             # the frame loop never waits on the worker
        assert wait_for(lambda: "timed out" in c.state()["ai"].get("error", "").lower(), 4.0)
        assert c.state()["ai"]["available"] is False
    finally:
        c.close()
        slow.stop()


# ================================================================ review fixes

def test_restore_false_never_writes_or_renames_state_files(tmp_path):
    good = json.dumps({"version": 1, "scenes": [None] * 16}).encode()
    for content in (good, b"{corrupt json"):
        (tmp_path / "scenes.json").write_bytes(content)
        (tmp_path / "midi_map.json").write_bytes(content)
        c = Controller(tmp_path, midi=False, ai=False, restore=False)        # read-only: headless / test runs
        c.set("zoom", 0.7)
        c.scene_save(3)
        c.scene_clear(0)
        c.midi.bind("cc", 0, 5, "param:hue")
        c.apply([{"cmd": "midi_clear", "target": None}, {"cmd": "midi_defaults"}])
        for _ in range(400):
            c.update(1 / 60, None)
        c.close()
        assert (tmp_path / "scenes.json").read_bytes() == content           # byte-identical
        assert (tmp_path / "midi_map.json").read_bytes() == content
        assert sorted(f.name for f in tmp_path.iterdir()) == ["midi_map.json", "scenes.json"]   # no .bad, no live_state
    assert c.scenes[3] is not None                                           # still works in memory


def test_transient_read_error_does_not_destroy_scenes(tmp_path, monkeypatch):
    c = make(tmp_path)
    c.set("twirl", 0.42)
    c.scene_save(12)
    c.midi.clear()
    c.midi.bind("cc", 0, 5, "param:hue")
    scenes_before = (tmp_path / "scenes.json").read_bytes()
    map_before = (tmp_path / "midi_map.json").read_bytes()
    real = Path.read_text

    def flaky(self, *a, **kw):
        if self.name in ("scenes.json", "midi_map.json"):
            raise PermissionError("locked by another process")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", flaky)
    c2 = make(tmp_path)                                    # the files exist but cannot be read right now
    assert c2.scenes_loaded is False and c2.scenes[0]["name"] == "Home" and c2.scenes[12] is None
    assert len(c2.midi.bindings) == len(midimod.default_bindings())
    c2.scene_save(5)                                       # still unreadable: nothing may be written
    c2.midi.bind("cc", 0, 9, "tap")
    assert (tmp_path / "scenes.json").read_bytes() == scenes_before
    assert (tmp_path / "midi_map.json").read_bytes() == map_before
    assert not list(tmp_path.glob("*.bad"))                # and nothing was quarantined
    monkeypatch.setattr(Path, "read_text", real)
    c2.set("zoom", 0.9)
    c2.scene_save(6)                                       # readable again: load first, then save on top
    saved = json.loads((tmp_path / "scenes.json").read_text())["scenes"]
    assert saved[12]["base"]["twirl"] == 0.42 and saved[6]["base"]["zoom"] == 0.9
    c2.midi.bind("cc", 0, 10, "autopilot")
    targets = [b["target"] for b in json.loads((tmp_path / "midi_map.json").read_text())["bindings"]]
    assert targets == ["param:hue", "autopilot"]


def test_corrupt_files_are_quarantined_under_unique_names(tmp_path):
    for _ in range(3):
        (tmp_path / "scenes.json").write_text("{nope")
        (tmp_path / "midi_map.json").write_text("[]")
        make(tmp_path)
    assert len(list(tmp_path.glob("scenes.json.*.bad"))) == 3            # earlier evidence is never overwritten
    assert len(list(tmp_path.glob("midi_map.json.*.bad"))) == 3
    assert not (tmp_path / "scenes.json").exists()


def test_recall_during_a_morph_starts_from_what_is_on_screen(tmp_path):
    c = make(tmp_path, bpm=120.0)
    c.scenes = [None] * 16
    c.set("swim", 0.3)
    c.scene_save(0)                                        # scenes 0 and 1: swim 0.3, no mods
    c.scene_save(1)
    c.set_mod("swim", 0, "bass", 0.4)                      # live look: swim pumped by the bass
    c.morph_bars = 1.0
    c.scene_recall(0)
    beats(c, 2)                                            # halfway: the outgoing mod is at half weight
    before = c.update(0, (1.0, 0, 0)).values["swim"]
    assert before == pytest.approx(0.5, abs=0.02)
    c.scene_recall(1)                                      # interrupt the morph
    after = c.update(0, (1.0, 0, 0)).values["swim"]
    assert after == pytest.approx(before, abs=0.01)        # no jump (it used to snap to 0.3)
    beats(c, 2)
    assert c.update(0, (1.0, 0, 0)).values["swim"] == pytest.approx(0.4, abs=0.02)   # fading from there
    beats(c, 2.2)
    assert c.update(0, (1.0, 0, 0)).values["swim"] == pytest.approx(0.3, abs=1e-3)


def test_removing_a_param_from_drift_releases_its_offset(tmp_path):
    c = make(tmp_path)
    c.apply([{"cmd": "autopilot", "drift": True, "drift_amt": 1.0, "drift_params": ["zoom", "hue"]}])
    run(c, 30.0)
    c._drift["zoom"][1] = 0.5                              # a definite offset
    assert abs(run(c, 1 / 60).values["zoom"]) > 0.1
    c.apply([{"cmd": "autopilot", "drift_params": ["hue"]}])
    assert "zoom" not in c._drift
    assert run(c, 1 / 60).values["zoom"] == pytest.approx(0.0, abs=1e-6)
    c.apply([{"cmd": "autopilot", "drift_params": ["hue", "zoom"]}])
    assert abs(run(c, 1 / 60).values["zoom"]) < 0.02       # back on the list: starts from zero, not the old offset


def test_midi_clock_tempo_survives_bunched_ticks(tmp_path):
    c = make(tmp_path)
    c.apply([{"cmd": "clock_src", "src": "midi"}])
    iv = 60.0 / 120.0 / 24
    for k in range(24 * 6):                                # the driver delivers ticks 5 at a time
        c.ft.t += iv
        if k % 5 == 4:
            for _ in range(5):
                c.midi.inject("clock", t=c.ft.t)
        c.update(iv, None)
    assert c.bpm == pytest.approx(120.0, rel=0.015)        # per-tick intervals would have read 24 or 300 BPM
    assert c.clock_active == "midi"


def test_midi_clock_follows_one_port_only(tmp_path):
    c = make(tmp_path)
    c.apply([{"cmd": "clock_src", "src": "midi"}])
    a, b = 60.0 / 100.0 / 24, 60.0 / 140.0 / 24
    ta = tb = c.ft.t
    for _ in range(2400):                                  # two clock sources at once: 100 BPM on A, 140 on B
        c.ft.t += 0.002
        while ta <= c.ft.t:
            c.midi.inject("clock", port="A", t=ta)
            ta += a
        while tb <= c.ft.t:
            c.midi.inject("clock", port="B", t=tb)
            tb += b
        c.update(0.002, None)
    assert c.bpm == pytest.approx(100.0, abs=0.5)          # locked to the first port heard, not a mix of both
    st = c.state()["clock"]
    assert st["clock_from"] == "A" and st["clock_ports"] == ["A", "B"] and st["clock_port"] is None
    c.apply([{"cmd": "clock_port", "port": "B"}])
    for _ in range(1200):
        c.ft.t += 0.002
        while ta <= c.ft.t:
            c.midi.inject("clock", port="A", t=ta)
            ta += a
        while tb <= c.ft.t:
            c.midi.inject("clock", port="B", t=tb)
            tb += b
        c.update(0.002, None)
    assert c.bpm == pytest.approx(140.0, abs=0.5)
    with pytest.raises(ValueError):
        validate({"cmd": "clock_port", "port": 5})


class FakeLink:
    made = 0

    def __init__(self, bpm):
        FakeLink.made += 1
        self.tempo, self.beat, self.enabled, self.quantum, self.toggles = bpm, 3.25, False, 4, 0

    def __setattr__(self, k, v):
        if k == "enabled" and "enabled" in self.__dict__ and v != self.__dict__["enabled"]:
            self.__dict__["toggles"] += 1
        self.__dict__[k] = v

    def force_beat(self, b):
        self.beat = b


def test_link_is_created_once_off_thread_and_bad_values_cannot_raise(tmp_path):
    FakeLink.made = 0
    c = make(tmp_path)
    c.linkb = control.LinkBridge(c.bpm, factory=FakeLink)
    main = threading.get_ident()
    made_on = []
    c.linkb._factory = lambda bpm: (made_on.append(threading.get_ident()), FakeLink(bpm))[1]
    c.apply([{"cmd": "clock_src", "src": "link"}])
    c.update(1 / 60, None)                                 # never blocks: internal clock until Link is up
    assert wait_for(lambda: (c.update(1 / 60, None), c.clock_active == "link")[1])
    assert c.beat == 3.25 and made_on and made_on[0] != main
    link = c.linkb.link
    for bad in (float("nan"), float("inf"), "x", None):
        link.beat = bad
        f = c.update(1 / 60, None)                         # a non-finite / broken reading falls back, never raises
        assert math.isfinite(f.clock["beat"]) and c.clock_active == "internal"
    link.beat, link.tempo = 8.0, float("nan")
    c.update(1 / 60, None)
    assert c.clock_active == "internal" and math.isfinite(c.bpm)
    link.tempo = 128.0
    c.update(1 / 60, None)
    assert c.clock_active == "link" and c.bpm == 128.0 and c.beat == 8.0
    for src in ("internal", "link", "internal", "link"):   # toggling never rebuilds the session
        c.apply([{"cmd": "clock_src", "src": src}])
        want = src == "link"
        assert wait_for(lambda: (c.update(1 / 60, None), c.linkb.enabled == want)[1])
    assert FakeLink.made == 1 and c.linkb.link is link and link.toggles == 5
    c.apply([{"cmd": "bpm", "v": 140}, {"cmd": "resync"}])
    assert wait_for(lambda: link.tempo == 140 and link.beat == 0.0)      # writes go through the Link thread
    c.close()
    assert link.enabled is False
    broken = control.LinkBridge(120, factory=lambda bpm: 1 / 0)          # aalink missing / failing: stays internal
    c2 = make(tmp_path)
    c2.linkb = broken
    c2.apply([{"cmd": "clock_src", "src": "link"}])
    for _ in range(20):
        c2.update(1 / 60, None)
        time.sleep(0.005)
    assert c2.clock_active == "internal" and "ZeroDivisionError" in c2.state()["clock"]["link_error"]


def autopilot_setup(tmp_path):
    c = make(tmp_path, bpm=240.0)
    c.scenes = [None] * 16
    for i in (0, 1, 2):
        c.set("zoom", i * 0.3)
        c.scene_save(i)
    c.scene_recall(0)
    c.apply([{"cmd": "autopilot", "on": True, "bars": 4, "mode": "next"}])     # every 16 beats
    return c


def test_manual_recall_holds_autopilot_off_for_a_full_period(tmp_path):
    c = autopilot_setup(tmp_path)
    beats(c, 14)
    c.scene_recall(2)                                      # the performer picks a scene 2 beats before autopilot's turn
    beats(c, 4)                                            # beat 18: autopilot's old slot has passed
    assert c.cur == 2
    beats(c, 10)                                           # beat 28: still inside the performer's full period
    assert c.cur == 2
    beats(c, 3)                                            # beat 31 > 14 + 16
    assert c.cur == 0


def test_manual_recall_wins_the_same_frame_as_autopilot(tmp_path):
    c = autopilot_setup(tmp_path)
    c.apply([{"cmd": "quantize", "q": "bar"}])
    beats(c, 13)
    c.scene_recall(2)                                      # queued for beat 16, exactly autopilot's boundary
    assert c.pending[0] == 2
    beats(c, 3.5)
    assert c.cur == 2 and c.pending is None                # autopilot ("next" would be 1) did not override it
    beats(c, 14)
    assert c.cur == 2                                      # and waits a full period from there
    beats(c, 3)
    assert c.cur == 0


def test_panic_switches_autopilot_off(tmp_path):
    c = autopilot_setup(tmp_path)
    c.apply([{"cmd": "panic"}])
    assert c.auto["on"] is False and c.state()["autopilot"]["on"] is False
    cur = c.cur
    beats(c, 40)
    assert c.cur == cur


def test_quantize_change_does_not_fire_a_pending_recall_early(tmp_path):
    c = make(tmp_path, bpm=120.0)
    c.set("zoom", 1.0)
    c.scene_save(3)
    c.set("zoom", 0.0)
    c.apply([{"cmd": "quantize", "q": "phrase"}])
    beats(c, 1.5)
    c.scene_recall(3)
    for q in ("beat", "bar", "off", "phrase", "bar"):
        c.apply([{"cmd": "quantize", "q": q}])
        c.update(0, None)
        assert c.base["zoom"] == 0.0 and c.pending is not None, q      # changing the setting is not a trigger
    beats(c, 2.4)                                          # beat 3.9: the new grid is the bar
    assert c.base["zoom"] == 0.0
    beats(c, 0.2)
    assert c.base["zoom"] == 1.0


def test_art_index_follows_deletes_and_steps_in_library_order(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    for k, name in enumerate(["zebra.png", "apple.png", "mango.png"]):       # created in this order
        (art / name).write_bytes(b"x")
        os_time = 1_700_000_000 + k * 10
        import os
        os.utime(art / name, (os_time, os_time))
    (art / "apple.depth.png").write_bytes(b"x")
    names = control.list_art(art)
    assert sorted(names) == ["apple.png", "mango.png", "zebra.png"]          # depth maps are not pieces
    c = Controller(tmp_path, midi=False, ai=False, restore=False, art_dir=art)
    c.set("art_index", 2)
    c.apply([{"cmd": "art_deleted", "i": 0}])              # a piece before the current one went away
    assert c.base["art_index"] == 1                        # same piece stays on screen
    c.apply([{"cmd": "art_deleted", "i": 1}])              # the current one itself: index stays (next piece slides in)
    c.apply([{"cmd": "art_deleted", "i": 5}])              # one after it: nothing to do
    assert c.base["art_index"] == 1
    c.apply([{"cmd": "art_step", "d": 1}, {"cmd": "art_step", "d": 1}])
    assert c.base["art_index"] == 0                        # wraps over the 3 pieces
    with pytest.raises(ValueError):
        validate({"cmd": "art_deleted", "i": -1})


def test_led_updates_never_block_the_caller(tmp_path):
    sent = []

    class SlowPort:
        def send(self, msg):
            time.sleep(0.25)                               # a wedged USB device
            sent.append((msg.channel, msg.note, msg.velocity))

        def close(self):
            pass

    m = midimod.MidiManager(None, scan=False)
    m._outs = {"APC mini mk2 1": SlowPort()}
    m._led_thread = threading.Thread(target=m._led_run, daemon=True)
    m._led_thread.start()
    t0 = time.perf_counter()
    for _ in range(50):
        m.leds({0: 5, 100: 1})
    assert time.perf_counter() - t0 < 0.05                 # the render thread only hands the dict over
    assert wait_for(lambda: len(sent) == 2, 3.0)
    assert sorted(sent) == [(0, 100, 1), (6, 0, 5)]        # pads on channel 7, round buttons on channel 1
    m.close()
    assert not hasattr(m, "_lock")                         # no lock is ever held across a device call
