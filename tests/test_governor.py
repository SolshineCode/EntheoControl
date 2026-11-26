"""Resource governor (governor.py): pure policy, fake sensors + fake clock, no GL."""
import json
import math
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import governor as gv  # noqa: E402
from params import PARAMS, defaults  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeSensors:
    def __init__(self, **kw):
        self.d = dict(gpu_util=97.0, vram_mb=2000.0, vram_total_mb=4096.0, gpu_temp=65.0, cpu=20.0, ram=40.0,
                      batt_pct=None, plugged=None, batt_secs=None)
        self.d.update(kw)

    def snapshot(self):
        return dict(self.d)

    def start(self):
        pass

    def stop(self):
        pass


def make(mode="suggest", **kw):
    clk, logs, holds = Clock(), [], []
    sens = kw.pop("sensors", None) or FakeSensors()
    g = gv.Governor(sensors=sens, clock=clk, log=logs.append, mode=mode, ai_hold=holds.append, **kw)
    return g, clk, logs, holds, sens


def look(**over):
    v = defaults()
    v.update(l_fractal=1.0, fractal_res=0.5, fractal_detail=0.6, l_beings=0.8, being_count=6.0, serpent=0.0)
    v.update(over)
    return v


def run(g, clk, seconds, ms, vals=None, base=None):
    """Feed frames of `ms` wall time for `seconds`. Returns the last processed vals."""
    vals = vals if vals is not None else look()
    base = base if base is not None else dict(vals)
    out = None
    n = int(seconds * 1000.0 / ms)
    for _ in range(max(1, n)):
        clk.t += ms / 1000.0
        out = dict(vals)
        g.frame(out, base)
    return out


# ------------------------------------------------------------------ frame stats

def test_frame_stats_percentiles_and_long_frames():
    fs = gv.FrameStats(maxlen=100)
    for i in range(90):
        fs.add(16.0)
    for i in range(10):
        fs.add(40.0)
    assert fs.p50() == pytest.approx(16.0)
    assert fs.p95() == pytest.approx(40.0)
    assert fs.long_frames() == 10
    fs.clear()
    assert fs.n == 0 and fs.p95() is None


def test_idle_gaps_are_not_counted_as_frames():
    g, clk, *_ = make()
    run(g, clk, 2.0, 16.7)
    clk.t += 1.5                         # projector unplugged: the loop idled
    g.frame(look(), look())
    assert max(g.stats.values()) < 20.0


# ------------------------------------------------------------------ suggest mode

def test_healthy_show_does_nothing():
    g, clk, logs, holds, _ = make("auto")
    out = run(g, clk, 30.0, 16.7)
    st = g.status()
    assert st["level"] == "ok" and st["suggestion"] is None and st["active"] == []
    assert out == look()


def test_suggest_mode_names_the_costly_layer_and_never_acts_alone():
    g, clk, logs, holds, _ = make("suggest")
    g.pass_ms.update(l_fractal=9.0, l_beings=1.0, scene=6.0)
    out = run(g, clk, 4.0, 25.0)
    sug = g.status()["suggestion"]
    assert sug is not None and sug["id"] == "res:fractal_res"
    assert "Fractal" in sug["text"] and "9" in sug["text"] and "0.35" in sug["text"]
    assert sug["countdown"] is None
    out = run(g, clk, 30.0, 25.0)
    assert out["fractal_res"] == 0.5 and g.status()["active"] == []
    g.command({"cmd": "gov", "op": "apply", "id": "res:fractal_res"})
    out = run(g, clk, 0.1, 25.0)
    assert out["fractal_res"] == pytest.approx(0.35)
    assert any("[governor]" in m and "fractal_res" in m for m in logs)


def test_suggestion_clears_when_the_strain_goes_away():
    g, clk, *_ = make("suggest")
    run(g, clk, 4.0, 25.0)
    assert g.status()["suggestion"] is not None
    run(g, clk, 12.0, 16.7)
    assert g.status()["suggestion"] is None


def test_ignore_snoozes_suggestions_for_ten_minutes():
    g, clk, *_ = make("suggest")
    run(g, clk, 4.0, 25.0)
    g.command({"cmd": "gov", "op": "ignore", "id": g.status()["suggestion"]["id"], "minutes": 10})
    run(g, clk, 300.0, 25.0)
    assert g.status()["suggestion"] is None
    run(g, clk, 305.0, 25.0)
    assert g.status()["suggestion"] is not None


def test_always_auto_switches_mode_and_applies():
    store = {}
    g, clk, *_ = make("suggest", store=store)
    run(g, clk, 4.0, 25.0)
    g.command({"cmd": "gov", "op": "always_auto", "id": g.status()["suggestion"]["id"]})
    out = run(g, clk, 0.1, 25.0)
    assert g.mode == "auto" and store["governor"]["mode"] == "auto"
    assert out["fractal_res"] < 0.5


# ------------------------------------------------------------------ auto mode, hysteresis

def test_auto_warns_then_applies_after_countdown():
    g, clk, logs, *_ = make("auto")
    run(g, clk, 4.0, 25.0)
    sug = g.status()["suggestion"]
    assert sug is not None and sug["countdown"] is not None and 0 < sug["countdown"] <= 5.0
    out = run(g, clk, 3.0, 25.0)
    assert out["fractal_res"] == 0.5                  # still counting down
    out = run(g, clk, 2.6, 25.0)
    assert out["fractal_res"] == pytest.approx(0.35)


def test_auto_countdown_can_be_cancelled():
    g, clk, *_ = make("auto")
    run(g, clk, 3.5, 25.0)
    g.command({"cmd": "gov", "op": "cancel"})
    out = run(g, clk, 8.0, 25.0)
    assert out["fractal_res"] == 0.5 and g.status()["active"] == []


def test_one_step_at_a_time():
    g, clk, *_ = make("auto")
    times = []
    for i in range(120):                              # 60 s of sustained strain
        run(g, clk, 0.5, 25.0)
        times.append(len(g.status()["active"]))
    # steps arrive one by one, at least ~8 s apart (3 s sustained + 5 s countdown, after a settle)
    firsts = [times.index(k) for k in sorted(set(times)) if k > 0]
    assert firsts == sorted(firsts)
    assert all(b - a >= 14 for a, b in zip(firsts, firsts[1:]))   # 14 half-seconds = 7 s
    assert len(g.status()["active"]) >= 2


def test_recovery_needs_fifteen_seconds_of_headroom_and_is_lifo():
    g, clk, *_ = make("auto")
    for _ in range(60):
        run(g, clk, 0.5, 25.0)
    run(g, clk, 3.0, 16.7)                                          # strain over (window flushed)
    applied = [a["id"] for a in g.status()["active"]]
    assert len(applied) >= 2
    run(g, clk, 7.0, 16.7)
    assert [a["id"] for a in g.status()["active"]] == applied     # not yet: 10 s of headroom
    run(g, clk, 8.0, 16.7)
    now_ids = [a["id"] for a in g.status()["active"]]
    assert now_ids == applied[:-1]                                   # last in, first out, one step
    run(g, clk, 15.0, 16.7)
    assert [a["id"] for a in g.status()["active"]] == applied[:-2]


def test_no_oscillation_when_restore_brings_the_strain_back():
    g, clk, *_ = make("auto")
    restores = 0
    prev = 0
    for cycle in range(40):                           # 40 x (bad 10 s / good 17 s) ~ 18 minutes
        for _ in range(20):
            run(g, clk, 0.5, 25.0)
        for _ in range(34):
            run(g, clk, 0.5, 16.7)
            n = len(g.status()["active"])
            if n < prev:
                restores += 1
            prev = n
        prev = len(g.status()["active"])
    assert restores <= 8                               # backoff: a step that keeps coming back sticks


def test_level_thresholds():
    g, clk, *_ = make("off")
    run(g, clk, 3.0, 16.7)
    assert g.status()["level"] == "ok"
    run(g, clk, 4.0, 22.0)
    assert g.status()["level"] == "strained"
    run(g, clk, 4.0, 40.0)
    assert g.status()["level"] == "critical"


def test_vram_temp_ram_thresholds():
    for kw, lvl, reason in ((dict(vram_mb=3700.0), "critical", "VRAM"), (dict(gpu_temp=89.0), "strained", "hot"),
                            (dict(thermal=True, throttle=0x40), "strained", "thermal"),
                            (dict(ram=93.0), "strained", "RAM")):
        g, clk, *_ = make("off", sensors=FakeSensors(**kw))
        run(g, clk, 3.0, 16.7)
        st = g.status()
        assert st["level"] == lvl, kw
        assert any(reason in r for r in st["reasons"]), st["reasons"]


# ------------------------------------------------------------------ what it touches

def test_degrade_order_res_then_detail_counts_then_ai_then_floor():
    g, clk, logs, holds, _ = make("auto", sensors=FakeSensors())
    g.pass_ms.update(l_fractal=3.0, l_land=8.0)
    vals = look(l_land=1.0, land_res=0.5, serpent=1.0, serpent_count=6.0)
    g.ai_status = dict(available=True, busy=True)
    order = []
    for _ in range(400):
        run(g, clk, 0.5, 45.0, vals, dict(vals))
        for a in g.status()["active"]:
            if a["id"] not in order:
                order.append(a["id"])
    assert order[0] == "res:land_res" and order[1] == "res:fractal_res"     # costliest pass first
    cats = [i.split(":")[0] for i in order]
    assert cats.index("ai") > max(i for i, c in enumerate(cats) if c in ("detail", "count"))
    assert cats[-1] == "floor" or "floor" in cats
    assert set(cats[:2]) == {"res"}
    assert holds and holds[-1] is True
    assert not any("tick failed" in m for m in logs), [m for m in logs if "tick failed" in m][:2]


def test_apply_under_persistent_sensor_strain_does_not_break_the_tick():
    g, clk, logs, *_ = make("auto", sensors=FakeSensors(vram_mb=3700.0, thermal=True, throttle=0x20))
    for _ in range(60):
        run(g, clk, 0.5, 16.7)
    assert len(g.status()["active"]) >= 2
    assert not any("tick failed" in m for m in logs)


def test_serpent_text_does_not_claim_the_whole_scene_cost():
    g, clk, *_ = make("suggest")
    g.pass_ms["scene"] = 22.0
    g._raw = dict(serpent=1.0, serpent_count=8.0)
    t = g._text(g._step["count:serpent_count"])
    assert "main scene" in t and "4 instead of 8" in t


def test_never_blacks_out_or_touches_opacities_master_or_triggers():
    g, clk, *_ = make("auto", sensors=FakeSensors(vram_mb=3900.0, gpu_temp=95.0, ram=97.0))
    vals = look(l_land=1.0, l_swarm=1.0, l_art=1.0, serpent=1.0)
    out = None
    for _ in range(600):
        out = run(g, clk, 0.5, 60.0, vals, dict(vals))
    untouched = [n for n in PARAMS if n.startswith("l_") or n in ("master", "blackout", "flash", "strobe", "serpent",
                                                                   "trails", "luma_key")]
    for n in untouched:
        assert out[n] == vals[n], n


def test_layers_that_are_off_are_never_governed():
    g, clk, *_ = make("auto")
    vals = look(l_fractal=0.0, l_beings=0.0)
    out = None
    for _ in range(120):
        out = run(g, clk, 0.5, 25.0, vals, dict(vals))
    assert out["fractal_res"] == 0.5 and out["being_count"] == 6.0
    assert not any(a["id"].endswith("fractal_res") or a["id"].endswith("being_count") for a in g.status()["active"])


def test_visible_reductions_fade_over_about_a_second():
    g, clk, *_ = make("suggest")
    run(g, clk, 4.0, 25.0)
    g.command({"cmd": "gov", "op": "apply", "id": "detail:fractal_detail"})
    v1 = run(g, clk, 0.5, 20.0)["fractal_detail"]
    v2 = run(g, clk, 0.7, 20.0)["fractal_detail"]
    assert 0.3 < v1 < 0.6 and v2 == pytest.approx(0.3, abs=1e-6)
    g.command({"cmd": "gov", "op": "restore", "id": "detail:fractal_detail"})
    v3 = run(g, clk, 0.5, 20.0)["fractal_detail"]
    v4 = run(g, clk, 0.7, 20.0)["fractal_detail"]
    assert 0.3 < v3 < 0.6 and v4 == pytest.approx(0.6)


def test_user_moving_a_governed_fader_wins():
    g, clk, logs, *_ = make("suggest")
    vals = look()
    run(g, clk, 4.0, 25.0, vals, dict(vals))
    g.command({"cmd": "gov", "op": "apply", "id": "res:fractal_res"})
    assert run(g, clk, 1.0, 25.0, vals, dict(vals))["fractal_res"] == pytest.approx(0.35)
    vals2 = dict(vals, fractal_res=0.8)
    out = run(g, clk, 1.0, 25.0, vals2, dict(vals2))
    assert out["fractal_res"] == pytest.approx(0.8)
    assert all(a["id"] != "res:fractal_res" for a in g.status()["active"])
    assert any("user" in m for m in logs)


def test_vram_critical_pauses_ai_first_and_releases_it():
    sens = FakeSensors(vram_mb=3800.0)
    g, clk, logs, holds, _ = make("auto", sensors=sens)
    g.ai_status = dict(available=True, busy=True)
    for _ in range(20):
        run(g, clk, 0.5, 16.7)
    assert g.status()["active"][0]["id"] == "ai:pause" and holds == [True]
    sens.d["vram_mb"] = 2500.0
    run(g, clk, 16.0, 16.7)
    assert holds == [True, False] and g.status()["active"] == []


def test_generic_res_params_are_discovered():
    p = dict(PARAMS)
    p["l_text"] = (0.0, 1.0, 0.0, "layers", "Text")
    p["text_res"] = (0.25, 1.0, 0.75, "text", "Render scale")
    steps = gv.build_steps(p)
    s = next(s for s in steps if s.id == "res:text_res")
    assert s.gate == ("l_text",) and s.kind == "res"


def test_off_mode_does_nothing_and_releases():
    g, clk, *_ = make("suggest")
    run(g, clk, 4.0, 25.0)
    g.command({"cmd": "gov", "op": "apply", "id": "detail:fractal_detail"})
    run(g, clk, 2.0, 25.0)
    g.command({"cmd": "gov", "op": "mode", "mode": "off"})
    out = run(g, clk, 60.0, 40.0)
    assert out["fractal_detail"] == pytest.approx(0.6) and g.status()["suggestion"] is None
    assert g.status()["active"] == []


def test_restore_all_button():
    g, clk, *_ = make("auto")
    for _ in range(60):
        run(g, clk, 0.5, 25.0)
    assert g.status()["active"]
    g.command({"cmd": "gov", "op": "restore", "id": "all"})
    run(g, clk, 1.2, 25.0)
    assert g.status()["active"] == []


# ------------------------------------------------------------------ battery

def test_battery_warnings_and_low_power_profile():
    sens = FakeSensors(batt_pct=55.0, plugged=False, batt_secs=4800)
    g, clk, *_ = make("suggest", sensors=sens)
    run(g, clk, 2.0, 16.7)
    w = g.status()["warnings"]
    assert any("Running on battery: 55%" in x["text"] and "80 min" in x["text"] for x in w)
    sens.d["batt_pct"] = 18.0
    run(g, clk, 4.0, 16.7)
    sug = g.status()["suggestion"]
    assert sug and sug["id"] == "low_power"
    g.command({"cmd": "gov", "op": "apply", "id": "low_power"})
    out = run(g, clk, 1.5, 16.7)
    act = g.status()["active"]
    assert len(act) >= 3 and all("battery" in a["reasons"] for a in act)
    assert out["fractal_res"] < 0.5 and out["being_count"] < 6
    sens.d["batt_pct"] = 8.0
    run(g, clk, 2.0, 16.7)
    assert any(x["level"] == "critical" for x in g.status()["warnings"])
    sens.d.update(plugged=True)
    out = run(g, clk, 2.0, 16.7)
    assert g.status()["active"] == [] and out["fractal_res"] == 0.5
    assert not any("battery" in x["text"].lower() for x in g.status()["warnings"])


def test_auto_applies_low_power_on_low_battery():
    sens = FakeSensors(batt_pct=15.0, plugged=False, batt_secs=-1)
    g, clk, *_ = make("auto", sensors=sens)
    run(g, clk, 10.0, 16.7)
    assert any("battery" in a["reasons"] for a in g.status()["active"])
    assert not any("min" in x["text"] for x in g.status()["warnings"] if "battery" in x["text"])


# ------------------------------------------------------------------ persistence, commands, status

def test_mode_is_persisted_in_the_store():
    store = {"governor": {"mode": "auto"}}
    g, *_ = make(None, store=store)
    assert g.mode == "auto"
    g.command({"cmd": "gov", "op": "mode", "mode": "off"})
    assert store["governor"]["mode"] == "off"
    g2, *_ = make(None, store={"governor": {"mode": "bogus"}})
    assert g2.mode == "suggest"
    g3, *_ = make("auto", store={"governor": {"mode": "off"}})      # an explicit mode (CLI) wins
    assert g3.mode == "auto"


def test_validate_command():
    ok = gv.validate_command({"cmd": "gov", "op": "mode", "mode": "auto", "junk": 1})
    assert ok == {"cmd": "gov", "type": "gov", "op": "mode", "mode": "auto"}
    assert gv.validate_command({"cmd": "gov", "op": "ignore", "id": "res:fractal_res", "minutes": 999})["minutes"] == 120
    for bad in ({"cmd": "gov", "op": "nope"}, {"cmd": "gov", "op": "mode", "mode": "x"},
                {"cmd": "gov", "op": "apply", "id": "../../x"}, {"cmd": "gov", "op": "apply", "id": 3}):
        with pytest.raises(ValueError):
            gv.validate_command(bad)


def test_remote_accepts_gov_commands():
    import remote_control as rc
    assert rc.validate_command({"cmd": "gov", "op": "restore", "id": "all"})["op"] == "restore"
    with pytest.raises(ValueError):
        rc.validate_command({"cmd": "gov", "op": "x"})


def test_status_is_json_and_carries_telemetry():
    sens = FakeSensors(batt_pct=80.0, plugged=True)
    g, clk, *_ = make("suggest", sensors=sens)
    g.pass_ms.update(l_fractal=4.2, scene=5.0)
    run(g, clk, 2.0, 16.7)
    st = json.loads(json.dumps(g.status()))
    for k in ("mode", "level", "fps", "p50", "p95", "gpu_util", "vram_mb", "gpu_temp", "batt_pct", "plugged", "passes",
              "active", "suggestion", "warnings"):
        assert k in st, k
    assert 55 < st["fps"] < 65 and st["passes"]["l_fractal"] == 4.2


def test_frame_overhead_is_tiny():
    g, clk, *_ = make("suggest")
    vals, base = look(), look()
    run(g, clk, 4.0, 25.0)
    for i in ("res:fractal_res", "detail:fractal_detail", "count:being_count"):
        g.command({"cmd": "gov", "op": "apply", "id": i})
    n = 5000
    t0 = time.perf_counter()
    for _ in range(n):
        clk.t += 0.0167
        g.frame(dict(vals), base)
    per = (time.perf_counter() - t0) / n * 1000.0
    assert per < 0.2, f"{per:.3f} ms/frame"


# ------------------------------------------------------------------ pass timer + sensors

class FakeQuery:
    made = 0

    def __init__(self):
        FakeQuery.made += 1
        self.open = False
        self.elapsed = 0

    def __enter__(self):
        assert not self.open
        self.open = True
        return self

    def __exit__(self, *a):
        self.open = False
        self.elapsed = 4_000_000          # 4 ms


def test_pass_timer_samples_lags_and_averages():
    FakeQuery.made = 0
    out = {}
    pt = gv.PassTimer(FakeQuery, every=10, lag=2, sink=out)
    for f in range(40):
        pt.begin_frame()
        pt.start("l_fractal")
        pt.start("scene")                 # starting a new section ends the previous one
        pt.stop()
    assert out["l_fractal"] == pytest.approx(4.0) and out["scene"] == pytest.approx(4.0)
    assert FakeQuery.made == 2            # queries are reused


def test_pass_timer_disables_itself_on_failure():
    def boom():
        raise RuntimeError("no timer queries")
    pt = gv.PassTimer(boom, every=1, sink={})
    for _ in range(5):
        pt.begin_frame()
        pt.start("scene")
        pt.stop()
    assert pt.enabled is False


def test_parse_nvidia_smi_and_battery():
    assert gv.parse_smi("34, 2711, 4096, 68\n") == dict(gpu_util=34.0, vram_mb=2711.0, vram_total_mb=4096.0, gpu_temp=68.0)
    assert gv.parse_smi("[N/A], 100, 4096, [N/A]")["gpu_util"] is None
    assert gv.parse_smi("garbage") == {}
    d = gv.parse_smi("95, 3117, 4096, 74, 0x0000000000000024")
    assert d["throttle"] == 0x24 and d["thermal"] is True
    assert gv.parse_smi("95, 3117, 4096, 74, 0x0000000000000004")["thermal"] is False
    assert gv.parse_smi("95, 3117, 4096, 74, [N/A]")["throttle"] is None


def test_thermal_slowdown_is_strained_and_blocks_headroom():
    g, clk, *_ = make("off", sensors=FakeSensors(gpu_temp=74.0, throttle=0x20, thermal=True))
    run(g, clk, 3.0, 16.7)
    st = g.status()
    assert st["level"] == "strained" and any("thermal" in r for r in st["reasons"])
    assert g.headroom is False and st["throttle"] == "0x20"

    class B:
        percent, secsleft, power_plugged = 42, 3600, False
    assert gv.battery_from_psutil(B()) == dict(batt_pct=42.0, plugged=False, batt_secs=3600)
    B.secsleft = -2
    assert gv.battery_from_psutil(B())["batt_secs"] is None
    assert gv.battery_from_psutil(None) == {}


def test_sensors_poll_with_fake_readers_and_threads_stop():
    calls = []
    s = gv.Sensors(gpu=lambda: calls.append("g") or dict(vram_mb=1.0), battery=lambda: dict(batt_pct=50.0),
                   host=lambda: dict(cpu=5.0), interval=0.02, gpu_interval=0.02)
    s.poll_once(force=True)
    snap = s.snapshot()
    assert snap["vram_mb"] == 1.0 and snap["batt_pct"] == 50.0 and snap["cpu"] == 5.0
    s.start()
    time.sleep(0.1)
    s.stop()
    assert not s.alive()

    def bad():
        raise OSError("nvidia-smi missing")
    s2 = gv.Sensors(gpu=bad, battery=None, host=None)
    s2.poll_once(force=True)
    assert s2.snapshot().get("vram_mb") is None


def test_real_sensor_readers_never_raise():
    for fn in (gv.read_battery, gv.read_host):
        assert isinstance(fn(), dict)
    r = gv.make_gpu_reader()
    if r is not None:
        assert isinstance(r(), dict)


# ------------------------------------------------------------------ hooks in other modules

def test_controller_persists_extension_settings(tmp_path):
    import control
    c = control.Controller(state_dir=tmp_path, restore=True, midi=False, ai=False)
    c.ext["governor"] = {"mode": "auto"}
    assert c._autosave()
    c.close()
    c2 = control.Controller(state_dir=tmp_path, restore=True, midi=False, ai=False)
    assert c2.ext.get("governor") == {"mode": "auto"}
    c2.close()


def test_ai_client_hold_keeps_jobs_queued():
    import ai_link
    a = ai_link.AIClient([], start=False)
    a.submit({"prompt": "x"})
    a.hold = True
    assert a._take_due(time.monotonic() + 1) == []
    a.hold = False
    assert len(a._take_due(time.monotonic() + 1)) == 1


def test_exhausted_advice_is_logged_rarely():
    g, clk, logs, *_ = make("auto")
    vals = look(l_fractal=0.0, l_beings=0.0)
    g.pass_ms["scene"] = 40.0
    for i in range(240):                                  # 2 min critical, nothing to lower
        g.pass_ms["scene"] = 40.0 + i % 7
        run(g, clk, 0.5, 45.0, vals, dict(vals))
    adv = [m for m in logs if "Nothing left" in m]
    assert 1 <= len(adv) <= 3 and "960x600" in g.status()["advice"]


def test_panel_meter_toast_and_health_card_in_a_real_browser(tmp_path):
    pw_api = pytest.importorskip("playwright.sync_api")
    import socket
    import threading
    import control
    from remote_control import RemoteControl
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art", tex_dir=tmp_path / "tex")
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
    sens = FakeSensors(gpu_util=96.0, vram_mb=3100.0, gpu_temp=74.0, throttle=0x20, thermal=True, batt_pct=64.0,
                       plugged=False, batt_secs=5400)
    g = gv.Governor(sensors=sens, log=lambda m: None, mode="suggest", store=ctl.ext)
    g.pass_ms.update(l_fractal=9.1, scene=6.2, l_beings=1.4)
    stop, lock = threading.Event(), threading.Lock()
    vals = look()

    def loop():
        while not stop.is_set():
            with lock:
                for c in ctl.apply(rc.drain()):
                    if c.get("cmd") == "gov":
                        g.command(c)
                ctl.update(1 / 60, [0.5, 0.3, 0.1])
                g.frame(dict(vals), dict(vals))
                rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60,
                                      gov=g.status()))
            time.sleep(1 / 60)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        with pw_api.sync_playwright() as pw:
            br = None
            for ch in ("msedge", "chrome", None):
                try:
                    br = pw.chromium.launch(channel=ch, headless=True) if ch else pw.chromium.launch(headless=True)
                    break
                except Exception:
                    pass
            if br is None:
                pytest.skip("no browser")
            page = br.new_page(viewport=dict(width=1400, height=900))
            errs = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/?k=tok")
            page.wait_for_function("document.getElementById('govm').textContent.indexOf('GPU 96%')>=0", timeout=8000)
            txt = page.inner_text("#govm")
            assert "74" in txt and "64%" in txt and "throttling" in txt
            with lock:
                g.suggestion = dict(id="res:fractal_res", kind="perf", text=g._text(g._step["res:fractal_res"]),
                                    why="frame time p95 24.0 ms", deadline=None, t=0)
            page.wait_for_selector("#govt.on", timeout=5000)
            assert "Fractal layer is costing 9.1 ms/frame" in page.inner_text("#govt")
            page.click("#tabs button[data-tab=midi]")
            page.wait_for_selector(".govc #govpass .pr", timeout=5000)
            assert "Fractal" in page.inner_text(".govc")
            page.screenshot(path=str(ROOT / "test_frames" / "governor_panel.png"))
            page.click("#govt button.pri")
            page.wait_for_function("document.querySelector('#govred').textContent.indexOf('RESTORE')>=0", timeout=5000)
            with lock:
                assert [a["id"] for a in g.status()["active"]] == ["res:fractal_res"]
            page.wait_for_timeout(1200)
            page.locator(".govc").screenshot(path=str(ROOT / "test_frames" / "governor_health_card.png"))
            page.click(".govc #govmode button:has-text('AUTO')")
            page.wait_for_function("document.querySelector('#govmode button.on').textContent==='AUTO'", timeout=5000)
            assert ctl.ext["governor"]["mode"] == "auto"
            page.click(".govc #govprof button:has-text('BATTERY SAVER')")
            page.wait_for_function("document.querySelector('#govprof button.on') && document.querySelector('#govprof button.on').textContent==='BATTERY SAVER'", timeout=5000)
            assert ctl.ext["governor"]["power_profile"] == "saver"
            page.fill("#govshow", "4")
            page.click(".govc button:has-text('SET')")
            page.wait_for_function("document.querySelector('#govpow').textContent.indexOf('show:')>=0", timeout=5000)
            assert "cap 30 fps" in page.inner_text("#govpow")
            page.locator(".govc").screenshot(path=str(ROOT / "test_frames" / "governor_power_card.png"))
            assert not errs, errs
            br.close()
    finally:
        stop.set()
        t.join(timeout=2)
        ctl.close()
        rc.stop()


# ------------------------------------------------------------------ Gemini review regressions (2026-11-25)

def test_r1_sensor_readings_replace_and_expire():
    t = [0.0]
    seq = [dict(thermal=True, throttle=0x20, vram_mb=3700.0), dict(vram_mb=1000.0)]

    def gpu():
        if seq:
            return seq.pop(0)
        raise OSError("smi timeout")
    s = gv.Sensors(gpu=gpu, battery=None, host=None, gpu_interval=2.0, clock=lambda: t[0])
    s.poll_once(force=True)
    assert s.snapshot()["thermal"] is True
    t[0] += 2.1
    s.poll_once()
    snap = s.snapshot()
    assert "thermal" not in snap and snap["vram_mb"] == 1000.0          # omitted keys clear (replace, not merge)
    for _ in range(30):                                                  # reader keeps failing: reading goes stale
        t[0] += 2.1
        s.poll_once()
    assert "vram_mb" not in s.snapshot()


def test_r1_failing_reader_backs_off():
    t, calls = [0.0], []

    def bad():
        calls.append(t[0])
        raise OSError("timeout")
    s = gv.Sensors(gpu=bad, battery=None, host=None, gpu_interval=2.0, clock=lambda: t[0])
    for _ in range(600):                                                 # 60 s at 0.1 s steps
        s.poll_once()
        t[0] += 0.1
    assert len(calls) <= 8                                               # 2, 4, 8, 16, 30 s apart, not every 2 s


class _R:
    def __init__(self, code, out):
        self.returncode, self.stdout = code, out


def test_r1_throttle_field_only_dropped_after_repeated_rejection():
    fails = [True]                                                       # the 5-field query fails ONCE (contention)

    def run(args, **kw):
        five = "clocks_throttle_reasons" in args[1]
        if five and fails and fails.pop():
            return _R(1, "")
        return _R(0, "50, 1000, 4096, 70, 0x20" if five else "50, 1000, 4096, 70")
    rd = gv.make_smi_reader("nvidia-smi", run=run)
    rd()
    assert rd()["thermal"] is True                                       # still asking for throttle reasons

    def run_bad(args, **kw):
        five = "clocks_throttle_reasons" in args[1]
        return _R(2, "") if five else _R(0, "50, 1000, 4096, 70")
    rd2 = gv.make_smi_reader("nvidia-smi", run=run_bad)
    seen = []
    for _ in range(5):
        seen.append(rd2())
    assert seen[-1].get("thermal") is None and seen[-1]["vram_mb"] == 1000.0
    assert all("thermal" not in d or d["thermal"] is None for d in seen)


def test_r1_sensor_thread_stops_quickly():
    s = gv.Sensors(gpu=lambda: time.sleep(0.05) or {}, battery=None, host=None, interval=0.01, gpu_interval=0.01)
    s.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    s.stop()
    assert time.perf_counter() - t0 < 1.5


def test_r2_user_move_drops_every_reduction_on_that_param_and_cools_down_the_param():
    g, clk, logs, *_ = make("auto")
    vals = look()
    run(g, clk, 4.0, 25.0, vals, dict(vals))
    g.command({"cmd": "gov", "op": "apply", "id": "res:fractal_res"})
    assert run(g, clk, 0.5, 25.0, vals, dict(vals))["fractal_res"] == pytest.approx(0.35)
    vals2 = dict(vals, fractal_res=0.8)
    out = None
    for _ in range(100):                                                 # 50 s of continued strain in AUTO
        out = run(g, clk, 0.5, 25.0, vals2, dict(vals2))
    assert out["fractal_res"] == pytest.approx(0.8)
    assert not any(a["param"] == "fractal_res" for a in g.status()["active"])


def test_r2_panel_restore_of_a_step_releases_the_whole_param():
    g, clk, *_ = make("suggest")
    run(g, clk, 4.0, 25.0)
    g.command({"cmd": "gov", "op": "apply", "id": "res:fractal_res"})
    g.command({"cmd": "gov", "op": "apply", "id": "floor:fractal_res"})
    g.command({"cmd": "gov", "op": "restore", "id": "res:fractal_res"})
    assert run(g, clk, 0.5, 25.0)["fractal_res"] == 0.5


def test_r2_scene_morph_is_not_a_user_move_when_edits_are_known():
    edits = {}
    g, clk, logs, *_ = make("suggest", edit_source=lambda: edits)
    vals = look()
    run(g, clk, 4.0, 25.0, vals, dict(vals))
    g.command({"cmd": "gov", "op": "apply", "id": "res:fractal_res"})
    run(g, clk, 0.5, 25.0, vals, dict(vals))
    out = None
    for k in range(20):                                                  # a morph rewrites the base every frame
        v = dict(vals, fractal_res=0.5 + 0.02 * k)
        out = run(g, clk, 0.1, 25.0, v, dict(v))
    assert any(a["id"] == "res:fractal_res" for a in g.status()["active"])
    assert out["fractal_res"] == pytest.approx(gv._snap(0.88 * 0.7, 0.25))
    edits["fractal_res"] = 1                                             # a hand on the fader
    v = dict(vals, fractal_res=0.9)
    assert run(g, clk, 0.5, 25.0, v, dict(v))["fractal_res"] == pytest.approx(0.9)
    assert not any(a["id"] == "res:fractal_res" for a in g.status()["active"])


def test_r2_panic_clears_a_pending_countdown():
    pan = [0]
    g, clk, *_ = make("auto", panic_source=lambda: pan[0])
    run(g, clk, 4.0, 25.0)
    assert g.status()["suggestion"]["countdown"] is not None
    pan[0] += 1
    run(g, clk, 2.0, 25.0)
    assert g.status()["suggestion"] is None
    out = run(g, clk, 4.0, 25.0)
    assert out["fractal_res"] == 0.5


def test_r3_a_sticky_top_step_does_not_block_earlier_recoveries():
    g, clk, *_ = make("suggest")
    run(g, clk, 4.0, 25.0)
    g.command({"cmd": "gov", "op": "apply", "id": "res:fractal_res"})
    g.command({"cmd": "gov", "op": "apply", "id": "count:being_count"})
    g.sticky["count:being_count"] = (3, clk.t + 1800)
    run(g, clk, 20.0, 16.7)
    assert [a["id"] for a in g.status()["active"]] == ["count:being_count"]


def test_r3_sticky_backoff_decays_after_a_clean_stretch():
    g, clk, *_ = make("auto")
    g.sticky["res:fractal_res"] = (4, clk.t - 1)
    g.restored_at["res:fractal_res"] = clk.t - 1000
    run(g, clk, 2.0, 16.7)
    assert "res:fractal_res" not in g.sticky


def test_r6_nonfinite_values_never_reach_the_json():
    assert gv.parse_smi("nan, inf, 4096, 70") == dict(gpu_util=None, vram_mb=None, vram_total_mb=4096.0, gpu_temp=70.0)
    g, clk, *_ = make("suggest", sensors=FakeSensors(gpu_temp=float("nan"), ram=float("inf"), cpu=float("-inf")))
    g.pass_ms["scene"] = float("nan")
    run(g, clk, 2.0, 16.7)
    json.dumps(g.status(), allow_nan=False)


def test_r7_stale_suggestion_is_replaced_when_its_layer_goes_off():
    g, clk, *_ = make("suggest")
    vals = look(l_land=1.0, land_res=0.5)
    g.pass_ms.update(l_fractal=9.0, l_land=3.0)
    run(g, clk, 4.0, 25.0, vals, dict(vals))
    assert g.status()["suggestion"]["id"] == "res:fractal_res"
    v2 = dict(vals, l_fractal=0.0)
    run(g, clk, 1.5, 25.0, v2, dict(v2))
    assert g.status()["suggestion"]["id"] == "res:land_res"


def test_r7_auto_countdown_rechecks_before_applying():
    g, clk, *_ = make("auto")
    vals = look(l_beings=0.0)
    run(g, clk, 4.0, 25.0, vals, dict(vals))
    assert g.status()["suggestion"]["id"] == "res:fractal_res"
    v2 = dict(vals, l_fractal=0.0)
    run(g, clk, 6.0, 25.0, v2, dict(v2))
    assert not any(a["id"] == "res:fractal_res" for a in g.status()["active"])


def test_r7_apply_is_ignored_in_off_mode():
    g, clk, *_ = make("off")
    run(g, clk, 2.0, 25.0)
    g.command({"cmd": "gov", "op": "apply", "id": "res:fractal_res"})
    g.command({"cmd": "gov", "op": "low_power", "on": True})
    assert run(g, clk, 1.0, 25.0)["fractal_res"] == 0.5 and g.status()["active"] == []


def test_r8_pass_timer_lag_and_engine_closes_each_pass_section():
    assert gv.PassTimer(lambda: None).lag >= 5
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    assert src.count("gov.timer.stop()") >= 2


def test_r9_land_res_is_gated_on_the_landscape_only():
    steps = {s.id: s for s in gv.build_steps()}
    assert steps["res:land_res"].gate == ("l_land",) and steps["floor:land_res"].gate == ("l_land",)


def test_r12_ai_hold_release_wakes_the_link():
    import ai_link
    a = ai_link.AIClient([], start=False)
    a.set_hold(True)
    assert a.hold is True
    a._wake.clear()
    a.set_hold(False)
    assert a.hold is False and a._wake.is_set()


def test_r13_low_power_on_must_be_a_bool():
    with pytest.raises(ValueError):
        gv.validate_command({"cmd": "gov", "op": "low_power", "on": "false"})
    assert gv.validate_command({"cmd": "gov", "op": "low_power", "on": False})["on"] is False


# ------------------------------------------------------------------ PR1 B9: refresh-aware strain, vsync hiccups, cheap passes

def feed_pattern(g, clk, seconds, pattern, vals=None):
    vals = vals if vals is not None else look()
    t_end = clk.t + seconds
    i = 0
    while clk.t < t_end:
        clk.t += pattern[i % len(pattern)] / 1000.0
        g.frame(dict(vals), dict(vals))
        i += 1


def test_b9_a_few_missed_vsyncs_with_an_idle_gpu_are_not_strain():
    g, clk, *_ = make("suggest", sensors=FakeSensors(gpu_util=40.0))
    g.pass_ms.update(scene=5.9, l_fractal=0.3, l_beings=0.2)
    pattern = [16.7] * 16 + [33.3]                        # ~6 % missed vblanks: p95 = 33 ms, mean ~17.6 ms
    feed_pattern(g, clk, 40.0, pattern)
    st = g.status()
    assert st["p95"] >= 30.0
    assert st["level"] == "ok" and st["suggestion"] is None and st["active"] == []


def test_b9_fifty_hz_projector_is_not_strained():
    g, clk, *_ = make("auto", sensors=FakeSensors(gpu_util=60.0))
    g.refresh_hz = 50.0
    g.pass_ms.update(scene=6.0, l_fractal=4.0)
    feed_pattern(g, clk, 30.0, [20.0])
    st = g.status()
    assert st["level"] == "ok" and st["active"] == [] and g.headroom is True
    assert st["refresh_hz"] == 50.0


def test_b9_thirty_hz_display_with_busy_gpu_judged_against_its_own_budget():
    g, clk, *_ = make("off", sensors=FakeSensors(gpu_util=97.0))
    g.refresh_hz = 30.0
    feed_pattern(g, clk, 5.0, [33.3])
    assert g.status()["level"] == "ok"
    feed_pattern(g, clk, 5.0, [45.0])                     # 22 fps of 30 Hz with a busy GPU
    assert g.status()["level"] == "strained"


def test_b9_sustained_fps_drop_with_a_busy_gpu_is_strain():
    g, clk, *_ = make("off", sensors=FakeSensors(gpu_util=97.0))
    feed_pattern(g, clk, 5.0, [22.0])
    st = g.status()
    assert st["level"] == "strained" and any("fps" in r for r in st["reasons"])


def test_b9_fps_drop_without_gpu_pressure_is_reported_not_degraded():
    g, clk, logs, *_ = make("auto", sensors=FakeSensors(gpu_util=30.0))
    g.pass_ms.update(scene=4.0, l_fractal=2.0)
    feed_pattern(g, clk, 30.0, [25.0])
    st = g.status()
    assert st["level"] == "ok" and st["active"] == []
    assert "not the GPU" in st["note"]


def test_b9_gpu_pass_time_is_strain_evidence_without_utilisation():
    g, clk, *_ = make("off", sensors=FakeSensors(gpu_util=None))
    g.pass_ms.update(scene=9.0, l_fractal=6.0)            # 15 ms of a 16.7 ms budget
    feed_pattern(g, clk, 5.0, [22.0])
    assert g.status()["level"] == "strained"


def test_b9_passes_under_one_ms_are_never_degraded():
    g, clk, *_ = make("auto", sensors=FakeSensors(gpu_util=97.0))
    g.pass_ms.update(scene=15.0, l_fractal=0.3, l_beings=0.2)
    out = None
    for _ in range(80):                                   # 40 s strained
        feed_pattern(g, clk, 0.5, [25.0])
    out = dict(look())
    g.frame(out, dict(look()))
    assert out["fractal_res"] == 0.5 and out["fractal_detail"] == 0.6 and out["being_count"] == 6.0
    assert g.status()["active"] == []
    assert "960x600" in g.status()["advice"]


def test_b9_refresh_rate_follows_the_display():
    hz = [50.0]
    g, clk, *_ = make("off")
    g.refresh_source = lambda: hz[0]
    feed_pattern(g, clk, 6.0, [20.0])
    assert g.refresh_hz == 50.0
    hz[0] = None                                          # unknown: keep the last good value
    feed_pattern(g, clk, 6.0, [20.0])
    assert g.refresh_hz == 50.0


class _VM:
    def __init__(self, w, h, hz):
        self.size = type("S", (), dict(width=w, height=h))()
        self.refresh_rate = hz


class FakeGlfw:
    def __init__(self, win_pos):
        self.mons = {"laptop": ((0, 0), _VM(1920, 1080, 60)), "projector": ((1920, 0), _VM(1280, 800, 50))}
        self.win_pos = win_pos

    def get_window_monitor(self, w):
        return None

    def get_window_pos(self, w):
        return self.win_pos

    def get_window_size(self, w):
        return (1280, 800)

    def get_monitors(self):
        return list(self.mons)

    def get_monitor_pos(self, m):
        return self.mons[m][0]

    def get_video_mode(self, m):
        return self.mons[m][1]

    def get_primary_monitor(self):
        return "laptop"


def test_b9_display_refresh_reads_the_monitor_under_the_window():
    assert gv.display_refresh(FakeGlfw((1920, 0)), "win") == 50.0
    assert gv.display_refresh(FakeGlfw((100, 100)), "win") == 60.0
    assert gv.display_refresh(FakeGlfw((-5000, -5000)), "win") == 60.0  # off every monitor: primary

    class Broken(FakeGlfw):
        def get_monitors(self):
            raise RuntimeError("no glfw")
    assert gv.display_refresh(Broken((0, 0)), "win") is None


def test_b9_engine_publishes_refresh_source_and_gov_status_hook():
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    assert "gov.refresh_source" in src and "controller.gov_status = gov.status" in src


# ------------------------------------------------------------------ POWER BALANCE (2026-11-26)

class Batt:
    """Fake battery: capacity integrates a net power (W, + = charging) over the fake clock."""

    def __init__(self, sens, clk, mwh=20000.0, full=71920.0, plugged=True):
        self.sens, self.clk, self.mwh, self.full, self.t = sens, clk, mwh, full, clk.t
        self.sens.d.update(plugged=plugged, batt_full_mwh=full)
        self.update(0.0)

    def update(self, net_w):
        dt = self.clk.t - self.t
        self.t = self.clk.t
        self.mwh = min(self.full, max(0.0, self.mwh + net_w * 1000.0 * dt / 3600.0))
        q = math.floor(self.mwh / 200.0) * 200.0                  # firmware reports coarse capacity steps
        self.sens.d.update(batt_mwh=q, batt_pct=100.0 * q / self.full, batt_rate_mw=None)


def sim_power(g, clk, batt, seconds, net_for, vals=None, ms=16.7, chunk=1.0):
    """Run frames for `seconds`; the battery's net power each chunk = net_for(governor)."""
    vals = vals if vals is not None else look()
    for _ in range(int(seconds / chunk)):
        batt.update(net_for(g))
        run(g, clk, chunk, ms, vals, dict(vals))


def load_model(base_w):
    """Net battery W for a load: each power step saves some watts (fps cap saves most)."""
    def f(g):
        cap = g.fps_cap or 60
        saving = {60: 0.0, 50: 3.0, 45: 4.5, 30: 9.0}.get(cap, 0.0)
        return base_w + saving + (1.0 if g.power_step >= 4 else 0.0) + (2.0 if g.power_step >= 5 else 0.0)
    return f


def test_power_net_watts_from_capacity_slope():
    est = gv.NetPower()
    for i in range(0, 300, 5):                                   # -6 W for 5 min, capacity in 200 mWh steps
        mwh = 20000.0 - 6.0 * 1000.0 * i / 3600.0
        est.add(float(i), math.floor(mwh / 200.0) * 200.0, None)
    assert est.watts() == pytest.approx(-6.0, abs=1.5)
    est2 = gv.NetPower()
    est2.add(0.0, 20000.0, -8000.0)                              # firmware rate wins when it reports one
    assert est2.watts() == pytest.approx(-8.0, abs=0.1)
    assert gv.NetPower().watts() is None


def test_power_battery_state_reader_never_raises():
    d = gv.read_battery()
    assert isinstance(d, dict)
    if d:
        assert "batt_pct" in d


def test_power_on_ac_draining_steps_down_until_net_is_positive_then_recovers():
    sens = FakeSensors(gpu_util=60.0)
    g, clk, logs, *_ = make("auto", sensors=sens)
    batt = Batt(sens, clk)
    base = [-6.0]                                                # heavy show drains 6 W on the stock charger
    model = load_model(0.0)
    sim_power(g, clk, batt, 1500, lambda gg: base[0] + model(gg))
    assert g.fps_cap is not None and g.fps_cap <= 45
    net = g.status()["power"]["net_w"]
    assert net is not None and net >= -0.5
    steps_at_peak = g.power_step
    base[0] = 4.0                                                # the show got lighter: plenty of surplus
    sim_power(g, clk, batt, 3000, lambda gg: base[0] + model(gg))
    assert g.power_step < steps_at_peak and g.fps_cap in (None, 50, 45)
    assert any("[governor]" in m and "power" in m for m in logs)


def test_power_charging_show_is_left_alone():
    sens = FakeSensors(gpu_util=60.0)
    g, clk, *_ = make("auto", sensors=sens)
    batt = Batt(sens, clk)
    sim_power(g, clk, batt, 900, lambda gg: 5.0)
    assert g.power_step == 0 and g.fps_cap is None


def test_power_suggest_mode_asks_first():
    sens = FakeSensors(gpu_util=60.0)
    g, clk, *_ = make("suggest", sensors=sens)
    batt = Batt(sens, clk)
    sim_power(g, clk, batt, 400, lambda gg: -6.0)
    sug = g.status()["suggestion"]
    assert sug and sug["id"] == "power:1" and "50 fps" in sug["text"] and g.fps_cap is None
    g.command({"cmd": "gov", "op": "apply", "id": "power:1"})
    assert g.fps_cap == 50


def test_power_battery_only_meets_a_four_hour_target():
    sens = FakeSensors(gpu_util=60.0)
    g, clk, *_ = make("auto", sensors=sens)
    batt = Batt(sens, clk, mwh=60000.0, plugged=False)          # 60 Wh left
    g.command({"cmd": "gov", "op": "show_length", "hours": 4})
    model = load_model(-22.0)                                    # 22 W draw: 2.7 h, needs <= 15 W for 4 h
    sim_power(g, clk, batt, 2400, model)
    p = g.status()["power"]
    assert p["allowed_w"] == pytest.approx(batt.mwh / 1000.0 / p["show_left_h"], rel=0.15)
    assert -p["net_w"] <= p["allowed_w"] * 1.1
    assert g.power_step >= 3


def test_power_performance_profile_never_steps_and_saver_starts_capped():
    sens = FakeSensors(gpu_util=60.0)
    g, clk, *_ = make("auto", sensors=sens, store={})
    batt = Batt(sens, clk)
    g.command({"cmd": "gov", "op": "power_profile", "profile": "performance"})
    sim_power(g, clk, batt, 900, lambda gg: -8.0)
    assert g.power_step == 0 and g.fps_cap is None
    g.command({"cmd": "gov", "op": "power_profile", "profile": "saver"})
    run(g, clk, 1.0, 16.7)
    assert g.fps_cap == 30
    assert g.store["governor"]["power_profile"] == "saver"


def test_power_settings_persist():
    store = {"governor": {"mode": "auto", "power_profile": "saver", "show_end": 2e9}}
    g, *_ = make(None, store=store)
    assert g.power_profile == "saver" and g.show_end == 2e9
    g.command({"cmd": "gov", "op": "show_length", "hours": 0})
    assert store["governor"]["show_end"] is None


def test_standby_while_blacked_out_or_master_zero():
    g, clk, *_ = make("suggest")
    v = look(blackout=1.0)
    run(g, clk, 1.0, 16.7, v, dict(v))
    assert g.standby is True and g.fps_cap == gv.STANDBY_FPS
    v = look(master=0.0)
    run(g, clk, 0.2, 16.7, v, dict(v))
    assert g.standby is True
    v = look()
    run(g, clk, 0.05, 16.7, v, dict(v))
    assert g.standby is False and g.fps_cap is None


def test_standby_frames_do_not_count_as_strain():
    g, clk, *_ = make("auto")
    v = look(blackout=1.0)
    run(g, clk, 30.0, 100.0, v, dict(v))                         # 10 fps on purpose
    assert g.status()["level"] == "ok" and g.status()["active"] == []


def test_ai_model_is_offloaded_after_idle_minutes_and_not_while_busy():
    offl = []
    g, clk, *_ = make("suggest", ai_offload=lambda: offl.append(clk.t))
    g.ai_source = lambda: dict(available=True, loaded=True, on_gpu=True, busy=False, queue=0, pending=0)
    run(g, clk, 300.0, 50.0)
    assert offl == []
    run(g, clk, 320.0, 50.0)                                     # > 10 min idle (balanced)
    assert len(offl) == 1
    run(g, clk, 120.0, 50.0)
    assert len(offl) == 1                                        # once per idle stretch
    g.ai_source = lambda: dict(available=True, loaded=True, on_gpu=True, busy=True, queue=1, pending=0)
    run(g, clk, 700.0, 50.0)
    assert len(offl) == 1


def test_frame_pacer_holds_the_cap():
    t = [0.0]
    slept = []
    p = gv.FramePacer(clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    for _ in range(30):
        t[0] += 0.004                                            # 4 ms of work per frame
        p.wait(30)
    assert t[0] == pytest.approx(1.0, abs=0.05)                  # 30 frames in ~1 s
    n = len(slept)
    p.wait(None)
    assert len(slept) == n                                       # no cap: never sleeps


def test_camera_strain_includes_the_power_step():
    g, clk, *_ = make("auto")
    assert g.camera_strain() == 0
    g.power_step = 6
    assert g.camera_strain() >= 1


def test_power_commands_validate():
    assert gv.validate_command({"cmd": "gov", "op": "power_profile", "profile": "balanced"})["profile"] == "balanced"
    assert gv.validate_command({"cmd": "gov", "op": "show_length", "hours": 4.5})["hours"] == 4.5
    for bad in ({"cmd": "gov", "op": "power_profile", "profile": "turbo"},
                {"cmd": "gov", "op": "show_length", "hours": "4"}, {"cmd": "gov", "op": "show_length", "hours": 99}):
        with pytest.raises(ValueError):
            gv.validate_command(bad)


def test_ai_link_offload_and_worker_status_reports_gpu_residency():
    import ai_link
    a = ai_link.AIClient([], start=False)
    a.offload()
    assert a._offload is True
    import ai_worker
    assert "on_gpu" in ai_worker.Worker.__init__.__code__.co_names or "on_gpu" in open(
        ROOT / "ai_worker.py", encoding="utf-8").read()


def test_engine_power_hooks_present():
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    for s in ("gov.pace(", "gov.standby", "dtK", "gov.camera_strain()", "--test-fps"):
        assert s in src, s


def test_worker_offload_endpoint_moves_the_idle_model_off_the_gpu_and_it_comes_back(tmp_path):
    import urllib.request
    from PIL import Image
    import ai_worker as aw

    class P:
        base, on_gpu, offloads = "fake", True, 0

        def offload(self):
            self.on_gpu, self.offloads = False, self.offloads + 1

        def txt2img(self, prompt, negative, seed, size):
            self.on_gpu = True                      # RealPipes._ready() restores before every image
            return Image.new("RGB", (size, size), (9, 9, 9))

    p = P()
    w = aw.Worker(out_dir=tmp_path / "t", snapshot=tmp_path / "s.png", token="tok", port=0, pipes_factory=lambda: p)
    w.start()

    def post(path, body):
        r = urllib.request.Request(f"http://127.0.0.1:{w.port}{path}", data=json.dumps(body).encode(), method="POST",
                                   headers={"X-Token": "tok", "Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status

    def until(fn, timeout=10):
        t = time.time()
        while not fn() and time.time() - t < timeout:
            time.sleep(0.05)
        return fn()
    try:
        assert post("/job", {"prompt": "a"}) == 200
        assert until(lambda: w.status()["done"] and not w.status()["busy"])
        assert w.status()["on_gpu"] is True
        assert post("/offload", {}) == 200
        assert until(lambda: w.status()["on_gpu"] is False) and p.offloads == 1
        assert post("/job", {"prompt": "b"}) == 200
        assert until(lambda: len(w.status()["done"]) >= 2 and w.status()["on_gpu"] is True)
    finally:
        w.stop()


def test_b9_slow_nvml_queries_run_only_every_fifth_poll():
    calls = {"throttle": 0, "power": 0, "mem": 0}

    class M:
        used, total = 1 << 30, 4 << 30

    class U:
        gpu = 50

    class N:
        def nvmlDeviceGetMemoryInfo(self, h):
            calls["mem"] += 1
            return M()

        def nvmlDeviceGetUtilizationRates(self, h):
            return U()

        def nvmlDeviceGetTemperature(self, h, s):
            return 70

        def nvmlDeviceGetCurrentClocksThrottleReasons(self, h):
            calls["throttle"] += 1
            return 0x20

        def nvmlDeviceGetPowerUsage(self, h):
            calls["power"] += 1
            return 21500
    rd = gv.make_nvml_reader(N(), "h")
    out = [rd() for _ in range(10)]
    assert calls["mem"] == 10 and calls["throttle"] == 2 and calls["power"] == 2
    assert all(d["thermal"] is True and d["gpu_w"] == 21.5 for d in out)       # cached between slow reads


def test_power_thirty_fps_echo_looks_like_sixty_fps_headless(tmp_path):
    """GPU: run only under .locks/gpu_suite (the full suite). Measured 2026-11-26: diff 5.5 (dt-scaled) vs 15.0 (per-frame)."""
    import os
    import subprocess
    import numpy as np
    from PIL import Image
    look_args = ["--set", "trails=0.94", "--set", "l_echo=0.9", "--set", "rotate=0.3", "--set", "zoom=0.2",
                 "--test-audio", "silent"]

    def render(name, frames, fps, env=None):
        out = tmp_path / name
        r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--test", str(frames), "--test-fps", str(fps),
                            "--out", str(out), *look_args], cwd=ROOT, capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, **(env or {})))
        assert r.returncode == 0, r.stderr[-800:]
        return np.asarray(Image.open(out / f"frame_{frames:04d}.png").convert("RGB"), dtype=np.float32)
    a = render("a", 600, 60)
    b = render("b", 300, 30)
    c = render("c", 300, 30, {"LV_ECHO_PER_FRAME": "1"})
    d_dt, d_pf = np.abs(a - b).mean(), np.abs(a - c).mean()
    assert d_dt < 0.6 * d_pf and d_dt < 9.0, (d_dt, d_pf)
