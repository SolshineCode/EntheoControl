"""ONLINE plugin (plugins/online): assistant allowlist + safety, SUGGEST / AUTOPILOT, undo, OFF = nothing runs,
timeouts, malformed LLM output, Nano Banana texture contract + fallback to the local engine. Fake backends only:
no test here talks to a real CLI or the network."""
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import control  # noqa: E402
from control import Controller, validate  # noqa: E402
from params import PARAMS, TRIGGERS  # noqa: E402
from plugins.online import protocol  # noqa: E402


class FakeTime:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeBackend:
    name = "claude"
    label = "Fake Claude"

    def __init__(self, replies=None, ok=True, error=None):
        self.replies, self.ok, self.error, self.prompts = list(replies or []), ok, error, []

    def available(self):
        return self.ok

    def ask(self, prompt, cancel=None, progress=None):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        r = self.replies.pop(0) if self.replies else '{"say": "nothing to do", "commands": []}'
        return r if isinstance(r, str) else json.dumps(r)


class FakeAI:
    """Stands in for ai_link.AIClient (the local SD-Turbo worker link)."""
    def __init__(self):
        self.jobs, self.cancelled = [], 0

    def submit(self, job, delay=0.0):
        self.jobs.append(dict(job))

    def cancel(self):
        self.cancelled += 1

    def status(self):
        return {"available": False}

    def pop_shown(self):
        return []

    def close(self):
        pass


class FakeProvider:
    name = "fake"

    def __init__(self, ok=True, error=None, size=(1024, 640)):
        self.ok, self.error, self.size, self.calls = ok, error, size, []

    def available(self):
        return self.ok

    def generate(self, prompt, init, work, cancel=None):
        from PIL import Image
        self.calls.append((prompt, None if init is None else Path(init).name))
        if self.error:
            raise self.error
        out = Path(work) / "nanobanana-output" / "img.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", self.size, (200, 30, 90)).save(out)
        return out


def make(tmp_path, replies=None, on=True, mode="suggest", backend=None, providers=None, engine="local", **kw):
    ft = FakeTime()
    c = Controller(tmp_path, restore=False, autosave=False, midi=False, ai=False, time_fn=ft, seed=1)
    c.ft = ft
    c.quantize, c.morph_bars = "off", 0.0
    c.ai = FakeAI()
    c.fake = backend or FakeBackend(replies)
    (tmp_path / "textures").mkdir(exist_ok=True)
    c.online_opts = dict(root=tmp_path, backends={"claude": c.fake, "agy": FakeBackend(ok=False)},
                         image_providers=providers if providers is not None else [FakeProvider()],
                         log_dir=tmp_path / "logs", clock=ft, sync=True, **kw)
    if on:
        do(c, op="enable", on=True)
        if mode != "suggest":
            do(c, op="mode", mode=mode)
        if engine != "local":
            do(c, op="engine", engine=engine)
    return c


def do(c, **kw):
    return c.apply([dict(kw, cmd="online")])


def ask(c, text="make it calmer"):
    do(c, op="ask", text=text)
    c.ft.t += 0.02
    c.update(0.02, (0.5, 0.4, 0.2))
    return c.state()["online"]


def settle(c, seconds=1.0, dt=1 / 60):
    for _ in range(int(seconds / dt)):
        c.ft.t += dt
        c.update(dt, (0.5, 0.4, 0.2))


def reply(*cmds, say="ok"):
    return {"say": say, "commands": list(cmds)}


def last_prop(st):
    return st["props"][-1]


# ------------------------------------------------------------------------------------------------ OFF

def test_off_by_default_and_nothing_is_loaded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("subprocess")))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("subprocess")))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setattr(threading.Thread, "start", lambda self: calls.append(("thread", self.name)))
    c = make(tmp_path, on=False)
    assert c.online is None and protocol.settings(c)["on"] is False
    do(c, op="ask", text="strobe everything")
    do(c, op="apply", id=1)
    do(c, op="undo", id=1)
    do(c, op="cancel")
    do(c, op="mode", mode="autopilot")
    do(c, op="engine", engine="nano")                     # the choice is remembered, but OFF means local
    c.apply([{"cmd": "gen", "prompt": "a buddha", "n": 1}])
    c.apply([{"cmd": "gen", "prompt": "a buddha", "init": "snapshot"}])
    c.apply([{"cmd": "panic"}])
    for _ in range(5):
        c.update(1 / 60, (0.5, 0.5, 0.5))
    assert c.online is None
    assert calls == []
    assert len(c.ai.jobs) == 2                            # both image jobs went to the LOCAL engine
    st = c.state()["online"]
    assert st["on"] is False and "msgs" not in st
    assert not (tmp_path / "logs").exists()


def test_import_of_control_does_not_import_subprocess_capable_plugin_code():
    code = ("import sys, control; c = control.Controller('.', restore=False, midi=False, ai=False); c.update(0.016);"
            "bad = [m for m in sys.modules if m.startswith('plugins.online.') and not m.endswith('.protocol')];"
            "print('BAD' if bad else 'CLEAN', bad)")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert "CLEAN" in out.stdout, out.stdout + out.stderr


def test_switch_persists_and_restores(tmp_path):
    c = make(tmp_path, on=True, mode="autopilot")
    assert c.online is not None
    s = c._settings()["ext"]["online"]
    assert s == dict(on=True, mode="autopilot", backend="claude", engine="local", cli_google=False)
    do(c, op="enable", on=False)
    assert c.online is None and c._settings()["ext"]["online"]["on"] is False
    # hand-edited junk in live_state.json never switches it on
    c.ext["online"] = {"on": "yes", "mode": "rm -rf", "engine": 7, "cli_google": 1, "backend": "agy"}
    assert protocol.settings(c) == protocol.DEFAULTS


def test_validate_online_commands():
    assert validate({"cmd": "online", "op": "enable", "on": True}) == {"cmd": "online", "op": "enable", "on": True}
    for bad in ({"op": "enable", "on": 1}, {"op": "mode", "mode": "god"}, {"op": "backend", "backend": "bash"},
                {"op": "engine", "engine": "dalle"}, {"op": "ask", "text": ""}, {"op": "ask", "text": "x" * 1001},
                {"op": "apply", "id": "1"}, {"op": "shell", "cmd2": "dir"}, {"op": "undo", "id": -1}):
        with pytest.raises(ValueError):
            validate(dict(bad, cmd="online"))
    assert validate({"cmd": "online", "op": "ask", "text": " hi\x00\x07 ", "junk": 1}) == {"cmd": "online", "op": "ask", "text": "hi"}


# ------------------------------------------------------------------------------------------------ allowlist

def test_suggest_mode_shows_a_diff_and_changes_nothing_until_apply(tmp_path):
    c = make(tmp_path, [reply({"cmd": "set", "name": "zoom", "v": 0.6}, {"cmd": "palette", "i": 3}, say="Deeper, greener.")])
    before = dict(c.base)
    st = ask(c)
    assert c.base == before                               # SUGGEST: nothing applied
    p = last_prop(st)
    assert p["status"] == "pending"
    assert any("zoom" in d and "0.60" in d for d in p["diff"])
    assert any("Emerald Shipibo" in d for d in p["diff"])  # option names, not numbers
    m = st["msgs"][-1]
    assert m["role"] == "assistant" and m["text"] == "Deeper, greener." and m["backend"] == "claude" and "secs" in m
    do(c, op="apply", id=p["id"])
    assert c.base["zoom"] == 0.6 and c.base["palette"] == 3.0
    st = c.state()["online"]
    assert last_prop(st)["status"] == "applied" and st["history"][-1]["id"] == p["id"]
    do(c, op="undo", id=p["id"])
    assert c.base["zoom"] == before["zoom"] and c.base["palette"] == before["palette"]
    assert c.state()["online"]["history"][-1]["undone"] is True
    do(c, op="undo", id=p["id"])                          # undoing twice is a no-op
    assert c.base["zoom"] == before["zoom"]


def test_dismiss_and_apply_only_once(tmp_path):
    c = make(tmp_path, [reply({"cmd": "set", "name": "zoom", "v": 0.5}), reply({"cmd": "set", "name": "zoom", "v": -0.5})])
    p1 = last_prop(ask(c))
    do(c, op="dismiss", id=p1["id"])
    do(c, op="apply", id=p1["id"])
    assert c.base["zoom"] == 0.0
    c.ft.t += 5
    p2 = last_prop(ask(c))
    do(c, op="apply", id=p2["id"])
    c.set("zoom", 0.9)
    c.ft.t += 5
    do(c, op="apply", id=p2["id"])                        # already applied: must not fire again
    assert c.base["zoom"] == 0.9


@pytest.mark.parametrize("cmd", [
    {"cmd": "trigger", "name": "strobe", "on": True}, {"cmd": "latch", "name": "flash", "on": True},
    {"cmd": "panic"}, {"cmd": "strobe_rate", "v": 0.0625}, {"cmd": "bpm", "v": 300}, {"cmd": "surf", "op": "del"},
    {"cmd": "corner", "i": 0, "u": 0.5, "v": 0.5}, {"cmd": "save"}, {"cmd": "reset"}, {"cmd": "calib", "on": True},
    {"cmd": "mod", "name": "master", "slot": 0, "src": "lfo_square", "amt": 1, "rate": 64},
    {"cmd": "midi_defaults"}, {"cmd": "scene_clear", "i": 0}, {"cmd": "init"}, {"cmd": "safe", "i": 3},
    {"cmd": "online", "op": "mode", "mode": "autopilot"}, {"cmd": "gov", "op": "mode", "mode": "off"},
    {"cmd": "rec", "op": "start"}, {"cmd": "shell", "run": "del /s /q C:\\"}, {"cmd": "text_phrases", "phrases": ["x"]},
    {"cmd": "set", "name": "strobe", "v": 1}, {"cmd": "set", "name": "blackout", "v": 1},
    {"cmd": "set", "name": "flash", "v": 1}, {"cmd": "nudge", "name": "invert", "d": 1},
    {"cmd": "set", "name": "no_such_param", "v": 1}, {"cmd": "set", "name": "zoom", "v": "0.5"},
    {"cmd": "set", "name": "zoom", "v": float("nan")}, {"cmd": "set", "name": "zoom"},
    {"cmd": "bypass", "key": "layers", "enabled": False}, {"cmd": "bypass", "key": "master", "enabled": False},
    {"cmd": "scene_recall", "slot": 16}, {"cmd": "scene_recall", "slot": 0}, {"cmd": "scene_save", "slot": 99},
    {"cmd": "art_pick", "source": "ai", "name": "..\\..\\secrets.png"}, {"cmd": "art_pick", "source": "ai", "name": "nope.png"},
    {"cmd": "gen", "prompt": ""}, {"cmd": "gen", "prompt": "x" * 2000}, "set zoom 1", 7, None, ["set"],
])
def test_everything_outside_the_allowlist_is_rejected_even_in_autopilot(tmp_path, cmd):
    c = make(tmp_path, [reply(cmd)], mode="autopilot")
    c.scenes[15] = None
    base, held, scenes = dict(c.base), dict(c.held), json.dumps(c.scenes)
    bpm, rate, mods, byp = c.bpm, c.strobe_rate, json.dumps(c.mods), set(c.bypass)
    st = ask(c)
    assert (c.base, c.held, json.dumps(c.scenes), c.bpm, c.strobe_rate, json.dumps(c.mods), c.bypass) == \
           (base, held, scenes, bpm, rate, mods, byp)
    assert c.ai.jobs == [] and c.apply([]) == []          # no image job, nothing sent on to the engine
    p = last_prop(st)
    assert p["status"] == "rejected" and len(p["rejected"]) == 1 and p["rejected"][0]["why"]
    assert st["history"] == []
    assert st["mode"] == "autopilot"


def test_clamping_master_floor_ranges_and_chase_rate(tmp_path):
    c = make(tmp_path, [reply({"cmd": "set", "name": "master", "v": 0.0}, {"cmd": "set", "name": "zoom", "v": 99},
                              {"cmd": "set", "name": "folds", "v": -5}, {"cmd": "set", "name": "chase_rate", "v": 4},
                              {"cmd": "nudge", "name": "hue", "d": 5}, {"cmd": "palette", "i": 40},
                              {"cmd": "morph", "bars": 1e9})], mode="autopilot")
    c.set_bpm(300)                                        # 4 sweeps a bar at 300 BPM would be 5 Hz
    ask(c)
    from plugins.online import safety
    assert c.base["master"] == safety.MASTER_MIN          # never a blackout by fader
    assert c.base["zoom"] == PARAMS["zoom"][1] and c.base["folds"] == PARAMS["folds"][0]
    assert c.base["hue"] == PARAMS["hue"][1] and c.base["palette"] == PARAMS["palette"][1]
    assert c.morph_bars == 64.0
    hz = abs(c.base["chase_rate"]) * c.bpm / 240.0
    assert hz <= safety.MAX_FLASH_HZ + 1e-9 and c.base["chase_rate"] == pytest.approx(2.0)
    assert safety.MAX_FLASH_HZ < 3.0
    assert any("clamped" in n for n in last_prop(c.state()["online"])["notes"])


def test_photosensitive_no_value_can_alternate_faster_than_3hz(tmp_path):
    """One value per param per batch (last wins) and at most one applied batch a second."""
    from plugins.online import assistant, safety
    flicker = [{"cmd": "set", "name": "master", "v": 1.0 if i % 2 else 0.2} for i in range(20)]
    c = make(tmp_path, [reply(*flicker) for _ in range(6)], mode="autopilot")
    changes = []
    for _ in range(6):                                    # six replies inside 0.6 s
        ask(c)
        changes.append(c.base["master"])
        c.ft.t += 0.08
    distinct = sum(1 for a, b in zip([1.0] + changes, changes) if a != b)
    assert distinct <= 1                                  # one batch got through; the rest wait for the limiter
    assert 1.0 / assistant.MIN_APPLY_S <= safety.MAX_FLASH_HZ
    st = c.state()["online"]
    assert sum(1 for p in st["props"] if p["status"] == "applied") == 1
    waiting = [p for p in st["props"] if p["status"] == "pending"]
    assert waiting and all("rate limit" in p["why"] for p in waiting)
    for n in TRIGGERS:
        assert not c.held[n] and c.base[n] == PARAMS[n][2]


def test_batch_limits(tmp_path):
    from plugins.online import safety
    many = [{"cmd": "set", "name": n, "v": PARAMS[n][2]} for n in list(control.CONT)[:safety.MAX_CMDS + 6]]
    gens = [{"cmd": "gen", "prompt": f"lotus {i}"} for i in range(4)]
    c = make(tmp_path, [reply(*many), reply(*gens, {"cmd": "scene_save", "slot": 9}, {"cmd": "scene_save", "slot": 10})],
             mode="autopilot")
    p = last_prop(ask(c))
    assert len(p["diff"]) == safety.MAX_CMDS and len(p["rejected"]) == 6
    c.ft.t += 5
    p = last_prop(ask(c))
    assert len(c.ai.jobs) == safety.MAX_GEN               # two image jobs, the other two rejected
    assert c.scenes[8] is not None and c.scenes[9] is None
    assert len(p["rejected"]) == 3


def test_request_and_image_rate_limits(tmp_path):
    from plugins.online import assistant
    c = make(tmp_path, [reply({"cmd": "gen", "prompt": "a"}, {"cmd": "gen", "prompt": "b"}) for _ in range(12)], mode="autopilot")
    for _ in range(10):
        ask(c)
        c.ft.t += 1.5
    assert len(c.fake.prompts) == assistant.ASK_LIMIT[0]  # the CLI was only started 6 times in that minute
    assert len(c.ai.jobs) == assistant.GEN_LIMIT[0]
    assert any("Slow down" in m["text"] for m in c.state()["online"]["msgs"])
    c.ft.t += 61
    ask(c)
    assert len(c.fake.prompts) == assistant.ASK_LIMIT[0] + 1


# ------------------------------------------------------------------------------------------------ autopilot + undo

def test_autopilot_applies_and_every_batch_is_undoable(tmp_path):
    c = make(tmp_path, [reply({"cmd": "set", "name": "zoom", "v": 0.4}, {"cmd": "bypass", "key": "l_beads", "enabled": False},
                              {"cmd": "morph", "bars": 8}),
                        reply({"cmd": "set", "name": "zoom", "v": 0.8}, {"cmd": "set", "name": "hue", "v": 0.5})],
             mode="autopilot")
    st = ask(c)
    assert c.base["zoom"] == 0.4 and "l_beads" in c.bypass and c.morph_bars == 8.0
    first = last_prop(st)["id"]
    c.ft.t += 2
    st = ask(c)
    second = last_prop(st)["id"]
    assert c.base["zoom"] == 0.8 and c.base["hue"] == 0.5
    assert [h["id"] for h in st["history"]] == [first, second] and all(h["by"] == "autopilot" for h in st["history"])
    do(c, op="undo", id=second)
    assert c.base["zoom"] == 0.4 and c.base["hue"] == PARAMS["hue"][2]
    do(c, op="undo", id=first)
    assert c.base["zoom"] == 0.0 and "l_beads" not in c.bypass and c.morph_bars == 0.0
    assert c.bypass_fade == control.BYPASS_FADE           # the lengthened fade was part of the undo record


def test_no_hard_cuts_layer_fades_and_scene_morphs_are_lengthened(tmp_path):
    from plugins.online import safety
    c = make(tmp_path, [reply({"cmd": "bypass", "key": "l_kene", "enabled": False}, {"cmd": "scene_recall", "slot": 2},
                              {"cmd": "morph", "bars": 0})], mode="autopilot")
    c.scenes[1] = dict(name="Peak", base=dict({n: c.base[n] for n in control.CONT}, zoom=0.7), mods={})
    c.bypass_fade, c.morph_bars = 0.0, 0.0                # the performer likes hard cuts; the assistant may not make them
    st = ask(c)
    assert c.bypass_fade == safety.MIN_FADE_S
    assert c.morph_bars * 4 * 60.0 / c.bpm == pytest.approx(safety.MIN_MORPH_S)
    assert 0.0 < c.base["zoom"] < 0.7 or c._morph is not None   # morphing, not a cut
    notes = last_prop(st)["notes"]
    assert any("layer fade" in n for n in notes) and any("scene morph" in n for n in notes)
    do(c, op="undo", id=last_prop(st)["id"])
    assert c.bypass_fade == 0.0 and c.morph_bars == 0.0
    c.close()
    f = next((tmp_path / "logs").glob("online_*.log"))
    assert any(json.loads(x)["kind"] == "clamp" for x in f.read_text(encoding="utf-8").splitlines())


def test_scene_recall_save_and_undo(tmp_path):
    c = make(tmp_path, [reply({"cmd": "scene_recall", "slot": 2}), reply({"cmd": "scene_save", "slot": 2}),
                        reply({"cmd": "scene_recall", "name": "peak"})], mode="autopilot")
    c.scenes[1] = dict(name="Peak", base=dict({n: c.base[n] for n in control.CONT}, zoom=0.7, hue=0.3), mods={})
    old_scene = json.dumps(c.scenes[1])
    c.set("zoom", -0.2)
    ask(c)
    settle(c)
    assert c.base["zoom"] == pytest.approx(0.7) and c.cur == 1
    pid = last_prop(c.state()["online"])["id"]
    do(c, op="undo", id=pid)
    assert c.base["zoom"] == -0.2 and c.cur == -1
    c.ft.t += 2
    ask(c)                                                # save over slot 2 ...
    assert c.scenes[1]["base"]["zoom"] == -0.2
    assert "overwrites" in last_prop(c.state()["online"])["diff"][0]
    do(c, op="undo", id=last_prop(c.state()["online"])["id"])
    assert json.dumps(c.scenes[1]) == old_scene           # ... and undo puts the old scene back
    c.ft.t += 2
    ask(c)
    settle(c)
    assert c.base["zoom"] == pytest.approx(0.7)           # recall by name


def test_art_pick_text_and_gen_go_through_the_normal_command_paths(tmp_path):
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "gen_1_2.png").write_bytes(b"x")
    c = make(tmp_path, [reply({"cmd": "art_pick", "source": "ai", "name": "gen_1_2.png", "use": 2},
                              {"cmd": "text", "text": "LOVE"}, {"cmd": "gen", "prompt": "a golden buddha", "from_frame": True})],
             mode="autopilot")
    do(c, op="ask", text="buddha time")
    c.ft.t += 0.02
    c.update(0.02)
    out = c.apply([])
    kinds = [o["cmd"] for o in out]
    assert "art_pick" in kinds and "snapshot" in kinds
    pick = [o for o in out if o["cmd"] == "art_pick"][0]
    assert pick == control.passthrough("art_pick", source="ai", name="gen_1_2.png", use=2)
    assert c.text.state()["text"] == "LOVE"
    assert c.ai.jobs == [dict(prompt="a golden buddha", n=1, init="snapshot", show=True, strength=0.45)]
    assert "gen_1_2.png" in c.fake.prompts[0] and "zoom" in c.fake.prompts[0]


def test_autopilot_waits_while_blackout_is_held_and_never_releases_it(tmp_path):
    c = make(tmp_path, [reply({"cmd": "set", "name": "zoom", "v": 0.5})], mode="autopilot")
    c.trigger("blackout", True)
    st = ask(c)
    assert c.base["zoom"] == 0.0 and c.held["blackout"] is True
    assert "blackout" in last_prop(st)["why"]
    c.trigger("blackout", False)
    c.ft.t += 0.1
    c.update(0.1)
    assert c.base["zoom"] == 0.5


def test_panic_drops_pending_work_and_leaves_autopilot(tmp_path):
    c = make(tmp_path, [reply({"cmd": "set", "name": "zoom", "v": 0.5})], mode="suggest")
    p = last_prop(ask(c))
    do(c, op="mode", mode="autopilot")
    c.apply([{"cmd": "panic"}])
    st = c.state()["online"]
    assert st["mode"] == "suggest" and protocol.settings(c)["mode"] == "suggest"
    assert last_prop(st)["status"] == "dismissed"
    c.ft.t += 5
    do(c, op="apply", id=p["id"])
    assert c.base["zoom"] == PARAMS["zoom"][2]


# ------------------------------------------------------------------------------------------------ bad output / errors

@pytest.mark.parametrize("raw", [
    "", "   ", "Sure! I will make it calmer.", "{not json", '{"commands": "set zoom 1"}', '[{"cmd":"set"}]',
    '{"say": 5, "commands": [null, 3, "x", {"cmd": 7}, {"name": "zoom"}]}', "```json\n{\"commands\": [}\n```",
    '{"say": "' + "A" * 5000 + '"}', '{"commands": [' + ",".join(['{"cmd":"set","name":"zoom","v":1e999}'] * 3) + "]}",
    "\x00\x1b[31m{}", '{"commands": {"cmd": "set", "name": "zoom", "v": 1}}',
])
def test_malformed_llm_output_never_changes_the_show_or_raises(tmp_path, raw):
    c = make(tmp_path, [raw], mode="autopilot")
    base = dict(c.base)
    st = ask(c)
    assert c.base == base and st["history"] == []
    assert st["msgs"][-1]["role"] in ("assistant", "system")
    assert all(len(m["text"]) <= 600 for m in st["msgs"])
    json.dumps(st)


def test_json_inside_prose_and_code_fences_is_found(tmp_path):
    raw = 'Here you go:\n```json\n{"say": "ok", "commands": [{"cmd": "set", "name": "zoom", "v": 0.25}]}\n```\nEnjoy {braces}.'
    c = make(tmp_path, [raw], mode="autopilot")
    ask(c)
    assert c.base["zoom"] == 0.25


def test_backend_error_missing_backend_and_log(tmp_path):
    from plugins.online.backends import BackendTimeout
    c = make(tmp_path, backend=FakeBackend(error=BackendTimeout("no answer within 1 s")))
    st = ask(c)
    assert st["msgs"][-1]["role"] == "system" and "no answer" in st["msgs"][-1]["text"]
    do(c, op="ask", text="hi", backend="agy")            # opt-in not given
    assert "opt-in" in c.state()["online"]["msgs"][-1]["text"]
    do(c, op="cli_google", on=True)
    do(c, op="ask", text="hi", backend="agy")            # not installed (fake says unavailable)
    assert "not installed" in c.state()["online"]["msgs"][-1]["text"]
    assert c.state()["online"]["backends"]["agy"]["ok"] is False
    c.close()
    lines = [json.loads(x) for x in (tmp_path / "logs").glob("online_*.log").__next__().read_text(encoding="utf-8").splitlines()]
    kinds = [x["kind"] for x in lines]
    assert "request" in kinds and "error" in kinds and kinds[0] == "online"


def test_log_records_request_response_and_actions(tmp_path):
    c = make(tmp_path, [reply({"cmd": "set", "name": "zoom", "v": 0.5}, {"cmd": "panic"})], mode="autopilot")
    pid = last_prop(ask(c))["id"]
    do(c, op="undo", id=pid)
    c.close()
    f = next((tmp_path / "logs").glob("online_*.log"))
    lines = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
    kinds = [x["kind"] for x in lines]
    for k in ("request", "response", "apply", "undo"):
        assert k in kinds
    resp = [x for x in lines if x["kind"] == "response"][0]
    assert resp["backend"] == "claude" and resp["rejected"][0]["cmd"] == "panic" and "secs" in resp


def test_cli_timeout_kills_the_child(tmp_path):
    from plugins.online import backends
    t0 = time.monotonic()
    with pytest.raises(backends.BackendTimeout):
        backends.run_cli([sys.executable, "-c", "import time; time.sleep(60)"], "", timeout=0.6)
    assert time.monotonic() - t0 < 15
    rc, out, err = backends.run_cli([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"], "hello", timeout=30)
    assert rc == 0 and out.strip() == "HELLO"
    ev = threading.Event()
    ev.set()
    with pytest.raises(backends.BackendCancelled):
        backends.run_cli([sys.executable, "-c", "import time; time.sleep(60)"], "", timeout=30, cancel=ev)
    with pytest.raises(backends.BackendError):
        backends.run_cli([str(tmp_path / "no_such_tool.exe")], "", timeout=5)


def test_claude_backend_contract(tmp_path, monkeypatch):
    """Prompt on stdin (never argv), cwd = home, tools off, resolved through shutil.which."""
    from plugins.online import backends
    seen = {}

    def fake_run(argv, stdin_text="", cwd=None, timeout=0, **kw):
        seen.update(argv=argv, stdin=stdin_text, cwd=cwd, timeout=timeout)
        return 0, '{"say": "hi", "commands": []}', ""

    monkeypatch.setattr(backends, "run_cli", fake_run)
    monkeypatch.setattr(backends.shutil, "which", lambda n: r"C:\fake\npm\claude.CMD" if n == "claude" else None)
    b = backends.ClaudeBackend(timeout=42)
    assert b.available() and b.ask("SECRET PROMPT") .startswith("{")
    assert seen["argv"][0].endswith("claude.CMD") and "-p" in seen["argv"] and "--restricted" in seen["argv"]
    assert "SECRET PROMPT" not in " ".join(seen["argv"]) and seen["stdin"] == "SECRET PROMPT"
    assert Path(seen["cwd"]) == Path.home() and seen["timeout"] == 42
    assert not backends.AgyBackend(tmp_path).available()
    monkeypatch.setattr(backends, "run_cli", lambda *a, **k: (1, "", "not logged in"))
    with pytest.raises(backends.BackendError):
        b.ask("x")


def test_threaded_worker_never_blocks_update(tmp_path):
    class Slow(FakeBackend):
        def ask(self, prompt, cancel=None, progress=None):
            time.sleep(0.4)
            return json.dumps(reply({"cmd": "set", "name": "zoom", "v": 0.3}))

    c = make(tmp_path, backend=Slow(), on=False)
    c.online_opts["sync"] = False
    do(c, op="enable", on=True)
    do(c, op="mode", mode="autopilot")
    t0 = time.monotonic()
    do(c, op="ask", text="go")
    c.update(0.016)
    assert time.monotonic() - t0 < 0.25                   # the command + a frame returned while the CLI still runs
    assert c.state()["online"]["busy"]["backend"] == "claude"
    end = time.monotonic() + 5
    while c.base["zoom"] != 0.3 and time.monotonic() < end:
        time.sleep(0.02)
        c.update(0.016)
    assert c.base["zoom"] == 0.3 and c.state()["online"]["busy"] is None
    c.close()


def test_prompt_is_built_from_the_registry_and_state(tmp_path):
    from plugins.online import assistant
    c = make(tmp_path, on=False)
    c.set("zoom", 0.33)
    c.scenes[4] = dict(name="Jewel pulse", base={n: c.base[n] for n in control.CONT}, mods={})
    p = assistant.build_prompt(c, "more jaguar", tex=["gen_a.png"], art=["mask.jpg"])
    for name in PARAMS:
        assert (f"\n{name} [" in p) == (name not in TRIGGERS), name     # triggers are not even offered
    for needle in ("0.33", '5 "Jewel pulse"', "Jaguar spirit", "gen_a.png", "mask.jpg", "more jaguar", '"commands"', "l_beads"):
        assert needle in p, needle
    assert len(p) < 24000


# ------------------------------------------------------------------------------------------------ Nano Banana

def test_nano_engine_writes_the_texture_contract_and_shows_it(tmp_path):
    from PIL import Image
    prov = FakeProvider()
    c = make(tmp_path, providers=[prov], engine="nano")
    c.apply([{"cmd": "gen", "prompt": "a serene buddha", "n": 1, "seed": 7}])
    c.update(0.016)
    out = c.apply([])
    assert c.ai.jobs == []                                # not the local engine
    files = sorted((tmp_path / "textures").glob("gen_*.png"))
    assert len(files) == 1 and files[0].name.endswith("_7.png")
    assert not list((tmp_path / "textures").glob("*.tmp"))
    with Image.open(files[0]) as im:
        assert im.text["prompt"] == "a serene buddha" and im.text["seed"] == "7" and max(im.size) <= 768
    assert control.passthrough("show_texture", name=files[0].name) in out
    st = c.state()["online"]
    assert st["engine"] == "nano" and st["nano"]["done"][-1]["name"] == files[0].name
    assert "Google" in st["privacy"]
    assert prov.calls == [("a serene buddha", None)]


def test_nano_dream_from_frame_passes_the_snapshot(tmp_path):
    from PIL import Image
    (tmp_path / "snapshots").mkdir()
    Image.new("RGB", (64, 40), (1, 2, 3)).save(tmp_path / "snapshots" / "current.png")
    prov = FakeProvider()
    c = make(tmp_path, providers=[prov], engine="nano")
    out = c.apply([{"cmd": "gen", "prompt": "melt into jaguars", "init": "snapshot", "strength": 0.5}])
    assert control.passthrough("snapshot") in out         # the engine is asked for a fresh frame first
    assert prov.calls == [("melt into jaguars", "current.png")]


@pytest.mark.parametrize("why", ["error", "missing", "no_snapshot", "rate"])
def test_nano_falls_back_to_the_local_engine(tmp_path, why):
    from plugins.online import images
    prov = FakeProvider(ok=why != "missing", error=RuntimeError("quota exceeded") if why == "error" else None)
    c = make(tmp_path, providers=[prov], engine="nano")
    job = {"cmd": "gen", "prompt": "lotus", "n": 1, "seed": 5}
    if why == "no_snapshot":
        job["init"] = "snapshot"
    n = images.IMG_LIMIT[0] + 1 if why == "rate" else 1
    for _ in range(n):
        c.apply([job])
    c.update(0.016)
    assert len(c.ai.jobs) == 1 and c.ai.jobs[0]["prompt"] == "lotus" and c.ai.jobs[0]["seed"] == 5
    assert "locally" in c.state()["online"]["nano"]["note"]
    if why in ("error", "missing", "no_snapshot"):
        assert not list((tmp_path / "textures").glob("gen_*.png"))


def test_local_engine_choice_and_online_off_use_the_local_worker(tmp_path):
    prov = FakeProvider()
    c = make(tmp_path, providers=[prov], engine="local")
    c.apply([{"cmd": "gen", "prompt": "lotus"}])
    assert len(c.ai.jobs) == 1 and prov.calls == []
    do(c, op="engine", engine="nano")
    do(c, op="enable", on=False)                          # offline: the nano choice is ignored
    c.apply([{"cmd": "gen", "prompt": "lotus"}])
    assert len(c.ai.jobs) == 2 and prov.calls == []


def test_gemini_provider_contract(tmp_path, monkeypatch):
    """Only the nanobanana extension is loaded, the prompt rides on stdin, no key is passed in the environment."""
    from PIL import Image
    from plugins.online import images
    seen = {}

    def fake_run(argv, stdin_text="", cwd=None, timeout=0, **kw):
        seen.update(argv=argv, stdin=stdin_text, cwd=cwd, kw=kw)
        d = Path(cwd) / "nanobanana-output"
        d.mkdir()
        Image.new("RGB", (32, 32)).save(d / "x.png")
        return 0, "saved", ""

    monkeypatch.setattr(images, "run_cli", fake_run)
    p = images.GeminiNanoProvider(exe=r"C:\fake\gemini.CMD", ext_dir=tmp_path)
    (tmp_path / "frame_src.png").write_bytes(b"")
    Image.new("RGB", (8, 8)).save(tmp_path / "frame_src.png")
    work = tmp_path / "w"
    work.mkdir()
    img = p.generate('a "buddha" & echo pwned\n| del', tmp_path / "frame_src.png", work)
    assert img.name == "x.png" and (work / "frame.png").is_file()
    a = seen["argv"]
    assert a[a.index("-e") + 1] == "nanobanana" and "pwned" not in " ".join(a)
    assert "edit_image" in seen["stdin"] and "pwned" in seen["stdin"] and "env" not in seen["kw"]
    assert seen["stdin"].count("\n") == 2                 # the prompt was flattened to one line


def test_no_credentials_are_read_by_the_plugin():
    """Each CLI uses its own login and the only key is the one the user typed into the panel: the plugin source
    never opens .env files, CLI credential files or API key environment variables."""
    for f in (ROOT / "plugins" / "online").glob("*.py"):
        src = f.read_text(encoding="utf-8")
        for needle in ("API_KEY", "load_dotenv", "oauth_creds", "environ[", "gemini-extension.json", "settings.json"):
            assert needle not in src, (f.name, needle)
        assert ".env" not in src.replace(".environ", ""), f.name


# ------------------------------------------------------------------------------------------------ opt-in + API key

def test_google_clis_are_opt_in_and_claude_is_the_default(tmp_path):
    from plugins.online import images
    c = make(tmp_path)
    c.online_opts.pop("image_providers")                  # real provider wiring (nothing is called)
    do(c, op="enable", on=False)
    do(c, op="enable", on=True)
    st = c.state()["online"]
    assert st["backend"] == "claude" and st["cli_google"] is False and st["backends"]["agy"]["optin"] is True
    assert [type(p) for p in c.online.images.providers] == [images.GeminiApiProvider]
    assert st["nano"]["available"] == [] and st["has_key"] is False
    do(c, op="backend", backend="agy")                    # refused without the opt-in
    assert c.state()["online"]["backend"] == "claude"
    do(c, op="cli_google", on=True)
    do(c, op="backend", backend="agy")
    st = c.state()["online"]
    assert st["backend"] == "agy" and st["cli_google"] is True and "terms" in st["cli_note"]
    assert [type(p) for p in c.online.images.providers] == [images.GeminiApiProvider, images.GeminiNanoProvider,
                                                            images.AgyImageProvider]
    do(c, op="cli_google", on=False)                      # opting out again also leaves the agy backend
    st = c.state()["online"]
    assert st["backend"] == "claude" and len(c.online.images.providers) == 1


def test_api_key_is_stored_locally_and_never_leaves(tmp_path):
    key = "AIzaTESTKEY_0123456789abcdefghij"
    c = make(tmp_path)
    c.online_opts.pop("image_providers")
    do(c, op="enable", on=False)
    do(c, op="enable", on=True)
    do(c, op="key", key=key)
    assert json.loads((tmp_path / "online_keys.json").read_text())["gemini_api_key"] == key
    st = c.state()["online"]
    assert st["has_key"] is True and st["nano"]["available"] == ["api"]
    assert key not in json.dumps(st) and key not in json.dumps(c._settings())
    do(c, op="key", key="")
    assert not (tmp_path / "online_keys.json").exists() and c.state()["online"]["has_key"] is False
    c.close()
    assert key not in next((tmp_path / "logs").glob("online_*.log")).read_text(encoding="utf-8")
    for bad in ("short", "has space in the key 0123456789", "x" * 201, 5, None, "semi;colon0123456789abcdef"):
        with pytest.raises(ValueError):
            validate({"cmd": "online", "op": "key", "key": bad})
    assert "online_keys.json" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_gemini_api_provider_request_and_reply(tmp_path, monkeypatch):
    import base64
    import io
    import urllib.error
    from PIL import Image
    from plugins.online import images
    key = "AIzaTESTKEY_0123456789abcdefghij"
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (9, 8, 7)).save(buf, "PNG")
    png = base64.b64encode(buf.getvalue()).decode()
    seen = {}

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self, n=-1):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=0):
        seen.update(url=req.full_url, headers={k.lower(): v for k, v in req.header_items()}, body=json.loads(req.data))
        return Resp(json.dumps({"id": "x", "steps": [{"type": "thought", "summary": [{"type": "text", "text": "hm"}]},
                                                     {"type": "model_output", "content": [{"type": "text", "text": "ok"},
                                                                                          {"type": "image", "data": png}]}]}).encode())

    monkeypatch.setattr(images.urllib.request, "urlopen", fake_open)
    monkeypatch.setitem(sys.modules, "google", None)       # no SDK in this test: the plain HTTPS path
    ks = images.KeyStore(tmp_path / "online_keys.json")
    prov = images.GeminiApiProvider(ks)
    assert not prov.available()
    ks.set(key)
    assert prov.available()
    Image.new("RGB", (8, 8)).save(tmp_path / "frame.png")
    out = prov.generate("a buddha\nline two", tmp_path / "frame.png", tmp_path)
    with Image.open(out) as im:
        assert im.size == (16, 16)
    assert seen["url"] == images.API_URL and seen["headers"]["x-goog-api-key"] == key
    assert seen["body"]["model"] == images.API_MODEL
    assert seen["body"]["input"][0] == {"type": "text", "text": "a buddha line two"}
    assert seen["body"]["input"][1]["type"] == "image" and seen["body"]["input"][1]["mime_type"] == "image/png"

    def boom(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, io.BytesIO(b'{"error": {"message": "quota exceeded"}}'))

    monkeypatch.setattr(images.urllib.request, "urlopen", boom)
    with pytest.raises(images.BackendError) as ex:
        prov.generate("a buddha", None, tmp_path)
    assert "429" in str(ex.value) and "quota" in str(ex.value) and key not in str(ex.value)
    monkeypatch.setattr(images.urllib.request, "urlopen", lambda req, timeout=0: Resp(b'{"steps": []}'))
    with pytest.raises(images.BackendError):
        prov.generate("a buddha", None, tmp_path)


# ------------------------------------------------------------------------------------------------ panel (headless)

def test_panel_assistant_tab_switch_chat_apply_undo_and_engine_selector(tmp_path):
    """The real panel in a headless browser against a real controller with a fake backend."""
    import socket
    pw_api = pytest.importorskip("playwright.sync_api")
    from remote_control import RemoteControl
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art", tex_dir=tmp_path / "tex")
    rc.start()
    ctl = Controller(tmp_path, restore=False, autosave=False, midi=False, ai=False)
    ctl.ai = FakeAI()
    fake = FakeBackend([reply({"cmd": "set", "name": "zoom", "v": 0.5}, {"cmd": "trigger", "name": "strobe", "on": True}, say="Deeper.")])
    ctl.online_opts = dict(root=tmp_path, backends={"claude": fake, "agy": FakeBackend(ok=False)},
                           log_dir=None)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            ctl.apply(rc.drain())
            ctl.update(1 / 60, [0.5, 0.3, 0.1])
            rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60))
            time.sleep(1 / 60)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    errors = []
    try:
        with pw_api.sync_playwright() as pw:
            br = None
            for channel in ("msedge", "chrome", None):
                try:
                    br = pw.chromium.launch(channel=channel, headless=True) if channel else pw.chromium.launch(headless=True)
                    break
                except Exception:
                    continue
            if br is None:
                pytest.skip("no Chromium-family browser available")
            page = br.new_page(viewport=dict(width=1500, height=950))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(f"http://127.0.0.1:{port}/?k=tok")
            page.wait_for_function("window.__panel && window.__panel.state().online && window.__online", timeout=10000)
            assert page.inner_text("#onlsw") == "ONLINE OFF"
            page.click('#tabs button[data-tab="ai"]')
            assert not page.is_visible("#onlpriv")                 # no Google note while offline
            page.click('#onleng .seg button[data-v="nano"]')       # refused while offline
            page.wait_for_timeout(400)
            assert ctl.online is None and protocol.settings(ctl)["engine"] == "local"
            page.click('#tabs button[data-tab="assistant"]')
            assert "ONLINE is off" in page.inner_text("#onllog")
            page.click("#onlbig")
            page.wait_for_function("window.__online.state().on === true", timeout=5000)
            assert ctl.online is not None and page.inner_text("#onlsw") == "ONLINE ON"
            page.fill("#onlta", "go deeper")
            page.press("#onlta", "Enter")
            page.wait_for_selector("#onllog .onl-p.pending", timeout=8000)
            txt = page.inner_text("#onllog")
            assert "Deeper." in txt and "Zoom" in txt and "refused: trigger" in txt and "Fake Claude" in txt
            assert ctl.base["zoom"] == 0.0 and not ctl.held["strobe"]
            page.click('#onllog .onl-p.pending button[data-act="apply"]')
            page.wait_for_selector("#onllog .onl-p.applied", timeout=5000)
            assert ctl.base["zoom"] == 0.5
            assert "Zoom" in page.inner_text("#onlhist")
            page.screenshot(path=str(ROOT / "test_frames" / "online_assistant_tab.png"))
            page.click('#onllog .onl-p.applied button[data-act="undo"]')
            page.wait_for_selector("#onllog .onl-p.undone", timeout=5000)
            assert ctl.base["zoom"] == 0.0
            page.click('#tabs button[data-tab="ai"]')
            page.click('#onleng .seg button[data-v="nano"]')
            page.wait_for_function("window.__online.state().engine === 'nano'", timeout=5000)
            assert page.is_visible("#onlpriv") and "sent to Google" in page.inner_text("#onlpriv")
            assert "No online image route yet" in page.inner_text("#onlnano")
            page.fill("#onlkin", "AIzaTESTKEY_0123456789abcdefghij")
            page.click("#onlksave")
            page.wait_for_function("window.__online.state().has_key === true", timeout=5000)
            assert page.input_value("#onlkin") == "" and "AIzaTESTKEY" not in page.content()
            page.screenshot(path=str(ROOT / "test_frames" / "online_ai_dream.png"))
            page.click('#tabs button[data-tab="assistant"]')
            assert not page.is_checked("#onlcli")
            page.click('#onlbe button[data-v="agy"]')              # refused: opt-in not ticked
            page.wait_for_timeout(300)
            assert protocol.settings(ctl)["backend"] == "claude"
            page.check("#onlcli")
            page.wait_for_function("window.__online.state().cli_google === true", timeout=5000)
            page.click('#onlbe button[data-v="agy"]')
            page.wait_for_function("window.__online.state().backend === 'agy'", timeout=5000)
            page.uncheck("#onlcli")
            page.wait_for_function("window.__online.state().backend === 'claude'", timeout=5000)
            page.click('#tabs button[data-tab="perform"]')
            assert page.is_visible("#onlmini") and page.is_enabled("#onlmi")
            assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 2
            page.click("#onlsw")                                   # and off again from the top bar
            page.wait_for_function("window.__online.state().on === false", timeout=5000)
            assert ctl.online is None
            br.close()
    finally:
        stop.set()
        th.join(timeout=2)
        ctl.close()
        rc.stop()
    assert errors == []
