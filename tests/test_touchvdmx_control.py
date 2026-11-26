"""TouchVDMX merge, control side: step sequencer mod sources, scene A/B crossfader, new option tiles, /preview."""
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import control  # noqa: E402
from control import Controller, validate  # noqa: E402
from params import MOD_SOURCES, PARAMS  # noqa: E402
from remote_control import RemoteControl  # noqa: E402


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


def at_beat(c, beat):
    """Put the clock on `beat` exactly and evaluate one frame."""
    for _ in range(4):                                       # let the fader slew settle first
        c.ft.t += 0.5
        c.update(0.5, [0, 0, 0])
    c.beat = beat
    c.ft.t += 1e-3
    return c.update(0.0, [0, 0, 0]).values


def do(c, **cmd):
    rest = c.apply([cmd])
    assert rest == [], f"not consumed: {rest}"


# ---------------------------------------------------------------- contract


def test_mod_sources_are_appended():
    assert MOD_SOURCES[:9] == ["none", "bass", "mid", "high", "lfo_sine", "lfo_tri", "lfo_saw", "lfo_square", "random"]
    assert MOD_SOURCES[9:11] == ["seq", "seq_gate"]       # later features may append more after these
    assert control.meta()["sources"] == list(MOD_SOURCES)


def test_new_stepped_params_have_tiles():
    opts = {p["name"]: p for p in control.meta()["params"]}
    for name in ("land_a", "land_b", "land_blend", "land_cam", "land_style", "land_mirror"):
        p = opts[name]
        assert p["stepped"] and len(p["options"]) == int(PARAMS[name][1] - PARAMS[name][0]) + 1, name
    assert opts["land_a"]["options"] == opts["land_b"]["options"]
    assert "land" in control.meta()["groups"]


# ---------------------------------------------------------------- step sequencer


def test_seq_defaults_and_state(tmp_path):
    c = make(tmp_path)
    s = c.state()["seq"]
    assert len(s["vals"]) == 16 and len(s["gates"]) == 16
    assert all(0.0 <= v <= 1.0 for v in s["vals"]) and set(s["gates"]) <= {0, 1}


def test_seq_source_plays_the_steps(tmp_path):
    c = make(tmp_path)
    for i in range(16):
        do(c, cmd="seq", i=i, v=i / 15.0)
    c.set("zoom", -1.0)
    c.set_mod("zoom", 0, "seq", amt=1.0, rate=1.0)          # rate 1 = the 16 steps span one bar
    for step in (0, 3, 8, 15):
        v = at_beat(c, 8.0 + step * 0.25 + 0.2)               # late in the step: the glide is over
        assert v["zoom"] == pytest.approx(-1.0 + 2.0 * step / 15.0, abs=1e-6), step
    # rate 4: quarter-note steps, the pattern spans four bars
    c.set_mod("zoom", 0, "seq", amt=1.0, rate=4.0)
    assert at_beat(c, 16.0 + 5.9)["zoom"] == pytest.approx(-1.0 + 2.0 * 5 / 15.0, abs=1e-6)


def test_seq_glides_into_each_step(tmp_path):
    c = make(tmp_path)
    for i in range(16):
        do(c, cmd="seq", i=i, v=0.0 if i % 2 else 1.0)
    c.set("swim", 0.0)
    c.set_mod("swim", 0, "seq", amt=1.0, rate=1.0)
    start, mid, late = (at_beat(c, 4.25 + f * 0.25)["swim"] for f in (0.0, 0.075, 0.5))
    assert start == pytest.approx(1.0, abs=1e-6) and late == pytest.approx(0.0, abs=1e-6)
    assert 0.2 < mid < 0.8, "a step change must be a short glide, not a click"


def test_seq_gate_fires_only_on_gated_steps_and_decays(tmp_path):
    c = make(tmp_path)
    for i in range(16):
        do(c, cmd="seq", i=i, g=(i == 4))
    c.set("swim", 0.0)
    c.set_mod("swim", 0, "seq_gate", amt=1.0, rate=1.0)
    assert at_beat(c, 1.0)["swim"] == pytest.approx(1.0, abs=1e-6)            # step 4 starts on beat 1
    assert 0.0 < at_beat(c, 1.2)["swim"] < 0.1                                # nearly gone by the end of the step
    assert at_beat(c, 0.5)["swim"] == 0.0 and at_beat(c, 2.0)["swim"] == 0.0  # ungated steps are silent


def test_seq_commands_are_validated(tmp_path):
    assert validate(dict(cmd="seq", i=3, v=7.0)) == dict(cmd="seq", i=3, v=1.0)
    assert validate(dict(cmd="seq", i=3, g=True)) == dict(cmd="seq", i=3, g=True)
    for bad in (dict(cmd="seq", i=16, v=0.5), dict(cmd="seq", i=0), dict(cmd="seq", i=0, v="x"),
                dict(cmd="seq", i=0, g=1), dict(cmd="seq", i=True, v=0.5), dict(cmd="seq", i=0, v=float("nan"))):
        with pytest.raises(ValueError):
            validate(bad)
    c = make(tmp_path)
    before = c.state()["seq"]
    do(c, cmd="seq_rand")
    after = c.state()["seq"]
    assert after != before and len(after["vals"]) == 16 and all(0.0 <= v <= 1.0 for v in after["vals"])


def test_seq_survives_a_restart_and_bad_files(tmp_path):
    c = make(tmp_path)
    do(c, cmd="seq", i=2, v=0.123)
    do(c, cmd="seq", i=2, g=True)
    do(c, cmd="seq", i=3, g=False)
    assert c._autosave()
    c.close()
    c2 = Controller(tmp_path, midi=False, ai=False, restore=True)
    assert c2.seq["vals"][2] == pytest.approx(0.123) and c2.seq["gates"][2] == 1 and c2.seq["gates"][3] == 0
    c2.close()
    p = tmp_path / "live_state.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["settings"]["seq"] = dict(vals=["x"] * 16, gates=[5] * 3)
    p.write_text(json.dumps(d), encoding="utf-8")
    c3 = Controller(tmp_path, midi=False, ai=False, restore=True)
    assert len(c3.seq["vals"]) == 16 and all(isinstance(v, float) and 0 <= v <= 1 for v in c3.seq["vals"])
    assert len(c3.seq["gates"]) == 16 and set(c3.seq["gates"]) <= {0, 1}
    c3.close()


# ---------------------------------------------------------------- scene A/B crossfader


def two_scenes(c):
    c.set("zoom", -0.8); c.set("travel_mode", 0); c.set("palette", 1.0)
    c.set_mod("swim", 0, "bass", amt=0.5)
    c.scene_save(4)
    c.mods = {}
    c.set("zoom", 0.6); c.set("travel_mode", 3); c.set("palette", 3.0)
    c.set_mod("hue", 0, "lfo_sine", amt=0.2)
    c.scene_save(9)


def test_xfade_blends_two_scenes(tmp_path):
    c = make(tmp_path)
    two_scenes(c)
    do(c, cmd="xfade", a=4, b=9, x=0.0)
    assert c.base["zoom"] == pytest.approx(-0.8) and c.base["travel_mode"] == 0 and "swim" in c.mods and c.cur == 4
    do(c, cmd="xfade", x=0.25)
    assert c.base["zoom"] == pytest.approx(-0.8 + 1.4 * 0.25) and c.base["palette"] == pytest.approx(1.5)
    assert c.base["travel_mode"] == 0 and "swim" in c.mods and "hue" not in c.mods
    do(c, cmd="xfade", x=0.75)
    assert c.base["travel_mode"] == 3 and "hue" in c.mods and "swim" not in c.mods      # stepped + mods flip at the middle
    do(c, cmd="xfade", x=1.0)
    assert c.base["zoom"] == pytest.approx(0.6) and c.cur == 9
    s = c.state()["xfade"]
    assert s == dict(a=4, b=9, x=1.0)


def test_xfade_cancels_morph_and_pending_and_ignores_empty_decks(tmp_path):
    c = make(tmp_path)
    two_scenes(c)
    c.morph_bars, c.quantize = 8.0, "phrase"
    c.scene_recall(4)
    assert c.pending is not None
    do(c, cmd="xfade", a=4, b=9, x=0.5)
    assert c.pending is None and c._morph is None
    assert c.base["zoom"] == pytest.approx(-0.1)
    z = c.base["zoom"]
    do(c, cmd="xfade", a=4, b=12, x=1.0)                    # slot 13 is empty: the decks load, the fader does nothing
    assert c.base["zoom"] == z and c.state()["xfade"]["b"] == 12
    do(c, cmd="xfade", a=-1, b=-1)
    assert c.state()["xfade"]["a"] == -1


def test_xfade_follows_edits_to_the_scenes(tmp_path):
    c = make(tmp_path)
    two_scenes(c)
    do(c, cmd="xfade", a=4, b=9, x=1.0)
    c.scene_clear(9)
    z = c.base["zoom"]
    do(c, cmd="xfade", x=0.0)                               # deck B is gone: nothing moves, nothing raises
    assert c.base["zoom"] == z


def test_xfade_validation():
    assert validate(dict(cmd="xfade", x=3.0)) == dict(cmd="xfade", x=1.0)
    assert validate(dict(cmd="xfade", a=2, b=-1)) == dict(cmd="xfade", a=2, b=-1)
    for bad in (dict(cmd="xfade"), dict(cmd="xfade", a=16), dict(cmd="xfade", x="1"), dict(cmd="xfade", b=1.5),
                dict(cmd="xfade", x=float("inf"))):
        with pytest.raises(ValueError):
            validate(bad)
    assert control.valid_target("xfade")


def test_xfade_is_midi_learnable(tmp_path):
    c = make(tmp_path)
    two_scenes(c)
    do(c, cmd="xfade", a=4, b=9, x=0.0)
    c._act(dict(target="xfade"), "cc", 3)                   # near the on-screen position: picked up
    c._act(dict(target="xfade"), "cc", 127)
    assert c.base["zoom"] == pytest.approx(0.6)
    c._act(dict(target="xfade"), "cc", 0)
    assert c.base["zoom"] == pytest.approx(-0.8)


# ---------------------------------------------------------------- /preview (panel output monitor)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read(), r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type")


def test_preview_endpoint_and_watch_window():
    r = RemoteControl(port=free_port(), token="tok", quiet=True, host="127.0.0.1")
    r.start()
    try:
        base = f"http://127.0.0.1:{r.port}"
        assert r.preview_job() is None, "nobody is watching: the engine must not capture"
        assert get(base + "/preview")[0] == 403
        status, _, _ = get(base + "/preview?k=tok")
        assert status == 204                                   # asked, nothing captured yet
        assert r.preview_job() == ("out", None)                # ... which starts the captures
        assert r.preview_job() is None                         # one per interval
        jpeg = b"\xff\xd8\xff\xe0" + b"x" * 64
        r.set_preview(jpeg)
        status, body, ctype = get(base + "/preview?k=tok")
        assert (status, body, ctype) == (200, jpeg, "image/jpeg")
        r.set_preview(b"")                                     # junk never replaces a good frame
        r.set_preview("nope")
        r.set_preview(b"x" * (3 << 20))
        assert get(base + "/preview?k=tok")[1] == jpeg
        r._preview_shot -= 10.0
        assert r.preview_job() == ("out", None)
        for v in r._views.values():                            # the panel went away
            v[0] -= 10.0
        r._preview_shot -= 10.0
        assert r.preview_job() is None and r._views == {}
    finally:
        r.stop()


# ---------------------------------------------------------------- source picker (uploaded art + AI textures)


def _png(path, color, size=(96, 64)):
    from PIL import Image
    Image.new("RGB", size, color).save(path)


def test_sources_lists_art_and_ai_textures_newest_first(tmp_path):
    import os
    import remote_control as rcm
    tex, art = tmp_path / "textures", tmp_path / "art"
    tex.mkdir(); art.mkdir()
    for i, n in enumerate(["gen_a.png", "gen_b.png", "seed_tile.jpg"]):
        _png(tex / n, (40 * i, 10, 200))
        os.utime(tex / n, (1000 + i, 1000 + i))
    (tex / "notes.txt").write_text("x")
    (tex / "bad name!.png").write_bytes(b"x")
    _png(art / "piece.png", (200, 30, 30))
    r = RemoteControl(port=free_port(), token="tok", quiet=True, host="127.0.0.1", tex_dir=tex, art_dir=art)
    r.start()
    try:
        base = f"http://127.0.0.1:{r.port}"
        assert get(base + "/sources")[0] == 403
        status, body, _ = get(base + "/sources?k=tok")
        d = json.loads(body)
        assert status == 200 and [t["name"] for t in d["tex"]] == ["seed_tile.jpg", "gen_b.png", "gen_a.png"]
        assert [t["ai"] for t in d["tex"]] == [False, True, True]
        assert [a["name"] for a in d["art"]] == ["piece.png"]
        status, body, ctype = get(base + "/tex-thumb/gen_b.png?k=tok")
        assert status == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8" and len(body) < 20_000
        for bad in ("nope.png", "..%2Fpiece.png", "notes.txt", "bad%20name!.png"):
            assert get(base + f"/tex-thumb/{bad}?k=tok")[0] == 404, bad
        assert len(rcm.tex_listing(tex, cap=2)) == 2
        assert rcm.tex_listing(tmp_path / "missing") == []
    finally:
        r.stop()


def test_show_texture_command_reaches_the_engine_validated():
    import remote_control as rcm
    c = rcm.validate_command(dict(cmd="show_texture", name="gen_0001.png", junk=1))
    assert c["cmd"] == "show_texture" and c["name"] == "gen_0001.png" and "junk" not in c
    for bad in ("../x.png", "a/b.png", "x.exe", "", 5, None, "a" * 200 + ".png"):
        with pytest.raises(ValueError):
            rcm.validate_command(dict(cmd="show_texture", name=bad))


# ---------------------------------------------------------------- params.OPTIONS -> named tiles in the panel


def test_params_options_become_panel_tiles():
    known = {"travel_mode": 1, "being": 1, "zoom": 1}
    out = control.merge_options({"travel_mode": ["A", "B"], "gone": ["x"]},
                                {"being": ["Dragon", "Jaguar spirit", 3], "travel_mode": ["T0", "T1"], "zoom": "nope",
                                 "unknown": ["x"]}, known)
    assert out == {"travel_mode": ["T0", "T1"], "being": ["Dragon", "Jaguar spirit", "3"]}
    assert control.merge_options({"travel_mode": ["A"]}, None, known) == {"travel_mode": ["A"]}
    import params
    for name, labels in getattr(params, "OPTIONS", {}).items():          # whatever a module declares is live
        if name in PARAMS:
            assert control.OPTIONS[name] == [str(x)[:40] for x in labels]
            assert name == "palette" or name in control.DISCRETE


def test_art_pick_command_is_validated_and_passed_to_the_engine():
    import remote_control as rcm
    c = rcm.validate_command(dict(cmd="art_pick", source="ai", name="gen_0001.png", use=3, junk=1))
    assert c == dict(cmd="art_pick", type="art_pick", source="ai", name="gen_0001.png", use=3)
    assert rcm.validate_command(dict(cmd="art_pick"))["source"] == "art"
    for bad in (dict(source="disk"), dict(name="../x.png"), dict(name=5), dict(use=99), dict(use="2"), dict(use=True)):
        with pytest.raises(ValueError):
            rcm.validate_command(dict(bad, cmd="art_pick"))
    ctl = Controller(None, restore=False, midi=False, ai=False)
    try:
        assert ctl.apply([c]) == [c], "the controller must hand art_pick on to the engine untouched"
    finally:
        ctl.close()


# ---------------------------------------------------------------- layer panel: solo preview + thumbnails


def test_solo_views_are_per_panel_and_thumbs_have_their_own_timer():
    """Review fix 4: one shared solo / picture made panel A's LIVE OUTPUT show panel B's solo."""
    import remote_control as rcm
    assert "l_kene" in rcm.LAYER_KEYS and "serpent" in rcm.LAYER_KEYS and "l_swarm" not in rcm.LAYER_KEYS
    r = RemoteControl(port=free_port(), token="tok", quiet=True, host="127.0.0.1")
    r.start()
    try:
        base = f"http://127.0.0.1:{r.port}"
        A, B = base + "/preview?k=tok", base + "/preview?k=tok&solo=l_art,l_kene,bogus,master"
        assert r.preview_job() is None, "nobody is watching"
        assert get(A)[0] == 204 and get(B)[0] == 204
        jobs = []
        for _ in range(4):
            r._preview_shot = -1e9
            jobs.append(r.preview_job())
        assert set(jobs) == {("out", None), ("mix", ("l_art", "l_kene"))}, "both views are served, junk names dropped"
        r.set_preview(b"\xff\xd8OUT")
        r.set_preview(b"\xff\xd8MIX", ("l_art", "l_kene"))
        assert get(A)[1] == b"\xff\xd8OUT" and get(B)[1] == b"\xff\xd8MIX", "each panel gets ITS view"
        assert get(base + "/preview?k=tok&solo=l_kene,l_art")[0] == 204, "a different solo set is a different view"
        r.set_preview(b"\xff\xd8LATE", ("l_beads",))            # a view nobody watches any more is not stored
        assert ("l_beads",) not in r._views
        # thumbnails: only while asked for, on their own (slower) timer, pattern layers more often than pass layers
        r._preview_shot = time_now = __import__("time").monotonic()
        assert r.preview_job() is None, "no thumbnail job until a layer panel asks"
        assert get(base + "/layer-thumb/l_kene?k=tok")[0] == 204
        assert get(base + "/layer-thumb/..%2Fx?k=tok")[0] == 204 and get(base + "/layer-thumb/l_kene")[0] == 403
        seen = []
        for _ in range(len(rcm.THUMB_ORDER)):
            r._preview_shot, r._thumb_shot = time_now + 5, -1e9      # the monitor is not due; a thumbnail is
            seen.append(r.preview_job())
        assert all(k == "thumb" for k, _ in seen) and {a for _, a in seen} == set(rcm.LAYER_KEYS)
        names = [a for _, a in seen]
        assert names.count("l_kene") == 3 and names.count("l_art") == 1
        assert r.preview_job() is None, "thumbnails respect their interval"
        r.set_thumb("l_kene", b"\xff\xd8jpeg")
        r.set_thumb("bogus", b"\xff\xd8jpeg")
        status, body, ctype = get(base + "/layer-thumb/l_kene?k=tok")
        assert (status, body, ctype) == (200, b"\xff\xd8jpeg", "image/jpeg")
        r._thumbs_asked -= 10.0
        r._thumb_shot = -1e9
        r._preview_shot = time_now + 5
        assert r.preview_job() is None, "the layer panel went away: no more thumbnails"
    finally:
        r.stop()


def test_art_pick_accepts_hand_dropped_file_names():
    """Review fix 8: art names with spaces / unicode were rejected by the texture-name regex."""
    import remote_control as rcm
    for name in ("My Painting (final) v2.png", "caf\u00e9 \u2013 nuit.jpeg", "IMG 0001.webp"):
        assert rcm.validate_command(dict(cmd="art_pick", source="art", name=name))["name"] == name
    for bad in ("a/b.png", "..\\x.png", "..", "c:evil.png", "x\x00.png", "line\nbreak.png", "y" * 201):
        with pytest.raises(ValueError):
            rcm.validate_command(dict(cmd="art_pick", source="art", name=bad))
    with pytest.raises(ValueError):
        rcm.validate_command(dict(cmd="art_pick", source="ai", name="has space.png"))


def test_texture_thumbnails_decode_two_at_a_time(tmp_path, monkeypatch):
    """Review fix 9: opening the picker fired up to 64 concurrent decodes."""
    import threading
    import time as _t
    import remote_control as rcm
    from PIL import Image
    for i in range(8):
        Image.new("RGB", (64, 64), (i * 30, 0, 0)).save(tmp_path / f"gen_{i}.png")
    live, peak, lock = [0], [0], threading.Lock()
    real = Image.open

    def slow_open(*a, **k):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        _t.sleep(0.05)
        try:
            return real(*a, **k)
        finally:
            with lock:
                live[0] -= 1

    monkeypatch.setattr(Image, "open", slow_open)
    rcm._TEX_THUMBS.clear()
    out = []
    ths = [threading.Thread(target=lambda n=f"gen_{i}.png": out.append(rcm.tex_thumb(tmp_path, n))) for i in range(8)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert len(out) == 8 and all(o and o[:2] == b"\xff\xd8" for o in out) and peak[0] <= 2


def test_midi_crossfader_soft_takeover(tmp_path):
    """Review item 12: a physical fader far from the on-screen position must not make the look jump."""
    c = make(tmp_path)
    two_scenes(c)
    do(c, cmd="xfade", a=4, b=9, x=0.2)
    z = c.base["zoom"]
    c._act(dict(target="xfade"), "cc", 127)                 # the fader is at the far end: ignored
    c._act(dict(target="xfade"), "cc", 100)
    assert c.base["zoom"] == z and c.xfade["x"] == pytest.approx(0.2)
    c._act(dict(target="xfade"), "cc", 20)                  # it crossed the on-screen position: picked up
    assert c.xfade["x"] == pytest.approx(20 / 127)
    c._act(dict(target="xfade"), "cc", 127)                 # and now it follows
    assert c.xfade["x"] == 1.0 and c.base["zoom"] == pytest.approx(0.6)
    do(c, cmd="xfade", x=0.5)                               # the panel moved it: pick-up is needed again
    c._act(dict(target="xfade"), "cc", 0)
    assert c.xfade["x"] == pytest.approx(0.5)
    c._act(dict(target="xfade"), "cc", 64)                  # within reach of the position
    assert c.xfade["x"] == pytest.approx(64 / 127)




def test_state_is_only_serialised_when_a_panel_asks():
    """Optimisation: with no panel connected, publishing state must cost no JSON encoding at all."""
    r = RemoteControl(port=free_port(), token="tok", quiet=True, host="127.0.0.1")
    r.start()
    try:
        base = f"http://127.0.0.1:{r.port}"
        for i in range(50):
            r.publish_state({"i": i, "ctl": {"x": [1, 2, 3]}})
        assert r.state_published == 50 and r.state_encoded == 0
        status, body, _ = get(base + "/state?k=tok")
        assert status == 200 and json.loads(body)["i"] == 49, "a panel always gets the latest state"
        assert r.state_encoded == 1
        assert json.loads(get(base + "/state?k=tok")[1])["i"] == 49 and r.state_encoded == 1, "no new state: no re-encode"
        r.publish_state({"bad": float("nan")})                 # not JSON-able: the last good snapshot stays
        assert json.loads(get(base + "/state?k=tok")[1])["i"] == 49
        r.publish_state({"i": 77})
        assert json.loads(get(base + "/state?k=tok")[1]) == {"i": 77}
    finally:
        r.stop()
