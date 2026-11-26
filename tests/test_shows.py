"""Shows, presets, settings: save / load later, autosave + backups + quarantine, migration, .entheo zip import /
export (sandboxed), partial recall, media refs. Pure python, no GL: python -m pytest -q tests/test_shows.py"""
import io
import json
import zipfile
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import control  # noqa: E402
import shows as S  # noqa: E402

DT = 1 / 60


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(d, clock=None, restore=True):
    c = control.Controller(d, restore=restore, midi=False, ai=False, bpm=120.0, seed=1, threads=False,
                           time_fn=clock or Clock(), art_dir=Path(d) / "art")
    return c


def do(c, **cmd):
    out = c.apply([cmd])
    c.update(DT, (0, 0, 0))                           # runs the (synchronous) worker + applies its results
    return out


def run(c, n=3):
    out = []
    for _ in range(n):
        c.update(DT, (0, 0, 0))
        out += c.apply([])
    return out


def show_json(d, slug):
    return json.loads((Path(d) / "shows" / slug / "show.json").read_text(encoding="utf-8"))


def listing(c, kind):
    c.state()                                         # the blobs are refreshed when the state is published
    return json.loads(c.blob_cache[kind][1])          # THIS controller's (control.blob() serves whichever published last)


# ------------------------------------------------------------------ shows


def test_save_as_writes_a_versioned_manifest_atomically(tmp_path):
    c = make(tmp_path)
    c.set("zoom", 0.7)
    do(c, cmd="scene_save", i=3)
    do(c, cmd="tl", op="items_add", track=c.tl.first("marker")["id"], items=[{"t": 16, "name": "drop"}])
    do(c, cmd="show", op="save_as", name="Gorge Night!")
    m = show_json(tmp_path, "gorge-night")
    assert m["format"] == "entheo-show" and m["version"] == S.SHOW_VERSION and m["name"] == "Gorge Night!"
    assert m["controller"]["base"]["zoom"] == 0.7 and m["controller"]["scenes"][3]["base"]["zoom"] == 0.7
    assert m["timeline"]["tracks"][0]["items"][0]["name"] == "drop" and m["timelines"][0]["name"] == "Timeline 1"
    assert "online" not in m["controller"]["settings"]["ext"]
    assert c.shows.current == "gorge-night" and not c.shows.dirty
    assert not list((tmp_path / "shows" / "gorge-night").glob("*.tmp"))
    assert json.loads((tmp_path / "shows" / ".recent.json").read_text())[0] == "gorge-night"
    assert [s["slug"] for s in listing(c, "shows")["shows"]] == ["gorge-night"]


def test_open_restores_look_scenes_timeline_settings_but_keeps_the_rig(tmp_path):
    c = make(tmp_path)
    c.set("zoom", 0.7)
    c.set("hue", 0.3)
    do(c, cmd="scene_save", i=5)
    do(c, cmd="bpm", v=97.0)
    do(c, cmd="bypass", key="l_beads", on=True)
    do(c, cmd="midi_clear")
    do(c, cmd="tl", op="track_add", kind="auto", param="zoom")
    do(c, cmd="show", op="save_as", name="A")
    # change everything
    c.set("zoom", -0.4)
    do(c, cmd="scene_clear", i=5)
    do(c, cmd="bpm", v=140.0)
    do(c, cmd="bypass", key="l_beads", on=False)
    do(c, cmd="midi_defaults")
    do(c, cmd="tl", op="clear")
    n_bind = len(c.midi.bindings)
    do(c, cmd="morph", bars=0)
    do(c, cmd="show", op="open", slug="a")
    run(c)
    assert c.base["zoom"] == 0.7 and c.base["hue"] == 0.3 and c.scenes[5] is not None
    assert c.bpm == 97.0 and "l_beads" in c.bypass and "zoom" in c.tl.by_param
    assert len(c.midi.bindings) == n_bind            # the venue's MIDI map was kept (rig not asked for)
    do(c, cmd="show", op="open", slug="a", parts=["midi"])
    assert c.midi.bindings == []


def test_open_mapping_goes_out_as_a_validated_surf_command(tmp_path):
    (tmp_path / "surfaces.json").write_text(json.dumps({"version": 1, "surfaces": [
        {"id": 1, "name": "Box", "shape": "quad", "corners": [[10, 10], [500, 10], [500, 400], [10, 400]]}]}))
    c = make(tmp_path)
    do(c, cmd="show", op="save_as", name="Rig")
    out = c.apply([{"cmd": "show", "op": "open", "slug": "rig", "parts": ["mapping"]}])
    out += run(c)
    surf = [x for x in out if x.get("cmd") == "surf"]
    assert surf and surf[0]["op"] == "replace" and surf[0]["surfaces"][0]["name"] == "Box"


def test_backups_rotate_and_a_corrupt_show_is_quarantined(tmp_path):
    c = make(tmp_path)
    do(c, cmd="show", op="save_as", name="B")
    for i in range(12):
        c.set("zoom", i / 20)
        do(c, cmd="show", op="save")
    d = tmp_path / "shows" / "b"
    assert len(list((d / "backups").glob("show.*.json"))) <= S.KEEP_BACKUPS
    (d / "show.json").write_text("{not json")
    c.set("zoom", -0.9)
    do(c, cmd="show", op="open", slug="b")
    assert c.base["zoom"] == -0.9                     # nothing applied
    assert list(d.glob("show.json.*.bad")) and c.shows.notice[1] == "error"


def test_newer_versions_are_refused_and_old_live_state_migrates(tmp_path):
    with pytest.raises(ValueError):
        S.migrate_show({"format": "entheo-show", "version": S.SHOW_VERSION + 1})
    with pytest.raises(ValueError):
        S.migrate_show({"hello": 1})
    m = S.migrate_show({"base": {"zoom": 0.4}, "settings": {"bpm": 90}})
    plan = S.clean_show(m)
    assert plan["look"]["base"]["zoom"] == 0.4 and plan["settings"]["bpm"] == 90


def test_clean_show_strips_online_and_never_turns_a_camera_on(tmp_path):
    raw = {"format": "entheo-show", "version": 1, "name": "x", "controller": {"settings": {
        "ext": {"online": {"on": True}, "camera": {"on": True, "mirror": False}, "governor": {"mode": "auto"}}}},
        "timeline": {"tracks": [{"kind": "event", "items": [{"t": 1, "cmd": {"cmd": "camera", "op": "on"}},
                                                            {"t": 2, "cmd": {"cmd": "online", "op": "enable", "on": True}}]}]}}
    plan = S.clean_show(raw)
    ext = plan["settings"]["ext"]
    assert "online" not in ext and ext["governor"] == {"mode": "auto"}
    assert "on" not in ext.get("camera", {})
    ev = [t for t in plan["timelines"][0]["tracks"] if t["kind"] == "event"]
    assert ev and ev[0]["items"] == []


def test_autosave_is_debounced_and_recover_offers_it(tmp_path):
    clk = Clock()
    c = make(tmp_path, clk)
    do(c, cmd="show", op="save_as", name="C")
    c.set("zoom", 0.33)
    clk.t += 5
    run(c)
    assert not (tmp_path / "shows" / "c" / "autosave.json").exists()
    clk.t += control.SHOW_DEBOUNCE
    run(c)
    a = json.loads((tmp_path / "shows" / "c" / "autosave.json").read_text())
    assert a["controller"]["base"]["zoom"] == 0.33
    assert listing(c, "shows")["shows"][0]["autosave"] is True      # newer by the timestamp inside the files
    c.set("zoom", 0.0)
    do(c, cmd="morph", bars=0)
    do(c, cmd="show", op="open", slug="c", recover=True)
    assert c.base["zoom"] == 0.33


def test_restart_brings_the_open_shows_timeline_back_stopped(tmp_path):
    c = make(tmp_path)
    do(c, cmd="tl", op="items_add", track=c.tl.first("cue")["id"], items=[{"t": 8, "kind": "scene", "slot": 1}])
    do(c, cmd="tl", op="play")
    do(c, cmd="show", op="save_as", name="D")
    c.close()
    c2 = make(tmp_path)
    run(c2)
    assert c2.shows.current == "d" and c2.tl.first("cue")["items"][0]["t"] == 8 and not c2.tl.playing


def test_untitled_timeline_survives_a_crash_restart(tmp_path):
    clk = Clock()
    c = make(tmp_path, clk)
    do(c, cmd="tl", op="name", name="Unsaved set")
    do(c, cmd="tl", op="track_add", kind="text")
    clk.t += 100
    run(c)
    c2 = make(tmp_path)
    run(c2)
    assert c2.tl.data["name"] == "Unsaved set" and c2.shows.current is None


def test_new_show_from_clean_factory_current(tmp_path):
    c = make(tmp_path)
    do(c, cmd="tl", op="track_add", kind="text")
    c.set("zoom", 0.5)
    do(c, cmd="show", op="new", **{"from": "current"})
    assert c.base["zoom"] == 0.5 and not c.tl.first("text") and c.shows.current is None
    do(c, cmd="show", op="new", **{"from": "factory"})
    assert c.base["zoom"] == 0.0 and c.scenes[1]["name"] == "Breathe"
    do(c, cmd="show", op="new", **{"from": "clean"})
    assert c.canvas_prev is not None


def test_template_opens_as_untitled(tmp_path):
    c = make(tmp_path)
    c.set("zoom", 0.25)
    do(c, cmd="show", op="save_as", name="Tpl", template=True)
    c.set("zoom", 0.0)
    do(c, cmd="morph", bars=0)
    do(c, cmd="show", op="new", slug="tpl", **{"from": "template"})
    assert c.base["zoom"] == 0.25 and c.shows.current is None and c.shows.dirty


def test_slugs_are_sandboxed(tmp_path):
    for bad in ("../x", "a/b", "A", "", "x" * 60, "..", "_media", "con\0"):
        with pytest.raises(ValueError):
            control.validate({"cmd": "show", "op": "open", "slug": bad})
    for bad in ("look/../x", "look", "nope/x", "look/a b"):
        with pytest.raises(ValueError):
            control.validate({"cmd": "preset", "op": "recall", "ref": bad})
    m = S.ShowManager(tmp_path, start=False)
    with pytest.raises(ValueError):
        m.preset_path("look", "../../etc")
    assert S.slugify("../../ Hello World ") == "hello-world"


# ------------------------------------------------------------------ .entheo


def test_export_import_roundtrip_with_collected_media(tmp_path):
    a, b = tmp_path / "laptop_a", tmp_path / "laptop_b"
    (a / "art").mkdir(parents=True)
    (a / "art" / "piece one.png").write_bytes(b"\x89PNG fake")
    c = make(a)
    c.set("art_index", 0)
    do(c, cmd="show", op="save_as", name="Travel", media=True)
    data = S.export_zip(a / "shows" / "travel")
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "show.json" in names and "media/art/piece one.png" in names
    slug = S.import_zip(data, b / "shows")
    c2 = make(b)
    do(c2, cmd="show", op="open", slug=slug)
    assert (b / "art" / "piece one.png").read_bytes() == b"\x89PNG fake" and c2.shows.missing == []


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in entries.items():
            z.writestr(n, d)
    return buf.getvalue()


def test_import_rejects_zip_slip_and_junk(tmp_path):
    good = json.dumps({"format": "entheo-show", "version": 1, "name": "z"})
    for entries in ({"show.json": good, "../evil.py": "x"}, {"show.json": good, "media/art/../../x.png": "x"},
                    {"show.json": good, "run.bat": "x"}, {"show.json": "{bad"}, {"thumbs/scene_0.jpg": "x"},
                    {"show.json": json.dumps({"format": "entheo-show", "version": 99})}):
        with pytest.raises(ValueError):
            S.import_zip(_zip(entries), tmp_path / "shows")
    with pytest.raises(ValueError):
        S.import_zip(b"not a zip", tmp_path / "shows")
    assert not (tmp_path / "evil.py").exists()
    slug = S.import_zip(_zip({"show.json": good}), tmp_path / "shows")
    slug2 = S.import_zip(_zip({"show.json": good}), tmp_path / "shows")
    assert slug != slug2                              # never overwrites


# ------------------------------------------------------------------ presets


def test_look_preset_save_list_recall_partial_and_morph(tmp_path):
    c = make(tmp_path)
    c.set("zoom", 0.6)
    c.set("hue", 0.4)
    c.set_mod("zoom", 0, "lfo_sine", 0.3, 4)
    do(c, cmd="preset", op="save", kind="look", name="Deep Blue", tags=["Chill", "blue"])
    lst = listing(c, "presets")["presets"]
    assert lst[0]["ref"] == "look/deep-blue" and lst[0]["tags"] == ["chill", "blue"] and lst[0]["n"] > 50
    c.set("zoom", 0.0)
    c.set("hue", 0.0)
    c.mods = {}
    do(c, cmd="preset", op="recall", ref="look/deep-blue", groups=["color"])     # partial: colour only
    assert c.base["hue"] == pytest.approx(0.4) and c.base["zoom"] == 0.0 and "zoom" not in c.mods
    do(c, cmd="preset", op="recall", ref="look/deep-blue", morph=1)             # whole look, morphed over a bar
    run(c, 10)
    assert 0.0 < c.base["zoom"] < 0.6
    run(c, 140)
    assert c.base["zoom"] == pytest.approx(0.6) and c.mods["zoom"][0]["src"] == "lfo_sine"


def test_preset_tags_fav_delete_and_corrupt_files(tmp_path):
    c = make(tmp_path)
    do(c, cmd="preset", op="save", kind="mod", name="Wobble")
    do(c, cmd="preset", op="set", ref="mod/wobble", fav=True, tags=["live"], name="Wobble 2")
    p = listing(c, "presets")["presets"][0]
    assert p["fav"] and p["tags"] == ["live"] and p["name"] == "Wobble 2"
    (tmp_path / "presets" / "look").mkdir(parents=True, exist_ok=True)
    (tmp_path / "presets" / "look" / "broken.json").write_text("{nope")
    do(c, cmd="preset", op="rescan")
    assert [x["ref"] for x in listing(c, "presets")["presets"]] == ["mod/wobble"]
    assert list((tmp_path / "presets" / "look").glob("broken.json.*.bad"))
    do(c, cmd="preset", op="delete", ref="mod/wobble")
    assert listing(c, "presets")["presets"] == [] and list((tmp_path / "presets" / ".trash").iterdir())


def test_layer_fx_midi_presets_and_import_validation(tmp_path):
    c = make(tmp_path)
    c.set("l_fractal", 0.8)
    do(c, cmd="preset", op="save", kind="layer", name="Fractal", layer="l_fractal")
    p = json.loads((tmp_path / "presets" / "layer" / "fractal.json").read_text())
    assert p["data"]["layer"] == "l_fractal" and "l_fractal" in p["data"]["values"]
    assert all(control.PARAMS[n][3] in ("layers", "fractal") for n in p["data"]["values"])
    do(c, cmd="preset", op="save", kind="midi", name="APC")
    c.midi.bindings = []
    do(c, cmd="preset", op="recall", ref="midi/apc")
    assert len(c.midi.bindings) > 10
    # import: a shared file is cleaned; garbage and unknown params are dropped, never executed
    raw = {"format": "entheo-preset", "version": 1, "kind": "look", "name": "Shared",
           "data": {"values": {"zoom": 5, "no_such": 1, "flash": 1}, "mods": {"zoom": [{"src": "__import__", "amt": 1}]}}}
    clean = S.clean_preset(raw)
    assert clean["data"]["values"] == {"zoom": 1.0} and clean["data"]["mods"] == {}
    for bad in ({"format": "x"}, {"format": "entheo-preset", "version": 9, "kind": "look"},
                {"format": "entheo-preset", "version": 1, "kind": "shell"},
                {"format": "entheo-preset", "version": 1, "kind": "look", "data": {}}):
        with pytest.raises(ValueError):
            S.clean_preset(bad)


def test_timeline_preset_cue_fires_from_the_cache(tmp_path):
    c = make(tmp_path)
    c.set("hue", 0.8)
    do(c, cmd="preset", op="save", kind="look", name="Hot")
    c.set("hue", 0.0)
    do(c, cmd="tl", op="items_add", track=c.tl.first("cue")["id"],
       items=[{"t": 1, "kind": "preset", "preset": "look/hot", "morph": 0}])
    do(c, cmd="tl", op="play")
    for _ in range(70):
        c.update(DT, (0, 0, 0))
    assert c.base["hue"] == pytest.approx(0.8)


def test_readonly_controller_never_writes(tmp_path):
    c = make(tmp_path, restore=False)
    do(c, cmd="show", op="save_as", name="X")
    do(c, cmd="preset", op="save", kind="look", name="X")
    assert not (tmp_path / "shows").exists() and not (tmp_path / "presets").exists()


def test_media_upload_names_are_checked(tmp_path):
    for bad in ("../a.mp3", "a.exe", ".hidden.mp3", "a" * 200 + ".mp3"):
        with pytest.raises(ValueError):
            S.save_media(tmp_path, bad, b"x")
    assert S.save_media(tmp_path, "My Song (live).mp3", b"x") == "My Song (live).mp3"


# ------------------------------------------------------------------ HTTP routes (token-gated, sandboxed)


def test_http_routes_export_import_thumbs_and_sandbox(tmp_path):
    import socket
    import urllib.error
    import urllib.request
    from remote_control import RemoteControl
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    c = make(tmp_path)
    c.set("zoom", 0.4)
    do(c, cmd="show", op="save_as", name="Net")
    do(c, cmd="preset", op="save", kind="look", name="Net look")
    c.state()
    r = RemoteControl(port=port, token="tk", quiet=True, host="127.0.0.1")
    r.state_dir = tmp_path
    assert r.toggle()
    base = f"http://127.0.0.1:{port}"

    def get(path, k="tk"):
        try:
            with urllib.request.urlopen(f"{base}{path}{'&' if '?' in path else '?'}k={k}", timeout=5) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, b""

    def post(path, data, headers=None):
        req = urllib.request.Request(f"{base}{path}?k=tk", data=data, headers=headers or {}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, None
    try:
        assert get("/shows", k="bad")[0] == 403
        code, body = get("/shows")
        assert code == 200 and json.loads(body)["shows"][0]["slug"] == "net"
        assert json.loads(get("/timeline")[1])["v"] == 1
        code, z = get("/show-export/net?media=0")
        assert code == 200 and zipfile.ZipFile(io.BytesIO(z)).read("show.json")
        for bad in ("/show-export/..%2F..%2Fx", "/preset-export/look/..%2Fx", "/preset-thumb/x/y", "/media/..%2Fa.mp3",
                    "/scene-thumb/99"):
            assert get(bad)[0] in (400, 404)
        code, pj = get("/preset-export/look/net-look")
        assert code == 200 and json.loads(pj)["kind"] == "look"
        code, out = post("/preset-import", pj, {"Content-Type": "application/json"})
        assert code == 200 and out["ref"].startswith("look/")
        assert post("/preset-import", b'{"format":"evil"}')[0] == 400
        code, out = post("/show-import", z)
        assert code == 200 and out["slug"].startswith("net")
        assert post("/show-import", b"PK junk")[0] == 400
        assert post("/media-upload", b"RIFFxxxx", {"X-Filename": "loop.wav"})[0] == 200
        assert (tmp_path / "shows" / "_media" / "loop.wav").read_bytes() == b"RIFFxxxx"
        assert post("/media-upload", b"x", {"X-Filename": "..%2Fevil.wav"})[0] == 400
        assert post("/media-upload", b"x", {"X-Filename": "run.exe"})[0] == 400
        assert get("/media/loop.wav")[0] == 200
        assert not list((tmp_path / "shows" / ".incoming").glob("*.part"))
    finally:
        r.stop()


# ------------------------------------------------------------------ Neon review regressions


def test_n11_n13_zip_lies_bad_media_and_fake_thumbs(tmp_path):
    good = json.dumps({"format": "entheo-show", "version": 1, "name": "z"})
    for entries in ({"show.json": good, "media/art/evil.exe": "MZ"}, {"show.json": good, "media/audio/x.png": "x"},
                    {"show.json": good, "media/tex/a.py": "import os"}):
        with pytest.raises(ValueError):
            S.import_zip(_zip(entries), tmp_path / "shows")
    # a header that lies about the size: the real bytes are counted
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("show.json", good)
        z.writestr("thumbs/scene_0.jpg", b"\xff\xd8" + b"\0" * (S.MAX_THUMB + 10))
    data = bytearray(buf.getvalue())
    with pytest.raises(ValueError):
        S.import_zip(bytes(data), tmp_path / "shows")
    slug = S.import_zip(_zip({"show.json": good, "thumbs/scene_1.jpg": b"<script>"}), tmp_path / "shows")
    assert not (tmp_path / "shows" / slug / "thumbs" / "scene_1.jpg").exists()      # not a JPEG: dropped


def test_n34_partial_open_keeps_the_open_show(tmp_path):
    c = make(tmp_path)
    do(c, cmd="show", op="save_as", name="A")
    do(c, cmd="show", op="save_as", name="B")
    do(c, cmd="show", op="open", slug="a")
    assert c.shows.current == "a"
    do(c, cmd="show", op="open", slug="b", parts=["timeline"])
    assert c.shows.current == "a" and c.shows.dirty



# ------------------------------------------------------------------ PR2 review (research/neon_pr2_review.md) B16-B19


def test_b16_listing_never_serves_a_stale_summary_after_a_quick_rewrite(tmp_path):
    c = make(tmp_path)
    for i in range(5):                                # save + star + rename inside one file-clock tick, five times
        do(c, cmd="preset", op="save", kind="mod", name=f"Quick {i}")
        do(c, cmd="preset", op="set", ref=f"mod/quick-{i}", fav=True, tags=["x"], name=f"Q{i}")
        p = [x for x in listing(c, "presets")["presets"] if x["ref"] == f"mod/quick-{i}"][0]
        assert p["fav"] and p["tags"] == ["x"] and p["name"] == f"Q{i}"


def test_b17_invalid_utf8_is_corrupt_not_a_crash(tmp_path):
    c = make(tmp_path)
    do(c, cmd="preset", op="save", kind="mod", name="Good")
    (tmp_path / "presets" / "mod" / "bad.json").write_bytes(b'{"name": "\xff\xfe"}')
    d = tmp_path / "shows" / "broken"
    d.mkdir(parents=True)
    (d / "show.json").write_bytes(b"\xff\xfe\x00{")
    do(c, cmd="preset", op="rescan")
    do(c, cmd="show", op="rescan")
    assert [x["ref"] for x in listing(c, "presets")["presets"]] == ["mod/good"]
    assert list((tmp_path / "presets" / "mod").glob("bad.json.*.bad"))
    assert [s["broken"] for s in listing(c, "shows")["shows"] if s["slug"] == "broken"] == [True]
    do(c, cmd="show", op="open", slug="broken")
    assert c.shows.notice[1] == "error" and list(d.glob("show.json.*.bad"))
    assert S.read_json(d / "nope.json", 100) == (None, "missing")


def test_b19_advertised_rev_matches_the_served_blob(tmp_path):
    c = make(tmp_path)
    st = c.state()
    m = c.shows
    m.rev["presets"] += 7                            # the worker moved on after state() took its blobs
    st2 = c.state()
    m.blobs["presets"] = (m.rev["presets"], m.blobs["presets"][1])
    for s_ in (st, st2):
        for k in ("shows", "presets"):
            assert s_["show"]["rev"][k] == c.blob_cache[k][0] or s_ is st
    assert st2["show"]["rev"]["presets"] == control.blob("presets")[0]



# ------------------------------------------------------------------ B16 hardening (parked test from the Phase 0/1 agent, adopted)


def test_preset_listing_sees_a_rewrite_inside_one_mtime_tick(tmp_path, monkeypatch):
    """Regression (the old flake): NTFS keeps the same mtime_ns for rewrites inside one clock tick (~15 ms), so a
    listing cache keyed on mtime alone showed the summary from before `preset set`. It only failed when the save and
    the set landed in the same tick. Forced here: every rewrite keeps the file's previous mtime."""
    import os
    real = S.write_atomic

    def same_tick(path, data):
        path = Path(path)
        old = path.stat().st_mtime_ns if path.exists() else None
        ok = real(path, data)
        if old is not None:
            os.utime(path, ns=(old, old))
        return ok
    monkeypatch.setattr(S, "write_atomic", same_tick)
    c = make(tmp_path)
    do(c, cmd="preset", op="save", kind="mod", name="Wobble")
    assert listing(c, "presets")["presets"][0]["fav"] is False
    do(c, cmd="preset", op="set", ref="mod/wobble", fav=True)
    assert listing(c, "presets")["presets"][0]["fav"] is True
    f = tmp_path / "presets" / "mod" / "wobble.json"
    raw = json.loads(f.read_text())                   # an edit by another process, same tick again
    raw["fav"] = False
    same_tick(f, json.dumps(raw).encode())
    do(c, cmd="preset", op="rescan")
    assert listing(c, "presets")["presets"][0]["fav"] is False




def test_b16_stress_200_rapid_rewrites_never_list_stale(tmp_path):
    c = make(tmp_path)
    do(c, cmd="preset", op="save", kind="mod", name="Spin")
    f = tmp_path / "presets" / "mod" / "spin.json"
    mt0 = f.stat().st_mtime_ns
    for i in range(200):
        do(c, cmd="preset", op="set", ref="mod/spin", tags=[f"t{i}"], fav=bool(i % 2))
        p = listing(c, "presets")["presets"][0]
        assert p["tags"] == [f"t{i}"] and p["fav"] == bool(i % 2), i
        if i % 25 == 0:                               # and an outside edit that keeps the old mtime and size
            raw = json.loads(f.read_text())
            raw["name"] = "Spin" if raw["name"] != "Spin" else "Spun"
            data = json.dumps(raw).encode()
            m = f.stat().st_mtime_ns
            f.write_bytes(data)
            import os
            os.utime(f, ns=(m, m))
            do(c, cmd="preset", op="rescan")
            assert listing(c, "presets")["presets"][0]["name"] == raw["name"], i
    assert mt0                                        # (the file existed throughout)


def test_b16_file_hash_is_content_based(tmp_path):
    import os
    f = tmp_path / "a.png"
    f.write_bytes(b"one")
    h1 = S.file_hash(f)
    m = f.stat().st_mtime_ns
    f.write_bytes(b"two")                             # same size, same mtime, written in place by someone else
    os.utime(f, ns=(m, m))
    assert S.file_hash(f) != h1
    big = tmp_path / "big.wav"
    big.write_bytes(b"\0" * S.HASH_CACHE_MIN)
    hb = S.file_hash(big)
    S.write_atomic(big, b"\1" * S.HASH_CACHE_MIN)    # our own rewrite: the write generation misses the cache
    assert S.file_hash(big) != hb
