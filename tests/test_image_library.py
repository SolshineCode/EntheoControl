"""One image library: uploaded art, AI generations and dreamified images usable everywhere (CPU tests)."""
import json
import sys
import time
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import image_library as lib  # noqa: E402


def _png(path, color=(200, 80, 40), prompt=None, age=10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = None
    if prompt:
        meta = PngImagePlugin.PngInfo()
        meta.add_text("prompt", prompt)
    Image.new("RGB", (64, 48), color).save(path, pnginfo=meta)
    t = time.time() - age
    import os
    os.utime(path, (t, t))
    return path.name


@pytest.fixture
def dirs(tmp_path):
    art, tex = tmp_path / "art", tmp_path / "textures"
    _png(art / "20260101-000000-000_bird_ab12.png", age=50)
    _png(art / "20260101-000000-000_bird_ab12_dream_eyes_1.png", age=40)
    _png(tex / "gen_0001_a.png", prompt="a glowing jaguar in the vines", age=30)
    _png(tex / "gen_dream_0002_eyes.png", age=20)
    _png(tex / "seed_noise.png", age=10)
    _png(tex / "gen_0001_a.depth.png", age=5)                  # a depth map is never an image of its own
    return art, tex


# ---------------------------------------------------------------- listing

def test_listing_is_one_list_with_kind_title_and_flags(dirs):
    art, tex = dirs
    items = lib.listing(art, tex)
    ids = [i["id"] for i in items]
    assert "art:20260101-000000-000_bird_ab12.png" in ids and "ai:gen_0001_a.png" in ids
    assert not any(i.endswith(".depth.png") for i in ids)
    by = {i["id"]: i for i in items}
    assert by["ai:gen_0001_a.png"]["title"] == "a glowing jaguar in the vines"       # the prompt is the title
    assert by["ai:gen_dream_0002_eyes.png"]["dreamed"] and by["art:20260101-000000-000_bird_ab12_dream_eyes_1.png"]["dreamed"]
    assert not by["ai:gen_0001_a.png"]["dreamed"] and not by["ai:seed_noise.png"]["ai"]
    assert by["ai:gen_0001_a.png"]["ai"]
    assert all(set(i) >= {"id", "kind", "name", "title", "fav", "picked", "dreamed", "depth", "t"} for i in items)
    assert [i["name"] for i in items if i["kind"] == "art"] == lib.list_art(art)       # art keeps art_index order


def test_favourites_picks_and_titles_persist_atomically(dirs):
    art, tex = dirs
    lib.edit(art, tex, "fav", "ai", "gen_0001_a.png")
    lib.edit(art, tex, "pick", "art", "20260101-000000-000_bird_ab12.png")
    lib.edit(art, tex, "title", "art", "20260101-000000-000_bird_ab12.png", title="Bird  of\nparadise")
    by = {i["id"]: i for i in lib.listing(art, tex)}
    assert by["ai:gen_0001_a.png"]["fav"] and by["art:20260101-000000-000_bird_ab12.png"]["picked"]
    assert by["art:20260101-000000-000_bird_ab12.png"]["title"] == "Bird of paradise"
    assert json.loads((art / lib.META_NAME).read_text())["version"] == 1
    assert not list(art.glob("*.tmp")) and not list(art.glob(".*.tmp"))
    lib.edit(art, tex, "unfav", "ai", "gen_0001_a.png")
    assert not {i["id"]: i for i in lib.listing(art, tex)}["ai:gen_0001_a.png"]["fav"]


@pytest.mark.parametrize("kind,name", [("ai", "../x.png"), ("art", "..\\x.png"), ("ai", "C:x.png"), ("ai", "nope.png"),
                                       ("bogus", "gen_0001_a.png"), ("ai", "gen_0001_a.depth.png"), ("art", ".library.json")])
def test_names_are_path_safe_and_must_exist(dirs, kind, name):
    art, tex = dirs
    with pytest.raises(ValueError):
        lib.edit(art, tex, "fav", kind, name)
    if kind == "ai":
        with pytest.raises(ValueError):
            lib.import_ai(tex, art, name)


def test_broken_meta_file_is_an_empty_library_not_a_crash(dirs):
    art, tex = dirs
    (art / lib.META_NAME).write_text("{nope")
    assert lib.listing(art, tex)
    lib.edit(art, tex, "fav", "ai", "gen_0001_a.png")                   # and the next write repairs it
    assert json.loads((art / lib.META_NAME).read_text())["favs"] == ["ai:gen_0001_a.png"]


# ---------------------------------------------------------------- AI image -> art library

def test_import_ai_copies_with_prompt_title_and_dedupes(dirs):
    art, tex = dirs
    n0 = len(lib.list_art(art))
    name = lib.import_ai(tex, art, "gen_0001_a.png")
    assert name in lib.list_art(art) and lib.list_art(art)[-1] == name and len(lib.list_art(art)) == n0 + 1
    assert "jaguar" in name                                              # the prompt names the file
    assert {i["id"]: i for i in lib.listing(art, tex)}["art:" + name]["title"] == "a glowing jaguar in the vines"
    assert lib.import_ai(tex, art, "gen_0001_a.png") == name             # the same image twice = the same piece
    assert len(lib.list_art(art)) == n0 + 1
    with Image.open(art / name) as im:
        assert im.size == (64, 48) and im.format == "PNG"
    assert lib.import_ai(tex, art, "gen_0001_a.png", title="Jaguar") == name   # explicit title updates it
    assert {i["id"]: i for i in lib.listing(art, tex)}["art:" + name]["title"] == "Jaguar"


def test_import_respects_the_art_library_quota(dirs, monkeypatch):
    art, tex = dirs
    monkeypatch.setattr(lib, "ART_CAP_FILES", len(lib.list_art(art)))
    with pytest.raises(ValueError, match="full"):
        lib.import_ai(tex, art, "gen_0001_a.png")
    monkeypatch.setattr(lib, "ART_CAP_FILES", 999)
    monkeypatch.setattr(lib, "ART_CAP_BYTES", 10)
    with pytest.raises(ValueError, match="full"):
        lib.check_quota(art, 100)


def test_delete_either_kind_and_forget_its_flags(dirs):
    art, tex = dirs
    lib.edit(art, tex, "fav", "ai", "gen_dream_0002_eyes.png")
    lib.edit(art, tex, "pick", "ai", "gen_dream_0002_eyes.png")
    assert lib.delete(art, tex, "ai", "gen_dream_0002_eyes.png")
    assert not (tex / "gen_dream_0002_eyes.png").exists()
    meta = lib.load_meta(art)
    assert "ai:gen_dream_0002_eyes.png" not in meta["favs"] + meta["picked"]
    name = lib.import_ai(tex, art, "gen_0001_a.png")
    lib.forget(art, "art", name)
    assert "art:" + name not in lib.load_meta(art)["titles"]
    assert lib.import_ai(tex, art, "gen_0001_a.png") == name             # the file is still there: same piece


# ---------------------------------------------------------------- the AI-texture layer's image set

def test_layer_sets_ai_art_both_and_picked(dirs):
    art, tex = dirs
    ai = [k for k, _ in lib.layer_files(0, tex, art)]
    assert ai == sorted(["gen_0001_a.png", "gen_dream_0002_eyes.png", "seed_noise.png"])   # 0 = textures/ as before
    arts = [k for k, _ in lib.layer_files(1, tex, art)]
    assert arts and all(k.startswith("art/") for k in arts)
    assert set(k for k, _ in lib.layer_files(2, tex, art)) == set(ai) | set(arts)
    assert [k for k, _ in lib.layer_files(3, tex, art)] == ai                # nothing picked yet: AI images
    lib.edit(art, tex, "pick", "art", "20260101-000000-000_bird_ab12.png")
    lib.edit(art, tex, "pick", "ai", "seed_noise.png")
    picked = lib.load_meta(art)["picked"]
    assert sorted(k for k, _ in lib.layer_files(3, tex, art, picked)) == ["art/20260101-000000-000_bird_ab12.png", "seed_noise.png"]
    extra = [k for k, _ in lib.layer_files(0, tex, art, extra=("art/20260101-000000-000_bird_ab12.png",))]
    assert "art/20260101-000000-000_bird_ab12.png" in extra                 # a show request loads it anyway
    for k, p in lib.layer_files(2, tex, art):
        assert p.is_file() and p.resolve().parent in (art.resolve(), tex.resolve())
    assert lib.layer_files(1, tex, art / "empty") == lib.layer_files(0, tex, art / "empty")   # no art: AI stands in


def test_layer_key_roundtrip():
    assert lib.split_key("art/x.png") == ("art", "x.png") and lib.split_key("gen_1.png") == ("ai", "gen_1.png")
    assert lib.layer_key("art", "x.png") == "art/x.png" and lib.layer_key("ai", "gen_1.png") == "gen_1.png"


def test_img_set_param_is_appended_and_classified():
    import params
    names = list(params.PARAMS)
    assert names.index("img_set") > names.index("art_mix_melt")
    assert params.PARAMS["img_set"][:3] == (0.0, 3.0, 0.0)
    assert len(params.OPTIONS["img_set"]) == 4 and "img_set" in params.BYPASS_KEEP
    import live_visuals as lv
    assert "img_set" in lv.CPU_ONLY


# ---------------------------------------------------------------- commands

def test_art_pick_accepts_any_and_show_texture_accepts_art_keys():
    import remote_control as rc
    import control
    assert rc.validate_command({"cmd": "art_pick", "source": "any", "name": "gen_0001_a.png"})["source"] == "any"
    assert rc.validate_command({"cmd": "show_texture", "name": "art/bird.png"})["name"] == "art/bird.png"
    for bad in ("art/../x.png", "art/a/b.png", "../x.png", "art\\x.png"):
        with pytest.raises(ValueError):
            rc.validate_command({"cmd": "show_texture", "name": bad})


def test_dream_from_any_library_image():
    import control
    import ai_worker
    out = control.validate({"cmd": "gen", "prompt": "more eyes", "init": {"kind": "ai", "name": "gen_0001_a.png"}})
    assert out["init"] == {"kind": "ai", "name": "gen_0001_a.png"} and 0.2 <= out["strength"] <= 0.9
    assert control.validate({"cmd": "gen", "prompt": "x", "init": {"kind": "art", "name": "b.png"}})["init"]["kind"] == "art"
    for bad in ({"kind": "ai", "name": "../x.png"}, {"kind": "disk", "name": "a.png"}, "C:/x.png"):
        with pytest.raises(ValueError):
            control.validate({"cmd": "gen", "prompt": "x", "init": bad})
    job = ai_worker.validate_job({"prompt": "x", "init": {"kind": "art", "name": "b.png"}})
    assert job["init"] == {"kind": "art", "name": "b.png"}
    assert ai_worker.validate_job({"prompt": "x", "init": "snapshot"})["init"] == "snapshot"   # unchanged


def test_worker_resolves_init_images_inside_their_folder(dirs):
    import ai_worker
    art, tex = dirs
    p = ai_worker.init_path({"kind": "ai", "name": "gen_0001_a.png"}, tex, art)
    assert p == tex / "gen_0001_a.png"
    assert ai_worker.init_path({"kind": "art", "name": "20260101-000000-000_bird_ab12.png"}, tex, art).parent == art
    with pytest.raises(FileNotFoundError):
        ai_worker.init_path({"kind": "ai", "name": "missing.png"}, tex, art)


# ---------------------------------------------------------------- panel server (stdlib http, no GL)

def test_library_endpoint(tmp_path):
    import urllib.error
    import urllib.request
    sys.path.insert(0, str(ROOT / "tests"))
    from test_remote import free_port
    from remote_control import RemoteControl
    art, tex = tmp_path / "art", tmp_path / "textures"
    _png(tex / "gen_0001_a.png", prompt="spirit jaguar", age=30)
    _png(art / "20260101-000000-000_bird_ab12.png", age=50)
    r = RemoteControl(port=free_port(), token="tok", quiet=True, tex_dir=tex, art_dir=art, ai_url="http://127.0.0.1:9",
                      token_paths=[tmp_path / ".none"])
    r.start()
    try:
        def post(body):
            req = urllib.request.Request(f"http://127.0.0.1:{r.port}/library?k=tok", data=json.dumps(body).encode())
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
        assert post({"op": "fav", "kind": "ai", "name": "../x.png"})[0] == 400
        assert post({"op": "fav", "kind": "ai", "name": "gen_0001_a.png"})[0] == 200
        code, out = post({"op": "import", "kind": "ai", "name": "gen_0001_a.png"})
        assert code == 200 and out["name"] in lib.list_art(art) and out["index"] == 1 and "jaguar" in out["name"]
        with urllib.request.urlopen(f"http://127.0.0.1:{r.port}/sources?k=tok", timeout=5) as resp:
            src = json.loads(resp.read())
        assert src["art"] and src["tex"] and {i["id"] for i in src["items"]} >= {"ai:gen_0001_a.png", "art:" + out["name"]}
        assert next(i for i in src["items"] if i["id"] == "ai:gen_0001_a.png")["fav"]
        assert post({"op": "delete", "kind": "ai", "name": "gen_0001_a.png"})[0] == 200 and not (tex / "gen_0001_a.png").exists()
        assert post({"op": "delete", "kind": "art", "name": out["name"]})[0] == 200
        assert post({"op": "nuke", "kind": "art", "name": "20260101-000000-000_bird_ab12.png"})[0] == 400
    finally:
        r.stop()


def test_panel_one_picker_with_filters_and_the_same_actions_everywhere():
    html = (ROOT / "panel.html").read_text(encoding="utf-8")
    for s in ("SRC_FILTERS", "'DREAMED'", "function srcMenu(", "'USE AS ART'", "'IN THE AI LAYER'", "'SKIN'",
              "'SPRITE CREATURE'", "DREAM FROM", "DREAMIFY", "SAVE TO ART", "'DELETE'", "PICKED SET", "FAVOURITE",
              "fetch('library'", "window.__panel.dreamify=open", "PN.skinTo=function", "mkTiles('img_set'",
              "init:{kind:it.kind,name:it.name}", "'art/'+it.name"):
        assert s in html, s


@pytest.mark.gl
def test_ai_layer_shows_art_uploads_when_the_set_says_so(tmp_path):        # GL: needs .locks/gpu_suite
    import art_layer
    sys.path.insert(0, str(ROOT / "tests"))
    from test_art_mix import _batch, _img
    import numpy as np
    art = tmp_path / "art"
    art_layer.write_test_art(art)
    old = time.time() - 60
    import os
    for f in art.iterdir():
        os.utime(f, (old, old))                                             # "finished" files (older than 2 s)
    jobs = {"ai": dict(sets=dict(l_ai=1.0, img_set=0.0), frames=240), "art": dict(sets=dict(l_ai=1.0, img_set=1.0), frames=240)}
    res = _batch(tmp_path / "r", jobs, "--render", "640x400", "--art", art)
    d = np.abs(_img(res["ai"]["path"]) - _img(res["art"]["path"])).mean() / 255 * 100
    assert res["art"]["nonfinite"] == 0 and d > 1.0, f"img_set 1 shows the same picture as img_set 0 ({d:.2f} %)"
