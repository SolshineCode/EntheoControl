"""Text layer (text_layer.py): param contract, SDF rasterization, fonts, command handling + phrase cycle,
persistence in live_state.json, off = zero cost, movement, trails, non-finite guard, engine hook + frame criteria.

One hidden GLFW window for the module (never visible on the projector).
"""
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import control  # noqa: E402
import params  # noqa: E402
import remote_control  # noqa: E402
import text_layer as tl  # noqa: E402

OUT = (640, 400)
FONT = tl.resolve_font("Arial Bold")


@pytest.fixture(scope="module")
def ctx():
    c, _win = tl.hidden_context()
    yield c


@pytest.fixture()
def tp(ctx):
    p = tl.TextPass(ctx, OUT)
    yield p
    p.release()


def field(tp, t=1.0, dt=1 / 60, bands=(0.5, 0.3, 0.2), frames=1, **over):
    vals = tl.default_values(**over)
    for i in range(frames):
        tp.render(vals, tl.make_clock(t + i * dt), t + i * dt, dt, bands, tl.GALLERY_PALETTE)
    return tl.read_field(tp)


def settle(tp, **over):
    """Past the fade-in of new words."""
    return field(tp, frames=14, **over)


def ink(f):
    return (f[..., 0] > 0.0) & (f[..., 2] > 0.5)


# ---------------------------------------------------------------- contract


def test_params_appended_not_reordered():
    names = list(params.PARAMS)
    assert names[:5] == ["master", "flash", "strobe", "invert", "freeze"]
    assert names.index("art_glow") == 71, "the original params must keep their positions (MIDI CC order)"
    i = names.index("l_text")
    assert names[i:i + len(tl.PARAM_NAMES)] == tl.PARAM_NAMES, "the text params are one appended block"
    assert names.index("being_mix") < i, "appended after the blocks that were there before"
    for n in tl.PARAM_NAMES:
        lo, hi, d, group, label = params.PARAMS[n]
        assert lo <= d <= hi and label and group == ("layers" if n == "l_text" else "text")
    assert params.PARAMS["l_text"][2] == 0.0, "off by default: the current look must not change"
    assert params.PARAMS["text_mode"][:3] == (0.0, 7.0, 0.0)


def test_named_dropdowns():
    for name, labels in (("text_mode", tl.MODES), ("text_color", tl.COLOR_MODES), ("text_move", tl.MOVES)):
        lo, hi = params.PARAMS[name][:2]
        assert params.OPTIONS[name] == labels and len(labels) == int(hi - lo) + 1
        assert control.OPTIONS[name] == labels and name in control.DISCRETE
    meta = {p["name"]: p for p in control.meta()["params"]}
    assert meta["text_mode"]["options"] == tl.MODES and meta["text_mode"]["stepped"]
    assert "text" in control.meta()["groups"]


def test_shader_and_cpu_split_matches_the_engine():
    import live_visuals as lv
    assert tl.CPU_ONLY_PARAMS <= lv.CPU_ONLY
    for n in tl.GLSL_PARAMS:
        assert n not in lv.CPU_ONLY and re.search(rf"\bp_{n}\b", lv.SCENE_FS)
    assert "//@TEXT_GLSL" not in lv.SCENE_FS and "textMix(o, cov" in lv.SCENE_FS and "textWarp(pscr" in lv.SCENE_FS
    assert "texT" in lv.SCENE_FS and "texTT" in lv.SCENE_FS


# ---------------------------------------------------------------- SDF rasterization


def test_distance_transform_is_exact():
    m = np.zeros((48, 52), bool)
    m[18:32, 20:27] = True
    m[6, 40] = True
    sd = tl.sdf_from_mask(m, ss=1, radius=10)
    yy, xx = np.mgrid[0:48, 0:52]
    ins, out = np.argwhere(m), np.argwhere(~m)
    d_out = np.sqrt(((yy[..., None] - ins[:, 0]) ** 2 + (xx[..., None] - ins[:, 1]) ** 2).min(-1))
    d_in = np.sqrt(((yy[..., None] - out[:, 0]) ** 2 + (xx[..., None] - out[:, 1]) ** 2).min(-1))
    ref = np.clip(np.where(m, d_in - 0.5, 0.5 - d_out), -10, 10)
    assert np.abs(sd - ref).max() < 1e-3
    assert tl.sdf_from_mask(np.zeros((9, 9), bool), ss=3, radius=6).max() < 0      # blank glyph: all outside


def test_supersampled_sdf_grid():
    m = np.zeros((30, 60), bool)
    m[9:21, 12:48] = True
    sd = tl.sdf_from_mask(m, ss=3, radius=9)
    assert sd.shape == (10, 20)
    assert sd[5, 10] > 1.0 and sd[0, 0] == pytest.approx(-3.0, abs=0.01)           # output-pixel units, clipped at radius/ss
    assert np.isfinite(sd).all()


def test_glyph_sdf_shape_and_cache():
    g = tl.glyph_sdf(FONT["path"], "O")
    sd = g["sdf"].astype(np.float32)
    h, w = sd.shape
    assert sd.dtype == np.float32 and g["sdf"].dtype == np.float16
    assert abs(g["w"] - w / tl.EM) < 1e-6 and abs(g["h"] - h / tl.EM) < 1e-6
    assert sd.max() > 0.03, "the stroke of the O is inside (positive)"
    assert sd[h // 2, w // 2] < -0.05, "the hole of the O is outside (negative)"
    assert sd[0, 0] == pytest.approx(-tl.PAD, abs=0.004) and sd.min() >= -tl.PAD - 1e-3
    assert abs(float(np.abs(np.diff(sd, axis=1))[h // 2].max()) - 1.0 / tl.EM) < 0.5 / tl.EM, "unit gradient: one em per EM pixels"
    n = tl.RASTER_COUNT
    assert tl.glyph_sdf(FONT["path"], "O") is g and tl.RASTER_COUNT == n, "cached per (font, character)"
    assert tl.glyph_sdf(FONT["path"], " ") is None


def test_layout_lines_kerning_and_atlas():
    lay = tl.build_layout("AV To\nxy", FONT["path"], FONT["name"])
    assert lay.n == 6 and lay.n_lines == 2 and list(lay.line) == [0, 0, 0, 0, 1, 1]
    assert list(lay.pos) == [0, 1, 3, 4, 0, 1], "letter slots count the space"
    assert np.all(np.diff(lay.x[:4]) > 0) and lay.widths[0] > lay.widths[1] > 0
    a, v = tl.build_layout("A", FONT["path"]).adv[0], tl.build_layout("V", FONT["path"]).adv[0]
    assert lay.x[1] < a - 1e-3, "AV is kerned tighter than the plain advance"
    assert lay.atlas.dtype == np.float16 and lay.atlas.shape[1] == tl.ATLAS_W and np.isfinite(lay.atlas).all()
    assert lay.atlas.max() > 0.02 and lay.atlas.min() >= -tl.PAD - 1e-3
    assert (lay.uv >= 0).all() and (lay.uv <= 1).all() and v > 0


# ---------------------------------------------------------------- fonts


def test_font_list_and_fallback():
    fonts = tl.list_fonts()
    assert len(fonts) > 10 and all(set(f) == {"name", "path", "user"} for f in fonts)
    names = [f["name"] for f in fonts]
    assert len(set(n.casefold() for n in names)) == len(names) and "Arial" in names
    assert tl.resolve_font("arial")["name"] == "Arial", "names are case-insensitive"
    assert tl.resolve_font("no such font 123")["name"] in tl.DEFAULT_FONTS
    assert tl.resolve_font(None)["name"] in tl.DEFAULT_FONTS


def test_user_font_folder(tmp_path):
    shutil.copy(FONT["path"], tmp_path / "my_font.ttf")
    (tmp_path / "broken.ttf").write_bytes(b"not a font")
    (tmp_path / "notes.txt").write_text("x")
    fonts = tl.list_fonts(dirs=[(tmp_path, True)])
    assert [(f["name"], f["user"]) for f in fonts] == [("Arial Bold", True)]
    assert (ROOT / "fonts").is_dir() and tl.font_dirs()[0] == (ROOT / "fonts", True)


def test_font_preview_png():
    data = tl.font_preview_png("Arial")
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and 100 < len(data) < 20000
    assert tl.font_preview_png("Arial") is data, "cached"
    assert tl.font_preview_png("no such font 123") is None
    assert tl.font_preview_png("../../etc/passwd") is None


# ---------------------------------------------------------------- commands


def test_command_validation():
    v = control.validate
    assert v({"cmd": "text", "text": "  hi\r\nthere\t!\x00\x07  \n\n", "font": " Impact "}) == {
        "cmd": "text", "text": "  hi\nthere !", "font": "Impact"}
    assert v({"cmd": "text", "align": "left"}) == {"cmd": "text", "align": "left"}
    assert len(v({"cmd": "text", "text": "x" * 5000})["text"]) == tl.MAX_TEXT
    assert v({"cmd": "text", "text": "\n".join("abc" for _ in range(40))})["text"].count("\n") == tl.MAX_LINES - 1
    assert v({"cmd": "text", "text": ""})["text"] == "", "empty text clears the words"
    for bad in ({"cmd": "text"}, {"cmd": "text", "text": 5}, {"cmd": "text", "font": "x" * 200}, {"cmd": "text", "font": ["a"]},
                {"cmd": "text", "align": "up"}, {"cmd": "text_phrases", "phrases": "abc"},
                {"cmd": "text_phrases", "phrases": ["ok", 3]}, {"cmd": "text_phrases", "phrases": ["a"] * 99},
                {"cmd": "text_phrases", "phrases": ["  "]}, {"cmd": "text_del", "i": -1}, {"cmd": "text_del", "i": 1.5},
                {"cmd": "text_go"}, {"cmd": "text_go", "d": 5}, {"cmd": "text_go", "i": True},
                {"cmd": "text_cycle"}, {"cmd": "text_cycle", "every": "week"}, {"cmd": "text_cycle", "on": 1},
                {"cmd": "text_add", "text": float("nan")}):
        with pytest.raises(ValueError):
            v(bad)
    assert tl.TEXT_CMDS <= control.CONTROL_CMDS
    # the panel's HTTP validator lets them through (and only them)
    assert remote_control.validate_command({"type": "text", "text": "om"}) == {"cmd": "text", "text": "om"}
    with pytest.raises(ValueError):
        remote_control.validate_command({"cmd": "text", "text": None})


def test_state_phrases_and_stepping():
    s = tl.TextState(seed=1)
    assert s.pop_show() is None
    s.do(control.validate({"cmd": "text", "text": "one", "font": "Impact"}))
    assert s.pop_show() == dict(text="one", font="Impact", align="center", warm=[]) and s.pop_show() is None
    s.do({"cmd": "text_add"})
    s.do({"cmd": "text_add", "text": "two"})
    s.do({"cmd": "text_add", "text": "three"})
    assert s.phrases == ["one", "two", "three"] and s.idx == 0
    s.do({"cmd": "text_go", "d": 1})
    assert (s.text, s.idx) == ("two", 1) and s.pop_show()["text"] == "two"
    s.do({"cmd": "text_go", "d": -1}); s.do({"cmd": "text_go", "d": -1})
    assert (s.text, s.idx) == ("three", 2)
    s.do({"cmd": "text_go", "i": 9})
    assert s.idx == 2, "an index past the list is ignored"
    s.do({"cmd": "text_del", "i": 0})
    assert s.phrases == ["two", "three"] and s.idx == 1 and s.text == "three"
    s.do({"cmd": "text", "text": "free words"})
    assert s.idx == -1 and s.state()["phrases"] == ["two", "three"]
    s.do({"cmd": "text_phrases", "phrases": ["free words", "b"]})
    assert s.idx == 0


def test_phrase_cycle_on_the_beat_and_bar():
    s = tl.TextState(seed=3)
    s.do({"cmd": "text_phrases", "phrases": ["a", "b", "c"]})
    s.do({"cmd": "text", "text": "a"})
    s.do({"cmd": "text_cycle", "on": True, "every": "bar"})
    seen = []
    for i in range(0, 12 * 40 + 1):                    # 12 beats in 1/40 beat steps
        s.tick(i / 40)
        seen.append(s.text)
    assert [seen[int(b * 40)] for b in (0, 3.9, 4.0, 7.9, 8.0, 12.0)] == ["a", "a", "b", "b", "c", "a"]
    s.do({"cmd": "text_cycle", "every": "beat"})
    s.tick(100.0); s.tick(100.5)
    before = s.text
    s.tick(101.0)
    assert s.text != before
    s.do({"cmd": "text_cycle", "on": False})
    hold = s.text
    for b in range(102, 140):
        s.tick(float(b))
    assert s.text == hold, "cycle off: the words stay"
    s.tick(float("nan")); s.tick(None)                 # a broken clock never raises


def test_random_cycle_never_repeats():
    s = tl.TextState(seed=5)
    s.do({"cmd": "text_phrases", "phrases": ["a", "b", "c", "d"]})
    s.do({"cmd": "text_cycle", "on": True, "every": "beat", "order": "random"})
    seq = []
    for b in range(60):
        s.tick(float(b))
        seq.append(s.text)
    assert set(seq) == {"a", "b", "c", "d"} and all(x != y for x, y in zip(seq, seq[1:]))


def test_cycle_starts_the_list_when_free_text_is_up():
    s = tl.TextState()
    s.do({"cmd": "text", "text": "typed"})
    s.do({"cmd": "text_phrases", "phrases": ["a", "b"]})
    s.do({"cmd": "text_cycle", "on": True, "every": "beat"})
    s.tick(7.2)
    assert s.text == "a"
    s.tick(8.01)
    assert s.text == "b"


# ---------------------------------------------------------------- controller: engine command + persistence


def make_controller(d, restore=True):
    return control.Controller(d, restore=restore, midi=False, ai=False)


def test_controller_raises_text_show_and_persists(tmp_path):
    c = make_controller(tmp_path)
    rest = c.apply([{"cmd": "text", "text": "LOVE\nIS", "font": "Impact"}, {"cmd": "text_add"},
                    {"cmd": "text_add", "text": "ALL IS ONE"}, {"cmd": "text_cycle", "on": True, "every": "bar2"},
                    {"cmd": "set", "name": "text_mode", "v": 3}, {"cmd": "set", "name": "l_text", "v": 0.8}])
    assert rest == [], "text commands are the controller's"
    c.update(0.01, [0, 0, 0])
    out = c.apply([])
    assert out == [dict(cmd="text_show", type="text_show", text="LOVE\nIS", font="Impact", align="center",
                        warm=["LOVE\nIS", "ALL IS ONE"])]
    c.update(0.01, [0, 0, 0])
    assert c.apply([]) == [], "raised once per change"
    st = c.state()["text"]
    assert st == dict(text="LOVE\nIS", font="Impact", align="center", phrases=["LOVE\nIS", "ALL IS ONE"], idx=0,
                      cycle=dict(on=True, every="bar2", order="next"))
    json.dumps(c.state())
    assert c._autosave()
    saved = json.loads((tmp_path / "live_state.json").read_text(encoding="utf-8"))
    assert saved["settings"]["text"] == st and saved["base"]["text_mode"] == 3.0
    c.close()

    r = make_controller(tmp_path)
    assert r.state()["text"] == st and r.base["l_text"] == pytest.approx(0.8)
    r.update(0.01, [0, 0, 0])
    assert [x["cmd"] for x in r.apply([])] == ["text_show"], "the engine is told the restored words on the first frame"
    r.close()


def test_controller_cycles_phrases_with_its_clock(tmp_path):
    c = make_controller(tmp_path, restore=False)
    c.apply([{"cmd": "text_phrases", "phrases": ["a", "b", "c"]}, {"cmd": "text_cycle", "on": True, "every": "beat"},
             {"cmd": "bpm", "v": 120}])
    shown = []
    for _ in range(int(2.3 / 0.02)):                   # 2.3 s at 120 bpm = 4.6 beats
        c.update(0.02, [0, 0, 0])
        shown += [x["text"] for x in c.apply([]) if x.get("cmd") == "text_show"]
    assert shown[:5] == ["a", "b", "c", "a", "b"]
    c.close()


@pytest.mark.parametrize("junk", [None, 5, "x", [], {"text": 7, "phrases": "no", "cycle": 3, "font": ["a"]},
                                  {"text": "ok", "phrases": [1, 2], "cycle": {"on": "yes", "every": "never"}, "align": "x"}])
def test_restore_tolerates_garbage(tmp_path, junk):
    (tmp_path / "live_state.json").write_text(json.dumps(dict(version=1, base={}, mods={}, settings=dict(text=junk))))
    c = make_controller(tmp_path)
    st = c.state()["text"]
    assert st["phrases"] == [] and st["cycle"] == dict(on=False, every="bar", order="next") and st["align"] == "center"
    assert st["text"] == ("ok" if isinstance(junk, dict) and junk.get("text") == "ok" else "")
    c.update(0.01, [0, 0, 0])
    c.close()


def test_headless_controller_never_writes(tmp_path):
    c = make_controller(tmp_path, restore=False)
    c.apply([{"cmd": "text", "text": "x"}])
    for _ in range(5):
        c.update(0.5, [0, 0, 0])
    c.close()
    assert not (tmp_path / "live_state.json").exists()


# ---------------------------------------------------------------- the pass


def test_off_costs_nothing(tp):
    vals = tl.default_values(l_text=0.0)
    for i in range(5):
        tex = tp.render(vals, tl.make_clock(i / 60), i / 60, 1 / 60, (0.5, 0.3, 0.2), tl.GALLERY_PALETTE)
    assert tp.draws == 0 and tp.trail_draws == 0 and tp.uploads == 0 and tp._thread is None
    assert tex is tp._tex and tex.size == OUT
    f = tl.read_field(tp)
    assert np.allclose(f[..., 0], -tl.PAD) and not f[..., 2].any()
    tp.set_text("READY", FONT["name"], wait=True)
    tp.render(vals, tl.make_clock(1), 1.0, 1 / 60, (0, 0, 0), None)
    assert tp.draws == 0 and tp.uploads == 0, "words waiting, layer down: still nothing uploaded or drawn"
    settle(tp)
    assert tp.draws > 0 and tp.uploads == 1
    f = tl.read_field(tp, )
    tp.render(vals, tl.make_clock(2), 2.0, 1 / 60, (0, 0, 0), None)
    assert not ink(tl.read_field(tp)).any(), "fading the layer out clears the field"


def test_field_is_a_distance_field_with_letter_index(tp):
    tp.set_text("HI", FONT["name"], wait=True)
    f = settle(tp, text_size=0.8)
    m = ink(f)
    assert 0.01 < m.mean() < 0.4
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - OUT[0] / 2) < OUT[0] * 0.04 and abs(ys.mean() - OUT[1] / 2) < OUT[1] * 0.06, "centred"
    assert np.isfinite(f).all() and f[..., 0].min() == pytest.approx(-tl.PAD, abs=0.004) and f[..., 0].max() < tl.PAD + 0.01
    ids = f[..., 1]
    assert set(np.unique(np.round(ids[m], 3))) == {0.0, 1.0}
    assert ids[:, xs.min() + 2][m[:, xs.min() + 2]].max() < 0.01 and ids[:, xs.max() - 2][m[:, xs.max() - 2]].min() > 0.99, \
        "letter index 0 for H, 1 for I"
    em = f[..., 3][m]
    assert em.min() == pytest.approx(em.max(), rel=0.02) and em.mean() == pytest.approx(tl.em_size(0.8), rel=0.25)
    # the gradient of the distance is one em per (em height in pixels): the edge is crisp at any size
    row = f[OUT[1] // 2, :, 0]
    edge = np.abs(np.diff(row))[(row[:-1] > -0.1) & (row[:-1] < 0.05)]
    assert edge.max() == pytest.approx(1.0 / (em.mean() * OUT[1]), rel=0.3)


def test_rasterizes_only_when_words_or_font_change(tp):
    tp.set_text("ONE", FONT["name"], wait=True)
    settle(tp)
    n = tl.RASTER_COUNT
    for i in range(30):
        field(tp, t=2 + i / 60, text_size=0.2 + i / 60, text_track=i / 60, text_move=float(i % 7))
    assert tp.uploads == 1 and tl.RASTER_COUNT == n, "faders never re-rasterize"
    tp.set_text("ONE", FONT["name"], wait=True)
    field(tp)
    assert tp.uploads == 1, "the same words again: nothing to do"
    tp.set_text("NEO", FONT["name"], wait=True)
    field(tp)
    assert tp.uploads == 2 and tl.RASTER_COUNT == n, "same letters: straight from the glyph cache"
    tp.set_text(None, "Georgia", wait=True)
    field(tp)
    assert tp.uploads == 3 and tp.text == "NEO" and tp.font_name == "Georgia" and tl.RASTER_COUNT > n


def test_set_text_does_not_block_the_render_thread(tp):
    import time
    tl._GLYPHS.clear()
    t0 = time.perf_counter()
    tp.set_text("Sphinx of black quartz, judge my vow 0123456789", FONT["name"])
    dt = time.perf_counter() - t0
    assert dt < 0.05, f"set_text took {dt * 1000:.0f} ms on the caller's thread"
    assert tp.set_text("Sphinx of black quartz, judge my vow 0123456789", FONT["name"], wait=True)
    assert ink(settle(tp)).any()


def test_empty_and_whitespace_words(tp):
    tp.set_text("WORDS", FONT["name"], wait=True)
    assert ink(settle(tp)).any()
    for blank in ("", "   \n  "):
        tp.set_text(blank, wait=True)
        assert not ink(field(tp)).any()
    tp.set_text("☃\U0001F600 ok", "no such font", wait=True)       # glyphs the font may not have: never raises
    field(tp)


def test_size_position_tracking_lines(tp):
    tp.set_text("OM\nOM", FONT["name"], wait=True)
    small, big = ink(settle(tp, text_size=0.3)), ink(field(tp, text_size=0.7))
    assert big.sum() > small.sum() * 3

    def box(m):
        ys, xs = np.nonzero(m)
        return xs.min(), xs.max(), ys.min(), ys.max(), xs.mean(), ys.mean()

    b0 = box(ink(field(tp, text_size=0.5)))
    bx = box(ink(field(tp, text_size=0.5, text_x=0.5)))
    by = box(ink(field(tp, text_size=0.5, text_y=0.5)))
    assert bx[4] - b0[4] == pytest.approx(OUT[0] * 0.25, abs=4) and abs(bx[5] - b0[5]) < 2
    assert b0[5] - by[5] == pytest.approx(OUT[1] * 0.25, abs=4), "+y is up"
    bt = box(ink(field(tp, text_size=0.5, text_track=0.6)))
    assert (bt[1] - bt[0]) > (b0[1] - b0[0]) * 1.25 and abs(bt[4] - b0[4]) < 8, "tracking widens, stays centred"
    bl = box(ink(field(tp, text_size=0.5, text_line=2.0)))
    assert (bl[3] - bl[2]) > (b0[3] - b0[2]) * 1.3
    tilt = ink(field(tp, text_size=0.5, text_tilt=0.5))
    bt2 = box(tilt)
    assert abs((bt2[1] - bt2[0]) - (b0[3] - b0[2])) < 6 and abs((bt2[3] - bt2[2]) - (b0[1] - b0[0])) < 6, \
        "tilt 0.5 = 90 degrees: width and height swap"


def test_long_line_is_fitted_to_the_frame(tp):
    tp.set_text("A VERY LONG LINE OF WORDS THAT WOULD RUN OFF", FONT["name"], wait=True)
    m = ink(settle(tp, text_size=0.9))
    xs = np.nonzero(m)[1]
    assert xs.min() > 2 and xs.max() < OUT[0] - 3


def test_alignment(tp):
    def left_edges(align):
        tp.set_text("WWWWWW\nI", FONT["name"], align, wait=True)
        m = ink(settle(tp, text_size=0.5))
        top, bot = m[: OUT[1] // 2], m[OUT[1] // 2:]
        return np.nonzero(top)[1].min(), np.nonzero(top)[1].max(), np.nonzero(bot)[1].min(), np.nonzero(bot)[1].max()

    l = left_edges("left"); c = left_edges("center"); r = left_edges("right")
    assert abs(l[2] - l[0]) < 12 and abs(r[3] - r[1]) < 12
    assert c[2] > c[0] + 20 and c[3] < c[1] - 20


@pytest.mark.parametrize("move", range(7))
def test_movement_modes(tp, move):
    tp.set_text("WAVES OF LIGHT", FONT["name"], wait=True)
    settle(tp, text_move=float(move), text_size=0.45)
    pulse = 0.3 if move == 6 else 0.0                 # (the audio pulse is tested on its own)
    a = ink(field(tp, t=3.0, text_move=float(move), text_size=0.45, text_pulse=pulse))
    b = ink(field(tp, t=3.0, frames=40, text_move=float(move), text_size=0.45, text_pulse=pulse))
    assert a.any() and b.any(), "always something on screen"
    moved = (a ^ b).sum() / max(a.sum(), 1)
    if move == 0:
        assert moved < 0.02, "static: holds still"
    else:
        assert moved > 0.10, f"{tl.MOVES[move]} does not move"
    still = ink(field(tp, t=5.0, frames=20, text_move=float(move), text_speed=0.0, text_pulse=0.0, text_size=0.45))
    still2 = ink(field(tp, t=5.5, frames=20, text_move=float(move), text_speed=0.0, text_pulse=0.0, text_size=0.45))
    if move != 6:                                      # bounce follows the beat, not the speed
        assert (still ^ still2).sum() / max(still.sum(), 1) < 0.02, "speed 0 = it stops"


def test_spiral_letters_shrink_toward_the_centre(tp):
    tp.set_text("INTO THE TUNNEL WE GO", FONT["name"], wait=True)
    settle(tp, text_move=5.0)
    inst = tp.instances_for(tl.default_values(text_move=5.0), tl.make_clock(1.0), 1 / 60, (0, 0, 0))
    r = np.hypot(inst[:, 0], inst[:, 1])
    assert len(inst) > 8 and np.corrcoef(r, inst[:, 2])[0, 1] > 0.98, "size follows the radius"
    assert inst[:, 13].min() >= 0 and inst[:, 13].max() <= 1.0


def test_audio_pulse(tp):
    tp.set_text("BOOM", FONT["name"], wait=True)
    settle(tp)
    quiet = ink(field(tp, t=2.4, bands=(0.0, 0, 0), text_pulse=1.0)).sum()
    loud = ink(field(tp, t=2.4, bands=(1.0, 0, 0), text_pulse=1.0)).sum()
    flat = ink(field(tp, t=2.4, bands=(1.0, 0, 0), text_pulse=0.0)).sum()
    flat0 = ink(field(tp, t=2.4, bands=(0.0, 0, 0), text_pulse=0.0)).sum()
    assert loud > quiet * 1.12 and flat == flat0


def test_new_words_fade_in(tp):
    tp.set_text("A", FONT["name"], wait=True)
    settle(tp)
    tp.set_text("B", FONT["name"], wait=True)
    first = field(tp, t=9.0)
    assert 0.0 < first[..., 2].max() < 0.5, "the first frame of new words is faint"
    assert settle(tp)[..., 2].max() == pytest.approx(1.0, abs=0.01)


def test_trails_only_in_mode_7(tp):
    tp.set_text("ECHO", FONT["name"], wait=True)
    settle(tp, text_mode=0.0)
    assert tp.trail_draws == 0 and not tl.read_trail(tp)[..., 0].any()
    f = field(tp, frames=50, text_mode=7.0, text_move=1.0, text_speed=1.0, text_weave=1.0)
    tr = tl.read_trail(tp)[..., 0].astype(np.float32) / 255.0
    assert tp.trail_draws == 50 and np.isfinite(tr).all()
    now = ink(f)[::2, ::2]
    assert (tr[now] > 0.5).mean() > 0.85, "the letters themselves are at full energy"
    ghost = (tr > 0.08) & ~now
    assert ghost.sum() > now.sum() * 0.3, "a trail is left behind / around the moving letters"
    assert tr.max() <= 1.0 and (tr < 0.02).mean() > 0.4, "trails decay to black"
    field(tp, text_mode=0.0)
    assert not tl.read_trail(tp)[..., 0].any(), "leaving the mode clears the buffer"


def test_trails_decay_away(tp):
    tp.set_text("GONE", FONT["name"], wait=True)
    field(tp, frames=30, text_mode=7.0, text_weave=0.0)
    tp.set_text("", wait=True)
    field(tp, frames=120, text_mode=7.0, text_weave=0.0)
    assert tl.read_trail(tp)[..., 0].max() == 0


def test_nonfinite_guard(tp):
    tp.set_text("SAFE", FONT["name"], wait=True)
    settle(tp)
    bad = float("nan"), float("inf"), -float("inf"), None, "x", 1e30
    for i, b in enumerate(bad):
        vals = tl.default_values()
        for n in tl.PARAM_NAMES[1:]:
            vals[n] = b
        clock = dict(bpm=b, beat=b, beat_phase=b, bar_phase=b, phrase_phase=b)
        tp.render(vals, clock, b, b, (b, b, b), [(b, b, b)])
        f = tl.read_field(tp)
        assert np.isfinite(f).all(), f"non-finite field for {b!r}"
        inst = tp.instances_for(vals, clock, 0.0, (0, 0, 0))
        assert np.isfinite(inst).all()
    for mode in (7.0,):
        vals = tl.default_values(text_mode=mode, text_weave=float("nan"), text_speed=float("inf"))
        tp.render(vals, None, 1.0, None, None, None)
        assert np.isfinite(tl.read_trail(tp).astype(np.float32)).all()
    assert ink(settle(tp)).any(), "and it still draws afterwards"


def test_gl_state_is_restored(ctx, tp):
    import moderngl
    fbo = ctx.simple_framebuffer((32, 32))
    fbo.use()
    ctx.viewport = (0, 0, 32, 32)
    tp.set_text("GL", FONT["name"], wait=True)
    settle(tp, text_mode=7.0)
    assert ctx.fbo.glo == fbo.glo and ctx.viewport == (0, 0, 32, 32)
    gl = tp._gl
    if gl is not None:
        assert not gl.glIsEnabled(0x0B71), "depth test is left off"
    ctx.enable(moderngl.BLEND)
    field(tp)
    if gl is not None:
        assert gl.glIsEnabled(0x0BE2), "the caller's blend switch is put back"
    ctx.disable(moderngl.BLEND)
    fbo.release()


def test_pass_is_fast(ctx, tp):
    import time
    tp.set_text("LOVE IS THE WAY\nALL IS ONE", FONT["name"], wait=True)
    settle(tp)
    vals = tl.default_values(text_move=4.0, text_mode=7.0)
    ctx.finish()
    t0 = time.perf_counter()
    for i in range(120):
        tp.render(vals, tl.make_clock(i / 60), i / 60, 1 / 60, (0.5, 0.3, 0.2), tl.GALLERY_PALETTE)
    ctx.finish()
    ms = (time.perf_counter() - t0) / 120 * 1000
    assert ms < 4.0, f"text pass costs {ms:.2f} ms / frame"


# ---------------------------------------------------------------- engine hook: every mode, frame criteria


def _batch(out, jobs, *args, timeout=600):
    spec = out / "jobs.json"
    spec.write_text(json.dumps(jobs))
    r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--batch", str(spec), "--out", str(out), *map(str, args)],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
    assert "disabled" not in r.stdout and "render failed" not in r.stdout, r.stdout[-2000:]
    return {d["name"]: d for d in json.loads((out / "batch.json").read_text())}


def _img(out, name):
    from PIL import Image
    return np.asarray(Image.open(out / f"{name}.png").convert("RGB")).astype(np.float32) / 255.0


def test_engine_modes_and_frame_criteria(tmp_path):
    words = [["text", "LOVE IS\nTHE WAY", "Arial Black"]]
    jobs = [dict(name="base", sets={}, frames=70),
            dict(name="off", sets=dict(l_text=0.0, text_mode=3.0), frames=70, script={"0": words})]
    for m in range(8):
        sets = dict(l_text=1.0, text_mode=float(m), text_size=0.62)
        if m == 6:
            sets["text_size"] = 0.36
        jobs.append(dict(name=f"mode{m}", sets=sets, frames=70, script={"0": words}))
    jobs.append(dict(name="flat1", sets=dict(l_text=1.0, text_mode=1.0, symmetry=0.0, text_size=0.62), frames=70, script={"0": words}))
    jobs.append(dict(name="max", sets=dict(l_text=1.0, text_size=1.0, text_glow=1.0, text_outline=1.0, text_color=2.0,
                                           text_pulse=1.0), frames=70, script={"0": words}))
    done = _batch(tmp_path, jobs, "--render", "640x400")
    assert all(d["nonfinite"] == 0 for d in done.values())
    base, off = _img(tmp_path, "base"), _img(tmp_path, "off")
    assert np.abs(base - off).max() < 0.02, "l_text 0 leaves the picture untouched (whatever the words / mode)"
    imgs = {m: _img(tmp_path, f"mode{m}") for m in range(8)}
    for m, im in imgs.items():
        mx = im.max(axis=2)
        assert np.abs(im - base).mean() > 0.004, f"mode {m} ({tl.MODES[m]}) changes nothing"
        assert (mx < 8 / 255).mean() >= 0.25, f"mode {m}: only {(mx < 8 / 255).mean():.0%} true black"
        assert (im.min(axis=2) > 235 / 255).mean() < 0.02, f"mode {m}: blown out"
    for a in range(8):
        for b in range(a + 1, 8):
            assert np.abs(imgs[a] - imgs[b]).mean() > 0.004, f"modes {a} and {b} look the same"
    # what each mode is for
    lum = lambda im: im.max(axis=2)
    lit = lum(base) > 0.15
    assert lum(imgs[2])[lit].mean() < lum(base)[lit].mean() * 0.6, "stencil: most of the visual is masked away"
    assert (lum(imgs[2]) < 8 / 255).mean() > (lum(base) < 8 / 255).mean() + 0.1
    assert lum(imgs[3])[lit].mean() < lum(base)[lit].mean() * 0.97, "knockout removes light from the visual"
    assert lum(imgs[3]).mean() < lum(imgs[0]).mean()
    assert lum(imgs[0]).mean() > lum(base).mean(), "overlay adds light"
    big = _img(tmp_path, "max")
    assert (big.max(axis=2) < 8 / 255).mean() >= 0.25 and (big.min(axis=2) > 235 / 255).mean() < 0.03


def test_engine_keeps_its_speed_with_text_up(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "text_layer.py"), "--bench-engine", "--out", str(tmp_path)],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
    res = json.loads((tmp_path / "bench.json").read_text())
    assert res["text_overlay"] > 0 and res["base"] > 0
    assert res["text_overlay"] > 0.55 * res["base"], f"text halves the frame rate: {res}"


# ---------------------------------------------------------------- remote endpoints + the panel's TEXT tab


def _serve(tmp_path):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = remote_control.RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art",
                                      tex_dir=tmp_path / "tex")
    rc.start()
    return rc, port


def test_font_endpoints(tmp_path):
    import urllib.error
    import urllib.parse
    import urllib.request
    rc, port = _serve(tmp_path)
    try:
        base = f"http://127.0.0.1:{port}"
        d = json.loads(urllib.request.urlopen(f"{base}/fonts?k=tok", timeout=10).read())
        assert d["ok"] and len(d["fonts"]) > 10 and set(d["fonts"][0]) == {"name", "user"} and d["default"]
        r = urllib.request.urlopen(f"{base}/font-preview/{urllib.parse.quote('Arial Bold')}?k=tok", timeout=10)
        assert r.headers["Content-Type"] == "image/png" and r.read()[:4] == b"\x89PNG"
        for bad in ("/font-preview/nope?k=tok", "/font-preview/..%2F..%2Fx?k=tok"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + bad, timeout=10)
            assert e.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{base}/fonts", timeout=10)
        assert e.value.code == 403, "the token is required"
        req = urllib.request.Request(f"{base}/cmd?k=tok", data=json.dumps({"cmd": "text", "text": "hi"}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req, timeout=10).read())["ok"]
        assert rc.drain() == [{"cmd": "text", "text": "hi"}]
    finally:
        rc.stop()


@pytest.mark.parametrize("size", [(1600, 1000), (390, 844)])
def test_panel_text_tab(tmp_path, size):
    import threading
    import time
    pw_api = pytest.importorskip("playwright.sync_api")
    rc, port = _serve(tmp_path)
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
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
            page = br.new_page(viewport=dict(width=size[0], height=size[1]))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(f"http://127.0.0.1:{port}/?k=tok")
            page.wait_for_function("window.__panel && window.__panel.state().online", timeout=10000)
            # compact box on PERFORM: Enter sends, and typing words raises the layer
            page.fill("#pg-perform .txmini input", "OM")
            page.press("#pg-perform .txmini input", "Enter")
            page.wait_for_timeout(500)
            assert ctl.text.text == "OM" and ctl.base["l_text"] == 1.0
            page.click('#tabs button[data-tab="text"]')
            page.wait_for_selector("#pg-text .txf button img", timeout=10000)
            assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 2
            assert page.input_value("#pg-text textarea") == "OM", "the box follows the words on show"
            ta = page.locator("#pg-text textarea")
            ta.fill("")
            ta.type("LOVE")
            ta.press("Shift+Enter")
            ta.type("IS THE WAY")
            ta.press("Enter")
            page.wait_for_timeout(400)
            assert ctl.text.text == "LOVE\nIS THE WAY", "Shift+Enter = new line, Enter = send"
            page.click("#pg-text button:has-text('+ PHRASE')")
            ta.fill("ALL IS ONE")
            page.click("#pg-text button:has-text('+ PHRASE')")
            page.wait_for_timeout(500)
            assert ctl.text.phrases == ["LOVE\nIS THE WAY", "ALL IS ONE"]
            assert page.locator("#pg-text .txp").count() == 2
            page.click("#pg-text .txp >> nth=1 >> button.w")
            page.wait_for_timeout(400)
            assert ctl.text.text == "ALL IS ONE" and ctl.text.idx == 1
            page.click("#pg-text button:has-text('AUTO-CYCLE')")
            page.click("#pg-text button:text-is('BEAT')")
            page.wait_for_timeout(500)
            assert ctl.text.cycle == dict(on=True, every="beat", order="next")
            page.click("#pg-text button:has-text('AUTO-CYCLE')")
            page.click("#pg-text .txp >> nth=0 >> button.x")
            page.wait_for_timeout(400)
            assert ctl.text.phrases == ["ALL IS ONE"] and not ctl.text.cycle["on"]
            # font search + pick
            page.fill("#pg-text input[placeholder^='Search']", "impact")
            page.wait_for_timeout(200)
            assert page.locator("#pg-text .txf button").count() >= 1
            page.click("#pg-text .txf button >> nth=0")
            page.wait_for_timeout(400)
            assert ctl.text.font.lower() == "impact"
            assert page.evaluate("document.querySelector('#pg-text .txf img').naturalWidth") > 8, "font preview image loaded"
            # integration mode buttons are generated from OPTIONS; the colour picker sets a fixed colour
            page.click("#pg-text button.tile:has-text('Stencil mask')")
            page.evaluate("""() => { const p = document.querySelector('#pg-text input[type=color]');
                                     p.value = '#ff0000'; p.dispatchEvent(new Event('input', {bubbles: true})); }""")
            page.wait_for_timeout(500)
            assert ctl.base["text_mode"] == 2.0 and ctl.base["text_color"] == 1.0
            assert ctl.base["text_hue"] == pytest.approx(0.0, abs=0.01) and ctl.base["text_sat"] == pytest.approx(1.0, abs=0.01)
            for name in ("text_size", "text_track", "text_glow", "text_weave", "text_dissolve", "text_speed"):
                assert page.locator(f'#pg-text [data-mt="param:{name}"]').count() >= 1, name
            shot = ROOT / "test_frames" / "text"
            shot.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(shot / f"panel_text_tab_{size[0]}.png"), full_page=True)
            br.close()
    finally:
        stop.set()
        th.join(timeout=2)
        ctl.close()
        rc.stop()
    assert not errors, errors
