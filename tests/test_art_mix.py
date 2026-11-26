"""Combinable art modes (art_combine + art_mix_*): any mix of relief / stencil / kaleidoscope / tunnel / displace /
melt at once, while the old one-mode art_use selector keeps rendering exactly as before.
CPU tests run anywhere; the GL tests (marked gl) render with live_visuals --batch and need the GPU lock."""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import art_layer  # noqa: E402
import params  # noqa: E402

MODES = ("relief", "stencil", "kaleido", "tunnel", "displace", "melt")
MIX = tuple("art_mix_" + m for m in MODES)


# ---------------------------------------------------------------- params contract (CPU)

def test_mix_params_are_appended_after_the_old_art_params():
    names = list(params.PARAMS)
    for n in ("art_combine",) + MIX:
        assert n in names, n
        assert names.index(n) > names.index("art_source"), f"{n} must be appended, not inserted"
        assert params.PARAMS[n][3] == "art", n
    assert params.PARAMS["art_combine"][:3] == (0.0, 1.0, 0.0)
    for n in MIX:
        assert params.PARAMS[n][:3] == (0.0, 1.0, 0.0), n                  # off by default: old looks unchanged
    assert params.OPTIONS["art_combine"] and len(params.OPTIONS["art_combine"]) == 2
    assert art_layer.ART_MIX == MIX and art_layer.ART_MODES == MODES


def test_combine_is_stepped_and_the_amounts_are_faders():
    import control
    assert "art_combine" in control.DISCRETE
    for n in MIX:
        assert n not in control.DISCRETE and n in control.CONT, n           # MIDI / mod-able continuous faders


def test_bypass_classification():
    for n in MIX:
        assert params.NEUTRAL.get(n) == 0.0 and n not in params.BYPASS_KEEP, n
    assert "art_combine" in params.BYPASS_KEEP and "art_combine" not in params.NEUTRAL
    cover = params.bypass_cover()
    assert set(MIX) <= set(cover["art"]) and "l_art" in cover["art"]         # the ART switch drops every mode amount
    for n in MIX:
        assert cover[n] == (n,)                                              # and each mode has its own switch


def test_clips_lane_and_stepped():
    import clips
    assert "art_combine" in clips.STEPPED
    assert {clips.lane_of(n) for n in MIX} == {clips.lane_of("art_use")}


# ---------------------------------------------------------------- the weights the engine uses (CPU)

@pytest.mark.parametrize("use", range(6))
def test_selector_gives_one_hot_weights(use):
    w = art_layer.art_weights(dict(art_use=float(use), art_combine=0.0, art_mix_kaleido=0.7, art_mix_melt=1.0))
    assert w == tuple(1.0 if i == use else 0.0 for i in range(6))           # amounts are ignored while COMBINE is off


def test_combine_uses_the_amounts_clamped():
    w = art_layer.art_weights(dict(art_use=3.0, art_combine=1.0, art_mix_stencil=1.0, art_mix_kaleido=0.5,
                                   art_mix_melt=7.0, art_mix_tunnel=-2.0, art_mix_displace=float("nan")))
    assert w == (0.0, 1.0, 0.5, 0.0, 0.0, 1.0)
    assert art_layer.art_weights({}) == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)      # defaults: relief, as always
    assert art_layer.art_weights(dict(art_use=2.6)) == art_layer.art_weights(dict(art_use=3.0))   # floor(x+0.5), as the shader
    assert art_layer.art_weights(dict(art_use=2.5)) == art_layer.art_weights(dict(art_use=3.0))


def test_one_hot_sets_for_a_mode_button():
    s = art_layer.mode_sets(2)
    assert s == dict(art_use=2.0, art_mix_relief=0.0, art_mix_stencil=0.0, art_mix_kaleido=1.0, art_mix_tunnel=0.0,
                     art_mix_displace=0.0, art_mix_melt=0.0)
    assert art_layer.mode_sets(99)["art_use"] == 5.0 and art_layer.mode_sets(-3)["art_use"] == 0.0


# ---------------------------------------------------------------- shader contract (CPU, reads the source)

def _fs():
    import live_visuals as lv
    return re.sub(r"//[^\n]*", "", lv.SCENE_FS)


def test_shader_reads_per_mode_weights_with_uniform_branches():
    code = _fs()
    assert re.search(r"uniform\s+float\s+artW\[6\]", code)
    assert "artUse" not in code and "amode" not in code
    for k in range(6):                                                      # every mode is skipped while its weight is 0
        assert f"artW[{k}] > 0.0" in code, k
    import live_visuals as lv
    src = Path(lv.__file__).read_text(encoding="utf-8")
    assert "artW=" in src and "art_weights(" in src


def test_relief_pass_only_renders_when_its_weight_is_up():
    import live_visuals as lv
    src = Path(lv.__file__).read_text(encoding="utf-8")
    assert "art_w[0] <= 0.0" in src                                         # no relief weight: the flat path, no 3D pass
    assert "not any(art_w)" in src                                          # nothing enabled: nothing rendered at all


# ---------------------------------------------------------------- persistence (CPU)

def test_scenes_store_and_recall_the_amounts(tmp_path):
    from control import Controller
    c = Controller(tmp_path, midi=False, ai=False, restore=False, seed=1)
    for k, v in dict(art_combine=1.0, art_mix_stencil=1.0, art_mix_kaleido=0.6, l_art=1.0).items():
        c.set(k, v)
    c.scene_save(3)
    for k in ("art_combine", "art_mix_stencil", "art_mix_kaleido"):
        c.set(k, 0.0)
    c.morph_bars = 0
    assert c.scene_recall(3, now=True)
    for _ in range(5):
        c.update(0.05)
    assert c.base["art_combine"] == 1.0 and c.base["art_mix_stencil"] == 1.0
    assert abs(c.base["art_mix_kaleido"] - 0.6) < 1e-9
    snap = c.scenes[3]                                                  # what scenes.json gets
    assert {"art_combine", *MIX} <= set(snap["base"])


def test_theme_and_layer_preset_scopes_cover_the_amounts():
    import shows
    import themes
    names = set(shows.layer_params("l_art"))
    assert {"art_combine", *MIX} <= names
    assert {"art_combine", *MIX} <= set(themes.scope_names(dict(kind="group", key="art")))


# ---------------------------------------------------------------- panel (CPU, reads the source)

def test_panel_has_a_combine_toggle_with_per_mode_faders():
    html = (ROOT / "panel.html").read_text(encoding="utf-8")
    assert "function mkArtModes(" in html
    assert "art_combine" in html and "COMBINE" in html
    assert "'art_mix_'" in html or '"art_mix_"' in html
    assert "mkArtModes('art_use'" in html or "mkArtModes(tabId" in html


# ---------------------------------------------------------------- GL renders (need .locks/gpu_suite)

def _batch(out, jobs, *args, timeout=900):
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "jobs.json"
    spec.write_text(json.dumps([dict(name=n, **j) for n, j in jobs.items()]))
    r = subprocess.run([sys.executable, str(ROOT / "live_visuals.py"), "--batch", str(spec), "--out", str(out), *map(str, args)],
                       capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    done = {d["name"]: d for d in json.loads((out / "batch.json").read_text())}
    return {n: dict(d, path=out / d["file"]) for n, d in done.items()}


def _img(path):
    return np.asarray(Image.open(path)).astype(np.float32)


def _one_hot(i):
    return dict(art_combine=1.0, **{m: (1.0 if j == i else 0.0) for j, m in enumerate(MIX)})


@pytest.mark.gl
@pytest.mark.parametrize("src", [1.0, 0.0])
def test_combine_one_hot_is_pixel_identical_to_the_selector(tmp_path, src):
    art = tmp_path / "art"
    art_layer.write_test_art(art)
    jobs = {}
    for i, n in enumerate(MODES):
        jobs[f"sel_{n}"] = dict(sets=dict(l_art=1.0, art_use=float(i), art_source=src), frames=40)
        jobs[f"mix_{n}"] = dict(sets=dict(l_art=1.0, art_use=float((i + 3) % 6), art_source=src, **_one_hot(i)), frames=40)
    res = _batch(tmp_path / "r", jobs, "--render", "640x400", "--art", art)
    for n in MODES:
        a, b = _img(res[f"sel_{n}"]["path"]), _img(res[f"mix_{n}"]["path"])
        assert np.array_equal(a, b), f"{n}: COMBINE with only this mode differs from the selector (max {np.abs(a - b).max()})"


COMBOS = {
    "stencil+kaleido": dict(art_mix_stencil=1.0, art_mix_kaleido=1.0),
    "stencil+tunnel": dict(art_mix_stencil=1.0, art_mix_tunnel=1.0),
    "kaleido+tunnel": dict(art_mix_kaleido=1.0, art_mix_tunnel=1.0),
    "kaleido+displace": dict(art_mix_kaleido=1.0, art_mix_displace=1.0),
    "stencil+melt": dict(art_mix_stencil=1.0, art_mix_melt=1.0),
    "kaleido+melt": dict(art_mix_kaleido=1.0, art_mix_melt=1.0),
    "stencil+kaleido+displace": dict(art_mix_stencil=1.0, art_mix_kaleido=1.0, art_mix_displace=1.0),
    "partial k0.4 t0.8 s0.5": dict(art_mix_kaleido=0.4, art_mix_tunnel=0.8, art_mix_stencil=0.5),
    "all six": {m: 1.0 for m in MIX},
}


@pytest.mark.gl
def test_combinations_render_clean(tmp_path):
    jobs = {n: dict(sets=dict(l_art=1.0, art_source=1.0, art_combine=1.0, **s), frames=120) for n, s in COMBOS.items()}
    jobs["off"] = dict(sets={}, frames=120)
    res = _batch(tmp_path, jobs, "--render", "640x400")
    off = _img(res["off"]["path"])
    for n in COMBOS:
        im = _img(res[n]["path"])
        assert res[n]["nonfinite"] == 0, n
        assert (im.min(axis=2) > 235).mean() < 0.02, f"{n}: blown out"
        assert (im.max(axis=2) < 40).mean() >= 0.25, f"{n}: less than 25 % black ({(im.max(axis=2) < 40).mean():.2f})"
        assert np.abs(im - off).mean() / 255 * 100 > 2.0, f"{n}: the art does not show"
