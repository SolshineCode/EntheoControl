"""Multi-surface mapping through the real engine (headless live_visuals.py) and the panel command route."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import surfaces as S  # noqa: E402

FRAMES = 40


def cmd(op, **kw):
    return S.validate_cmd(dict(kw, cmd="surf", op=op))


def render(tmp_path, name, sset=None):
    out = tmp_path / name
    args = [sys.executable, str(ROOT / "live_visuals.py"), "--test", str(FRAMES), "--out", str(out)]
    if sset is not None:
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps(sset.to_dict()))
        args += ["--surfaces", str(f)]
    r = subprocess.run(args, capture_output=True, text=True, timeout=240, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "nonfinite=0" in r.stdout, r.stdout[-600:]
    return np.asarray(Image.open(out / f"frame_{FRAMES:04d}.png").convert("RGB")).astype(np.int16), r.stdout


@pytest.fixture(scope="module")
def stills(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("surf_engine")
    base, log = render(tmp, "base")
    one = S.SurfaceSet()                                        # a saved single full-frame surface: the old fast path
    gen = S.SurfaceSet()                                        # the same picture forced through the surface renderer
    gen.apply(cmd("add"))
    gen.apply(cmd("set", id=2, opacity=0))
    box = S.SurfaceSet()
    for k, pts in enumerate(([[330, 330], [650, 370], [650, 700], [330, 640]], [[650, 370], [900, 310], [900, 620], [650, 700]],
                             [[330, 330], [590, 280], [900, 310], [650, 370]])):
        if k:
            box.apply(cmd("add"))
        box.apply(cmd("corners", id=k + 1, pts=pts))
    glow = S.SurfaceSet.from_dict(box.to_dict())
    for i in (1, 2, 3):
        glow.apply(cmd("set", id=i, content="glow"))
    assert not gen.trivial() and one.trivial() and not box.trivial()
    return dict(base=base, log=log, one=render(tmp, "one", one)[0], gen=render(tmp, "gen", gen)[0],
                box=render(tmp, "box", box)[0], glow=render(tmp, "glow", glow)[0], boxset=box, tmp=tmp)


def test_single_full_frame_surface_is_pixel_identical_to_the_old_output(stills):
    assert stills["base"].max() > 60, "the reference frame is not black"
    assert "[map] 1 surface(s)" in stills["log"]
    assert np.array_equal(stills["one"], stills["base"]), "a plain surfaces.json must not change a single pixel"
    assert np.abs(stills["gen"] - stills["base"]).max() <= 1, "the surface renderer itself reproduces the frame"


def test_box_faces_are_lit_and_everything_else_is_black(stills):
    img, st = stills["box"], stills["boxset"]
    inside = np.zeros(img.shape[:2], dtype=bool)
    for y in range(0, 800, 4):
        for x in range(0, 1280, 4):
            inside[y, x] = any(S.point_in_poly((x + 0.5, y + 0.5), s["corners"]) for s in st.surfaces)
    grid = np.zeros_like(inside)
    grid[::4, ::4] = True
    near = np.zeros_like(inside)
    for dy in (-4, 0, 4):
        for dx in (-4, 0, 4):
            near |= np.roll(np.roll(inside, dy, 0), dx, 1)
    lit = img.max(axis=2) > 0
    assert not (lit & grid & ~near).any(), "light outside the box faces"
    assert img[100:250, 100:300].max() == 0 and img[720:, :].max() == 0
    assert (lit & inside).sum() > 0.5 * inside.sum(), "the faces show the picture"


def test_edge_glow_outlines_the_faces(stills):
    img = stills["glow"].max(axis=2)
    assert img[365:375, 640:660].max() > 180, "the shared corner of the three faces glows"
    assert img[480:560, 430:520].mean() < 25, "the middle of a glow-only face is dark"
    assert img[100:250, 100:300].max() == 0


def test_surf_commands_pass_the_panel_validator_and_reach_the_engine_untouched():
    import control
    from remote_control import validate_command
    c = validate_command({"type": "surf", "op": "corner", "id": 1, "i": 0, "x": 64, "y": 40, "junk": 1})
    assert c == {"cmd": "surf", "type": "surf", "op": "corner", "id": 1, "i": 0, "m": False, "x": 64.0, "y": 40.0}
    with pytest.raises(ValueError):
        validate_command({"cmd": "surf", "op": "explode"})
    assert "surf" not in control.CONTROL_CMDS, "the controller passes it through to the engine"


def test_projector_keys_that_change_the_mapping_are_ctrl_gated():
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    body = src[src.index("def on_key("):src.index("def mouse_uv(")]
    ctrl = body[body.index("elif ctrl:"):body.index("elif glfw.KEY_1 <= key")]
    plain = body[body.index("elif glfw.KEY_1 <= key"):]
    for key in ("KEY_R", "KEY_Z", "KEY_C", "KEY_G", "KEY_BACKSPACE"):
        assert key in ctrl and not re.search(r"key == glfw\." + key + r"\b", plain), key
    assert 'op="reset"' in ctrl and 'op="undo"' in ctrl and "mapper.save()" in plain
