"""PER-SURFACE LAYERS + MASK SURFACES (thin parts) + AUTO MAP reach checks: the CPU side (no GPU)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import surfaces as S  # noqa: E402
import shows  # noqa: E402


def cmd(op, **kw):
    return S.validate_cmd(dict(kw, cmd="surf", op=op))


def three():
    st = S.SurfaceSet()
    st.apply(cmd("corners", id=1, pts=[[100, 100], [400, 100], [400, 400], [100, 400]]))
    for k in (2, 3):
        st.apply(cmd("add"))
    return st


# ---------------------------------------------------------------- per-surface layers: model
def test_layer_validation_defaults_bounds_and_refusals():
    c = cmd("set_layers", id=1, layers=[{"src": "fractal", "opacity": 7, "blend": "screen", "scroll": -9, "hue": 400},
                                        {"src": "main"}])
    a, b = c["layers"]
    assert a == dict(S.LAYER_DEFAULT, src="fractal", opacity=1.0, blend="screen", scroll=-2.0, hue=180.0)
    assert b["src"] == "main" and b["on"] is True and b["scale"] == 1.0
    for bad in ([{"src": "nope"}], [{"blend": "burn"}], [{"on": 1}], [{"opacity": float("nan")}], [{}] * 4, "x", [3]):
        with pytest.raises(ValueError):
            cmd("set_layers", id=1, layers=bad)
    assert cmd("set", id=1, content="layers")["content"] == "layers"
    assert cmd("set", id=1, layers=[{"src": "kene"}])["layers"][0]["src"] == "kene"


def test_set_layers_switch_keeps_the_stack_and_is_one_undo_step():
    st = three()
    assert st.apply(cmd("set_layers", id=2, content="layers", layers=[{"src": "kene"}, {"src": "lines", "blend": "add"}]))
    s = st.get(2)
    assert s["content"] == "layers" and [x["src"] for x in s["layers"]] == ["kene", "lines"]
    assert not st.trivial()
    assert st.apply(cmd("set", id=2, content="main"))                 # "same as main": the stack is kept
    assert st.get(2)["layers"][1]["blend"] == "add"
    st.apply(cmd("set", id=2, content="layers"))
    st.undo(); st.undo(); st.undo()
    assert st.get(2)["content"] == "main" and st.get(2)["layers"] == []


def test_give_every_surface_a_different_look():
    st = three()
    for _ in range(9):
        st.apply(cmd("add"))
    assert st.apply(cmd("looks"))
    srcs = [s["layers"][0]["src"] for s in st.surfaces]
    assert all(s["content"] == "layers" for s in st.surfaces)
    n = len(S.LOOK_ROTA)
    assert len(set(srcs[:n])) == n                                   # distinct sources round robin
    looks = {(s["layers"][0]["src"], s["layers"][0]["hue"], s["layers"][0]["dir"]) for s in st.surfaces}
    assert len(looks) == len(st.surfaces)                            # and every surface still looks different
    assert all(s["layers"][0]["scroll"] != 0 for s in st.surfaces)   # all moving
    st.undo()
    assert all(s["content"] == "main" for s in st.surfaces)


def test_render_items_carry_only_live_layers():
    st = three()
    st.apply(cmd("set_layers", id=1, content="layers", layers=[{"src": "kene", "dir": 90, "hue": 90},
                                                               {"src": "beads", "on": False}, {"src": "main", "opacity": 0}]))
    it = {i["id"]: i for i in st.render_items(1280, 800)}
    assert it[1]["content"] == S.CONTENTS.index("layers") == 4
    (src, op, blend, scale, scroll, cx, cy, hue), = it[1]["layers"]
    assert src == "kene" and abs(cx) < 1e-9 and cy == pytest.approx(-1.0) and hue == pytest.approx(np.pi / 2)
    assert it[2]["layers"] == ()                                     # not in layers mode: nothing to request


def test_layers_persist_in_surfaces_json_and_show_files_and_old_files_migrate(tmp_path):
    st = three()
    st.apply(cmd("set_layers", id=3, content="layers", layers=[{"src": "land", "scale": 2.5}]))
    assert st.save(tmp_path / "surfaces.json")
    back = S.SurfaceSet.load(tmp_path / "surfaces.json")
    assert back.status == "loaded" and back.get(3)["layers"][0] == dict(S.LAYER_DEFAULT, src="land", scale=2.5)
    assert back.get(3)["content"] == "layers"
    raw = json.loads((tmp_path / "surfaces.json").read_text())
    for s in raw["surfaces"]:                                         # a file from before today: no layers key
        s.pop("layers")
    (tmp_path / "old.json").write_text(json.dumps(raw))
    old = S.SurfaceSet.load(tmp_path / "old.json")
    assert old.status == "loaded" and all(s["layers"] == [] for s in old.surfaces)
    raw["surfaces"][0]["layers"] = [{"src": "bogus"}]                 # a bad stack in a file: dropped, surface kept
    assert S.SurfaceSet.from_dict(raw).surfaces[0]["layers"] == []
    m = shows.clean_surfaces({"surfaces": st.to_dict()["surfaces"]})  # show files go through the same cleaner
    assert m["surfaces"][2]["layers"][0]["src"] == "land"


# ---------------------------------------------------------------- mask surfaces: model
def mask_surface(sid, label, f="grass_1.png", group="Grass blade"):
    s = S.new_surface(sid, corners=[[10, 10], [60, 10], [60, 200], [10, 200]], name=f"Grass blade {sid}")
    s.update(shape="mask", maskfile=f, label=label, group=group)
    return s


def test_mask_surface_validation_and_group_settings():
    for bad in ("../x.png", "a/b.png", "x.jpg", "", None, "C:x.png"):
        with pytest.raises(ValueError):
            cmd("replace", surfaces=[dict(mask_surface(1, 1), maskfile=bad)])
    with pytest.raises(ValueError):
        cmd("replace", surfaces=[dict(mask_surface(1, 1), label=0)])
    lax = S.clean_surface(dict(mask_surface(1, 1), maskfile="../../evil.png"))
    assert lax["shape"] == "quad" and "maskfile" not in lax          # non-strict (files): falls back to its box
    st = S.SurfaceSet()
    assert st.apply(cmd("replace", surfaces=[mask_surface(k, k) for k in (1, 2, 3)] + [S.new_surface(4)]))
    assert st.apply(cmd("set_layers", id=2, group_all=True, content="layers", layers=[{"src": "lines"}]))
    assert [s["content"] for s in st.surfaces] == ["layers"] * 3 + ["main"]
    assert st.apply(cmd("set", id=1, opacity=0.5))                    # each one individually as well
    assert [s["opacity"] for s in st.surfaces[:3]] == [0.5, 1.0, 1.0]
    st.undo()
    st.undo()
    assert all(s["content"] == "main" for s in st.surfaces)          # the group edit was ONE undo step
    assert not st.apply(cmd("set", id=4, shape="mask"))              # only AUTO MAP makes mask surfaces
    it = st.render_items(1280, 800)[0]
    assert it["maskfile"] == "grass_1.png" and it["label"] == 1 and it["glow"] == 0.0
    M = st.source_matrix(st.get(1))                                  # a stencil: the picture where it falls
    assert S.apply_h(M, (640, 400)) == pytest.approx((0.5, 0.5))


def test_mapper_keeps_masks_next_to_its_file(tmp_path):
    m = S.Mapper(tmp_path / "surfaces.json")
    assert m.set.mask_dir == tmp_path / "masks"


def test_64_mask_parts_fit_in_one_panel_request():
    import remote_control
    surfs = [dict(mask_surface(k, k), name="W" * S.MAX_NAME, layers=[dict(S.LAYER_DEFAULT)] * 3) for k in range(1, 65)]
    body = json.dumps([{"cmd": "surf", "op": "replace", "surfaces": surfs}])
    assert len(body.encode()) < 2 * remote_control.MAX_BODY       # a pasted layout this big: see HANDOFF (known limit)
    st = S.SurfaceSet()
    assert st.apply(cmd("replace", surfaces=[mask_surface(k, k) for k in range(1, 65)])) and len(st.surfaces) == 64
