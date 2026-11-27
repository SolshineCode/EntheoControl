"""AUTO MAP: projector-reach refusals (live-test bugs 2026-11-26 16:50) + DETECT THIN PARTS on a synthetic grass fan."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
cv2 = pytest.importorskip("cv2")
import automap as am  # noqa: E402
import surfaces  # noqa: E402
from test_automap import CH, CW, H_PC, PH, PW, Rig  # noqa: E402

BASE = (325.0, 300.0)
TIPS = [(170, 95), (215, 75), (265, 62), (320, 58), (375, 62), (430, 75), (480, 95)]


def cal(dense=False):
    c = am.Calibration(np.linalg.inv(H_PC), 0.5, (CW, CH), (PW, PH), "aruco")
    if dense:                                    # a known dense camera -> projector map (the same warp, per pixel)
        yy, xx = np.mgrid[0:CH, 0:CW]
        q = am.apply_h(np.linalg.inv(H_PC), np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64))
        px, py = q[:, 0].reshape(CH, CW), q[:, 1].reshape(CH, CW)
        ok = (px >= 0) & (py >= 0) & (px < PW) & (py < PH)
        c.dense = (px.astype(np.float32), py.astype(np.float32), ok)
    return c


def grass_rig():
    rig = Rig()
    rig.refl[:] = 0.12                                                # a dark room behind the plant
    m = np.zeros((CH, CW), np.uint8)
    for tx, ty in TIPS:                                               # 7 blades from one base, 3 px wide
        cv2.line(m, (int(BASE[0]), int(BASE[1])), (tx, ty), 1, 3, cv2.LINE_AA)
    rig.refl[m > 0] = 0.95
    return rig


# ---------------------------------------------------------------- bugs 1 + 2: refuse / flag unreachable outlines
def test_reach_check_flags_outside_and_blown_up_outlines():
    ok = [[300, 300], [500, 300], [500, 500], [300, 500]]
    assert am.reach_check(ok, (PW, PH)) is None
    below = [[300, 780], [500, 780], [500, 930], [300, 930]]          # armchair / planter: under the frame
    assert am.reach_check(below, (PW, PH)) == "outside projector reach"
    huge = [[-12, -26], [1264, -26], [1264, 757], [-12, 757]]         # the 'Cloth screen' blow-up
    assert am.reach_check(huge, (PW, PH)) == "exceeds the projector frame"
    assert am.reach_check(huge, (PW, PH), wall=True) is None          # the wall IS the whole projection
    wild = [[300, 300], [1500, 300], [1500, 500], [300, 500]]          # 220 px past the right edge
    assert am.reach_check(wild, (PW, PH)) == "exceeds the projector frame"


def test_to_surfaces_refuses_unreachable_faces_with_the_reason():
    c = cal()
    P2C = H_PC
    def cam(poly):
        return am.apply_h(P2C, np.array(poly, np.float64))
    found = [dict(poly=cam([[300, 300], [500, 300], [500, 500], [300, 500]]), quad=True, conf=1, area=0.1, name="Box"),
             dict(poly=cam([[300, 780], [500, 780], [500, 930], [300, 930]]), quad=True, conf=1, area=0.1, name="Armchair"),
             dict(poly=cam([[-12, -26], [1264, -26], [1264, 757], [-12, 757]]), quad=True, conf=1, area=0.9, name="Cloth")]
    why = []
    new, ids = am.to_surfaces(found, c, [], (PW, PH), report=why)
    assert [s["name"] for s in new] == ["Box"] and len(ids) == 1
    assert why == ["Armchair: outside projector reach", "Cloth: exceeds the projector frame"]
    am.flag_reach(found, c, (PW, PH))
    assert [f.get("reach", "") for f in found] == ["", "outside projector reach", "exceeds the projector frame"]


def test_create_reports_refusals_and_state_shows_reach(tmp_path):
    rig = Rig()
    got = []
    w = am.AutoMap(rig.show, rig.shown_id, rig.still, lambda: ([], (PW, PH)), got.append, calib_dir=tmp_path, hold=0.0)
    w.calib = cal()
    w.found = am.flag_reach([dict(poly=am.apply_h(H_PC, np.array([[300, 780], [500, 780], [500, 930], [300, 930]], float)),
                                  quad=True, conf=1, area=0.1, name="Pillow")], w.calib, (PW, PH))
    assert w.state()["found"][0]["reach"] == "outside projector reach"
    assert not w.create() and "Pillow: outside projector reach" in w.err and not got


# ---------------------------------------------------------------- thin parts
def test_thin_instances_split_a_fan_of_blades_and_flow_base_to_tip():
    rig = grass_rig()
    rig.show(np.full((PH, PW), 200, np.uint8))
    white = rig.still()
    rig.show(np.zeros((PH, PW), np.uint8))
    black = rig.still()
    m = am.object_mask(white, black)
    labels, u, v, parts = am.thin_instances(m)
    assert len(parts) == len(TIPS), [p["length"] for p in parts]
    for p in parts:                                                   # base at the bottom, tip at the top
        assert p["base"][1] > p["tip"][1]
        sel = labels == p["label"]
        ys = np.nonzero(sel)[0]
        lo, hi = sel & (np.arange(CH)[:, None] > np.percentile(ys, 85)), sel & (np.arange(CH)[:, None] < np.percentile(ys, 15))
        assert u[lo].mean() < 0.3 < 0.75 < u[hi].mean()
        assert v[sel].min() == 0.0 and v[sel].max() <= 1.0
    tips = sorted(p["tip"][0] for p in parts)                         # one part per blade, each ending at its tip
    assert np.allclose(tips, [t[0] for t in TIPS], atol=6)


@pytest.mark.parametrize("dense", [False, True])
def test_warp_parts_to_projector_and_create_a_group(tmp_path, dense):
    rig = grass_rig()
    ss = surfaces.SurfaceSet((PW, PH))
    ss.mask_dir = tmp_path / "masks"

    def surf(c):
        assert ss.apply(surfaces.validate_cmd(c, (PW, PH)))
    w = am.AutoMap(rig.show, rig.shown_id, rig.still, lambda: ([dict(s) for s in ss.surfaces], (PW, PH)), surf,
                   calib_dir=tmp_path, hold=0.0)
    w.calib, w.mask_dir = cal(dense), ss.mask_dir
    w._detect_thin(False)
    st = w.state()["thin"]
    assert st["n"] == len(TIPS) and st["seen"] == len(TIPS)
    img = w.thin["img"]
    assert img.shape == (PH, PW, 4) and set(np.unique(img[..., 0])) == set(range(len(TIPS) + 1))
    # each blade's tip lands where the projector must light it (projector px of the camera tip)
    for p in w.thin["parts"]:
        tx, ty = am.apply_h(np.linalg.inv(H_PC), np.array([p["tip"]], float))[0]
        y0, y1, x0, x1 = int(ty) - 14, int(ty) + 15, int(tx) - 14, int(tx) + 15
        assert (img[y0:y1, x0:x1, 0] == p["label"]).any()
        assert img[y0:y1, x0:x1, 1][img[y0:y1, x0:x1, 0] == p["label"]].mean() > 180     # u ~ 1 at the tip
    assert w.create_thin("Grass blade")
    made = [s for s in ss.surfaces if s["shape"] == "mask"]
    assert [s["name"] for s in made] == [f"Grass blade {k}" for k in range(1, len(TIPS) + 1)]
    assert {s["group"] for s in made} == {"Grass blade"} and {s["maskfile"] for s in made} == {"grass_blade_1.png"}
    from PIL import Image
    back = np.asarray(Image.open(ss.mask_dir / "grass_blade_1.png"))
    assert (back == img).all()
    assert len(ss.surfaces) == len(TIPS)                              # the untouched full-frame #1 was replaced
    assert w.create_thin("Grass blade")                               # again: a new group + a new file, nothing overwritten
    assert {s["group"] for s in ss.surfaces} == {"Grass blade", "Grass blade 2"}
    assert (ss.mask_dir / "grass_blade_2.png").exists()


def test_thin_commands_validate():
    assert am.validate_command({"op": "detect_thin", "use_found": True}) == {"cmd": "automap", "type": "automap",
                                                                              "op": "detect_thin", "use_found": True}
    assert am.validate_command({"op": "create_thin", "name": "  Leaf\x00 "})["name"] == "Leaf"
    with pytest.raises(ValueError):
        am.validate_command({"op": "create_thin", "name": 5})
