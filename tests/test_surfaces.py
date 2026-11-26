"""Multi-surface mapping model (surfaces.py): geometry, commands, undo, files, render data. No GPU needed."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import surfaces as S  # noqa: E402

FULL = [[0, 0], [1280, 0], [1280, 800], [0, 800]]


def cmd(op, **kw):
    return S.validate_cmd(dict(kw, cmd="surf", op=op))


def near(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ------------------------------------------------------------------ geometry

def test_rectangle_readouts():
    d = S.describe(FULL)
    assert near(d["angles"], [90] * 4) and near(d["edges"], [1280, 800, 1280, 800])
    assert d["rotation"] == 0 and abs(d["skew"]) < 1e-9 and abs(d["aspect"] - 1.6) < 1e-9
    assert near(d["keystone"], [0, 0]) and d["area"] == 1280 * 800 and d["center"] == [640, 400]
    assert d["pct"][2] == [100.0, 100.0]


def test_keystone_trapezoid_angles_and_keystone():
    q = [[250, 0], [750, 0], [1000, 1000], [0, 1000]]                    # top edge half as wide
    low = math.degrees(math.atan2(1000, 250))
    assert near(S.interior_angles(q), [180 - low, 180 - low, low, low], 1e-9)
    assert abs(sum(S.interior_angles(q)) - 360) < 1e-9
    kh, kv = S.keystone(q)
    assert abs(kh - (500 - 1000) / 750 * 100) < 1e-9 and abs(kv) < 1e-9
    assert abs(S.skew(q)) < 1e-9, "a symmetric trapezoid is keystoned, not skewed"


def test_rotation_is_clockwise_positive_and_skew_reads_a_lean():
    sq = [[400, 200], [800, 200], [800, 600], [400, 600]]
    r = S.rotate_pts(sq, 10)
    assert abs(S.rotation(r) - 10) < 1e-9 and near(S.interior_angles(r), [90] * 4, 1e-9)
    assert r[1][1] > r[0][1], "clockwise on screen: the right end of the top edge drops"
    lean = [[500, 200], [900, 200], [800, 600], [400, 600]]            # parallelogram leaning right at the top
    assert S.skew(lean) > 10 and abs(S.skew(sq)) < 1e-9
    assert abs(S.skew([[300, 200], [700, 200], [800, 600], [400, 600]]) + S.skew(lean)) < 1e-9


def test_concave_polygon_reports_a_reflex_angle():
    arrow = [[0, 0], [100, 0], [100, 100], [50, 50], [0, 100]]
    a = S.interior_angles(arrow)
    assert abs(a[3] - 270) < 1e-9 and abs(sum(a) - 540) < 1e-9
    assert not S.is_convex(arrow) and S.is_simple(arrow)
    assert not S.is_simple([[0, 0], [100, 100], [100, 0], [0, 100]])        # bow tie
    assert S.is_convex(FULL) and S.is_convex(FULL[::-1]) and not S.is_convex([[0, 0], [10, 0], [20, 0], [0, 10]])


def test_square_up_keeps_centre_area_aspect_tilt():
    k = [[128, 80], [1150, 40], [1020, 700], [260, 680]]
    s = S.square_up(k)
    assert near(S.interior_angles(s), [90] * 4, 1e-7)
    assert near(S.centroid(s), S.centroid(k)) and abs(abs(S.signed_area(s)) - abs(S.signed_area(k))) < 1e-6
    assert abs(S.aspect(s) - S.aspect(k)) < 1e-9 and abs(S.rotation(s) - S.rotation(k)) < 1e-9
    lvl = S.square_up(k, level=True)
    assert abs(S.rotation(lvl)) < 1e-9 and lvl[0][1] == lvl[1][1]
    mirrored = [k[1], k[0], k[3], k[2]]
    assert S.signed_area(S.square_up(mirrored)) < 0, "a mirrored quad stays mirrored"


def test_typed_angle_and_edge_move_one_corner_only():
    a = S.with_angle(FULL, 1, 75)
    assert abs(S.interior_angles(a)[1] - 75) < 1e-6 and a[0] == [0, 0] and a[2] == [1280, 800] and a[3] == [0, 800]
    e = S.with_edge(FULL, 0, 900)
    assert abs(S.edge_lengths(e)[0] - 900) < 1e-9 and e[0] == [0, 0] and e[1] == [900, 0]


def test_homography_maps_the_four_points_and_inverts():
    src = [(100, 50), (900, 120), (840, 700), (60, 640)]
    dst = [(0, 0), (1, 0), (1, 1), (0, 1)]
    H = S.homography(src, dst)
    for p, q in zip(src, dst):
        assert near(S.apply_h(H, p), q, 1e-9)
    Hi = S.invert3(H)
    assert near(S.apply_h(Hi, (0.5, 0.5)), S.apply_h(S.homography(dst, src), (0.5, 0.5)), 1e-7)
    mid = S.apply_h(Hi, (0.5, 0.5))
    assert S.point_in_poly(mid, src)
    with pytest.raises(ValueError):
        S.homography([(0, 0), (1, 1), (2, 2), (3, 3)], dst)


def test_homography_matches_the_engine_one():
    np = pytest.importorskip("numpy")
    src, dst = [(0, 0), (1, 0), (1, 1), (0, 1)], [(0.1, 0.05), (0.9, 0.1), (0.85, 0.95), (0.05, 0.9)]
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u]); A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    Hn = np.linalg.svd(np.array(A, dtype=float))[2][-1].reshape(3, 3)
    assert np.allclose(np.array(S.homography(src, dst)), Hn / Hn[2, 2], atol=1e-9)


def test_clip_poly():
    tri = [(640, -200), (1500, 900), (-200, 900)]
    out = S.clip_poly(tri, FULL)
    assert len(out) >= 4 and all(-1e-6 <= x <= 1280 + 1e-6 and -1e-6 <= y <= 800 + 1e-6 for x, y in out)
    assert abs(S.signed_area(out)) < 1280 * 800
    assert S.clip_poly([(2000, 2000), (2100, 2000), (2100, 2100)], FULL) == []
    inside = [(100, 100), (200, 100), (150, 200)]
    assert S.clip_poly(inside, FULL) == inside and S.clip_poly(inside, FULL[::-1]) == inside


# ------------------------------------------------------------------ the set

def test_default_set_is_one_full_frame_surface_and_trivial():
    st = S.SurfaceSet()
    assert len(st.surfaces) == 1 and st.surfaces[0]["corners"] == [[0, 0], [1280, 0], [1280, 800], [0, 800]]
    assert st.trivial() and st.sel == 1
    assert st.legacy_corners() == [[0, 1 - 1.0], [1, 0], [1, 1], [0, 1]]             # BL BR TR TL, v up
    it = st.render_items(1280, 800)
    assert len(it) == 1 and it[0]["box"] == (0, 0, 1280, 800)
    M = [it[0]["M"][0:3], it[0]["M"][3:6], it[0]["M"][6:9]]
    assert near(S.apply_h(M, (0.5, 0.5)), (0.5 / 1280, 0.5 / 800), 1e-12)           # gl pixel centre -> the same texel
    assert near(S.apply_h(M, (1279.5, 799.5)), (1279.5 / 1280, 799.5 / 800), 1e-12)


def test_anything_beyond_a_plain_corner_pin_is_not_trivial():
    for kw in (dict(opacity=0.5), dict(brightness=1.2), dict(feather=4), dict(glow=0.5), dict(content="solid"),
               dict(src="span"), dict(shape="poly")):
        st = S.SurfaceSet()
        assert st.apply(cmd("set", id=1, **kw)) and not st.trivial(), kw
    st = S.SurfaceSet()
    st.apply(cmd("add"))
    assert not st.trivial()
    st.apply(cmd("set", id=2, enabled=False))
    assert st.trivial(), "a disabled extra surface costs nothing"
    st.apply(cmd("overlay", mode="all"))
    assert not st.trivial()


def test_list_ops_add_dup_delete_rename_reorder_select():
    st = S.SurfaceSet()
    assert st.apply(cmd("add", name="Box front")) and st.sel == 2 and st.surfaces[1]["name"] == "Box front"
    assert st.apply(cmd("dup", id=2)) and [s["id"] for s in st.surfaces] == [1, 2, 3] and st.sel == 3
    assert st.surfaces[2]["name"] == "Box front copy" and st.surfaces[2]["corners"] != st.surfaces[1]["corners"]
    assert st.surfaces[2]["color"] != st.surfaces[1]["color"], "every surface gets its own colour"
    assert st.apply(cmd("rename", id=3, name="  Box top \n")) and st.get(3)["name"] == "Box top"
    assert st.apply(cmd("move", id=3, to=0)) and [s["id"] for s in st.surfaces] == [3, 1, 2]
    assert [s["order"] for s in st.surfaces] == [0, 1, 2]
    assert st.apply(cmd("select", id=1)) and st.sel == 1 and not st.apply(cmd("select", id=99))
    assert st.apply(cmd("del", id=1)) and st.sel in (2, 3) and len(st.surfaces) == 2
    st.apply(cmd("del", id=2))
    assert not st.apply(cmd("del", id=3)), "the last surface cannot be deleted"
    for _ in range(40):
        st.apply(cmd("add"))
    assert len(st.surfaces) == S.MAX_SURFACES and len({s["id"] for s in st.surfaces}) == S.MAX_SURFACES


def test_corner_moves_refuse_to_fold_and_clamp_to_the_overscan():
    st = S.SurfaceSet()
    assert st.apply(cmd("corner", id=1, i=0, x=100, y=50)) and st.surfaces[0]["corners"][0] == [100, 50]
    assert not st.apply(cmd("corner", id=1, i=0, x=1279, y=799)), "TL onto BR folds the quad"
    assert st.surfaces[0]["corners"][0] == [100, 50]
    assert st.apply(cmd("nudge", id=1, i=0, dx=-0.25, dy=1)) and st.surfaces[0]["corners"][0] == [99.75, 51]
    st.apply(cmd("corner", id=1, i=2, x=1e9, y=1e9))
    assert st.surfaces[0]["corners"][2] == [1280 * 1.5, 800 * 1.5]
    assert not st.apply(cmd("corner", id=7, i=0, x=1, y=1))                          # unknown id
    assert st.apply(cmd("corners", id=1, pts=[[10, 10], [500, 20], [480, 300], [20, 310]]))
    assert not st.apply(cmd("corners", id=1, pts=[[10, 10], [500, 20], [20, 310], [480, 300]]))   # bow tie


def test_shift_square_angle_edge_rotate_reset():
    st = S.SurfaceSet()
    st.apply(cmd("corners", id=1, pts=[[300, 200], [820, 230], [790, 600], [280, 560]]))
    c0 = [list(p) for p in st.surfaces[0]["corners"]]
    assert st.apply(cmd("shift", id=1, dx=10, dy=-5)) and st.surfaces[0]["corners"][0] == [310, 195]
    assert not st.apply(cmd("shift", id=1, dx=5000, dy=0)), "a shift never squashes the quad against the limit"
    assert st.apply(cmd("rotate", id=1, deg=0)) and abs(S.rotation(st.surfaces[0]["corners"])) < 1e-9
    assert st.apply(cmd("angle", id=1, i=2, deg=90)) and abs(S.interior_angles(st.surfaces[0]["corners"])[2] - 90) < 1e-5
    assert st.apply(cmd("edge", id=1, i=0, len=400)) and abs(S.edge_lengths(st.surfaces[0]["corners"])[0] - 400) < 1e-9
    assert st.apply(cmd("square", id=1)) and near(S.interior_angles(st.surfaces[0]["corners"]), [90] * 4, 1e-6)
    assert st.apply(cmd("reset", id=1)) and st.surfaces[0]["corners"] == [[0, 0], [1280, 0], [1280, 800], [0, 800]]
    assert c0 != st.surfaces[0]["corners"]


def test_undo_redo_and_gesture_coalescing():
    st = S.SurfaceSet()
    for k in range(30):                                               # one drag = one undo step
        st.apply(cmd("corner", id=1, i=0, x=k, y=k, g="d1"))
    assert st.surfaces[0]["corners"][0] == [29, 29] and st.state()["undo"] == 1
    st.apply(cmd("corner", id=1, i=0, x=50, y=50, g="d2"))
    st.apply(cmd("add"))
    assert st.undo() and len(st.surfaces) == 1
    assert st.undo() and st.surfaces[0]["corners"][0] == [29, 29]
    assert st.undo() and st.surfaces[0]["corners"][0] == [0, 0] and not st.undo()
    assert st.redo() and st.surfaces[0]["corners"][0] == [29, 29]
    st.apply(cmd("corner", id=1, i=1, x=1000, y=5))                   # a new edit drops the redo branch
    assert not st.redo()
    assert st.apply(cmd("undo")) and st.apply(cmd("redo")) and st.surfaces[0]["corners"][1] == [1000, 5]


def test_undo_depth_is_twenty_and_refused_edits_leave_no_step():
    st = S.SurfaceSet()
    for k in range(35):
        st.apply(cmd("corner", id=1, i=0, x=k + 1, y=0))
    assert st.state()["undo"] == S.UNDO_DEPTH == 20
    n = 0
    while st.undo():
        n += 1
    assert n == 20 and st.surfaces[0]["corners"][0] == [15, 0]
    st2 = S.SurfaceSet()
    assert not st2.apply(cmd("corner", id=1, i=0, x=1279, y=799)) and st2.state()["undo"] == 0 and st2.rev == 0
    assert not st2.apply(cmd("corner", id=1, i=0, x=0, y=0)), "an edit that changes nothing is not an edit"
    st2.apply(cmd("select", id=1)); st2.apply(cmd("overlay", mode="all", grid=True))
    assert st2.state()["undo"] == 0 and st2.overlay == dict(mode="all", grid=True) and not st2.dirty


def test_mask_points_add_delete_move_and_poly_outline():
    st = S.SurfaceSet()
    st.apply(cmd("corners", id=1, pts=[[200, 100], [800, 100], [800, 700], [200, 700]]))
    assert st.apply(cmd("mask_add", id=1)) and st.surfaces[0]["shape"] == "poly" and len(st.surfaces[0]["mask"]) == 5
    assert st.apply(cmd("mask", id=1, pts=[[500, 100], [800, 700], [200, 700]]))
    assert st.polygon(st.surfaces[0]) == [(500, 100), (800, 700), (200, 700)]
    assert not st.apply(cmd("mask_del", id=1, i=0)), "a mask keeps at least 3 points"
    assert st.apply(cmd("corner", id=1, m=True, i=0, x=500, y=-300))             # the mask may leave the quad...
    poly = st.polygon(st.surfaces[0])
    assert min(y for _, y in poly) >= 100 - 1e-6, "...but the picture never leaves the quad"
    assert st.apply(cmd("mask", id=1, pts=[[0, 0], [50, 0], [50, 50]])) and st.polygon(st.surfaces[0]) == []
    assert st.render_items(1280, 800) == [], "a mask that misses its quad draws nothing"
    assert not st.apply(cmd("mask", id=1, pts=[[300, 200], [700, 600], [700, 200], [300, 600]])), "self-crossing mask"
    for _ in range(10):
        st.apply(cmd("mask_add", id=1))
    assert len(st.surfaces[0]["mask"]) == S.MAX_MASK


def test_set_fields_and_glow_content_turns_the_outline_on():
    st = S.SurfaceSet()
    assert st.apply(cmd("set", id=1, opacity=2, brightness=-1, feather=9999, color=[1, 0.5, 2]))
    s = st.surfaces[0]
    assert (s["opacity"], s["brightness"], s["feather"], s["color"]) == (1.0, 0.0, S.MAX_FEATHER, [1.0, 0.5, 1.0])
    assert st.apply(cmd("set", id=1, content="glow")) and s is not st.surfaces[0] or st.surfaces[0]["glow"] == 1.0
    assert st.apply(cmd("set", id=1, src="rect", rect=[0.25, 0.5, 0.9, 0.9]))
    assert st.surfaces[0]["source"] == dict(mode="rect", rect=[0.25, 0.5, 0.75, 0.5])


@pytest.mark.parametrize("bad", [
    {}, {"op": "nope"}, {"op": "corner", "id": 1, "i": 4, "x": 0, "y": 0}, {"op": "corner", "id": 1, "i": 0, "x": "1", "y": 0},
    {"op": "corner", "id": 1, "i": 0, "x": float("nan"), "y": 0}, {"op": "corner", "id": True, "i": 0, "x": 0, "y": 0},
    {"op": "corner", "id": 1, "i": 0, "x": 0}, {"op": "set", "id": 1}, {"op": "set", "id": 1, "content": "disco"},
    {"op": "set", "id": 1, "enabled": 1}, {"op": "rename", "id": 1, "name": "   "}, {"op": "rename", "id": 1, "name": 5},
    {"op": "corners", "id": 1, "pts": [[0, 0]]}, {"op": "mask", "id": 1, "pts": [[0, 0], [1, 1]]},
    {"op": "move", "id": 1, "to": -1}, {"op": "overlay", "mode": "loud"}, {"op": "add", "g": "x" * 99},
    {"op": "set", "id": 1, "color": [1, 2]}, {"op": "replace", "surfaces": []}, {"op": "replace", "surfaces": [{"corners": 3}]},
])
def test_validate_rejects_malformed_commands(bad):
    with pytest.raises(ValueError):
        S.validate_cmd(bad)


def test_validate_normalizes_and_carries_cmd_and_type():
    c = S.validate_cmd({"op": "nudge", "id": 3, "i": 2, "dx": 1, "dy": -0.25, "junk": 1, "g": 7})
    assert c == {"cmd": "surf", "type": "surf", "op": "nudge", "id": 3, "i": 2, "m": False, "dx": 1.0, "dy": -0.25, "g": 7}
    assert S.validate_cmd({"op": "rename", "id": 1, "name": "x" * 99})["name"] == "x" * S.MAX_NAME
    assert not S.SurfaceSet().apply({"op": "corner"}), "apply never raises on a short command"


def test_replace_imports_a_whole_layout_as_one_undo_step():
    a = S.SurfaceSet()
    a.apply(cmd("add")); a.apply(cmd("set", id=2, content="glow"))
    b = S.SurfaceSet()
    assert b.apply(cmd("replace", surfaces=a.to_dict()["surfaces"]))
    assert b.to_dict()["surfaces"] == a.to_dict()["surfaces"]
    assert b.undo() and len(b.surfaces) == 1


# ------------------------------------------------------------------ sources / render data

def three_faces():
    st = S.SurfaceSet()
    st.apply(cmd("corners", id=1, pts=[[200, 300], [500, 300], [500, 600], [200, 600]]))
    st.apply(cmd("add")); st.apply(cmd("corners", id=2, pts=[[500, 300], [700, 200], [700, 500], [500, 600]]))
    st.apply(cmd("add")); st.apply(cmd("corners", id=3, pts=[[200, 300], [500, 300], [700, 200], [400, 200]]))
    return st


def test_span_is_one_continuous_undistorted_picture_across_surfaces():
    st = three_faces()
    for i in (1, 2, 3):
        st.apply(cmd("set", id=i, src="span"))
    x0, y0, cw, ch = st.span_box()
    assert abs(cw / ch - 1.6) < 1e-9 and x0 <= 200 and y0 <= 200 and x0 + cw >= 700 and y0 + ch >= 600
    shared = (500, 450)                                               # a point on the edge two faces share
    uv = [S.apply_h(st.source_matrix(st.get(i)), shared) for i in (1, 2)]
    assert near(uv[0], uv[1], 1e-12), "no seam: both faces sample the same spot"
    a, b = S.apply_h(st.source_matrix(st.get(1)), (300, 400)), S.apply_h(st.source_matrix(st.get(1)), (400, 400))
    c = S.apply_h(st.source_matrix(st.get(1)), (300, 500))
    assert abs((b[0] - a[0]) * cw - 100) < 1e-9 and abs((c[1] - a[1]) * ch - 100) < 1e-9, "uniform scale: no stretch"
    assert all(0 <= v <= 1 for i in (1, 2, 3) for p in st.get(i)["corners"] for v in S.apply_h(st.source_matrix(st.get(i)), p))
    st.apply(cmd("set", id=3, enabled=False))
    assert st.span_box() == (x0, y0, cw, ch), "review L5: switching a face off must not move the picture on the others"
    st.apply(cmd("set", id=2, src="full")); st.apply(cmd("set", id=3, src="full"))
    assert st.span_box() != (x0, y0, cw, ch), "faces that leave span mode do leave the canvas"


def test_full_and_rect_sources_map_corners_to_the_frame_or_the_sub_rect():
    st = three_faces()
    s = st.get(2)
    M = st.source_matrix(s)
    for p, q in zip(s["corners"], [(0, 0), (1, 0), (1, 1), (0, 1)]):
        assert near(S.apply_h(M, p), q, 1e-9)
    st.apply(cmd("set", id=2, src="rect", rect=[0.5, 0.25, 0.5, 0.5]))
    M = st.source_matrix(st.get(2))
    for p, q in zip(st.get(2)["corners"], [(0.5, 0.25), (1, 0.25), (1, 0.75), (0.5, 0.75)]):
        assert near(S.apply_h(M, p), q, 1e-9)


def test_render_items_are_in_gl_pixels_scaled_to_the_framebuffer():
    st = three_faces()
    st.apply(cmd("set", id=2, enabled=False))
    st.apply(cmd("set", id=3, content="glow", feather=10))
    it = st.render_items(640, 400)                                   # half-size preview window
    assert [i["id"] for i in it] == [1, 3] and it[1]["content"] == 2 and it[1]["glow"] == 1.0 and it[1]["feather"] == 5.0
    assert it[0]["poly"] == [(100, 250), (250, 250), (250, 100), (100, 100)]          # y up, halved
    assert it[0]["box"] == (99, 99, 251, 251)
    M = [it[0]["M"][0:3], it[0]["M"][3:6], it[0]["M"][6:9]]
    assert near(S.apply_h(M, (100, 250)), (0, 1), 1e-9) and near(S.apply_h(M, (250, 100)), (1, 0), 1e-9)   # TL -> uv (0, 1)
    assert it[0]["selected"] is False and it[1]["selected"] is True
    st.apply(cmd("corners", id=1, pts=[[-600, -380], [-100, -380], [-100, -100], [-600, -100]]))
    assert [i["id"] for i in st.render_items(640, 400)] == [3], "a surface wholly off the projector is skipped"


def test_pick_finds_the_topmost_surface_and_its_source_point():
    st = three_faces()
    s, uv = st.pick(350, 450)
    assert s["id"] == 1 and near(uv, (0.5, 0.5), 1e-9)
    assert st.pick(5, 5) is None
    st.apply(cmd("add")); st.apply(cmd("corners", id=4, pts=[[300, 400], [400, 400], [400, 500], [300, 500]]))
    assert st.pick(350, 450)[0]["id"] == 4


def test_twelve_surfaces_render_data_is_cheap():
    import time
    st = S.SurfaceSet()
    for k in range(11):
        st.apply(cmd("add"))
        st.apply(cmd("mask_add", id=k + 2))
    t0 = time.perf_counter()
    for _ in range(20):
        assert len(st.render_items(1280, 800)) == 12
    assert (time.perf_counter() - t0) / 20 < 0.02


# ------------------------------------------------------------------ files

def test_legacy_corner_round_trip():
    leg = [[0.1, 0.05], [0.9, 0.1], [0.85, 0.95], [0.05, 0.9]]         # BL BR TR TL, v up
    px = S.legacy_to_px(leg)
    assert near(px[0], (0.05 * 1280, 0.1 * 800)) and near(px[3], (0.1 * 1280, 0.95 * 800))    # TL, BL in pixels
    back = S.px_to_legacy(px)
    assert all(near(a, b, 1e-12) for a, b in zip(back, leg))
    st = S.SurfaceSet()
    assert st.set_legacy_corners(leg) and all(near(a, b, 1e-12) for a, b in zip(st.legacy_corners(), leg))
    assert st.trivial(), "a moved corner pin is still the old fast path"


def test_missing_file_migrates_mapping_json_into_surface_1(tmp_path):
    leg = [[0.1, 0.05], [0.9, 0.1], [0.85, 0.95], [0.05, 0.9]]
    (tmp_path / "mapping.json").write_text(json.dumps(leg))
    st = S.SurfaceSet.load(tmp_path / "surfaces.json", legacy=tmp_path / "mapping.json")
    assert st.status == "migrated" and len(st.surfaces) == 1 and st.trivial()
    assert all(near(a, b, 1e-12) for a, b in zip(st.legacy_corners(), leg))
    assert not (tmp_path / "surfaces.json").exists(), "loading never writes"
    assert json.loads((tmp_path / "mapping.json").read_text()) == leg
    for junk in ("{", "[[0,0],[1,1],[2,2],[3,3]]", "[1,2,3,4]", '{"a":1}'):
        (tmp_path / "mapping.json").write_text(junk)
        st = S.SurfaceSet.load(tmp_path / "surfaces.json", legacy=tmp_path / "mapping.json")
        assert st.status == "default" and st.surfaces[0]["corners"] == [[0, 0], [1280, 0], [1280, 800], [0, 800]]
    assert S.SurfaceSet.load(tmp_path / "none.json").status == "default"
    assert S.SurfaceSet.load(None, legacy=tmp_path / "nope.json").status == "default"


def test_save_load_round_trip_and_legacy_file_follows(tmp_path):
    st = three_faces()
    st.apply(cmd("set", id=2, content="glow", feather=6, src="span", opacity=0.8))
    st.apply(cmd("mask", id=3, pts=[[250, 290], [500, 295], [650, 210]]))
    st.apply(cmd("overlay", mode="all", grid=True))
    assert st.dirty and st.save(tmp_path / "surfaces.json", legacy=tmp_path / "mapping.json") and not st.dirty
    raw = json.loads((tmp_path / "surfaces.json").read_text())
    assert raw["version"] == 1 and raw["output"] == [1280, 800] and "overlay" not in raw and len(raw["surfaces"]) == 3
    assert set(raw["surfaces"][0]) == {"id", "name", "order", "enabled", "shape", "corners", "mask", "source", "content",
                                       "color", "opacity", "brightness", "glow", "feather"}
    back = S.SurfaceSet.load(tmp_path / "surfaces.json")
    assert back.status == "loaded" and back.to_dict() == st.to_dict() and back.overlay["mode"] == "off"
    assert not list(tmp_path.glob("*.tmp"))
    leg = json.loads((tmp_path / "mapping.json").read_text())
    assert all(near(a, b, 1e-12) for a, b in zip(leg, S.px_to_legacy(st.surfaces[0]["corners"])))


@pytest.mark.parametrize("junk", ["{not json", "[]", '{"surfaces": 5}', '{"surfaces": []}', '{"surfaces": [{"corners": "x"}]}',
                                  '{"surfaces": [{"corners": [[0,0],[1,1],[2,2],[3,3]]}]}', '{"output": [0, 5], "surfaces": []}'])
def test_corrupt_file_is_quarantined_and_defaults_come_back(tmp_path, junk):
    f = tmp_path / "surfaces.json"
    f.write_text(junk)
    st = S.SurfaceSet.load(f)
    assert st.status == "quarantined" and st.can_save and len(st.surfaces) == 1
    bad = list(tmp_path.glob("surfaces.json.*.bad"))
    assert len(bad) == 1 and bad[0].read_text() == junk and not f.exists()
    assert st.save(f) and S.SurfaceSet.load(f).status == "loaded"


def test_read_only_load_never_moves_the_file_and_headless_never_writes(tmp_path):
    f = tmp_path / "surfaces.json"
    f.write_text("{broken")
    st = S.SurfaceSet.load(f, writable=False)
    assert st.status == "quarantined" and f.read_text() == "{broken" and not list(tmp_path.glob("*.bad"))
    assert not S.SurfaceSet().save(None)


def test_unreadable_file_is_left_alone_and_saving_is_off(tmp_path, monkeypatch):
    f = tmp_path / "surfaces.json"
    good = S.SurfaceSet()
    good.apply(cmd("add"))
    good.save(f)
    text = f.read_text()
    (tmp_path / "mapping.json").write_text("[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]")
    real = Path.read_text

    def locked(self, *a, **k):
        if self.name == "surfaces.json":
            raise PermissionError("locked")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", locked)
    st = S.SurfaceSet.load(f, legacy=tmp_path / "mapping.json")
    assert st.status == "unreadable" and not st.can_save
    assert st.surfaces[0]["corners"][0] == pytest.approx([128, 80]), "review L1: the old corner pin still applies"
    assert not st.save(f)
    monkeypatch.undo()
    assert f.read_text() == text and not list(tmp_path.glob("*.bad"))


def test_one_broken_surface_does_not_take_the_others_down(tmp_path):
    st = three_faces()
    raw = st.to_dict()
    raw["surfaces"][1]["corners"] = [[0, 0], [1, 1]]
    raw["surfaces"][2]["opacity"] = "loud"
    raw["surfaces"][2]["id"] = raw["surfaces"][0]["id"]               # duplicate id
    raw["surfaces"].append("junk")
    (tmp_path / "surfaces.json").write_text(json.dumps(raw))
    back = S.SurfaceSet.load(tmp_path / "surfaces.json")
    assert back.status == "loaded" and len(back.surfaces) == 2 and back.surfaces[1]["opacity"] == 1.0
    assert len({s["id"] for s in back.surfaces}) == 2


# ------------------------------------------------------------------ engine bridge

def test_mapper_autosaves_after_a_quiet_moment_and_keeps_the_legacy_bridge(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(S.time, "monotonic", lambda: clock[0])
    m = S.Mapper(tmp_path / "surfaces.json", legacy=tmp_path / "mapping.json")
    assert m.corners() == [[0, 0], [1, 0], [1, 1], [0, 1]] and m.target_id() == 1
    assert m.command(cmd("corner", id=1, i=0, x=64, y=40))
    m.tick()
    assert not (tmp_path / "surfaces.json").exists(), "not while the user is still dragging"
    clock[0] += 2.0
    m.tick()
    assert json.loads((tmp_path / "surfaces.json").read_text())["surfaces"][0]["corners"][0] == [64, 40]
    assert json.loads((tmp_path / "mapping.json").read_text())[3] == [0.05, 0.95]           # TL in the old format
    leg = m.corners()
    leg[1] = [0.9, 0.1]                                               # old `corner` command / mouse drag: BR
    assert m.push_corners(leg) and near(m.set.surfaces[0]["corners"][2], (1152, 720), 1e-9)
    m.command(cmd("overlay", mode="all"))
    assert m._edit_t is not None
    m.flush()
    assert S.SurfaceSet.load(tmp_path / "surfaces.json").surfaces[0]["corners"][2] == m.set.surfaces[0]["corners"][2]
    assert m.command(cmd("save")) and not m.set.dirty


def test_mapper_headless_is_read_only_and_click_helpers(tmp_path):
    f = tmp_path / "s.json"
    st = S.SurfaceSet()
    st.apply(cmd("corners", id=1, pts=[[200, 300], [500, 300], [500, 600], [200, 600]]))
    st.apply(cmd("add")); st.apply(cmd("corners", id=2, pts=[[700, 100], [1100, 100], [1100, 400], [700, 400]]))
    st.save(f)
    text = f.read_text()
    m = S.Mapper(f, writable=False)
    assert m.set.status == "loaded" and m.command(cmd("del", id=1)) and not m.save()
    m.flush()
    assert f.read_text() == text
    m = S.Mapper(f, writable=False)
    assert m.ripple_uv(350 / 1280, 1 - 450 / 800) == pytest.approx((0.5, 0.5)) and m.ripple_uv(0.01, 0.01) is None
    assert m.nearest_corner(1090 / 1280, 1 - 110 / 800) == (2, 2) and m.set.sel == 2      # TR of surface 2 = legacy index 2
    assert m.target_id() == 2 and m.corners()[2] == pytest.approx([1100 / 1280, 1 - 100 / 800])
    none = S.Mapper(None, legacy=tmp_path / "nope.json")
    assert none.set.trivial() and not none.command(cmd("save"))


# ------------------------------------------------------------------ Opus review regressions

def test_zero_size_framebuffer_is_not_an_error():
    """Review H1: a minimised projector window reports a 0 x 0 framebuffer."""
    st = S.SurfaceSet()
    st.apply(cmd("add"))
    assert st.render_items(0, 0) == [] and st.render_items(1280, 0) == [] and st.render_items(0, 800) == []
    assert len(st.render_items(1280, 800)) == 2


def test_output_gate_skips_zero_size_and_never_gives_up_on_the_surface_path():
    """Review H1: five bad frames used to switch multi-surface output off until a restart."""
    g = S.OutputGate()
    for _ in range(50):                                   # minimised for a while: nothing drawn, nothing counted
        assert g.plan((0, 0), False, True, 1.0) == "skip"
    assert g.fails == 0 and g.plan((1280, 800), False, True, 1.0) == "multi"
    assert g.plan((1280, 800), True, True, 1.0) == "pin" and g.mode == "pin"            # one plain surface: old path
    assert g.plan((1280, 800), False, False, 1.0) == "pin"                             # no renderer at all
    t = 10.0
    for k in range(S.OutputGate.MAX_FAILS):
        assert g.plan((1280, 800), False, True, t) == "multi"
        line = g.failed(t, "boom")
        assert "boom" in line
        t += 1 / 60
    assert g.mode == "fallback" and "trying again" in line
    assert g.plan((1280, 800), False, True, t) == "pin", "fallback = the old pin (black outside it)"
    assert g.plan((1280, 800), False, True, t + 1.9) == "pin"
    assert g.plan((0, 0), False, True, t + 1.95) == "skip" and g.mode == "fallback"
    assert g.plan((1280, 800), False, True, t + 2.1) == "multi", "it tries the surfaces again every RETRY_S"
    g.failed(t + 2.1, "still broken")
    assert g.plan((1280, 800), False, True, t + 2.2) == "pin" and g.plan((1280, 800), False, True, t + 4.2) == "multi"
    g.ok()
    assert g.mode == "multi" and g.fails == 0 and g.error == ""
    assert all(g.plan((1280, 800), False, True, t + 4.2 + k / 60) == "multi" for k in range(200))
    for hours in (1, 5):                                  # hours later it is still retrying, not dead
        for _ in range(S.OutputGate.MAX_FAILS):
            g.failed(hours * 3600.0)
        assert g.plan((1280, 800), False, True, hours * 3600.0 + 2.01) == "multi"


def test_engine_output_pass_uses_the_gate_and_the_pin_shader_is_black_outside_the_pin():
    """Review H1, engine side (source level: no GL while a show is live)."""
    import re
    src = (ROOT / "live_visuals.py").read_text(encoding="utf-8")
    a = src.index("# ---- output pass")
    out = src[a:src.index("if recorder is not None", a)] if "if recorder is not None" in src[a:] else src[a:a + 3000]
    assert "mapper.gate.plan(" in out and "mapper.gate.ok()" in out and "mapper.gate.failed(" in out
    assert 'map_ref["fails"]' not in src, "the old give-up-forever counter is gone"
    assert re.search(r'if plan == "pin":\s+target\.use\(\)', out), "nothing is drawn on a skipped (0 x 0) frame"
    fs = src[src.index('OUT_FS = """'):src.index('METER_FS = """')]
    assert "vec3 col = vec3(0);" in fs and "if (all(greaterThanEqual(c,vec2(0))) && all(lessThanEqual(c,vec2(1))))" in fs
    assert "surfaces=mapper.state()" in src, "the panel is told when the engine is on the fallback"


def failing_writes(monkeypatch, fail):
    real = S._atomic_write
    calls = []

    def write(path, obj):
        calls.append(str(path))
        return False if fail[0] else real(path, obj)

    monkeypatch.setattr(S, "_atomic_write", write)
    return calls


def test_failed_autosave_is_retried_with_backoff_and_flushed_at_exit(tmp_path, monkeypatch):
    """Review M6: one failed write (AV / indexer holding the file) used to drop the edits for good."""
    clock, fail = [100.0], [True]
    monkeypatch.setattr(S.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(S.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    calls = failing_writes(monkeypatch, fail)
    f = tmp_path / "surfaces.json"
    m = S.Mapper(f)
    m.command(cmd("corner", id=1, i=0, x=64, y=40))
    clock[0] += 1.6
    m.tick()
    assert len(calls) == 1 and m.save_failed and m.state()["save_failed"] and m.set.dirty and not f.exists()
    m.tick()
    assert len(calls) == 1, "not hammered every frame"
    waits = []
    for _ in range(7):                                    # 1.5, 3, 6, 12, 24, 30, 30
        t0 = clock[0]
        while len(calls) == len(waits) + 1:
            clock[0] += 0.1
            m.tick()
        waits.append(round(clock[0] - t0, 1))
    assert waits[0] == pytest.approx(1.5, abs=0.11) and waits[1] == pytest.approx(3.0, abs=0.11)
    assert waits == sorted(waits) and waits[-1] == pytest.approx(S.Mapper.RETRY_MAX_S, abs=0.11) == waits[-2]
    n = len(calls)
    assert m.flush(tries=3) is False and len(calls) == n + 3, "exit tries a few times, then says so"
    fail[0] = False
    assert m.flush() is True and not m.save_failed and not m.set.dirty
    assert json.loads(f.read_text())["surfaces"][0]["corners"][0] == [64, 40]
    n = len(calls)
    clock[0] += 100
    m.tick()
    assert m.flush() is True and len(calls) == n, "nothing pending, nothing written"


def test_flush_writes_dirty_edits_even_when_no_save_is_pending(tmp_path, monkeypatch):
    """Review M6: flush() looked only at the pending timer, which a failed save had cleared."""
    f = tmp_path / "surfaces.json"
    m = S.Mapper(f)
    m.set.apply(cmd("corner", id=1, i=0, x=10, y=10))     # an edit that never went through command()
    assert m._edit_t is None and m.set.dirty
    assert m.flush() and json.loads(f.read_text())["surfaces"][0]["corners"][0] == [10, 10]
    ro = S.Mapper(f, writable=False)
    ro.command(cmd("corner", id=1, i=0, x=20, y=20))
    assert ro.flush() is True and json.loads(f.read_text())["surfaces"][0]["corners"][0] == [10, 10]


def test_unreadable_surfaces_file_still_uses_the_old_corner_pin(tmp_path, monkeypatch):
    """Review L1."""
    (tmp_path / "surfaces.json").write_text("{}")
    (tmp_path / "mapping.json").write_text("[[0.25,0.25],[0.75,0.25],[0.75,0.75],[0.25,0.75]]")
    real = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("locked"))
                        if self.name == "surfaces.json" else real(self, *a, **k))
    m = S.Mapper(tmp_path / "surfaces.json", legacy=tmp_path / "mapping.json")
    assert m.set.status == "unreadable" and not m.set.can_save
    assert m.corners() == [pytest.approx(p) for p in ([0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75])]
    m.command(cmd("corner", id=1, i=0, x=5, y=5))
    assert m.flush() is True and not m.save_failed, "saving is off by design here: no retry storm"
    monkeypatch.undo()
    assert (tmp_path / "surfaces.json").read_text() == "{}" and json.loads((tmp_path / "mapping.json").read_text())[0] == [0.25, 0.25]


def test_lossy_load_keeps_a_copy_of_the_original_before_the_first_save(tmp_path):
    """Review L2: dropped surfaces used to vanish from disk at the next edit."""
    st = three_faces()
    raw = st.to_dict()
    raw["surfaces"][1]["corners"] = [[0, 0], [1, 1]]
    f = tmp_path / "surfaces.json"
    f.write_text(json.dumps(raw))
    original = f.read_text()
    m = S.Mapper(f)
    assert len(m.set.surfaces) == 2 and m.set.backup_first and not list(tmp_path.glob("*.bak")), "loading never writes"
    assert m.command(cmd("save"))
    baks = list(tmp_path.glob("surfaces.json.*.bak"))
    assert len(baks) == 1 and baks[0].read_text() == original and len(json.loads(f.read_text())["surfaces"]) == 2
    m.command(cmd("corner", id=1, i=0, x=210, y=310))
    assert m.command(cmd("save")) and len(list(tmp_path.glob("*.bak"))) == 1, "one copy, not one per save"
    clean = S.Mapper(f)
    assert not clean.set.backup_first and clean.command(cmd("save")) is True and len(list(tmp_path.glob("*.bak"))) == 1


def test_corrupt_file_that_cannot_be_moved_aside_is_copied_not_overwritten(tmp_path, monkeypatch):
    """Review L2: if the quarantine rename fails, the first save must not destroy the only copy."""
    f = tmp_path / "surfaces.json"
    f.write_text("{corrupt")
    monkeypatch.setattr(S, "_quarantine", lambda p: None)
    m = S.Mapper(f)
    assert m.set.status == "quarantined" and m.set.backup_first and f.read_text() == "{corrupt"
    monkeypatch.setattr(S, "_backup", lambda p: False)
    m.command(cmd("corner", id=1, i=0, x=8, y=8))
    assert m.save() is False and f.read_text() == "{corrupt" and m.save_failed, "no copy, no overwrite (and it retries)"
    monkeypatch.undo()
    monkeypatch.setattr(S, "_quarantine", lambda p: None)
    assert m.save() is True and not m.save_failed
    baks = list(tmp_path.glob("surfaces.json.*.bak"))
    assert len(baks) == 1 and baks[0].read_text() == "{corrupt" and json.loads(f.read_text())["version"] == 1


def test_glow_shader_has_no_pow_of_a_negative_base():
    """Review L3 (source level): pow(x, 2.0) with x < 0 is undefined in GLSL."""
    src = (ROOT / "surface_render.py").read_text(encoding="utf-8")
    fs = src[src.index('FS = """'):src.index('def _font')]
    assert "pow(" not in fs and "exp(-kc * kc)" in fs and "exp(-k2 * k2)" in fs


def test_name_tags_of_deleted_surfaces_are_released_and_the_font_is_cached():
    """Review L6 (no GL: the renderer's bookkeeping with stand-in textures)."""
    pytest.importorskip("moderngl")
    import surface_render as SR

    class Tex:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    r = SR.SurfaceRenderer.__new__(SR.SurfaceRenderer)
    a, b, c = Tex(), Tex(), Tex()
    r._labels = {1: ("k", a, (10, 10)), 2: ("k", b, (10, 10)), 7: ("k", c, (10, 10))}
    r._prune({1, 7})
    assert sorted(r._labels) == [1, 7] and b.released and not a.released and not c.released
    assert SR._font(34) is SR._font(34), "the TrueType file is opened once, not for every name tag"


def test_a_full_layout_import_fits_in_one_request():
    """Review L10: 24 surfaces with 8-point masks as one surf `replace`."""
    import remote_control
    st = S.SurfaceSet()
    for _ in range(S.MAX_SURFACES - 1):
        st.apply(cmd("add"))
    for s in st.surfaces:
        s["name"] = "W" * S.MAX_NAME
        st.apply(cmd("mask", id=s["id"], pts=[[400.123456, 300.123456], [700.123456, 310.123456],
                 [820.123456, 400.123456], [800.123456, 520.123456], [700.123456, 610.123456], [520.123456, 640.123456],
                 [410.123456, 560.123456], [380.123456, 420.123456]]))
    assert all(s["shape"] == "poly" and len(s["mask"]) == 8 for s in st.surfaces)
    body = json.dumps([{"cmd": "surf", "op": "replace", "surfaces": st.to_dict()["surfaces"]}])
    assert len(body.encode()) < remote_control.MAX_BODY
    c = remote_control.validate_command(json.loads(body)[0])
    fresh = S.SurfaceSet()
    assert fresh.apply(c) and len(fresh.surfaces) == S.MAX_SURFACES
