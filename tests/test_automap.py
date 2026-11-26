"""AUTO MAP: synthetic warps only (no projector, no webcam, no GPU)."""
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
cv2 = pytest.importorskip("cv2")
import automap as am  # noqa: E402
import surfaces  # noqa: E402

PW, PH = 1280, 800
CW, CH = 640, 360
# projector px -> camera px: the projection fills the middle of the camera view, a little keystoned
SRC = np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])
DST = np.float32([[110, 40], [540, 55], [520, 320], [125, 305]])
H_PC = cv2.getPerspectiveTransform(SRC, DST).astype(np.float64)        # projector -> camera


class Rig:
    """Fake projector + camera: a still = ambient + reflectance x (the shown pattern warped into the camera) + noise."""

    def __init__(self, H=H_PC, faces=(), seed=0, H_right=None):
        self.H, self.H_right = H, H_right
        self.rng = np.random.default_rng(seed)
        self.refl = np.full((CH, CW), 0.55, np.float32)                 # the wall
        for poly, r in faces:                                          # brighter box faces (camera px)
            m = np.zeros((CH, CW), np.uint8)
            cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 1)
            self.refl[m > 0] = r
        self.pattern = None
        self.drawn = 0
        self.lock = threading.Lock()

    def show(self, img):
        with self.lock:
            if img is not None:
                self.pattern = img
                self.drawn += 1

    def shown_id(self):
        return self.drawn

    def warp(self, img):
        out = cv2.warpPerspective(img.astype(np.float32), self.H, (CW, CH), flags=cv2.INTER_AREA)
        if self.H_right is not None:                                   # a folded wall: right half uses another H
            r = cv2.warpPerspective(img.astype(np.float32), self.H_right, (CW, CH), flags=cv2.INTER_AREA)
            out[:, CW // 2:] = r[:, CW // 2:]
        return out

    def still(self, skip=2):
        lit = self.warp(self.pattern)
        img = 12.0 + self.refl * lit * 1.3 + self.rng.normal(0, 2.0, (CH, CW))
        return np.clip(cv2.GaussianBlur(img, (3, 3), 0), 0, 255).astype(np.uint8)


def _caps(rig, seq):
    out = {}
    for k, img in seq:
        rig.show(img)
        out[k] = rig.still().astype(np.float32)
    return out


def test_gray_code_recovers_the_homography():
    rig = Rig()
    px, py, ok = am.gray_decode(_caps(rig, am.gray_sequence(PW, PH)), PW, PH)
    cal = am.calib_from_gray(px, py, ok, (CW, CH), (PW, PH))
    Hcp = np.linalg.inv(H_PC)
    pts = np.array([[200, 100], [400, 180], [480, 280]], float)
    assert np.abs(cal.cam_to_proj(pts) - am.apply_h(Hcp, pts)).max() < 4.0     # one stripe = 4 projector px
    assert ok.mean() > 0.3


def test_gray_code_dense_map_handles_a_folded_wall():
    Hr = H_PC.copy()
    Hr[0, 2] += 25.0                                                    # right half shifted 25 camera px
    rig = Rig(H_right=Hr)
    px, py, ok = am.gray_decode(_caps(rig, am.gray_sequence(PW, PH)), PW, PH)
    cal = am.calib_from_gray(px, py, ok, (CW, CH), (PW, PH))
    for cx, Hx in ((200, H_PC), (450, Hr)):
        p = np.array([[cx, 180.0]])
        truth = am.apply_h(np.linalg.inv(Hx), p)
        assert np.abs(cal.cam_to_proj(p) - truth).max() < 5.0            # dense map: right on both halves


def test_aruco_calibration_and_refusal():
    rig = Rig()
    board, corners = am.aruco_board(PW, PH)
    rig.show(board)
    found = am.detect_aruco(rig.still())
    cal = am.calib_from_aruco(corners, found, (CW, CH), (PW, PH))
    assert cal.rms < 2.0
    p = np.array([[300, 200.0]])
    assert np.abs(cal.cam_to_proj(p) - am.apply_h(np.linalg.inv(H_PC), p)).max() < 3.0
    few = {k: v for k, v in list(found.items())[:3]}
    with pytest.raises(am.CalibError, match="only 3 of 24 markers"):
        am.calib_from_aruco(corners, few, (CW, CH), (PW, PH))
    part = am.calib_from_aruco(corners, {k: v for k, v in list(found.items())[:5]}, (CW, CH), (PW, PH))
    assert part.warning and "extrapolated" in part.warning and part.coverage < 0.75


def test_calibration_save_load_roundtrip(tmp_path):
    cal = am.Calibration(np.eye(3) * 2, 0.7, (CW, CH), (PW, PH), "aruco", dense=None, seen="x")
    cal.save(tmp_path)
    back = am.Calibration.load(tmp_path)
    assert back.method == "aruco" and np.allclose(back.H, cal.H) and back.dense is None
    assert not list(tmp_path.glob("*.png")) and not list(tmp_path.glob("*.jpg"))     # numbers only


BOX_A = [[230, 120], [330, 125], [328, 230], [232, 226]]
BOX_B = [[360, 150], [440, 146], [442, 250], [362, 252]]


def _scene_rig():
    return Rig(faces=[(BOX_A, 1.0), (BOX_B, 0.85)])


def test_candidates_and_click_to_pick_find_the_box_faces():
    rig = _scene_rig()
    rig.show(np.zeros((PH, PW), np.uint8)); black = rig.still()
    rig.show(np.full((PH, PW), 100, np.uint8)); grey = rig.still()
    c = am.candidates(grey, black)
    quads = [f["poly"] for f in c if f["quad"]]

    def near(box):
        return any(np.abs(am.order_quad(np.array(box, float)) - q).max() < 8 for q in quads)
    assert near(BOX_A) and near(BOX_B)
    f = am.pick_face(grey, black, 280, 175)
    assert f is not None and f["quad"] and np.abs(f["poly"] - am.order_quad(np.array(BOX_A, float))).max() < 8
    assert am.pick_face(grey, black, 5, 5) is None


def test_ai_polygons_are_validated():
    ok = am.validate_polys([{"points": BOX_A, "name": "front"}], (CW, CH))
    assert ok[0]["quad"] and ok[0]["name"] == "front"
    for bad in ([], [{"points": [[0, 0], [1, 1]]}], [{"points": [[0, 0], [700, 0], [700, 100], [0, 100]]}],
                [{"points": [[100, 100], [200, 200], [200, 100], [100, 200]]}], [{"points": [[0, 0], [1, 0], [1, 1]]}],
                [{"points": [[0, 0], [float("nan"), 0], [5, 5]]}]):
        with pytest.raises(ValueError):
            am.validate_polys(bad, (CW, CH))


def test_to_surfaces_replaces_a_lone_full_frame_and_keeps_real_ones():
    cal = am.Calibration(np.linalg.inv(H_PC), 0.5, (CW, CH), (PW, PH), "aruco")
    found = am.validate_polys([{"points": BOX_A}, {"points": [[400, 100], [450, 100], [470, 150], [430, 190], [390, 150]]}],
                              (CW, CH))
    full = [surfaces.new_surface(1, (PW, PH))]
    new, ids = am.to_surfaces(found, cal, full, (PW, PH), ["Front"])
    assert ids == [1, 2] and new[0]["name"] == "Front" and new[1]["shape"] == "poly" and len(new[1]["mask"]) == 5
    truth = am.apply_h(np.linalg.inv(H_PC), am.order_quad(np.array(BOX_A, float)))
    assert np.abs(np.array(new[0]["corners"]) - truth).max() < 1e-6
    real = [surfaces.new_surface(1, (PW, PH), "Mine", [[10, 10], [500, 10], [500, 400], [10, 400]])]
    new, ids = am.to_surfaces(found[:1], cal, real, (PW, PH))
    assert [s["name"] for s in new] == ["Mine", "Auto 1"] and ids == [2]
    ss = surfaces.SurfaceSet((PW, PH))
    assert ss.apply(surfaces.validate_cmd({"op": "replace", "surfaces": new}, (PW, PH)))   # the engine command works


def test_refine_converges_with_a_biased_calibration():
    """The projected outline and the physical edge meet within REFINE_TOL even though the calibration is off."""
    bias = np.array([[1, 0, 9.0], [0, 1, -6.0], [0, 0, 1]])
    cal = am.Calibration(np.linalg.inv(H_PC) @ np.linalg.inv(bias), 1.0, (CW, CH), (PW, PH), "aruco")   # wrong by ~10 cam px
    rig = _scene_rig()
    rig.show(np.full((PH, PW), 100, np.uint8)); grey = rig.still()
    rig.show(np.zeros((PH, PW), np.uint8)); black = rig.still().astype(np.float32)
    phys = am.order_quad(np.array(BOX_A, float))
    q = cal.cam_to_proj(phys)                                            # where the biased calibration puts it
    target = am.physical_quad(grey, cal.proj_to_cam(q))
    assert target is not None and np.abs(target - phys).max() < 2.5
    errs = []
    for _ in range(4):
        rig.show(am.outline_image([q], (PW, PH)))
        seen = am.projected_quad(np.clip(rig.still() - black, 0, 255), cal.proj_to_cam(q))
        q, e = am.refine_step(q, cal, target, seen)
        errs.append(e)
    assert errs[0] > 6 and errs[-1] < 1.5


def test_wizard_end_to_end_on_the_fake_rig(tmp_path):
    rig = _scene_rig()
    ss = surfaces.SurfaceSet((PW, PH))
    lock = threading.Lock()

    def surf(c):
        with lock:
            ss.apply(surfaces.validate_cmd(c, (PW, PH)))
    w = am.AutoMap(rig.show, rig.shown_id, rig.still, lambda: ([dict(s) for s in ss.surfaces], (PW, PH)), surf,
                   calib_dir=tmp_path, hold=0.0)

    def run(fn, *a):
        assert fn(*a) is not False
        w._thread.join(timeout=120)
        assert not w.busy and not w.err, w.err
    run(w.check)
    assert "whole projection" in w.msg
    run(w.calibrate, True)
    assert w.calib.method == "aruco+graycode" and (tmp_path / "usb" / "pose1" / "calibration.json").is_file()
    run(w.detect)
    assert len(w.found) >= 2
    assert w.create(keep=[0, 1], names=["A", "B"])
    assert [s["name"] for s in ss.surfaces] == ["A", "B"]
    n_undo = len(ss._undo) if hasattr(ss, "_undo") else None
    run(w.verify)
    assert w.report and all(r["after"] is not None and r["after"] < 2.0 for r in w.report)
    assert w.image_jpeg()[:2] == b"\xff\xd8"
    st = w.state()
    assert st["calib"]["dense"] and len(st["found"]) == len(w.found)
    if n_undo is not None:
        assert len(ss._undo) - n_undo <= 1                              # the whole refinement is one undo step


def test_wizard_refuses_when_the_camera_sees_nothing(tmp_path):
    class Blind(Rig):
        def still(self, skip=2):
            return np.full((CH, CW), 12, np.uint8)
    rig = Blind()
    w = am.AutoMap(rig.show, rig.shown_id, rig.still, lambda: ([], (PW, PH)), lambda c: None, calib_dir=tmp_path, hold=0.0)
    w.check(); w._thread.join(10)
    assert "does not see the projection" in w.err
    w.calibrate(); w._thread.join(10)
    assert "markers seen" in w.err


def test_patterns_are_photosensitive_safe():
    seq = am.gray_sequence(PW, PH)
    assert max(int(img.max()) for _, img in seq) <= am.PATTERN_LEVEL < 255
    assert am.HOLD_S >= 0.5                                              # <= 2 pattern changes per second


def test_validate_command():
    assert am.validate_command({"op": "pick", "x": 0.5, "y": 0.2}) == {"cmd": "automap", "type": "automap", "op": "pick", "x": 0.5, "y": 0.2}
    assert am.validate_command({"op": "create", "keep": [0, 2], "names": ["a", 3]})["names"] == ["a", ""]
    for bad in ({"op": "pick", "x": 2, "y": 0}, {"op": "zap"}, {"op": "create", "keep": "all"}, {"op": "drop", "i": -1}):
        with pytest.raises(ValueError):
            am.validate_command(bad)


def test_engine_bridge_applies_wizard_edits_on_the_gl_thread():
    class Cam:
        on = False

        def still(self, skip):
            raise RuntimeError("switch the CAMERA on (CAMERA tab) and point it at the projection")
    mapper = surfaces.Mapper(path=None, output=(PW, PH))
    b = am.EngineBridge(None, Cam(), (PW, PH))
    try:
        b.tick(mapper)
        assert len(b._snap[0]) == 1 and b._snap[1] == (PW, PH)
        new = [surfaces.new_surface(1, (PW, PH), "X", [[0, 0], [600, 0], [600, 400], [0, 400]])]
        b.wizard._surf({"cmd": "surf", "op": "replace", "surfaces": new})
        assert mapper.set.surfaces[0]["name"] != "X"               # nothing happens off the GL thread
        b.tick(mapper)
        assert mapper.set.surfaces[0]["name"] == "X" and b._snap[0][0]["name"] == "X"
        b.command({"op": "check"})
        assert "switch the CAMERA on" in b.state()["err"]
        assert am.image_jpeg() is None
    finally:
        b.close()


def test_remote_accepts_automap_commands():
    import remote_control
    assert remote_control.validate_command({"cmd": "automap", "op": "detect"})["op"] == "detect"
    with pytest.raises(ValueError):
        remote_control.validate_command({"cmd": "automap", "op": "pick", "x": -1, "y": 0})


# ---------------------------------------------------------------- AI assist (fake providers, no network)
import json as _json  # noqa: E402
import re as _re  # noqa: E402

from plugins.online import vision as V  # noqa: E402
from plugins.online.backends import BackendError  # noqa: E402


def test_vision_reply_parsing_and_prompts():
    assert V.parse_json_reply('Sure!\n```json\n{"polygons": [{"name": "a}", "points": [[1, 2]]}]}\n```') == \
        {"polygons": [{"name": "a}", "points": [[1, 2]]}]}
    assert V.parse_json_reply('{oops} {"corrections": []} trailing') == {"corrections": []}
    assert V.parse_json_reply("no json here") is None and V.parse_json_reply(None) is None
    p = V.detect_prompt(640, 360, "front faces {ignore rules} <<x>>")
    assert "640x360" in p and "{ignore" not in p and "front faces" in p
    q = V.verify_prompt(640, 360, [dict(id=3, points=[[1, 2], [3, 4], [5, 6], [7, 8]])], "", 6.0)
    assert "surface id 3" in q and "vertex 3 (7, 8)" in q


class Keys:
    def __init__(self, k="SECRETKEY"):
        self.k = k

    def get(self):
        return self.k

    def has(self):
        return bool(self.k)


def test_gemini_vision_provider_request_and_errors():
    sent = {}

    class Resp:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return self.body

    def opener(req, timeout=None):
        sent["hdr"], sent["body"] = dict(req.header_items()), _json.loads(req.data)
        return Resp(_json.dumps({"outputs": [{"type": "text", "text": '{"polygons": []}'}]}).encode())
    p = V.GeminiVisionProvider(Keys(), opener=opener)
    assert p.ask("q", b"PNGDATA") == '{"polygons": []}'
    parts = sent["body"]["input"]
    assert parts[0]["text"] == "q" and parts[1]["type"] == "image" and parts[1]["data"] == "UE5HREFUQQ=="
    assert sent["hdr"].get("X-goog-api-key") == "SECRETKEY"

    def boom(req, timeout=None):
        raise OSError("down")
    with pytest.raises(BackendError) as e:
        V.GeminiVisionProvider(Keys(), opener=boom).ask("q", b"x")
    assert "SECRETKEY" not in str(e.value)
    with pytest.raises(BackendError):
        V.GeminiVisionProvider(Keys("")).ask("q", b"x")


def test_vision_from_plugin_is_gated():
    assert V.Vision.from_plugin(None) is None

    class Plug:
        keys, cli_google, root, log = Keys(), False, Path("."), None
    assert [p.name for p in V.Vision.from_plugin(Plug()).providers] == ["gemini-api"]
    Plug.cli_google = True                                    # "Allow Google CLI tools": agy joins, after the API route
    assert [p.name for p in V.Vision.from_plugin(Plug()).providers] == ["gemini-api", "agy"]
    assert hasattr(V, "AgyVisionProvider")


class FakeProv:
    name = "fake"

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def available(self):
        return True

    def ask(self, prompt, png, work=None, cancel=None):
        self.calls.append((prompt, png))
        return self.fn(prompt, png)


def _ai_wizard(tmp_path, reply, online=True, person=None):
    rig = _scene_rig()
    ss = surfaces.SurfaceSet((PW, PH))

    def surf(c):
        ss.apply(surfaces.validate_cmd(c, (PW, PH)))
    prov = FakeProv(reply)
    vis = V.Vision([prov])
    w = am.AutoMap(rig.show, rig.shown_id, rig.still, lambda: ([dict(s) for s in ss.surfaces], (PW, PH)), surf,
                   calib_dir=tmp_path, hold=0.0, vision=(lambda: vis) if online else (lambda: None),
                   person=lambda: (person, True))                     # MediaPipe segmentation present
    return w, prov, ss, rig


def _wait(w):
    if w._thread is not None:
        w._thread.join(timeout=120)
    assert not w.busy


def test_ai_detect_needs_online_shows_the_exact_picture_and_adds_validated_candidates(tmp_path):
    good = {"polygons": [{"name": "front A", "points": BOX_A},
                         {"name": "bad", "points": [[0, 0], [2000, 0], [5, 5]]}]}
    w, prov, ss, rig = _ai_wizard(tmp_path, lambda p, png: "Here you go: " + _json.dumps(good), online=False)
    assert not w.ai_prepare("detect") and "needs ONLINE" in w.err
    w, prov, ss, rig = _ai_wizard(tmp_path, lambda p, png: "Here you go: " + _json.dumps(good))
    w.calibrate(False); _wait(w)
    w.detect(); _wait(w)
    n0 = len(w.found)
    assert w.ai_prepare("detect", "the front faces")
    st = w.state()["ai"]
    assert st["online"] and st["pending"]["kind"] == "detect" and not prov.calls      # nothing sent yet
    assert w.image[0] == "ai_send"
    assert w.ai_send(); _wait(w)
    assert "the front faces" in prov.calls[0][0]
    shown = cv2.imdecode(np.frombuffer(prov.calls[0][1], np.uint8), cv2.IMREAD_GRAYSCALE)
    assert (shown == np.clip(w.scene[0], 0, 255).astype(np.uint8)).all()        # what was shown is what was sent
    new = w.found[n0:]
    assert len(new) == 1 and new[0]["ai"] and new[0]["name"] == "front A" and "refused" in w.msg
    assert w.state()["found"][-1]["ai"] is True


def test_ai_detect_redacts_people_and_needs_confirmation(tmp_path):
    person = np.zeros((90, 160), np.float32)
    person[10:60, 20:40] = 1.0                                              # someone on the left
    w, prov, ss, rig = _ai_wizard(tmp_path, lambda p, png: '{"polygons": []}', person=person)
    w.calibrate(False); _wait(w)
    w.detect(); _wait(w)
    assert w.ai_prepare("detect")
    pend = w.state()["ai"]["pending"]
    assert pend["person"] and pend["redacted"] > 0
    assert not w.ai_send() and "send anyway" in w.err and not prov.calls
    assert w.ai_send(confirm_person=True); _wait(w)
    img = cv2.imdecode(np.frombuffer(prov.calls[0][1], np.uint8), cv2.IMREAD_GRAYSCALE)
    region = img[40:200, 90:150]                                            # the person area is one flat grey
    assert int(region.max()) - int(region.min()) == 0
    w.ai_prepare("detect"); w.ai_discard()
    assert w.ai is None and "nothing was sent" in w.msg


def _made_with_bias(w, ss):
    """Surfaces created from a biased calibration (corners ~10 cam px off), no classical refine."""
    bias = np.array([[1, 0, 9.0], [0, 1, -6.0], [0, 0, 1]])
    w.calib = am.Calibration(np.linalg.inv(H_PC) @ np.linalg.inv(bias), 1.0, (CW, CH), (PW, PH), "aruco")
    w.found = am.validate_polys([{"points": BOX_A}], (CW, CH))
    assert w.create()


def _truth_corrections(prompt, png, sign=1.0):
    """What a good model would answer: the listed outline corners -> the real box corners (camera px).
    The listed corners are where the BIASED calibration believes the outline is; the projector really draws it
    elsewhere, so this is a plausible, imperfect answer (it does not know the bias either)."""
    pts = [(float(a), float(b)) for a, b in _re.findall(r"vertex \d \(([-\d.]+), ([-\d.]+)\)", prompt)]
    sid = int(_re.search(r"surface id (\d+)", prompt).group(1))
    # the true outline position in the camera = H_PC applied to the projector quad = belief + bias offset
    bias = np.array([9.0, -6.0])
    phys = am.order_quad(np.array(BOX_A, float))
    out = []
    for i, p in enumerate(pts):
        d = (phys[i] - (np.array(p) - bias)) * sign
        out.append({"id": sid, "vertex": i, "dx": float(d[0]), "dy": float(d[1])})
    return _json.dumps({"corrections": out})


def test_ai_verify_applies_bounded_corrections_as_one_undo_step(tmp_path):
    w, prov, ss, rig = _ai_wizard(tmp_path, lambda p, png: _truth_corrections(p, png))
    _made_with_bias(w, ss)
    before = np.array(ss.surfaces[0]["corners"])
    n_undo = len(ss._undo)
    assert w.ai_prepare("verify", ""); _wait(w)
    assert w.ai is not None and w.ai["kind"] == "verify" and w.image[0] == "ai_send"
    assert w.ai_send(); _wait(w)
    assert not w.err, w.err
    after = np.array(ss.surfaces[0]["corners"])
    r = w.report[0]
    assert r["ai"] and r["kept"] and r["after"] < r["before"], r
    step = am.apply_h(H_PC, after) - am.apply_h(H_PC, before)
    assert np.linalg.norm(step, axis=1).max() <= am.AI_MAX_NUDGE + 0.6           # bounded per press (camera px)
    assert len(ss._undo) - n_undo == 1


def test_ai_verify_reverts_when_the_error_gets_worse(tmp_path):
    w, prov, ss, rig = _ai_wizard(tmp_path, lambda p, png: _truth_corrections(p, png, sign=-1.0))
    _made_with_bias(w, ss)
    before = [[list(p) for p in s["corners"]] for s in ss.surfaces]
    w.ai_prepare("verify"); _wait(w)
    w.ai_send(); _wait(w)
    assert w.report and not w.report[0]["kept"] and "reverted" in w.msg
    assert [[list(p) for p in s["corners"]] for s in ss.surfaces] == before


def test_ai_replies_are_clamped_and_filtered():
    shapes = [dict(id=1, points=[[0, 0], [1, 0], [1, 1], [0, 1]])]
    c = am.ai_corrections({"corrections": [{"id": 1, "vertex": 0, "dx": 100, "dy": 0}, {"id": 1, "vertex": 0, "dx": 1, "dy": 1},
                                           {"id": 9, "vertex": 0, "dx": 1, "dy": 1}, {"id": 1, "vertex": 7, "dx": 1, "dy": 1},
                                           {"id": 1, "vertex": 2, "dx": "nan", "dy": 1}, "junk"]}, shapes)
    assert list(c) == [1] and np.allclose(c[1][0], [am.AI_MAX_NUDGE, 0]) and np.allclose(c[1][1:], 0)
    with pytest.raises(am.CalibError):
        am.ai_corrections({"nope": 1}, shapes)
    good, bad = am.ai_polygons({"polygons": [{"points": BOX_A}] * 15}, (CW, CH))
    assert len(good) == am.AI_MAX_POLYS and any("ignored" in b for b in bad)


def test_ai_commands_validate():
    assert am.validate_command({"op": "ai_prepare", "kind": "detect", "instruction": "boxes"})["instruction"] == "boxes"
    assert am.validate_command({"op": "ai_send", "confirm_person": "yes"})["confirm_person"] is False
    with pytest.raises(ValueError):
        am.validate_command({"op": "ai_prepare", "kind": "everything"})


def test_vision_rate_limit_and_fallthrough():
    bad = FakeProv(lambda p, png: "not json")
    good = FakeProv(lambda p, png: '{"polygons": []}')
    v = V.Vision([bad, good])
    assert v.ask("detect", "q", b"x") == ({"polygons": []}, "fake")
    for _ in range(V.CALL_LIMIT[0] - 1):
        v.ask("detect", "q", b"x")
    with pytest.raises(BackendError, match="limit"):
        v.ask("detect", "q", b"x")


# ---------------------------------------------------------------- built-in webcam: mirrored, wide-angle, low-res, noisy
BW, BH = 480, 270                                                     # a low-res built-in webcam
K1 = -0.20                                                            # barrel distortion


class WebcamRig(Rig):
    """Oblique, partial, barrel-distorted, mirrored, noisy camera. dst = where the projector corners land (pinhole)."""

    def __init__(self, dst, mirrored=True, k1=K1, noise=5.0, gain=1.0, seed=3):
        Hp = cv2.getPerspectiveTransform(SRC, np.float32(dst)).astype(np.float64)
        super().__init__(H=Hp, seed=seed)
        self.mirror, self.k1, self.noise, self.gain = mirrored, k1, noise, gain
        self.refl = np.full((BH, BW), 0.6, np.float32)
        K = am.lens_K((BW, BH))
        ys, xs = np.mgrid[0:BH, 0:BW]
        und = am.undistort_pts(np.c_[xs.ravel(), ys.ravel()], K, (k1, 0.0)) if k1 else np.c_[xs.ravel(), ys.ravel()]
        self.pad = 200                                                # the lens sees beyond the pinhole frame
        self.mapx = (und[:, 0].reshape(BH, BW) + self.pad).astype(np.float32)
        self.mapy = (und[:, 1].reshape(BH, BW) + self.pad).astype(np.float32)
        T = np.array([[1, 0, self.pad], [0, 1, self.pad], [0, 0, 1]], float)
        self.Hpad = T @ Hp
        self.reflp = np.full((BH + 2 * self.pad, BW + 2 * self.pad), 0.6, np.float32)

    def still(self, skip=2):
        lit = cv2.warpPerspective(self.pattern.astype(np.float32), self.Hpad, (BW + 2 * self.pad, BH + 2 * self.pad),
                                  flags=cv2.INTER_AREA)
        img = 10.0 + self.reflp * lit * self.gain
        img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)        # the lens
        img = img + self.rng.normal(0, self.noise, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img[:, ::-1].copy() if self.mirror else img

    def truth(self, cam_pts):
        """Unmirrored, distorted camera px -> projector px (ground truth)."""
        u = am.undistort_pts(cam_pts, am.lens_K((BW, BH)), (self.k1, 0.0)) if self.k1 else np.asarray(cam_pts, float)
        return am.apply_h(np.linalg.inv(self.H), u)


OBLIQUE_PARTIAL = [[70, 40], [800, 8], [780, 268], [85, 238]]       # right part of the projection falls off the frame
OBLIQUE_FULL = [[60, 35], [425, 22], [415, 245], [70, 232]]


def _wizard(rig, tmp_path, profile="builtin"):
    ss = surfaces.SurfaceSet((PW, PH))
    w = am.AutoMap(rig.show, rig.shown_id, rig.still, lambda: ([dict(s) for s in ss.surfaces], (PW, PH)),
                   lambda c: ss.apply(surfaces.validate_cmd(c, (PW, PH))), calib_dir=tmp_path, hold=0.0, profile=profile)
    return w, ss


def _go(w, fn, *a):
    assert fn(*a) is not False, w.err
    w._thread.join(timeout=180)
    assert not w.busy


def test_builtin_webcam_mirror_lens_and_accuracy_at_the_edges(tmp_path):
    rig = WebcamRig(OBLIQUE_FULL)
    w, ss = _wizard(rig, tmp_path)
    _go(w, w.check)
    assert not w.err, w.err
    assert w.mirrored is True and "mirrored" in w.msg
    assert w.adapt["stripe"] >= 8 and w.adapt["snr"] > 5
    _go(w, w.calibrate, True)
    assert not w.err, w.err
    c = w.calib
    assert c.mirrored and abs(c.dist[0] - K1) < 0.08, c.dist
    # accuracy across the visible projection, edges included (unmirrored distorted camera px)
    grid = np.array([[x, y] for x in np.linspace(80, 400, 6) for y in np.linspace(45, 225, 5)], float)
    inside = (grid[:, 0] > 0)
    err = np.linalg.norm(c.cam_to_proj(grid[inside], dense=False) - rig.truth(grid[inside]), axis=1)
    assert err.mean() < 3.0 and err.max() < 9.0, err                  # projector px (~3 camera px in the far corners)
    plain = am.Calibration(am.fit_h(*_corr(rig, c))[0], 0, (BW, BH), (PW, PH), "plain")
    err0 = np.linalg.norm(plain.cam_to_proj(grid, dense=False) - rig.truth(grid), axis=1)
    assert err0.max() > 2.5 * err.max() and err0.mean() > 2.5 * err.mean()   # the lens model makes the edges right
    assert (tmp_path / "builtin" / "pose1" / "calibration.json").is_file()
    again = am.AutoMap(rig.show, rig.shown_id, rig.still, lambda: ([], (PW, PH)), lambda c: None,
                       calib_dir=tmp_path, hold=0.0, profile="builtin")
    assert again.calib is not None and again.mirrored is True and again.calib.dist == c.dist
    usb = am.AutoMap(rig.show, rig.shown_id, rig.still, lambda: ([], (PW, PH)), lambda c: None,
                     calib_dir=tmp_path, hold=0.0, profile="usb")
    assert usb.calib is None                                         # profiles are kept apart


def _corr(rig, c):
    """Exact correspondences for a lens-free fit to compare against (unmirrored distorted cam px -> projector px)."""
    g = np.array([[x, y] for x in np.linspace(80, 400, 9) for y in np.linspace(45, 225, 7)], float)
    return g, rig.truth(g)


def test_partial_view_calibrates_with_a_warning_and_two_poses_cover_it(tmp_path):
    rig = WebcamRig(OBLIQUE_PARTIAL, mirrored=False)
    w, ss = _wizard(rig, tmp_path)
    _go(w, w.check)
    assert any("RIGHT" in a for a in w.framing["advice"])
    _go(w, w.calibrate, True)
    assert not w.err, w.err
    assert w.calib.warning and "extrapolated" in w.calib.warning and w.calib.coverage < 0.9
    one = am.union_coverage([w.calib], (PW, PH))
    assert w.set_pose(2)
    rig2 = WebcamRig([[-110, 30], [380, 15], [370, 255], [-95, 240]], mirrored=False, seed=4)   # turned: sees the right part
    w._still, w._show, w._shown_id = rig2.still, rig2.show, rig2.shown_id
    _go(w, w.calibrate, True)
    assert not w.err, w.err
    both = am.union_coverage(list(w.poses.values()), (PW, PH))
    assert one < 0.9 and both > 0.97 and "together" in w.msg
    assert (tmp_path / "builtin" / "pose2" / "calibration.json").is_file()


def test_dim_noisy_camera_gets_brighter_patterns_and_wider_stripes(tmp_path):
    rig = WebcamRig(OBLIQUE_FULL, mirrored=False, noise=14.0, gain=0.35)
    w, ss = _wizard(rig, tmp_path)
    _go(w, w.check)
    assert w.adapt["snr"] < 8 and w.adapt["level"] == am.LEVEL_MAX and w.adapt["stripe"] >= 8


def test_framing_advice_directions():
    lit = np.zeros((90, 160), bool)
    lit[0:50, 40:120] = True
    assert any("UP" in a for a in am.framing_advice(lit)[1])
    lit = np.zeros((90, 160), bool)
    lit[20:70, 100:160] = True
    assert any("RIGHT" in a for a in am.framing_advice(lit)[1])
    lit = np.zeros((90, 160), bool)
    lit[30:50, 70:90] = True
    assert "small" in am.framing_advice(lit)[1][0]
    lit = np.zeros((90, 160), bool)
    lit[10:80, 10:150] = True
    assert "good" in am.framing_advice(lit)[1][0]
    assert "does not see" in am.framing_advice(np.zeros((90, 160), bool))[1][0]


def test_lens_fit_is_zero_for_a_pinhole_camera():
    pts = np.array([[x, y] for x in np.linspace(20, 620, 12) for y in np.linspace(20, 340, 8)], float)
    dist, c0, c1 = am.fit_lens(pts, am.apply_h(np.linalg.inv(H_PC), pts) + np.random.default_rng(0).normal(0, 0.3, (96, 2)),
                               (CW, CH))
    assert dist == (0.0, 0.0)


def test_camera_profile_and_ir_refusal():
    class Cam:
        on, s, devices = True, {"device": 0}, [{"index": 0, "name": "Integrated Camera", "ir": False},
                                               {"index": 1, "name": "Logitech BRIO", "ir": False},
                                               {"index": 2, "name": "Integrated IR Camera", "ir": True}]
    c = Cam()
    assert am.camera_profile(c) == "builtin" and am._camera_ready(c) is None
    c.s = {"device": 1}
    assert am.camera_profile(c) == "usb"
    c.s = {"device": 2}
    assert "infrared" in am._camera_ready(c)
    c.devices, c.s = None, {"device": 0}
    assert am.camera_profile(c) == "builtin"


def test_webcam_commands_validate():
    assert am.validate_command({"op": "profile", "name": "builtin"})["name"] == "builtin"
    assert am.validate_command({"op": "pose", "n": 2})["n"] == 2
    assert am.validate_command({"op": "framing", "on": False})["on"] is False
    for bad in ({"op": "profile", "name": "ir"}, {"op": "pose", "n": 3}, {"op": "pose", "n": True}):
        with pytest.raises(ValueError):
            am.validate_command(bad)


# ---------------------------------------------------------------- regression tests for the Neon review (AUTO MAP side)
import time as _time  # noqa: E402


def test_degenerate_homographies_are_refused_not_propagated(tmp_path):
    Hv = np.array([[1, 0, 0], [0, 1, 0], [0.01, 0, 1.0]])          # vanishing line at x = -100
    out = am.apply_h(Hv, [[-100.0, 5.0], [-300.0, 5.0], [10.0, 5.0]])
    assert np.isnan(out[0]).all() and np.isnan(out[1]).all() and np.isfinite(out[2]).all()
    with pytest.raises(am.CalibError):
        am.Calibration(np.zeros((3, 3)), 0.5, (CW, CH), (PW, PH), "x")
    with pytest.raises(am.CalibError):
        am.Calibration(np.eye(3), float("nan"), (CW, CH), (PW, PH), "x")
    (tmp_path / "calibration.json").write_text('{"H": [[0,0,0],[0,0,0],[0,0,0]], "rms": 1, "cam": [640,360], '
                                               '"proj": [1280,800], "method": "x"}')
    assert am.Calibration.load(tmp_path) is None                   # a corrupt file never crashes the engine
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 600, (40, 2))
    dst = rng.uniform(0, 1200, (40, 2))                            # no plane at all: RANSAC would fit a lucky 4
    with pytest.raises(am.CalibError):
        am.fit_h(src, dst)


def test_concave_quads_names_and_server_side_ai_validation():
    concave = [[100, 100], [300, 100], [200, 150], [200, 300]]
    f = am.validate_polys([{"points": concave, "name": None}], (CW, CH))[0]
    assert not f["quad"] and f["poly"].tolist() == concave and f["name"] == ""
    assert am.validate_polys([{"points": BOX_A, "name": "   "}], (CW, CH))[0]["name"] == ""
    touch = [[100, 100], [300, 100], [100, 300], [300, 300]]      # bow-tie
    with pytest.raises(ValueError):
        am.validate_polys([{"points": touch}], (CW, CH))
    for bad in ({"op": "ai_polys", "polys": [{"points": [[0, 0], [1, "x"], [2, 2]]}]},
                {"op": "ai_polys", "polys": [{"points": [[0, 0], [float("nan"), 1], [2, 2]]}]},
                {"op": "ai_polys", "polys": []}, {"op": "ai_polys", "polys": [[[0, 0]] * 9]}):
        with pytest.raises(ValueError):
            am.validate_command(bad)
    ok = am.validate_command({"op": "ai_polys", "polys": [{"points": BOX_A, "name": None}]})
    assert ok["polys"][0]["name"] == ""
    cal = am.Calibration(np.linalg.inv(H_PC), 0.5, (CW, CH), (PW, PH), "aruco")
    many = am.validate_polys([{"points": BOX_A}] * 12, (CW, CH)) * 3
    new, ids = am.to_surfaces(many, cal, [], (PW, PH), [None, "  ", "x"])
    assert len(new) == surfaces.MAX_SURFACES and new[0]["name"] == "Auto 1" and new[2]["name"] == "x"
    ss = surfaces.SurfaceSet((PW, PH))
    assert ss.apply(surfaces.validate_cmd({"op": "replace", "surfaces": new}, (PW, PH)))


def test_create_refuses_with_a_reason_instead_of_claiming_success(tmp_path):
    w, prov, ss, rig = _ai_wizard(tmp_path, lambda p, png: "{}")
    w.calib = am.Calibration(np.linalg.inv(H_PC), 0.5, (CW, CH), (PW, PH), "aruco")
    Hv = np.array([[1, 0, 0], [0, 1, 0], [-0.004, 0, 1.0]])        # everything right of x=250 is past the vanishing line
    w.calib = am.Calibration(Hv, 0.5, (CW, CH), (PW, PH), "x")
    w.found = am.validate_polys([{"points": [[400, 100], [500, 100], [500, 200], [400, 200]]}], (CW, CH))
    assert not w.create() and "could" in w.err and w.made == []


def test_busy_wizard_refuses_edits_and_pictures_are_dropped_when_the_camera_goes_off(tmp_path):
    w, prov, ss, rig = _ai_wizard(tmp_path, lambda p, png: "{}")
    w.calibrate(False); _wait(w)
    w.detect(); _wait(w)
    n = len(w.found)
    w.busy = True
    assert not w.create() and not w.add_polys([{"points": BOX_A}])
    w.command({"op": "drop", "i": 0})
    assert len(w.found) == n
    w.busy = False
    st = w.state()
    st["report"].append("x"); st["adapt"]["stripe"] = 99
    assert w.report == [] and w.adapt["stripe"] != 99                # state() hands out copies

    class Cam:
        on = False
    b = am.EngineBridge(None, Cam(), (PW, PH))
    try:
        b.wizard.image, b.wizard.scene = ("scene", np.zeros((4, 4, 3), np.uint8), 1), (1, 2)
        mapper = surfaces.Mapper(path=None, output=(PW, PH))
        b.tick(mapper)
        assert b.wizard.image is None and b.wizard.scene is None and am.image_jpeg() is None
    finally:
        b.close()


def test_pattern_hold_counts_from_display_and_blackout_or_panic_cancel(tmp_path):
    """A renderer that only shows the pattern 0.3 s after it was asked: the hold starts when it is really up."""
    shown, asked = [0], []

    def show(img):
        if img is not None:
            asked.append(_time.monotonic())

    def shown_id():
        if asked and _time.monotonic() - asked[-1] > 0.3 and shown[0] < len(asked):
            shown[0] = len(asked)
        return shown[0]
    w = am.AutoMap(show, shown_id, lambda skip: np.zeros((CH, CW), np.uint8), lambda: ([], (PW, PH)), lambda c: None,
                   calib_dir=tmp_path, hold=0.4)
    t0 = _time.monotonic()
    w._project(np.zeros((PH, PW), np.uint8))
    assert _time.monotonic() - t0 >= 0.3 + 0.4 * am.PROFILES["usb"]["settle"] - 0.02

    class Ctl:
        panics = 0

    class Cam:
        on = True
    ctl = Ctl()
    b = am.EngineBridge(None, Cam(), (PW, PH), ctl)
    try:
        b.wizard._shown_id = lambda: 10 ** 9                       # pretend every pattern is drawn at once
        b.wizard.hold = 0.2
        b.wizard._still = lambda skip: np.zeros((CH, CW), np.uint8)
        b.wizard.calibrate(True)
        mapper = surfaces.Mapper(path=None, output=(PW, PH))
        ctl.panics = 1
        b.tick(mapper)
        b.wizard._thread.join(10)
        assert not b.wizard.busy and "PANIC" in b.wizard.err + "cancelled"
        b.wizard.calibrate(True)
        b.tick(mapper, suppress=True)                             # blackout held
        b.wizard._thread.join(10)
        assert not b.wizard.busy
        assert b.draw(None, (PW, PH), suppress=True) is False
    finally:
        b.close()


def test_refine_with_a_scale_bias_never_diverges():
    """Calibration wrong by a scale about the box (a face nearer than the wall): damped steps converge."""
    S = np.array([[1.06, 0, -17.0], [0, 1.06, -10.0], [0, 0, 1]])
    cal = am.Calibration(np.linalg.inv(H_PC) @ S, 1.0, (CW, CH), (PW, PH), "aruco")
    rig = _scene_rig()
    rig.show(np.full((PH, PW), 100, np.uint8)); grey = rig.still().astype(np.float32)
    rig.show(np.zeros((PH, PW), np.uint8)); black = rig.still().astype(np.float32)
    phys = am.order_quad(np.array(BOX_A, float))
    q = cal.cam_to_proj(phys)
    target = am.physical_quad(np.clip(grey - black, 0, 255), cal.proj_to_cam(q))
    assert target is not None
    errs = []
    for _ in range(6):
        rig.show(am.outline_image([q], (PW, PH)))
        seen = am.projected_quad(np.clip(rig.still() - black, 0, 255), cal.proj_to_cam(q))
        q, e = am.refine_step(q, cal, target, seen)
        errs.append(e)
    assert all(b <= a + 0.3 for a, b in zip(errs, errs[1:])) and errs[-1] < 1.5, errs


def test_overlay_texture_uses_byte_alignment():
    import inspect
    assert "alignment=1" in inspect.getsource(am.AutoMapOverlay.draw)


# ---------------------------------------------------------------- PR-2 review (research/neon_pr2_review.md) regressions

def test_b5_cancel_takes_the_pattern_off_the_wall_at_once(tmp_path):
    shown = []
    w = am.AutoMap(lambda img: shown.append(img), lambda: 10 ** 9, lambda skip: (_time.sleep(3.0), np.zeros((CH, CW)))[1],
                   lambda: ([], (PW, PH)), lambda c: None, calib_dir=tmp_path, hold=0.0)
    w.check()                                             # the fake camera takes 3 s per still
    _time.sleep(0.2)
    a = _time.monotonic()
    w.cancel("panic")
    assert shown[-1] is None and _time.monotonic() - a < 0.05    # off the wall now, not when the step unwinds
    w._thread.join(10)
    assert "panic" in w.err


def test_b5_hold_is_interruptible_and_levels_are_capped(tmp_path):
    w = am.AutoMap(lambda img: None, lambda: 10 ** 9, lambda skip: np.zeros((CH, CW)), lambda: ([], (PW, PH)),
                   lambda c: None, calib_dir=tmp_path, hold=5.0)
    threading.Timer(0.2, w.cancel).start()
    a = _time.monotonic()
    with pytest.raises(am.CalibError):
        w._project(np.zeros((PH, PW), np.uint8))
    assert _time.monotonic() - a < 1.0
    assert max(int(img.max()) for _, img in am.gray_sequence(PW, PH, 4, 255)) <= am.LEVEL_MAX
    assert all(p["level"] <= am.LEVEL_MAX for p in am.PROFILES.values())


def test_b5_panic_clears_the_overlay_in_the_bridge():
    class Ov:
        def __init__(self):
            self.last = "x"

        def show(self, img):
            self.last = img

    class Ctl:
        panics = 0

    class Cam:
        on = True
    ctl = Ctl()
    b = am.EngineBridge(None, Cam(), (PW, PH), ctl)
    try:
        b.overlay = Ov()
        ctl.panics = 1
        b.tick(surfaces.Mapper(path=None, output=(PW, PH)))
        assert b.overlay.last is None
    finally:
        b.overlay = None
        b.close()


def test_b10_redaction_uses_the_mask_taken_with_the_photo_and_asks_without_segmentation(tmp_path):
    live = {"m": None}
    person = np.zeros((90, 160), np.float32)
    person[10:60, 20:40] = 1.0
    w, prov, ss, rig = _ai_wizard(tmp_path, lambda p, png: '{"polygons": []}')
    w._person = lambda: (live["m"], True)
    w.calibrate(False); _wait(w)
    live["m"] = person                                    # someone stands there while the photo is taken
    w.detect(); _wait(w)
    live["m"] = np.zeros((90, 160), np.float32)           # ...and walks away before AI is pressed
    assert w.ai_prepare("detect")
    assert w.ai["person"] and w.ai["redacted"] > 0        # still greyed out and still asks
    w.ai_discard()
    w._person = lambda: (np.zeros((90, 160), np.float32), False)   # motion fallback: cannot see people standing still
    w.detect(); _wait(w)
    assert w.ai_prepare("detect") and w.ai["person"]      # so it always asks


def test_b11_the_vision_call_limit_holds_across_requests():
    class Plug:
        keys, cli_google, root, log = Keys(), False, Path("."), None
    plug = Plug()
    v1, v2 = V.Vision.from_plugin(plug), V.Vision.from_plugin(plug)
    assert v1 is v2
    v1.providers = [FakeProv(lambda p, png: '{"polygons": []}')]
    for _ in range(V.CALL_LIMIT[0]):
        V.Vision.from_plugin(plug).ask("detect", "q", b"x")
    with pytest.raises(BackendError, match="limit"):
        V.Vision.from_plugin(plug).ask("detect", "q", b"x")


def test_sf1_escaped_quotes_inside_json_strings():
    txt = 'noise {"polygons": [{"name": "the \\"front\\" face", "points": [[1, 2], [3, 4], [5, 6]]}]} tail'
    assert V.parse_json_reply(txt)["polygons"][0]["name"] == 'the "front" face'
