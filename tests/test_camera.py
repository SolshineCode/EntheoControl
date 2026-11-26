"""Crowd camera: pure-python tests with fake frames / a fake worker (no webcam, no GPU)."""
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camera_layer as cl  # noqa: E402
import camera_worker as cw  # noqa: E402
import params  # noqa: E402


def test_wire_format_roundtrip_and_rejects_garbage():
    mask = (np.arange(cw.MASK_W * cw.MASK_H) % 256).astype(np.uint8).reshape(cw.MASK_H, cw.MASK_W)
    flow = np.random.default_rng(1).normal(size=(cw.FLOW_H, cw.FLOW_W, 2)).astype(np.float32)
    info = dict(motion=0.3, poses=[{"l_wrist": [0.1, 0.2, 0.9]}])
    r = cw.unpack(cw.pack(7, 12.5, mask, flow, info))
    assert r["seq"] == 7 and r["t"] == 12.5 and (r["mask"] == mask).all()
    assert np.allclose(r["flow"], flow, atol=2e-3) and r["info"] == info
    with pytest.raises(ValueError):
        cw.unpack(b"junk")
    with pytest.raises(ValueError):
        cw.unpack(cw.pack(1, 0, mask, flow, info)[:-3])


def test_energies_and_pool():
    fl = np.zeros((90, 160, 2), np.float32)
    e = cw.energies(fl, np.zeros((90, 160), np.uint8))
    assert e["motion"] == 0 and e["presence"] == 0
    fl[:, :53, 0] = 1.0                                   # only the left third moves
    m = np.zeros((90, 160), np.uint8); m[:, :80] = 255
    e = cw.energies(fl, m)
    assert e["left"] > 0.9 and e["centre"] < 0.05 and e["right"] == 0 and 0 < e["motion"] < e["left"]
    assert abs(e["presence"] - 0.5) < 1e-6
    p = cw.pool(fl, (18, 32))
    assert p.shape == (18, 32, 2) and abs(p[..., 0].mean() - fl[..., 0].mean()) < 1e-5


def test_ir_cameras_are_flagged():
    assert cw.IR_NAME.search("Integrated IR Camera") and not cw.IR_NAME.search("Integrated Camera")


def test_worker_analyse_with_fake_models():
    pytest.importorskip("cv2")
    w = cw.Worker("tok", models=None)
    seg = lambda rgb, ts: np.ones(rgb.shape[:2], np.float32) * 0.8
    pose = lambda rgb, ts, n: [{"l_wrist": [0.5, 0.5, 1.0]}]
    a = np.zeros((360, 640, 3), np.uint8)
    m, f, info, g, tm = w.analyse(a, 0, 0.0, seg, pose, True, [], None)
    b = np.roll(a, 4, axis=1); b[100:200, 100:200] = 255
    m, f, info, g, tm = w.analyse(b, 33, 1 / 30, seg, pose, True, [], g)
    assert m.shape == (cw.MASK_H, cw.MASK_W) and f.shape == (cw.FLOW_H, cw.FLOW_W, 2)
    assert info["presence"] == 1.0 and info["poses"][0]["l_wrist"][2] == 1.0 and set(tm) == {"seg", "flow", "pose"}


def test_worker_config_is_clamped():
    w = cw.Worker("tok")
    assert w.configure({"fps": 99, "res": -3, "max_poses": 9}) == dict(fps=30, res=0, pose_every=2, max_poses=3)
    with pytest.raises(ValueError):
        w.configure({"fps": "fast"})


def test_params_contract():
    for n in ("l_camera", "crowd_mode", "crowd_push", "crowd_swarm", "crowd_beings", "crowd_ripples", "crowd_serpent"):
        assert n in params.PARAMS
    assert params.PARAMS["l_camera"][2] == 0.0                 # off by default
    assert len(params.OPTIONS["crowd_mode"]) == params.PARAMS["crowd_mode"][1] + 1
    assert all(s in params.MOD_SOURCES for s in cl.SIGNALS)
    assert params.MOD_SOURCES.index("seq_gate") < params.MOD_SOURCES.index("crowd_motion")   # appended
    assert "l_camera" in params.bypass_cover()["camera"]


def test_settings_and_commands_validate():
    s = cl.clean_settings({"scale": 99, "crop": "junk", "mirror": "yes", "device": 3})
    assert s["scale"] == 4.0 and s["crop"] == [0, 0, 1, 1] and s["mirror"] is True and s["device"] == 3
    assert cl.validate_command({"op": "on"})["op"] == "on"
    c = cl.validate_command({"op": "set", "pos_x": 5, "mirror": False, "evil": 1})
    assert c == {"cmd": "camera", "type": "camera", "op": "set", "pos_x": 1.0, "mirror": False}
    for bad in ({"op": "rm"}, {"op": "set"}, {"op": "set", "mirror": 1}, {"op": "set", "crop": [0, 0, 1]},
                {"op": "set", "scale": float("nan")}):
        with pytest.raises(ValueError):
            cl.validate_command(bad)


def test_remote_validates_camera_commands():
    import remote_control
    assert remote_control.validate_command({"cmd": "camera", "op": "off"})["op"] == "off"
    with pytest.raises(ValueError):
        remote_control.validate_command({"cmd": "camera", "op": "nope"})


def test_alignment_maps_corners_and_mirrors():
    s = cl.clean_settings({})
    H = cl.align_matrix(s)
    u, v = cl.apply_h(H, 0.0, 0.0)                      # camera top-left, mirrored -> screen top-right
    assert abs(u - 1) < 1e-9 and abs(v - 1) < 1e-9
    s = cl.clean_settings({"mirror": False, "scale": 0.5, "pos_x": 0.5})
    H = cl.align_matrix(s)
    u, v = cl.apply_h(H, 0.5, 0.5)
    assert abs(u - 0.75) < 1e-9 and abs(v - 0.5) < 1e-9
    u, v = cl.apply_h(H, 0.0, 1.0)                      # bottom-left of camera -> bottom-left of the quad (y up)
    assert abs(u - 0.5) < 1e-9 and abs(v - 0.25) < 1e-9
    s = cl.clean_settings({"key_x": 0.6, "crop": [0.1, 0.2, 0.9, 0.8]})
    H = cl.align_matrix(s)
    for cam, scr in zip([(0.9, 0.2), (0.1, 0.2), (0.1, 0.8), (0.9, 0.8)], cl.screen_quad(s)):
        assert np.allclose(cl.apply_h(H, *cam), scr)


def _body(wrist_y=0.8, wrist_x=0.3, hip_x=0.5):
    # screen uv (y up)
    return {"l_shoulder": (0.45, 0.6), "r_shoulder": (0.55, 0.6), "l_hip": (hip_x - 0.04, 0.4),
            "r_hip": (hip_x + 0.04, 0.4), "l_wrist": (wrist_x, wrist_y), "r_wrist": (0.7, 0.45),
            "l_elbow": (0.4, 0.7)}


def test_signals_from_pose_and_slew_limits():
    sg = cl.Signals()
    sg.on_data(dict(motion=0.5, left=0.2, centre=0.1, right=0.0, presence=0.3), [_body()], 1 / 15)
    assert sg.target["hand_height"] == 1.0 and 0 < sg.target["hands_spread"] <= 1
    assert sg.target["crowd_motion"] == 0.5
    v = sg.step(1 / 60)["crowd_motion"]
    assert 0 < v < 0.1                                  # attack 0.10 s: no jump
    for _ in range(120):
        sg.step(1 / 60)
    assert abs(sg.v["crowd_motion"] - 0.5) < 1e-3
    sg.on_data(dict(motion=0.5), [_body(wrist_y=0.6, wrist_x=0.3)], 1 / 15)   # wrist moved -> speed
    assert sg.target["hand_speed"] > 0.3
    sg.lost()
    for _ in range(600):
        sg.step(1 / 60)
    assert sg.idle()


def test_signal_cannot_follow_a_fast_flicker():
    """A crowd signal toggling at 6 Hz (a strobing crowd) reaches a fader with a small swing only."""
    sg = cl.Signals()
    vals = []
    for i in range(600):
        if i % 5 == 0:
            on = (i // 5) % 2 == 0
            sg.target = dict.fromkeys(cl.SIGNALS, 1.0 if on else 0.0)
        vals.append(sg.step(1 / 60)["crowd_motion"])
    tail = vals[300:]
    assert max(tail) - min(tail) < 0.15


def test_ripple_gate_rate_limit():
    g = cl.RippleGate()
    out, t = [], 0.0
    hands = {"a": (0.2, 0.5)}
    g.feed(t, hands, 1 / 15, 1.0)
    for i in range(150):                                # 10 s of wild hand waving at 15 Hz data
        t += 1 / 15
        hands = {"a": (0.2 + 0.3 * (i % 2), 0.5), "b": (0.8 - 0.3 * (i % 2), 0.5)}
        out += [(t, p) for p in g.feed(t, hands, 1 / 15, 1.0)]
    assert 0 < len(out) <= 10 * cl.RIPPLE_HZ + 1
    assert all(b[0] - a[0] >= 1 / cl.RIPPLE_HZ - 1e-9 for a, b in zip(out, out[1:]))
    assert g.feed(t + 5, {"a": (0, 0)}, 1 / 15, 0.0) == []


def test_steer_offsets_follow_flee_and_stay_in_frame():
    pos = np.array([[0.0, 0.0], [0.7, 0.4]])
    off = np.zeros((2, 2))
    for _ in range(300):
        cl.steer_offsets(pos, [(0.3, 0.1)], off, 1.0, 1 / 60, 1.6)
    assert np.hypot(*(pos[0] + off[0] - (0.3, 0.1))) < 0.1
    assert (np.abs(pos + off) <= [0.75 + 1e-9, 0.45 + 1e-9]).all()
    off = np.zeros((2, 2))
    for _ in range(300):
        cl.steer_offsets(pos, [(0.05, 0.0)], off, -1.0, 1 / 60, 1.6)
    assert off[0, 0] < -0.1                              # fled away from the hand
    for _ in range(600):
        cl.steer_offsets(pos, [], off, 0.0, 1 / 60, 1.6)
    assert np.abs(off).max() < 1e-3


def test_arm_track_head_at_wrist():
    b = _body(wrist_y=0.9, wrist_x=0.4)
    ang, off, hx = cl.arm_track(b, 1.6)
    wx, wy = (0.4 - 0.5) * 1.6, 0.4
    ca, sa = math.cos(ang), math.sin(ang)
    assert abs((ca * wx + sa * wy) + 0.03 - hx) < 1e-9 and abs(-sa * wx + ca * wy - off) < 1e-9
    assert cl.arm_track({}, 1.6) is None


# ---------------------------------------------------------------- the layer with a fake worker


class FakeProc:
    pid = 4242

    def __init__(self):
        self.returncode, self.killed = None, False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.killed = True


class FakeLink:
    instances = []

    def __init__(self, port, token, unpack):
        self.q, self.status, self.error, self.debug, self.debug_want, self.cfg, self.quitted = [], {}, "", b"J", 0, [], False
        FakeLink.instances.append(self)

    def take(self):
        return self.q.pop(0) if self.q else None

    def configure(self, cfg):
        self.cfg.append(cfg)

    def quit(self):
        self.quitted = True


class FakeCtl:
    def __init__(self):
        self.ext, self.signals, self.ripples = {}, {}, []

    def ripple(self, x, y):
        self.ripples.append((x, y))


def _layer(tmp_path, clock):
    (tmp_path / ".remote_token").write_text("tok")
    (tmp_path / "camera_worker.py").write_text("#")
    ctl = FakeCtl()
    procs = []
    lay = cl.CameraLayer(None, ctl, here=tmp_path, spawn=lambda p, s, h: procs.append(FakeProc()) or procs[-1],
                         link_factory=FakeLink, clock=clock, sync=True)
    return lay, ctl, procs


def _result(seq, t, poses=()):
    m = np.zeros((cw.MASK_H, cw.MASK_W), np.uint8); m[20:70, 60:100] = 255
    return dict(seq=seq, t=t, mask=m, flow=np.zeros((cw.FLOW_H, cw.FLOW_W, 2), np.float32),
                info=dict(motion=0.6, left=0.1, centre=0.6, right=0.1, presence=0.2, poses=list(poses)))


def test_layer_off_by_default_costs_nothing(tmp_path):
    lay, ctl, procs = _layer(tmp_path, time.monotonic)
    assert not lay.on and lay.state()["status"] in ("off", "unavailable") and not procs
    for _ in range(10):
        lay.pre_update(1 / 60)
    assert lay.uniforms({"l_camera": 1.0}) == {"p_l_camera": 0.0, "p_crowd_push": 0.0}
    assert ctl.signals == {} and lay.uploads == 0


def test_layer_on_data_signals_off_releases(tmp_path):
    if not cl.available()[0]:
        pytest.skip("cv2 missing")
    now = [100.0]
    lay, ctl, procs = _layer(tmp_path, lambda: now[0])
    lay.command({"op": "on"})
    assert lay.on and procs and lay.state()["status"] == "starting"
    link = FakeLink.instances[-1]
    pose = {"l_shoulder": [0.45, 0.4, 1], "r_shoulder": [0.55, 0.4, 1], "l_hip": [0.46, 0.6, 1], "r_hip": [0.54, 0.6, 1],
            "l_wrist": [0.3, 0.2, 1], "r_wrist": [0.7, 0.5, 1], "l_elbow": [0.4, 0.3, 1]}
    for i in range(60):
        now[0] += 1 / 30
        link.q.append(_result(i + 1, now[0], [pose]))
        lay.pre_update(1 / 30)
    st = lay.state()
    assert st["status"] == "running" and st["bodies"] == 1
    assert ctl.signals["crowd_motion"] > 0.5 and ctl.signals["hand_height"] > 0.5
    u = lay.uniforms({"l_camera": 1.0, "crowd_soft": 0.4})
    assert u["p_l_camera"] == pytest.approx(0.0)            # no GL context: nothing bound, layer stays dark
    lay.command({"op": "set", "pos_x": 0.2})
    assert ctl.ext["camera"]["pos_x"] == 0.2 and "on" not in ctl.ext["camera"]
    lay.command({"op": "off"})
    assert not lay.on and link.quitted and procs[0].returncode == 0
    for _ in range(200):
        lay.pre_update(1 / 60)
    assert all(v == 0.0 for v in ctl.signals.values())


def test_layer_hand_ripples_reach_the_controller_rate_limited(tmp_path):
    if not cl.available()[0]:
        pytest.skip("cv2 missing")
    now = [0.0]
    lay, ctl, _ = _layer(tmp_path, lambda: now[0])
    lay.command({"op": "on"})
    link = FakeLink.instances[-1]
    lay._vals = {"crowd_ripples": 1.0, "l_camera": 1.0}
    for i in range(150):
        now[0] += 1 / 15
        x = 0.2 + 0.5 * (i % 2)
        link.q.append(_result(i + 1, now[0], [{"l_wrist": [x, 0.5, 1.0]}]))
        lay.pre_update(1 / 15)
    assert 0 < len(ctl.ripples) <= 10 * cl.RIPPLE_HZ + 1


def test_governor_strain_lowers_quality_then_restores(tmp_path):
    if not cl.available()[0]:
        pytest.skip("cv2 missing")
    now = [0.0]
    lay, ctl, _ = _layer(tmp_path, lambda: now[0])
    lay.command({"op": "on"})
    link = FakeLink.instances[-1]
    for _ in range(300):                                  # 5 s strained
        now[0] += 1 / 60
        lay.pre_update(1 / 60, strain=1)
    assert lay.quality == 1 and link.cfg[-1]["fps"] == 15
    for _ in range(60 * 25):
        now[0] += 1 / 60
        lay.pre_update(1 / 60, strain=0)
    assert lay.quality == 0 and link.cfg[-1]["fps"] == 30


def test_worker_crash_is_reported(tmp_path):
    if not cl.available()[0]:
        pytest.skip("cv2 missing")
    now = [0.0]
    lay, ctl, procs = _layer(tmp_path, lambda: now[0])
    lay.command({"op": "on"})
    procs[-1].returncode = 3                               # e.g. opencv missing in the child
    lay.pre_update(1 / 60)
    assert not lay.on and lay.state()["status"] == "error"


def test_controller_reads_crowd_mod_sources():
    import control
    c = control.Controller(restore=False, midi=False, ai=False)
    try:
        c.set("zoom", 0.0)
        c.set_mod("zoom", 0, "crowd_motion", amt=1.0)
        c.signals["crowd_motion"] = 0.8
        for _ in range(30):
            fr = c.update(1 / 60, np.zeros(3))
        assert fr.values["zoom"] > 0.5
        c.signals.clear()
        for _ in range(30):
            fr = c.update(1 / 60, np.zeros(3))
        assert abs(fr.values["zoom"]) < 0.05
    finally:
        c.close()


def test_glsl_snippet_declares_what_the_engine_calls():
    for name in ("camWarp", "camScene", "camShow", "camM", "camF", "camH", "p_l_camera"):
        assert name in cl.SCENE_GLSL and (name in cl.NULL_GLSL or name in ("camM", "camF", "camH"))


def test_governor_status_shows_the_camera_cost_only_while_on():
    import governor
    g = governor.Governor(sensors=None) if "sensors" in governor.Governor.__init__.__code__.co_varnames else None
    if g is None:
        pytest.skip("governor signature changed")
    assert g.status()["camera"] is None
    g.camera_source = lambda: dict(active=True, engine_ms=0.1, fps=10)
    assert g.status()["camera"]["fps"] == 10
    g.camera_source = lambda: dict(active=False)
    assert g.status()["camera"] is None


# ---------------------------------------------------------------- regression tests for the Neon review (camera side)

def test_crop_edge_moves_clamp_instead_of_resetting():
    s = cl.clean_settings({"crop": [0.5, 0.5, 0.55, 0.9]})
    assert s["crop"] == pytest.approx([0.5, 0.5, 0.6, 0.9])


def test_signal_6hz_flicker_is_tiny_and_slew_limited():
    sg = cl.Signals()
    vals = []
    for i in range(600):
        if i % 5 == 0:
            sg.target = dict.fromkeys(cl.SIGNALS, 1.0 if (i // 5) % 2 == 0 else 0.0)
        vals.append(sg.step(1 / 60)["crowd_motion"])
    tail = vals[300:]
    assert max(tail) - min(tail) < 0.15
    assert max(abs(b - a) for a, b in zip(vals, vals[1:])) <= cl.SLEW_MAX / 60 + 1e-9


def test_hand_speed_never_pairs_left_with_right():
    sg = cl.Signals()
    b = {"l_shoulder": (0.45, 0.6), "r_shoulder": (0.55, 0.6), "l_hip": (0.46, 0.4), "r_hip": (0.54, 0.4),
         "l_wrist": (0.2, 0.5), "r_wrist": (0.8, 0.5)}
    sg.on_data({}, [b], 1 / 15)
    b2 = dict(b)
    del b2["l_wrist"]                                     # left wrist lost: only the (still) right one remains
    sg.on_data({}, [b2], 1 / 15)
    assert sg.target["hand_speed"] < 0.01


class SlowLink(FakeLink):
    def quit(self):
        time.sleep(1.0)
        self.quitted = True


def test_camera_off_never_blocks_the_render_thread(tmp_path):
    (tmp_path / ".remote_token").write_text("tok")
    (tmp_path / "camera_worker.py").write_text("#")
    if not cl.available()[0]:
        pytest.skip("cv2 missing")
    procs = []
    lay = cl.CameraLayer(None, FakeCtl(), here=tmp_path, spawn=lambda p, s, h, t=None: procs.append(FakeProc()) or procs[-1],
                         link_factory=SlowLink)
    lay.command({"op": "on"})
    t0 = time.monotonic()
    while lay.link is None and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    a = time.perf_counter()
    lay.command({"op": "off"})
    assert time.perf_counter() - a < 0.05 and not lay.on
    lay.command({"op": "on"})                             # waits for the teardown on its own thread, not here
    assert time.perf_counter() - a < 0.1
    lay._teardown.join(3)
    t0 = time.monotonic()
    while lay.link is None and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert lay.link is not None and lay.token and len(procs) == 2
    lay.close()


def test_silhouettes_glide_every_frame(tmp_path):
    if not cl.available()[0]:
        pytest.skip("cv2 missing")
    now = [0.0]
    lay, ctl, _ = _layer(tmp_path, lambda: now[0])
    lay.command({"op": "on"})
    link = FakeLink.instances[-1]
    lay._vals = {"crowd_soft": 0.0}
    now[0] += 0.1
    link.q.append(_result(1, now[0]))
    lay.pre_update(1 / 60)
    assert lay._mask_target is not None and lay.mask is not None
    lay.mask = np.zeros_like(lay._mask_target)
    lay.pre_update(1 / 60)
    assert float(lay.mask.max()) <= 1.0 - math.exp(-(1 / 60) / cl.MASK_TAU_MIN) + 1e-6


def test_worker_http_auth_still_gating_and_quit():
    pytest.importorskip("cv2")
    import http.client
    import json as _j
    import socket

    class Cap:
        def read(self):
            time.sleep(0.02)
            return True, np.zeros((360, 640, 3), np.uint8)

        def release(self):
            pass

    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    w = cw.Worker("secret", port=port, capture_factory=lambda d: Cap(),
                  models=(lambda rgb, ts: np.zeros(rgb.shape[:2], np.float32), lambda rgb, ts, n: []))
    th = threading.Thread(target=w.run, daemon=True)
    th.start()
    time.sleep(0.6)

    def req(method, path, token="secret", body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request(method, path, body=body, headers={"X-Token": token, "Content-Type": "application/json"})
        r = c.getresponse(); data = r.read(); c.close()
        return r.status, data
    assert req("GET", "/status", token="wrong")[0] == 403
    assert req("GET", "/still?skip=1")[0] == 403                      # stills are off unless AUTO MAP asks
    assert req("POST", "/config", body=_j.dumps({"allow_still_s": 30}).encode())[0] == 200
    st, data = req("GET", "/still?skip=1")
    assert st == 200 and cw.unpack_still(data).shape == (360, 640)
    assert req("POST", "/config", body=_j.dumps({"allow_still_s": 0}).encode())[0] == 200
    assert req("GET", "/still?skip=1")[0] == 403
    st, data = req("GET", "/frame?after=0")
    assert st in (200, 204)
    assert req("POST", "/quit")[0] == 200
    th.join(5)
    assert not th.is_alive()


def test_shader_guards_the_vanishing_line_and_respects_the_edge():
    assert "c.z > 1e-4" in cl.SCENE_GLSL and "float edge" in cl.SCENE_GLSL
