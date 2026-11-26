"""Headless browser: the CAMERA tab, the top-bar indicator and the AUTO MAP card (fake engine state, no camera, no GPU)."""
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pw_api = pytest.importorskip("playwright.sync_api")
import control  # noqa: E402
from remote_control import RemoteControl  # noqa: E402

CAM = dict(on=True, status="running", reason="", notes="", available=True, quality=0, bodies=1, cost_ms=0.1,
           settings=dict(device=0, res=1, fps=30, pose=True, max_poses=2, mirror=True, crop=[0, 0, 1, 1], pos_x=0, pos_y=0,
                         scale=1, key_x=0, key_y=0, gain=1, show_ir=False, auto_quality=True),
           devices=[dict(index=0, name="Integrated Camera", ir=False), dict(index=1, name="Integrated IR Camera", ir=True)],
           signals=dict(crowd_motion=0.4, hand_height=0.8), fps=10, grab_fps=10, ms=dict(seg=6, flow=4, pose=30),
           backend="mediapipe", res_labels=["480x270", "320x180", "240x135"])
AM = dict(step="detect", busy=False, msg="3 face(s) found", err="", progress=0, image=0, made=[], report=[],
          calib=dict(method="aruco+graycode", rms=0.8, seen="24/24 markers", dense=True, cam=[640, 360], proj=[1280, 800],
                     lens=[-0.15, 0.05], coverage=0.97, warning="", mirrored=True),
          found=[dict(quad=True, conf=1.0, area=0.04, name="", ai=False, wall=False),
                 dict(quad=True, conf=1.0, area=0.03, name="front A", ai=True, wall=False),
                 dict(quad=True, conf=0.4, area=0.4, name="Wall (whole projection)", ai=False, wall=True)],
          ai=dict(online=True, providers=["gemini-api"], pending=dict(kind="detect", person=True, redacted=0.1,
                                                                    prompt="find the faces", bytes=1234)),
          profile="builtin", profiles=dict(builtin="Built-in webcam", usb="USB webcam"), pose=1, poses=[1],
          mirrored=True, adapt=dict(stripe=8, level=200, snr=40, scale=0.3),
          framing=dict(coverage=0.5, advice=["tilt the camera UP a little (the top of the projection is cut off)"]))


@pytest.fixture
def served(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = RemoteControl(port=port, token="tok", quiet=True, host="127.0.0.1", art_dir=tmp_path / "art", tex_dir=tmp_path / "tex")
    rc.start()
    ctl = control.Controller(tmp_path, midi=False, ai=False, art_dir=tmp_path / "art")
    sent, stop = [], threading.Event()

    def loop():
        while not stop.is_set():
            rest = ctl.apply(rc.drain())
            sent.extend(c for c in rest or () if c.get("cmd") in ("camera", "automap"))
            ctl.update(1 / 60, [0.2, 0.1, 0.1])
            rc.publish_state(dict(ctl.state(), corners=[[0, 0], [1, 0], [1, 1], [0, 1]], res=[1280, 800], fps=60,
                                  camera=CAM, automap=AM))
            time.sleep(1 / 30)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}/?k=tok", sent
    stop.set()
    t.join(2)
    ctl.close()
    rc.stop()


def launch(pw):
    for ch in ("msedge", "chrome", None):
        try:
            return pw.chromium.launch(channel=ch, headless=True) if ch else pw.chromium.launch(headless=True)
        except Exception:
            continue
    pytest.skip("no Chromium-family browser")


@pytest.mark.parametrize("size", [(1600, 1000), (390, 844)])
def test_camera_tab_indicator_and_automap_card(served, size):
    url, sent = served
    errors = []
    with pw_api.sync_playwright() as pw:
        br = launch(pw)
        page = br.new_page(viewport=dict(width=size[0], height=size[1]))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "automap-image" not in m.text else None)
        page.goto(url)
        page.wait_for_function("window.__camera && window.__camera.state().on === true", timeout=10000)
        assert "CAMERA ON" in page.inner_text("#camsw")
        page.click('#tabs button[data-tab="camera"]')
        page.wait_for_timeout(400)
        assert "click = OFF" in page.inner_text("#cambig")
        assert page.eval_on_selector_all("#camdev option", "o => o.map(x => x.textContent)") == ["0: Integrated Camera"]
        assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 2
        page.click("#camsw")                                         # one click in the top bar = CAMERA OFF
        page.wait_for_timeout(400)
        assert {"cmd": "camera", "type": "camera", "op": "off"} in sent
        page.click('#tabs button[data-tab="mapping"]')
        page.wait_for_selector("#automap")
        page.wait_for_timeout(500)
        rows = page.inner_text("#automap")
        assert "AI" in rows and "front A" in page.eval_on_selector_all("#automap input[type=text]", "e => e.map(x => x.value)")
        walls = page.eval_on_selector_all("#automap .amf input[type=checkbox]", "e => e.map(x => x.checked)")
        assert walls == [True, True, False]                          # the wall candidate is not ticked
        assert page.is_visible("#amaisend") and "send anyway" in rows
        n = len(sent)
        page.click("#amcal")                                         # first tap only arms
        page.wait_for_timeout(300)
        assert not any(c.get("op") == "calibrate" for c in sent[n:]) and "TAP AGAIN" in page.inner_text("#amcal")
        page.click("#amcal")
        page.wait_for_timeout(400)
        assert any(c.get("op") == "calibrate" for c in sent[n:])
        assert "BUILT-IN" in page.inner_text("#automap") and "tilt the camera UP" in rows
        assert page.evaluate("document.documentElement.scrollWidth - window.innerWidth") <= 2
        br.close()
    assert errors == [], errors
